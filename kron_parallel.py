
# Script 3: Parallelizing tiny Kronecker terms across the GPU
#
# Main systems question
# ---------------------
# Given a fixed tiny-factor Kronecker representation, can we expose
# BOTH axes of independent work to the GPU?
#
#   1. output entries (i,j)
#   2. Kronecker / separation terms s
#
# Fixed representation:
#
#   W = sum_{s=1}^S
#         F0_s kron F1_s kron F2_s kron F3_s kron F4_s
#
# with
#
#   F0 : 1 x 3   (3 values)
#   F1 : 2 x 4   (8 values)
#   F2 : 2 x 4   (8 values)
#   F3 : 2 x 4   (8 values)
#   F4 : 2 x 4   (8 values)
#
# Full W shape:
#
#   16 x 768
#
# Every contribution to one output entry requires only five factor
# scalar loads and a few multiplies.
#
# We compare:
#
#   A. serial-rank native
#      one GPU thread owns one output entry and loops over all S terms.
#
#   B. rank-parallel native
#      the 2-D work grid is:
#
#          output tile x rank chunk
#
#      Different rank chunks are evaluated concurrently on the GPU.
#      A second tiny reduction kernel combines the partial sums.
#
#   C. repeated torch.kron
#      framework-level expansion baseline.
#
# This is the clean experiment for the claim:
#
#   tiny Kronecker factors make each term contribution so small that
#   many independent factor contributions can be issued concurrently
#   across GPU threads/warps rather than serialized as separate matrix
#   operations.
#
# IMPORTANT:
# We do not literally assign one matrix to one physical CUDA core.
# CUDA schedules threads/warps on SMs and CUDA cores.
#
# Run:
#   python kron_term_parallel.py

import math
import statistics

import torch
import triton
import triton.language as tl


assert torch.cuda.is_available()

DEV = "cuda"
DT = torch.float32

torch.manual_seed(17)
torch.cuda.manual_seed_all(17)

print("GPU   :", torch.cuda.get_device_name(0))
print("torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("Triton:", triton.__version__)

NROW = 16
NCOL = 768
NEL = NROW * NCOL


# ============================================================
# Timing
# ============================================================

def sync():
    torch.cuda.synchronize()


def median_us(
    fn,
    warmup=10,
    iters=100,
    repeats=7,
):
    for _ in range(warmup):
        fn()

    sync()

    vals = []

    for _ in range(repeats):
        a = torch.cuda.Event(
            enable_timing=True
        )
        b = torch.cuda.Event(
            enable_timing=True
        )

        a.record()

        for _ in range(iters):
            fn()

        b.record()
        sync()

        vals.append(
            a.elapsed_time(b)
            * 1000.0
            / iters
        )

    return statistics.median(vals)


def rel_l2(a, b, eps=1e-12):
    return (
        (a.float() - b.float()).norm()
        / b.float().norm().clamp_min(eps)
    ).item()


# ============================================================
# Input generation
# ============================================================

def make_factors(S, seed=17):
    g = torch.Generator(
        device="cpu"
    )
    g.manual_seed(
        seed + S
    )

    def rn(*shape):
        return (
            0.5
            * torch.randn(
                *shape,
                generator=g,
                dtype=torch.float32,
            )
        ).cuda().contiguous()

    return [
        rn(S, 1, 3),
        rn(S, 2, 4),
        rn(S, 2, 4),
        rn(S, 2, 4),
        rn(S, 2, 4),
    ]


# ============================================================
# A. Serial-rank native kernel
#
# Parallel over output entries.
# Serial over S inside each output lane.
# ============================================================

@triton.jit
def serial_rank_k5_kernel(
    F0,
    F1,
    F2,
    F3,
    F4,
    OUT,
    S,
    NEL_CONST: tl.constexpr,
    NCOL_CONST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = (
        tl.program_id(0) * BLOCK
        + tl.arange(0, BLOCK)
    )

    mask = e < NEL_CONST

    i = e // NCOL_CONST
    j = e - i * NCOL_CONST

    i1 = i // 8
    i2 = (i // 4) % 2
    i3 = (i // 2) % 2
    i4 = i % 2

    j0 = j // 256
    j1 = (j // 64) % 4
    j2 = (j // 16) % 4
    j3 = (j // 4) % 4
    j4 = j % 4

    acc = tl.zeros(
        (BLOCK,),
        dtype=tl.float32,
    )

    for s in tl.range(
        0,
        S,
        loop_unroll_factor=1,
    ):
        a = tl.load(
            F0
            + s * 3
            + j0,
            mask=mask,
            other=0.0,
        )

        b = tl.load(
            F1
            + s * 8
            + i1 * 4
            + j1,
            mask=mask,
            other=0.0,
        )

        c = tl.load(
            F2
            + s * 8
            + i2 * 4
            + j2,
            mask=mask,
            other=0.0,
        )

        d = tl.load(
            F3
            + s * 8
            + i3 * 4
            + j3,
            mask=mask,
            other=0.0,
        )

        f = tl.load(
            F4
            + s * 8
            + i4 * 4
            + j4,
            mask=mask,
            other=0.0,
        )

        acc += (
            a * b * c * d * f
        )

    tl.store(
        OUT + e,
        acc,
        mask=mask,
    )


def serial_rank(
    factors,
    out,
    S,
    block=256,
):
    grid = (
        triton.cdiv(
            NEL,
            block,
        ),
    )

    serial_rank_k5_kernel[grid](
        *factors,
        out,
        S,
        NEL_CONST=NEL,
        NCOL_CONST=NCOL,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )


# ============================================================
# B1. Rank-parallel partial kernel
#
# Grid:
#
#   program_id(0) = output tile
#   program_id(1) = rank chunk
#
# Each program computes:
#
#   partial[rank_chunk, output_entry]
#
# Multiple rank chunks therefore execute concurrently.
# ============================================================

@triton.jit
def rank_parallel_partial_kernel(
    F0,
    F1,
    F2,
    F3,
    F4,
    PARTIAL,
    S,
    NEL_CONST: tl.constexpr,
    NCOL_CONST: tl.constexpr,
    RANK_CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_block = tl.program_id(0)
    rank_block = tl.program_id(1)

    e = (
        out_block * BLOCK
        + tl.arange(0, BLOCK)
    )

    mask = e < NEL_CONST

    i = e // NCOL_CONST
    j = e - i * NCOL_CONST

    i1 = i // 8
    i2 = (i // 4) % 2
    i3 = (i // 2) % 2
    i4 = i % 2

    j0 = j // 256
    j1 = (j // 64) % 4
    j2 = (j // 16) % 4
    j3 = (j // 4) % 4
    j4 = j % 4

    s0 = (
        rank_block
        * RANK_CHUNK
    )

    acc = tl.zeros(
        (BLOCK,),
        dtype=tl.float32,
    )

    for r in tl.static_range(
        0,
        RANK_CHUNK,
    ):
        s = s0 + r
        smask = mask & (s < S)

        a = tl.load(
            F0
            + s * 3
            + j0,
            mask=smask,
            other=0.0,
        )

        b = tl.load(
            F1
            + s * 8
            + i1 * 4
            + j1,
            mask=smask,
            other=0.0,
        )

        c = tl.load(
            F2
            + s * 8
            + i2 * 4
            + j2,
            mask=smask,
            other=0.0,
        )

        d = tl.load(
            F3
            + s * 8
            + i3 * 4
            + j3,
            mask=smask,
            other=0.0,
        )

        f = tl.load(
            F4
            + s * 8
            + i4 * 4
            + j4,
            mask=smask,
            other=0.0,
        )

        acc += (
            a * b * c * d * f
        )

    base = (
        rank_block
        * NEL_CONST
        + e
    )

    tl.store(
        PARTIAL + base,
        acc,
        mask=mask,
    )


# ============================================================
# B2. Reduction kernel
#
# One output thread sums the much smaller number of rank chunks.
# ============================================================

@triton.jit
def reduce_partial_kernel(
    PARTIAL,
    OUT,
    NEL_CONST: tl.constexpr,
    NRANK_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = (
        tl.program_id(0) * BLOCK
        + tl.arange(0, BLOCK)
    )

    mask = e < NEL_CONST

    acc = tl.zeros(
        (BLOCK,),
        dtype=tl.float32,
    )

    for rb in tl.static_range(
        0,
        NRANK_BLOCKS,
    ):
        acc += tl.load(
            PARTIAL
            + rb * NEL_CONST
            + e,
            mask=mask,
            other=0.0,
        )

    tl.store(
        OUT + e,
        acc,
        mask=mask,
    )


def rank_parallel(
    factors,
    partial,
    out,
    S,
    rank_chunk,
    block=256,
):
    nrank_blocks = (
        S + rank_chunk - 1
    ) // rank_chunk

    grid1 = (
        triton.cdiv(
            NEL,
            block,
        ),
        nrank_blocks,
    )

    rank_parallel_partial_kernel[
        grid1
    ](
        *factors,
        partial,
        S,
        NEL_CONST=NEL,
        NCOL_CONST=NCOL,
        RANK_CHUNK=rank_chunk,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )

    grid2 = (
        triton.cdiv(
            NEL,
            block,
        ),
    )

    reduce_partial_kernel[
        grid2
    ](
        partial,
        out,
        NEL_CONST=NEL,
        NRANK_BLOCKS=nrank_blocks,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )


# ============================================================
# Framework baseline
# ============================================================

@torch.no_grad()
def repeated_torch_kron(
    factors,
):
    S = factors[0].shape[0]

    out = torch.zeros(
        NROW,
        NCOL,
        device=DEV,
        dtype=DT,
    )

    for s in range(S):
        term = factors[0][s]

        for f in factors[1:]:
            term = torch.kron(
                term,
                f[s],
            )

        out.add_(term)

    return out


# ============================================================
# Autotune rank-parallel schedule
# ============================================================

RANK_CHUNKS = [
    2,
    4,
    8,
    16,
    32,
    64,
]


@torch.no_grad()
def tune_parallel(
    factors,
    S,
):
    rows = []

    for rc in RANK_CHUNKS:
        nrank_blocks = (
            S + rc - 1
        ) // rc

        partial = torch.empty(
            nrank_blocks,
            NEL,
            device=DEV,
            dtype=DT,
        )

        out = torch.empty(
            NROW,
            NCOL,
            device=DEV,
            dtype=DT,
        )

        try:
            rank_parallel(
                factors,
                partial,
                out,
                S,
                rc,
            )

            sync()

            us = median_us(
                lambda: rank_parallel(
                    factors,
                    partial,
                    out,
                    S,
                    rc,
                ),
                warmup=5,
                iters=100,
                repeats=5,
            )

            rows.append(
                {
                    "rank_chunk": rc,
                    "rank_blocks":
                        nrank_blocks,
                    "us": us,
                    "partial": partial,
                    "out": out,
                }
            )

        except Exception as e:
            rows.append(
                {
                    "rank_chunk": rc,
                    "error": repr(e),
                }
            )

    valid = [
        r for r in rows
        if "us" in r
    ]

    if not valid:
        raise RuntimeError(rows)

    best = min(
        valid,
        key=lambda r: r["us"],
    )

    return best, rows


# ============================================================
# One experiment
# ============================================================

@torch.no_grad()
def run_case(
    S,
    time_framework,
):
    factors = make_factors(
        S
    )

    out_serial = torch.empty(
        NROW,
        NCOL,
        device=DEV,
        dtype=DT,
    )

    # Serial-rank native.
    serial_rank(
        factors,
        out_serial,
        S,
    )
    sync()

    serial_us = median_us(
        lambda: serial_rank(
            factors,
            out_serial,
            S,
        ),
        warmup=10,
        iters=200,
        repeats=7,
    )

    # Rank-parallel native.
    best, configs = tune_parallel(
        factors,
        S,
    )

    rank_parallel(
        factors,
        best["partial"],
        best["out"],
        S,
        best["rank_chunk"],
    )
    sync()

    err_parallel = rel_l2(
        best["out"].view(
            NROW,
            NCOL,
        ),
        out_serial,
    )

    result = {
        "S": S,
        "serial_us":
            serial_us,
        "parallel_us":
            best["us"],
        "parallel_speedup":
            serial_us
            / best["us"],
        "rank_chunk":
            best["rank_chunk"],
        "rank_blocks":
            best["rank_blocks"],
        "parallel_rel_l2":
            err_parallel,
    }

    if time_framework:
        iters = (
            5 if S <= 64
            else 2
        )

        kron_us = median_us(
            lambda: repeated_torch_kron(
                factors
            ),
            warmup=1,
            iters=iters,
            repeats=5,
        )

        result[
            "torch_kron_us"
        ] = kron_us

        result[
            "parallel_vs_kron"
        ] = (
            kron_us
            / best["us"]
        )

    # Throughput of Kronecker-term/output contributions.
    contributions = (
        S * NEL
    )

    result[
        "parallel_contrib_per_us"
    ] = (
        contributions
        / best["us"]
    )

    result[
        "serial_contrib_per_us"
    ] = (
        contributions
        / serial_us
    )

    return result


# ============================================================
# Main
# ============================================================

def main():
    print()
    print("=" * 128)
    print(
        "TINY KRONECKER TERM PARALLELISM"
    )
    print("=" * 128)

    print()
    print(
        "Fixed factorization: "
        "(1x3) x (2x4) x (2x4) x (2x4) x (2x4)"
    )
    print(
        "Each separation term stores only 35 FP32 values."
    )
    print(
        "Full represented matrix: 16 x 768 = 12,288 outputs."
    )

    print()
    print(
        f"{'S':>5} | "
        f"{'serial-rank':>12} "
        f"{'rank-par':>12} "
        f"{'speedup':>9} | "
        f"{'chunk':>6} "
        f"{'rblocks':>7} | "
        f"{'parallel contrib/us':>20} "
        f"{'rel err':>10}"
    )

    print("-" * 104)

    framework_set = {
        8,
        32,
        64,
        128,
        256,
        512,
    }

    results = []

    for S in [
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    ]:
        r = run_case(
            S,
            time_framework=(
                S in framework_set
            ),
        )

        results.append(r)

        print(
            f"{S:5d} | "
            f"{r['serial_us']:10.2f}us "
            f"{r['parallel_us']:10.2f}us "
            f"{r['parallel_speedup']:8.2f}x | "
            f"{r['rank_chunk']:6d} "
            f"{r['rank_blocks']:7d} | "
            f"{r['parallel_contrib_per_us']:20.1f} "
            f"{r['parallel_rel_l2']:10.2e}"
        )

    print()
    print("=" * 128)
    print(
        "FRAMEWORK EXPANSION VS RANK-PARALLEL NATIVE"
    )
    print("=" * 128)

    print(
        f"{'S':>5} | "
        f"{'torch.kron':>12} "
        f"{'rank-par':>12} "
        f"{'speedup':>10}"
    )

    print("-" * 48)

    for r in results:
        if "torch_kron_us" not in r:
            continue

        print(
            f"{r['S']:5d} | "
            f"{r['torch_kron_us']:10.2f}us "
            f"{r['parallel_us']:10.2f}us "
            f"{r['parallel_vs_kron']:9.2f}x"
        )

    print()
    print("=" * 128)
    print("HOW TO READ THIS")
    print("=" * 128)

    print(
        r"""
This experiment does NOT vary K.

It fixes one tiny-factor Kronecker representation and changes only
the execution schedule.

serial-rank:
    output entries are parallel, but all S Kronecker terms are
    accumulated sequentially inside each output thread.

rank-parallel:
    output entries AND groups of Kronecker terms become independent
    GPU work. The GPU evaluates many tiny factor contributions
    concurrently and reduces the partial sums afterward.

The most important quantity is:

    serial-rank latency / rank-parallel latency

as S grows.

If the ratio grows above 1 at large S, that is direct evidence that
the separation/factor dimension itself contains exploitable GPU
parallelism.

The paper wording should be:

    Small Kronecker factors reduce each term contribution to a handful
    of scalar operations. This allows the runtime to expose thousands
    of independent term/output contributions concurrently to CUDA
    threads and warps, rather than dispatching the factors as a
    sequence of small matrix operations.

Not:

    one Kronecker matrix is placed on one CUDA core.
"""
    )


if __name__ == "__main__":
    main()


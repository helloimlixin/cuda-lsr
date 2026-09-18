
# Script 1: Exact-W tiny-Kronecker-factor GPU benchmark
#
# Main hypothesis:
#   If Kronecker factor matrices become extremely small, the work needed
#   for one output entry becomes only a few scalar loads/multiplies.
#   We can expose many independent output entries to GPU threads instead
#   of treating every factor as a separate matrix multiplication.
#
# IMPORTANT TERMINOLOGY:
#   We are NOT statically pinning one factor matrix to one physical CUDA core.
#   Rather, each GPU thread computes one (or a few) output entries, and the
#   tiny factor values it needs are used as register-local scalar operands.
#   CUDA schedules those threads across CUDA cores.
#
# This script performs TWO experiments:
#
#   A) Exact-W factor-granularity experiment
#      K=3, K=4, and K=5 represent the SAME 16 x 768 matrix.
#
#      K=3 largest factor: 4 x 48 = 192 values
#      K=4 largest factor: 2 x 12 = 24 values
#      K=5 largest factor: 2 x 4  = 8 values
#
#      The K=4 and K=3 representations are constructed by exactly merging
#      neighboring K=5 factors using Kronecker associativity.
#
#      This isolates the question:
#
#          Does finer factor granularity help the GPU even when W is fixed?
#
#   B) Framework-expansion vs Kronecker-native execution
#
#      Compare the native entrywise kernel against:
#        1) repeated torch.kron
#        2) vectorized batched Kronecker construction
#
# Recommended environment:
#   NVIDIA GPU
#   PyTorch CUDA build
#   Triton 3.x
#
# Example:
#   python exact_w_tiny_kron.py
#
# Optional:
#   python exact_w_tiny_kron.py --s-values 8 16 32 64 128 256

import argparse
import math
import statistics

import torch
import triton
import triton.language as tl


# ============================================================
# Environment
# ============================================================

assert torch.cuda.is_available(), "CUDA GPU required"

DEV = "cuda"
DT = torch.float32

torch.manual_seed(17)
torch.cuda.manual_seed_all(17)

print("GPU   :", torch.cuda.get_device_name(0))
print("torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("Triton:", triton.__version__)


# ============================================================
# Utilities
# ============================================================

def sync():
    torch.cuda.synchronize()


def rel_l2(a, b, eps=1e-12):
    return (
        (a.float() - b.float()).norm()
        / b.float().norm().clamp_min(eps)
    ).item()


def median_us(fn, warmup=10, iters=100, repeats=7):
    for _ in range(warmup):
        fn()

    sync()
    vals = []

    for _ in range(repeats):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)

        a.record()

        for _ in range(iters):
            fn()

        b.record()
        sync()

        vals.append(
            a.elapsed_time(b) * 1000.0 / iters
        )

    return statistics.median(vals)


def batched_kron(A, B):
    """
    A: [S,a,b]
    B: [S,c,d]

    returns:
        [S,a*c,b*d]
    """
    S, a, b = A.shape
    S2, c, d = B.shape
    assert S == S2

    return (
        torch.einsum(
            "sab,scd->sacbd",
            A,
            B,
        )
        .reshape(
            S,
            a * c,
            b * d,
        )
        .contiguous()
    )


# ============================================================
# Construct mathematically equivalent K=5 / K=4 / K=3 factors
#
# Full output dimension:
#   1 * 2 * 2 * 2 * 2 = 16
#
# Full input dimension:
#   3 * 4 * 4 * 4 * 4 = 768
#
# K=5:
#   (1x3) kron (2x4) kron (2x4) kron (2x4) kron (2x4)
#
# K=4:
#   [(1x3) kron (2x4)] kron (2x4) kron (2x4) kron (2x4)
#   = (2x12) kron (2x4) kron (2x4) kron (2x4)
#
# K=3:
#   [(2x12) kron (2x4)] kron (2x4) kron (2x4)
#   = (4x48) kron (2x4) kron (2x4)
#
# Therefore all three represent exactly the same matrix in exact
# arithmetic. FP32 differs only because multiplication/reduction
# order changes.
# ============================================================

def make_equivalent_factors(S, seed=17):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + S)

    def rn(shape):
        # Modest scale keeps sums numerically well behaved.
        return (
            0.5
            * torch.randn(
                *shape,
                generator=g,
                dtype=torch.float32,
            )
        ).cuda().contiguous()

    f0 = rn((S, 1, 3))
    f1 = rn((S, 2, 4))
    f2 = rn((S, 2, 4))
    f3 = rn((S, 2, 4))
    f4 = rn((S, 2, 4))

    k5 = [
        f0,
        f1,
        f2,
        f3,
        f4,
    ]

    g01 = batched_kron(
        f0,
        f1,
    )

    k4 = [
        g01,
        f2,
        f3,
        f4,
    ]

    g012 = batched_kron(
        g01,
        f2,
    )

    k3 = [
        g012,
        f3,
        f4,
    ]

    return {
        3: k3,
        4: k4,
        5: k5,
    }


# ============================================================
# Kronecker-native Triton kernels
#
# Each lane/thread handles one dense output entry W[i,j].
#
# For that output entry it:
#   1) decodes (i,j) into tensor-mode coordinates,
#   2) reads ONE scalar from every factor,
#   3) multiplies those K scalars,
#   4) accumulates over separation terms S.
#
# No intermediate Kronecker matrix is materialized.
# ============================================================

@triton.jit
def direct_k3_kernel(
    F0,  # [S,4,48]
    F1,  # [S,2,4]
    F2,  # [S,2,4]
    OUT, # [16,768]
    S,
    NEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = (
        tl.program_id(0) * BLOCK
        + tl.arange(0, BLOCK)
    )

    mask = e < NEL

    i = e // 768
    j = e - i * 768

    # Output modes: [4,2,2]
    i0 = i // 4
    i1 = (i // 2) % 2
    i2 = i % 2

    # Input modes: [48,4,4]
    j0 = j // 16
    j1 = (j // 4) % 4
    j2 = j % 4

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
            + s * (4 * 48)
            + i0 * 48
            + j0,
            mask=mask,
            other=0.0,
        )

        b = tl.load(
            F1
            + s * (2 * 4)
            + i1 * 4
            + j1,
            mask=mask,
            other=0.0,
        )

        c = tl.load(
            F2
            + s * (2 * 4)
            + i2 * 4
            + j2,
            mask=mask,
            other=0.0,
        )

        acc += a * b * c

    tl.store(
        OUT + e,
        acc,
        mask=mask,
    )


@triton.jit
def direct_k4_kernel(
    F0,  # [S,2,12]
    F1,  # [S,2,4]
    F2,  # [S,2,4]
    F3,  # [S,2,4]
    OUT,
    S,
    NEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = (
        tl.program_id(0) * BLOCK
        + tl.arange(0, BLOCK)
    )

    mask = e < NEL

    i = e // 768
    j = e - i * 768

    # Output modes: [2,2,2,2]
    i0 = i // 8
    i1 = (i // 4) % 2
    i2 = (i // 2) % 2
    i3 = i % 2

    # Input modes: [12,4,4,4]
    j0 = j // 64
    j1 = (j // 16) % 4
    j2 = (j // 4) % 4
    j3 = j % 4

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
            + s * (2 * 12)
            + i0 * 12
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

        acc += (
            a * b * c * d
        )

    tl.store(
        OUT + e,
        acc,
        mask=mask,
    )


@triton.jit
def direct_k5_kernel(
    F0,  # [S,1,3]
    F1,  # [S,2,4]
    F2,  # [S,2,4]
    F3,  # [S,2,4]
    F4,  # [S,2,4]
    OUT,
    S,
    NEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = (
        tl.program_id(0) * BLOCK
        + tl.arange(0, BLOCK)
    )

    mask = e < NEL

    i = e // 768
    j = e - i * 768

    # Output modes: [1,2,2,2,2]
    # i0 is always 0.
    i1 = i // 8
    i2 = (i // 4) % 2
    i3 = (i // 2) % 2
    i4 = i % 2

    # Input modes: [3,4,4,4,4]
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


# ============================================================
# Kernel wrappers
# ============================================================

NEL = 16 * 768


def native_reconstruct(
    factors,
    K,
    out,
    block=256,
):
    S = factors[0].shape[0]

    grid = (
        triton.cdiv(
            NEL,
            block,
        ),
    )

    num_warps = (
        2 if block <= 64
        else 4
        if block <= 256
        else 8
    )

    if K == 3:
        direct_k3_kernel[grid](
            factors[0],
            factors[1],
            factors[2],
            out,
            S,
            NEL=NEL,
            BLOCK=block,
            num_warps=num_warps,
            num_stages=1,
        )

    elif K == 4:
        direct_k4_kernel[grid](
            factors[0],
            factors[1],
            factors[2],
            factors[3],
            out,
            S,
            NEL=NEL,
            BLOCK=block,
            num_warps=num_warps,
            num_stages=1,
        )

    elif K == 5:
        direct_k5_kernel[grid](
            factors[0],
            factors[1],
            factors[2],
            factors[3],
            factors[4],
            out,
            S,
            NEL=NEL,
            BLOCK=block,
            num_warps=num_warps,
            num_stages=1,
        )

    else:
        raise ValueError(K)

    return out


# ============================================================
# Framework baselines
# ============================================================

@torch.no_grad()
def repeated_torch_kron(
    factors,
):
    S = factors[0].shape[0]

    out = torch.zeros(
        16,
        768,
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


@torch.no_grad()
def vectorized_kron(
    factors,
):
    term = factors[0]

    for f in factors[1:]:
        term = batched_kron(
            term,
            f,
        )

    return term.sum(dim=0)


# ============================================================
# Search best block for native kernel
# ============================================================

@torch.no_grad()
def find_best_native(
    factors,
    K,
    out,
    blocks=(64, 128, 256, 512),
):
    rows = []

    for block in blocks:
        try:
            native_reconstruct(
                factors,
                K,
                out,
                block=block,
            )
            sync()

            us = median_us(
                lambda: native_reconstruct(
                    factors,
                    K,
                    out,
                    block=block,
                ),
                warmup=10,
                iters=200,
                repeats=7,
            )

            rows.append(
                {
                    "block": block,
                    "us": us,
                }
            )

        except Exception as e:
            rows.append(
                {
                    "block": block,
                    "error": repr(e),
                }
            )

    valid = [
        r
        for r in rows
        if "us" in r
    ]

    if not valid:
        raise RuntimeError(rows)

    return (
        min(
            valid,
            key=lambda r: r["us"],
        ),
        rows,
    )


# ============================================================
# One S case
# ============================================================

@torch.no_grad()
def run_case(
    S,
    time_framework=True,
):
    reps = make_equivalent_factors(
        S,
        seed=17,
    )

    outs = {
        K: torch.empty(
            16,
            768,
            device=DEV,
            dtype=DT,
        )
        for K in [3, 4, 5]
    }

    best = {}

    for K in [3, 4, 5]:
        b, configs = find_best_native(
            reps[K],
            K,
            outs[K],
        )

        best[K] = {
            "us": b["us"],
            "block": b["block"],
            "configs": configs,
        }

        native_reconstruct(
            reps[K],
            K,
            outs[K],
            block=b["block"],
        )

    sync()

    # All three must represent the same W.
    err_35 = rel_l2(
        outs[3],
        outs[5],
    )

    err_45 = rel_l2(
        outs[4],
        outs[5],
    )

    # Compare the native K=5 result against the vectorized
    # framework reconstruction for correctness.
    v5 = vectorized_kron(
        reps[5]
    )

    sync()

    err_native_vec = rel_l2(
        outs[5],
        v5,
    )

    result = {
        "S": S,

        "k3_us": best[3]["us"],
        "k4_us": best[4]["us"],
        "k5_us": best[5]["us"],

        "k3_block": best[3]["block"],
        "k4_block": best[4]["block"],
        "k5_block": best[5]["block"],

        "k5_speedup_vs_k3":
            best[3]["us"]
            / best[5]["us"],

        "k5_speedup_vs_k4":
            best[4]["us"]
            / best[5]["us"],

        "k3_vs_k5_rel_l2":
            err_35,

        "k4_vs_k5_rel_l2":
            err_45,

        "native_vs_vectorized_rel_l2":
            err_native_vec,
    }

    if time_framework:
        # Vectorized baseline.
        vec_us = median_us(
            lambda: vectorized_kron(
                reps[5]
            ),
            warmup=3,
            iters=10,
            repeats=5,
        )

        # repeated torch.kron baseline.
        #
        # Keep iteration count small because this path launches
        # many framework operations.
        kron_iters = (
            5 if S <= 64
            else 3
        )

        kron_us = median_us(
            lambda: repeated_torch_kron(
                reps[5]
            ),
            warmup=1,
            iters=kron_iters,
            repeats=5,
        )

        result.update(
            {
                "vectorized_k5_us":
                    vec_us,

                "torch_kron_k5_us":
                    kron_us,

                "native_k5_speedup_vs_vectorized":
                    vec_us
                    / best[5]["us"],

                "native_k5_speedup_vs_torch_kron":
                    kron_us
                    / best[5]["us"],
            }
        )

    del reps, outs, v5
    torch.cuda.empty_cache()

    return result


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--s-values",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            96,
            128,
            192,
            256,
        ],
    )

    p.add_argument(
        "--framework-s-values",
        type=int,
        nargs="+",
        default=[
            8,
            32,
            64,
            128,
            256,
        ],
        help=(
            "S values for the slower framework "
            "baseline measurements."
        ),
    )

    args = p.parse_args()

    framework_set = set(
        args.framework_s_values
    )

    print()
    print("=" * 124)
    print("EXACT-W TINY-KRONECKER-FACTOR EXPERIMENT")
    print("=" * 124)
    print()
    print("Full operator: 16 x 768")
    print()
    print(
        "K=3 factor shapes: "
        "(4x48), (2x4), (2x4) "
        "-> max factor = 192 values"
    )
    print(
        "K=4 factor shapes: "
        "(2x12), (2x4), (2x4), (2x4) "
        "-> max factor = 24 values"
    )
    print(
        "K=5 factor shapes: "
        "(1x3), (2x4), (2x4), (2x4), (2x4) "
        "-> max factor = 8 values"
    )
    print()
    print(
        "K=3 and K=4 are obtained by exact Kronecker "
        "merges of the K=5 factors."
    )

    print()
    print(
        f"{'S':>4} | "
        f"{'K3':>10} "
        f"{'K4':>10} "
        f"{'K5':>10} | "
        f"{'K5/K3':>8} "
        f"{'K5/K4':>8} | "
        f"{'err 3/5':>10} "
        f"{'err 4/5':>10}"
    )
    print("-" * 96)

    results = []

    for S in args.s_values:
        r = run_case(
            S,
            time_framework=(
                S in framework_set
            ),
        )

        results.append(r)

        print(
            f"{S:4d} | "
            f"{r['k3_us']:8.2f}us "
            f"{r['k4_us']:8.2f}us "
            f"{r['k5_us']:8.2f}us | "
            f"{r['k5_speedup_vs_k3']:7.3f}x "
            f"{r['k5_speedup_vs_k4']:7.3f}x | "
            f"{r['k3_vs_k5_rel_l2']:10.2e} "
            f"{r['k4_vs_k5_rel_l2']:10.2e}"
        )

    print()
    print("=" * 124)
    print("FRAMEWORK EXPANSION VS K=5 NATIVE EXECUTION")
    print("=" * 124)

    print(
        f"{'S':>4} | "
        f"{'torch.kron':>12} "
        f"{'vectorized':>12} "
        f"{'native K5':>12} | "
        f"{'vs kron':>10} "
        f"{'vs vector':>10} "
        f"{'rel err':>10}"
    )
    print("-" * 92)

    for r in results:
        if (
            "torch_kron_k5_us"
            not in r
        ):
            continue

        print(
            f"{r['S']:4d} | "
            f"{r['torch_kron_k5_us']:10.2f}us "
            f"{r['vectorized_k5_us']:10.2f}us "
            f"{r['k5_us']:10.2f}us | "
            f"{r['native_k5_speedup_vs_torch_kron']:9.2f}x "
            f"{r['native_k5_speedup_vs_vectorized']:9.2f}x "
            f"{r['native_vs_vectorized_rel_l2']:10.2e}"
        )

    print()
    print("=" * 124)
    print("WHAT THIS EXPERIMENT PROVES")
    print("=" * 124)

    print(
        r"""
There are TWO separate claims.

1. Framework -> native execution

   torch.kron / vectorized framework construction may be much slower
   than directly evaluating the independent factor-entry products on
   the GPU.

2. Factor granularity, holding W fixed

   K=3, K=4, and K=5 are mathematically equivalent representations
   of the same operator.

   Therefore:

       K3 latency / K5 latency

   isolates whether making the constituent factors much smaller helps
   the GPU execution itself.

The hardware interpretation should be stated carefully:

   We do not literally pin one factor matrix to one physical CUDA core.

   Instead, tiny factors reduce each output calculation to a few
   scalar factor accesses and multiplications. Thousands of independent
   output calculations are exposed to CUDA threads; their scalar
   arithmetic is then scheduled across CUDA cores, with the tiny
   working set amenable to registers/cache.

That is the main hypothesis this paper should test.
"""
    )


if __name__ == "__main__":
    main()


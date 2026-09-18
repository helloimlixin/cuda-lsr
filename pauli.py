
# Script 1: Kronecker-native Pauli-sum Hamiltonian application
#
# Real application:
#   quantum many-body Hamiltonians written as sums of Pauli strings
#
#       H = sum_{s=1}^S c_s P_s,
#       P_s = P_{s,1} \otimes ... \otimes P_{s,K},
#       P_{s,k} in {I, X, Y, Z}.
#
# Every local Kronecker factor is exactly 2 x 2.
#
# The dense Hamiltonian has shape:
#
#       2^K x 2^K
#
# and therefore becomes impossible to materialize quickly.
#
# Instead, a Pauli string acts on a basis index using:
#
#   * a bit flip mask,
#   * a sign/parity mask,
#   * a phase from the number of Y factors.
#
# This script compares:
#
#   1) Kronecker-native Triton:
#      state-vector entries x Pauli-term partitions are parallelized
#      directly on the GPU; H is never constructed.
#
#   2) Chunked PyTorch non-materializing baseline:
#      vectorized gather/phase application over batches of Pauli strings.
#
#   3) Python/PyTorch one-term-at-a-time baseline where practical.
#
#   4) Dense complex Hamiltonian matvec where small enough to materialize.
#
# The Kronecker-native kernel evaluates the SAME Pauli-sum operator.
#
# Recommended environment:
#   PyTorch CUDA build
#   Triton 3.x
#
# Example:
#   python pauli_kron_native.py
#
# Useful larger sweep:
#   python pauli_kron_native.py \
#       --k-values 8 10 12 14 16 18 20 \
#       --s-values 64 256 1024 4096

import argparse
import math
import statistics
import time

import torch
import triton
import triton.language as tl


# ============================================================
# Environment
# ============================================================

assert torch.cuda.is_available(), "CUDA GPU required"

DEV = "cuda"

torch.manual_seed(17)
torch.cuda.manual_seed_all(17)

print("GPU   :", torch.cuda.get_device_name(0))
print("torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("Triton:", triton.__version__)


# ============================================================
# Timing helpers
# ============================================================

def sync():
    torch.cuda.synchronize()


def median_us(fn, warmup=5, iters=30, repeats=5):
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


def rel_l2_pair(ar, ai, br, bi, eps=1e-12):
    num = torch.sqrt(
        ((ar.float() - br.float()) ** 2).sum()
        + ((ai.float() - bi.float()) ** 2).sum()
    )

    den = torch.sqrt(
        (br.float() ** 2).sum()
        + (bi.float() ** 2).sum()
    ).clamp_min(eps)

    return (num / den).item()


def rel_l2_complex(a, b, eps=1e-12):
    return (
        (a - b).abs().norm()
        / b.abs().norm().clamp_min(eps)
    ).item()


# ============================================================
# Random Pauli Hamiltonian
#
# Local coding:
#   0 = I
#   1 = X
#   2 = Y
#   3 = Z
#
# For each Pauli string:
#
#   flip_mask: bit set for X or Y
#   sign_mask: bit set for Z or Y
#   ny_mod4  : number of Y factors modulo four
#
# For input computational-basis index c:
#
#   r = c XOR flip_mask
#
# and the phase is:
#
#   i^(#Y) * (-1)^popcount(c & sign_mask)
#
# ============================================================

def encode_pauli_strings(local_ops):
    """
    local_ops: CPU int64 tensor [S,K], values 0..3

    returns CPU int32:
        flip_mask [S]
        sign_mask [S]
        ny_mod4  [S]
    """
    S, K = local_ops.shape

    flip = torch.zeros(S, dtype=torch.int64)
    sign = torch.zeros(S, dtype=torch.int64)

    for k in range(K):
        op = local_ops[:, k]
        bit = 1 << k

        flip |= (
            ((op == 1) | (op == 2)).to(torch.int64)
            * bit
        )

        sign |= (
            ((op == 2) | (op == 3)).to(torch.int64)
            * bit
        )

    ny = (
        (local_ops == 2)
        .sum(dim=1)
        .remainder(4)
        .to(torch.int64)
    )

    # K <= 30 in this benchmark, so int32 masks are enough.
    return (
        flip.to(torch.int32),
        sign.to(torch.int32),
        ny.to(torch.int32),
    )


def make_problem(K, S, seed=17):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 10007 * K + S)

    local_ops = torch.randint(
        0,
        4,
        (S, K),
        generator=g,
        dtype=torch.int64,
    )

    flip, sign, ny = encode_pauli_strings(local_ops)

    coeff = (
        torch.randn(
            S,
            generator=g,
            dtype=torch.float32,
        )
        / math.sqrt(S)
    )

    N = 1 << K

    psi_r = torch.randn(
        N,
        generator=g,
        dtype=torch.float32,
    )

    psi_i = torch.randn(
        N,
        generator=g,
        dtype=torch.float32,
    )

    # Normalize state.
    norm = torch.sqrt(
        (psi_r ** 2).sum()
        + (psi_i ** 2).sum()
    )

    psi_r = (
        psi_r / norm
    ).cuda().contiguous()

    psi_i = (
        psi_i / norm
    ).cuda().contiguous()

    return {
        "K": K,
        "S": S,
        "N": N,
        "local_ops": local_ops,
        "flip": flip.cuda().contiguous(),
        "sign": sign.cuda().contiguous(),
        "ny": ny.cuda().contiguous(),
        "coeff": coeff.cuda().contiguous(),
        "psi_r": psi_r,
        "psi_i": psi_i,
    }


# ============================================================
# Triton helpers
# ============================================================

@triton.jit
def parity32(x):
    # We only use <= 30 bits.
    x = x ^ (x >> 16)
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    return x & 1


# ============================================================
# Kronecker-native partial kernel
#
# Grid:
#   x = state-vector output tile
#   y = Pauli-term partition
#
# Different output entries and different term partitions are
# independent GPU work.
#
# No dense Hamiltonian and no intermediate Kronecker matrix.
# ============================================================

@triton.jit
def pauli_partial_kernel(
    PSI_R,
    PSI_I,
    FLIP,
    SIGN,
    NY,
    COEFF,
    PART_R,
    PART_I,
    N_CONST: tl.constexpr,
    S_CONST: tl.constexpr,
    P_CONST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_block = tl.program_id(0)
    part = tl.program_id(1)

    r = (
        out_block * BLOCK
        + tl.arange(0, BLOCK)
    )

    rmask = r < N_CONST

    # Exact integer partition of [0,S).
    s0 = (
        S_CONST * part
    ) // P_CONST

    s1 = (
        S_CONST * (part + 1)
    ) // P_CONST

    acc_r = tl.zeros(
        (BLOCK,),
        dtype=tl.float32,
    )

    acc_i = tl.zeros(
        (BLOCK,),
        dtype=tl.float32,
    )

    for s in tl.range(
        s0,
        s1,
        loop_unroll_factor=1,
    ):
        flip = tl.load(
            FLIP + s
        )

        sign_mask = tl.load(
            SIGN + s
        )

        phase_code = (
            tl.load(NY + s)
            & 3
        )

        coeff = tl.load(
            COEFF + s
        )

        # Input basis index contributing to output r.
        c = r ^ flip

        xr = tl.load(
            PSI_R + c,
            mask=rmask,
            other=0.0,
        )

        xi = tl.load(
            PSI_I + c,
            mask=rmask,
            other=0.0,
        )

        parity = parity32(
            c & sign_mask
        )

        # Multiply by i^(#Y).
        #
        # code 0:  1 * (xr + i xi)
        # code 1:  i * (...) = -xi + i xr
        # code 2: -1 * (...) = -xr - i xi
        # code 3: -i * (...) =  xi - i xr
        pr = tl.where(
            phase_code == 0,
            xr,
            tl.where(
                phase_code == 1,
                -xi,
                tl.where(
                    phase_code == 2,
                    -xr,
                    xi,
                ),
            ),
        )

        pi = tl.where(
            phase_code == 0,
            xi,
            tl.where(
                phase_code == 1,
                xr,
                tl.where(
                    phase_code == 2,
                    -xi,
                    -xr,
                ),
            ),
        )

        # (-1)^parity.
        sgn = tl.where(
            parity == 0,
            1.0,
            -1.0,
        )

        scale = (
            coeff * sgn
        )

        acc_r += (
            scale * pr
        )

        acc_i += (
            scale * pi
        )

    base = (
        part * N_CONST
        + r
    )

    tl.store(
        PART_R + base,
        acc_r,
        mask=rmask,
    )

    tl.store(
        PART_I + base,
        acc_i,
        mask=rmask,
    )


# ============================================================
# Reduction over Pauli-term partitions
# ============================================================

@triton.jit
def pauli_reduce_kernel(
    PART_R,
    PART_I,
    OUT_R,
    OUT_I,
    N_CONST: tl.constexpr,
    P_CONST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = (
        tl.program_id(0) * BLOCK
        + tl.arange(0, BLOCK)
    )

    mask = r < N_CONST

    ar = tl.zeros(
        (BLOCK,),
        dtype=tl.float32,
    )

    ai = tl.zeros(
        (BLOCK,),
        dtype=tl.float32,
    )

    for p in tl.range(
        0,
        P_CONST,
        loop_unroll_factor=1,
    ):
        ar += tl.load(
            PART_R
            + p * N_CONST
            + r,
            mask=mask,
            other=0.0,
        )

        ai += tl.load(
            PART_I
            + p * N_CONST
            + r,
            mask=mask,
            other=0.0,
        )

    tl.store(
        OUT_R + r,
        ar,
        mask=mask,
    )

    tl.store(
        OUT_I + r,
        ai,
        mask=mask,
    )


def native_apply(
    prob,
    P,
    partial_r,
    partial_i,
    out_r,
    out_i,
    block=256,
):
    N = prob["N"]
    S = prob["S"]

    grid1 = (
        triton.cdiv(
            N,
            block,
        ),
        P,
    )

    pauli_partial_kernel[
        grid1
    ](
        prob["psi_r"],
        prob["psi_i"],
        prob["flip"],
        prob["sign"],
        prob["ny"],
        prob["coeff"],
        partial_r,
        partial_i,
        N_CONST=N,
        S_CONST=S,
        P_CONST=P,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )

    grid2 = (
        triton.cdiv(
            N,
            block,
        ),
    )

    pauli_reduce_kernel[
        grid2
    ](
        partial_r,
        partial_i,
        out_r,
        out_i,
        N_CONST=N,
        P_CONST=P,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )


# ============================================================
# PyTorch reference helpers
# ============================================================

def parity32_torch(x):
    x = x ^ (x >> 16)
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    return x & 1


@torch.no_grad()
def torch_chunked_apply(
    prob,
    chunk_s=16,
):
    N = prob["N"]
    S = prob["S"]

    idx = torch.arange(
        N,
        device=DEV,
        dtype=torch.int64,
    )

    psi = torch.complex(
        prob["psi_r"],
        prob["psi_i"],
    )

    out = torch.zeros(
        N,
        device=DEV,
        dtype=torch.complex64,
    )

    # i^n.
    phase_lut = torch.tensor(
        [
            1.0 + 0.0j,
            0.0 + 1.0j,
            -1.0 + 0.0j,
            0.0 - 1.0j,
        ],
        device=DEV,
        dtype=torch.complex64,
    )

    flip = prob["flip"].to(
        torch.int64
    )

    sign = prob["sign"].to(
        torch.int64
    )

    ny = prob["ny"].to(
        torch.int64
    )

    for s0 in range(
        0,
        S,
        chunk_s,
    ):
        s1 = min(
            S,
            s0 + chunk_s,
        )

        # [B,N]
        c = (
            idx.unsqueeze(0)
            ^ flip[s0:s1].unsqueeze(1)
        )

        vals = psi[c]

        par = parity32_torch(
            c
            & sign[
                s0:s1
            ].unsqueeze(1)
        )

        signed_phase = (
            (1.0 - 2.0 * par.float())
            .to(torch.complex64)
            * phase_lut[
                ny[s0:s1]
            ].unsqueeze(1)
        )

        out.add_(
            (
                prob["coeff"][
                    s0:s1
                ]
                .to(torch.complex64)
                .unsqueeze(1)
                * signed_phase
                * vals
            ).sum(dim=0)
        )

    return out


@torch.no_grad()
def torch_loop_apply(
    prob,
):
    N = prob["N"]
    S = prob["S"]

    idx = torch.arange(
        N,
        device=DEV,
        dtype=torch.int64,
    )

    psi = torch.complex(
        prob["psi_r"],
        prob["psi_i"],
    )

    out = torch.zeros(
        N,
        device=DEV,
        dtype=torch.complex64,
    )

    phase_lut = torch.tensor(
        [
            1.0 + 0.0j,
            0.0 + 1.0j,
            -1.0 + 0.0j,
            0.0 - 1.0j,
        ],
        device=DEV,
        dtype=torch.complex64,
    )

    flip = prob["flip"].to(
        torch.int64
    )

    sign = prob["sign"].to(
        torch.int64
    )

    ny = prob["ny"].to(
        torch.int64
    )

    for s in range(S):
        c = idx ^ flip[s]

        par = parity32_torch(
            c & sign[s]
        )

        phase = (
            (1.0 - 2.0 * par.float())
            .to(torch.complex64)
            * phase_lut[ny[s]]
        )

        out.add_(
            prob["coeff"][s]
            .to(torch.complex64)
            * phase
            * psi[c]
        )

    return out


# ============================================================
# Dense Hamiltonian, only for small N.
#
# Build cost is NOT included in matvec timing.
# ============================================================

@torch.no_grad()
def build_dense_hamiltonian(
    prob,
):
    N = prob["N"]
    S = prob["S"]

    idx = torch.arange(
        N,
        device=DEV,
        dtype=torch.int64,
    )

    H = torch.zeros(
        N,
        N,
        device=DEV,
        dtype=torch.complex64,
    )

    phase_lut = torch.tensor(
        [
            1.0 + 0.0j,
            0.0 + 1.0j,
            -1.0 + 0.0j,
            0.0 - 1.0j,
        ],
        device=DEV,
        dtype=torch.complex64,
    )

    flip = prob["flip"].to(
        torch.int64
    )

    sign = prob["sign"].to(
        torch.int64
    )

    ny = prob["ny"].to(
        torch.int64
    )

    for s in range(S):
        c = idx
        r = (
            c ^ flip[s]
        )

        par = parity32_torch(
            c & sign[s]
        )

        vals = (
            prob["coeff"][s]
            .to(torch.complex64)
            * (
                1.0
                - 2.0
                * par.float()
            ).to(
                torch.complex64
            )
            * phase_lut[ny[s]]
        )

        H.index_put_(
            (r, c),
            vals,
            accumulate=True,
        )

    return H


# ============================================================
# Native partition tuning
# ============================================================

def candidate_partitions(
    S,
    N,
    max_partial_mb,
):
    vals = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    ]

    out = []

    for P in vals:
        if P > S:
            continue

        # Two FP32 partial arrays.
        mb = (
            2
            * P
            * N
            * 4
            / 1024**2
        )

        if mb <= max_partial_mb:
            out.append(P)

    return out


@torch.no_grad()
def tune_native(
    prob,
    max_partial_mb=512,
):
    N = prob["N"]
    S = prob["S"]

    rows = []

    for P in candidate_partitions(
        S,
        N,
        max_partial_mb,
    ):
        partial_r = torch.empty(
            P,
            N,
            device=DEV,
            dtype=torch.float32,
        )

        partial_i = torch.empty_like(
            partial_r
        )

        out_r = torch.empty(
            N,
            device=DEV,
            dtype=torch.float32,
        )

        out_i = torch.empty_like(
            out_r
        )

        native_apply(
            prob,
            P,
            partial_r,
            partial_i,
            out_r,
            out_i,
        )

        sync()

        work = N * S

        # Large cases need fewer iterations.
        if work <= 2_000_000:
            iters = 100
        elif work <= 20_000_000:
            iters = 30
        else:
            iters = 10

        us = median_us(
            lambda: native_apply(
                prob,
                P,
                partial_r,
                partial_i,
                out_r,
                out_i,
            ),
            warmup=3,
            iters=iters,
            repeats=5,
        )

        rows.append(
            {
                "P": P,
                "us": us,
                "partial_r": partial_r,
                "partial_i": partial_i,
                "out_r": out_r,
                "out_i": out_i,
                "partial_mb":
                    2
                    * P
                    * N
                    * 4
                    / 1024**2,
            }
        )

    if not rows:
        raise RuntimeError(
            "No legal P under partial-memory cap"
        )

    best = min(
        rows,
        key=lambda r: r["us"],
    )

    return best, rows


# ============================================================
# One benchmark case
# ============================================================

@torch.no_grad()
def run_case(
    K,
    S,
    torch_chunk,
    max_partial_mb,
    run_loop,
    run_dense,
):
    prob = make_problem(
        K,
        S,
    )

    N = prob["N"]

    dense_bytes = (
        N * N * 8
    )

    dense_gib = (
        dense_bytes
        / 1024**3
    )

    # ----------------------------------------
    # Native tune.
    # ----------------------------------------
    best, native_rows = tune_native(
        prob,
        max_partial_mb=max_partial_mb,
    )

    native_apply(
        prob,
        best["P"],
        best["partial_r"],
        best["partial_i"],
        best["out_r"],
        best["out_i"],
    )

    sync()

    # ----------------------------------------
    # Chunked PyTorch reference + timing.
    # ----------------------------------------
    ref = torch_chunked_apply(
        prob,
        chunk_s=torch_chunk,
    )

    sync()

    err_native = rel_l2_pair(
        best["out_r"],
        best["out_i"],
        ref.real,
        ref.imag,
    )

    work = N * S

    if work <= 2_000_000:
        torch_iters = 10
    elif work <= 20_000_000:
        torch_iters = 3
    else:
        torch_iters = 1

    torch_us = median_us(
        lambda: torch_chunked_apply(
            prob,
            chunk_s=torch_chunk,
        ),
        warmup=1,
        iters=torch_iters,
        repeats=3,
    )

    result = {
        "K": K,
        "S": S,
        "N": N,
        "dense_gib": dense_gib,
        "native_us": best["us"],
        "native_P": best["P"],
        "native_partial_mb":
            best["partial_mb"],
        "torch_chunk_us":
            torch_us,
        "speedup_vs_chunked":
            torch_us / best["us"],
        "native_rel_l2":
            err_native,
        "contrib_per_us":
            work / best["us"],
        "native_rows":
            [
                {
                    "P": r["P"],
                    "us": r["us"],
                    "partial_mb":
                        r["partial_mb"],
                }
                for r in native_rows
            ],
    }

    # ----------------------------------------
    # One-term-at-a-time PyTorch baseline.
    # Only where requested/practical.
    # ----------------------------------------
    if run_loop:
        loop_out = torch_loop_apply(
            prob
        )
        sync()

        loop_err = rel_l2_complex(
            loop_out,
            ref,
        )

        loop_us = median_us(
            lambda: torch_loop_apply(
                prob
            ),
            warmup=1,
            iters=1,
            repeats=3,
        )

        result.update(
            {
                "torch_loop_us":
                    loop_us,
                "speedup_vs_loop":
                    loop_us
                    / best["us"],
                "loop_rel_l2":
                    loop_err,
            }
        )

    # ----------------------------------------
    # Dense matvec baseline.
    # Build cost excluded.
    # ----------------------------------------
    if run_dense:
        H = build_dense_hamiltonian(
            prob
        )

        psi = torch.complex(
            prob["psi_r"],
            prob["psi_i"],
        )

        dense_ref = (
            H @ psi
        )

        sync()

        dense_err = rel_l2_complex(
            dense_ref,
            ref,
        )

        dense_us = median_us(
            lambda: H @ psi,
            warmup=5,
            iters=30,
            repeats=5,
        )

        result.update(
            {
                "dense_us":
                    dense_us,
                "dense_speedup_native":
                    dense_us
                    / best["us"],
                "dense_rel_l2":
                    dense_err,
            }
        )

        del H, psi, dense_ref

    del prob, ref
    torch.cuda.empty_cache()

    return result


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[
            8,
            12,
            16,
            20,
        ],
    )

    p.add_argument(
        "--s-values",
        type=int,
        nargs="+",
        default=[
            64,
            256,
            1024,
        ],
    )

    p.add_argument(
        "--torch-chunk",
        type=int,
        default=16,
    )

    p.add_argument(
        "--max-partial-mb",
        type=float,
        default=512.0,
    )

    p.add_argument(
        "--max-work",
        type=int,
        default=150_000_000,
        help=(
            "Skip K,S cases with "
            "(2^K)*S above this threshold."
        ),
    )

    p.add_argument(
        "--dense-max-k",
        type=int,
        default=10,
    )

    p.add_argument(
        "--dense-max-s",
        type=int,
        default=256,
    )

    p.add_argument(
        "--loop-max-work",
        type=int,
        default=2_000_000,
    )

    args = p.parse_args()

    print()
    print("=" * 132)
    print(
        "KRONECKER-NATIVE PAULI HAMILTONIAN APPLICATION"
    )
    print("=" * 132)

    print()
    print(
        "Each local factor is one of I, X, Y, Z: "
        "a 2 x 2 Kronecker factor."
    )
    print(
        "Native path applies the Pauli sum directly "
        "without materializing H."
    )

    results = []

    print()
    print(
        f"{'K':>3} "
        f"{'S':>6} "
        f"{'N=2^K':>10} | "
        f"{'dense GiB':>10} | "
        f"{'torch vec':>11} "
        f"{'native':>11} "
        f"{'P':>4} "
        f"{'speedup':>9} | "
        f"{'rel err':>10}"
    )

    print("-" * 104)

    for K in args.k_values:
        if K > 30:
            raise ValueError(
                "This script uses int32 bit masks; "
                "keep K <= 30."
            )

        for S in args.s_values:
            N = 1 << K
            work = N * S

            if work > args.max_work:
                print(
                    f"{K:3d} "
                    f"{S:6d} "
                    f"{N:10d} | "
                    f"{(N*N*8/1024**3):10.2f} | "
                    f"{'SKIP':>11} "
                    f"{'SKIP':>11} "
                    f"{'-':>4} "
                    f"{'-':>9} | "
                    f"{'work cap':>10}"
                )
                continue

            run_loop = (
                work
                <= args.loop_max_work
            )

            run_dense = (
                K
                <= args.dense_max_k
                and S
                <= args.dense_max_s
            )

            r = run_case(
                K=K,
                S=S,
                torch_chunk=args.torch_chunk,
                max_partial_mb=args.max_partial_mb,
                run_loop=run_loop,
                run_dense=run_dense,
            )

            results.append(r)

            print(
                f"{K:3d} "
                f"{S:6d} "
                f"{r['N']:10d} | "
                f"{r['dense_gib']:10.2f} | "
                f"{r['torch_chunk_us']:9.2f}us "
                f"{r['native_us']:9.2f}us "
                f"{r['native_P']:4d} "
                f"{r['speedup_vs_chunked']:8.2f}x | "
                f"{r['native_rel_l2']:10.2e}"
            )

    print()
    print("=" * 132)
    print("DETAILS")
    print("=" * 132)

    for r in results:
        print()
        print(
            f"K={r['K']}, S={r['S']}, N={r['N']}"
        )

        print(
            f"  estimated dense-H storage : "
            f"{r['dense_gib']:.3f} GiB"
        )

        print(
            f"  best native P             : "
            f"{r['native_P']}"
        )

        print(
            f"  native partial storage    : "
            f"{r['native_partial_mb']:.2f} MiB"
        )

        print(
            f"  native latency            : "
            f"{r['native_us']:.2f} us"
        )

        print(
            f"  chunked PyTorch latency   : "
            f"{r['torch_chunk_us']:.2f} us"
        )

        print(
            f"  speedup vs chunked        : "
            f"{r['speedup_vs_chunked']:.2f}x"
        )

        print(
            f"  native relative error     : "
            f"{r['native_rel_l2']:.3e}"
        )

        print(
            "  native partition sweep:"
        )

        for q in r["native_rows"]:
            print(
                f"    P={q['P']:2d}: "
                f"{q['us']:9.2f} us, "
                f"partials={q['partial_mb']:.2f} MiB"
            )

        if "torch_loop_us" in r:
            print(
                f"  term-loop PyTorch         : "
                f"{r['torch_loop_us']:.2f} us"
            )

            print(
                f"  speedup vs term-loop      : "
                f"{r['speedup_vs_loop']:.2f}x"
            )

        if "dense_us" in r:
            print(
                f"  dense H @ psi             : "
                f"{r['dense_us']:.2f} us"
            )

            print(
                f"  dense/native ratio        : "
                f"{r['dense_speedup_native']:.2f}x"
            )

            print(
                f"  dense correctness error   : "
                f"{r['dense_rel_l2']:.3e}"
            )

    print()
    print("=" * 132)
    print("PAPER INTERPRETATION")
    print("=" * 132)

    print(
        r"""
This is a real tensor-product application, not a synthetic Kronecker
construction benchmark.

A K-qubit Hamiltonian would require a dense complex matrix with

    2^K x 2^K

entries.  Pauli decomposition instead stores S products of tiny 2x2
local factors.

The Kronecker-native implementation never constructs those 2x2 tensor
products.  Their structure is compiled into bit permutations, parity
tests, and phases, and the GPU executes the resulting work jointly
across:

    * state-vector output entries, and
    * independent Pauli/Kronecker terms.

The relevant systems claim is:

    When the local Kronecker factors are extremely small, materializing
    or dispatching them as matrices is unnecessary. Their algebra can be
    mapped directly to fine-grained GPU work, preserving the compact
    representation while exposing enough parallelism for the device.

The dense-storage column is also important. It shows the regime where
a conventional dense Hamiltonian ceases to be a meaningful baseline,
while the Kronecker-native operator remains linear in the state-vector
size times the number of Pauli terms.
"""
    )


if __name__ == "__main__":
    main()


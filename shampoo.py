
# Script 2: Distributed-Shampoo-style batched CoupledNewton inverse roots
#
# Application:
#   Shampoo / Distributed Shampoo periodically computes inverse roots of
#   many small-to-medium preconditioner matrices.
#
# Main systems question:
#   Can same-shaped inverse-root problems be evaluated concurrently on
#   the GPU instead of dispatching one matrix function at a time?
#
# This script compares:
#
#   1) PER-FACTOR COUPLED NEWTON
#      Production-style Python loop: every factor runs the same recurrence
#      independently.
#
#   2) BATCHED COUPLED NEWTON
#      Same mathematical recurrence, but all same-shaped factors are kept
#      in [B,n,n] tensors and advanced with batched GEMMs. Each matrix has
#      an independent convergence mask.
#
#   3) HYBRID
#      Run batched first, then recompute only the difficult tail with the
#      per-factor production path. This mirrors the production-safe design
#      used in our earlier GPT-2 experiments.
#
# Representative GPT-2-like buckets:
#
#   n=128, root=4, B=2
#   n=256, root=2, B=86
#   n=256, root=4, B=128
#   n=512, root=2, B=128
#   n=512, root=4, B=128
#
# The matrices are synthetic SPD factors with representative dimensions.
# The benchmark is algorithm-preserving: baseline and batched paths use
# the same CoupledNewton recurrence.
#
# Optional:
#   --quick
#       smaller bucket counts for a fast smoke test.
#
# Run:
#   python shampoo_coupled_newton_batched.py
#
# Notes:
#   * FP32 is intentional because production optimizer roots are commonly
#     stored/evaluated in FP32.
#   * This is the matrix-function application benchmark. It is not yet the
#     full GPT-2/WikiText training experiment.

import argparse
import math
import statistics
import time

import torch


# ============================================================
# Environment
# ============================================================

assert torch.cuda.is_available(), "CUDA GPU required"

DEV = "cuda"
DT = torch.float32

torch.manual_seed(17)
torch.cuda.manual_seed_all(17)

torch.set_float32_matmul_precision("highest")
try:
    torch.backends.cuda.matmul.allow_tf32 = False
except Exception:
    pass

print("GPU   :", torch.cuda.get_device_name(0))
print("torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("TF32  :", torch.backends.cuda.matmul.allow_tf32)


# ============================================================
# Utilities
# ============================================================

def sync():
    torch.cuda.synchronize()


def median_ms(fn, warmup=2, iters=3, repeats=5):
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
        vals.append(a.elapsed_time(b) / iters)

    return statistics.median(vals)


def batch_rel_l2(a, b, eps=1e-12):
    """
    Per-matrix relative Frobenius error.
    """
    num = (
        (a.float() - b.float())
        .flatten(1)
        .norm(dim=1)
    )

    den = (
        b.float()
        .flatten(1)
        .norm(dim=1)
        .clamp_min(eps)
    )

    return num / den


def residual_error(A, X, root):
    """
    Diagnostic:
        || X^root A - I ||_inf
    evaluated per matrix.

    This is not the production stopping criterion; it is only an
    end-of-run quality diagnostic.
    """
    B, n, _ = A.shape

    if root == 2:
        Xp = X @ X
    elif root == 4:
        X2 = X @ X
        Xp = X2 @ X2
    else:
        raise ValueError(root)

    M = Xp @ A

    I = torch.eye(
        n,
        device=A.device,
        dtype=A.dtype,
    ).expand(B, n, n)

    return (
        (M - I)
        .abs()
        .flatten(1)
        .amax(dim=1)
    )


# ============================================================
# Synthetic SPD factor generation
#
# We intentionally vary the ridge level across factors so a small tail
# is harder than the rest. This is useful for exercising the fallback.
# ============================================================

@torch.no_grad()
def make_spd_batch(
    B,
    n,
    seed,
    hard_frac=0.15,
):
    g = torch.Generator(
        device=DEV
    )
    g.manual_seed(seed)

    # Random Gram matrices.
    X = torch.randn(
        B,
        n,
        n,
        device=DEV,
        dtype=DT,
        generator=g,
    )

    A = (
        X @ X.transpose(-1, -2)
    ) / float(n)

    # Normalize each factor to O(1) Frobenius scale.
    fro = (
        A.flatten(1)
        .norm(dim=1)
        .clamp_min(1e-12)
    )

    A = (
        A
        / fro[:, None, None]
        * math.sqrt(n)
    )

    # Easy majority: larger ridge.
    # Hard tail: much smaller ridge.
    num_hard = max(
        1,
        int(round(B * hard_frac)),
    )

    ridge = torch.full(
        (B,),
        1e-2,
        device=DEV,
        dtype=DT,
    )

    ridge[-num_hard:] = 1e-5

    I = torch.eye(
        n,
        device=DEV,
        dtype=DT,
    )

    A = (
        A
        + ridge[:, None, None] * I
    )

    return A.contiguous()


# ============================================================
# Production-style CoupledNewton recurrence
#
# Same recurrence used in the earlier Meta/Distributed-Shampoo study:
#
#   alpha = -1/root
#   z = (root+1)/(2 ||A||_F)
#   X = z^(-alpha) I
#   M = z A
#
#   M_p = alpha M + (1-alpha) I
#   X   = X @ M_p
#   M   = M_p^root @ M
#
# Stop on:
#   || M - I ||_inf <= tolerance
# ============================================================

def matrix_power_small(M, root):
    if root == 2:
        return M @ M

    if root == 4:
        M2 = M @ M
        return M2 @ M2

    raise ValueError(
        "This benchmark supports root=2 or root=4."
    )


@torch.no_grad()
def coupled_newton_one(
    A,
    root,
    epsilon=0.0,
    tolerance=1e-6,
    max_iterations=100,
):
    n = A.shape[0]

    I = torch.eye(
        n,
        device=A.device,
        dtype=A.dtype,
    )

    if epsilon != 0.0:
        Ar = A + epsilon * I
    else:
        Ar = A

    alpha = -1.0 / float(root)

    A_nrm = torch.linalg.norm(
        Ar
    )

    z = (
        (root + 1.0)
        / (2.0 * A_nrm)
    )

    # -alpha = 1/root
    X = (
        z ** (-alpha)
    ) * I

    M = z * Ar

    err = (
        M - I
    ).abs().max()

    it = 0

    while (
        float(err) > tolerance
        and it < max_iterations
    ):
        Mp = (
            alpha * M
            + (1.0 - alpha) * I
        )

        X = X @ Mp

        M = (
            matrix_power_small(
                Mp,
                root,
            )
            @ M
        )

        err = (
            M - I
        ).abs().max()

        it += 1

    return X, it, float(err)


@torch.no_grad()
def per_factor_coupled_newton(
    A,
    root,
    epsilon=0.0,
    tolerance=1e-6,
    max_iterations=100,
):
    """
    Production-style execution:
        loop over factors and run CoupledNewton independently.
    """
    B = A.shape[0]

    outs = []
    iters = []
    errors = []

    for b in range(B):
        X, it, err = coupled_newton_one(
            A[b],
            root=root,
            epsilon=epsilon,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

        outs.append(X)
        iters.append(it)
        errors.append(err)

    return (
        torch.stack(
            outs,
            dim=0,
        ),
        torch.tensor(
            iters,
            device=DEV,
            dtype=torch.int32,
        ),
        torch.tensor(
            errors,
            device=DEV,
            dtype=torch.float32,
        ),
    )


# ============================================================
# Batched CoupledNewton
#
# Same recurrence, but all matrices are evaluated together.
#
# Important:
#   each matrix retains an independent convergence mask.
# ============================================================

@torch.no_grad()
def batched_coupled_newton(
    A,
    root,
    epsilon=0.0,
    tolerance=1e-6,
    max_iterations=100,
):
    B, n, _ = A.shape

    I = torch.eye(
        n,
        device=A.device,
        dtype=A.dtype,
    ).expand(B, n, n)

    if epsilon != 0.0:
        Ar = (
            A
            + epsilon * I
        )
    else:
        Ar = A

    alpha = -1.0 / float(root)

    A_nrm = (
        Ar.flatten(1)
        .norm(dim=1)
    )

    z = (
        (root + 1.0)
        / (2.0 * A_nrm)
    )

    X = (
        z.pow(-alpha)
        [:, None, None]
        * I
    ).contiguous()

    M = (
        z[:, None, None]
        * Ar
    ).contiguous()

    err = (
        (M - I)
        .abs()
        .flatten(1)
        .amax(dim=1)
    )

    active = (
        err > tolerance
    )

    iters = torch.zeros(
        B,
        device=DEV,
        dtype=torch.int32,
    )

    for _ in range(
        max_iterations
    ):
        if not bool(
            active.any()
        ):
            break

        Mp = (
            alpha * M
            + (1.0 - alpha) * I
        )

        X_new = X @ Mp

        M_new = (
            matrix_power_small(
                Mp,
                root,
            )
            @ M
        )

        mask3 = active[
            :, None, None
        ]

        X = torch.where(
            mask3,
            X_new,
            X,
        )

        M = torch.where(
            mask3,
            M_new,
            M,
        )

        iters = (
            iters
            + active.to(
                torch.int32
            )
        )

        err = (
            (M - I)
            .abs()
            .flatten(1)
            .amax(dim=1)
        )

        active = (
            err > tolerance
        )

    return X, iters, err


# ============================================================
# Hybrid production-safe path
#
# First run batched.  Any factor whose iteration count exceeds the
# conservative threshold is recomputed using the per-factor path.
#
# Thresholds come from the earlier GPT-2 factor diagnostics:
#
#   root=2 -> fallback if iterations > 15
#   root=4 -> fallback if iterations > 16
# ============================================================

@torch.no_grad()
def hybrid_coupled_newton(
    A,
    root,
    epsilon=0.0,
    tolerance=1e-6,
    max_iterations=100,
):
    Xb, iters, berr = (
        batched_coupled_newton(
            A,
            root=root,
            epsilon=epsilon,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    )

    threshold = (
        15 if root == 2
        else 16
    )

    hard = (
        iters > threshold
    )

    hard_idx = hard.nonzero(
        as_tuple=False
    ).flatten()

    X = Xb.clone()

    for idx in hard_idx.tolist():
        Xi, _, _ = coupled_newton_one(
            A[idx],
            root=root,
            epsilon=epsilon,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

        X[idx].copy_(Xi)

    return (
        X,
        iters,
        hard,
        berr,
    )


# ============================================================
# Benchmark one bucket
# ============================================================

@torch.no_grad()
def run_bucket(
    n,
    root,
    B,
    seed,
    tolerance,
    max_iterations,
):
    print()
    print(
        f"bucket n={n}, root={root}, B={B}"
    )

    A = make_spd_batch(
        B=B,
        n=n,
        seed=seed,
    )

    # --------------------------------------------------------
    # Correctness reference: production per-factor path.
    # --------------------------------------------------------
    ref, ref_iters, ref_stop = (
        per_factor_coupled_newton(
            A,
            root=root,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    )

    sync()

    # --------------------------------------------------------
    # Batched output.
    # --------------------------------------------------------
    bat, bat_iters, bat_stop = (
        batched_coupled_newton(
            A,
            root=root,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    )

    sync()

    rel_bat = batch_rel_l2(
        bat,
        ref,
    )

    # --------------------------------------------------------
    # Hybrid output.
    # --------------------------------------------------------
    hyb, hyb_iters, hard, _ = (
        hybrid_coupled_newton(
            A,
            root=root,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    )

    sync()

    rel_hyb = batch_rel_l2(
        hyb,
        ref,
    )

    # --------------------------------------------------------
    # Timings.
    # --------------------------------------------------------
    #
    # Baseline is slow; one timed execution per repetition is enough.
    base_ms = median_ms(
        lambda: per_factor_coupled_newton(
            A,
            root=root,
            tolerance=tolerance,
            max_iterations=max_iterations,
        ),
        warmup=1,
        iters=1,
        repeats=3,
    )

    bat_ms = median_ms(
        lambda: batched_coupled_newton(
            A,
            root=root,
            tolerance=tolerance,
            max_iterations=max_iterations,
        ),
        warmup=2,
        iters=3,
        repeats=5,
    )

    hyb_ms = median_ms(
        lambda: hybrid_coupled_newton(
            A,
            root=root,
            tolerance=tolerance,
            max_iterations=max_iterations,
        ),
        warmup=1,
        iters=2,
        repeats=3,
    )

    # --------------------------------------------------------
    # Diagnostics.
    # --------------------------------------------------------
    threshold = (
        15 if root == 2
        else 16
    )

    fallback_pct = (
        hard.float()
        .mean()
        .item()
        * 100.0
    )

    resid_ref = residual_error(
        A,
        ref,
        root,
    )

    resid_hyb = residual_error(
        A,
        hyb,
        root,
    )

    row = {
        "n": n,
        "root": root,
        "B": B,

        "base_ms":
            base_ms,

        "bat_ms":
            bat_ms,

        "hyb_ms":
            hyb_ms,

        "bat_speedup":
            base_ms / bat_ms,

        "hyb_speedup":
            base_ms / hyb_ms,

        "fallback_pct":
            fallback_pct,

        "threshold":
            threshold,

        "bat_max_rel":
            rel_bat.max().item(),

        "bat_med_rel":
            rel_bat.median().item(),

        "hyb_max_rel":
            rel_hyb.max().item(),

        "hyb_med_rel":
            rel_hyb.median().item(),

        "ref_iter_median":
            ref_iters.float()
            .median()
            .item(),

        "ref_iter_max":
            ref_iters.max()
            .item(),

        "bat_iter_median":
            bat_iters.float()
            .median()
            .item(),

        "bat_iter_max":
            bat_iters.max()
            .item(),

        "ref_resid_max":
            resid_ref.max()
            .item(),

        "hyb_resid_max":
            resid_hyb.max()
            .item(),
    }

    return row


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--quick",
        action="store_true",
    )

    p.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
    )

    p.add_argument(
        "--max-iterations",
        type=int,
        default=100,
    )

    args = p.parse_args()

    if args.quick:
        buckets = [
            (128, 4, 2),
            (256, 2, 16),
            (256, 4, 32),
            (512, 2, 32),
            (512, 4, 32),
        ]
    else:
        buckets = [
            (128, 4, 2),
            (256, 2, 86),
            (256, 4, 128),
            (512, 2, 128),
            (512, 4, 128),
        ]

    print()
    print("=" * 132)
    print(
        "DISTRIBUTED-SHAMPOO-STYLE "
        "BATCHED COUPLEDNEWTON"
    )
    print("=" * 132)

    print()
    print(
        "Baseline and optimized paths use the same "
        "CoupledNewton recurrence."
    )
    print(
        "Only execution granularity changes: "
        "factor-by-factor vs same-shape batched GEMMs."
    )

    results = []

    for j, (
        n,
        root,
        B,
    ) in enumerate(buckets):
        row = run_bucket(
            n=n,
            root=root,
            B=B,
            seed=17 + 100 * j,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
        )

        results.append(row)

    print()
    print("=" * 132)
    print("SUMMARY")
    print("=" * 132)

    print(
        f"{'shape/root':>12} "
        f"{'B':>5} | "
        f"{'per-factor':>11} "
        f"{'batched':>11} "
        f"{'hybrid':>11} | "
        f"{'batch spd':>9} "
        f"{'hyb spd':>9} | "
        f"{'fallback':>9} "
        f"{'hyb max err':>12}"
    )

    print("-" * 116)

    for r in results:
        print(
            f"{r['n']:4d}/r{r['root']:<5d} "
            f"{r['B']:5d} | "
            f"{r['base_ms']:9.2f}ms "
            f"{r['bat_ms']:9.2f}ms "
            f"{r['hyb_ms']:9.2f}ms | "
            f"{r['bat_speedup']:8.2f}x "
            f"{r['hyb_speedup']:8.2f}x | "
            f"{r['fallback_pct']:8.2f}% "
            f"{r['hyb_max_rel']:12.2e}"
        )

    print()
    print("=" * 132)
    print("DETAILS")
    print("=" * 132)

    for r in results:
        print()
        print(
            f"n={r['n']}, root={r['root']}, B={r['B']}"
        )

        print(
            f"  per-factor latency      : "
            f"{r['base_ms']:.3f} ms"
        )

        print(
            f"  batched latency         : "
            f"{r['bat_ms']:.3f} ms"
        )

        print(
            f"  hybrid latency          : "
            f"{r['hyb_ms']:.3f} ms"
        )

        print(
            f"  batched speedup         : "
            f"{r['bat_speedup']:.3f}x"
        )

        print(
            f"  hybrid speedup          : "
            f"{r['hyb_speedup']:.3f}x"
        )

        print(
            f"  fallback threshold      : "
            f"{r['threshold']} iterations"
        )

        print(
            f"  fallback fraction       : "
            f"{r['fallback_pct']:.2f}%"
        )

        print(
            f"  reference iter median   : "
            f"{r['ref_iter_median']:.1f}"
        )

        print(
            f"  reference iter max      : "
            f"{r['ref_iter_max']}"
        )

        print(
            f"  batched iter median     : "
            f"{r['bat_iter_median']:.1f}"
        )

        print(
            f"  batched iter max        : "
            f"{r['bat_iter_max']}"
        )

        print(
            f"  batched max rel error   : "
            f"{r['bat_max_rel']:.3e}"
        )

        print(
            f"  hybrid max rel error    : "
            f"{r['hyb_max_rel']:.3e}"
        )

        print(
            f"  ref residual max        : "
            f"{r['ref_resid_max']:.3e}"
        )

        print(
            f"  hybrid residual max     : "
            f"{r['hyb_resid_max']:.3e}"
        )

    print()
    print("=" * 132)
    print("PAPER INTERPRETATION")
    print("=" * 132)

    print(
        r"""
This application exercises a different form of Kronecker-native
parallelism than the Pauli experiment.

Shampoo represents optimizer state using many independent factor
matrices.  The algorithmic structure is already compact, but a
factor-by-factor implementation still produces a long sequence of
small or medium matrix operations.

The batched path preserves the exact CoupledNewton recurrence while
grouping same-shaped factor operations into batched GEMMs.  The hybrid
path then recomputes only the numerically difficult tail with the
production per-factor routine.

The systems claim is therefore:

    Kronecker-factored optimizers expose concurrency across factor
    matrices.  Keeping that structure visible at execution time allows
    the GPU to process many inverse-root problems concurrently instead
    of serializing them as independent matrix-function calls.

For the paper, synthetic-SPD results from this script should be treated
as a controlled matrix-function benchmark.  The stronger application
claim remains the end-to-end GPT-2/WikiText-103 result, where the same
idea reduced periodic optimizer latency and total training wall time.
"""
    )


if __name__ == "__main__":
    main()


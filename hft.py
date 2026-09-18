
# HFT Kronecker Opportunity Screening Suite
#
# Workloads:
#   A) Cross-sectional signal whitening / short-horizon risk normalization
#   B) Multi-term cross-asset / cross-feature propagation
#   C) Event-driven multivariate Hawkes update (dense vs Kronecker-native Triton)
#
# Recommended: CUDA GPU, recent PyTorch + Triton.
#
# These are systems microbenchmarks on synthetic data. They are NOT trading
# strategies and do not make claims about predictive alpha.

import math
import statistics
import torch

assert torch.cuda.is_available(), "CUDA GPU required"

DEV = "cuda"
DT = torch.float32

print("GPU   :", torch.cuda.get_device_name(0))
print("torch :", torch.__version__)

try:
    import triton
    import triton.language as tl
    print("Triton:", triton.__version__)
except Exception as e:
    raise RuntimeError("Triton is required for the Hawkes benchmark") from e


# ============================================================
# Utilities
# ============================================================

def sync():
    torch.cuda.synchronize()


def bench_ms(fn, warmup=20, iters=100, repeats=7):
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


def rel_l2(a, b, eps=1e-12):
    return ((a.float() - b.float()).norm() /
            b.float().norm().clamp_min(eps)).item()


def random_spd_inv_sqrt(n, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)

    M = torch.randn(n, n, generator=g, dtype=DT, device="cpu")
    Q, _ = torch.linalg.qr(M)

    # Moderate condition number.
    eig = torch.logspace(
        -0.5, 0.5, n,
        dtype=DT,
        device="cpu",
    )

    C = Q @ torch.diag(eig) @ Q.T
    w, V = torch.linalg.eigh(C)

    C_inv_sqrt = (
        (V * w.rsqrt().unsqueeze(0))
        @ V.T
    )

    return C_inv_sqrt.cuda().contiguous()


# ============================================================
# A. Cross-sectional signal whitening / risk normalization
#
# Example interpretation:
#   rows    = assets
#   columns = features / horizons / venues
#
# Covariance model:
#   Sigma ~= Sigma_asset (x) Sigma_feature
#
# Whitening:
#   vec(Y) = (A^{-1/2} (x) B^{-1/2}) vec(X)
#          = vec(A^{-1/2} X B^{-T/2})
#
# Dense baseline materializes the d x d whitening operator.
# Factored path uses two GEMMs.
# ============================================================

def kron_apply_shared_2d(X, A, B):
    """
    X: [M, na, nf]
    A: [na, na]
    B: [nf, nf]

    Returns A X B^T without materializing kron(A,B).

    Implemented as two large GEMMs rather than M tiny GEMMs.
    """
    M, na, nf = X.shape

    # Left multiply all feature columns from all snapshots in one GEMM.
    X_left = (
        X.permute(1, 0, 2)
         .reshape(na, M * nf)
    )

    Y_left = (
        A @ X_left
    ).reshape(
        na, M, nf
    ).permute(
        1, 0, 2
    ).contiguous()

    # Right multiply all asset rows from all snapshots in one GEMM.
    Y = (
        Y_left.reshape(M * na, nf)
        @ B.T
    ).reshape(M, na, nf)

    return Y


@torch.no_grad()
def benchmark_whitening(
    na=128,
    nf=16,
    snapshots=(256, 1024, 4096, 16384),
):
    print()
    print("=" * 110)
    print("A. CROSS-SECTIONAL SIGNAL WHITENING")
    print("=" * 110)

    A = random_spd_inv_sqrt(na, seed=1)
    B = random_spd_inv_sqrt(nf, seed=2)

    d = na * nf

    # For row-major flattening:
    # vec_row(A X B^T) = (A kron B) vec_row(X)
    W = torch.kron(A, B).contiguous()

    factor_storage = A.numel() + B.numel()
    dense_storage = W.numel()

    print(
        f"assets={na}, features={nf}, d={d}"
    )
    print(
        f"dense operator storage : {dense_storage:,} floats"
    )
    print(
        f"factor storage         : {factor_storage:,} floats"
    )
    print(
        f"weight compression     : "
        f"{dense_storage/factor_storage:.1f}x"
    )

    print()
    print(
        f"{'snapshots':>10} | "
        f"{'dense':>10} "
        f"{'factored':>10} "
        f"{'speedup':>9} "
        f"{'rel err':>10}"
    )
    print("-" * 62)

    for M in snapshots:
        X = torch.randn(
            M, na, nf,
            device=DEV,
            dtype=DT,
        )

        Xf = X.reshape(M, d)

        def dense_fn():
            return Xf @ W.T

        def fact_fn():
            return kron_apply_shared_2d(
                X, A, B
            )

        ref = dense_fn().reshape(M, na, nf)
        got = fact_fn()
        err = rel_l2(got, ref)

        dense_ms = bench_ms(
            dense_fn,
            warmup=10,
            iters=30,
            repeats=5,
        )

        fact_ms = bench_ms(
            fact_fn,
            warmup=10,
            iters=30,
            repeats=5,
        )

        print(
            f"{M:10d} | "
            f"{dense_ms*1e3:8.1f}us "
            f"{fact_ms*1e3:8.1f}us "
            f"{dense_ms/fact_ms:8.2f}x "
            f"{err:10.2e}"
        )

        del X, Xf, ref, got
        torch.cuda.empty_cache()


# ============================================================
# B. Multi-term cross-asset / cross-feature propagation
#
# Example:
#   Y = sum_s A_s X B_s^T
#
# Interpretations:
#   - cross-asset transient-impact operator
#   - multi-venue signal propagation
#   - cross-sectional alpha transform
#   - asset x horizon state transition
#
# Equivalent dense operator:
#   W = sum_s A_s (x) B_s
# ============================================================

@torch.no_grad()
def benchmark_multiterm_propagator(
    na=64,
    nf=8,
    S=4,
    snapshots=(256, 1024, 4096, 16384),
):
    print()
    print("=" * 110)
    print("B. MULTI-TERM CROSS-ASSET / CROSS-FEATURE PROPAGATION")
    print("=" * 110)

    g = torch.Generator(
        device=DEV
    ).manual_seed(17)

    As = (
        torch.randn(
            S, na, na,
            device=DEV,
            dtype=DT,
            generator=g,
        )
        / math.sqrt(na)
    ).contiguous()

    Bs = (
        torch.randn(
            S, nf, nf,
            device=DEV,
            dtype=DT,
            generator=g,
        )
        / math.sqrt(nf)
    ).contiguous()

    d = na * nf

    W = torch.zeros(
        d, d,
        device=DEV,
        dtype=DT,
    )

    for s in range(S):
        W += torch.kron(
            As[s],
            Bs[s],
        )

    print(
        f"assets={na}, features={nf}, "
        f"terms={S}, d={d}"
    )

    dense_storage = W.numel()
    fact_storage = As.numel() + Bs.numel()

    print(
        f"weight compression: "
        f"{dense_storage/fact_storage:.1f}x"
    )

    print()
    print(
        f"{'snapshots':>10} | "
        f"{'dense':>10} "
        f"{'factored':>10} "
        f"{'speedup':>9} "
        f"{'rel err':>10}"
    )
    print("-" * 62)

    for M in snapshots:
        X = torch.randn(
            M, na, nf,
            device=DEV,
            dtype=DT,
            generator=g,
        )

        Xf = X.reshape(M, d)

        def dense_fn():
            return Xf @ W.T

        def fact_fn():
            out = torch.zeros_like(X)

            for s in range(S):
                out.add_(
                    kron_apply_shared_2d(
                        X,
                        As[s],
                        Bs[s],
                    )
                )

            return out

        ref = dense_fn().reshape(
            M, na, nf
        )
        got = fact_fn()

        err = rel_l2(
            got,
            ref,
        )

        dense_ms = bench_ms(
            dense_fn,
            warmup=10,
            iters=30,
            repeats=5,
        )

        fact_ms = bench_ms(
            fact_fn,
            warmup=10,
            iters=30,
            repeats=5,
        )

        print(
            f"{M:10d} | "
            f"{dense_ms*1e3:8.1f}us "
            f"{fact_ms*1e3:8.1f}us "
            f"{dense_ms/fact_ms:8.2f}x "
            f"{err:10.2e}"
        )

        del X, Xf, ref, got
        torch.cuda.empty_cache()


# ============================================================
# C. Multivariate Hawkes intensity state update
#
# A common market-microstructure model is a multivariate Hawkes process:
#
#   h <- exp(-beta dt) h + K[:, event]
#
# where event dimensions can be structured as
#
#   asset x event_type
#
# and
#
#   K = K_asset (x) K_event.
#
# We compare two equally fused Triton kernels:
#
#   dense: reads a column from a materialized dense K
#   kron : generates that column entry from two small factors
#
# This isolates the value of preserving Kronecker structure.
# ============================================================

@triton.jit
def hawkes_dense_kernel(
    STATE,
    EVENT,
    DECAY,
    KCOL,
    OUT,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = (
        tl.program_id(0) * BLOCK
        + tl.arange(0, BLOCK)
    )

    total = N * D
    mask = offs < total

    n = offs // D
    j = offs - n * D

    event = tl.load(
        EVENT + n,
        mask=mask,
        other=0,
    )

    decay = tl.load(
        DECAY + n,
        mask=mask,
        other=0.0,
    )

    old = tl.load(
        STATE + offs,
        mask=mask,
        other=0.0,
    )

    # KCOL stores K^T so the selected event column
    # is contiguous in memory.
    excite = tl.load(
        KCOL + event * D + j,
        mask=mask,
        other=0.0,
    )

    tl.store(
        OUT + offs,
        decay * old + excite,
        mask=mask,
    )


@triton.jit
def hawkes_kron_kernel(
    STATE,
    EVENT_ASSET,
    EVENT_TYPE,
    DECAY,
    KA,
    KE,
    OUT,
    N: tl.constexpr,
    NA: tl.constexpr,
    NE: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = (
        tl.program_id(0) * BLOCK
        + tl.arange(0, BLOCK)
    )

    total = N * D
    mask = offs < total

    n = offs // D
    j = offs - n * D

    out_asset = j // NE
    out_type = j - out_asset * NE

    event_asset = tl.load(
        EVENT_ASSET + n,
        mask=mask,
        other=0,
    )

    event_type = tl.load(
        EVENT_TYPE + n,
        mask=mask,
        other=0,
    )

    decay = tl.load(
        DECAY + n,
        mask=mask,
        other=0.0,
    )

    old = tl.load(
        STATE + offs,
        mask=mask,
        other=0.0,
    )

    a = tl.load(
        KA + out_asset * NA + event_asset,
        mask=mask,
        other=0.0,
    )

    e = tl.load(
        KE + out_type * NE + event_type,
        mask=mask,
        other=0.0,
    )

    tl.store(
        OUT + offs,
        decay * old + a * e,
        mask=mask,
    )


def hawkes_dense(
    state,
    event,
    decay,
    Kcol,
    out,
):
    N, D = state.shape
    BLOCK = 256

    grid = (
        triton.cdiv(
            N * D,
            BLOCK,
        ),
    )

    hawkes_dense_kernel[grid](
        state,
        event,
        decay,
        Kcol,
        out,
        N=N,
        D=D,
        BLOCK=BLOCK,
        num_warps=4,
    )

    return out


def hawkes_kron(
    state,
    event_asset,
    event_type,
    decay,
    KA,
    KE,
    out,
):
    N, D = state.shape
    NA = KA.shape[0]
    NE = KE.shape[0]

    assert D == NA * NE

    BLOCK = 256

    grid = (
        triton.cdiv(
            N * D,
            BLOCK,
        ),
    )

    hawkes_kron_kernel[grid](
        state,
        event_asset,
        event_type,
        decay,
        KA,
        KE,
        out,
        N=N,
        NA=NA,
        NE=NE,
        D=D,
        BLOCK=BLOCK,
        num_warps=4,
    )

    return out


@torch.no_grad()
def benchmark_hawkes(
    na=128,
    ne=8,
    universes=(1024, 8192, 32768),
):
    print()
    print("=" * 110)
    print("C. MULTIVARIATE HAWKES EVENT UPDATE")
    print("=" * 110)

    g = torch.Generator(
        device=DEV
    ).manual_seed(123)

    KA = (
        torch.rand(
            na, na,
            device=DEV,
            dtype=DT,
            generator=g,
        )
        * 0.02
    ).contiguous()

    KE = (
        torch.rand(
            ne, ne,
            device=DEV,
            dtype=DT,
            generator=g,
        )
        * 0.02
    ).contiguous()

    D = na * ne

    Kdense = torch.kron(
        KA,
        KE,
    )

    # Column-major logical access made contiguous:
    # Kcol[event, output].
    Kcol = (
        Kdense.T
        .contiguous()
    )

    dense_weights = Kdense.numel()
    factor_weights = (
        KA.numel()
        + KE.numel()
    )

    print(
        f"assets={na}, event_types={ne}, d={D}"
    )
    print(
        f"dense excitation weights : "
        f"{dense_weights:,}"
    )
    print(
        f"factor weights           : "
        f"{factor_weights:,}"
    )
    print(
        f"weight compression       : "
        f"{dense_weights/factor_weights:.1f}x"
    )

    print()
    print(
        f"{'streams':>10} | "
        f"{'dense':>10} "
        f"{'kron':>10} "
        f"{'speedup':>9} "
        f"{'rel err':>10}"
    )
    print("-" * 62)

    for N in universes:
        state = torch.randn(
            N, D,
            device=DEV,
            dtype=DT,
            generator=g,
        )

        event_asset = torch.randint(
            0, na,
            (N,),
            device=DEV,
            generator=g,
            dtype=torch.int32,
        )

        event_type = torch.randint(
            0, ne,
            (N,),
            device=DEV,
            generator=g,
            dtype=torch.int32,
        )

        event = (
            event_asset * ne
            + event_type
        ).contiguous()

        # exp(-beta * dt), synthetic here.
        decay = (
            0.90
            + 0.099
            * torch.rand(
                N,
                device=DEV,
                dtype=DT,
                generator=g,
            )
        ).contiguous()

        out_dense = torch.empty_like(
            state
        )
        out_kron = torch.empty_like(
            state
        )

        hawkes_dense(
            state,
            event,
            decay,
            Kcol,
            out_dense,
        )

        hawkes_kron(
            state,
            event_asset,
            event_type,
            decay,
            KA,
            KE,
            out_kron,
        )

        sync()

        err = rel_l2(
            out_kron,
            out_dense,
        )

        dense_ms = bench_ms(
            lambda: hawkes_dense(
                state,
                event,
                decay,
                Kcol,
                out_dense,
            ),
            warmup=20,
            iters=100,
            repeats=7,
        )

        kron_ms = bench_ms(
            lambda: hawkes_kron(
                state,
                event_asset,
                event_type,
                decay,
                KA,
                KE,
                out_kron,
            ),
            warmup=20,
            iters=100,
            repeats=7,
        )

        print(
            f"{N:10d} | "
            f"{dense_ms*1e3:8.1f}us "
            f"{kron_ms*1e3:8.1f}us "
            f"{dense_ms/kron_ms:8.2f}x "
            f"{err:10.2e}"
        )

        del (
            state,
            event_asset,
            event_type,
            event,
            decay,
            out_dense,
            out_kron,
        )

        torch.cuda.empty_cache()


# ============================================================
# Run
# ============================================================

benchmark_whitening()

benchmark_multiterm_propagator()

benchmark_hawkes()

print()
print("=" * 110)
print("HOW TO INTERPRET")
print("=" * 110)
print("""
A useful follow-up case has at least one of these:
  1) factorized path is already faster in PyTorch;
  2) factorized path is close but uses dramatically less storage;
  3) factorized path loses only because it launches many small ops,
     suggesting a fused CUDA/Triton kernel could recover the gap;
  4) speedup increases strongly with number of assets/streams/snapshots.

The strongest candidates should then be ported to the same custom-CUDA
style as the Kronecker Kalman kernel.
""")


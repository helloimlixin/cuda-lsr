
# Fused Kronecker-sum propagator for HFT-style cross-asset / cross-feature updates
#
# Computes, exactly up to FP32 roundoff:
#
#   Y[m] = sum_{s=1}^S A_s X[m] B_s^T
#
# where:
#   X[m] : [64, 8]
#   A_s  : [64, 64]
#   B_s  : [8, 8]
#
# This is equivalent to:
#
#   vec(Y[m]) = [sum_s A_s (x) B_s] vec(X[m])
#
# but never materializes the 512 x 512 dense operator at runtime.
#
# Important systems point:
#   The previous PyTorch factored path used ~2S separate GEMMs.
#   This CUDA kernel performs the entire S-term operator in ONE launch.
#
# Stage 1 (inside one CUDA block per snapshot):
#   Z_s = A_s X
#
# Stage 2:
#   Y = sum_s Z_s B_s^T
#
# The A factors are prepacked as A^T-like [S, reduction, output-row]
# so all threads in a warp read contiguous A values at each reduction step.
#
# Supported S: 1, 2, 4, 8
# Target shape: asset x feature = 64 x 8
#
# Recommended:
#   NVIDIA GPU + nvcc + PyTorch CUDA build.
#
# This is a systems microbenchmark, not a trading strategy.

import os
import sys
import math
import shutil
import tempfile
import statistics
import subprocess

import torch
from torch.utils.cpp_extension import load_inline


# ============================================================
# Environment
# ============================================================

assert torch.cuda.is_available(), "CUDA GPU required"

DEV = "cuda"
DT = torch.float32

torch.manual_seed(17)
torch.cuda.manual_seed_all(17)

# Keep this comparison FP32-vs-FP32.
torch.set_float32_matmul_precision("highest")
try:
    torch.backends.cuda.matmul.allow_tf32 = False
except Exception:
    pass

print("GPU   :", torch.cuda.get_device_name(0))
print("torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("float32 matmul precision:", torch.get_float32_matmul_precision())


# ============================================================
# CUDA extension
# ============================================================

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>


template<int S>
__global__ __launch_bounds__(256)
void kron_sum_64x8_kernel(
    const float* __restrict__ X,       // [M,64,8]
    const float* __restrict__ AT,      // [S,64(reduction),64(output)]
    const float* __restrict__ B,       // [S,8,8] ; B[s,j,b]
    float* __restrict__ Y,             // [M,64,8]
    int M
) {
    constexpr int NA = 64;
    constexpr int NF = 8;
    constexpr int D  = NA * NF;

    const int m = blockIdx.x;
    const int t = threadIdx.x;

    if (m >= M) return;

    // X for this snapshot.
    __shared__ float sx[D];

    // Z[s,i,b] = sum_a A[s,i,a] X[a,b].
    __shared__ float sz[S * D];

    // B is tiny and shared by all output calculations.
    __shared__ float sb[S * NF * NF];

    // --------------------------------------------------------
    // Load X and B.
    // --------------------------------------------------------
    for (int e = t; e < D; e += blockDim.x) {
        sx[e] = X[(size_t)m * D + e];
    }

    for (int e = t; e < S * NF * NF; e += blockDim.x) {
        sb[e] = B[e];
    }

    __syncthreads();

    // --------------------------------------------------------
    // Stage 1:
    //
    // One thread owns one (s,i) row at a time and computes
    // all eight feature outputs.
    //
    // A is stored as AT[s,a,i], not A[s,i,a].
    // For fixed a, neighboring threads i read contiguous values.
    // --------------------------------------------------------
    for (int si = t; si < S * NA; si += blockDim.x) {
        const int s = si / NA;
        const int i = si - s * NA;

        float r0 = 0.f;
        float r1 = 0.f;
        float r2 = 0.f;
        float r3 = 0.f;
        float r4 = 0.f;
        float r5 = 0.f;
        float r6 = 0.f;
        float r7 = 0.f;

        #pragma unroll 8
        for (int a = 0; a < NA; ++a) {
            // AT[s,a,i] = A[s,i,a]
            const float av = AT[
                ((size_t)s * NA + a) * NA + i
            ];

            const float* xrow = &sx[a * NF];

            r0 = fmaf(av, xrow[0], r0);
            r1 = fmaf(av, xrow[1], r1);
            r2 = fmaf(av, xrow[2], r2);
            r3 = fmaf(av, xrow[3], r3);
            r4 = fmaf(av, xrow[4], r4);
            r5 = fmaf(av, xrow[5], r5);
            r6 = fmaf(av, xrow[6], r6);
            r7 = fmaf(av, xrow[7], r7);
        }

        float* zrow = &sz[
            ((size_t)s * NA + i) * NF
        ];

        zrow[0] = r0;
        zrow[1] = r1;
        zrow[2] = r2;
        zrow[3] = r3;
        zrow[4] = r4;
        zrow[5] = r5;
        zrow[6] = r6;
        zrow[7] = r7;
    }

    __syncthreads();

    // --------------------------------------------------------
    // Stage 2:
    //
    // Y[i,j] = sum_s sum_b Z[s,i,b] B[s,j,b].
    //
    // 512 outputs. With 256 threads, each thread computes
    // at most two outputs.
    // --------------------------------------------------------
    for (int e = t; e < D; e += blockDim.x) {
        const int i = e / NF;
        const int j = e - i * NF;

        float acc = 0.f;

        #pragma unroll
        for (int s = 0; s < S; ++s) {
            const float* zrow = &sz[
                ((size_t)s * NA + i) * NF
            ];

            const float* brow = &sb[
                ((size_t)s * NF + j) * NF
            ];

            #pragma unroll
            for (int b = 0; b < NF; ++b) {
                acc = fmaf(
                    zrow[b],
                    brow[b],
                    acc
                );
            }
        }

        Y[(size_t)m * D + e] = acc;
    }
}


static void check_tensor(
    const torch::Tensor& x,
    const char* name
) {
    TORCH_CHECK(
        x.is_cuda(),
        name,
        " must be CUDA"
    );
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat32,
        name,
        " must be float32"
    );
    TORCH_CHECK(
        x.is_contiguous(),
        name,
        " must be contiguous"
    );
}


void kron_sum_out(
    torch::Tensor x,
    torch::Tensor at,
    torch::Tensor b,
    torch::Tensor out
) {
    check_tensor(x,   "x");
    check_tensor(at,  "at");
    check_tensor(b,   "b");
    check_tensor(out, "out");

    TORCH_CHECK(
        x.dim() == 3
        && x.size(1) == 64
        && x.size(2) == 8,
        "x must have shape [M,64,8]"
    );

    const int M = x.size(0);
    const int S = at.size(0);

    TORCH_CHECK(
        at.dim() == 3
        && at.size(1) == 64
        && at.size(2) == 64,
        "at must have shape [S,64,64]"
    );

    TORCH_CHECK(
        b.dim() == 3
        && b.size(0) == S
        && b.size(1) == 8
        && b.size(2) == 8,
        "b must have shape [S,8,8]"
    );

    TORCH_CHECK(
        out.sizes() == x.sizes(),
        "out must match x shape"
    );

    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 block(256);
    dim3 grid(M);

    #define DISPATCH_S(SVAL)                                       \
        if (S == SVAL) {                                          \
            kron_sum_64x8_kernel<SVAL>                             \
                <<<grid, block, 0, stream>>>(                      \
                    x.data_ptr<float>(),                           \
                    at.data_ptr<float>(),                          \
                    b.data_ptr<float>(),                           \
                    out.data_ptr<float>(),                         \
                    M                                              \
                );                                                 \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                        \
            return;                                                \
        }

    DISPATCH_S(1)
    DISPATCH_S(2)
    DISPATCH_S(4)
    DISPATCH_S(8)

    #undef DISPATCH_S

    TORCH_CHECK(
        false,
        "Supported S values are 1, 2, 4, 8; got ",
        S
    );
}
"""


CPP_SRC = r"""
#include <torch/extension.h>

void kron_sum_out(
    torch::Tensor x,
    torch::Tensor at,
    torch::Tensor b,
    torch::Tensor out
);
"""


def build_extension():
    try:
        import ninja  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "ninja",
            ]
        )

    cap = torch.cuda.get_device_capability()
    os.environ[
        "TORCH_CUDA_ARCH_LIST"
    ] = f"{cap[0]}.{cap[1]}"

    build_dir = os.path.join(
        tempfile.gettempdir(),
        "hft_kron_sum_build",
    )

    shutil.rmtree(
        build_dir,
        ignore_errors=True,
    )
    os.makedirs(
        build_dir,
        exist_ok=True,
    )

    print(
        "building CUDA extension in",
        build_dir,
        "...",
    )

    ext = load_inline(
        name="hft_kron_sum_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["kron_sum_out"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
        ],
        build_directory=build_dir,
        verbose=False,
    )

    print("build OK")
    return ext


ext = build_extension()


# ============================================================
# Benchmark utilities
# ============================================================

def sync():
    torch.cuda.synchronize()


def median_ms(
    fn,
    warmup=20,
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
            / iters
        )

    return statistics.median(
        vals
    )


def rel_l2(
    a,
    b,
    eps=1e-12,
):
    return (
        (
            a.float()
            - b.float()
        ).norm()
        /
        b.float()
        .norm()
        .clamp_min(eps)
    ).item()


# ============================================================
# PyTorch factored reference
# ============================================================

def kron_apply_shared_2d(
    X,
    A,
    B,
):
    """
    X: [M,64,8]
    A: [64,64]
    B: [8,8]

    Returns A X B^T.
    """
    M = X.shape[0]

    left = (
        A
        @ X.permute(
            1, 0, 2
        ).reshape(
            64,
            M * 8,
        )
    )

    left = (
        left.reshape(
            64,
            M,
            8,
        )
        .permute(
            1, 0, 2
        )
        .contiguous()
    )

    return (
        left.reshape(
            M * 64,
            8,
        )
        @ B.T
    ).reshape(
        M,
        64,
        8,
    )


# ============================================================
# Workload generation
# ============================================================

def make_case(
    M,
    S,
    seed=17,
):
    g = torch.Generator(
        device=DEV
    )
    g.manual_seed(
        seed
        + 1000 * M
        + S
    )

    X = torch.randn(
        M,
        64,
        8,
        device=DEV,
        dtype=DT,
        generator=g,
    ).contiguous()

    A = (
        torch.randn(
            S,
            64,
            64,
            device=DEV,
            dtype=DT,
            generator=g,
        )
        / math.sqrt(64)
    ).contiguous()

    B = (
        torch.randn(
            S,
            8,
            8,
            device=DEV,
            dtype=DT,
            generator=g,
        )
        / math.sqrt(8)
    ).contiguous()

    # Prepack A for coalesced access in the CUDA kernel:
    #
    # AT[s,a,i] = A[s,i,a].
    #
    # This is a reusable model/operator packing step and is NOT
    # included in steady-state latency.
    AT = (
        A.transpose(1, 2)
        .contiguous()
    )

    W = torch.zeros(
        512,
        512,
        device=DEV,
        dtype=DT,
    )

    for s in range(S):
        W.add_(
            torch.kron(
                A[s],
                B[s],
            )
        )

    Y_dense = torch.empty_like(
        X
    )

    Y_cuda = torch.empty_like(
        X
    )

    return (
        X,
        A,
        AT,
        B,
        W,
        Y_dense,
        Y_cuda,
    )


# ============================================================
# Main benchmark
# ============================================================

@torch.no_grad()
def run_case(
    M,
    S,
):
    (
        X,
        A,
        AT,
        B,
        W,
        Y_dense,
        Y_cuda,
    ) = make_case(
        M,
        S,
    )

    Xflat = X.view(
        M,
        512,
    )

    Ydense_flat = Y_dense.view(
        M,
        512,
    )

    WT = W.T.contiguous()

    # ----------------------------------------
    # Dense baseline, preallocated output.
    # ----------------------------------------
    def dense_fn():
        torch.mm(
            Xflat,
            WT,
            out=Ydense_flat,
        )

    # ----------------------------------------
    # Existing PyTorch factored path.
    # ----------------------------------------
    def torch_factored_fn():
        out = torch.zeros_like(
            X
        )

        for s in range(S):
            out.add_(
                kron_apply_shared_2d(
                    X,
                    A[s],
                    B[s],
                )
            )

        return out

    # ----------------------------------------
    # One-launch CUDA Kronecker sum.
    # ----------------------------------------
    def cuda_fn():
        ext.kron_sum_out(
            X,
            AT,
            B,
            Y_cuda,
        )

    # Correctness.
    dense_fn()
    cuda_fn()
    sync()

    err_cuda = rel_l2(
        Y_cuda,
        Y_dense,
    )

    Y_torch = (
        torch_factored_fn()
    )

    err_torch = rel_l2(
        Y_torch,
        Y_dense,
    )

    # Timing.
    dense_ms = median_ms(
        dense_fn,
        warmup=20,
        iters=100,
        repeats=7,
    )

    cuda_ms = median_ms(
        cuda_fn,
        warmup=20,
        iters=100,
        repeats=7,
    )

    # PyTorch factored path is much slower,
    # so use fewer repetitions.
    torch_factored_ms = median_ms(
        torch_factored_fn,
        warmup=5,
        iters=10,
        repeats=5,
    )

    # ----------------------------------------
    # Analytical arithmetic.
    # ----------------------------------------
    dense_mac = (
        512 * 512
    )

    structured_mac = (
        S
        * (
            64 * 64 * 8
            + 64 * 8 * 8
        )
    )

    arithmetic_ratio = (
        dense_mac
        / structured_mac
    )

    dense_params = (
        512 * 512
    )

    structured_params = (
        S
        * (
            64 * 64
            + 8 * 8
        )
    )

    compression = (
        dense_params
        / structured_params
    )

    row = {
        "M": M,
        "S": S,
        "dense_us":
            dense_ms * 1000.0,
        "torch_factored_us":
            torch_factored_ms * 1000.0,
        "cuda_us":
            cuda_ms * 1000.0,
        "cuda_speedup_vs_dense":
            dense_ms / cuda_ms,
        "cuda_speedup_vs_torch_factored":
            torch_factored_ms / cuda_ms,
        "torch_factored_over_dense":
            torch_factored_ms / dense_ms,
        "cuda_rel_l2":
            err_cuda,
        "torch_factored_rel_l2":
            err_torch,
        "dense_mac_per_snapshot":
            dense_mac,
        "structured_mac_per_snapshot":
            structured_mac,
        "dense_over_structured_mac":
            arithmetic_ratio,
        "weight_compression":
            compression,
    }

    del (
        X,
        A,
        AT,
        B,
        W,
        WT,
        Y_dense,
        Y_cuda,
        Y_torch,
    )

    torch.cuda.empty_cache()

    return row


def print_header():
    print()
    print("=" * 132)
    print(
        "FUSED CROSS-ASSET / CROSS-FEATURE "
        "KRONECKER-SUM PROPAGATOR"
    )
    print("=" * 132)

    print(
        f"{'S':>3} {'M':>8} | "
        f"{'dense':>10} "
        f"{'torch fact':>12} "
        f"{'CUDA fused':>11} | "
        f"{'CUDA/dense':>11} "
        f"{'CUDA/torch':>11} | "
        f"{'MAC ratio':>9} "
        f"{'compression':>11} "
        f"{'rel err':>10}"
    )

    print("-" * 132)


def print_row(r):
    print(
        f"{r['S']:3d} "
        f"{r['M']:8d} | "
        f"{r['dense_us']:8.1f}us "
        f"{r['torch_factored_us']:10.1f}us "
        f"{r['cuda_us']:9.1f}us | "
        f"{r['cuda_speedup_vs_dense']:10.2f}x "
        f"{r['cuda_speedup_vs_torch_factored']:10.2f}x | "
        f"{r['dense_over_structured_mac']:8.2f}x "
        f"{r['weight_compression']:10.1f}x "
        f"{r['cuda_rel_l2']:10.2e}"
    )


if __name__ == "__main__":
    print_header()

    results = []

    # S=4 is the original HFT screening workload.
    #
    # S sweep is useful because it shows the crossover
    # between representational capacity and execution cost.
    for S in [1, 2, 4, 8]:
        for M in [
            256,
            1024,
            4096,
            16384,
        ]:
            r = run_case(
                M,
                S,
            )

            results.append(r)
            print_row(r)

    print()
    print("=" * 132)
    print("INTERPRETATION")
    print("=" * 132)

    print(
        """
The most important row is S=4, because it matches the previous
cross-asset / cross-feature benchmark.

For S=4:

    dense MACs / snapshot      = 262,144
    structured MACs / snapshot = 147,456
    theoretical MAC reduction  = 1.78x
    weight compression          = 15.75x

The goal of this kernel is to determine whether eliminating the 2S
framework GEMM launches is enough to turn those analytical advantages
into a real wall-clock win.

If CUDA/dense > 1:
    We have an exact structured HFT-style operator that is both smaller
    and faster than the materialized dense transform.

If CUDA/dense is close to 1:
    Profile the kernel before changing the model. The next targets are:
      * vectorized / wider A loads;
      * more snapshots per block;
      * tensor-core / WMMA execution for stage 1;
      * persistent A/B factor caching;
      * fusing downstream nonlinear/state-update work.

If CUDA/dense remains far below 1:
    The dense 512x512 GEMM is simply too efficient on this GPU at this
    shape. Increase asset dimension before abandoning the idea, because
    the dense operator grows quadratically while the factorized operator
    does not.
"""
    )



# Script 4: Causal rank-parallelism sweep for tiny Kronecker factors
#
# Goal
# ----
# Isolate ONE variable:
#
#     how many independent groups of Kronecker terms are exposed
#     concurrently to the GPU?
#
# Fixed representation:
#
#   W = sum_{s=1}^S
#       F0_s kron F1_s kron F2_s kron F3_s kron F4_s
#
#   F0 : [1,3]
#   F1..F4 : [2,4]
#
# Full W: [16,768]
#
# Each term has only 35 FP32 factor values.
#
# We sweep rank partitions P:
#
#   P = 1,2,4,8,16,32,64,...
#
# The CUDA kernel body is IDENTICAL for every P.
#
# blockIdx.y selects the rank partition.
#
# Partition p processes:
#
#   s_begin = floor(S*p/P)
#   s_end   = floor(S*(p+1)/P)
#
# Every configuration then runs the SAME reduction kernel.
#
# Thus:
#
#   latency(P=1) / latency(P)
#
# is a clean measure of the benefit from exposing the rank/term
# dimension as additional GPU parallelism.
#
# Unlike the previous Triton script:
#   * there is no static-range unrolling difference between variants;
#   * P=1 and P>1 use the exact same compiled CUDA kernel;
#   * timing differences come from the execution schedule.
#
# Run:
#   python kron_rank_parallel_causal.py

import os
import sys
import shutil
import tempfile
import statistics
import subprocess

import torch
from torch.utils.cpp_extension import load_inline


assert torch.cuda.is_available()

DEV = "cuda"
DT = torch.float32

torch.manual_seed(17)
torch.cuda.manual_seed_all(17)

print("GPU   :", torch.cuda.get_device_name(0))
print("torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)

props = torch.cuda.get_device_properties(0)
print("SMs   :", props.multi_processor_count)


# ============================================================
# CUDA extension
# ============================================================

CPP_SRC = r"""
#include <torch/extension.h>

void kron_rank_partials(
    torch::Tensor f0,
    torch::Tensor f1,
    torch::Tensor f2,
    torch::Tensor f3,
    torch::Tensor f4,
    torch::Tensor partial,
    int64_t S,
    int64_t P
);

void kron_rank_reduce(
    torch::Tensor partial,
    torch::Tensor out,
    int64_t P
);
"""


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>


constexpr int NROW = 16;
constexpr int NCOL = 768;
constexpr int NEL  = NROW * NCOL;


// ============================================================
// Same kernel for ALL partition counts P.
//
// Grid:
//   x = output tile
//   y = rank partition
//
// Each thread owns one output entry within its tile and loops only
// over the contiguous rank range assigned to blockIdx.y.
// ============================================================

__global__ void kron_rank_partials_kernel(
    const float* __restrict__ F0,   // [S,1,3]
    const float* __restrict__ F1,   // [S,2,4]
    const float* __restrict__ F2,
    const float* __restrict__ F3,
    const float* __restrict__ F4,
    float* __restrict__ PARTIAL,    // [P,NEL]
    int S,
    int P
) {
    const int e =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (e >= NEL) {
        return;
    }

    const int part = blockIdx.y;

    // Exact integer partitioning of [0,S).
    const int s_begin =
        (int)(((long long)S * part) / P);

    const int s_end =
        (int)(((long long)S * (part + 1)) / P);

    const int i = e / NCOL;
    const int j = e - i * NCOL;

    // Output coordinates for:
    // [1,2,2,2,2] x [3,4,4,4,4]
    const int i1 = i / 8;
    const int i2 = (i / 4) & 1;
    const int i3 = (i / 2) & 1;
    const int i4 = i & 1;

    const int j0 = j / 256;
    const int j1 = (j / 64) & 3;
    const int j2 = (j / 16) & 3;
    const int j3 = (j / 4) & 3;
    const int j4 = j & 3;

    float acc = 0.0f;

    // IMPORTANT:
    // This exact runtime loop is used for every P.
    for (int s = s_begin; s < s_end; ++s) {
        const float a =
            F0[(size_t)s * 3 + j0];

        const float b =
            F1[(size_t)s * 8 + i1 * 4 + j1];

        const float c =
            F2[(size_t)s * 8 + i2 * 4 + j2];

        const float d =
            F3[(size_t)s * 8 + i3 * 4 + j3];

        const float f =
            F4[(size_t)s * 8 + i4 * 4 + j4];

        acc = fmaf(
            ((a * b) * c) * d,
            f,
            acc
        );
    }

    PARTIAL[
        (size_t)part * NEL + e
    ] = acc;
}


// ============================================================
// Same reduction kernel for every P.
// ============================================================

__global__ void kron_rank_reduce_kernel(
    const float* __restrict__ PARTIAL,
    float* __restrict__ OUT,
    int P
) {
    const int e =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (e >= NEL) {
        return;
    }

    float acc = 0.0f;

    for (int p = 0; p < P; ++p) {
        acc += PARTIAL[
            (size_t)p * NEL + e
        ];
    }

    OUT[e] = acc;
}


static void check_f(
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


void kron_rank_partials(
    torch::Tensor f0,
    torch::Tensor f1,
    torch::Tensor f2,
    torch::Tensor f3,
    torch::Tensor f4,
    torch::Tensor partial,
    int64_t S,
    int64_t P
) {
    check_f(f0, "f0");
    check_f(f1, "f1");
    check_f(f2, "f2");
    check_f(f3, "f3");
    check_f(f4, "f4");
    check_f(partial, "partial");

    TORCH_CHECK(
        f0.size(0) == S
        && f0.size(1) == 1
        && f0.size(2) == 3,
        "f0 shape"
    );

    TORCH_CHECK(
        f1.size(0) == S
        && f1.size(1) == 2
        && f1.size(2) == 4,
        "f1 shape"
    );

    TORCH_CHECK(
        f2.sizes() == f1.sizes(),
        "f2 shape"
    );

    TORCH_CHECK(
        f3.sizes() == f1.sizes(),
        "f3 shape"
    );

    TORCH_CHECK(
        f4.sizes() == f1.sizes(),
        "f4 shape"
    );

    TORCH_CHECK(
        partial.dim() == 2
        && partial.size(0) == P
        && partial.size(1) == NEL,
        "partial must be [P,12288]"
    );

    constexpr int THREADS = 256;

    dim3 block(THREADS);
    dim3 grid(
        (NEL + THREADS - 1) / THREADS,
        (unsigned)P
    );

    auto stream =
        at::cuda::getCurrentCUDAStream();

    kron_rank_partials_kernel
        <<<grid, block, 0, stream>>>(
            f0.data_ptr<float>(),
            f1.data_ptr<float>(),
            f2.data_ptr<float>(),
            f3.data_ptr<float>(),
            f4.data_ptr<float>(),
            partial.data_ptr<float>(),
            (int)S,
            (int)P
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void kron_rank_reduce(
    torch::Tensor partial,
    torch::Tensor out,
    int64_t P
) {
    check_f(partial, "partial");
    check_f(out, "out");

    TORCH_CHECK(
        partial.dim() == 2
        && partial.size(0) == P
        && partial.size(1) == NEL,
        "partial must be [P,12288]"
    );

    TORCH_CHECK(
        out.numel() == NEL,
        "out must contain 12288 floats"
    );

    constexpr int THREADS = 256;

    dim3 block(THREADS);
    dim3 grid(
        (NEL + THREADS - 1) / THREADS
    );

    auto stream =
        at::cuda::getCurrentCUDAStream();

    kron_rank_reduce_kernel
        <<<grid, block, 0, stream>>>(
            partial.data_ptr<float>(),
            out.data_ptr<float>(),
            (int)P
        );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def build_extension():
    try:
        import ninja  # noqa
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
        "kron_rank_parallel_causal_build",
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
        name="kron_rank_parallel_causal_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=[
            "kron_rank_partials",
            "kron_rank_reduce",
        ],
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
# Helpers
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


def rel_l2(
    a,
    b,
    eps=1e-12,
):
    return (
        (a.float() - b.float()).norm()
        / b.float().norm().clamp_min(eps)
    ).item()


# ============================================================
# Inputs
# ============================================================

def make_factors(
    S,
    seed=17,
):
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
# Two-kernel P-partition operator
# ============================================================

@torch.no_grad()
def run_partitioned(
    factors,
    partial,
    out,
    S,
    P,
):
    ext.kron_rank_partials(
        *factors,
        partial,
        S,
        P,
    )

    ext.kron_rank_reduce(
        partial,
        out,
        P,
    )


# ============================================================
# Reference via P=1
# ============================================================

@torch.no_grad()
def run_s_case(
    S,
):
    factors = make_factors(S)

    partitions = [
        p
        for p in [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
        ]
        if p <= S
    ]

    rows = []

    ref = None
    base_us = None

    for P in partitions:
        partial = torch.empty(
            P,
            16 * 768,
            device=DEV,
            dtype=DT,
        )

        out = torch.empty(
            16,
            768,
            device=DEV,
            dtype=DT,
        )

        # Compile/warm once.
        run_partitioned(
            factors,
            partial,
            out,
            S,
            P,
        )
        sync()

        us = median_us(
            lambda: run_partitioned(
                factors,
                partial,
                out,
                S,
                P,
            ),
            warmup=10,
            iters=100,
            repeats=7,
        )

        if P == 1:
            ref = out.clone()
            base_us = us
            err = 0.0
        else:
            run_partitioned(
                factors,
                partial,
                out,
                S,
                P,
            )
            sync()

            err = rel_l2(
                out,
                ref,
            )

        terms_per_partition = (
            S / P
        )

        rows.append(
            {
                "S": S,
                "P": P,
                "terms_per_partition":
                    terms_per_partition,
                "us": us,
                "speedup_vs_p1":
                    base_us / us,
                "rel_l2":
                    err,
                "partial_kb":
                    partial.numel()
                    * 4
                    / 1024,
            }
        )

    return rows


# ============================================================
# Main
# ============================================================

def main():
    print()
    print("=" * 112)
    print(
        "CAUSAL SWEEP: "
        "KRONECKER-TERM PARALLELISM"
    )
    print("=" * 112)

    print()
    print(
        "Fixed factors: "
        "(1x3) x (2x4)^4"
    )
    print(
        "35 FP32 factor values per Kronecker term."
    )
    print(
        "P=1 and P>1 execute the SAME compiled CUDA kernel."
    )
    print()

    all_rows = []

    for S in [
        64,
        128,
        256,
        512,
        1024,
    ]:
        rows = run_s_case(S)
        all_rows.extend(rows)

        print()
        print(
            f"S = {S}"
        )

        print(
            f"{'P':>5} "
            f"{'terms/P':>10} | "
            f"{'latency':>11} "
            f"{'vs P=1':>9} | "
            f"{'partial KB':>11} "
            f"{'rel err':>10}"
        )

        print("-" * 68)

        for r in rows:
            print(
                f"{r['P']:5d} "
                f"{r['terms_per_partition']:10.1f} | "
                f"{r['us']:9.2f}us "
                f"{r['speedup_vs_p1']:8.2f}x | "
                f"{r['partial_kb']:10.1f} "
                f"{r['rel_l2']:10.2e}"
            )

        best = min(
            rows,
            key=lambda x: x["us"],
        )

        print(
            f"best S={S}: "
            f"P={best['P']}, "
            f"{best['us']:.2f} us, "
            f"{best['speedup_vs_p1']:.2f}x "
            f"vs P=1"
        )

    print()
    print("=" * 112)
    print("PAPER INTERPRETATION")
    print("=" * 112)

    print(
        r"""
This experiment varies only one thing: the degree of parallelism over
the Kronecker-term dimension.

For fixed S:

    P = 1
        one rank partition; every output thread processes all terms.

    P > 1
        the same term range is split into P independent partitions.
        CUDA launches those partitions concurrently across the GPU,
        followed by the same reduction kernel.

Therefore:

    T(P=1) / T(P)

directly measures the benefit of exposing the tiny Kronecker-term
dimension as GPU parallel work.

The intended systems claim is:

    Because each Kronecker term consists of only a handful of tiny
    factor accesses and scalar products, the term dimension can be
    partitioned aggressively and evaluated concurrently across CUDA
    thread blocks. This converts a sequence of fine-grained structured
    operations into a large two-dimensional parallel workload over
    output entries and Kronecker terms.

Do not call this "one matrix per CUDA core." The scheduling unit is
threads/warps/CTAs; CUDA maps those onto SM execution resources.
"""
    )


if __name__ == "__main__":
    main()


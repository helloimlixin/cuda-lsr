
# Script 2: Exact-W on-chip-resident tiny-factor benchmark
#
# Purpose
# -------
# Script 1 showed:
#
#   * huge native-vs-framework wins;
#   * but a direct "one output entry per thread" kernel does NOT make
#     finer factorization faster by itself.
#
# That is expected: once every output is already parallel, splitting
# a factor only adds scalar loads/multiplies.
#
# This script tests the actual hardware hypothesis:
#
#     Fine Kronecker factors have a much smaller FACTOR WORKING SET.
#     A CTA can cooperatively stage many complete separation terms
#     into on-chip shared memory and reuse them across many output
#     entries.
#
# SAME represented operator:
#
#   K=3: (4x48), (2x4), (2x4)
#        208 floats / separation term
#
#   K=4: (2x12), (2x4), (2x4), (2x4)
#         48 floats / separation term
#
#   K=5: (1x3), (2x4), (2x4), (2x4), (2x4)
#         35 floats / separation term
#
# At S=256:
#
#   K=3 factor footprint = 208 KB
#   K=4 factor footprint =  48 KB
#   K=5 factor footprint =  35 KB
#
# Thus K=4/K=5 can potentially make the entire factor set resident
# in a ~48 KB CTA working set, whereas K=3 cannot.
#
# Kernel strategy
# ---------------
# Each CTA owns a tile of dense output entries.
#
# For a chunk of separation terms:
#   1. threads cooperatively copy the COMPLETE factor matrices for
#      that chunk from global memory into shared memory;
#   2. synchronize;
#   3. every output thread accumulates all terms using the resident
#      factors;
#   4. move to the next chunk.
#
# This is deliberately different from Script 1.
#
# We autotune:
#   * output threads per CTA: 128, 256, 512
#   * resident separation-rank chunk
#
# The important comparison is still exact-W:
#
#       latency(K=3) / latency(K=5)
#
# but now the execution schedule is explicitly designed to exploit
# small factor footprints.
#
# Run:
#   python exact_w_resident_kron.py
#
# Recommended:
#   NVIDIA GPU, PyTorch CUDA build, nvcc available.

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

print("GPU   :", torch.cuda.get_device_name(0))
print("torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)

props = torch.cuda.get_device_properties(0)

print("SM count:", props.multi_processor_count)
print("shared memory / block:", props.shared_memory_per_block)

if hasattr(props, "shared_memory_per_block_optin"):
    print(
        "opt-in shared memory / block:",
        props.shared_memory_per_block_optin,
    )


# ============================================================
# Build exact-equivalent factors
# ============================================================

def batched_kron(A, B):
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


def make_equivalent_factors(S, seed=17):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + S)

    def rn(shape):
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


def flatten_pack(factors):
    """
    Pack each separation term contiguously:

        [term0 factor0, factor1, ...,
         term1 factor0, factor1, ...]

    This makes cooperative whole-factor staging straightforward.
    """
    S = factors[0].shape[0]

    rows = []

    for s in range(S):
        rows.append(
            torch.cat(
                [
                    f[s].reshape(-1)
                    for f in factors
                ],
                dim=0,
            )
        )

    return torch.stack(
        rows,
        dim=0,
    ).contiguous()


# ============================================================
# CUDA kernel
# ============================================================

CPP_SRC = r"""
#include <torch/extension.h>

void resident_k3(
    torch::Tensor packed,
    torch::Tensor out,
    int64_t S,
    int64_t chunk,
    int64_t threads
);

void resident_k4(
    torch::Tensor packed,
    torch::Tensor out,
    int64_t S,
    int64_t chunk,
    int64_t threads
);

void resident_k5(
    torch::Tensor packed,
    torch::Tensor out,
    int64_t S,
    int64_t chunk,
    int64_t threads
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
// K=3
//
// Per term layout:
//   F0 [4,48] : offsets   0 .. 191
//   F1 [2,4]  : offsets 192 .. 199
//   F2 [2,4]  : offsets 200 .. 207
//
// E = 208 floats / term.
// ============================================================

template<int CHUNK>
__global__ void resident_k3_kernel(
    const float* __restrict__ P,
    float* __restrict__ OUT,
    int S
) {
    constexpr int E = 208;

    extern __shared__ float sh[];

    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = e < NEL;

    int i = 0;
    int j = 0;

    int i0 = 0, i1 = 0, i2 = 0;
    int j0 = 0, j1 = 0, j2 = 0;

    if (active) {
        i = e / NCOL;
        j = e - i * NCOL;

        i0 = i / 4;
        i1 = (i / 2) & 1;
        i2 = i & 1;

        j0 = j / 16;
        j1 = (j / 4) & 3;
        j2 = j & 3;
    }

    float acc = 0.0f;

    for (int s0 = 0; s0 < S; s0 += CHUNK) {
        const int nr = min(CHUNK, S - s0);
        const int total = nr * E;

        // Cooperatively stage COMPLETE factors.
        for (
            int q = threadIdx.x;
            q < total;
            q += blockDim.x
        ) {
            const int r = q / E;
            const int u = q - r * E;

            sh[q] = P[
                (size_t)(s0 + r) * E
                + u
            ];
        }

        __syncthreads();

        if (active) {
            #pragma unroll 1
            for (int r = 0; r < nr; ++r) {
                const float* p = sh + r * E;

                const float a = p[
                    i0 * 48 + j0
                ];

                const float b = p[
                    192
                    + i1 * 4
                    + j1
                ];

                const float c = p[
                    200
                    + i2 * 4
                    + j2
                ];

                acc = fmaf(
                    a * b,
                    c,
                    acc
                );
            }
        }

        __syncthreads();
    }

    if (active) {
        OUT[e] = acc;
    }
}


// ============================================================
// K=4
//
// Per term layout:
//   F0 [2,12] :  0 .. 23
//   F1 [2,4]  : 24 .. 31
//   F2 [2,4]  : 32 .. 39
//   F3 [2,4]  : 40 .. 47
//
// E = 48 floats / term.
// ============================================================

template<int CHUNK>
__global__ void resident_k4_kernel(
    const float* __restrict__ P,
    float* __restrict__ OUT,
    int S
) {
    constexpr int E = 48;

    extern __shared__ float sh[];

    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = e < NEL;

    int i = 0;
    int j = 0;

    int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    int j0 = 0, j1 = 0, j2 = 0, j3 = 0;

    if (active) {
        i = e / NCOL;
        j = e - i * NCOL;

        i0 = i / 8;
        i1 = (i / 4) & 1;
        i2 = (i / 2) & 1;
        i3 = i & 1;

        j0 = j / 64;
        j1 = (j / 16) & 3;
        j2 = (j / 4) & 3;
        j3 = j & 3;
    }

    float acc = 0.0f;

    for (int s0 = 0; s0 < S; s0 += CHUNK) {
        const int nr = min(CHUNK, S - s0);
        const int total = nr * E;

        for (
            int q = threadIdx.x;
            q < total;
            q += blockDim.x
        ) {
            const int r = q / E;
            const int u = q - r * E;

            sh[q] = P[
                (size_t)(s0 + r) * E
                + u
            ];
        }

        __syncthreads();

        if (active) {
            #pragma unroll 1
            for (int r = 0; r < nr; ++r) {
                const float* p = sh + r * E;

                const float a = p[
                    i0 * 12 + j0
                ];

                const float b = p[
                    24
                    + i1 * 4
                    + j1
                ];

                const float c = p[
                    32
                    + i2 * 4
                    + j2
                ];

                const float d = p[
                    40
                    + i3 * 4
                    + j3
                ];

                acc = fmaf(
                    (a * b) * c,
                    d,
                    acc
                );
            }
        }

        __syncthreads();
    }

    if (active) {
        OUT[e] = acc;
    }
}


// ============================================================
// K=5
//
// Per term layout:
//   F0 [1,3] :  0 ..  2
//   F1 [2,4] :  3 .. 10
//   F2 [2,4] : 11 .. 18
//   F3 [2,4] : 19 .. 26
//   F4 [2,4] : 27 .. 34
//
// E = 35 floats / term.
// ============================================================

template<int CHUNK>
__global__ void resident_k5_kernel(
    const float* __restrict__ P,
    float* __restrict__ OUT,
    int S
) {
    constexpr int E = 35;

    extern __shared__ float sh[];

    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = e < NEL;

    int i = 0;
    int j = 0;

    int i1 = 0, i2 = 0, i3 = 0, i4 = 0;
    int j0 = 0, j1 = 0, j2 = 0, j3 = 0, j4 = 0;

    if (active) {
        i = e / NCOL;
        j = e - i * NCOL;

        i1 = i / 8;
        i2 = (i / 4) & 1;
        i3 = (i / 2) & 1;
        i4 = i & 1;

        j0 = j / 256;
        j1 = (j / 64) & 3;
        j2 = (j / 16) & 3;
        j3 = (j / 4) & 3;
        j4 = j & 3;
    }

    float acc = 0.0f;

    for (int s0 = 0; s0 < S; s0 += CHUNK) {
        const int nr = min(CHUNK, S - s0);
        const int total = nr * E;

        for (
            int q = threadIdx.x;
            q < total;
            q += blockDim.x
        ) {
            const int r = q / E;
            const int u = q - r * E;

            sh[q] = P[
                (size_t)(s0 + r) * E
                + u
            ];
        }

        __syncthreads();

        if (active) {
            #pragma unroll 1
            for (int r = 0; r < nr; ++r) {
                const float* p = sh + r * E;

                const float a = p[
                    j0
                ];

                const float b = p[
                    3
                    + i1 * 4
                    + j1
                ];

                const float c = p[
                    11
                    + i2 * 4
                    + j2
                ];

                const float d = p[
                    19
                    + i3 * 4
                    + j3
                ];

                const float f = p[
                    27
                    + i4 * 4
                    + j4
                ];

                acc = fmaf(
                    ((a * b) * c) * d,
                    f,
                    acc
                );
            }
        }

        __syncthreads();
    }

    if (active) {
        OUT[e] = acc;
    }
}


// ============================================================
// Launch helpers
// ============================================================

template<typename Kernel>
void optin_shared(
    Kernel kernel,
    int bytes
) {
    // Harmless when bytes are within the default limit.
    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        bytes
    );
}


#define LAUNCH_CASE(KERNEL, CH)                                      \
    case CH: {                                                       \
        const int shmem = CH * elems_per_term * sizeof(float);       \
        optin_shared(KERNEL<CH>, shmem);                             \
        KERNEL<CH><<<grid, block, shmem, stream>>>(                  \
            packed.data_ptr<float>(),                                \
            out.data_ptr<float>(),                                   \
            (int)S                                                   \
        );                                                           \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                              \
        return;                                                      \
    }


void check_common(
    torch::Tensor packed,
    torch::Tensor out,
    int64_t S,
    int64_t threads
) {
    TORCH_CHECK(
        packed.is_cuda(),
        "packed must be CUDA"
    );

    TORCH_CHECK(
        out.is_cuda(),
        "out must be CUDA"
    );

    TORCH_CHECK(
        packed.scalar_type() == torch::kFloat32,
        "packed must be float32"
    );

    TORCH_CHECK(
        out.scalar_type() == torch::kFloat32,
        "out must be float32"
    );

    TORCH_CHECK(
        packed.is_contiguous(),
        "packed must be contiguous"
    );

    TORCH_CHECK(
        out.is_contiguous(),
        "out must be contiguous"
    );

    TORCH_CHECK(
        out.numel() == NEL,
        "out must contain 16*768 entries"
    );

    TORCH_CHECK(
        packed.size(0) == S,
        "packed first dim must equal S"
    );

    TORCH_CHECK(
        threads == 128
        || threads == 256
        || threads == 512,
        "threads must be 128, 256, or 512"
    );
}


void resident_k3(
    torch::Tensor packed,
    torch::Tensor out,
    int64_t S,
    int64_t chunk,
    int64_t threads
) {
    constexpr int elems_per_term = 208;

    check_common(
        packed,
        out,
        S,
        threads
    );

    TORCH_CHECK(
        packed.size(1) == elems_per_term,
        "K3 packed width must be 208"
    );

    dim3 block((unsigned)threads);
    dim3 grid(
        (NEL + threads - 1)
        / threads
    );

    auto stream =
        at::cuda::getCurrentCUDAStream();

    switch ((int)chunk) {
        LAUNCH_CASE(
            resident_k3_kernel,
            8
        )
        LAUNCH_CASE(
            resident_k3_kernel,
            16
        )
        LAUNCH_CASE(
            resident_k3_kernel,
            32
        )
        LAUNCH_CASE(
            resident_k3_kernel,
            48
        )

        default:
            TORCH_CHECK(
                false,
                "K3 chunk must be 8/16/32/48"
            );
    }
}


void resident_k4(
    torch::Tensor packed,
    torch::Tensor out,
    int64_t S,
    int64_t chunk,
    int64_t threads
) {
    constexpr int elems_per_term = 48;

    check_common(
        packed,
        out,
        S,
        threads
    );

    TORCH_CHECK(
        packed.size(1) == elems_per_term,
        "K4 packed width must be 48"
    );

    dim3 block((unsigned)threads);
    dim3 grid(
        (NEL + threads - 1)
        / threads
    );

    auto stream =
        at::cuda::getCurrentCUDAStream();

    switch ((int)chunk) {
        LAUNCH_CASE(
            resident_k4_kernel,
            32
        )
        LAUNCH_CASE(
            resident_k4_kernel,
            64
        )
        LAUNCH_CASE(
            resident_k4_kernel,
            128
        )
        LAUNCH_CASE(
            resident_k4_kernel,
            256
        )

        default:
            TORCH_CHECK(
                false,
                "K4 chunk must be 32/64/128/256"
            );
    }
}


void resident_k5(
    torch::Tensor packed,
    torch::Tensor out,
    int64_t S,
    int64_t chunk,
    int64_t threads
) {
    constexpr int elems_per_term = 35;

    check_common(
        packed,
        out,
        S,
        threads
    );

    TORCH_CHECK(
        packed.size(1) == elems_per_term,
        "K5 packed width must be 35"
    );

    dim3 block((unsigned)threads);
    dim3 grid(
        (NEL + threads - 1)
        / threads
    );

    auto stream =
        at::cuda::getCurrentCUDAStream();

    switch ((int)chunk) {
        LAUNCH_CASE(
            resident_k5_kernel,
            32
        )
        LAUNCH_CASE(
            resident_k5_kernel,
            64
        )
        LAUNCH_CASE(
            resident_k5_kernel,
            128
        )
        LAUNCH_CASE(
            resident_k5_kernel,
            256
        )

        default:
            TORCH_CHECK(
                false,
                "K5 chunk must be 32/64/128/256"
            );
    }
}

#undef LAUNCH_CASE
"""


# ============================================================
# Compile extension
# ============================================================

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
        "exact_w_resident_kron_build",
    )

    shutil.rmtree(
        build_dir,
        ignore_errors=True,
    )

    os.makedirs(
        build_dir,
        exist_ok=True,
    )

    print()
    print(
        "building CUDA extension in",
        build_dir,
        "...",
    )

    ext = load_inline(
        name="exact_w_resident_kron_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=[
            "resident_k3",
            "resident_k4",
            "resident_k5",
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
        ],
        build_directory=build_dir,
        verbose=False,
    )

    print("build OK")
    return ext


ext = build_extension()


# ============================================================
# Benchmark helpers
# ============================================================

def sync():
    torch.cuda.synchronize()


def median_us(
    fn,
    warmup=10,
    iters=200,
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
# Dispatch
# ============================================================

CHUNKS = {
    3: [8, 16, 32, 48],
    4: [32, 64, 128, 256],
    5: [32, 64, 128, 256],
}

THREADS = [
    128,
    256,
    512,
]


def launch(
    K,
    packed,
    out,
    S,
    chunk,
    threads,
):
    if K == 3:
        ext.resident_k3(
            packed,
            out,
            S,
            chunk,
            threads,
        )

    elif K == 4:
        ext.resident_k4(
            packed,
            out,
            S,
            chunk,
            threads,
        )

    elif K == 5:
        ext.resident_k5(
            packed,
            out,
            S,
            chunk,
            threads,
        )

    else:
        raise ValueError(K)


@torch.no_grad()
def autotune(
    K,
    packed,
    out,
    S,
):
    rows = []

    for chunk in CHUNKS[K]:
        for threads in THREADS:
            try:
                launch(
                    K,
                    packed,
                    out,
                    S,
                    chunk,
                    threads,
                )

                sync()

                us = median_us(
                    lambda: launch(
                        K,
                        packed,
                        out,
                        S,
                        chunk,
                        threads,
                    ),
                    warmup=5,
                    iters=100,
                    repeats=5,
                )

                rows.append(
                    {
                        "K": K,
                        "chunk": chunk,
                        "threads": threads,
                        "us": us,
                    }
                )

            except Exception as e:
                rows.append(
                    {
                        "K": K,
                        "chunk": chunk,
                        "threads": threads,
                        "error": repr(e),
                    }
                )

    valid = [
        r for r in rows
        if "us" in r
    ]

    if not valid:
        raise RuntimeError(
            f"No valid K={K} configs: {rows}"
        )

    best = min(
        valid,
        key=lambda x: x["us"],
    )

    return best, rows


# ============================================================
# Correctness reference
# ============================================================

@torch.no_grad()
def vectorized_reconstruct(
    factors,
):
    term = factors[0]

    for f in factors[1:]:
        term = batched_kron(
            term,
            f,
        )

    return term.sum(
        dim=0
    )


# ============================================================
# Run
# ============================================================

@torch.no_grad()
def run_case(S):
    reps = make_equivalent_factors(
        S,
        seed=17,
    )

    packed = {
        K: flatten_pack(
            reps[K]
        )
        for K in [3, 4, 5]
    }

    out = {
        K: torch.empty(
            16,
            768,
            device=DEV,
            dtype=DT,
        )
        for K in [3, 4, 5]
    }

    best = {}
    all_cfg = {}

    for K in [3, 4, 5]:
        b, rows = autotune(
            K,
            packed[K],
            out[K],
            S,
        )

        best[K] = b
        all_cfg[K] = rows

        launch(
            K,
            packed[K],
            out[K],
            S,
            b["chunk"],
            b["threads"],
        )

    sync()

    ref = vectorized_reconstruct(
        reps[5]
    )

    sync()

    err = {
        K: rel_l2(
            out[K],
            ref,
        )
        for K in [3, 4, 5]
    }

    result = {
        "S": S,

        "k3_us": best[3]["us"],
        "k4_us": best[4]["us"],
        "k5_us": best[5]["us"],

        "k3_chunk": best[3]["chunk"],
        "k4_chunk": best[4]["chunk"],
        "k5_chunk": best[5]["chunk"],

        "k3_threads": best[3]["threads"],
        "k4_threads": best[4]["threads"],
        "k5_threads": best[5]["threads"],

        "k3_over_k5":
            best[3]["us"]
            / best[5]["us"],

        "k4_over_k5":
            best[4]["us"]
            / best[5]["us"],

        "err3": err[3],
        "err4": err[4],
        "err5": err[5],

        "all_cfg": all_cfg,
    }

    del reps, packed, out, ref
    torch.cuda.empty_cache()

    return result


def main():
    print()
    print("=" * 132)
    print(
        "EXACT-W ON-CHIP-RESIDENT "
        "TINY-FACTOR BENCHMARK"
    )
    print("=" * 132)

    print()
    print("Per-separation-term factor footprint:")
    print("  K=3 : 208 floats = 832 bytes")
    print("  K=4 :  48 floats = 192 bytes")
    print("  K=5 :  35 floats = 140 bytes")

    print()
    print("At S=256:")
    print("  K=3 : 208 KB")
    print("  K=4 :  48 KB")
    print("  K=5 :  35 KB")

    print()
    print(
        f"{'S':>4} | "
        f"{'K3':>10} "
        f"{'K4':>10} "
        f"{'K5':>10} | "
        f"{'K3/K5':>8} "
        f"{'K4/K5':>8} | "
        f"{'K3 cfg':>13} "
        f"{'K4 cfg':>13} "
        f"{'K5 cfg':>13} | "
        f"{'max err':>10}"
    )

    print("-" * 132)

    results = []

    for S in [
        8,
        16,
        32,
        64,
        128,
        256,
    ]:
        r = run_case(S)
        results.append(r)

        mxerr = max(
            r["err3"],
            r["err4"],
            r["err5"],
        )

        print(
            f"{S:4d} | "
            f"{r['k3_us']:8.2f}us "
            f"{r['k4_us']:8.2f}us "
            f"{r['k5_us']:8.2f}us | "
            f"{r['k3_over_k5']:7.3f}x "
            f"{r['k4_over_k5']:7.3f}x | "
            f"{r['k3_threads']:3d}t/"
            f"{r['k3_chunk']:<3d}r "
            f"{r['k4_threads']:3d}t/"
            f"{r['k4_chunk']:<3d}r "
            f"{r['k5_threads']:3d}t/"
            f"{r['k5_chunk']:<3d}r | "
            f"{mxerr:10.2e}"
        )

    print()
    print("=" * 132)
    print("INTERPRETATION")
    print("=" * 132)

    print(
        r"""
This is the causal test we actually want.

Script 1:
    one output entry per thread, direct global factor loads.

    Result:
        K=5 was not inherently faster than K=3.

    Interpretation:
        merely increasing K does not create useful parallelism once
        all output entries are already independent.

Script 2:
    complete factor chunks are made CTA-resident and reused.

    Now factor GRANULARITY changes the amount of structured state that
    can reside on-chip at once:

        K=3: 208 floats / term
        K=4:  48 floats / term
        K=5:  35 floats / term

The key quantity is:

        K3 latency / K5 latency

especially at S=64, 128, 256.

If this ratio rises above 1 as S increases, the evidence supports the
actual systems claim:

    fine Kronecker factorization reduces the factor working set enough
    to increase on-chip residency and reuse.

Do NOT describe this as literally assigning one matrix to one CUDA core.

A precise description is:

    Tiny Kronecker factors make the per-term structured state small
    enough to be staged on-chip and reused by many CUDA threads, while
    independent output products provide the thread-level parallelism
    scheduled across CUDA cores.
"""
    )


if __name__ == "__main__":
    main()


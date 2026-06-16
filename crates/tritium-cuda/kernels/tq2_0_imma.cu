// IMMA int8 ternary mpGEMM — compute-bound prefill kernel (v0.30, ADR 0005 / WF-A).
//
// SKELETON. The tiled add-only kernel (`tq2_0_add.cu`) wins memory-bound decode
// (batch=1); this kernel targets the compute-bound prefill (large M) with the
// int8 tensor cores: int8 activations × ternary weights via `mma.m16n8k32`
// (`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`), 16x8x32 tiles, ternary
// weights unpacked to int8 in double-buffered shared memory. Mirrors bitnet.cpp's
// W1.58A8 GPU kernel / BitBLAS / GPTQ-Marlin.
//
// REQUIRES sm_80+ (Ampere): the `m16n8k32` int8 `mma` shape is not available on
// sm_75 (Turing) — so WF-A compiles THIS kernel for compute_80 (not the
// compute_75 floor `tq2_0_add.cu` uses), and `build.rs` must emit a second PTX
// target accordingly. It is intentionally NOT yet wired into `build.rs`'s nvcc
// invocation; WF-A adds the compile + the host launch + the autotune/codegen
// (`src/autotune.rs`, `src/codegen.rs`) integration.
//
// Contract (unchanged from `reference_mpgemm`, fused per ADR 0005):
//   out[m,n] = act_scale[m] * weight_scale[n] * sum_k qact[m,k] * trit[n,k]
// where qact is the per-token int8 absmax quant (W1.58A8, Qp=127). Correctness is
// held to the vs-reference + cross-kernel parity gate (IMMA == add-only == ref).

#include <cuda_runtime.h>

extern "C" {

// Placeholder entry point so the file is a valid translation unit. WF-A replaces
// this with the tiled `mma.m16n8k32` kernel (template parameters supplied by the
// autotune `TileConfig`). Parameters are illustrative and will change.
__global__ void tq2_0_imma_placeholder(const signed char* /*qact*/,
                                       const unsigned char* /*qweight_i2sint8*/,
                                       const float* /*act_scale*/,
                                       const float* /*weight_scale*/,
                                       float* /*out*/, int /*M*/, int /*N*/,
                                       int /*K*/) {
  // WF-A: implement the IMMA tiled contraction here.
}

}  // extern "C"

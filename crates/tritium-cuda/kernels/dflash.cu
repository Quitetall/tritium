// DFlash block-diffusion drafter for the resident Qwen3.6 executor.
//
// The drafter (z-lab Qwen3.6-27B-DFlash: five Qwen3 layers conditioned on target
// hidden states) proposes a block of tokens that the target then verifies in one
// batched pass. Its numerics never reach the output -- the target accepts a
// drafted token only if it equals the target's own greedy token -- so these
// kernels trade precision for speed freely: int8 weights with one f32 scale per
// row, f16 activations into the tensor cores, CUDA transcendentals.
//
// Built for compute_80 (`mma.m16n8k16`).

#include <cuda_fp16.h>
#include <stdint.h>

namespace dflash {

constexpr uint32_t kChunk = 64;  // K columns per weight load (16 per quad thread)
constexpr uint32_t kWarps = 4;   // warps per block, splitting K

__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two int8 weights as an f16 pair, exactly: bias each byte to unsigned, place it
// in the low mantissa byte of 1024.0 (0x64XX), subtract 1024 + 128.
__device__ __forceinline__ uint32_t int8_pair_to_half2(uint32_t biased, uint32_t select) {
  const uint32_t bits = __byte_perm(biased, 0x64646464U, select);
  const __half2 value = __hsub2(*reinterpret_cast<const __half2*>(&bits), __float2half2_rn(1152.0f));
  return *reinterpret_cast<const uint32_t*>(&value);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xFFFFFFFFU, value, offset);
  return value;
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = (blockDim.x + 31) >> 5;
  value = warp_sum(value);
  __syncthreads();
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  float total = 0.0f;
  for (int index = 0; index < warps; ++index) total += scratch[index];
  return total;
}

__device__ __forceinline__ float block_max(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = (blockDim.x + 31) >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, offset));
  }
  __syncthreads();
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  float best = -INFINITY;
  for (int index = 0; index < warps; ++index) best = fmaxf(best, scratch[index]);
  return best;
}

// Y[m, n] = X[m, k] . (W[n, k] * scale[n]) for at most 8 * NT rows of X.
//
// A block of `kWarps` warps owns sixteen output rows; warp w takes every
// kWarps-th 64-column chunk and the block sums the warps' partials in shared
// memory. Within a chunk, quad thread c holds columns 16c .. 16c + 15, which the
// activation prepass lays out to match.
template <uint32_t NT>
__device__ __forceinline__ void w8_rows(const int8_t* __restrict__ weight,
                                        const float* __restrict__ scale,
                                        const __half* __restrict__ act,
                                        float* __restrict__ out,
                                        uint32_t n,
                                        uint32_t k,
                                        uint32_t m,
                                        uint32_t out_stride) {
  __shared__ float partial[kWarps][NT][4][32];
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp = threadIdx.x >> 5U;
  const uint32_t g = lane >> 2U;
  const uint32_t c = lane & 3U;
  const uint32_t row0 = blockIdx.x * 16U + g;
  const uint32_t n_tiles = (m + 7U) / 8U;
  const uint32_t chunks = k / kChunk;

  float acc[NT][4];
#pragma unroll
  for (uint32_t t = 0; t < NT; ++t) {
#pragma unroll
    for (uint32_t e = 0; e < 4U; ++e) acc[t][e] = 0.0f;
  }
  const int8_t* w0 = weight + static_cast<size_t>(row0) * k + 16U * c;
  const int8_t* w1 = w0 + static_cast<size_t>(8U) * k;
  for (uint32_t chunk = warp; chunk < chunks; chunk += kWarps) {
    const uint4 r0 = __ldcs(reinterpret_cast<const uint4*>(w0 + chunk * kChunk));
    const uint4 r1 = __ldcs(reinterpret_cast<const uint4*>(w1 + chunk * kChunk));
    const uint32_t words0[4] = {r0.x ^ 0x80808080U, r0.y ^ 0x80808080U, r0.z ^ 0x80808080U,
                                r0.w ^ 0x80808080U};
    const uint32_t words1[4] = {r1.x ^ 0x80808080U, r1.y ^ 0x80808080U, r1.z ^ 0x80808080U,
                                r1.w ^ 0x80808080U};
#pragma unroll
    for (uint32_t s = 0; s < 4U; ++s) {
      uint32_t a[4];
      a[0] = int8_pair_to_half2(words0[s], 0x5140U);
      a[1] = int8_pair_to_half2(words1[s], 0x5140U);
      a[2] = int8_pair_to_half2(words0[s], 0x5342U);
      a[3] = int8_pair_to_half2(words1[s], 0x5342U);
      const size_t slot = (static_cast<size_t>(chunk) * 16U + s * 4U + c) * 4U;
#pragma unroll
      for (uint32_t t = 0; t < NT; ++t) {
        if (t >= n_tiles) break;
        const uint2 b = __ldg(reinterpret_cast<const uint2*>(act + static_cast<size_t>(t * 8U + g) * k + slot));
        mma16816(acc[t], a, b.x, b.y);
      }
    }
  }
#pragma unroll
  for (uint32_t t = 0; t < NT; ++t) {
#pragma unroll
    for (uint32_t e = 0; e < 4U; ++e) partial[warp][t][e][lane] = acc[t][e];
  }
  __syncthreads();
  if (warp != 0U) return;
  const float s0 = scale[row0];
  const float s1 = scale[row0 + 8U];
#pragma unroll
  for (uint32_t t = 0; t < NT; ++t) {
    if (t >= n_tiles) break;
#pragma unroll
    for (uint32_t e = 0; e < 4U; ++e) {
      float value = 0.0f;
#pragma unroll
      for (uint32_t w = 0; w < kWarps; ++w) value += partial[w][t][e][lane];
      const uint32_t token = t * 8U + 2U * c + (e & 1U);
      if (token >= m) continue;
      const uint32_t row = e < 2U ? row0 : row0 + 8U;
      out[static_cast<size_t>(token) * out_stride + row] = value * (e < 2U ? s0 : s1);
    }
  }
}

// Y = X . W for 4-bit weights: each 64-column chunk of a row has one f16 scale
// and 32 bytes of offset-8 nibbles. Quad thread c owns columns 16c .. 16c + 15 of
// a chunk as two words; word w holds columns 8w .. 8w + 7 (of the 16) with
// column 2k in bits 4k .. 4k + 3 and column 2k + 1 in bits 16 + 4k .. 19 + 4k,
// so `(word >> 4k) & 0x000F000F` is a column pair.
__device__ __forceinline__ uint32_t nibble_pair(uint32_t word, uint32_t k, __half2 scale) {
  const uint32_t bits = ((word >> (4U * k)) & 0x000F000FU) | 0x64006400U;
  const __half2 value = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&bits),
                                        __float2half2_rn(1032.0f)),
                                scale);
  return *reinterpret_cast<const uint32_t*>(&value);
}

// Block-staged 4-bit GEMM: a block of eight warps owns 128 output rows (sixteen
// per warp) and K-split `blockIdx.y` of `gridDim.y` in 256-column chunks. Each
// chunk's activation tile is staged once in shared memory for all eight warps
// (a warp-per-16-rows kernel re-read the activations from L2, four times the
// weight bytes). With one split the block writes `out`; otherwise
// `partial[split][m][n]` for `dflash_reduce`.
constexpr uint32_t kBlockWarps = 8U;
constexpr uint32_t kStageCols = 256U;
constexpr uint32_t kStageStride = kStageCols + 8U;  // halves; distinct banks per row

template <uint32_t NT>
__device__ __forceinline__ void w4_block(const uint8_t* __restrict__ weight,
                                         const __half* __restrict__ scale,
                                         const __half* __restrict__ act,
                                         float* __restrict__ out,
                                         float* __restrict__ partial,
                                         uint32_t n,
                                         uint32_t k,
                                         uint32_t m,
                                         uint32_t out_stride) {
  __shared__ __align__(16) __half staged[NT * 8U][kStageStride];
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp = threadIdx.x >> 5U;
  const uint32_t g = lane >> 2U;
  const uint32_t c = lane & 3U;
  const uint32_t row0 = (blockIdx.x * kBlockWarps + warp) * 16U + g;
  const bool active = (blockIdx.x * kBlockWarps + warp) * 16U < n;
  const uint32_t n_tiles = (m + 7U) / 8U;
  const uint32_t chunks = k / kStageCols;
  const uint32_t split = blockIdx.y;
  const uint32_t splits = gridDim.y;
  const uint32_t first = chunks * split / splits;
  const uint32_t last = chunks * (split + 1U) / splits;
  const uint32_t groups = k / kChunk;

  float acc[NT][4];
#pragma unroll
  for (uint32_t t = 0; t < NT; ++t) {
#pragma unroll
    for (uint32_t e = 0; e < 4U; ++e) acc[t][e] = 0.0f;
  }
  const uint8_t* w0 = weight + static_cast<size_t>(row0) * (k / 2U) + 8U * c;
  const uint8_t* w1 = w0 + static_cast<size_t>(8U) * (k / 2U);
  const __half* s0 = scale + static_cast<size_t>(row0) * groups;
  const __half* s1 = s0 + static_cast<size_t>(8U) * groups;
  const uint32_t staged_rows = n_tiles * 8U;
  // The chunk's weights (64 bytes and 8 scales per thread) are loaded before the
  // activation staging and the next chunk's while this one computes: one 8-byte
  // load in flight per thread left the kernel at a quarter of DRAM bandwidth.
  uint2 r0[4], r1[4];
  __half s0v[4], s1v[4];
  auto load = [&](uint32_t chunk) {
#pragma unroll
    for (uint32_t q = 0; q < 4U; ++q) {
      const uint32_t group = chunk * 4U + q;
      r0[q] = __ldcs(reinterpret_cast<const uint2*>(w0 + group * 32U));
      r1[q] = __ldcs(reinterpret_cast<const uint2*>(w1 + group * 32U));
      s0v[q] = s0[group];
      s1v[q] = s1[group];
    }
  };
  if (active && first < last) load(first);
  for (uint32_t chunk = first; chunk < last; ++chunk) {
    __syncthreads();
    // 512 bytes per staged row, as 32 16-byte copies.
    for (uint32_t index = threadIdx.x; index < staged_rows * 32U; index += blockDim.x) {
      const uint32_t row = index / 32U;
      const uint32_t part = index % 32U;
      const uint4 value = __ldg(reinterpret_cast<const uint4*>(
          act + static_cast<size_t>(row) * k + static_cast<size_t>(chunk) * kStageCols + part * 8U));
      *reinterpret_cast<uint4*>(&staged[row][part * 8U]) = value;
    }
    __syncthreads();
    if (!active) continue;
    uint2 c0[4], c1[4];
    __half2 sc0[4], sc1[4];
#pragma unroll
    for (uint32_t q = 0; q < 4U; ++q) {
      c0[q] = r0[q];
      c1[q] = r1[q];
      sc0[q] = __half2half2(s0v[q]);
      sc1[q] = __half2half2(s1v[q]);
    }
    if (chunk + 1U < last) load(chunk + 1U);
#pragma unroll
    for (uint32_t q = 0; q < 4U; ++q) {
#pragma unroll
      for (uint32_t s = 0; s < 4U; ++s) {
        const uint32_t word0 = s < 2U ? c0[q].x : c0[q].y;
        const uint32_t word1 = s < 2U ? c1[q].x : c1[q].y;
        const uint32_t pair = (s & 1U) * 2U;
        uint32_t a[4];
        a[0] = nibble_pair(word0, pair, sc0[q]);
        a[1] = nibble_pair(word1, pair, sc1[q]);
        a[2] = nibble_pair(word0, pair + 1U, sc0[q]);
        a[3] = nibble_pair(word1, pair + 1U, sc1[q]);
        const uint32_t column = q * 64U + (s * 4U + c) * 4U;
#pragma unroll
        for (uint32_t t = 0; t < NT; ++t) {
          if (t >= n_tiles) break;
          const uint2 b = *reinterpret_cast<const uint2*>(&staged[t * 8U + g][column]);
          mma16816(acc[t], a, b.x, b.y);
        }
      }
    }
  }
  if (!active) return;
#pragma unroll
  for (uint32_t t = 0; t < NT; ++t) {
    if (t >= n_tiles) break;
#pragma unroll
    for (uint32_t e = 0; e < 4U; ++e) {
      const uint32_t token = t * 8U + 2U * c + (e & 1U);
      if (token >= m) continue;
      const uint32_t row = e < 2U ? row0 : row0 + 8U;
      if (splits == 1U) {
        out[static_cast<size_t>(token) * out_stride + row] = acc[t][e];
      } else {
        partial[(static_cast<size_t>(split) * m + token) * n + row] = acc[t][e];
      }
    }
  }
}

}  // namespace dflash

// f32 activation rows -> f16 in the W8 GEMM's slot order: slot
// ((chunk * 16 + s * 4 + c) * 4 + j) holds column chunk * 64 + 16c + 4s + j.
// Rows in [m, gridDim.x) are zero.
extern "C" __global__ void dflash_split_act(const float* __restrict__ x,
                                            __half* __restrict__ out,
                                            uint32_t m,
                                            uint32_t k) {
  const uint32_t row = blockIdx.x;
  for (uint32_t slot = threadIdx.x; slot < k; slot += blockDim.x) {
    const uint32_t chunk = slot / 64U;
    const uint32_t within = slot % 64U;
    const uint32_t s = within / 16U;
    const uint32_t c = (within / 4U) % 4U;
    const uint32_t j = within % 4U;
    const uint32_t column = chunk * 64U + 16U * c + 4U * s + j;
    out[static_cast<size_t>(row) * k + slot] =
        __float2half_rn(row < m ? x[static_cast<size_t>(row) * k + column] : 0.0f);
  }
}

extern "C" __global__ void __launch_bounds__(128) dflash_w8_rows16(const int8_t* __restrict__ weight,
                                                                   const float* __restrict__ scale,
                                                                   const __half* __restrict__ act,
                                                                   float* __restrict__ out,
                                                                   uint32_t n,
                                                                   uint32_t k,
                                                                   uint32_t m,
                                                                   uint32_t out_stride) {
  dflash::w8_rows<2U>(weight, scale, act, out, n, k, m, out_stride);
}

extern "C" __global__ void __launch_bounds__(128) dflash_w8_rows64(const int8_t* __restrict__ weight,
                                                                   const float* __restrict__ scale,
                                                                   const __half* __restrict__ act,
                                                                   float* __restrict__ out,
                                                                   uint32_t n,
                                                                   uint32_t k,
                                                                   uint32_t m,
                                                                   uint32_t out_stride) {
  dflash::w8_rows<8U>(weight, scale, act, out, n, k, m, out_stride);
}

// Layer-major captured features `[layers, rows, hidden]` (row stride `stride`
// rows per layer) -> row-major `[rows, layers * hidden]`.
extern "C" __global__ void dflash_concat_features(const float* __restrict__ features,
                                                  float* __restrict__ out,
                                                  uint32_t layers,
                                                  uint32_t rows,
                                                  uint32_t hidden,
                                                  uint32_t stride) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = static_cast<uint64_t>(layers) * rows * hidden;
  if (index >= total) return;
  const uint32_t column = static_cast<uint32_t>(index % hidden);
  const uint32_t row = static_cast<uint32_t>((index / hidden) % rows);
  const uint32_t layer = static_cast<uint32_t>(index / (static_cast<uint64_t>(hidden) * rows));
  out[(static_cast<size_t>(row) * layers + layer) * hidden + column] =
      features[(static_cast<size_t>(layer) * stride + row) * hidden + column];
}

// Qwen3 head norm (x / rms(x) * w over each 128-wide head) then full NeoX RoPE at
// position `position0 + row`. Block (head, row), one thread per lane.
//
// `in` rows hold the heads at float offset `in_offset` (row stride `in_stride`);
// head `h` of row `row` goes to `out[row * out_stride + h * 128 ..]`. For keys the
// caller points `out` at the cache row of `position0`. With `value_cache` set,
// the same heads at `value_offset` in `in` are copied unnormalized to it with
// `out`'s layout.
extern "C" __global__ void dflash_norm_rope(const float* __restrict__ in,
                                            uint32_t in_stride,
                                            uint32_t in_offset,
                                            const float* __restrict__ norm,
                                            const float* __restrict__ inv_freq,
                                            float* __restrict__ out,
                                            uint32_t out_stride,
                                            int position0,
                                            float eps,
                                            float* __restrict__ value_cache,
                                            uint32_t value_offset) {
  __shared__ float lanes[128];
  __shared__ float scratch[32];
  const uint32_t head = blockIdx.x;
  const uint32_t row = blockIdx.y;
  const uint32_t lane = threadIdx.x;
  const float* source = in + static_cast<size_t>(row) * in_stride + in_offset + head * 128U;
  const float x = source[lane];
  const float inverse = rsqrtf(dflash::block_sum(x * x, scratch) / 128.0f + eps);
  lanes[lane] = x * inverse * norm[lane];
  __syncthreads();
  const int position = position0 + static_cast<int>(row);
  const uint32_t pair = lane % 64U;
  float sine;
  float cosine;
  sincosf(static_cast<float>(position) * inv_freq[pair], &sine, &cosine);
  const float a = lanes[pair];
  const float b = lanes[pair + 64U];
  const float rotated = lane < 64U ? a * cosine - b * sine : b * cosine + a * sine;
  out[static_cast<size_t>(row) * out_stride + head * 128U + lane] = rotated;
  if (value_cache != nullptr) {
    value_cache[static_cast<size_t>(row) * out_stride + head * 128U + lane] =
        in[static_cast<size_t>(row) * in_stride + value_offset + head * 128U + lane];
  }
}

// Drafter attention: query row `r` (block (head, r)) over cached keys
// [first(r), last(r)). Sliding causal layers see keys up to their own position
// within `window`; the full layer sees every cached key and the whole block.
extern "C" __global__ void dflash_attention(const float* __restrict__ query,
                                            const float* __restrict__ key_cache,
                                            const float* __restrict__ value_cache,
                                            float* __restrict__ out,
                                            int position0,
                                            int block_rows,
                                            int causal,
                                            int window,
                                            int heads,
                                            int kv_heads,
                                            float scale) {
  extern __shared__ float scores[];
  __shared__ float scratch[32];
  const int head = blockIdx.x;
  const int row = blockIdx.y;
  const int kv_head = head / (heads / kv_heads);
  const int kv_width = kv_heads * 128;
  const int own = position0 + row;
  const int last = causal ? own + 1 : position0 + block_rows;
  const int first = causal && window > 0 ? max(0, own + 1 - window) : 0;
  const int span = last - first;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const float* q = query + static_cast<size_t>(row) * heads * 128 + head * 128;

  for (int index = warp; index < span; index += warps) {
    const float* k = key_cache + static_cast<size_t>(first + index) * kv_width + kv_head * 128;
    float dot = 0.0f;
    for (int d = lane; d < 128; d += 32) dot += q[d] * k[d];
    dot = dflash::warp_sum(dot);
    if (lane == 0) scores[index] = dot * scale;
  }
  __syncthreads();
  float local = -INFINITY;
  for (int index = threadIdx.x; index < span; index += blockDim.x) local = fmaxf(local, scores[index]);
  const float peak = dflash::block_max(local, scratch);
  float sum = 0.0f;
  for (int index = threadIdx.x; index < span; index += blockDim.x) {
    const float weight = __expf(scores[index] - peak);
    scores[index] = weight;
    sum += weight;
  }
  const float inverse = 1.0f / dflash::block_sum(sum, scratch);
  __syncthreads();
  for (int d = threadIdx.x; d < 128; d += blockDim.x) {
    float total = 0.0f;
    for (int index = 0; index < span; ++index) {
      total += scores[index] *
               value_cache[static_cast<size_t>(first + index) * kv_width + kv_head * 128 + d];
    }
    out[static_cast<size_t>(row) * heads * 128 + head * 128 + d] = total * inverse;
  }
}

// out[r][j] = silu(gate_up[r][j]) * gate_up[r][width + j]: the SwiGLU product of a
// fused gate|up projection, `rows` rows of `2 * width`.
extern "C" __global__ void dflash_swiglu_fused(const float* __restrict__ gate_up,
                                               float* __restrict__ out,
                                               uint32_t width,
                                               uint32_t rows) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<uint64_t>(width) * rows) return;
  const uint32_t row = static_cast<uint32_t>(index / width);
  const uint32_t column = static_cast<uint32_t>(index % width);
  const float gate = gate_up[static_cast<size_t>(row) * 2U * width + column];
  const float up = gate_up[static_cast<size_t>(row) * 2U * width + width + column];
  out[index] = gate / (1.0f + __expf(-gate)) * up;
}

extern "C" __global__ void __launch_bounds__(256) dflash_w4_block16(const uint8_t* __restrict__ weight,
                                                                    const __half* __restrict__ scale,
                                                                    const __half* __restrict__ act,
                                                                    float* __restrict__ out,
                                                                    float* __restrict__ partial,
                                                                    uint32_t n,
                                                                    uint32_t k,
                                                                    uint32_t m,
                                                                    uint32_t out_stride) {
  dflash::w4_block<2U>(weight, scale, act, out, partial, n, k, m, out_stride);
}

extern "C" __global__ void __launch_bounds__(256) dflash_w4_block64(const uint8_t* __restrict__ weight,
                                                                    const __half* __restrict__ scale,
                                                                    const __half* __restrict__ act,
                                                                    float* __restrict__ out,
                                                                    float* __restrict__ partial,
                                                                    uint32_t n,
                                                                    uint32_t k,
                                                                    uint32_t m,
                                                                    uint32_t out_stride) {
  dflash::w4_block<8U>(weight, scale, act, out, partial, n, k, m, out_stride);
}

// Sum K-split partials `[splits, m, n]` in split order into `out` (row stride
// `out_stride`).
extern "C" __global__ void dflash_reduce(const float* __restrict__ partial,
                                         float* __restrict__ out,
                                         uint32_t n,
                                         uint32_t m,
                                         uint32_t splits,
                                         uint32_t out_stride) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<uint64_t>(m) * n) return;
  const uint32_t token = static_cast<uint32_t>(index / n);
  const uint32_t row = static_cast<uint32_t>(index % n);
  float sum = 0.0f;
  for (uint32_t s = 0; s < splits; ++s) sum += partial[(static_cast<size_t>(s) * m + token) * n + row];
  out[static_cast<size_t>(token) * out_stride + row] = sum;
}

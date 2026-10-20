// SALT V2 B3 x f32 GEMM on the f16 tensor cores (prefill and verification).
//
// The row-stream GEMV (`salt_v2_stream_rows_f32`) applies each decoded weight word
// to every activation row through CUDA-core FMAs, and pays the activation's L1
// traffic once per row: at M = 25 a row still cost 0.69 of a whole M = 1 pass.
// Here a warp owns sixteen weight rows, decodes each word once into tensor-core
// fragments, and multiplies it against up to 64 activation rows with
// `mma.m16n8k16`.
//
// **Numerics.** Within the RFC 0001 fast tier, not bit-identical to the GEMV.
// A weight is decoded exactly in f32 -- sum over planes of (f16 scale) x trit -- and
// split into f16 `hi + lo`; an activation, pre-scaled per row by a power of two so
// it cannot overflow f16, is split the same way. Three products (hi.hi, hi.lo,
// lo.hi) accumulate in f32, so each product carries ~22 significant bits; only the
// K-sum order differs from the GEMV. The order is fixed by the shape alone (rows,
// K, and the split count, never M), so a row's result does not depend on which
// other rows share the batch: prefill and verification are batch-invariant.
//
// **K permutation.** An MMA sums its sixteen k-slots in hardware, so which column
// sits in which slot is free as long as both operands agree. Quad thread `c` owns
// byte `c` of every payload word. Words 0-11 make three groups of four words and
// five k16 steps each: steps 5G .. 5G + 3 take digits 0-3 of byte c of word
// 4G + s, step 5G + 4 takes digit 4 of byte c of words 4G .. 4G + 3. That keeps
// one word's weights live at a time. Word 12's sixteen real trits are step 15,
// four per thread. `salt_v2_mma_split_act` writes activations in the same order.
//
// Requires B3, scale group 128, `k % 256 == 0`, and every fused tensor's row count
// a multiple of 16; the host checks all four.

#include <cuda_fp16.h>
#include <stdint.h>

namespace salt_mma {

constexpr uint32_t kTile = 256;
constexpr uint32_t kRankStrideTiles = 256;
// Activation rows one warp multiplies: eight n8 MMA tiles.
constexpr uint32_t kMaxRows = 64;

// Mirrors `SaltStreamTensor` in salt_v2.cu and `salt_v2_runtime.rs`.
struct Tensor {
  unsigned long long payload;
  unsigned long long scales;
  unsigned long long index_metadata;
  unsigned long long output;
  uint32_t rows;
  uint32_t tile_count;
  uint32_t plane_count;
  uint32_t allocation_map_bytes;
  uint32_t rank_prefix_count;
  uint32_t terminal_map_value;
  uint32_t first_row;
  uint32_t reserved;
};

__device__ __forceinline__ uint32_t planes_of(const unsigned char* meta,
                                              uint32_t map_bytes,
                                              uint32_t terminal,
                                              uint32_t tile) {
  const uint64_t bit = static_cast<uint64_t>(tile) * 2U;
  const uint64_t allocated = static_cast<uint64_t>(map_bytes) * 8U;
  const uint32_t code = bit < allocated ? (meta[bit >> 3] >> (bit & 7U)) & 3U
                                        : (terminal >> (bit - allocated)) & 3U;
  return code < 3U ? code + 1U : 0U;
}

__device__ __forceinline__ uint32_t rank_prefix(const unsigned char* meta,
                                                uint32_t map_bytes,
                                                uint32_t index) {
  const unsigned char* bytes = meta + map_bytes + index * 4U;
  return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8U) |
         (static_cast<uint32_t>(bytes[2]) << 16U) | (static_cast<uint32_t>(bytes[3]) << 24U);
}

// Plane rank of `tile`'s first plane: the stored prefix of its rank block plus
// the planes of the tiles before it in that block. The four threads of a quad
// split the scan; every thread of the quad returns the same value.
__device__ __forceinline__ uint32_t rank_at(const Tensor& t, uint32_t tile, uint32_t c) {
  const unsigned char* meta = reinterpret_cast<const unsigned char*>(t.index_metadata);
  const uint32_t block = tile / kRankStrideTiles;
  uint32_t partial = 0U;
  for (uint32_t before = block * kRankStrideTiles + c; before < tile; before += 4U) {
    partial += planes_of(meta, t.allocation_map_bytes, t.terminal_map_value, before);
  }
  partial += __shfl_xor_sync(0xFFFFFFFFU, partial, 1);
  partial += __shfl_xor_sync(0xFFFFFFFFU, partial, 2);
  return (block == 0U ? 0U : rank_prefix(meta, t.allocation_map_bytes, block - 1U)) + partial;
}

__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
  const __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// Split f32 weights into f16 hi and lo halves, two at a time.
__device__ __forceinline__ void split2(float a, float b, uint32_t& hi, uint32_t& lo) {
  const __half2 h = __floats2half2_rn(a, b);
  const float2 back = __half22float2(h);
  hi = *reinterpret_cast<const uint32_t*>(&h);
  lo = pack_half2(a - back.x, b - back.y);
}

__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A B3 digit (0, 1, 2 for trit -1, 0, +1) as a float trit without a convert:
// `digit | 0x4B000000` is the float 2^23 + digit.
__device__ __forceinline__ float trit_of(uint32_t packed, uint32_t d) {
  return __int_as_float(static_cast<int>(((packed >> (2U * d)) & 3U) | 0x4B000000U)) -
         8388609.0f;
}

// A word group's payload for one quad thread, loaded one group ahead of its use:
// `w[r][p][i]` is word `i` of the group for row `r` (g, g + 8) and plane `p`, and
// `s[r][p]` that plane-tile's two scales. Group 3 is word 12 alone.
struct GroupWords {
  uint32_t w[2][3][4];
  __half2 s[2][3];
};

template <uint32_t G>
__device__ __forceinline__ void load_group(GroupWords& out,
                                           const uint32_t* words,
                                           const __half2* scale_pairs,
                                           const uint32_t (&rank)[2],
                                           const uint32_t (&planes)[2]) {
#pragma unroll
  for (uint32_t r = 0; r < 2U; ++r) {
#pragma unroll
    for (uint32_t p = 0; p < 3U; ++p) {
      if (p < planes[r]) {
        const uint32_t* plane_words = words + static_cast<size_t>(rank[r] + p) * 13U + G * 4U;
        out.s[r][p] = scale_pairs[rank[r] + p];
#pragma unroll
        for (uint32_t i = 0; i < (G < 3U ? 4U : 1U); ++i) out.w[r][p][i] = __ldg(plane_words + i);
      }
    }
  }
}

// The trit table: entry `b` holds the five trits of B3 byte `b` as one byte each,
// in bytes 0-4 of a `uint2`: 0x3F for +1, 0xBF for -1, 0 for 0. Placed in the top
// byte of an otherwise zero word, those are the floats +0.5, -0.5 and 0, so one
// PRMT turns a trit into a float that an FMA against twice the scale makes exact.
__device__ __forceinline__ uint2 trit_entry(uint32_t code) {
  uint32_t bytes[2] = {0U, 0U};
  uint32_t rest = code;
  for (uint32_t d = 0; d < 5U; ++d) {
    const uint32_t digit = rest % 3U;
    rest /= 3U;
    const uint32_t value = digit == 0U ? 0xBFU : (digit == 2U ? 0x3FU : 0U);
    bytes[d / 4U] |= value << (8U * (d % 4U));
  }
  return make_uint2(bytes[0], bytes[1]);
}

// Trit `d` of a table entry as a float in {-0.5, 0, +0.5}. Selector nibble 3 picks
// byte `d` of (lo, hi); nibbles 0-2 pick hi's byte 1, which is always zero.
__device__ __forceinline__ float half_trit(uint2 entry, uint32_t d) {
  return __uint_as_float(__byte_perm(entry.x, entry.y, (d << 12U) | 0x555U));
}

// Byte `c` of `word` through the trit table.
__device__ __forceinline__ uint2 byte_entry(const uint2* table, uint32_t word, uint32_t select) {
  return table[__byte_perm(word, 0U, select)];
}

// Word 12's real trits 240..255: thread `c` takes 240 + 4c .. 240 + 4c + 3.
__device__ __forceinline__ void decode_tail(float (&w)[20],
                                            const GroupWords& in,
                                            uint32_t r,
                                            uint32_t planes,
                                            uint32_t c,
                                            const uint2* table) {
#pragma unroll
  for (uint32_t j = 0; j < 4U; ++j) w[j] = 0.0f;
#pragma unroll
  for (uint32_t p = 0; p < 3U; ++p) {
    if (p < planes) {
      const uint32_t bits = in.w[r][p][0];
      const float sc = 2.0f * __high2float(in.s[r][p]);
#pragma unroll
      for (uint32_t j = 0; j < 4U; ++j) {
        const uint32_t trit = 4U * c + j;
        const uint2 entry = byte_entry(table, bits, 0x4440U | (trit / 5U));
        w[j] = fmaf(half_trit(entry, trit % 5U), sc, w[j]);
      }
    }
  }
}

// Activations live in shared memory one 256-column tile at a time, a row per
// activation row padded to `kActStride` halves so the fragment loads of a
// half-warp hit 32 distinct banks.
constexpr uint32_t kActStride = kTile + 16U;

__device__ __forceinline__ void cp_async16(void* smem, const void* global) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(global));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <uint32_t N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Stage tile `tile`'s activation rows [0, rows) into `hi`/`lo`.
__device__ __forceinline__ void stage_tile(__half* hi,
                                           __half* lo,
                                           const __half* __restrict__ act_hi,
                                           const __half* __restrict__ act_lo,
                                           uint32_t tile,
                                           uint32_t rows,
                                           uint32_t k) {
  const uint32_t chunks = rows * (kTile / 8U);  // 16-byte chunks per array
  for (uint32_t index = threadIdx.x; index < 2U * chunks; index += blockDim.x) {
    const bool second = index >= chunks;
    const uint32_t chunk = second ? index - chunks : index;
    const uint32_t row = chunk / (kTile / 8U);
    const uint32_t part = chunk % (kTile / 8U);
    const size_t from = static_cast<size_t>(row) * k + static_cast<size_t>(tile) * kTile + part * 8U;
    __half* to = (second ? lo : hi) + row * kActStride + part * 8U;
    cp_async16(to, (second ? act_lo : act_hi) + from);
  }
}

// Multiply one group's decoded weights (`STEPS` k16 steps starting at `step0`)
// against the staged activation tile, accumulating into `acc`.
//
// Tensor-core f32 accumulation truncates; summing each word group into a fresh
// fragment and folding it into `acc` with a rounded add keeps that bias from
// growing with K.
template <uint32_t NT, uint32_t STEPS>
__device__ __forceinline__ void group_mma(float (&acc)[NT][4],
                                          const float (&w0)[20],
                                          const float (&w1)[20],
                                          const __half* hi,
                                          const __half* lo,
                                          uint32_t step0,
                                          uint32_t g,
                                          uint32_t c,
                                          uint32_t n_tiles) {
  float part[NT][4];
#pragma unroll
  for (uint32_t n = 0; n < NT; ++n) {
#pragma unroll
    for (uint32_t e = 0; e < 4U; ++e) part[n][e] = 0.0f;
  }
#pragma unroll
  for (uint32_t s = 0; s < STEPS; ++s) {
    uint32_t a_hi[4];
    uint32_t a_lo[4];
    split2(w0[4U * s], w0[4U * s + 1U], a_hi[0], a_lo[0]);
    split2(w1[4U * s], w1[4U * s + 1U], a_hi[1], a_lo[1]);
    split2(w0[4U * s + 2U], w0[4U * s + 3U], a_hi[2], a_lo[2]);
    split2(w1[4U * s + 2U], w1[4U * s + 3U], a_hi[3], a_lo[3]);
    const uint32_t column = (step0 + s) * 16U + c * 4U;
#pragma unroll
    for (uint32_t n = 0; n < NT; ++n) {
      if (n >= n_tiles) break;
      const uint32_t at = (n * 8U + g) * kActStride + column;
      const uint2 bh = *reinterpret_cast<const uint2*>(hi + at);
      const uint2 bl = *reinterpret_cast<const uint2*>(lo + at);
      mma16816(part[n], a_hi, bh.x, bh.y);
      mma16816(part[n], a_hi, bl.x, bl.y);
      mma16816(part[n], a_lo, bh.x, bh.y);
    }
  }
#pragma unroll
  for (uint32_t n = 0; n < NT; ++n) {
    if (n >= n_tiles) break;
#pragma unroll
    for (uint32_t e = 0; e < 4U; ++e) acc[n][e] = __fadd_rn(acc[n][e], part[n][e]);
  }
}

// The GEMM body for at most `8 * NT` activation rows. A block of warps shares one
// K-split (`blockIdx.y` of `gridDim.y`, tiles split evenly by index) and one staged
// activation tile at a time; each warp owns sixteen fused rows. With one split a
// warp writes each tensor's `[m, rows]` output directly; otherwise it writes
// `partial[split][m][total]` for `salt_v2_mma_reduce` to sum in split order.
template <uint32_t NT, bool DOUBLE>
__device__ __forceinline__ void mma_rows(const __half* __restrict__ act_hi,
                                         const __half* __restrict__ act_lo,
                                         const float* __restrict__ row_scale,
                                         const Tensor* __restrict__ tensors,
                                         uint32_t tensor_count,
                                         uint32_t total_rows,
                                         uint32_t k,
                                         uint32_t m,
                                         float* __restrict__ partial) {
  __shared__ uint2 table[256];
  extern __shared__ __align__(16) unsigned char act_shared[];
  for (uint32_t code = threadIdx.x; code < 256U; code += blockDim.x) {
    table[code] = trit_entry(code);
  }

  const uint32_t n_tiles = (m + 7U) / 8U;
  const uint32_t staged_rows = n_tiles * 8U;
  const uint32_t buffer_halves = NT * 8U * kActStride;
  __half* buffers = reinterpret_cast<__half*>(act_shared);

  const uint32_t tiles_per_row = k / kTile;
  const uint32_t split = blockIdx.y;
  const uint32_t splits = gridDim.y;
  const uint32_t tile_begin = tiles_per_row * split / splits;
  const uint32_t tile_end = tiles_per_row * (split + 1U) / splits;

  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t g = lane >> 2U;
  const uint32_t c = lane & 3U;
  // Byte-permute selector moving byte `c` of a word to byte 0, zeroing the rest.
  const uint32_t select = 0x4440U | c;
  const uint32_t first_fused = (blockIdx.x * (blockDim.x >> 5U) + (threadIdx.x >> 5U)) * 16U;
  // Warps past the last row still take part in staging and barriers.
  const bool active = first_fused < total_rows;

  uint32_t index = 0U;
  while (active && index + 1U < tensor_count && first_fused >= tensors[index + 1U].first_row) {
    ++index;
  }
  const Tensor t = tensors[index];
  const uint32_t row0 = (active ? first_fused - t.first_row : 0U) + g;
  const uint32_t rows[2] = {row0, row0 + 8U};
  const unsigned char* meta = reinterpret_cast<const unsigned char*>(t.index_metadata);
  const uint32_t* words = reinterpret_cast<const uint32_t*>(t.payload);
  const __half2* scale_pairs = reinterpret_cast<const __half2*>(t.scales);

  uint32_t rank[2] = {0U, 0U};
  uint32_t planes[2] = {0U, 0U};
  if (active) {
#pragma unroll
    for (uint32_t r = 0; r < 2U; ++r) {
      rank[r] = rank_at(t, rows[r] * tiles_per_row + tile_begin, c);
      planes[r] = planes_of(meta, t.allocation_map_bytes, t.terminal_map_value,
                            rows[r] * tiles_per_row + tile_begin);
    }
  }

  float acc[NT][4];
#pragma unroll
  for (uint32_t n = 0; n < NT; ++n) {
#pragma unroll
    for (uint32_t e = 0; e < 4U; ++e) acc[n][e] = 0.0f;
  }

  for (uint32_t tile = tile_begin; tile < tile_end; ++tile) {
    const uint32_t which = DOUBLE ? (tile - tile_begin) & 1U : 0U;
    __half* hi = buffers + which * 2U * buffer_halves;
    __half* lo = hi + buffer_halves;
    if (DOUBLE) {
      if (tile == tile_begin) {
        stage_tile(buffers, buffers + buffer_halves, act_hi, act_lo, tile_begin, staged_rows, k);
        cp_async_commit();
      }
      if (tile + 1U < tile_end) {
        __half* next_hi = buffers + (which ^ 1U) * 2U * buffer_halves;
        stage_tile(next_hi, next_hi + buffer_halves, act_hi, act_lo, tile + 1U, staged_rows, k);
      }
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      __syncthreads();
      stage_tile(hi, lo, act_hi, act_lo, tile, staged_rows, k);
      cp_async_commit();
      cp_async_wait<0>();
    }
    __syncthreads();

    if (active) {
      uint32_t next_rank[2];
      uint32_t next_planes[2] = {0U, 0U};
#pragma unroll
      for (uint32_t r = 0; r < 2U; ++r) {
        next_rank[r] = rank[r] + planes[r];
        if (tile + 1U < tile_end) {
          next_planes[r] = planes_of(meta, t.allocation_map_bytes, t.terminal_map_value,
                                     rows[r] * tiles_per_row + tile + 1U);
          // The next tile's plane-tiles into L1 while this one computes.
          const char* first = reinterpret_cast<const char*>(words + static_cast<size_t>(next_rank[r]) * 13U);
          const uint32_t bytes = next_planes[r] * 52U;
          for (uint32_t offset = c * 128U; offset < bytes; offset += 4U * 128U) {
            asm volatile("prefetch.global.L1 [%0];" ::"l"(first + offset));
          }
          asm volatile("prefetch.global.L1 [%0];" ::"l"(first + bytes - 1U));
        }
      }
      // Quad lanes split the tile's payload loads: lane c loads words c, c + 4,
      // c + 8 and c + 12 of every plane-tile of both rows, all issued up front,
      // and each step takes the word it needs from the owning lane by shuffle.
      // Every quad lane needs every word (each decodes its own byte of it), so
      // loading them all per lane re-requested each line four times and thrashed
      // L1 (33% hit rate, 375 MB of L2 reads for 59 MB of weights).
      uint32_t held[2][3][4];
#pragma unroll
      for (uint32_t r = 0; r < 2U; ++r) {
#pragma unroll
        for (uint32_t p = 0; p < 3U; ++p) {
#pragma unroll
          for (uint32_t j = 0; j < 4U; ++j) {
            const uint32_t word = 4U * j + c;
            held[r][p][j] = p < planes[r] && word < 13U
                                ? __ldg(words + static_cast<size_t>(rank[r] + p) * 13U + word)
                                : 0U;
          }
        }
      }
      const uint32_t quad_base = lane & ~3U;
      // Scales of every plane-tile of the two rows, doubled for the half trits.
      float2 sc[2][3];
#pragma unroll
      for (uint32_t r = 0; r < 2U; ++r) {
#pragma unroll
        for (uint32_t p = 0; p < 3U; ++p) {
          if (p < planes[r]) {
            const float2 half_scale = __half22float2(scale_pairs[rank[r] + p]);
            sc[r][p] = make_float2(2.0f * half_scale.x, 2.0f * half_scale.y);
          }
        }
      }
#pragma unroll
      for (uint32_t group = 0; group < 3U; ++group) {
        float part[NT][4];
#pragma unroll
        for (uint32_t n = 0; n < NT; ++n) {
#pragma unroll
          for (uint32_t e = 0; e < 4U; ++e) part[n][e] = 0.0f;
        }
        float w4[2][4];
#pragma unroll
        for (uint32_t s = 0; s < 5U; ++s) {
          float w[2][4];
          if (s < 4U) {
            const uint32_t word = 4U * group + s;
#pragma unroll
            for (uint32_t r = 0; r < 2U; ++r) {
#pragma unroll
              for (uint32_t j = 0; j < 4U; ++j) w[r][j] = 0.0f;
              w4[r][s] = 0.0f;
#pragma unroll
              for (uint32_t p = 0; p < 3U; ++p) {
                const uint32_t bits =
                    __shfl_sync(0xFFFFFFFFU, held[r][p][word >> 2U], quad_base | (word & 3U));
                if (p < planes[r]) {
                  const uint2 entry = byte_entry(table, bits, select);
#pragma unroll
                  for (uint32_t d = 0; d < 5U; ++d) {
                    float scale_d;
                    if (word < 6U) {
                      scale_d = sc[r][p].x;
                    } else if (word == 6U) {
                      scale_d = 5U * c + d < 8U ? sc[r][p].x : sc[r][p].y;
                    } else {
                      scale_d = sc[r][p].y;
                    }
                    const float t = half_trit(entry, d);
                    if (d < 4U) {
                      w[r][d] = fmaf(t, scale_d, w[r][d]);
                    } else {
                      w4[r][s] = fmaf(t, scale_d, w4[r][s]);
                    }
                  }
                }
              }
            }
          } else {
#pragma unroll
            for (uint32_t r = 0; r < 2U; ++r) {
#pragma unroll
              for (uint32_t j = 0; j < 4U; ++j) w[r][j] = w4[r][j];
            }
          }
          uint32_t a_hi[4];
          uint32_t a_lo[4];
          split2(w[0][0], w[0][1], a_hi[0], a_lo[0]);
          split2(w[1][0], w[1][1], a_hi[1], a_lo[1]);
          split2(w[0][2], w[0][3], a_hi[2], a_lo[2]);
          split2(w[1][2], w[1][3], a_hi[3], a_lo[3]);
          const uint32_t column = (group * 5U + s) * 16U + c * 4U;
#pragma unroll
          for (uint32_t n = 0; n < NT; ++n) {
            if (n >= n_tiles) break;
            const uint32_t at = (n * 8U + g) * kActStride + column;
            const uint2 bh = *reinterpret_cast<const uint2*>(hi + at);
            const uint2 bl = *reinterpret_cast<const uint2*>(lo + at);
            mma16816(part[n], a_hi, bh.x, bh.y);
            mma16816(part[n], a_hi, bl.x, bl.y);
            mma16816(part[n], a_lo, bh.x, bh.y);
          }
        }
#pragma unroll
        for (uint32_t n = 0; n < NT; ++n) {
          if (n >= n_tiles) break;
#pragma unroll
          for (uint32_t e = 0; e < 4U; ++e) acc[n][e] = __fadd_rn(acc[n][e], part[n][e]);
        }
      }
      {
        GroupWords current;
        float w0[20];
        float w1[20];
        load_group<3U>(current, words, scale_pairs, rank, planes);
        decode_tail(w0, current, 0U, planes[0], c, table);
        decode_tail(w1, current, 1U, planes[1], c, table);
        group_mma<NT, 1U>(acc, w0, w1, hi, lo, 15U, g, c, n_tiles);
      }
#pragma unroll
      for (uint32_t r = 0; r < 2U; ++r) {
        rank[r] = next_rank[r];
        planes[r] = next_planes[r];
      }
    }
    if (DOUBLE) __syncthreads();
  }
  if (!active) return;

  // c0, c1: weight row g, activation rows 2c, 2c + 1 of the n-tile; c2, c3: row g + 8.
#pragma unroll
  for (uint32_t n = 0; n < NT; ++n) {
    if (n >= n_tiles) break;
#pragma unroll
    for (uint32_t e = 0; e < 4U; ++e) {
      const uint32_t act_row = n * 8U + 2U * c + (e & 1U);
      if (act_row >= m) continue;
      const uint32_t row = rows[e < 2U ? 0 : 1];
      const float value = acc[n][e] * row_scale[act_row];
      if (splits == 1U) {
        reinterpret_cast<float*>(t.output)[static_cast<size_t>(act_row) * t.rows + row] = value;
      } else {
        partial[(static_cast<size_t>(split) * m + act_row) * total_rows + t.first_row + row] =
            value;
      }
    }
  }
}

}  // namespace salt_mma

// Activation rows -> permuted f16 hi/lo halves plus one power-of-two scale per row.
//
// One block per row. Row `r`'s scale is 2^e with the smallest e >= 0 that brings
// max|x| under 2^14, so hi never overflows f16; the GEMM multiplies its outputs
// back by 2^e, which is exact. Rows in [m, padded) are zero.
//
// Within each 256-column tile the halves are stored in the GEMM's slot order:
// slot ((step * 4 + c) * 4 + j) holds the trit quad thread `c` feeds to k16 step
// `step` in its slot `j` (see the K permutation note at the top).
extern "C" __global__ void salt_v2_mma_split_act(const float* __restrict__ activation,
                                                 __half* __restrict__ hi,
                                                 __half* __restrict__ lo,
                                                 float* __restrict__ row_scale,
                                                 uint32_t m,
                                                 uint32_t k) {
  __shared__ float reduce[32];
  const uint32_t row = blockIdx.x;
  const float* x = activation + static_cast<size_t>(row) * k;
  float peak = 0.0f;
  if (row < m) {
    for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) peak = fmaxf(peak, fabsf(x[i]));
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    peak = fmaxf(peak, __shfl_xor_sync(0xFFFFFFFFU, peak, offset));
  }
  if ((threadIdx.x & 31U) == 0U) reduce[threadIdx.x >> 5U] = peak;
  __syncthreads();
  if (threadIdx.x < 32U) {
    peak = threadIdx.x < (blockDim.x >> 5U) ? reduce[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      peak = fmaxf(peak, __shfl_xor_sync(0xFFFFFFFFU, peak, offset));
    }
    if (threadIdx.x == 0U) reduce[0] = peak;
  }
  __syncthreads();
  peak = reduce[0];
  int e = 0;
  // A non-finite peak keeps e = 0 and propagates through the products.
  while (e < 126 && peak >= 16384.0f) {
    peak *= 0.5f;
    ++e;
  }
  const float down = ldexpf(1.0f, -e);
  if (threadIdx.x == 0U) row_scale[row] = ldexpf(1.0f, e);

  __half* out_hi = hi + static_cast<size_t>(row) * k;
  __half* out_lo = lo + static_cast<size_t>(row) * k;
  for (uint32_t slot = threadIdx.x; slot < k; slot += blockDim.x) {
    const uint32_t tile = slot / salt_mma::kTile;
    const uint32_t within = slot % salt_mma::kTile;
    const uint32_t step = within / 16U;
    const uint32_t c = (within / 4U) % 4U;
    const uint32_t j = within % 4U;
    uint32_t trit;
    if (step < 15U) {
      // Steps 5G .. 5G + 3 take digits 0-3 of byte c of word 4G + s; step 5G + 4
      // takes digit 4 of byte c of words 4G .. 4G + 3.
      const uint32_t group = step / 5U;
      const uint32_t s = step % 5U;
      trit = s < 4U ? 20U * (4U * group + s) + 5U * c + j : 20U * (4U * group + j) + 5U * c + 4U;
    } else {
      trit = 240U + 4U * c + j;
    }
    const float value = row < m ? x[tile * salt_mma::kTile + trit] * down : 0.0f;
    const __half h = __float2half_rn(value);
    out_hi[slot] = h;
    out_lo[slot] = __float2half_rn(value - __half2float(h));
  }
}

// Up to 16 activation rows (verification). Single-buffered activation tile: the
// shared memory a second buffer took is worth more as L1 for the payload words
// (launched with a 50% shared-memory carveout; measured ~12% faster).
extern "C" __global__ void __launch_bounds__(256) salt_v2_mma_rows16_f32(const __half* __restrict__ act_hi,
                                                  const __half* __restrict__ act_lo,
                                                  const float* __restrict__ row_scale,
                                                  const salt_mma::Tensor* __restrict__ tensors,
                                                  uint32_t tensor_count,
                                                  uint32_t total_rows,
                                                  uint32_t k,
                                                  uint32_t m,
                                                  float* __restrict__ partial) {
  salt_mma::mma_rows<2U, false>(act_hi, act_lo, row_scale, tensors, tensor_count, total_rows, k, m,
                         partial);
}

// Up to 64 activation rows (prefill).
extern "C" __global__ void salt_v2_mma_rows64_f32(const __half* __restrict__ act_hi,
                                                  const __half* __restrict__ act_lo,
                                                  const float* __restrict__ row_scale,
                                                  const salt_mma::Tensor* __restrict__ tensors,
                                                  uint32_t tensor_count,
                                                  uint32_t total_rows,
                                                  uint32_t k,
                                                  uint32_t m,
                                                  float* __restrict__ partial) {
  salt_mma::mma_rows<8U, false>(act_hi, act_lo, row_scale, tensors, tensor_count, total_rows, k, m,
                         partial);
}

// Sum K-split partials in split order and scatter into each tensor's output.
extern "C" __global__ void salt_v2_mma_reduce(const float* __restrict__ partial,
                                              const salt_mma::Tensor* __restrict__ tensors,
                                              uint32_t tensor_count,
                                              uint32_t total_rows,
                                              uint32_t m,
                                              uint32_t splits) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<uint64_t>(m) * total_rows) return;
  const uint32_t act_row = static_cast<uint32_t>(index / total_rows);
  const uint32_t fused = static_cast<uint32_t>(index % total_rows);
  float sum = 0.0f;
  for (uint32_t s = 0; s < splits; ++s) {
    sum += partial[(static_cast<size_t>(s) * m + act_row) * total_rows + fused];
  }
  uint32_t t = 0U;
  while (t + 1U < tensor_count && fused >= tensors[t + 1U].first_row) ++t;
  const salt_mma::Tensor tensor = tensors[t];
  reinterpret_cast<float*>(tensor.output)[static_cast<size_t>(act_row) * tensor.rows + fused -
                                          tensor.first_row] = sum;
}

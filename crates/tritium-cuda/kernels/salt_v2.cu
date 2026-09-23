// SALT V2 correctness kernel (plan 0043 Stage 6).
//
// The kernel consumes the physical D2/B3/S34 bytes directly. It never writes a
// dense dequantized weight: each plane/group reduces activation values through
// trit-directed add/sub/skip first, applies its one f16 scale, and immediately
// accumulates that contribution into one output scalar.
// `build.rs` compiles this unit with --fmad=false; the explicit round-to-nearest
// multiply/add intrinsics also freeze the CPU-reference reduction order.

#include <cuda_fp16.h>
#include <stdint.h>

namespace {

constexpr uint32_t kAllocationTile = 256;
constexpr uint32_t kRankStrideTiles = 256;
// `plane_count_for_tile` returns `code + 1` for a two-bit `code < 3`, so a
// tile carries at most three planes. The warp kernel reserves that many
// ordered contribution slots per scale group.
constexpr uint32_t kMaxPlanesPerTile = 3;
// Every byte value, because the scalar decoder this table must match decodes
// whatever byte it is handed, not only B3's 243 canonical codes.
constexpr uint32_t kB3TableEntries = 256;
// The table's footprint measured in the shared array's own element size.
constexpr uint32_t kB3TableWords = kB3TableEntries * sizeof(unsigned short) / sizeof(float);

__device__ __forceinline__ int decode_trit(
    const unsigned char* payload,
    uint64_t payload_bytes,
    uint64_t base,
    uint32_t logical_len,
    uint32_t plane_bytes,
    uint32_t local_index,
    uint32_t codec) {
  if (local_index >= logical_len || base + plane_bytes > payload_bytes) {
    return 0;
  }

  if (codec == 0) {  // D2: four `trit + 1` codes per byte.
    const uint32_t byte_index = local_index >> 2;
    if (byte_index >= plane_bytes) return 0;
    const uint32_t shift = (local_index & 3U) * 2U;
    const unsigned char code = (payload[base + byte_index] >> shift) & 3U;
    return code <= 2U ? static_cast<int>(code) - 1 : 0;
  }

  if (codec == 1) {  // B3: five little-endian radix-3 digits per byte.
    const uint32_t byte_index = local_index / 5U;
    if (byte_index >= plane_bytes) return 0;
    uint32_t value = payload[base + byte_index];
    // Peel down to the requested digit with a CONSTANT divisor. The previous
    // form indexed a `constexpr unsigned char kPlace[5]` with a runtime value,
    // which nvcc cannot keep in registers: it materialized the table in LOCAL
    // memory, and ncu measured those loads at 46.65% of every L1TEX sector this
    // kernel requested (0.3 of 32 bytes per sector actually used). Dividing by
    // the literal 3 instead compiles to a multiply-shift and touches no memory.
    const uint32_t digit_index = local_index - byte_index * 5U;
#pragma unroll
    for (uint32_t skipped = 0; skipped < 4U; ++skipped) {
      if (skipped < digit_index) value /= 3U;
    }
    return static_cast<int>(value % 3U) - 1;
  }

  // S34: one five-bit code per group of four. Low two bits choose the zero;
  // the remaining three bits are signs in increasing nonzero position order.
  const uint32_t group = local_index >> 2;
  const uint32_t bit_index = group * 5U;
  const uint32_t byte_index = bit_index >> 3;
  const uint32_t shift = bit_index & 7U;
  if (byte_index >= plane_bytes) return 0;
  uint32_t word = payload[base + byte_index];
  if (byte_index + 1U < plane_bytes) {
    word |= static_cast<uint32_t>(payload[base + byte_index + 1U]) << 8U;
  }
  const uint32_t code = (word >> shift) & 31U;
  const uint32_t slot = local_index & 3U;
  const uint32_t zero_slot = code & 3U;
  if (slot == zero_slot) return 0;
  const uint32_t sign_index = slot - (slot > zero_slot ? 1U : 0U);
  return (code & (1U << (2U + sign_index))) != 0U ? 1 : -1;
}

// The scalar loop's add/sub/skip, expressed as one addend.
//
// `digit` is the codec's raw code: 0 means -1, 1 means the zero trit, 2 means
// +1 (D2's out-of-range 3 also means zero). Returning a literal `+0.0f` for the
// zero trit is exactly the skip it replaces. `__fadd_rn` yields -0 only when
// both operands are -0, and an accumulator that starts at +0 can never reach
// -0 under round-to-nearest (x + (-x) rounds to +0), so `acc + 0.0f == acc`
// holds for every accumulator this kernel can build. Selecting an addend
// instead of branching removes two divergent branches per coefficient; ncu
// measured only 18.8 of 32 threads active per cycle before this.
__device__ __forceinline__ float trit_addend(uint32_t digit, float activation_value) {
  return digit == 0U ? -activation_value : (digit == 2U ? activation_value : 0.0f);
}

// Reduce one scale group's coefficients into a single accumulator.
//
// Produces bit-identical results to the `decode_trit`-per-coefficient loop this
// replaces: coefficients are consumed in the same ascending `local` order, the
// same `__fsub_rn`/`__fadd_rn` intrinsics run in the same sequence, and a zero
// trit is skipped rather than added. The difference is purely how many loads it
// takes to get there. The scalar form re-fetched the SAME payload byte once per
// trit — five times per byte under B3, four under D2 — and ncu measured the
// result at 1.1 useful bytes per 32-byte sector transmitted. Here the byte is
// loaded once and its codes are unpacked in registers.
//
// `segment_activation` points at the coefficient `local_start` maps to, so the
// activation index is the loop's own offset.
__device__ __forceinline__ float reduce_group_segment(
    const unsigned char* payload,
    uint64_t payload_bytes,
    uint64_t base,
    uint32_t logical_len,
    uint32_t plane_bytes,
    const float* segment_activation,
    uint32_t local_start,
    uint32_t segment_len,
    uint32_t codec,
    const unsigned short* b3_digits) {
  float accumulator = 0.0f;
  // `decode_trit` returns 0 for every coefficient when the plane's bytes fall
  // outside the payload, so the whole segment contributes nothing.
  if (base + plane_bytes > payload_bytes) return accumulator;
  const uint32_t local_end = local_start + segment_len;
  uint32_t local = local_start;

  // Both `local >= logical_len` and `byte_index >= plane_bytes` are monotone in
  // `local`, so a coefficient that decodes as zero implies every later one does.
  // Clamping the bound once lets the loops below run without a per-coefficient
  // range test.
  const uint32_t limit = min(local_end, logical_len);

  if (codec == 1U) {  // B3: five little-endian radix-3 digits per byte.
    while (local < limit) {
      const uint32_t byte_index = local / 5U;
      if (byte_index >= plane_bytes) break;
      const uint32_t code = payload[base + byte_index];
      // Recovering five radix-3 digits costs five constant divisions, roughly 33
      // instructions for five trits, and ncu puts this kernel at 75.8% SM
      // throughput -- it is instruction-bound, so that is the cost that matters.
      // `b3_digits` is a 256-entry table holding all five digits of a byte at two
      // bits each, built once per block in shared memory, which turns the whole
      // group into one load and five shift/mask pairs. Callers without the table
      // pass null and keep the divisions.
      uint32_t packed;
      if (b3_digits != nullptr) {
        packed = b3_digits[code];
      } else {
        packed = (code % 3U) | (((code / 3U) % 3U) << 2U) | (((code / 9U) % 3U) << 4U) |
                 (((code / 27U) % 3U) << 6U) | (((code / 81U) % 3U) << 8U);
      }
      const uint32_t d0 = packed & 3U;
      const uint32_t d1 = (packed >> 2U) & 3U;
      const uint32_t d2 = (packed >> 4U) & 3U;
      const uint32_t d3 = (packed >> 6U) & 3U;
      const uint32_t d4 = (packed >> 8U) & 3U;
      const uint32_t run_base = byte_index * 5U;
      const uint32_t run_end = min(limit, run_base + 5U);
      const float* run_activation = segment_activation + (local - local_start);
      if (local == run_base && run_end == run_base + 5U) {
        // The common case: a whole byte's run lies inside the segment. Five
        // independent addends, no loop, no per-coefficient index arithmetic.
        accumulator = __fadd_rn(accumulator, trit_addend(d0, run_activation[0]));
        accumulator = __fadd_rn(accumulator, trit_addend(d1, run_activation[1]));
        accumulator = __fadd_rn(accumulator, trit_addend(d2, run_activation[2]));
        accumulator = __fadd_rn(accumulator, trit_addend(d3, run_activation[3]));
        accumulator = __fadd_rn(accumulator, trit_addend(d4, run_activation[4]));
        local = run_end;
        continue;
      }
      for (; local < run_end; ++local) {
        const uint32_t offset = local - run_base;
        const uint32_t digit = offset == 0U ? d0
                             : offset == 1U ? d1
                             : offset == 2U ? d2
                             : offset == 3U ? d3
                                            : d4;
        accumulator = __fadd_rn(
            accumulator, trit_addend(digit, segment_activation[local - local_start]));
      }
    }
    return accumulator;
  }

  if (codec == 0U) {  // D2: four `trit + 1` codes per byte.
    // `decode_trit` maps code 0 -> -1, 1 -> 0, 2 -> +1, and out-of-range 3 -> 0,
    // which is what `trit_addend` selects on.
    while (local < limit) {
      const uint32_t byte_index = local >> 2;
      if (byte_index >= plane_bytes) break;
      const uint32_t code = payload[base + byte_index];
      const uint32_t c0 = code & 3U;
      const uint32_t c1 = (code >> 2U) & 3U;
      const uint32_t c2 = (code >> 4U) & 3U;
      const uint32_t c3 = (code >> 6U) & 3U;
      const uint32_t run_base = byte_index << 2;
      const uint32_t run_end = min(limit, run_base + 4U);
      const float* run_activation = segment_activation + (local - local_start);
      if (local == run_base && run_end == run_base + 4U) {
        accumulator = __fadd_rn(accumulator, trit_addend(c0, run_activation[0]));
        accumulator = __fadd_rn(accumulator, trit_addend(c1, run_activation[1]));
        accumulator = __fadd_rn(accumulator, trit_addend(c2, run_activation[2]));
        accumulator = __fadd_rn(accumulator, trit_addend(c3, run_activation[3]));
        local = run_end;
        continue;
      }
      for (; local < run_end; ++local) {
        const uint32_t offset = local - run_base;
        const uint32_t pair = offset == 0U ? c0 : offset == 1U ? c1 : offset == 2U ? c2 : c3;
        accumulator = __fadd_rn(
            accumulator, trit_addend(pair, segment_activation[local - local_start]));
      }
    }
    return accumulator;
  }

  // S34 keeps the shared scalar decoder: it already spans two bytes per code
  // and is not the codec any shipped bundle uses.
  for (; local < local_end; ++local) {
    const int trit = decode_trit(payload, payload_bytes, base, logical_len,
                                 plane_bytes, local, codec);
    const float activation_value = segment_activation[local - local_start];
    if (trit < 0) {
      accumulator = __fsub_rn(accumulator, activation_value);
    } else if (trit > 0) {
      accumulator = __fadd_rn(accumulator, activation_value);
    }
  }
  return accumulator;
}

__device__ __forceinline__ uint32_t plane_count_for_tile(
    const unsigned char* index_metadata,
    uint32_t allocation_map_bytes,
    uint32_t terminal_map_value,
    uint32_t tile) {
  const uint64_t bit = static_cast<uint64_t>(tile) * 2U;
  const uint64_t allocated_bits = static_cast<uint64_t>(allocation_map_bytes) * 8U;
  uint32_t code;
  if (bit < allocated_bits) {
    code = (index_metadata[bit >> 3] >> (bit & 7U)) & 3U;
  } else {
    code = (terminal_map_value >> (bit - allocated_bits)) & 3U;
  }
  return code < 3U ? code + 1U : 0U;
}

__device__ __forceinline__ uint32_t read_rank_prefix(
    const unsigned char* index_metadata,
    uint32_t allocation_map_bytes,
    uint32_t prefix_index) {
  const unsigned char* bytes =
      index_metadata + allocation_map_bytes + prefix_index * 4U;
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8U) |
         (static_cast<uint32_t>(bytes[2]) << 16U) |
         (static_cast<uint32_t>(bytes[3]) << 24U);
}

__device__ __forceinline__ uint32_t plane_payload_bytes(
    uint32_t codec, uint32_t logical_len) {
  if (codec == 0U) return (logical_len + 3U) / 4U;
  if (codec == 1U) return (logical_len + 4U) / 5U;
  const uint32_t groups = (logical_len + 3U) / 4U;
  return (groups * 5U + 7U) / 8U;
}

}  // namespace

#if defined(TRITIUM_DEVICE_LOSS_QUALIFICATION)
// Release qualification only: a one-thread device trap poisons this CUDA
// context, exercising the same sticky driver-error path as a fatal device
// exception. Production code can launch this symbol only after its private,
// signal-gated qualification arm is set; it is never part of model dispatch.
extern "C" __global__ void tritium_qualification_poison_context() {
  asm volatile("trap;");
}
#endif

extern "C" __global__ void salt_v2_forward_exact(
    const float* activation,
    const unsigned char* payload,
    const __half* scales,
    const unsigned char* index_metadata,
    float* output,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t codec,
    uint32_t scale_group_size,
    uint32_t tile_count,
    uint32_t plane_count,
    uint64_t payload_bytes,
    uint64_t scale_count,
    uint32_t allocation_map_bytes,
    uint32_t rank_prefix_count,
    uint32_t terminal_map_value) {
  const uint64_t output_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t output_count = static_cast<uint64_t>(m) * n;
  if (output_index >= output_count) return;

  const uint32_t mi = static_cast<uint32_t>(output_index / n);
  const uint32_t row = static_cast<uint32_t>(output_index % n);
  const uint64_t row_base = static_cast<uint64_t>(row) * k;
  const uint64_t row_end = row_base + k;
  uint64_t coefficient = row_base;
  float accumulator = 0.0f;
  while (coefficient < row_end) {
    const uint32_t tile = static_cast<uint32_t>(coefficient / kAllocationTile);
    const uint32_t local_start =
        static_cast<uint32_t>(coefficient % kAllocationTile);
    if (tile >= tile_count) break;

    const uint32_t rank_block = tile / kRankStrideTiles;
    uint32_t begin = 0U;
    if (rank_block != 0U) {
      const uint32_t prefix_index = rank_block - 1U;
      if (prefix_index >= rank_prefix_count) break;
      begin = read_rank_prefix(index_metadata, allocation_map_bytes, prefix_index);
    }
    const uint32_t scan_start = rank_block * kRankStrideTiles;
    for (uint32_t prior = scan_start; prior < tile; ++prior) {
      begin += plane_count_for_tile(
          index_metadata, allocation_map_bytes, terminal_map_value, prior);
    }
    const uint32_t planes = plane_count_for_tile(
        index_metadata, allocation_map_bytes, terminal_map_value, tile);
    const uint32_t end = begin + planes;
    if (planes == 0U || end > plane_count) break;

    const uint64_t total_coefficients = static_cast<uint64_t>(n) * k;
    const uint64_t tile_base = static_cast<uint64_t>(tile) * kAllocationTile;
    if (tile_base >= total_coefficients) break;
    const uint32_t logical_len = static_cast<uint32_t>(
        min(static_cast<uint64_t>(kAllocationTile), total_coefficients - tile_base));
    const uint32_t group = local_start / scale_group_size;
    const uint32_t group_end =
        min((group + 1U) * scale_group_size, logical_len);
    if (local_start >= group_end) break;
    const uint64_t segment_len =
        min(static_cast<uint64_t>(group_end - local_start),
            row_end - coefficient);

    const uint32_t full_payload_bytes = plane_payload_bytes(codec, kAllocationTile);
    const uint32_t current_payload_bytes = plane_payload_bytes(codec, logical_len);
    const uint32_t full_scale_count =
        (kAllocationTile + scale_group_size - 1U) / scale_group_size;
    const uint32_t current_scale_count =
        (logical_len + scale_group_size - 1U) / scale_group_size;
    for (uint32_t plane = begin; plane < end; ++plane) {
      const uint32_t local_plane = plane - begin;
      const uint64_t payload_base =
          static_cast<uint64_t>(begin) * full_payload_bytes +
          static_cast<uint64_t>(local_plane) * current_payload_bytes;
      const uint64_t scale_base = static_cast<uint64_t>(begin) * full_scale_count +
                                  static_cast<uint64_t>(local_plane) * current_scale_count;
      const uint64_t scale_index = scale_base + group;
      if (scale_index >= scale_count) continue;
      const float group_accumulator = reduce_group_segment(
          payload, payload_bytes, payload_base, logical_len,
          current_payload_bytes,
          activation + static_cast<uint64_t>(mi) * k + (coefficient - row_base),
          local_start, static_cast<uint32_t>(segment_len), codec, nullptr);
      const float contribution = __fmul_rn(
          group_accumulator, __half2float(scales[scale_index]));
      accumulator = __fadd_rn(accumulator, contribution);
    }
    coefficient += segment_len;
  }
  output[output_index] = accumulator;
}

// Prefill-oriented variant. One block owns a tile of output rows and stages
// each 256-coefficient activation tile once in shared memory. Reduction order,
// codec decode, and scale application intentionally match the scalar kernel;
// this changes memory traffic and launch geometry only.
extern "C" __global__ void salt_v2_forward_tiled(
    const float* activation,
    const unsigned char* payload,
    const __half* scales,
    const unsigned char* index_metadata,
    float* output,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t codec,
    uint32_t scale_group_size,
    uint32_t tile_count,
    uint32_t plane_count,
    uint64_t payload_bytes,
    uint64_t scale_count,
    uint32_t allocation_map_bytes,
    uint32_t rank_prefix_count,
    uint32_t terminal_map_value) {
  const uint32_t row = static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint32_t mi = static_cast<uint32_t>(blockIdx.y);
  const bool active = mi < m && row < n;

  __shared__ float activation_tile[kAllocationTile];
  const uint64_t row_base = static_cast<uint64_t>(row) * k;
  const uint32_t tiles_per_row = (k + kAllocationTile - 1U) / kAllocationTile;
  const uint32_t full_payload_bytes = plane_payload_bytes(codec, kAllocationTile);
  const uint32_t full_scale_count =
      (kAllocationTile + scale_group_size - 1U) / scale_group_size;
  float accumulator = 0.0f;

  for (uint32_t row_tile = 0; row_tile < tiles_per_row; ++row_tile) {
    const uint32_t local_tile_start = row_tile * kAllocationTile;
    const uint32_t logical_len =
        min(kAllocationTile, k - local_tile_start);
    for (uint32_t local = threadIdx.x; local < logical_len;
         local += blockDim.x) {
      activation_tile[local] =
          activation[static_cast<uint64_t>(mi) * k + local_tile_start + local];
    }
    __syncthreads();

    const uint64_t coefficient = row_base + local_tile_start;
    const uint32_t tile = static_cast<uint32_t>(coefficient / kAllocationTile);
    if (active && tile < tile_count) {
      const uint32_t rank_block = tile / kRankStrideTiles;
      uint32_t begin = 0U;
      if (rank_block != 0U) {
        const uint32_t prefix_index = rank_block - 1U;
        if (prefix_index < rank_prefix_count) {
          begin = read_rank_prefix(index_metadata, allocation_map_bytes, prefix_index);
        }
      }
      const uint32_t scan_start = rank_block * kRankStrideTiles;
      for (uint32_t prior = scan_start; prior < tile; ++prior) {
        begin += plane_count_for_tile(
            index_metadata, allocation_map_bytes, terminal_map_value, prior);
      }
      const uint32_t planes = plane_count_for_tile(
          index_metadata, allocation_map_bytes, terminal_map_value, tile);
      const uint32_t end = begin + planes;
      const uint64_t total_coefficients = static_cast<uint64_t>(n) * k;
      const uint64_t tile_base = static_cast<uint64_t>(tile) * kAllocationTile;
      if (planes != 0U && end <= plane_count && tile_base < total_coefficients) {
        const uint32_t physical_len = static_cast<uint32_t>(min(
            static_cast<uint64_t>(kAllocationTile), total_coefficients - tile_base));
        const uint32_t current_payload_bytes = plane_payload_bytes(codec, physical_len);
        const uint32_t current_scale_count =
            (physical_len + scale_group_size - 1U) / scale_group_size;
        uint32_t local_start = 0U;
        while (local_start < logical_len) {
          const uint32_t group = local_start / scale_group_size;
          const uint32_t group_end =
              min((group + 1U) * scale_group_size, logical_len);
          const uint32_t segment_len = group_end - local_start;
          for (uint32_t plane = begin; plane < end; ++plane) {
            const uint32_t local_plane = plane - begin;
            const uint64_t payload_base =
                static_cast<uint64_t>(begin) * full_payload_bytes +
                static_cast<uint64_t>(local_plane) * current_payload_bytes;
            const uint64_t scale_base =
                static_cast<uint64_t>(begin) * full_scale_count +
                static_cast<uint64_t>(local_plane) * current_scale_count;
            const uint64_t scale_index = scale_base + group;
            if (scale_index >= scale_count) continue;
            const float group_accumulator = reduce_group_segment(
                payload, payload_bytes, payload_base, physical_len,
                current_payload_bytes, activation_tile + local_start,
                local_start, segment_len, codec, nullptr);
            const float contribution = __fmul_rn(
                group_accumulator, __half2float(scales[scale_index]));
            accumulator = __fadd_rn(accumulator, contribution);
          }
          local_start = group_end;
        }
      }
    }
    __syncthreads();
  }
  if (active) {
    output[static_cast<uint64_t>(mi) * n + row] = accumulator;
  }
}

// Warp-per-row decode.
//
// `salt_v2_forward_exact` hands one thread a whole `k`-long row. At decode
// (m = 1) that is N threads for the entire launch, and ncu measured the
// consequences: 0.56 waves per SM, so the grid does not fill the machine even
// once; 5.13 of 12 active warps per scheduler with 1.18 eligible; and 1.4
// useful bytes per 32-byte sector, because adjacent lanes own rows whose
// payloads sit `k` apart.
//
// Here a warp owns one output and each lane owns whole scale groups, strided by
// 32. The launch carries 32x the threads, and lanes read payload bytes tens of
// bytes apart rather than `k`.
//
// The reduction stays bit-identical to the scalar kernel. Every value that
// kernel accumulates is one `group_sum * scale` for a (tile, group, plane),
// added in tile-major, then group, then plane order. A lane writes each of its
// contributions into the slot naming its position in that sequence --
// `unit * kMaxPlanesPerTile + plane_local` -- and lane 0 replays the additions
// in index order with the same `__fadd_rn`. Untouched slots hold +0.0f, a no-op
// for the reason `trit_addend` documents, so the padding is invisible.
//
// A scalar-kernel `break` truncates the rest of the row, so lanes agree on the
// earliest unit that breaks and lane 0 stops there.
//
// Requires `k % kAllocationTile == 0` and `kAllocationTile % scale_group_size == 0`,
// which together make every segment exactly `scale_group_size` long and put unit
// `i` at column `i * scale_group_size`. The host checks both and falls back to
// the scalar kernel otherwise.
extern "C" __global__ void salt_v2_forward_warp(
    const float* activation,
    const unsigned char* payload,
    const __half* scales,
    const unsigned char* index_metadata,
    float* output,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t codec,
    uint32_t scale_group_size,
    uint32_t tile_count,
    uint32_t plane_count,
    uint64_t payload_bytes,
    uint64_t scale_count,
    uint32_t allocation_map_bytes,
    uint32_t rank_prefix_count,
    uint32_t terminal_map_value,
    uint32_t groups_per_row) {
  extern __shared__ float salt_v2_warp_shared[];

  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp_in_block = threadIdx.x >> 5U;
  const uint32_t warps_per_block = blockDim.x >> 5U;
  const uint32_t slots_per_row = groups_per_row * kMaxPlanesPerTile;

  // Block-wide B3 digit table, ahead of the per-warp contribution slots. Entry
  // `c` holds all five radix-3 digits of byte `c` at two bits each, so decoding
  // a byte becomes one shared load and five shift/mask pairs instead of five
  // constant divisions -- roughly 33 instructions for five trits, on a kernel
  // ncu measures at 75.8% SM throughput. Built for every byte value, not only
  // B3's 243 canonical codes, because the scalar decoder this has to match
  // applies `(c / 3^i) % 3` to whatever byte it is handed.
  unsigned short* b3_digits = reinterpret_cast<unsigned short*>(salt_v2_warp_shared);
  for (uint32_t code = threadIdx.x; code < kB3TableEntries; code += blockDim.x) {
    b3_digits[code] = static_cast<unsigned short>(
        (code % 3U) | (((code / 3U) % 3U) << 2U) | (((code / 9U) % 3U) << 4U) |
        (((code / 27U) % 3U) << 6U) | (((code / 81U) % 3U) << 8U));
  }
  // Every warp in the block must reach this, including one whose output row is
  // out of range, so the bounds check below cannot precede a block-wide barrier.
  __syncthreads();

  float* slots = salt_v2_warp_shared + kB3TableWords +
                 static_cast<size_t>(warp_in_block) * slots_per_row;

  const uint64_t output_index =
      static_cast<uint64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const uint64_t output_count = static_cast<uint64_t>(m) * n;
  // Warp-uniform: every lane of a warp shares `output_index`.
  if (output_index >= output_count) return;

  const uint32_t mi = static_cast<uint32_t>(output_index / n);
  const uint32_t row = static_cast<uint32_t>(output_index % n);
  const uint64_t row_base = static_cast<uint64_t>(row) * k;
  const uint64_t row_end = row_base + k;
  const uint64_t total_coefficients = static_cast<uint64_t>(n) * k;
  const float* row_activation = activation + static_cast<uint64_t>(mi) * k;

  for (uint32_t slot = lane; slot < slots_per_row; slot += 32U) {
    slots[slot] = 0.0f;
  }
  __syncwarp();

  const uint32_t full_payload_bytes = plane_payload_bytes(codec, kAllocationTile);
  const uint32_t full_scale_count =
      (kAllocationTile + scale_group_size - 1U) / scale_group_size;

  uint32_t break_unit = groups_per_row;
  for (uint32_t unit = lane; unit < groups_per_row; unit += 32U) {
    const uint32_t column = unit * scale_group_size;
    const uint64_t coefficient = row_base + column;
    const uint32_t tile = static_cast<uint32_t>(coefficient / kAllocationTile);
    if (column >= k || tile >= tile_count) { break_unit = unit; break; }

    const uint32_t rank_block = tile / kRankStrideTiles;
    uint32_t begin = 0U;
    if (rank_block != 0U) {
      const uint32_t prefix_index = rank_block - 1U;
      if (prefix_index >= rank_prefix_count) { break_unit = unit; break; }
      begin = read_rank_prefix(index_metadata, allocation_map_bytes, prefix_index);
    }
    const uint32_t scan_start = rank_block * kRankStrideTiles;
    for (uint32_t prior = scan_start; prior < tile; ++prior) {
      begin += plane_count_for_tile(
          index_metadata, allocation_map_bytes, terminal_map_value, prior);
    }
    const uint32_t planes = plane_count_for_tile(
        index_metadata, allocation_map_bytes, terminal_map_value, tile);
    const uint32_t end = begin + planes;
    if (planes == 0U || end > plane_count) { break_unit = unit; break; }

    const uint64_t tile_base = static_cast<uint64_t>(tile) * kAllocationTile;
    if (tile_base >= total_coefficients) { break_unit = unit; break; }
    const uint32_t logical_len = static_cast<uint32_t>(
        min(static_cast<uint64_t>(kAllocationTile), total_coefficients - tile_base));

    const uint32_t local_start =
        static_cast<uint32_t>(coefficient % kAllocationTile);
    const uint32_t group = local_start / scale_group_size;
    const uint32_t group_end = min((group + 1U) * scale_group_size, logical_len);
    if (local_start >= group_end) { break_unit = unit; break; }
    const uint64_t segment_len =
        min(static_cast<uint64_t>(group_end - local_start), row_end - coefficient);

    const uint32_t current_payload_bytes = plane_payload_bytes(codec, logical_len);
    const uint32_t current_scale_count =
        (logical_len + scale_group_size - 1U) / scale_group_size;

    for (uint32_t plane = begin; plane < end; ++plane) {
      const uint32_t local_plane = plane - begin;
      const uint64_t payload_base =
          static_cast<uint64_t>(begin) * full_payload_bytes +
          static_cast<uint64_t>(local_plane) * current_payload_bytes;
      const uint64_t scale_base =
          static_cast<uint64_t>(begin) * full_scale_count +
          static_cast<uint64_t>(local_plane) * current_scale_count;
      const uint64_t scale_index = scale_base + group;
      // The scalar kernel skips this plane and keeps going; the slot stays +0.0f.
      if (scale_index >= scale_count) continue;
      const float group_accumulator = reduce_group_segment(
          payload, payload_bytes, payload_base, logical_len,
          current_payload_bytes, row_activation + column, local_start,
          static_cast<uint32_t>(segment_len), codec, b3_digits);
      slots[unit * kMaxPlanesPerTile + local_plane] =
          __fmul_rn(group_accumulator, __half2float(scales[scale_index]));
    }
  }

  // Earliest breaking unit across the warp: the scalar kernel would have
  // abandoned the row there, so nothing at or beyond it may contribute.
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    break_unit = min(break_unit, __shfl_xor_sync(0xffffffffU, break_unit, offset));
  }
  __syncwarp();

  if (lane == 0U) {
    const uint32_t live_slots = break_unit * kMaxPlanesPerTile;
    float accumulator = 0.0f;
    for (uint32_t slot = 0; slot < live_slots; ++slot) {
      accumulator = __fadd_rn(accumulator, slots[slot]);
    }
    output[output_index] = accumulator;
  }
}

// Fast SALT V2 forward: the warp kernel without the ordered replay.
//
// `salt_v2_forward_warp` reproduces the scalar kernel's addition order exactly,
// and everything expensive about it is in service of that: three shared
// contribution slots per group, a barrier, and a serial `__fadd_rn` chain on
// lane 0 spanning the whole row. This variant drops all of it -- each lane
// accumulates its own groups into a register and the warp finishes with a
// shuffle tree-reduce.
//
// That reassociates the K-sum, so results are close but not bit-identical. It is
// the same trade `salt_mpgemm_tiled_f32` already makes, and the reason
// `SaltV2ForwardMode` distinguishes a fast entry point from the exact one: this
// kernel is gated on relative error against the CPU reference, never equality.
//
// Dropping the slots also frees the shared memory they occupied. Only the B3
// digit table remains, so a block no longer trades occupancy against row width
// -- which for `down_proj` at K = 17408 was 26 KiB per block.
extern "C" __global__ void salt_v2_forward_warp_fast(
    const float* activation,
    const unsigned char* payload,
    const __half* scales,
    const unsigned char* index_metadata,
    float* output,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t codec,
    uint32_t scale_group_size,
    uint32_t tile_count,
    uint32_t plane_count,
    uint64_t payload_bytes,
    uint64_t scale_count,
    uint32_t allocation_map_bytes,
    uint32_t rank_prefix_count,
    uint32_t terminal_map_value,
    uint32_t groups_per_row) {
  extern __shared__ float salt_v2_warp_shared[];

  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp_in_block = threadIdx.x >> 5U;
  const uint32_t warps_per_block = blockDim.x >> 5U;

  // Block-wide B3 digit table, ahead of the per-warp contribution slots. Entry
  // `c` holds all five radix-3 digits of byte `c` at two bits each, so decoding
  // a byte becomes one shared load and five shift/mask pairs instead of five
  // constant divisions -- roughly 33 instructions for five trits, on a kernel
  // ncu measures at 75.8% SM throughput. Built for every byte value, not only
  // B3's 243 canonical codes, because the scalar decoder this has to match
  // applies `(c / 3^i) % 3` to whatever byte it is handed.
  unsigned short* b3_digits = reinterpret_cast<unsigned short*>(salt_v2_warp_shared);
  for (uint32_t code = threadIdx.x; code < kB3TableEntries; code += blockDim.x) {
    b3_digits[code] = static_cast<unsigned short>(
        (code % 3U) | (((code / 3U) % 3U) << 2U) | (((code / 9U) % 3U) << 4U) |
        (((code / 27U) % 3U) << 6U) | (((code / 81U) % 3U) << 8U));
  }
  // Every warp in the block must reach this, including one whose output row is
  // out of range, so the bounds check below cannot precede a block-wide barrier.
  __syncthreads();


  const uint64_t output_index =
      static_cast<uint64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const uint64_t output_count = static_cast<uint64_t>(m) * n;
  // Warp-uniform: every lane of a warp shares `output_index`.
  if (output_index >= output_count) return;

  const uint32_t mi = static_cast<uint32_t>(output_index / n);
  const uint32_t row = static_cast<uint32_t>(output_index % n);
  const uint64_t row_base = static_cast<uint64_t>(row) * k;
  const uint64_t row_end = row_base + k;
  const uint64_t total_coefficients = static_cast<uint64_t>(n) * k;
  const float* row_activation = activation + static_cast<uint64_t>(mi) * k;

  float lane_accumulator = 0.0f;

  const uint32_t full_payload_bytes = plane_payload_bytes(codec, kAllocationTile);
  const uint32_t full_scale_count =
      (kAllocationTile + scale_group_size - 1U) / scale_group_size;

  uint32_t break_unit = groups_per_row;
  for (uint32_t unit = lane; unit < groups_per_row; unit += 32U) {
    const uint32_t column = unit * scale_group_size;
    const uint64_t coefficient = row_base + column;
    const uint32_t tile = static_cast<uint32_t>(coefficient / kAllocationTile);
    if (column >= k || tile >= tile_count) { break_unit = unit; break; }

    const uint32_t rank_block = tile / kRankStrideTiles;
    uint32_t begin = 0U;
    if (rank_block != 0U) {
      const uint32_t prefix_index = rank_block - 1U;
      if (prefix_index >= rank_prefix_count) { break_unit = unit; break; }
      begin = read_rank_prefix(index_metadata, allocation_map_bytes, prefix_index);
    }
    const uint32_t scan_start = rank_block * kRankStrideTiles;
    for (uint32_t prior = scan_start; prior < tile; ++prior) {
      begin += plane_count_for_tile(
          index_metadata, allocation_map_bytes, terminal_map_value, prior);
    }
    const uint32_t planes = plane_count_for_tile(
        index_metadata, allocation_map_bytes, terminal_map_value, tile);
    const uint32_t end = begin + planes;
    if (planes == 0U || end > plane_count) { break_unit = unit; break; }

    const uint64_t tile_base = static_cast<uint64_t>(tile) * kAllocationTile;
    if (tile_base >= total_coefficients) { break_unit = unit; break; }
    const uint32_t logical_len = static_cast<uint32_t>(
        min(static_cast<uint64_t>(kAllocationTile), total_coefficients - tile_base));

    const uint32_t local_start =
        static_cast<uint32_t>(coefficient % kAllocationTile);
    const uint32_t group = local_start / scale_group_size;
    const uint32_t group_end = min((group + 1U) * scale_group_size, logical_len);
    if (local_start >= group_end) { break_unit = unit; break; }
    const uint64_t segment_len =
        min(static_cast<uint64_t>(group_end - local_start), row_end - coefficient);

    const uint32_t current_payload_bytes = plane_payload_bytes(codec, logical_len);
    const uint32_t current_scale_count =
        (logical_len + scale_group_size - 1U) / scale_group_size;

    for (uint32_t plane = begin; plane < end; ++plane) {
      const uint32_t local_plane = plane - begin;
      const uint64_t payload_base =
          static_cast<uint64_t>(begin) * full_payload_bytes +
          static_cast<uint64_t>(local_plane) * current_payload_bytes;
      const uint64_t scale_base =
          static_cast<uint64_t>(begin) * full_scale_count +
          static_cast<uint64_t>(local_plane) * current_scale_count;
      const uint64_t scale_index = scale_base + group;
      // The scalar kernel skips this plane and keeps going; the slot stays +0.0f.
      if (scale_index >= scale_count) continue;
      const float group_accumulator = reduce_group_segment(
          payload, payload_bytes, payload_base, logical_len,
          current_payload_bytes, row_activation + column, local_start,
          static_cast<uint32_t>(segment_len), codec, b3_digits);
      lane_accumulator = __fadd_rn(
          lane_accumulator,
          __fmul_rn(group_accumulator, __half2float(scales[scale_index])));
    }
  }

  // Earliest breaking unit across the warp: the scalar kernel would have
  // abandoned the row there, so nothing at or beyond it may contribute.
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    break_unit = min(break_unit, __shfl_xor_sync(0xffffffffU, break_unit, offset));
  }
  if (break_unit != groups_per_row) {
    // The scalar kernel abandons the rest of a row here, and every condition
    // that reaches it is malformed metadata. Reconstructing which lanes'
    // partials survive that truncation would cost more than the reduction this
    // kernel exists to avoid, so fast mode refuses the row instead: the host
    // rejects any non-finite output, so this fails the call rather than
    // silently answering differently from the exact kernel.
    if (lane == 0U) {
      output[output_index] = __int_as_float(0x7FC00000);
    }
    return;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    lane_accumulator =
        __fadd_rn(lane_accumulator, __shfl_down_sync(0xFFFFFFFFU, lane_accumulator, offset));
  }
  if (lane == 0U) {
    output[output_index] = lane_accumulator;
  }
}

// Row-streaming SALT V2 GEMV (fast tier, B3 at scale group 128).
//
// Every earlier kernel gives each lane its own scale group and lets it walk that
// group's bytes one at a time, so a warp's 32 addresses are spread by the group
// stride and every load is a separate transaction. On the Qwen3.6 bundle that
// ran the MLP's 17408-row projections at ~95 GB/s, 9% of peak.
//
// The layout already allows better. A B3 plane-tile is 52 bytes -- exactly
// thirteen 32-bit words -- at offset `rank * 52`, and one row's plane-tiles are
// contiguous in rank order. So a row is one aligned run of `13 * plane_tiles`
// words, and a warp can read it the way any dense GEMV reads a row: lane `i`
// takes word `i`, then `i + 32`, fully coalesced. The only per-row bookkeeping is
// which tile each plane-tile belongs to, which the warp derives once from the
// 2-bit allocation map into a small shared table.
//
// Word `w` of a plane-tile holds trits `20w .. 20w + 19`. The 128-trit scale
// boundary falls inside word 6, at its ninth trit, so two accumulators -- trits
// 0-7 and 8-19 -- are enough to fold every word into the right scales without a
// single extra multiply-add. Word 12 carries four padding trits past 256; the
// codec guarantees they are zero trits, so their activations are never loaded.
//
// A trit becomes a float without an int-to-float convert, which runs at a quarter
// of FFMA rate on this architecture: `digit | 0x4B000000` is the float 2^23 +
// digit, and subtracting 2^23 + 1 leaves exactly `digit - 1`.
//
// The K-sum is reassociated, so this is a fast-tier kernel gated on relative
// error, never equality. Rows whose metadata is malformed are refused with a NaN
// for the host's non-finite check, as in `salt_v2_forward_warp_fast`.
//
// Requires codec B3, scale group 128, and `k % 256 == 0`; the host checks all three.
__device__ __forceinline__ float b3_trit(uint32_t packed, uint32_t digit) {
  return __int_as_float(static_cast<int>(((packed >> (2U * digit)) & 3U) | 0x4B000000U)) -
         8388609.0f;
}

// One row of the row-streaming GEMV, shared by the single-tensor and fused
// multi-tensor kernels. Must be called by a whole warp; `tile_of` is that warp's
// slice of the per-row plane-tile table and `b3_digits` the block's digit table.
// Row setup shared by the f32 and A8 row-streaming GEMVs: the row's starting plane
// rank, and its plane-tile -> tile table in `tile_of`. Returns false, for the whole
// warp, when the row's metadata is malformed.
__device__ __forceinline__ bool stream_row_setup(unsigned char* tile_of,
                                                 uint32_t lane,
                                                 const unsigned char* __restrict__ index_metadata,
                                                 uint32_t row,
                                                 uint32_t k,
                                                 uint32_t tile_count,
                                                 uint32_t plane_count,
                                                 uint32_t allocation_map_bytes,
                                                 uint32_t rank_prefix_count,
                                                 uint32_t terminal_map_value,
                                                 uint32_t tile_table_bytes,
                                                 uint32_t& row_rank_out,
                                                 uint32_t& plane_tiles_out) {
  const uint32_t tiles_per_row = k / kAllocationTile;
  const uint32_t first_tile = row * tiles_per_row;

  // The row's starting plane rank: the stored prefix for its rank block plus the
  // planes of the tiles between that block's start and the row.
  bool malformed = first_tile + tiles_per_row > tile_count;
  const uint32_t rank_block = first_tile / kRankStrideTiles;
  uint32_t row_rank = 0U;
  if (rank_block != 0U) {
    if (rank_block - 1U < rank_prefix_count) {
      row_rank = read_rank_prefix(index_metadata, allocation_map_bytes, rank_block - 1U);
    } else {
      malformed = true;
    }
  }
  uint32_t before = 0U;
  for (uint32_t tile = rank_block * kRankStrideTiles + lane; tile < first_tile; tile += 32U) {
    before += plane_count_for_tile(index_metadata, allocation_map_bytes, terminal_map_value, tile);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    before += __shfl_xor_sync(0xFFFFFFFFU, before, offset);
  }
  row_rank += before;

  // Plane-tile -> tile table for this row, by an exclusive scan of plane counts.
  uint32_t carried = 0U;
  for (uint32_t chunk = 0; chunk < tiles_per_row; chunk += 32U) {
    const uint32_t local = chunk + lane;
    uint32_t planes = 0U;
    if (local < tiles_per_row && !malformed) {
      planes = plane_count_for_tile(
          index_metadata, allocation_map_bytes, terminal_map_value, first_tile + local);
      if (planes == 0U) malformed = true;
    }
    uint32_t inclusive = planes;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const uint32_t up = __shfl_up_sync(0xFFFFFFFFU, inclusive, offset);
      if (lane >= static_cast<uint32_t>(offset)) inclusive += up;
    }
    const uint32_t start = carried + inclusive - planes;
    if (start + planes <= tile_table_bytes) {
      for (uint32_t plane = 0; plane < planes; ++plane) {
        tile_of[start + plane] = static_cast<unsigned char>(local);
      }
    } else if (planes != 0U) {
      malformed = true;
    }
    carried += __shfl_sync(0xFFFFFFFFU, inclusive, 31);
  }
  const uint32_t plane_tiles = carried;
  if (row_rank + plane_tiles > plane_count) malformed = true;
  if (__any_sync(0xFFFFFFFFU, malformed)) return false;
  __syncwarp();
  row_rank_out = row_rank;
  plane_tiles_out = plane_tiles;
  return true;

}

__device__ __forceinline__ void stream_row(const unsigned short* __restrict__ b3_digits,
                                           unsigned char* tile_of,
                                           uint32_t lane,
                                           const float* __restrict__ row_activation,
                                           const unsigned char* __restrict__ payload,
                                           const __half* __restrict__ scales,
                                           const unsigned char* __restrict__ index_metadata,
                                           float* out_slot,
                                           uint32_t row,
                                           uint32_t k,
                                           uint32_t tile_count,
                                           uint32_t plane_count,
                                           uint32_t allocation_map_bytes,
                                           uint32_t rank_prefix_count,
                                           uint32_t terminal_map_value,
                                           uint32_t tile_table_bytes) {
  uint32_t row_rank = 0U;
  uint32_t plane_tiles = 0U;
  if (!stream_row_setup(tile_of, lane, index_metadata, row, k, tile_count, plane_count,
                        allocation_map_bytes, rank_prefix_count, terminal_map_value,
                        tile_table_bytes, row_rank, plane_tiles)) {
    if (lane == 0U) *out_slot = __int_as_float(0x7FC00000);
    return;
  }

  const uint32_t* words = reinterpret_cast<const uint32_t*>(payload) +
                          static_cast<size_t>(row_rank) * 13U;
  const __half2* scale_pairs = reinterpret_cast<const __half2*>(scales) + row_rank;
  const uint32_t total_words = plane_tiles * 13U;

  float accumulator = 0.0f;
  for (uint32_t index = lane; index < total_words; index += 32U) {
    const uint32_t plane_tile = index / 13U;
    const uint32_t word = index - plane_tile * 13U;
    const uint32_t bits = __ldcs(words + index);
    const float2 scale = __half22float2(scale_pairs[plane_tile]);
    const float4* chunk = reinterpret_cast<const float4*>(
        row_activation + static_cast<size_t>(tile_of[plane_tile]) * kAllocationTile +
        word * 20U);
    const float4 a0 = __ldg(chunk);
    const float4 a1 = __ldg(chunk + 1);
    const float4 a2 = __ldg(chunk + 2);
    const float4 a3 = __ldg(chunk + 3);
    // Word 12's last four trits are padding past the tile: zero trits whose
    // activations belong to the next tile, so they are never read.
    const float4 a4 = word == 12U ? make_float4(0.0f, 0.0f, 0.0f, 0.0f) : __ldg(chunk + 4);
    const float a[20] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w, a2.x, a2.y,
                         a2.z, a2.w, a3.x, a3.y, a3.z, a3.w, a4.x, a4.y, a4.z, a4.w};
    const uint32_t p0 = b3_digits[bits & 0xFFU];
    const uint32_t p1 = b3_digits[(bits >> 8U) & 0xFFU];
    const uint32_t p2 = b3_digits[(bits >> 16U) & 0xFFU];
    const uint32_t p3 = b3_digits[bits >> 24U];
    // Trits 0-7 (byte 0, three of byte 1) and 8-19 fold into separate sums, so the
    // one straddling word can take a different scale on each side.
    float low = 0.0f;
    float high = 0.0f;
#pragma unroll
    for (uint32_t digit = 0; digit < 5U; ++digit) low = fmaf(b3_trit(p0, digit), a[digit], low);
#pragma unroll
    for (uint32_t digit = 0; digit < 3U; ++digit) {
      low = fmaf(b3_trit(p1, digit), a[5U + digit], low);
    }
#pragma unroll
    for (uint32_t digit = 3; digit < 5U; ++digit) {
      high = fmaf(b3_trit(p1, digit), a[5U + digit], high);
    }
#pragma unroll
    for (uint32_t digit = 0; digit < 5U; ++digit) {
      high = fmaf(b3_trit(p2, digit), a[10U + digit], high);
    }
#pragma unroll
    for (uint32_t digit = 0; digit < 5U; ++digit) {
      high = fmaf(b3_trit(p3, digit), a[15U + digit], high);
    }
    // Words 0-5 are wholly group 0, 7-12 wholly group 1; word 6 splits at trit 8.
    const float low_scale = word <= 6U ? scale.x : scale.y;
    const float high_scale = word <= 5U ? scale.x : scale.y;
    accumulator = fmaf(low_scale, low, accumulator);
    accumulator = fmaf(high_scale, high, accumulator);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    accumulator += __shfl_xor_sync(0xFFFFFFFFU, accumulator, offset);
  }
  if (lane == 0U) *out_slot = accumulator;
}

extern "C" __global__ void salt_v2_stream_f32(
    const float* __restrict__ activation,
    const unsigned char* __restrict__ payload,
    const __half* __restrict__ scales,
    const unsigned char* __restrict__ index_metadata,
    float* __restrict__ output,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t tile_count,
    uint32_t plane_count,
    uint32_t allocation_map_bytes,
    uint32_t rank_prefix_count,
    uint32_t terminal_map_value,
    uint32_t tile_table_bytes) {
  extern __shared__ unsigned char stream_shared[];
  unsigned short* b3_digits = reinterpret_cast<unsigned short*>(stream_shared);
  for (uint32_t code = threadIdx.x; code < kB3TableEntries; code += blockDim.x) {
    b3_digits[code] = static_cast<unsigned short>(
        (code % 3U) | (((code / 3U) % 3U) << 2U) | (((code / 9U) % 3U) << 4U) |
        (((code / 27U) % 3U) << 6U) | (((code / 81U) % 3U) << 8U));
  }
  // Block-wide barrier before any warp may leave for an out-of-range row.
  __syncthreads();

  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp_in_block = threadIdx.x >> 5U;
  const uint32_t warps_per_block = blockDim.x >> 5U;
  unsigned char* tile_of = stream_shared + kB3TableEntries * sizeof(unsigned short) +
                           static_cast<size_t>(warp_in_block) * tile_table_bytes;

  const uint64_t output_index =
      static_cast<uint64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  if (output_index >= static_cast<uint64_t>(m) * n) return;
  const uint32_t mi = static_cast<uint32_t>(output_index / n);
  const uint32_t row = static_cast<uint32_t>(output_index % n);
  stream_row(b3_digits, tile_of, lane, activation + static_cast<size_t>(mi) * k, payload,
             scales, index_metadata, output + output_index, row, k, tile_count, plane_count,
             allocation_map_bytes, rank_prefix_count, terminal_map_value, tile_table_bytes);
}

// One tensor of a fused row-stream launch. Mirrored field-for-field by
// `SaltStreamTensor` in `salt_v2_runtime.rs`; any change here must change there.
struct SaltStreamTensor {
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

// Several projections of one input in one launch (m = 1).
//
// A Qwen3.6 layer feeds the same normalized vector to up to four projections.
// Launched separately, the small ones cost latency, not bandwidth: DeltaNet's
// 48-row b and a projections took ~10 us each for almost no bytes, and the
// attention k and v ~12 us each. Here the tensors share one row space --
// tensor t owns rows [first_row, first_row + rows) -- and each warp finds its
// tensor from a four-entry table and streams that row exactly as
// `salt_v2_stream_f32` would, writing to the tensor's own output buffer.
extern "C" __global__ void salt_v2_stream_f32_multi(
    const float* __restrict__ activation,
    const SaltStreamTensor* __restrict__ tensors,
    uint32_t tensor_count,
    uint32_t total_rows,
    uint32_t k,
    uint32_t tile_table_bytes) {
  extern __shared__ unsigned char stream_shared[];
  unsigned short* b3_digits = reinterpret_cast<unsigned short*>(stream_shared);
  for (uint32_t code = threadIdx.x; code < kB3TableEntries; code += blockDim.x) {
    b3_digits[code] = static_cast<unsigned short>(
        (code % 3U) | (((code / 3U) % 3U) << 2U) | (((code / 9U) % 3U) << 4U) |
        (((code / 27U) % 3U) << 6U) | (((code / 81U) % 3U) << 8U));
  }
  __syncthreads();

  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp_in_block = threadIdx.x >> 5U;
  const uint32_t warps_per_block = blockDim.x >> 5U;
  unsigned char* tile_of = stream_shared + kB3TableEntries * sizeof(unsigned short) +
                           static_cast<size_t>(warp_in_block) * tile_table_bytes;
  const uint32_t fused_row = blockIdx.x * warps_per_block + warp_in_block;
  if (fused_row >= total_rows) return;

  uint32_t index = 0U;
  while (index + 1U < tensor_count && fused_row >= tensors[index + 1U].first_row) ++index;
  const SaltStreamTensor tensor = tensors[index];
  const uint32_t row = fused_row - tensor.first_row;
  stream_row(b3_digits, tile_of, lane, activation,
             reinterpret_cast<const unsigned char*>(tensor.payload),
             reinterpret_cast<const __half*>(tensor.scales),
             reinterpret_cast<const unsigned char*>(tensor.index_metadata),
             reinterpret_cast<float*>(tensor.output) + row, row, k, tensor.tile_count,
             tensor.plane_count, tensor.allocation_map_bytes, tensor.rank_prefix_count,
             tensor.terminal_map_value, tile_table_bytes);
}

// ---------------------------------------------------------------------------
// Load-time repack: dense plane 0 + per-row extra planes (the "D0X" layout).
//
// The row-stream kernels are co-limited by L1 wavefronts (~76%, 20 of every ~33
// per 32 words are activation loads) and issue (~71%). Reusing a lane's 20
// activations across several rows cuts the activation share, but on the ragged
// B3 layout it diverged (plane p >= 1 runs whenever any lane's tile has it) and
// needed per-row rank and tile-table setup. Every tile has at least one plane,
// so plane 0 is dense: repacked row-major as 13 words per (row, tile), it has no
// raggedness, no setup and fully coalesced rows. The remaining planes -- 1.5% of
// down_proj's plane-tiles, ~20% of gate|up's -- go in a per-row list.
//
//   dense_payload  u32  [rows][tiles][13]   first plane of each tile
//   dense_scales   u32  [rows][tiles]       its half2 scale pair
//   row_ptr        u32  [rows + 1]          extra-plane prefix per row
//   extra_tile     u8   [extras]            tile of each extra plane, ascending per row
//   extra_payload  u32  [extras][13]
//   extra_scales   u32  [extras]
//
// The bytes are the B3 bytes unchanged, so the repacked tensor holds the same
// values; only their order (and so the K-sum association) differs.
// ---------------------------------------------------------------------------

// Pass 1: per row, how many tiles carry at least two and exactly three planes.
// Summed on the host they give each plane's coverage, which picks how many planes
// go dense. Malformed rows are counted in `bad` (the host refuses the tensor).
extern "C" __global__ void salt_v2_repack_count(const unsigned char* __restrict__ index_metadata,
                                                uint32_t* __restrict__ two_or_more,
                                                uint32_t* __restrict__ three,
                                                uint32_t* __restrict__ bad,
                                                uint32_t rows,
                                                uint32_t k,
                                                uint32_t tile_count,
                                                uint32_t allocation_map_bytes,
                                                uint32_t terminal_map_value) {
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t row = blockIdx.x * (blockDim.x >> 5U) + (threadIdx.x >> 5U);
  if (row >= rows) return;
  const uint32_t tiles = k / kAllocationTile;
  const uint32_t first_tile = row * tiles;
  uint32_t two = 0U;
  uint32_t full = 0U;
  uint32_t malformed = first_tile + tiles > tile_count ? 1U : 0U;
  for (uint32_t tile = lane; tile < tiles && malformed == 0U; tile += 32U) {
    const uint32_t planes = plane_count_for_tile(index_metadata, allocation_map_bytes,
                                                 terminal_map_value, first_tile + tile);
    if (planes == 0U || planes > 3U) malformed = 1U;
    two += planes >= 2U ? 1U : 0U;
    full += planes == 3U ? 1U : 0U;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    two += __shfl_xor_sync(0xFFFFFFFFU, two, offset);
    full += __shfl_xor_sync(0xFFFFFFFFU, full, offset);
    malformed |= __shfl_xor_sync(0xFFFFFFFFU, malformed, offset);
  }
  if (lane == 0U) {
    two_or_more[row] = two;
    three[row] = full;
    if (malformed != 0U) atomicAdd(bad, 1U);
  }
}

// Pass 2: scatter every plane-tile of a row. Plane j of tile t goes dense when
// j < dense_planes, else to the row's extra list. Dense slots a tile does not
// fill stay zero -- zero scale, so they contribute exactly nothing. A lane owns a
// tile: a one-time copy, so simplicity over coalescing.
extern "C" __global__ void salt_v2_repack_write(const unsigned char* __restrict__ payload,
                                                const __half* __restrict__ scales,
                                                const unsigned char* __restrict__ index_metadata,
                                                const uint32_t* __restrict__ row_ptr,
                                                uint32_t* __restrict__ dense_payload,
                                                uint32_t* __restrict__ dense_scales,
                                                unsigned char* __restrict__ extra_tile,
                                                uint32_t* __restrict__ extra_payload,
                                                uint32_t* __restrict__ extra_scales,
                                                uint32_t rows,
                                                uint32_t k,
                                                uint32_t allocation_map_bytes,
                                                uint32_t rank_prefix_count,
                                                uint32_t terminal_map_value,
                                                uint32_t dense_planes) {
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t row = blockIdx.x * (blockDim.x >> 5U) + (threadIdx.x >> 5U);
  if (row >= rows) return;
  const uint32_t tiles = k / kAllocationTile;
  const uint32_t positions = tiles * 13U;
  const uint32_t first_tile = row * tiles;
  // Row's starting rank, as `stream_row_setup` derives it.
  const uint32_t rank_block = first_tile / kRankStrideTiles;
  uint32_t row_rank = 0U;
  if (rank_block != 0U && rank_block - 1U < rank_prefix_count) {
    row_rank = read_rank_prefix(index_metadata, allocation_map_bytes, rank_block - 1U);
  }
  uint32_t before = 0U;
  for (uint32_t tile = rank_block * kRankStrideTiles + lane; tile < first_tile; tile += 32U) {
    before += plane_count_for_tile(index_metadata, allocation_map_bytes, terminal_map_value, tile);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    before += __shfl_xor_sync(0xFFFFFFFFU, before, offset);
  }
  row_rank += before;

  const uint32_t* words = reinterpret_cast<const uint32_t*>(payload);
  const uint32_t* scale_pairs = reinterpret_cast<const uint32_t*>(scales);
  uint32_t plane_carry = 0U;
  uint32_t extra_carry = row_ptr[row];
  for (uint32_t chunk = 0; chunk < tiles; chunk += 32U) {
    const uint32_t tile = chunk + lane;
    const uint32_t planes =
        tile < tiles ? plane_count_for_tile(index_metadata, allocation_map_bytes,
                                            terminal_map_value, first_tile + tile)
                     : 0U;
    const uint32_t extras = planes > dense_planes ? planes - dense_planes : 0U;
    uint32_t plane_scan = planes;
    uint32_t extra_scan = extras;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const uint32_t up_planes = __shfl_up_sync(0xFFFFFFFFU, plane_scan, offset);
      const uint32_t up_extras = __shfl_up_sync(0xFFFFFFFFU, extra_scan, offset);
      if (lane >= static_cast<uint32_t>(offset)) {
        plane_scan += up_planes;
        extra_scan += up_extras;
      }
    }
    const uint32_t first_plane = row_rank + plane_carry + plane_scan - planes;
    const uint32_t first_extra = extra_carry + extra_scan - extras;
    for (uint32_t plane = 0; plane < planes; ++plane) {
      const size_t source = first_plane + plane;
      if (plane < dense_planes) {
        const size_t slot = (static_cast<size_t>(plane) * rows + row) * tiles + tile;
        for (uint32_t word = 0; word < 13U; ++word) {
          dense_payload[(static_cast<size_t>(plane) * rows + row) * positions + tile * 13U + word] =
              words[source * 13U + word];
        }
        dense_scales[slot] = scale_pairs[source];
      } else {
        const size_t slot = first_extra + plane - dense_planes;
        for (uint32_t word = 0; word < 13U; ++word) {
          extra_payload[slot * 13U + word] = words[source * 13U + word];
        }
        extra_scales[slot] = scale_pairs[source];
        extra_tile[slot] = static_cast<unsigned char>(tile);
      }
    }
    plane_carry += __shfl_sync(0xFFFFFFFFU, plane_scan, 31);
    extra_carry += __shfl_sync(0xFFFFFFFFU, extra_scan, 31);
  }
}

// One tensor of a fused D0X launch. Mirrored by `d0x_descriptor` in
// `salt_v2_repack.rs`.
struct D0xTensor {
  unsigned long long dense_payload;
  unsigned long long dense_scales;
  unsigned long long row_ptr;
  unsigned long long extra_tile;
  unsigned long long extra_payload;
  unsigned long long extra_scales;
  unsigned long long output;
  uint32_t rows;
  uint32_t first_row;
  uint32_t dense_planes;
  uint32_t reserved;
};

// Rows per block, and warps per block, of the D0X GEMV.
constexpr uint32_t kD0xRows = 4U;
constexpr uint32_t kD0xWarps = 8U;

// One B3 word against 20 activations, as a scaled contribution.
__device__ __forceinline__ float d0x_word(const unsigned short* __restrict__ b3_digits,
                                          uint32_t bits,
                                          const float (&a)[20],
                                          uint32_t word,
                                          uint32_t scale_bits,
                                          float accumulator) {
  const uint32_t p0 = b3_digits[bits & 0xFFU];
  const uint32_t p1 = b3_digits[(bits >> 8U) & 0xFFU];
  const uint32_t p2 = b3_digits[(bits >> 16U) & 0xFFU];
  const uint32_t p3 = b3_digits[bits >> 24U];
  float low = 0.0f;
  float high = 0.0f;
#pragma unroll
  for (uint32_t digit = 0; digit < 5U; ++digit) low = fmaf(b3_trit(p0, digit), a[digit], low);
#pragma unroll
  for (uint32_t digit = 0; digit < 3U; ++digit) {
    low = fmaf(b3_trit(p1, digit), a[5U + digit], low);
  }
#pragma unroll
  for (uint32_t digit = 3; digit < 5U; ++digit) {
    high = fmaf(b3_trit(p1, digit), a[5U + digit], high);
  }
#pragma unroll
  for (uint32_t digit = 0; digit < 5U; ++digit) {
    high = fmaf(b3_trit(p2, digit), a[10U + digit], high);
  }
#pragma unroll
  for (uint32_t digit = 0; digit < 5U; ++digit) {
    high = fmaf(b3_trit(p3, digit), a[15U + digit], high);
  }
  __half2 pair;
  *reinterpret_cast<uint32_t*>(&pair) = scale_bits;
  const float2 scale = __half22float2(pair);
  const float low_scale = word <= 6U ? scale.x : scale.y;
  const float high_scale = word <= 5U ? scale.x : scale.y;
  accumulator = fmaf(low_scale, low, accumulator);
  return fmaf(high_scale, high, accumulator);
}

__device__ __forceinline__ void d0x_activations(const float* __restrict__ activation,
                                                uint32_t tile,
                                                uint32_t word,
                                                float (&a)[20]) {
  const float4* chunk = reinterpret_cast<const float4*>(
      activation + static_cast<size_t>(tile) * kAllocationTile + word * 20U);
  const float4 a0 = __ldg(chunk);
  const float4 a1 = __ldg(chunk + 1);
  const float4 a2 = __ldg(chunk + 2);
  const float4 a3 = __ldg(chunk + 3);
  // Word 12's last four trits are padding: never read past the tile.
  const float4 a4 = word == 12U ? make_float4(0.0f, 0.0f, 0.0f, 0.0f) : __ldg(chunk + 4);
  a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
  a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
  a[8] = a2.x; a[9] = a2.y; a[10] = a2.z; a[11] = a2.w;
  a[12] = a3.x; a[13] = a3.y; a[14] = a3.z; a[15] = a3.w;
  a[16] = a4.x; a[17] = a4.y; a[18] = a4.z; a[19] = a4.w;
}

// Fused D0X GEMV (m = 1). A row group of `kD0xRows` consecutive rows of one
// tensor (the host requires each member's rows to be a multiple of it) is owned by
// `group_warps` warps (1, 2, 4 or 8) that split K between them; a block holds
// `kD0xWarps / group_warps` groups. One warp per group suits short K (no barrier
// work per group); several suit long K, where one warp per group would leave too
// few warps (5120-row down_proj: 1280). Dense pass: a thread owns (tile, word)
// positions, loads their 20 activations once and applies them to every dense
// plane of every row -- coalesced, divergence-free. Extra pass: the rows'
// remaining planes as one flattened word stream. Fast tier: relative error.
extern "C" __global__ void __launch_bounds__(kD0xWarps * 32U)
    salt_v2_d0x_f32(const float* __restrict__ activation,
                    const D0xTensor* __restrict__ tensors,
                    uint32_t tensor_count,
                    uint32_t total_rows,
                    uint32_t k,
                    uint32_t group_warps) {
  __shared__ unsigned short b3_digits[kB3TableEntries];
  __shared__ float partial[kD0xWarps][kD0xRows];
  for (uint32_t code = threadIdx.x; code < kB3TableEntries; code += blockDim.x) {
    b3_digits[code] = static_cast<unsigned short>(
        (code % 3U) | (((code / 3U) % 3U) << 2U) | (((code / 9U) % 3U) << 4U) |
        (((code / 27U) % 3U) << 6U) | (((code / 81U) % 3U) << 8U));
  }
  __syncthreads();
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp = threadIdx.x >> 5U;
  const uint32_t group = warp / group_warps;
  const uint32_t group_thread = threadIdx.x - group * group_warps * 32U;
  const uint32_t group_threads = group_warps * 32U;
  const uint32_t first = (blockIdx.x * (kD0xWarps / group_warps) + group) * kD0xRows;
  const bool active = first < total_rows;

  float accumulator[kD0xRows];
#pragma unroll
  for (uint32_t r = 0; r < kD0xRows; ++r) accumulator[r] = 0.0f;
  uint32_t row0 = 0U;
  D0xTensor tensor;
  if (active) {
    uint32_t index = 0U;
    while (index + 1U < tensor_count && first >= tensors[index + 1U].first_row) ++index;
    tensor = tensors[index];
    row0 = first - tensor.first_row;
    const uint32_t tiles = k / kAllocationTile;
    const uint32_t positions = tiles * 13U;
    const size_t plane_words = static_cast<size_t>(tensor.rows) * positions;
    const size_t plane_scales = static_cast<size_t>(tensor.rows) * tiles;
    const uint32_t* dense = reinterpret_cast<const uint32_t*>(tensor.dense_payload) +
                            static_cast<size_t>(row0) * positions;
    const uint32_t* dense_scales = reinterpret_cast<const uint32_t*>(tensor.dense_scales) +
                                   static_cast<size_t>(row0) * tiles;
    const uint32_t dense_planes = tensor.dense_planes;

    for (uint32_t position = group_thread; position < positions; position += group_threads) {
      const uint32_t tile = position / 13U;
      const uint32_t word = position - tile * 13U;
      float a[20];
      d0x_activations(activation, tile, word, a);
#pragma unroll
      for (uint32_t plane = 0; plane < 3U; ++plane) {
        if (plane >= dense_planes) break;
        uint32_t bits[kD0xRows];
        uint32_t scale_bits[kD0xRows];
#pragma unroll
        for (uint32_t r = 0; r < kD0xRows; ++r) {
          bits[r] = __ldcs(dense + plane * plane_words + static_cast<size_t>(r) * positions +
                           position);
          scale_bits[r] = __ldg(dense_scales + plane * plane_scales +
                                static_cast<size_t>(r) * tiles + tile);
        }
#pragma unroll
        for (uint32_t r = 0; r < kD0xRows; ++r) {
          accumulator[r] = d0x_word(b3_digits, bits[r], a, word, scale_bits[r], accumulator[r]);
        }
      }
    }

    // Extra planes of the group's rows, flattened.
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(tensor.row_ptr) + row0;
    uint32_t bound[kD0xRows + 1U];
#pragma unroll
    for (uint32_t r = 0; r <= kD0xRows; ++r) bound[r] = row_ptr[r];
    const unsigned char* extra_tile = reinterpret_cast<const unsigned char*>(tensor.extra_tile);
    const uint32_t* extra_payload = reinterpret_cast<const uint32_t*>(tensor.extra_payload);
    const uint32_t* extra_scales = reinterpret_cast<const uint32_t*>(tensor.extra_scales);
    const uint32_t extra_words = (bound[kD0xRows] - bound[0]) * 13U;
    for (uint32_t at = group_thread; at < extra_words; at += group_threads) {
      const uint32_t extra = bound[0] + at / 13U;
      const uint32_t word = at % 13U;
      const uint32_t bits = __ldcs(extra_payload + static_cast<size_t>(extra) * 13U + word);
      float a[20];
      d0x_activations(activation, extra_tile[extra], word, a);
      const float contribution =
          d0x_word(b3_digits, bits, a, word, __ldg(extra_scales + extra), 0.0f);
#pragma unroll
      for (uint32_t r = 0; r < kD0xRows; ++r) {
        if (extra >= bound[r] && extra < bound[r + 1U]) accumulator[r] += contribution;
      }
    }
  }

#pragma unroll
  for (uint32_t r = 0; r < kD0xRows; ++r) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      accumulator[r] += __shfl_xor_sync(0xFFFFFFFFU, accumulator[r], offset);
    }
  }
  if (group_warps == 1U) {
    if (active && lane < kD0xRows) {
      float value = accumulator[0];
#pragma unroll
      for (uint32_t r = 1; r < kD0xRows; ++r) value = lane == r ? accumulator[r] : value;
      reinterpret_cast<float*>(tensor.output)[row0 + lane] = value;
    }
    return;
  }
  if (lane == 0U) {
#pragma unroll
    for (uint32_t r = 0; r < kD0xRows; ++r) partial[warp][r] = accumulator[r];
  }
  __syncthreads();
  if (active && group_thread < kD0xRows) {
    float sum = 0.0f;
    for (uint32_t w = 0; w < group_warps; ++w) sum += partial[group * group_warps + w][group_thread];
    reinterpret_cast<float*>(tensor.output)[row0 + group_thread] = sum;
  }
}

// ---------------------------------------------------------------------------
// A8 row-streaming GEMV (relaxed tier): int8 activations, dp4a.
//
// ncu puts the f32 row-stream kernels at 71-80% L1/TEX throughput, above both
// DRAM and SM: every 4-byte weight word needs 80 bytes of f32 activations, so
// the activation loads, not the weights, bind. Quantizing activations to int8
// per 128-coefficient group -- the same groups the weights are scaled over --
// cuts that to 20 bytes per word, and `__dp4a` does four multiply-adds per
// instruction on exact int32.
//
// This changes numerics: activations carry ~8 bits instead of 24, so this is a
// relaxed-tier kernel gated on output quality (RFC 0001's bars), not on its
// distance from the f32 kernels.
// ---------------------------------------------------------------------------

// Quantize `rows` rows of `k` activations to int8 with one f32 scale per
// 128-coefficient group: q = round(x / s), s = absmax / 127. One warp per group.
extern "C" __global__ void salt_v2_quant_act_g128(const float* __restrict__ input,
                                                  signed char* __restrict__ quantized,
                                                  float* __restrict__ group_scale,
                                                  uint32_t groups) {
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t group = blockIdx.x * (blockDim.x >> 5U) + (threadIdx.x >> 5U);
  if (group >= groups) return;
  const float4 values = reinterpret_cast<const float4*>(input + static_cast<size_t>(group) * 128U)[lane];
  float peak = fmaxf(fmaxf(fabsf(values.x), fabsf(values.y)), fmaxf(fabsf(values.z), fabsf(values.w)));
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    peak = fmaxf(peak, __shfl_xor_sync(0xFFFFFFFFU, peak, offset));
  }
  const float scale = peak / 127.0f;
  const float inverse = peak > 0.0f ? 127.0f / peak : 0.0f;
  char4 packed;
  packed.x = static_cast<signed char>(__float2int_rn(values.x * inverse));
  packed.y = static_cast<signed char>(__float2int_rn(values.y * inverse));
  packed.z = static_cast<signed char>(__float2int_rn(values.z * inverse));
  packed.w = static_cast<signed char>(__float2int_rn(values.w * inverse));
  reinterpret_cast<char4*>(quantized + static_cast<size_t>(group) * 128U)[lane] = packed;
  if (lane == 0U) group_scale[group] = scale;
}

// Byte -> its five trits as signed int8, little-endian in the low five bytes.
__device__ __forceinline__ void build_b3_int8_table(unsigned long long* table) {
  for (uint32_t code = threadIdx.x; code < kB3TableEntries; code += blockDim.x) {
    unsigned long long packed = 0ULL;
    uint32_t value = code;
#pragma unroll
    for (uint32_t digit = 0; digit < 5U; ++digit) {
      const int trit = static_cast<int>(value % 3U) - 1;
      value /= 3U;
      packed |= static_cast<unsigned long long>(static_cast<unsigned char>(trit)) << (8U * digit);
    }
    table[code] = packed;
  }
}

__device__ __forceinline__ void stream_row_i8(const unsigned long long* __restrict__ b3_int8,
                                              unsigned char* tile_of,
                                              uint32_t lane,
                                              const signed char* __restrict__ row_quantized,
                                              const float* __restrict__ row_group_scale,
                                              const unsigned char* __restrict__ payload,
                                              const __half* __restrict__ scales,
                                              const unsigned char* __restrict__ index_metadata,
                                              float* out_slot,
                                              uint32_t row,
                                              uint32_t k,
                                              uint32_t tile_count,
                                              uint32_t plane_count,
                                              uint32_t allocation_map_bytes,
                                              uint32_t rank_prefix_count,
                                              uint32_t terminal_map_value,
                                              uint32_t tile_table_bytes) {
  uint32_t row_rank = 0U;
  uint32_t plane_tiles = 0U;
  if (!stream_row_setup(tile_of, lane, index_metadata, row, k, tile_count, plane_count,
                        allocation_map_bytes, rank_prefix_count, terminal_map_value,
                        tile_table_bytes, row_rank, plane_tiles)) {
    if (lane == 0U) *out_slot = __int_as_float(0x7FC00000);
    return;
  }
  const uint32_t* words = reinterpret_cast<const uint32_t*>(payload) +
                          static_cast<size_t>(row_rank) * 13U;
  const __half2* scale_pairs = reinterpret_cast<const __half2*>(scales) + row_rank;
  const uint32_t total_words = plane_tiles * 13U;

  float accumulator = 0.0f;
  for (uint32_t index = lane; index < total_words; index += 32U) {
    const uint32_t plane_tile = index / 13U;
    const uint32_t word = index - plane_tile * 13U;
    const uint32_t bits = __ldcs(words + index);
    const float2 weight_scale = __half22float2(scale_pairs[plane_tile]);
    const uint32_t tile = tile_of[plane_tile];
    const int* chunk = reinterpret_cast<const int*>(
        row_quantized + static_cast<size_t>(tile) * kAllocationTile + word * 20U);
    const int a0 = __ldg(chunk);
    const int a1 = __ldg(chunk + 1);
    const int a2 = __ldg(chunk + 2);
    const int a3 = __ldg(chunk + 3);
    // Word 12's last four trits are zero padding past the tile; never read them.
    const int a4 = word == 12U ? 0 : __ldg(chunk + 4);

    const unsigned long long t0 = b3_int8[bits & 0xFFU];
    const unsigned long long t1 = b3_int8[(bits >> 8U) & 0xFFU];
    const unsigned long long t2 = b3_int8[(bits >> 16U) & 0xFFU];
    const unsigned long long t3 = b3_int8[bits >> 24U];
    // Twenty trits, five per byte, regrouped four at a time for dp4a.
    const int w0 = static_cast<int>(t0);
    const int w1 = static_cast<int>(((t0 >> 32U) & 0xFFULL) | (t1 << 8U));
    const int w2 = static_cast<int>(((t1 >> 24U) & 0xFFFFULL) | (t2 << 16U));
    const int w3 = static_cast<int>(((t2 >> 16U) & 0xFFFFFFULL) | (t3 << 24U));
    const int w4 = static_cast<int>(t3 >> 8U);

    // Trits 0-7 are dp4a groups 0-1 and 8-19 groups 2-4, so word 6's split at its
    // ninth trit falls on a group boundary and costs nothing.
    const int low = __dp4a(w1, a1, __dp4a(w0, a0, 0));
    const int high = __dp4a(w4, a4, __dp4a(w3, a3, __dp4a(w2, a2, 0)));
    const uint32_t group_base = tile * 2U;
    const float low_scale = (word <= 6U ? weight_scale.x : weight_scale.y) *
                            row_group_scale[group_base + (word <= 6U ? 0U : 1U)];
    const float high_scale = (word <= 5U ? weight_scale.x : weight_scale.y) *
                             row_group_scale[group_base + (word <= 5U ? 0U : 1U)];
    accumulator = fmaf(low_scale, static_cast<float>(low), accumulator);
    accumulator = fmaf(high_scale, static_cast<float>(high), accumulator);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    accumulator += __shfl_xor_sync(0xFFFFFFFFU, accumulator, offset);
  }
  if (lane == 0U) *out_slot = accumulator;
}

// Shared layout for the A8 kernels: the 256-entry int8 trit table (2 KiB), then
// one plane-tile table per warp.
constexpr uint32_t kB3Int8TableBytes = kB3TableEntries * sizeof(unsigned long long);

extern "C" __global__ void salt_v2_stream_i8(const signed char* __restrict__ quantized,
                                             const float* __restrict__ group_scale,
                                             const unsigned char* __restrict__ payload,
                                             const __half* __restrict__ scales,
                                             const unsigned char* __restrict__ index_metadata,
                                             float* __restrict__ output,
                                             uint32_t m,
                                             uint32_t n,
                                             uint32_t k,
                                             uint32_t tile_count,
                                             uint32_t plane_count,
                                             uint32_t allocation_map_bytes,
                                             uint32_t rank_prefix_count,
                                             uint32_t terminal_map_value,
                                             uint32_t tile_table_bytes) {
  extern __shared__ unsigned long long stream_i8_shared[];
  build_b3_int8_table(stream_i8_shared);
  __syncthreads();
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp_in_block = threadIdx.x >> 5U;
  const uint32_t warps_per_block = blockDim.x >> 5U;
  unsigned char* tile_of = reinterpret_cast<unsigned char*>(stream_i8_shared) +
                           kB3Int8TableBytes + static_cast<size_t>(warp_in_block) * tile_table_bytes;
  const uint64_t output_index =
      static_cast<uint64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  if (output_index >= static_cast<uint64_t>(m) * n) return;
  const uint32_t mi = static_cast<uint32_t>(output_index / n);
  const uint32_t row = static_cast<uint32_t>(output_index % n);
  stream_row_i8(stream_i8_shared, tile_of, lane, quantized + static_cast<size_t>(mi) * k,
                group_scale + static_cast<size_t>(mi) * (k / 128U), payload, scales,
                index_metadata, output + output_index, row, k, tile_count, plane_count,
                allocation_map_bytes, rank_prefix_count, terminal_map_value, tile_table_bytes);
}

extern "C" __global__ void salt_v2_stream_i8_multi(const signed char* __restrict__ quantized,
                                                   const float* __restrict__ group_scale,
                                                   const SaltStreamTensor* __restrict__ tensors,
                                                   uint32_t tensor_count,
                                                   uint32_t total_rows,
                                                   uint32_t k,
                                                   uint32_t tile_table_bytes) {
  extern __shared__ unsigned long long stream_i8_shared[];
  build_b3_int8_table(stream_i8_shared);
  __syncthreads();
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp_in_block = threadIdx.x >> 5U;
  const uint32_t warps_per_block = blockDim.x >> 5U;
  unsigned char* tile_of = reinterpret_cast<unsigned char*>(stream_i8_shared) +
                           kB3Int8TableBytes + static_cast<size_t>(warp_in_block) * tile_table_bytes;
  const uint32_t fused_row = blockIdx.x * warps_per_block + warp_in_block;
  if (fused_row >= total_rows) return;
  uint32_t index = 0U;
  while (index + 1U < tensor_count && fused_row >= tensors[index + 1U].first_row) ++index;
  const SaltStreamTensor tensor = tensors[index];
  const uint32_t row = fused_row - tensor.first_row;
  stream_row_i8(stream_i8_shared, tile_of, lane, quantized, group_scale,
                reinterpret_cast<const unsigned char*>(tensor.payload),
                reinterpret_cast<const __half*>(tensor.scales),
                reinterpret_cast<const unsigned char*>(tensor.index_metadata),
                reinterpret_cast<float*>(tensor.output) + row, row, k, tensor.tile_count,
                tensor.plane_count, tensor.allocation_map_bytes, tensor.rank_prefix_count,
                tensor.terminal_map_value, tile_table_bytes);
}

// Reconstruct selected semantic matrix rows directly from the resident codec
// payload. `rows` may repeat and its order is preserved, which makes this the
// token-embedding primitive for a `[vocab, hidden]` SALT V2 tensor.
extern "C" __global__ void salt_v2_gather_rows(
    const unsigned char* payload,
    const __half* scales,
    const unsigned char* index_metadata,
    const uint32_t* rows,
    float* output,
    uint32_t selected_rows,
    uint32_t n,
    uint32_t k,
    uint32_t codec,
    uint32_t scale_group_size,
    uint32_t tile_count,
    uint32_t plane_count,
    uint64_t payload_bytes,
    uint64_t scale_count,
    uint32_t allocation_map_bytes,
    uint32_t rank_prefix_count,
    uint32_t terminal_map_value) {
  const uint64_t output_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t output_count = static_cast<uint64_t>(selected_rows) * k;
  if (output_index >= output_count) return;

  const uint32_t selection = static_cast<uint32_t>(output_index / k);
  const uint32_t column = static_cast<uint32_t>(output_index % k);
  const uint32_t row = rows[selection];
  if (row >= n) return;
  const uint64_t total_coefficients = static_cast<uint64_t>(n) * k;
  const uint64_t coefficient = static_cast<uint64_t>(row) * k + column;
  const uint32_t tile = static_cast<uint32_t>(coefficient / kAllocationTile);
  const uint32_t local = static_cast<uint32_t>(coefficient % kAllocationTile);
  if (tile >= tile_count) return;

  const uint32_t rank_block = tile / kRankStrideTiles;
  uint32_t begin = 0U;
  if (rank_block != 0U) {
    const uint32_t prefix_index = rank_block - 1U;
    if (prefix_index >= rank_prefix_count) return;
    begin = read_rank_prefix(index_metadata, allocation_map_bytes, prefix_index);
  }
  const uint32_t scan_start = rank_block * kRankStrideTiles;
  for (uint32_t prior = scan_start; prior < tile; ++prior) {
    begin += plane_count_for_tile(
        index_metadata, allocation_map_bytes, terminal_map_value, prior);
  }
  const uint32_t planes = plane_count_for_tile(
      index_metadata, allocation_map_bytes, terminal_map_value, tile);
  const uint32_t end = begin + planes;
  if (planes == 0U || end > plane_count) return;

  const uint64_t tile_base = static_cast<uint64_t>(tile) * kAllocationTile;
  if (tile_base >= total_coefficients) return;
  const uint32_t logical_len = static_cast<uint32_t>(
      min(static_cast<uint64_t>(kAllocationTile), total_coefficients - tile_base));
  if (local >= logical_len) return;
  const uint32_t group = local / scale_group_size;
  const uint32_t full_payload_bytes = plane_payload_bytes(codec, kAllocationTile);
  const uint32_t current_payload_bytes = plane_payload_bytes(codec, logical_len);
  const uint32_t full_scale_count =
      (kAllocationTile + scale_group_size - 1U) / scale_group_size;
  const uint32_t current_scale_count =
      (logical_len + scale_group_size - 1U) / scale_group_size;

  float accumulator = 0.0f;
  for (uint32_t plane = begin; plane < end; ++plane) {
    const uint32_t local_plane = plane - begin;
    const uint64_t payload_base =
        static_cast<uint64_t>(begin) * full_payload_bytes +
        static_cast<uint64_t>(local_plane) * current_payload_bytes;
    const uint64_t scale_base = static_cast<uint64_t>(begin) * full_scale_count +
                                static_cast<uint64_t>(local_plane) * current_scale_count;
    const uint64_t scale_index = scale_base + group;
    if (scale_index >= scale_count) continue;
    const int trit = decode_trit(
        payload, payload_bytes, payload_base, logical_len,
        current_payload_bytes, local, codec);
    if (trit < 0) {
      accumulator = __fsub_rn(accumulator, __half2float(scales[scale_index]));
    } else if (trit > 0) {
      accumulator = __fadd_rn(accumulator, __half2float(scales[scale_index]));
    }
  }
  output[output_index] = accumulator;
}

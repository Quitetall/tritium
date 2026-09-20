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
    uint32_t codec) {
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
      // Five INDEPENDENT divisions by literals, which nvcc lowers to
      // multiply-shift. The previous `value /= 3` walk made each digit depend on
      // the one before it; ncu attributed 37.9% of this kernel's issue stalls to
      // exactly that kind of fixed-latency execution dependency.
      const uint32_t d0 = code % 3U;
      const uint32_t d1 = (code / 3U) % 3U;
      const uint32_t d2 = (code / 9U) % 3U;
      const uint32_t d3 = (code / 27U) % 3U;
      const uint32_t d4 = (code / 81U) % 3U;
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
          local_start, static_cast<uint32_t>(segment_len), codec);
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
                local_start, segment_len, codec);
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
  extern __shared__ float contribution_slots[];

  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp_in_block = threadIdx.x >> 5U;
  const uint32_t warps_per_block = blockDim.x >> 5U;
  const uint32_t slots_per_row = groups_per_row * kMaxPlanesPerTile;
  float* slots =
      contribution_slots + static_cast<size_t>(warp_in_block) * slots_per_row;

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
          static_cast<uint32_t>(segment_len), codec);
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

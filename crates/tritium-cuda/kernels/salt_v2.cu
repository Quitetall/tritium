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
    constexpr unsigned char kPlace[5] = {1, 3, 9, 27, 81};
    const uint32_t byte_index = local_index / 5U;
    if (byte_index >= plane_bytes) return 0;
    const unsigned char code = payload[base + byte_index];
    const unsigned char digit = (code / kPlace[local_index % 5U]) % 3U;
    return static_cast<int>(digit) - 1;
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
      float group_accumulator = 0.0f;
      for (uint64_t offset = 0; offset < segment_len; ++offset) {
        const uint32_t local = local_start + static_cast<uint32_t>(offset);
        const int trit = decode_trit(
            payload, payload_bytes, payload_base, logical_len,
            current_payload_bytes, local, codec);
        const float activation_value =
            activation[static_cast<uint64_t>(mi) * k +
                       (coefficient - row_base) + offset];
        if (trit < 0) {
          group_accumulator = __fsub_rn(group_accumulator, activation_value);
        } else if (trit > 0) {
          group_accumulator = __fadd_rn(group_accumulator, activation_value);
        }
      }
      const float contribution = __fmul_rn(
          group_accumulator, __half2float(scales[scale_index]));
      accumulator = __fadd_rn(accumulator, contribution);
    }
    coefficient += segment_len;
  }
  output[output_index] = accumulator;
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

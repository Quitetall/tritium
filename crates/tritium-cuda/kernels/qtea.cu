// Checked QTEA V1 bytes; no dense weight buffer. ADR0172.
// All offsets are derived from the format-owned checked layout on the host.
#include <cuda_runtime.h>

using U64 = unsigned long long;

__device__ __forceinline__ unsigned read_u32(const unsigned char* p) {
    return unsigned(p[0]) | (unsigned(p[1]) << 8) |
           (unsigned(p[2]) << 16) | (unsigned(p[3]) << 24);
}
__device__ __forceinline__ float read_f32(const unsigned char* p) {
    return __uint_as_float(read_u32(p));
}
__device__ __forceinline__ float fp8(unsigned char code) {
    const unsigned exponent = (code & 127u) >> 3;
    const unsigned fraction = code & 7u;
    const float magnitude = exponent == 0
        ? __fmul_rn(float(fraction), 0.001953125f)
        : __fmul_rn(float(8u + fraction), __uint_as_float((exponent + 117u) << 23));
    return __uint_as_float(__float_as_uint(magnitude) ^ (unsigned(code & 128u) << 24));
}

__device__ __forceinline__ float weight(
    const unsigned char* wire, unsigned row, unsigned column,
    unsigned rows, unsigned columns, unsigned group, unsigned residual_count,
    U64 trits, U64 params, U64 multipliers, U64 residuals, U64 record_bytes,
    U64 position_bytes) {
    const U64 flat = U64(row) * columns + column;
    const unsigned powers[5] = {1, 3, 9, 27, 81};
    const int trit = int((wire[trits + flat / 5] / powers[flat % 5]) % 3) - 1;
    const unsigned char* pair = wire + params + (U64(column / group) * rows + row) * 8;
    float result = __fadd_rn(
        __fmul_rn(__fmul_rn(read_f32(pair), read_f32(wire + multipliers + U64(column) * 4)),
                  float(trit)), read_f32(pair + 4));
    // Original-column records are sorted and unique. No dense column index.
    unsigned lo = 0, hi = residual_count;
    while (lo < hi) {
        const unsigned mid = lo + (hi - lo) / 2;
        if (read_u32(wire + residuals + U64(mid) * record_bytes) < column)
            lo = mid + 1;
        else hi = mid;
    }
    if (lo < residual_count) {
        const unsigned char* record = wire + residuals + U64(lo) * record_bytes;
        if (read_u32(record) == column) {
            const unsigned quartet = row / 4;
            const unsigned position = (record[8 + quartet / 4] >> ((quartet % 4) * 2)) & 3;
            if (position == row % 4) {
                result = __fadd_rn(result, __fdiv_rn(
                    fp8(record[8 + position_bytes + quartet]), read_f32(record + 4)));
            }
        }
    }
    return result;
}

#define QTEA_GEOMETRY unsigned rows, unsigned columns, unsigned group, unsigned residual_count, \
    U64 trits, U64 params, U64 multipliers, U64 residuals, U64 record_bytes, U64 position_bytes
#define QTEA_WEIGHT(row, col) weight(wire, row, col, rows, columns, group, residual_count, \
    trits, params, multipliers, residuals, record_bytes, position_bytes)

extern "C" __global__ void qtea_matmul_ordered(
    const unsigned char* wire, const float* input, float* output,
    unsigned count, QTEA_GEOMETRY) {
    // Exactly four complete warps per block, one output per warp. Bounds return
    // is uniform across a whole warp; no surviving lane skips a barrier.
    __shared__ float products[4][32];
    const unsigned lane = threadIdx.x % 32;
    const unsigned warp = threadIdx.x / 32;
    const unsigned item = blockIdx.x * 4 + warp;
    if (item >= count) return;
    const unsigned row = item % rows;
    const U64 input_base = U64(item / rows) * columns;
    float sum = 0.0f;
    for (unsigned start = 0; start < columns; start += 32) {
        const unsigned column = start + lane;
        products[warp][lane] = column < columns
            ? __fmul_rn(input[input_base + column], QTEA_WEIGHT(row, column)) : 0.0f;
        __syncwarp(0xffffffffu);
        if (lane == 0) {
            const unsigned valid = min(32u, columns - start);
            for (unsigned i = 0; i < valid; ++i)
                sum = __fadd_rn(sum, products[warp][i]);
        }
        __syncwarp(0xffffffffu); // Reader finishes before next overwrite.
    }
    if (lane == 0) output[item] = sum;
}

extern "C" __global__ void qtea_gather(
    const unsigned char* wire, const unsigned* row_ids, float* output,
    unsigned count, QTEA_GEOMETRY) {
    const unsigned item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < count) {
        const unsigned row = row_ids[item / columns];
        const unsigned column = item % columns;
        output[item] = QTEA_WEIGHT(row, column);
    }
}

// Separate explicit numerical mode: device expf is not host exp bit parity.
// Status is private per-call scratch and is inspected before host publication.
extern "C" __global__ void qtea_swiglu_device(
    float* gate, const float* up, unsigned* status, unsigned count) {
    const U64 item = U64(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item < count) {
        const float z = gate[item];
        const float u = up[item];
        const float value = __fmul_rn(__fdiv_rn(z, __fadd_rn(1.0f, expf(-z))), u);
        if (!isfinite(z) || !isfinite(u) || !isfinite(value)) atomicOr(status, 1u);
        gate[item] = value;
    }
}

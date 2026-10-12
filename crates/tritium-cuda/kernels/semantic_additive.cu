// Private packed semantic payload. Canonical plane encodings; no dense shadow.
// --fmad=false pins plane/group-major multiply-then-add reference arithmetic.
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t u32le(const uint8_t *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
        (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
__device__ __forceinline__ uint32_t h(const uint8_t *p, uint32_t i) {
    return u32le(p + 4 * i);
}
__device__ __forceinline__ int digit(const uint8_t *p, uint32_t codec, uint32_t i) {
    if (codec == 0) return int((p[i / 4] >> (2 * (i % 4))) & 3) - 1;
    if (codec == 1) {
        uint32_t code = p[i / 5];
        for (uint32_t j = 0; j < i % 5; ++j) code /= 3;
        return int(code % 3) - 1;
    }
    uint32_t bit = (i / 4) * 5, byte = bit / 8, shift = bit % 8;
    uint32_t word = p[byte];
    if (shift > 3) word |= uint32_t(p[byte + 1]) << 8;
    uint32_t code = (word >> shift) & 31, lane = i % 4, zero = code & 3;
    if (lane == zero) return 0;
    uint32_t sign = lane - (lane > zero ? 1 : 0);
    return code & (1 << (2 + sign)) ? 1 : -1;
}
__device__ __forceinline__ float scale(const uint8_t *p, uint32_t plane, uint32_t index) {
    uint32_t width = h(p, 5);
    const uint8_t *s = p + h(p, 9) + index * width;
    float result;
    if (width == 2) {
        unsigned short bits = (unsigned short)(uint32_t(s[0]) | (uint32_t(s[1]) << 8));
        result = __half2float(__ushort_as_half(bits));
    } else result = __uint_as_float(u32le(s));
    if (!h(p, 6)) {
        float ratio = __uint_as_float(h(p, 8));
        for (uint32_t i = 0; i < plane; ++i) result *= ratio;
    }
    return result;
}
__device__ __forceinline__ float add_digit(float sum, float value, int trit) {
    if (trit == 1) return sum + value;
    if (trit == -1) return sum - value;
    return sum;
}
__device__ __forceinline__ uint32_t row_meta(const uint8_t *p, uint32_t plane, uint32_t row, bool scales) {
    uint32_t planes = h(p, 2), rows = h(p, 0);
    uint32_t i = planes + (scales ? planes * (rows + 1) : 0) + plane * (rows + 1) + row;
    return u32le(p + h(p, 10) + 4 * i);
}
__device__ __forceinline__ uint32_t plane_start(const uint8_t *p, uint32_t plane) {
    return h(p, 13) ? u32le(p + h(p, 10) + 4 * plane) : 60 + plane * h(p, 12);
}
extern "C" __global__ void semantic_additive_matmul(
    const uint8_t *p, const float *act, float *out, uint32_t batch
) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t rows = h(p, 0), cols = h(p, 1), planes = h(p, 2);
    if (at >= uint64_t(batch) * rows) return;
    uint32_t row = at % rows;
    const float *a = act + (at / rows) * cols;
    bool compact = h(p, 13), tensor = h(p, 7);
    uint32_t group = tensor ? cols : h(p, 3), codec = h(p, 4);
    float sum = 0.0f;
    for (uint32_t plane = 0; plane < planes; ++plane) {
        const uint8_t *data = p + plane_start(p, plane);
        if (!compact) {
            for (uint32_t start = 0; start < cols; start += group) {
                uint32_t end = min(start + group, cols);
                float partial = 0.0f;
                for (uint32_t col = start; col < end; ++col)
                    partial = add_digit(partial, a[col], digit(data, codec, row * cols + col));
                uint32_t unit = tensor ? 0 : row * ((cols + group - 1) / group) + start / group;
                uint32_t index = h(p, 6) ? plane * h(p, 14) + unit : unit;
                sum += partial * scale(p, plane, index);
            }
        } else {
            uint32_t tile_width = h(p, 12), tiles = (cols + tile_width - 1) / tile_width;
            uint32_t coefficient = row_meta(p, plane, row, false);
            uint32_t si = row_meta(p, plane, row, true);
            float tensor_partial = 0.0f;
            for (uint32_t tile = 0; tile < tiles; ++tile) {
                uint32_t count = p[h(p, 11) + row * tiles + tile];
                uint32_t col = tile * tile_width, len = min(tile_width, cols - col);
                if (count <= plane) {
                    if (!tensor && !h(p, 6) && count > 0) si += (len + group - 1) / group;
                    continue;
                }
                uint32_t width = tensor ? len : group;
                for (uint32_t start = 0; start < len; start += width) {
                    float partial = tensor ? tensor_partial : 0.0f;
                    uint32_t end = min(start + width, len);
                    for (uint32_t offset = start; offset < end; ++offset)
                        partial = add_digit(partial, a[col + offset], digit(data, codec, coefficient + offset));
                    if (tensor) tensor_partial = partial;
                    else sum += partial * scale(p, plane, si + start / group);
                }
                coefficient += len;
                if (!tensor) si += (len + group - 1) / group;
            }
            if (tensor) sum += tensor_partial * scale(p, plane, plane);
        }
    }
    out[at] = sum;
}
extern "C" __global__ void semantic_additive_gather(
    const uint8_t *p, const uint32_t *ids, float *out, uint32_t selected
) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t rows = h(p, 0), cols = h(p, 1), planes = h(p, 2);
    if (at >= uint64_t(selected) * cols) return;
    uint32_t row = ids[at / cols], col = at % cols, group = h(p, 3), codec = h(p, 4);
    bool compact = h(p, 13), tensor = h(p, 7);
    float sum = 0.0f;
    for (uint32_t plane = 0; plane < planes; ++plane) {
        uint32_t coefficient, index;
        if (!compact) {
            coefficient = row * cols + col;
            uint32_t unit = tensor ? 0 : row * ((cols + group - 1) / group) + col / group;
            index = h(p, 6) ? plane * h(p, 14) + unit : unit;
        } else {
            uint32_t tile_width = h(p, 12), tiles = (cols + tile_width - 1) / tile_width, target = col / tile_width;
            if (p[h(p, 11) + row * tiles + target] <= plane) continue;
            coefficient = row_meta(p, plane, row, false);
            index = row_meta(p, plane, row, true);
            for (uint32_t tile = 0; tile < target; ++tile) {
                uint32_t count = p[h(p, 11) + row * tiles + tile];
                uint32_t len = min(tile_width, cols - tile * tile_width);
                if (count > plane) coefficient += len;
                if (!tensor && (h(p, 6) ? count > plane : count > 0)) index += (len + group - 1) / group;
            }
            coefficient += col % tile_width;
            index = tensor ? plane : index + (col % tile_width) / group;
        }
        sum += float(digit(p + plane_start(p, plane), codec, coefficient)) * scale(p, plane, index);
    }
    out[at] = sum;
}

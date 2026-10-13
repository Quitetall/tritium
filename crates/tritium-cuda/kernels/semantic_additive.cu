// Private packed semantic payload. Canonical plane encodings; no dense shadow.
// --fmad=false pins plane/group-major multiply-then-add reference arithmetic.
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void semantic_decoder_dense_gather(
    const float *table, const uint32_t *ids, float *output, uint32_t cols, uint32_t count
) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at < count) output[at] = table[uint64_t(ids[at / cols]) * cols + at % cols];
}
// Same 256-slot fold/tree as the canonical host inference RMSNorm; sqrt is the
// explicit device-f32 profile. In-place writes occur after the reduction barrier.
extern "C" __global__ void semantic_decoder_norm(float *x, const float *w, uint32_t cols, float eps) {
    __shared__ float sum[256];
    uint32_t lane = threadIdx.x;
    uint64_t base = uint64_t(blockIdx.x) * cols;
    float s = 0.0f;
    for (uint32_t c = lane; c < cols; c += 256) { float v = x[base + c]; s += v * v; }
    sum[lane] = s;
    __syncthreads();
    for (uint32_t width = 128; width > 0; width >>= 1) {
        if (lane < width) sum[lane] += sum[lane + width];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(sum[0] / float(cols) + eps);
    for (uint32_t c = lane; c < cols; c += 256) x[base + c] = (x[base + c] * inv) * w[c];
}
extern "C" __global__ void semantic_decoder_bias(float *x, const float *bias, uint32_t cols, uint32_t count) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at < count) x[at] += bias[at % cols];
}
extern "C" __global__ void semantic_decoder_add(float *x, const float *delta, uint32_t count) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at < count) x[at] += delta[at];
}
extern "C" __global__ void semantic_decoder_relu2(float *gate, const float *up, uint32_t count) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at < count) { float r = fmaxf(gate[at], 0.0f); gate[at] = (r * r) * up[at]; }
}
extern "C" __global__ void semantic_decoder_rope(float *x, const float *table,
    uint32_t heads, uint32_t dim, uint32_t pairs
) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= pairs) return;
    uint32_t half = dim / 2, token = at / (heads * half), pair = at % half;
    uint64_t base = (at / half) * dim;
    float a = x[base + pair], b = x[base + pair + half];
    float c = table[uint64_t(token) * dim + pair], s = table[uint64_t(token) * dim + pair + half];
    x[base + pair] = a * c - b * s;
    x[base + pair + half] = b * c + a * s;
}

// Explicit device-f32 profile. Online softmax uses O(head_dim) block scratch;
// no [query, context] score matrix and no KV host readback. Default host graph
// and existing exact matrix/basis/legacy-resident kernels remain unchanged.
extern "C" __global__ void semantic_causal_attention(
    const float *query, const float *key, const float *value, float *output,
    uint32_t query_heads, uint32_t kv_heads, uint32_t dim, uint32_t offset, float scale
) {
    extern __shared__ float shared[];
    const uint32_t lane = threadIdx.x;
    const uint32_t head = blockIdx.x % query_heads;
    const uint32_t row = blockIdx.x / query_heads;
    const uint32_t kv_head = head / (query_heads / kv_heads);
    const uint64_t q_at = (uint64_t(row) * query_heads + head) * dim;
    const float q = lane < dim ? query[q_at + lane] : 0.0f;
    float accumulator = 0.0f;
    float maximum = -INFINITY, denominator = 0.0f;
    for (uint32_t token = 0; token <= offset + row; ++token) {
        const uint64_t kv_at = (uint64_t(token) * kv_heads + kv_head) * dim;
        shared[lane] = lane < dim ? q * key[kv_at + lane] : 0.0f;
        __syncthreads();
        for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) shared[lane] += shared[lane + stride];
            __syncthreads();
        }
        if (lane == 0) {
            float score = shared[0] * scale;
            float next_max = fmaxf(maximum, score);
            float alpha = expf(maximum - next_max);
            float weight = expf(score - next_max);
            denominator = denominator * alpha + weight;
            maximum = next_max;
            shared[blockDim.x] = alpha;
            shared[blockDim.x + 1] = weight;
            shared[blockDim.x + 2] = denominator;
        }
        __syncthreads();
        if (lane < dim) {
            accumulator = accumulator * shared[blockDim.x]
                + shared[blockDim.x + 1] * value[kv_at + lane];
        }
        __syncthreads();
    }
    if (lane < dim) output[q_at + lane] = accumulator / shared[blockDim.x + 2];
}

extern "C" __global__ void semantic_swiglu(float *gate, const float *up, uint32_t count) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at < count) {
        float z = gate[at];
        gate[at] = (z / (1.0f + expf(-z))) * up[at];
    }
}

__device__ __forceinline__ uint64_t semantic_splitmix(uint64_t value) {
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}
__device__ __forceinline__ bool semantic_negative(uint64_t seed, uint64_t domain, uint32_t col) {
    uint64_t tagged = semantic_splitmix(seed ^ UINT64_C(0x5452495449554d31));
    uint64_t keyed = semantic_splitmix(tagged ^ domain);
    return semantic_splitmix(keyed ^ (uint64_t(col) * UINT64_C(0x9e3779b97f4a7c15))) & 1;
}
extern "C" __global__ void semantic_basis_sign(
    float *values, uint32_t count, uint32_t cols, uint64_t seed, uint64_t domain
) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at < count && semantic_negative(seed, domain, uint32_t(at % cols))) values[at] = -values[at];
}
extern "C" __global__ void semantic_basis_butterfly(
    float *values, uint32_t pairs, uint32_t cols, uint32_t block, uint32_t width
) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= pairs) return;
    uint32_t row_pairs = cols / 2, block_pairs = block / 2;
    uint32_t pair = at % row_pairs, local = pair % block_pairs;
    uint64_t left = (at / row_pairs) * cols + (pair / block_pairs) * block
        + (local / width) * (2 * width) + local % width;
    uint64_t right = left + width;
    float a = values[left], b = values[right];
    values[left] = a + b;
    values[right] = a - b;
}
extern "C" __global__ void semantic_basis_finish(
    float *values, uint32_t count, uint32_t cols, float normalization,
    uint32_t inverse_sign, uint64_t seed, uint64_t domain
) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= count) return;
    float value = values[at] * normalization;
    if (inverse_sign && semantic_negative(seed, domain, uint32_t(at % cols))) value = -value;
    values[at] = value;
}

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

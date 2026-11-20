// Private packed semantic payload. Canonical plane encodings; no dense shadow.
// --fmad=false pins plane/group-major multiply-then-add reference arithmetic.
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void semantic_delta_convolution(
    const float *raw, const float *weights, float *history, float *output,
    uint32_t width, uint32_t kernel, uint32_t tokens
) {
    uint64_t channel = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= width) return;
    uint64_t base = channel * kernel;
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t tap = 1; tap < kernel; ++tap) history[base + tap - 1] = history[base + tap];
        history[base + kernel - 1] = raw[uint64_t(token) * width + channel];
        float sum = 0.0f;
        for (uint32_t tap = 0; tap < kernel; ++tap) sum += history[base + tap] * weights[base + tap];
        output[uint64_t(token) * width + channel] = sum / (1.0f + expf(-sum));
    }
}
// Explicit device-f32 profile. Existing exact host-normalized recurrence stays
// unchanged. A block owns one value head, traversing tokens in order on device.
extern "C" __global__ void semantic_delta_mixer(
    const float *qkv, const float *z, const float *beta_logits, const float *decay_logits,
    const float *a_log, const float *dt_bias, const float *norm,
    float *state, float *output, uint32_t heads, uint32_t dk, uint32_t dv,
    uint32_t group, uint32_t tokens, float query_scale, float eps
) {
    extern __shared__ float shared[];
    float *kk = shared, *qq = shared + dk, *core = shared + 2 * dk, *scalars = core + dv;
    uint32_t head = blockIdx.x, lane = threadIdx.x, key_head = head / group;
    uint64_t key_width = uint64_t(heads / group) * dk;
    uint64_t value_width = uint64_t(heads) * dv, conv_width = 2 * key_width + value_width;
    uint64_t state_base = uint64_t(head) * dk * dv;
    for (uint32_t token = 0; token < tokens; ++token) {
        uint64_t raw_base = uint64_t(token) * conv_width;
        if (lane == 0) {
            float q_sq = 0.0f, k_sq = 0.0f;
            for (uint32_t i = 0; i < dk; ++i) {
                float q = qkv[raw_base + uint64_t(key_head) * dk + i];
                float k = qkv[raw_base + key_width + uint64_t(key_head) * dk + i];
                q_sq += q * q; k_sq += k * k;
            }
            float q_inv = (1.0f / sqrtf(q_sq + 1e-6f)) * query_scale;
            float k_inv = 1.0f / sqrtf(k_sq + 1e-6f);
            for (uint32_t i = 0; i < dk; ++i) {
                qq[i] = qkv[raw_base + uint64_t(key_head) * dk + i] * q_inv;
                kk[i] = qkv[raw_base + key_width + uint64_t(key_head) * dk + i] * k_inv;
            }
            uint64_t at = uint64_t(token) * heads + head;
            scalars[0] = 1.0f / (1.0f + expf(-beta_logits[at]));
            float step = decay_logits[at] + dt_bias[head];
            float softplus = step > 20.0f ? step : log1pf(expf(step));
            scalars[1] = expf(-expf(a_log[head]) * softplus);
        }
        __syncthreads();
        uint64_t value_at = uint64_t(token) * value_width + uint64_t(head) * dv + lane;
        if (lane < dv) {
            float memory = 0.0f;
            for (uint32_t i = 0; i < dk; ++i) {
                uint64_t at = state_base + uint64_t(i) * dv + lane;
                state[at] *= scalars[1]; memory += kk[i] * state[at];
            }
            float delta = scalars[0] * (qkv[raw_base + 2 * key_width + uint64_t(head) * dv + lane] - memory);
            float mixed = 0.0f;
            for (uint32_t i = 0; i < dk; ++i) {
                uint64_t at = state_base + uint64_t(i) * dv + lane;
                state[at] += kk[i] * delta; mixed += qq[i] * state[at];
            }
            core[lane] = mixed;
        }
        __syncthreads();
        if (lane == 0) {
            float variance = 0.0f;
            for (uint32_t i = 0; i < dv; ++i) variance += core[i] * core[i];
            scalars[2] = 1.0f / sqrtf(variance / float(dv) + eps);
        }
        __syncthreads();
        if (lane < dv) {
            float gate = z[value_at] / (1.0f + expf(-z[value_at]));
            output[value_at] = ((core[lane] * scalars[2]) * norm[lane]) * gate;
        }
        __syncthreads();
    }
}

extern "C" __global__ void semantic_decoder_dense_gather(
    const float *table, const uint32_t *ids, float *output, uint32_t cols, uint32_t count
) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at < count) output[at] = table[uint64_t(ids[at / cols]) * cols + at % cols];
}
// Same 256-slot fold/tree as the canonical host inference RMSNorm; sqrt is the
// explicit device-f32 profile. In-place writes occur after the reduction barrier.
extern "C" __global__ void semantic_decoder_norm(float *x, const float *w, uint32_t cols, float eps, uint32_t zero_centered) {
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
    for (uint32_t c = lane; c < cols; c += 256) {
        float weight = zero_centered ? 1.0f + w[c] : w[c];
        x[base + c] = (x[base + c] * inv) * weight;
    }
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
    uint32_t heads, uint32_t dim, uint32_t rotary, uint32_t pairs
) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= pairs) return;
    uint32_t half = rotary / 2, token = at / (heads * half), pair = at % half;
    uint64_t base = (at / half) * dim;
    float a = x[base + pair], b = x[base + pair + half];
    float c = table[uint64_t(token) * rotary + pair], s = table[uint64_t(token) * rotary + pair + half];
    x[base + pair] = a * c - b * s;
    x[base + pair + half] = b * c + a * s;
}
extern "C" __global__ void semantic_qwen_query_gate(const float *fused, float *query, float *gate, uint32_t dim, uint32_t count) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at < count) {
        uint64_t source = (at / dim) * (2 * dim) + at % dim;
        query[at] = fused[source]; gate[at] = fused[source + dim];
    }
}
extern "C" __global__ void semantic_qwen_attention_gate(float *attention, const float *gate, uint32_t count) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at < count) attention[at] *= 1.0f / (1.0f + expf(-gate[at]));
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

extern "C" __global__ void semantic_swiglu(float *gate, const float *up, uint32_t *invalid, uint32_t count) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at < count) {
        float z = gate[at];
        float u = up[at];
        float value = (z / (1.0f + expf(-z))) * u;
        if (!isfinite(z) || !isfinite(u) || !isfinite(value)) atomicOr(invalid, 1u);
        gate[at] = value;
    }
}

// Preserve invalid transformed input even when a pruned matrix masks it.
extern "C" __global__ void semantic_activation_check_finite(
    const float *values, uint32_t *invalid, uint32_t count) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at < count && !isfinite(values[at])) atomicOr(invalid, 1u);
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
template<bool cooperative>
__device__ __forceinline__ float semantic_ordered_partial(
    float partial, const float *a, const uint8_t *data, uint32_t codec,
    uint32_t coefficient, uint32_t begin, uint32_t end
);
template<bool cooperative>
__device__ __forceinline__ float semantic_additive_dot(
    const uint8_t *p, const float *a, uint32_t row
) {
    uint32_t rows = h(p, 0), cols = h(p, 1), planes = h(p, 2);
    bool compact = h(p, 13), tensor = h(p, 7);
    uint32_t group = tensor ? cols : h(p, 3), codec = h(p, 4);
    float sum = 0.0f;
    for (uint32_t plane = 0; plane < planes; ++plane) {
        const uint8_t *data = p + plane_start(p, plane);
        if (!compact) {
            for (uint32_t start = 0; start < cols; start += group) {
                uint32_t end = min(start + group, cols);
                float partial = semantic_ordered_partial<cooperative>(
                    0.0f, a, data, codec, row * cols, start, end);
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
                    partial = semantic_ordered_partial<cooperative>(
                        partial, a + col, data, codec, coefficient, start, end);
                    if (tensor) tensor_partial = partial;
                    else sum += partial * scale(p, plane, si + start / group);
                }
                coefficient += len;
                if (!tensor) si += (len + group - 1) / group;
            }
            if (tensor) sum += tensor_partial * scale(p, plane, plane);
        }
    }
    return sum;
}
// Cooperative reads, deliberately NOT a parallel reduction: lane zero folds
// every digit in exactly the scalar kernel's column order. All 32 lanes take
// the same shuffle path, including ragged tails and skipped compact tiles.
template<bool cooperative>
__device__ __forceinline__ float semantic_ordered_partial(
    float partial, const float *a, const uint8_t *data, uint32_t codec,
    uint32_t coefficient, uint32_t begin, uint32_t end
) {
    if (!cooperative) {
        for (uint32_t at = begin; at < end; ++at)
            partial = add_digit(partial, a[at], digit(data, codec, coefficient + at));
        return partial;
    }
    uint32_t lane = threadIdx.x & 31;
    for (uint32_t start = begin; start < end; start += 32) {
        uint32_t at = start + lane;
        float value = at < end ? a[at] : 0.0f;
        int trit = at < end ? digit(data, codec, coefficient + at) : 0;
        uint32_t count = min(32u, end - start);
        for (uint32_t source = 0; source < count; ++source) {
            float v = __shfl_sync(0xffffffffu, value, source);
            int t = __shfl_sync(0xffffffffu, trit, source);
            if (lane == 0) partial = add_digit(partial, v, t);
        }
    }
    return partial;
}
extern "C" __global__ void semantic_additive_matmul(
    const uint8_t *p, const float *act, float *out, uint32_t batch
) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t rows = h(p, 0), cols = h(p, 1);
    if (at >= uint64_t(batch) * rows) return;
    out[at] = semantic_additive_dot<false>(p, act + (at / rows) * cols, at % rows);
}
extern "C" __global__ void semantic_additive_matmul_ordered_warp(
    const uint8_t *p, const float *act, float *out, uint32_t batch
) {
    uint64_t at = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    uint32_t rows = h(p, 0), cols = h(p, 1);
    if (at >= uint64_t(batch) * rows) return;
    float sum = semantic_additive_dot<true>(p, act + (at / rows) * cols, at % rows);
    if ((threadIdx.x & 31) == 0) out[at] = sum;
}
// Private cross-row kernels. Dispatch selects the measured 32-row geometry
// only; the 8-row variant remains an explicit comparison diagnostic.
// Uniform B3, free F16 group-128 only. Cooperative lanes stage at most 27
// canonical bytes per row/group (28-byte stride avoids a 32-byte bank stride).
// One consumer thread owns each row and retains *all* scalar column, group and
// plane accumulation order. No parallel reduction, repacked weights, dense
// shadow or retained workspace. Every thread reaches both group barriers,
// including ragged row/column tails and partially occupied final blocks.
template<uint32_t tile_rows>
__device__ __forceinline__ void semantic_b3_rows(
    const uint8_t *p, const float *act, float *out, uint32_t batch, float *shared
) {
    uint32_t rows = h(p, 0), cols = h(p, 1), planes = h(p, 2);
    if (h(p, 13) || h(p, 4) != 1 || h(p, 3) != 128 ||
        h(p, 5) != 2 || h(p, 6) != 1 || h(p, 7) || blockIdx.y >= batch) return;
    float *values = shared;
    uint8_t *codes = reinterpret_cast<uint8_t *>(shared + 128);
    uint32_t lane = threadIdx.x, first_row = blockIdx.x * tile_rows;
    uint32_t row = first_row + lane;
    float sum = 0.0f;
    for (uint32_t plane = 0; plane < planes; ++plane) {
        const uint8_t *data = p + plane_start(p, plane);
        for (uint32_t start = 0; start < cols; start += 128) {
            uint32_t len = min(128u, cols - start);
            values[lane] = lane < len ? act[uint64_t(blockIdx.y) * cols + start + lane] : 0.0f;
            for (uint32_t i = lane; i < tile_rows * 28; i += 128) {
                uint32_t local_row = i / 28, byte = i % 28;
                uint32_t coefficient = (first_row + local_row) * cols + start;
                uint32_t needed = (coefficient % 5 + len + 4) / 5;
                codes[i] = first_row + local_row < rows && byte < needed
                    ? data[coefficient / 5 + byte] : 0;
            }
            __syncthreads();
            if (lane < tile_rows && row < rows) {
                uint32_t phase = (row * cols + start) % 5;
                const uint8_t *row_codes = codes + lane * 28;
                float partial = 0.0f;
                for (uint32_t col = 0; col < len; ++col)
                    partial = add_digit(partial, values[col], digit(row_codes, 1, phase + col));
                uint32_t unit = row * ((cols + 127) / 128) + start / 128;
                sum += partial * scale(p, plane, plane * h(p, 14) + unit);
            }
            __syncthreads();
        }
    }
    if (lane < tile_rows && row < rows) out[uint64_t(blockIdx.y) * rows + row] = sum;
}
extern "C" __global__ void semantic_additive_matmul_b3_rows_8(
    const uint8_t *p, const float *act, float *out, uint32_t batch
) {
    extern __shared__ float shared[];
    semantic_b3_rows<8>(p, act, out, batch, shared);
}
extern "C" __global__ void semantic_additive_matmul_b3_rows_32(
    const uint8_t *p, const float *act, float *out, uint32_t batch
) {
    extern __shared__ float shared[];
    semantic_b3_rows<32>(p, act, out, batch, shared);
}
// Group-parallel kernel, selected only for measured decode families. Each group starts at +0 and folds
// its columns in the original order. After a CTA-wide barrier, thread zero
// folds the completed partials in the original plane/group order, including
// the separate f32 multiply/add (--fmad=false). This is NOT a tree reduction.
// At most 3 * ceil(17408/128) = 408 partials fit the 512-float shared scratch;
// no persistent/global partial buffer, repacking or dense shadow is needed.
extern "C" __global__ void semantic_additive_matmul_b3_groups(
    const uint8_t *p, const float *act, float *out, uint32_t batch
) {
    uint32_t rows = h(p, 0), cols = h(p, 1), planes = h(p, 2);
    if (blockDim.x != 128 || blockIdx.x >= rows || blockIdx.y >= batch ||
        !cols || cols > 17408 || !planes || planes > 3 || h(p, 13) ||
        h(p, 4) != 1 || h(p, 3) != 128 || h(p, 5) != 2 ||
        h(p, 6) != 1 || h(p, 7)) return;
    extern __shared__ float partials[];
    uint32_t row = blockIdx.x, groups = (cols + 127) / 128;
    const float *a = act + uint64_t(blockIdx.y) * cols;
    for (uint32_t unit = threadIdx.x; unit < planes * groups; unit += blockDim.x) {
        uint32_t plane = unit / groups, group = unit - plane * groups;
        const uint8_t *data = p + plane_start(p, plane);
        uint32_t start = group * 128, end = min(start + 128, cols);
        float partial = 0.0f;
        for (uint32_t col = start; col < end; ++col)
            partial = add_digit(partial, a[col], digit(data, 1, row * cols + col));
        partials[unit] = partial;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (uint32_t plane = 0; plane < planes; ++plane)
            for (uint32_t group = 0; group < groups; ++group)
                sum += partials[plane * groups + group] *
                    scale(p, plane, plane * h(p, 14) + row * groups + group);
        out[uint64_t(blockIdx.y) * rows + row] = sum;
    }
}
__device__ __forceinline__ float semantic_stored_weight(const uint8_t *p, uint32_t row, uint32_t col) {
    uint32_t cols = h(p, 1), planes = h(p, 2), group = h(p, 3), codec = h(p, 4);
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
    return sum;
}
extern "C" __global__ void semantic_additive_gather(
    const uint8_t *p, const uint32_t *ids, float *out, uint32_t selected
) {
    uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t cols = h(p, 1);
    if (at >= uint64_t(selected) * cols) return;
    out[at] = semantic_stored_weight(p, ids[at / cols], at % cols);
}
// Private-upload check only, after canonical bytes and native metadata validation.
// Grid-stride indexing covers large/pruned logical geometry without a full row.
extern "C" __global__ void semantic_additive_validate_finite(const uint8_t *p, uint32_t *invalid) {
    uint32_t cols = h(p, 1);
    uint64_t total = uint64_t(h(p, 0)) * cols;
    uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; at < total; at += stride) {
        if (!isfinite(semantic_stored_weight(p, at / cols, at % cols))) atomicExch(invalid, 1u);
    }
}

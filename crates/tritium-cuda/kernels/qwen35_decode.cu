// Qwen3.5/Qwen3.6 resident decode kernels (SALT decode campaign, Phase 2).
//
// Everything a Qwen3.6 decode step does outside its projections, on the device:
// the zero-centered RMSNorms and residual adds, the Gated DeltaNet's causal conv,
// gating scalars and gated norm, full attention's query/gate split, head norms,
// partial RoPE, KV append, GQA attention and output gate, the SwiGLU product, and
// the greedy argmax. Before this, each ran on the host between SALT launches, so
// every one of ~500 projections per token paid an upload, a blocking download and
// two host scans -- 23 of every 37.6 ms per token was spent outside any kernel.
//
// These are fast-tier kernels. Reductions run in tree order, and transcendentals
// use CUDA's own `expf`/`sincosf`/`log1pf`, so results are close to the host path
// rather than bit-identical to it; the resident executor is gated on relative
// error and greedy-token identity against the host forward, never on equality.
// A bit-exact exact tier needs canonical transcendentals shared with the host,
// which is separate, owner-gated work.
//
// Every kernel that depends on the decode position reads it from `ctrl[0]`, so a
// launch sequence can later be captured once into a CUDA graph and replayed.

#include <stdint.h>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xFFFFFFFFU, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFU, value, offset));
  }
  return value;
}

// Block-wide sum. `scratch` holds one float per warp; every thread gets the total.
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
  value = warp_max(value);
  __syncthreads();
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  float total = -INFINITY;
  for (int index = 0; index < warps; ++index) total = fmaxf(total, scratch[index]);
  return total;
}

__device__ __forceinline__ float sigmoidf(float value) { return 1.0f / (1.0f + expf(-value)); }

__device__ __forceinline__ float siluf(float value) { return value / (1.0f + expf(-value)); }

}  // namespace

// residual += branch (when `add`), then out = rmsnorm(residual) * (1 + w) or * w.
//
// One block per row. Fusing the add into the norm means the residual stream is
// read once per sub-layer instead of twice.
extern "C" __global__ void q35_add_rmsnorm(float* residual,
                                           const float* branch,
                                           const float* weight,
                                           float* out,
                                           int n,
                                           float eps,
                                           int add,
                                           int one_plus) {
  __shared__ float scratch[32];
  float* row = residual + static_cast<size_t>(blockIdx.x) * n;
  const float* branch_row = branch + static_cast<size_t>(blockIdx.x) * n;
  float* out_row = out + static_cast<size_t>(blockIdx.x) * n;
  float squares = 0.0f;
  for (int index = threadIdx.x; index < n; index += blockDim.x) {
    float value = row[index];
    if (add) {
      value += branch_row[index];
      row[index] = value;
    }
    squares += value * value;
  }
  const float inverse = rsqrtf(block_sum(squares, scratch) / static_cast<float>(n) + eps);
  for (int index = threadIdx.x; index < n; index += blockDim.x) {
    const float scale = one_plus ? 1.0f + weight[index] : weight[index];
    out_row[index] = row[index] * inverse * scale;
  }
}

// Full-attention prep for one decode token. Blocks [0, n_head) are query heads,
// blocks [n_head, n_head + n_kv) are key/value heads; one thread per head lane.
//
// q_proj emits query and gate interleaved per head ([head][query | gate]); this
// splits them, RMSNorms each query and key head with its (1 + w) weights, applies
// NeoX partial RoPE over the first `rotary_dim` lanes at position `ctrl[0]`, and
// appends the rotated key and the value to the cache row for that position.
extern "C" __global__ void q35_attn_prep(const float* fused,
                                         const float* key_in,
                                         const float* value_in,
                                         const float* q_norm,
                                         const float* k_norm,
                                         const float* inv_freq,
                                         float* query_out,
                                         float* gate_out,
                                         float* key_cache,
                                         float* value_cache,
                                         const int* ctrl,
                                         int n_head,
                                         int n_kv,
                                         int head_dim,
                                         int rotary_dim,
                                         float eps) {
  extern __shared__ float lanes[];  // head_dim floats
  __shared__ float scratch[32];
  const int lane = threadIdx.x;
  const int position = ctrl[0];
  const bool is_query = static_cast<int>(blockIdx.x) < n_head;
  const int head = is_query ? blockIdx.x : blockIdx.x - n_head;
  const int kv_width = n_kv * head_dim;

  float value = 0.0f;
  if (lane < head_dim) {
    if (is_query) {
      const float* pair = fused + static_cast<size_t>(head) * 2 * head_dim;
      value = pair[lane];
      gate_out[head * head_dim + lane] = pair[head_dim + lane];
    } else {
      value = key_in[head * head_dim + lane];
      value_cache[static_cast<size_t>(position) * kv_width + head * head_dim + lane] =
          value_in[head * head_dim + lane];
    }
  }
  const float inverse =
      rsqrtf(block_sum(lane < head_dim ? value * value : 0.0f, scratch) /
                 static_cast<float>(head_dim) +
             eps);
  const float* norm = is_query ? q_norm : k_norm;
  if (lane < head_dim) lanes[lane] = value * inverse * (1.0f + norm[lane]);
  __syncthreads();

  if (lane < head_dim) {
    float rotated = lanes[lane];
    const int half = rotary_dim / 2;
    if (lane < rotary_dim) {
      const int pair_lane = lane < half ? lane : lane - half;
      float sine;
      float cosine;
      sincosf(static_cast<float>(position) * inv_freq[pair_lane], &sine, &cosine);
      const float a = lanes[pair_lane];
      const float b = lanes[pair_lane + half];
      rotated = lane < half ? a * cosine - b * sine : b * cosine + a * sine;
    }
    if (is_query) {
      query_out[head * head_dim + lane] = rotated;
    } else {
      key_cache[static_cast<size_t>(position) * kv_width + head * head_dim + lane] = rotated;
    }
  }
}

// GQA attention for one decode token over positions [0, ctrl[0]].
//
// One block per query head. Warps split the context for the q.k scores (each
// lane covers head_dim / 32 lanes of the dot product), the softmax runs over the
// scores in shared memory, and then each thread owns one output lane and sums
// the value rows, which keeps those reads coalesced. `max_context` bounds the
// shared score buffer; the host sizes the launch from it.
extern "C" __global__ void q35_attention(const float* query,
                                         const float* key_cache,
                                         const float* value_cache,
                                         float* out,
                                         const int* ctrl,
                                         int n_head,
                                         int n_kv,
                                         int head_dim,
                                         float scale) {
  extern __shared__ float scores[];
  __shared__ float scratch[32];
  const int head = blockIdx.x;
  const int kv_head = head / (n_head / n_kv);
  const int kv_width = n_kv * head_dim;
  const int context = ctrl[0] + 1;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const float* q = query + head * head_dim;

  for (int position = warp; position < context; position += warps) {
    const float* k = key_cache + static_cast<size_t>(position) * kv_width + kv_head * head_dim;
    float dot = 0.0f;
    for (int index = lane; index < head_dim; index += 32) dot += q[index] * k[index];
    dot = warp_sum(dot);
    if (lane == 0) scores[position] = dot * scale;
  }
  __syncthreads();

  float local_max = -INFINITY;
  for (int position = threadIdx.x; position < context; position += blockDim.x) {
    local_max = fmaxf(local_max, scores[position]);
  }
  const float peak = block_max(local_max, scratch);
  float local_sum = 0.0f;
  for (int position = threadIdx.x; position < context; position += blockDim.x) {
    const float weight = expf(scores[position] - peak);
    scores[position] = weight;
    local_sum += weight;
  }
  const float inverse = 1.0f / block_sum(local_sum, scratch);
  __syncthreads();

  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    float total = 0.0f;
    for (int position = 0; position < context; ++position) {
      total += scores[position] *
               value_cache[static_cast<size_t>(position) * kv_width + kv_head * head_dim + index];
    }
    out[head * head_dim + index] = total * inverse;
  }
}

// x *= sigmoid(gate): full attention's output gate.
extern "C" __global__ void q35_sigmoid_mul(float* x, const float* gate, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) x[index] *= sigmoidf(gate[index]);
}

// out = silu(gate) * up: the SwiGLU product.
extern "C" __global__ void q35_swiglu(const float* gate, const float* up, float* out, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) out[index] = siluf(gate[index]) * up[index];
}

// Depthwise causal conv for one DeltaNet decode token, one thread per channel.
//
// The per-channel state holds the last `kernel` raw inputs, oldest first. It is
// shifted, the new input appended, and the taps summed against it, then SiLU --
// the host's `depthwise_causal_conv` exactly, just without the host.
extern "C" __global__ void q35_deltanet_conv(const float* raw,
                                             float* conv_state,
                                             const float* conv_weight,
                                             float* out,
                                             int width,
                                             int kernel) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= width) return;
  float* state = conv_state + static_cast<size_t>(channel) * kernel;
  const float* weight = conv_weight + static_cast<size_t>(channel) * kernel;
  float sum = 0.0f;
  for (int tap = 0; tap < kernel - 1; ++tap) {
    const float shifted = state[tap + 1];
    state[tap] = shifted;
    sum += shifted * weight[tap];
  }
  const float incoming = raw[channel];
  state[kernel - 1] = incoming;
  sum += incoming * weight[kernel - 1];
  out[channel] = siluf(sum);
}

// DeltaNet gating scalars for one decode token.
//
// Blocks [0, key_heads) L2-normalize that head's query and key (the query also
// takes 1/sqrt(dk)) into `qq` and `kk`, one thread per lane. The last block
// computes every value head's beta = sigmoid(b) and decay =
// exp(-exp(a_log) * softplus(a + dt_bias)). These are the scalars the host used
// to compute before each device recurrence step.
extern "C" __global__ void q35_deltanet_prep(const float* convolved,
                                             const float* b_logits,
                                             const float* a_logits,
                                             const float* a_log,
                                             const float* dt_bias,
                                             float* kk,
                                             float* qq,
                                             float* beta,
                                             float* decay,
                                             int key_heads,
                                             int value_heads,
                                             int key_head_dim,
                                             float query_scale,
                                             float l2_epsilon) {
  __shared__ float scratch[32];
  const int head = blockIdx.x;
  const int lane = threadIdx.x;
  if (head == key_heads) {
    for (int value_head = lane; value_head < value_heads; value_head += blockDim.x) {
      beta[value_head] = sigmoidf(b_logits[value_head]);
      const float x = a_logits[value_head] + dt_bias[value_head];
      const float softplus = x > 20.0f ? x : log1pf(expf(x));
      decay[value_head] = expf(-expf(a_log[value_head]) * softplus);
    }
    return;
  }
  const int key_width = key_heads * key_head_dim;
  const float q = lane < key_head_dim ? convolved[head * key_head_dim + lane] : 0.0f;
  const float k = lane < key_head_dim ? convolved[key_width + head * key_head_dim + lane] : 0.0f;
  const float q_inverse = rsqrtf(block_sum(q * q, scratch) + l2_epsilon) * query_scale;
  const float k_inverse = rsqrtf(block_sum(k * k, scratch) + l2_epsilon);
  if (lane < key_head_dim) {
    qq[head * key_head_dim + lane] = q * q_inverse;
    kk[head * key_head_dim + lane] = k * k_inverse;
  }
}

// Per value head: out = core * rsqrt(mean(core^2) + eps) * w * silu(z).
// One block per value head, one thread per lane.
extern "C" __global__ void q35_gated_rmsnorm(const float* core,
                                             const float* z,
                                             const float* weight,
                                             float* out,
                                             int value_head_dim,
                                             float eps) {
  __shared__ float scratch[32];
  const int head = blockIdx.x;
  const int lane = threadIdx.x;
  const int index = head * value_head_dim + lane;
  const float value = lane < value_head_dim ? core[index] : 0.0f;
  const float inverse =
      rsqrtf(block_sum(value * value, scratch) / static_cast<float>(value_head_dim) + eps);
  if (lane < value_head_dim) out[index] = value * inverse * weight[lane] * siluf(z[index]);
}

// Greedy argmax, stage one: each block reduces a strided slice of the logits to
// its best (value, index). Ties keep the highest index and NaN never wins, which
// is the host's `sample_greedy` rule.
extern "C" __global__ void q35_argmax_partial(const float* logits,
                                              int n,
                                              float* partial_value,
                                              int* partial_index) {
  __shared__ float values[1024];
  __shared__ int indices[1024];
  float best = -INFINITY;
  int best_index = -1;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n;
       index += gridDim.x * blockDim.x) {
    const float value = logits[index];
    if (!isnan(value) && (best_index < 0 || value >= best)) {
      best = value;
      best_index = index;
    }
  }
  values[threadIdx.x] = best;
  indices[threadIdx.x] = best_index;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const int other = threadIdx.x + stride;
      const bool take = indices[other] >= 0 &&
                        (indices[threadIdx.x] < 0 || values[other] > values[threadIdx.x] ||
                         (values[other] == values[threadIdx.x] &&
                          indices[other] > indices[threadIdx.x]));
      if (take) {
        values[threadIdx.x] = values[other];
        indices[threadIdx.x] = indices[other];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_value[blockIdx.x] = values[0];
    partial_index[blockIdx.x] = indices[0];
  }
}

// Greedy argmax, stage two: one thread folds the block partials and writes the
// chosen token id, which is also the next step's embedding row.
extern "C" __global__ void q35_argmax_final(const float* partial_value,
                                            const int* partial_index,
                                            int partials,
                                            uint32_t* token) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  float best = -INFINITY;
  int best_index = -1;
  for (int index = 0; index < partials; ++index) {
    const int candidate = partial_index[index];
    if (candidate < 0) continue;
    const float value = partial_value[index];
    if (best_index < 0 || value > best || (value == best && candidate > best_index)) {
      best = value;
      best_index = candidate;
    }
  }
  token[0] = best_index < 0 ? 0U : static_cast<uint32_t>(best_index);
}

// Qwen3.5/Qwen3.6 Gated DeltaNet recurrent step.
//
// 48 of Qwen3.6's 64 layers are `linear_attention`, and its per-token state
// update ran on the host: an nsys profile of decode put
// `Qwen35DeltaNet::recurrent_forward` at 31.8% of all CPU samples and ~145 ms of
// every 270 ms token, against 107 ms of GPU kernel. It is also why activations
// cannot stay device-resident -- a host layer sits between every projection, so
// each one pays a round trip.
//
// The recurrence is sequential across tokens but wide within one. For a head,
// `memory`, the state update and `mixed` all touch only column `value_lane` of
// the `[key_head_dim, value_head_dim]` state, so the `value_head_dim` lanes are
// independent. `value_lane` is also the fastest-varying state index, so mapping
// threads to it makes every state access coalesced.
//
// Compiled with `--fmad=false`, and every operation is an explicit round-to-
// nearest intrinsic, so results match the host reduction bit for bit. Each
// scalar that needs a transcendental -- `beta`, `decay`, and the L2 inverses
// folded into `kk`/`qq` -- is computed host-side and passed in, because `expf`
// and `f32::exp` are not required to agree and the state must.

#include <stdint.h>

extern "C" __global__ void qwen35_deltanet_recurrent_step(
    float* state,           // [value_heads, key_head_dim, value_head_dim]
    const float* kk,        // [key_heads, key_head_dim], already k * l2_inverse(k)
    const float* qq,        // [key_heads, key_head_dim], already q * l2_inverse(q) * query_scale
    const float* value,     // [value_heads, value_head_dim]
    const float* beta,      // [value_heads]
    const float* decay,     // [value_heads]
    float* core,            // [value_heads, value_head_dim]
    uint32_t key_head_dim,
    uint32_t value_head_dim,
    uint32_t group_size) {
  extern __shared__ float projected[];  // kk then qq, `key_head_dim` each

  const uint32_t head = blockIdx.x;
  // Lanes may be split across `gridDim.y` blocks per head: they are independent,
  // and one block per head is only 48 blocks on a 128-SM part. The host path
  // launches with gridDim.y == 1, which is the original single-block form.
  const uint32_t value_lane = blockIdx.y * blockDim.x + threadIdx.x;
  const uint32_t key_head = head / group_size;
  const float* head_kk = kk + static_cast<size_t>(key_head) * key_head_dim;
  const float* head_qq = qq + static_cast<size_t>(key_head) * key_head_dim;

  for (uint32_t index = threadIdx.x; index < key_head_dim; index += blockDim.x) {
    projected[index] = head_kk[index];
    projected[key_head_dim + index] = head_qq[index];
  }
  __syncthreads();
  if (value_lane >= value_head_dim) return;

  float* head_state = state + static_cast<size_t>(head) * key_head_dim * value_head_dim;
  const float head_decay = decay[head];

  // Pass 1: decay this lane's column and reduce it against the key. The host
  // decays the whole head first and then reduces; decaying each element as it is
  // read is the same value, because a lane only ever reads its own column.
  //
  // The decayed column is not written back here. Pass 2 recomputes the identical
  // `__fmul_rn(state, decay)` and writes only the final value, so the state
  // takes one write per step instead of two, and the column pass 2 rereads
  // (a 32-lane block's is 16 KiB) is still in L1.
  float memory = 0.0f;
  for (uint32_t key_lane = 0; key_lane < key_head_dim; ++key_lane) {
    const size_t index = static_cast<size_t>(key_lane) * value_head_dim + value_lane;
    const float decayed = __fmul_rn(head_state[index], head_decay);
    memory = __fadd_rn(memory, __fmul_rn(projected[key_lane], decayed));
  }

  const float delta = __fmul_rn(
      beta[head],
      __fsub_rn(value[static_cast<size_t>(head) * value_head_dim + value_lane], memory));

  // Pass 2: apply the delta rule and reduce the updated column against the query.
  float mixed = 0.0f;
  for (uint32_t key_lane = 0; key_lane < key_head_dim; ++key_lane) {
    const size_t index = static_cast<size_t>(key_lane) * value_head_dim + value_lane;
    const float decayed = __fmul_rn(head_state[index], head_decay);
    const float updated = __fadd_rn(decayed, __fmul_rn(projected[key_lane], delta));
    head_state[index] = updated;
    mixed = __fadd_rn(mixed, __fmul_rn(projected[key_head_dim + key_lane], updated));
  }

  core[static_cast<size_t>(head) * value_head_dim + value_lane] = mixed;
}

// `qwen35_deltanet_recurrent_step` over `tokens` consecutive tokens, the state
// held in registers between them.
//
// Each thread owns one `value_lane` column of its head's state, exactly as in the
// single-step kernel, and for every token runs the same two passes with the same
// round-to-nearest intrinsics in the same order. The column is read from global
// memory once before the first token and written once after the last, instead of
// once per token; the values in between are the ones the single-step kernel
// would have stored and re-read, so the result is bit-identical to `tokens`
// launches of it. `KD` is the key head width (the column's length).
//
// Token t reads `kk`/`qq` row t (`key_width` floats), `value` row t (of
// `value_stride` floats), `beta`/`decay` row t (`value_heads` floats) and writes
// `core` row t (`value_heads * value_head_dim` floats). Launched with
// `blockDim.x == 32` and `gridDim.y * 32 == value_head_dim`.
template <uint32_t KD>
__device__ __forceinline__ void recurrent_rows_body(float* state,
                                                    const float* kk,
                                                    const float* qq,
                                                    const float* value,
                                                    const float* beta,
                                                    const float* decay,
                                                    float* core,
                                                    uint32_t value_head_dim,
                                                    uint32_t group_size,
                                                    uint32_t tokens,
                                                    uint32_t key_width,
                                                    uint32_t value_stride,
                                                    uint32_t value_heads,
                                                    float* projected) {
  const uint32_t head = blockIdx.x;
  const uint32_t value_lane = blockIdx.y * blockDim.x + threadIdx.x;
  const uint32_t key_head = head / group_size;
  const bool live = value_lane < value_head_dim;
  float* head_state = state + static_cast<size_t>(head) * KD * value_head_dim;

  float column[KD];
  if (live) {
#pragma unroll
    for (uint32_t key_lane = 0; key_lane < KD; ++key_lane) {
      column[key_lane] = head_state[static_cast<size_t>(key_lane) * value_head_dim + value_lane];
    }
  }

  for (uint32_t t = 0; t < tokens; ++t) {
    const float* head_kk = kk + static_cast<size_t>(t) * key_width + key_head * KD;
    const float* head_qq = qq + static_cast<size_t>(t) * key_width + key_head * KD;
    __syncthreads();  // the previous token's reads of `projected` are done
    for (uint32_t index = threadIdx.x; index < KD; index += blockDim.x) {
      projected[index] = head_kk[index];
      projected[KD + index] = head_qq[index];
    }
    __syncthreads();
    if (!live) continue;

    const float head_decay = decay[static_cast<size_t>(t) * value_heads + head];
    // The single-step kernel computes `decayed` in both passes; it is the same
    // value, so here pass 1 keeps it in the column for pass 2.
    float memory = 0.0f;
#pragma unroll
    for (uint32_t key_lane = 0; key_lane < KD; ++key_lane) {
      column[key_lane] = __fmul_rn(column[key_lane], head_decay);
      memory = __fadd_rn(memory, __fmul_rn(projected[key_lane], column[key_lane]));
    }
    // Re-read `projected` in pass 2 instead of holding 128 of its values in
    // registers next to the column.
    asm volatile("" ::: "memory");
    const float delta = __fmul_rn(
        beta[static_cast<size_t>(t) * value_heads + head],
        __fsub_rn(value[static_cast<size_t>(t) * value_stride + head * value_head_dim + value_lane],
                  memory));
    float mixed = 0.0f;
#pragma unroll
    for (uint32_t key_lane = 0; key_lane < KD; ++key_lane) {
      const float updated = __fadd_rn(column[key_lane], __fmul_rn(projected[key_lane], delta));
      column[key_lane] = updated;
      mixed = __fadd_rn(mixed, __fmul_rn(projected[KD + key_lane], updated));
    }
    core[static_cast<size_t>(t) * value_heads * value_head_dim + head * value_head_dim +
         value_lane] = mixed;
  }

  if (live) {
#pragma unroll
    for (uint32_t key_lane = 0; key_lane < KD; ++key_lane) {
      head_state[static_cast<size_t>(key_lane) * value_head_dim + value_lane] = column[key_lane];
    }
  }
}

extern "C" __global__ void __launch_bounds__(32) qwen35_deltanet_recurrent_rows128(float* state,
                                                             const float* kk,
                                                             const float* qq,
                                                             const float* value,
                                                             const float* beta,
                                                             const float* decay,
                                                             float* core,
                                                             uint32_t value_head_dim,
                                                             uint32_t group_size,
                                                             uint32_t tokens,
                                                             uint32_t key_width,
                                                             uint32_t value_stride,
                                                             uint32_t value_heads) {
  __shared__ float projected[2 * 128];
  recurrent_rows_body<128>(state, kk, qq, value, beta, decay, core, value_head_dim, group_size,
                           tokens, key_width, value_stride, value_heads, projected);
}

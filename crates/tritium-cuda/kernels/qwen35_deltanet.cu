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
  float memory = 0.0f;
  for (uint32_t key_lane = 0; key_lane < key_head_dim; ++key_lane) {
    const size_t index = static_cast<size_t>(key_lane) * value_head_dim + value_lane;
    const float decayed = __fmul_rn(head_state[index], head_decay);
    head_state[index] = decayed;
    memory = __fadd_rn(memory, __fmul_rn(projected[key_lane], decayed));
  }

  const float delta = __fmul_rn(
      beta[head],
      __fsub_rn(value[static_cast<size_t>(head) * value_head_dim + value_lane], memory));

  // Pass 2: apply the delta rule and reduce the updated column against the query.
  float mixed = 0.0f;
  for (uint32_t key_lane = 0; key_lane < key_head_dim; ++key_lane) {
    const size_t index = static_cast<size_t>(key_lane) * value_head_dim + value_lane;
    const float updated =
        __fadd_rn(head_state[index], __fmul_rn(projected[key_lane], delta));
    head_state[index] = updated;
    mixed = __fadd_rn(mixed, __fmul_rn(projected[key_head_dim + key_lane], updated));
  }

  core[static_cast<size_t>(head) * value_head_dim + value_lane] = mixed;
}

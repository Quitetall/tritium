// attention.metal — v3 Q-blocked prefill GQA attention: the MSL twin of
// tritium-cuda's `gqa_attention_batch_v3_f32` (kernels/decode.cu, the
// `gqa_attention_batch_v3_body` family). f32 KV only for now (the CUDA `_h`
// f16-KV twin is follow-up work; the Metal backend has no KV-dtype machinery
// yet).
//
// STATUS: compile-verified only — authored on a Linux box with no Metal
// device; first executed (and gated against the pinned-order host reference)
// on the self-hosted Apple-Silicon CI lane. See `attn.rs` module docs.
//
// ## Thread mapping (CUDA -> Metal)
//
//   CUDA block, 256 threads = 8 warps x 32 lanes
//     -> Metal threadgroup, ATTN_V3_THREADS = 256 threads
//        = ATTN_V3_SIMDS = 8 simdgroups x 32 lanes.
//   blockIdx.x = head        -> threadgroup_position_in_grid.x
//   blockIdx.y = row block   -> threadgroup_position_in_grid.y
//   threadIdx.x (tid)        -> thread_index_in_threadgroup
//   tid >> 5 (warp id)       -> simdgroup_index_in_threadgroup
//   tid & 31 (lane id)       -> thread_index_in_simdgroup
//   __shfl_xor_sync          -> simd_shuffle_xor  (butterfly max only —
//                               exact under any order; the pinned sums below
//                               are NOT replaced by simd_sum)
//   __shfl_sync(.., 0)       -> simd_shuffle(value, 0)
//   __syncthreads()          -> threadgroup_barrier(mem_device|mem_threadgroup)
//                               (CUDA __syncthreads orders shared AND global
//                               within the block; scores live in device memory)
//   __syncwarp()             -> simdgroup_barrier(mem_device|mem_threadgroup)
//
// The warp<->simdgroup identification (simdgroup = 32 consecutive linear
// thread indices; RPW = BQ / SIMDS = 1 -> one query row per simdgroup in
// phases 1/2, exactly the CUDA shape) requires the pipeline's
// thread_execution_width to be 32; the HOST checks that at dispatch and
// falls back to the pinned-order CPU reference otherwise.
//
// ## Pinned orders (the ADR 0022 twin contract, per (row, head))
//
//   * phase 1 dot: one lane owns the whole d-ascending add chain per key;
//   * phase 2: butterfly fmax (order-exact), elementwise exp, softmax SUM
//     sequential j-ascending on lane 0 only, one 1/sum divide, elementwise
//     scale;
//   * phase 3: per output dim, sequential j-ascending V fold across
//     ascending chunks with the rev-1 `w == 0.0f` zero-skip; keys past a
//     row's causal limit stage a 0 weight and take the same skip.
//
// Arithmetic discipline: this source is compiled with FAST MATH DISABLED
// (the host passes setFastMathEnabled:NO — unlike mpgemm.metal, which keeps
// the default), so +,*,/ are IEEE round-to-nearest, mirroring CUDA's
// __fadd_rn/__fmul_rn/__fdiv_rn. Every mul-add in a pinned chain is split
// into two statements (named temporary) so no conforming compiler may
// contract it into an fma. The ONE deliberate deviation from the CUDA twin:
// CUDA's exp_f32 computes exp in f64 and rounds once (matching glibc expf
// bit-for-bit); Metal has no double, so exp_f32 here is precise::exp — a few
// ULP from the host exp. The Mac gate therefore asserts a tight tolerance
// vs the host reference, not to_bits equality (see attn.rs module docs).

#include <metal_stdlib>
using namespace metal;

// MUST mirror decode.cu's ATTN_V3_* defines and attn.rs's constants (the
// attn.rs consts-pin test parses these lines; keep them bare numbers).
// ATTN_V3_HDMAX is decode.cu's ATTN_V2_HDMAX (v3 reuses the v2 bound there;
// Metal has no v2 kernel to borrow a name from).
#define ATTN_V3_THREADS 256
#define ATTN_V3_BQ 8
#define ATTN_V3_KCH 32
#define ATTN_V3_HDMAX 128
#define ATTN_V3_SIMDS (ATTN_V3_THREADS / 32)
// Query rows owned per simdgroup in phases 1 and 2 (simdgroup s owns rows
// s, s+SIMDS, ...).
#define ATTN_V3_RPW (ATTN_V3_BQ / ATTN_V3_SIMDS)
// RPW = 0 (more simdgroups than BQ rows) compiles phases 1/2 to nothing —
// every output garbage; catch it in every build (decode.cu's static_assert).
static_assert(ATTN_V3_BQ % ATTN_V3_SIMDS == 0 && ATTN_V3_RPW >= 1,
              "ATTN_V3_BQ must be a positive multiple of ATTN_V3_SIMDS");
// Threadgroup staging must fit Apple Silicon's 32 KB threadgroup memory.
static_assert((ATTN_V3_KCH * (ATTN_V3_HDMAX + 1) + ATTN_V3_BQ * ATTN_V3_HDMAX +
               ATTN_V3_BQ * (ATTN_V3_KCH + 1)) * 4 <= 32768,
              "v3 threadgroup staging exceeds the 32 KB threadgroup budget");

// exp_f32 — the softmax exponential. CUDA: __double2float_rn(exp((double)x)),
// bit-identical to glibc expf. Metal HAS NO double, so this is the port's one
// pinned-value deviation: precise::exp (fast-math is disabled for this
// library, but spell precise:: anyway so a compile-option drift cannot
// silently swap in the fast variant).
static inline float exp_f32(float x) { return precise::exp(x); }

// Scalar launch params, set_bytes at buffer(5). Field order/offsets match the
// repr(C) `AttnV3Params` on the Rust side (pinned there by const asserts).
struct AttnV3Params {
    uint ctx_max;        // scores stride per (row, head): [m, n_head, ctx_max]
    uint n_head;
    uint n_head_kv;
    uint head_dim;       // host guarantees 1 <= head_dim <= ATTN_V3_HDMAX
    float scale;
    uint causal_offset;  // keys already in the KV arena before this prefill
    uint m;              // query rows in this prefill
};

// Grid: (n_head, ceil(m / ATTN_V3_BQ)) threadgroups of ATTN_V3_THREADS —
// exactly the CUDA launch. q/out are [m, n_head, head_dim]; k/v are KV arenas
// [>= causal_offset + m, n_head_kv, head_dim]; scores is the global
// [m, n_head, ctx_max] scratch (the lever that removes v2's ctx cap). All
// per-row bases widen to 64-bit ulong (the v0.6.9 indexing discipline,
// mirroring the CUDA (long long) casts).
kernel void gqa_attention_batch_v3_f32(
    device const float* q        [[buffer(0)]],
    device const float* k        [[buffer(1)]],
    device const float* v        [[buffer(2)]],
    device       float* out_buf  [[buffer(3)]],
    device       float* scores   [[buffer(4)]],
    constant AttnV3Params& p     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tidx [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lidx [[thread_index_in_simdgroup]])
{
    const int h = int(tgid.x);
    const int row0 = int(tgid.y) * ATTN_V3_BQ;
    const int tid = int(tidx);
    const int m = int(p.m);
    const int n_head = int(p.n_head);
    const int n_head_kv = int(p.n_head_kv);
    const int head_dim = int(p.head_dim);
    const int ctx_max = int(p.ctx_max);
    const int causal_offset = int(p.causal_offset);
    const float scale = p.scale;

    const int nrows = min(ATTN_V3_BQ, m - row0);
    if (nrows <= 0 || h >= n_head) return;  // threadgroup-uniform: no barrier hazard
    const int n_rep = n_head / n_head_kv;
    const int kv = h / n_rep;
    // One past the highest key any row of this block attends.
    const int ctx_top = causal_offset + row0 + nrows;

    // K/V chunk stage (reused between phases 1 and 3), the BQ query rows, and
    // the per-chunk score strip phase 3 reads broadcast from threadgroup mem.
    threadgroup float s_kv[ATTN_V3_KCH * (ATTN_V3_HDMAX + 1)];
    threadgroup float s_q[ATTN_V3_BQ * ATTN_V3_HDMAX];
    threadgroup float s_w[ATTN_V3_BQ][ATTN_V3_KCH + 1];

    for (int idx = tid; idx < nrows * head_dim; idx += ATTN_V3_THREADS) {
        const int r = idx / head_dim;
        const int d = idx - r * head_dim;
        s_q[r * ATTN_V3_HDMAX + d] =
            q[(ulong(row0 + r) * ulong(n_head) + ulong(h)) * ulong(head_dim) + ulong(d)];
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    const int kstride = head_dim + 1;

    // Phase 1: scaled dots. Each staged KCH-key chunk feeds all BQ rows:
    // lane l owns key l for rows sgid + rr*SIMDS (lanes sharing a key
    // broadcast the staged K row; the 32 lanes of a simdgroup share a query
    // row -> broadcast s_q reads). Keys past a row's causal limit write
    // nothing.
    for (int c0 = 0; c0 < ctx_top; c0 += ATTN_V3_KCH) {
        const int nk = min(ATTN_V3_KCH, ctx_top - c0);
        for (int idx = tid; idx < nk * head_dim; idx += ATTN_V3_THREADS) {
            const int j = idx / head_dim;
            const int d = idx - j * head_dim;
            s_kv[j * kstride + d] =
                k[(ulong(c0 + j) * ulong(n_head_kv) + ulong(kv)) * ulong(head_dim) + ulong(d)];
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        const int key = int(lidx);
        if (key < nk) {
            const threadgroup float* kr = &s_kv[key * kstride];
            const int jglob = c0 + key;
            for (int rr = 0; rr < ATTN_V3_RPW; ++rr) {
                const int r = int(sgid) + rr * ATTN_V3_SIMDS;
                if (r < nrows && jglob <= causal_offset + row0 + r) {
                    const threadgroup float* qr = &s_q[r * ATTN_V3_HDMAX];
                    // PINNED: the d-ascending dot chain (rev-1's order). The
                    // named temporary keeps mul and add in separate
                    // statements — no fma contraction.
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        const float prod = qr[d] * kr[d];
                        dot = dot + prod;
                    }
                    scores[(ulong(row0 + r) * ulong(n_head) + ulong(h)) * ulong(ctx_max) +
                           ulong(jglob)] = dot * scale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }

    // Phase 2: per-row softmax; simdgroup s owns rows s, s+SIMDS, ...
    // Butterfly fmax (exact under any order), elementwise exp, ORDER-PINNED
    // sequential sum on lane 0, elementwise inv scale. Phase-1 scores for a
    // row's stripe j were written by THIS lane (jglob % 32 == lane, KCH == 32),
    // so the strided reads below need no extra cross-thread ordering.
    const int lane = int(lidx);
    for (int rr = 0; rr < ATTN_V3_RPW; ++rr) {
        const int r = int(sgid) + rr * ATTN_V3_SIMDS;
        if (r < nrows) {
            device float* sc =
                scores + (ulong(row0 + r) * ulong(n_head) + ulong(h)) * ulong(ctx_max);
            const int ctx = causal_offset + row0 + r + 1;
            float local_max = -INFINITY;
            for (int j = lane; j < ctx; j += 32) local_max = fmax(local_max, sc[j]);
            for (int off = 16; off > 0; off >>= 1) {
                local_max = fmax(local_max, simd_shuffle_xor(local_max, ushort(off)));
            }
            const float mx = local_max;
            for (int j = lane; j < ctx; j += 32) {
                sc[j] = exp_f32(sc[j] - mx);
            }
            // Lane 0 is about to read every lane's exp writes (device mem).
            simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            float inv = 0.0f;
            if (lane == 0) {
                // PINNED: THE softmax sum — sequential j-ascending on one lane.
                float sum = 0.0f;
                for (int j = 0; j < ctx; ++j) {
                    sum = sum + sc[j];
                }
                inv = 1.0f / sum;
            }
            inv = simd_shuffle(inv, ushort(0));
            for (int j = lane; j < ctx; j += 32) {
                sc[j] = sc[j] * inv;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // Phase 3: V fold. Per chunk: stage V into the reused s_kv, stage the
    // chunk's normalized weights for all rows into s_w (0 past a row's causal
    // limit), then thread d folds every row — one V load feeds BQ chains,
    // each chain j-ascending with the rev-1 zero-skip.
    float acc[ATTN_V3_BQ];
    for (int r = 0; r < ATTN_V3_BQ; ++r) acc[r] = 0.0f;
    for (int c0 = 0; c0 < ctx_top; c0 += ATTN_V3_KCH) {
        const int nk = min(ATTN_V3_KCH, ctx_top - c0);
        for (int idx = tid; idx < nk * head_dim; idx += ATTN_V3_THREADS) {
            const int j = idx / head_dim;
            const int d = idx - j * head_dim;
            s_kv[j * kstride + d] =
                v[(ulong(c0 + j) * ulong(n_head_kv) + ulong(kv)) * ulong(head_dim) + ulong(d)];
        }
        for (int idx = tid; idx < nrows * nk; idx += ATTN_V3_THREADS) {
            const int r = idx / nk;
            const int jj = idx - r * nk;
            const int jglob = c0 + jj;
            s_w[r][jj] = (jglob <= causal_offset + row0 + r)
                ? scores[(ulong(row0 + r) * ulong(n_head) + ulong(h)) * ulong(ctx_max) +
                         ulong(jglob)]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (tid < head_dim) {
            const int d = tid;
            for (int jj = 0; jj < nk; ++jj) {
                const float vv = s_kv[jj * kstride + d];
                for (int r = 0; r < ATTN_V3_BQ; ++r) {
                    // Rows >= nrows read UNSTAGED s_w garbage here — safe by
                    // design (as in the CUDA twin): their acc[r] can only
                    // poison a register the nrows-guarded store below never
                    // emits (no FP traps on this path).
                    const float w = s_w[r][jj];
                    if (w != 0.0f) {
                        // PINNED: the per-(row, d) j-ascending fold chain.
                        const float prod = w * vv;
                        acc[r] = acc[r] + prod;
                    }
                }
            }
        }
        // Fold reads done before the next chunk restages s_kv/s_w.
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    if (tid < head_dim) {
        for (int r = 0; r < ATTN_V3_BQ; ++r) {
            if (r < nrows) {
                out_buf[(ulong(row0 + r) * ulong(n_head) + ulong(h)) * ulong(head_dim) +
                        ulong(tid)] = acc[r];
            }
        }
    }
}

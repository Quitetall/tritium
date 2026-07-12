// IMMA int8 ternary mpGEMM — compute-bound prefill kernel (v0.30, ADR 0005 / WF-A).
//
// The tiled add-only kernel (`tq2_0_add.cu`) wins memory-bound decode (batch=1);
// this kernel targets the compute-bound prefill (large M) with the int8 tensor
// cores: int8 activations × ternary weights via the `mma.m16n8k32` shape
// (`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`), 16×8×32 tiles, with the
// 2-bit ternary weights unpacked to int8 in double-buffered shared memory. It
// mirrors bitnet.cpp's W1.58A8 GPU kernel / BitBLAS / GPTQ-Marlin.
//
// REQUIRES sm_80+ (Ampere): the `m16n8k32` int8 `mma` shape is not available on
// sm_75 (Turing) — so `build.rs` compiles THIS kernel for compute_80 (not the
// compute_75 floor `tq2_0_add.cu` uses) and emits a second PTX target.
//
// ## Contract (fused per ADR 0005)
//
//   out[m,n] = act_scale[m] * weight_scale[n] * Σ_k qact[m,k] * trit[n,k]
//
// `qact` is the per-token int8 absmax quant of the activations (W1.58A8, Qp=127),
// supplied here as int8 `[M, K]` row-major. `trit[n,k] ∈ {-1,0,+1}`. Correctness
// is held to the vs-reference + cross-kernel parity gate (IMMA == add-only == ref).
//
// ## Accumulation precision (LESSON from the tiled add-only kernel)
//
// A parallel reduction over the ternary sum drifts past the 1e-4 float tolerance
// vs the sequential f32 reference unless the partial sums are kept exact. Here the
// contraction is done in **int32** by the `mma` instruction — `int8 × int8`
// products in `[-127*1, 127*1]` summed over K never overflow int32 for any BitNet
// shape (|acc| ≤ 127·K ≤ 127·6912 ≈ 8.8e5, far under 2^31), so the integer
// accumulate is *exact*. The only floating-point step is the final
// `act_scale[m] · weight_scale[n] · acc`, a single f32 multiply per output. With
// an exact integer contraction the IMMA result matches the f32 reference to ~1e-4
// (the reference's own f32-accumulate rounding is the only divergence), so this
// kernel needs no widened reduction bar (cf. ADR 0002/0004 — the float path's
// 1e-4 is the reference's rounding, not a defect of this kernel).
//
// ## I2sInt8 weight layout (the contract with `tritium_format::convert_i2s_to_int8`)
//
// Weights stay 2-bit packed in VRAM and are laid out to match exactly what the
// `mma` B operand wants — an `8×32` int8 tile read row-major (`B[n_in_tile,
// k_in_tile]`, the PTX "col" operand of an `N×K` matrix). The packing is, in
// order, the most significant first:
//
//   * N is padded up to a multiple of 8 (`N_TILE`), K up to a multiple of 32
//     (`K_TILE`). Padding rows/cols carry trit 0 (code 1) so they contribute 0.
//   * The grid is `ceil(N/8)` n-tiles × `ceil(K/32)` k-tiles. Tiles are stored
//     n-tile-major then k-tile-major: tile `(nt, kt)` is at flat tile index
//     `nt * num_ktiles + kt`.
//   * Each tile is 8·32 = 256 ternary codes in `(n_in_tile, k_in_tile)` row-major
//     order, 4 codes per byte (2 bits each, low pair = first element), so 64 bytes
//     per tile. `code = trit + 1 ∈ {0,1,2}` — the same `+1` offset as I2_S/TQ2_0.
//
// So `bytes.len() == num_ntiles · num_ktiles · 64`. This is a SIMPLE, documented
// interleave (correctness-first; perf tuning is WF-B's job): the kernel reads one
// byte, extracts the four 2-bit codes for four consecutive k, and writes int8
// `{-1,0,+1}` into the shared B tile at the position the fragment load expects.

#include <cuda_runtime.h>

// mma.m16n8k32 tile dims.
#define IMMA_M 16
#define IMMA_N 8
#define IMMA_K 32
// 8×32 ternary codes per weight tile, 4 codes/byte → 64 bytes.
#define IMMA_WTILE_BYTES (IMMA_N * IMMA_K / 4)

extern "C" {

// Per-token int8 absmax activation quant (W1.58A8), on device — the fused
// `mpgemm_with_act_quant` override's first step. One block per activation row `m`;
// the block's threads find the row absmax `γ = max_k |act|` via a shared-memory
// reduction, then quantize `q = clamp(round_ties_even(act · 127/γ), -128, 127)`
// and emit the per-token dequant multiplier `scale = γ / 127`. A fully-zero row
// yields zero quants and a zero scale (its dequantized contribution is 0 either
// way). This must match `tritium_spec::quantize_act_int8` / `tritium-nn`'s
// `ops::act_quant` exactly — round-half-to-even is load-bearing for greedy token
// parity, so `rintf` (IEEE round-to-nearest-even under the default rounding mode)
// is used, not `roundf` (which rounds halves away from zero).
//
// Qp = 127 (the positive int8 cap); -128 is reachable. `qout` is int8 [M, K],
// `scale_out` is f32 [M].
#define A8_QB 127.0f
#define ACT_QUANT_THREADS 256

__global__ void act_quant_int8_per_token(
    const float* __restrict__ act,    // [M, K] row-major
    signed char* __restrict__ qout,   // [M, K] int8
    float* __restrict__ scale_out,    // [M]
    const int m,
    const int k) {
    const int row = blockIdx.x;
    if (row >= m) {
        return;
    }
    const float* arow = act + (long long)row * k;
    signed char* qrow = qout + (long long)row * k;

    // Block-wide absmax reduction over this row's K elements.
    __shared__ float s_max[ACT_QUANT_THREADS];
    float local = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const float a = fabsf(arow[i]);
        if (a > local) {
            local = a;
        }
    }
    s_max[threadIdx.x] = local;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            const float other = s_max[threadIdx.x + off];
            if (other > s_max[threadIdx.x]) {
                s_max[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }
    const float gamma = s_max[0];

    if (gamma == 0.0f) {
        // Zero row: all-zero quants, zero scale. (Matches the host default.)
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            qrow[i] = 0;
        }
        if (threadIdx.x == 0) {
            scale_out[row] = 0.0f;
        }
        return;
    }

    const float s = A8_QB / gamma;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        // rintf = round-half-to-even under the default rounding mode; clamp to the
        // asymmetric int8 range [-128, +127] (a value rounding to +128 saturates).
        float q = rintf(arow[i] * s);
        q = fminf(fmaxf(q, -128.0f), A8_QB);
        qrow[i] = (signed char)q;
    }
    if (threadIdx.x == 0) {
        scale_out[row] = gamma / A8_QB;
    }
}

// One warp computes one 16×8 output tile, looping over the K dimension in 32-wide
// steps. Activations (int8 `[M,K]` row-major) and the unpacked ternary weights are
// staged into double-buffered shared memory; the `mma.m16n8k32` instruction does
// the int32 contraction; lane stores fold the per-token and per-channel scales.
//
// Launch geometry (host side, `src/cuda.rs`):
//   * block = 32 threads (one warp) — correctness-first, one tile per block.
//   * grid  = (ceil(N/8), ceil(M/16)) — one block per (m-tile, n-tile).
//   * shared = 2 · (16·32 int8 A + 8·32 int8 B) bytes (double-buffered).
//
// `weights` is the I2sInt8 packing documented at the top of this file; `m,n,k` are
// the *logical* (unpadded) dimensions, and `num_ktiles = ceil(k/32)` is the packed
// k-tile stride the host passes so the kernel does not recompute it.
__global__ void tq2_0_imma_mpgemm(
    const signed char* __restrict__ qact,    // int8 [M, K] row-major
    const unsigned char* __restrict__ weights, // I2sInt8 packed (see header)
    const float* __restrict__ act_scale,     // [M]   per-token dequant multiplier
    const float* __restrict__ weight_scale,  // [N]   per-output-channel scale
    float* __restrict__ out,                 // [M, N] row-major, overwritten
    const int m,
    const int n,
    const int k,
    const int num_ktiles) {                  // ceil(k / 32), the packed k-tile count
    const int lane = threadIdx.x;            // 0..31, one warp per block
    const int group = lane >> 2;             // groupID = laneid / 4  (0..7)
    const int tig = lane & 3;                // threadID_in_group = laneid % 4 (0..3)

    const int m_tile = blockIdx.y * IMMA_M;  // first output row of this tile
    const int n_tile = blockIdx.x * IMMA_N;  // first output col of this tile

    // Double-buffered shared staging: int8 A (16×32) + int8 B (8×32) per buffer.
    // 16-byte aligned so the fragment registers below load as single u32s.
    __shared__ __align__(16) signed char s_a[2][IMMA_M * IMMA_K];
    __shared__ __align__(16) signed char s_b[2][IMMA_N * IMMA_K];

    // Per-thread int32 accumulators (the 4 C-fragment registers), exact over K.
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    // Stage one k-tile (`kt`) of A and B into shared buffer `buf`. All 32 lanes
    // cooperate. Out-of-range rows/cols (the padded tail past `m`/`n`/`k`) are
    // written as 0 so they contribute nothing to the int32 sum.
    auto stage = [&](int kt, int buf) {
        const int k0 = kt * IMMA_K;
        // A tile: 16×32 = 512 int8. Lane l writes elements l, l+32, … .
        for (int idx = lane; idx < IMMA_M * IMMA_K; idx += 32) {
            const int r = idx / IMMA_K;          // 0..15  (row within tile)
            const int c = idx % IMMA_K;          // 0..31  (k within tile)
            const int gm = m_tile + r;
            const int gk = k0 + c;
            s_a[buf][idx] =
                (gm < m && gk < k) ? qact[(long long)gm * k + gk] : (signed char)0;
        }
        // B tile: 8×32 = 256 ternary codes, packed 4/byte in (n_in_tile, k_in_tile)
        // row-major. The 64 packed bytes for tile (nt, kt) start at
        // `(nt*num_ktiles + kt) * 64`. Lane l unpacks bytes l, l+32 (2 bytes each).
        const int nt = blockIdx.x;
        const long long tile_byte0 =
            ((long long)nt * num_ktiles + kt) * IMMA_WTILE_BYTES;
        for (int byte_idx = lane; byte_idx < IMMA_WTILE_BYTES; byte_idx += 32) {
            const unsigned int packed = weights[tile_byte0 + byte_idx];
            // This byte holds 4 consecutive (n_in_tile, k_in_tile) codes: element
            // index e = byte_idx*4 + j, with n_in_tile = e/32, k_in_tile = e%32.
            for (int j = 0; j < 4; ++j) {
                const unsigned int code = (packed >> (2 * j)) & 3u;
                // trit = code - 1: 0→-1, 1→0, 2→+1.
                const signed char trit = (signed char)((int)code - 1);
                s_b[buf][byte_idx * 4 + j] = trit;
            }
        }
    };

    // Prologue: stage the first k-tile into buffer 0 (when K is non-empty).
    int cur = 0;
    if (num_ktiles > 0) {
        stage(0, 0);
    }

    for (int kt = 0; kt < num_ktiles; ++kt) {
        const int nxt = cur ^ 1;
        // Prefetch the next k-tile into the other buffer while this one computes.
        if (kt + 1 < num_ktiles) {
            stage(kt + 1, nxt);
        }
        __syncthreads();  // current buffer fully staged (and next, if any)

        // --- Load the A fragment (4 regs, 4 packed int8 each) from s_a[cur]. ---
        // a0: row=group,   k=4*tig + {0..3}      a1: row=group+8, same k
        // a2: row=group,   k=16 + 4*tig + {0..3} a3: row=group+8, same k
        // Each lane's quad is 4 CONSECUTIVE shared bytes, so one little-endian
        // u32 load per register — identical bytes/bits to the old per-byte pack
        // (offsets are 4-aligned: IMMA_K=32 stride, kbase 0/16, 4*tig).
        unsigned int a0, a1, a2, a3;
        {
            const signed char* A = s_a[cur];
            a0 = *(const unsigned int*)(A + group * IMMA_K + 4 * tig);
            a1 = *(const unsigned int*)(A + (group + 8) * IMMA_K + 4 * tig);
            a2 = *(const unsigned int*)(A + group * IMMA_K + 16 + 4 * tig);
            a3 = *(const unsigned int*)(A + (group + 8) * IMMA_K + 16 + 4 * tig);
        }

        // --- Load the B fragment (2 regs) from s_b[cur], same u32 trick. ---
        unsigned int b0, b1;
        {
            const signed char* B = s_b[cur];
            b0 = *(const unsigned int*)(B + group * IMMA_K + 4 * tig);
            b1 = *(const unsigned int*)(B + group * IMMA_K + 16 + 4 * tig);
        }

        // int32 accumulate: D = A · Bᵀ + C, exact for int8×ternary (no overflow).
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

        __syncthreads();  // done reading s_a[cur]/s_b[cur]; safe to reuse next iter
        cur = nxt;
    }

    // --- Store the 16×8 C fragment, folding scales. ---
    // c0: row=group,   col=2*tig
    // c1: row=group,   col=2*tig+1
    // c2: row=group+8, col=2*tig
    // c3: row=group+8, col=2*tig+1
    // out[m,n] = ((float)acc · weight_scale[n]) · act_scale[m] — the EXACT
    // association the dp4a family uses (tq2_0_add.cu tiled_i8_scaled:
    // `(float)acc * scales[ni] * act_scale[mi]`). The integer acc is
    // order-free, and this is a pure multiply chain (no FMA contraction
    // possible), so matching the association makes IMMA output BIT-IDENTICAL
    // to the dp4a path — the ADR 0026 Track P prefill dispatch relies on it.
    auto store = [&](int acc, int row_in_tile, int col_in_tile) {
        const int gm = m_tile + row_in_tile;
        const int gn = n_tile + col_in_tile;
        if (gm < m && gn < n) {
            out[(long long)gm * n + gn] =
                (float)acc * weight_scale[gn] * act_scale[gm];
        }
    };
    store(c0, group, 2 * tig);
    store(c1, group, 2 * tig + 1);
    store(c2, group + 8, 2 * tig);
    store(c3, group + 8, 2 * tig + 1);
}

}  // extern "C"

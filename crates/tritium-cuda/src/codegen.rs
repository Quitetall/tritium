//! nvrtc JIT codegen for the IMMA kernel (v0.30, ADR 0005 / WF-B).
//!
//! The IMMA (`mma.m16n8k32`) kernel is templated over the [`TileConfig`]
//! parameters from [`super::autotune`]; this module renders the CUDA source for a
//! chosen tile and compiles it to a loadable PTX image via cudarc's nvrtc binding
//! at runtime (the `nvrtc` cargo feature is enabled on `cudarc`, and `libnvrtc` is
//! dlopen'd on first use by the `fallback-dynamic-loading` path). AOT default PTX
//! (built by `build.rs` for the common BitNet shapes) covers the hot path; JIT
//! covers the long tail and the autotune search.
//!
//! ## Determinism is load-bearing
//!
//! A JIT-compiled tile must produce **bit-identical** output to the AOT cubin for
//! the same tile (the cold-cache == warm-cache gate). This is guaranteed *by
//! construction*, not by luck:
//!
//!   * The contraction is an **int32** `mma.m16n8k32` accumulate. Integer addition
//!     is exact and associative, so reordering the K loop or splitting the work
//!     across more warps/tiles never changes the accumulated value — every covered
//!     output sees the identical `Σ_k qact·trit`.
//!   * The only floating-point step is the single per-output
//!     `(float)acc · weight_scale[n] · act_scale[m]` fold (a `float` multiply chain
//!     in a fixed left-to-right order). It is emitted identically for every tile,
//!     so the rounding is identical.
//!
//! Codegen therefore only varies **launch geometry and shared-memory staging**
//! (how many 16×8 `mma` sub-tiles a block covers, how the warps partition them,
//! how deep the `cp.async` prefetch pipeline is) — never the arithmetic.
//!
//! ## Rev 4 staging (CODEGEN_REV 4)
//!
//! The rev-3 renderer staged with per-byte SIMT loads, kept B *unpacked* in
//! shared (4 bytes per 2-bit code), and scanned warp ownership with a
//! `st % WARPS` modulo test under `#pragma unroll 1`. Rev 4 replaces all three:
//!
//!   * **`cp.async` pipeline**: A tiles and the *packed* B tiles are staged
//!     global→shared with 16-byte `cp.async.cg` copies, `stages` buffers deep,
//!     one commit group per K step (empty groups pad the prologue so
//!     `wait_group(STAGES-2)` always certifies the current buffer). A row tail
//!     shorter than a 16B chunk only exists when `k % 16 != 0`; that uniform
//!     case takes a byte-staging fallback path (same shared bytes, same zeros).
//!   * **Packed B in shared**: the 64-byte I2sInt8 tiles are staged verbatim
//!     (4× less shared traffic + footprint than unpacked codes) and expanded to
//!     the int8 fragment u32 at `mma`-operand load time (a few ALU ops).
//!     Zero-filled/skipped B tiles decode to trit −1 garbage, which is SAFE by
//!     invariant: an out-of-range *k*-tile has an all-zero staged A (so the mma
//!     contributes exactly 0 regardless of B), and an out-of-range *n*-tile only
//!     feeds accumulators whose stores are masked by `gn < n`. Both invariants
//!     are load-bearing — do not "fix" the garbage decode.
//!   * **Warp-grid ownership**: the block's `M_SUBTILES × N_SUBTILES` grid is
//!     partitioned contiguously over a `WARPS_M × WARPS_N` warp grid (computed
//!     by [`TileConfig::warp_grid`]); each compute warp owns a `WM_PER × WN_PER`
//!     rectangle, loads each B fragment once per owned column and reuses each A
//!     fragment across its row — no modulo scan, no wasted fragment loads.
//!     Warps beyond the grid (`warps > subtiles`) still cooperate in staging.
//!
//! [`TileConfig`]: super::autotune::TileConfig
//! [`TileConfig::warp_grid`]: super::autotune::TileConfig::warp_grid

use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use tritium_spec::BackendError;

use super::autotune::TileConfig;

/// `mma.m16n8k32` output-tile dimensions — the hardware shape the kernel is built
/// on. A [`TileConfig`]'s `tile_m`/`tile_n`/`tile_k` are integer multiples of these
/// (validated by [`TileConfig::is_valid`]); the block covers `tile_m/16 × tile_n/8`
/// of these `mma` sub-tiles.
const MMA_M: u16 = 16;
const MMA_N: u16 = 8;
const MMA_K: u16 = 32;
/// Threads in a warp.
const WARP: u16 = 32;

/// The exported kernel symbol the host resolves after loading the JIT module. Must
/// match the `extern "C"` name rendered by [`render_imma_source`] and the AOT
/// kernel's `tq2_0_imma_mpgemm` so the same `load_function` call serves both paths.
pub(crate) const JIT_KERNEL_NAME: &str = "tq2_0_imma_mpgemm";

/// Render the IMMA kernel CUDA source specialised to `cfg`.
///
/// The output is a single translation unit exporting `tq2_0_imma_mpgemm` with the
/// **exact** signature the host launches (`const signed char* qact, const unsigned
/// char* weights, const float* act_scale, const float* weight_scale, float* out,
/// int m, int n, int k, int num_ktiles`). Only the launch geometry and staging vary
/// with `cfg`; the per-output arithmetic (int32 `mma` accumulate + the single scale
/// fold) is fixed, so every rendered config is bit-identical on shared inputs.
///
/// `cfg` must satisfy [`TileConfig::is_valid`]; callers funnel through
/// [`compile_imma`], which checks it first.
pub(crate) fn render_imma_source(cfg: TileConfig) -> String {
    let m_subtiles = (cfg.tile_m / MMA_M) as u32;
    let n_subtiles = (cfg.tile_n / MMA_N) as u32;
    let subtiles = m_subtiles * n_subtiles;
    let warps = cfg.warps as u32;
    let block_threads = warps * WARP as u32;
    let stages = cfg.stages as u32;
    // K-tiles consumed per main-loop step (tile_k / 32).
    let ktiles_per_step = (cfg.tile_k / MMA_K) as u32;
    // Contiguous warp-grid partition of the sub-tile grid (see autotune.rs).
    let (warps_m, warps_n) = cfg.warp_grid();
    let wm_per = m_subtiles / warps_m as u32;
    let wn_per = n_subtiles / warps_n as u32;

    // Shared staging per stage: unpacked int8 A (TILE_M × TILE_K) + the PACKED
    // 64-byte B tiles (N_SUBTILES × KTILES_PER_STEP of them).
    let a_stage_bytes = (cfg.tile_m as u32) * (cfg.tile_k as u32);
    let b_stage_bytes = n_subtiles * ktiles_per_step * 64;

    format!(
        r#"// AUTO-GENERATED by tritium-cuda::codegen (WF-B nvrtc JIT, rev 4). Do not edit.
//
// IMMA int8 ternary mpGEMM, specialised to tile_m={tile_m} tile_n={tile_n}
// tile_k={tile_k} warps={warps} stages={stages} (warp grid {warps_m}x{warps_n}).
// Numerics are identical to the AOT kernel (kernels/tq2_0_imma.cu): exact int32
// mma accumulate + one f32 scale fold per output. Only launch geometry and the
// cp.async staging vary with the tile.

#define IMMA_M {mma_m}
#define IMMA_N {mma_n}
#define IMMA_K {mma_k}
#define IMMA_WTILE_BYTES (IMMA_N * IMMA_K / 4)

#define TILE_M {tile_m}
#define TILE_N {tile_n}
#define TILE_K {tile_k}
#define M_SUBTILES {m_subtiles}
#define N_SUBTILES {n_subtiles}
#define SUBTILES {subtiles}
#define WARPS {warps}
#define STAGES {stages}
#define KTILES_PER_STEP {ktiles_per_step}
#define WARPS_M {warps_m}
#define WARPS_N {warps_n}
#define WM_PER {wm_per}
#define WN_PER {wn_per}
#define A_STAGE {a_stage_bytes}
#define B_STAGE {b_stage_bytes}

// 16-byte global->shared cp.async (sm_80+; this kernel already requires it for
// mma.m16n8k32). `bytes` is 0 or 16: 0 zero-fills the whole chunk, 16 copies it.
// The source pointer must be 16-byte aligned whenever bytes == 16.
__device__ __forceinline__ void cp16(void* smem, const void* gmem, int bytes) {{
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(gmem), "r"(bytes));
}}

// Expand one packed I2sInt8 byte (4 × 2-bit codes, low pair = first element)
// into the little-endian u32 of 4 int8 trits the mma B operand wants:
// byte j of the result = (int8)(((p >> 2j) & 3) - 1).
__device__ __forceinline__ unsigned expand_b(unsigned p) {{
    unsigned r = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {{
        const int t = (int)((p >> (2 * j)) & 3u) - 1;
        r |= ((unsigned)t & 0xFFu) << (8 * j);
    }}
    return r;
}}

extern "C" __global__ void __launch_bounds__({block_threads}) tq2_0_imma_mpgemm(
    const signed char* __restrict__ qact,       // int8 [M, K] row-major
    const unsigned char* __restrict__ weights,  // I2sInt8 packed
    const float* __restrict__ act_scale,        // [M]
    const float* __restrict__ weight_scale,     // [N]
    float* __restrict__ out,                     // [M, N] row-major
    const int m,
    const int n,
    const int k,
    const int num_ktiles) {{
    const int warp = threadIdx.x >> 5;           // 0..WARPS-1
    const int lane = threadIdx.x & 31;           // 0..31
    const int group = lane >> 2;                 // mma row/col group (0..7)
    const int tig = lane & 3;                    // thread-in-group (0..3)

    const int block_m = blockIdx.y * TILE_M;
    const int block_n = blockIdx.x * TILE_N;
    const int n_tiles = (n + IMMA_N - 1) / IMMA_N;  // n-tiles present in `weights`

    // STAGES-deep staging: unpacked int8 A + PACKED 64-byte B tiles per stage.
    // 16-byte aligned: cp.async destinations and the u32 fragment loads (every
    // fragment offset is a multiple of 4; cp16 chunks are multiples of 16).
    __shared__ __align__(16) signed char s_a[STAGES][A_STAGE];
    __shared__ __align__(16) unsigned char s_bp[STAGES][B_STAGE];

    // This warp's contiguous WM_PER x WN_PER rectangle of the sub-tile grid.
    // Warps at id >= WARPS_M*WARPS_N stage but do not compute.
    const int wgm = warp % WARPS_M;
    const int wgn = warp / WARPS_M;
    const bool computes = warp < WARPS_M * WARPS_N;

    int c0[WM_PER * WN_PER], c1[WM_PER * WN_PER], c2[WM_PER * WN_PER], c3[WM_PER * WN_PER];
    #pragma unroll
    for (int s = 0; s < WM_PER * WN_PER; ++s) {{
        c0[s] = 0; c1[s] = 0; c2[s] = 0; c3[s] = 0;
    }}

    const int num_steps = (num_ktiles + KTILES_PER_STEP - 1) / KTILES_PER_STEP;
    // Fast staging path: every in-range A chunk is a full, 16B-aligned 16-byte
    // copy exactly when k is a multiple of 16 (chunk starts are then aligned and
    // never straddle the row tail). Uniform across the block.
    const bool k16 = (k % 16) == 0;

    // Stage K-step `step` into buffer `buf`. All warps cooperate. Rows/k past
    // m/k stage as ZERO (cp.async zero-fill / explicit zero writes) — that zero
    // A is what makes the packed-B garbage decode safe (see module doc).
    auto stage = [&](int step, int buf) {{
        const int kt0 = step * KTILES_PER_STEP;
        if (k16) {{
            // A: TILE_M*TILE_K/16 16-byte chunks.
            for (int c = threadIdx.x; c < TILE_M * TILE_K / 16; c += blockDim.x) {{
                const int r = c / (TILE_K / 16);
                const int kc = c % (TILE_K / 16);
                const int gm = block_m + r;
                const int gk = kt0 * IMMA_K + kc * 16;
                const int bytes = (gm < m && gk < k) ? 16 : 0;
                // A zero-size copy still wants a valid source address: park it
                // at the buffer base.
                const signed char* src =
                    bytes ? qact + (long long)gm * k + gk : qact;
                cp16(&s_a[buf][r * TILE_K + kc * 16], src, bytes);
            }}
        }} else {{
            // Generality fallback (k % 16 != 0): per-byte staging, identical
            // shared bytes to the fast path (in-range copies + zero tails).
            for (int idx = threadIdx.x; idx < TILE_M * TILE_K; idx += blockDim.x) {{
                const int r = idx / TILE_K;
                const int c = idx % TILE_K;
                const int gm = block_m + r;
                const int gk = kt0 * IMMA_K + c;
                s_a[buf][idx] =
                    (gm < m && gk < k) ? qact[(long long)gm * k + gk] : (signed char)0;
            }}
        }}
        // B: the packed 64-byte tiles for this step, verbatim. Tile (nt, kt) is
        // at byte (nt*num_ktiles + kt)*64; both are always 16B-aligned. An
        // out-of-range tile zero-fills — its decode (trit -1) is annihilated by
        // zero A (kt oob) or masked at store (nt oob).
        for (int c = threadIdx.x; c < B_STAGE / 16; c += blockDim.x) {{
            const int tile_idx = c / 4;                 // 4 chunks per 64B tile
            const int chunk = c % 4;
            const int nn = tile_idx / KTILES_PER_STEP;
            const int kk = tile_idx % KTILES_PER_STEP;
            const int nt = blockIdx.x * N_SUBTILES + nn;
            const int kt = kt0 + kk;
            const int bytes = (nt < n_tiles && kt < num_ktiles) ? 16 : 0;
            const unsigned char* src = bytes
                ? weights + ((long long)nt * num_ktiles + kt) * IMMA_WTILE_BYTES + chunk * 16
                : weights;
            cp16(&s_bp[buf][tile_idx * IMMA_WTILE_BYTES + chunk * 16], src, bytes);
        }}
    }};

    // Prologue: fill the pipeline with exactly STAGES-1 commit groups (empty
    // groups pad past num_steps) so wait_group(STAGES-2) below always certifies
    // the oldest outstanding stage.
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {{
        if (s < num_steps) {{
            stage(s, s);
        }}
        asm volatile("cp.async.commit_group;\n" ::);
    }}

    for (int step = 0; step < num_steps; ++step) {{
        const int buf = step % STAGES;
        // Certify stage `step` landed (committed = STAGES-1+step groups; all but
        // the newest STAGES-2 are complete after this wait). The syncthreads also
        // fences the previous iteration's compute off the buffer the commit
        // below will overwrite.
        asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2));
        __syncthreads();
        {{
            const int nxt = step + STAGES - 1;
            if (nxt < num_steps) {{
                stage(nxt, nxt % STAGES);
            }}
            asm volatile("cp.async.commit_group;\n" ::);
        }}

        if (computes) {{
            #pragma unroll
            for (int kk = 0; kk < KTILES_PER_STEP; ++kk) {{
                const int kbase = kk * IMMA_K;
                // B fragments once per owned column: row `group` of tile
                // (nn, kk) is 8 packed bytes at T + group*8; b0 covers
                // k = 4*tig..4*tig+3 (byte tig), b1 covers 16+4*tig (byte 4+tig).
                unsigned b0[WN_PER], b1[WN_PER];
                #pragma unroll
                for (int j = 0; j < WN_PER; ++j) {{
                    const int nn = wgn * WN_PER + j;
                    const unsigned char* T =
                        s_bp[buf] + (nn * KTILES_PER_STEP + kk) * IMMA_WTILE_BYTES;
                    b0[j] = expand_b(T[group * 8 + tig]);
                    b1[j] = expand_b(T[group * 8 + 4 + tig]);
                }}
                #pragma unroll
                for (int i = 0; i < WM_PER; ++i) {{
                    const int mm = wgm * WM_PER + i;
                    const signed char* A = s_a[buf] + mm * (IMMA_M * TILE_K);
                    const unsigned a0 =
                        *(const unsigned*)(A + group * TILE_K + kbase + 4 * tig);
                    const unsigned a1 =
                        *(const unsigned*)(A + (group + 8) * TILE_K + kbase + 4 * tig);
                    const unsigned a2 =
                        *(const unsigned*)(A + group * TILE_K + kbase + 16 + 4 * tig);
                    const unsigned a3 =
                        *(const unsigned*)(A + (group + 8) * TILE_K + kbase + 16 + 4 * tig);
                    #pragma unroll
                    for (int j = 0; j < WN_PER; ++j) {{
                        const int slot = i * WN_PER + j;
                        asm volatile(
                            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};\n"
                            : "+r"(c0[slot]), "+r"(c1[slot]), "+r"(c2[slot]), "+r"(c3[slot])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0[j]), "r"(b1[j]));
                    }}
                }}
            }}
        }}
        // No trailing barrier: the next iteration's wait_group + __syncthreads
        // is what fences every warp's compute off the buffer the next commit
        // overwrites, and stage(step+STAGES-1) writes buffer (step-1)%STAGES,
        // never the one being computed (STAGES >= 2).
    }}

    // Store each owned sub-tile, folding scales (single f32 multiply chain, the
    // same order as the AOT kernel: (float)acc * weight_scale * act_scale — the
    // dp4a family's exact association (bit-identity contract, ADR 0026)).
    if (computes) {{
        #pragma unroll
        for (int i = 0; i < WM_PER; ++i) {{
            #pragma unroll
            for (int j = 0; j < WN_PER; ++j) {{
                const int slot = i * WN_PER + j;
                const int tile_m0 = block_m + (wgm * WM_PER + i) * IMMA_M;
                const int tile_n0 = block_n + (wgn * WN_PER + j) * IMMA_N;
                auto store = [&](int acc, int row_in_tile, int col_in_tile) {{
                    const int gm = tile_m0 + row_in_tile;
                    const int gn = tile_n0 + col_in_tile;
                    if (gm < m && gn < n) {{
                        out[(long long)gm * n + gn] =
                            (float)acc * weight_scale[gn] * act_scale[gm];
                    }}
                }};
                store(c0[slot], group, 2 * tig);
                store(c1[slot], group, 2 * tig + 1);
                store(c2[slot], group + 8, 2 * tig);
                store(c3[slot], group + 8, 2 * tig + 1);
            }}
        }}
    }}
}}
"#,
        mma_m = MMA_M,
        mma_n = MMA_N,
        mma_k = MMA_K,
        tile_m = cfg.tile_m,
        tile_n = cfg.tile_n,
        tile_k = cfg.tile_k,
        m_subtiles = m_subtiles,
        n_subtiles = n_subtiles,
        subtiles = subtiles,
        warps = warps,
        stages = stages,
        ktiles_per_step = ktiles_per_step,
        warps_m = warps_m,
        warps_n = warps_n,
        wm_per = wm_per,
        wn_per = wn_per,
        a_stage_bytes = a_stage_bytes,
        b_stage_bytes = b_stage_bytes,
        block_threads = block_threads,
    )
}

/// JIT-compile the IMMA kernel for `cfg` to a loadable PTX image via nvrtc, built
/// for `arch` (e.g. `"compute_80"` for forward-compatible PTX, or `"sm_89"` for a
/// device-specific cubin). Returns the [`Ptx`] image ready for
/// [`cudarc::driver::CudaContext::load_module`].
///
/// # Errors
/// [`BackendError::InvalidInput`] if `cfg` is not [`TileConfig::is_valid`];
/// [`BackendError::Backend`] (with the nvrtc compile log) on a compilation failure.
pub(crate) fn compile_imma(cfg: TileConfig, arch: &'static str) -> Result<Ptx, BackendError> {
    if !cfg.is_valid() {
        return Err(BackendError::InvalidInput(format!(
            "invalid IMMA tile config {cfg:?}: tile dims must be positive multiples of \
             16x8x32 and warps/stages >= 1"
        )));
    }
    let src = render_imma_source(cfg);
    let opts = CompileOptions {
        arch: Some(arch),
        // Match the AOT build's numeric flags. The AOT path compiles with `nvcc -O3`
        // and no `--fmad`/`--ftz` overrides, so the default FMA-contraction and
        // denormal behaviour apply; nvrtc defaults to the same, so we leave these at
        // None rather than forcing them and risking a divergence from the AOT cubin.
        // (The single scale fold is a multiply chain, not an FMA candidate, so FMA
        // contraction does not affect it regardless.)
        ..Default::default()
    };
    compile_ptx_with_opts(&src, opts).map_err(|e| {
        BackendError::Backend(format!("nvrtc compile IMMA tile {cfg:?} for {arch}: {e}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendered_source_has_kernel_symbol() {
        let src = render_imma_source(TileConfig::AOT_EQUIVALENT);
        assert!(
            src.contains("void __launch_bounds__"),
            "expected launch-bounds qualifier in rendered source"
        );
        assert!(
            src.contains(JIT_KERNEL_NAME),
            "expected the kernel symbol {JIT_KERNEL_NAME} in rendered source"
        );
        // The mma instruction must survive templating verbatim (load-bearing for the
        // exact-int32 accumulate).
        assert!(
            src.contains("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"),
            "expected the m16n8k32 mma instruction in rendered source"
        );
        // Rev 4: the cp.async pipeline must be present.
        assert!(
            src.contains("cp.async.cg.shared.global"),
            "expected cp.async staging in rendered source"
        );
        assert!(
            src.contains("cp.async.commit_group") && src.contains("cp.async.wait_group"),
            "expected cp.async pipeline control in rendered source"
        );
    }

    #[test]
    fn aot_equivalent_renders_single_warp_single_subtile() {
        // The AOT-equivalent config must render exactly one 16x8 sub-tile and one
        // warp per block, matching kernels/tq2_0_imma.cu's launch shape.
        let src = render_imma_source(TileConfig::AOT_EQUIVALENT);
        assert!(
            src.contains("#define SUBTILES 1"),
            "AOT-equiv must be 1 sub-tile"
        );
        assert!(src.contains("#define WARPS 1"), "AOT-equiv must be 1 warp");
        assert!(src.contains("#define M_SUBTILES 1"));
        assert!(src.contains("#define N_SUBTILES 1"));
        assert!(src.contains("#define WARPS_M 1"));
        assert!(src.contains("#define WARPS_N 1"));
    }

    #[test]
    fn multi_warp_config_renders_consistent_geometry() {
        let cfg = TileConfig {
            tile_m: 32,
            tile_n: 16,
            tile_k: 64,
            warps: 4,
            stages: 2,
        };
        let src = render_imma_source(cfg);
        assert!(src.contains("#define M_SUBTILES 2"));
        assert!(src.contains("#define N_SUBTILES 2"));
        assert!(src.contains("#define SUBTILES 4"));
        assert!(src.contains("#define WARPS 4"));
        assert!(src.contains("#define KTILES_PER_STEP 2"));
        // 4 warps over a 2x2 sub-tile grid: 2x2 warp grid, one sub-tile each.
        assert!(src.contains("#define WARPS_M 2"));
        assert!(src.contains("#define WARPS_N 2"));
        assert!(src.contains("#define WM_PER 1"));
        assert!(src.contains("#define WN_PER 1"));
    }

    #[test]
    fn oversubscribed_warps_idle_in_compute() {
        // BASELINE has 4 warps but a single 16x8 sub-tile: the warp grid must
        // stay 1x1 (3 warps stage-only), never a grid that outruns the
        // sub-tiles.
        let src = render_imma_source(TileConfig::BASELINE);
        assert!(src.contains("#define WARPS 4"));
        assert!(src.contains("#define WARPS_M 1"));
        assert!(src.contains("#define WARPS_N 1"));
        assert!(src.contains("#define WM_PER 1"));
        assert!(src.contains("#define WN_PER 1"));
    }

    #[test]
    fn packed_b_stage_is_quarter_of_unpacked() {
        // Rev 4 stages B packed: B_STAGE must be tile_n*tile_k/4 bytes.
        let cfg = TileConfig {
            tile_m: 128,
            tile_n: 64,
            tile_k: 64,
            warps: 8,
            stages: 3,
        };
        let src = render_imma_source(cfg);
        // 8 n-subtiles * 2 k-tiles * 64 bytes = 1024 = 64*64/4.
        assert!(src.contains("#define B_STAGE 1024"));
        assert!(src.contains("#define A_STAGE 8192"));
    }
}

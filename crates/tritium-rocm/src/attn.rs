//! Host-side structure for the v3 Q-blocked prefill attention port
//! ([`gqa_attention_v3.hip`](../kernels/gqa_attention_v3.hip)) — the HIP twin
//! of tritium-cuda's `gqa_attention_batch_v3_f32` (`kernels/decode.cu`),
//! Track E2. The Metal port (crates/tritium-metal/src/attn.rs +
//! attention.metal) is the sibling of this module; the structure below
//! mirrors it deliberately so the two non-CUDA ports stay reviewable against
//! each other.
//!
//! Everything in this module is plain Rust with **no HIP dependency**, so it
//! compiles — and its tests run — in the default (no-`rocm`) build on every
//! platform, including the cpu-only CI matrix. It holds:
//!
//! * the launch constants that MUST mirror the `#define`s in
//!   `gqa_attention_v3.hip` (pinned by
//!   [`tests::attn_v3_consts_match_hip_defines`]), which in turn mirror
//!   decode.cu's `ATTN_V3_*` family (pinned on the CUDA side by
//!   `attn_v2_consts_match_decode_cu_defines`);
//! * the dispatch geometry + validation the ROCm backend uses (pure
//!   functions, unit-tested here);
//! * the `TRITIUM_ATTN_V3` kill-switch parser (the tritium-cuda env pattern:
//!   unset/`1` = on, `0` = off, anything else is a loud reject);
//! * [`gqa_attention_prefill_ref`], the **pinned-order host reference**: the
//!   same per-(row, head) summation orders as the kernel, in plain `f32` ops
//!   (Rust `f32` `+`/`*` are IEEE round-to-nearest with no implicit FMA —
//!   the host-side meaning of CUDA's `__fadd_rn`/`__fmul_rn`). Since this
//!   backend has no rev-1/v2 device attention to fall back to, the reference
//!   is both the kill-switch fallback and the oracle the MI300X-lane gate
//!   compares the GPU kernel against.
//!
//! ## Wavefront contract (wave64 CDNA vs wave32 RDNA)
//!
//! The kernel keeps CUDA's 32-lane groups as LOGICAL lanes and uses width-32
//! shuffles, so it is wave-size-agnostic by construction — the full analysis
//! (including why the 64-lane-row alternative was rejected, and the one
//! structural deviation: CUDA's in-guard `__syncwarp` becomes a hoisted
//! unconditional `__syncthreads`) lives in the kernel banner. The host still
//! probes the physical `warpSize` at backend init (`attn_v3_wave_probe`) and
//! refuses the device rung unless it is 32 or 64 — the dispatch check
//! mirroring Metal's `thread_execution_width == 32` refusal.
//!
//! ## Bit parity with the host (unlike Metal, HIP has doubles)
//!
//! CUDA v3 is gated `to_bits`-identical to rev-1 because both use
//! `exp_f32(x) = __double2float_rn(exp((double)x))` — f64 exp rounded once,
//! matching glibc `expf`. HIP HAS `double` (native f64 on CDNA), so the HIP
//! kernel keeps that exact spelling; AMD's ocml f64 `exp` is ~1 ULP like
//! CUDA's, so bit-identity with this module's reference (`f32::exp` = glibc
//! `expf`) is EXPECTED. It is verified rather than assumed: the MI300X gate
//! (`rocm::tests::attn_v3_matches_pinned_host_reference_or_skip`) hard-fails
//! on any drift past a tight exp-only tolerance (a real order/mapping bug)
//! and asserts bit-equality as a second, separately-diagnosed tier
//! (`TRITIUM_ROCM_ATTN_STRICT_BITS=0` demotes only that tier to a report, so
//! the one budgeted session can finish its gates in a single pass even if
//! ocml's exp deviates by a ULP — the finding gets recorded either way).
//!
//! ## MI300X cloud-session runbook (Track E2 — run these, don't improvise)
//!
//! One budgeted session validates this port. On the rented box (Hot Aisle
//! MI300X VF or equivalent, ROCm 7.x):
//!
//! ```text
//! # 0. go/no-go (memory: functional VF works, broken SR-IOV wedges):
//! timeout 20 rocminfo
//!
//! # 1. from the Tritium checkout root — build kernels via hipcc and run
//! #    BOTH device gates in one pass (tq2_0 conformance + the E2 attention
//! #    gate). Needs ROCM_PATH or /opt/rocm; build.rs finds hipcc itself.
//! #    REQUIRE_DEVICE turns a load-time .co rejection into a hard failure
//! #    instead of a self-skip — mandatory on the paid session.
//! TRITIUM_ROCM_REQUIRE_DEVICE=1 cargo test -p tritium-rocm --features rocm --release -- --nocapture
//!
//! # 2. if (and only if) the gate's BIT-EQUALITY tier fails while the
//! #    tolerance tier passes (an exp-ULP finding, not a bug): re-run with
//! #    the strict-bits tier demoted to a report, record the printed
//! #    mismatch count + max ULP in the session notes, and file the
//! #    follow-up to pick a documented tolerance gate (the Metal precedent):
//! TRITIUM_ROCM_ATTN_STRICT_BITS=0 \
//!   cargo test -p tritium-rocm --features rocm --release -- --nocapture
//!
//! # 3. kill-switch sanity (host-reference fallback path):
//! TRITIUM_ATTN_V3=0 cargo test -p tritium-rocm --features rocm --release
//!
//! # optional bench (only if budget remains; correctness is the session's
//! # deliverable): none prepared — do not improvise one on rental time.
//! ```
//!
//! ## MFMA int8 GEMM — design memo (E2 second item; analysis only)
//!
//! What an IMMA-equivalent compute-bound prefill mpGEMM would take on CDNA3,
//! against the CUDA structure (`kernels/tq2_0_imma.cu` AOT + the rev-5 nvrtc
//! codegen in `tritium-cuda/src/codegen.rs`):
//!
//! * **Instruction**: `__builtin_amdgcn_mfma_i32_16x16x32_i8` (gfx942; the
//!   32x32x16 sibling also exists). One MFMA is wavefront-wide (wave64):
//!   A 16x32 i8, B 32x16 i8, C/D 16x16 i32 per instruction — vs CUDA's
//!   per-warp `mma.m16n8k32` (16x32 x 32x8). The determinism argument
//!   carries over unchanged: the contraction is exact associative int32, so
//!   tiling/scheduling never changes a value, and the single per-output
//!   `(float)acc * weight_scale[n] * act_scale[m]` fold pins the only
//!   floating-point rounding.
//! * **I2sInt8 interleave**: the CUDA weight tile is an 8n x 32k packed
//!   2-bit tile (64 B) laid out for `mma`'s B fragment. MFMA's B fragment is
//!   lane-major over a 32k x 16n operand — the 8x32 tile does NOT map 1:1.
//!   Two options: (i) keep `tritium_format::convert_i2s_to_int8` unchanged
//!   and re-derive the per-lane unpack addressing at LDS-load time (pure
//!   index math, like the rev-4 "packed B in shared" expansion — preferred,
//!   zero format churn); (ii) a gfx942-specific interleave in tritium-format
//!   (format churn, couples the wire format to one vendor — not preferred).
//! * **Pipeline**: no `cp.async` on CDNA3. First cut: plain LDS staging with
//!   manual double buffering (the rev-3 shape); `global_load_lds` /
//!   `llvm.amdgcn.load.to.lds` builtins are the later optimization, not a
//!   correctness need.
//! * **Codegen**: `hiprtc` is API-compatible with nvrtc, so the rev-5
//!   renderer's *host* machinery (autotune, cache, JIT-vs-AOT parity gate)
//!   would port mechanically, but the rendered source needs an MFMA backend
//!   (different fragment ownership, no `ldmatrix`, XOR swizzle re-derived
//!   for 64-lane LDS banking).
//! * **Estimated effort**: fixed-tile AOT kernel (the `tq2_0_imma_mpgemm`
//!   analogue) ≈ 2–4 focused days plus one MI300X tuning/validation session;
//!   the full codegen/autotune port is a separate campaign (roughly the size
//!   of the original rev-3→rev-5 arc). **Recommendation**: out of scope for
//!   the one budgeted E2 session — decide after the attention gate lands.

/// Threads per block — mirrors decode.cu `ATTN_V3_THREADS` (256 = 8 CUDA
/// warps). On HIP the block splits into [`ATTN_V3_WARPS`] logical 32-lane
/// groups regardless of the physical wavefront width (see the kernel
/// banner); the host refuses the device path if the probed `warpSize` is
/// not 32 or 64.
pub const ATTN_V3_THREADS: u32 = 256;
/// Query rows blocked per (block, head) — decode.cu `ATTN_V3_BQ`.
pub const ATTN_V3_BQ: usize = 8;
/// Keys staged per K/V chunk — decode.cu `ATTN_V3_KCH`.
pub const ATTN_V3_KCH: usize = 32;
/// Max `head_dim` the kernel's LDS staging is sized for — decode.cu
/// `ATTN_V2_HDMAX` (the v3 CUDA kernel reuses the v2 bound; the HIP source
/// names it `ATTN_V3_HDMAX` because this crate has no v2).
pub const ATTN_V3_HDMAX: usize = 128;
/// Lanes per LOGICAL group the kernel's warp mapping assumes (CUDA warp
/// width; physical wavefronts may be 64 — the kernel shuffles at width 32).
pub const ATTN_V3_LANES: usize = 32;
/// Logical groups per block (CUDA `ATTN_V3_WARPS`).
pub const ATTN_V3_WARPS: usize = ATTN_V3_THREADS as usize / ATTN_V3_LANES;
/// Query rows owned per logical group in phases 1/2 (CUDA `ATTN_V3_RPW`).
pub const ATTN_V3_RPW: usize = ATTN_V3_BQ / ATTN_V3_WARPS;

// The CUDA kernel's static_assert, host-side: RPW = 0 (more groups than BQ
// rows) would compile phases 1/2 to nothing.
const _: () = assert!(ATTN_V3_BQ.is_multiple_of(ATTN_V3_WARPS) && ATTN_V3_RPW >= 1);

/// Parse a `TRITIUM_ATTN_V3` value: unset / `"1"` → enabled (default),
/// `"0"` → disabled, anything else is a loud reject — the exact behaviour of
/// tritium-cuda's `attn_env` closure (and tritium-metal's twin parser) for
/// the same variable.
///
/// Pure (takes the value, not the environment) so it is unit-testable without
/// racy `set_var` in parallel test threads; the backend wraps it with a
/// `std::env::var` read.
///
/// # Errors
/// A description of the malformed value, for the backend to surface as
/// `BackendError::InvalidInput`.
pub fn parse_attn_v3(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(v) => Err(format!("TRITIUM_ATTN_V3={v:?} — use 1 (default) or 0")),
    }
}

/// Launch grid for a v3 dispatch: `(n_head, ceil(m / ATTN_V3_BQ))` — the
/// exact CUDA grid (`grid_dim = (n_head, ceil(m/BQ), 1)`), with
/// [`ATTN_V3_THREADS`]`×1×1` threads per block.
#[must_use]
pub fn v3_grid(m: usize, n_head: usize) -> (u32, u32) {
    // Callers bound both against the dispatch grid ceiling before launching;
    // the casts cannot truncate because validate_v3_launch caps the inputs
    // at i32::MAX.
    (n_head as u32, m.div_ceil(ATTN_V3_BQ) as u32)
}

/// Element count of the global `[m, n_head, ctx_max]` scores scratch the v3
/// kernel requires (in CUDA it is what lifts v2's ctx cap; here it is simply
/// the kernel's contract). `None` on overflow.
#[must_use]
pub fn v3_scores_len(m: usize, n_head: usize, ctx_max: usize) -> Option<usize> {
    m.checked_mul(n_head)?.checked_mul(ctx_max)
}

/// Validate a prefill-attention call against the kernel's host contract.
/// Length arguments (not slices) keep it pure and trivially unit-testable.
/// The twin of tritium-metal's `validate_v3_launch` (same checks, same
/// order); duplicated rather than shared because no common non-backend crate
/// is a clean home for backend launch contracts.
///
/// Checks shape coherence (`n_head % n_head_kv`, `causal_offset + m <=
/// ctx_max`), buffer lengths (`q`/`out` exactly `m·n_head·head_dim`; `k`/`v`
/// arenas at least `ctx_top·n_head_kv·head_dim` rows), the kernel's `i32`
/// scalar range (the HIP kernel takes `int` scalars), and scores-scratch
/// overflow. It does NOT enforce `head_dim <= ATTN_V3_HDMAX` — that is a
/// *dispatch* bound (too-large heads fall back to the host reference,
/// mirroring CUDA's v3 → baseline priority), not an input error.
///
/// # Errors
/// A description of the first violated invariant.
#[allow(clippy::too_many_arguments)]
pub fn validate_v3_launch(
    q_len: usize,
    k_len: usize,
    v_len: usize,
    out_len: usize,
    ctx_max: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    causal_offset: usize,
    m: usize,
) -> Result<(), String> {
    if m == 0 || n_head == 0 || n_head_kv == 0 || head_dim == 0 {
        return Err(format!(
            "degenerate attention shape: m={m} n_head={n_head} n_head_kv={n_head_kv} head_dim={head_dim}"
        ));
    }
    if !n_head.is_multiple_of(n_head_kv) {
        return Err(format!(
            "n_head {n_head} not a multiple of n_head_kv {n_head_kv}"
        ));
    }
    let ctx_top = causal_offset
        .checked_add(m)
        .ok_or_else(|| format!("causal_offset {causal_offset} + m {m} overflows"))?;
    if ctx_top > ctx_max {
        return Err(format!(
            "causal_offset {causal_offset} + m {m} = {ctx_top} exceeds ctx_max {ctx_max}"
        ));
    }
    let q_expect = m
        .checked_mul(n_head)
        .and_then(|x| x.checked_mul(head_dim))
        .ok_or_else(|| "q length m*n_head*head_dim overflows".to_owned())?;
    if q_len != q_expect {
        return Err(format!("q len {q_len} != m*n_head*head_dim = {q_expect}"));
    }
    if out_len != q_expect {
        return Err(format!(
            "out len {out_len} != m*n_head*head_dim = {q_expect}"
        ));
    }
    let kv_min = ctx_top
        .checked_mul(n_head_kv)
        .and_then(|x| x.checked_mul(head_dim))
        .ok_or_else(|| "kv arena length ctx_top*n_head_kv*head_dim overflows".to_owned())?;
    if k_len < kv_min {
        return Err(format!(
            "k len {k_len} < ctx_top*n_head_kv*head_dim = {kv_min}"
        ));
    }
    if v_len < kv_min {
        return Err(format!(
            "v len {v_len} < ctx_top*n_head_kv*head_dim = {kv_min}"
        ));
    }
    // The HIP kernel takes `int` scalars (and widens per-row bases to 64
    // bits in-kernel, the v0.6.9 discipline) — reject anything that would go
    // negative in the i32 cast rather than dispatching a wrapped value.
    // (`ctx_top <= ctx_max` above keeps the kernel's `causal_offset + row0 +
    // nrows` int sum in range too.)
    for (name, val) in [
        ("ctx_max", ctx_max),
        ("n_head", n_head),
        ("n_head_kv", n_head_kv),
        ("head_dim", head_dim),
        ("causal_offset", causal_offset),
        ("m", m),
    ] {
        if val > i32::MAX as usize {
            return Err(format!(
                "{name} {val} exceeds the kernel's i32 param range (the HIP \
                 kernel takes int scalars)"
            ));
        }
    }
    if v3_scores_len(m, n_head, ctx_max).is_none() {
        return Err("scores scratch m*n_head*ctx_max overflows".to_owned());
    }
    Ok(())
}

/// Pinned-order host reference for the v3 prefill attention — the CPU twin
/// of `gqa_attention_batch_v3_f32` (and, transitively, of the CUDA
/// rev-1/v2/v3 family, whose per-(row, head) orders it reproduces).
/// Duplicated from tritium-metal's `attn::gqa_attention_prefill_ref` (the
/// same function, verbatim) with a cross-link rather than shared through a
/// dependency: the only common ancestors of the two backend crates are
/// spec/core/format, none of which is a clean home for a backend-family
/// reference, and a metal<->rocm dependency either way would be wrong.
///
/// Layouts (row-major, exactly the kernel's):
/// * `q`, `out`: `[m, n_head, head_dim]`
/// * `k`, `v`: `[>= causal_offset + m, n_head_kv, head_dim]` (KV arena rows)
///
/// Per (row `r`, head `h`), with `ctx = causal_offset + r + 1` and
/// `kv = h / (n_head / n_head_kv)`:
/// 1. **dot chain** — for each key `j < ctx`, a single sequential
///    d-ascending `dot = dot + q[d]*k[d]` chain, then one `dot * scale`
///    rounding (kernel phase 1: one lane owns the whole chain).
/// 2. **softmax** — max via [`f32::max`] (order-exact), elementwise
///    `exp(x - max)`, then the ORDER-PINNED sequential j-ascending sum and
///    one `1.0 / sum` division, elementwise `* inv` (kernel phase 2: the
///    sequential sum runs on logical lane 0).
/// 3. **V fold** — per output dim `d`, a sequential j-ascending
///    `acc = acc + w*v` chain with the rev-1 `w == 0.0` skip (kernel
///    phase 3 folds chunk-ascending then within-chunk-ascending = the same
///    global j order).
///
/// The exponential is Rust [`f32::exp`] (glibc `expf`, correctly rounded) —
/// the CUDA `exp_f32` was built to match it, and the HIP kernel keeps CUDA's
/// f64-round-once spelling, so device output is expected bit-identical (the
/// MI300X gate verifies; see the module docs).
///
/// Caller must have validated via [`validate_v3_launch`]; this function
/// indexes accordingly.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_prefill_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    scale: f32,
    causal_offset: usize,
    m: usize,
) {
    let n_rep = n_head / n_head_kv;
    let mut weights: Vec<f32> = Vec::new();
    for r in 0..m {
        let ctx = causal_offset + r + 1;
        for h in 0..n_head {
            let kv = h / n_rep;
            let q_row = &q[(r * n_head + h) * head_dim..(r * n_head + h) * head_dim + head_dim];

            // Phase 1: scaled dots, one d-ascending chain per key.
            weights.clear();
            weights.reserve(ctx);
            for j in 0..ctx {
                let k_row =
                    &k[(j * n_head_kv + kv) * head_dim..(j * n_head_kv + kv) * head_dim + head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let prod = q_row[d] * k_row[d];
                    dot += prod;
                }
                weights.push(dot * scale);
            }

            // Phase 2: max (order-exact), exp, PINNED sequential sum, inv scale.
            let mut mx = f32::NEG_INFINITY;
            for &s in &weights {
                mx = mx.max(s);
            }
            for s in weights.iter_mut() {
                *s = (*s - mx).exp();
            }
            let mut sum = 0.0f32;
            for &s in &weights {
                sum += s;
            }
            let inv = 1.0 / sum;
            for s in weights.iter_mut() {
                *s *= inv;
            }

            // Phase 3: per-dim sequential j-ascending V fold with the zero-skip.
            let o_row =
                &mut out[(r * n_head + h) * head_dim..(r * n_head + h) * head_dim + head_dim];
            for (d, o) in o_row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (j, &w) in weights.iter().enumerate() {
                    if w != 0.0 {
                        let prod = w * v[(j * n_head_kv + kv) * head_dim + d];
                        acc += prod;
                    }
                }
                *o = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The N2-guardrail pattern from tritium-cuda
    /// (`attn_v2_consts_match_decode_cu_defines`), pointed at the HIP source:
    /// the Rust dispatch constants above must equal `gqa_attention_v3.hip`'s
    /// `#define`s — a host value drifting LARGER than the kernel's LDS
    /// sizing would dispatch out-of-bounds block indices that only an AMD
    /// box could catch. Parsed from source; runs on every platform.
    /// (decode.cu's own `ATTN_V3_*`/`ATTN_V2_HDMAX` values are pinned by the
    /// CUDA-side test; a cross-crate `include_str!` would break packaging,
    /// so the CUDA<->HIP agreement is documented rather than parsed here —
    /// the same note as the Metal twin.)
    #[test]
    fn attn_v3_consts_match_hip_defines() {
        let src = include_str!("../kernels/gqa_attention_v3.hip");
        let get = |name: &str| -> usize {
            src.lines()
                .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
                .unwrap_or_else(|| panic!("#define {name} missing from gqa_attention_v3.hip"))
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("#define {name} is not a bare number"))
        };
        assert_eq!(get("ATTN_V3_THREADS"), ATTN_V3_THREADS as usize);
        assert_eq!(get("ATTN_V3_BQ"), ATTN_V3_BQ);
        assert_eq!(get("ATTN_V3_KCH"), ATTN_V3_KCH);
        assert_eq!(get("ATTN_V3_HDMAX"), ATTN_V3_HDMAX);
    }

    #[test]
    fn kill_switch_parses_like_cuda_attn_env() {
        assert_eq!(parse_attn_v3(None), Ok(true));
        assert_eq!(parse_attn_v3(Some("1")), Ok(true));
        assert_eq!(parse_attn_v3(Some("0")), Ok(false));
        assert!(parse_attn_v3(Some("yes")).is_err());
        assert!(parse_attn_v3(Some("")).is_err());
    }

    #[test]
    fn grid_geometry_matches_cuda_launch() {
        assert_eq!(v3_grid(1, 32), (32, 1));
        assert_eq!(v3_grid(8, 32), (32, 1));
        assert_eq!(v3_grid(9, 32), (32, 2));
        assert_eq!(v3_grid(512, 24), (24, 64));
        assert_eq!(v3_scores_len(512, 24, 2048), Some(512 * 24 * 2048));
        assert_eq!(v3_scores_len(usize::MAX, 2, 2), None);
    }

    #[test]
    fn validate_rejects_bad_shapes_and_lengths() {
        let ok = validate_v3_launch(
            5 * 8 * 64, // q
            5 * 2 * 64, // k (ctx_top = 5)
            5 * 2 * 64, // v
            5 * 8 * 64, // out
            5,          // ctx_max
            8,
            2,
            64,
            0,
            5,
        );
        assert_eq!(ok, Ok(()));
        // n_head not a multiple of n_head_kv.
        assert!(
            validate_v3_launch(
                5 * 7 * 64,
                5 * 2 * 64,
                5 * 2 * 64,
                5 * 7 * 64,
                5,
                7,
                2,
                64,
                0,
                5
            )
            .is_err()
        );
        // ctx_top exceeds ctx_max.
        assert!(
            validate_v3_launch(
                5 * 8 * 64,
                5 * 2 * 64,
                5 * 2 * 64,
                5 * 8 * 64,
                4,
                8,
                2,
                64,
                0,
                5
            )
            .is_err()
        );
        // Short K arena.
        assert!(
            validate_v3_launch(
                5 * 8 * 64,
                4 * 2 * 64,
                5 * 2 * 64,
                5 * 8 * 64,
                5,
                8,
                2,
                64,
                0,
                5
            )
            .is_err()
        );
        // Wrong q length.
        assert!(
            validate_v3_launch(
                5 * 8 * 64 - 1,
                5 * 2 * 64,
                5 * 2 * 64,
                5 * 8 * 64,
                5,
                8,
                2,
                64,
                0,
                5
            )
            .is_err()
        );
        // m = 0.
        assert!(validate_v3_launch(0, 0, 0, 0, 5, 8, 2, 64, 0, 0).is_err());
        // ctx_max past i32::MAX: the kernel takes int scalars, so a value
        // that fits u32 but not i32 must still be rejected (arena lengths
        // only need to cover ctx_top, so this reaches the range check).
        assert!(
            validate_v3_launch(
                5 * 8 * 64,
                5 * 2 * 64,
                5 * 2 * 64,
                5 * 8 * 64,
                i32::MAX as usize + 1,
                8,
                2,
                64,
                0,
                5
            )
            .is_err()
        );
        // Oversized arenas are allowed (KV arenas may be over-allocated).
        assert_eq!(
            validate_v3_launch(
                5 * 8 * 64,
                9 * 2 * 64,
                9 * 2 * 64,
                5 * 8 * 64,
                9,
                8,
                2,
                64,
                0,
                5
            ),
            Ok(())
        );
    }

    /// Deterministic input generator (the xorshift pattern the Metal twin's
    /// tests use — no external rng).
    #[allow(clippy::type_complexity)]
    fn random_case(
        m: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        causal_offset: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let ctx_top = causal_offset + m;
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15
            ^ ((m as u64) << 1)
            ^ ((n_head as u64) << 17)
            ^ ((head_dim as u64) << 33)
            ^ (causal_offset as u64);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f32 / u64::MAX as f32) * 2.0 - 1.0
        };
        let q: Vec<f32> = (0..m * n_head * head_dim).map(|_| next()).collect();
        let k: Vec<f32> = (0..ctx_top * n_head_kv * head_dim)
            .map(|_| next())
            .collect();
        let v: Vec<f32> = (0..ctx_top * n_head_kv * head_dim)
            .map(|_| next())
            .collect();
        (q, k, v)
    }

    /// f64-accumulated independent oracle (different order + precision, no
    /// zero-skip) — validates the reference's MATH; the reference's ORDERS
    /// are validated by construction + review against decode.cu.
    #[allow(clippy::too_many_arguments)]
    fn oracle_f64(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
        causal_offset: usize,
        m: usize,
    ) {
        let n_rep = n_head / n_head_kv;
        for r in 0..m {
            let ctx = causal_offset + r + 1;
            for h in 0..n_head {
                let kv = h / n_rep;
                let scores: Vec<f64> = (0..ctx)
                    .map(|j| {
                        (0..head_dim)
                            .map(|d| {
                                f64::from(q[(r * n_head + h) * head_dim + d])
                                    * f64::from(k[(j * n_head_kv + kv) * head_dim + d])
                            })
                            .sum::<f64>()
                            * f64::from(scale)
                    })
                    .collect();
                let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|&s| (s - mx).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for d in 0..head_dim {
                    let acc: f64 = exps
                        .iter()
                        .enumerate()
                        .map(|(j, &e)| e / sum * f64::from(v[(j * n_head_kv + kv) * head_dim + d]))
                        .sum();
                    out[(r * n_head + h) * head_dim + d] = acc as f32;
                }
            }
        }
    }

    #[test]
    fn reference_matches_f64_oracle() {
        // Staircase + tail (m not a BQ multiple), GQA + MHA, head_dim at and
        // below the HDMAX cap, deep ctx (causal_offset > 0).
        for &(m, n_head, n_head_kv, head_dim, causal_offset) in &[
            (1usize, 4usize, 4usize, 64usize, 0usize),
            (11, 8, 2, 64, 0),
            (5, 8, 2, 80, 37),
            (16, 4, 1, 128, 3),
        ] {
            let (q, k, v) = random_case(m, n_head, n_head_kv, head_dim, causal_offset);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut got = vec![0.0f32; m * n_head * head_dim];
            let mut want = vec![0.0f32; m * n_head * head_dim];
            gqa_attention_prefill_ref(
                &q,
                &k,
                &v,
                &mut got,
                n_head,
                n_head_kv,
                head_dim,
                scale,
                causal_offset,
                m,
            );
            oracle_f64(
                &q,
                &k,
                &v,
                &mut want,
                n_head,
                n_head_kv,
                head_dim,
                scale,
                causal_offset,
                m,
            );
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let denom = w.abs().max(1.0);
                assert!(
                    ((g - w) / denom).abs() <= 1e-4,
                    "[{i}] got {g} want {w} (m={m} n_head={n_head} kv={n_head_kv} hd={head_dim} co={causal_offset})"
                );
            }
        }
    }

    /// Row independence, bitwise: row r of an m-row call equals a 1-row call
    /// at causal_offset + r — the structural property the Q-blocking (and
    /// its causal staircase) must preserve. This is the CPU stand-in for the
    /// CUDA staircase gate; the MI300X lane repeats it against the device
    /// kernel via the per-shape sweep in `rocm::tests`.
    #[test]
    fn reference_rows_are_independent_bitwise() {
        let (m, n_head, n_head_kv, head_dim, causal_offset) = (11, 8, 2, 64, 5);
        let (q, k, v) = random_case(m, n_head, n_head_kv, head_dim, causal_offset);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut batch = vec![0.0f32; m * n_head * head_dim];
        gqa_attention_prefill_ref(
            &q,
            &k,
            &v,
            &mut batch,
            n_head,
            n_head_kv,
            head_dim,
            scale,
            causal_offset,
            m,
        );
        for r in 0..m {
            let q_row = &q[r * n_head * head_dim..(r + 1) * n_head * head_dim];
            let mut single = vec![0.0f32; n_head * head_dim];
            gqa_attention_prefill_ref(
                q_row,
                &k,
                &v,
                &mut single,
                n_head,
                n_head_kv,
                head_dim,
                scale,
                causal_offset + r,
                1,
            );
            let batch_row = &batch[r * n_head * head_dim..(r + 1) * n_head * head_dim];
            for (i, (&b, &s)) in batch_row.iter().zip(&single).enumerate() {
                assert_eq!(
                    b.to_bits(),
                    s.to_bits(),
                    "row {r} elem {i}: batch {b} != single {s}"
                );
            }
        }
    }
}

//! Host-side structure for the v3 Q-blocked prefill attention port
//! ([`attention.metal`](../src/attention.metal)) — the MSL twin of
//! tritium-cuda's `gqa_attention_batch_v3_f32` (`kernels/decode.cu`).
//!
//! Everything in this module is plain Rust with **no Metal dependency**, so it
//! compiles — and its tests run — on every platform, including the cpu-only
//! Linux CI matrix. It holds:
//!
//! * the launch constants that MUST mirror the `#define`s in
//!   `attention.metal` (pinned by [`tests::attn_v3_consts_match_attention_metal_defines`]),
//!   which in turn mirror decode.cu's `ATTN_V3_*` family (pinned on the CUDA
//!   side by `attn_v2_consts_match_decode_cu_defines`);
//! * the dispatch geometry + validation the macOS backend uses (pure
//!   functions, unit-tested here);
//! * the `TRITIUM_ATTN_V3` kill-switch parser (the tritium-cuda env pattern:
//!   unset/`1` = on, `0` = off, anything else is a loud reject);
//! * [`gqa_attention_prefill_ref`], the **pinned-order host reference**: the
//!   same per-(row, head) summation orders as the kernel, in plain `f32` ops
//!   (Rust `f32` `+`/`*` are IEEE round-to-nearest with no implicit FMA —
//!   the host-side meaning of CUDA's `__fadd_rn`/`__fmul_rn`). Since the
//!   Metal backend has no rev-1/v2 device attention to fall back to, this
//!   reference is both the kill-switch fallback and the oracle the Mac-lane
//!   conformance gate compares the GPU kernel against.
//!
//! ## The one deliberate deviation from the CUDA twin contract
//!
//! CUDA v3 is gated `to_bits`-identical to rev-1 because both use
//! `exp_f32(x) = __double2float_rn(exp((double)x))` — f64 exp rounded once,
//! matching glibc `expf`. **Metal has no `double`**, so the MSL kernel uses
//! `metal::precise::exp` instead; every summation ORDER is still pinned
//! verbatim, but the exponential may differ from Rust's `f32::exp` by a few
//! ULP. The Mac-lane gate therefore asserts a tight per-element tolerance
//! (attributable solely to `exp`) rather than bit equality — see
//! `backend::tests`. Closing this to full bit parity would need a float-float
//! (double-single) `exp` in MSL; recorded as follow-up work, not improvised
//! here.

/// Threads per threadgroup — mirrors decode.cu `ATTN_V3_THREADS` (256 = 8
/// CUDA warps). On Metal the threadgroup splits into
/// `ATTN_V3_THREADS / ATTN_V3_SIMD_WIDTH` = 8 simdgroups of 32; the host
/// refuses the device path if the compiled pipeline's
/// `thread_execution_width` is not 32 (see the backend dispatch).
pub const ATTN_V3_THREADS: u32 = 256;
/// Query rows blocked per (threadgroup, head) — decode.cu `ATTN_V3_BQ`.
pub const ATTN_V3_BQ: usize = 8;
/// Keys staged per K/V chunk — decode.cu `ATTN_V3_KCH`.
pub const ATTN_V3_KCH: usize = 32;
/// Max `head_dim` the kernel's threadgroup staging is sized for — decode.cu
/// `ATTN_V2_HDMAX` (the v3 CUDA kernel reuses the v2 bound; the MSL source
/// names it `ATTN_V3_HDMAX` because Metal has no v2).
pub const ATTN_V3_HDMAX: usize = 128;
/// Lanes per simdgroup the kernel's warp mapping assumes (CUDA warp width).
pub const ATTN_V3_SIMD_WIDTH: usize = 32;
/// Simdgroups per threadgroup (CUDA `ATTN_V3_WARPS`).
pub const ATTN_V3_SIMDS: usize = ATTN_V3_THREADS as usize / ATTN_V3_SIMD_WIDTH;
/// Query rows owned per simdgroup in phases 1/2 (CUDA `ATTN_V3_RPW`).
pub const ATTN_V3_RPW: usize = ATTN_V3_BQ / ATTN_V3_SIMDS;

// The CUDA kernel's static_assert, host-side: RPW = 0 (more simdgroups than
// BQ rows) would compile phases 1/2 to nothing.
const _: () = assert!(ATTN_V3_BQ.is_multiple_of(ATTN_V3_SIMDS) && ATTN_V3_RPW >= 1);

/// Parse a `TRITIUM_ATTN_V3` value: unset / `"1"` → enabled (default),
/// `"0"` → disabled, anything else is a loud reject — the exact behaviour of
/// tritium-cuda's `attn_env` closure for the same variable.
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

/// Threadgroup grid for a v3 launch: `(n_head, ceil(m / ATTN_V3_BQ))` — the
/// exact CUDA grid (`grid_dim = (n_head, ceil(m/BQ), 1)`), with
/// [`ATTN_V3_THREADS`]`×1×1` threads per threadgroup.
#[must_use]
pub fn v3_threadgroups(m: usize, n_head: usize) -> (u64, u64) {
    (n_head as u64, m.div_ceil(ATTN_V3_BQ) as u64)
}

/// Element count of the global `[m, n_head, ctx_max]` scores scratch the v3
/// kernel requires (the lever that lifts v2's ctx cap). `None` on overflow.
#[must_use]
pub fn v3_scores_len(m: usize, n_head: usize, ctx_max: usize) -> Option<usize> {
    m.checked_mul(n_head)?.checked_mul(ctx_max)
}

/// Validate a prefill-attention call against the kernel's host contract.
/// Length arguments (not slices) keep it pure and trivially unit-testable.
///
/// Checks shape coherence (`n_head % n_head_kv`, `causal_offset + m <=
/// ctx_max`), buffer lengths (`q`/`out` exactly `m·n_head·head_dim`; `k`/`v`
/// arenas at least `ctx_top·n_head_kv·head_dim` rows), the kernel's `u32`
/// scalar range, and scores-scratch overflow. It does NOT enforce
/// `head_dim <= ATTN_V3_HDMAX` — that is a *dispatch* bound (too-large heads
/// fall back to the host reference, mirroring CUDA's v3 → baseline priority),
/// not an input error.
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
    // The kernel takes its scalars as u32 (`AttnV3Params`) and widens per-row
    // bases to 64 bits in-kernel (the v0.6.9 discipline); reject anything that
    // would truncate the u32 params rather than dispatching a wrapped value.
    for (name, val) in [
        ("ctx_max", ctx_max),
        ("n_head", n_head),
        ("n_head_kv", n_head_kv),
        ("head_dim", head_dim),
        ("causal_offset", causal_offset),
        ("m", m),
    ] {
        if val > u32::MAX as usize {
            return Err(format!("{name} {val} exceeds the kernel's u32 param range"));
        }
    }
    if v3_scores_len(m, n_head, ctx_max).is_none() {
        return Err("scores scratch m*n_head*ctx_max overflows".to_owned());
    }
    Ok(())
}

/// Pinned-order host reference for the v3 prefill attention — the CPU twin of
/// `gqa_attention_batch_v3_f32` (and, transitively, of the CUDA
/// rev-1/v2/v3 family, whose per-(row, head) orders it reproduces).
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
///    sequential sum runs on lane 0).
/// 3. **V fold** — per output dim `d`, a sequential j-ascending
///    `acc = acc + w*v` chain with the rev-1 `w == 0.0` skip (kernel
///    phase 3 folds chunk-ascending then within-chunk-ascending = the same
///    global j order).
///
/// The exponential is Rust [`f32::exp`] (glibc `expf`, correctly rounded) —
/// the target CUDA's `exp_f32` was built to match; the Metal kernel's
/// `precise::exp` may differ by a few ULP (see the module docs).
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
    /// (`attn_v2_consts_match_decode_cu_defines`), pointed at the MSL source:
    /// the Rust dispatch constants above must equal `attention.metal`'s
    /// `#define`s — a host value drifting LARGER than the kernel's shared
    /// sizing would dispatch out-of-bounds threadgroup indices that only a
    /// Mac could catch. Parsed from source; runs on every platform.
    /// (decode.cu's own `ATTN_V3_*`/`ATTN_V2_HDMAX` values are pinned by the
    /// CUDA-side test; a cross-crate `include_str!` would break packaging, so
    /// the CUDA<->MSL agreement is documented rather than parsed here.)
    #[test]
    fn attn_v3_consts_match_attention_metal_defines() {
        let src = include_str!("attention.metal");
        let get = |name: &str| -> usize {
            src.lines()
                .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
                .unwrap_or_else(|| panic!("#define {name} missing from attention.metal"))
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
    fn threadgroup_geometry_matches_cuda_grid() {
        assert_eq!(v3_threadgroups(1, 32), (32, 1));
        assert_eq!(v3_threadgroups(8, 32), (32, 1));
        assert_eq!(v3_threadgroups(9, 32), (32, 2));
        assert_eq!(v3_threadgroups(512, 24), (24, 64));
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

    /// Deterministic input generator (the xorshift pattern the backend tests
    /// already use — no external rng).
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
    /// at causal_offset + r — the structural property the Q-blocking (and its
    /// causal staircase) must preserve. This is the CPU stand-in for the CUDA
    /// staircase gate; the Mac lane repeats it against the device kernel.
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

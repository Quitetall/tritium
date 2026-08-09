//! RFC 0001 / ADR 0036 L3b **lever 6** acceptance + measurement: the fast
//! kernel tier on the BATCHED tree routes — the fused online-softmax twins
//! over the paged/slots `TreeCtrlAddr` axes — vs the exact tier (model + GPU
//! gated, run explicitly):
//!
//! ```text
//! cargo test -p tritium-nn --features cuda --release --test tier_fast_slots_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The gates (RFC 0001 as amended by Amendment 1, b9fb8d5, extended to the
//! routes lever 6 adds):
//!   * `cuda_l3b_slots_in_situ_kernel_drift` — Amendment 1's BINDING drift
//!     bar on the SLOTS route (fused_slots twins): ≤1e-4 max rel at the
//!     swapped kernel's output inside a real multi-slot verify, via the
//!     `TRITIUM_TREE_ATTN_DUMP` seam on the batch's own scratch
//!     (`tree_attn_dump_batch`). Layer 0 is the assertion point (its inputs
//!     are tier-invariant); deeper layers + verify logits are REPORTED.
//!   * `cuda_l3b_paged_in_situ_kernel_drift` — the same bar on the PAGED
//!     single-slot route (fused_ctrl_paged twins), non-identity page table.
//!   * `cuda_l3b_slots_tier_identity_and_abba` — the quality-triplet
//!     identity leg + RFC checklist items 4/5 on the multi-slot spec shape
//!     (`tree_verify_greedy_slots`, N ∈ {2, 4}, the batch_spec_kv_bench
//!     harness): per (N, rung, ctx), ABBA (fast, exact, exact, fast); every
//!     leg asserts committed == the leg's own plain-greedy stream (ADR 0014
//!     losslessness ON the fused forward), and across tiers the committed
//!     streams and verify counts must be identical (deterministic drafts ⇒
//!     τ exactly unchanged).
//!
//! The dense-route gates live in `tier_fast_bench.rs`; the drafter-model
//! bitwise slot gates (`acceptance.rs`) double as fast-tier checks when run
//! under `TRITIUM_KERNEL_TIER=fast` (graph legs ride the fused twins, eager
//! legs the exact pair — token equality there is greedy identity again).

#![cfg(feature = "cuda")]

use std::path::Path;

static GGUF_PATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let dir = std::env::var("TRITIUM_MODEL_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.cache/tritium-models",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    format!("{dir}/bitnet-2b4t-gguf/ggml-model-i2_s.gguf")
});
const REF_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/reference/bitnet_accept.json"
);

/// Spec committed-token target PER SLOT (timed portion of the ABBA legs).
const SPEC_TOKENS: usize = 50;
/// Room past the prefill: reference stream + spec commits + tree slack.
const TAIL_BUDGET: usize = 168;

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

/// Max relative error, the ADR-0004 convention (per `row_len` chunk, the
/// reference chunk's max magnitude floored at 1e-3 as denominator).
fn max_rel_err_rows(got: &[f32], want: &[f32], row_len: usize) -> f32 {
    let mut worst = 0.0f32;
    for (g, w) in got.chunks_exact(row_len).zip(want.chunks_exact(row_len)) {
        let scale = w.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-3);
        for (&a, &b) in g.iter().zip(w) {
            let rel = (a - b).abs() / scale;
            if rel > worst {
                worst = rel;
            }
        }
    }
    worst
}

/// Env guard for one leg's (tier, rung) pair; restores on drop.
struct EnvGuard {
    tier: Option<std::ffi::OsString>,
    kv: Option<std::ffi::OsString>,
}
impl EnvGuard {
    fn set(tier: &str, rung: &str) -> Self {
        let prev_tier = std::env::var_os("TRITIUM_KERNEL_TIER");
        let prev_kv = std::env::var_os("TRITIUM_KV");
        // SAFETY: these ignored benches run single-threaded (--test-threads=1
        // per the header command); no other thread touches the environment.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("TRITIUM_KERNEL_TIER", tier);
            std::env::set_var("TRITIUM_KV", rung);
        }
        Self {
            tier: prev_tier,
            kv: prev_kv,
        }
    }
}
impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: single-threaded bench (see `EnvGuard::set`).
        #[allow(unsafe_code)]
        unsafe {
            match self.tier.take() {
                Some(v) => std::env::set_var("TRITIUM_KERNEL_TIER", v),
                None => std::env::remove_var("TRITIUM_KERNEL_TIER"),
            }
            match self.kv.take() {
                Some(v) => std::env::set_var("TRITIUM_KV", v),
                None => std::env::remove_var("TRITIUM_KV"),
            }
        }
    }
}

/// Scoped `TRITIUM_TREE_ATTN_DUMP=1` (baked at scratch alloc — set before
/// every leg's model load); restores on drop.
struct DumpEnvGuard;
impl DumpEnvGuard {
    fn set() -> Self {
        // SAFETY: single-threaded bench (see `EnvGuard::set`).
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("TRITIUM_TREE_ATTN_DUMP", "1");
        }
        DumpEnvGuard
    }
}
impl Drop for DumpEnvGuard {
    fn drop(&mut self) {
        // SAFETY: single-threaded bench (see `EnvGuard::set`).
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("TRITIUM_TREE_ATTN_DUMP");
        }
    }
}

fn load_inputs() -> Option<(Vec<u8>, Vec<u32>)> {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model bench)", *GGUF_PATH);
        return None;
    }
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(REF_PATH).expect("read reference"))
            .expect("parse reference");
    let base: Vec<u32> = reference["token_ids"]
        .as_array()
        .expect("token_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");
    Some((bytes, base))
}

struct LegResult {
    tier: &'static str,
    ctx: usize,
    /// Aggregate committed tokens/s of the multi-slot spec phase.
    spec_tps: f64,
    verifies: usize,
    /// Per-slot committed streams (identity gate across tiers).
    committed: Vec<Vec<u32>>,
}

/// One multi-slot leg: fresh model under (tier, rung), single-seq prefill to
/// `target`, adopt into every slot of an N-row dense batch, then the
/// mock-draft multi-slot spec engine (the batch_spec_kv_bench shape: chain
/// root + 4 correct + 1 wrong per slot, m_total = 6·N).
fn run_slots_leg(
    tier: &'static str,
    rung: &'static str,
    n: usize,
    target: usize,
    bytes: &[u8],
    base: &[u32],
) -> Option<LegResult> {
    let _env = EnvGuard::set(tier, rung);
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == "cuda")?
        .init;
    let backend = match init() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: cuda backend failed to init ({e})");
            return None;
        }
    };
    let file = tritium_format::read_gguf(bytes).expect("parse gguf");
    let mut runner = tritium_nn::ModelRunner::load(&file, bytes, backend).expect("load model");
    let n_ctx = runner.config.n_ctx as usize;
    assert!(
        target + TAIL_BUDGET <= n_ctx,
        "leg target {target} + tail {TAIL_BUDGET} > n_ctx {n_ctx}"
    );
    let prompt: Vec<u32> = base.iter().cycle().take(target).copied().collect();
    let positions: Vec<usize> = (0..target).collect();

    // Single-seq prefill + the leg's greedy reference stream. Plain decode is
    // tier-untouched (the fused twins live on the tree routes only), so this
    // stream is tier-invariant — asserted across tiers by the caller.
    let logits = runner.forward(&prompt, &positions).expect("prefill");
    let mut want: Vec<u32> = vec![argmax(&logits) as u32];
    {
        let mut pos = target;
        while want.len() < SPEC_TOKENS + 12 {
            let logits = runner
                .forward(&[*want.last().expect("non-empty")], &[pos])
                .expect("stream step");
            want.push(argmax(&logits) as u32);
            pos += 1;
        }
    }

    let mut batch = runner.new_batch(n).expect("new_batch");
    for r in 0..n {
        runner
            .adopt_into_batch_row(&mut batch, r, target)
            .expect("adopt");
        batch.set_position(r, target).expect("pos");
    }
    let rm = runner
        .resident_cuda()
        .expect("resident model")
        .expect("cuda resident model present");

    let rows: Vec<usize> = (0..n).collect();
    let mut committed: Vec<Vec<u32>> = vec![vec![want[0]]; n];
    let mut verifies = 0usize;
    let make_tree = |c: usize, root: u32| -> (Vec<u32>, Vec<i32>) {
        let mut t = vec![root];
        t.extend(want[c..c + 4].iter().copied());
        t.push((want[c + 3] + 5) % 128_256);
        (t, (0..6i32).map(|i| i - 1).collect())
    };
    // Warm verify (captures the slots tree graph off the clock).
    {
        let trees: Vec<(Vec<u32>, Vec<i32>)> = (0..n)
            .map(|r| make_tree(committed[r].len(), *committed[r].last().expect("ne")))
            .collect();
        let refs: Vec<(&[u32], &[i32])> = trees
            .iter()
            .map(|(t, p)| (t.as_slice(), p.as_slice()))
            .collect();
        let outs = rm
            .tree_verify_greedy_slots(&mut batch, &rows, &refs)
            .expect("warm slots verify");
        for (r, out) in outs.into_iter().enumerate() {
            committed[r].extend(out);
        }
        verifies += 1;
    }
    let timed_start = committed[0].len();
    let t0 = std::time::Instant::now();
    while committed[0].len() - timed_start < SPEC_TOKENS {
        let trees: Vec<(Vec<u32>, Vec<i32>)> = (0..n)
            .map(|r| make_tree(committed[r].len(), *committed[r].last().expect("ne")))
            .collect();
        let refs: Vec<(&[u32], &[i32])> = trees
            .iter()
            .map(|(t, p)| (t.as_slice(), p.as_slice()))
            .collect();
        let outs = rm
            .tree_verify_greedy_slots(&mut batch, &rows, &refs)
            .expect("slots verify");
        for (r, out) in outs.into_iter().enumerate() {
            assert!(!out.is_empty(), "verify must commit >= 1 token");
            committed[r].extend(out);
        }
        verifies += 1;
    }
    let spec_dt = t0.elapsed();
    let spec_committed: usize = committed.iter().map(|c| c.len() - timed_start).sum();

    // ADR 0014 losslessness ON this leg's own forward (under fast this is the
    // RFC greedy-identity gate on the fused slots kernel).
    for (r, c) in committed.iter().enumerate() {
        let m = c.len().min(want.len());
        assert_eq!(
            &c[..m],
            &want[..m],
            "[{tier}/{rung} N={n}] slot {r} spec committed diverged from plain greedy"
        );
    }
    drop(batch);

    Some(LegResult {
        tier,
        ctx: target,
        spec_tps: spec_committed as f64 / spec_dt.as_secs_f64(),
        verifies,
        committed,
    })
}

/// RFC quality-triplet identity + ABBA on the multi-slot spec shape.
#[test]
#[ignore = "lever-6 slots ABBA + identity: run explicitly with --ignored --nocapture --test-threads=1"]
fn cuda_l3b_slots_tier_identity_and_abba() {
    let Some((bytes, base)) = load_inputs() else {
        return;
    };
    // Cell selection for CONTENDED boxes (GPU shared with a training run +
    // sibling bench sessions): TRITIUM_L6_N / TRITIUM_L6_CTX (short|long) /
    // TRITIUM_L6_RUNG each optionally pin one axis so a single invocation
    // runs one ABBA cell; unset = the full matrix.
    let want_n: Option<usize> = std::env::var("TRITIUM_L6_N")
        .ok()
        .map(|v| v.parse().expect("TRITIUM_L6_N"));
    let want_ctx = std::env::var("TRITIUM_L6_CTX").ok();
    let want_rung = std::env::var("TRITIUM_L6_RUNG").ok();
    // n_ctx asserted inside run_slots_leg; 4096 is the 2B4T window.
    for n in [2usize, 4] {
        if want_n.is_some_and(|w| w != n) {
            continue;
        }
        for &target in &[512usize, 4096 - TAIL_BUDGET] {
            let ctx_name = if target == 512 { "short" } else { "long" };
            if want_ctx.as_deref().is_some_and(|w| w != ctx_name) {
                continue;
            }
            for rung in ["f32", "f16"] {
                if want_rung.as_deref().is_some_and(|w| w != rung) {
                    continue;
                }
                let mut results: Vec<LegResult> = Vec::new();
                for tier in ["fast", "exact", "exact", "fast"] {
                    let Some(r) = run_slots_leg(tier, rung, n, target, &bytes, &base) else {
                        return;
                    };
                    println!(
                        "[leg {tier}/{rung} N={n} ctx≈{}] multi-slot spec {:.1} agg tok/s \
                         ({} verifies)",
                        r.ctx, r.spec_tps, r.verifies
                    );
                    results.push(r);
                }
                // Cross-tier identity (greedy 256-horizon equivalent on the
                // slots shape): committed streams and verify counts must
                // match token-for-token — with deterministic drafts this pins
                // τ exactly.
                let (f0, e0) = (&results[0], &results[1]);
                assert_eq!(
                    f0.committed, e0.committed,
                    "[{rung} N={n} ctx≈{target}] fast-tier committed streams diverged from exact"
                );
                assert_eq!(
                    f0.verifies, e0.verifies,
                    "[{rung} N={n} ctx≈{target}] verify count changed — τ moved"
                );
                let mean = |tier: &str| -> f64 {
                    let v: Vec<f64> = results
                        .iter()
                        .filter(|r| r.tier == tier)
                        .map(|r| r.spec_tps)
                        .collect();
                    v.iter().sum::<f64>() / v.len() as f64
                };
                let (sf, se) = (mean("fast"), mean("exact"));
                println!(
                    "L3b slots ABBA [{rung} N={n} ctx≈{}] multi-slot spec: fast {sf:.1} vs \
                     exact {se:.1} agg tok/s ({:+.1}%)",
                    results[0].ctx,
                    (sf / se - 1.0) * 100.0,
                );
            }
        }
    }
}

/// One in-situ probe leg on the SLOTS route: fresh model under (tier, rung)
/// with the dump seam on, prefill, adopt into N=2 slots, ONE fixed slots
/// verify (trees from the reference stream — tier-invariant), read the
/// batch scratch's per-layer attention dump.
fn run_slots_dump_leg(
    tier: &'static str,
    rung: &'static str,
    target: usize,
    bytes: &[u8],
    base: &[u32],
) -> Option<(Vec<f32>, usize, usize)> {
    const N: usize = 2;
    let _env = EnvGuard::set(tier, rung);
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == "cuda")?
        .init;
    let backend = match init() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: cuda backend failed to init ({e})");
            return None;
        }
    };
    let file = tritium_format::read_gguf(bytes).expect("parse gguf");
    let mut runner = tritium_nn::ModelRunner::load(&file, bytes, backend).expect("load model");
    let prompt: Vec<u32> = base.iter().cycle().take(target).copied().collect();
    let positions: Vec<usize> = (0..target).collect();
    runner.forward(&prompt, &positions).expect("probe prefill");
    let mut batch = runner.new_batch(N).expect("new_batch");
    for r in 0..N {
        runner
            .adopt_into_batch_row(&mut batch, r, target)
            .expect("adopt");
        batch.set_position(r, target).expect("pos");
    }
    let rm = runner
        .resident_cuda()
        .expect("resident model")
        .expect("cuda resident model present");
    let rows: Vec<usize> = (0..N).collect();
    // Distinct fixed chains per slot (6 nodes each, m_total = 12).
    let trees: Vec<(Vec<u32>, Vec<i32>)> = (0..N)
        .map(|r| {
            let t: Vec<u32> = base
                .iter()
                .cycle()
                .skip(target + 8 * r)
                .take(6)
                .copied()
                .collect();
            (t, (0..6i32).map(|i| i - 1).collect())
        })
        .collect();
    let refs: Vec<(&[u32], &[i32])> = trees
        .iter()
        .map(|(t, p)| (t.as_slice(), p.as_slice()))
        .collect();
    rm.tree_verify_greedy_slots(&mut batch, &rows, &refs)
        .expect("probe slots verify");
    let dump = rm
        .tree_attn_dump_batch(&batch)
        .expect("attn dump read")
        .expect("TRITIUM_TREE_ATTN_DUMP was set at build — batch dump must exist");
    drop(batch);
    Some(dump)
}

/// Amendment 1's binding in-situ drift bar on the SLOTS route (N=2,
/// m_total=12): layer 0 asserted ≤1e-4 (tier-invariant inputs), deeper
/// layers reported.
#[test]
#[ignore = "lever-6 slots in-situ gate: run explicitly with --ignored --nocapture --test-threads=1"]
fn cuda_l3b_slots_in_situ_kernel_drift() {
    let Some((bytes, base)) = load_inputs() else {
        return;
    };
    let _dump_env = DumpEnvGuard::set();
    const M_TOTAL: usize = 12;
    // TRITIUM_L6_CTX / TRITIUM_L6_RUNG pin one cell (contended-box driver;
    // see cuda_l3b_slots_tier_identity_and_abba).
    let want_ctx = std::env::var("TRITIUM_L6_CTX").ok();
    let want_rung = std::env::var("TRITIUM_L6_RUNG").ok();
    for &target in &[512usize, 4096 - TAIL_BUDGET] {
        let ctx_name = if target == 512 { "short" } else { "long" };
        if want_ctx.as_deref().is_some_and(|w| w != ctx_name) {
            continue;
        }
        for rung in ["f32", "f16"] {
            if want_rung.as_deref().is_some_and(|w| w != rung) {
                continue;
            }
            let Some((ed, es, eq)) = run_slots_dump_leg("exact", rung, target, &bytes, &base)
            else {
                return;
            };
            let Some((fd, fs, fq)) = run_slots_dump_leg("fast", rung, target, &bytes, &base) else {
                return;
            };
            assert_eq!((es, eq, ed.len()), (fs, fq, fd.len()));
            assert!(
                ed[..M_TOTAL * eq].iter().any(|&x| x != 0.0),
                "[{rung} ctx≈{target}] slots dump layer 0 is all zeros — the graph \
                 route (and its dump nodes) did not run"
            );
            let n_layers = ed.len() / es;
            let layer = |d: &[f32], li: usize| d[li * es..li * es + M_TOTAL * eq].to_vec();
            let mut in_situ = 0.0f32;
            let mut deep_worst = (0usize, 0.0f32);
            for li in 0..n_layers {
                let d = max_rel_err_rows(&layer(&fd, li), &layer(&ed, li), eq);
                if li == 0 {
                    in_situ = d;
                } else if d > deep_worst.1 {
                    deep_worst = (li, d);
                }
            }
            println!(
                "[slots {rung} ctx≈{target}] IN-SITU kernel-output drift (layer 0) = \
                 {in_situ:.3e} (bar 1e-4) | deeper layers reported: worst {:.3e} at \
                 layer {} (i8-floor composition, not the kernel)",
                deep_worst.1, deep_worst.0,
            );
            assert!(
                in_situ <= 1e-4,
                "[slots {rung} ctx≈{target}] RFC 0001 Amendment 1 in-situ drift gate \
                 failed: {in_situ:.3e} > 1e-4 at the fused slots kernel's output"
            );
        }
    }
}

/// One in-situ probe leg on the PAGED single-slot route: N=1 paged batch,
/// interleaved-page reservation is unnecessary here (one slot), but the
/// mapping still exercises the table policy (fused_ctrl_paged) end to end.
fn run_paged_dump_leg(
    tier: &'static str,
    rung: &'static str,
    target: usize,
    bytes: &[u8],
    base: &[u32],
) -> Option<(Vec<f32>, usize, usize)> {
    const M: usize = 8;
    let _env = EnvGuard::set(tier, rung);
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == "cuda")?
        .init;
    let backend = match init() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: cuda backend failed to init ({e})");
            return None;
        }
    };
    let file = tritium_format::read_gguf(bytes).expect("parse gguf");
    let mut runner = tritium_nn::ModelRunner::load(&file, bytes, backend).expect("load model");
    let prompt: Vec<u32> = base.iter().cycle().take(target).copied().collect();
    let positions: Vec<usize> = (0..target).collect();
    runner.forward(&prompt, &positions).expect("probe prefill");
    // Pool with slack; reserve prefix + padded tree rows for slot 0.
    let pool_pages = (target + 96).div_ceil(tritium_cuda::KV_PAGE_TOKENS) + 2;
    let mut batch = runner
        .new_batch_paged(1, pool_pages)
        .expect("new_batch_paged");
    batch
        .reserve_pages(0, target + 96)
        .expect("reserve slot pages");
    runner
        .adopt_into_batch_row(&mut batch, 0, target)
        .expect("adopt");
    batch.set_position(0, target).expect("pos");
    let tree: Vec<u32> = base.iter().cycle().skip(target).take(M).copied().collect();
    let parents: Vec<i32> = (0..M as i32).map(|i| i - 1).collect();
    runner
        .tree_verify_greedy_slot(&mut batch, 0, &tree, &parents)
        .expect("paged slot verify");
    let rm = runner
        .resident_cuda()
        .expect("resident model")
        .expect("cuda resident model present");
    let dump = rm
        .tree_attn_dump_batch(&batch)
        .expect("attn dump read")
        .expect("TRITIUM_TREE_ATTN_DUMP was set at build — batch dump must exist");
    drop(batch);
    Some(dump)
}

/// Amendment 1's binding in-situ drift bar on the PAGED route (single slot,
/// 8-node chain through the page table).
#[test]
#[ignore = "lever-6 paged in-situ gate: run explicitly with --ignored --nocapture --test-threads=1"]
fn cuda_l3b_paged_in_situ_kernel_drift() {
    let Some((bytes, base)) = load_inputs() else {
        return;
    };
    let _dump_env = DumpEnvGuard::set();
    const M: usize = 8;
    for &target in &[512usize, 4096 - TAIL_BUDGET] {
        for rung in ["f32", "f16"] {
            let Some((ed, es, eq)) = run_paged_dump_leg("exact", rung, target, &bytes, &base)
            else {
                return;
            };
            let Some((fd, fs, fq)) = run_paged_dump_leg("fast", rung, target, &bytes, &base) else {
                return;
            };
            assert_eq!((es, eq, ed.len()), (fs, fq, fd.len()));
            assert!(
                ed[..M * eq].iter().any(|&x| x != 0.0),
                "[{rung} ctx≈{target}] paged dump layer 0 is all zeros — the graph \
                 route (and its dump nodes) did not run"
            );
            let n_layers = ed.len() / es;
            let layer = |d: &[f32], li: usize| d[li * es..li * es + M * eq].to_vec();
            let mut in_situ = 0.0f32;
            let mut deep_worst = (0usize, 0.0f32);
            for li in 0..n_layers {
                let d = max_rel_err_rows(&layer(&fd, li), &layer(&ed, li), eq);
                if li == 0 {
                    in_situ = d;
                } else if d > deep_worst.1 {
                    deep_worst = (li, d);
                }
            }
            println!(
                "[paged {rung} ctx≈{target}] IN-SITU kernel-output drift (layer 0) = \
                 {in_situ:.3e} (bar 1e-4) | deeper layers reported: worst {:.3e} at \
                 layer {} (i8-floor composition, not the kernel)",
                deep_worst.1, deep_worst.0,
            );
            assert!(
                in_situ <= 1e-4,
                "[paged {rung} ctx≈{target}] RFC 0001 Amendment 1 in-situ drift gate \
                 failed: {in_situ:.3e} > 1e-4 at the fused paged kernel's output"
            );
        }
    }
}

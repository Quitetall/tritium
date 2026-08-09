//! RFC 0001 / ADR 0036 L3b acceptance + measurement: the fast kernel tier
//! (`TRITIUM_KERNEL_TIER=fast`, first member = the fused tree attention) vs
//! the exact tier, per KV rung and context regime (model + GPU gated, run
//! explicitly):
//!
//! ```text
//! cargo test -p tritium-nn --features cuda --release --test tier_fast_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The gates (RFC 0001 as amended by Amendment 1, b9fb8d5):
//!   * `cuda_l3b_in_situ_kernel_drift` — Amendment 1's BINDING drift bar:
//!     ≤1e-4 max rel at the swapped kernel's OUTPUT inside a real forward.
//!     Via the `TRITIUM_TREE_ATTN_DUMP` seam, the layer-0 tree-attention
//!     output (fast vs exact at bit-identical inputs — prefill and the
//!     pre-attention trunk are tier-invariant) is asserted; deeper layers'
//!     drift and the final verify-logits drift are REPORTED (their inputs
//!     diverge through the i8 activation lattice — the amendment's ~1e-2
//!     structural floor, which measures composition, not the kernel).
//!   * `cuda_l3b_tier_drift_and_identity` — the quality-triplet identity leg
//!     plus the Amendment 1 reporting duty: per (rung, ctx-regime), the fast
//!     tier's spec-committed stream is token-identical to the exact tier's
//!     over a >=256-token horizon (committed == plain-greedy losslessness is
//!     also asserted inside every leg, which pins the ADR 0014 acceptance
//!     walk), and the final verify-logits drift is measured and PRINTED —
//!     not gated: Amendment 1 retired the 2e-3 final-logits bar (the i8
//!     re-round floor; see `cuda_l3b_drift_diagnostic`) but requires the
//!     number stay reported in adoption records. Deterministic drafts and
//!     greedy identity together imply tau is EXACTLY unchanged: same trees
//!     in, same accepted paths out.
//!   * `cuda_l3b_spec_abba` — RFC checklist items 4 and 5: ABBA
//!     (fast, exact, exact, fast) spec e2e + ms/verify per (rung, ctx),
//!     mock-draft engine (fixed acceptance profile, drafting cost excluded —
//!     the spec_kv_bench rig). Plain decode is printed as a control: the
//!     fast tier does not touch the plain path, so it must be ~unchanged.
//!
//! ABBA order-alternation; each leg loads a fresh model under its
//! (`TRITIUM_KERNEL_TIER`, `TRITIUM_KV`) pair. Long-ctx legs run BOTH rungs;
//! f16 owns that regime (L6), so the long-ctx claim is measured ON f16 per
//! the RFC's composition clause.

#![cfg(feature = "cuda")]

use std::path::Path;

/// Model cache root: override via `TRITIUM_MODEL_DIR`; default
/// `~/.cache/tritium-models`; tests skip cleanly when absent.
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

/// Committed-token horizon for the identity gate (RFC: 256-token greedy
/// identity) and the timed portion of the ABBA legs.
const SPEC_TOKENS: usize = 256;
/// Plain reference stream length: enough for the identity horizon's drafts.
const PLAIN_STREAM: usize = SPEC_TOKENS + 16;
/// Untimed plain steps for the ABBA legs' control number.
const PLAIN_STEPS: usize = 64;
/// Room past the prefill: the plain stream + tree pad slack.
const TAIL_BUDGET: usize = PLAIN_STREAM + 48;

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

/// Max relative error, the ADR-0004 convention (`acceptance.rs`
/// `max_rel_err`, the metric behind the `cpu_cuda_parity` 2e-3 bar): the
/// reference vector's max magnitude (floored) is the denominator. Applied
/// per logits row (`row_len`-sized chunks) like the per-step parity gate.
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

struct LegResult {
    tier: &'static str,
    ctx: usize,
    plain_tps: f64,
    spec_tps: f64,
    ms_per_verify: f64,
    verifies: usize,
    graph_buckets: usize,
    /// Committed stream (identity gate across tiers).
    committed: Vec<u32>,
    /// One fixed-tree verify's logits right after prefill (drift gate).
    probe_logits: Vec<f32>,
}

/// One leg: fresh model under (tier, rung), prefill to `target`, plain
/// stream, a fixed-tree logits probe, then the mock-draft spec engine.
fn run_leg(
    tier: &'static str,
    rung: &'static str,
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
    let positions: Vec<usize> = (0..prompt.len()).collect();

    // ── Plain leg (control + reference stream) ──
    let logits = runner.forward(&prompt, &positions).expect("prefill");
    let mut next = argmax(&logits) as u32;
    let mut pos = prompt.len();
    let logits = runner.forward(&[next], &[pos]).expect("warm step");
    next = argmax(&logits) as u32;
    pos += 1;
    let mut stream: Vec<u32> = vec![next];
    let t0 = std::time::Instant::now();
    for _ in 0..PLAIN_STEPS {
        let logits = runner.forward(&[next], &[pos]).expect("decode");
        next = argmax(&logits) as u32;
        pos += 1;
        stream.push(next);
    }
    let plain_dt = t0.elapsed();
    while stream.len() < PLAIN_STREAM {
        let logits = runner.forward(&[next], &[pos]).expect("stream tail");
        next = argmax(&logits) as u32;
        pos += 1;
        stream.push(next);
    }

    // ── Logits probe: fresh state, one FIXED tree, full verify logits ──
    // (the tier drift gate compares these across tiers at identical state;
    // the tree is drawn from the plain stream, which is tier-invariant).
    runner.reset();
    let l0 = runner.forward(&prompt, &positions).expect("probe prefill");
    let first = argmax(&l0) as u32;
    let probe_tree: Vec<u32> = {
        let mut t = vec![first];
        t.extend(stream[0..6].iter().copied());
        t.push((stream[5] + 5) % 128_256);
        t
    };
    let probe_parents: Vec<i32> = (0..8i32).map(|i| i - 1).collect();
    let probe_logits = runner
        .tree_verify_logits(&probe_tree, &probe_parents)
        .expect("probe verify logits");
    // Abandon the pending tree (no commit): the spec run below re-prefills.

    // ── Spec leg: fresh state, mock chain drafts from the plain stream ──
    runner.reset();
    let l0 = runner.forward(&prompt, &positions).expect("spec prefill");
    let first = argmax(&l0) as u32;
    let rm = runner
        .resident_cuda()
        .expect("resident model")
        .expect("cuda resident model present");
    let want = &stream;
    let mut committed: Vec<u32> = vec![first];
    let mut verifies = 0usize;
    let make_tree = |c: usize, root: u32| -> (Vec<u32>, Vec<i32>) {
        let mut t = vec![root];
        t.extend(want[c..c + 6].iter().copied());
        t.push((want[c + 5] + 5) % 128_256);
        (t, (0..8i32).map(|i| i - 1).collect())
    };
    {
        let (t, p) = make_tree(0, first);
        let out = rm.tree_verify_greedy(&t, &p).expect("warm verify");
        committed.extend(&out);
        verifies += 1;
    }
    let timed_start = committed.len();
    let t0 = std::time::Instant::now();
    while committed.len() - timed_start < SPEC_TOKENS {
        let c = committed.len() - 1;
        let (t, p) = make_tree(c, *committed.last().expect("non-empty"));
        let out = rm.tree_verify_greedy(&t, &p).expect("spec verify");
        assert!(!out.is_empty(), "verify must commit >= 1 token");
        committed.extend(&out);
        verifies += 1;
    }
    let spec_dt = t0.elapsed();
    let spec_committed = committed.len() - timed_start;
    let graph_buckets = rm.tree_graph_bucket_count();

    // Losslessness within the leg: committed == the leg's own plain greedy.
    // Under the FAST tier this is the RFC greedy-identity + ADR 0014
    // acceptance gate on the fused kernel's own forward.
    let n = (committed.len() - 1).min(want.len());
    assert_eq!(
        &committed[1..1 + n],
        &want[..n],
        "[{tier}/{rung}] spec committed stream diverged from plain greedy at ctx≈{target}"
    );

    Some(LegResult {
        tier,
        ctx: target,
        plain_tps: PLAIN_STEPS as f64 / plain_dt.as_secs_f64(),
        spec_tps: spec_committed as f64 / spec_dt.as_secs_f64(),
        ms_per_verify: spec_dt.as_secs_f64() * 1e3 / (verifies - 1).max(1) as f64,
        verifies,
        graph_buckets,
        committed,
        probe_logits,
    })
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

/// Short and long context targets (long sized like spec_kv_bench: near the
/// 4K window minus the tail budget).
fn ctx_targets(n_ctx_hint: usize) -> [usize; 2] {
    [512, n_ctx_hint - TAIL_BUDGET]
}

#[test]
#[ignore = "L3b RFC gates: run explicitly with --ignored --nocapture --test-threads=1"]
fn cuda_l3b_tier_drift_and_identity() {
    let Some((bytes, base)) = load_inputs() else {
        return;
    };
    // n_ctx is asserted inside run_leg; 4096 is the 2B4T window.
    for &target in &ctx_targets(4096) {
        for rung in ["f32", "f16"] {
            let Some(exact) = run_leg("exact", rung, target, &bytes, &base) else {
                return;
            };
            let Some(fast) = run_leg("fast", rung, target, &bytes, &base) else {
                return;
            };
            assert!(
                fast.graph_buckets >= 1,
                "[fast/{rung}] verify ran eager — the fast tier lives on the graph \
                 tree route; nothing was measured"
            );
            // Final verify-logits drift (8 nodes × vocab at identical state,
            // fast vs exact) — REPORTED, NOT GATED: RFC 0001 Amendment 1
            // retired the 2e-3 final-logits bar (the i8 activation lattice
            // re-rounds under ANY sub-lattice perturbation — a ~1e-2
            // structural floor independent of kernel accuracy; see
            // `cuda_l3b_drift_diagnostic`) but requires the number stay
            // reported in every adoption record. The BINDING drift bar is
            // `cuda_l3b_in_situ_kernel_drift`; the binding e2e legs are the
            // quality triplet (ppl ratio, the identity asserts below, τ).
            assert_eq!(exact.probe_logits.len(), fast.probe_logits.len());
            let row = exact.probe_logits.len() / 8; // 8 tree nodes x vocab
            let drift = max_rel_err_rows(&fast.probe_logits, &exact.probe_logits, row);
            // RFC item 3 — greedy identity over the 256-token horizon: the
            // committed streams must be token-identical across tiers. With
            // identical (deterministic) drafts this also pins τ exactly.
            assert_eq!(
                exact.committed, fast.committed,
                "[{rung} ctx≈{target}] fast-tier committed stream diverged from exact"
            );
            assert_eq!(
                exact.verifies, fast.verifies,
                "[{rung} ctx≈{target}] verify count changed — τ moved"
            );
            println!(
                "[{rung} ctx≈{target}] final-logits drift(verify logits, {} vals) = \
                 {drift:.3e} (REPORTED — bar retired by RFC 0001 Amendment 1; \
                 expected ~the 1e-2 i8 re-round floor) | committed identity {}/{} \
                 tokens | verifies {}",
                fast.probe_logits.len(),
                fast.committed.len(),
                exact.committed.len(),
                fast.verifies,
            );
        }
    }
}

/// Scoped `TRITIUM_TREE_ATTN_DUMP=1` (the in-situ drift seam is baked at
/// model build, so it must be set before every leg's load); restores on drop
/// so later legs/tests don't pay the dump buffer + copy nodes.
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

/// One in-situ probe leg: fresh model under (tier, rung) with the attention
/// dump enabled, prefill to `target`, one fixed 8-node chain verify (tokens
/// drawn deterministically from the reference stream — tier-invariant by
/// construction), then read back the per-layer attention outputs + the
/// verify logits. Returns `(dump, layer_stride, q_width, logits)`.
fn run_dump_leg(
    tier: &'static str,
    rung: &'static str,
    target: usize,
    bytes: &[u8],
    base: &[u32],
) -> Option<(Vec<f32>, usize, usize, Vec<f32>)> {
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
        target + 16 <= n_ctx,
        "probe target {target} + tree > n_ctx {n_ctx}"
    );
    let prompt: Vec<u32> = base.iter().cycle().take(target).copied().collect();
    let positions: Vec<usize> = (0..prompt.len()).collect();
    runner.forward(&prompt, &positions).expect("probe prefill");
    let tree: Vec<u32> = base.iter().cycle().skip(target).take(8).copied().collect();
    let parents: Vec<i32> = (0..8i32).map(|i| i - 1).collect();
    let logits = runner
        .tree_verify_logits(&tree, &parents)
        .expect("probe verify logits");
    let rm = runner
        .resident_cuda()
        .expect("resident model")
        .expect("cuda resident model present");
    assert!(
        rm.tree_graph_bucket_count() >= 1,
        "[{tier}/{rung}] probe verify ran eager — the dump seam (and the fast \
         tier's fused kernel) live on the graph route; nothing was measured"
    );
    let (dump, stride, q_width) = rm
        .tree_attn_dump()
        .expect("attn dump read")
        .expect("TRITIUM_TREE_ATTN_DUMP was set at build — dump must exist");
    Some((dump, stride, q_width, logits))
}

/// RFC 0001 Amendment 1's BINDING drift bar: ≤1e-4 max rel at the swapped
/// kernel's OUTPUT inside a real forward. Both tiers run the identical
/// (prompt, tree) verify on the graph route with the per-layer attention
/// dump enabled; layer 0 is the assertion point because its kernel inputs
/// are bit-identical across tiers (prefill and the pre-attention trunk —
/// embed, rmsnorm+quant, q/k/v matmuls, RoPE, kv-append — are tier-invariant
/// and deterministic), so the layer-0 diff is purely the fused kernel in
/// situ. Deeper layers are REPORTED, not gated: their inputs diverge through
/// the i8 activation lattice (the amendment's ~1e-2 structural floor), so a
/// cross-tier diff there measures composition, not the kernel. The final
/// verify-logits drift is also printed (the amendment's reporting duty).
#[test]
#[ignore = "L3b Amendment 1 in-situ gate: run explicitly with --ignored --nocapture --test-threads=1"]
fn cuda_l3b_in_situ_kernel_drift() {
    let Some((bytes, base)) = load_inputs() else {
        return;
    };
    let _dump_env = DumpEnvGuard::set();
    const M: usize = 8; // probe tree nodes (chain)
    for &target in &ctx_targets(4096) {
        for rung in ["f32", "f16"] {
            let Some((ed, es, eq, el)) = run_dump_leg("exact", rung, target, &bytes, &base) else {
                return;
            };
            let Some((fd, fs, fq, fl)) = run_dump_leg("fast", rung, target, &bytes, &base) else {
                return;
            };
            assert_eq!((es, eq, ed.len()), (fs, fq, fd.len()));
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
            let row = el.len() / M;
            let logit_drift = max_rel_err_rows(&fl, &el, row);
            println!(
                "[{rung} ctx≈{target}] IN-SITU kernel-output drift (layer 0, \
                 fast vs exact at identical inputs) = {in_situ:.3e} (bar 1e-4) | \
                 deeper layers reported: worst {:.3e} at layer {} (i8-floor \
                 composition, not the kernel) | final-logits drift {logit_drift:.3e} \
                 (reported)",
                deep_worst.1, deep_worst.0,
            );
            assert!(
                in_situ <= 1e-4,
                "[{rung} ctx≈{target}] RFC 0001 Amendment 1 in-situ drift gate \
                 failed: {in_situ:.3e} > 1e-4 at the fused kernel's output (layer 0, \
                 bit-identical inputs)"
            );
        }
    }
}

/// Teacher-forced perplexity THROUGH THE TREE-VERIFY FORWARD (the path the
/// fast tier claims): windows are fed as 8-node chain trees via
/// `tree_verify_logits` + full-path `tree_commit`, so every scored position's
/// logits come out of the (fast or exact) tree attention. The stock WT-103
/// harness (`lm_head_ppl.rs`) scores through the plain prefill path, which
/// the tree tier never touches — running it would be vacuous for L3b; this
/// variant makes the RFC's ppl(fast)/ppl(exact) <= 1.001 bar binding.
fn verify_path_ppl(
    tier: &'static str,
    rung: &'static str,
    bytes: &[u8],
    ids: &[u32],
) -> Option<f64> {
    const SEQ_LEN: usize = 103;
    const CHAIN: usize = 8;
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
    let parents: Vec<i32> = (0..CHAIN as i32).map(|i| i - 1).collect();
    let path: Vec<usize> = (0..CHAIN).collect();
    let (mut nll, mut count) = (0.0f64, 0usize);
    for w in ids.chunks_exact(SEQ_LEN) {
        runner.reset();
        let seed = runner.forward(&[w[0]], &[0]).expect("seed");
        let vocab = seed.len();
        let mut fed = 1usize; // cache holds w[..fed]
        while fed + CHAIN < SEQ_LEN {
            let tree: Vec<u32> = w[fed..fed + CHAIN].to_vec();
            let logits = runner
                .tree_verify_logits(&tree, &parents)
                .expect("verify logits");
            for (j, row) in logits.chunks_exact(vocab).enumerate() {
                let target = w[fed + j + 1] as usize;
                // log softmax at target, f64 accumulation (max-shifted).
                let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse: f64 = row.iter().map(|&x| ((x as f64) - mx).exp()).sum();
                nll -= (row[target] as f64) - mx - lse.ln();
                count += 1;
            }
            runner.tree_commit(&path).expect("commit chain");
            fed += CHAIN;
        }
    }
    let ppl = (nll / count as f64).exp();
    eprintln!("[{tier}/{rung}] verify-path ppl {ppl:.6} over {count} positions");
    Some(ppl)
}

/// Diagnostic (not a gate): where does the fast tier's verify-logit drift
/// come from? Prints per-node drift (ADR-0004 convention), the count of
/// positions past 2e-3, and an exact-vs-exact control (must be 0 — the path
/// is deterministic; a nonzero control means the probe itself is broken).
#[test]
#[ignore = "L3b drift diagnostic: run explicitly with --ignored --nocapture --test-threads=1"]
fn cuda_l3b_drift_diagnostic() {
    let Some((bytes, base)) = load_inputs() else {
        return;
    };
    for &target in &[512usize] {
        let Some(e1) = run_leg("exact", "f32", target, &bytes, &base) else {
            return;
        };
        let Some(e2) = run_leg("exact", "f32", target, &bytes, &base) else {
            return;
        };
        let Some(f1) = run_leg("fast", "f32", target, &bytes, &base) else {
            return;
        };
        let row = e1.probe_logits.len() / 8;
        let ctrl = max_rel_err_rows(&e2.probe_logits, &e1.probe_logits, row);
        println!("[control exact-vs-exact ctx≈{target}] drift = {ctrl:.3e}");
        for node in 0..8 {
            let a = &f1.probe_logits[node * row..(node + 1) * row];
            let b = &e1.probe_logits[node * row..(node + 1) * row];
            let scale = b.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-3);
            let mut worst = 0.0f32;
            let mut over = 0usize;
            let mut argmax_same = true;
            for (&x, &y) in a.iter().zip(b) {
                let rel = (x - y).abs() / scale;
                if rel > worst {
                    worst = rel;
                }
                if rel > 2e-3 {
                    over += 1;
                }
            }
            if argmax(a) != argmax(b) {
                argmax_same = false;
            }
            println!(
                "  node {node}: worst {worst:.3e} | {over}/{row} positions > 2e-3 | \
                 argmax same: {argmax_same}"
            );
        }
    }
}

#[test]
#[ignore = "L3b verify-path ppl gate: needs the 2B4T bundle + WT-103 corpus"]
fn cuda_l3b_verify_path_ppl_ratio() {
    let Some((bytes, _)) = load_inputs() else {
        return;
    };
    let corpus = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        format!(
            "{}/blut/data/corpus_wt103.jsonl",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let Ok(raw) = std::fs::read_to_string(&corpus) else {
        eprintln!("skipping: corpus absent at {corpus}");
        return;
    };
    // 50 windows x 103 tokens — the same budget as the stock 5100-position
    // harness (scored positions here: 12 chains x 8 per window = 4800).
    let mut ids: Vec<u32> = Vec::new();
    for line in raw.lines().take(50) {
        let v: serde_json::Value = serde_json::from_str(line).expect("corpus line");
        let toks: Vec<u32> = v["tokens"]
            .as_array()
            .expect("tokens")
            .iter()
            .map(|t| t.as_u64().expect("id") as u32)
            .collect();
        assert!(toks.len() >= 103, "corpus window shorter than 103");
        ids.extend(&toks[..103]);
    }
    for rung in ["f32", "f16"] {
        let Some(pe) = verify_path_ppl("exact", rung, &bytes, &ids) else {
            return;
        };
        let Some(pf) = verify_path_ppl("fast", rung, &bytes, &ids) else {
            return;
        };
        let ratio = pf / pe;
        println!(
            "[{rung}] verify-path ppl fast/exact = {ratio:.6} ({:+.4}%; bar <= 1.001)",
            (ratio - 1.0) * 100.0
        );
        assert!(
            ratio <= 1.001,
            "[{rung}] fast-tier verify-path ppl ratio {ratio:.6} exceeds 1.001"
        );
    }
}

#[test]
#[ignore = "L3b ABBA bench: run explicitly with --ignored --nocapture --test-threads=1"]
fn cuda_l3b_spec_abba() {
    let Some((bytes, base)) = load_inputs() else {
        return;
    };
    for &target in &ctx_targets(4096) {
        for rung in ["f32", "f16"] {
            let mut results: Vec<LegResult> = Vec::new();
            for tier in ["fast", "exact", "exact", "fast"] {
                let Some(r) = run_leg(tier, rung, target, &bytes, &base) else {
                    return;
                };
                println!(
                    "[leg {tier}/{rung} ctx≈{}] plain {:.1} tok/s | spec {:.1} tok/s | \
                     {:.2} ms/verify ({} verifies, {} buckets)",
                    r.ctx, r.plain_tps, r.spec_tps, r.ms_per_verify, r.verifies, r.graph_buckets
                );
                assert!(
                    r.graph_buckets >= 1,
                    "[{tier}/{rung}] spec leg ran eager — the tier lives on the graph route"
                );
                results.push(r);
            }
            let mean = |tier: &str, f: &dyn Fn(&LegResult) -> f64| -> f64 {
                let v: Vec<f64> = results.iter().filter(|r| r.tier == tier).map(f).collect();
                v.iter().sum::<f64>() / v.len() as f64
            };
            let (sf, se) = (
                mean("fast", &|r| r.spec_tps),
                mean("exact", &|r| r.spec_tps),
            );
            let (vf, ve) = (
                mean("fast", &|r| r.ms_per_verify),
                mean("exact", &|r| r.ms_per_verify),
            );
            let (pf, pe) = (
                mean("fast", &|r| r.plain_tps),
                mean("exact", &|r| r.plain_tps),
            );
            println!(
                "L3b ABBA [{rung} ctx≈{}] (mock-draft spec engine, drafting excluded):\n\
                 spec:  fast {sf:.1} vs exact {se:.1} tok/s ({:+.1}%)\n\
                 verify: fast {vf:.3} vs exact {ve:.3} ms ({:+.1}%)\n\
                 plain (control, tier-untouched): fast {pf:.1} vs exact {pe:.1} tok/s ({:+.1}%)",
                results[0].ctx,
                (sf / se - 1.0) * 100.0,
                (vf / ve - 1.0) * 100.0,
                (pf / pe - 1.0) * 100.0,
            );
        }
    }
}

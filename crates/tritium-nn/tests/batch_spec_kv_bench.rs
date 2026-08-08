//! ADR 0036 L6 **stage 2** composition measurement: long-context BATCHED
//! decode + multi-slot spec, f16 KV vs f32 KV (model + GPU gated, run
//! explicitly):
//!
//! ```text
//! cargo test -p tritium-nn --features cuda --release --test batch_spec_kv_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The stage-2 sibling of `spec_kv_bench` (the solo-spec L6 ABBA): each leg
//! loads a fresh model under its `TRITIUM_KV` rung, prefills ONE long prompt
//! (~n_ctx − 168 ≈ 3.9K) through the single-sequence path, adopts it into
//! every slot of an N-row batch, and measures
//!   (a) **undrafted** batched plain decode (the on-device-sampling argmax
//!       graph, N rows in lockstep), and
//!   (b) **drafted** multi-slot spec — `tree_verify_greedy_slots` driven by
//!       synthetic fixed-quality chain drafts (per slot: root + 4 correct +
//!       1 wrong tail, m_total = 6·N, an exact bucket at N ∈ {2, 4}), drawn
//!       from the rung's own greedy stream so the acceptance profile is
//!       identical across rungs; draft-generation cost is EXCLUDED.
//! Losslessness is asserted in every leg: batch rows == the rung's
//! single-sequence greedy stream (plain), spec committed == the same stream.
//! ABBA order-alternation (f16, f32, f32, f16) per N.

#![cfg(feature = "cuda")]

use std::path::Path;

/// Model cache root: override via `TRITIUM_MODEL_DIR`; default `~/.cache/tritium-models`; tests skip cleanly when absent.
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

/// Timed lockstep plain steps per leg (every step advances all N rows).
const PLAIN_STEPS: usize = 48;
/// Spec committed-token target PER SLOT (timed portion).
const SPEC_TOKENS: usize = 50;
/// Room past the prefill: plain steps + spec commits + tree slack.
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

struct KvEnvGuard(Option<std::ffi::OsString>);
impl KvEnvGuard {
    fn set(rung: &str) -> Self {
        let prev = std::env::var_os("TRITIUM_KV");
        // SAFETY: this ignored bench runs single-threaded (--test-threads=1
        // per the header command); no other thread touches the environment.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("TRITIUM_KV", rung);
        }
        Self(prev)
    }
}
impl Drop for KvEnvGuard {
    fn drop(&mut self) {
        // SAFETY: single-threaded bench (see `KvEnvGuard::set`).
        #[allow(unsafe_code)]
        unsafe {
            match self.0.take() {
                Some(v) => std::env::set_var("TRITIUM_KV", v),
                None => std::env::remove_var("TRITIUM_KV"),
            }
        }
    }
}

struct LegResult {
    rung: &'static str,
    ctx: usize,
    /// Aggregate committed tokens/s of the batched plain (argmax-graph) phase.
    plain_tps: f64,
    /// Aggregate committed tokens/s of the multi-slot spec phase.
    spec_tps: f64,
    verifies: usize,
}

/// One measured leg under `rung` with an N-slot batch.
fn run_leg(rung: &'static str, n: usize, bytes: &[u8], base: &[u32]) -> Option<LegResult> {
    let _kv = KvEnvGuard::set(rung);
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
        n_ctx >= 4096,
        "stage-2 gate wants >=4K context (n_ctx={n_ctx})"
    );
    let plen = n_ctx - TAIL_BUDGET;
    let prompt: Vec<u32> = base.iter().cycle().take(plen).copied().collect();
    let positions: Vec<usize> = (0..plen).collect();

    // Single-seq prefill + the rung-local greedy reference stream (want[0] =
    // the prefill argmax; the batch rows are argmax-gated identical to this).
    let logits = runner.forward(&prompt, &positions).expect("prefill");
    let mut want: Vec<u32> = vec![argmax(&logits) as u32];
    {
        let mut pos = plen;
        while want.len() < PLAIN_STEPS.max(SPEC_TOKENS) + 12 {
            let logits = runner
                .forward(&[*want.last().expect("non-empty")], &[pos])
                .expect("stream step");
            want.push(argmax(&logits) as u32);
            pos += 1;
        }
    }

    // Adopt the SAME prefix into every slot of a dense N-row batch.
    let mut batch = runner.new_batch(n).expect("new_batch");
    for r in 0..n {
        runner
            .adopt_into_batch_row(&mut batch, r, plen)
            .expect("adopt");
        batch.set_position(r, plen).expect("pos");
    }
    let rm = runner
        .resident_cuda()
        .expect("resident model")
        .expect("cuda resident model present");

    // ── Phase A: undrafted batched plain (argmax graph), lockstep. ──
    let mut feed: Vec<u32> = vec![want[0]; n];
    // Warm the batched argmax graph off the clock.
    let ids = rm
        .decode_batch_graph_argmax(&mut batch, &feed)
        .expect("warm batch step");
    for (r, &id) in ids.iter().enumerate() {
        assert_eq!(id, want[1], "warm step row {r} diverged from single greedy");
    }
    feed = ids;
    // step_idx: next expected want index.
    let t0 = std::time::Instant::now();
    for step_idx in (2usize..).take(PLAIN_STEPS) {
        let ids = rm
            .decode_batch_graph_argmax(&mut batch, &feed)
            .expect("batch step");
        for (r, &id) in ids.iter().enumerate() {
            assert_eq!(
                id, want[step_idx],
                "[{rung} N={n}] plain batch row {r} diverged at step {step_idx}"
            );
        }
        feed = ids;
    }
    let plain_dt = t0.elapsed();

    // ── Phase B: drafted multi-slot spec. Rewind every slot to the prefix
    // (rows past the watermark are dead; verifies re-append them). ──
    for r in 0..n {
        batch.set_position(r, plen).expect("rewind");
    }
    let rows: Vec<usize> = (0..n).collect();
    let mut committed: Vec<Vec<u32>> = vec![vec![want[0]]; n];
    let mut verifies = 0usize;
    // Chain per slot: root + 4 correct + 1 wrong tail (m = 6; m_total = 6N,
    // an exact bucket for N ∈ {2, 4}); each verify commits 5 tokens.
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

    // Losslessness within the rung, per slot.
    for (r, c) in committed.iter().enumerate() {
        let m = c.len().min(want.len());
        assert_eq!(
            &c[..m],
            &want[..m],
            "[{rung} N={n}] slot {r} spec committed stream diverged from plain greedy"
        );
    }
    drop(batch);

    Some(LegResult {
        rung,
        ctx: plen,
        plain_tps: (n * PLAIN_STEPS) as f64 / plain_dt.as_secs_f64(),
        spec_tps: spec_committed as f64 / spec_dt.as_secs_f64(),
        verifies,
    })
}

#[test]
#[ignore = "stage-2 ABBA bench: run explicitly with --ignored --nocapture --test-threads=1"]
fn cuda_l6_stage2_batch_spec_kv_abba() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model bench)", *GGUF_PATH);
        return;
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

    for n in [2usize, 4] {
        let mut results: Vec<LegResult> = Vec::new();
        for rung in ["f16", "f32", "f32", "f16"] {
            let Some(r) = run_leg(rung, n, &bytes, &base) else {
                return;
            };
            println!(
                "[N={n} leg {rung}] ctx≈{} plain {:.1} agg tok/s | multi-slot \
                 spec {:.1} agg tok/s ({} verifies)",
                r.ctx, r.plain_tps, r.spec_tps, r.verifies
            );
            results.push(r);
        }
        let mean = |rung: &str, f: &dyn Fn(&LegResult) -> f64| -> f64 {
            let v: Vec<f64> = results.iter().filter(|r| r.rung == rung).map(f).collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        let (p16, p32) = (mean("f16", &|r| r.plain_tps), mean("f32", &|r| r.plain_tps));
        let (s16, s32) = (mean("f16", &|r| r.spec_tps), mean("f32", &|r| r.spec_tps));
        println!(
            "stage-2 ABBA N={n} @ctx≈{} (mock drafts, drafting cost excluded):\n\
             batched plain:   f16 {p16:.1} vs f32 {p32:.1} agg tok/s ({:+.1}%)\n\
             multi-slot spec: f16 {s16:.1} vs f32 {s32:.1} agg tok/s ({:+.1}%)",
            results[0].ctx,
            (p16 / p32 - 1.0) * 100.0,
            (s16 / s32 - 1.0) * 100.0,
        );
    }
}

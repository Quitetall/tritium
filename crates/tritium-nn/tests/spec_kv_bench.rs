//! ADR 0036 L6 measurement: long-context (>=4K) solo spec decode, f16 KV vs
//! f32 KV, with plain decode as the reference pair (model + GPU gated, run
//! explicitly):
//!
//! ```text
//! cargo test -p tritium-nn --features cuda --release --test spec_kv_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! ABBA order-alternation (f16, f32, f32, f16); each leg loads a fresh model
//! under its `TRITIUM_KV` rung, prefills near the context limit and measures
//! (a) plain graph decode and (b) the tree-verify spec engine on the GRAPH
//! route driven by synthetic fixed-quality drafts (an m=8 chain per verify:
//! 6 correct + 1 wrong tail, drawn from the rung's own greedy stream — so
//! the acceptance profile is identical across rungs and the A/B isolates the
//! KV rung in the verify forward). Draft-generation cost is EXCLUDED (mock
//! drafter): the numbers bind on the verify+commit engine, which is the
//! surface L6 changed. Losslessness (spec committed == plain greedy within
//! the rung) is asserted in every leg.

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

/// Plain timed decode steps per leg.
const PLAIN_STEPS: usize = 64;
/// Spec committed-token target per leg (timed portion).
const SPEC_TOKENS: usize = 56;
/// Room left past the prefill: plain steps + spec commits + tree pad slack.
const TAIL_BUDGET: usize = PLAIN_STEPS + SPEC_TOKENS + 48;

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
    plain_tps: f64,
    spec_tps: f64,
    verifies: usize,
    graph_buckets: usize,
}

/// One measured leg under `rung`: fresh model, long prefill, plain decode
/// timing, then a fresh prefill and the mock-draft spec engine timing.
fn run_leg(rung: &'static str, bytes: &[u8], base: &[u32]) -> Option<LegResult> {
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
        "the L6 gate wants >=4K context (n_ctx={n_ctx})"
    );
    let target = n_ctx - TAIL_BUDGET;
    let prompt: Vec<u32> = base.iter().cycle().take(target).copied().collect();
    let positions: Vec<usize> = (0..prompt.len()).collect();

    // ── Plain leg ──
    let logits = runner.forward(&prompt, &positions).expect("prefill");
    let mut next = argmax(&logits) as u32;
    let mut pos = prompt.len();
    // Warm the M=1 decode graph.
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
    // Extend the reference stream far enough for the spec leg's drafts.
    while stream.len() < SPEC_TOKENS + 16 {
        let logits = runner.forward(&[next], &[pos]).expect("stream tail");
        next = argmax(&logits) as u32;
        pos += 1;
        stream.push(next);
    }

    // ── Spec leg: fresh state, same prompt; verify-driven decode on the
    // graph tree route with fixed-quality synthetic drafts ──
    runner.reset();
    let l0 = runner.forward(&prompt, &positions).expect("spec prefill");
    let first = argmax(&l0) as u32;
    // The plain leg's first post-prefill token was produced by the SAME rung
    // and state; the streams must agree from the start.
    let rm = runner
        .resident_cuda()
        .expect("resident model")
        .expect("cuda resident model present");
    let want = &stream; // rung-local greedy reference (starts after warm step)

    // committed[0] is the prefill argmax; want[] starts one step later.
    let mut committed: Vec<u32> = vec![first];
    let mut verifies = 0usize;
    // Chain tree per verify: root = last committed; 6 correct continuations
    // + 1 wrong tail (constant acceptance profile: 6 drafts + bonus = 7).
    let make_tree = |c: usize, root: u32| -> (Vec<u32>, Vec<i32>) {
        let mut t = vec![root];
        t.extend(want[c..c + 6].iter().copied());
        t.push((want[c + 5] + 5) % 128_256);
        (t, (0..8i32).map(|i| i - 1).collect())
    };
    // Warm verify (captures the bucket-8 tree graph outside the clock).
    {
        let (t, p) = make_tree(0, first);
        let out = rm.tree_verify_greedy(&t, &p).expect("warm verify");
        committed.extend(&out);
        verifies += 1;
    }
    let timed_start = committed.len();
    let t0 = std::time::Instant::now();
    while committed.len() - timed_start < SPEC_TOKENS {
        let c = committed.len() - 1; // index into want for continuations
        let (t, p) = make_tree(c, *committed.last().expect("non-empty"));
        let out = rm.tree_verify_greedy(&t, &p).expect("spec verify");
        assert!(!out.is_empty(), "verify must commit >= 1 token");
        committed.extend(&out);
        verifies += 1;
    }
    let spec_dt = t0.elapsed();
    let spec_committed = committed.len() - timed_start;
    let graph_buckets = rm.tree_graph_bucket_count();

    // Losslessness at long ctx, within the rung: committed == plain greedy.
    let n = (committed.len() - 1).min(want.len());
    assert_eq!(
        &committed[1..1 + n],
        &want[..n],
        "[{rung}] spec committed stream diverged from plain greedy at ctx≈{target}"
    );

    Some(LegResult {
        rung,
        ctx: target,
        plain_tps: PLAIN_STEPS as f64 / plain_dt.as_secs_f64(),
        spec_tps: spec_committed as f64 / spec_dt.as_secs_f64(),
        verifies,
        graph_buckets,
    })
}

#[test]
#[ignore = "L6 ABBA bench: run explicitly with --ignored --nocapture --test-threads=1"]
fn cuda_l6_long_ctx_spec_kv_abba() {
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

    // ABBA: f16, f32, f32, f16.
    let mut results: Vec<LegResult> = Vec::new();
    for rung in ["f16", "f32", "f32", "f16"] {
        let Some(r) = run_leg(rung, &bytes, &base) else {
            return;
        };
        println!(
            "[leg {rung}] ctx≈{} plain {:.1} tok/s | spec {:.1} tok/s \
             ({} verifies, {} tree-graph buckets)",
            r.ctx, r.plain_tps, r.spec_tps, r.verifies, r.graph_buckets
        );
        // The L6 route assertion: BOTH rungs must have run the graph route
        // (unless the eager counterfactual is explicitly requested for A/B).
        assert!(
            // Intentional duplication of tritium-cuda's env_flag_on
            // contract (unset/"0" = off, "1" = on) — that helper is
            // pub(super) and unreachable from this crate's tests. If the
            // contract ever broadens, update both.
            r.graph_buckets >= 1
                || matches!(std::env::var("TRITIUM_TREE_EAGER").as_deref(), Ok("1")),
            "[{rung}] spec leg ran eager — the L6 bench must exercise the \
             graph tree route"
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
        "L6 ABBA @ctx≈{} (mock-draft spec engine, drafting cost excluded):\n\
         plain: f16 {p16:.1} vs f32 {p32:.1} tok/s ({:+.1}%)\n\
         spec:  f16 {s16:.1} vs f32 {s32:.1} tok/s ({:+.1}%)",
        results[0].ctx,
        (p16 / p32 - 1.0) * 100.0,
        (s16 / s32 - 1.0) * 100.0,
    );
}

//! 1c decision measurement (ADR 0032 truncate-reconcile, batched leg): for a
//! KEPT multi-spec row whose drafter watermark fell a probe-period behind
//! (gap = 64) at long context, is the masked k=1 `draft_batch` catch-up loop
//! cheaper than the enrollment path's reset + full re-prefill + adopt?
//! (batch.rs routes gaps > 1 to the latter; the gap-close loop's guard would
//! only be raised to probe-period gaps if THIS measurement says the masked
//! loop wins at the 4K shape.) Model + GPU gated, run explicitly:
//!
//! ```text
//! cargo test -p tritium-nn --features cuda --release --test drafter_catchup_bench -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

use tritium_nn::ModelRunner;

static DRAFTER_PATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let home = std::env::var("HOME").unwrap_or_default();
    let longctx = format!("{home}/blut/data/drafter-8L768-longctx.gguf");
    if std::path::Path::new(&longctx).exists() {
        longctx
    } else {
        format!("{home}/blut/data/drafter-8L768-s3.gguf")
    }
});

fn load_on(name: &str, bytes: &[u8]) -> Option<ModelRunner> {
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == name)
        .map(|e| e.init)?;
    let backend = match init() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: {name} backend failed to init ({e})");
            return None;
        }
    };
    let file = tritium_format::read_gguf(bytes).expect("parse gguf");
    Some(ModelRunner::load(&file, bytes, backend).expect("load model"))
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

/// A/B at the probe shape: p = 4032, watermark = 3968 (gap 64), pool N = 4
/// with the other three rows dead (the shipped gap-close masks non-gap rows
/// dead, so the catch-up steps run at full batch width with one live row —
/// exactly what a raised guard would execute).
#[test]
#[ignore = "bench: run explicitly on a quiet box"]
fn drafter_probe_catchup_ab() {
    const P: usize = 4032; // pending's position (ctx 4096 cap - headroom)
    const GAP: usize = 64; // SpecGovernor::PROBE_PERIOD
    const REPS: usize = 5;
    let never_eos = u32::MAX;

    if !std::path::Path::new(&*DRAFTER_PATH).exists() {
        eprintln!("skipping: {} absent (drafter-gated bench)", *DRAFTER_PATH);
        return;
    }
    let bytes = std::fs::read(&*DRAFTER_PATH).expect("read drafter gguf");
    let Some(mut runner) = load_on("cuda", &bytes) else {
        return;
    };
    println!("drafter: {} (n_ctx={})", *DRAFTER_PATH, runner.config.n_ctx);
    assert!(
        (runner.config.n_ctx as usize) > P + 1,
        "drafter n_ctx too small for the probe shape"
    );

    let hist: Vec<u32> = (0..P as u32).map(|i| 1 + (i % 997)).collect();
    let mut batch = runner.new_batch(4).expect("new_batch");
    for r in 1..4 {
        batch.set_live(r, false).expect("set_live");
    }

    // Untimed helper: put row 0's drafter KV at watermark `n`.
    let enroll_at = |runner: &mut ModelRunner, batch: &mut tritium_cuda::BatchKv, n: usize| {
        runner.reset();
        let positions: Vec<usize> = (0..n).collect();
        runner.forward(&hist[..n], &positions).expect("prefill");
        runner.adopt_into_batch_row(batch, 0, n).expect("adopt");
        batch.set_position(0, n).expect("set_position");
    };

    // Warmup all three legs once (clock + graph capture + IMMA shadows).
    enroll_at(&mut runner, &mut batch, P);
    enroll_at(&mut runner, &mut batch, P - GAP);
    for _ in 0..GAP {
        let dpos = batch.positions()[0];
        let feeds = [hist[dpos], 0, 0, 0];
        runner
            .draft_batch(&mut batch, &feeds, 1, never_eos)
            .expect("warmup catch-up step");
    }
    assert_eq!(batch.positions()[0], P, "warmup catch-up must reach p");
    enroll_at(&mut runner, &mut batch, P - GAP);
    runner
        .adopt_from_batch_row(&batch, 0, P - GAP)
        .expect("warmup adopt-from");
    let gap_pos: Vec<usize> = (P - GAP..P).collect();
    runner
        .forward(&hist[P - GAP..P], &gap_pos)
        .expect("warmup gap forward");
    runner
        .adopt_into_batch_row(&mut batch, 0, P)
        .expect("warmup adopt back");
    batch.set_position(0, P).expect("warmup re-position");

    let mut t_prefill = Vec::with_capacity(REPS);
    let mut t_catchup = Vec::with_capacity(REPS);
    let mut t_delta = Vec::with_capacity(REPS);
    for rep in 0..REPS {
        // Rotate the leg order each rep (abc / bca / cab ...) so no leg
        // always rides the same clock state.
        for leg in 0..3 {
            match (rep + leg) % 3 {
                0 => {
                    // (a) enrollment re-prefill: reset + M=P prefill + adopt.
                    let t0 = std::time::Instant::now();
                    enroll_at(&mut runner, &mut batch, P);
                    t_prefill.push(t0.elapsed().as_secs_f64() * 1e3);
                }
                1 => {
                    // (b) masked catch-up: 64 k=1 draft_batch steps.
                    enroll_at(&mut runner, &mut batch, P - GAP); // untimed restore
                    let t0 = std::time::Instant::now();
                    for _ in 0..GAP {
                        let dpos = batch.positions()[0];
                        let feeds = [hist[dpos], 0, 0, 0];
                        runner
                            .draft_batch(&mut batch, &feeds, 1, never_eos)
                            .expect("catch-up step");
                    }
                    t_catchup.push(t0.elapsed().as_secs_f64() * 1e3);
                    assert_eq!(batch.positions()[0], P, "catch-up must reach p");
                }
                _ => {
                    // (c) delta re-sync: adopt-from + gap forward + adopt back.
                    enroll_at(&mut runner, &mut batch, P - GAP); // untimed restore
                    let t0 = std::time::Instant::now();
                    runner
                        .adopt_from_batch_row(&batch, 0, P - GAP)
                        .expect("adopt-from");
                    let gp: Vec<usize> = (P - GAP..P).collect();
                    runner.forward(&hist[P - GAP..P], &gp).expect("gap forward");
                    runner
                        .adopt_into_batch_row(&mut batch, 0, P)
                        .expect("adopt back");
                    batch.set_position(0, P).expect("re-position");
                    t_delta.push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
    }

    let mp = median(&mut t_prefill);
    let mc = median(&mut t_catchup);
    let md = median(&mut t_delta);
    println!("re-prefill (reset + M={P} prefill + adopt): median {mp:.2} ms  {t_prefill:.1?}");
    println!(
        "masked catch-up ({GAP} x k=1 draft_batch @ N=4): median {mc:.2} ms ({:.2} ms/step)  {t_catchup:.1?}",
        mc / GAP as f64
    );
    println!(
        "delta re-sync (adopt-from + M={GAP} gap forward + adopt back): median {md:.2} ms  {t_delta:.1?}"
    );
    println!(
        "verdicts at the probe shape: delta/re-prefill = {:.2}x, catch-up/re-prefill = {:.2}x ({})",
        md / mp,
        mc / mp,
        if md < mp {
            "delta wins — the enrollment delta path earns its keep"
        } else {
            "re-prefill wins — record and keep the full path"
        }
    );
}

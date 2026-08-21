//! Task-#65 dual-graph dispatch gate (model + GPU gated, `cuda` feature):
//! under the fast tier, a dense-solo small-bucket verify must replay the
//! FUSED capture below `TREE_NB_MIN_PREFIX` and the NODE-BLOCKED capture at
//! or above it — two graphs per bucket, selected per replay by prefix
//! length. Teeth: the capture count goes 1 → 2 when the same bucket crosses
//! the threshold, the NB capture line appears exactly at the crossing, and
//! both verifies commit sane tokens (numeric bounds are the kernel gates'
//! job — `tree_fused_nb_fast_tier_within_1e4_of_exact_pair`).
//!
//! A separate test binary so the tier env cannot leak into concurrently
//! running tests (the tier_fast_bench isolation pattern); the single test
//! here runs with the binary to itself.

#![cfg(feature = "cuda")]

use tritium_cpu as _;
use tritium_cuda as _;

static DRAFTER_PATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    format!(
        "{}/blut/data/drafter-8L768-longctx.gguf",
        std::env::var("HOME").unwrap_or_default()
    )
});

#[test]
fn nb_dual_graph_dispatch_crosses_threshold() {
    if !std::path::Path::new(&*DRAFTER_PATH).exists() {
        eprintln!("skipping: {} absent (drafter-gated test)", *DRAFTER_PATH);
        return;
    }
    // SAFETY: the only test in this binary; no other thread touches the
    // environment (the tier_fast_bench EnvGuard precedent).
    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var("TRITIUM_KERNEL_TIER", "fast");
        std::env::remove_var("TRITIUM_TREE_NB"); // the Auto default under test
        std::env::remove_var("TRITIUM_KV");
    }

    let bytes = std::fs::read(&*DRAFTER_PATH).expect("read drafter gguf");
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == "cuda")
        .expect("cuda registered")
        .init;
    let Ok(backend) = init() else {
        eprintln!("skipping: no cuda device");
        return;
    };
    let file = tritium_format::read_gguf(&bytes).expect("parse gguf");
    let mut runner = tritium_nn::ModelRunner::load(&file, &bytes, backend).expect("load");

    // Below the threshold: prefix 100 → the FUSED capture (count 1).
    let short: Vec<u32> = (1u32..=100).collect();
    let positions: Vec<usize> = (0..short.len()).collect();
    runner.forward(&short, &positions).expect("short prefill");
    let chain = |base: u32| -> (Vec<u32>, Vec<i32>) {
        // 7-node chain (bucket 8): root + 6 children, arbitrary in-vocab ids.
        let tokens: Vec<u32> = (0..7u32).map(|i| base + i).collect();
        let parents: Vec<i32> = (0..7i32).map(|i| i - 1).collect();
        (tokens, parents)
    };
    {
        let rm = runner.resident_cuda().expect("resident").expect("cuda");
        let (t, p) = chain(11);
        let out = rm.tree_verify_greedy(&t, &p).expect("short verify");
        assert!(!out.is_empty(), "short verify must commit");
        assert_eq!(
            rm.tree_graph_variant_counts(),
            (1, 0),
            "exactly the FUSED capture below the threshold"
        );
    }

    // Cross the threshold: fresh sequence at prefix 2000 → the NB capture
    // joins (count 2), same bucket.
    runner.reset();
    let long: Vec<u32> = (0..2000u32).map(|i| 1 + (i % 997)).collect();
    let positions: Vec<usize> = (0..long.len()).collect();
    runner.forward(&long, &positions).expect("long prefill");
    {
        let rm = runner.resident_cuda().expect("resident").expect("cuda");
        let (t, p) = chain(23);
        let out = rm.tree_verify_greedy(&t, &p).expect("long verify");
        assert!(!out.is_empty(), "long verify must commit");
        assert_eq!(
            rm.tree_graph_variant_counts(),
            (1, 1),
            "the NB capture must join at prefix >= TREE_NB_MIN_PREFIX \
             (dual graphs for one bucket, direction pinned)"
        );
        // And replaying BELOW the threshold again must not re-capture.
    }
    runner.reset();
    runner
        .forward(&short, &(0..short.len()).collect::<Vec<_>>())
        .expect("short again");
    {
        let rm = runner.resident_cuda().expect("resident").expect("cuda");
        let (t, p) = chain(37);
        let out = rm.tree_verify_greedy(&t, &p).expect("short verify 2");
        assert!(!out.is_empty());
        assert_eq!(
            rm.tree_graph_variant_counts(),
            (1, 1),
            "no third capture — both variants reused"
        );
    }
    println!("dual-graph dispatch: fused below / NB above the 1536 threshold, 2 captures");
}

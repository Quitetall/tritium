//! **Step 2: SALT as it was actually designed** — curvature/error-allocated plane counts at a bits
//! budget, instead of the uniform `T` every experiment so far has used.
//!
//! Every SALT number to date gave *every* group the same plane count. That is not the method: SALT V2
//! allocates planes per group from a global bits budget, so cheap groups take one plane and
//! loss-critical ones take three. The preregistered Compact profile targets ≤2.25 **physical** bpw —
//! roughly one plane on average — which uniform `T=3` (6.19 physical bpw) blows through by 2.75×.
//!
//! This scores allocated SALT against uniform SALT on one axis, so "does allocation earn its
//! complexity?" gets a number rather than an assumption.
//!
//! Sensitivity is **uniform** here: without calibration activations there is no real Hessian, and the
//! repo's own finding is that energy-weighted ≈ uniform in that case. So the allocator is spending its
//! budget purely on marginal reconstruction-error reduction. Curvature-informed allocation (a real
//! Fisher/Hessian from the corpus) is the next rung, and is exactly where PT²-LLM's
//! activation-awareness would enter.
//!
//! `#[ignore]`d; run:
//! ```text
//! TRITIUM_CORPUS=<corpus.json> cargo test -p tritium-nn --release \
//!   --test salt_allocated_ppl -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{extract, perplexity_windowed};
use tritium_nn::ModelRunner;
use tritium_quantize::{AllocConfig, GroupInput, allocate};
use tritium_train::ops::ste;

const EVAL_WINDOW: usize = 512;
/// Packed cost of one ternary plane in TQ2_0 (66 bytes per 256 trits).
const PHYSICAL_BPW_PER_PLANE: f64 = 66.0 * 8.0 / 256.0;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

fn eval_ids() -> Vec<u32> {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_string()
    });
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("corpus json")).expect("parse");
    j["eval_ids"]
        .as_array()
        .expect("eval_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect()
}

#[test]
#[ignore = "slow allocated-SALT sweep; needs SmolLM2-135M; run explicitly"]
fn allocated_salt_beats_uniform_at_matched_bits() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let itf_iters: usize = std::env::var("TRITIUM_ITF_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let eval = eval_ids();
    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);

    // One group per output row — the natural kernel tile the allocator documents.
    let total_weights: usize = fp.iter().map(Vec::len).sum();
    println!(
        "fp reference {ppl_fp:.3} ppl | {} tensors, {total_weights} weights, one group per output row\n",
        fp.len()
    );
    println!(
        "{:<26} {:>9} {:>9} {:>13} {:>9}",
        "config", "log bpw", "phys bpw", "ppl", "× fp"
    );
    println!("{}", "-".repeat(70));

    // Uniform reference points, both fitters.
    for t in 1..=3usize {
        for (label, q) in [
            (
                "uniform greedy",
                fp.iter()
                    .zip(&shapes)
                    .map(|(w, &(n, k))| ste::salt_quantize_forward(w, n, k, t))
                    .collect::<Vec<_>>(),
            ),
            (
                "uniform ITF",
                fp.iter()
                    .zip(&shapes)
                    .map(|(w, &(n, k))| ste::salt_quantize_forward_itf(w, n, k, t, itf_iters))
                    .collect::<Vec<_>>(),
            ),
        ] {
            let ppl = perplexity_windowed(&q, &arch, &eval, EVAL_WINDOW);
            println!(
                "{:<26} {:>9.3} {:>9.3} {:>13.3} {:>8.2}×",
                format!("{label} T={t}"),
                t as f64 * 1.584_962_5,
                t as f64 * PHYSICAL_BPW_PER_PLANE,
                ppl,
                ppl / ppl_fp
            );
        }
    }

    // Allocated: sweep the logical budget. 1.585 = all-base (T=1 everywhere); 4.75 = all T=3.
    for target_logical in [1.75f64, 2.0, 2.5, 3.17, 4.0] {
        // Build one group per row across every tensor, in a stable order.
        let mut group_weights: Vec<&[f32]> = Vec::new();
        for (w, &(rows, cols)) in fp.iter().zip(&shapes) {
            for r in 0..rows {
                group_weights.push(&w[r * cols..(r + 1) * cols]);
            }
        }
        let groups: Vec<GroupInput> = group_weights
            .iter()
            .map(|w| GroupInput {
                weights: w,
                sensitivity: 1.0, // uniform: no calibration Hessian available
            })
            .collect();
        let cfg = AllocConfig::from_bpw(target_logical, total_weights, 1, 3);
        let alloc = match allocate(&groups, &cfg) {
            Ok(a) => a,
            Err(e) => {
                println!("  allocation failed at {target_logical:.2} logical bpw: {e:?}");
                continue;
            }
        };
        let sizes: Vec<usize> = group_weights.iter().map(|w| w.len()).collect();
        let log_bpw = alloc.avg_bpw(&sizes);
        let phys_bpw = alloc
            .plane_counts
            .iter()
            .zip(&sizes)
            .map(|(&t, &s)| t as f64 * s as f64 * PHYSICAL_BPW_PER_PLANE)
            .sum::<f64>()
            / total_weights as f64;

        // Quantize each row with the plane count it was allocated.
        let mut gi = 0usize;
        let quantized: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(rows, cols))| {
                let mut out = vec![0.0f32; w.len()];
                for r in 0..rows {
                    let t = alloc.plane_counts[gi];
                    gi += 1;
                    let src = &w[r * cols..(r + 1) * cols];
                    let q = ste::salt_quantize_forward_itf(src, 1, cols, t, itf_iters);
                    out[r * cols..(r + 1) * cols].copy_from_slice(&q);
                }
                out
            })
            .collect();

        let ppl = perplexity_windowed(&quantized, &arch, &eval, EVAL_WINDOW);
        let hist = (1..=3)
            .map(|t| alloc.plane_counts.iter().filter(|&&c| c == t).count())
            .collect::<Vec<_>>();
        println!(
            "{:<26} {:>9.3} {:>9.3} {:>13.3} {:>8.2}×   planes T1/T2/T3 = {}/{}/{}",
            format!("ALLOCATED ITF @{target_logical:.2}"),
            log_bpw,
            phys_bpw,
            ppl,
            ppl / ppl_fp,
            hist[0],
            hist[1],
            hist[2]
        );
    }

    println!(
        "\nRead: compare an ALLOCATED row against the UNIFORM row with the same physical bpw. If \
         allocation does not win at matched bits, the allocator is not earning its complexity — and \
         the Compact profile (≤2.25 physical bpw) is the configuration the paper would publish."
    );
}

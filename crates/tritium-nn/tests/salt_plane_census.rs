//! **Per-plane statistics of the balanced-ternary ladder, and what a random-access codec can charge.**
//!
//! `salt_symbol_entropy.rs` measures the **joint** `3^T` symbol and found a −11.4% (T=3) / −9.0%
//! (T=4) gap against dense packing. That figure is the *upper bound* on any lossless scheme, and it
//! is not reachable in the decode path: a variable-length code destroys the O(1) offset arithmetic
//! the multiply-free kernel depends on, which is why the standing decision is entropy-code **at
//! rest** only. Per-plane statistics have never been measured at all.
//!
//! This asks the deployable question: **how much of that gap survives the random-access
//! constraint?**
//!
//! # The structural prediction being tested
//!
//! `ladder_digits` emits balanced-ternary digits **most significant first** — plane 0 carries
//! `3^(T-1)`, plane `T-1` carries `3^0`. A weight's digit at the coarse end is nonzero only when
//! `|k|` is large, and real weight groups are heavy near zero. So:
//!
//! - **plane 0 should be SPARSE** (high zero fraction, low entropy),
//! - **plane T-1 should be near-UNIFORM** over `{-1,0,+1}` (entropy ≈ log2(3) = 1.585),
//!
//! and if that holds, charging every plane the same rate is structurally wrong.
//!
//! # The codecs being priced (all fixed-rate, all random-access)
//!
//! - **B3** — 5 trits/byte. `ceil(group/5)*8/group` bits/trit; 1.625 at group 128. Rate is
//!   independent of the data, so it is the honest floor for a dense plane.
//! - **D2** — 2 bits/trit. Strictly worse than B3 for unconstrained ternary; reported only so the
//!   table shows why it is never selected.
//! - **TB1** — bitmap + signs: `1 + (1-p)` bits/trit at zero fraction `p`, **plus** a per-block
//!   rank prefix so sign lookup stays O(1) (one `u16` per 256 trits = 0.0625 bits/trit). Beats B3
//!   once `p` exceeds ≈0.66 including that overhead.
//!
//! Entropy is reported alongside as the unreachable bound, never as an achievable rate.
//!
//! `#[ignore]`d; run:
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//! TRITIUM_MODEL_DIR=$HOME/.cache/tritium-models/smollm2-360m \
//!   cargo test -p tritium-nn --release --test salt_plane_census -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, calibrate, extract, fold};
use tritium_nn::ModelRunner;
use tritium_train::ops::ste::{self, RotationPolicy};

const GROUP: usize = 128;
const GRID: usize = 16;
const FOLD_ALPHA: f64 = 0.75;
const CALIB_WINDOWS: usize = 8;
const CALIB_SEQ: usize = 512;
/// One `u16` rank prefix per 256 trits keeps TB1 sign lookup O(1).
const TB1_RANK_BITS_PER_TRIT: f64 = 16.0 / 256.0;

/// Rotation policy for the census. **The rotated/unrotated split is only informative under
/// `Auto`**: with `Always` every group rotates, so the "rot zero%" column equals the overall
/// column and carries no information. Set `TRITIUM_CENSUS_ROTATE=auto` to exercise it.
fn rotation() -> RotationPolicy {
    match std::env::var("TRITIUM_CENSUS_ROTATE").as_deref() {
        Ok("auto") => RotationPolicy::Auto,
        Ok("never") => RotationPolicy::Never,
        _ => RotationPolicy::Always,
    }
}

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = std::env::var("TRITIUM_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-360m"));
    PathBuf::from(dir)
}

fn corpus_train() -> Vec<u32> {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_string()
    });
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("corpus json")).expect("parse");
    j["train_ids"]
        .as_array()
        .expect("train_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect()
}

/// Symbol counts for one plane index, optionally restricted to rotated or unrotated groups.
#[derive(Clone, Copy, Default)]
struct PlaneStats {
    neg: u64,
    zero: u64,
    pos: u64,
}

impl PlaneStats {
    fn observe(&mut self, trits: &[i8]) {
        for &t in trits {
            match t {
                -1 => self.neg += 1,
                1 => self.pos += 1,
                _ => self.zero += 1,
            }
        }
    }

    fn total(self) -> u64 {
        self.neg + self.zero + self.pos
    }

    fn zero_fraction(self) -> f64 {
        if self.total() == 0 {
            return 0.0;
        }
        self.zero as f64 / self.total() as f64
    }

    /// Shannon entropy of the `{-1,0,+1}` marginal, in bits per trit. The unreachable bound.
    fn entropy_bits(self) -> f64 {
        let n = self.total() as f64;
        if n == 0.0 {
            return 0.0;
        }
        [self.neg, self.zero, self.pos]
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.log2()
            })
            .sum()
    }

    /// TB1: a presence bitmap plus one sign bit per nonzero, plus the rank prefix that keeps
    /// lookup O(1). Data-dependent but fixed-rate once `p` is known for the plane.
    fn tb1_bits(self) -> f64 {
        1.0 + (1.0 - self.zero_fraction()) + TB1_RANK_BITS_PER_TRIT
    }
}

/// B3 over one `group`-trit run: `ceil(group/5)` bytes. 1.625 at group 128, not the asymptotic 1.6.
fn b3_bits_per_trit(group: usize) -> f64 {
    (group.div_ceil(5) * 8) as f64 / group as f64
}

#[test]
#[ignore = "slow per-plane census; set TRITIUM_MODEL_DIR + TRITIUM_CORPUS; run explicitly"]
fn ladder_plane_census_and_codec_pricing() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() && !dir.join("model.safetensors.index.json").exists()
    {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let train = corpus_train();

    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp,
            &arch,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (fp_folded, _arch_folded) = fold(&fp, &shapes, &arch, &calib, FOLD_ALPHA);

    let b3 = b3_bits_per_trit(GROUP);
    println!(
        "{} | g{GROUP} | grid {GRID} | fold α={FOLD_ALPHA} | rotation={:?}\n\
         B3 = {b3:.4} bits/trit (the dense floor); D2 = 2.0000; TB1 = 1 + (1-p) + {TB1_RANK_BITS_PER_TRIT:.4} rank\n",
        dir.file_name().unwrap_or_default().to_string_lossy(),
        rotation(),
    );
    if matches!(rotation(), RotationPolicy::Always | RotationPolicy::Never) {
        println!(
            "  (rot zero% duplicates the overall column under this policy — every group takes the\n\
             same branch. Use TRITIUM_CENSUS_ROTATE=auto for a meaningful split.)\n"
        );
    }

    for t in [3usize, 4] {
        // Per-plane symbol counts, split by whether the group's fitter chose to rotate.
        let mut all = vec![PlaneStats::default(); t];
        let mut rot = vec![PlaneStats::default(); t];
        let mut unrot = vec![PlaneStats::default(); t];

        for (w, &(rows, cols)) in fp_folded.iter().zip(&shapes) {
            let fits = ste::geometric_ladder_fit(w, rows, cols, t, GROUP, GRID, rotation());
            let mask = ste::geometric_rotation_mask(w, rows, cols, t, GROUP, GRID, rotation());
            for (gi, (_s0, planes)) in fits.iter().enumerate() {
                let rotated = mask.get(gi).copied().unwrap_or(0) != 0;
                for (p, trits) in planes.iter().enumerate() {
                    all[p].observe(trits);
                    if rotated {
                        rot[p].observe(trits);
                    } else {
                        unrot[p].observe(trits);
                    }
                }
            }
        }

        println!("── T={t} ─────────────────────────────────────────────────────────────────────");
        println!(
            "{:<6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>11}",
            "plane", "zero%", "H(bits)", "B3", "TB1", "best", "rot zero%", "unrot zero%"
        );
        let mut dense_total = 0.0;
        let mut best_total = 0.0;
        let mut entropy_total = 0.0;
        for p in 0..t {
            let s = all[p];
            let tb1 = s.tb1_bits();
            let best = b3.min(tb1);
            dense_total += b3;
            best_total += best;
            entropy_total += s.entropy_bits();
            println!(
                "{:<6} {:>8.2}% {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>8.2}% {:>10.2}%",
                p,
                100.0 * s.zero_fraction(),
                s.entropy_bits(),
                b3,
                tb1,
                best,
                100.0 * rot[p].zero_fraction(),
                100.0 * unrot[p].zero_fraction(),
            );
        }

        // Scales are identical across all three accountings, so they cancel in the comparison and
        // are deliberately excluded: this prices the TRIT PAYLOAD only.
        let saving = 100.0 * (1.0 - best_total / dense_total);
        let bound = 100.0 * (1.0 - entropy_total / dense_total);
        println!(
            "\n  trit payload: dense(all B3) {dense_total:.4}  best-per-plane {best_total:.4}  \
             entropy bound {entropy_total:.4}  (bits/weight)\n  \
             random-access saving {saving:+.2}%   vs entropy bound {bound:+.2}%   \
             → {:.0}% of the bound is reachable\n",
            if bound.abs() > 1e-9 {
                100.0 * saving / bound
            } else {
                0.0
            }
        );
    }

    println!(
        "NOTE: entropy is the UNREACHABLE bound — a variable-length code breaks the O(1) offset\n\
         arithmetic the multiply-free decode path needs. Only the `best` column is deployable, and\n\
         TB1's rate already includes the rank prefix that keeps sign lookup O(1). Scales are\n\
         excluded throughout: they are identical across the three accountings and would only\n\
         dilute the comparison."
    );
}

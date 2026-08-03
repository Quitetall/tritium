//! **What do SALT's trit planes actually cost, as opposed to what we charge for them?**
//!
//! Every bpw figure this repo has ever quoted assumes each plane is packed densely: `log2(3)` bits
//! per trit (B3's 1.6, D2's 2.0), times `T`, plus scales. That is the cost of storing `T`
//! *independent* uniform ternary symbols.
//!
//! The planes are not independent and they are not uniform. `fit_joint_ternary` assigns all `T`
//! planes for one weight jointly by exact enumeration of `3^T` states — and the sweep that landed
//! the 1.08x-fp T=3 result observed only **13 of 27** states ever being used. If that holds on real
//! weights, the joint symbol carries `log2(13) = 3.70` bits where dense packing charges `3 x 1.585
//! = 4.75`, and the true rate is lower still once the *distribution* over those states is accounted
//! for (the near-zero residual state dominates by construction).
//!
//! This test measures that gap. It changes no reconstruction and therefore no perplexity — entropy
//! coding is lossless, so whatever rate it finds applies at the quality already measured. It is a
//! repricing, not a new method.
//!
//! Three rates are reported per `T`:
//!
//! - **dense** — what we charge today: `T * bits_per_trit(codec) + scales/group`.
//! - **alphabet** — `log2(|distinct states used|)`, a fixed-width code over the observed alphabet.
//!   Cheapest possible *non-statistical* code; a decoder needs only the state table.
//! - **entropy** — `H(symbol)` measured per group and averaged, the rate an ideal per-group
//!   entropy coder achieves. Measured per *group* rather than globally because scales are already
//!   per-group, so that is the unit any real codec would adapt over; a global figure would flatter
//!   itself by pooling distributions that a real decoder cannot pool.
//!
//! The honest caveats, stated up front because they decide whether any of this ships:
//!
//! 1. **`gr-alpha` does NOT charge for the per-group alphabet table and is therefore optimistic —
//!    do not quote it as a rate.** A decoder that reads `log2(|alphabet_g|)`-bit symbols has to be
//!    told *which* states are in group `g`'s alphabet, and that side information is unpriced here.
//!    At `T=8` a group using 40 of 6561 states would need `40·log2(6561) ≈ 508` bits of table for
//!    128 weights — about 4 bpw, which would eat the entire saving. Only **`dense`** and
//!    **`gl-alpha`** (one table for the whole model, amortised to nothing) are chargeable as-is.
//! 2. `entropy` is a lower bound on a real codec's rate, not a rate. Arithmetic coding approaches
//!    it; a practical GPU-decodable format will not reach it, and an adaptive coder that avoids
//!    transmitting the distribution is inherently sequential.
//! 3. **Bandwidth saved is not time saved.** The open question already on record for B3
//!    (`research-ternary-sota-mid2026.md:160`) applies with more force here: a variable-length or
//!    table-driven decode costs more per weight than a shift-and-mask. This test cannot answer it.
//! 4. Symbol statistics are measured on one 135M model's weights after the salience fold. Whether
//!    the collapse survives other architectures and scales is unmeasured.
//!
//! So the result to take from this is **how much information the symbols actually carry**, which
//! bounds what any codec could achieve. Turning that bound into a shippable rate is a codec-design
//! problem this test does not solve.
//!
//! Run:
//! ```text
//! cargo test -p tritium-nn --release --test salt_symbol_entropy -- --ignored --nocapture
//! ```

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use common::{Calib, calibrate, extract, fold};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_nn::ModelRunner;
use tritium_quantize::{JointFitConfig, JointFitMetric, fit_joint_ternary};
use tritium_train::ops::ste::{self, RotationPolicy};

const GROUP: usize = 128;
const CALIB_WINDOWS: usize = 8;
const CALIB_SEQ: usize = 512;
/// Δ candidates per group for the ladder fitter.
const GRID: usize = 16;

/// Which fitter's symbols to price.
///
/// `ladder` is the default and the one that matters. The joint fitter enumerates `3^T` states per
/// weight with EM restarts — measuring it over 134M weights took 90 minutes for `T=1` alone and
/// timed out before `T=2`. The ladder is O(T), so the same sweep runs to `T=8` in seconds. It is
/// also the more interesting subject: a *uniform* grid on a peaked weight distribution should put
/// far more mass on the near-zero symbols than a free-scale fitter, which adapts its levels to the
/// data and thereby flattens its own symbol distribution. If entropy coding pays anywhere, it pays
/// on the ladder.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fitter {
    Ladder,
    Joint,
}

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
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

/// Per-group symbol statistics, accumulated across every group in the model.
#[derive(Default)]
struct SymbolStats {
    /// Every joint state observed anywhere, and how often.
    global: HashMap<u32, u64>,
    /// Sum over groups of `n_g * H_g` — the numerator of the group-adaptive rate.
    weighted_group_entropy_bits: f64,
    /// Sum over groups of `n_g * log2(|alphabet_g|)` — a per-group fixed-width code.
    weighted_group_alphabet_bits: f64,
    weights: u64,
    groups: u64,
}

impl SymbolStats {
    /// Fold one group's joint symbols in. `symbols` is one `u32` per weight in the group,
    /// packing plane `p`'s trit into base-3 digit `p`.
    fn observe_group(&mut self, symbols: &[u32]) {
        if symbols.is_empty() {
            return;
        }
        let mut local: HashMap<u32, u64> = HashMap::new();
        for &s in symbols {
            *local.entry(s).or_default() += 1;
            *self.global.entry(s).or_default() += 1;
        }
        let n = symbols.len() as f64;
        let h: f64 = local
            .values()
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.log2()
            })
            .sum();
        self.weighted_group_entropy_bits += n * h;
        self.weighted_group_alphabet_bits += n * (local.len() as f64).log2();
        self.weights += symbols.len() as u64;
        self.groups += 1;
    }

    /// Ideal per-group entropy-coded rate, bits per weight.
    fn entropy_bits_per_weight(&self) -> f64 {
        self.weighted_group_entropy_bits / self.weights as f64
    }

    /// Per-group fixed-width code over each group's own observed alphabet, bits per weight.
    fn group_alphabet_bits_per_weight(&self) -> f64 {
        self.weighted_group_alphabet_bits / self.weights as f64
    }

    /// One fixed-width code shared by the whole model, bits per weight.
    fn global_alphabet_bits_per_weight(&self) -> f64 {
        (self.global.len() as f64).log2()
    }

    /// Entropy of the pooled distribution — reported only to show how much of the win is
    /// group-local structure a global code would throw away.
    fn global_entropy_bits_per_weight(&self) -> f64 {
        let total: u64 = self.global.values().sum();
        self.global
            .values()
            .map(|&c| {
                let p = c as f64 / total as f64;
                -p * p.log2()
            })
            .sum()
    }
}

/// Joint-fit one group and return its per-weight base-3 joint symbols, or `None` if the fit failed.
///
/// Rotation is applied around the fit exactly as `salt_joint.rs::joint_group` does, and the
/// rotate/plain choice uses the same SSE rule — the symbols measured here are the ones the
/// already-measured 1.08x configuration actually produces, not a different fit.
fn group_symbols(bs: &[f32], cfg: JointFitConfig) -> Option<Vec<u32>> {
    let sse = |recon: &[f32], target: &[f32]| -> f64 {
        recon
            .iter()
            .zip(target)
            .map(|(&x, &y)| f64::from(x - y) * f64::from(x - y))
            .sum()
    };
    let encode = |trits: &[Vec<i8>]| -> Vec<u32> {
        let n = trits[0].len();
        (0..n)
            .map(|i| {
                let mut sym = 0u32;
                let mut radix = 1u32;
                for plane in trits {
                    sym += u32::try_from(plane[i] + 1).expect("trit in {-1,0,1}") * radix;
                    radix *= 3;
                }
                sym
            })
            .collect()
    };

    let plain = fit_joint_ternary(bs, JointFitMetric::Identity, cfg).ok();
    let rotatable = bs.len().is_power_of_two() && bs.len() > 1;
    if !rotatable {
        return plain.map(|f| encode(&f.trits));
    }

    let mut rotated_in = bs.to_vec();
    ste::fast_hadamard(&mut rotated_in);
    let rot = fit_joint_ternary(&rotated_in, JointFitMetric::Identity, cfg).ok();

    match (plain, rot) {
        (Some(p), Some(r)) => {
            // The rotated fit's error must be compared in the ORIGINAL basis: rotate its
            // reconstruction back before scoring, exactly as the sweep does.
            let mut back = r.reconstruction.clone();
            ste::fast_hadamard(&mut back);
            if sse(&back, bs) < sse(&p.reconstruction, bs) {
                Some(encode(&r.trits))
            } else {
                Some(encode(&p.trits))
            }
        }
        (Some(p), None) => Some(encode(&p.trits)),
        (None, Some(r)) => Some(encode(&r.trits)),
        (None, None) => None,
    }
}

/// Base-3 encode one group's plane-major trits into one symbol per weight.
fn encode_trits(trits: &[Vec<i8>]) -> Vec<u32> {
    let n = trits[0].len();
    (0..n)
        .map(|i| {
            let mut sym = 0u32;
            let mut radix = 1u32;
            for plane in trits {
                sym += u32::try_from(plane[i] + 1).expect("trit in {-1,0,1}") * radix;
                radix *= 3;
            }
            sym
        })
        .collect()
}

fn fitter() -> Fitter {
    match std::env::var("TRITIUM_ENTROPY_FITTER").as_deref() {
        Ok("joint") => Fitter::Joint,
        _ => Fitter::Ladder,
    }
}

fn plane_counts(f: Fitter) -> Vec<usize> {
    let default = if f == Fitter::Joint {
        "1,2,3"
    } else {
        "1,2,3,4,6,8"
    };
    std::env::var("TRITIUM_ENTROPY_T")
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .filter(|t| (1..=8).contains(t))
        .collect()
}

#[test]
#[ignore = "sweeps every tensor; needs SmolLM2-135M; run explicitly"]
fn joint_symbols_cost_less_than_dense_planes_charge() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);

    // Measure the symbols of the configuration that produced the headline number, which includes
    // the salience fold — the fold changes the weights, and therefore the residual statistics.
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
    let (fp, _arch) = fold(&fp, &shapes, &arch, &calib, 0.75);

    let fitter = fitter();
    println!(
        "SmolLM2-135M, salience fold alpha=0.75, {}-group fit, {} fitter.\n\
         Reconstruction (and therefore perplexity) is IDENTICAL across all rate columns —\n\
         entropy coding is lossless. Only the price changes.\n",
        GROUP,
        if fitter == Fitter::Joint {
            "joint 3^T (Auto rotation)"
        } else {
            "balanced-ternary ladder (Always rotation)"
        }
    );
    println!(
        "{:<4} {:>7} {:>9} {:>10} {:>10} {:>10} {:>9}",
        "T", "states", "dense", "gl-alpha", "gr-alpha", "entropy", "vs dense"
    );
    println!("{}", "-".repeat(66));

    for t in plane_counts(fitter) {
        let cfg = JointFitConfig {
            planes: t.min(3),
            ..JointFitConfig::default()
        };
        let mut stats = SymbolStats::default();
        let mut failed_groups = 0u64;

        for (w, &(rows, cols)) in fp.iter().zip(&shapes) {
            match fitter {
                // O(T) per weight: one whole-tensor call, no per-group enumeration.
                Fitter::Ladder => {
                    for (_, planes) in ste::geometric_ladder_fit(
                        w,
                        rows,
                        cols,
                        t,
                        GROUP,
                        GRID,
                        RotationPolicy::Always,
                    ) {
                        stats.observe_group(&encode_trits(&planes));
                    }
                }
                Fitter::Joint => {
                    for r in 0..rows {
                        let row = &w[r * cols..(r + 1) * cols];
                        for bs in row.chunks(GROUP) {
                            match group_symbols(bs, cfg) {
                                Some(syms) => stats.observe_group(&syms),
                                None => failed_groups += 1,
                            }
                        }
                    }
                }
            }
        }

        // What the artifact costs today, from the packer's own ledger — the ladder stores ONE f16
        // anchor per group where the free-scale path stores T, so the two have different baselines
        // and must not be charged the same.
        let rot_bit = 1.0 / GROUP as f64;
        let dense = rot_bit
            + match fitter {
                Fitter::Ladder => {
                    ste::ternary_bits_per_weight_geometric(t, GROUP, SaltV2Codec::B3, GROUP)
                }
                Fitter::Joint => {
                    ste::ternary_bits_per_weight_codec(t, GROUP, SaltV2Codec::B3, GROUP)
                }
            };
        // Symbol columns replace only the TRIT payload; scales and the rotation bit still cost.
        let scale_overhead = rot_bit
            + match fitter {
                Fitter::Ladder => 16.0 / GROUP as f64,
                Fitter::Joint => t as f64 * 16.0 / GROUP as f64,
            };
        let gl_alpha = stats.global_alphabet_bits_per_weight() + scale_overhead;
        let gr_alpha = stats.group_alphabet_bits_per_weight() + scale_overhead;
        let entropy = stats.entropy_bits_per_weight() + scale_overhead;

        println!(
            "{:<4} {:>3}/{:<3} {:>9.3} {:>10.3} {:>10.3} {:>10.3} {:>8.1}%",
            t,
            stats.global.len(),
            3usize.pow(t as u32),
            dense,
            gl_alpha,
            gr_alpha,
            entropy,
            100.0 * (entropy - dense) / dense,
        );
        if failed_groups > 0 {
            println!("     (note: {failed_groups} groups failed to fit and are EXCLUDED)");
        }
        println!(
            "     pooled-entropy cross-check {:.3} bpw ({} groups, {} weights)",
            stats.global_entropy_bits_per_weight() + scale_overhead,
            stats.groups,
            stats.weights,
        );
    }

    println!(
        "\ndense    = what every bpw figure in this repo charges (T x B3 + scales)\n\
         gl-alpha = one fixed-width code over the whole model's observed state set\n\
         gr-alpha = fixed-width code per group — OPTIMISTIC, the per-group alphabet table\n\
                    is NOT charged for; not quotable as a rate (see caveat 1)\n\
         entropy  = ideal per-group entropy coder: a LOWER BOUND, not an achievable rate.\n\
         None of these are a speed claim. GPU decode cost of a table-driven symbol is\n\
         unmeasured and could exceed the bandwidth saved."
    );
}

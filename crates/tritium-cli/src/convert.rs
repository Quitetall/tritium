//! `tritium convert` — model-aware SALT conversion: load → calibrate → fold → ladder-fit → bundle.
//!
//! # Why this is not a flag on `tritium quantize`
//!
//! `quantize` reads a bare safetensors file. That is enough to fit weights, and it is exactly why
//! the artifact it writes is a *different configuration* from every published SALT number: those
//! were all measured under the AWQ-style salience fold, which needs activation statistics from a
//! calibration corpus and, more importantly, needs the **layer graph**.
//!
//! The fold scales the columns of each projection by a per-input-channel salience `s` and divides
//! the *preceding norm* by the same `s`. The product is unchanged — it is an exact
//! reparameterisation — but only if both halves are applied. Knowing which norm feeds which
//! projection is a property of the model, not of any tensor, so the fold has to run on a loaded
//! model. Hence a separate command.
//!
//! Measured value of the fold on SmolLM2-360M: **2.3× at T=2, 8.5% at T=3, 1.1% at T=4**.
//!
//! # What the artifact has to contain, and why
//!
//! `ModelWeights::load_salt` reads the 2-D matrices from the SALT bundle and — in its own words —
//! *"only 1D norms come from the fp master"*. A converter that wrote just a `.tslb` next to the
//! original model would therefore produce a model that **loads and is silently wrong**: the
//! weights would carry the fold and the norms would not.
//!
//! So `convert` writes a self-contained output directory:
//!
//! | file | contents |
//! |---|---|
//! | `config.json` | copied verbatim from the source |
//! | `model.safetensors` | the fp 1-D norms, **after** the fold |
//! | `model.tslb` | the SALT bundle: embedding + every projection |
//! | tokenizer assets | copied when present, so the directory is usable on its own |
//!
//! This is the same asymmetry that makes the Hadamard rotation unrepresentable — a rotated fit
//! reconstructs `W·H` and the bundle has nowhere to record `H`. The fold escapes it only because
//! its other half lands in a tensor the format already carries. Rotation stays off here for
//! exactly that reason.
//!
//! # Reading the fidelity receipt
//!
//! `convert` writes `receipt.json` by default: the resolved recipe, the byte cost, and per-tensor
//! relative Frobenius error decoded from the artifact that was just written (so f16 block-scale
//! rounding is included, not idealised away).
//!
//! **That error is not a quality ranking across recipes.** Measured on SmolLM2-135M at T=4:
//! `--fold-alpha 0` gives 0.0252 and `--fold-alpha 0.75` gives **0.0380** — the fold is 51% worse
//! by weight-space error while being better by perplexity, because it deliberately moves error to
//! where activations are small. The same inversion is already on record here for per-group plane
//! allocation: 12.4% lower weight SSE, 12% *worse* perplexity. The receipt says so in its own
//! `interpretation` field, since that file is what a downstream consumer reads.
//!
//! # Configuration that has never been measured
//!
//! Fold-without-rotation is a *new* point: the published numbers are fold **and** rotation, and
//! `quantize`'s numbers are neither. The measured anchors for the no-fold/no-rotation path this
//! shares are 1.335× fp at T=3 and 1.024× at T=4 (SmolLM2-360M, g256). What the fold is worth on
//! top of those is not yet known, so this command reports its recipe and makes no quality claim.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use tritium_format::{SaltRow, salt_rows_to_dense, write_salt_bundle};
use tritium_nn::calibrate::{Calib, calibrate, extract, fold, norm_tensors, weight_names};
use tritium_nn::{HfJsonTokenizer, ModelRunner, Tokenizer};

use crate::quantize_ladder::{LadderConfig, quantize_tensor_ladder};

/// Calibration window length. Matches the research harness's `EVAL_WINDOW` so a `convert` run and
/// a harness run see the same context structure; the fold only reads per-channel second moments,
/// which are not sensitive to this, but keeping them equal removes a variable when comparing.
const CALIB_WINDOW: usize = 512;

/// Tokenizer assets copied through so the output directory stands alone. Missing ones are skipped
/// without comment — not every source snapshot has all of them.
const TOKENIZER_ASSETS: [&str; 5] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
];

/// Resolved conversion settings.
pub(crate) struct ConvertConfig {
    pub(crate) calib: Option<PathBuf>,
    pub(crate) calib_tokens: usize,
    pub(crate) fold_alpha: f64,
    pub(crate) ladder: LadderConfig,
}

pub(crate) fn run(model: &Path, out: &Path, cfg: &ConvertConfig) -> Result<()> {
    cfg.ladder.validate()?;
    if !(0.0..=1.0).contains(&cfg.fold_alpha) {
        bail!(
            "--fold-alpha must be in [0, 1] (0 = identity fold, 1 = full salience); got {}",
            cfg.fold_alpha
        );
    }
    if cfg.calib.is_none() && cfg.fold_alpha != 0.0 {
        bail!(
            "--fold-alpha {} was requested but there is no calibration corpus to measure salience \
             from. Pass --calib <corpus>, or --fold-alpha 0 to convert without the fold (which is \
             what `tritium quantize` already does).",
            cfg.fold_alpha
        );
    }

    let runner = ModelRunner::from_hf(model, Box::new(tritium_cpu::CpuBackend::new()))
        .with_context(|| format!("load fp model {}", model.display()))?;
    let (arch, fp, shapes) = extract(&runner);
    let names = weight_names(&arch);
    if names.len() != fp.len() {
        bail!(
            "internal: {} weight names for {} weights — weight_names and extract have drifted",
            names.len(),
            fp.len()
        );
    }

    // Fold first: it rewrites both the weights and the norms, and the ladder must fit the folded
    // weights, not the originals.
    let (weights, arch, fold_desc) = match &cfg.calib {
        Some(path) => {
            let tokens = load_calibration_tokens(path, model, cfg.calib_tokens, arch.vocab)?;
            let windows = tokens.len() / CALIB_WINDOW;
            if windows == 0 {
                bail!(
                    "calibration corpus {} has {} tokens, need at least {CALIB_WINDOW} for one \
                     window",
                    path.display(),
                    tokens.len()
                );
            }
            let mut acc = Calib::new(&arch);
            for w in 0..windows {
                calibrate(
                    &fp,
                    &arch,
                    &tokens[w * CALIB_WINDOW..(w + 1) * CALIB_WINDOW],
                    &mut acc,
                );
            }
            let (w, arch) = fold(&fp, &shapes, &arch, &acc, cfg.fold_alpha);
            (
                w,
                arch,
                format!(
                    "salience fold alpha={} over {windows} x {CALIB_WINDOW} calibration tokens",
                    cfg.fold_alpha
                ),
            )
        }
        None => (fp, arch, "no calibration fold".to_owned()),
    };

    // Fit every projection with the ladder. `shapes[i]` is `(n_out, k_in)` for `weights[i]`.
    let mut quantized: Vec<(String, Vec<SaltRow>)> = Vec::with_capacity(weights.len());
    let mut total_params = 0usize;
    let mut fidelity: Vec<TensorFidelity> = Vec::with_capacity(weights.len());
    for ((name, w), &(rows, k)) in names.iter().zip(&weights).zip(&shapes) {
        if w.len() != rows * k {
            bail!("{name}: {} values for shape [{rows}, {k}]", w.len());
        }
        let fitted = quantize_tensor_ladder(w, rows, k, &cfg.ladder)
            .with_context(|| format!("ladder-quantize {name}"))?;
        // Score the PACKED rows, not a re-run of the fitter: the f16 block scales are rounded on
        // the way into TQ2_0, so decoding the artifact is the only way to report the error the
        // user's file actually has rather than the one the fit intended.
        let decoded = salt_rows_to_dense(&fitted)
            .map_err(|e| anyhow::anyhow!("decode {name} for the fidelity receipt: {e}"))?;
        fidelity.push(TensorFidelity::measure(name, rows, k, w, &decoded)?);
        total_params += rows * k;
        quantized.push((name.clone(), fitted));
    }

    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;

    let refs: Vec<(&str, &[SaltRow])> = quantized
        .iter()
        .map(|(n, r)| (n.as_str(), r.as_slice()))
        .collect();
    let bundle = write_salt_bundle(&refs).context("serialize SALT bundle")?;
    let bundle_path = out.join("model.tslb");
    std::fs::write(&bundle_path, &bundle)
        .with_context(|| format!("write {}", bundle_path.display()))?;

    // The folded norms. Without these the bundle above reconstructs a model whose weights carry a
    // fold its norms do not — a wrong model that loads cleanly.
    let norms = norm_tensors(&arch);
    let norms_path = out.join("model.safetensors");
    std::fs::write(&norms_path, write_f32_safetensors(&norms))
        .with_context(|| format!("write {}", norms_path.display()))?;

    let config_src = model.join("config.json");
    std::fs::copy(&config_src, out.join("config.json"))
        .with_context(|| format!("copy {}", config_src.display()))?;
    let mut copied_assets = 0usize;
    for asset in TOKENIZER_ASSETS {
        let src = model.join(asset);
        if src.exists() {
            std::fs::copy(&src, out.join(asset))
                .with_context(|| format!("copy {}", src.display()))?;
            copied_assets += 1;
        }
    }

    let receipt_path = out.join("receipt.json");
    let whole_model_error = write_receipt(
        &receipt_path,
        model,
        cfg,
        &fold_desc,
        &fidelity,
        total_params,
        bundle.len(),
    )?;

    // Load the artifact back before claiming success. Every failure mode this command can have —
    // a tensor name the loader does not look up, a shape the bundle disagrees with, a norm written
    // under the wrong key — produces a file that is *structurally valid* and simply never resolves.
    // A converter that reports success without reloading cannot tell those apart from a good run.
    ModelRunner::from_salt(out, &bundle_path, Box::new(tritium_cpu::CpuBackend::new()))
        .with_context(|| {
            format!(
                "the converted model at {} did not load back — the artifact is written but not \
                 usable",
                out.display()
            )
        })?;

    let bpw = cfg.ladder.realizable_bpw();
    println!(
        "converted {} tensors ({:.2}M params) | ladder geometric, {} planes, g{}, grid {}, no \
         rotation, {fold_desc} → {} ({:.1} MiB bundle + {:.1} KiB norms + config + {copied_assets} \
         tokenizer files, {bpw:.4} bpw)",
        names.len(),
        total_params as f64 / 1e6,
        cfg.ladder.planes,
        cfg.ladder.group,
        cfg.ladder.grid,
        out.display(),
        bundle.len() as f64 / (1024.0 * 1024.0),
        norms.iter().map(|(_, v)| v.len() * 4).sum::<usize>() as f64 / 1024.0,
    );
    println!(
        "  fidelity: {:.4} relative Frobenius error whole-model, measured on the decoded artifact \
         (per-tensor breakdown in {})",
        whole_model_error,
        receipt_path.display()
    );
    println!("  verified: reloaded from {}", out.display());
    Ok(())
}

/// Reconstruction fidelity of one tensor, measured on the decoded artifact.
struct TensorFidelity {
    name: String,
    rows: usize,
    cols: usize,
    /// `‖W − Ŵ‖_F / ‖W‖_F`. Scale-free, so tensors of very different magnitudes are comparable —
    /// which raw SSE is not, and which is the whole point of reporting this per tensor.
    relative_frobenius: f64,
    /// `Σ(W − Ŵ)²` and `Σ W²`, kept so the whole-model figure is a true aggregate rather than an
    /// average of ratios (which would weight a 576-element tensor like a 100M-element one).
    sq_error: f64,
    sq_weight: f64,
}

impl TensorFidelity {
    fn measure(name: &str, rows: usize, cols: usize, w: &[f32], decoded: &[f32]) -> Result<Self> {
        if decoded.len() != w.len() {
            bail!(
                "{name}: decoded {} values for a {}-value tensor — the packer and the decoder \
                 disagree about this tensor's layout",
                decoded.len(),
                w.len()
            );
        }
        let mut sq_error = 0.0f64;
        let mut sq_weight = 0.0f64;
        for (&a, &b) in w.iter().zip(decoded) {
            let d = f64::from(a) - f64::from(b);
            sq_error += d * d;
            sq_weight += f64::from(a) * f64::from(a);
        }
        Ok(Self {
            name: name.to_owned(),
            rows,
            cols,
            relative_frobenius: if sq_weight > 0.0 {
                (sq_error / sq_weight).sqrt()
            } else {
                0.0
            },
            sq_error,
            sq_weight,
        })
    }
}

/// Write the fidelity receipt.
///
/// Emitted by default rather than behind a flag: a quantized model with no record of *how* it was
/// quantized and *how much* it lost is the thing that makes quantized checkpoints untrustworthy,
/// and the receipt is cheap — it is decoded from the artifact that was just written.
#[allow(clippy::too_many_arguments)]
fn write_receipt(
    path: &Path,
    source: &Path,
    cfg: &ConvertConfig,
    fold_desc: &str,
    fidelity: &[TensorFidelity],
    total_params: usize,
    bundle_bytes: usize,
) -> Result<f64> {
    let sq_error: f64 = fidelity.iter().map(|f| f.sq_error).sum();
    let sq_weight: f64 = fidelity.iter().map(|f| f.sq_weight).sum();
    let whole_model = if sq_weight > 0.0 {
        (sq_error / sq_weight).sqrt()
    } else {
        0.0
    };

    let tensors: Vec<serde_json::Value> = fidelity
        .iter()
        .map(|f| {
            serde_json::json!({
                "name": f.name,
                "shape": [f.rows, f.cols],
                "relative_frobenius_error": f.relative_frobenius,
            })
        })
        .collect();

    let receipt = serde_json::json!({
        "source": {
            "directory": source.display().to_string(),
            "config_sha256": file_digest(&source.join("config.json"))?,
        },
        "recipe": {
            "fitter": "geometric ladder",
            "planes": cfg.ladder.planes,
            "group": cfg.ladder.group,
            "grid": cfg.ladder.grid,
            // Recorded explicitly because its absence is load-bearing: the SALT bundle has nowhere
            // to store a rotation matrix, so a rotated fit would reconstruct W*H instead of W.
            "rotation": "none",
            "fold_alpha": if cfg.calib.is_some() { cfg.fold_alpha } else { 0.0 },
            "calibration": fold_desc,
        },
        "cost": {
            "parameters": total_params,
            "bits_per_weight": cfg.ladder.realizable_bpw(),
            "bundle_bytes": bundle_bytes,
            "container": "TQ2_0 planes (2 bits/trit + one f16 scale per 256)",
        },
        "fidelity": {
            "whole_model_relative_frobenius_error": whole_model,
            "measured_on": "the decoded artifact, including f16 block-scale rounding",
            // Load-bearing caveat, in the artifact rather than only in the docs, because the
            // receipt is what a downstream consumer actually reads. Measured on SmolLM2-135M at
            // T=4: alpha=0 gives 0.0252 and alpha=0.75 gives 0.0380 -- the fold is 51% WORSE by
            // this metric while being better by perplexity, because it deliberately trades
            // weight-space error for error where activations are large. This repo has already
            // measured the same inversion for per-group plane allocation (12.4% lower weight SSE,
            // 12% worse perplexity). Comparing this number across recipes ranks them backwards.
            "interpretation": "Weight-space error against the fp master. Valid for comparing \
                               TENSORS within this model, and for detecting a corrupt conversion. \
                               NOT a quality ranking across recipes: a larger fold_alpha \
                               increases this number and improves perplexity, because the fold \
                               trades weight-space error for error where activations are small.",
            "tensors": tensors,
        },
    });

    std::fs::write(path, serde_json::to_vec_pretty(&receipt)?)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(whole_model)
}

fn file_digest(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(crate::hex::hex_digest(
        &<sha2::Sha256 as sha2::Digest>::digest(&bytes),
    ))
}

/// Load calibration token ids.
///
/// Two accepted forms, distinguished by content rather than by extension so a renamed file cannot
/// be silently misread:
///
/// - a JSON object with a `train_ids` array of integers — the format the research corpora use, so
///   a `convert` run can be compared against a harness run on identical tokens;
/// - anything else: UTF-8 text, tokenized with the source model's `tokenizer.json`.
fn load_calibration_tokens(
    path: &Path,
    model: &Path,
    limit: usize,
    vocab: usize,
) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;

    let ids: Vec<u32> = if let Some(ids) = parse_train_ids(&bytes)? {
        ids
    } else {
        let text = String::from_utf8(bytes).with_context(|| {
            format!(
                "{} is neither a corpus JSON with `train_ids` nor UTF-8 text",
                path.display()
            )
        })?;
        tokenize_with_model(model, &text)?
    };

    if let Some(&bad) = ids.iter().find(|&&id| id as usize >= vocab) {
        bail!(
            "calibration corpus {} contains token id {bad}, outside the model's vocabulary of \
             {vocab} — it was tokenized for a different model",
            path.display()
        );
    }
    Ok(ids.into_iter().take(limit).collect())
}

/// Recognize the research corpus format. `Ok(None)` means "not that format"; an error means it
/// looked like one and was malformed, which is worth reporting rather than falling through to a
/// nonsense tokenization.
fn parse_train_ids(bytes: &[u8]) -> Result<Option<Vec<u32>>> {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return Ok(None);
    };
    let Some(array) = value.get("train_ids") else {
        return Ok(None);
    };
    let array = array
        .as_array()
        .context("corpus JSON has `train_ids` but it is not an array")?;
    array
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .context("`train_ids` must contain non-negative integers that fit in u32")
        })
        .collect::<Result<Vec<u32>>>()
        .map(Some)
}

/// Tokenize raw calibration text with the **source model's own** tokenizer.
///
/// Using the model's tokenizer rather than any default is not a convenience: token ids are only
/// meaningful relative to the vocabulary they were produced for, and calibrating on ids from a
/// different tokenizer would measure the activation statistics of gibberish while looking
/// perfectly healthy.
fn tokenize_with_model(model: &Path, text: &str) -> Result<Vec<u32>> {
    let tokenizer_json = model.join("tokenizer.json");
    let tokenizer_config = model.join("tokenizer_config.json");
    if !tokenizer_json.exists() {
        bail!(
            "calibration text needs a tokenizer, but {} has no tokenizer.json. Pass a corpus JSON \
             with a `train_ids` array instead (the format under ~/.cache/tritium-corpora).",
            model.display()
        );
    }
    let tokenizer = HfJsonTokenizer::from_files(&tokenizer_json, &tokenizer_config)
        .with_context(|| format!("load tokenizer from {}", model.display()))?;
    tokenizer
        .encode(text)
        .map_err(|e| anyhow::anyhow!("tokenize calibration text: {e}"))
}

/// Minimal f32 safetensors writer for the fp side-file.
///
/// Deliberately not reusing a general writer: this file has exactly one job — carry the 1-D norms
/// that `load_salt` reads from the fp master — and every tensor in it is a 1-D f32 vector.
fn write_f32_safetensors(tensors: &[(String, &[f32])]) -> Vec<u8> {
    // BTreeMap so the header is deterministic: two runs on the same model produce byte-identical
    // files, which is what makes an artifact diffable and hashable.
    let mut entries: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut offset = 0usize;
    for (name, values) in tensors {
        let end = offset + values.len() * 4;
        entries.insert(name.as_str(), (offset, end));
        offset = end;
    }

    let header = entries
        .iter()
        .map(|(name, (start, end))| {
            format!(
                r#""{name}":{{"dtype":"F32","shape":[{}],"data_offsets":[{start},{end}]}}"#,
                (end - start) / 4
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let mut header = format!("{{{header}}}").into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }

    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    // Payload order must match the offsets, which came from `tensors` order, not the sorted header.
    for (_, values) in tensors {
        for &v in *values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

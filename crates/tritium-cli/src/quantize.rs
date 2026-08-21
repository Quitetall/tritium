//! `tritium quantize` — offline SALT quantization of an fp model to a SALT bundle.
//!
//! Reads an fp16/bf16/f32 safetensors model, SALT-quantizes every 2D weight tensor at a
//! target bits-per-weight, and writes a single-file SALT artifact. `sidecar` preserves
//! the dense v1 bundle; `sidecar-progressive` stores compact sparse residuals. 1D tensors
//! (norms, biases) are copied-through conceptually but not quantized here — the bundle
//! holds only the quantized matrices.
//!
//! `--scale-group tensor` uses one per-tensor AbsMean for the base plane (matches deployed
//! BitNet b1.58 I2_S; required for a b1.58 master); `block` (default) is per-256-block, best
//! for a normally-trained fp master.

use std::path::Path;

use anyhow::{Context, Result};
use clap::ValueEnum;
use tritium_format::{
    DEFAULT_SPARSE_RESIDUAL_DENSITY, SafeTensors, SaltRow, write_progressive_salt_bundle,
    write_salt_bundle, write_salt_gguf,
};
use tritium_quantize::fisher::tile_sensitivity;
use tritium_quantize::{BaseScaleScope, QuantConfig, Sensitivity, quantize_tensor};

use crate::SaltSensitivityArg;
use crate::quantize_ladder::{LadderArg, LadderConfig, quantize_tensor_ladder};

/// Base-plane scale granularity (CLI mirror of [`BaseScaleScope`]).
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum ScaleGroupArg {
    /// Per-256-block AbsMean (default; best for a normally-trained fp master).
    Block,
    /// Per-tensor AbsMean base plane (matches deployed BitNet b1.58 I2_S).
    Tensor,
}

/// Output container for the quantized model.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum OutputFormat {
    /// Single-file SALT bundle (`.tslb`).
    Sidecar,
    /// Progressive SALT bundle with sparse residual planes (`.tslb`).
    SidecarProgressive,
    /// A GGUF container holding the SALT rows (tritium-private tensor type).
    Gguf,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    input: &Path,
    output: &Path,
    bpw: f64,
    scale_group: ScaleGroupArg,
    sensitivity: SaltSensitivityArg,
    fisher: Option<&Path>,
    format: OutputFormat,
    ladder: LadderArg,
    ladder_cfg: LadderConfig,
) -> Result<()> {
    if ladder == LadderArg::Geometric {
        ladder_cfg.validate()?;
        // The ladder assigns every plane from one anchor; there is no per-group plane budget for
        // a Fisher or energy sensitivity to allocate. Silently ignoring these would be a knob that
        // does nothing — refuse instead.
        if fisher.is_some() {
            anyhow::bail!(
                "--fisher allocates planes per group and only applies to --ladder itf; the \
                 geometric ladder gives every group the same plane count. Pass --ladder itf, or \
                 drop --fisher."
            );
        }
        if !matches!(sensitivity, SaltSensitivityArg::Uniform) {
            anyhow::bail!(
                "--sensitivity {sensitivity:?} allocates planes per group and only applies to \
                 --ladder itf; the geometric ladder gives every group the same plane count."
            );
        }
    }
    let sg = match scale_group {
        ScaleGroupArg::Block => BaseScaleScope::Block,
        ScaleGroupArg::Tensor => BaseScaleScope::Tensor,
    };
    let sens = match sensitivity {
        SaltSensitivityArg::Uniform => Sensitivity::Uniform,
        SaltSensitivityArg::Energy => Sensitivity::Energy,
    };

    let bytes = std::fs::read(input).with_context(|| format!("read {}", input.display()))?;
    let st = SafeTensors::parse(&bytes).context("parse safetensors")?;

    // Optional diagonal-Fisher sensitivity sidecar (plan 0039). When present it overrides
    // `--sensitivity`: each weight is allocated by loss curvature per 256-block tile.
    let fisher_bytes = fisher
        .map(|p| std::fs::read(p).with_context(|| format!("read Fisher sidecar {}", p.display())))
        .transpose()?;
    let fisher_st = fisher_bytes
        .as_ref()
        .map(|b| SafeTensors::parse(b).context("parse Fisher sidecar safetensors"))
        .transpose()?;

    // Quantize every 2D weight matrix; skip 1D tensors (norms/biases) and degenerate shapes.
    let names: Vec<String> = st
        .names()
        .filter(|n| {
            st.shape(n)
                .is_some_and(|s| s.len() == 2 && s[0] >= 2 && s[1] >= 2)
        })
        .map(str::to_owned)
        .collect();
    if names.is_empty() {
        anyhow::bail!("no 2D weight tensors found in {}", input.display());
    }

    let mut quantized: Vec<(String, Vec<SaltRow>)> = Vec::with_capacity(names.len());
    let mut total_params = 0usize;
    let mut total_bits = 0.0f64;
    for name in &names {
        let shape = st.shape(name).expect("filtered to Some");
        let (rows, k) = (shape[0], shape[1]);
        let w = st
            .tensor_f32(name)
            .with_context(|| format!("read tensor {name}"))?;
        let sensitivity = match &fisher_st {
            Some(fst) => {
                // The Fisher must be present with the SAME shape as the weight — not merely the same
                // element count: a transposed [k, rows] sidecar would pass a length check yet mis-map
                // every per-tile sensitivity (silent-wrong), so compare the shape exactly.
                match fst.shape(name) {
                    None => anyhow::bail!(
                        "Fisher sidecar has no tensor {name} (every quantized weight needs a Fisher entry)"
                    ),
                    Some([r, c]) if *r == rows && *c == k => {}
                    Some(s) => anyhow::bail!(
                        "Fisher for {name} has shape {s:?}, expected [{rows}, {k}] (the weight's shape)"
                    ),
                }
                let f = fst
                    .tensor_f32(name)
                    .with_context(|| format!("read Fisher for tensor {name}"))?;
                if let Some(bad) = f.iter().find(|v| !v.is_finite() || **v < 0.0) {
                    anyhow::bail!("Fisher for {name} must be finite and ≥ 0; found {bad}");
                }
                let f64v: Vec<f64> = f.iter().map(|&x| f64::from(x)).collect();
                Sensitivity::Custom(tile_sensitivity(&f64v, rows, k))
            }
            None => sens.clone(),
        };
        let cfg = QuantConfig {
            budget_bpw: bpw,
            t_min: 1,
            t_max: 3,
            sensitivity,
            scale_group: sg,
        };
        match ladder {
            LadderArg::Geometric => {
                let rows_out = quantize_tensor_ladder(&w, rows, k, &ladder_cfg)
                    .with_context(|| format!("ladder-quantize {name}"))?;
                total_params += rows * k;
                total_bits += ladder_cfg.realizable_bpw() * (rows * k) as f64;
                quantized.push((name.clone(), rows_out));
            }
            LadderArg::Itf => {
                let qt = quantize_tensor(&w, rows, k, &cfg)
                    .with_context(|| format!("quantize {name}"))?;
                total_params += rows * k;
                total_bits += qt.logical_bpw() * (rows * k) as f64;
                quantized.push((name.clone(), qt.salt_rows));
            }
        }
    }

    let refs: Vec<(&str, &[SaltRow])> = quantized
        .iter()
        .map(|(n, r)| (n.as_str(), r.as_slice()))
        .collect();
    let (out_bytes, container) = match format {
        OutputFormat::Sidecar => (
            write_salt_bundle(&refs).context("serialize SALT bundle")?,
            "SALT bundle",
        ),
        OutputFormat::SidecarProgressive => (
            write_progressive_salt_bundle(&refs, DEFAULT_SPARSE_RESIDUAL_DENSITY)
                .context("serialize progressive SALT bundle")?,
            "progressive SALT bundle",
        ),
        OutputFormat::Gguf => (
            write_salt_gguf(&refs).context("serialize SALT GGUF")?,
            "SALT GGUF",
        ),
    };
    std::fs::write(output, &out_bytes).with_context(|| format!("write {}", output.display()))?;

    let avg_bpw = if total_params > 0 {
        total_bits / total_params as f64
    } else {
        0.0
    };
    let sens_desc = if fisher.is_some() {
        "Fisher".to_string()
    } else {
        format!("{sensitivity:?}")
    };
    // Report the recipe that actually ran. The ladder ignores --bpw (its rate is fixed by the
    // plane count and the container), so printing a "bpw target" for it would describe a knob that
    // had no effect — the same class of mislabelling that hides configuration mismatches.
    let recipe = match ladder {
        LadderArg::Geometric => format!(
            "ladder geometric, {} planes, g{}, grid {}, no rotation, no calibration fold",
            ladder_cfg.planes, ladder_cfg.group, ladder_cfg.grid
        ),
        LadderArg::Itf => format!(
            "itf free-scale, {bpw:.3} bpw target, {scale_group:?} scale, {sens_desc} sensitivity"
        ),
    };
    println!(
        "quantized {} tensors ({:.2}M params) | {recipe} → {} {} ({:.1} MiB, {:.3} avg bpw)",
        names.len(),
        total_params as f64 / 1e6,
        container,
        output.display(),
        out_bytes.len() as f64 / (1024.0 * 1024.0),
        avg_bpw,
    );
    if ladder == LadderArg::Geometric {
        println!(
            "  note: a SALT bundle stores each plane as TQ2_0 (2.0625 bits/trit incl. its f16 \
             scale), so this file costs {:.4} bpw. The ladder's one-anchor-per-group saving and \
             the 1.625 B3 rate are NOT expressible in this container; no bits-per-weight claim \
             against integer quantization is made here.",
            ladder_cfg.realizable_bpw()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_accepts_progressive_sidecar_name() {
        assert!(matches!(
            OutputFormat::from_str("sidecar-progressive", true),
            Ok(OutputFormat::SidecarProgressive)
        ));
    }
}

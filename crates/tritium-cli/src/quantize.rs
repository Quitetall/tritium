//! `tritium quantize` — offline SALT quantization of an fp model to a SALT bundle.
//!
//! Reads an fp16/bf16/f32 safetensors model, SALT-quantizes every 2D weight tensor at a
//! target bits-per-weight, and writes a single-file [`write_salt_bundle`] artifact. 1D
//! tensors (norms, biases) are copied-through conceptually but not quantized here — the
//! bundle holds only the quantized matrices.
//!
//! `--scale-group tensor` uses one per-tensor AbsMean for the base plane (matches deployed
//! BitNet b1.58 I2_S; required for a b1.58 master); `block` (default) is per-256-block, best
//! for a normally-trained fp master.

use std::path::Path;

use anyhow::{Context, Result};
use clap::ValueEnum;
use tritium_format::{SafeTensors, SaltRow, write_salt_bundle, write_salt_gguf};
use tritium_quantize::{QuantConfig, ScaleGroup, Sensitivity, quantize_tensor};

/// Base-plane scale granularity (CLI mirror of [`ScaleGroup`]).
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
    /// A GGUF container holding the SALT rows (tritium-private tensor type).
    Gguf,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    input: &Path,
    output: &Path,
    bpw: f64,
    scale_group: ScaleGroupArg,
    format: OutputFormat,
) -> Result<()> {
    let sg = match scale_group {
        ScaleGroupArg::Block => ScaleGroup::Block,
        ScaleGroupArg::Tensor => ScaleGroup::Tensor,
    };

    let bytes = std::fs::read(input).with_context(|| format!("read {}", input.display()))?;
    let st = SafeTensors::parse(&bytes).context("parse safetensors")?;

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
        let cfg = QuantConfig {
            budget_bpw: bpw,
            t_min: 1,
            t_max: 3,
            sensitivity: Sensitivity::Uniform,
            scale_group: sg,
        };
        let qt = quantize_tensor(&w, rows, k, &cfg).with_context(|| format!("quantize {name}"))?;
        total_params += rows * k;
        total_bits += qt.logical_bpw() * (rows * k) as f64;
        quantized.push((name.clone(), qt.salt_rows));
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
    println!(
        "quantized {} tensors ({:.2}M params) at {:.3} bpw target, {:?} scale → {} {} ({:.1} MiB, {:.3} avg bpw)",
        names.len(),
        total_params as f64 / 1e6,
        bpw,
        scale_group,
        container,
        output.display(),
        out_bytes.len() as f64 / (1024.0 * 1024.0),
        avg_bpw,
    );
    Ok(())
}

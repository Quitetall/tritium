//! Balanced-ternary geometric ladder for `tritium quantize`.
//!
//! The ladder assigns all `T` planes from one anchor per group: `s_p = s₀·3^-(p-1)`, so the
//! reachable levels are a uniform grid over every integer in `±(3^T−1)/2` and the fit is one
//! `round()` plus base-3 digit extraction. It replaces the free-scale ITF fitter, which picks each
//! plane's scale from the residual left by the last and whose ratios drift upward until a later
//! plane takes a *larger* scale than its predecessor.
//!
//! # What this path can and cannot express — measured, not assumed
//!
//! Two of the levers behind the published SALT numbers **cannot be represented in a SALT bundle**,
//! so the artifact this writes is a different configuration from the research runs and is measured
//! as such:
//!
//! - **No salience fold.** The AWQ-style fold needs activation statistics from a calibration
//!   corpus; `tritium quantize` reads a safetensors file and nothing else.
//! - **No rotation.** [`RotationPolicy::Never`] is the only correct choice here: the ladder fits in
//!   the rotated basis, and the bundle carries no rotation metadata, so a rotated artifact would
//!   silently reconstruct `W·H` instead of `W`.
//!
//! Measured on SmolLM2-360M, WikiText-2 32,768-token held-out (fp 14.909), in **exactly this
//! configuration** — no fold, no rotation, `g256`:
//!
//! | planes | ladder | ITF (old fitter) |
//! |---|---|---|
//! | 2 | 4820.2 (323×) | 327.1 (21.9×) |
//! | 3 | **19.898 (1.335×)** | 76.560 (5.135×) |
//! | 4 | **15.268 (1.024×)** | — |
//!
//! **At `T≥3` the ladder wins by ~3.9× and costs fewer bits. At `T=2` it LOSES to ITF by 14.7×**,
//! because its fixed 1/3 spacing assumes a well-conditioned distribution and rotation is what
//! supplies that; without rotation the free-scale fitter adapts to heavy tails and the rigid grid
//! cannot. Both `T=2` settings are unusable in absolute terms (21.9× and 323× fp), so the practical
//! rule is `T≥3`, which [`LadderConfig::validate`] enforces.
//!
//! # Byte accounting
//!
//! A SALT bundle stores each plane as TQ2_0: 64 bytes of trits plus one f16 scale per 256-trit
//! block = **2.0625 bits/trit/plane**. The 1.625 bits/trit B3 rate and the ladder's
//! one-anchor-per-group saving are *not* realizable in this container — both it and SALT V2 are
//! per-plane-scaled. This path therefore makes no bits-per-weight claim against integer
//! quantization; it reports what the file actually costs.

use anyhow::{Result, bail};
use clap::ValueEnum;
use half::f16;
use tritium_core::Trit;
use tritium_format::{QK_K, SaltRow, TQ2_0_BLOCK_BYTES, pack_tq2_0_block};
use tritium_train::ops::ste::{self, RotationPolicy};

/// Which fitter `tritium quantize` drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum LadderArg {
    /// Balanced-ternary geometric ladder (default): one anchor per group, `s_p = s₀·3^-(p-1)`.
    Geometric,
    /// The previous free-scale iterative ternary fit. Kept so every published number stays
    /// reproducible, and because it beats the ladder below 3 planes without rotation.
    Itf,
}

/// Resolved ladder settings for one run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LadderConfig {
    pub(crate) planes: usize,
    pub(crate) group: usize,
    pub(crate) grid: usize,
}

impl LadderConfig {
    /// Reject configurations this path cannot write correctly or should not write at all.
    pub(crate) fn validate(&self) -> Result<()> {
        if !(1..=9).contains(&self.planes) {
            bail!(
                "--planes must be 1..=9 (3^9 = 19683 is the largest level count exact in f32), got {}",
                self.planes
            );
        }
        // A TQ2_0 block carries ONE f16 scale for 256 trits. The ladder's scale is per group, so a
        // block straddling two groups would need two anchors and could not be encoded.
        if !self.group.is_multiple_of(QK_K) {
            bail!(
                "--group must be a multiple of {QK_K} for the SALT bundle: a TQ2_0 block holds one \
                 f16 scale per {QK_K} trits, so a smaller group would put two anchors in one block \
                 (got {})",
                self.group
            );
        }
        if self.planes < 3 {
            bail!(
                "--ladder geometric with --planes {} is a footgun without rotation: measured 323x fp \
                 at T=2 on SmolLM2-360M versus 21.9x for --ladder itf, because the ladder's fixed \
                 1/3 spacing needs the well-conditioned distribution that a Hadamard rotation \
                 supplies, and a SALT bundle cannot carry rotation metadata. Use --planes 3 or more \
                 (T=4 measures 1.024x fp), or --ladder itf if you must go lower.",
                self.planes
            );
        }
        Ok(())
    }

    /// Realizable bits per weight in a SALT bundle: TQ2_0 charges 2 bits/trit plus one f16 scale
    /// per 256-trit block, **per plane**. Deliberately not the B3 rate, which this container does
    /// not use.
    pub(crate) fn realizable_bpw(&self) -> f64 {
        let per_plane = 2.0 + 16.0 / QK_K as f64;
        self.planes as f64 * per_plane
    }
}

/// Fit one 2-D weight tensor with the ladder and pack it into per-output-channel [`SaltRow`]s.
///
/// `wf` is row-major `[rows, cols]`. Rotation is fixed to [`RotationPolicy::Never`] — see the module
/// docs; passing anything else would produce an artifact that reconstructs the wrong weights.
pub(crate) fn quantize_tensor_ladder(
    wf: &[f32],
    rows: usize,
    cols: usize,
    cfg: &LadderConfig,
) -> Result<Vec<SaltRow>> {
    let groups_per_row = cols.div_ceil(cfg.group);
    let fits = ste::geometric_ladder_fit(
        wf,
        rows,
        cols,
        cfg.planes,
        cfg.group,
        cfg.grid,
        RotationPolicy::Never,
    );
    if fits.len() != rows * groups_per_row {
        bail!(
            "ladder fit returned {} groups, expected {} ({rows} rows x {groups_per_row} groups)",
            fits.len(),
            rows * groups_per_row
        );
    }

    let blocks_per_row = cols.div_ceil(QK_K);
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut planes: Vec<Vec<u8>> = Vec::with_capacity(cfg.planes);
        for p in 0..cfg.planes {
            let mut packed = vec![0u8; blocks_per_row * TQ2_0_BLOCK_BYTES];
            for b in 0..blocks_per_row {
                let start = b * QK_K;
                let len = QK_K.min(cols - start);
                // `group % QK_K == 0` (validated), so every trit in this block belongs to one
                // group and therefore shares one anchor.
                let g = start / cfg.group;
                let (s0, group_planes) = &fits[r * groups_per_row + g];

                // s_p = s0 * 3^-p, built by repeated division so the exponent is exact.
                let mut scale = f64::from(*s0);
                for _ in 0..p {
                    scale /= 3.0;
                }

                let off = start - g * cfg.group;
                let mut trits = [Trit::ZERO; QK_K];
                for (i, slot) in trits.iter_mut().enumerate().take(len) {
                    let v = group_planes[p].get(off + i).copied().unwrap_or(0);
                    *slot = Trit::from_i8(v).expect("ladder digits are in {-1,0,1}");
                }
                let block = &mut packed[b * TQ2_0_BLOCK_BYTES..(b + 1) * TQ2_0_BLOCK_BYTES];
                pack_tq2_0_block(&trits, f16::from_f64(scale), block)
                    .map_err(|e| anyhow::anyhow!("pack TQ2_0 block {b} of plane {p}: {e}"))?;
            }
            planes.push(packed);
        }
        out.push(SaltRow { k: cols, planes });
    }
    Ok(out)
}

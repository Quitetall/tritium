//! Tensor-level SALT quantization — the offline bridge from fp32 weights to
//! packed TQ2_0 residual-sidecar rows (ADR 0001/0006).
//!
//! Ties the three foundation pieces together for a whole weight matrix:
//! 1. split each row into 256-block **groups** (the kernel-tile granularity);
//! 2. greedily [`residual_expand`] each group and [`allocate`] planes across all
//!    groups under one bits-per-weight budget (rate-distortion water-filling);
//! 3. assemble each row's planes into a [`SaltRow`] of standard TQ2_0 rows.
//!
//! The result dequantizes (via [`tritium_format::dequant_salt_row`]) to exactly
//! the f16-scaled plane sum — the same lossy f16 scale the legacy TQ2_0 pipeline
//! already uses, so `budget = base` reproduces flat BitNet AbsMean bit-for-bit.
//!
//! **Storage note:** this foundation stores *dense* planes — a row carries
//! `max_b T_b` planes, padding lower-`T` blocks with zero trits/scale (which
//! contribute nothing on dequant). [`QuantizedTensor::logical_bpw`] reports the
//! allocator's bpw (`≤ budget`); the *sparse* residual plane that makes on-disk
//! bytes match that figure is the later GPU-gated step (ADR 0001 §5).

use half::f16;
use tritium_core::Trit;
use tritium_format::{FormatError, QK_K, SaltRow, TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row};

use crate::{AllocConfig, AllocError, GroupInput, allocate, residual_expand};

/// How to score each group's loss sensitivity `H_g` for the allocator.
#[derive(Clone, Copy, Debug, Default)]
pub enum Sensitivity<'a> {
    /// All groups equally sensitive — allocate purely by reconstruction-error
    /// reduction (minimize total MSE under the budget). The default.
    #[default]
    Uniform,
    /// Group energy `‖w_g‖²` — a magnitude proxy when no Hessian is available.
    Energy,
    /// Caller-supplied per-group sensitivities (e.g. a GPTQ Hessian diagonal),
    /// row-major `r·nb + b`. Must have exactly one entry per group.
    Custom(&'a [f64]),
}

/// Knobs for [`quantize_tensor`].
#[derive(Clone, Copy, Debug)]
pub struct QuantConfig<'a> {
    /// Target **average bits-per-weight** (`≥ t_min·log2 3`; `1.585` = all base).
    pub budget_bpw: f64,
    /// Dense base planes every group gets (default `1`).
    pub t_min: usize,
    /// Plane cap per group (default `3`, tile-uniform `{1,2,3}`).
    pub t_max: usize,
    /// Sensitivity scoring.
    pub sensitivity: Sensitivity<'a>,
}

impl Default for QuantConfig<'_> {
    fn default() -> Self {
        Self {
            budget_bpw: 2.0,
            t_min: 1,
            t_max: 3,
            sensitivity: Sensitivity::Uniform,
        }
    }
}

/// A SALT-quantized weight matrix: one [`SaltRow`] per output channel.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct QuantizedTensor {
    /// Output channels (rows of the original matrix).
    pub rows: usize,
    /// Input features per row (`K`).
    pub k: usize,
    /// One packed SALT row per output channel.
    pub salt_rows: Vec<SaltRow>,
    /// Allocated plane count per group, row-major `r·nb + b`.
    pub plane_counts: Vec<usize>,
}

impl QuantizedTensor {
    /// The allocator's logical bits-per-weight (`Σ|g|·log2(3)·T_g / N`), which is
    /// `≤ budget_bpw`. Dense storage may exceed this until the sparse plane lands.
    pub fn logical_bpw(&self) -> f64 {
        let total: usize = self.rows * self.k;
        if total == 0 {
            return 0.0;
        }
        let nb = num_blocks(self.k);
        let bits: f64 = self
            .plane_counts
            .iter()
            .enumerate()
            .map(|(g, &t)| {
                // group g = (row, block); its size is 256 except the last block.
                let start = (g % nb) * QK_K;
                let size = (start + QK_K).min(self.k) - start;
                size as f64 * crate::TRIT_BITS * t as f64
            })
            .sum();
        bits / total as f64
    }
}

/// Why a tensor could not be quantized.
#[derive(Clone, Debug, PartialEq)]
pub enum QuantError {
    /// `weights.len()` ≠ `rows · k`.
    ShapeMismatch {
        rows: usize,
        k: usize,
        got: usize,
    },
    /// [`Sensitivity::Custom`] had the wrong number of entries.
    SensitivityLen {
        expected: usize,
        got: usize,
    },
    /// The plane allocator rejected the inputs.
    Alloc(AllocError),
    /// A plane failed to pack.
    Format(FormatError),
}

impl core::fmt::Display for QuantError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QuantError::ShapeMismatch { rows, k, got } => {
                write!(f, "shape mismatch: {rows}×{k} needs {} weights, got {got}", rows * k)
            }
            QuantError::SensitivityLen { expected, got } => {
                write!(f, "custom sensitivity: expected {expected} entries, got {got}")
            }
            QuantError::Alloc(e) => write!(f, "allocation: {e}"),
            QuantError::Format(e) => write!(f, "format: {e}"),
        }
    }
}

impl std::error::Error for QuantError {}

impl From<AllocError> for QuantError {
    fn from(e: AllocError) -> Self {
        QuantError::Alloc(e)
    }
}
impl From<FormatError> for QuantError {
    fn from(e: FormatError) -> Self {
        QuantError::Format(e)
    }
}

/// Quantize a row-major `rows × k` fp32 weight matrix to packed SALT rows.
///
/// # Errors
/// [`QuantError::ShapeMismatch`] if `weights.len() != rows·k`;
/// [`QuantError::SensitivityLen`] on a mis-sized [`Sensitivity::Custom`];
/// propagates [`AllocError`] / [`FormatError`].
pub fn quantize_tensor(
    weights: &[f32],
    rows: usize,
    k: usize,
    cfg: &QuantConfig,
) -> Result<QuantizedTensor, QuantError> {
    if weights.len() != rows.saturating_mul(k) {
        return Err(QuantError::ShapeMismatch {
            rows,
            k,
            got: weights.len(),
        });
    }
    let nb = num_blocks(k);
    let n_groups = rows * nb;
    if let Sensitivity::Custom(s) = cfg.sensitivity
        && s.len() != n_groups
    {
        return Err(QuantError::SensitivityLen {
            expected: n_groups,
            got: s.len(),
        });
    }

    // Build group slices + sensitivities (row-major over blocks).
    let mut groups: Vec<GroupInput> = Vec::with_capacity(n_groups);
    for r in 0..rows {
        let row = &weights[r * k..r * k + k];
        for b in 0..nb {
            let start = b * QK_K;
            let end = (start + QK_K).min(k);
            let gw = &row[start..end];
            let sensitivity = match cfg.sensitivity {
                Sensitivity::Uniform => 1.0,
                Sensitivity::Energy => gw.iter().map(|&x| x as f64 * x as f64).sum(),
                Sensitivity::Custom(s) => s[r * nb + b],
            };
            groups.push(GroupInput { weights: gw, sensitivity });
        }
    }

    let total_w = rows * k;
    let acfg = AllocConfig::from_bpw(cfg.budget_bpw, total_w, cfg.t_min, cfg.t_max);
    let alloc = allocate(&groups, &acfg)?;

    // Assemble each row's dense planes.
    let mut salt_rows = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &weights[r * k..r * k + k];
        let stacks: Vec<_> = (0..nb)
            .map(|b| {
                let start = b * QK_K;
                let end = (start + QK_K).min(k);
                residual_expand(&row[start..end], alloc.plane_counts[r * nb + b])
            })
            .collect();
        let t_row = stacks.iter().map(|s| s.plane_count()).max().unwrap_or(0);

        let mut planes = Vec::with_capacity(t_row);
        for p in 0..t_row {
            let mut row_trits = vec![Trit::ZERO; k];
            let mut row_scales = vec![f16::ZERO; nb];
            for (b, st) in stacks.iter().enumerate() {
                if p < st.plane_count() {
                    let plane = &st.planes[p];
                    let start = b * QK_K;
                    row_scales[b] = f16::from_f32(plane.scale);
                    row_trits[start..start + plane.trits.len()].copy_from_slice(&plane.trits);
                }
            }
            let mut bytes = vec![0u8; nb * TQ2_0_BLOCK_BYTES];
            pack_tq2_0_row(&row_trits, &row_scales, &mut bytes)?;
            planes.push(bytes);
        }
        salt_rows.push(SaltRow { k, planes });
    }

    Ok(QuantizedTensor {
        rows,
        k,
        salt_rows,
        plane_counts: alloc.plane_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::absmean_ternary;
    use tritium_format::dequant_salt_row;

    /// Deterministic pseudo-random fp32 tensor (LCG → [-2, 2)).
    fn make_tensor(rows: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..rows * k)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 2.0
            })
            .collect()
    }

    /// Independent reference: dequant by re-expanding each group with f16 scales.
    fn reference_dequant(weights: &[f32], rows: usize, k: usize, plane_counts: &[usize]) -> Vec<f32> {
        let nb = num_blocks(k);
        let mut out = vec![0.0f32; rows * k];
        for r in 0..rows {
            for b in 0..nb {
                let start = b * QK_K;
                let end = (start + QK_K).min(k);
                let gw = &weights[r * k + start..r * k + end];
                let stack = residual_expand(gw, plane_counts[r * nb + b]);
                for plane in &stack.planes {
                    let s = f16::from_f32(plane.scale).to_f32();
                    for (i, t) in plane.trits.iter().enumerate() {
                        out[r * k + start + i] += s * t.to_f32();
                    }
                }
            }
        }
        out
    }

    // ── End-to-end gate: dequant(quantized) == independent reference, bit-exact.
    // Catches assembly bugs (block offsets, scale placement, partial-block pad).
    #[test]
    fn dequant_matches_reference() {
        for &(rows, k) in &[(1usize, 256usize), (3, 700), (4, 256), (2, 257), (5, 1000)] {
            let w = make_tensor(rows, k, 0xC0FFEE ^ (k as u64));
            let cfg = QuantConfig { budget_bpw: 2.6, ..Default::default() };
            let qt = quantize_tensor(&w, rows, k, &cfg).unwrap();
            let reference = reference_dequant(&w, rows, k, &qt.plane_counts);
            for (r, salt) in qt.salt_rows.iter().enumerate() {
                let deq = dequant_salt_row(salt).unwrap();
                for i in 0..k {
                    assert_eq!(
                        deq[i].to_bits(),
                        reference[r * k + i].to_bits(),
                        "rows={rows} k={k} row {r} elem {i}"
                    );
                }
            }
        }
    }

    // ── Gate: budget == base ⇒ T=1 everywhere == flat AbsMean (BitNet floor). ─
    #[test]
    fn budget_floor_is_flat_absmean() {
        let (rows, k) = (3usize, 512usize);
        let w = make_tensor(rows, k, 0xABBA);
        // budget_bpw exactly the base (t_min=1 ⇒ log2 3 bpw)
        let cfg = QuantConfig { budget_bpw: crate::TRIT_BITS, t_min: 1, t_max: 3, sensitivity: Sensitivity::Uniform };
        let qt = quantize_tensor(&w, rows, k, &cfg).unwrap();
        assert!(qt.plane_counts.iter().all(|&t| t == 1), "all groups at base");

        // Every group dequantizes to flat AbsMean with the f16-rounded scale.
        let nb = num_blocks(k);
        for r in 0..rows {
            let deq = dequant_salt_row(&qt.salt_rows[r]).unwrap();
            for b in 0..nb {
                let start = b * QK_K;
                let end = (start + QK_K).min(k);
                let gw = &w[r * k + start..r * k + end];
                let flat = absmean_ternary(gw);
                let s = f16::from_f32(flat.scale).to_f32();
                for (i, t) in flat.trits.iter().enumerate() {
                    assert_eq!(deq[start + i].to_bits(), (s * t.to_f32()).to_bits());
                }
            }
        }
    }

    // ── Gate: the allocator's logical bpw never exceeds the budget. ──────────
    #[test]
    fn logical_bpw_within_budget() {
        for &bpw in &[crate::TRIT_BITS, 2.0, 2.5, 3.0, 4.0] {
            let w = make_tensor(4, 800, 0xD00D ^ bpw.to_bits());
            let cfg = QuantConfig { budget_bpw: bpw, ..Default::default() };
            let qt = quantize_tensor(&w, 4, 800, &cfg).unwrap();
            assert!(
                qt.logical_bpw() <= bpw + 1e-9,
                "logical {} > budget {bpw}",
                qt.logical_bpw()
            );
        }
    }

    // ── Gate: determinism — same tensor+config ⇒ byte-identical packed rows. ──
    #[test]
    fn quantize_is_deterministic() {
        let w = make_tensor(3, 600, 0x1357);
        let cfg = QuantConfig { budget_bpw: 2.4, ..Default::default() };
        let a = quantize_tensor(&w, 3, 600, &cfg).unwrap();
        let b = quantize_tensor(&w, 3, 600, &cfg).unwrap();
        assert_eq!(a, b);
    }

    // Degenerate shapes must not panic: k=0 ⇒ empty rows, rows=0 ⇒ no rows.
    #[test]
    fn degenerate_shapes_are_safe() {
        let empty: [f32; 0] = [];
        let two_empty = quantize_tensor(&empty, 2, 0, &QuantConfig::default()).unwrap();
        assert_eq!(two_empty.salt_rows.len(), 2);
        assert!(two_empty.salt_rows.iter().all(|r| r.planes.is_empty()));
        assert_eq!(two_empty.logical_bpw(), 0.0);

        let no_rows = quantize_tensor(&empty, 0, 256, &QuantConfig::default()).unwrap();
        assert!(no_rows.salt_rows.is_empty());
        assert_eq!(no_rows.logical_bpw(), 0.0);
    }

    #[test]
    fn shape_mismatch_errors() {
        let w = vec![0.0f32; 10];
        assert!(matches!(
            quantize_tensor(&w, 3, 4, &QuantConfig::default()),
            Err(QuantError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn custom_sensitivity_wrong_len_errors() {
        let w = make_tensor(2, 256, 1);
        let bad = [1.0f64; 3]; // need 2 groups, gave 3
        let cfg = QuantConfig { sensitivity: Sensitivity::Custom(&bad), ..Default::default() };
        assert!(matches!(
            quantize_tensor(&w, 2, 256, &cfg),
            Err(QuantError::SensitivityLen { expected: 2, got: 3 })
        ));
    }

    // Higher budget ⇒ no group loses planes (more capacity only adds).
    #[test]
    fn higher_budget_never_reduces_planes() {
        let w = make_tensor(3, 512, 0x9999);
        let lo = quantize_tensor(&w, 3, 512, &QuantConfig { budget_bpw: 2.0, ..Default::default() }).unwrap();
        let hi = quantize_tensor(&w, 3, 512, &QuantConfig { budget_bpw: 3.5, ..Default::default() }).unwrap();
        for (l, h) in lo.plane_counts.iter().zip(&hi.plane_counts) {
            assert!(h >= l, "more budget reduced a plane count");
        }
    }
}

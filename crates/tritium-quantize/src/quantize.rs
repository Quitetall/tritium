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
use tritium_core::{Trit, absmean};
use tritium_format::{FormatError, QK_K, SaltRow, TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row};

use crate::{
    AllocConfig, AllocError, GroupInput, Plane, PlaneStack, TRIT_BITS, allocate, residual_expand,
    ternary_at_scale,
};

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

/// Granularity of the **base** (T=1) plane's AbsMean scale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScaleGroup {
    /// Per-256-block: every block fits its own AbsMean (the SALT/TQ2_0 default). Best for a
    /// normally-trained fp master — it reconstructs the weights most faithfully.
    #[default]
    Block,
    /// Per-tensor: ONE AbsMean over the whole matrix for the base plane (residual planes stay
    /// per-block). This reproduces deployed **BitNet b1.58 I2_S**, which is QAT-trained against
    /// a single per-tensor ternary scale — the per-block fit reconstructs the heavy-tailed
    /// *latent* master too faithfully and yields weights the model was never trained for. Use
    /// for b1.58 masters so SALT's floor (`budget = log2 3`) matches the deployed checkpoint.
    Tensor,
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
    /// Base-plane scale granularity (default [`ScaleGroup::Block`]).
    pub scale_group: ScaleGroup,
}

impl Default for QuantConfig<'_> {
    fn default() -> Self {
        Self {
            budget_bpw: 2.0,
            t_min: 1,
            t_max: 3,
            sensitivity: Sensitivity::Uniform,
            scale_group: ScaleGroup::Block,
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
    ShapeMismatch { rows: usize, k: usize, got: usize },
    /// [`Sensitivity::Custom`] had the wrong number of entries.
    SensitivityLen { expected: usize, got: usize },
    /// The plane allocator rejected the inputs.
    Alloc(AllocError),
    /// A plane failed to pack.
    Format(FormatError),
}

impl core::fmt::Display for QuantError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QuantError::ShapeMismatch { rows, k, got } => {
                write!(
                    f,
                    "shape mismatch: {rows}×{k} needs {} weights, got {got}",
                    rows * k
                )
            }
            QuantError::SensitivityLen { expected, got } => {
                write!(
                    f,
                    "custom sensitivity: expected {expected} entries, got {got}"
                )
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

    // Each path yields, per row, the `nb` block plane-stacks plus the realized plane count
    // per group; the assembly below is shared.
    let (row_stacks, plane_counts) = match cfg.scale_group {
        ScaleGroup::Block => expand_per_block(weights, rows, k, nb, cfg)?,
        ScaleGroup::Tensor => expand_per_tensor(weights, rows, k, nb, cfg)?,
    };

    let mut salt_rows = Vec::with_capacity(rows);
    for stacks in &row_stacks {
        salt_rows.push(assemble_salt_row(k, nb, stacks)?);
    }

    Ok(QuantizedTensor {
        rows,
        k,
        salt_rows,
        plane_counts,
    })
}

/// `[start, end)` of block `b` in a `k`-wide row (the last block is short when `k % 256 != 0`).
#[inline]
fn block_range(b: usize, k: usize) -> (usize, usize) {
    let start = b * QK_K;
    (start, (start + QK_K).min(k))
}

/// One group's loss sensitivity `H_g` for the allocator. `idx` is the row-major group index
/// `r·nb + b` (for [`Sensitivity::Custom`]).
fn group_sensitivity(sensitivity: Sensitivity, gw: &[f32], idx: usize) -> f64 {
    match sensitivity {
        Sensitivity::Uniform => 1.0,
        Sensitivity::Energy => gw.iter().map(|&x| x as f64 * x as f64).sum(),
        Sensitivity::Custom(s) => s[idx],
    }
}

/// Per-256-block SALT (the default): every group fits its own AbsMean base + greedy residual
/// planes, allocated under the budget. Returns the per-row block stacks + per-group counts.
type RowStacks = (Vec<Vec<PlaneStack>>, Vec<usize>);

fn expand_per_block(
    weights: &[f32],
    rows: usize,
    k: usize,
    nb: usize,
    cfg: &QuantConfig,
) -> Result<RowStacks, QuantError> {
    let n_groups = rows * nb;
    let mut groups: Vec<GroupInput> = Vec::with_capacity(n_groups);
    for r in 0..rows {
        let row = &weights[r * k..r * k + k];
        for b in 0..nb {
            let (start, end) = block_range(b, k);
            let gw = &row[start..end];
            groups.push(GroupInput {
                weights: gw,
                sensitivity: group_sensitivity(cfg.sensitivity, gw, r * nb + b),
            });
        }
    }
    let total_w = rows * k;
    let acfg = AllocConfig::from_bpw(cfg.budget_bpw, total_w, cfg.t_min, cfg.t_max);
    let alloc = allocate(&groups, &acfg)?;

    let mut row_stacks = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &weights[r * k..r * k + k];
        let stacks = (0..nb)
            .map(|b| {
                let (start, end) = block_range(b, k);
                residual_expand(&row[start..end], alloc.plane_counts[r * nb + b])
            })
            .collect();
        row_stacks.push(stacks);
    }
    Ok((row_stacks, alloc.plane_counts))
}

/// Per-tensor base SALT (BitNet b1.58): the T=1 base plane uses ONE AbsMean over the whole
/// matrix (so `budget = log2 3` reproduces the deployed I2_S ternary); extra planes are then
/// allocated per-256-block on the *residual* left by the base, via the same machinery.
fn expand_per_tensor(
    weights: &[f32],
    rows: usize,
    k: usize,
    nb: usize,
    cfg: &QuantConfig,
) -> Result<RowStacks, QuantError> {
    let total_w = rows * k;
    // The per-tensor base costs one plane on every group; below that floor nothing fits.
    if cfg.budget_bpw + 1e-9 < TRIT_BITS {
        return Err(QuantError::Alloc(AllocError::BudgetTooSmall {
            base_bits: total_w as f64 * TRIT_BITS,
            budget_bits: cfg.budget_bpw * total_w as f64,
        }));
    }

    // One per-tensor AbsMean scale; the base plane forces it on every element of the matrix.
    let base_scale = absmean(weights);
    let base = ternary_at_scale(weights, base_scale);
    let mut residual = weights.to_vec();
    for (r, t) in residual.iter_mut().zip(&base.trits) {
        *r -= base_scale * t.to_f32();
    }

    // Allocate the EXTRA planes (beyond the base) on the residual, per block.
    let n_groups = rows * nb;
    let mut groups: Vec<GroupInput> = Vec::with_capacity(n_groups);
    for r in 0..rows {
        let rrow = &residual[r * k..r * k + k];
        for b in 0..nb {
            let (start, end) = block_range(b, k);
            let gw = &rrow[start..end];
            groups.push(GroupInput {
                weights: gw,
                sensitivity: group_sensitivity(cfg.sensitivity, gw, r * nb + b),
            });
        }
    }
    let extra_bpw = (cfg.budget_bpw - TRIT_BITS).max(0.0);
    let acfg = AllocConfig::from_bpw(
        extra_bpw,
        total_w,
        cfg.t_min.saturating_sub(1),
        cfg.t_max.saturating_sub(1),
    );
    let extra = allocate(&groups, &acfg)?;

    let mut row_stacks = Vec::with_capacity(rows);
    let mut plane_counts = vec![0usize; n_groups];
    for r in 0..rows {
        let rrow = &residual[r * k..r * k + k];
        let stacks = (0..nb)
            .map(|b| {
                let (start, end) = block_range(b, k);
                let g = r * nb + b;
                let base_block = Plane {
                    scale: base_scale,
                    trits: base.trits[r * k + start..r * k + end].to_vec(),
                };
                let resid = residual_expand(&rrow[start..end], extra.plane_counts[g]);
                // Plane count = the always-present base + the extra planes the allocator gave
                // (mirrors the per-block path, which reports requested counts).
                plane_counts[g] = 1 + extra.plane_counts[g];
                let mut planes = Vec::with_capacity(1 + resid.planes.len());
                planes.push(base_block);
                planes.extend(resid.planes);
                PlaneStack { planes }
            })
            .collect();
        row_stacks.push(stacks);
    }
    Ok((row_stacks, plane_counts))
}

/// Assemble one row's `nb` block plane-stacks into a packed [`SaltRow`] of dense TQ2_0
/// planes (plane `p` gathers block `b`'s `p`-th plane, padding shorter blocks with zeros).
fn assemble_salt_row(k: usize, nb: usize, stacks: &[PlaneStack]) -> Result<SaltRow, QuantError> {
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
    Ok(SaltRow { k, planes })
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
    fn reference_dequant(
        weights: &[f32],
        rows: usize,
        k: usize,
        plane_counts: &[usize],
    ) -> Vec<f32> {
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
            let cfg = QuantConfig {
                budget_bpw: 2.6,
                ..Default::default()
            };
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

    /// Independent reference for the per-tensor path: a single per-tensor base plane plus
    /// per-block residual planes re-expanded from the residual (driven by `plane_counts`).
    fn reference_dequant_tensor(
        weights: &[f32],
        rows: usize,
        k: usize,
        plane_counts: &[usize],
    ) -> Vec<f32> {
        let nb = num_blocks(k);
        let base_scale = tritium_core::absmean(weights);
        let base = crate::ternary_at_scale(weights, base_scale);
        let mut residual = weights.to_vec();
        for (rsd, t) in residual.iter_mut().zip(&base.trits) {
            *rsd -= base_scale * t.to_f32();
        }
        let bs = f16::from_f32(base_scale).to_f32();
        let mut out = vec![0.0f32; rows * k];
        for r in 0..rows {
            for b in 0..nb {
                let start = b * QK_K;
                let end = (start + QK_K).min(k);
                // base plane (uniform per-tensor scale).
                for i in start..end {
                    out[r * k + i] += bs * base.trits[r * k + i].to_f32();
                }
                // residual planes: plane_counts[g] − 1 of them, re-expanded on the residual.
                let extra = plane_counts[r * nb + b] - 1;
                let stack = residual_expand(&residual[r * k + start..r * k + end], extra);
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

    // ── Gate: Tensor-mode dequant(quantized) == independent reference, ABOVE the floor.
    // Closes the multi-plane (base + per-block residual) assembly path the floor gate misses.
    #[test]
    fn tensor_dequant_matches_reference() {
        for &(rows, k) in &[(1usize, 256usize), (3, 700), (4, 256), (2, 257), (5, 1000)] {
            let w = make_tensor(rows, k, 0x5A17 ^ (k as u64));
            let cfg = QuantConfig {
                budget_bpw: 2.6,
                scale_group: ScaleGroup::Tensor,
                ..Default::default()
            };
            let qt = quantize_tensor(&w, rows, k, &cfg).unwrap();
            let reference = reference_dequant_tensor(&w, rows, k, &qt.plane_counts);
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
            assert!(
                qt.logical_bpw() <= 2.6 + 1e-9,
                "tensor logical bpw {} > budget",
                qt.logical_bpw()
            );
        }
    }

    // ── Gate: ScaleGroup::Tensor floor == PER-TENSOR AbsMean (deployed BitNet I2_S).
    // The deployed b1.58 uses ONE absmean scale for the whole tensor; the per-256-block
    // default does not reproduce it (it reconstructs the heavy-tailed master far more
    // faithfully → weights the QAT model was never trained for). At the floor budget every
    // group must be a single per-tensor base plane.
    #[test]
    fn tensor_group_floor_is_per_tensor_absmean() {
        let (rows, k) = (3usize, 512usize);
        let w = make_tensor(rows, k, 0xBEEF);
        let cfg = QuantConfig {
            budget_bpw: crate::TRIT_BITS,
            t_min: 1,
            t_max: 3,
            sensitivity: Sensitivity::Uniform,
            scale_group: ScaleGroup::Tensor,
        };
        let qt = quantize_tensor(&w, rows, k, &cfg).unwrap();
        assert!(
            qt.plane_counts.iter().all(|&t| t == 1),
            "floor ⇒ base plane only"
        );

        let tensor_absmean = tritium_core::absmean(&w);
        let s = f16::from_f32(tensor_absmean).to_f32();
        for r in 0..rows {
            let deq = dequant_salt_row(&qt.salt_rows[r]).unwrap();
            for i in 0..k {
                // Mirror `quantize_one`'s `as i8` cast (collapses -0.0 → 0) so the reference
                // doesn't carry a negative-zero the integer trit can't represent.
                let trit = ((w[r * k + i] / tensor_absmean).round().clamp(-1.0, 1.0) as i8) as f32;
                assert_eq!(deq[i].to_bits(), (s * trit).to_bits(), "r={r} i={i}");
            }
        }
    }

    // ── Gate: budget == base ⇒ T=1 everywhere == flat AbsMean (BitNet floor). ─
    #[test]
    fn budget_floor_is_flat_absmean() {
        let (rows, k) = (3usize, 512usize);
        let w = make_tensor(rows, k, 0xABBA);
        // budget_bpw exactly the base (t_min=1 ⇒ log2 3 bpw)
        let cfg = QuantConfig {
            budget_bpw: crate::TRIT_BITS,
            t_min: 1,
            t_max: 3,
            sensitivity: Sensitivity::Uniform,
            scale_group: ScaleGroup::Block,
        };
        let qt = quantize_tensor(&w, rows, k, &cfg).unwrap();
        assert!(
            qt.plane_counts.iter().all(|&t| t == 1),
            "all groups at base"
        );

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
            let cfg = QuantConfig {
                budget_bpw: bpw,
                ..Default::default()
            };
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
        let cfg = QuantConfig {
            budget_bpw: 2.4,
            ..Default::default()
        };
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
        let cfg = QuantConfig {
            sensitivity: Sensitivity::Custom(&bad),
            ..Default::default()
        };
        assert!(matches!(
            quantize_tensor(&w, 2, 256, &cfg),
            Err(QuantError::SensitivityLen {
                expected: 2,
                got: 3
            })
        ));
    }

    // Higher budget ⇒ no group loses planes (more capacity only adds).
    #[test]
    fn higher_budget_never_reduces_planes() {
        let w = make_tensor(3, 512, 0x9999);
        let lo = quantize_tensor(
            &w,
            3,
            512,
            &QuantConfig {
                budget_bpw: 2.0,
                ..Default::default()
            },
        )
        .unwrap();
        let hi = quantize_tensor(
            &w,
            3,
            512,
            &QuantConfig {
                budget_bpw: 3.5,
                ..Default::default()
            },
        )
        .unwrap();
        for (l, h) in lo.plane_counts.iter().zip(&hi.plane_counts) {
            assert!(h >= l, "more budget reduced a plane count");
        }
    }
}

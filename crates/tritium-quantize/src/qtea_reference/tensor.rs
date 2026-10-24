// Copyright (c) 2026 The QTEA Authors. Derived from qtea.py at
// e39bd8c0297e3427daeb404427c74cb6e202b683. MIT notice in LICENSE-MIT.

use super::{
    ArithmeticError, GroupParameters, SalientRanking, SparseResidualColumn, budget, check_input,
    check_iterations, filled, finite,
};
use crate::{CurvatureSourceId, InputGram};
use tritium_core::Trit;

mod export;
mod sweep;

/// Native scalar-f32/SPD-only single-tensor numerical policy, not a solver catalog ID.
pub const TENSOR_FIT_ID: &str = "qtea.cpu-tensor.f32.v1";

/// Explicit full-tensor recipe; validated before fitting/allocation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorFitConfig {
    /// Original column-group and feedback-block width,2..=4096.
    pub group_size: usize,
    /// Fraction of salient columns rounded to whole groups, finite0..=1.
    pub salient_ratio: f64,
    /// Positive finite initial dead-zone multiplier.
    pub delta_coefficient: f32,
    /// Original-group ITF rounds,0..=64.
    pub fit_iterations: u32,
    /// Per-column-v refinement rounds,0..=64.
    pub rescale_iterations: u32,
    /// Nonnegative finite fraction of mean diagonal, with no adaptive retry.
    pub damping: f32,
    /// Nonnegative finite exponential feedback decay.
    pub decay: f64,
    /// Positive relative Cholesky pivot floor, less than1; not a condition-number estimate.
    pub relative_pivot_floor: f32,
}

impl Default for TensorFitConfig {
    fn default() -> Self {
        Self {
            group_size: 128,
            salient_ratio: 0.05,
            delta_coefficient: 0.96,
            fit_iterations: 1,
            rescale_iterations: 2,
            damping: 0.007,
            decay: 1.0,
            relative_pivot_floor: 1e-7,
        }
    }
}

impl TensorFitConfig {
    fn validate(self, columns: usize) -> Result<(), ArithmeticError> {
        if !(2..=4096).contains(&self.group_size) || !columns.is_multiple_of(self.group_size) {
            return Err(ArithmeticError::InvalidShape);
        }
        check_iterations(self.fit_iterations)?;
        check_iterations(self.rescale_iterations)?;
        if !self.salient_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.salient_ratio)
            || !self.delta_coefficient.is_finite()
            || self.delta_coefficient <= 0.0
            || !self.damping.is_finite()
            || self.damping < 0.0
            || !self.decay.is_finite()
            || self.decay < 0.0
            || !self.relative_pivot_floor.is_finite()
            || self.relative_pivot_floor <= 0.0
            || self.relative_pivot_floor >= 1.0
        {
            return Err(ArithmeticError::InvalidConfig);
        }
        Ok(())
    }

    fn digest(self) -> [u8; 32] {
        let mut hash = blake3::Hasher::new_derive_key("tritium qtea tensor recipe v1");
        hash.update(TENSOR_FIT_ID.as_bytes());
        hash.update(&(self.group_size as u64).to_le_bytes());
        hash.update(&self.salient_ratio.to_bits().to_le_bytes());
        hash.update(&self.delta_coefficient.to_bits().to_le_bytes());
        hash.update(&self.fit_iterations.to_le_bytes());
        hash.update(&self.rescale_iterations.to_le_bytes());
        hash.update(&self.damping.to_bits().to_le_bytes());
        hash.update(&self.decay.to_bits().to_le_bytes());
        hash.update(&self.relative_pivot_floor.to_bits().to_le_bytes());
        *hash.finalize().as_bytes()
    }
}

/// Sparse residual attached to one original input column.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorResidual {
    column: usize,
    residual: SparseResidualColumn,
}

impl TensorResidual {
    /// Original (not fitting-order) input-column index.
    pub const fn column(&self) -> usize {
        self.column
    }
    /// Actual unpacked1:4 FP8 arithmetic representation.
    pub const fn residual(&self) -> &SparseResidualColumn {
        &self.residual
    }
}

/// Semantic fitted tensor, not an admitted compressed artifact/backend handle.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorFit {
    rows: usize,
    columns: usize,
    config: TensorFitConfig,
    groups: Vec<GroupParameters>,
    trits: Vec<Trit>,
    multipliers: Vec<f32>,
    residuals: Vec<TensorResidual>,
    order: Vec<usize>,
    source: CurvatureSourceId,
    gram_digest: [u8; 32],
    weights_digest: [u8; 32],
    recipe_digest: [u8; 32],
}

impl TensorFit {
    /// Conservative new-live logical byte ceiling,64C²+64RC+256C+4096.
    ///
    /// Includes owned output, nested metadata and live fitting/factor/hash scratch;
    /// excludes borrowed inputs, caller clones/retained results and allocator/RSS.
    ///
    /// # Errors
    /// Rejects nonpositive/unsupported geometry and arithmetic overflow.
    pub fn required_work_bytes(rows: usize, columns: usize) -> Result<usize, ArithmeticError> {
        if rows == 0
            || rows > (1 << 24)
            || !rows.is_multiple_of(4)
            || columns == 0
            || columns > (1 << 24)
        {
            return Err(ArithmeticError::InvalidShape);
        }
        columns
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(64))
            .and_then(|n| {
                rows.checked_mul(columns)
                    .and_then(|rc| rc.checked_mul(64))
                    .and_then(|rc| n.checked_add(rc))
            })
            .and_then(|n| columns.checked_mul(256).and_then(|c| n.checked_add(c)))
            .and_then(|n| n.checked_add(4096))
            .ok_or(ArithmeticError::InvalidShape)
    }

    /// Fit a complete tensor against an actual source-bound input Gram.
    ///
    /// Uses declared damping only, SPD-only native lower Bunch-Kaufman1x1
    /// ordering, inverse-Cholesky column feedback and sparse FP8 compensation.
    /// Native scalar rounding is not universal BLAS/LAPACK bitwise identity.
    /// No input mutation, recipe substitution, diagonal fallback, adaptive retry,
    /// package write or solver registration occurs. Group/constant-row refusals
    /// from arithmetic V1 remain. Curvature binding is not independent custody.
    ///
    /// # Errors
    /// Rejects config/shape/budget/allocation, unbound/mismatched/invalid Gram,
    /// failed/small pivots, degenerate groups and nonfinite input/intermediates.
    pub fn fit(
        weights: &[f32],
        rows: usize,
        columns: usize,
        gram: &InputGram,
        config: TensorFitConfig,
        max_work_bytes: usize,
    ) -> Result<Self, ArithmeticError> {
        let required = Self::required_work_bytes(rows, columns)?;
        config.validate(columns)?;
        if rows.checked_mul(columns) != Some(weights.len())
            || gram.dimension() != columns
            || gram.as_slice().len() != columns * columns
        {
            return Err(ArithmeticError::InvalidShape);
        }
        let source = gram.source_id().ok_or(ArithmeticError::UnboundGram)?;
        budget(Some(required), max_work_bytes)?;
        check_input(weights)?;
        let mut original = filled(weights.len(), 0.0)?;
        original.copy_from_slice(weights);
        let mut h = filled(columns * columns, 0.0)?;
        for (output, &value) in h.iter_mut().zip(gram.as_slice()) {
            *output = finite(value as f32)?;
        }
        let mut diagonal = filled(columns, 0.0)?;
        for column in 0..columns {
            for row in 0..columns {
                if h[row * columns + column] != h[column * columns + row] {
                    return Err(ArithmeticError::InvalidGram);
                }
            }
            if h[column * columns + column] < 0.0 {
                return Err(ArithmeticError::InvalidGram);
            }
            if h[column * columns + column] == 0.0 {
                if (0..columns).any(|row| h[row * columns + column] != 0.0) {
                    return Err(ArithmeticError::InvalidGram);
                }
                h[column * columns + column] = 1.0;
                for row in 0..rows {
                    original[row * columns + column] = 0.0;
                }
            }
            diagonal[column] = h[column * columns + column];
        }
        let group_count = columns / config.group_size;
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(group_count)
            .map_err(|_| ArithmeticError::AllocationFailed)?;
        let mut group_weights = filled(rows * config.group_size, 0.0)?;
        for group in 0..group_count {
            for row in 0..rows {
                group_weights[row * config.group_size..(row + 1) * config.group_size]
                    .copy_from_slice(
                        &original[row * columns + group * config.group_size
                            ..row * columns + (group + 1) * config.group_size],
                    );
            }
            groups.push(GroupParameters::fit(
                &group_weights,
                rows,
                config.group_size,
                config.delta_coefficient,
                config.fit_iterations,
                12 * rows + config.group_size,
            )?);
        }
        let ranking = SalientRanking::rank(
            &original,
            rows,
            columns,
            &diagonal,
            config.group_size,
            config.salient_ratio,
            columns * (size_of::<usize>() + 4),
        )?;
        let salient = ranking.salient_count();
        let mut order = filled(columns, 0_usize)?;
        order.copy_from_slice(ranking.columns());
        drop(ranking);
        super::factor::order_rest(
            &h,
            columns,
            &mut order[salient..],
            config.damping,
            config.relative_pivot_floor,
        )?;
        let permuted = super::factor::permuted(&h, columns, &order, config.damping)?;
        let upper = super::factor::inverse_upper(permuted, columns, config.relative_pivot_floor)?;
        let mut hash = blake3::Hasher::new_derive_key("tritium qtea original weights v1");
        hash.update(&(rows as u64).to_le_bytes());
        hash.update(&(columns as u64).to_le_bytes());
        for weight in weights {
            hash.update(&weight.to_bits().to_le_bytes());
        }
        let mut residuals = Vec::new();
        residuals
            .try_reserve_exact(salient)
            .map_err(|_| ArithmeticError::AllocationFailed)?;
        let mut result = Self {
            rows,
            columns,
            config,
            groups,
            trits: filled(weights.len(), Trit::ZERO)?,
            multipliers: filled(columns, 1.0)?,
            residuals,
            order,
            source,
            gram_digest: gram.digest(),
            weights_digest: *hash.finalize().as_bytes(),
            recipe_digest: config.digest(),
        };
        sweep::run(&mut result, &original, &upper, salient)?;
        result
            .residuals
            .sort_unstable_by_key(|record| record.column);
        Ok(result)
    }

    /// Output rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }
    /// Input columns.
    pub const fn columns(&self) -> usize {
        self.columns
    }
    /// Exact admitted scalar recipe.
    pub const fn config(&self) -> TensorFitConfig {
        self.config
    }
    /// Original row-major ternary codes.
    pub fn trits(&self) -> &[Trit] {
        &self.trits
    }
    /// Original group order; each group owns one alpha/beta/delta per output row.
    pub fn groups(&self) -> &[GroupParameters] {
        &self.groups
    }
    /// Column-v in original column order.
    pub fn column_multipliers(&self) -> &[f32] {
        &self.multipliers
    }
    /// FP8 residual records sorted by original column.
    pub fn residuals(&self) -> &[TensorResidual] {
        &self.residuals
    }
    /// Original column indices in actual fitting order.
    pub fn fitting_order(&self) -> &[usize] {
        &self.order
    }
    /// Exact calibration source/cache/token identities supplied with the Gram.
    pub const fn curvature_source(&self) -> CurvatureSourceId {
        self.source
    }
    /// Canonical input Gram identity, unchanged by damping or permutation.
    pub const fn gram_digest(&self) -> [u8; 32] {
        self.gram_digest
    }
    /// Actual original f32 weight bits and geometry, before dead-channel handling.
    pub const fn weights_digest(&self) -> [u8; 32] {
        self.weights_digest
    }
    /// Recipe and fixed numerical-policy identity.
    pub const fn recipe_digest(&self) -> [u8; 32] {
        self.recipe_digest
    }

    /// Decode the semantic fitted tensor into exact caller-sized row-major storage.
    ///
    /// # Errors
    /// Wrong shape refuses before mutation. This is reference reconstruction,
    /// not execution through a compressed backend or a stored package.
    pub fn decode_into(&self, output: &mut [f32]) -> Result<(), ArithmeticError> {
        if output.len() != self.trits.len() {
            return Err(ArithmeticError::InvalidShape);
        }
        for row in 0..self.rows {
            for column in 0..self.columns {
                let group = &self.groups[column / self.config.group_size];
                output[row * self.columns + column] = (group.alpha()[row]
                    * self.multipliers[column])
                    * self.trits[row * self.columns + column].to_f32()
                    + group.beta()[row];
            }
        }
        for record in &self.residuals {
            for (block, (&index, &code)) in record
                .residual
                .indices()
                .iter()
                .zip(record.residual.fp8_codes())
                .enumerate()
            {
                output[(block * 4 + usize::from(index)) * self.columns + record.column] +=
                    super::sparse::decode(code) / record.residual.scale_multiplier();
            }
        }
        Ok(())
    }
}

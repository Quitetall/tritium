use super::{TensorFit, TensorFitConfig};
use crate::qtea_reference::{ArithmeticError, budget, check_input};
use crate::{CurvatureSourceId, InputGram};
use tritium_format::qtea::{
    QteaBindings, QteaByteLedger, QteaError, QteaLimits, QteaMetadata, QteaRecipe,
};

/// Immutable borrowed numerical inputs, not machine admission or a completed fit.
///
/// Planning scans weights and the full Gram inO(RC+C²), but allocates no arrays,
/// does not factor curvature and does not write output. Actual fitting can still
/// refuse a pivot, degenerate group or numerical update. Native input/recipe
/// hashes remain unchanged; the new plan digest is not a file or checkpoint ID.
///
/// The borrow prevents changing the source while a plan is usable:
/// ```compile_fail,E0502
/// use tritium_quantize::{InputGram, qtea_reference::{TensorFit, TensorFitConfig}};
/// fn invalid(mut weights: Vec<f32>, gram: &InputGram) {
///     let config = TensorFitConfig { group_size: 2, ..TensorFitConfig::default() };
///     let plan = TensorFit::plan(&weights, 4, 4, gram, config, usize::MAX).unwrap();
///     weights[0] = 2.0;
///     let _ = plan.fit();
/// }
/// ```
#[derive(Clone)]
pub struct TensorFitPlan<'a> {
    pub(super) weights: &'a [f32],
    pub(super) gram: &'a InputGram,
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) config: TensorFitConfig,
    pub(super) source: CurvatureSourceId,
    pub(super) weights_digest: [u8; 32],
    pub(super) gram_digest: [u8; 32],
    pub(super) recipe_digest: [u8; 32],
    input_digest: [u8; 32],
    required_work_bytes: usize,
}

impl core::fmt::Debug for TensorFitPlan<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorFitPlan")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .field("input_digest", &self.input_digest)
            .field("required_work_bytes", &self.required_work_bytes)
            .finish_non_exhaustive()
    }
}

impl<'a> TensorFitPlan<'a> {
    pub(super) fn new(
        weights: &'a [f32],
        rows: usize,
        columns: usize,
        gram: &'a InputGram,
        config: TensorFitConfig,
        max_work_bytes: usize,
    ) -> Result<Self, ArithmeticError> {
        let required_work_bytes = TensorFit::required_work_bytes(rows, columns)?;
        config.validate(columns)?;
        if rows.checked_mul(columns) != Some(weights.len())
            || gram.dimension() != columns
            || gram.as_slice().len() != columns * columns
        {
            return Err(ArithmeticError::InvalidShape);
        }
        let source = gram.source_id().ok_or(ArithmeticError::UnboundGram)?;
        budget(Some(required_work_bytes), max_work_bytes)?;
        check_input(weights)?;
        let values = gram.as_slice();
        if values.iter().any(|x| !(*x as f32).is_finite()) {
            return Err(ArithmeticError::NonFiniteArithmetic);
        }
        for column in 0..columns {
            for row in 0..columns {
                if values[row * columns + column] as f32 != values[column * columns + row] as f32 {
                    return Err(ArithmeticError::InvalidGram);
                }
            }
            let diagonal = values[column * columns + column] as f32;
            if diagonal < 0.
                || (diagonal == 0.
                    && (0..columns).any(|row| values[row * columns + column] as f32 != 0.))
            {
                return Err(ArithmeticError::InvalidGram);
            }
        }
        let mut weights_hash = blake3::Hasher::new_derive_key("tritium qtea original weights v1");
        weights_hash.update(&(rows as u64).to_le_bytes());
        weights_hash.update(&(columns as u64).to_le_bytes());
        for value in weights {
            weights_hash.update(&value.to_bits().to_le_bytes());
        }
        let weights_digest = *weights_hash.finalize().as_bytes();
        let gram_digest = gram.digest();
        let recipe_digest = config.digest();
        let input_digest = numerical_input_digest(metadata(
            rows,
            columns,
            config,
            source,
            weights_digest,
            gram_digest,
        ))?;
        Ok(Self {
            weights,
            gram,
            rows,
            columns,
            config,
            source,
            weights_digest,
            gram_digest,
            recipe_digest,
            input_digest,
            required_work_bytes,
        })
    }

    /// Numerical input fingerprint; not owner/name, raw checkpoint or artifact identity.
    pub const fn input_digest(&self) -> [u8; 32] {
        self.input_digest
    }
    /// Shared validated geometry/recipe and actual input fingerprints.
    pub fn metadata(&self) -> QteaMetadata {
        metadata(
            self.rows,
            self.columns,
            self.config,
            self.source,
            self.weights_digest,
            self.gram_digest,
        )
    }
    /// Conservative native fitting logical bound, excluding borrowed inputs/RSS.
    pub const fn required_work_bytes(&self) -> usize {
        self.required_work_bytes
    }
    /// Plan exact wire bytes/caps before fitting/export, with no allocation.
    ///
    /// # Errors
    /// Propagates owning format metadata or receiver-cap refusal.
    pub fn encoded_ledger(&self, limits: QteaLimits) -> Result<QteaByteLedger, QteaError> {
        self.metadata().byte_ledger(limits)
    }
    /// Execute the existing scalar numerical policy against these exact borrows.
    ///
    /// # Errors
    /// Fitting can still reject conditioning, group/data arithmetic or allocation.
    /// Planning is not a positive fit or qualification receipt.
    pub fn fit(self) -> Result<TensorFit, ArithmeticError> {
        TensorFit::fit_prepared(self)
    }
}

/// Reconstruct the existing mathematical input fingerprint from declared wire
/// metadata. This checks basic geometry/source IDs, not original weights/Gram,
/// execution, recipe validity, machine admission or source authenticity.
/// # Errors
/// Empty/overflowing geometry or unbound source identity.
pub fn numerical_input_digest(meta: QteaMetadata) -> Result<[u8; 32], ArithmeticError> {
    if meta.rows == 0 || meta.columns == 0 || meta.rows.checked_mul(meta.columns).is_none() {
        return Err(ArithmeticError::InvalidShape);
    }
    let source = CurvatureSourceId::new(
        meta.bindings.source_model,
        meta.bindings.activation_cache,
        meta.bindings.token_stream,
    )
    .map_err(|_| ArithmeticError::UnboundGram)?;
    let mut hash = blake3::Hasher::new_derive_key("tritium qtea numerical input plan v1");
    hash.update(&(meta.rows as u64).to_le_bytes());
    hash.update(&(meta.columns as u64).to_le_bytes());
    for digest in [
        meta.bindings.weights,
        meta.bindings.gram,
        meta.recipe.digest(),
        source.digest(),
    ] {
        hash.update(&digest);
    }
    Ok(*hash.finalize().as_bytes())
}

pub(super) fn metadata(
    rows: usize,
    columns: usize,
    c: TensorFitConfig,
    source: CurvatureSourceId,
    weights: [u8; 32],
    gram: [u8; 32],
) -> QteaMetadata {
    QteaMetadata {
        rows,
        columns,
        recipe: QteaRecipe {
            group_size: c.group_size,
            salient_ratio: c.salient_ratio,
            delta_coefficient: c.delta_coefficient,
            fit_iterations: c.fit_iterations,
            rescale_iterations: c.rescale_iterations,
            damping: c.damping,
            decay: c.decay,
            relative_pivot_floor: c.relative_pivot_floor,
        },
        bindings: QteaBindings {
            weights,
            gram,
            source_model: source.source_model_digest(),
            activation_cache: source.activation_cache_digest(),
            token_stream: source.token_stream_digest(),
        },
    }
}

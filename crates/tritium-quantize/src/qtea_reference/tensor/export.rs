use super::TensorFit;
use tritium_format::qtea::{QteaError, QteaGroupRef, QteaLimits, QteaResidualRef, QteaTensor};

impl TensorFit {
    /// Export exact executable parameters to the bounded residual-bearing format.
    /// Fitting-only group delta stays offline; recipe/input identities bind replay.
    /// Borrowed iterators avoid flattening group arrays or a dense weight copy.
    /// Input/recipe identities are retained, not independently authenticated.
    /// No model package, registry entry or qualification receipt is published.
    ///
    /// # Errors
    /// Refuses format/identity/cap/allocation errors with no partial result.
    pub fn to_qtea_tensor(&self, limits: QteaLimits) -> Result<QteaTensor, QteaError> {
        let meta = super::plan::metadata(
            self.rows,
            self.columns,
            self.config,
            self.source,
            self.weights_digest,
            self.gram_digest,
        );
        let encoded = QteaTensor::from_parts(
            meta,
            &self.trits,
            &self.multipliers,
            &self.order,
            self.groups.iter().map(|g| QteaGroupRef {
                alpha: g.alpha(),
                beta: g.beta(),
            }),
            self.residuals.iter().map(|r| QteaResidualRef {
                column: r.column(),
                scale_multiplier: r.residual().scale_multiplier(),
                indices: r.residual().indices(),
                codes: r.residual().fp8_codes(),
            }),
            limits,
        )?;
        if encoded.view().recipe_digest() != self.recipe_digest {
            return Err(QteaError::InvalidMetadata);
        }
        Ok(encoded)
    }
}

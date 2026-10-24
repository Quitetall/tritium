use super::TensorFit;
use tritium_format::qtea::{
    QteaBindings, QteaError, QteaGroupRef, QteaLimits, QteaMetadata, QteaRecipe, QteaResidualRef,
    QteaTensor,
};

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
        let c = self.config;
        let meta = QteaMetadata {
            rows: self.rows,
            columns: self.columns,
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
                weights: self.weights_digest,
                gram: self.gram_digest,
                source_model: self.source.source_model_digest(),
                activation_cache: self.source.activation_cache_digest(),
                token_stream: self.source.token_stream_digest(),
            },
        };
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

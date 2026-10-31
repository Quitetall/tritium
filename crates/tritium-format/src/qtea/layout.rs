use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct Layout {
    pub meta: QteaMetadata,
    pub salient: usize,
    pub params: usize,
    pub multipliers: usize,
    pub order: usize,
    pub residuals: usize,
    pub position_bytes: usize,
    pub record_bytes: usize,
    pub trailer: usize,
    pub ledger: QteaByteLedger,
}

impl Layout {
    pub(super) fn new(meta: QteaMetadata) -> Result<Self, QteaError> {
        let (r, c, g) = (meta.rows, meta.columns, meta.recipe.group_size);
        if r == 0
            || c == 0
            || r > (1 << 24)
            || c > (1 << 24)
            || !r.is_multiple_of(4)
            || !(2..=4096).contains(&g)
            || !c.is_multiple_of(g)
        {
            return Err(QteaError::InvalidGeometry);
        }
        let p = meta.recipe;
        if !p.salient_ratio.is_finite()
            || !(0.0..=1.0).contains(&p.salient_ratio)
            || !p.delta_coefficient.is_finite()
            || p.delta_coefficient <= 0.
            || p.fit_iterations > 64
            || p.rescale_iterations > 64
            || !p.damping.is_finite()
            || p.damping < 0.
            || !p.decay.is_finite()
            || p.decay < 0.
            || !p.relative_pivot_floor.is_finite()
            || p.relative_pivot_floor <= 0.
            || p.relative_pivot_floor >= 1.
        {
            return Err(QteaError::InvalidMetadata);
        }
        let b = meta.bindings;
        if [
            b.weights,
            b.gram,
            b.source_model,
            b.activation_cache,
            b.token_stream,
        ]
        .contains(&[0; 32])
        {
            return Err(QteaError::InvalidMetadata);
        }
        let groups = c / g;
        let salient = if p.salient_ratio == 0. {
            0
        } else {
            ((groups as f64 * p.salient_ratio).round_ties_even() as usize)
                .max(1)
                .min(groups)
                * g
        };
        let mul = |a: usize, b: usize| a.checked_mul(b).ok_or(QteaError::InvalidGeometry);
        let add = |a: usize, b: usize| a.checked_add(b).ok_or(QteaError::InvalidGeometry);
        let trit_bytes = mul(r, c)?.div_ceil(5);
        let parameter_bytes = mul(mul(r, groups)?, 8)?;
        let params = add(QTEA_HEADER_BYTES, trit_bytes)?;
        let multipliers = add(params, parameter_bytes)?;
        let order = add(multipliers, mul(c, 4)?)?;
        let residuals = add(order, mul(c, 4)?)?;
        let position_bytes = (r / 4).div_ceil(4);
        let record_bytes = add(add(8, position_bytes)?, r / 4)?;
        let trailer = add(residuals, mul(salient, record_bytes)?)?;
        let total = add(trailer, 32)?;
        Ok(Self {
            meta,
            salient,
            params,
            multipliers,
            order,
            residuals,
            position_bytes,
            record_bytes,
            trailer,
            ledger: QteaByteLedger {
                header_bytes: QTEA_HEADER_BYTES,
                trit_bytes,
                group_parameter_bytes: parameter_bytes,
                column_multiplier_bytes: mul(c, 4)?,
                order_bytes: mul(c, 4)?,
                residual_metadata_bytes: mul(salient, 8)?,
                residual_position_bytes: mul(salient, position_bytes)?,
                residual_code_bytes: mul(salient, r / 4)?,
                checksum_bytes: 32,
                total_bytes: total,
            },
        })
    }

    pub(super) fn admit(self, limits: QteaLimits) -> Result<Self, QteaError> {
        if self.ledger.total_bytes > limits.max_tensor_bytes {
            return Err(QteaError::TensorLimit);
        }
        if self.meta.columns > limits.max_validation_bytes {
            return Err(QteaError::ValidationLimit);
        }
        Ok(self)
    }

    pub(super) fn read(header: &[u8], limits: QteaLimits) -> Result<Self, QteaError> {
        if header.len() < QTEA_HEADER_BYTES
            || header[..8] != QTEA_MAGIC
            || u32_at(header, 8) != QTEA_VERSION
            || u32_at(header, 12) != QTEA_HEADER_BYTES as u32
            || header[56..64]
                .iter()
                .chain(&header[100..104])
                .chain(&header[304..384])
                .any(|b| *b != 0)
        {
            return Err(QteaError::InvalidHeader);
        }
        let count = |offset| {
            usize::try_from(u64_at(header, offset)).map_err(|_| QteaError::InvalidGeometry)
        };
        let digest = |offset| {
            header[offset..offset + 32]
                .try_into()
                .expect("fixed header digest")
        };
        let meta = QteaMetadata {
            rows: count(24)?,
            columns: count(32)?,
            recipe: QteaRecipe {
                group_size: count(40)?,
                salient_ratio: f64::from_bits(u64_at(header, 64)),
                delta_coefficient: f32_at(header, 72),
                fit_iterations: u32_at(header, 76),
                rescale_iterations: u32_at(header, 80),
                damping: f32_at(header, 84),
                decay: f64::from_bits(u64_at(header, 88)),
                relative_pivot_floor: f32_at(header, 96),
            },
            bindings: QteaBindings {
                weights: digest(104),
                gram: digest(136),
                source_model: digest(200),
                activation_cache: digest(232),
                token_stream: digest(264),
            },
        };
        let layout = Self::new(meta)?.admit(limits)?;
        if count(16)? != layout.ledger.total_bytes
            || count(48)? != layout.salient
            || count(296)? != layout.trailer - QTEA_HEADER_BYTES
            || header[168..200] != meta.recipe.digest()
        {
            return Err(QteaError::InvalidHeader);
        }
        Ok(layout)
    }

    pub(super) fn write(self, header: &mut [u8]) {
        header[..8].copy_from_slice(&QTEA_MAGIC);
        header[8..12].copy_from_slice(&QTEA_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&(QTEA_HEADER_BYTES as u32).to_le_bytes());
        for (offset, value) in [
            (16, self.ledger.total_bytes),
            (24, self.meta.rows),
            (32, self.meta.columns),
            (40, self.meta.recipe.group_size),
            (48, self.salient),
            (296, self.trailer - QTEA_HEADER_BYTES),
        ] {
            header[offset..offset + 8].copy_from_slice(&(value as u64).to_le_bytes());
        }
        let p = self.meta.recipe;
        header[64..72].copy_from_slice(&p.salient_ratio.to_bits().to_le_bytes());
        for (offset, value) in [
            (72, p.delta_coefficient),
            (84, p.damping),
            (96, p.relative_pivot_floor),
        ] {
            header[offset..offset + 4].copy_from_slice(&value.to_bits().to_le_bytes());
        }
        header[76..80].copy_from_slice(&p.fit_iterations.to_le_bytes());
        header[80..84].copy_from_slice(&p.rescale_iterations.to_le_bytes());
        header[88..96].copy_from_slice(&p.decay.to_bits().to_le_bytes());
        let b = self.meta.bindings;
        for (offset, value) in [
            (104, b.weights),
            (136, b.gram),
            (168, p.digest()),
            (200, b.source_model),
            (232, b.activation_cache),
            (264, b.token_stream),
        ] {
            header[offset..offset + 32].copy_from_slice(&value);
        }
    }
}

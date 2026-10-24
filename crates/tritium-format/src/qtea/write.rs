use super::{layout::Layout, *};
use tritium_core::Trit;

impl QteaTensor {
    /// Encode borrowed fitted arrays without flattening groups or a dense tensor.
    /// Writer provider lengths are checked against actual yielded item counts.
    /// New storage is one bounded encoded buffer plus column-sized validation.
    ///
    /// # Errors
    /// Refuses shape/recipe/source, provider counts, canonical/numerical data,
    /// caps or failed allocation. No partially encoded result is returned.
    pub fn from_parts<'a>(
        meta: QteaMetadata,
        trits: &[Trit],
        multipliers: &[f32],
        order: &[usize],
        mut groups: impl ExactSizeIterator<Item = QteaGroupRef<'a>>,
        mut residuals: impl ExactSizeIterator<Item = QteaResidualRef<'a>>,
        limits: QteaLimits,
    ) -> Result<Self, QteaError> {
        let l = Layout::new(meta)?.admit(limits)?;
        let (r, c, g) = (meta.rows, meta.columns, meta.recipe.group_size);
        if trits.len() != r * c
            || multipliers.len() != c
            || order.len() != c
            || groups.len() != c / g
            || residuals.len() != l.salient
        {
            return Err(QteaError::InvalidGeometry);
        }
        let mut bytes = allocate(l.ledger.total_bytes)?;
        if bytes.capacity() > limits.max_tensor_bytes {
            return Err(QteaError::TensorLimit);
        }
        l.write(&mut bytes[..QTEA_HEADER_BYTES]);
        crate::salt_v2::SaltV2Codec::B3
            .pack_into(trits, &mut bytes[QTEA_HEADER_BYTES..l.params])
            .map_err(|_| QteaError::InvalidBody)?;
        for group in 0..c / g {
            let parameters = groups.next().ok_or(QteaError::InvalidGeometry)?;
            if [parameters.alpha.len(), parameters.beta.len()] != [r; 2] {
                return Err(QteaError::InvalidGeometry);
            }
            for row in 0..r {
                let offset = l.params + (group * r + row) * 8;
                for (i, value) in [parameters.alpha[row], parameters.beta[row]]
                    .iter()
                    .enumerate()
                {
                    bytes[offset + i * 4..offset + (i + 1) * 4]
                        .copy_from_slice(&value.to_bits().to_le_bytes());
                }
            }
        }
        if groups.next().is_some() {
            return Err(QteaError::InvalidGeometry);
        }
        for (i, value) in multipliers.iter().enumerate() {
            bytes[l.multipliers + i * 4..l.multipliers + (i + 1) * 4]
                .copy_from_slice(&value.to_bits().to_le_bytes());
        }
        for (i, &value) in order.iter().enumerate() {
            let value = u32::try_from(value).map_err(|_| QteaError::InvalidGeometry)?;
            bytes[l.order + i * 4..l.order + (i + 1) * 4].copy_from_slice(&value.to_le_bytes());
        }
        for i in 0..l.salient {
            let record = residuals.next().ok_or(QteaError::InvalidGeometry)?;
            if record.indices.len() != r / 4 || record.codes.len() != r / 4 {
                return Err(QteaError::InvalidGeometry);
            }
            let offset = l.residuals + i * l.record_bytes;
            let column = u32::try_from(record.column).map_err(|_| QteaError::InvalidGeometry)?;
            bytes[offset..offset + 4].copy_from_slice(&column.to_le_bytes());
            bytes[offset + 4..offset + 8]
                .copy_from_slice(&record.scale_multiplier.to_bits().to_le_bytes());
            for (block, &position) in record.indices.iter().enumerate() {
                if position >= 4 {
                    return Err(QteaError::InvalidBody);
                }
                bytes[offset + 8 + block / 4] |= position << ((block % 4) * 2);
            }
            bytes[offset + 8 + l.position_bytes..offset + l.record_bytes]
                .copy_from_slice(record.codes);
        }
        if residuals.next().is_some() {
            return Err(QteaError::InvalidGeometry);
        }
        let digest = fingerprint(&bytes[..l.trailer]);
        bytes[l.trailer..].copy_from_slice(&digest);
        Self::from_owned_bytes(bytes, limits)
    }
}

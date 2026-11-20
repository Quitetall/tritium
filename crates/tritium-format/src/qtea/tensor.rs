use super::{layout::Layout, *};
use tritium_core::Trit;

/// Owned canonical encoded tensor; source identities are not independent custody.
#[derive(Clone, Debug)]
pub struct QteaTensor {
    pub(super) bytes: Vec<u8>,
    pub(super) layout: Layout,
}

/// Immutable checked borrowed bytes. No dense weight/trit buffer is retained.
#[derive(Clone, Copy, Debug)]
pub struct QteaTensorView<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) layout: Layout,
}

/// Checked sparse correction record in original row/column coordinates.
#[derive(Clone, Copy, Debug)]
pub struct QteaResidualView<'a> {
    bytes: &'a [u8],
    rows: usize,
    positions: usize,
}

impl QteaTensor {
    /// Take encoded storage without copying it, validating all bytes and capacity.
    ///
    /// # Errors
    /// Refuses malformed/noncanonical/numerically invalid data or caller caps.
    pub fn from_owned_bytes(bytes: Vec<u8>, limits: QteaLimits) -> Result<Self, QteaError> {
        if bytes.capacity() > limits.max_tensor_bytes {
            return Err(QteaError::TensorLimit);
        }
        let layout = QteaTensorView::from_bytes(&bytes, limits)?.layout;
        Ok(Self { bytes, layout })
    }

    /// Read exactly one bounded file, checking the header before body allocation/read.
    /// Requires EOF after the integrity trailer; consumes one byte on trailing-data refusal.
    ///
    /// # Errors
    /// Refuses malformed data/caps/allocation and propagates I/O failures.
    pub fn read_from(
        reader: &mut impl std::io::Read,
        limits: QteaLimits,
    ) -> Result<Self, QteaError> {
        let mut header = [0u8; QTEA_HEADER_BYTES];
        reader.read_exact(&mut header)?;
        let layout = Layout::read(&header, limits)?;
        let mut bytes = allocate(layout.ledger.total_bytes)?;
        if bytes.capacity() > limits.max_tensor_bytes {
            return Err(QteaError::TensorLimit);
        }
        bytes[..QTEA_HEADER_BYTES].copy_from_slice(&header);
        reader.read_exact(&mut bytes[QTEA_HEADER_BYTES..])?;
        let mut tail = [0u8; 1];
        if reader.read(&mut tail)? != 0 {
            return Err(QteaError::InvalidHeader);
        }
        Self::from_owned_bytes(bytes, limits)
    }

    /// Write all canonical bytes. An error is not successful publication.
    ///
    /// # Errors
    /// Propagates short/failed writer I/O.
    pub fn write_to(&self, writer: &mut impl std::io::Write) -> Result<(), QteaError> {
        writer.write_all(&self.bytes)?;
        Ok(())
    }
    /// Already-validated borrowed execution view; no revalidation/allocation.
    pub fn view(&self) -> QteaTensorView<'_> {
        QteaTensorView {
            bytes: &self.bytes,
            layout: self.layout,
        }
    }
    /// Every serialized byte, including header and checksum.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Actual retained Vec element capacity, not allocator overhead/RSS.
    pub fn buffer_capacity_bytes(&self) -> usize {
        self.bytes.capacity()
    }
    /// Fixed Rust handle storage on this target, separately from retained bytes.
    pub const fn handle_size_bytes() -> usize {
        size_of::<Self>()
    }
}

impl<'a> QteaTensorView<'a> {
    /// Validate immutable borrowed bytes under explicit encoded/validation caps.
    ///
    /// # Errors
    /// Refuses length/version/identity/integrity, noncanonical data, invalid
    /// numerical reconstruction, allocation failure or insufficient caller limits.
    pub fn from_bytes(bytes: &'a [u8], limits: QteaLimits) -> Result<Self, QteaError> {
        let layout = Layout::read(bytes, limits)?;
        if bytes.len() != layout.ledger.total_bytes {
            return Err(QteaError::InvalidHeader);
        }
        if bytes[layout.trailer..] != fingerprint(&bytes[..layout.trailer]) {
            return Err(QteaError::InvalidBody);
        }
        let view = Self { bytes, layout };
        view.validate_body()?;
        Ok(view)
    }
    /// Original geometry, exact recipe and producer-supplied bindings.
    pub const fn metadata(&self) -> QteaMetadata {
        self.layout.meta
    }
    /// Exact serialized component sizes, including all metadata and padding.
    pub const fn byte_ledger(&self) -> QteaByteLedger {
        self.layout.ledger
    }
    /// Allocation-free offsets derived from this checked view's admitted layout.
    /// Does not change wire bytes or authenticate source/qualification claims.
    pub const fn execution_layout(&self) -> QteaExecutionLayout {
        QteaExecutionLayout {
            trits_offset: QTEA_HEADER_BYTES,
            group_parameters_offset: self.layout.params,
            column_multipliers_offset: self.layout.multipliers,
            residuals_offset: self.layout.residuals,
            residual_record_bytes: self.layout.record_bytes,
            residual_position_bytes: self.layout.position_bytes,
            residual_count: self.layout.salient,
        }
    }
    /// Domain-separated digest over all header/body bytes, not a signature.
    pub fn content_digest(&self) -> [u8; 32] {
        self.bytes[self.layout.trailer..]
            .try_into()
            .expect("checked trailer")
    }
    /// Preserved native numerical recipe identity.
    pub fn recipe_digest(&self) -> [u8; 32] {
        self.bytes[168..200].try_into().expect("checked header")
    }
    /// Entire borrowed encoded storage; ownership/caller capacity is not implied.
    pub fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Read one original-coordinate ternary value directly from its radix byte.
    ///
    /// # Errors
    /// Refuses an out-of-range row or column.
    pub fn trit(&self, row: usize, column: usize) -> Result<Trit, QteaError> {
        self.check_index(row, column)?;
        Ok(self.trit_at(row * self.layout.meta.columns + column))
    }
    /// Exact original group alpha/beta, without alignment assumptions.
    ///
    /// # Errors
    /// Refuses an out-of-range original group or row.
    pub fn group_parameters(&self, group: usize, row: usize) -> Result<(f32, f32), QteaError> {
        let m = self.layout.meta;
        if row >= m.rows || group >= m.columns / m.recipe.group_size {
            return Err(QteaError::InvalidIndex);
        }
        Ok(self.parameters_at(group, row))
    }
    /// Original input column-v value.
    ///
    /// # Errors
    /// Refuses an out-of-range column.
    pub fn column_multiplier(&self, column: usize) -> Result<f32, QteaError> {
        if column >= self.layout.meta.columns {
            return Err(QteaError::InvalidIndex);
        }
        Ok(f32_at(self.bytes, self.layout.multipliers + column * 4))
    }
    /// Actual fitting order as original column indices, with no allocated array.
    pub fn fitting_order(&self) -> impl ExactSizeIterator<Item = usize> + 'a {
        let bytes = self.bytes;
        let offset = self.layout.order;
        let n = self.layout.meta.columns;
        (0..n).map(move |i| u32_at(bytes, offset + i * 4) as usize)
    }
    /// Sparse correction records sorted by original column, without copying.
    pub fn residuals(&self) -> impl ExactSizeIterator<Item = QteaResidualView<'a>> + 'a {
        let bytes = self.bytes;
        let l = self.layout;
        (0..l.salient).map(move |i| QteaResidualView {
            bytes: &bytes[l.residuals + i * l.record_bytes..l.residuals + (i + 1) * l.record_bytes],
            rows: l.meta.rows,
            positions: l.position_bytes,
        })
    }
    /// Reconstruct one affine base scalar, not a dense weight matrix.
    ///
    /// # Errors
    /// Refuses an out-of-range coordinate. Admitted base arithmetic is finite.
    pub fn base_weight(&self, row: usize, column: usize) -> Result<f32, QteaError> {
        self.check_index(row, column)?;
        Ok(self.base_at(row, column))
    }
    fn check_index(&self, row: usize, column: usize) -> Result<(), QteaError> {
        if row >= self.layout.meta.rows || column >= self.layout.meta.columns {
            Err(QteaError::InvalidIndex)
        } else {
            Ok(())
        }
    }
    fn trit_at(&self, index: usize) -> Trit {
        let digit = (self.bytes[QTEA_HEADER_BYTES + index / 5] / [1, 3, 9, 27, 81][index % 5]) % 3;
        match digit {
            0 => Trit::NEG,
            1 => Trit::ZERO,
            _ => Trit::POS,
        }
    }
    fn parameters_at(&self, group: usize, row: usize) -> (f32, f32) {
        let offset = self.layout.params + (group * self.layout.meta.rows + row) * 8;
        (f32_at(self.bytes, offset), f32_at(self.bytes, offset + 4))
    }
    fn base_at(&self, row: usize, column: usize) -> f32 {
        let m = self.layout.meta;
        let (a, b) = self.parameters_at(column / m.recipe.group_size, row);
        (a * f32_at(self.bytes, self.layout.multipliers + column * 4))
            * self.trit_at(row * m.columns + column).to_f32()
            + b
    }
    fn validate_body(&self) -> Result<(), QteaError> {
        let l = self.layout;
        let m = l.meta;
        let n = m.rows * m.columns;
        for (i, &byte) in self.bytes[QTEA_HEADER_BYTES..l.params].iter().enumerate() {
            if byte >= 243 {
                return Err(QteaError::InvalidBody);
            }
            for (slot, power) in [1, 3, 9, 27, 81].iter().enumerate() {
                if i * 5 + slot >= n && (byte / power) % 3 != 1 {
                    return Err(QteaError::InvalidBody);
                }
            }
        }
        for offset in (l.params..l.multipliers).step_by(8) {
            let (a, b) = (f32_at(self.bytes, offset), f32_at(self.bytes, offset + 4));
            if !a.is_finite() || a <= 0. || !b.is_finite() {
                return Err(QteaError::InvalidBody);
            }
        }
        for offset in (l.multipliers..l.order).step_by(4) {
            let v = f32_at(self.bytes, offset);
            if !v.is_finite() || v <= 0. {
                return Err(QteaError::InvalidBody);
            }
        }
        let mut flags = allocate(m.columns)?;
        for (i, column) in self.fitting_order().enumerate() {
            if column >= m.columns || flags[column] != 0 {
                return Err(QteaError::InvalidBody);
            }
            flags[column] = if i < l.salient { 3 } else { 1 };
        }
        let mut previous = None;
        for record in self.residuals() {
            let column = record.column();
            if column >= m.columns || previous.is_some_and(|p| p >= column) || flags[column] != 3 {
                return Err(QteaError::InvalidBody);
            }
            previous = Some(column);
            let scale = record.scale_multiplier();
            if !scale.is_finite() || scale <= 0. {
                return Err(QteaError::InvalidBody);
            }
            let remainder = (m.rows / 4) % 4;
            if remainder != 0 && record.bytes[8 + l.position_bytes - 1] >> (remainder * 2) != 0 {
                return Err(QteaError::InvalidBody);
            }
            for block in 0..m.rows / 4 {
                let code = record.bytes[8 + l.position_bytes + block];
                if code & 127 == 127 {
                    return Err(QteaError::InvalidBody);
                }
                let value = fp8(code) / scale;
                let row = block * 4 + usize::from(record.position_at(block));
                if !value.is_finite() || !(self.base_at(row, column) + value).is_finite() {
                    return Err(QteaError::InvalidBody);
                }
            }
        }
        for row in 0..m.rows {
            for column in 0..m.columns {
                if !self.base_at(row, column).is_finite() {
                    return Err(QteaError::InvalidBody);
                }
            }
        }
        Ok(())
    }
}

impl QteaResidualView<'_> {
    /// Original input-column index.
    pub fn column(&self) -> usize {
        u32_at(self.bytes, 0) as usize
    }
    /// Positive finite encoding multiplier, not its reciprocal.
    pub fn scale_multiplier(&self) -> f32 {
        f32_at(self.bytes, 4)
    }
    /// Selected correction for this row, or None when this record skips the row.
    ///
    /// # Errors
    /// Refuses an out-of-range row. Selected signed zero is preserved as Some.
    pub fn value_at_row(&self, row: usize) -> Result<Option<f32>, QteaError> {
        if row >= self.rows {
            return Err(QteaError::InvalidIndex);
        }
        let block = row / 4;
        Ok((usize::from(self.position_at(block)) == row % 4)
            .then(|| fp8(self.bytes[8 + self.positions + block]) / self.scale_multiplier()))
    }
    fn position_at(&self, block: usize) -> u8 {
        (self.bytes[8 + block / 4] >> ((block % 4) * 2)) & 3
    }
}

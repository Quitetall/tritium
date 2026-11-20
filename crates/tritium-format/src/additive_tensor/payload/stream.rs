//! Incremental counterpart of the shared canonical range validator.
use super::*;

/// Validate raw additive bytes in arbitrary successive chunks without retaining
/// a whole encoded payload or trit master. Counts stay borrowed; geometry stores
/// at most one range per u8-bounded plane, with fixed codec/scalar carry scratch.
/// This checks canonical bytes/scales, not decoded-weight sums or source custody.
pub struct AdditiveStreamValidator<'a> {
    plan: PayloadPlan<'a>,
    codec: SaltV2Codec,
    scratch: Vec<Trit>,
    pending: [u8; SCRATCH_TRITS],
    pending_len: usize,
    plane: usize,
    trit_at: usize,
    total: usize,
    maximum_scale: f32,
    poisoned: bool,
}
impl core::fmt::Debug for AdditiveStreamValidator<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AdditiveStreamValidator")
            .field("encoded_length", &self.plan.length)
            .field("plane_count", &self.plan.planes.len())
            .field("consumed", &self.total)
            .field("pending", &self.pending_len)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}
impl<'a> AdditiveStreamValidator<'a> {
    /// Construct from uniform geometry and exact raw byte length.
    /// # Errors
    /// Invalid geometry/law/codec/allocation/transport, sizing or allocation errors.
    pub fn new_uniform(
        layout: AdditiveLayout,
        planes: u8,
        length: usize,
    ) -> Result<Self, AdditiveTensorError> {
        Self::new(PayloadPlan::with_allocation(
            layout,
            Allocation::Uniform(planes),
            length,
        )?)
    }
    /// Construct with borrowed immutable per-tile counts and exact raw byte length.
    /// # Errors
    /// Same errors as new_uniform, plus malformed per-tile counts.
    pub fn new_per_tile(
        layout: AdditiveLayout,
        counts: &'a [u8],
        length: usize,
    ) -> Result<Self, AdditiveTensorError> {
        Self::new(PayloadPlan::with_allocation(
            layout,
            Allocation::PerTile(counts),
            length,
        )?)
    }
    fn new(plan: PayloadPlan<'a>) -> Result<Self, AdditiveTensorError> {
        Ok(Self {
            codec: salt_codec(plan.layout.codec)?,
            plan,
            scratch: reserved_payload(SCRATCH_TRITS)?,
            pending: [0; SCRATCH_TRITS],
            pending_len: 0,
            plane: 0,
            trit_at: 0,
            total: 0,
            maximum_scale: 0.,
            poisoned: false,
        })
    }
    /// Consume successive bytes; chunks need not align to planes/codewords/scalars.
    /// # Errors
    /// Trailing bytes, noncanonical codes/tails, invalid scales or prior poison.
    /// Any error poisons the validator; no later success can hide ignored errors.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), AdditiveTensorError> {
        self.push_with(bytes, |_| {}, |_| {})
    }
    fn push_with(
        &mut self,
        bytes: &[u8],
        mut trits: impl FnMut(&[Trit]),
        mut scales: impl FnMut(f32),
    ) -> Result<(), AdditiveTensorError> {
        if self.poisoned {
            return Err(AdditiveTensorError::StreamPoisoned);
        }
        let result = self.push_inner(bytes, &mut trits, &mut scales);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    fn push_inner(
        &mut self,
        mut bytes: &[u8],
        trit_values: &mut impl FnMut(&[Trit]),
        scale_values: &mut impl FnMut(f32),
    ) -> Result<(), AdditiveTensorError> {
        let total = self
            .total
            .checked_add(bytes.len())
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        if total > self.plan.length {
            return Err(AdditiveTensorError::PayloadLength {
                expected: self.plan.length,
                got: total,
            });
        }
        self.total = total;
        while !bytes.is_empty() {
            while self.plane < self.plan.planes.len()
                && self.trit_at == self.plan.planes[self.plane].trits
            {
                self.plane += 1;
                self.trit_at = 0;
            }
            let trits = if self.plane < self.plan.planes.len() {
                Some((self.plan.planes[self.plane].trits - self.trit_at).min(SCRATCH_TRITS))
            } else {
                None
            };
            let wanted = match trits {
                Some(n) => encoded_plane_len(self.codec, n)?,
                None => scale_width(self.plan.layout.law.precision)?,
            };
            let count = bytes.len().min(wanted - self.pending_len);
            self.pending[self.pending_len..self.pending_len + count]
                .copy_from_slice(&bytes[..count]);
            self.pending_len += count;
            bytes = &bytes[count..];
            if self.pending_len != wanted {
                continue;
            }
            if let Some(trits) = trits {
                let byte_at = encoded_plane_len(self.codec, self.trit_at)?;
                decode_window(
                    self.codec,
                    &self.pending[..wanted],
                    trits,
                    &mut self.scratch,
                    self.trit_at,
                    byte_at,
                )?;
                trit_values(&self.scratch);
                self.trit_at += trits;
            } else {
                validate_scale_chunk(
                    self.plan.layout,
                    self.plan.planes.len(),
                    &self.pending[..wanted],
                )?;
                let value = stored_scale(self.plan.layout.law.precision, &self.pending[..wanted])?;
                self.maximum_scale = self.maximum_scale.max(value);
                scale_values(value);
            }
            self.pending_len = 0;
        }
        Ok(())
    }
    /// Complete exact-length canonical validation and return maximum stored scale.
    /// The maximum is only a numerical bound input, not decoded-weight validation.
    /// Caller must separately validate decoded sums and source identity/custody.
    /// # Errors
    /// Ignored earlier errors, short payload or incomplete codeword/scalar.
    pub fn finish(self) -> Result<f32, AdditiveTensorError> {
        if self.poisoned {
            return Err(AdditiveTensorError::StreamPoisoned);
        }
        if self.total != self.plan.length || self.pending_len != 0 {
            return Err(AdditiveTensorError::PayloadLength {
                expected: self.plan.length,
                got: self.total,
            });
        }
        Ok(self.maximum_scale)
    }
}

/// Bounded owned decoding through the same incremental canonical validator.
///
/// Retains one decoded owner, not its encoded payload. Counts remain borrowed
/// until finish; the returned tensor then owns one compact index. Logical caps
/// include coefficients, widened scales, index and fixed codec/carry scratch,
/// excluding metadata, allocation slack and external source buffers/RSS.
pub struct AdditiveStreamDecoder<'a> {
    validator: AdditiveStreamValidator<'a>,
    trits: Vec<Trit>,
    scales: Vec<f32>,
}
impl core::fmt::Debug for AdditiveStreamDecoder<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AdditiveStreamDecoder")
            .field("validator", &self.validator)
            .field("decoded_trits", &self.trits.len())
            .field("decoded_scales", &self.scales.len())
            .finish_non_exhaustive()
    }
}
impl<'a> AdditiveStreamDecoder<'a> {
    /// Forecast logical decoded payload plus fixed codec/carry scratch. No
    /// coefficient, scale or compact index payload is allocated by this query.
    pub fn working_bytes(
        layout: AdditiveLayout,
        planes: &PlaneCounts,
        length: usize,
    ) -> Result<u64, AdditiveTensorError> {
        Self::sizes(&PayloadPlan::new(layout, planes, length)?)
    }
    fn sizes(plan: &PayloadPlan<'_>) -> Result<u64, AdditiveTensorError> {
        let index = match plan.allocation {
            Allocation::Uniform(_) => 0,
            Allocation::PerTile(counts) => PerTilePlan::new(plan.layout, counts)
                .and_then(|p| p.buffer_len_bytes())
                .map_err(AdditiveTensorError::InvalidView)?,
        };
        plan.scale_count
            .checked_mul(4)
            .and_then(|n| n.checked_add(plan.trit_count))
            .and_then(|n| n.checked_add(index))
            .and_then(|n| n.checked_add(2 * SCRATCH_TRITS))
            .map(|n| n as u64)
            .ok_or(AdditiveTensorError::LengthOverflow)
    }
    /// Validate geometry and cap before allocating decoded owner payloads.
    /// Unsupported transport/laws and malformed lengths remain fail-closed.
    pub fn new(
        layout: AdditiveLayout,
        planes: &'a PlaneCounts,
        length: usize,
        max_working_bytes: u64,
    ) -> Result<Self, AdditiveTensorError> {
        let plan = PayloadPlan::new(layout, planes, length)?;
        let required = Self::sizes(&plan)?;
        if required > max_working_bytes {
            return Err(AdditiveTensorError::DecodeWorkingLimit {
                required,
                limit: max_working_bytes,
            });
        }
        let trits = reserved_payload(plan.trit_count)?;
        let scales = reserved_payload(plan.scale_count)?;
        Ok(Self {
            validator: AdditiveStreamValidator::new(plan)?,
            trits,
            scales,
        })
    }
    /// Consume arbitrary successive windows. Any error poisons the shared
    /// validator; ignoring it can never make finish succeed.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), AdditiveTensorError> {
        let Self {
            validator,
            trits,
            scales,
        } = self;
        validator.push_with(
            bytes,
            |values| trits.extend_from_slice(values),
            |value| scales.push(value),
        )
    }
    /// Return the single validated owner only after exact complete validation.
    /// Enclosing blob/package/source verification remains the loader's duty.
    pub fn finish(self) -> Result<AdditiveTensor, AdditiveTensorError> {
        let Self {
            validator,
            trits,
            scales,
        } = self;
        let layout = validator.plan.layout;
        let allocation = validator.plan.allocation;
        let (t, s, length, total) = (
            validator.plan.trit_count,
            validator.plan.scale_count,
            validator.plan.length,
            validator.total,
        );
        validator.finish()?;
        if trits.len() != t || scales.len() != s {
            return Err(AdditiveTensorError::PayloadLength {
                expected: length,
                got: total,
            });
        }
        match allocation {
            Allocation::Uniform(count) => AdditiveTensor::new(layout, count, trits, scales),
            Allocation::PerTile(counts) => {
                AdditiveTensor::new_per_tile(layout, copy_payload(counts)?, trits, scales)
            }
        }
    }
}

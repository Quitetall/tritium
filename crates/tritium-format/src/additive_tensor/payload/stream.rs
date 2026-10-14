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
        if self.poisoned {
            return Err(AdditiveTensorError::StreamPoisoned);
        }
        let result = self.push_inner(bytes);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    fn push_inner(&mut self, mut bytes: &[u8]) -> Result<(), AdditiveTensorError> {
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
                self.trit_at += trits;
            } else {
                validate_scale_chunk(
                    self.plan.layout,
                    self.plan.planes.len(),
                    &self.pending[..wanted],
                )?;
                self.maximum_scale = self.maximum_scale.max(stored_scale(
                    self.plan.layout.law.precision,
                    &self.pending[..wanted],
                )?);
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

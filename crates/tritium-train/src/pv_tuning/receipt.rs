use super::PvTuningError;
use super::wire::Reader;

const MAGIC: [u8; 4] = *b"PVR1";
const CHECKSUM_BYTES: usize = 32;
const BODY_BYTES: usize = 100;

/// Deterministic evidence for one alternating P/V step.
#[derive(Clone, Debug, PartialEq)]
pub struct PvStepReceipt {
    pub(super) optimizer_step: u64,
    pub(super) selected_code_units: usize,
    pub(super) changed_code_units: usize,
    pub(super) trust_limited_code_units: usize,
    pub(super) changed_scales: usize,
    pub(super) v_surrogate_before: f64,
    pub(super) v_surrogate_after: f64,
    pub(super) relative_code_change: f64,
    pub(super) representation_digest: [u8; 32],
}

impl PvStepReceipt {
    /// Completed one-based optimizer step.
    #[must_use]
    pub const fn optimizer_step(&self) -> u64 {
        self.optimizer_step
    }

    /// Scalar or structured code units selected for projection.
    #[must_use]
    pub const fn selected_code_units(&self) -> usize {
        self.selected_code_units
    }

    /// Selected units whose deployed codes changed.
    #[must_use]
    pub const fn changed_code_units(&self) -> usize {
        self.changed_code_units
    }

    /// Improving candidates rejected only by trust-ratio budget.
    #[must_use]
    pub const fn trust_limited_code_units(&self) -> usize {
        self.trust_limited_code_units
    }

    /// f16 scale bit patterns changed by P step.
    #[must_use]
    pub const fn changed_scales(&self) -> usize {
        self.changed_scales
    }

    /// Squared-distance projection surrogate before V step.
    #[must_use]
    pub const fn v_surrogate_before(&self) -> f64 {
        self.v_surrogate_before
    }

    /// Squared-distance projection surrogate after V step; never greater than before.
    #[must_use]
    pub const fn v_surrogate_after(&self) -> f64 {
        self.v_surrogate_after
    }

    /// L2 code-movement norm divided by post-P representation norm.
    #[must_use]
    pub const fn relative_code_change(&self) -> f64 {
        self.relative_code_change
    }

    /// Digest of exact post-step deployed representation.
    #[must_use]
    pub const fn representation_digest(&self) -> [u8; 32] {
        self.representation_digest
    }

    /// Canonical checksum-bound wire form used by durable model campaign overlays.
    #[must_use]
    pub fn checkpoint_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(BODY_BYTES + CHECKSUM_BYTES);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&self.optimizer_step.to_le_bytes());
        for value in [
            self.selected_code_units,
            self.changed_code_units,
            self.trust_limited_code_units,
            self.changed_scales,
        ] {
            out.extend_from_slice(&(value as u64).to_le_bytes());
        }
        for value in [
            self.v_surrogate_before,
            self.v_surrogate_after,
            self.relative_code_change,
        ] {
            out.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        out.extend_from_slice(&self.representation_digest);
        let checksum = blake3::hash(&out);
        out.extend_from_slice(checksum.as_bytes());
        out
    }

    /// Parse a strict campaign receipt, rejecting corruption and invalid metrics.
    pub fn resume(bytes: &[u8]) -> Result<Self, PvTuningError> {
        if bytes.len() != BODY_BYTES + CHECKSUM_BYTES {
            return Err(PvTuningError::checkpoint(
                "PV receipt payload length mismatch",
            ));
        }
        let (body, checksum) = bytes.split_at(BODY_BYTES);
        if blake3::hash(body).as_bytes() != checksum {
            return Err(PvTuningError::checkpoint("PV receipt checksum mismatch"));
        }
        let mut reader = Reader::new(body);
        if reader.array::<4>()? != MAGIC {
            return Err(PvTuningError::checkpoint("bad PV receipt magic"));
        }
        let optimizer_step = reader.u64()?;
        if optimizer_step == 0 {
            return Err(PvTuningError::checkpoint("PV receipt step is zero"));
        }
        let selected_code_units = reader.usize()?;
        let changed_code_units = reader.usize()?;
        let trust_limited_code_units = reader.usize()?;
        let changed_scales = reader.usize()?;
        if changed_code_units > selected_code_units
            || trust_limited_code_units > selected_code_units
        {
            return Err(PvTuningError::checkpoint(
                "PV receipt unit counts are inconsistent",
            ));
        }
        let v_surrogate_before = reader.f64()?;
        let v_surrogate_after = reader.f64()?;
        let relative_code_change = reader.f64()?;
        if !v_surrogate_before.is_finite()
            || !v_surrogate_after.is_finite()
            || !relative_code_change.is_finite()
            || v_surrogate_before < 0.0
            || v_surrogate_after < 0.0
            || relative_code_change < 0.0
            || v_surrogate_after > v_surrogate_before
        {
            return Err(PvTuningError::checkpoint("PV receipt metrics are invalid"));
        }
        let representation_digest = reader.array::<32>()?;
        if reader.remaining() != 0 {
            return Err(PvTuningError::checkpoint("PV receipt has trailing bytes"));
        }
        Ok(Self {
            optimizer_step,
            selected_code_units,
            changed_code_units,
            trust_limited_code_units,
            changed_scales,
            v_surrogate_before,
            v_surrogate_after,
            relative_code_change,
            representation_digest,
        })
    }
}

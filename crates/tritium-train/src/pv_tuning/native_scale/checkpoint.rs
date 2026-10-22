use super::super::wire::{Reader, read_adam_state, write_adam_state};
use super::*;

const MAGIC: [u8; 4] = *b"TNSR";
const VERSION: u8 = 1;
pub(super) const FIXED_BYTES: u64 = 4 + 1 + 32 + 32 + 8 + 8 + 32;

impl NativeScaleRefinementSession {
    /// Exact native scale/moment checkpoint, not a hard inference artifact.
    /// The exact parent representation and frozen request are required to resume.
    pub fn checkpoint_bytes(&self) -> Result<Vec<u8>, PvTuningError> {
        let len = usize::try_from(self.ledger.checkpoint_bytes).map_err(|_| size_overflow())?;
        let mut out = reserved(len)?;
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.extend_from_slice(self.plan_id.as_bytes());
        out.extend_from_slice(&self.parent);
        out.extend_from_slice(&self.completed_updates.to_le_bytes());
        out.extend_from_slice(&(self.owners.len() as u64).to_le_bytes());
        for owner in &self.owners {
            out.extend_from_slice(&(owner.weight.scales().len() as u64).to_le_bytes());
            for &scale in owner.weight.scales() {
                match owner.weight.layout().law.precision {
                    ScalePrecision::F16 => {
                        out.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes())
                    }
                    ScalePrecision::Bf16 => {
                        out.extend_from_slice(&bf16::from_f32(scale).to_bits().to_le_bytes())
                    }
                    ScalePrecision::F32 => out.extend_from_slice(&scale.to_bits().to_le_bytes()),
                    _ => {
                        return Err(PvTuningError::checkpoint(
                            "unsupported native scale precision",
                        ));
                    }
                }
            }
            write_adam_state(&mut out, &owner.state);
        }
        out.extend_from_slice(blake3::hash(&out).as_bytes());
        debug_assert_eq!(out.len(), len);
        Ok(out)
    }

    /// Resume exact native state with frozen parent tensors in declared owner order.
    ///
    /// Rejects wrong parents/plans, legacy formats, truncation, extra bytes,
    /// invalid moments/scales and out-of-range cursors before returning any
    /// session. Valid bytes do not prove gradients or objective execution.
    pub fn resume(
        plan: NativeRefinementPlan,
        parents: Vec<AdditiveTensor>,
        bytes: &[u8],
    ) -> Result<Self, PvTuningError> {
        let admission = admit(&plan, &parents)?;
        if bytes.len() as u64 != admission.ledger.checkpoint_bytes {
            return Err(PvTuningError::checkpoint(
                "native checkpoint length mismatch",
            ));
        }
        let (body, checksum) = bytes.split_at(bytes.len() - 32);
        if blake3::hash(body).as_bytes() != checksum {
            return Err(PvTuningError::checkpoint(
                "native checkpoint checksum mismatch",
            ));
        }
        let mut reader = Reader::new(body);
        if reader.array::<4>()? != MAGIC || reader.u8()? != VERSION {
            return Err(PvTuningError::checkpoint(
                "unsupported native checkpoint format",
            ));
        }
        if reader.array::<32>()? != *admission.plan_id.as_bytes()
            || reader.array::<32>()? != admission.parent
        {
            return Err(PvTuningError::checkpoint(
                "native checkpoint plan/parent mismatch",
            ));
        }
        let completed = reader.u64()?;
        if completed > plan.training.len() as u64 || reader.usize()? != parents.len() {
            return Err(PvTuningError::checkpoint(
                "native checkpoint cursor/owner count mismatch",
            ));
        }
        let mut owners = reserved(parents.len())?;
        for mut weight in parents {
            let len = weight.scales().len();
            if reader.usize()? != len {
                return Err(PvTuningError::checkpoint(
                    "native checkpoint scale count mismatch",
                ));
            }
            let mut values = reserved(len)?;
            for _ in 0..len {
                values.push(match weight.layout().law.precision {
                    ScalePrecision::F16 => f16::from_bits(reader.u16()?).to_f32(),
                    ScalePrecision::Bf16 => bf16::from_bits(reader.u16()?).to_f32(),
                    ScalePrecision::F32 => f32::from_bits(reader.u32()?),
                    _ => {
                        return Err(PvTuningError::checkpoint(
                            "unsupported native scale precision",
                        ));
                    }
                });
            }
            let state = read_adam_state(&mut reader, len)?;
            if completed > 0 {
                check_corrected_moments(admission.optimizer, completed, &state)
                    .map_err(PvTuningError::checkpoint)?;
            }
            if completed == 0
                && (values
                    .iter()
                    .zip(weight.scales())
                    .any(|(a, b)| a.to_bits() != b.to_bits())
                    || state.m.iter().chain(&state.v).any(|v| v.to_bits() != 0))
            {
                return Err(PvTuningError::checkpoint(
                    "native zero-step checkpoint differs from initial state",
                ));
            }
            weight.replace_scales(values).map_err(|e| {
                PvTuningError::checkpoint(format!("invalid native checkpoint scales: {e:?}"))
            })?;
            owners.push(Owner { weight, state });
        }
        if reader.remaining() != 0 {
            return Err(PvTuningError::checkpoint(
                "native checkpoint has trailing bytes",
            ));
        }
        Ok(Self::from_admitted(plan, admission, owners, completed))
    }
}

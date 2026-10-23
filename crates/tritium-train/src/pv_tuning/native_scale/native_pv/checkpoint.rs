use super::*;
use crate::pv_tuning::wire::{Reader, write_adam_state};
use tritium_core::Trit;

const MAGIC: [u8; 4] = *b"TNPV";
const VERSION: u8 = 1;
pub(super) const FIXED_BYTES: u64 = 4 + 1 + 32 + 32 + 8 + 8 + 32;

impl NativePvRefinementSession {
    /// Exact TNPV state, not an inference package or numerical replay proof.
    /// Serialization is separately admitted by the frozen checkpoint byte cap.
    pub fn checkpoint_bytes(&self) -> Result<Vec<u8>, PvTuningError> {
        let length = usize::try_from(self.ledger.checkpoint_bytes).map_err(|_| size_overflow())?;
        let mut bytes = reserved(length)?;
        bytes.extend_from_slice(&MAGIC);
        bytes.push(VERSION);
        bytes.extend_from_slice(self.plan_id.as_bytes());
        bytes.extend_from_slice(&self.parent);
        bytes.extend_from_slice(&self.completed.to_le_bytes());
        bytes.extend_from_slice(&(self.owners.len() as u64).to_le_bytes());
        for owner in &self.owners {
            bytes.extend_from_slice(&(owner.weight.trits().len() as u64).to_le_bytes());
            bytes.extend(owner.weight.trits().iter().map(|code| code.get() as u8));
            bytes.extend_from_slice(&(owner.weight.scales().len() as u64).to_le_bytes());
            for &value in owner.weight.scales() {
                match owner.weight.layout().law.precision {
                    ScalePrecision::F16 => {
                        bytes.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes())
                    }
                    ScalePrecision::Bf16 => {
                        bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes())
                    }
                    ScalePrecision::F32 => bytes.extend_from_slice(&value.to_bits().to_le_bytes()),
                    _ => {
                        return Err(PvTuningError::checkpoint(
                            "unsupported native PV scale precision",
                        ));
                    }
                }
            }
            write_adam_state(&mut bytes, &owner.scale);
            bytes.extend_from_slice(&(owner.code.m.len() as u64).to_le_bytes());
            write_adam_state(&mut bytes, &owner.code);
        }
        bytes.extend_from_slice(blake3::hash(&bytes).as_bytes());
        debug_assert_eq!(bytes.len(), length);
        Ok(bytes)
    }

    /// Resume exact code/scale/moment state against the original parents and plan.
    ///
    /// Rejects other checkpoint families, bad caps/identity/shape/checksum, invalid
    /// codec/scales/moments and changed zero-step state. Valid contents do not
    /// prove any gradient, objective, parent custody or prior training execution.
    pub fn resume(
        plan: NativeRefinementPlan,
        parents: Vec<AdditiveTensor>,
        bytes: &[u8],
    ) -> Result<Self, PvTuningError> {
        let mut session = admit(plan, &parents)?;
        if bytes.len() as u64 != session.ledger.checkpoint_bytes {
            return Err(PvTuningError::checkpoint(
                "native PV checkpoint length mismatch",
            ));
        }
        let (body, checksum) = bytes.split_at(bytes.len() - 32);
        if blake3::hash(body).as_bytes() != checksum {
            return Err(PvTuningError::checkpoint(
                "native PV checkpoint checksum mismatch",
            ));
        }
        let mut reader = Reader::new(body);
        if reader.array::<4>()? != MAGIC || reader.u8()? != VERSION {
            return Err(PvTuningError::checkpoint(
                "unsupported native PV checkpoint format",
            ));
        }
        if reader.array::<32>()? != *session.plan_id.as_bytes()
            || reader.array::<32>()? != session.parent
        {
            return Err(PvTuningError::checkpoint(
                "native PV checkpoint plan/parent mismatch",
            ));
        }
        session.completed = reader.u64()?;
        if session.completed > session.plan.training.len() as u64
            || reader.usize()? != parents.len()
        {
            return Err(PvTuningError::checkpoint(
                "native PV checkpoint cursor/owner mismatch",
            ));
        }
        for mut weight in parents {
            let active = active_count(&weight)?;
            if reader.usize()? != weight.trits().len() {
                return Err(PvTuningError::checkpoint(
                    "native PV checkpoint trit count mismatch",
                ));
            }
            let mut codes = reserved(weight.trits().len())?;
            for _ in 0..weight.trits().len() {
                codes.push(
                    Trit::from_i8(reader.u8()? as i8).map_err(|_| {
                        PvTuningError::checkpoint("invalid native PV checkpoint trit")
                    })?,
                );
            }
            let scales = weight.scales().len();
            if reader.usize()? != scales {
                return Err(PvTuningError::checkpoint(
                    "native PV checkpoint scale count mismatch",
                ));
            }
            let mut values = reserved(scales)?;
            for _ in 0..scales {
                values.push(match weight.layout().law.precision {
                    ScalePrecision::F16 => f16::from_bits(reader.u16()?).to_f32(),
                    ScalePrecision::Bf16 => bf16::from_bits(reader.u16()?).to_f32(),
                    ScalePrecision::F32 => f32::from_bits(reader.u32()?),
                    _ => {
                        return Err(PvTuningError::checkpoint(
                            "unsupported native PV scale precision",
                        ));
                    }
                });
            }
            let scale = read_state(&mut reader, scales)?;
            if reader.usize()? != active {
                return Err(PvTuningError::checkpoint(
                    "native PV checkpoint code count mismatch",
                ));
            }
            let code = read_state(&mut reader, active)?;
            if session.completed == 0 {
                if codes != weight.trits()
                    || values
                        .iter()
                        .zip(weight.scales())
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                    || scale
                        .m
                        .iter()
                        .chain(&scale.v)
                        .chain(&code.m)
                        .chain(&code.v)
                        .any(|v| v.to_bits() != 0)
                {
                    return Err(PvTuningError::checkpoint(
                        "native PV zero-step state differs from parent",
                    ));
                }
            } else {
                check_corrected_moments(session.continuous, session.completed, &scale)
                    .map_err(PvTuningError::checkpoint)?;
                check_corrected_moments(session.code_optimizer, session.completed, &code)
                    .map_err(PvTuningError::checkpoint)?;
            }
            weight.replace_scales(values).map_err(|e| {
                PvTuningError::checkpoint(format!("native PV checkpoint scales: {e:?}"))
            })?;
            weight.replace_trits(codes).map_err(|e| {
                PvTuningError::checkpoint(format!("native PV checkpoint codes: {e:?}"))
            })?;
            session.owners.push(PvOwner {
                weight,
                scale,
                code,
            });
        }
        if reader.remaining() != 0 {
            return Err(PvTuningError::checkpoint(
                "native PV checkpoint trailing body",
            ));
        }
        Ok(session)
    }
}

fn read_state(reader: &mut Reader<'_>, length: usize) -> Result<AdamState, PvTuningError> {
    let mut read_values = |variance: bool| -> Result<Vec<f32>, PvTuningError> {
        let mut values = reserved(length)?;
        for _ in 0..length {
            let value = f32::from_bits(reader.u32()?);
            if !value.is_finite() || (variance && value.is_sign_negative()) {
                return Err(PvTuningError::checkpoint(
                    "native PV checkpoint invalid optimizer moment",
                ));
            }
            values.push(value);
        }
        Ok(values)
    };
    Ok(AdamState {
        m: read_values(false)?,
        v: read_values(true)?,
    })
}

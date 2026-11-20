use super::*;
use std::io::{Cursor, Seek};

mod stream;
pub(super) use stream::*;
pub(super) const FIXED_BYTES: u64 = 117;

impl NativePvRefinementSession {
    /// Exact TNPV state, not an inference package or numerical replay proof.
    /// Serialization is separately admitted by the frozen checkpoint byte cap.
    pub fn checkpoint_bytes(&self) -> Result<Vec<u8>, PvTuningError> {
        let length = usize::try_from(self.ledger.checkpoint_bytes).map_err(|_| size_overflow())?;
        let mut bytes = reserved(length)?;
        bytes.extend_from_slice(
            &Header {
                plan: self.plan_id,
                parent: self.parent,
                completed: self.completed,
                owners: self.owners.len() as u64,
            }
            .bytes(),
        );
        for owner in &self.owners {
            write_owner::<4>(&mut bytes, owner).map_err(numerical_error)?;
        }
        bytes.extend_from_slice(blake3::hash(&bytes).as_bytes());
        debug_assert_eq!(bytes.len(), length);
        Ok(bytes)
    }

    /// Resume exact code/scale/moment state against the original parents and plan.
    /// Checksum integrity is not proof of gradient/objective execution or custody.
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
        let mut reader = Cursor::new(body);
        let header = read_header(&mut reader).map_err(numerical_error)?;
        if header.plan != session.plan_id || header.parent != session.parent {
            return Err(PvTuningError::checkpoint(
                "native PV checkpoint plan/parent mismatch",
            ));
        }
        if header.completed > session.plan.training.len() as u64
            || header.owners != parents.len() as u64
        {
            return Err(PvTuningError::checkpoint(
                "native PV checkpoint cursor/owner mismatch",
            ));
        }
        session.completed = header.completed;
        let updater = PvUpdater {
            continuous: session.continuous,
            code_optimizer: session.code_optimizer,
            fraction: session.fraction,
            trust: session.trust,
        };
        for weight in parents {
            session.owners.push(
                read_owner::<4>(&mut reader, weight, header.completed, updater)
                    .map_err(numerical_error)?,
            );
        }
        if reader
            .stream_position()
            .map_err(|e| PvTuningError::checkpoint(e.to_string()))?
            != body.len() as u64
        {
            return Err(PvTuningError::checkpoint(
                "native PV checkpoint trailing body",
            ));
        }
        Ok(session)
    }
}

fn numerical_error(error: WireError) -> PvTuningError {
    match error {
        WireError::Invalid(error) => error,
        WireError::Io(error) => {
            PvTuningError::checkpoint(format!("native PV checkpoint I/O: {error}"))
        }
    }
}

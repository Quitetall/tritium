use crate::optim::AdamState;

use super::PvTuningSession;

impl PvTuningSession {
    /// Digest exact representation, optimizer state, completed step, and active cursor.
    #[must_use]
    pub fn state_digest(&self) -> [u8; 32] {
        let mut hash = blake3::Hasher::new();
        hash.update(b"tritium.pv-tuning-state.v1\0");
        hash.update(&self.parent_digest);
        hash.update(&self.config.recipe_digest());
        hash.update(&self.completed_step.to_le_bytes());
        hash.update(&self.weight.digest());
        hash_adam(&mut hash, &self.code_state);
        hash_adam(&mut hash, &self.scale_state);
        match &self.blockwise {
            None => {
                hash.update(&[0]);
            }
            Some(state) => {
                hash.update(&[1]);
                hash.update(&state.optimizer_step.to_le_bytes());
                hash.update(&(state.max_block_elements as u64).to_le_bytes());
                hash.update(&(state.next_offset as u64).to_le_bytes());
                hash.update(&(state.scale_gradient.len() as u64).to_le_bytes());
                for value in &state.scale_gradient {
                    hash.update(&value.to_bits().to_le_bytes());
                }
            }
        };
        *hash.finalize().as_bytes()
    }
}

fn hash_adam(hash: &mut blake3::Hasher, state: &AdamState) {
    hash.update(&(state.m.len() as u64).to_le_bytes());
    for value in &state.m {
        hash.update(&value.to_bits().to_le_bytes());
    }
    hash.update(&(state.v.len() as u64).to_le_bytes());
    for value in &state.v {
        hash.update(&value.to_bits().to_le_bytes());
    }
}

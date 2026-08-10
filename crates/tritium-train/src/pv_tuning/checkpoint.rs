use half::f16;

use super::wire::{Reader, read_adam_state, write_adam_state};
use super::{
    PvBlockwiseState, PvTernaryPlane, PvTernaryStructure, PvTernaryWeight, PvTuningConfig,
    PvTuningError, PvTuningSession,
};

const MAGIC: [u8; 4] = *b"TPV1";
const VERSION: u8 = 1;
const CHECKSUM_BYTES: usize = 32;
const FIXED_BODY_BYTES: usize = 103;
const BLOCKWISE_MAGIC: [u8; 5] = *b"TPVB1";

impl PvTuningSession {
    /// Serialize representation, Adam states, recipe/parent identities, and step with
    /// end-to-end BLAKE3 checksum.
    ///
    /// # Errors
    /// Returns error if host geometry cannot be represented by format.
    pub fn checkpoint_bytes(&self) -> Result<Vec<u8>, PvTuningError> {
        if self.blockwise.is_some() {
            return Err(PvTuningError::checkpoint(
                "in-flight blockwise step requires a campaign checkpoint",
            ));
        }
        self.base_checkpoint_bytes()
    }

    fn base_checkpoint_bytes(&self) -> Result<Vec<u8>, PvTuningError> {
        let rows = u64::try_from(self.weight.rows)
            .map_err(|_| PvTuningError::checkpoint("rows exceed checkpoint range"))?;
        let cols = u64::try_from(self.weight.cols)
            .map_err(|_| PvTuningError::checkpoint("cols exceed checkpoint range"))?;
        let group_size = u64::try_from(self.weight.group_size)
            .map_err(|_| PvTuningError::checkpoint("group_size exceeds checkpoint range"))?;
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.extend_from_slice(&self.config.digest());
        out.extend_from_slice(&self.parent_digest);
        out.extend_from_slice(&self.completed_step.to_le_bytes());
        out.extend_from_slice(&rows.to_le_bytes());
        out.extend_from_slice(&cols.to_le_bytes());
        out.extend_from_slice(&group_size.to_le_bytes());
        out.push(self.weight.structure.wire_tag());
        out.push(self.weight.planes.len() as u8);
        for plane in &self.weight.planes {
            for &trit in &plane.trits {
                out.push(trit as u8);
            }
            for scale in &plane.scales {
                out.extend_from_slice(&scale.to_bits().to_le_bytes());
            }
        }
        write_adam_state(&mut out, &self.code_state);
        write_adam_state(&mut out, &self.scale_state);
        let checksum = blake3::hash(&out);
        out.extend_from_slice(checksum.as_bytes());
        Ok(out)
    }

    /// Serialize an in-flight bounded-gradient cursor without materializing its remaining gradient.
    pub fn blockwise_checkpoint_bytes(&self) -> Result<Vec<u8>, PvTuningError> {
        let state = self
            .blockwise
            .as_ref()
            .ok_or_else(|| PvTuningError::checkpoint("no blockwise step is active"))?;
        let base = self.base_checkpoint_bytes()?;
        let base_len = u64::try_from(base.len())
            .map_err(|_| PvTuningError::checkpoint("checkpoint size exceeds wire range"))?;
        let optimizer_step = state.optimizer_step.to_le_bytes();
        let max_block_elements = u64::try_from(state.max_block_elements)
            .map_err(|_| PvTuningError::checkpoint("block size exceeds wire range"))?
            .to_le_bytes();
        let next_offset = u64::try_from(state.next_offset)
            .map_err(|_| PvTuningError::checkpoint("block cursor exceeds wire range"))?
            .to_le_bytes();
        let scale_count = u64::try_from(state.scale_gradient.len())
            .map_err(|_| PvTuningError::checkpoint("scale count exceeds wire range"))?
            .to_le_bytes();
        let scale_bytes = state
            .scale_gradient
            .len()
            .checked_mul(8)
            .ok_or_else(|| PvTuningError::checkpoint("checkpoint size overflow"))?;
        let capacity = BLOCKWISE_MAGIC
            .len()
            .checked_add(8 * 5)
            .and_then(|size| size.checked_add(base.len()))
            .and_then(|size| size.checked_add(scale_bytes))
            .and_then(|size| size.checked_add(CHECKSUM_BYTES))
            .ok_or_else(|| PvTuningError::checkpoint("checkpoint size overflow"))?;
        let mut out = Vec::new();
        out.try_reserve_exact(capacity)
            .map_err(|_| PvTuningError::checkpoint("checkpoint allocation failed"))?;
        out.extend_from_slice(&BLOCKWISE_MAGIC);
        out.extend_from_slice(&base_len.to_le_bytes());
        out.extend_from_slice(&base);
        out.extend_from_slice(&optimizer_step);
        out.extend_from_slice(&max_block_elements);
        out.extend_from_slice(&next_offset);
        out.extend_from_slice(&scale_count);
        for value in &state.scale_gradient {
            out.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let checksum = blake3::hash(&out);
        out.extend_from_slice(checksum.as_bytes());
        Ok(out)
    }

    /// Resume only when checksum, parent, recipe, geometry, and payload all match.
    ///
    /// # Errors
    /// Any mismatch, truncation, trailing byte, invalid float, or invalid representation
    /// returns [`PvTuningError::Checkpoint`].
    pub fn resume(
        parent: PvTernaryWeight,
        config: PvTuningConfig,
        bytes: &[u8],
    ) -> Result<Self, PvTuningError> {
        parent
            .validate()
            .map_err(|error| PvTuningError::checkpoint(error.to_string()))?;
        let body_len = bytes
            .len()
            .checked_sub(CHECKSUM_BYTES)
            .ok_or_else(|| PvTuningError::checkpoint("checkpoint is truncated"))?;
        let (body, checksum) = bytes.split_at(body_len);
        if blake3::hash(body).as_bytes() != checksum {
            return Err(PvTuningError::checkpoint("checkpoint checksum mismatch"));
        }
        if body.len() != expected_body_len(&parent)? {
            return Err(PvTuningError::checkpoint(
                "checkpoint payload length mismatch",
            ));
        }
        parse_body(parent, config, body)
    }

    /// Resume a strict in-flight blockwise cursor bound to its exact parent and recipe.
    pub fn resume_blockwise(
        parent: PvTernaryWeight,
        config: PvTuningConfig,
        bytes: &[u8],
    ) -> Result<Self, PvTuningError> {
        let body_len = bytes
            .len()
            .checked_sub(CHECKSUM_BYTES)
            .ok_or_else(|| PvTuningError::checkpoint("checkpoint is truncated"))?;
        let (body, checksum) = bytes.split_at(body_len);
        if blake3::hash(body).as_bytes() != checksum {
            return Err(PvTuningError::checkpoint("checkpoint checksum mismatch"));
        }
        let mut reader = Reader::new(body);
        if reader.array::<5>()? != BLOCKWISE_MAGIC {
            return Err(PvTuningError::checkpoint("bad blockwise checkpoint magic"));
        }
        let base_len = reader.usize()?;
        let base = reader.bytes(base_len)?;
        let mut session = Self::resume(parent, config, base)?;
        let optimizer_step = reader.u64()?;
        let expected_step = session
            .completed_step
            .checked_add(1)
            .ok_or_else(|| PvTuningError::checkpoint("step counter overflow"))?;
        if optimizer_step != expected_step {
            return Err(PvTuningError::checkpoint(
                "blockwise optimizer step is not the exact next step",
            ));
        }
        let max_block_elements = reader.usize()?;
        if max_block_elements == 0 {
            return Err(PvTuningError::checkpoint(
                "blockwise max block size is zero",
            ));
        }
        let next_offset = reader.usize()?;
        if next_offset > session.weight.len() {
            return Err(PvTuningError::checkpoint(
                "blockwise cursor exceeds weight length",
            ));
        }
        let scale_count = reader.usize()?;
        if scale_count != session.weight.total_scale_count() {
            return Err(PvTuningError::checkpoint(
                "blockwise scale-gradient count mismatch",
            ));
        }
        let mut scale_gradient = Vec::new();
        scale_gradient
            .try_reserve_exact(scale_count)
            .map_err(|_| PvTuningError::checkpoint("scale-gradient allocation failed"))?;
        for _ in 0..scale_count {
            let value = reader.f64()?;
            if !value.is_finite() {
                return Err(PvTuningError::checkpoint(
                    "scale gradient contains a non-finite value",
                ));
            }
            scale_gradient.push(value);
        }
        if reader.remaining() != 0 {
            return Err(PvTuningError::checkpoint(
                "blockwise checkpoint has trailing bytes",
            ));
        }
        session.blockwise = Some(PvBlockwiseState {
            optimizer_step,
            max_block_elements,
            next_offset,
            scale_gradient,
        });
        Ok(session)
    }
}

fn parse_body(
    parent: PvTernaryWeight,
    config: PvTuningConfig,
    body: &[u8],
) -> Result<PvTuningSession, PvTuningError> {
    let mut reader = Reader::new(body);
    if reader.array::<4>()? != MAGIC {
        return Err(PvTuningError::checkpoint("bad checkpoint magic"));
    }
    if reader.u8()? != VERSION {
        return Err(PvTuningError::checkpoint("unsupported checkpoint version"));
    }
    if reader.array::<32>()? != config.digest() {
        return Err(PvTuningError::checkpoint("recipe identity mismatch"));
    }
    let parent_digest = parent.digest();
    if reader.array::<32>()? != parent_digest {
        return Err(PvTuningError::checkpoint("parent identity mismatch"));
    }
    let completed_step = reader.u64()?;
    if reader.usize()? != parent.rows
        || reader.usize()? != parent.cols
        || reader.usize()? != parent.group_size
    {
        return Err(PvTuningError::checkpoint("weight geometry mismatch"));
    }
    if PvTernaryStructure::from_wire_tag(reader.u8()?)
        .ok_or_else(|| PvTuningError::checkpoint("unknown structure tag"))?
        != parent.structure
        || usize::from(reader.u8()?) != parent.planes.len()
    {
        return Err(PvTuningError::checkpoint("weight representation mismatch"));
    }
    let mut planes = Vec::with_capacity(parent.planes.len());
    for _ in 0..parent.planes.len() {
        let mut trits = Vec::with_capacity(parent.len());
        for _ in 0..parent.len() {
            trits.push(reader.u8()? as i8);
        }
        let mut scales = Vec::with_capacity(parent.scale_count_per_plane());
        for _ in 0..parent.scale_count_per_plane() {
            scales.push(f16::from_bits(reader.u16()?));
        }
        planes.push(PvTernaryPlane { trits, scales });
    }
    let weight = PvTernaryWeight::new(
        parent.rows,
        parent.cols,
        parent.group_size,
        parent.structure,
        planes,
    )
    .map_err(|error| PvTuningError::checkpoint(error.to_string()))?;
    let code_state = read_adam_state(&mut reader, parent.len())?;
    let scale_state = read_adam_state(&mut reader, parent.total_scale_count())?;
    if reader.remaining() != 0 {
        return Err(PvTuningError::checkpoint("checkpoint has trailing bytes"));
    }
    Ok(PvTuningSession {
        parent_digest,
        config,
        weight,
        scale_state,
        code_state,
        completed_step,
        blockwise: None,
    })
}

pub(super) fn encoded_checkpoint_len(session: &PvTuningSession) -> Result<usize, PvTuningError> {
    let base = expected_body_len(&session.weight)?
        .checked_add(CHECKSUM_BYTES)
        .ok_or_else(|| PvTuningError::checkpoint("checkpoint size overflow"))?;
    let Some(state) = &session.blockwise else {
        return Ok(base);
    };
    state
        .scale_gradient
        .len()
        .checked_mul(core::mem::size_of::<f64>())
        .and_then(|scale_bytes| {
            BLOCKWISE_MAGIC
                .len()
                .checked_add(8 * 5)
                .and_then(|size| size.checked_add(base))
                .and_then(|size| size.checked_add(scale_bytes))
                .and_then(|size| size.checked_add(CHECKSUM_BYTES))
        })
        .ok_or_else(|| PvTuningError::checkpoint("checkpoint size overflow"))
}

fn expected_body_len(parent: &PvTernaryWeight) -> Result<usize, PvTuningError> {
    let codes = parent
        .len()
        .checked_mul(parent.planes.len())
        .ok_or_else(|| PvTuningError::checkpoint("checkpoint size overflow"))?;
    let scales = parent
        .total_scale_count()
        .checked_mul(2)
        .ok_or_else(|| PvTuningError::checkpoint("checkpoint size overflow"))?;
    let code_state = parent
        .len()
        .checked_mul(8)
        .ok_or_else(|| PvTuningError::checkpoint("checkpoint size overflow"))?;
    let scale_state = parent
        .total_scale_count()
        .checked_mul(8)
        .ok_or_else(|| PvTuningError::checkpoint("checkpoint size overflow"))?;
    FIXED_BODY_BYTES
        .checked_add(codes)
        .and_then(|size| size.checked_add(scales))
        .and_then(|size| size.checked_add(code_state))
        .and_then(|size| size.checked_add(scale_state))
        .ok_or_else(|| PvTuningError::checkpoint("checkpoint size overflow"))
}

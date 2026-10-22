//! Original-parent numerical verification, not callback/quality authentication.
use super::publication::{NativeScaleChildError as Error, read_tensor};
use super::*;
use std::io::{Read, Seek};
use tritium_evidence::{NativeRefinementParentView, NativeScaleChildReceiptView, VerifiedLog};
use tritium_format::trit_model::{TritModelReader, decode_additive_payload};
use tritium_schema::{PlaneCounts, TensorStorage};
use tritium_spec::AdditiveMetadata;

/// Logical decoded state and one encoded parent plus explicit caller payload.
#[derive(Clone, Copy, Debug)]
pub struct NativeScaleParentReadLimits {
    /// State (including moments), largest encoded owner and extra payload ceiling.
    pub max_working_bytes: u64,
    /// Additional overlapping caller payload. The child verifier adds its actual
    /// expected checkpoint separately; direct decoding adds only this value.
    pub extra_payload_bytes: u64,
}
fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b)
        .ok_or(Error::Limit("native parent size overflow"))
}

/// Decode actual selected originals once under the same common state ledger.
/// Source/log admission is rechecked; no complete encoded model is copied.
pub fn decode_native_scale_parents<R: Read + Seek>(
    reader: &mut TritModelReader<R>,
    parent: &NativeRefinementParentView,
    plan: &NativeRefinementPlan,
    limits: NativeScaleParentReadLimits,
) -> Result<Vec<AdditiveTensor>, Error> {
    plan.validate_parent(reader.manifest(), parent.binding())
        .map_err(|_| Error::Binding("native parent/request mismatch"))?;
    let mut retained = 0;
    let mut largest = 0;
    for name in &plan.owners {
        let meta = reader.tensor_meta(&plan.parent.level, name)?;
        let TensorStorage::Additive { layout, planes } = &meta.storage else {
            return Err(Error::Binding("native owner is not additive"));
        };
        let m = match planes {
            PlaneCounts::Uniform(p) => AdditiveMetadata::uniform(*layout, *p),
            PlaneCounts::PerTile(c) => AdditiveMetadata::per_tile(*layout, c),
            _ => return Err(Error::Binding("native allocation unsupported")),
        }
        .map_err(|_| Error::Binding("native owner metadata invalid"))?;
        let mut bytes = m
            .auxiliary_bytes()
            .map_err(|_| Error::Limit("native index size"))? as u64;
        for plane in 0..m.plane_count() {
            bytes = add(
                bytes,
                m.plane_trit_count(plane)
                    .map_err(|_| Error::Limit("native coefficient size"))? as u64,
            )?;
        }
        bytes = add(
            bytes,
            (m.scale_count() as u64)
                .checked_mul(12)
                .ok_or(Error::Limit("native scale size"))?,
        )?;
        retained = add(retained, bytes)?;
        largest = largest.max(reader.tensor_byte_len(&plan.parent.level, name)?);
    }
    if limits.max_working_bytes == 0
        || retained > plan.limits.max_state_bytes
        || add(add(retained, largest)?, limits.extra_payload_bytes)? > limits.max_working_bytes
    {
        return Err(Error::Limit(
            "native decoded parent/state exceeds byte limit",
        ));
    }
    parent.verify(reader)?;
    let mut parents = Vec::new();
    parents
        .try_reserve_exact(plan.owners.len())
        .map_err(|_| Error::Allocation {
            requested: plan.owners.len(),
        })?;
    for name in &plan.owners {
        let bytes = read_tensor(reader, &plan.parent.level, name)?;
        parents.push(decode_additive_payload(
            reader.tensor_meta(&plan.parent.level, name)?,
            &bytes,
        )?);
    }
    reader.verify()?;
    Ok(parents)
}

/// Actual original-parent, journal, checkpoint and encoded-child numerical binding.
/// This does not authenticate objective/data/teacher execution or grant release approval.
#[derive(Clone, Debug)]
pub struct VerifiedNativeScaleChild {
    receipt: NativeScaleChildReceiptView,
    replay: VerifiedNativeScaleReplay,
    package: tritium_format::PackageId,
}
impl VerifiedNativeScaleChild {
    /// Original parent-admitted child record view.
    #[must_use]
    pub fn receipt(&self) -> &NativeScaleChildReceiptView {
        &self.receipt
    }
    /// Actual numerical replay input/output binding.
    #[must_use]
    pub fn replay(&self) -> &VerifiedNativeScaleReplay {
        &self.replay
    }
    /// Exact observed child container identity.
    #[must_use]
    pub const fn package(&self) -> tritium_format::PackageId {
        self.package
    }
}

/// Explicit original sources and frozen request for a numerical child audit.
#[derive(Debug)]
pub struct NativeScaleChildVerification<'a, P, C, J> {
    /// Original retained parent bytes.
    pub parent: &'a mut TritModelReader<P>,
    /// Original parent evidence view, not resealed metadata.
    pub parent_evidence: &'a NativeRefinementParentView,
    /// Actual child bytes.
    pub child: &'a mut TritModelReader<C>,
    /// Original verified child log.
    pub log: &'a VerifiedLog,
    /// Frozen requested plan.
    pub plan: &'a NativeRefinementPlan,
    /// Complete recorded gradient journal.
    pub journal: &'a mut J,
    /// Actual expected checkpoint bytes, externally admitted before allocation.
    pub checkpoint: &'a [u8],
}

/// Replay from actual originals and bind actual deployed owner bytes, not just record IDs.
pub fn verify_native_scale_child<P: Read + Seek, C: Read + Seek, J: Read + Seek>(
    source: NativeScaleChildVerification<'_, P, C, J>,
    parent_limits: NativeScaleParentReadLimits,
    replay_limits: NativeScaleJournalLimits,
) -> Result<VerifiedNativeScaleChild, Error> {
    let mut views = NativeScaleChildReceiptView::fold(
        source.log,
        source.parent,
        source.parent_evidence,
        source.plan,
    )?;
    if views.len() != 1 {
        return Err(Error::Binding("complete native child receipt absent"));
    }
    let receipt = views.remove(0);
    receipt.verify_child(source.child)?;
    if source.checkpoint.len() as u64 != receipt.replay().checkpoint_bytes
        || blake3::hash(source.checkpoint).as_bytes() != receipt.replay().checkpoint.as_bytes()
    {
        return Err(Error::Binding("native child checkpoint bytes differ"));
    }
    let limits = NativeScaleParentReadLimits {
        extra_payload_bytes: add(
            parent_limits.extra_payload_bytes,
            source.checkpoint.len() as u64,
        )?,
        ..parent_limits
    };
    let parents =
        decode_native_scale_parents(source.parent, source.parent_evidence, source.plan, limits)?;
    let (session, replay) = replay_native_scale_journal_with_state(
        source.plan.clone(),
        parents,
        source.journal,
        receipt.replay().journal,
        source.checkpoint,
        replay_limits,
    )?;
    for (name, weight) in session.plan.owners.iter().zip(session.weights()) {
        let length = source
            .parent
            .tensor_byte_len(&session.plan.parent.level, name)?;
        // The same parent cap also admits state+checkpoint+one encoded owner at verification.
        if add(
            add(
                session.ledger.state_payload_bytes,
                limits.extra_payload_bytes,
            )?,
            length,
        )? > limits.max_working_bytes
        {
            return Err(Error::Limit(
                "native child encoded verification exceeds byte limit",
            ));
        }
        let bytes = weight.encode_payload()?;
        let meta = source.child.tensor_meta(&session.plan.child_level, name)?;
        if bytes.len() as u64
            != source
                .child
                .tensor_byte_len(&session.plan.child_level, name)?
            || blake3::hash(&bytes).as_bytes() != meta.blob.id.as_bytes()
        {
            return Err(Error::Binding(
                "native child bytes differ from replayed owner state",
            ));
        }
    }
    source.parent_evidence.verify(source.parent)?;
    receipt.verify_child(source.child)?;
    Ok(VerifiedNativeScaleChild {
        package: source.child.package_id(),
        receipt,
        replay,
    })
}

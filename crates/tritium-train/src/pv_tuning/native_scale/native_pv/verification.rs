//! Parent-backed P/V numerical child verification, not objective or release acceptance.
use super::super::publication::NativePvChildError as Error;
use super::super::verification::{
    NativePvParentReadLimits, decode_native_pv_parents, verify_encoded_state,
};
use super::*;
use std::io::{Read, Seek};
use tritium_evidence::{NativePvChildReceiptView, NativeRefinementParentView, VerifiedLog};
use tritium_format::trit_model::TritModelReader;
fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b)
        .ok_or(Error::Limit("native PV verifier size overflow"))
}
/// Actual original-parent, journal, checkpoint and encoded-child numerical binding.
/// This does not authenticate objective/data/teacher execution or grant release approval.
#[derive(Clone, Debug)]
pub struct VerifiedNativePvChild {
    receipt: NativePvChildReceiptView,
    replay: VerifiedNativePvReplay,
    package: tritium_format::PackageId,
}
impl VerifiedNativePvChild {
    /// Original parent-admitted child record view.
    #[must_use]
    pub fn receipt(&self) -> &NativePvChildReceiptView {
        &self.receipt
    }
    /// Actual numerical replay input/output binding.
    #[must_use]
    pub fn replay(&self) -> &VerifiedNativePvReplay {
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
pub struct NativePvChildVerification<'a, P, C, J> {
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
pub fn verify_native_pv_child<P: Read + Seek, C: Read + Seek, J: Read + Seek>(
    source: NativePvChildVerification<'_, P, C, J>,
    parent_limits: NativePvParentReadLimits,
    replay_limits: NativePvJournalLimits,
) -> Result<VerifiedNativePvChild, Error> {
    let mut views = NativePvChildReceiptView::fold(
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
    let limits = NativePvParentReadLimits {
        extra_payload_bytes: add(
            parent_limits.extra_payload_bytes,
            source.checkpoint.len() as u64,
        )?,
        ..parent_limits
    };
    let parents =
        decode_native_pv_parents(source.parent, source.parent_evidence, source.plan, limits)?;
    let (session, replay) = replay_native_pv_journal_with_state(
        source.plan.clone(),
        parents,
        source.journal,
        receipt.replay().journal,
        source.checkpoint,
        replay_limits,
    )?;
    verify_encoded_state(
        super::super::publication::State::Pv(&session, &replay),
        source.parent,
        source.child,
        limits.extra_payload_bytes,
        limits.max_working_bytes,
    )?;
    source.parent_evidence.verify(source.parent)?;
    receipt.verify_child(source.child)?;
    Ok(VerifiedNativePvChild {
        package: source.child.package_id(),
        receipt,
        replay,
    })
}

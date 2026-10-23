//! Typed P/V replay around the common bounded bundle transport.
use super::*;
use crate::pv_tuning::native_scale::journal::bundle::{
    NativeScaleResumeBundleError, NativeScaleResumeBundleLimits, wire,
};
use std::{fs::File, path::Path};
use tritium_schema::NativePvResumeManifest;

/// Shared encoded/working payload limits, not physical memory or disk admission.
pub type NativePvResumeBundleLimits = NativeScaleResumeBundleLimits;
/// Shared transport errors; actual P/V replay errors use the distinct PvJournal variant.
pub type NativePvResumeBundleError = NativeScaleResumeBundleError;
type BundleResult<T> = std::result::Result<T, NativePvResumeBundleError>;
struct PvLive<'a, W> {
    session: &'a NativePvRefinementSession,
    journal: &'a mut NativePvJournalWriter<W>,
}
impl<W: Read + Write + Seek> wire::Live for PvLive<'_, W> {
    type Stream = W;
    fn check(&self) -> BundleResult<()> {
        Ok(self.journal.check(self.session)?)
    }
    fn checkpoint_bytes(&self) -> u64 {
        self.session.size_ledger().checkpoint_bytes
    }
    fn snapshot(&self) -> BundleResult<(Vec<u8>, wire::Manifest)> {
        let (checkpoint, point) = self.journal.snapshot_unflushed(self.session)?;
        Ok((
            checkpoint,
            wire::Manifest::Pv(NativePvResumeManifest {
                plan: self.session.plan().clone(),
                point,
            }),
        ))
    }
    fn flush(&mut self) -> BundleResult<()> {
        Ok(self.journal.flush()?)
    }
    fn stream(&mut self) -> (&mut W, &mut bool) {
        (&mut self.journal.output, &mut self.journal.poisoned)
    }
}
/// Byte-consistent immutable P/V recovery input; actual-parent replay required.
#[derive(Debug)]
pub struct NativePvResumeBundleReader<R>(wire::ReaderCore<R>);
impl<R: Read + Seek> NativePvResumeBundleReader<R> {
    /// Verify complete expected TNPB bytes/caps before bounded metadata/checkpoint
    /// allocation. This is not numerical, objective or quality qualification.
    pub fn open(
        source: R,
        expected: BlobId,
        limits: NativePvResumeBundleLimits,
    ) -> BundleResult<Self> {
        wire::ReaderCore::open(source, expected, limits, wire::Kind::Pv).map(Self)
    }
    /// Exact frozen P/V request and untrusted emitted observations.
    pub fn manifest(&self) -> &NativePvResumeManifest {
        let wire::Manifest::Pv(manifest) = self.0.manifest() else {
            unreachable!("typed PV bundle")
        };
        manifest
    }
    /// Exact complete original bundle identity, including checksum.
    pub const fn id(&self) -> BlobId {
        self.0.id()
    }
    /// Exact original encoded bytes, not physical disk allocation/RSS.
    pub const fn bytes(&self) -> u64 {
        self.0.bytes()
    }
    /// Recheck the retained source against its exact original identity.
    pub fn verify_unchanged(&mut self) -> BundleResult<()> {
        self.0.verify_unchanged()
    }
    /// Replay original parents and exact checkpoint, then copy only the admitted
    /// prefix into a new empty journal. No extracted prefix, overwrite or alternate
    /// optimizer. Caller owns cleanup of an unusable private destination tail.
    pub fn resume<W: Write + Seek>(
        self,
        parents: Vec<tritium_format::AdditiveTensor>,
        output: W,
        limits: NativePvJournalLimits,
    ) -> BundleResult<(NativePvRefinementSession, NativePvJournalWriter<W>)> {
        let (source, manifest, checkpoint) = self.0.into_parts()?;
        let wire::Manifest::Pv(manifest) = manifest else {
            unreachable!("typed PV bundle")
        };
        Ok(resume_native_pv_journal(
            NativePvJournalResume {
                plan: manifest.plan,
                parents,
                source,
                point: &manifest.point,
                checkpoint: &checkpoint,
                tail: NativePvResumeTail::ExactPrefix,
            },
            output,
            limits,
        )?)
    }
}
/// Stage/flush/reopen TNPB in an empty owned stream; no fsync/publication implied.
/// Restored destination-only failure permits retry; source uncertainty or unwinding
/// prefix I/O poisons the writer. No numerical state or quality claim is fabricated.
pub fn stage_native_pv_resume_bundle<W: Read + Write + Seek, O: Read + Write + Seek>(
    session: &NativePvRefinementSession,
    journal: &mut NativePvJournalWriter<W>,
    output: O,
    limits: NativePvResumeBundleLimits,
) -> BundleResult<NativePvResumeBundleReader<O>> {
    wire::stage_bundle(&mut PvLive { session, journal }, output, limits)
        .map(NativePvResumeBundleReader)
}
/// Committed P/V recovery inputs with explicit postcommit caveats.
#[derive(Debug)]
pub struct PublishedNativePvResumeBundle {
    /// Byte-consistent bundle; original-parent numerical replay still required.
    pub bundle: NativePvResumeBundleReader<File>,
    /// Postcommit cleanup/directory-sync failures, never a false failed commit.
    pub warnings: Vec<tritium_quantize::engine::PublicationWarning>,
}
/// Fsync then no-clobber hard-link `resume.tnpb` in a new directory. Removes only
/// owned files before commit; not crash/security or physical-memory qualification.
pub fn publish_native_pv_resume_bundle<W: Read + Write + Seek>(
    session: &NativePvRefinementSession,
    journal: &mut NativePvJournalWriter<W>,
    directory: &Path,
    limits: NativePvResumeBundleLimits,
) -> BundleResult<PublishedNativePvResumeBundle> {
    let result = wire::publish(&mut PvLive { session, journal }, directory, limits)?;
    Ok(PublishedNativePvResumeBundle {
        bundle: NativePvResumeBundleReader(result.bundle),
        warnings: result.warnings,
    })
}

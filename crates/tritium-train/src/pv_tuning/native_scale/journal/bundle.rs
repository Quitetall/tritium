//! Typed scale recovery; shared transport never changes the numerical profile.
use super::{
    NativeScaleJournalError, NativeScaleJournalLimits, NativeScaleJournalResume,
    NativeScaleJournalWriter, NativeScaleRefinementSession, NativeScaleResumeTail, Reader, SCRATCH,
    resume_native_scale_journal,
};
use std::{
    fs::File,
    io::{Read, Seek, Write},
    path::Path,
};
use tritium_schema::{BlobId, NativeScaleResumeManifest};
pub(in crate::pv_tuning::native_scale) mod wire;
pub use wire::{NativeScaleResumeBundleError, NativeScaleResumeBundleLimits};
type Result<T> = std::result::Result<T, NativeScaleResumeBundleError>;

struct ScaleLive<'a, W> {
    session: &'a NativeScaleRefinementSession,
    journal: &'a mut NativeScaleJournalWriter<W>,
}
impl<W: Read + Write + Seek> wire::Live for ScaleLive<'_, W> {
    type Stream = W;
    fn check(&self) -> Result<()> {
        Ok(self.journal.check(self.session)?)
    }
    fn checkpoint_bytes(&self) -> u64 {
        self.session.size_ledger().checkpoint_bytes
    }
    fn snapshot(&self) -> Result<(Vec<u8>, wire::Manifest)> {
        let (checkpoint, point) = self.journal.snapshot_unflushed(self.session)?;
        Ok((
            checkpoint,
            wire::Manifest::Scale(NativeScaleResumeManifest {
                plan: self.session.plan().clone(),
                point,
            }),
        ))
    }
    fn flush(&mut self) -> Result<()> {
        Ok(self.journal.flush()?)
    }
    fn stream(&mut self) -> (&mut W, &mut bool) {
        (&mut self.journal.output, &mut self.journal.poisoned)
    }
}

/// Retained byte-consistent scale recovery input; actual-parent replay required.
#[derive(Debug)]
pub struct NativeScaleResumeBundleReader<R>(wire::ReaderCore<R>);
impl<R: Read + Seek> NativeScaleResumeBundleReader<R> {
    /// Admit complete TNSB bytes before bounded JSON/checkpoint allocation.
    pub fn open(
        source: R,
        expected: BlobId,
        limits: NativeScaleResumeBundleLimits,
    ) -> Result<Self> {
        wire::ReaderCore::open(source, expected, limits, wire::Kind::Scale).map(Self)
    }
    /// Read-only frozen scale request and emitted observations, not qualification.
    pub fn manifest(&self) -> &NativeScaleResumeManifest {
        let wire::Manifest::Scale(manifest) = self.0.manifest() else {
            unreachable!("typed scale bundle")
        };
        manifest
    }
    /// Exact original complete file identity.
    pub const fn id(&self) -> BlobId {
        self.0.id()
    }
    /// Exact original encoded bytes, not physical disk allocation/RSS.
    pub const fn bytes(&self) -> u64 {
        self.0.bytes()
    }
    /// Recheck retained bytes against their original complete identity.
    pub fn verify_unchanged(&mut self) -> Result<()> {
        self.0.verify_unchanged()
    }
    /// Replay original parents then copy into a new empty journal; never mutate
    /// the saved source. Caller owns cleanup of a failed private output tail.
    pub fn resume<W: Write + Seek>(
        self,
        parents: Vec<tritium_format::AdditiveTensor>,
        output: W,
        limits: NativeScaleJournalLimits,
    ) -> Result<(NativeScaleRefinementSession, NativeScaleJournalWriter<W>)> {
        let (source, manifest, checkpoint) = self.0.into_parts()?;
        let wire::Manifest::Scale(manifest) = manifest else {
            unreachable!("typed scale bundle")
        };
        Ok(resume_native_scale_journal(
            NativeScaleJournalResume {
                plan: manifest.plan,
                parents,
                source,
                point: &manifest.point,
                checkpoint: &checkpoint,
                tail: NativeScaleResumeTail::ExactPrefix,
            },
            output,
            limits,
        )?)
    }
}

/// Stage/flush/reopen a complete TNSB in an empty owned stream. No fsync or
/// publication implied; source uncertainty poisons the live writer.
pub fn stage_native_scale_resume_bundle<W: Read + Write + Seek, O: Read + Write + Seek>(
    session: &NativeScaleRefinementSession,
    journal: &mut NativeScaleJournalWriter<W>,
    output: O,
    limits: NativeScaleResumeBundleLimits,
) -> Result<NativeScaleResumeBundleReader<O>> {
    wire::stage_bundle(&mut ScaleLive { session, journal }, output, limits)
        .map(NativeScaleResumeBundleReader)
}
/// Committed scale recovery inputs and explicit postcommit caveats.
#[derive(Debug)]
pub struct PublishedNativeScaleResumeBundle {
    /// Byte-consistent bundle; not numerical/quality qualification.
    pub bundle: NativeScaleResumeBundleReader<File>,
    /// Postcommit cleanup/directory-sync failures, never a false failed commit.
    pub warnings: Vec<tritium_quantize::engine::PublicationWarning>,
}
/// Fsync then no-clobber hard-link `resume.tnsb` in a new directory. Removes
/// only owned files before commit; not crash/security or physical qualification.
pub fn publish_native_scale_resume_bundle<W: Read + Write + Seek>(
    session: &NativeScaleRefinementSession,
    journal: &mut NativeScaleJournalWriter<W>,
    directory: &Path,
    limits: NativeScaleResumeBundleLimits,
) -> Result<PublishedNativeScaleResumeBundle> {
    let result = wire::publish(&mut ScaleLive { session, journal }, directory, limits)?;
    Ok(PublishedNativeScaleResumeBundle {
        bundle: NativeScaleResumeBundleReader(result.bundle),
        warnings: result.warnings,
    })
}

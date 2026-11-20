//! Read-only observations/publication without exposing independently mutable state.
use super::*;
use crate::pv_tuning::*;
use std::path::Path;
use tritium_evidence::EvidenceRecorder;
use tritium_quantize::engine::DirectoryTensorSink;
use tritium_schema::BlobId;

impl<R: Read + Seek> NativeCpuPvPagedSession<R> {
    /// Publish standard streamed recovery input; no mutable numerical state escapes.
    pub fn save_resume_bundle<J: Read + Write + Seek>(
        &mut self,
        journal: &mut NativePvPagedJournalWriter<J>,
        output: &Path,
        limits: NativePvPagedResumeBundleLimits,
    ) -> Result<PublishedNativePvPagedResumeBundle, NativeScaleResumeBundleError> {
        publish_native_pv_paged_resume_bundle(&mut self.session, journal, output, limits)
    }
    /// Numerically audit the actual current checkpoint against complete history.
    /// Returned replay state is separate; this packed cohort does not change.
    pub fn replay_recorded<J: Read + Seek, W: Read + Write + Seek, E>(
        &mut self,
        journal: J,
        expected_journal: BlobId,
        parents: impl FnMut(usize, &str) -> Result<AdditiveTensor, E>,
        checkpoints: impl FnMut(u64) -> Result<W, E>,
        limits: NativePvPagedReplayLimits,
    ) -> Result<(NativePvPagedSession<W>, VerifiedNativePvReplay), NativePvPagedReplayError<E>>
    {
        replay_native_pv_paged_journal(
            NativePvPagedJournalReplay {
                plan: self.session.plan().clone(),
                source: journal,
                expected_journal,
                checkpoint: self.session.journal_checkpoint(),
            },
            parents,
            checkpoints,
            limits,
        )
    }
    /// Common immutable child publication; never replace this live cohort.
    pub fn publish_child<P: Read + Seek, J: Read + Seek, S: Read + Write + Seek>(
        &mut self,
        source: NativePvChildSource<'_, P, J>,
        spool: &mut S,
        output: &mut DirectoryTensorSink,
        limits: NativePvPagedChildLimits,
        recorder: EvidenceRecorder,
    ) -> Result<PublishedNativePvPagedChild, NativePvChildError> {
        publish_native_pv_paged_child(&mut self.session, source, spool, output, limits, recorder)
    }
}

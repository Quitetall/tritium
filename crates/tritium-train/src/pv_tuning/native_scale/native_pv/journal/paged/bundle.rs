//! Seek-backed adapter to the shared frozen TNPB transport, not a new format.
use super::*;
use crate::pv_tuning::native_scale::journal::bundle::wire;
use std::{cell::RefCell, fs::File, path::Path};
use tritium_schema::NativePvResumeManifest;
type BundleResult<T> = std::result::Result<T, wire::NativeScaleResumeBundleError>;

/// Paged transport limits; resident checkpoint-inclusive caps are unchanged.
#[derive(Clone, Copy, Debug)]
pub struct NativePvPagedResumeBundleLimits {
    /// Complete TNPB bytes, including checksum.
    pub max_bundle_bytes: u64,
    /// Encoded JSON ceiling before allocation/decoding.
    pub max_manifest_bytes: usize,
    /// Actual serialized checkpoint length, not a heap allocation budget.
    pub max_checkpoint_bytes: u64,
    /// Encoded metadata plus8192byte I/O scratch, excluding decoded metadata,
    /// numerical/source/stream buffers, allocator slack and physical RSS/VRAM.
    pub max_working_bytes: u64,
}
impl NativePvPagedResumeBundleLimits {
    fn wire(self) -> wire::NativeScaleResumeBundleLimits {
        wire::NativeScaleResumeBundleLimits {
            max_bundle_bytes: self.max_bundle_bytes,
            max_manifest_bytes: self.max_manifest_bytes,
            max_checkpoint_bytes: self.max_checkpoint_bytes,
            max_working_bytes: self.max_working_bytes,
        }
    }
}
struct Live<'a, R, W> {
    session: &'a mut NativePvPagedSession<R>,
    journal: &'a mut NativePvPagedJournalWriter<W>,
}
impl<R: Read + Seek, W: Read + Write + Seek> wire::Live for Live<'_, R, W> {
    type Stream = W;
    fn check(&mut self) -> BundleResult<()> {
        let result = verify(self.session).and_then(|state| self.journal.check(state));
        if result.is_err() {
            self.journal.inner.poisoned = true;
        }
        Ok(result?)
    }
    fn checkpoint_bytes(&self) -> u64 {
        self.session.size_ledger().logical.checkpoint_bytes
    }
    fn streamed(&self) -> bool {
        true
    }
    fn snapshot(&mut self) -> BundleResult<(Option<Vec<u8>>, wire::Manifest)> {
        self.check()?;
        Ok((
            None,
            wire::Manifest::Pv(NativePvResumeManifest {
                plan: self.session.plan().clone(),
                point: self.journal.point(self.session.journal_state()),
            }),
        ))
    }
    fn write_checkpoint(
        &mut self,
        _checkpoint: Option<&[u8]>,
        point: &wire::Point,
        mut write: impl FnMut(&[u8]) -> BundleResult<()>,
    ) -> BundleResult<()> {
        // Remains poisoned on unwind or uncertain source I/O. A destination-only
        // error is retryable only after a fresh complete source/state check.
        self.journal.inner.poisoned = true;
        let source = self.session.journal_checkpoint();
        let mut source_ok = false;
        let result = (|| {
            if source.seek(SeekFrom::End(0))? != point.checkpoint_bytes
                || source.seek(SeekFrom::Start(0))? != 0
            {
                return Err(wire::NativeScaleResumeBundleError::Invalid(
                    "paged bundle checkpoint length/seek differs",
                ));
            }
            let mut scratch = [0; SCRATCH];
            let mut hash = blake3::Hasher::new();
            let mut left = point.checkpoint_bytes;
            while left > 0 {
                let n = left.min(SCRATCH as u64) as usize;
                source_ok = false;
                source.read_exact(&mut scratch[..n])?;
                hash.update(&scratch[..n]);
                source_ok = true;
                write(&scratch[..n])?;
                left -= n as u64;
            }
            source_ok = false;
            if hash.finalize().as_bytes() != point.checkpoint.as_bytes()
                || source.seek(SeekFrom::End(0))? != point.checkpoint_bytes
            {
                return Err(wire::NativeScaleResumeBundleError::Invalid(
                    "paged bundle checkpoint transfer differs",
                ));
            }
            source_ok = true;
            Ok(())
        })();
        if source_ok {
            self.journal.inner.poisoned = false;
            self.check()?;
        }
        result
    }
    fn flush(&mut self) -> BundleResult<()> {
        Ok(self.journal.inner.flush()?)
    }
    fn stream(&mut self) -> (&mut W, &mut bool) {
        (
            &mut self.journal.inner.output,
            &mut self.journal.inner.poisoned,
        )
    }
}

/// Original transport error or original numerical/provider/factory replay error.
#[derive(Debug)]
#[non_exhaustive]
pub enum NativePvPagedResumeBundleError<E> {
    /// Frozen common TNPB transport failure.
    Bundle(wire::NativeScaleResumeBundleError),
    /// Original paged numerical replay failure, unchanged.
    Replay(NativePvPagedReplayError<E>),
}
impl<E: core::fmt::Display> core::fmt::Display for NativePvPagedResumeBundleError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Bundle(e) => e.fmt(f),
            Self::Replay(e) => e.fmt(f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for NativePvPagedResumeBundleError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bundle(e) => Some(e),
            Self::Replay(e) => Some(e),
        }
    }
}
impl<E> From<wire::NativeScaleResumeBundleError> for NativePvPagedResumeBundleError<E> {
    fn from(e: wire::NativeScaleResumeBundleError) -> Self {
        Self::Bundle(e)
    }
}

/// Byte-bound recovery source retaining metadata and byte regions, no checkpoint Vec.
/// Opening alone does not authenticate numerical, objective or quality claims.
#[derive(Debug)]
pub struct NativePvPagedResumeBundleReader<R>(wire::ReaderCore<R>);
impl<R: Read + Seek> NativePvPagedResumeBundleReader<R> {
    /// Verify both complete byte passes and strict manifest under paged limits.
    pub fn open(
        source: R,
        expected: BlobId,
        limits: NativePvPagedResumeBundleLimits,
    ) -> BundleResult<Self> {
        wire::ReaderCore::open_streamed(source, expected, limits.wire(), wire::Kind::Pv).map(Self)
    }
    /// Immutable original request and untrusted emitted point.
    pub fn manifest(&self) -> &NativePvResumeManifest {
        let wire::Manifest::Pv(m) = self.0.manifest() else {
            unreachable!("typed PV bundle")
        };
        m
    }
    /// Exact complete retained bundle identity.
    pub const fn id(&self) -> BlobId {
        self.0.id()
    }
    /// Exact encoded bytes, not physical disk or memory usage.
    pub const fn bytes(&self) -> u64 {
        self.0.bytes()
    }
    /// Rehash the complete retained source, not just admitted regions.
    pub fn verify_unchanged(&mut self) -> BundleResult<()> {
        self.0.verify_unchanged()
    }
    /// Numerically reconstruct from actual original owners using checkpoint and
    /// prefix views of this single source, then return a new continuing journal.
    /// No extracted files or source writes. Factory/private outputs remain caller
    /// owned on refusal; final whole-bundle identity is checked before return.
    pub fn resume<W: Read + Write + Seek, J: Write + Seek, E>(
        self,
        output: J,
        parents: impl FnMut(usize, &str) -> std::result::Result<AdditiveTensor, E>,
        make_checkpoint: impl FnMut(u64) -> std::result::Result<W, E>,
        limits: NativePvPagedReplayLimits,
    ) -> std::result::Result<
        (NativePvPagedSession<W>, NativePvPagedJournalWriter<J>),
        NativePvPagedResumeBundleError<E>,
    > {
        let parts = self.0.into_streamed_parts()?;
        let wire::Manifest::Pv(manifest) = parts.manifest else {
            unreachable!("typed PV bundle")
        };
        let shared = RefCell::new(parts.source);
        let mut checkpoint = wire::Prefix::new(
            wire::Shared(&shared),
            parts.checkpoint_offset,
            manifest.point.checkpoint_bytes,
        )?;
        let prefix = wire::Prefix::new(
            wire::Shared(&shared),
            parts.prefix_offset,
            manifest.point.journal_prefix_bytes,
        )?;
        let (mut session, mut journal) = resume_native_pv_paged_journal(
            NativePvPagedJournalResume {
                plan: manifest.plan,
                source: prefix,
                point: &manifest.point,
                checkpoint: &mut checkpoint,
                tail: NativePvResumeTail::ExactPrefix,
            },
            output,
            parents,
            make_checkpoint,
            limits,
        )
        .map_err(NativePvPagedResumeBundleError::Replay)?;
        if journal.checkpoint(&mut session).map_err(|e| {
            NativePvPagedResumeBundleError::Replay(NativePvPagedReplayError::Journal(e))
        })? != manifest.point
        {
            return Err(wire::NativeScaleResumeBundleError::Invalid(
                "paged bundle recovered point differs",
            )
            .into());
        }
        wire::verify_raw(&mut shared.into_inner(), parts.total, parts.id)?;
        // No further journal output effects after the whole-bundle guard.
        // Those final source reads must not redirect reconstructed state.
        let state = verify(&mut session).map_err(|e| {
            NativePvPagedResumeBundleError::Replay(NativePvPagedReplayError::Journal(e))
        })?;
        if journal.point(state) != manifest.point {
            return Err(wire::NativeScaleResumeBundleError::Invalid(
                "paged bundle workspace changed during final source check",
            )
            .into());
        }
        Ok((session, journal))
    }
}

/// Save/reopen standard TNPB from actual paged state without checkpoint buffering.
/// Destination-only failure may permit retry after fresh source/cursor checks;
/// source uncertainty/unwind poisons the writer. Flush alone is not durability.
pub fn stage_native_pv_paged_resume_bundle<
    R: Read + Seek,
    W: Read + Write + Seek,
    O: Read + Write + Seek,
>(
    session: &mut NativePvPagedSession<R>,
    journal: &mut NativePvPagedJournalWriter<W>,
    output: O,
    limits: NativePvPagedResumeBundleLimits,
) -> BundleResult<NativePvPagedResumeBundleReader<O>> {
    wire::stage_bundle(&mut Live { session, journal }, output, limits.wire())
        .map(NativePvPagedResumeBundleReader)
}
/// Committed standard TNPB and explicit postcommit cleanup/sync caveats.
#[derive(Debug)]
pub struct PublishedNativePvPagedResumeBundle {
    /// Byte-bound recovery input, not a numerical/quality capability.
    pub bundle: NativePvPagedResumeBundleReader<File>,
    /// Postcommit warnings are not reported as failed commits.
    pub warnings: Vec<tritium_quantize::engine::PublicationWarning>,
}
/// Reuse fsync and new-directory no-clobber hard-link publication. Never replace
/// existing paths or remove foreign files; not crash/security qualification.
pub fn publish_native_pv_paged_resume_bundle<R: Read + Seek, W: Read + Write + Seek>(
    session: &mut NativePvPagedSession<R>,
    journal: &mut NativePvPagedJournalWriter<W>,
    directory: &Path,
    limits: NativePvPagedResumeBundleLimits,
) -> BundleResult<PublishedNativePvPagedResumeBundle> {
    let result = wire::publish(&mut Live { session, journal }, directory, limits.wire())?;
    Ok(PublishedNativePvPagedResumeBundle {
        bundle: NativePvPagedResumeBundleReader(result.bundle),
        warnings: result.warnings,
    })
}

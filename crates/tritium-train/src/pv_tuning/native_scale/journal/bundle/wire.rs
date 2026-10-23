//! Self-contained byte-bound recovery inputs, never a model or qualification receipt.
use super::*;
use crate::pv_tuning::NativePvJournalError;
use std::io::{self, Read, Seek, SeekFrom, Write};
use tritium_schema::{
    BlobId, NativePvResumeManifest, NativeRecoveryRecipe, NativeRefinementPlan,
    NativeRefinementPlanId, NativeScaleResumeManifest,
};
mod publication;
mod stage;
pub(in crate::pv_tuning::native_scale) use publication::publish;
pub(in crate::pv_tuning::native_scale) use stage::stage_bundle;

#[derive(Clone, Copy, Debug)]
pub(in crate::pv_tuning::native_scale) enum Kind {
    Scale,
    Pv,
}
impl Kind {
    fn magic(self) -> [u8; 8] {
        match self {
            Self::Scale => *b"TNSB\0\0\0\x01",
            Self::Pv => *b"TNPB\0\0\0\x01",
        }
    }
    fn suffix(self) -> &'static str {
        match self {
            Self::Scale => "tnsb",
            Self::Pv => "tnpb",
        }
    }
}
#[derive(Debug)]
pub(in crate::pv_tuning::native_scale) enum Manifest {
    Scale(NativeScaleResumeManifest),
    Pv(NativePvResumeManifest),
}
pub(in crate::pv_tuning::native_scale) struct Point {
    pub plan: NativeRefinementPlanId,
    pub completed_updates: u64,
    pub journal_prefix: BlobId,
    pub journal_prefix_bytes: u64,
    pub checkpoint: BlobId,
    pub checkpoint_bytes: u64,
}
impl Manifest {
    fn parse(kind: Kind, bytes: &[u8]) -> Result<Self> {
        let result = match kind {
            Kind::Scale => Self::Scale(serde_json::from_slice(bytes)?),
            Kind::Pv => Self::Pv(serde_json::from_slice(bytes)?),
        };
        if !matches!(
            (&result, &result.plan().recipe),
            (Self::Scale(_), NativeRecoveryRecipe::ScaleOnly { .. })
                | (Self::Pv(_), NativeRecoveryRecipe::AlternatingPv { .. })
        ) {
            return Err(invalid(
                "native resume manifest recipe differs from bundle family",
            ));
        }
        Ok(result)
    }
    fn kind(&self) -> Kind {
        match self {
            Self::Scale(_) => Kind::Scale,
            Self::Pv(_) => Kind::Pv,
        }
    }
    fn write<W: Write>(&self, output: W) -> std::result::Result<(), serde_json::Error> {
        match self {
            Self::Scale(m) => serde_json::to_writer(output, m),
            Self::Pv(m) => serde_json::to_writer(output, m),
        }
    }
    fn plan(&self) -> &NativeRefinementPlan {
        match self {
            Self::Scale(m) => &m.plan,
            Self::Pv(m) => &m.plan,
        }
    }
    fn point(&self) -> Point {
        match self {
            Self::Scale(m) => Point {
                plan: m.point.plan,
                completed_updates: m.point.completed_updates,
                journal_prefix: m.point.journal_prefix,
                journal_prefix_bytes: m.point.journal_prefix_bytes,
                checkpoint: m.point.checkpoint,
                checkpoint_bytes: m.point.checkpoint_bytes,
            },
            Self::Pv(m) => Point {
                plan: m.point.plan,
                completed_updates: m.point.completed_updates,
                journal_prefix: m.point.journal_prefix,
                journal_prefix_bytes: m.point.journal_prefix_bytes,
                checkpoint: m.point.checkpoint,
                checkpoint_bytes: m.point.checkpoint_bytes,
            },
        }
    }
}
// Implemented only by concrete profile adapters beside their own writer. The
// shared transport controls prefix poisoning/restoration, never numerical replay.
pub(in crate::pv_tuning::native_scale) trait Live {
    type Stream: Read + Write + Seek;
    fn check(&mut self) -> Result<()>;
    fn checkpoint_bytes(&self) -> u64;
    fn snapshot(&mut self) -> Result<(Option<Vec<u8>>, Manifest)>;
    fn streamed(&self) -> bool {
        false
    }
    fn write_checkpoint(
        &mut self,
        checkpoint: Option<&[u8]>,
        _point: &Point,
        mut write: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        for chunk in checkpoint
            .ok_or_else(|| invalid("native resume checkpoint missing"))?
            .chunks(SCRATCH)
        {
            write(chunk)?;
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<()>;
    fn stream(&mut self) -> (&mut Self::Stream, &mut bool);
}

/// Logical encoded/working limits, not disk, allocator, metadata heap or RSS limits.
#[derive(Clone, Copy, Debug)]
pub struct NativeScaleResumeBundleLimits {
    /// Complete exact profile bundle bytes, including checksum.
    pub max_bundle_bytes: u64,
    /// Serialized JSON bytes, before metadata allocation/decoding.
    pub max_manifest_bytes: usize,
    /// Complete serialized profile checkpoint bytes.
    pub max_checkpoint_bytes: u64,
    /// Serialized manifest + checkpoint + one 8192-byte I/O buffer.
    pub max_working_bytes: u64,
}

/// Original I/O/JSON/replay failure or strict byte/cap refusal.
#[derive(Debug)]
#[non_exhaustive]
pub enum NativeScaleResumeBundleError {
    /// Changed or malformed recovery inputs.
    Invalid(&'static str),
    /// Checked overflow or explicit logical byte ceiling.
    Limit(&'static str),
    /// Fallible serialized payload reservation failure.
    Allocation,
    /// Original stream/filesystem failure.
    Io(io::Error),
    /// Strict schema JSON decoding/encoding failure.
    Json(serde_json::Error),
    /// Existing common snapshot/replay/continuation failure.
    Journal(NativeScaleJournalError),
    /// Actual P/V snapshot/replay/continuation failure; never relabeled scale.
    PvJournal(NativePvJournalError),
}
impl std::fmt::Display for NativeScaleResumeBundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) | Self::Limit(s) => f.write_str(s),
            Self::Allocation => f.write_str("native resume bundle allocation failed"),
            Self::Io(e) => e.fmt(f),
            Self::Json(e) => e.fmt(f),
            Self::Journal(e) => e.fmt(f),
            Self::PvJournal(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for NativeScaleResumeBundleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::Journal(e) => Some(e),
            Self::PvJournal(e) => Some(e),
            _ => None,
        }
    }
}
impl From<io::Error> for NativeScaleResumeBundleError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<serde_json::Error> for NativeScaleResumeBundleError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
impl From<NativeScaleJournalError> for NativeScaleResumeBundleError {
    fn from(e: NativeScaleJournalError) -> Self {
        Self::Journal(e)
    }
}
impl From<NativePvJournalError> for NativeScaleResumeBundleError {
    fn from(e: NativePvJournalError) -> Self {
        Self::PvJournal(e)
    }
}
type Result<T> = std::result::Result<T, NativeScaleResumeBundleError>;
fn invalid(s: &'static str) -> NativeScaleResumeBundleError {
    NativeScaleResumeBundleError::Invalid(s)
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or(NativeScaleResumeBundleError::Limit(
        "native resume bundle size overflow",
    ))
}
fn admit(l: NativeScaleResumeBundleLimits, m: u64, c: u64, p: u64, streamed: bool) -> Result<u64> {
    let total = add(add(add(64, m)?, c)?, p)?;
    if m == 0
        || c == 0
        || p == 0
        || l.max_bundle_bytes == 0
        || l.max_manifest_bytes == 0
        || l.max_checkpoint_bytes == 0
        || l.max_working_bytes == 0
        || m > l.max_manifest_bytes as u64
        || c > l.max_checkpoint_bytes
        || total > l.max_bundle_bytes
        || add(add(m, if streamed { 0 } else { c })?, SCRATCH as u64)? > l.max_working_bytes
    {
        return Err(NativeScaleResumeBundleError::Limit(
            "native resume bundle byte limit exceeded",
        ));
    }
    Ok(total)
}
fn seek<R: Seek>(s: &mut R, to: SeekFrom, expected: u64) -> Result<()> {
    if s.seek(to)? != expected {
        return Err(invalid("native resume bundle seek differs"));
    }
    Ok(())
}
fn bytes(n: u64) -> Result<Vec<u8>> {
    let n = usize::try_from(n)
        .map_err(|_| NativeScaleResumeBundleError::Limit("native resume bundle address width"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| NativeScaleResumeBundleError::Allocation)?;
    out.resize(n, 0);
    Ok(out)
}

pub(in crate::pv_tuning::native_scale) fn verify_raw<R: Read + Seek>(
    s: &mut R,
    total: u64,
    expected: BlobId,
) -> Result<()> {
    seek(s, SeekFrom::End(0), total)?;
    seek(s, SeekFrom::Start(0), 0)?;
    let mut hash = blake3::Hasher::new();
    let mut scratch = [0; SCRATCH];
    let mut left = total
        .checked_sub(32)
        .ok_or_else(|| invalid("native resume bundle length"))?;
    while left > 0 {
        let n = left.min(SCRATCH as u64) as usize;
        s.read_exact(&mut scratch[..n])?;
        hash.update(&scratch[..n]);
        left -= n as u64;
    }
    let body = hash.finalize();
    let mut footer = [0; 32];
    s.read_exact(&mut footer)?;
    hash.update(&footer);
    if footer != *body.as_bytes() || hash.finalize().as_bytes() != expected.as_bytes() {
        return Err(invalid("native resume bundle identity/checksum differs"));
    }
    seek(s, SeekFrom::End(0), total)
}

/// Retained byte-consistent recovery inputs; actual-parent replay still required.
/// No mutable source/metadata access or model/quality admission is exposed.
#[derive(Debug)]
pub(in crate::pv_tuning::native_scale) struct ReaderCore<R> {
    source: R,
    manifest: Manifest,
    checkpoint: Option<Vec<u8>>,
    checkpoint_offset: u64,
    prefix_offset: u64,
    total: u64,
    id: BlobId,
}
impl<R: Read + Seek> ReaderCore<R> {
    /// Admit exact complete bytes before bounded JSON/checkpoint allocation.
    pub(in crate::pv_tuning::native_scale) fn open(
        source: R,
        expected: BlobId,
        limits: NativeScaleResumeBundleLimits,
        kind: Kind,
    ) -> Result<Self> {
        Self::open_mode(source, expected, limits, kind, false)
    }
    pub(in crate::pv_tuning::native_scale) fn open_streamed(
        source: R,
        expected: BlobId,
        limits: NativeScaleResumeBundleLimits,
        kind: Kind,
    ) -> Result<Self> {
        Self::open_mode(source, expected, limits, kind, true)
    }
    fn open_mode(
        mut source: R,
        expected: BlobId,
        limits: NativeScaleResumeBundleLimits,
        kind: Kind,
        streamed: bool,
    ) -> Result<Self> {
        // Limits on the header itself are checked before touching a caller stream.
        if limits.max_bundle_bytes < 64
            || limits.max_manifest_bytes == 0
            || limits.max_checkpoint_bytes == 0
            || limits.max_working_bytes < SCRATCH as u64
        {
            return Err(NativeScaleResumeBundleError::Limit(
                "native resume bundle byte limit exceeded",
            ));
        }
        seek(&mut source, SeekFrom::Start(0), 0)?;
        let mut header = [0; 32];
        source.read_exact(&mut header)?;
        if header[..8] != kind.magic() {
            return Err(invalid("native resume bundle magic differs"));
        }
        let m = u64::from_le_bytes(header[8..16].try_into().expect("fixed header"));
        let c = u64::from_le_bytes(header[16..24].try_into().expect("fixed header"));
        let p = u64::from_le_bytes(header[24..32].try_into().expect("fixed header"));
        let total = admit(limits, m, c, p, streamed)?;
        verify_raw(&mut source, total, expected)?;
        seek(&mut source, SeekFrom::Start(0), 0)?;
        let mut input = Reader {
            source,
            hash: blake3::Hasher::new(),
            bytes: 0,
        };
        if input.array::<32>()? != header {
            return Err(invalid("native resume bundle header changed"));
        }
        let mut metadata = bytes(m)?;
        for chunk in metadata.chunks_mut(SCRATCH) {
            input.read(chunk)?;
        }
        let manifest = Manifest::parse(kind, &metadata)?;
        drop(metadata);
        let point = manifest.point();
        if tritium_evidence::native_refinement_plan_id(manifest.plan())
            .map_err(|_| invalid("native resume bundle plan invalid"))?
            != point.plan
            || point.completed_updates > manifest.plan().training.len() as u64
            || point.checkpoint_bytes != c
            || c > manifest.plan().limits.max_checkpoint_bytes
            || point.journal_prefix_bytes != p
        {
            return Err(invalid("native resume bundle manifest binding differs"));
        }
        let checkpoint_offset = input.bytes;
        let mut scratch = [0; SCRATCH];
        let mut checkpoint = if streamed { None } else { Some(bytes(c)?) };
        let mut checkpoint_hash = blake3::Hasher::new();
        if let Some(data) = checkpoint.as_mut() {
            for chunk in data.chunks_mut(SCRATCH) {
                input.read(chunk)?;
                checkpoint_hash.update(chunk);
            }
        } else {
            let mut left = c;
            while left > 0 {
                let n = left.min(SCRATCH as u64) as usize;
                input.read(&mut scratch[..n])?;
                checkpoint_hash.update(&scratch[..n]);
                left -= n as u64;
            }
        }
        if checkpoint_hash.finalize().as_bytes() != point.checkpoint.as_bytes() {
            return Err(invalid("native resume bundle checkpoint differs"));
        }
        let prefix_offset = input.bytes;
        let mut hash = blake3::Hasher::new();
        let mut left = p;
        while left > 0 {
            let n = left.min(SCRATCH as u64) as usize;
            input.read(&mut scratch[..n])?;
            hash.update(&scratch[..n]);
            left -= n as u64;
        }
        let body = input.hash.finalize();
        if hash.finalize().as_bytes() != point.journal_prefix.as_bytes()
            || input.array::<32>()? != *body.as_bytes()
            || input.bytes != total
            || input.hash.finalize().as_bytes() != expected.as_bytes()
        {
            return Err(invalid("native resume bundle second-pass binding differs"));
        }
        seek(&mut input.source, SeekFrom::End(0), total)?;
        Ok(Self {
            source: input.source,
            manifest,
            checkpoint,
            checkpoint_offset,
            prefix_offset,
            total,
            id: expected,
        })
    }
    /// Frozen request/observations, not numerical or objective authentication.
    pub(in crate::pv_tuning::native_scale) fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    /// Exact original complete file identity.
    pub(in crate::pv_tuning::native_scale) const fn id(&self) -> BlobId {
        self.id
    }
    /// Exact original file bytes, not RSS or physical disk allocation.
    pub(in crate::pv_tuning::native_scale) const fn bytes(&self) -> u64 {
        self.total
    }
    /// Recheck the retained source against its original complete identity.
    pub(in crate::pv_tuning::native_scale) fn verify_unchanged(&mut self) -> Result<()> {
        verify_raw(&mut self.source, self.total, self.id)
    }
    /// Replay actual original parents, then copy the prefix to a new empty output.
    /// No extraction file or alternate optimizer is used. Caller owns failed output cleanup.
    pub(in crate::pv_tuning::native_scale) fn into_parts(
        mut self,
    ) -> Result<(Prefix<R>, Manifest, Vec<u8>)> {
        self.verify_unchanged()?;
        let Self {
            source,
            manifest,
            checkpoint,
            prefix_offset,
            ..
        } = self;
        let source = Prefix::new(source, prefix_offset, manifest.point().journal_prefix_bytes)?;
        Ok((
            source,
            manifest,
            checkpoint.ok_or_else(|| invalid("resident checkpoint missing"))?,
        ))
    }
    pub(in crate::pv_tuning::native_scale) fn into_streamed_parts(
        mut self,
    ) -> Result<StreamedParts<R>> {
        if self.checkpoint.is_some() {
            return Err(invalid("streamed checkpoint expected"));
        }
        self.verify_unchanged()?;
        Ok(StreamedParts {
            source: self.source,
            manifest: self.manifest,
            checkpoint_offset: self.checkpoint_offset,
            prefix_offset: self.prefix_offset,
            total: self.total,
            id: self.id,
        })
    }
}

pub(in crate::pv_tuning::native_scale) struct StreamedParts<R> {
    pub source: R,
    pub manifest: Manifest,
    pub checkpoint_offset: u64,
    pub prefix_offset: u64,
    pub total: u64,
    pub id: BlobId,
}

pub(in crate::pv_tuning::native_scale) use crate::pv_tuning::journal_io::Shared;

pub(in crate::pv_tuning::native_scale) struct Prefix<R> {
    source: R,
    start: u64,
    len: u64,
    position: u64,
}
impl<R: Seek> Prefix<R> {
    pub(in crate::pv_tuning::native_scale) fn new(
        mut source: R,
        start: u64,
        len: u64,
    ) -> Result<Self> {
        seek(&mut source, SeekFrom::Start(start), start)?;
        Ok(Self {
            source,
            start,
            len,
            position: 0,
        })
    }
}
impl<R: Read + Seek> Read for Prefix<R> {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        let n = (self.len - self.position).min(b.len() as u64) as usize;
        if n == 0 {
            return Ok(0);
        }
        // Checkpoint/prefix regions may share one source: every read restores
        // its own absolute cursor rather than trusting the other region's seek.
        let absolute = self
            .start
            .checked_add(self.position)
            .ok_or_else(|| io::Error::other("native resume prefix seek overflow"))?;
        if self.source.seek(SeekFrom::Start(absolute))? != absolute {
            return Err(io::Error::other("native resume prefix seek differs"));
        }
        let n = self.source.read(&mut b[..n])?;
        self.position += n as u64;
        Ok(n)
    }
}
impl<R: Seek> Seek for Prefix<R> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let next = match from {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::End(n) => self.len as i128 + n as i128,
            SeekFrom::Current(n) => self.position as i128 + n as i128,
        };
        if next < 0 || next > self.len as i128 {
            return Err(io::Error::other("native resume prefix seek outside region"));
        }
        let next = next as u64;
        let absolute = self
            .start
            .checked_add(next)
            .ok_or_else(|| io::Error::other("native resume prefix seek overflow"))?;
        if self.source.seek(SeekFrom::Start(absolute))? != absolute {
            return Err(io::Error::other("native resume prefix seek differs"));
        }
        self.position = next;
        Ok(next)
    }
}

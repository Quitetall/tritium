//! Self-contained byte-bound recovery inputs, never a model or qualification receipt.
use super::{
    NativeScaleJournalError, NativeScaleJournalLimits, NativeScaleJournalResume,
    NativeScaleJournalWriter, NativeScaleRefinementSession, NativeScaleResumeTail, Reader, SCRATCH,
    resume_native_scale_journal,
};
use std::io::{self, Read, Seek, SeekFrom, Write};
use tritium_schema::{BlobId, NativeScaleResumeManifest};
mod publication;
mod stage;
pub use publication::{PublishedNativeScaleResumeBundle, publish_native_scale_resume_bundle};
pub use stage::stage_native_scale_resume_bundle;
const MAGIC: [u8; 8] = *b"TNSB\0\0\0\x01";

/// Logical encoded/working limits, not disk, allocator, metadata heap or RSS limits.
#[derive(Clone, Copy, Debug)]
pub struct NativeScaleResumeBundleLimits {
    /// Complete exact TNSB bytes, including checksum.
    pub max_bundle_bytes: u64,
    /// Serialized JSON bytes, before metadata allocation/decoding.
    pub max_manifest_bytes: usize,
    /// Complete serialized TNSR bytes.
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
}
impl std::fmt::Display for NativeScaleResumeBundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) | Self::Limit(s) => f.write_str(s),
            Self::Allocation => f.write_str("native resume bundle allocation failed"),
            Self::Io(e) => e.fmt(f),
            Self::Json(e) => e.fmt(f),
            Self::Journal(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for NativeScaleResumeBundleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::Journal(e) => Some(e),
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
type Result<T> = std::result::Result<T, NativeScaleResumeBundleError>;
fn invalid(s: &'static str) -> NativeScaleResumeBundleError {
    NativeScaleResumeBundleError::Invalid(s)
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or(NativeScaleResumeBundleError::Limit(
        "native resume bundle size overflow",
    ))
}
fn admit(l: NativeScaleResumeBundleLimits, m: u64, c: u64, p: u64) -> Result<u64> {
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
        || add(add(m, c)?, SCRATCH as u64)? > l.max_working_bytes
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

fn verify_raw<R: Read + Seek>(s: &mut R, total: u64, expected: BlobId) -> Result<()> {
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
pub struct NativeScaleResumeBundleReader<R> {
    source: R,
    manifest: NativeScaleResumeManifest,
    checkpoint: Vec<u8>,
    prefix_offset: u64,
    total: u64,
    id: BlobId,
}
impl<R: Read + Seek> NativeScaleResumeBundleReader<R> {
    /// Admit exact complete bytes before bounded JSON/checkpoint allocation.
    pub fn open(
        mut source: R,
        expected: BlobId,
        limits: NativeScaleResumeBundleLimits,
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
        if header[..8] != MAGIC {
            return Err(invalid("native resume bundle magic differs"));
        }
        let m = u64::from_le_bytes(header[8..16].try_into().expect("fixed header"));
        let c = u64::from_le_bytes(header[16..24].try_into().expect("fixed header"));
        let p = u64::from_le_bytes(header[24..32].try_into().expect("fixed header"));
        let total = admit(limits, m, c, p)?;
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
        let manifest: NativeScaleResumeManifest = serde_json::from_slice(&metadata)?;
        drop(metadata);
        let point = &manifest.point;
        if tritium_evidence::native_refinement_plan_id(&manifest.plan)
            .map_err(|_| invalid("native resume bundle plan invalid"))?
            != point.plan
            || point.completed_updates > manifest.plan.training.len() as u64
            || point.checkpoint_bytes != c
            || c > manifest.plan.limits.max_checkpoint_bytes
            || point.journal_prefix_bytes != p
        {
            return Err(invalid("native resume bundle manifest binding differs"));
        }
        let mut checkpoint = bytes(c)?;
        for chunk in checkpoint.chunks_mut(SCRATCH) {
            input.read(chunk)?;
        }
        if blake3::hash(&checkpoint).as_bytes() != point.checkpoint.as_bytes() {
            return Err(invalid("native resume bundle checkpoint differs"));
        }
        let prefix_offset = input.bytes;
        let mut hash = blake3::Hasher::new();
        let mut scratch = [0; SCRATCH];
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
            prefix_offset,
            total,
            id: expected,
        })
    }
    /// Frozen request/observations, not numerical or objective authentication.
    pub fn manifest(&self) -> &NativeScaleResumeManifest {
        &self.manifest
    }
    /// Exact original complete file identity.
    pub const fn id(&self) -> BlobId {
        self.id
    }
    /// Exact original file bytes, not RSS or physical disk allocation.
    pub const fn bytes(&self) -> u64 {
        self.total
    }
    /// Recheck the retained source against its original complete identity.
    pub fn verify_unchanged(&mut self) -> Result<()> {
        verify_raw(&mut self.source, self.total, self.id)
    }
    /// Replay actual original parents, then copy the prefix to a new empty output.
    /// No extraction file or alternate optimizer is used. Caller owns failed output cleanup.
    pub fn resume<W: Write + Seek>(
        mut self,
        parents: Vec<tritium_format::AdditiveTensor>,
        output: W,
        limits: NativeScaleJournalLimits,
    ) -> Result<(NativeScaleRefinementSession, NativeScaleJournalWriter<W>)> {
        self.verify_unchanged()?;
        let Self {
            source,
            manifest,
            checkpoint,
            prefix_offset,
            ..
        } = self;
        let source = Prefix::new(source, prefix_offset, manifest.point.journal_prefix_bytes)?;
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

struct Prefix<R> {
    source: R,
    start: u64,
    len: u64,
    position: u64,
}
impl<R: Seek> Prefix<R> {
    fn new(mut source: R, start: u64, len: u64) -> Result<Self> {
        seek(&mut source, SeekFrom::Start(start), start)?;
        Ok(Self {
            source,
            start,
            len,
            position: 0,
        })
    }
}
impl<R: Read> Read for Prefix<R> {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        let n = (self.len - self.position).min(b.len() as u64) as usize;
        if n == 0 {
            return Ok(0);
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

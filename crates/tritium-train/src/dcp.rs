//! Distributed checkpoint (DCP) with resharding + crash-atomic writes (plan 0016).
//!
//! A distributed checkpoint is a **directory** holding one shard file per rank plus a manifest:
//! ```text
//! <dir>/manifest.tdcp                       ← the commit point (written/renamed LAST)
//! <dir>/step_<step>_shard_<rank:04>.tdcp    ← one per rank at save time
//! ```
//! The on-disk *global* state (param + optimizer planes) is **world-agnostic**: a shard is just a
//! contiguous slice of the flattened-and-padded buffers ([`FlatShardPlan`]). So [`load`] always
//! reassembles the same global buffers regardless of the save-time world `K`, and **resharding** to a
//! new world `J` is just `FlatShardPlan::new(leaf_lens, J)` applied by the caller — the substrate for
//! the ADR-0008 "save-on-K / load-on-J ⇒ identical forward" gate.
//!
//! **Crash atomicity (no torn checkpoint).** Every file is written to a `.tmp` sibling, `fsync`ed, and
//! `rename`d into place (atomic on POSIX); the parent directory is `fsync`ed so the rename is durable.
//! Shards are committed first, then the manifest — so the manifest is the single commit point. A crash
//! before the manifest is committed leaves the *previous* manifest (and its complete shards) intact:
//! [`load`] reads only what the live manifest names, ignoring half-written `.tmp` files and orphaned
//! shards from an interrupted save. A committed manifest therefore only ever names fully-written
//! shards; an externally-corrupted (truncated) shard is detected by the length check and returns a
//! [`DcpError`] rather than loading garbage. Old-step shards are *not* garbage-collected here (a disk
//! cost, not a correctness one — keep-last-N GC is a deferred cleanup).
//!
//! The optimizer state is carried as **parallel f32 planes** aligned with the flat parameter (AdamW →
//! `[m, v]`), so the DCP is optimizer-agnostic: it shards and reassembles the param and every plane by
//! the same element range. Byte framing reuses the never-panic [`Cursor`] from [`crate::checkpoint`].

use crate::checkpoint::{CheckpointError, Cursor};
use crate::fsdp::{FlatShardError, FlatShardPlan};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-rank shard file magic: `b"TDCP"` (Tritium Distributed-Checkpoint, Per-rank).
pub const DCP_SHARD_MAGIC: [u8; 4] = *b"TDCP";
/// Manifest file magic: `b"TDCM"` (Tritium Distributed-Checkpoint, Manifest).
pub const DCP_MANIFEST_MAGIC: [u8; 4] = *b"TDCM";
/// Current DCP format version.
pub const DCP_VERSION: u8 = 1;

/// Upper bound on optimizer-state planes a manifest may declare. AdamW uses 2; any real first-order
/// optimizer uses a small handful. A larger value in a (corrupt) manifest is rejected rather than used
/// to size an allocation — the never-panic guard against an unbounded `Vec` from untrusted bytes.
pub const MAX_STATE_PLANES: usize = 16;

/// Maximum number of `f32` values exchanged with a streaming state callback at once.
///
/// The DCP implementation owns one buffer of this size while saving or loading. The filesystem
/// writer may additionally own its small standard-library buffer, but no allocation scales with the
/// model or shard size. Public so callers can size pinned/mapped staging buffers to the same bound.
pub const STREAM_BUFFER_ELEMENTS: usize = 16 * 1024;

const SHARD_HEADER_BYTES: usize = 4 + 1 + 5 * 8;

/// Process-unique counter for atomic-write temp filenames (so concurrent saves to one dir cannot
/// collide on the same `.tmp` source).
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Why a distributed-checkpoint operation failed. Parsing/loading never panics: a corrupt, truncated,
/// stale, or missing piece always yields one of these.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DcpError {
    /// A filesystem operation failed; carries the OS error string.
    Io(String),
    /// A file's first bytes were not the expected magic.
    BadMagic,
    /// A format version this build cannot read.
    UnsupportedVersion(u8),
    /// A buffer ended before `needed` bytes were available (`had` = total length).
    Truncated {
        /// Byte offset the read required.
        needed: usize,
        /// Bytes actually present.
        had: usize,
    },
    /// Bytes remained after a complete parse (count of leftover bytes).
    TrailingBytes(usize),
    /// A shard header field disagreed with the manifest (a stale / mismatched / wrong-rank shard).
    ShardMismatch {
        /// Which field disagreed (`"step"` / `"world"` / `"rank"` / `"n_planes"` / `"shard_len"`).
        field: &'static str,
        /// The value the manifest implies.
        expected: u64,
        /// The value the shard carried.
        got: u64,
    },
    /// The manifest named `world` shards but shard `rank` was absent / unreadable.
    MissingShard(usize),
    /// The manifest's own fields were internally inconsistent (e.g. `chunk` disagrees with the plan
    /// recomputed from `leaf_lens` + `world`) — a corrupt manifest.
    LayoutMismatch {
        /// Which derived quantity disagreed (`"total"` / `"chunk"`).
        field: &'static str,
        /// The value recomputed from `leaf_lens` + `world`.
        expected: usize,
        /// The value the manifest carried.
        got: usize,
    },
    /// The manifest's layout was structurally invalid (`world == 0`, or a length that overflows
    /// `usize`) — caught on the untrusted load path before it could panic [`FlatShardPlan`].
    InvalidManifest(&'static str),
    /// A length field exceeded `usize` (e.g. a 64-bit checkpoint read on a 32-bit target, or a
    /// corrupt field). Carries the offending field name + value.
    ValueTooLarge {
        /// The field that was too large.
        field: &'static str,
        /// The value read.
        value: u64,
    },
    /// The manifest declared more optimizer-state planes than [`MAX_STATE_PLANES`] — rejected rather
    /// than used to size an allocation.
    TooManyPlanes {
        /// The plane count the manifest declared.
        got: usize,
        /// The maximum allowed ([`MAX_STATE_PLANES`]).
        max: usize,
    },
    /// `save` would overwrite a committed checkpoint with a non-greater step (a non-monotonic save,
    /// which could tear the live checkpoint). The previously-committed step and the attempted step.
    NonMonotonicSave {
        /// The step the committed manifest already holds.
        committed: u64,
        /// The step the rejected save attempted.
        attempted: u64,
    },
    /// A streaming source or sink violated the declared global-state layout.
    InvalidState(&'static str),
}

impl core::fmt::Display for DcpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DcpError::Io(m) => write!(f, "distributed-checkpoint I/O error: {m}"),
            DcpError::BadMagic => write!(f, "bad DCP magic"),
            DcpError::UnsupportedVersion(v) => write!(f, "unsupported DCP version {v}"),
            DcpError::Truncated { needed, had } => {
                write!(f, "DCP file truncated: needed {needed} bytes, had {had}")
            }
            DcpError::TrailingBytes(n) => write!(f, "{n} trailing bytes after DCP record"),
            DcpError::ShardMismatch {
                field,
                expected,
                got,
            } => write!(
                f,
                "DCP shard {field} mismatch: manifest {expected}, shard {got}"
            ),
            DcpError::MissingShard(r) => write!(f, "DCP shard {r} missing or unreadable"),
            DcpError::LayoutMismatch {
                field,
                expected,
                got,
            } => write!(
                f,
                "DCP manifest {field} inconsistent: recomputed {expected}, manifest {got}"
            ),
            DcpError::InvalidManifest(why) => write!(f, "invalid DCP manifest: {why}"),
            DcpError::ValueTooLarge { field, value } => {
                write!(f, "DCP {field} value {value} exceeds usize")
            }
            DcpError::TooManyPlanes { got, max } => {
                write!(f, "DCP manifest declares {got} state planes (max {max})")
            }
            DcpError::NonMonotonicSave {
                committed,
                attempted,
            } => write!(
                f,
                "non-monotonic DCP save: committed step {committed}, attempted {attempted}"
            ),
            DcpError::InvalidState(why) => write!(f, "invalid streaming DCP state: {why}"),
        }
    }
}

impl std::error::Error for DcpError {}

impl From<CheckpointError> for DcpError {
    fn from(e: CheckpointError) -> Self {
        match e {
            CheckpointError::BadMagic => DcpError::BadMagic,
            CheckpointError::UnsupportedVersion(v) => DcpError::UnsupportedVersion(v),
            CheckpointError::Truncated { needed, had } => DcpError::Truncated { needed, had },
            CheckpointError::TrailingBytes(n) => DcpError::TrailingBytes(n),
        }
    }
}

/// The world-agnostic global contents of a distributed checkpoint: the completed-step count, the leaf
/// layout (so the flat buffers can be split back into per-leaf params), the full flat parameter, and
/// the optimizer-state planes (each the same length as `param`).
#[derive(Clone, Debug, PartialEq)]
pub struct DistCheckpoint {
    /// Completed optimizer steps (the next step uses `step + 1`).
    pub step: u64,
    /// Per-leaf element counts, in leaf order — the global flat layout.
    pub leaf_lens: Vec<usize>,
    /// The full flat parameter (length `Σ leaf_lens`).
    pub param: Vec<f32>,
    /// Optimizer-state planes aligned with `param` (AdamW → `[m, v]`), each length `param.len()`.
    pub planes: Vec<Vec<f32>>,
}

impl DistCheckpoint {
    /// Total element count (`Σ leaf_lens`).
    #[must_use]
    pub fn total(&self) -> usize {
        self.leaf_lens.iter().sum()
    }
}

/// Identifies one flat state plane exchanged through [`StateSource`] and [`StateSink`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatePlane {
    /// The model's flat master-parameter plane.
    Parameter,
    /// Optimizer state plane by zero-based index (AdamW uses `0 = m`, `1 = v`).
    Optimizer(usize),
}

/// Pull-based source for bounded-memory checkpoint saves.
///
/// `save_from` requests non-overlapping ranges no larger than [`STREAM_BUFFER_ELEMENTS`]. A source
/// may copy from host memory, map a file, or synchronously download one device chunk. Implementations
/// must fill all of `out` or return an error.
pub trait StateSource {
    /// Completed optimizer step stored in the checkpoint.
    fn step(&self) -> u64;
    /// Per-leaf lengths describing the global flat layout.
    fn leaf_lens(&self) -> &[usize];
    /// Number of optimizer-state planes.
    fn plane_count(&self) -> usize;
    /// Fill one global flat range beginning at `offset`.
    ///
    /// # Errors
    /// Returns [`DcpError`] when the source cannot supply this range.
    fn read_chunk(
        &mut self,
        plane: StatePlane,
        offset: usize,
        out: &mut [f32],
    ) -> Result<(), DcpError>;
}

/// Push-based destination for bounded-memory checkpoint loads.
///
/// `load_into` validates every shard header and file length before calling [`StateSink::begin`], then
/// delivers non-overlapping ranges no larger than [`STREAM_BUFFER_ELEMENTS`]. If an I/O error occurs
/// during the subsequent payload pass, the sink may contain a partial state and must discard it.
pub trait StateSink {
    /// Declare the global state before any values are delivered.
    ///
    /// # Errors
    /// Returns [`DcpError`] when the sink cannot prepare the declared layout.
    fn begin(&mut self, step: u64, leaf_lens: &[usize], plane_count: usize)
    -> Result<(), DcpError>;

    /// Store one global flat range beginning at `offset`.
    ///
    /// # Errors
    /// Returns [`DcpError`] when the sink cannot accept this range.
    fn write_chunk(
        &mut self,
        plane: StatePlane,
        offset: usize,
        values: &[f32],
    ) -> Result<(), DcpError>;

    /// Commit the fully-delivered state.
    ///
    /// # Errors
    /// Returns [`DcpError`] when the sink cannot commit the state.
    fn finish(&mut self) -> Result<(), DcpError> {
        Ok(())
    }
}

fn state_slice(
    checkpoint: &DistCheckpoint,
    plane: StatePlane,
    offset: usize,
    len: usize,
) -> Result<&[f32], DcpError> {
    let source = match plane {
        StatePlane::Parameter => &checkpoint.param,
        StatePlane::Optimizer(index) => checkpoint
            .planes
            .get(index)
            .ok_or(DcpError::InvalidState("optimizer plane index out of range"))?,
    };
    let end = offset
        .checked_add(len)
        .ok_or(DcpError::InvalidState("source range overflows usize"))?;
    source
        .get(offset..end)
        .ok_or(DcpError::InvalidState("source range out of bounds"))
}

struct BorrowedCheckpointSource<'a>(&'a DistCheckpoint);

impl StateSource for BorrowedCheckpointSource<'_> {
    fn step(&self) -> u64 {
        self.0.step
    }

    fn leaf_lens(&self) -> &[usize] {
        &self.0.leaf_lens
    }

    fn plane_count(&self) -> usize {
        self.0.planes.len()
    }

    fn read_chunk(
        &mut self,
        plane: StatePlane,
        offset: usize,
        out: &mut [f32],
    ) -> Result<(), DcpError> {
        out.copy_from_slice(state_slice(self.0, plane, offset, out.len())?);
        Ok(())
    }
}

impl StateSink for DistCheckpoint {
    fn begin(
        &mut self,
        step: u64,
        leaf_lens: &[usize],
        plane_count: usize,
    ) -> Result<(), DcpError> {
        let total = leaf_lens.iter().try_fold(0usize, |sum, &len| {
            sum.checked_add(len)
                .ok_or(DcpError::InvalidState("sink layout overflows usize"))
        })?;
        self.step = step;
        self.leaf_lens = leaf_lens.to_vec();
        self.param = vec![0.0; total];
        self.planes = vec![vec![0.0; total]; plane_count];
        Ok(())
    }

    fn write_chunk(
        &mut self,
        plane: StatePlane,
        offset: usize,
        values: &[f32],
    ) -> Result<(), DcpError> {
        let destination = match plane {
            StatePlane::Parameter => &mut self.param,
            StatePlane::Optimizer(index) => self
                .planes
                .get_mut(index)
                .ok_or(DcpError::InvalidState("optimizer plane index out of range"))?,
        };
        let end = offset
            .checked_add(values.len())
            .ok_or(DcpError::InvalidState("sink range overflows usize"))?;
        destination
            .get_mut(offset..end)
            .ok_or(DcpError::InvalidState("sink range out of bounds"))?
            .copy_from_slice(values);
        Ok(())
    }
}

/// Manifest fields (the global metadata needed to locate + validate the shards).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Manifest {
    step: u64,
    world: usize,
    n_planes: usize,
    total: usize,
    chunk: usize,
    leaf_lens: Vec<usize>,
}

fn serialize_manifest(m: &Manifest) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&DCP_MANIFEST_MAGIC);
    out.push(DCP_VERSION);
    out.extend_from_slice(&m.step.to_le_bytes());
    out.extend_from_slice(&(m.world as u64).to_le_bytes());
    out.extend_from_slice(&(m.n_planes as u64).to_le_bytes());
    out.extend_from_slice(&(m.total as u64).to_le_bytes());
    out.extend_from_slice(&(m.chunk as u64).to_le_bytes());
    out.extend_from_slice(&(m.leaf_lens.len() as u64).to_le_bytes());
    for &len in &m.leaf_lens {
        out.extend_from_slice(&(len as u64).to_le_bytes());
    }
    out
}

fn parse_manifest(bytes: &[u8]) -> Result<Manifest, DcpError> {
    let mut c = Cursor::new(bytes);
    if c.take_magic()? != DCP_MANIFEST_MAGIC {
        return Err(DcpError::BadMagic);
    }
    let version = c.u8()?;
    if version != DCP_VERSION {
        return Err(DcpError::UnsupportedVersion(version));
    }
    let step = c.u64()?;
    let world = u64_to_usize("world", c.u64()?)?;
    let n_planes = u64_to_usize("n_planes", c.u64()?)?;
    // Bound n_planes BEFORE it is ever used to size an allocation (here, in load(), or in
    // parse_shard) — an unbounded value from a corrupt manifest must not drive a giant `Vec`.
    if n_planes > MAX_STATE_PLANES {
        return Err(DcpError::TooManyPlanes {
            got: n_planes,
            max: MAX_STATE_PLANES,
        });
    }
    let total = u64_to_usize("total", c.u64()?)?;
    let chunk = u64_to_usize("chunk", c.u64()?)?;
    let leaf_count = u64_to_usize("leaf_count", c.u64()?)?;
    // Each leaf_len is a fixed 8-byte u64, so the remaining bytes cap the count exactly — clamp the
    // pre-allocation to what the buffer can actually hold (a crafted huge leaf_count can't pre-alloc).
    let mut leaf_lens = Vec::with_capacity(leaf_count.min(c.remaining() / 8));
    for _ in 0..leaf_count {
        leaf_lens.push(u64_to_usize("leaf_len", c.u64()?)?);
    }
    if c.remaining() != 0 {
        return Err(DcpError::TrailingBytes(c.remaining()));
    }
    Ok(Manifest {
        step,
        world,
        n_planes,
        total,
        chunk,
        leaf_lens,
    })
}

#[cfg(test)]
fn serialize_shard(
    step: u64,
    world: usize,
    rank: usize,
    param: &[f32],
    planes: &[&[f32]],
) -> Vec<u8> {
    let shard_len = param.len();
    debug_assert!(
        planes.iter().all(|p| p.len() == shard_len),
        "every plane shard must match the param shard length"
    );
    let mut out = Vec::new();
    out.extend_from_slice(&DCP_SHARD_MAGIC);
    out.push(DCP_VERSION);
    out.extend_from_slice(&step.to_le_bytes());
    out.extend_from_slice(&(world as u64).to_le_bytes());
    out.extend_from_slice(&(rank as u64).to_le_bytes());
    out.extend_from_slice(&(planes.len() as u64).to_le_bytes());
    out.extend_from_slice(&(shard_len as u64).to_le_bytes());
    for &x in param {
        out.extend_from_slice(&x.to_le_bytes());
    }
    for plane in planes {
        for &x in *plane {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

/// Parse + validate a shard against the manifest. Returns `(param_shard, plane_shards)`.
#[cfg(test)]
fn parse_shard(
    bytes: &[u8],
    m: &Manifest,
    rank: usize,
) -> Result<(Vec<f32>, Vec<Vec<f32>>), DcpError> {
    let mut c = Cursor::new(bytes);
    if c.take_magic()? != DCP_SHARD_MAGIC {
        return Err(DcpError::BadMagic);
    }
    let version = c.u8()?;
    if version != DCP_VERSION {
        return Err(DcpError::UnsupportedVersion(version));
    }
    let step = c.u64()?;
    check_field("step", m.step, step)?;
    let world = c.u64()?;
    check_field("world", m.world as u64, world)?;
    let got_rank = c.u64()?;
    check_field("rank", rank as u64, got_rank)?;
    let n_planes = c.u64()?;
    check_field("n_planes", m.n_planes as u64, n_planes)?;
    let shard_len = c.u64()?;
    check_field("shard_len", m.chunk as u64, shard_len)?;
    let shard_len = u64_to_usize("shard_len", shard_len)?;
    let param = c.f32_vec(shard_len)?;
    let mut planes = Vec::with_capacity(m.n_planes);
    for _ in 0..m.n_planes {
        planes.push(c.f32_vec(shard_len)?);
    }
    if c.remaining() != 0 {
        return Err(DcpError::TrailingBytes(c.remaining()));
    }
    Ok((param, planes))
}

fn check_field(field: &'static str, expected: u64, got: u64) -> Result<(), DcpError> {
    if expected == got {
        Ok(())
    } else {
        Err(DcpError::ShardMismatch {
            field,
            expected,
            got,
        })
    }
}

fn u64_to_usize(field: &'static str, v: u64) -> Result<usize, DcpError> {
    usize::try_from(v).map_err(|_| DcpError::ValueTooLarge { field, value: v })
}

fn shard_path(dir: &Path, step: u64, rank: usize) -> PathBuf {
    dir.join(format!("step_{step}_shard_{rank:04}.tdcp"))
}

fn manifest_path(dir: &Path) -> PathBuf {
    dir.join("manifest.tdcp")
}

fn write_shard_header<W: Write>(
    writer: &mut W,
    step: u64,
    world: usize,
    rank: usize,
    n_planes: usize,
    shard_len: usize,
) -> Result<(), DcpError> {
    writer.write_all(&DCP_SHARD_MAGIC).map_err(io)?;
    writer.write_all(&[DCP_VERSION]).map_err(io)?;
    for value in [
        step,
        world as u64,
        rank as u64,
        n_planes as u64,
        shard_len as u64,
    ] {
        writer.write_all(&value.to_le_bytes()).map_err(io)?;
    }
    Ok(())
}

fn parse_shard_header(bytes: &[u8], manifest: &Manifest, rank: usize) -> Result<(), DcpError> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take_magic()? != DCP_SHARD_MAGIC {
        return Err(DcpError::BadMagic);
    }
    let version = cursor.u8()?;
    if version != DCP_VERSION {
        return Err(DcpError::UnsupportedVersion(version));
    }
    check_field("step", manifest.step, cursor.u64()?)?;
    check_field("world", manifest.world as u64, cursor.u64()?)?;
    check_field("rank", rank as u64, cursor.u64()?)?;
    check_field("n_planes", manifest.n_planes as u64, cursor.u64()?)?;
    check_field("shard_len", manifest.chunk as u64, cursor.u64()?)?;
    if cursor.remaining() != 0 {
        return Err(DcpError::TrailingBytes(cursor.remaining()));
    }
    Ok(())
}

fn expected_shard_bytes(manifest: &Manifest) -> Result<usize, DcpError> {
    let planes = manifest
        .n_planes
        .checked_add(1)
        .ok_or(DcpError::InvalidManifest(
            "shard plane count overflows usize",
        ))?;
    let payload = manifest
        .chunk
        .checked_mul(planes)
        .and_then(|n| n.checked_mul(core::mem::size_of::<f32>()))
        .ok_or(DcpError::InvalidManifest(
            "shard byte length overflows usize",
        ))?;
    SHARD_HEADER_BYTES
        .checked_add(payload)
        .ok_or(DcpError::InvalidManifest(
            "shard byte length overflows usize",
        ))
}

fn open_validated_shard(
    dir: &Path,
    manifest: &Manifest,
    rank: usize,
) -> Result<std::fs::File, DcpError> {
    let mut file = std::fs::File::open(shard_path(dir, manifest.step, rank))
        .map_err(|_| DcpError::MissingShard(rank))?;
    let actual_u64 = file.metadata().map_err(io)?.len();
    let actual = usize::try_from(actual_u64).map_err(|_| DcpError::ValueTooLarge {
        field: "shard_file_len",
        value: actual_u64,
    })?;
    let expected = expected_shard_bytes(manifest)?;
    if actual < expected {
        return Err(DcpError::Truncated {
            needed: expected,
            had: actual,
        });
    }
    if actual > expected {
        return Err(DcpError::TrailingBytes(actual - expected));
    }
    let mut header = [0u8; SHARD_HEADER_BYTES];
    file.read_exact(&mut header).map_err(io)?;
    parse_shard_header(&header, manifest, rank)?;
    Ok(file)
}

/// Write `bytes` to `path` crash-atomically: write a process-unique `.tmp` sibling, `fsync` it,
/// `rename` it into place, then `fsync` the parent directory so the rename is durable.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), DcpError> {
    atomic_write_with(path, |file| file.write_all(bytes).map_err(io))
}

/// Run `write` against a temporary sibling, then fsync and rename it into place.
fn atomic_write_with(
    path: &Path,
    write: impl FnOnce(&mut std::fs::File) -> Result<(), DcpError>,
) -> Result<(), DcpError> {
    // A process-unique temp suffix (pid + atomic counter) so concurrent saves into one directory can
    // never collide on the same `.tmp` source and truncate each other.
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(
        ".tmp.{}.{}",
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let tmp = PathBuf::from(tmp);
    let write_result = (|| {
        let mut f = std::fs::File::create(&tmp).map_err(io)?;
        write(&mut f)?;
        f.sync_all().map_err(io)?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    if let Err(error) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(io(error));
    }
    if let Some(parent) = path.parent() {
        // Directory fsync makes the rename durable across a crash (POSIX). Opening a dir as a file is
        // best-effort (some platforms refuse it), but if it DID open, a sync_all error is a real
        // durability failure and is surfaced rather than swallowed.
        if let Ok(d) = std::fs::File::open(parent) {
            d.sync_all().map_err(io)?;
        }
    }
    Ok(())
}

fn io(e: std::io::Error) -> DcpError {
    DcpError::Io(e.to_string())
}

/// Save `ckpt` as a `world`-rank distributed checkpoint under `dir` (created if absent). Writes one
/// shard per rank (each atomically), then commits the manifest (atomically) — the manifest is the
/// single commit point, so a crash before it commits leaves the previously-committed checkpoint intact.
///
/// **Monotonic-step contract.** Saves must advance the step: each new step writes its own shard files
/// (named by step), so it never overwrites the files the live manifest references. A save whose step
/// is `<=` the committed step is rejected with [`DcpError::NonMonotonicSave`] *before any file is
/// written* — re-saving an already-committed step would overwrite live shards in place, which (under a
/// mid-write crash) could tear the checkpoint. Within this contract, the "old-or-new, never-torn"
/// guarantee holds unconditionally.
///
/// # Errors
/// [`DcpError::NonMonotonicSave`] if `ckpt.step` is not greater than the committed step;
/// [`DcpError::Io`] on any filesystem failure.
///
/// # Panics
/// If `world == 0`, or if `ckpt.param` / any plane length disagrees with `Σ ckpt.leaf_lens` (a caller
/// bug — the buffers must match the declared layout).
pub fn save(dir: &Path, ckpt: &DistCheckpoint, world: usize) -> Result<(), DcpError> {
    let total = ckpt.total();
    assert_eq!(
        ckpt.param.len(),
        total,
        "param length {} != Σ leaf_lens {total}",
        ckpt.param.len()
    );
    for (i, p) in ckpt.planes.iter().enumerate() {
        assert_eq!(
            p.len(),
            total,
            "plane {i} length {} != Σ leaf_lens {total}",
            p.len()
        );
    }
    let mut source = BorrowedCheckpointSource(ckpt);
    save_from(dir, &mut source, world)
}

fn write_source_plane<W: Write>(
    writer: &mut W,
    source: &mut (impl StateSource + ?Sized),
    plane: StatePlane,
    lo: usize,
    hi: usize,
    total: usize,
    scratch: &mut [f32],
) -> Result<(), DcpError> {
    let mut position = lo;
    while position < hi {
        let len = (hi - position).min(scratch.len());
        let values = &mut scratch[..len];
        values.fill(0.0);
        let source_len = total.saturating_sub(position).min(len);
        if source_len != 0 {
            source.read_chunk(plane, position, &mut values[..source_len])?;
        }
        for value in values {
            writer.write_all(&value.to_le_bytes()).map_err(io)?;
        }
        position += len;
    }
    Ok(())
}

/// Save from a pull-based state provider without materializing or cloning the full model state.
///
/// The on-disk representation is exactly DCP wire v1. Each callback range is at most
/// [`STREAM_BUFFER_ELEMENTS`] values. Shards are temp-written, fsynced, and renamed first; the
/// manifest is committed last, preserving the same old-or-new crash contract as [`save`].
///
/// # Errors
/// Returns [`DcpError::NonMonotonicSave`] for a non-advancing step, [`DcpError::TooManyPlanes`] for
/// an unsupported state-plane count, or propagates source and filesystem errors.
///
/// # Panics
/// Panics if `world == 0`, matching [`save`]'s existing contract.
pub fn save_from(
    dir: &Path,
    source: &mut (impl StateSource + ?Sized),
    world: usize,
) -> Result<(), DcpError> {
    let step = source.step();
    let leaf_lens = source.leaf_lens().to_vec();
    let n_planes = source.plane_count();
    if n_planes > MAX_STATE_PLANES {
        return Err(DcpError::TooManyPlanes {
            got: n_planes,
            max: MAX_STATE_PLANES,
        });
    }
    // Monotonic-step guard: refuse to overwrite a committed checkpoint with a non-greater step (which
    // would mutate live-manifest-referenced shard files in place). Checked before writing anything.
    if let Ok(prev_bytes) = std::fs::read(manifest_path(dir))
        && let Ok(prev) = parse_manifest(&prev_bytes)
        && step <= prev.step
    {
        return Err(DcpError::NonMonotonicSave {
            committed: prev.step,
            attempted: step,
        });
    }
    let plan = FlatShardPlan::new(&leaf_lens, world);
    let total = plan.total();

    std::fs::create_dir_all(dir).map_err(io)?;
    let mut scratch = vec![0.0f32; STREAM_BUFFER_ELEMENTS];
    // 1. Write every shard atomically.
    for rank in 0..world {
        let (lo, hi) = plan.shard_range(rank);
        atomic_write_with(&shard_path(dir, step, rank), |file| {
            let mut writer = BufWriter::new(file);
            write_shard_header(&mut writer, step, world, rank, n_planes, plan.chunk())?;
            write_source_plane(
                &mut writer,
                source,
                StatePlane::Parameter,
                lo,
                hi,
                total,
                &mut scratch,
            )?;
            for plane in 0..n_planes {
                write_source_plane(
                    &mut writer,
                    source,
                    StatePlane::Optimizer(plane),
                    lo,
                    hi,
                    total,
                    &mut scratch,
                )?;
            }
            writer.flush().map_err(io)
        })?;
    }
    // 2. Commit the manifest LAST (the single commit point).
    let manifest = Manifest {
        step,
        world,
        n_planes,
        total,
        chunk: plan.chunk(),
        leaf_lens,
    };
    atomic_write(&manifest_path(dir), &serialize_manifest(&manifest))?;
    Ok(())
}

/// Load the committed distributed checkpoint from `dir`, reassembling the world-agnostic global
/// buffers (padding dropped). The result is independent of the save-time world; reshard to a new world
/// `J` with [`FlatShardPlan::new`]`(&ckpt.leaf_lens, J)`.
///
/// # Errors
/// [`DcpError`] if the manifest or any named shard is missing, truncated, stale, or inconsistent.
pub fn load(dir: &Path) -> Result<DistCheckpoint, DcpError> {
    let mut checkpoint = DistCheckpoint {
        step: 0,
        leaf_lens: Vec::new(),
        param: Vec::new(),
        planes: Vec::new(),
    };
    load_into(dir, &mut checkpoint)?;
    Ok(checkpoint)
}

fn validated_manifest(dir: &Path) -> Result<(Manifest, FlatShardPlan), DcpError> {
    let manifest_bytes = std::fs::read(manifest_path(dir)).map_err(io)?;
    let m = parse_manifest(&manifest_bytes)?;
    // Recompute the layout from the (untrusted) leaf_lens + world via the non-panicking constructor: a
    // corrupt manifest (world == 0, overflowing lengths) returns an error instead of panicking the
    // loader. Then check the manifest's own total/chunk agree with the recomputed plan.
    let plan = FlatShardPlan::try_new(&m.leaf_lens, m.world).map_err(|e| match e {
        FlatShardError::ZeroWorld => DcpError::InvalidManifest("world size is zero"),
        FlatShardError::Overflow => DcpError::InvalidManifest("layout length overflows usize"),
    })?;
    if plan.total() != m.total {
        return Err(DcpError::LayoutMismatch {
            field: "total",
            expected: plan.total(),
            got: m.total,
        });
    }
    if plan.chunk() != m.chunk {
        return Err(DcpError::LayoutMismatch {
            field: "chunk",
            expected: plan.chunk(),
            got: m.chunk,
        });
    }
    Ok((m, plan))
}

fn read_plane_into<R: Read>(
    reader: &mut R,
    sink: &mut (impl StateSink + ?Sized),
    plane: StatePlane,
    lo: usize,
    shard_len: usize,
    total: usize,
    scratch: &mut Vec<f32>,
) -> Result<(), DcpError> {
    let mut local_offset = 0usize;
    while local_offset < shard_len {
        let len = (shard_len - local_offset).min(STREAM_BUFFER_ELEMENTS);
        scratch.clear();
        for _ in 0..len {
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes).map_err(io)?;
            scratch.push(f32::from_le_bytes(bytes));
        }
        let global_offset = lo + local_offset;
        let valid_len = total.saturating_sub(global_offset).min(len);
        if valid_len != 0 {
            sink.write_chunk(plane, global_offset, &scratch[..valid_len])?;
        }
        local_offset += len;
    }
    Ok(())
}

/// Load a committed checkpoint into a push-based destination without materializing global buffers.
///
/// Manifest layout plus every shard header and exact file length are validated before `sink.begin`.
/// Payload chunks are then delivered in global-offset order, with DCP padding omitted. The sink owns
/// transactional publication: [`StateSink::finish`] is called only after the full payload succeeds.
/// If payload I/O or a sink write fails after `begin`, the sink may contain partial state and must
/// discard it; `finish` is not called.
///
/// # Errors
/// Returns [`DcpError`] for missing, truncated, trailing, stale, or inconsistent checkpoint files, or
/// propagates an error from the sink.
pub fn load_into(dir: &Path, sink: &mut (impl StateSink + ?Sized)) -> Result<(), DcpError> {
    let (manifest, plan) = validated_manifest(dir)?;

    // Retain the exact files that passed validation. Reopening by path below would introduce a
    // TOCTOU window where a shard could be replaced between validation and payload streaming.
    let files = (0..manifest.world)
        .map(|rank| open_validated_shard(dir, &manifest, rank))
        .collect::<Result<Vec<_>, _>>()?;

    sink.begin(manifest.step, &manifest.leaf_lens, manifest.n_planes)?;
    let mut scratch = Vec::with_capacity(STREAM_BUFFER_ELEMENTS);
    for (rank, file) in files.into_iter().enumerate() {
        let mut reader = BufReader::new(file);
        let (lo, hi) = plan.shard_range(rank);
        read_plane_into(
            &mut reader,
            sink,
            StatePlane::Parameter,
            lo,
            hi - lo,
            manifest.total,
            &mut scratch,
        )?;
        for plane in 0..manifest.n_planes {
            read_plane_into(
                &mut reader,
                sink,
                StatePlane::Optimizer(plane),
                lo,
                hi - lo,
                manifest.total,
                &mut scratch,
            )?;
        }
    }
    sink.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TmpDir(PathBuf);

    impl TmpDir {
        fn new(tag: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "tritium_dcp_stream_{}_{}_{}",
                std::process::id(),
                tag,
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct TrackingSource<'a> {
        checkpoint: &'a DistCheckpoint,
        max_chunk: usize,
        calls_until_failure: Option<usize>,
    }

    impl StateSource for TrackingSource<'_> {
        fn step(&self) -> u64 {
            self.checkpoint.step
        }

        fn leaf_lens(&self) -> &[usize] {
            &self.checkpoint.leaf_lens
        }

        fn plane_count(&self) -> usize {
            self.checkpoint.planes.len()
        }

        fn read_chunk(
            &mut self,
            plane: StatePlane,
            offset: usize,
            out: &mut [f32],
        ) -> Result<(), DcpError> {
            if let Some(calls) = &mut self.calls_until_failure {
                if *calls == 0 {
                    return Err(DcpError::Io("injected source failure".into()));
                }
                *calls -= 1;
            }
            self.max_chunk = self.max_chunk.max(out.len());
            let src = match plane {
                StatePlane::Parameter => &self.checkpoint.param,
                StatePlane::Optimizer(index) => &self.checkpoint.planes[index],
            };
            out.copy_from_slice(&src[offset..offset + out.len()]);
            Ok(())
        }
    }

    #[derive(Default)]
    struct CollectingSink {
        step: u64,
        leaf_lens: Vec<usize>,
        param: Vec<f32>,
        planes: Vec<Vec<f32>>,
        max_chunk: usize,
        finished: bool,
    }

    impl StateSink for CollectingSink {
        fn begin(
            &mut self,
            step: u64,
            leaf_lens: &[usize],
            plane_count: usize,
        ) -> Result<(), DcpError> {
            let total = leaf_lens.iter().sum();
            self.step = step;
            self.leaf_lens = leaf_lens.to_vec();
            self.param = vec![0.0; total];
            self.planes = vec![vec![0.0; total]; plane_count];
            Ok(())
        }

        fn write_chunk(
            &mut self,
            plane: StatePlane,
            offset: usize,
            values: &[f32],
        ) -> Result<(), DcpError> {
            self.max_chunk = self.max_chunk.max(values.len());
            let dst = match plane {
                StatePlane::Parameter => &mut self.param,
                StatePlane::Optimizer(index) => &mut self.planes[index],
            };
            dst[offset..offset + values.len()].copy_from_slice(values);
            Ok(())
        }

        fn finish(&mut self) -> Result<(), DcpError> {
            self.finished = true;
            Ok(())
        }
    }

    fn mk_manifest() -> Manifest {
        Manifest {
            step: 7,
            world: 4,
            n_planes: 2,
            total: 42,
            chunk: 11,
            leaf_lens: vec![24, 18],
        }
    }

    #[test]
    fn manifest_roundtrips() {
        let m = mk_manifest();
        assert_eq!(parse_manifest(&serialize_manifest(&m)).unwrap(), m);
    }

    #[test]
    fn manifest_bad_magic_detected() {
        let mut b = serialize_manifest(&mk_manifest());
        b[0] ^= 0xFF;
        assert_eq!(parse_manifest(&b), Err(DcpError::BadMagic));
    }

    #[test]
    fn manifest_trailing_bytes_detected() {
        let mut b = serialize_manifest(&mk_manifest());
        b.push(0);
        assert_eq!(parse_manifest(&b), Err(DcpError::TrailingBytes(1)));
    }

    #[test]
    fn shard_roundtrips() {
        let m = mk_manifest();
        let param: Vec<f32> = (0..m.chunk).map(|i| i as f32).collect();
        let p0: Vec<f32> = (0..m.chunk).map(|i| -(i as f32)).collect();
        let p1: Vec<f32> = (0..m.chunk).map(|i| i as f32 * 2.0).collect();
        let bytes = serialize_shard(m.step, m.world, 2, &param, &[&p0, &p1]);
        let (gp, gpl) = parse_shard(&bytes, &m, 2).unwrap();
        assert_eq!(gp, param);
        assert_eq!(gpl, vec![p0, p1]);
    }

    #[test]
    fn shard_wrong_rank_is_mismatch() {
        let m = mk_manifest();
        let z = vec![0.0f32; m.chunk];
        let bytes = serialize_shard(m.step, m.world, 1, &z, &[&z, &z]);
        // parsed expecting rank 0, but the shard carries rank 1.
        assert!(matches!(
            parse_shard(&bytes, &m, 0),
            Err(DcpError::ShardMismatch { field: "rank", .. })
        ));
    }

    #[test]
    fn shard_stale_step_is_mismatch() {
        let m = mk_manifest();
        let z = vec![0.0f32; m.chunk];
        let bytes = serialize_shard(m.step + 1, m.world, 0, &z, &[&z, &z]);
        assert!(matches!(
            parse_shard(&bytes, &m, 0),
            Err(DcpError::ShardMismatch { field: "step", .. })
        ));
    }

    #[test]
    fn shard_truncated_detected() {
        let m = mk_manifest();
        let z = vec![1.0f32; m.chunk];
        let bytes = serialize_shard(m.step, m.world, 0, &z, &[&z, &z]);
        let cut = &bytes[..bytes.len() - 4];
        assert!(matches!(
            parse_shard(cut, &m, 0),
            Err(DcpError::Truncated { .. })
        ));
    }

    fn streaming_checkpoint(step: u64, len: usize) -> DistCheckpoint {
        DistCheckpoint {
            step,
            leaf_lens: vec![len / 3, len - len / 3],
            param: (0..len).map(|i| i as f32 * 0.25 - 7.0).collect(),
            planes: vec![
                (0..len).map(|i| -(i as f32) * 0.5).collect(),
                (0..len).map(|i| i as f32 * i as f32 * 0.001).collect(),
            ],
        }
    }

    #[test]
    fn streaming_source_and_sink_roundtrip_with_bounded_chunks() {
        let checkpoint = streaming_checkpoint(9, STREAM_BUFFER_ELEMENTS * 2 + 17);
        let dir = TmpDir::new("roundtrip");
        let mut source = TrackingSource {
            checkpoint: &checkpoint,
            max_chunk: 0,
            calls_until_failure: None,
        };
        save_from(&dir.0, &mut source, 2).unwrap();
        assert_eq!(source.max_chunk, STREAM_BUFFER_ELEMENTS);

        let mut sink = CollectingSink::default();
        load_into(&dir.0, &mut sink).unwrap();
        assert!(sink.finished);
        assert_eq!(sink.max_chunk, STREAM_BUFFER_ELEMENTS);
        assert_eq!(sink.step, checkpoint.step);
        assert_eq!(sink.leaf_lens, checkpoint.leaf_lens);
        assert_eq!(sink.param, checkpoint.param);
        assert_eq!(sink.planes, checkpoint.planes);
    }

    #[test]
    fn streaming_save_is_byte_exact_wire_v1() {
        let checkpoint = streaming_checkpoint(14, 29);
        let dir = TmpDir::new("wire");
        let mut source = TrackingSource {
            checkpoint: &checkpoint,
            max_chunk: 0,
            calls_until_failure: None,
        };
        save_from(&dir.0, &mut source, 4).unwrap();

        let plan = FlatShardPlan::new(&checkpoint.leaf_lens, 4);
        let mut param = checkpoint.param.clone();
        param.resize(plan.padded_len(), 0.0);
        let mut planes = checkpoint.planes.clone();
        for plane in &mut planes {
            plane.resize(plan.padded_len(), 0.0);
        }
        for rank in 0..4 {
            let (lo, hi) = plan.shard_range(rank);
            let plane_slices: Vec<_> = planes.iter().map(|p| &p[lo..hi]).collect();
            let expected = serialize_shard(checkpoint.step, 4, rank, &param[lo..hi], &plane_slices);
            assert_eq!(
                std::fs::read(shard_path(&dir.0, checkpoint.step, rank)).unwrap(),
                expected
            );
        }
        let manifest = Manifest {
            step: checkpoint.step,
            world: 4,
            n_planes: checkpoint.planes.len(),
            total: checkpoint.total(),
            chunk: plan.chunk(),
            leaf_lens: checkpoint.leaf_lens.clone(),
        };
        assert_eq!(
            std::fs::read(manifest_path(&dir.0)).unwrap(),
            serialize_manifest(&manifest)
        );
    }

    #[test]
    fn streaming_load_rejects_truncation_before_mutating_sink() {
        let checkpoint = streaming_checkpoint(3, 97);
        let dir = TmpDir::new("truncate");
        save(&dir.0, &checkpoint, 2).unwrap();
        let path = shard_path(&dir.0, checkpoint.step, 1);
        let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        file.set_len(file.metadata().unwrap().len() - 4).unwrap();

        let mut sink = CollectingSink::default();
        assert!(matches!(
            load_into(&dir.0, &mut sink),
            Err(DcpError::Truncated { .. })
        ));
        assert!(sink.leaf_lens.is_empty());
        assert!(!sink.finished);
    }

    #[test]
    fn streaming_load_rejects_corrupt_header_before_mutating_sink() {
        let checkpoint = streaming_checkpoint(4, 97);
        let dir = TmpDir::new("corrupt_header");
        save(&dir.0, &checkpoint, 2).unwrap();
        let path = shard_path(&dir.0, checkpoint.step, 0);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] ^= 0xff;
        std::fs::write(path, bytes).unwrap();

        let mut sink = CollectingSink::default();
        assert_eq!(load_into(&dir.0, &mut sink), Err(DcpError::BadMagic));
        assert!(sink.leaf_lens.is_empty());
        assert!(!sink.finished);
    }

    #[test]
    fn failed_streaming_save_does_not_replace_committed_manifest() {
        let old = streaming_checkpoint(10, STREAM_BUFFER_ELEMENTS + 9);
        let new = streaming_checkpoint(11, STREAM_BUFFER_ELEMENTS + 9);
        let dir = TmpDir::new("interrupted");
        save(&dir.0, &old, 2).unwrap();

        let mut source = TrackingSource {
            checkpoint: &new,
            max_chunk: 0,
            calls_until_failure: Some(2),
        };
        assert_eq!(
            save_from(&dir.0, &mut source, 2),
            Err(DcpError::Io("injected source failure".into()))
        );
        assert_eq!(load(&dir.0).unwrap(), old);
    }
}

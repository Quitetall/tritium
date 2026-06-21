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

/// Write `bytes` to `path` crash-atomically: write a process-unique `.tmp` sibling, `fsync` it,
/// `rename` it into place, then `fsync` the parent directory so the rename is durable.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), DcpError> {
    use std::io::Write;
    // A process-unique temp suffix (pid + atomic counter) so concurrent saves into one directory can
    // never collide on the same `.tmp` source and truncate each other.
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(
        ".tmp.{}.{}",
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let tmp = PathBuf::from(tmp);
    {
        let mut f = std::fs::File::create(&tmp).map_err(io)?;
        f.write_all(bytes).map_err(io)?;
        f.sync_all().map_err(io)?;
    }
    std::fs::rename(&tmp, path).map_err(io)?;
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
    // Monotonic-step guard: refuse to overwrite a committed checkpoint with a non-greater step (which
    // would mutate live-manifest-referenced shard files in place). Checked before writing anything.
    if let Ok(prev_bytes) = std::fs::read(manifest_path(dir))
        && let Ok(prev) = parse_manifest(&prev_bytes)
        && ckpt.step <= prev.step
    {
        return Err(DcpError::NonMonotonicSave {
            committed: prev.step,
            attempted: ckpt.step,
        });
    }
    let plan = FlatShardPlan::new(&ckpt.leaf_lens, world);
    let padded = plan.padded_len();
    // Pad the flat buffers to the shard-aligned length (zeros beyond `total`, exactly as
    // FlatShardPlan::flatten pads — these padding elements are dropped on load).
    let mut param = ckpt.param.clone();
    param.resize(padded, 0.0);
    let planes: Vec<Vec<f32>> = ckpt
        .planes
        .iter()
        .map(|p| {
            let mut pp = p.clone();
            pp.resize(padded, 0.0);
            pp
        })
        .collect();

    std::fs::create_dir_all(dir).map_err(io)?;
    // 1. Write every shard atomically.
    for rank in 0..world {
        let (lo, hi) = plan.shard_range(rank);
        let plane_shards: Vec<&[f32]> = planes.iter().map(|p| &p[lo..hi]).collect();
        let bytes = serialize_shard(ckpt.step, world, rank, &param[lo..hi], &plane_shards);
        atomic_write(&shard_path(dir, ckpt.step, rank), &bytes)?;
    }
    // 2. Commit the manifest LAST (the single commit point).
    let manifest = Manifest {
        step: ckpt.step,
        world,
        n_planes: ckpt.planes.len(),
        total,
        chunk: plan.chunk(),
        leaf_lens: ckpt.leaf_lens.clone(),
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
    let padded = plan.padded_len();
    let mut param = vec![0.0f32; padded];
    let mut planes = vec![vec![0.0f32; padded]; m.n_planes];
    for rank in 0..m.world {
        let bytes = std::fs::read(shard_path(dir, m.step, rank))
            .map_err(|_| DcpError::MissingShard(rank))?;
        let (ps, plns) = parse_shard(&bytes, &m, rank)?;
        let (lo, hi) = plan.shard_range(rank);
        param[lo..hi].copy_from_slice(&ps);
        for (plane, src) in planes.iter_mut().zip(&plns) {
            plane[lo..hi].copy_from_slice(src);
        }
    }
    param.truncate(m.total);
    for plane in &mut planes {
        plane.truncate(m.total);
    }
    Ok(DistCheckpoint {
        step: m.step,
        leaf_lens: m.leaf_lens,
        param,
        planes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

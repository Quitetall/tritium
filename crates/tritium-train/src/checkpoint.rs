//! `TOPT` training checkpoint: a step counter plus, per trainable leaf, its current
//! parameter values and the optimizer's state for it — everything needed to resume a
//! run bit-exactly. Serialized little-endian.
//!
//! Bit-exact resume falls out for free: f32 little-endian round-trips exactly and the
//! optimizer update is a fixed sequence of f32 ops on identical inputs, so `N` steps
//! uninterrupted equals `N/2` + checkpoint + restore + `N/2` on the same machine
//! (plan 0008). Parsing mirrors the never-panic, bounds-checked discipline of
//! `tritium-format::salt_bundle` but stays self-contained — a local [`CheckpointError`]
//! and [`Cursor`], no `tritium-format` dependency, since a training checkpoint is not a
//! SALT artifact.
//!
//! Layout (little-endian):
//! ```text
//! magic b"TOPT" (4) | version u8
//! step u64
//! leaf_count u32
//! per leaf: len u64 | param[len] f32 | <optimizer state bytes (e.g. AdamW = m[len] f32, v[len] f32)>
//! ```
//! `len` (an element count) is all the deserializer needs; shape is not a checkpoint
//! concern because it is not an [`Optimizer`](crate::optim::Optimizer) concern.

use crate::optim::Optimizer;

/// Checkpoint magic: `b"TOPT"` (Tritium OPTimizer checkpoint).
pub const CHECKPOINT_MAGIC: [u8; 4] = *b"TOPT";

/// Current checkpoint format version. A second optimizer (later) adds a discriminator
/// behind a version bump.
pub const CHECKPOINT_VERSION: u8 = 1;

/// Why a checkpoint failed to parse. Parsing never panics: a corrupt or truncated
/// buffer always yields one of these.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointError {
    /// First four bytes were not [`CHECKPOINT_MAGIC`].
    BadMagic,
    /// Format version this build cannot read.
    UnsupportedVersion(u8),
    /// The buffer ended before `needed` bytes were available (`had` = total length).
    Truncated {
        /// Byte offset the read required.
        needed: usize,
        /// Bytes actually present.
        had: usize,
    },
    /// Bytes remained after a complete parse (count of leftover bytes).
    TrailingBytes(usize),
}

impl core::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CheckpointError::BadMagic => write!(f, "bad checkpoint magic (expected TOPT)"),
            CheckpointError::UnsupportedVersion(v) => {
                write!(f, "unsupported checkpoint version {v}")
            }
            CheckpointError::Truncated { needed, had } => {
                write!(f, "checkpoint truncated: needed {needed} bytes, had {had}")
            }
            CheckpointError::TrailingBytes(n) => write!(f, "{n} trailing bytes after checkpoint"),
        }
    }
}

impl std::error::Error for CheckpointError {}

/// A little-endian cursor that errors (never panics) on a short read, and refuses to
/// pre-allocate for a length the buffer cannot actually hold.
#[derive(Debug)]
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// Wrap a byte buffer at offset 0.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Bytes not yet consumed.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], CheckpointError> {
        let end = self.pos.checked_add(n).ok_or(CheckpointError::Truncated {
            needed: usize::MAX,
            had: self.bytes.len(),
        })?;
        if end > self.bytes.len() {
            return Err(CheckpointError::Truncated {
                needed: end,
                had: self.bytes.len(),
            });
        }
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    /// Read a 4-byte magic tag.
    ///
    /// # Errors
    /// [`CheckpointError::Truncated`] if fewer than 4 bytes remain.
    pub fn take_magic(&mut self) -> Result<[u8; 4], CheckpointError> {
        let s = self.take(4)?;
        Ok([s[0], s[1], s[2], s[3]])
    }

    /// Read a `u8`.
    pub fn u8(&mut self) -> Result<u8, CheckpointError> {
        Ok(self.take(1)?[0])
    }

    /// Read a little-endian `u32`.
    pub fn u32(&mut self) -> Result<u32, CheckpointError> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    /// Read a little-endian `u64`.
    pub fn u64(&mut self) -> Result<u64, CheckpointError> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes(s.try_into().expect("8 bytes")))
    }

    /// Read `n` little-endian `f32`s. Validates the bytes are present *before*
    /// allocating, so a crafted huge `n` can neither overflow the offset arithmetic
    /// nor trigger a giant allocation — the byte count `n*4` and the absolute end
    /// offset are both computed with checked arithmetic.
    pub fn f32_vec(&mut self, n: usize) -> Result<Vec<f32>, CheckpointError> {
        let end = n.checked_mul(4).and_then(|b| self.pos.checked_add(b));
        match end {
            Some(e) if e <= self.bytes.len() => {}
            _ => {
                return Err(CheckpointError::Truncated {
                    needed: end.unwrap_or(usize::MAX),
                    had: self.bytes.len(),
                });
            }
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let s = self.take(4)?;
            out.push(f32::from_le_bytes([s[0], s[1], s[2], s[3]]));
        }
        Ok(out)
    }
}

/// One trainable leaf's slice of a checkpoint: its current parameter values and the
/// optimizer state for it. `param.len()` is the element count `len`.
#[derive(Clone, Debug, PartialEq)]
pub struct LeafCheckpoint<S> {
    /// Current parameter values (row-major, flat).
    pub param: Vec<f32>,
    /// Optimizer state for this leaf.
    pub state: S,
}

/// A full training checkpoint: the completed-step count and one [`LeafCheckpoint`] per
/// trainable leaf, in the caller's leaf order.
#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint<S> {
    /// Number of completed optimizer steps (the next step uses `step + 1`).
    pub step: u64,
    /// Per-leaf params + optimizer state, in the caller's order.
    pub leaves: Vec<LeafCheckpoint<S>>,
}

/// Serialize a checkpoint to the `TOPT` byte format.
pub fn write_checkpoint<O: Optimizer>(opt: &O, ckpt: &Checkpoint<O::State>) -> Vec<u8> {
    // The `leaf_count` header is `u32`; a checkpoint with ≥2³² leaves would silently
    // truncate it. That is unreachable for any real model, but assert rather than
    // corrupt the file. (Per-leaf `len` is `u64` ≥ `usize` on supported targets, so
    // it cannot truncate.)
    debug_assert!(
        u32::try_from(ckpt.leaves.len()).is_ok(),
        "leaf_count exceeds u32::MAX"
    );
    let mut out = Vec::new();
    out.extend_from_slice(&CHECKPOINT_MAGIC);
    out.push(CHECKPOINT_VERSION);
    out.extend_from_slice(&ckpt.step.to_le_bytes());
    out.extend_from_slice(&(ckpt.leaves.len() as u32).to_le_bytes());
    for leaf in &ckpt.leaves {
        out.extend_from_slice(&(leaf.param.len() as u64).to_le_bytes());
        for &p in &leaf.param {
            out.extend_from_slice(&p.to_le_bytes());
        }
        opt.write_state(&leaf.state, &mut out);
    }
    out
}

/// Parse a `TOPT` checkpoint. Enforces magic + version and bounds-checks every field;
/// a corrupt or truncated buffer errors rather than panicking.
///
/// # Errors
/// [`CheckpointError`] on bad magic, unsupported version, truncation, or trailing bytes.
pub fn read_checkpoint<O: Optimizer>(
    opt: &O,
    bytes: &[u8],
) -> Result<Checkpoint<O::State>, CheckpointError> {
    let mut c = Cursor::new(bytes);
    if c.take(4)? != CHECKPOINT_MAGIC {
        return Err(CheckpointError::BadMagic);
    }
    let version = c.u8()?;
    if version != CHECKPOINT_VERSION {
        return Err(CheckpointError::UnsupportedVersion(version));
    }
    let step = c.u64()?;
    let leaf_count = c.u32()? as usize;
    let mut leaves = Vec::with_capacity(leaf_count.min(c.remaining()));
    for _ in 0..leaf_count {
        let len = usize::try_from(c.u64()?).map_err(|_| CheckpointError::Truncated {
            needed: usize::MAX,
            had: c.remaining(),
        })?;
        let param = c.f32_vec(len)?;
        let state = opt.read_state(len, &mut c)?;
        leaves.push(LeafCheckpoint { param, state });
    }
    if c.remaining() != 0 {
        return Err(CheckpointError::TrailingBytes(c.remaining()));
    }
    Ok(Checkpoint { step, leaves })
}

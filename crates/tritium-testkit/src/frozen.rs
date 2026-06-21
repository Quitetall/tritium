//! The frozen, versioned conformance vector set.
//!
//! From v0.70 on, "correct" for a backend means: reproduces a **committed,
//! immutable** set of [`ConformanceVector`]s — not a set regenerated from a seed
//! at test time. Freezing the suite is the prerequisite ADR 0009 names for
//! backend breadth: every new backend (`tritium-wgpu`, `tritium-wasm`,
//! `tritium-metal`, `tritium-rocm`), every v0.80 "matches the native reference"
//! interop gate, and every v1.0 release re-run grades against this one artifact,
//! so the reference must not drift underneath them.
//!
//! The committed file lives at `vectors/<VECTOR_SET_VERSION>.jsonl` in this
//! crate. [`frozen_vectors_path`] resolves it from this crate's
//! `CARGO_MANIFEST_DIR` (a compile-time absolute path), so a consuming crate —
//! whose test process runs with *its own* manifest dir as the working directory —
//! still finds the one canonical set.
//!
//! ## Re-freezing
//!
//! The set is deliberately immutable. To widen coverage (a new format, non-block
//! `K`, more shapes) you re-freeze as a **new version**: regenerate via the
//! `freeze_vectors` example, commit the new `vectors/<ver>.jsonl`, and bump
//! [`VECTOR_SET_VERSION`]. The [`tests::frozen_set_matches_pinned_generator`] gate
//! makes any *accidental* drift (a changed generator, a changed reference kernel,
//! a hand-edited file) a hard test failure.

use std::path::{Path, PathBuf};

use crate::load_vectors;
use crate::vector::ConformanceVector;

/// Version tag of the committed frozen conformance set.
///
/// Bump this — and commit a new `vectors/<version>.jsonl` — only as a deliberate,
/// reviewed re-freeze when op/format coverage genuinely widens. It is the
/// `version` half of "frozen and versioned" (ADR 0009).
pub const VECTOR_SET_VERSION: &str = "v070";

/// The pinned seed the frozen set was generated from (matches the historical CPU
/// conformance gate, so freezing changed *nothing* about what CPU validates).
pub const FROZEN_SEED: u64 = 0xC0FFEE;

/// The pinned random-vector count. The fixed boundary set is appended
/// unconditionally by [`generate_vectors`], so the frozen file holds more than
/// this many vectors.
pub const FROZEN_COUNT: usize = 64;

/// Absolute path to the committed frozen vector file.
///
/// Resolved from this crate's `CARGO_MANIFEST_DIR` at compile time, so it points
/// at the testkit's own `vectors/` directory regardless of which crate's test
/// binary calls it (cargo runs each crate's tests with that crate as the cwd).
#[must_use]
pub fn frozen_vectors_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("vectors")
        .join(format!("{VECTOR_SET_VERSION}.jsonl"))
}

/// Load the committed frozen conformance vector set.
///
/// # Panics
/// Panics only if the committed artifact is missing or unparseable — a build-tree
/// invariant enforced by [`tests::frozen_set_matches_pinned_generator`]. Callers
/// are test harnesses, for which a missing canonical reference is a hard error,
/// not a recoverable condition.
#[must_use]
pub fn frozen_vectors() -> Vec<ConformanceVector> {
    let path = frozen_vectors_path();
    load_vectors(&path).unwrap_or_else(|e| {
        panic!(
            "frozen conformance set {} is missing or corrupt: {e}. \
             Regenerate with `cargo run -p tritium-testkit --example freeze_vectors`.",
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate_vectors;

    /// The teeth. The committed artifact MUST equal what the pinned generator
    /// produces today. This fails if the generator, the reference kernel, or the
    /// committed file drift — forcing a *deliberate* re-freeze (regenerate the
    /// file via the `freeze_vectors` example and bump [`VECTOR_SET_VERSION`])
    /// rather than letting the "frozen" reference silently move.
    #[test]
    fn frozen_set_matches_pinned_generator() {
        let regenerated = generate_vectors(FROZEN_SEED, FROZEN_COUNT);
        let frozen = frozen_vectors();
        assert_eq!(
            frozen, regenerated,
            "frozen {VECTOR_SET_VERSION}.jsonl drifted from \
             generate_vectors(0x{FROZEN_SEED:X}, {FROZEN_COUNT}). If intentional, regenerate via \
             `cargo run -p tritium-testkit --example freeze_vectors` and bump VECTOR_SET_VERSION."
        );
    }

    /// The frozen set is non-trivial: random cases plus the full boundary set,
    /// covering both packing formats. Cheap structural guard against an empty or
    /// half-written artifact passing the equality check by accident.
    #[test]
    fn frozen_set_is_nonempty_and_covers_boundaries() {
        let v = frozen_vectors();
        assert!(
            v.len() > FROZEN_COUNT,
            "frozen set must include the random cases AND the appended boundary set, got {}",
            v.len()
        );
        assert!(
            v.iter().any(|x| x.id.starts_with("boundary-")),
            "frozen set must include the degenerate boundary cases"
        );
        assert!(
            v.iter().any(|x| x.format == "tq2_0"),
            "frozen set must exercise tq2_0"
        );
        assert!(
            v.iter().any(|x| x.format == "tq1_0"),
            "frozen set must exercise tq1_0"
        );
    }
}

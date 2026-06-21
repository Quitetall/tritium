//! Regenerate the committed frozen conformance vector set.
//!
//! Run this **deliberately** (and review the resulting diff) only when re-freezing
//! for a new `VECTOR_SET_VERSION` — it overwrites
//! `vectors/<VECTOR_SET_VERSION>.jsonl` from the pinned seed/count:
//!
//! ```text
//! cargo run -p tritium-testkit --example freeze_vectors
//! ```
//!
//! The `frozen_set_matches_pinned_generator` gate fails until the committed file
//! matches `generate_vectors(FROZEN_SEED, FROZEN_COUNT)`, so this example is the
//! single sanctioned way to (re)produce that file.

use tritium_testkit::{
    FROZEN_COUNT, FROZEN_SEED, VECTOR_SET_VERSION, frozen_vectors_path, generate_vectors,
    save_vectors,
};

fn main() {
    let vectors = generate_vectors(FROZEN_SEED, FROZEN_COUNT);
    let path = frozen_vectors_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create the vectors/ directory");
    }
    save_vectors(&path, &vectors).expect("write the frozen vector set");
    eprintln!(
        "froze {} vectors (seed=0x{FROZEN_SEED:X}, count={FROZEN_COUNT}, version={VECTOR_SET_VERSION}) -> {}",
        vectors.len(),
        path.display()
    );
}

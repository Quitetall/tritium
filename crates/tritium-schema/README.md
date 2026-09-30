# tritium-schema

The canonical Rust vocabulary for Tritium artifact and execution schemas. This
crate is `no_std` by default when built without its `std` feature and has no
runtime dependencies. Wire encodings and law admission are added in later
steps; this initial foundation defines stable semantic values only.

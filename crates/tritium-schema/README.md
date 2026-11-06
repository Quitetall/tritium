# tritium-schema

The canonical Rust vocabulary for Tritium artifact and execution schemas. This
crate is `no_std` by default when built without its `std` feature and has no
default runtime dependencies. The optional `schema-gen` feature derives the
JSON Schema projections from these Rust types; run
`cargo run -p tritium-schema --features schema-gen --bin tritium-schema-projections`
to update them, or pass `--check` to detect drift.

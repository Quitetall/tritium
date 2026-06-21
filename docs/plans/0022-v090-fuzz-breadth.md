# 0022 — Fuzz breadth: a target for every untrusted-byte parser  (serves: v0.90 / ADR 0011; point release `v0.5.9`)

## Goal

Complete the U5 invariant — *every* parser that ingests untrusted model-file bytes has a cargo-fuzz
target — so the v0.90 security review's trust boundary is exhaustively fuzzed. Reachable now, CPU-only,
purely additive (new fuzz targets; no library change). Success: 5 new targets build, run clean, and
join the scheduled CI fuzz sweep.

## Context

`tritium-format/fuzz` already fuzzed 3 parsers (`gguf_parse`, `salt_gguf_parse`, `sparse_plane_parse`).
The remaining untrusted-byte entrypoints had **no** target: `read_tqbin`, `read_tqidx`,
`read_salt_bundle`, `SafeTensors::parse`, `read_legacy_as_salt`. These are exactly the model-file
trust-boundary parsers the v0.90 threat model will enumerate.

## Changes
- **5 new `crates/tritium-format/fuzz/fuzz_targets/*.rs`**: `tqbin_parse`, `tqidx_parse`,
  `salt_bundle_parse`, `safetensors_parse`, `salt_legacy_parse`. Each feeds raw bytes and discards the
  result (the parsers are total by construction; a crash/UB is the only failure). `salt_legacy_parse`
  takes the first 2 bytes as `k` and the rest as the row, so the fuzzer explores both the
  length-mismatch early-return and the valid-length deep decode.
- **`fuzz/Cargo.toml`**: a `[[bin]]` per new target (also refreshes the stale `tritium-format`
  `0.4.0 → 0.5.8` pin in `fuzz/Cargo.lock`).
- **`.github/workflows/ci.yml`**: the scheduled fuzz lane now loops over all 8 targets (~450 s each,
  ~1 h total); adding a parser is now a one-line edit + a target + a `[[bin]]`.

## Gate (verified locally, nightly + cargo-fuzz 0.13.2)
Each new target built, linked real libFuzzer, and ran a 15 s smoke with **zero crashes/leaks** and
healthy coverage growth:
```
tqbin_parse   DONE  cov 78
tqidx_parse   DONE  cov 149
salt_bundle   DONE  cov 236
safetensors   DONE  cov 1062
salt_legacy   DONE  cov 37
```
RSS stayed ~0.5 GB (no unbounded alloc). The 24 h-cumulative sweep is the CI lane's job.

## Follow-up (noted, not in scope)
`tritium-train`'s `dcp` load path also parses untrusted bytes (hardened never-panic in 0016) — it lives
in a different crate with no fuzz harness yet; add a `tritium-train/fuzz` target in a later increment.

## Done criterion
5 targets build + run clean; CI lane sweeps all 8; version `0.5.9`; CHANGELOG + ROADMAP updated;
reviewed; tagged `v0.5.9` + pushed.

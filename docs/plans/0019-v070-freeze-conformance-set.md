# 0019 — Freeze + version the conformance vector set  (serves: v0.70 / ADR 0009; point release `v0.5.7`)

## Goal

Turn the conformance suite from a *regenerated-per-run* seed into a **committed, versioned,
immutable artifact** — the single reference every future backend (wgpu, wasm, metal, rocm), every
v0.80 "matches the native reference" interop gate, and every v1.0 release re-run grades against. The
literal prerequisite ADR 0009 names ("the conformance vector set must be frozen and versioned"), and
the highest-value off-GPU item: it gates all downstream backend/interop/release work and needs no
hardware. Success: a committed `vectors/v070.jsonl`, a `frozen_vectors()` API resolved independent of
the caller's cwd, and a gate that fails if the committed set ever drifts from the pinned generator.

## Context (verified against HEAD `db7f734`)

- `tritium-testkit` already has all the machinery: `generate_vectors(seed,count)` (deterministic
  xorshift + `reference_mpgemm`, byte-identical per `(seed,count)`), `save_vectors`/`load_vectors`
  (lossless JSONL round-trip, tested), `run_conformance`. **Zero `.jsonl` is committed** (`git ls-files`
  confirms) and there is **no version/hash stamp** — the suite is a moving target.
- The CPU primary gate is `generate_vectors(0xC0FFEE, 64)` (`tritium-cpu/src/lib.rs:376`). Freezing the
  **same `(0xC0FFEE, 64)`** means the CPU gate grades against a *provably identical* set — zero
  behavioral change, just locked to a committed file.
- The CUDA conformance test (`cuda.rs:6849`, `generate_vectors(0xC0FFEE, 16)`) is in the user's active
  perf-optimization file. It is **left untouched**: the freeze gate (`frozen == generate_vectors(SEED,
  COUNT)`) proves the generator still reproduces the artifact bit-for-bit, so CUDA's seed-generated
  16-vector set is provably the frozen set's prefix. Repoint it opportunistically when `cuda.rs` is
  next edited — not now, to avoid conflicting with WIP.

## Steps

### Step 1 — `frozen` module (RED first)
- **Files:** `crates/tritium-testkit/src/frozen.rs` (new), `crates/tritium-testkit/src/lib.rs` (wire).
- Add `pub const VECTOR_SET_VERSION = "v070"`, `pub const FROZEN_SEED = 0xC0FFEE`, `pub const
  FROZEN_COUNT = 64`; `frozen_vectors_path()` joining `env!("CARGO_MANIFEST_DIR")/vectors/<ver>.jsonl`
  (compile-time absolute → resolves from any consuming crate's cwd); `frozen_vectors()` =
  `load_vectors(path)` (panic-with-context only if the committed artifact is missing/corrupt — a
  build-tree invariant the gate enforces).
- Unit tests: `frozen_set_matches_pinned_generator` (the teeth: committed `==`
  `generate_vectors(FROZEN_SEED, FROZEN_COUNT)`); `frozen_set_is_nonempty_and_covers_boundaries`.
- **Command:** `cargo test -p tritium-testkit frozen::`
- **Expected (RED):** `frozen_set_matches_pinned_generator` FAILS — the artifact does not exist yet
  (`load_vectors` errors → panic with the missing-path message). This proves the gate actually checks
  the committed file.

### Step 2 — materialize the artifact (GREEN)
- **Files:** `crates/tritium-testkit/examples/freeze_vectors.rs` (new),
  `crates/tritium-testkit/vectors/v070.jsonl` (generated, committed).
- The example writes `generate_vectors(FROZEN_SEED, FROZEN_COUNT)` to `frozen_vectors_path()`.
- **Command:** `cargo run -p tritium-testkit --example freeze_vectors` then
  `cargo test -p tritium-testkit frozen::` and `ls -lh crates/tritium-testkit/vectors/v070.jsonl`.
- **Expected (GREEN):** example prints `wrote 89 vectors …` (64 random + 25 boundary); both frozen
  tests pass. Record the file size. **If > ~8 MB:** reduce `FROZEN_COUNT` (note the change), regenerate.

### Step 3 — repoint the CPU gate
- **Files:** `crates/tritium-cpu/src/lib.rs` (the `conformance_zero_failures` test at ~376 + its import).
- `generate_vectors(0xC0FFEE, 64)` → `tritium_testkit::frozen_vectors()`. Keep
  `conformance_zero_failures_other_seeds` (the generative complement) as-is.
- **Command:** `cargo test -p tritium-cpu conformance`
- **Expected:** `conformance_zero_failures` + `conformance_zero_failures_other_seeds` pass.

## Gate
```
cargo test -p tritium-testkit            # incl. frozen:: gate + existing roundtrip tests
cargo test -p tritium-cpu conformance    # frozen-graded CPU gate green
cargo clippy -p tritium-testkit -p tritium-cpu --all-targets -- -D warnings
cargo fmt --check
```
All green; the committed `vectors/v070.jsonl` is the frozen reference; the teeth fail on drift.

## Commit
```
feat(testkit): freeze + version the conformance vector set (v0.5.7, serves v0.70/ADR 0009)

Commit the conformance suite as an immutable, versioned artifact
(vectors/v070.jsonl = generate_vectors(0xC0FFEE, 64)) instead of regenerating
it per test run. frozen_vectors() resolves the file via the testkit crate's
CARGO_MANIFEST_DIR so any backend crate grades against the one reference
regardless of cwd. The frozen_set_matches_pinned_generator gate fails if the
generator, the reference kernel, or the file ever drift — a re-freeze must be
deliberate (regenerate via the freeze_vectors example + bump VECTOR_SET_VERSION).

The prerequisite ADR 0009 names for all v0.70 backend breadth: every new backend
(wgpu/wasm/metal/rocm) and every later interop/release re-run now grades against
this committed set. CPU gate repointed to frozen_vectors() (identical set, just
locked). cuda.rs left untouched (active WIP); the drift gate proves its
seed-generated set equals the frozen prefix.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

## Review
Adversarial multi-lens review of the commit (project policy: `feature-dev:code-reviewer` subagent /
review workflow — NOT lamu). Triage findings, verify before fixing.

## Done criterion
`vectors/v070.jsonl` committed; `frozen_vectors()` exported; the drift gate + CPU conformance green;
clippy/fmt clean; version bumped to `0.5.7`; CHANGELOG + ROADMAP row updated; tagged `v0.5.7` + pushed.

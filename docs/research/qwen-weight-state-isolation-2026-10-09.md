# Qwen weight-state isolation

Date: 2026-10-09
Evidence class: host implementation tests; not release qualification
Parent: ADR 0033 / plans 0043 and 0051
Contract: private research commit
`5bf90a94de0e17bbc8eb9c53be72f51cd21b0484`,
`adr/0033-qwen-weight-state-isolation-2026-10-09.md`.

## Finding and change

Previously, a successful in-place SALT scale update retained the language
runner identity. Caches and outputs created using the old weights could still
be passed to the changed runner. Temporary projection probes likewise used
the parent identity, so a candidate output could be evaluated with the parent
head or aligned with its MTP graph. Identity must denote one live weight state.

Successful scale updates now start a new identity epoch. Old caches/outputs
are rejected by forward execution, cache observations and per-row head
evaluation. Resetting an old cache does not rebind it. Failed updates preserve
the existing identity; successful identical updates conservatively invalidate
previous objects too.

Temporary probes receive their own identity. Their restoration guard restores
the original identity only after restoring the original projection, on return,
error and panic unwind. Parent objects remain valid after successful
restoration; candidate objects cannot cross back into the parent. Existing MTP
exact-target checks apply unchanged. Permanent mutation does not reauthorize
the old drafter: reload an immutable child to assemble target/MTP together.

The packed load receipt describes the initial load, not qualification of a
subsequent in-place candidate. No production oracle is authorized by this
change, and the numerical tolerances and release gates are unchanged.

## Verification state

The original managed validation unit
`tritium-qwen-state-validation-20261009.service`, invocation
`fbe2afd403004624a7be26ca8d308f48`, terminated with exit 101: nine integration
tests passed and two new assertions failed. The assertions expected
`NnError::Provenance`, but the existing public `forward` foreign-cache contract
returns `NnError::Backend("Qwen3.5 text cache belongs to a different runner")`.
A targeted reproduction confirmed that rejection; the tests now check the
existing error variant and reason. Snapshot/head and MTP provenance checks
retain their `Provenance` assertions. Temporary diagnostic logging was removed.

Subsequent commands used the existing
`/mnt/4tb/tmp/tritium-research-target` cache, `RUSTC_WRAPPER=`,
`CARGO_BUILD_JOBS=2` and `TMPDIR=/mnt/4tb/tmp`:

```sh
timeout 300 cargo test --locked -p tritium-nn --test qwen35_text_runner
timeout 300 cargo test --locked -p tritium-nn --lib
timeout 1200 cargo clippy --locked -p tritium-nn --all-targets -- -D warnings
```

The integration suite passed 11/11 tests; the library suite passed 167/167.
Neither suite had ignored tests. The first scoped Clippy run reached its
300-second limit (exit 124) without a terminal lint result. Its follow-up is
managed by `tritium-qwen-state-lint-20261009.service`, invocation
`587e01aa7d2f41eb814cf63ab8198ff1`, with a 1,200-second limit. That job completed
successfully (exit 0); Cargo reported 3m 32s. Scoped
`rustfmt --edition 2024 --check` and `git diff --check` also passed.

New tests cover successful/rejected/identical
scale updates, stale cache cursor/reset
behavior, native snapshots and head outputs, temporary parent/candidate cache
and output separation, MTP alignment, and success/error/unwind/invalid-override
restoration. Controlled host tests do not establish physical CUDA behavior.

## Remaining obligations

Complete pushed-tree checks and CI before calling this software slice verified.
Production MTP needs independent authorization bound to
the executed artifact: the current dense-source oracle identity is not a
license to relabel a lossy packed candidate as that dense model. Final Qwen
fitting/refinement, source/quality/runtime/physical-byte gates, backend/browser/
distributed qualification, packaging/deployment/model-zoo evidence and
independent reproduction/human release approval remain open.

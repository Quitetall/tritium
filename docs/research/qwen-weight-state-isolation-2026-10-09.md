# Qwen weight-state isolation

Date: 2026-10-09
Evidence class: implementation with validation pending; not release qualification
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

Scoped `rustfmt --edition 2024 --check` and `git diff --check` passed. The first
narrow Cargo test invocation reached its 300-second limit (`exit 124`) after
waiting for the shared build cache and starting dependency compilation. It did
not produce a test result.

The managed validation unit
`tritium-qwen-state-validation-20261009.service`, invocation
`fbe2afd403004624a7be26ca8d308f48`, waits for the already-running managed push
before sequentially running the following with the existing SSD cache and two
build workers:

```sh
timeout 600 cargo test --locked -p tritium-nn --test qwen35_text_runner
timeout 600 cargo test --locked -p tritium-nn --lib
timeout 600 cargo clippy --locked -p tritium-nn --all-targets -- -D warnings
```

Validation remains pending until terminal results are inspected. New tests
cover successful/rejected/identical scale updates, stale cache cursor/reset
behavior, native snapshots and head outputs, temporary parent/candidate cache
and output separation, MTP alignment, and success/error/unwind/invalid-override
restoration. Controlled host tests do not establish physical CUDA behavior.

## Remaining obligations

Inspect the managed validation results and repair any failures before calling
this slice verified. Production MTP needs independent authorization bound to
the executed artifact: the current dense-source oracle identity is not a
license to relabel a lossy packed candidate as that dense model. Final Qwen
fitting/refinement, source/quality/runtime/physical-byte gates, backend/browser/
distributed qualification, packaging/deployment/model-zoo evidence and
independent reproduction/human release approval remain open.

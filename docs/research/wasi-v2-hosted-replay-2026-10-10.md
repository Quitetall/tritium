# Hosted WASI V2 execution and exact replay — 2026-10-10

Executed clean source: `38e0f2411d423bae909d44f81696d83c92bea9c9`.
This adds WASI to the existing source-38 CPU/CUDA/native-wgpu evidence subset.
It is not all-seven backend qualification, performance evidence, model-quality
evidence, a new candidate's evidence or full-release approval.

## Hosted artifact, not just a green job

GitHub [CI run 38036743231, WASI job 114168705032](https://github.com/Quitetall/tritium/actions/runs/38036743231/job/114168705032)
checked out and verified the exact source above. The retained job log shows
Wasmtime `48.0.1 (7bac2c277 2026-08-24)`, rustc
`1.99.0 (b940084d7 2026-09-28)`, 14 guest tests passing without skips, actual
guest sealer execution and strict Rust admission of its emitted bundle.

Artifact `11664068563` was downloaded without triggering a new CI run. Its
original ZIP SHA-256 matches GitHub's recorded digest:
`971db8e3d71cc343c9dacefeee1d1be1fe57e77ef5f1d88617b8a6dd11062d8e`.
The single original JSON member is retained unchanged:

- [WASI bundle](evidence/wasi-v2-38e0f241/ba4cbd6b3a12b2dbed857ee977491e4c2984ce5df351285a7f28adee2bba1830.json).
- BLAKE3: `ba4cbd6b3a12b2dbed857ee977491e4c2984ce5df351285a7f28adee2bba1830`.
- SHA-256: `9f81bb4aaee550d437f4b06c02351f6ff9222f2a42938c768c827e79120e97d4`.
- Build: `tritium-wasm@1.1.0-rc.2+source-git:38e0f2411d423bae909d44f81696d83c92bea9c9`.
- Physical runtime identity: `wasmtime:48.0.1:x86_64`.

The guest executed all 36 frozen operations and 117 vectors: 72 successful
executions and 45 expected errors. Strict Rust `training_capability_table
--schema v2 DIGEST=PATH` and separate Python `source_contract`/`validate_bundle`
admission passed. Runtime identity is operator-supplied with retained host
provenance, not cryptographic runtime attestation.

## Local source-38 reproduction

The clean source-38 managed checkout was built with two Cargo jobs in the
existing SSD cache, `RUSTFLAGS="-D warnings"`, and source/check-only/Git
redirect environment variables unset. Local rustc was
`1.98.0 (88d9e12ae 2026-08-18)`. A temporary official Wasmtime v48.0.1
Linux/x86_64 download was checksum-verified against its release-asset metadata;
no global runtime installation or PATH change was made.

Local guest tests passed 14/14 without skips. Python backend admission and
qualification suites passed 5/5 before any source edits. The corrected local
guest was run twice. Both outputs matched the hosted JSON byte for byte using
`cmp`, and separate Python admission passed for both. The corrected guest
module SHA-256 is
`9c89723c9b9ff2b3c79bec16a852fe5b7b8769f5bee4d4653923e33480e568d3`.
Different host compiler versions did not change these particular receipt bytes;
this is not a general cross-compiler equivalence claim.

The runtime archive SHA-256 was
`4c2e31b68ad99e0a519f225a261fda099eb15f056d4a24fdb3c2a46517bde1df`;
the extracted binary SHA-256 was
`5a61e28214e31c2a52154103407bae5265a36732b3af502609e5ffe19f249463`.

## Diagnostic failure and producer repair

The first local command was incorrectly launched through `systemd-run` with
default environment expansion. Systemd expanded shell-local `${version}` and
`${arch}` before bash computed them, compiling the stamp `wasmtime::` into the
guest. Its bundle `213db9346c02eb2f87ce497cc225a0c3f90c33a760db48f48cb9bfd50c1bf4ba`
is diagnostic only. Only the physical-device field differed from the hosted
bundle. A minimized red/green systemd probe confirmed the cause; rebuilding
with `--expand-environment=no` produced the exact hosted bytes at unchanged
source-38. This was an invocation defect, not a hosted CI failure.

The sealer's prefix/length guard also accepted that incomplete identity, despite
its documented non-placeholder requirement. The implementation repair requires
exactly `wasmtime:<version>:<architecture>` with nonempty components and no
whitespace, control characters or extra fields. It retains arbitrary legitimate
version/build and architecture labels: no architecture whitelist or SemVer
policy is introduced. Example regression tests exercise the helper actually
called by `main`; the WASI CI test selection includes examples via
`--all-targets`. No receipt-consumer schema or frozen admission rule is changed.
These repair tests are separate development evidence, not execution of the
repair at the older source-38 identity.

The regression was observed red inside Wasmtime before the repair: the original
guard returned `Some("wasmtime::")` where the test required `None`. Repeating
with uncaptured output retained that exact assertion. After repair,
`cargo test --locked --target wasm32-wasip1 -p tritium-wasm --all-targets --
--nocapture` passed 14 library tests and 2 example tests, without skips; example
execution took less than 0.01s after compilation. CI also retains an explicit
`--doc` invocation rather than losing the previous doctest selection. A first
native test build hit its 180-second compile deadline before reaching tests;
that run is recorded as a timeout, not a pass or a regression failure.

## Limits, custody and remaining gates

Reported peak resident bytes `4192`, scratch bytes `132032` and zero host
transfers are tiny-corpus backend counters, not process RSS, model-size or
independent steady-state transfer/synchronization measurements. The successful
hosted WASI lane does not promote skipped CUDA/ROCm/Metal/performance CI jobs.
ROCm, Metal and MCU physical semantic bundles remain open. A future candidate
must regenerate every source-bound backend bundle; these bytes are not relabeled.
Browser WebGPU/WASM packaging and Chrome/Firefox/Safari training/fault evidence
remain distinct obligations.

Original hosted ZIP/JSON, retrieval metadata, job log, local guest modules,
build/test and diagnosis logs are retained at
`/home/brianklam/Projects/Tritium/archive/verification/wasi-v2-38e0f241-20261010/`.
Verified replay JSON duplicates and temporary runtime downloads are removed
after testing; one malformed diagnostic bundle remains clearly labeled. Public
and archive JSON bytes are compared after transfer. August scratch, model
weights, shared cache and unrelated work are not deleted. The owned repair
branch is removed after integration; the pinned managed checkout is preserved.

The full release still requires serving empirical qualification; PyTorch/HF
lifecycle/PTQ/refinement/distributed evidence; remaining physical backend and
browser/performance matrices; whole-model ONNX and complete packaging/tutorial/
Colab provenance; authorized recipe freeze and Qwen language/MTP quality,
physical bytes, runtime/memory and reproduction; actual production security/
deployment; audited model zoo/community; independent clearance, signing,
explicit human activation and authorized publication.

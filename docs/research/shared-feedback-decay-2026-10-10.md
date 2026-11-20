# Shared additive-quantization feedback decay

Code source: `5828a72c478503e03d462ae8ecbafdce1adaf798`.
Original source: `333375283ce26bd3b73c6bb7a38eab0c476b906a`.
Scope: accepted ADR 0044 D11 / plan 0055 P3 prerequisite, not a completed phase
or model-quality, performance, backend or release qualification.

## Change

`tritium-core::FeedbackDecay` owns validated constant/ramped propagation
fractions. Invalid, non-finite or out-of-range fractions reject rather than
clamp. A ramp uses the global source-column order, not a group-local ordinal.
The explicit policy is allocation-free and `no_std`; the historical
SmolLM2-derived log-density heuristic remains `std`-only and is not a universal
quality rule.

The shipping NN ladder fitter and the f64 reference feedback fitter now use
this policy. The existing NN `auto_decay` interface delegates to the shared
rule. The reference gains opt-in `fit_with_feedback_decay`, stores O(columns)
decay factors, retains them across reconstruction replacement/suffix refit,
and validates curvature even when decay is zero. Unit decay bypasses the new
multiplication to preserve the original arithmetic. Existing default calls
remain undecayed.

This does not consolidate the two working-precision implementations. Shipping
NN fitting still uses its existing f32 working weights/f64 compensation; the
reference is still an f64 oracle. F32Accum64 remains the intended engine
default until a recorded model-quality A/B justifies a change. No recipe,
package identity, tensor format, runtime kernel or precision default changed.
The future QuantPlan must bind the schedule before new recipes are admitted.

## Original-source evidence

Before changing production source, sixteen shipping-fitter payload digests
were captured from the original revision: exact scale bits and trit bytes over
decay 1.0/0.5, ramp off/on, rotation off/on and post-pass/in-loop scale refit.
The committed public integration test matches all sixteen frozen digests.
This is a small numerical fixture, not model calibration or quality evidence.

A separately executed Rust probe compiles the actual original
`salt_v2_feedback.rs` Git-object snapshot, alongside the new public interface.
It compares f64 bits for working weights, reconstructions and each group-fit
input before and after reconstruction replacement/suffix refitting. All 80
cases pass: source widths 1/2/3/7/16, group widths 1/2/3/8, rows 1/2, and unit
decay with each ramp setting. The original snapshot produces three dead-code
warnings because this probe deliberately exercises only part of its interface;
production Clippy remains warning-free.

The first payload-capture fixture failed to compile with E0689 because its
integer width lacked a type annotation. The corrected fixture then captured
the baseline before production edits. Both logs are retained; the fixture
error is not a reproduced quantizer defect.

## Verification on the clean code commit

Host: Linux x86_64, Intel Core i9-14900K; CPU tests only. Commands ran in the
isolated checkout, with `RUSTC_WRAPPER=` and `CARGO_BUILD_JOBS=2` for Cargo checks:

```sh
cargo test --locked -p tritium-core -p tritium-quantize --lib --tests
cargo test --locked -p tritium-nn --lib salt_fit::
cargo test --locked -p tritium-nn --test feedback_decay_legacy
cargo check --locked -p tritium-core --no-default-features
cargo clippy --locked -p tritium-core -p tritium-quantize -p tritium-nn --all-targets -- -D warnings
cargo fmt --all --check
git diff --check
```

Results: **318 passed / 0 failed / 10 ignored** in the core/quantizer suite,
**6 passed** in the NN fitting library tests, **2 passed** in the frozen-payload
integration suite; no_std check, scoped Clippy and formatting pass.
The ten declared ignores are four manual solver-profile tests, one Qwen
allocation-curve test, one GPT-2 reconstruction-curve test, and four Qwen
single-plane quality tests. None were run or promoted to passing evidence.
Hand-worked decay tests cover scalar active-Schur correction, multi-column
groups, global ramp order, retained refit policy and invalid-metric rejection
before invoking the fitter at zero decay.

The original-source probe command was:

```sh
CARGO_TARGET_DIR=/mnt/4tb/tmp/tritium-research-target \
  cargo run --offline --locked \
  --manifest-path /mnt/4tb/tmp/tritium-feedback-decay-20261010/probe/Cargo.toml
```

Its source, lockfile and original snapshot are preserved below; scratch is
removed after custody checks. Reproduction requires a checkout of the code
revision and updating the probe's two local dependency paths to that checkout.

## Custody and remaining work

Durable archive:
`/home/brianklam/Projects/Tritium/archive/verification/shared-feedback-decay-5828a72c-20261010`.
It retains the eight changed source files, original feedback source, probe and
logs. Its separate stdlib verifier compares snapshots against externally
supplied original Git objects without importing the new numerical module;
checksums establish retained bytes only.

This is only a P3 prerequisite. The accepted one-schema/one-core plan still
requires P0's complete decode baseline, P1 schema/core/evidence foundations,
P2 format/backend conformance, the full P3 F32Accum64 engine and exact shipping
parity, direct-fit allocator/quality A/B, and P4-P7 loader/runner/surface
migration and verified deletion. Full Stage-7 measurements/auxiliary producers
and G64/G256 solver admission remain open; no full grid was run here.

Other v1.1 fronts remain measured-observability admission, representative
PTQ/refinement and matched baselines, Qwen language/MTP quality/bytes/runtime/
reproduction, physical backend/browser/multi-device performance, serving and
production fault/security qualification, final packages/ONNX/Colab/model zoo/
community, second-machine/operator clearance and explicit human public-release
authorization. No large capture/fitting, paid compute, model-weight deletion,
signing, publication, deployment or human activation was performed.

# Bounded native Qwen cache observations

Date: 2026-10-09
Evidence class: local software tests; no final 27B or GPU qualification
Parent: ADR 0033 / plan 0051
Contract: private research amendment
`258a1bb041abd7ee4d82447ca006da4d9edbbca4`,
`adr/0033-native-cache-observations-2026-10-09.md`.
MTP worker contract extension:
`ca741c2514ff78e8d58bbe1f971f03b22f887f1c` in the same amendment.

## Gap and implemented behavior

The prior worker compared ONNX caches with deterministic ONNX replay, not with
native execution. Identical logits and replay caches could therefore hide
cache-value drift. Language prompt/decode cases now require native cache
observations and include their maximum absolute numeric error, plus both
observed/replay logits, in the existing frozen `1e-3` error gate. Exact ONNX
replay remains mandatory. No receipt schema, production MTP admission or
empirical acceptance is changed.

Authorized MTP prefill/decode transactions now have the same requirement.
Both observed and replay K/V outputs must match native observations in the
exact ordered `present_k.0`, `present_v.0` inventory; incomplete inventories
cannot qualify even if both executions omit the same state. Cache errors join
the existing maximum-absolute-error gate alongside hidden-state/logit errors.
Missing or malformed native observations fail closed. This worker behavior
does not create the still-unavailable production MTP authorization interface.

Rust language and MTP runners return immutable owned `Qwen35ReferenceState`
observations from an exact-runner committed cache. Names and flattened layouts
match exported graphs: layer-indexed convolution/recurrent state, rotated
keys and values. A shared helper checks positive axes, geometry, overflow,
aggregate payload budget and finite values before copying. It rejects empty
state, foreign runners, inconsistent layer cursors and device-owned DeltaNet
recurrence instead of exposing stale host buffers.

Python observations are disabled by default. `include_states=True` exposes
copied names, shapes and values; `state_steps` selects unique transaction
indices. Retained native values share one positive budget capped at 256 MiB,
excluding additional Python getter copies. A failed observation publishes no
partial output list and each invocation owns a fresh cache.

The pinned Qwen config has 48 recurrent layers, 48 value heads, and 128x128
FP32 recurrence: 144 MiB per snapshot before convolution/KV. Two snapshots
exceed 256 MiB. The worker observes prompt separately and selects only decode
while replaying prompt plus continuation, retaining the full state inventory.

## Verification

Before implementation, ten controlled Python regressions failed: missing
native comparison and undetected decode-only cache drift. Rust integration
tests failed with `E0599` for the absent runner observation interface.

Commands use the existing SSD build cache, with `RUSTC_WRAPPER=` and
`CARGO_BUILD_JOBS=2`:

```sh
timeout 180 cargo test --locked -p tritium-nn --test qwen35_text_runner
timeout 180 cargo test --locked -p tritium-nn --lib
timeout 300 cargo check --locked -p tritium-py --no-default-features
```

Results: eight hybrid-runner integration tests and 164 native library tests
passed; Python binding compilation passed. Native tests cover layout, exact
payload budget, cursor/reset isolation, malformed/non-finite/overflow state,
foreign runner rejection and draft-only MTP observations without promotion.

The qualification-worker, archive-binding and ONNX Python regression suite
passed 85 tests without skips. A subsequent worker rerun with transaction
selection and its added regression passed 33 tests. Producer, verifier and
workflow regressions passed 11 tests. These controlled software tests do not
establish final-candidate numerical parity or GPU state observation.

The native Python adapter's five `qwen::tests` also passed, including shared
snapshot-budget accounting and selecting decode without retaining prefill.
The pre-push hook now honors explicit `CARGO_TARGET_DIR`, avoiding a second
build cache when the project SSD cache was already selected. Its legacy
default remains unchanged; three controlled shell-execution tests passed for
explicit, unset and empty target settings, with formatting/Clippy still called.

The MTP extension first reproduced 18 failures with unchanged logits but
drifted observed/replay K or V, absent/malformed native observations and
reordered ONNX inventories. After implementation, the worker suite passed 52
tests, including a nineteenth regression for mutually matching but incomplete
KV inventories. The combined worker, bundle-binding, Torch ONNX, producer and
receipt-verifier suite passed 112 tests without skips:

```sh
timeout 90 env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=crates/tritium-py/python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m pytest -q -p no:cacheprovider --tb=short \
  crates/tritium-py/tests/test_qualify_onnx_worker.py \
  crates/tritium-py/tests/test_bundle_binding.py \
  crates/tritium-py/tests/test_torch_onnx.py \
  scripts/tests/test_qualify_onnx_inference.py \
  scripts/tests/test_verify_onnx_inference_receipt.py
```

These tests use controlled adapters and generated test payloads; they do not
execute the final Qwen candidate or establish production MTP authorization.

## Remaining obligations

- Production MTP promotion and Python `reference_mtp` remain unavailable.
  Core draft-only snapshots do not grant production authorization.
- Native-versus-ONNX MTP cache comparison is enforced by the worker but still
  needs an authorized native oracle and final-candidate execution evidence.
- Execute full prompt/decode/DeltaNet/MTP comparisons against the exact final
  Qwen candidate and retain independent qualification evidence.
- Device-owned state observation needs an explicit download interface and
  physical hardware verification; current observations fail closed.
- Release fitting, quality, runtime, physical-byte, packaging, backend,
  model-zoo and independent reproduction gates remain separate obligations.

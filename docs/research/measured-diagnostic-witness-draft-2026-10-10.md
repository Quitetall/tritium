# Measured diagnostic witness draft — 2026-10-10

Status: **draft substrate tested; release admission unchanged and NOT qualified**.
Private proposed ADR 0053 requires human adoption before the observability v2
producer/validator/registry migration. No public API is exported by this change.

## Reproduced admission debt

The current `tritium.installed-observability.v1` producer writes literal memory
4096, decode time 3.5 and teacher KL 0.125. Its validator requires those samples.
Replaying actual local, hosted and source-free receipts retained under
`observability-binding-65d3b3c1-20261010` admits all three without a measurement
witness. The dedicated negative probe fails repeatedly, including after this
draft. Historical receipts are unchanged. They cannot establish the measured
diagnostic obligation in ADR 0033; no v2 release credit is claimed.

## Implemented private substrate

Measured code source: `e4a203172fe96f1fb876ef3dfc208fc325f5984a`.

- `_diagnostic_measurements.py` checks exact native/release/archive/executing
  payload identities before constructing the existing tied 4x2 CPU SALT QAT
  fixture. The default `SaltSTE` is preserved; a first draft incorrectly
  required `AbsMeanSTE`, and the actual-wheel test caught that mismatch.
- Five warmups and 31 synchronous `perf_counter_ns` intervals retain each
  actual output. The scalar is median **CPU forward milliseconds**, including
  the Python/native operation boundary, not full-model/token decoding speed.
- Linux `/proc/self/statm` and page size retain **current whole-process RSS**
  after inference and before telemetry. This includes framework/allocator
  state; it is not model-only, peak, training or GPU memory.
- Dense-teacher and actual hard-student logits yield mean KL(teacher||student)
  at temperature one using float64 log-softmax. Retained weights, projection,
  inputs and logits permit separate stdlib arithmetic replay.
- `_diagnostic_witness.py` checks strict fields, frozen projection/forward
  arithmetic, all timed outputs, interval ordering/count, RSS arithmetic,
  finite scalars, external candidate binding and canonical identity. Re-signing
  inconsistent values does not make them acceptable. A witness alone cannot
  prove execution or grant release credit; no hostile-Python sandbox is claimed.

The private schema is `tritium.diagnostic-measurement-witness.draft1`, not
`tritium.installed-observability.v2`. Existing release receipts and admission
functions are deliberately untouched while adoption is pending.

## Executed evidence

Exact-source local wheel SHA256:
`d7faf1f9ee3615f162bb0b6dfc6ee1c321dd6629af7d6f48e2811b6d3db79ccf`.
Maturin completed the final native release build in 49.78 seconds. It warned
that the local `linux_x86_64` wheel cannot be uploaded to PyPI; this is not a
portable release package. Installation used force-reinstall, no index, no
dependencies and no bytecode compilation in a disposable SSD venv.

The exact installed wheel produced witness
`sha256:08292098dadded6c0154cf7e80e944cd05a122cceb9c2ad0ec182407fb293922`:

- CPU fixture forward median: `0.057997` ms.
- Whole-process current RSS: `911695872` bytes.
- Fixture teacher KL: `0.019607035739791598`.

These are one local tiny-fixture observation, not benchmark or model-quality
qualification. CPU work only; no flagship fitting or cloud compute was run.
The independent `python3 -I -S -B verify_draft.py` process imports no Torch or
installed Tritium and derives the same scalars from retained raw evidence,
while binding the source/release/run/archive digest supplied externally.

Focused installed-wheel pytest: **58 passed in 0.39s**, covering new draft
arithmetic/tampering/actual-candidate rejections and existing observability
receipt/provenance tests. The initial source pytest failed because the checkout
had no native extension; an explicitly stdlib-only `--noconftest` subset passed
34 tests before the wheel build. That subset is not installed qualification.
Normal commit hooks and actionlint pass. CI now includes the new draft tests;
candidate CI results remain separate from these local measurements.

Durable evidence:
`/home/brianklam/Projects/Tritium/archive/verification/diagnostic-witness-e4a20317-20261010`.
It retains the final wheel, marked earlier draft wheel, raw witness, verifier,
validator snapshot, logs and repeated legacy counterexample. No old campaign
outputs or unapproved August caches were deleted.

## Remaining measured-diagnostics migration

After human adoption: change the actual producer, portable admission/replay,
registry, CI retention and contract docs together; bind all three real telemetry
adapters to witness-derived values, preserve explicit historical v1 inspection,
reject v1 for measured release admission, and rerun the original counterexample.
Then execute source/compiler/network-free exact-candidate evidence and a separate
verifier, followed by clean CI and independent release clearance. Until those
steps pass, diagnostics qualification is **UNKNOWN/MISSING**, not repaired.

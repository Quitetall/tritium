# Frozen V2 CPU conformance — 2026-10-10

Executed source: `b7b03ace4bfbd83f1e40b063f1f27e7cfe5d02b8`.
This is the CPU lane's executed semantic-conformance evidence, not the complete
seven-backend release receipt or a final-release approval.

## Observed result

The clean-source CPU reference adapter executed all 36 operations and 117
frozen `TrainingOpManifestV2` cases: 72 successful executions and 45 expected
errors. The producer grades actual output buffers and receipts against the
frozen vectors before sealing; it does not merely serialize expected outputs.
A second execution produced byte-identical sealed bytes, verified with `cmp`.

The separate Rust `training_capability_table` process admitted the bundle with
the strict `ReleaseCandidate` policy (no `--allow-dirty`). The separate Python
verifier checked the frozen source-contract hashes, exact build revision and
CPU bundle coverage/capabilities. The two Python backend qualification/admission
suites passed 5/5 in 11.398s.

Physical identity is
`cpu:linux:x86_64:Intel(R) Core(TM) i9-14900K:32-logical`; `/proc/cpuinfo`
independently identifies the i9-14900K. Embedded build identity is
`tritium-train@1.1.0-rc.2+source-git:b7b03ace4bfbd83f1e40b063f1f27e7cfe5d02b8`.
The isolated checkout was clean at this same SHA before and after execution.
No source identity override was injected during compilation.

The immutable [sealed bundle](evidence/cpu-v2-b7b03ace/0d9b585dfc4093ffc4bf5a3d08360b854de0a99c9787d1706383410e755fd549.json)
is retained without edits:

- BLAKE3: `0d9b585dfc4093ffc4bf5a3d08360b854de0a99c9787d1706383410e755fd549`.
- SHA-256: `11322be6c18d62b89290a017842f87b71dbaa478f8cdd65bf0b4abd7cfb5c9fa`.
- Reported corpus peak resident bytes: 4192; peak scratch bytes: 132032.
- Reported host transfers: zero.

Those byte counters describe backend allocations for small conformance vectors,
not whole-process RSS, Qwen model memory, compression ratio or throughput.

## Reproduction

Build from the exact clean source. The observed build used the existing SSD
Cargo cache, two jobs, `RUSTFLAGS=-Dwarnings`, a 600s outer timeout and an owned
systemd user unit. `TRITIUM_SOURCE_ID`, `GIT_DIR`, `GIT_WORK_TREE` and
`GIT_INDEX_FILE` were unset. Compilation completed in 1m26s; no performance
claim is made from build duration.

```bash
cargo build --locked -p tritium-train -p tritium-testkit \
  --example seal_cpu_training_receipts --example training_capability_table
timeout 60 "$CARGO_TARGET_DIR/debug/examples/seal_cpu_training_receipts" \
  --schema v2 "$OUTPUT/receipts"
timeout 60 "$CARGO_TARGET_DIR/debug/examples/training_capability_table" \
  --schema v2 \
  "0d9b585dfc4093ffc4bf5a3d08360b854de0a99c9787d1706383410e755fd549=$OUTPUT/receipts/0d9b585dfc4093ffc4bf5a3d08360b854de0a99c9787d1706383410e755fd549.json"
timeout 120 env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/4tb/tmp \
  python3 -m unittest scripts.tests.test_verify_training_backend_receipt \
  scripts.tests.test_qualify_training_backends
```

For the Python CPU admission, load
`scripts/verify-training-backend-receipt.py`, call `source_contract(repo)`, then
`validate_bundle("cpu", bundle, executed_source_revision, operations, vectors)`.
The full seven-family aggregator was not invoked with a partial/fabricated
inventory; its frozen all-seven requirement remains intact.

The local durable archive is
`/home/brianklam/Projects/Tritium/archive/verification/cpu-manifest-b7b03ace-20261010/`.
It retains original bundle bytes, Rust capability output, Python validation
summary, replay binding, executable hashes and build journal. Public and
archived bundle bytes were compared and rehashed after transfer.

## Remaining obligations and cleanup

CUDA, ROCm, Metal, native wgpu, WASI and MCU physical V2 conformance lanes remain
required, along with the physical Chrome/Firefox/Safari matrix, full performance
traces and independent final replay. No GPU lane, model, cloud, OCI scan or
deployment was run. CPU success does not substitute for those targets or
establish SOTA model quality. A new source revision must regenerate its own
source-bound receipts, not relabel this bundle.

The full v1.1 goal also still requires serving empirical latency/resource/fault
evidence, PyTorch/HF lifecycle/PTQ/refinement/distributed qualification,
whole-model ONNX and complete packaging/tutorial/Colab evidence, authorized
recipe freeze and Qwen language/MTP quality/runtime/reproduction, production
deployment/security qualification, audited model zoo/community deliverables,
independent clearance, signing and explicit human activation/publication.

Per-run scratch results were moved into the durable project archive; the
byte-identical duplicate replay bundle was removed. The managed source checkout
could not be archived because the app reports it is protected by a pinned
task/workspace. It was preserved, not manually removed. The build unit reached
terminal completion and unloaded. The shared build cache was reused without
cleaning. Unrelated edits, global installation, current GPU workload and August
campaign artifacts were preserved.

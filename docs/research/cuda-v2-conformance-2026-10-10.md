# Frozen V2 CUDA conformance — 2026-10-10

Executed clean source:
`172a37f2d66c7e0dae142337d35ac407bc099758`.
This is native CUDA semantic execution and physical-device binding evidence,
not the complete seven-backend qualification or final-release approval.

## Observed result

The RTX 4090 executed all 36 frozen V2 operations and 117 cases: 72 successful
executions and 45 expected errors. Two executions produced byte-identical
sealed bundles, checked with `cmp`. The separate Rust capability-table reader
admitted the bundle using strict `ReleaseCandidate` policy; no dirty allowance
was used. Separate Python source-contract and CUDA bundle validation passed.
An independent `nvidia-smi` query matched the complete physical identity:

`cuda:0:GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351:NVIDIA GeForce RTX 4090`.

The embedded build is
`tritium-cuda@1.1.0-rc.2+source-git:172a37f2d66c7e0dae142337d35ac407bc099758`.
The isolated checkout was clean before and after native build/test and before
bundle validation. Compilation explicitly unset check-only mode, injected
source identity and Git environment redirects; it used real NVCC kernels.
The observed driver was 615.71.09, NVCC 13.4.92, compute capability 8.9.

The full native CUDA portable-training suite passed 3/3 with no ignored or
filtered tests in 0.50s on the clean executed source. The two Python backend
qualification/admission suites passed 5/5 in 0.029s. Build/test timings are
observations, not performance qualification.

The immutable [sealed bundle](evidence/cuda-v2-172a37f2/ac7c4e1800116341d301641e62fb19c256639eb0b98ea12817ac241a9c3131ce.json)
has BLAKE3
`ac7c4e1800116341d301641e62fb19c256639eb0b98ea12817ac241a9c3131ce`
and SHA-256
`4d6f718d31fc79ac800e8164824b42b529ed2b20272197e49e39cef1b0557538`.
Reported peak resident bytes are 4192, peak scratch bytes 132032 and host
transfers zero. These are small-vector backend counters, not whole-process
RSS, Qwen memory/compression or independent transfer-profile measurements.

## Reproduction

Use an exact clean checkout of the executed source, real CUDA hardware and
the native toolkit. The observed build reused the SSD Cargo cache, two jobs,
`RUSTFLAGS=-Dwarnings` and a bounded owned systemd user job.

```bash
cargo build --locked -p tritium-cuda -p tritium-testkit \
  --features tritium-cuda/cuda \
  --example seal_cuda_training_receipts --example training_capability_table
cargo test --locked -p tritium-cuda --features cuda \
  --test portable_training -- --nocapture
timeout 60 "$CARGO_TARGET_DIR/debug/examples/seal_cuda_training_receipts" \
  --schema v2 "$OUTPUT/receipts" 0
timeout 60 "$CARGO_TARGET_DIR/debug/examples/training_capability_table" \
  --schema v2 \
  "ac7c4e1800116341d301641e62fb19c256639eb0b98ea12817ac241a9c3131ce=$OUTPUT/receipts/ac7c4e1800116341d301641e62fb19c256639eb0b98ea12817ac241a9c3131ce.json"
nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap --format=csv,noheader
```

Load `scripts/verify-training-backend-receipt.py`, call `source_contract(repo)`
and `validate_bundle("cuda", bundle, executed_source, operations, vectors)`.
Compare its returned physical identity with `cuda:<index>:<driver UUID>:<name>`
from the separate driver query, and compare two independent sealer outputs.
Do not use the all-seven aggregator with invented or partial backend inventory.
Different hardware or source may correctly produce a different bundle digest.

Original bytes, driver validation, generated capability table, executable
hashes and native build/test journal are retained at
`/home/brianklam/Projects/Tritium/archive/verification/cuda-manifest-172a37f2-20261010/`.
The earlier UUID-deficient bundle is preserved separately as diagnostic-only
evidence at `cuda-pre-repair-b7b03ace-20261010/` in that archive parent.

## Remaining obligations and cleanup

CPU's earlier bundle is bound to `b7b03ace`, not this source. Any common-source
candidate must regenerate its own CPU and CUDA evidence. ROCm, Metal, native
wgpu, WASI and MCU physical lanes, the browser matrix and all-target performance
traces remain required. No transfer/global-sync profiler, model-quality run,
cloud, deployment or release-signing action occurred here.

The full goal still requires serving empirical evidence; PyTorch/HF lifecycle,
PTQ/refinement and distributed qualification; whole-model ONNX and complete
packaging/tutorial/Colab evidence; authorized recipe freeze and Qwen language/
MTP quality, physical bytes, runtime/memory and reproduction; production
security/deployment qualification; audited model zoo/community; independent
clearance, signing and explicit human activation/publication.

Per-run scratch results were moved to the durable archive and the identical
replay copy removed. No August campaign scratch, model snapshot, unrelated
edits, active workload or shared build cache was deleted. The existing managed
source worktree remains protected by its pinned task/workspace; no manual
archival bypass was used.

# Common-source local V2 backend conformance — 2026-10-10

Executed clean source:
`38e0f2411d423bae909d44f81696d83c92bea9c9`.
This records native semantic execution on three backend families and four
configurations, not all-seven qualification, performance or release approval.

## Executed configurations

Every configuration executed all 36 frozen V2 operations and 117 cases:
72 successful executions and 45 expected errors. A second execution of each
sealer produced byte-identical bytes, checked with `cmp`. The producer grades
actual output buffers and receipts before sealing; these are not serialized
expected-output fixtures.

| Configuration | Recorded physical identity | Immutable bundle |
|---|---|---|
| CPU / i9-14900K | Linux/x86_64, CPU model, 32 logical CPUs | [CPU](evidence/local-v2-38e0f241/cpu/42ddcee7891319d7b50eafd751ef712d63562172ee62fdb41f4813d123a5b58a.json) |
| CUDA / RTX 4090 | Ordinal, driver UUID and device name | [CUDA](evidence/local-v2-38e0f241/cuda/b1cab93da3a7453c18bcaa6d7f9be7972d26a293d3dcf0ce3efaabedf60ab636.json) |
| Native wgpu / RTX 4090 | Adapter name and DiscreteGpu class | [wgpu NVIDIA](evidence/local-v2-38e0f241/wgpu-nvidia/7b366dc67bf260c5301cb38ab46ebeb4b05333019dde85996672ab1272506cb8.json) |
| Native wgpu / Intel RPL-S | Adapter name and IntegratedGpu class | [wgpu Intel](evidence/local-v2-38e0f241/wgpu-intel/30ac7befe7fd1f3dc074224218741a066bf43ee496a45eb910bbb65ef6b34b5c.json) |

Each filename is its bundle's BLAKE3 identity. SHA-256 identities, in table order:

- `7992526a03df3ba339c0e498cb067fe3d700c3ff60c24245ce10eae419609790`.
- `c70ecd86f1d5b3092d3d57b662a868b55c75feef1a1f097e5d8e5e29316e5fa2`.
- `987629aa7dc04b9066d01598ccbf99f1f764fc1bedaaa018e0548ca93677c1d8`.
- `a589ab181c8723c519ef6ac171c9fe0d3cc324dae5cc59870321fadc4ca67abd`.

## Separate validation and identity limits

Strict Rust `ReleaseCandidate` admission passed for each bundle without
`--allow-dirty`. Separate Python `source_contract` and `validate_bundle`
checks passed at the same source. The CPU identity matched `/proc/cpuinfo`.
The CUDA identity matched a separate `nvidia-smi` query:
`GPU-1790118a-a6d7-4eaf-fcac-dcacac5f4351`, RTX 4090, driver 615.71.09.

Native wgpu used Vulkan. Explicit `TRITIUM_WGPU_ADAPTER` values selected each
GPU. Independent `vulkaninfo --summary` enumeration uniquely matched the
name/class pairs on this machine: NVIDIA vendor/device `0x10de/0x2684`, driver
615.71.09; Intel `0x8086/0xa780`, Mesa 26.2.4-arch3.1. Vulkan reported NVIDIA
UUID `1790118a-a6d7-4eaf-fcac-dcacac5f4351` and Intel UUID
`868080a7-0400-0000-0002-000000000000`.

Those UUIDs are independently observed context, **not embedded wgpu receipt
identity**. The implementation currently records only adapter name/class.
This bounded local association does not prove unique-instance binding on a
machine with identically named adapters and does not inherit CUDA's UUID claim.

Combining both wgpu bundles in one generated capability table correctly failed
with `DuplicateBackend` for `wgpu.portable.v1:wgpu`. That existing one-row-per-
logical-backend rule was preserved. Both bundles were admitted separately;
the three-family local table uses the NVIDIA wgpu bundle only. The all-seven
release aggregator was not run against a fabricated/partial inventory.

## Tests and reproduction

At the exact clean source, adapter-selected wgpu integration runs passed 1/1
on NVIDIA (1.58s) and Intel (0.54s), with zero skipped/ignored/filtered tests.
The non-skipping sealers independently proved complete execution. The three
CPU integration suites passed 13/13; CUDA passed 3/3 (0.70s); the two Python
backend admission/qualification suites passed 5/5 (5.989s).

Native builds unset `TRITIUM_SOURCE_ID`, `TRITIUM_CHECK_ONLY` and Git environment
redirects, used two Cargo jobs, the existing SSD cache and
`RUSTFLAGS=-Dwarnings`. The checkout was clean before/after build/test and
before separate bundle validation. No GPU workload was stopped. Build/test
durations under shared workstation load are not throughput evidence.

```bash
cargo build --locked -p tritium-wgpu -p tritium-testkit \
  --features tritium-wgpu/wgpu \
  --example seal_wgpu_training_receipts --example training_capability_table
TRITIUM_WGPU_ADAPTER='NVIDIA GeForce RTX 4090' \
  cargo test --locked -p tritium-wgpu --features wgpu \
  --test portable_training -- --nocapture
TRITIUM_WGPU_ADAPTER='Intel(R) Graphics (RPL-S)' \
  timeout 120 "$CARGO_TARGET_DIR/debug/examples/seal_wgpu_training_receipts" \
  "$OUTPUT/intel/receipts"
```

Build the CPU/CUDA sealers using the commands in the earlier CPU and CUDA
execution records, but from this exact clean source; use `--schema v2`.
Run `training_capability_table --schema v2 DIGEST=PATH` separately for each
wgpu bundle. Call the Python verifier's `source_contract(repo)` and
`validate_bundle(family, path, executed_source, operations, vectors)`, then
compare live driver identities and original/replay bytes. Different source
or hardware can correctly produce different bundle digests.

## Remaining gates and cleanup

All four report 4192 peak resident bytes, 132032 peak scratch bytes and zero
host transfers. These are tiny-vector backend counters, not whole-process
RSS, model compression or independent steady-state transfer/sync profiling.
ROCm, Metal, WASI and MCU physical lanes remain required, as do all-target
performance traces and real Chrome/Firefox/Safari training/fault evidence.
Native Vulkan is not browser WebGPU. A new candidate source requires new
source-bound bundles; these bytes are not relabeled for later commits.

The full release still requires serving empirical evidence; PyTorch/HF
lifecycle/refinement/distributed qualification; whole-model ONNX and complete
packaging/tutorial/Colab evidence; authorized recipe freeze and Qwen language/
MTP quality, physical bytes, runtime/memory and reproduction; production
security/deployment; audited model zoo/community; independent clearance,
signing, explicit human activation and authorized publication.

Original bundles, driver enumeration, generated validators/tables, executable
hashes and native build/test logs are retained at
`/home/brianklam/Projects/Tritium/archive/verification/local-backends-v2-38e0f241-20261010/`
(296 KiB). Public/archive bytes were compared after transfer. Verified replay
duplicates were removed and owned scratch moved to that durable archive.
No August scratch, model snapshot, foreign edits or active workload was removed.
No new worktree or branch was created; the existing 58 MiB managed source
checkout remains protected by its pinned task/workspace and was not manually
removed. The shared build cache was reused without cleaning.

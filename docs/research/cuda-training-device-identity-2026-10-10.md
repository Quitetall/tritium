# CUDA training receipt device identity — 2026-10-10

## Finding and repair

The clean-source native V2 sealer at
`b7b03ace4bfbd83f1e40b063f1f27e7cfe5d02b8` executed all 117 frozen cases,
but its receipt identity was only `cuda:0:NVIDIA GeForce RTX 4090`.
Strict Rust admission and Python structural validation accepted that bundle;
a separate comparison with the driver's `nvidia-smi` UUID failed twice.
Semantic/structural acceptance does not establish a unique physical binding.

The training adapter used `TernaryBackend::device_id()` when constructing
physical evidence. The existing `physical_device_id()` contract explicitly
requires the driver-reported UUID for CUDA release evidence. The repair uses
that accessor while leaving the logical backend selector
`cuda.portable.v1:cuda:0`, receipt schemas, vector bytes and admission rules
unchanged. This corrects the implementation of an existing contract, not a
new format or a relaxed gate.

## Executed regression

On the local RTX 4090, the added regression ran the real native adapter's
complete V2 corpus and failed before the repair:
`CUDA training receipts omitted the physical driver UUID` (0.38s).
After the accessor correction, the full portable CUDA integration suite passed
3/3 with no ignored or filtered tests (0.36s): V2, V3 and physical-UUID binding.
These red/green runs used an isolated development tree based on `5caee9f6`;
they are implementation evidence, not clean-source release qualification.
The regression also requires all 72 successful V2 executions to retain a
receipt, preventing an empty receipt inventory from passing UUID checks.

```bash
cargo test --locked -p tritium-cuda --features cuda \
  --test portable_training -- --nocapture
```

The observed native toolchain was NVCC 13.4.92, driver 615.71.09 and compute
capability 8.9. Builds used two Cargo jobs, the existing SSD cache and
`RUSTFLAGS=-Dwarnings`. Check-only mode and source-identity injection were
unset. No debug logging was added, and no other GPU workload was stopped.

## Evidence boundary and next gate

The pre-repair content-addressed bundle
`1a2b864662921f9930dd45c5f3b1a53cffb81e01e305cbb9d4a1e445fbddd59f`
remains diagnostic evidence, not qualified physical-device evidence. Its bytes
must not be edited or relabeled. Corrected receipts require new execution
from a clean repair commit, separate strict readers, independent driver UUID
comparison and byte-identical replay.

Even a corrected CUDA semantic bundle does not clear steady-state transfer or
synchronization profiling, throughput, all-seven physical backend admission,
the browser matrix, model quality/runtime, independent reproduction, signing
or human public-release authorization. All remain separate obligations.

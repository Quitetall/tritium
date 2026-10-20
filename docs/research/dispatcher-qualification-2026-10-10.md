# Current-source portable wheel and dispatcher qualification

Measured source: `98489c95560629f4bd43016420d1738f63aa91e1`.
Release binding: `1.1.0-rc.2`. Scope: retained dispatcher evidence, not full
release admission, Qwen quality or whole-model performance.

## Bounded manylinux build resources

Both manylinux builders now pass the caller's `CARGO_BUILD_JOBS` into Docker,
defaulting to two jobs. Previously the host limit did not reach the compiler
inside the container. The pinned image, toolchain, platform contract and
numerical implementation are unchanged.

The actual-shell regression initially failed six subcases (CPU/CUDA, each with
default, one and four jobs). External Git/rustup/nvcc/Docker/Python are stubbed;
this is command-wiring evidence, not wheel qualification. The second invocation
overlapped the repair and passed; the retained `build-jobs-red-repeat.log` name
is misleading and must not be counted as a second red run. The focused suite
passed nine tests in 11.436 seconds; the committed builder/qualification contract
suite passed 18 tests in 26.772 seconds. `actionlint` and normal commit/push
formatting, syntax and Clippy checks passed.

## Exact portable artifact

The clean measured source was built using the existing CUDA manylinux cache and
two compiler jobs. Native compilation completed in 45.11 seconds. Auditwheel
and `verify-wheel.py` accepted the resulting `manylinux_2_28_x86_64` artifact;
that structural check reported `install_smoke: false`. An actual isolated
installation and the qualifications below subsequently exercised its runtime.

- Name: `pytritium-1.1.0rc2-cp39-abi3-manylinux_2_28_x86_64.whl`.
- Size: `2957204` bytes.
- SHA256: `7d5b84b68d0cca2fe49feb6b7f764f9b7a5f40f1d8428177ba052e8083d2738d`.
- Compiled source identity: `source-git:98489c95560629f4bd43016420d1738f63aa91e1`.
- Compiled backends: CPU and CUDA.

The wheel was retained on durable disk before installation into an owned
scratch virtual environment. No global installation, model routing, publication
or running service was changed. The successful managed build and CUDA
qualification units subsequently unloaded; their missing handles were not
treated as grounds to restart completed work.

## Physical CUDA gate

Run `cuda-dispatch-98489c95-20261010` passed all eight frozen CUDA cases and
both dedicated sanitizer cases with exactly one zero-error summary. Coverage
includes compiled fp16/autocast master-cache behavior, warm native dispatch
without composite matmul/projection or host transfers, ragged fp16/fp32 tails,
non-finite semantics, mutation/storage invalidation and cross-stream owner
lifetime. The CUDA source fixture and JUnit results are retained with the
receipt, rather than inferred from a summary.

- Receipt: `sha256:1672622f5388a33c1dd9fc9e71aa58d009ee9dd4b9f64515e8af45525ac834d7`.
- Device: RTX 4090, capability 8.9, UUID `1790118a-a6d7-4eaf-fcac-dcacac5f4351`.
- Python: 3.14.7; Torch: 2.11.0+cu130.
- CUDA driver: 615.71.09; runtime: 13.0.
- Compute Sanitizer: 2026.3.0.0.
- Test-source Git blob: `6ae4a48f83f414ac2a09e112b3ff945a252eeba1`.

A separate verifier process accepted the receipt against the exact wheel and
original source objects. The repeated sanitizer-log digest across older runs
is the digest of the same short zero-error summary; the current tool version,
source and wheel bindings come from this run's receipt, not that digest alone.

## CPU wrapper-overhead gate

Run `cpu-overhead-98489c95-20261010` exercised the same installed CUDA wheel's
CPU backend from a source-free working directory on the i9-14900K. It pinned
logical CPU 0 and one Rayon/Torch/interop/OMP/MKL thread. The frozen policy is
unchanged: ten paired warmups, 31 alternating paired samples, 10,000 bootstrap
resamples, 95% upper bound and maximum ratio 1.05.

| Frozen case | Median wrapper/direct ratio | Bootstrap upper ratio |
| --- | ---: | ---: |
| Decode forward | 1.004968 | 1.009868 |
| Decode backward | 1.003836 | 1.005572 |
| Microbatch forward | 1.003625 | 1.011041 |
| Microbatch backward | 1.000801 | 1.008522 |
| Prefill forward | 0.996031 | 1.008124 |
| Prefill backward | 0.997002 | 1.005369 |

All six cases passed. Receipt:
`sha256:19f85170e5e8658709ae6b3ec8c9988369345fa58bad54cba4738dfb2b7c04e7`.
Raw trace SHA256:
`2daf1b867b3e94a4139f767d52a0a429ead6cd8ce9dc500d427d5dae4bed1ade`.
A separate verifier process accepted the retained receipt and recalculated
trace aggregation. These ratios measure adapter overhead against direct
execution, not Qwen tokens/second, CPU health or a 13600K-to-14900K speedup.

## Reverification and custody

Durable archive:
`/home/brianklam/Projects/Tritium/archive/verification/cuda-dispatch-98489c95-20261010`.
It retains the wheel, both receipts, raw CPU timings, CUDA fixture/JUnit/
sanitizer outputs, build/verification logs and original source snapshots.
The archive's separate stdlib verifier compares snapshots against externally
supplied original Git objects without importing the qualification Git helper.
Checksums establish custody; they do not create release admission.

From the source checkout, with `ARCHIVE` pointing to the directory above:

```sh
python3 -B scripts/verify-torch-dispatch-cuda-receipt.py \
  "$ARCHIVE/qualification/receipt.json" --repo "$PWD" \
  --source-revision 98489c95560629f4bd43016420d1738f63aa91e1 \
  --release 1.1.0-rc.2 \
  --wheel "$ARCHIVE/wheels/pytritium-1.1.0rc2-cp39-abi3-manylinux_2_28_x86_64.whl"
python3 -B scripts/verify-torch-dispatch-overhead-receipt.py \
  "$ARCHIVE/cpu-overhead/receipt.json" \
  --source-revision 98489c95560629f4bd43016420d1738f63aa91e1 \
  --release 1.1.0-rc.2 \
  --wheel "$ARCHIVE/wheels/pytritium-1.1.0rc2-cp39-abi3-manylinux_2_28_x86_64.whl"
```

Hosted jobs for the measured source completed successfully: CI `38057271036`,
wheels `38057271075`, CodeQL `38057271024`, docs `38057271215` and CPU capstone
`38057271091`. These are source/CI results, not model or full-release gates.

## Remaining obligations

Final candidate admission must bind source-frozen package bytes and all required
receipts; this development artifact is not a signed or published candidate.
The final CPU-only portable package, two-device distributed training and the
other native/backend/performance matrix still require their own evidence.

Stage-7 also has a software blocker, not only a compute backlog: the documented
measurement and baseline/refinement command names are placeholders rather than
tracked full-coverage runners. Orchestration alone, or a SmolLM2 smoke, does not
satisfy the frozen S2KF recipe grid. The reference fitter currently admits G128
only, while the package format already represents G64/G128/G256. Supporting
those fitting alternatives needs the existing preregistered conformance and
measurement gates, not a duplicate wire-format proposal.

Measured observability admission (ADR 0053 adoption pending), representative
PTQ/refinement and baselines, Qwen language/MTP quality/bytes/runtime/reproduction,
physical multi-device/backend/browser performance, serving fault/residency/
concurrency, production security/deployment, final packaging/zoo/community,
second-machine/operator clearance and explicit human release activation remain
open. No paid compute or large capture/fitting was started.

# Distributed installed-candidate admission

## Repair and scope

Source `d92f71248c710a0b096d196ada0f989a82fc13a9` binds every distributed
worker's installed native source, release, wheel archive and package bytes
before the worker's CUDA queries, NCCL initialization or model construction. The public
qualifier forwards those identities to each rank and forces installation of
the named wheel, even when a same-version distribution is already visible.

The frozen worker is a checkout-owned script, not a wheel member. Its source
is bound by the outer qualifier's existing exact-clean-revision check; shared
admission binds the installed package without incorrectly requiring wheel
ownership of the external script. The qualifier's public arguments, v1
receipt/fragments, 127,943,680-parameter workload, training/physical-device
conditions and scaling thresholds are unchanged. This enforces ADR 0033's
existing exact-candidate contract, not a new training or release gate.

## Red loop and causes

Baseline installed package: official CPU wheel for source
`5ca0c91c28930197cffb35949fe065ec155db19c`, SHA-256
`dbedb830cf88bf5fc0339cba0f3c3f2e4969c528b76b6402b4f6ff732c15996d`.
Three main-preflight call-site tests failed twice (0.19/0.14 seconds): wrong
source, wrong release and opaque non-wheel bytes reached a forbidden hardware
query. The initial red loop supplied parsed namespaces to isolate preflight,
not a claim that the preceding CLI accepted flags it did not yet expose.
Real CLI argument parsing is covered by the repaired regression suite.

The previous worker checked only installed-distribution ownership; the launcher
did not forward candidate identity. A separate actual pip probe requesting the
retained `65d3b3c...` wheel returned success but kept the already installed
`5ca0c91c...` native source because both have version `1.1.0rc2`. Repeating
with `--force-reinstall` installed exact native source `65d3b3c...`. A launcher
argv regression also failed before repair on the missing reinstall flag.

## Local exact-candidate checks

Fresh local wheel: 10,582,923 bytes, SHA-256
`358a323c5714c3c7095d80a66e8208ff590b548c72488030ede3e4d9049b0bdc`.
Its native stamp is exact source `d92f7124...`, release `1.1.0rc2`.
The local `linux_x86_64` tag is not publishable on PyPI. CPython 3.14.7,
Torch 2.11.0+cu130, Transformers 5.5.3, Accelerate 1.14.0 and existing
framework dependencies were reused in an
isolated venv; this is not a clean-dependency/compiler-free environment.

Thirty-six focused distributed/HF/estimator admission tests pass against that
wheel. Eight launcher/receipt unit tests, actionlint, staged formatting/Python
compilation and normal pre-push gates pass without bypasses. The eight new
preflight tests cover wrong source/release/wheel through namespace and real CLI
entry, plus valid-candidate physical guards in both DDP and FSDP modes.
Installed-wheel CI pins Accelerate and executes this test file; it does not
silently substitute a skipped accelerator-dependent preflight suite.

Five separate real CLI processes reject three invalid candidates and stop both
valid-candidate modes at the two-physical-device guard. Actual two-rank torchrun
launches also stop both ranks at that guard in DDP and FSDP modes. The current
host has one RTX 4090. No fragment, checkpoint or qualification receipt is
published by these negative probes; they do not measure distributed training.

Three existing semantic regressions also pass against the fresh local candidate:
actual two-rank CPU/Gloo DDP step/checkpoint, CPU FSDP sharded-state resume and
safe HF export, and fresh-process CPU bf16 Accelerate training. The selected
tests exclude the CUDA qualification fixture, which labels a synthetic test
source rather than a candidate wheel. CPU semantic checks do not discharge the
distinct physical accelerator gate.

The official hosted manylinux wheel is 10,573,931 bytes, SHA-256
`dc3958d56bbf2afd57d1fde9d92681b4cc60276e86f1cfe151e1d0e0b8714a38`.
Artifact `11668998819` has ZIP SHA-256
`6262741346b6bea0a58b8b35cd7c357b2ac308f3517d7b0c5f07ba7acc6a37ec`.
Eight preflight tests and the same five real CLI/two two-rank rejection probes
also pass locally against that exact installed hosted wheel. These local
dependencies differ from CI's pinned Accelerate 1.10.0 runtime; hosted results
must be observed separately. A separate process validates wheel bytes,
source-bound probe results, both-rank stderr and absence of training artifacts.

Hosted exact-source workflow:
[38051204583](https://github.com/Quitetall/tritium/actions/runs/38051204583).
Installed-wheel job `114211059703` reports **180 passed, four skipped**;
source-free job `114211059619` also passed at that exact source. The complete
workflow subsequently passed, including all three platform wheels, ABI matrix
and pinned SmolLM2 job. Completed metadata is retained separately from the
initial running snapshot. CUDA is skipped, not qualified. A green workflow
does not turn one-GPU rejection or CPU semantic checks into accelerator
qualification or clear the full public release.

## Custody and remaining gates

Durable evidence destination:
`/home/brianklam/Projects/Tritium/archive/verification/distributed-binding-d92f7124-20261010`.
Retain exact candidate wheels, red/green logs, actual pip probe logs, real CLI
and two-rank stderr, runtime/device inventory and hosted metadata. Validate
retained evidence/checksums before removing the owned scratch/venv. Old August
caches, historical Qwen campaigns and shared build caches are outside cleanup.
Foreign staged diff identity remains
`c4dbfb4610a65b3c57fdf6ffc2d7c817f83ea53ad562c7d03420778382348e75`.

Actual two-distinct-GPU NCCL/fp16 DDP/FSDP checkpoint/RNG/no-transfer,
memory/throughput/scaling qualification remains open. CPU semantics and correct
one-GPU rejection cannot satisfy it. Source-bound diagnostic measurements,
final catalog qualification, representative PTQ/refinement/matched baselines,
flagship Qwen quality/runtime/reproduction and other full-release gates also
remain open. No release inventory or qualification receipt is invented here.

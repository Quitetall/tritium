# Whole-Qwen qualification: bind executed bytes to candidate archives

Status: local software regressions pass. No whole-Qwen qualification receipt,
model-quality result, runtime benchmark or release approval is produced here.
This implements the archive-binding obligation in ADR 0033 / plan 0051.

## Gap and resulting behavior

The producer previously supplied archive identity records plus caller-selected
unpacked directories. The installed worker checked the wheel bytes and native/
ONNX package-lineage agreement, but could not verify either directory against
the actual candidate archive. Agreement between two loaded models did not prove
that either was the archived release artifact.

The producer now derives both archive paths from the authenticated candidate
inventory and sends `--onnx-archive` and `--native-archive` to the installed
worker. Nested archive paths retain their exact inventory location rather than
being guessed from a basename. Receipt and execution-trace schemas, numeric
tolerances, model coverage and MTP promotion requirements remain unchanged.

The shipped `BoundBundle` verifier checks archive filename, byte length and
SHA-256, then compares every regular flat POSIX ustar member with the unpacked
directory. `.tar`, `.tar.zst` and `.tzst` are supported. Missing, extra,
duplicate, unknown or unsafe members, links, symlink traversal, FIFOs, truncated
streams and nonzero trailing payload fail before model execution. Member/file
comparison uses chunks of at most 1 MiB. Archive/member/uncompressed physical
bounds match the release bundle inventory's existing bounds; zstd decoding has
a 128 MiB window limit. Decoder buffers and verification streams close before
inference; no complete model copy or persistent dense weight shadow is created
by this check.

No-follow directory/file handles stay open through execution. File identity,
size, mutation timestamps and link count are rechecked before a successful
result leaves the context. Linux inode watches also detect parent directories
that are moved, deleted or unmounted. A queue event, including loss of a watch
or overflow, fails closed. Ordinary sibling-file writes in shared ancestors
do not invalidate a run. The current custody implementation requires Linux
no-follow handles, `/proc/self/fd` and inotify; other physical platforms remain
unqualified for this worker.

The existing negative-fault tests retain hardlinked weights to avoid copying a
potentially tens-of-gigabytes arena. Their links are now prepared before the
custody snapshot and removed after its final check. Copied graph/manifest files
are rechecked against the bound source before intentional corruption; linked
weight identity is also checked. Thus the fault setup's own ctime/link-count
changes cannot conceal a mutation during inference or cause a false rejection.
Zstd archive qualification requires the Python `zstandard` package in the
offline installed-wheel environment; local compressed-archive tests used
version 0.25.0. Missing dependency and corrupt compressed streams fail closed.

## Regression evidence

The producer regression first failed with
`AssertionError: '--onnx-archive' not found`. A second regression exposed a
transient ancestor swap after the initial descriptor implementation:
`Failed: DID NOT RAISE <class '...BundleBindingError'>`. The directory-move
watch closes that reproduced loophole.

Final focused commands:

```sh
TMPDIR=/mnt/4tb/tmp PYTHONPATH=crates/tritium-py/python timeout 90 python -m pytest -q \
  crates/tritium-py/tests/test_bundle_binding.py \
  crates/tritium-py/tests/test_qualify_onnx_worker.py \
  --basetemp=/mnt/4tb/tmp/tritium-bundle-binding

TMPDIR=/mnt/4tb/tmp timeout 90 python -m unittest \
  scripts.tests.test_qualify_onnx_inference \
  scripts.tests.test_verify_onnx_inference_receipt -q
```

These pass 31 Python worker/binding tests and 7 producer/verifier tests locally.
They cover uncompressed/compressed archive equality, a mismatch in the last
chunk of a multi-megabyte member, path and byte substitutions, invalid topology,
mutation-and-restoration, parent swaps, public-worker rejection before model
load, retained custody through execution, shared sibling writes and fault-link
cleanup. The CPU wheel lane now runs these worker/binding regressions against
the installed candidate wheel with pinned binary `zstandard==0.25.0` and
source fallback disabled. Workflow/source-identity tests passed 9 tests and
`actionlint .github/workflows/wheels.yml` passed locally. The hosted lane must
still execute the new commit. Public execution is intercepted in the small fixtures; these are
software tests, not Qwen inference evidence. Bytecode compilation and
`git diff --check` also pass. Per-run fixture directories are removed.

## Binding work still open

- Obtain and independently admit the production Qwen MTP oracle; the worker
  still refuses a model with `mtp_verified=False` or no `reference_mtp`.
- Execute this worker from the final installed wheel against the real admitted
  Qwen language/MTP model and ONNX archives in a source/compiler-free environment.
- Admit the resulting execution trace through the independent receipt verifier
  and exact final candidate inventory.
- Complete the other plan-0044 gates: PTQ/refinement/quality/runtime and physical
  bytes, algorithm/distributed evidence, physical backend/browser and CUDA/Colab
  matrices, deployment, audited zoo, second-machine reproduction, signed release
  assembly and authorized activation.

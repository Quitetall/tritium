# Hugging Face installed-candidate provenance repair

## Scope and result

Source `e0d31ce898c96d7f74e90152976578cfba15c22d` repairs the candidate
binding in the HF QAT lifecycle and whole-model hard-export producers and
installed replay validators. Native source identity, installed release,
wheel ZIP/RECORD inventory, installed payload hashes, and ownership of the
package/native/binding-helper origins are checked before model execution.
Receipt schemas and numerical acceptance thresholds are unchanged.

This is tiny-model frontend lifecycle evidence, not pretrained-model quality,
native accelerated HF training, distributed execution, Qwen language/MTP
qualification, SOTA performance, or full release approval. Other qualification
producers, including the QAT tutorial and SmolLM2 wrapper, still need their own
candidate-binding audit. No large-model campaign, paid compute, publication, or
human activation was started by this repair.

## Red-capable evidence

The original installed CPU wheel came from source
`52b56ce9b4b191326764e2d1d4821d7a7dcd5ead`, wheel SHA-256
`14561f830ca2795946bdbc558107fadcd443efd74f851239d9a5c3ae181a94ee`.
Its HF producer and installed validator both accepted:

- a caller source of forty `a` characters despite a different native stamp;
- opaque `not a wheel archive` bytes as the candidate `.whl`.

The reproducer trained the tiny tied Llama, wrote through the official receipt
writer, replayed the installed validator, and asserted that admission must
reject the bad identity. Both modes failed that assertion (exit 1). Six public
producer call-site regressions and four public replay call-site regressions
also failed before the repair by reaching forbidden model creation/loading.
An initial missing-receipt harness invocation was corrected; it was not used
as evidence of the provenance bug.

The regression suite covers actual producer/replay admission, plus explicitly
synthetic wheel/RECORD integrity fixtures. Fixture-created receipts are not
release qualification. Existing portable-tree tests explicitly mock candidate
admission rather than presenting synthetic wheel bytes as installed evidence.

## Clean candidate execution

Local host: CPython 3.14.7, Torch 2.11.0+cu130, Transformers 5.5.3;
CPU-only execution with CUDA hidden, one Torch/BLAS thread, network-offline
HF loading, isolated `python -I` and no Tritium source overlay. The temporary
venv reused existing framework dependencies, so it is not the compiler-free
or fresh-dependency qualification environment.

Two newly installed wheels stamped with exact source `e0d31ce8...` passed
producer execution, separately invoked installed replay, and the 24-test HF
provenance/lifecycle/export suite:

| Candidate | Bytes | SHA-256 | Tests |
| --- | ---: | --- | --- |
| Local `linux_x86_64` | 10,582,203 | `18858b08dfa29120e24530e7c0cae389dfd4dff8265b43ba585340766e44a9a1` | 24 passed |
| Hosted `manylinux_2_28_x86_64` | 10,573,209 | `f2a3039839f8a81a398e5cc41a92f66979476f95f95efba51778e3c05a9d79b0` | 24 passed |

The local Linux tag is not a publishable PyPI wheel. Both wheel structures and
isolated, dependency-free native install smoke checks passed. An initial
local verifier invocation requested a compatibility receipt without its
required target ID and exited 2; the successful local rerun did not claim
manylinux compatibility. The hosted-wheel rerun used the canonical Linux CPU
target and exact source revision.

Hosted-wheel local receipt identities:

- `tritium.hf-lifecycle.v1`:
  `sha256:b6f3bd70fc87e81816792426247237f0e2a8bec4ec6c6040880f335268769a83`.
  One optimizer step, eight unique converted weights, tied weights preserved,
  exact save/AutoModel-reload logits; checkpoint tree 15,990 bytes.
- `tritium.hf-export-reload.v1`:
  `sha256:c9b089f7ea39a5915d921e2e4cd286bccacbe79e8c1df5ce4fbcdfd54c0ebe9a`.
  Eight unique converted weights, tied packed weights preserved, no dense
  weight shadows, exact hard-export/reload logits and generation; artifact
  tree 18,092 bytes, state 6,112 bytes/35 tensors.

Replays omitting the external wheel also passed installed RECORD/native-source
integrity checks. Such a replay does not independently establish archive-byte
identity; release qualification uses the explicit candidate wheel.

The original wrong-source and opaque-wheel reproducer modes now reject before
qualification. Temporary developer-overlay test results were retained but are
not promoted to clean installed-wheel qualification.

## Hosted and local gates

[Wheel workflow 38045016770](https://github.com/Quitetall/tritium/actions/runs/38045016770)
binds source `e0d31ce898c96d7f74e90152976578cfba15c22d`.
Linux artifact `11666722720` has ZIP SHA-256
`3eac9e0b83f2df71b088a13dcf6fa0d1ddddc0c218fa483dfabadc4b7ea13b33`.
At the recorded observation, the Linux/macOS/Windows wheel builds, installed
Torch/Python 3.13 job `114193300520`, and compiler/source-free Python 3.13 job
`114193300579` succeeded. The latter runs and validates both HF receipt kinds.
The CUDA job was skipped, not qualified; remaining workflow jobs were still
running. A later observation confirmed the complete workflow succeeded. Its
installed-wheel pytest invocation included all three HF test files and reported
143 passed/4 skipped across the combined PTQ/ONNX/HF suite. Workflow success
does not imply CUDA qualification or full release completion.

The source-free artifact `11666947477` has ZIP SHA-256
`678fd248c481a5ed3644067eee6e3e8eeb570950e5767e7f2ed10ea5084a0945`.
Its CPU Torch 2.11.0/Transformers 5.5.3 HF lifecycle receipt is
`sha256:b5d2f120f56cf9c8e93976fa68bbf9f1e8384aac83cabcd753ab4f14293543fd`;
hard-export receipt is
`sha256:84e3d191783f45e3a309076958c200b798e0582c534b38e3fb86f87bf45b0df8`.
Both bind the same `f2a30398...` wheel and exact `e0d31ce8...` source.
Their retained trees independently passed local portable byte/identity
validation against that wheel. This portable check is not an installed replay
of the CI runtime on a differently located/versioned local installation.
The initial local portable-check command used a nonexistent module name and
failed import before validation; the corrected import from `tutorial_receipt`
passed. That harness failure remains retained separately.

Local checks: eight workflow/source-binding unit tests, actionlint for the
modified workflow, staged-tree format/Python compilation hooks, and normal
pre-push all-feature Clippy/platform/workflow checks passed. The exact source
commit was pushed without bypassing hooks. Foreign staged contents retained
SHA-256 `c4dbfb4610a65b3c57fdf6ffc2d7c817f83ea53ad562c7d03420778382348e75`.

## Replay commands and custody

Install the retained candidate wheel without dependency/source-build fallback,
then use a new output directory and the recorded source/release/run identity:

```sh
python -I -m tritium.torch.hf_lifecycle --output-dir OUT \
  --wheel-artifact WHEEL --source-revision e0d31ce898c96d7f74e90152976578cfba15c22d \
  --release 1.1.0-rc.2 --run-id NEW_RUN
python -I -m tritium.torch.hf_lifecycle --check-receipt OUT/receipt.json \
  --wheel-artifact WHEEL --source-revision e0d31ce898c96d7f74e90152976578cfba15c22d \
  --release 1.1.0-rc.2
```

Repeat with `tritium.torch.hf_export_lifecycle` and a distinct output directory.
Installed replay binds the receipt's original installation path; generating a
new independent run is not rewriting or relocating its immutable provenance.

Durable local custody:
`/home/brianklam/Projects/Tritium/archive/verification/hf-provenance-e0d31ce8-20261010`.
It retains both candidate wheels, original hosted metadata/SBOM, tiny checkpoint
payloads, receipt identities, raw red/green logs, and command/gate observations.
Generated model payloads and wheels are not committed to Git. The ephemeral
venv, pytest directories, and reproducer are removed after checksum verification.

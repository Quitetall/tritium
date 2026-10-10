# Hosted SmolLM2 tutorial and CPU wheel matrix

Status: measured hosted CPU tutorial and ABI3 evidence; aggregate v1.1 release
qualification remains open.

## Executed candidate

[Actions run 37985901639](https://github.com/Quitetall/tritium/actions/runs/37985901639)
executed source `70974aeaba2924d7d864f5f961dd91e421e97bc7` as
`1.1.0-rc.2`. Its pinned SmolLM2 PTQ/QAT CPU tutorial, source-free installed
tutorial, installed Torch lifecycle, three CPU wheel builds and ABI3 matrix all
passed. The CUDA wheel lane was skipped. CI, docs, CodeQL and capstone workflows
also passed at this source revision; these do not replace empirical release
receipts.

The tutorial runner reported AMD EPYC 9V45 96-Core Processor hardware with four
available logical CPUs, Python 3.13.16, Torch 2.11.0 CPU, Transformers 5.5.3,
ONNX 1.22.0, ORT 1.27.0 and ONNXScript 0.7.1. Its installed wheel was
`pytritium-1.1.0rc2-cp39-abi3-manylinux_2_28_x86_64.whl`, 10,560,923 bytes,
SHA-256 `5ec84fcb7c7e03b522e23560871f09368ac084d6cca04be76160a824a5ac3ccc`.
The pinned model was `HuggingFaceTB/SmolLM2-135M-Instruct` at
`12fd25f77366fa6b3b4b768ec3050bf629380bac`.

## Measured tutorial result

The public installed-wheel workflow completed PTQ calibration/conversion,
compact native checkpoint save/reload, token generation, ONNX export/replay,
QAT backward/update and optimizer checkpoint/resume in **98.794800962 seconds**,
excluding first model download. The receipt records 211 selected parameters,
61 preserved parameters and 272 resumed optimizer-state entries.

ONNX replay used `ORT_DISABLE_ALL` and unchanged `rtol=atol=1e-4`. Maximum
absolute error was `8.738040924072266e-5`; maximum normalized tolerance ratio
was `0.6461082696914673`. Thus this exact AMD-hosted candidate clears the
tutorial parity assertion that failed in earlier runs. The constant/reshape
type inference repair in `70974aea` followed the graph-local diagnosis in
[the diagnostic note](onnx-torch-runtime-parity-2026-10-09.md). This result
does not establish parity for every CPU, model or runtime configuration, nor
does it measure the overhead of FP64 projection accumulation in isolation.

Selected FP32 dense weight bytes were 537,919,488 and compact checkpoint bytes
were 30,824,451, a 17.45x ratio. This compares selected FP32 tensors with a
serialized checkpoint; it is not a BF16 baseline, full-model resident-memory
measurement or Qwen compression result. Zero-trit fraction was 0.5007834481.
The generated continuation repeats `interpre`; no language-quality or
near-lossless claim is earned by this execution test.

## Retained evidence and verification

The original [tutorial receipt](evidence/onnx-hosted-70974aea/receipt.json) has
SHA-256 `1679046742da2ee7692467438cc40f73e700218eca9305aa23795122b5b387e2`,
matching the digest printed by the hosted qualification producer. It came from
Actions artifact `11643900403`, whose archive identity is
`sha256:254026bc8abb06a30014c09fae1f8705f40a8a768fe46602e1aa2e360ee2ff7b`.
The receipt was retrieved with bounded ZIP byte-range reads; the complete
1,100,126,608-byte archive was not retained locally. The archive includes model
and optimizer payloads and remains available in the linked workflow.

The original [ABI3 matrix receipt](evidence/onnx-hosted-70974aea/abi3/python-abi3-39-plus.json)
and all 16 cell files are retained beside it. Its receipt ID is
`sha256:a5a6ce73e0a8e15e325bff0a3984b1bbcd79435d6730ded5e6b8c1d75e4ce6ef`.
Cells cover CPython 3.9–3.14 on Linux/Windows and 3.11–3.14 on macOS arm64.
Local verification validated the tutorial receipt with
`scripts/qualify-smollm2-release-tutorial.py::validate_receipt`, validated the
matrix with `scripts/aggregate-wheel-smoke.py::validate_receipt`, rehashed every
cell against that matrix, and applied its `_validate_cell` contract to each
original cell. Windows CRLF bytes are preserved through local Git attributes.
This verifies retained evidence integrity; the physical execution was performed
by the hosted workflow. Candidate assembly must still rehash the actual wheels
and bind these receipts to its exact artifact inventory.

The wheel workflow now uploads `smollm2-cpu-tutorial-receipt` separately from
the full replay archive. Future receipt inspection can retrieve the small file
without downloading model payloads. Workflow/source-identity suites passed
9 tests, and `actionlint .github/workflows/wheels.yml` passed. The whole-Qwen
ONNX producer/verifier suites also passed 6 tests; no whole-Qwen run occurred.

## Remaining release work

1. Execute the CUDA/Colab and physical browser/backend matrices against the
   final candidate; this hosted CPU result cannot satisfy those cells.
2. Bind the installed whole-Qwen worker's unpacked model and ONNX directories
   to the exact candidate archives. Record identities alone do not prove which
   extracted bytes executed.
3. Obtain and independently admit production Qwen MTP oracle evidence. The
   compiled authorization ledger currently contains only a small fixture, and
   `QwenModel.mtp_verified` remains false.
4. Finish the Qwen PTQ/refined candidates and their language/MTP quality,
   runtime, physical-byte and reproduction gates.
5. Finish deployment, audited-zoo, signed candidate assembly and independent
   release closure. Public activation still requires owner authorization.

Plan 0051 and parent plan 0044 remain in progress. Temporary downloaded evidence
used during this check is removed after retaining the small original receipts.

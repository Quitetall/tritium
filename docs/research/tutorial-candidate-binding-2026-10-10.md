# Installed tutorial candidate binding and pinned SmolLM2 execution

## Result and boundary

Public source `8cf96e6bf39bad385353daf7d81cfa945e9073d4` extends the HF
candidate-admission repair to the installed QAT tutorial, its installed replay,
and the pinned SmolLM2 qualifier. A shared internal guard verifies the native
source stamp, installed release, wheel ZIP/RECORD inventory, installed payload
hashes, and ownership of executing entry points before model execution. The HF
hard-export caller is explicitly included in the executing-file inventory.
No public API, receipt schema, numerical tolerance, or frozen time gate changed.

The guard has no Transformers dependency. A real Torch-only tutorial completed
training, optimizer checkpoint/resume, hard export and installed replay while
Transformers imports were blocked. A subprocess regression preserves that
optional-dependency boundary. This follow-up test and this record are later
than the measured source; they do not relabel the retained wheels or receipts.

This is candidate-provenance and small-model workflow evidence, not Qwen
language/MTP quality, SOTA performance, distributed training, GPU qualification,
an audited model zoo, or independent public-release clearance.

## Reproduction and checks

The baseline installed wheel was the retained official CPU candidate for
`e0d31ce898c96d7f74e90152976578cfba15c22d`. Seven of eight new admission
cases failed before repair, twice: a wrong source or opaque non-wheel payload
reached forbidden model creation/loading in the producer, replay, or SmolLM2
wrapper. The existing SmolLM2 release-version check already passed. Failures
are retained in `red-provenance.log` and `red-repeat.log`; the initial missing
pytest setup is a separate harness failure, not provenance evidence.

Fresh local native-wheel checks reported 38 passed and 9 subtests passed. The
additional Transformers-optional import regression passed separately. Against
the retained official hosted wheel, the final focused suite reported **39
passed and 9 subtests passed**, covering:

```sh
TRITIUM_TEST_INSTALLED_WHEEL=1 python -I -m pytest -q \
  crates/tritium-py/tests/test_tutorial_candidate_provenance.py \
  crates/tritium-py/tests/test_tutorial_qat.py \
  crates/tritium-py/tests/test_hf_candidate_provenance.py \
  crates/tritium-py/tests/test_hf_lifecycle_receipt.py \
  crates/tritium-py/tests/test_hf_export_lifecycle.py
```

These local venvs use installed Tritium payloads without a Tritium source
overlay, but reuse existing framework dependencies. They are not fresh-dependency
or compiler-free environments. One hosted-install attempt skipped an already
visible same-version package; a force-reinstall established an actual secondary
venv installation. An initial pytest invocation omitted installed-wheel mode
and failed during source-package binding, before tests. Both failures are
retained separately; neither is counted as a passing run.

Thirteen workflow/tutorial/SmolLM2 contract unit tests and actionlint passed.
The source commit passed normal staged-tree hooks and pre-push gates and was
pushed without bypasses. The user's unrelated staged changes retained binary
diff SHA-256 `c4dbfb4610a65b3c57fdf6ffc2d7c817f83ea53ad562c7d03420778382348e75`.

## Candidate identities and hosted execution

| Candidate wheel | Bytes | SHA-256 |
| --- | ---: | --- |
| Local `linux_x86_64` | 10,582,695 | `1cc9e13f8100c4846f1e3b6d94d83ecfed12b302e8654abca324085ea339e222` |
| Hosted `manylinux_2_28_x86_64` | 10,573,830 | `0f1d5e1be1ed25c904e03ca366b02ebdef87ac68780fa85dab2caae21b9c3d80` |

Both are `pytritium-1.1.0rc2-cp39-abi3` and stamp exact source `8cf96e6b...`.
The local Linux tag is not a publishable PyPI wheel.

[Workflow 38046697193](https://github.com/Quitetall/tritium/actions/runs/38046697193)
completed successfully at that exact source. The installed Torch/Python 3.13
job `114198168742` reported **152 passed, 4 skipped** across its broader suite.
Source/compiler-free job `114198168661` executed and replayed the QAT, HF
lifecycle and HF hard-export fixtures. The Python abi3 admission job passed;
the CUDA job was **skipped, not qualified**.

The retained source-free artifact is `11668000955`, ZIP SHA-256
`a19b92b176cc65dc195bb3addbeb706cc93acc265807d2a47b3bd1d3545f2aa5`.
Its tutorial, HF lifecycle and HF export receipt identities respectively are:

- `sha256:a254fc75bfc5d2977b8fb3fdc384d1e0fc36bd01d58cb8b25e7026944c7374f7`
- `sha256:28b65e2ac52f9fcfa3120fadcfe2ec6fe396e7b0eb7dca293dafb1ee60d83d41`
- `sha256:df6b89475d2b6e6056e40df9eb91151b8c30d2fed8f9803d5f15d5873d3e8e4a`

Fresh local execution and separately invoked installed replay also passed for
all three producer kinds against the official hosted wheel. Eight retained
trees pass separate portable byte/source/release/wheel validation: those three
local trees, the three source-free trees, the hosted functional QAT tree, and
the local Torch-only QAT tree. Portable validation is not a replay of a moved
CI installation. Receipts are preserved unchanged, including original origins.

## Pinned SmolLM2 workflow, not quality qualification

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, immutable model revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. The local CPU attempt used its
existing offline cache, the clean local wheel, an eight-CPU-equivalent quota,
12 GiB memory cap, no swap allocation, and a 600-second process deadline.

The actual run completed PTQ calibration/conversion, native HF checkpoint
roundtrip, generation, real ONNX export and held-out replay, QAT step, and
optimizer checkpoint/resume in **180.574 seconds excluding download**. Hosted
CPU job `114198168761` completed the same frozen workflow in **165.805 seconds**.
The frozen 300-second limit and ONNX tolerances `rtol=atol=1e-4` were unchanged.
Systemd's retained journal reports 188.424 seconds total wall time, 843.879
seconds CPU time and 4.7G peak memory for the local process. The collected unit
no longer exposes those resource fields; the journal, not a fabricated current
unit reading, is the evidence. An earlier launch failed before Python because
`/usr/bin/time` was unavailable; the successful retry omitted that wrapper.

Both runs converted 211 selected parameter groups, retained 272 optimizer state
entries, and measured 30,824,451 compact checkpoint bytes versus 537,919,488
selected dense weight bytes. This **17.45x selected-weight/checkpoint ratio is
not a whole-model, training-memory, or resident-memory compression claim**.
The QAT checkpoint deliberately retains latent weights and optimizer state.

Local receipt SHA-256 is
`24e0a502b21aed97851c8b52dbc1c433fbf38d6f26fb674856124c4c5656166b`;
hosted receipt SHA-256 is
`304365a416fa98d22db55b2b2c64b76b746a6aa5984bf723aee926bedffb08b9`.
Both receipts pass the separate qualifier's field validation; the local native
checkpoint digest also matches. Full local output payloads are retained. Only
the lightweight hosted SmolLM2 receipt and logs were downloaded, so this record
does not claim local verification of its complete hosted model tree.

Generated continuation in both runs was repetitive (`interpre` repeated).
Successful execution and export parity do not demonstrate language quality.
Representative held-out quality and flagship-model qualification remain open.

## Durable custody and next gate

Local evidence is outside scratch at
`/home/brianklam/Projects/Tritium/archive/verification/tutorial-binding-8cf96e6b-20261010`.
It retains candidate wheels and hosted metadata/SBOM, original red/green and
harness-failure logs, actual tiny checkpoint/receipt trees, full local SmolLM2
outputs, hosted run/job/artifact metadata, and the portable validation script.
Archive checksums and relocated portable validation are checked before deleting
this task's venvs, pytest trees and scratch. Weights and wheels are not in Git.
The unapproved August caches and old campaign outputs are outside this cleanup.

Next cheap work: audit remaining candidate-binding producers (especially
observability and distributed/estimator wrappers) against actual bad-source and
bad-wheel call sites. Physical backend/browser, representative refinement,
Qwen quality/runtime/reproduction, production deployment/security, model-zoo,
and independent final-release authorization gates remain separate obligations.

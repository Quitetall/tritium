# Local release-candidate evidence

Tritium admits unpublished release artifacts without treating their presence as
release readiness. `scripts/assemble-release-candidate.py` creates deterministic
artifact identities and SLSA provenance. `scripts/release-status` then rehashes
every byte and prints `CANDIDATE_EVIDENCE_VALID`. That status does **not** mean
`LOCAL_RC_READY`; model-zoo, browser, serving, package-matrix, signing and
second-machine gates remain separate.

The `release` Actions workflow has a non-publishing `candidate` dispatch mode.
It accepts only a full commit ID reachable from the default branch (or uses
that branch's current tip), then builds the same wheel, crate, npm, SBOM, and
release-input bundle as the tag-based path. Candidate mode must leave PyPI,
GitHub Releases, and crates.io untouched. The default dispatch mode remains
`publish` and requires an existing reviewed release tag; tag pushes also retain
the existing publish behavior. This workflow creates package evidence, not a
release-candidate admission or activation receipt.

### Packed embedding selected-row decode (2026-10-05)

`AdditiveTernaryEmbedding` now decodes packed trit bytes only for token IDs in
the current input, in bounded chunks of at most 2^18 weight elements before
per-plane accumulation. It does not build a dense vocabulary-by-hidden-size
weight table. A CPU/CUDA regression compares outputs exactly with the dense
reference for duplicate IDs, two planes, a partial final scale group and long
inputs spanning multiple chunks; it also checks int32/int64 IDs and empty
sequences while making the full-matrix decoder unavailable to the layer.
`PYTHONPATH=crates/tritium-py/python python3 -m pytest -q
crates/tritium-py/tests` passed 369 tests, with 27 skips. This is correctness
and bounded-temporary evidence for the Python reference path, not a native
fused-kernel or speed claim.

### Packed linear output-row decode (2026-10-07)

`AdditiveTernaryLinear` decodes packed weights in output-row tiles capped at
2^22 weight elements, instead of materializing the entire output-by-input
matrix on every forward call. Bias is cast to the input dtype as before, and
the input feature dimension is checked explicitly. The chunk-bound regression
constructs a 300,000-by-16 layer and verifies the exact two tiles (262,144 and
37,856 rows); the full `test_module_onnx.py` file passed all 9 tests.

The exact local candidate wheel
`pytritium-1.1.0rc2-cp39-abi3-linux_x86_64.whl` (SHA-256
`b21343f103aec89c0e843f72730dafb23dbdc49c639677e531d96f684888c94b`) ran the
pinned SmolLM2-135M tutorial on CPU in 255.23 seconds, excluding first model
download. The wheel binds source-tree object
`88fa42cf1b29b4f4d07bd36c99234ce316eb3fd4` (commit `917fd4ae`), model revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, and receipt run ID
`local-4m-88fa42cf`. The receipt SHA-256 is
`fe61ada1279767f06c8891b0683de35f1443a66e759e49e776aceed552f0125e`; it records
ONNX replay max absolute error `8.01e-5`, max tolerance ratio `0.368`, and
selected dense/checkpoint bytes `537,919,488` / `92,192,265` (5.83x). The wheel
and complete 2.2 GiB tutorial output are preserved under
`/mnt/2tb/tritium-smollm2-917fd4ae-local/`.

This is local CPU evidence only: the wheel's `linux_x86_64` tag is not a
manylinux release artifact, and the hosted candidate-wheel/tutorial gate had
not completed when this record was updated. It does not establish a native
fused kernel or cross-machine performance claim.

## Gate status (measured 2026-09-03)

### Hugging Face distributed CPU software checks (2026-10-04)

At source commit `e8362849`,
`/home/brianklam/.cache/tritium-py313-ci/bin/python -m pytest -q
crates/tritium-py/tests/test_huggingface_distributed.py` passed three checks:
two-rank CPU DDP training plus checkpoint reload, two-rank CPU FSDP training
plus sharded checkpoint resume/export, and Accelerate CPU bf16 execution. The
CUDA-only Accelerate test was skipped because this environment has CPU-only
PyTorch (`2.11.0+cpu`; Transformers `5.5.3`).

This verifies useful distributed software paths, but it is not the
candidate-bound `distributed-training` release receipt and does not qualify
multi-GPU execution, CUDA checkpointing, or performance. The `pytorch-hf` gate
remains PARTIAL until its required two-or-more-GPU evidence is registered.

The source-tree PyTorch dispatcher checks also passed on this branch head:
`/home/brianklam/.cache/tritium-py313-ci/bin/python -m pytest -q
crates/tritium-py/tests/test_torch_dispatch.py -k
'opcheck_and_fullgraph_compile or supports_functorch_grad_and_vmap'` reported
2 passed, 34 deselected. This checks the CPU ternary op's `torch.library`
opcheck, full-graph eager-backend compilation, `torch.func.grad`, and vmap
behavior. It is narrow local software evidence, not a built-wheel check, GPU
qualification, or a distributed-training receipt.

The focused Qwen3.6 capture integration suite also passed on CPU:
`/home/brianklam/.cache/tritium-py313-ci/bin/python -m pytest -q
crates/tritium-py/tests/test_kronecker_capture.py
crates/tritium-py/tests/test_qwen36_components.py` reported 41 passed and 1
CUDA-only skip. These synthetic checks cover Qwen component resolution,
capture-session resume/publication, token-stream binding, and small-model
capture behavior. They do not establish the pinned checkpoint's real 506-record
calibration capture, MTP production parity, or a release receipt.

### PR CI on release-input-admission (2026-10-04)

GitHub CI, docs, capstone CPU smoke, wheels, and CodeQL all passed for branch
head `cecf528bce32d000fe7cb1e116cd3c9dcb276d46` in run
[`37218555369`](https://github.com/Quitetall/tritium/actions/runs/37218555369).
The CI matrix included the receipt-backed compatibility-matrix job. However,
the downloaded crate and npm qualification receipts bind source revision
`b4b475da036791e93015439095eb5829bd13ae94`, the PR merge tree, rather than the
branch-head candidate. They are not admissible as package evidence for the
branch-head revision. The exact-source workflow supports an explicit source
revision, but no separate run has been dispatched for `cecf528`; package-matrix
admission therefore remains open for that candidate. Hardware and release
qualification are not implied by this PR CI result.

### Exact-source PR CI refresh (2026-10-04)

GitHub Actions run
[`37253818275`](https://github.com/Quitetall/tritium/actions/runs/37253818275)
completed successfully for exact branch head
`f2b2b3a35058638171df4928f8e120f453bcb760` on
`feat/release-input-admission`. The workflow's source-bound jobs verified and
used the PR head revision. The required CI aggregate passed, including the
cross-platform CPU test/lint/format matrix, source-bound compatibility and
package checks, CPU serving contract, ONNX CPU custom op, Burn/Candle CPU
interop, MSRV, API stability, supply-chain, SBOM, and workflow lint checks.

The run explicitly skipped fuzz parsers, CUDA conformance/parity, real-model
serving E2E, ROCm, performance regression, wgpu, and Metal. This is exact-source
software CI evidence only. It does not replace the package-matrix release
receipt, physical backend evidence, real-model serving or quality gates, or
independent release qualification.

The next exact-source run for head
`d32b2cab40b4f5ccedd74ea878062997485c7405` also completed successfully:
[CI run `37254882104`](https://github.com/Quitetall/tritium/actions/runs/37254882104)
passed its required aggregate, and
[wheel run `37254882146`](https://github.com/Quitetall/tritium/actions/runs/37254882146)
passed. CI skipped fuzz parsers, CUDA conformance/parity, real-model serving
E2E, ROCm, performance regression, wgpu, and Metal. The result validates the
exact software revision and package workflow, but does not close the omitted
hardware, performance, real-model, or independent release gates.

### PR CI refresh — research note commit (2026-10-04)

GitHub CI run
[`37256062399`](https://github.com/Quitetall/tritium/actions/runs/37256062399)
completed successfully for PR head `8289f9103a329c06dabca62856d7dc529bd280d3`
on `feat/release-input-admission`. The required aggregate and all enabled jobs
passed, including Linux/macOS/Windows CPU formatting, lint and tests, the
receipt-backed compatibility matrix, web package, crate packaging, API
stability, WASI, mocked serving, ONNX CPU, Burn/Candle interop, MSRV, supply
chain, SBOM and workflow lint. Source-bound evidence jobs use the PR head and
verify it with `verify-workflow-source.py`; the general PR CPU matrix validates
GitHub's merge result, as the workflow documents. The run skipped fuzz,
CUDA/ROCm/Metal/wgpu execution, real-model serving and performance regression.
This is software-CI evidence only; it does not close physical backend, model
quality, runtime-performance, flagship, or independent release gates.

### Installed CPU wheel, HF lifecycle, and ONNX smoke (2026-10-04)

At source revision `2f53bf00658f9d8412b98acff387763efb4e191d`, a CPU abi3 wheel
was built from a clean `git archive` of that commit, keeping unrelated dirty
working-tree edits out of the artifact. `scripts/verify-wheel.py
/mnt/2tb/tritium-wheel-2f53bf00-clean --install-smoke` passed. The wheel is
`pytritium-1.1.0rc2-cp39-abi3-linux_x86_64.whl`, SHA-256
`ea6b94ca7ee1f8fc766ec6d63dd2d13fa1e20af6c068bd1413bb17767442701a`,
10,228,737 bytes. Its `linux_x86_64` tag is host-local and is **not** a
manylinux/release wheel.

The installed-wheel functional smoke passed on CPU with PyTorch 2.11.0+cu130,
Transformers 5.5.3, and safetensors 0.8.0. It exercised native ternary matmul,
Hugging Face QAT forward/backward and optimizer update/resume, safe checkpoint
save/reload, and tied-weight identity. Receipt:
`/mnt/2tb/tritium-wheel-2f53bf00-clean/functional-receipt.json`, SHA-256
`e6882d1179f54b694bbde145bdcff373a948a11445c79b2701c4a0d5e463376c`, receipt
ID `sha256:ae75353764ac5c4fa7ebc7680f3e4ed07e0bb040e8f3f6e153f3566327b57eae`.
Its Tritium package and extension were loaded from the isolated wheel venv and
checked against the forbidden source-checkout path; dependency packages were
available from the host Python site packages, so this is not a fully isolated
dependency-install qualification.

From outside the checkout, the installed-wheel HF lifecycle receipt tests and
QAT tutorial receipt tests passed (2 + 5). The ONNX tests passed 28/28 using
ONNX 1.23.1, ONNX Runtime 1.30.0, and ONNX Script 0.7.2; they exercise tiny
module/graph artifacts, not whole-Qwen inference. Separately, the source-tree
Python suite passed 363 tests and skipped 27: CUDA-extension tests, the two
installed-wheel-only files, ONNX before its dependencies were added, and
external cross-project migration tests. These local checks do not close the
manylinux package matrix, CUDA wheel, two-physical-GPU distributed training,
full-Qwen ONNX, model-quality, or independent-release gates.

### Installed CUDA manylinux wheel smoke (2026-10-04)

At source revision `a6487b23f84c248f1440c5b71488db10621ad44c`,
`scripts/build-cuda-manylinux-wheel.sh` built the CUDA-enabled abi3 wheel in
the pinned manylinux 2.28 container against the host CUDA 13.4 toolkit. The
wheel is
`pytritium-1.1.0rc2-cp39-abi3-manylinux_2_28_x86_64.whl`, SHA-256
`7e4bfe1c8ffc8e16a165a77b9a3807af2f873c8c49224379e1ab1a240655c35a`,
2,889,220 bytes. `scripts/verify-wheel.py` passed its exact platform-tag,
wheel-integrity, and isolated-install checks. Its wheel smoke receipt is
`/mnt/2tb/tritium-cuda-wheel-a6487b23/wheel-smoke.json` (SHA-256
`e94562b037e033415505b62b63c65411869a510095a71cba48c4f423e18e25b3`),
bound to CPython 3.14.7 and target `linux-x86_64-cuda13-sm89`.

The installed-wheel functional smoke then passed on the RTX 4090 with native
device `cuda:0`: native ternary matmul, Hugging Face QAT forward/backward,
optimizer update and resume, safetensors save/reload, and tied-weight identity.
Receipt:
`/mnt/2tb/tritium-cuda-wheel-a6487b23/functional-receipt.json`, SHA-256
`feb46e27d24534bdbb92352d867b3f7b302e67bb035f9d616d3f6b3fe39bb1d0`,
receipt ID `sha256:2a8af2b9db4d1923e2b63af848002d341c56d6b760dc97352c406ab002f301b6`.
The native ternary operation used CUDA; the tiny PyTorch/HF QAT lifecycle ran
on CPU. This is one local Linux/Python/GPU package smoke, not a wheel-matrix
qualification, Qwen test, CUDA QAT proof, performance result, or independent
release gate. The wheel and receipts are durable local evidence under
`/mnt/2tb/tritium-cuda-wheel-a6487b23/`.

The installed-wheel QAT tutorial was also run separately on `cuda:0` from
outside the checkout. It completed a two-plane tied-embedding QAT step with a
finite nonzero gradient, optimizer save/resume, hard export, and strict artifact
reload; `--check-receipt` reopened the receipt and artifact successfully. Its
receipt is
`/mnt/2tb/tritium-cuda-wheel-a6487b23/tutorial-cuda/receipt.json` (SHA-256
`4de946cc4128bf3595e341d9351ba62d4f737837623015f87eccbc79441cdc98`,
receipt ID `sha256:abc1f22c92faa68093ff9e7cedc162548c6e7c2a639871062c10f71462ed8ed0`).
The hard artifact is a 2,533-byte tiny fixture, not a language model. This run
used the venv's normal Python mode so its CUDA-enabled PyTorch from host
user-site packages was visible; isolated `python -I` saw a non-CUDA PyTorch and
did not run this tutorial. It therefore verifies installed Tritium wheel
behavior on this host, but not a self-contained dependency environment or the
cross-platform CUDA packaging matrix.

### Current-revision CPU manylinux wheel and installed facade (2026-10-04)

At exact source revision `d885a80a770c19332d09b79bb844773d1aadb7b3`,
`/home/brianklam/.cache/tritium-prepush/worktree` was a clean detached worktree.
`scripts/build-cpu-manylinux-wheel.sh` produced
`/mnt/2tb/tritium-wheel-d885a80/pytritium-1.1.0rc2-cp39-abi3-manylinux_2_28_x86_64.whl`,
10,514,698 bytes, SHA-256
`7cb0e9c2ffb0adc3c6d132625e6341369b99984e50f7ea71a224a435643ac299`. The
manylinux platform and isolated install check passed. The wheel-smoke receipt
at `/mnt/2tb/tritium-wheel-d885a80/wheel-smoke.json` has SHA-256
`509b8809b940475407fc49fc29d27d45ef4ac2ad75bcc85ac6a9d10bd758bf7b`.

The installed-wheel functional smoke passed on CPU. Its receipt at
`/mnt/2tb/tritium-wheel-d885a80/functional-receipt.json` has SHA-256
`8e48e3eb7ae9f579e5367e3b0610b1ccfa8544212d0582b17fdc176610315e10` and
receipt ID
`sha256:c5b03196e628321d3e8995bc533a510581a8d892f15668d3d99da69e7225349c`.
It exercised native CPU ternary matmul, HF QAT forward/backward, optimizer
update/resume, safetensors save/reload, and tied-weight identity under CPython
3.14.7, PyTorch 2.11.0+cu130, Transformers 5.5.3, and safetensors 0.7.0. The
test venv installed the Tritium wheel but inherited dependency packages from
the host Python site; it is not a fully isolated dependency-install test.

The 24 tests in `crates/tritium-py/tests/test_torch_onnx.py` also passed from
that installed-wheel venv, with `tritium` resolved under the venv's
`site-packages`. The generation-adapter test uses a fake native runtime; this
does not qualify a real ORT session or whole-Qwen generation. These are
current-revision local package checks, not the cross-platform matrix, aggregate
package gate, or independent release qualification.

### Current-head CPU manylinux wheel refresh (2026-10-04)

The exact clean candidate worktree at
`/home/brianklam/.cache/tritium-prepush/worktree` was advanced to
`9f19b20b53a2d19aca3c6839b2d67611ce592811` and remained clean. The pinned
manylinux build produced
`/mnt/2tb/tritium-wheel-9f19b20/pytritium-1.1.0rc2-cp39-abi3-manylinux_2_28_x86_64.whl`,
10,514,737 bytes, SHA-256
`4f7770f7cfd9c0d877c7e11926e04995a6c3051dcc6985b79842f32069724662`. Its
platform-tag and install smoke passed. The verifier receipt is
`/mnt/2tb/tritium-wheel-9f19b20/wheel-smoke.json`, SHA-256
`aaaec33867016cdafca1e81d881a3ee60098bbe7ed4a548eca41d823a900173d`.

The installed-wheel CPU functional smoke passed from the wheel's venv. Its
receipt is `/mnt/2tb/tritium-wheel-9f19b20/functional-receipt.json`, SHA-256
`228debaf0ef27a5b76864a0147d000106cec293a54ed71702d47c8ad4288bd3f`, receipt
ID `sha256:270d6d4c8caf6270b1e30257500124e7b5652ef1c636288c5d8810983d68e7a2`.
It exercises native ternary matmul, HF QAT forward/backward, optimizer
step/checkpoint resume, safetensors save/reload and tied-weight identity on
CPython 3.14.7 with PyTorch 2.11.0+cu130, Transformers 5.5.3 and safetensors
0.7.0. The venv installed the Tritium wheel but inherited dependency packages
from host site-packages, so this is not a fully isolated dependency test.

All 24 `test_torch_onnx.py` tests also passed from that installed-wheel venv;
the import resolved to its `site-packages`. The generation facade tests use a
fake native runtime, not a real ONNX Runtime session or whole-Qwen execution.
These checks are local candidate-revision package evidence only. They do not
close the aggregate package gate, cross-platform matrix, real-Qwen ONNX,
model-quality or independent-release gates.

### Current-revision browser npm archive (2026-10-04)

At clean detached source revision
`7867ee606b1c273e7dc2b72e762db4fba79fdb1d`, the offline browser package
workflow passed with `npm run check`: generated-file checks, the pinned WASM
build, strict TypeScript, package build, 145 Node tests, and archive
verification. The locally built runtime was Node `v24.21.0` with npm `12.0.2`;
this does not establish a Node 22 run or the cross-platform package matrix.

The exact archive is
`/mnt/2tb/tritium-npm-archive-7867ee60/tritium-ai-web-1.1.0-rc.2.tgz`,
627,367 bytes, SHA-256
`1d6363c21e49ffcbb80ddbd85f9eb4857706f43779070d3606aa4d154459200b`.
Its strict npm qualification receipt
`/mnt/2tb/tritium-npm-archive-7867ee60/npm-archive-receipt.json` validated
with receipt ID
`sha256:30ecad1f696cc44005a5426147517613ec73ed576e91c996c504309ea98d4659`.
The package-lock CycloneDX SBOM was reproduced exactly from the archive-bound
receipt and locked dependency inventory; it contains 49 dependency components.
This is local package evidence, not a physical-browser WebGPU result or a
release-candidate package-matrix pass.

### Current-head browser npm archive on Node 22 (2026-10-05)

At clean detached source revision
`f003db81e7de8429325db114c7c8467becc5280c`, the full offline browser package
workflow passed with Node `v22.23.3` and npm `12.0.2`. Generated-file checks,
the pinned WASM build, strict TypeScript, all 145 Node tests, offline install,
and archive verification passed. The strict receipt was independently reopened
by `scripts/verify-npm-archive-receipt.py`'s validator.

Archive: `release/v1.1/evidence/npm-node22-f003/tritium-ai-web-1.1.0-rc.2.tgz`,
628,405 bytes, SHA-256
`5c2e1eed2fd6ad5540f006fca8d0a2b9b684a2e313518af7d7b8fad91b306447`.
Receipt: `release/v1.1/evidence/npm-node22-f003/npm-archive-receipt.json`,
ID `sha256:67408cc2fa5f088d107402bc81ebd43db7dda47cda732a1867e765bcb520bfc3`.
The SBOM is retained beside the archive. This adds a Node 22 local package
result for the exact current head; it does not establish the cross-platform
package matrix, candidate CI admission, or physical-browser WebGPU conformance.

### Physical browser WebGPU lane fragments (2026-10-04)

The exact RC.2 npm archive for clean source revision
`7523eb94e4d9d092eee78c155daa4ef0d2473d63` (627,893 bytes, SHA-256
`f9a869590467156dbb7d9aee83ff7eeb0d8b37246e95442ae5eb6243cef5cda4`) was
run through the physical WebGPU WebDriver lane in Chrome 154.0.8037.92 and
Firefox 157. Both traces report all 72 valid and 45 expected-invalid vectors,
zero skipped cases, the complete prepare/forward/backward/step/checkpoint/
resume/export/reload lifecycle, all six injected fault classes, zero
steady-state readbacks, and an exported artifact byte-identical to the native
reference. Chrome reports an NVIDIA RTX 4090 adapter. Firefox reports a
browser-sanitized NVIDIA renderer string; its exact adapter model is unknown.

The lane fragments and npm archive are retained at
`/mnt/2tb/tritium-v11-browser-7523eb94/`. Trace SHA-256 values are
`90ca8a1d6f8254760e8d25666b584de418d78474b61188bca8ab464caa288903` (Chrome)
and `991bca2c5458d8daa90d4688ee74aaaeabffc2109622da9c2d4233a3bcc78ad4`
(Firefox). These are contributor-run lane fragments, not a combined
`browser-conformance` receipt: physical Safari, same-candidate aggregation and
registry admission remain open. This also does not establish Node 22 or
cross-platform package-matrix coverage.

The same lane producer was rerun against the exact RC.2 npm archive for source
revision `b95092301a544a460df0d90dc3771c26f44c2baf` (628,123 bytes, SHA-256
`822bbaeeda9278b21a1791c18f0f403509e2681f8af13032936ce7fd0b49376e`) using
the clean source worktree, its exact-source native reference, and retained npm
qualification receipt. Chrome 157.0.8081.0 and Firefox 157 each completed all
72 valid and 45 invalid-input cases with zero skips; both passed the full
prepare/forward/backward/optimizer/checkpoint/resume/export/reload lifecycle,
six injected fault classes, native artifact parity, and zero steady-state
readbacks. Chrome identifies a non-fallback NVIDIA RTX 4090 adapter. Firefox
identifies a non-fallback NVIDIA renderer but sanitizes the exact adapter model.
Their trace SHA-256 values are
`5294b2569e1486d3ab4a6d7a7ae1012d88f587189be52ae049c819dfe6c1c5b8` (Chrome)
and `129b34281c53d44e009b7ada46341b266e6bce185a015ed048674b0a65eb980d`
(Firefox). Lane and trace files are retained under
`release/v1.1/evidence/browser-ci-b950923/`. These exact-source fragments still
do not satisfy `browser-conformance`: the physical Safari lane, same-candidate
aggregation, and registry admission remain open.

### ONNX Python facade source regression refresh (2026-10-04)

`python -m pytest crates/tritium-py/tests/test_torch_onnx.py -q` passed 24
tests on the current source checkout. In addition to strict manifest admission,
typed artifact routing, batch-one forward and MTP calls, and greedy cached
generation, the suite now exercises an opt-in Transformers `GenerationMixin`
adapter that carries Tritium's tuple cache through the standard `.generate()`
loop. The adapter is limited to batch-one CPU decoding without padding or beam
search. The test uses a fake native runtime: it does not execute the candidate
installed wheel, export a real authenticated Qwen bundle, or qualify whole-model
Qwen generation. Those candidate-bound ONNX gates remain open. The same focused
suite was rerun on 2026-10-05 at source revision
`5afd487b6dcc97b3447c70ca8852adc19ecdd17a` with
`/home/brianklam/.cache/tritium-py313-ci/bin/python -m pytest -q
crates/tritium-py/tests/test_torch_onnx.py`; it again passed 24 tests. The
rerun confirms the source-level facade result at the current release branch
head only; it does not upgrade the test's fake runtime to real Qwen ORT evidence.

### Serving software regression refresh (2026-10-04)

At source revision `c12812ada218dfbd8cfb524e31852b09deab18df`, the local
CPU-feature serving suite passed with
`RUSTC_WRAPPER='' cargo test --locked -p tritium-serve --features serve`:
42 library tests, 4 binary tests, 2 CLI tests, 31 contract tests, and 1
OpenTelemetry parentage test passed (80 total, 0 failed). The `batch_serve`,
`e2e`, and `spec_lookup` integration targets registered zero tests under this
feature selection; this run does not qualify CUDA, real-artifact serving, OCI,
Kubernetes deployment, or model quality. The matching local check
`RUSTC_WRAPPER='' cargo clippy --locked -p tritium-serve --features serve
--all-targets -- -D warnings` also passed. These are contributor-run software
checks, not independent release receipts; the serving gates below remain open.

### Stage-7 campaign orchestration regression checks (2026-10-04)

At source revision `b127fae2ee094647e5d6c61dadf465903701dd99`, Python 3.14.7
ran
`python -m pytest -q scripts/tests/test_run_stage7_recipe_freeze.py
scripts/tests/test_qualify_stage7_recipe_freeze.py
scripts/tests/test_verify_stage7_qualification_receipt.py`: 63 passed in
16.32 seconds. These synthetic tests exercise campaign orchestration, resume,
qualification, and strict receipt verification. They do not run the SmolLM2
recipe-freeze measurements, establish a terminal recipe decision, or create a
candidate-bound `stage7-recipe-freeze` release receipt. That empirical gate
remains open.

### GDN sensitivity receipt verifier alignment (2026-10-05)

The local GDN sensitivity receipt verifier now requires both output- and
state-divergence curves at every frozen sequence-depth point. It reports
DeltaNet maximum and full-attention median terminal state divergence as
diagnostics; the frozen routing decision remains based only on terminal output
divergence. The focused verifier and probe-preflight suites pass (16 tests),
Python compilation passes, and `git diff --check` passes. These are local
software checks only: no measurement receipt was produced and no Qwen weights
were loaded.

The production measurement producer is still missing. The Qwen runtime now has
an internal paired-sampling seam that can collect final hidden rows and selected
DeltaNet recurrent states at frozen token positions; fixture tests exercise it.
Proposed [ADR 0049](adr/0049-qwen36-gdn-sensitivity-metric.md) now defines
absolute RMS output/state metrics, fixed sample positions and state-layer
selection, but it is still `PROPOSED`; the v2 receipt schema and verifier are
not implemented. Therefore no accepted metric contract or source-bound
measurement producer exists yet. No eight-probe measurement has run against the
approved calibration pack, and no measurement receipt has been produced. Do not
start those probes until ADR 0049 is accepted, the v2 verifier and producer are
implemented and independently checked, and campaign authorization is explicit.
No probe execution or flagship campaign was started for this software change.

### Admitted scale-refined child execution (2026-10-05)

`Qwen36AdmittedExecutionSession::replay_refined_candidate` now reopens an
immutable scale-update child under the exact admitted parent profile, validates
parent/child identity and physical ledgers, freshly executes final logits and
the frozen output scopes on the sealed built-in backend, and mints the separate
`TSQ36RC v1` refined-execution receipt. The focused synthetic end-to-end test
`admitted_qwen_execution_binds_campaign_packages_backend_tokens_and_outputs`
passed, and `cargo test --locked -p tritium-salt --lib` passed (65 passed, 2
ignored). `cargo clippy --locked -p tritium-salt --all-targets -- -D warnings`
also passed. This verifies the campaign API and receipt path on a small CPU fixture
only. No pinned Qwen checkpoint was loaded; this is not Stage-9 quality,
performance, physical-size, CUDA, or release evidence.

#### B3 lineage decision (2026-10-06)

The user selected the immutable-child-package path for sliding-window scale
updates. The existing `TSQ36RC v1` path is the required replay/lineage mechanism;
the base `TSQ36EX v1` stays final-logit-only, and child scope evidence must not
be mislabeled as a `TSQ36SB` binding. This decision resolves the lineage choice,
not the B3 optimizer. On 2026-10-07, commits `2689b865`, `e032d89e`, and
`7537780c` added a bounded-memory `FixedTritScaleUpdateCandidateBuilder`: it
streams activation windows, emits canonical f16 scale updates, binds the owned
candidate to the frozen spec/parent/seed, and derives residuals as
`teacher - (current projection - active tile-plane contribution)`. The builder
now computes that active contribution directly from the fixed trits, base scales,
and activation window. Shape and finite-value errors are rejected. The
output-reconstruction integration suite passed (23 tests), package Clippy passed
with warnings denied, formatting passed, and the focused Qwen admission fixture
(1 test) passed. That fixture covers the existing immutable-child replay path
separately; the new builder is not yet connected to it. These are CPU fixtures,
not Qwen quality or hardware evidence. Production B3 remains open: a Qwen adapter
must supply aligned teacher/current projection outputs and exact base trits/scales;
deterministic per-window candidates must be scored and selected against the frozen
objective; selected updates must then be joined to child materialization and exact
replay. No full Qwen campaign or paid compute is authorized by this decision.

On 2026-10-06, the bounded Qwen paired-projection visitor was exposed as a
campaign-facing API. It resolves canonical MLP, DeltaNet, and full-attention
projection names, validates teacher/current geometry and activation arithmetic,
and synchronously lends finite outputs to the caller without retaining them.
The loaded SALT V2 fixture exercises it. The local ADR 0028 working copy records
the provenance boundary: callers bind teacher identity, activation identity,
projection name/index, spec, and parent. That amendment has not been promoted
from the private research repository into this public checkout. This visitor is
only an adapter seam; dense teacher acquisition, full campaign data capture,
and production candidate orchestration remain open.

The admitted-child CPU fixture now closes another software seam: it fits two
seeded scale candidates under a two-restart spec, materializes each as its own
immutable child, evaluates each loaded child into an output candidate receipt,
selects through the frozen objective, maps the winning receipt back to the exact
fitted candidate and child lineage, and freshly replays that child into
`TSQ36RC`. The fixture uses each child's own outputs as its teacher, so it
exercises identity joining and deterministic tie selection, not dense-teacher
quality ranking. The full `tritium-salt` suite passed (65 unit tests, 2 ignored;
all integration suites passed) and Clippy passed with warnings denied. This
proves software composition on synthetic CPU data, not Qwen quality, full-model
coordination, CUDA, or production qualification.

`OutputReconstructionReceipt::selected_fitted_scale_update_candidate` now
provides that join as a reusable checked API. It verifies spec and parent
identity, the full frozen restart count, unique fitted IDs/seeds, and an exact
ID-plus-seed match for every scored restart before returning the selected owned
update set. Focused tests reject wrong-parent and missing-restart inputs; the
admitted-child fixture now uses this resolver instead of manually matching
candidate IDs. Additional tests prove candidate-list reordering is harmless and
substituting an unscored fit is rejected. This adds no receipt fields or
wire-version changes, and enforces ADR 0050's exact child-candidate binding.

The admitted-session API now composes the visitor with the frozen spec, a
verified parent execution, an activation source, and a parent-bound scale-fit
builder. It reopens each requested block window against the activation-cache
digest, chooses the cache for the named projection's layer, and feeds paired
teacher/current outputs into the active plane fit. It checks package admission
again after the window. The builder accepts zero scales only when the matching
trit group is all zero, matching the SALT V2 package contract; it rejects
negative zero and zero-scaled nonzero groups. A synthetic admitted-Qwen fixture
now starts a fit from a strict packed parent plane and produces a spec- and
parent-bound candidate. The focused fixture, quantize tests, formatting, and
scoped Clippy passed. This is software-path evidence only: the fixture uses
synthetic activations and teacher weights, not the pinned Qwen teacher or
admitted production captures. Full deterministic candidate scoring/selection
and Qwen empirical evidence remain open.

The fixture now feeds the actual fitted candidate through the existing child
package writer. It reopens the candidate under its exact frozen spec, checks the
parent digest against a strict reader, emits `SaltV2ScaleUpdateChild` lineage,
then reloads the child and replays it through `TSQ36RC v1`; it no longer
substitutes hand-authored scales. The focused fixture and `tritium-salt` Clippy
passed. This proves the synthetic candidate-to-child-to-replay composition,
not a shipped campaign coordinator. A production admission-bound materializer
still needs its ADR/API decision; deterministic whole-model scoring/selection,
restart orchestration, admitted production captures, and Qwen empirical evidence
remain open.

The quantize suite now also composes four independently seeded fits with the
frozen output scorer and restart selector. It verifies that the selected
`OutputCandidateReceipt` maps back to the exact fitted scale-update candidate,
not merely a separately hand-labeled candidate ID. All 26 output-reconstruction
tests and quantize Clippy passed. This remains synthetic: it does not score a
loaded child model against dense Qwen teacher outputs or establish model quality.

### Dynamic packed-embedding ONNX export repair (2026-10-07)

Commit `cf87c4fc` removes the dynamic-slice `copy_` from the `torch.export`
capture path in `AdditiveTernaryWeight._dense_rows`. Dynamo capture decodes the
selected token rows as one functional tensor; ordinary eager execution retains
the bounded 2^18-weight-element chunk path. The regression uses a 576-wide,
three-plane embedding and replays dynamic sequences on both sides of the eager
chunk threshold. The full source-tree Python suite passed (374 passed, 27
skipped), and the focused test passed on Python 3.13 / Torch 2.11.0 CPU with a
tiny tied-weight Llama through the public ONNX exporter. That local environment
used ONNX 1.23.1, ONNX Runtime 1.30.0 and ONNXScript 0.7.2, not the pinned
SmolLM2 lane's exact ONNX dependency versions. The pinned model is not cached
locally, so no second full tutorial run was started.

Exact-source hosted run `37570157130` for `cf87c4fc` built Linux, macOS and
Windows wheels, and its source-free tutorial and installed-wheel checks passed.
At the last status check, the pinned SmolLM2 CPU tutorial was still running; it
has no terminal result yet. Therefore the previously observed SmolLM2 exporter
failure is not yet confirmed fixed by the exact candidate-wheel gate. This
source-tree evidence does not qualify the release candidate.

### Large flattened-index ONNX parity repair (2026-10-07)

Commit `41dc1f4c` replaces `torch.div(..., rounding_mode="floor")` in packed
row decoding with integer `torch.floor_divide`. ONNX had lowered the former to
float32 divide/floor; flattened positions above 2^24 could therefore select
incorrect packed-byte indices in large linear layers. The regression test
crosses the first affected index and checks exported ONNX output against eager
execution. The `test_module_onnx.py` suite passed (8 tests), as did the related
dynamic-sequence embedding export test and `git diff --check`.

Exact-revision hosted wheel workflow `37572547065` completed successfully.
Its pinned SmolLM2-1.7B CPU PTQ/QAT tutorial step passed against revision
`41dc1f4c`, including the ONNX replay/parity gate; Linux, macOS, and Windows
wheel builds, the installed-wheel Torch test, source-free tutorial, and the
abi3 matrix also passed. The tutorial evidence artifact was uploaded as
`smollm2-cpu-tutorial` (artifact `11461188114`, 1,298,452,082 bytes). At this
recording, Rust CodeQL remained in progress. CUDA, ROCm, Metal, wgpu, real-model
serving, and performance lanes were skipped by runner policy, so this does not
close those release gates or qualify the flagship Qwen artifact.

The restart seed now also controls the actual initial scale vector: a
domain-separated BLAKE3 derivation binds each initial scale to the frozen spec,
parent package, seed, tensor, tile, plane, and scale-group index. The synthetic
rank-deficient fit verifies that the seeded starts can produce distinct f16
update vectors, and repeating one seed reproduces the exact candidate ID and
update bytes. This closes a software mismatch where distinct candidate IDs
previously labeled the same zero-start fit. It remains a synthetic solver test,
not evidence that four starts improve Qwen quality; the production scorer must
choose among candidates using the frozen output objective.

### Flagship campaign status refresh (2026-10-04)

The canonical read-only probe was rerun against the durable workspace
`/mnt/4tb/tritium-qwen36-campaign-20260813` with
`python scripts/qwen36-ptq-status.py --work-dir
/mnt/4tb/tritium-qwen36-campaign-20260813 --json`. It still reports
`status=stalled`, 0 of 506 published masters, zero seals, and one staged
447,083,070-byte record whose recorded PID is not alive. This does not change
the prior provenance finding: the legacy S2KF evidence is not admitted for
fitting without a source-bound capture transcript or a recapture from the
approved calibration pack. No campaign was started or modified for this
refresh.

The probe was repeated on 2026-10-04. It again reports `stalled`, 0/506
published masters, zero seals, and the same 447,083,070-byte temporary record
with a dead recorded PID. The durable workspace contains 506 `.s2kf` capture
records (about 3.8 GiB); these are activation/evidence records, not fitted
master tensors. The source-admission receipt describes 27,318,026,240
additive coefficients across the 506 target tensors, but a matching tensor
count does not establish that these old captures used the approved 512-sequence
calibration pack. The capture-binding receipt is still missing. The campaign
was not restarted, and no fitting or model replay was run.

The local synthetic resume/seal regression
`RUSTC_WRAPPER='' cargo test --locked -p tritium-salt
campaign_resumes_seals_and_preserves_the_base_workspace -- --nocapture`
passed (1 test, 0 failed). It exercises restart, sealing, and preservation of
the base workspace using a fixture. It does not reopen the recovered 506 Qwen
captures, fit model tensors, or establish model quality or release readiness.

### Local pinned-checkpoint inventory refresh (2026-10-04)

Read-only inventory found all 15 named safetensors shards plus config, index,
and tokenizer files in each of these local directories:

- `/mnt/4tb/models/Qwen3.6-27B`
- `/mnt/4tb/models/qwen36-27b-6a9e13bd`
- `/mnt/4tb/qwen36-27b-source-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`

The inspected Hugging Face cache metadata names pinned revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. The durable source directory
`/mnt/4tb/qwen36-27b-source-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` was
checked against the existing pinned official identity receipt
`sha256:154f7807dc5aa829dd061020c4cf8e10db1aefafd2f6f95d6ab8301d5c01dbc9`:
all 29 file sizes and declared digests matched (15 safetensors shards and
`tokenizer.json` by SHA-256; 13 small repository files by Git blob SHA-1).
Its source-admission parent is `sha256:0a45d3b593893aaf660d34ecd31cc66bf28ae4fd19d411ffa0671d2747ca2fd4`.
The shard files total 55,563,006,400 bytes; this is 150,496 bytes above the
campaign's 55,562,855,904-byte raw tensor payload because the safetensors files
also contain headers/metadata. That difference is expected and is not a model
weight mismatch. This verifies that one local snapshot matches the pinned
official file manifest; it does not turn the base source-admission receipt's
`official_payload_authenticated: false` field into true or admit calibration
evidence. The other two directories have the expected filenames and sizes but
were not independently rehashed. Campaign status remains stalled (0/506
published masters, zero seals). No source weights were altered, deleted, or
fitted.

### Qwen source and calibration preflight recovery (2026-10-04)

The source-admission receipt still referenced a proof under `/mnt/4tb/tmp` that
no longer exists. The same 221,951-byte `ingest.tq36` proof was found in both
`/mnt/4tb/qwen36-source-admission/...` and the durable campaign workspace; both
copies hash to the receipt's expected SHA-256
`09b59e8e41d7e0f947e31d2fc8f4fb635804f162f0c7558a2df6ff6d98b834e0`. The
old campaign receipt did not match the parent of the existing official-source
identity receipt, so the strict no-overwrite
`scripts/rebind-qwen36-source-evidence.py` was used with the matching admitted
receipt and verified proof. It produced a fresh, strictly reopened receipt pair
and copied proof under
`/mnt/4tb/qwen36-source-admission/rebound-20261004/` (rebind receipt
`sha256:f316adcfe1da1343f1cdf52168ade72e48b59b6275e0dd3a9c5ed721234ee7f8`).
Original evidence was left unchanged.

Using the verified local checkpoint, rebound official-source identity, and the
existing token pack, `scripts/verify-qwen36-calibration-pack.py` passed and
produced calibration-pack receipt
`sha256:b5b2ba2801cc5125e6cc7561117268bf29a20f153a7e662c47a1ca2f078beb63` plus
pre-capture replay contract
`sha256:089f3800c6817c3a079fc146d9f087093031bee6b416c58eff708e28e0c5b55a`.
The pack contains 512 calibration sequences (1,048,576 ordered tokens). The
`scripts/capture-qwen36-from-pack.py` preflight passed against these exact
identities and batch digest
`sha256:ca913e334bf22c73755d27b11848599008790f672600b8085daf5ec53022202c`;
it explicitly reported `NOT STARTED` because `--execute` was not supplied.

The campaign workspace has 506 existing `.s2kf` files but no discovered
pack-linked capture-binding receipt. These records are not admitted as the new
verified pack's captured calibration evidence merely because the count matches
the tensor count. The canonical fitter still reports 0/506 published masters
and no seals. Next gates are a candidate-bound Stage-7 qualification, actual
pack-linked model replay/capture, strict capture-binding verification of all
506 records, and only then fitting. No GPU model load or capture was started.

### CUDA paged-KV cancellation smoke (2026-10-03)

At source revision `8e8f6e0eecd6521adaf7e5911b580af4ffebba2a`, the focused
CUDA BitNet serving test
`cargo test --locked -p tritium-serve --features cuda --test batch_serve
cuda_batched_admission_interleaves_live_slot -- --exact --nocapture
--test-threads=1` passed on the local RTX 4090: one test passed, zero failed,
in 249.85 seconds. The 2,048-token admission window took 428 ms while the
other slot emitted 16 tokens (maximum measured inter-token gap 35.5 ms); the
test's active, prefill, and queued-cancellation assertions also observed exact
KV reservation release and zero release failures. The cold warm-up request
logged 183 seconds, so this is not a decode-performance claim. This was a
source-tree integration test using the local BitNet GGUF compatibility fixture,
not a candidate-bound production-bundle receipt or the strict schema-v3
readiness gate. The checkout also contained unrelated, uncommitted EAT-O work.

### CUDA cold-start phase diagnosis (2026-10-04)

The same focused test was rerun with phase sums emitted after its first warm-up
request. It passed (1 passed, 0 failed; 255.64 seconds). The first request
measured 197.641 seconds from HTTP acceptance to decode-worker admission,
0.006 seconds in prefill, 0.141 seconds in decode, 197.649 seconds to first
token, and 197.791 seconds end to end. This localizes the cold delay before
model prefill/decode; it is not evidence of slow ternary token execution.

Source inspection explains the phase boundary: the batched worker constructs
the resident decoder and paged-KV pool before it receives queued jobs, while
the router previously marked the worker alive as soon as the thread was
spawned. That let chat requests queue during CUDA batch initialization while
`/readyz` incorrectly reported ready. The router now tracks batch-worker
readiness separately from liveness, rejects chat until decoder and KV-pool
initialization succeeds, and clears readiness if the worker exits. The focused
RTX 4090 test passed (1 passed, 0 failed; 223.90 seconds): `/readyz` was 503,
`/healthz` was 200, and early chat was 503 during startup; readiness later
became 200 after 165.043 seconds, and the existing admission/interleaving
checks passed. This is local source-tree evidence using the BitNet GGUF fixture,
not a candidate-bound schema-v3 readiness/deployment receipt, Qwen evidence, or
a release qualification. Plan 0052's production artifact and deployment gates
remain open.

### IMMA startup policy comparison (2026-10-04)

The same RTX 4090 integration test was repeated with only
`TRITIUM_IMMA_TUNE` changed. Readiness time was measured from immediately before
`build_router_batched` until `/readyz` returned 200:

| Policy | Readiness time | Meaning |
|---|---:|---|
| default (`tune`) | 175.060 s | Runtime may search for tile choices, then load/compile the selected functions. |
| `load` | 55.486 s | Avoids the runtime search; loads cached choices or uses the AOT choice. |
| `off` | 10.004 s | Skips IMMA prefill setup; this is not performance-equivalent to the other policies. |

Each run passed the same focused CUDA admission/interleaving test. These are
single-run observations on one GPU, one BitNet fixture, and this source tree;
they identify IMMA policy/setup as a major contributor to this cold-start case,
but do not provide a general startup guarantee or isolate exact additive costs.
`load` is a useful current operator workaround when its precomputed/AOT choice
is acceptable. The default policy has not been changed: altering it would affect
runtime behavior and needs a contract decision plus broader cold/warm and
performance validation. `off` is diagnostic only, not a recommended equivalent
serving configuration.

### Flagship campaign verification refresh (2026-09-29)

The previous refresh reported the pinned Qwen3.6-27B additive-master campaign
as sealed with all 506 tensor-master slots. The discovered campaign workspace
does not support that claim. Its path is
`/mnt/4tb/tmp/qwen36-ptq-bef4058c-v1/qwen36-source/tsc1_9553bf20975ed88ab3a673522930f9b585ae2e205959ea3dd00ee79c9587c0ba/tsc1_7e0c191fefc020e74bb0ea1da33d11f69a517a231970d6c9174ee66494e52aa1/tensor-work/v1`.
The canonical status probe reports `stalled`, 506 expected masters, zero
published additive-slot receipts, no completion seal, and one 447,083,070-byte
staged record whose owner PID is no longer alive. The workspace has 360
preserved-source slot references and 360 underlying `.twr` records; those are
not fitted additive masters. Two shallow source workspaces under
`/mnt/4tb/tritium-qwen36-work*` are idle and contain no campaign catalog.
Therefore the long-running additive-master fitting step is **not complete in
the discovered workspace**. The prior completion claim remains unresolved
because the PTQ bundle variants below refer to the same campaign/completion IDs
but no matching canonical seal was found in the Qwen workspaces or
`release/v1.1/`. Do not interpret either the old claim or bundle metadata as a
release-gate pass.

Two bundle variants exist under `/mnt/4tb/tmp` with the same campaign and
completion IDs but different selection IDs. Both manifests declare
`complete_model: false`, `official_payload_authenticated: false`, and
`source_identity_status: measured-awaiting-official-registration`. They are
useful conversion outputs, not admissible complete-model release artifacts.
Neither variant is bound to a candidate-specific flagship receipt. The
historical gate inventory below predates this refresh and must not be read as
current campaign liveness.

### Source identity and retained-bundle follow-up (2026-09-30)

A second read-only probe compared the dead staged record's header with its
actual length: it declares a 1,077,709,406-byte record but contains only
447,083,070 bytes (about 41.5%). This is an incomplete tensor stream, not a
recoverable published master. Resuming the current campaign store scavenges
crash-left temporary records; do not promote this file or describe it as a
completed tensor.

The two observed bundles are
`/mnt/4tb/tmp/qwen36-ptq-b3-r2-r3-565abdee` and
`/mnt/4tb/tmp/qwen36-ptq-b56b4b7-v1-bundle`. Their manifests carry the same
campaign ID, completion ID, and measured source-model ID, but distinct
selection IDs. Both remain explicitly unauthenticated and incomplete; their
different profile sizes do not establish two independently completed master
campaigns.

The inspected durable source directory,
`/mnt/4tb/qwen36-27b-source-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, holds
about 1.1 GiB of Hugging Face cache data, including partial shard downloads,
not the complete checkpoint. The pinned
[official Qwen revision](https://huggingface.co/Qwen/Qwen3.6-27B/tree/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9)
contains 15 weight shards totaling 55,562,855,904 bytes; its
[revision API metadata](https://huggingface.co/api/models/Qwen/Qwen3.6-27B/revision/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9?blobs=true)
provides the pinned per-file SHA-256 values for source verification.

There is also a code-level admission prerequisite: `Qwen36SourceIdentityStatus`
currently has only `MeasuredAwaitingOfficialRegistration`, whose
`official_payload_authenticated()` result is false. A manifest edit cannot
authenticate these bundles. Before another flagship run can produce admissible
evidence, an independently verified official source identity must be
registered through a separate source-identity path, and the complete pinned
source payload must be available and verified. No download or campaign restart
was performed for this follow-up.

### Pinned source fetch and checksum verification (2026-09-30)

The complete pinned Hugging Face snapshot has since been downloaded into the
durable source directory above. `hf cache verify Qwen/Qwen3.6-27B --revision
6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 --local-dir
/mnt/4tb/qwen36-27b-source-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
--fail-on-missing-files` verified all 29 repository files and reported that all
checksums match. Excluding Hugging Face's local `.cache` metadata, the directory
contains exactly those 29 files totaling 55,586,107,940 bytes. Passing
`--fail-on-extra-files` is not appropriate on this `--local-dir`: the CLI counts
its own `.cache/huggingface` lock/metadata files as extras. Those files were
preserved; no cleanup was needed.

This verifies the downloaded files against the pinned Hub revision but does
not by itself satisfy Tritium's code-level source identity admission. The
source-admission receipt intentionally remains unauthenticated. A separate
official-identity verifier and registration are still required before source
admission can authorize the 506-tensor campaign. That campaign has not been
restarted, and the prior incomplete bundle variants remain inadmissible.

The Rust `qwen36-preflight` then completed against this verified snapshot. It
measured source model ID
`trm1_126eb094f936c87bf7aeff60e57dadf5351ff082a48b8d63c7553919029cd3ca`,
manifest content ID
`tsc1_9553bf20975ed88ab3a673522930f9b585ae2e205959ea3dd00ee79c9587c0ba`,
and proof ID
`tsc1_7e0c191fefc020e74bb0ea1da33d11f69a517a231970d6c9174ee66494e52aa1`.
The generated 221,951-byte proof is byte-identical to the proof already in the
stalled campaign workspace (SHA-256
`09b59e8e41d7e0f947e31d2fc8f4fb635804f162f0c7558a2df6ff6d98b834e0`). This
confirms the preserved workspace used the exact same content-bound source.

The source-admission receipt produced with the matching CI wheel and accepted
by `verify-qwen36-source-admission-receipt.py` has receipt ID
`sha256:718abe3e52eab53cc7e945fc0232dc18a42e3bac413544b855494594ae7ba08b`.
It confirms the 1,199-tensor inventory (506 additive, 360 preserved, 15 MTP)
but correctly still reports
`identity_status=measured-awaiting-official-registration` and
`official_payload_authenticated=false`. The current master-campaign status
probe still reports `stalled`, zero of 506 published master receipts, no seal,
and the same dead 447,083,070-byte staged record. Source admission is not
fitting completion or a deployable model.

### Separate official source identity verification (2026-09-30)

The new `verify-qwen36-official-source-identity.py` path separately fetches
the pinned Hugging Face revision metadata, verifies every local file against
the official file inventory (LFS SHA-256 for 16 files and Git blob SHA-1 for
13 ordinary Git files), and requires the measured source-admission IDs to
match the frozen Qwen3.6 identity. It verified 29 files totaling
55,586,107,940 bytes, with official manifest digest
`7911b682b615162590074c15baa429ff23c64b7c1d66bd2e134ef6fa3a2a3a3f`.

Its generated receipt is
`release/v1.1/evidence/qwen36-official-source-identity-2026-09-30/receipt.json`
(receipt ID
`sha256:154f7807dc5aa829dd061020c4cf8e10db1aefafd2f6f95d6ab8301d5c01dbc9`),
bound to the measured source-admission receipt
`sha256:0a45d3b593893aaf660d34ecd31cc66bf28ae4fd19d411ffa0671d2747ca2fd4`.
This receipt does not mutate or replace source-admission evidence, and it is
not yet a registered release gate. It establishes exact official snapshot
bytes bound to the already measured semantic ID; integrating that registration
into campaign authorization and the release registry remains open. The prior
incomplete bundle variants remain inadmissible, and the 506-tensor campaign
has not been restarted.

### Source identity release-registry linkage (2026-10-01)

The release evidence evaluator now requires both `source-admission` and
`official-source-identity` receipts for the `qwen-source-admission` gate. The
official receipt must name the exact source-admission receipt ID as its sole
registry parent; both entries must bind the same candidate source artifact, and
repository, revision, semantic model ID, manifest ID, and proof ID must agree.
Each registry entry ID must also equal the ID derived from its immutable
receipt bytes. An admission receipt by itself therefore remains `MISSING`, not
a source-identity pass.

This implements the registry-side linkage only. The no-replace producer is
`scripts/register-qwen36-source-identity.py`; it copies the verified identity
receipt into the evidence root, adds the exact admission parent, and validates
the full candidate registry before retaining the new registry. Its fixture
tests pass. An attempt to extend the retained `3662cc3f` registry rolled back
its outputs because the old crate-archive receipt's lock digest no longer
matches the current Cargo.lock. A current same-revision candidate and refreshed
package evidence are needed for actual registry publication. The Python Qwen
reconciliation wrapper and both public Rust PTQ reconciliation entrypoints now
require a validated source-identity authorization. The shared Rust driver binds
that authorization to the retained preflight before it opens or resumes the
campaign workspace. Candidate-only source admission remains available for
research, but cannot invoke the canonical Qwen PTQ reconciler without the
official-identity receipt pair. This closes the execution-path gap; it does not
register the source gate or refresh stale package evidence. No fitting or
campaign restart was performed.

### Durable source-proof copy (2026-10-01)

The 221,951-byte proof named by the retained source-admission receipt was copied
byte-for-byte from `/mnt/4tb/tmp` to
`/mnt/2tb/tritium-release-evidence/qwen-source-admission/sha256-0a45d3b593893aaf660d34ecd31cc66bf28ae4fd19d411ffa0671d2747ca2fd4/source-proof.tq36`.
Its SHA-256 is
`09b59e8e41d7e0f947e31d2fc8f4fb635804f162f0c7558a2df6ff6d98b834e0`, matching
the receipt. `scripts/rebind-qwen36-source-evidence.py` then produced a new
durable receipt pair under
`/mnt/2tb/tritium-release-evidence/qwen-source-admission/rebound-2026-10-01/`.
The admission receipt changes only the proof path; the official-identity
receipt changes only its admission parent and derived receipt ID. The
`rebind.json` records both parent IDs, both new IDs, and the exact changed
fields. The reissued pair passes the source-admission and official-identity
receipt validators. This is a host-local relocation that reuses the existing
official inventory and Hub-response digest; it is not a fresh Hub/checkpoint
verification or a new release-registry admission. A standalone Rust check
using the in-progress `Qwen36SourceIdentityAuthorization` consumer opened this
reissued pair and verified the proof bytes and receipt IDs. The current tracked
registry still references the original receipt pair, so `/mnt/4tb/tmp` must
not be pruned until a new validated registry is published against the current
candidate.

### Revalidated durable source identity (2026-10-01)

The official-identity verifier was rerun against the complete pinned snapshot
at `/mnt/4tb/qwen36-27b-source-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
It passed all 29 files (55,586,107,940 bytes) against the pinned Hub inventory
and emitted
`/mnt/2tb/tritium-release-evidence/qwen-source-admission/recheck-2026-10-01/official-source-identity.json`
with receipt ID
`sha256:868dd248ff845a39e7a02414196649730b492b8b625f27b2815dc4594e2582df`.
The receipt binds the rebound durable source-admission receipt
`sha256:abfb820bbc4fd65aff43b2c11e1471bd7bf7e9b72acd42073ff857a7cefe03d2`
and the same measured source-model, manifest and proof IDs. This is a refreshed
official-byte check and an authorization input; it does not rewrite the
source-admission receipt's `official_payload_authenticated=false`, register the
pair to a current release candidate, or produce tensor masters. The 506-master
campaign remains stalled and was not restarted.

### Fresh pinned-source identity check (2026-10-07)

The official-source verifier was rerun against the durable 52-GiB snapshot and
the rebound source-admission receipt. It passed all 29 local files against the
immutable Hugging Face revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`,
covering 55,586,107,940 bytes. The fresh receipt is
`/mnt/2tb/tritium-release-evidence/qwen-source-admission/recheck-2026-10-07/official-source-identity.json`
with ID
`sha256:c045300d5de262d4d72887d711ddfc6f413e83adc3b560332e32abecc6414eca`.
It binds source-admission receipt
`sha256:abfb820bbc4fd65aff43b2c11e1471bd7bf7e9b72acd42073ff857a7cefe03d2`
and the already measured model, manifest, and proof identities. This confirms
the durable source bytes still match the pinned public snapshot; it does not
register a current release candidate, change
`official_payload_authenticated: false`, admit calibration evidence, or prove
PTQ quality. No model replay, fitting, or campaign was started.

The frozen calibration pack was revalidated against this fresh identity using
`/mnt/4tb/tritium-qwen36-campaign-20260813/token-pack/manifest.json`. The new
pack receipt is
`/mnt/2tb/tritium-release-evidence/qwen-calibration/recheck-2026-10-07/calibration-pack.json`
(`sha256:5faa6dafa3d05fca967a19b9516d0c80aa69a3a330b14e3156c7c4ff0cc15e9f`);
its replay contract is
`/mnt/2tb/tritium-release-evidence/qwen-calibration/recheck-2026-10-07/replay-contract.json`
(`sha256:3df79a3aac599069dd88dd98e77e74866a5153ca42d37c0ea81a7797456120b3`).
Both preserve the existing frozen pack ID and batch digest
`sha256:ca913e334bf22c73755d27b11848599008790f672600b8085daf5ec53022202c`
for 512 sequences / 1,048,576 calibration tokens. The capture-from-pack
preflight passed against these exact receipts and printed `NOT STARTED` because
`--execute` was not supplied. The new receipts establish source/pack/replay
consistency; they do not prove actual model replay or admit the legacy 506 S2KF
records. Candidate-bound Stage-7 qualification and actual capture remain gates.

### Legacy Qwen calibration evidence audit (2026-10-01)

The 506-file S2KF directory
`/mnt/4tb/tmp/qwen36-evidence-a417374-1seq-clean-20260821` passes the
installed native structural inspector `inspect_qwen36_ptq_evidence`: evidence
ID `tsc1_11df78c64eaa9a27ec43f4a226d701c30bd897bbeca5c0b8692bae24631e2c71`,
506 records, input-Hessian curvature, pinned source-model ID
`126eb094f936c87bf7aeff60e57dadf5351ff082a48b8d63c7553919029cd3ca`, and
common activation-cache/token-stream identities. Its token-stream digest is
`d30fbc02209285448253dde628fe4c5285700cf86d66296c500eb6b1c38dcd2b`.

That digest exactly matches the first 2,048-token member of token pack
`sha256:e17652c928e5d378f19c3d3344c167844101df9ab2ff4362a05f117ff5889f38`.
The pack's calibration partition contains 512 members (1,048,576 tokens), with
the frozen C4/OpenWebMath/StarCoderData 50/25/25 composition. ADR 0043 requires
that coverage for scored rungs 2–4. This is therefore a provenance mismatch
that blocks treating the legacy evidence as campaign-grade; it is not proof
that the capture consumed only one sequence, because the capture API accepts
the token-stream digest from its caller and the old capture invocation ledger
has not been found. Do not start fitting from this evidence. Recover a
source-bound capture transcript or recapture from the admitted frozen pack,
recording the exact partition/window receipt, before resuming tensor masters.
The pack receipt records the full calibration partition token digest as
`sha256:98008cb043f6df722a81cca127f71eb8cb84f05445fbd7a35f1aff948f63fe15`,
which differs from the legacy S2KF token-stream identity above. It does not
change the unknown about what the historical capture actually consumed.

The evidence remains preserved in `/mnt/4tb/tmp`; no file was moved or removed.
The source-model ID match and structural inspection do not establish calibration
coverage, accuracy, or a completed master campaign.

A fresh read-only `scripts/qwen36-ptq-status.py` probe of the durable campaign
workspace reports `stalled`, 0 of 506 published masters, no seal, and one dead
temporary record of 447,083,070 bytes (its recorded PID is no longer alive).
That partial file is retained; it is not a complete tensor master and is not
evidence that the campaign can resume against approved calibration provenance.

The probe was refreshed on 2026-10-08 using
`python scripts/qwen36-ptq-status.py --work-dir
/mnt/4tb/tritium-qwen36-campaign-20260813 --json`. It still reports zero
published masters, zero seals, and that same dead staged record; expected
payload for the 506 fitted tensors is 23,156,295,680 bytes. No rate or ETA is
available without a live writer. This operational probe made no campaign
changes. The historical S2KF records remain calibration-evidence inputs, not
fitted weights, and the missing capture-to-approved-token-pack binding still
prevents resuming fitting from them.
A targeted search for the legacy and approved token-stream digests and
capture/replay transcript markers across the campaign workspace, the recovered
506-record evidence directory, and the Qwen calibration evidence directory
found only the calibration-pack and replay-contract receipts; no capture
invocation transcript surfaced. This search is bounded to those roots and file
types, so it does not establish that no copy exists elsewhere.

The Qwen token pack is distinct from Stage 7's SmolLM2-1.7B recipe-freeze pack.
The Qwen pack declares tokenizer digest
`sha256:72943ec7247b68e70aa6e5651a5b0abb870b07a1c1f90bd5da9badece7294407`,
vocabulary size 248,320, and 512 calibration sequences. The pinned Qwen source
config also declares text vocabulary size 248,320. By contrast,
`release/v1.1/campaign-84284a4-cuda.json` names
`HuggingFaceTB/SmolLM2-1.7B`, binds a different token-evidence manifest digest
(`sha256:2664bb998a231865baf55cb76806c67e53022b479ddfa901346f9a1d7cb9a0ae`),
and records tokenizer digest
`sha256:4b0c039b16d1fb8cb6d06c8e1698671d03c9ef51372f1ffff1fe0aa0fd555ced`.
Its Stage 7 recipe-freeze receipt cannot qualify Qwen's calibration tokenizer
or capture.

The tokenizer identity and token pack have now been independently verified by
`scripts/verify-qwen36-calibration-pack.py` against the official source-identity
receipt. The content-addressed receipt
`/mnt/2tb/tritium-release-evidence/qwen-calibration/calibration-pack-e464c020.json`
binds the official tokenizer identity, complete pack, exact 512-sequence
calibration partition, dataset revisions and source-member identities. It
confirms that the pack's tokenizer digest
`sha256:72943ec7247b68e70aa6e5651a5b0abb870b07a1c1f90bd5da9badece7294407`
matches the canonical inventory of the four pinned Qwen tokenizer assets, and
that the calibration partition contains 1,048,576 ordered tokens. This receipt
is pack-provenance evidence only: it does not prove that the model replay
consumed those tokens or bind an actual replay digest to an ordered S2KF
evidence-set digest. Those capture/evidence links remain required before
fitting; the legacy 506-record set is still not admitted for campaign use.

A deterministic pre-capture replay contract is now persisted at
`/mnt/2tb/tritium-release-evidence/qwen-calibration/replay-contract-2026-10-01.json`
with ID
`sha256:bb1e8d9d4f7a8564a76c6ca14378a0d1698dba01a3170c0e473c7c8936aadfcc`.
It binds the verified pack receipt to the expected PyTorch batch digest
`sha256:ca913e334bf22c73755d27b11848599008790f672600b8085daf5ec53022202c`
under a fixed one-sequence-per-batch policy. The capture API already checks
that digest against each replay before publishing records. The contract is not
evidence that replay occurred. `scripts/qwen36_calibration_replay.py` now
reopens the official pack and receipt, retains only the verified 4,194,304-byte
calibration token window, and yields the exact batches required by that digest.
Opening it against the durable Qwen pack reproduced the same pack receipt,
contract ID, and capture batch digest. This remains tooling verification, not a
model replay: the actual capture must use this factory, persist a native capture
receipt binding its digest to the ordered S2KF set, and freshly reopen all 506
records before fitting.

The replay helper now exposes `capture_binding()` and durable
`write_capture_binding()` for the native capture result. The separate
`scripts/verify-qwen36-capture-binding.py` command revalidates the source pack,
contract, native session identity and ordered S2KF evidence-set digest. Its
reopen path is covered with a fake native session in unit tests only; no real
Qwen capture receipt or complete 506-record evidence namespace exists yet.
On 2026-10-05, the capture-from-pack, calibration-replay, and calibration-pack
validator suites passed locally (22 tests and 10 subtests). They validate
software behavior with fixtures; they do not provide a real native reopen or
admit the recovered legacy captures.

### Hosted package evidence from PR #51 (2026-10-01)

PR #51 (`bbbafd99cada6dab821cea63e5cdf971f1ce79fb`) has successful hosted
package workflows. Its artifacts are retained under
`/mnt/2tb/tritium-release-evidence/ci-bbbafd99/`. The workflows ran against
the PR synthetic merge revision
`fb670342ee85e340397c68705127b0268648261b`, not the branch head. The
`crate-archive`, `npm-archive`, `compatibility-matrix`, and `clean-install`
receipts each pass their owning validator at that exact revision. The
compatibility receipt covers 16 CPython/platform cells; the clean-install
receipt validates an installed Linux CPU wheel with PyTorch QAT forward and
backward, optimizer update and resume, Hugging Face safetensors save/reload,
and tied-weight identity checks.

This harvest does **not** close the `packages` release gate: these receipts
are not registered against a candidate manifest, and their synthetic merge
revision differs from the checked-out branch head. Do not transplant them into
the older `release/v1.1` candidate or registry. Rebuild/harvest from the final
release revision and validate all four receipt kinds against one candidate
before recording a gate pass. Hosted CUDA, Metal, ROCm, wgpu, real-model
serving, fuzz, and performance-regression lanes were skipped by runner policy;
those remain unqualified.

### Local exact-source CUDA dispatcher evidence (`577bdb2e`)

At source revision `577bdb2e21faf2b43ff35dd1101a5079d5dbf331`, the portable
`manylinux_2_28_x86_64` CUDA wheel passed the physical RTX 4090 dispatcher
qualifier: 8/8 native CUDA tests passed, and the two selected tail-path tests
passed Compute Sanitizer 2026.3.0 with zero errors. The independently verified
receipt is
`sha256:f2f8b3c62503e419ae9ff8ea5b7195f2d9132e5c0fa3c06f9bc9b381cd25afaf`;
it binds wheel SHA-256
`295b3bb9f6a22c9276bfffb346e28179a6f2cc6c1aa94f80554da1ee439d93d9`, CUDA
13.0, driver 615.71.09, Torch 2.11.0+cu130, and RTX 4090 UUID
`1790118a-a6d7-4eaf-fcac-dcacac5f4351`. The verifier passed against that exact
source worktree and wheel. Receipt and raw outputs are retained in the local
candidate workspace at `release/v1.1/evidence/torch-dispatch-cuda-577bdb2/`.

This run used Python 3.14.7, while the hosted CUDA workflow is pinned to
Python 3.13. It is local, source-bound hardware evidence; it is not registered
to a candidate manifest, does not replace the hosted pinned-environment run,
and does not close the release gate. Re-run and register against the final
release candidate before making a release claim.

### Exact pushed-head CUDA rerun (`08a52cba`)

The receipt was rebuilt and rerun on the exact pushed PR head
`08a52cba8b8104018cc427500f43b708ba829ae9`, eliminating the source-revision
gap from the earlier `577bdb2e` run. The portable CUDA wheel passed all eight
native CUDA dispatcher tests on the RTX 4090; both selected tail-path tests
passed Compute Sanitizer 2026.3.0 with zero errors. The independently verified
receipt is
`sha256:49ea0d53fe3417bcaa93fc69af71a6183a9a035eae53ba9782d28b636dec10a9`.
It binds wheel SHA-256
`0556e947831803a67597e4718600655aa4e3c387812c06e631e80c86ecbccb9a`, CUDA
13.0, driver 615.71.09, Torch 2.11.0+cu130, and the same RTX 4090 UUID. Raw
outputs are retained under
`release/v1.1/evidence/torch-dispatch-cuda-08a52cba/` in the local candidate
workspace.

The wheel was built in the pinned manylinux/Python 3.13 image, but qualification
executed with the host's Python 3.14.7. The hosted Python 3.13 CUDA job remains
skipped. This exact-source local receipt is not registered to a candidate
manifest and does not close the release gate.

### Physical native-wgpu training corpus (`b3a556d1`)

The native wgpu receipt sealer ran from a clean detached worktree at
`b3a556d1c6eec85772ea2ed9aa95018705aece57` on the physical RTX 4090 via
Vulkan. The receipt binds manifest digest
`9093a1a7f9a3422c399943782aadf4df6b11833cf2253db0db56ff2d9dedb098`, vector
digest `38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7`,
36 operations and all 117 frozen cases. The reopened development capability
table reports 4,192 peak resident bytes and 132,032 peak scratch bytes. Receipt
digest:
`adeeeff34a2c3a7b8fe3952af9aa2144492aaf7e9583849baf0f42afa40ecb08`.

This is physical development evidence, not candidate-registry admission or
the seven-backend release gate. The hosted/self-hosted wgpu CI lane was skipped;
candidate artifact binding, release admission, and the other required backend
receipts remain separate obligations. The receipt is retained at
`/mnt/2tb/tritium-wgpu-b3a-receipts/` in the local candidate workspace.

### Latest local verification (2026-09-18, `570a8802`)

`scripts/verify-gates.sh release` completed with exit status 0 after the
`rustls` security update to `0.23.45`. The run observed clean formatting,
warning-free workspace Clippy, workspace tests, the 558.63-second CPU fidelity
ladder, exact 32-token CPU greedy parity, 481 Python tests, community-contract
validation, `cargo deny check`, release-version validation, and RC semver
reporting. Semver reports intentional RC API changes but remains non-blocking
until `TRITIUM_SEMVER_MODE=block` is selected.

This is local gate evidence only. It does not promote any missing model,
hardware, browser, deployment, independent-review, or public-activation gate.

### Latest CI verification (2026-09-18, run `35383127123`, commit `bf4c181d`)

The required CI workflow completed successfully. Supply-chain, compatibility
matrix, Linux/macOS/Windows CPU validation, Metal, MSRV, Burn, Candle, ONNX,
WASI, Web package, SBOM, API stability, publish readiness, workflow lint and
CPU benchmark lanes all passed. CUDA, ROCm, wgpu, real-model serving, fuzzing
and performance-regression lanes were skipped by runner policy; those skips are
not release evidence. The compatibility receipt exists in CI output but is not
yet harvested into a same-revision release candidate registry.

This CI result validates source and packaging workflows only. It does not
promote the Qwen flagship, hardware, deployment, model-zoo, reproduction,
signing or public-activation gates.

### Current-head local package probes (2026-09-18, `6211a513`)

The current clean checkout passes `check-publish.sh` and local crate archive
qualification for all 23 publishable crates. The npm archive probe passes its
strict TypeScript, offline-install and 143-test suite. A pinned
`manylinux_2_28_x86_64` abi3 wheel passes structure/install smoke and the
six-operation functional smoke (`native_ternary_matmul`, QAT backward,
optimizer step/checkpoint resume, HF safetensors reload and tied-weight
identity). Receipts are persisted under the ignored
`release/v1.1/evidence/{crate-archive,npm-archive,clean-install}-6211a513/`
tree. They bind this exact source revision but are not yet assembled into a
candidate manifest or registered in the release evidence registry.

The preceding source revision (`2f5adf6728ad4654b2811b6a08359ad111004fbb`)
also passed local package probes (not yet registered in a release candidate):
`check-publish.sh` and 23-crate qualification, npm archive qualification, and
one pinned manylinux CPU wheel clean-install plus differentiable smoke. Receipts are under the ignored
`release/v1.1/evidence/{crate-archive,npm-archive,clean-install}-2f5adf67/`
tree and bind that full revision. They remain local evidence until candidate
assembly, compatibility-matrix harvesting, and registry binding.

The twelve gates and their 38 evidence kinds are defined in code, not here —
`scripts/release-evidence-status.py`, constants `GATES` and `KNOWN_KINDS`. That is
deliberate: a partial or adversarial registry cannot remove a gate.

The table below is the measured **union** across every local registry under
`release/v1.1/` (23 registries, 105 receipts). A union is more generous than any
real report can be: `evaluate()` additionally requires a registry to bind one
exact candidate manifest at one exact `source_revision`. Read it as an upper
bound on progress.

> **Corrected 2026-09-05 — the coherence problem is smaller than this section
> first claimed.** The original text said "no single revision comes close to
> satisfying a gate set". That is true of the *local* registries and false of
> what CI produces. **Every release run emits a coherent, same-revision evidence
> set that has never been harvested.** The rc.2 run (33955449151) alone produced
> receipts at `d16c0dda` for **seven** evidence kinds:
>
> | kind | receipt schema | where |
> |---|---|---|
> | `clean-install` | `tritium.wheel-functional-qualification.v1` | `wheel-functional-*` |
> | `compatibility-matrix` | `tritium.abi3-matrix-qualification.v1` | `abi3-compatibility-receipt` |
> | `api-signature` | `tritium.installed-api-signature.v1` | `wheel-functional-*` |
> | `installed-qat-tutorial` | `tritium.installed-qat-tutorial.v3` | `wheel-functional-*`, `wheel-tutorial-*` |
> | `export-reload` | `tritium.hf-export-reload.v1` | `wheel-tutorial-*` |
> | `frontend-lifecycle` | `tritium.hf-lifecycle.v1` | `wheel-tutorial-*` |
> | `observability` | `tritium.installed-observability.v1` | `wheel-tutorial-*` |
>
> The `release-bundle` artifact also carries the three wheels and three
> CycloneDX SBOMs already named to the artifact-ID convention this document
> specifies, so candidate assembly needs no hand-built inputs. Confirmed by
> doing it: the rc.2 candidate assembles clean (`assemble-release-candidate: OK:
> 1.1.0-rc.2 (3 artifacts)`) at
> `release/v1.1/candidates/d16c0dda-rc2-harvest/`, manifest SHA-256
> `e766391e7ba16004…`. `release-status` then declines it only because it
> requires the candidate's `source_revision` to equal the checked-out HEAD —
> a correct guard, satisfied by running it from a worktree at `d16c0dda`.
>
> Map schemas to kinds by reading each **validator's** import in
> `release-evidence-status.py:13-46`, not by matching filenames: `clean-install`
> resolves through `wheel-functional-smoke.py`, *not* `verify-wheel.py`, so
> `tritium.compatibility-receipt.v1` — which sits in the `release-bundle` and
> looks like the obvious candidate — is an input to the abi3 aggregation rather
> than a registry kind. Registering the set still requires per-kind binding work:
> each registry entry's `id` must equal its receipt's own `receipt_id`, and the
> `tritium.compatibility-receipt.v1` files carry neither that nor a `release`
> field.
>
> What this changes: the barrier to a coherent gate report is **registration,
> not production**, for a substantial share of the evidence. Two gates are now
> within reach rather than blocked —
>
> - **`packages`** needs `clean-install`, `compatibility-matrix`, `crate-archive`,
>   `npm-archive`. rc.2 supplies the first two at `d16c0dda`; the other two are
>   not emitted by the release run and would have to be produced at that revision.
> - **`pytorch-hf`** needs eight kinds. rc.2 supplies **five** of them
>   (`installed-qat-tutorial`, `frontend-lifecycle`, `export-reload`,
>   `observability`, `api-signature`). Of the remaining three,
>   `torch-dispatch-overhead` and `torch-dispatch-cuda` are producible on this
>   box's GPU, and only `distributed-training` needs hardware we lack — which
>   makes this gate a **GPU-rental away from PASS**, not blocked.
>
> Every artifact above is still downloadable (`expired=false`), and each release
> regenerates them, so nothing here is time-critical.

**15 of the 38 evidence kinds have been produced. 23 have not.**
The 15 are `api-signature`, `clean-install`, `crate-archive`, `cuda-training`,
`estimator-validation`, `export-reload`, `frontend-lifecycle`,
`installed-qat-tutorial`, `npm-archive`, `observability`, `oci-security-cpu`,
`oci-security-cuda`, `source-admission`, `torch-dispatch-cuda` and
`torch-dispatch-overhead`. The last two OCI ones were produced on 2026-09-03
against the existing rev `3e07eabb` archives, and both pass with **zero
high/critical vulnerabilities and zero secret findings**. The 23 that remain
appear in the "Missing kinds" column below; the two lists sum to the 38 that
`GATES` requires.

### Reproducing the OCI security receipts

Three things are non-obvious enough to be worth recording, because each one
costs an hour to rediscover:

1. `docker load` **rejects** these archives. They are pure OCI layout
   (`index.json`), not Docker format (`manifest.json`).
2. **podman cannot be used as the bridge.** It reads `oci-archive:` and keeps
   the digest locally, but every push recompresses blobs through its storage and
   changes the manifest digest — with `--format oci` too. `regctl image copy
   "ocidir://<extracted>@sha256:<digest>" localhost:5000/<repo>:<tag>` preserves
   it exactly.
3. The digest to match is **not** the index digest. `verify-oci-archive.py`
   prints the child image-manifest digest; the archive's `index.json` entry is
   the index above it. `docker pull <repo>@<child digest>` is what puts the value
   `qualify-oci-runtime.py` demands into `RepoDigests`.

`qualify-oci-security.py` also expects the Trivy vulnerability database to
already exist as an ordinary file — run `trivy image --download-db-only
--cache-dir <dir>` first; it will not fetch one for you.

| Gate | Status | Missing kinds | What the missing kinds require |
|---|---|---|---|
| `qwen-source-admission` | EVIDENCE | — | — |
| `packages` | PARTIAL | `compatibility-matrix` | **Not blocked — CI produces this on every release and the release workflow now carries it into the payload.** The rc.2 `abi3-compatibility-receipt` passes `aggregate-wheel-smoke.py`'s own validator: `tritium.abi3-matrix-qualification.v1`, bound to `d16c0dda`, `passed: true`, 16 cells spanning CPython 3.9.25–3.14.7 across three platforms and three distinct wheels. Harvesting it advances the union 15→16. It does **not** by itself close the gate: crate, npm, clean-install, and compatibility receipts still must bind one exact revision before a coherent `packages` PASS. For local candidate `6723dcda`, the PR-150 wheels run (`37157008596`) records the GitHub merge commit `16fff0db` (base plus PR head) in its receipt, while the candidate manifest binds only PR head `6723dcda`; that matrix is not admissible to this candidate. |
| `pytorch-hf` | PARTIAL | `distributed-training` | Two or more GPUs. |
| `native-backends` | PARTIAL | `backend-manifest`, `performance` | All seven trace families, in order — `FAMILIES = ("cpu", "cuda", "rocm", "metal", "wgpu", "wasi", "mcu")`. Needs AMD *and* Apple *and* an MCU board. |
| `estimators-refinement` | PARTIAL | `refinement`, `baseline-ablation` | Separate local SALT campaign runs and baseline ablations; no current receipt is registered for these kinds. |
| `flagship-qwen` | **NOT RUNNING — canonical record says stalled** | `conversion-refinement`, `quality`, `task-retention`, `runtime`, `physical-bytes` | The durable workspace `/mnt/4tb/tritium-qwen36-campaign-20260813` exists and the 2026-10-04 canonical probe found 0/506 published masters, no seal, and one dead incomplete staged record. The 506 legacy captures still lack an admitted capture-binding receipt for the approved pack. Complete the Stage 7 recipe freeze and establish capture provenance before any fitting resume. |
| `stage7-freeze` | NONE | `stage7-recipe-freeze` | Complete the 1.7B recipe freeze before unsealing/running the pinned Qwen flagship, as required by plan 0043. |
| `onnx` | NONE | `onnx-inference` | Whole-Qwen ONNX execution traces — downstream of the flagship artifact. |
| `browser` | **UNREGISTERED FRAGMENTS** | `browser-conformance` | Chrome and Firefox traces exist for source `7523eb94`, but no combined receipt is registered. **Three** lanes are required: `--chrome-lane`, `--firefox-lane`, `--safari-lane`. Safari is gated on macOS and needs Apple hardware. |
| `serving` | PARTIAL | `oci-runtime-{cpu,cuda}`, `serving-deployment-{cpu,cuda}` | Both `oci-security-*` kinds are **done** (2026-09-03). The remaining four need an admissible serving bundle, which is not available. The loader computes `manifest_package_id` from exact `tritium.json` bytes (raw lowercase BLAKE3); SALT, preserved, and config packages use domain-separated `trp1_…` IDs. ADR 0033 now records that distinction, and local startup/OCI validators plus regression tests enforce it. This fixes the identified software contract mismatch, but no production runtime or deployment receipt has been produced from the current source; serving remains open. Deployment additionally needs Kubernetes, a Helm chart archive, and a `--bundle-manifest`. |
| `zoo-community` | NONE | `model-zoo`, `generated-claims`, `governance-docs` | All three come from **one** `qualify-zoo-community.py` call. It requires a `--governance-review` whose `independent_from_maintainers` field must be `True` (`verify-zoo-community-receipt.py:426-429`) and a named reviewer with an `organization` — i.e. a second person. It also requires four frozen model entries, the fourth being the flagship. Source-to-generated drift checks passed locally on 2026-10-04 (`generate-release-claims.py --check`, `generate-compatibility.py --check`; 9 focused tests passed), but these checks are not candidate-bound zoo/community receipts. |
| `reproduction-signoff` | NONE | `second-machine`, `independent-review` | A second machine, plus a reviewer whose identity differs from the reproduction operator. |

**Three kinds require a second person, not two.** `independent-review` and
`second-machine` are the obvious ones; `governance-docs` is the third, because its
attestation must assert `independent_from_maintainers`. And because
`qualify-zoo-community.py` emits its three receipts from a single call, that one
requirement holds `model-zoo` and `generated-claims` hostage alongside it. There
is no flag to produce one of the three alone.

Grouping the 23 remaining kinds by what actually unblocks them:

- **An admissible serving bundle** (13): the five `flagship-qwen` kinds,
  `stage7-recipe-freeze`, `onnx-inference`, `model-zoo` and `generated-claims`
  (coupled as described above), plus `oci-runtime-{cpu,cuda}` and
  `serving-deployment-{cpu,cuda}`. This is one dependency, not several — every
  one of them ultimately waits on the in-flight conversion producing a bundle
  whose manifest `tritium-serve` will admit. The two deployment kinds need
  Kubernetes on top of that.
- **The CPU, once the conversion frees it** (2): `refinement`,
  `baseline-ablation`. No new dependency.
- **Already produced by CI, awaiting registration** (1): `compatibility-matrix`.
  Corrected 2026-09-05 — this was previously grouped under hardware we lack,
  which was wrong. GitHub's macOS and Windows runners supply exactly the
  platforms this box cannot, the receipt is regenerated every release, and the
  rc.2 one validates today. Harvesting it closes the `packages` gate outright.
- **Hardware this project does not have** (4): `distributed-training` (≥2 GPUs,
  rentable), `browser-conformance` (macOS, for the Safari lane), and
  `backend-manifest` + `performance` (AMD + Apple + MCU).
- **A second person** (3, listed above), of which `second-machine` also needs a
  second machine.

Note that `release/v1.1/` is git-ignored, so this evidence exists only on the
machine that produced it. It is neither backed up nor independently reviewable,
which is a distinct risk from the gates themselves.

## Candidate layout

Place exact unpublished payloads and their generated SBOMs below the ignored
`release/v1.1/` directory. Every CycloneDX SBOM must set
`metadata.component.bom-ref` to the artifact ID used below. Its root component
must also bind the exact artifact filename, byte count and SHA-256 through
`tritium:artifact:file`, `tritium:artifact:bytes` and `hashes`. SPDX documents
use their document `name` as the artifact ID and must describe exactly one
package whose `packageFileName` and SHA256 checksum bind the artifact. Inputs
live outside the candidate directory because candidate admission rejects every
unmanifested file.

```json
{
  "schema": "tritium.release-inputs.v1",
  "release": "1.1.0-rc.2",
  "source_revision": "FULL_40_CHARACTER_GIT_REVISION",
  "builder": {
    "id": "https://github.com/OWNER/REPOSITORY/actions/workflows/release.yml",
    "build_type": "https://tritium.ai/build/package/v1",
    "invocation_id": "EXACT_WORKFLOW_RUN_ID"
  },
  "artifacts": [
    {
      "id": "pytritium-linux-cpu",
      "kind": "python-wheel",
      "path": "pytritium-1.1.0rc2-cp39-abi3-manylinux_2_28_x86_64.whl",
      "sbom": "pytritium-linux-cpu.cdx.json"
    }
  ]
}
```

## Assemble and verify

```bash
RUSTC_WRAPPER= cargo build --release -p tritium-cli
export TRITIUM_BIN="$(cargo metadata --no-deps --format-version 1 \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/release/tritium"
test -x "$TRITIUM_BIN"
python scripts/assemble-release-candidate.py \
  --inputs /tmp/tritium-v1.1-inputs.json \
  --output release/v1.1/manifest.json \
  --digest-tool "$TRITIUM_BIN"
scripts/release-status \
  --candidate release/v1.1/manifest.json \
  --digest-tool "$TRITIUM_BIN"
```

The build uses an empty `RUSTC_WRAPPER` to avoid inheriting a stale local
`sccache` setting. Keep this shell open for the later commands below: they use
`$TRITIUM_BIN`, which is derived from Cargo's configured target directory rather
than assuming the binary lives under the repository's `target/` directory.

Assembly sorts artifacts by ID, streams SHA-256/BLAKE3 through shipped CLI,
writes canonical in-toto/SLSA v1 statements, fsyncs data and directories, then
strictly reloads generated candidate. It never overwrites existing manifest or
provenance. For `model-bundle`, `onnx-bundle`, and `helm-chart` inputs, a missing
named SBOM is generated before provenance publication; a supplied SBOM must
reproduce the same canonical inventory exactly. Failure removes every SBOM and
metadata file created by that assembly attempt.

Verification requires:

- canonical `1.1.0-rc.N` version matching Cargo, Python, npm and compatibility
  mirrors;
- clean checkout at exact source revision;
- ordinary contained files with no symlink traversal or unmanifested payload;
- exact bytes, SHA-256 and BLAKE3 for every artifact;
- digest-bound CycloneDX/SPDX SBOM tied to exact artifact ID;
- digest-bound in-toto/SLSA v1 provenance tied to artifact SHA-256, source
  revision and builder identity.

For the browser package, build the exact npm archive first, then generate its
closed CycloneDX inventory from archive bytes. The generator rejects path
traversal, links, duplicate members, package identity drift and unsafe archive
topology; every regular member is hashed and linked from the root component:

```bash
npm pack ./packages/tritium-web --pack-destination release/v1.1
python scripts/generate-npm-sbom.py \
  --archive release/v1.1/tritium-ai-web-1.1.0-rc.2.tgz \
  --artifact-id tritium-web-node22 \
  --source-revision "$(git rev-parse HEAD)" \
  --output release/v1.1/tritium-web-node22.cdx.json
```

Npm archive qualification remains separate: an SBOM proves package bytes and
topology, while browser training and offline-install receipts prove runtime
behavior.

Canonical flat `model-bundle` and `onnx-bundle` tar archives get complete member
inventories through `scripts/generate-bundle-sbom.py`. Canonical input uses
POSIX ustar as `.tar` or one zstd-compressed `.tar.zst`/`.tzst`; auto-detected
or mislabeled compression is rejected. The generator rejects
links, symlinked or replaced parent paths, directories, path traversal,
duplicate portable names, unexpected files, nonzero trailing payload,
manifest byte-ledger/digest drift and source mutation. Each member is streamed
through `tritium release digest-stream`; ONNX BLAKE3 values and model profile,
preserved-tensor and Hugging Face package IDs must match exact archived bytes.
Lineage authority comes from the separate ONNX inference qualification receipt;
the SBOM generator does not promote self-described lineage strings to evidence.
`.tar.zst` requires the `zstd` decoder. Example:

```bash
python scripts/generate-bundle-sbom.py \
  --artifact release/v1.1/qwen-onnx.tar.zst \
  --artifact-id qwen-onnx \
  --kind onnx-bundle \
  --source-revision "$(git rev-parse HEAD)" \
  --digest-tool "$TRITIUM_BIN" \
  --output release/v1.1/qwen-onnx.cdx.json
```

Helm candidates use Tritium's source-closed packager, not ambient `helm package`
defaults. It emits one deterministic gzip stream containing a sorted POSIX
ustar `tritium/` tree with canonical modes, owners, timestamps, padding, and no
directory or link members. `Chart.yaml` must bind exact candidate release
through both `version` and `appVersion` and retain frozen chart API, Kubernetes
floor, and receipt-schema annotations. Source symlinks, duplicate portable
paths, mutation, oversized input, output overwrite, noncanonical gzip, trailing
streams, unsafe tar paths, links, and metadata drift fail closed.

```bash
python scripts/package-helm-chart.py \
  --source deploy/helm/tritium \
  --release 1.1.0-rc.2 \
  --output release/v1.1/tritium-1.1.0-rc.2.tgz
python scripts/generate-deployment-sbom.py \
  --artifact release/v1.1/tritium-1.1.0-rc.2.tgz \
  --artifact-id tritium-helm \
  --kind helm-chart \
  --release 1.1.0-rc.2 \
  --source-revision "$(git rev-parse HEAD)" \
  --digest-tool "$TRITIUM_BIN" \
  --output release/v1.1/tritium-helm.cdx.json
```

Helm CycloneDX inventory binds exact compressed bytes and every chart member's
SHA-256, raw BLAKE3, transport package ID, and byte count. Candidate admission
regenerates whole document independently. Embedded or root-only chart metadata
cannot substitute for complete archive inventory.

`scripts/build-oci-candidate` emits `<archive>.cdx.json` after its independent
OCI verifier passes. The same deployment generator accepts `--kind oci-image`.
It binds exact transport-tar SHA-256/bytes and inventories every layout, index,
manifest, config, layer, and attestation blob through SHA-256, raw BLAKE3,
transport package ID, and byte count. Admission requires a closed descriptor
graph with no unreferenced blobs, one Linux/amd64 image, hardened runtime labels
and identity, and image-manifest-bound semantic SPDX plus SLSA v1 statements.
OCI builds require `TRITIUM_OCI_BUILDER_ID` as a safe HTTPS identity. Admission
checks BuildKit `mode=max` structure, exact source-revision build argument,
resolved dependencies, LLB definition, builder/invocation identity and a
non-empty SPDX package inventory; predicate URLs or empty objects cannot satisfy
those gates. Candidate admission carries embedded BuildKit builder identity
through archive, build receipt, SBOM and outer SLSA external parameters, and
carries embedded BuildKit invocation identity through archive, SBOM and those
external parameters. Outer SLSA run details retain the distinct release-packaging
workflow identity. Hidden
PAX/GNU tar extension records are rejected rather than omitted from transport
inventory. Build output is staged, verified and atomically published. Compressed,
unsafe, linked, duplicate, corrupt, unaligned, trailing, unbound, or mutated
archives fail closed. Candidate assembly generates a missing OCI SBOM and
candidate admission regenerates the whole document; embedded BuildKit
attestations alone cannot substitute for exact transport inventory. This closes
SBOM infrastructure, not physical image/runtime/security qualification or
publication.

OCI build receipts use `tritium.oci-build.v2` when deployment artifacts are
added after the package set is frozen. The receipt retains the exact manifest
hash used to build the image and binds a canonical `package_inventory_sha256`
over every non-deployment artifact. Final candidates can therefore add their
OCI image and Helm chart without creating a circular manifest hash, while any
package, path or byte drift still fails closed. `tritium.oci-build.v1` remains
readable only when its exact manifest hash still matches.

## Aggregate evidence status

An evidence registry lives outside the candidate directory, whose closed file
allowlist remains unchanged. It binds the exact candidate-manifest SHA-256 and
references only validated receipt schemas and candidate artifact IDs. Admitted
empirical kinds include artifact-bound CUDA fp16 training and installed-wheel
clean-install lifecycle receipts. Each binds source/release/run/machine identity,
exact wheel bytes and frozen operation coverage. Unrecognized or self-asserted
kinds fail closed.

Python abi3 matrix qualification is separate evidence: one content-addressed,
run-bound receipt must contain every admitted CPython/platform cell, reuse one
exact wheel per target, and match Linux, Windows and macOS wheel identities in
candidate manifest. Matrix evidence cannot substitute for local crate/npm/image
archives.

Rust archive qualification consumes exact candidate-version `.crate` set from
one clean revision. Every archive must have safe topology, matching
`Cargo.toml.orig`, clean `.cargo_vcs_info.json`, and exact bytes. Harness extracts
all archives outside source checkout, patches internal registry dependencies to
those extracted packages, stages exact `Cargo.lock` dependencies with
`cargo vendor --locked`, then uses empty `CARGO_HOME` for locked
`cargo check --offline --all-targets` across every library-bearing package.
Registry requires receipt inventory equal
candidate `rust-crate` inventory. Npm archive qualification remains independent.

Serving qualification is split by flavor. `serving-deployment-cpu` and
`serving-deployment-cuda` each anchor one candidate OCI image plus the candidate
Helm chart. Place the exact bundle manifest and OCI build receipt named and
hashed by each deployment-v2 receipt beneath the evidence-registry directory.
The deployment entry must name exactly the matching `oci-runtime-*` and
`oci-security-*` receipt IDs as parents. Registry validation replays the full
offline deployment validator, requires all three receipts to bind the same
candidate image, and requires the runtime and Kubernetes startup receipts to be
exactly equal. CPU evidence cannot satisfy the CUDA deployment gate.

```bash
scripts/release-status \
  --candidate release/v1.1/manifest.json \
  --registry release/v1.1-evidence/registry.json \
  --json-output release/v1.1-evidence/status.json \
  --digest-tool "$TRITIUM_BIN"
```

The ADR 0033 gate list is compiled into the status tool rather than supplied by
the registry. Empty and partial registries therefore enumerate `MISSING` gates;
one valid CUDA receipt cannot green the broader native-backend gate, and one
functional wheel or compatibility matrix cannot replace complete local-archive
evidence.

PyTorch dispatcher evidence is intentionally split. `torch-dispatch-overhead`
binds exact installed-wheel CPU forward/backward overhead distributions to the
five-percent policy. `torch-dispatch-cuda` binds the exact CUDA wheel, committed
dispatcher test source, physical GPU identity, all seven native CUDA cases, and
compute-sanitizer JUnit/log bytes with one zero-error summary. Both kinds are
required; CUDA training evidence cannot substitute for dispatcher residency,
tail, cache-lifetime, stream-ordering, or memcheck coverage.
Public activation is always `EXTERNAL_AUTH_REQUIRED` and is not inferred from
local evidence.

### Exact-source wheel CI evidence — 2026-10-04

GitHub Actions wheel run [37220268418](https://github.com/Quitetall/tritium/actions/runs/37220268418)
completed successfully from branch source revision
`6ad03ea7fdc9731a428e2f94bdd3a244bd81bb01` (not the synthetic pull-request
merge tree). It built Linux x86_64, Windows x64, and macOS arm64 CPU wheels;
the source-free tutorial and installed PyTorch 2.11 / Python 3.13 functional
job also passed. The Linux wheel SHA-256 was
`7fbceb2182d0e6b418222cba54a1ed757bbb277a05aa607f0e96612457c0e3cd`.

The run's ABI3 matrix passed for CPython 3.9–3.14 on Linux x86_64, Windows
x64, and macOS arm64. Its matrix receipt is
`sha256:2a87b3316fb9431e80e6a402497bcfb1419c896df9a57db7aa7cd8fd187ef730`
(`github-37220268418-1-abi3-matrix`). The installed-wheel functional receipt
is `sha256:2928c47113038399cd579cb6d7f01f2ab899f039291b9d0587d56d2d7b34e850`
(`github-37220268418-1-cpu-functional`); it covers native ternary matmul,
QAT forward/backward, optimizer update/resume, Hugging Face safetensors
save/reload, and tied-weight identity. These are exact-source package checks,
not a flagship-model quality result.

The installed-wheel PyTorch dispatch-overhead receipt
`sha256:5140cf6a73f0d1f739cf5e634fbdb451c8868b22dcc88987a2fcc3474120b3ec`
(`github-37220268418-1-torch-dispatch-overhead`) independently verified
against the exact wheel and source. All six decode, microbatch, and prefill
forward/backward cases passed the 5% ceiling; the largest measured bootstrap
upper ratio was `1.00925`. This is CPU wrapper-overhead evidence on a four-vCPU
AMD EPYC 7763 runner. It does not qualify CUDA dispatch or replace the separate
physical-GPU `torch-dispatch-cuda` receipt.

The CUDA wheel lane was skipped because this was a pull-request run. The
workflow receipts are not yet a complete v1.1 candidate registry: crate/npm
archives, physical-GPU and other hardware gates, model-quality evidence,
second-machine reproduction, independent release review, and explicit human
activation remain separate requirements. Other required checks for the pull
request were still running when this record was written.

### Serving deployment contract checks — 2026-10-04

At source revision `34acd332e5bce398d1c59324108e6fc7feb58047`, the focused
deployment contract suite passed 117 tests and 61 subtests:

```text
pytest -q scripts/tests/test_deployment_contract.py \
  scripts/tests/test_qualify_kubernetes_deployment.py \
  scripts/tests/test_qualify_oci_security.py \
  scripts/tests/test_qualify_oci_runtime.py \
  scripts/tests/test_verify_oci_archive.py
117 passed, 61 subtests passed
```

`./scripts/check-deployment-manifests` also passed using the repository's
digest-pinned Helm 3.18.4 container with networking disabled and image pulls
disabled. It linted the chart, rejected CUDA configuration without GPU
resources, and rendered the default and CUDA/KEDA/ServiceMonitor configurations
with the required CUDA `Recreate` strategy. These are contract and chart
rendering checks only: no Kubernetes cluster, schema-v3 model bundle, OCI
runtime, NVIDIA deployment, autoscaling event or rollback was exercised.
Plan 0052's empirical serving and deployment gates remain open.

## Local sign-off

Evidence readiness and maintainer sign-off are separate layers. A complete
registry produces `LOCAL_RC_EVIDENCE_READY_UNSIGNED` with exit status 2; it does
not produce `LOCAL_RC_READY`. Seal that exact canonical report with an SSH
signing key. Sign-off re-runs canonical candidate admission, including every
artifact identity, SBOM, provenance and closed-directory check, through the
same digest tool used for release status. Then verify it against a reviewed
`allowed_signers` file:

```bash
python scripts/local-rc-signoff.py seal \
  --report release/v1.1-evidence/status.json \
  --registry release/v1.1-evidence/registry.json \
  --candidate release/v1.1/manifest.json \
  --digest-tool "$TRITIUM_BIN" \
  --principal release-maintainer --key /secure/release-key \
  --output release/v1.1-evidence/signoff.json
python scripts/local-rc-signoff.py verify \
  --report release/v1.1-evidence/status.json \
  --registry release/v1.1-evidence/registry.json \
  --candidate release/v1.1/manifest.json \
  --digest-tool "$TRITIUM_BIN" \
  --principal release-maintainer \
  --statement release/v1.1-evidence/signoff.json \
  --signature release/v1.1-evidence/signoff.json.sig \
  --allowed-signers /secure/tritium-release-allowed-signers
```

Before sealing, registry must contain admitted
`tritium.second-machine-reproduction.v1` and
`tritium.independent-release-review.v1` receipts. Independent-review entry must
parent every other registry receipt and list same IDs in
`reviewed_receipt_ids`; reviewer and reproduction operator identities and
organizations must differ. Copied primary-host results or reviewer transport
failure remain blockers, never passing evidence.

The statement binds candidate-manifest, registry and report SHA-256 identities,
release revision and signer principal. Any evidence change invalidates it. Key
generation, signer authorization and the local tag remain explicit maintainer
actions; no publication or tag push is inferred.

### Portable training contract CPU checks — 2026-10-04

At source revision `1cb50800ceebeb02d9262ebaaee4fef6848e3b4a`, focused local
contract checks passed:

```text
cargo test --locked -p tritium-spec --test train_backend_contract --test training_vectors
4 backend-contract tests passed; 6 vector/schema tests passed
cargo test --locked -p tritium-train --test portable_vectors
5 tests passed, including canonical V2 and V3 CPU corpus replay
PYTHONPATH=crates/tritium-py/python pytest -q crates/tritium-py/tests/test_training_manifest.py
10 passed
```

These results check the CPU reference and cross-language Python manifest
reader. They do not qualify an accelerator, browser runtime, physical backend,
performance, or release candidate. The planned TypeScript check
(`npx tsc -p bindings/typescript/tsconfig.json --noEmit`) could not run because
the TypeScript compiler is not installed in this checkout; `npx` reported that
it would not supply the missing compiler implicitly. No dependency was
installed. The language-parity and physical-backend gates therefore remain
open.

### Estimator and refinement CPU regression checks — 2026-10-04

At working revision `361b9b08aee5e4376a3d70453934dbeed129dd69`, the local
estimator, refinement-core and PyTorch stage-4 regression suites reported:
User-owned EAT-O files were modified in the worktree during these runs; these
suites do not cover those files.

```text
PYTHONPATH=crates/tritium-py/python pytest -q -rs \
  crates/tritium-py/tests/test_estimator_catalog.py \
  crates/tritium-py/tests/test_refinement_core.py \
  crates/tritium-py/tests/test_torch_stage4.py
28 passed, 7 skipped
```

All seven skips were optional LamQuant codec-neural or BLUT-LAMU integrations
not installed on this machine. These CPU/synthetic tests cover estimator
projection, gradient, conversion, refinement, and adapter-parity behavior;
they do not measure a large-model quality gain or qualify Qwen PTQ/refinement,
accelerator execution, or release readiness. The absent integrations remain
unverified rather than passing.

### TypeScript and generated-web-source check correction — 2026-10-04

The previous note that the TypeScript compiler was unavailable was too broad:
the compiler exists under `packages/tritium-web/node_modules`, but is not on
the repository-root `npx` path. At revision
`0fd7685c26b687f3e846237a23c39adfb2108d9b`, these direct checks passed:

```text
packages/tritium-web/node_modules/.bin/tsc -p bindings/typescript/tsconfig.json --noEmit
packages/tritium-web/node_modules/.bin/tsc -p packages/tritium-web/tsconfig.json --noEmit
```

The generated-web-source check also passed as the first stage of
`npm --prefix packages/tritium-web run typecheck`. That combined command then
stopped at `build:wasm`: the script requires a completely clean Git worktree,
and this checkout contains unrelated user-owned EAT-O changes. It did not
compile the WASM guest or reach its own TypeScript stage. `wasm-bindgen` is not
installed on this machine, so the complete WASM build remains unverified. The
direct TypeScript passes do not close browser or physical WebGPU gates.

### Clean-worktree web package recheck — 2026-10-05

The earlier WASM limitation is now superseded for the current pushed source
revision `fd927eab0b3db893f3fe59e272907d9e026125e6`. A clean detached worktree
passed the full `npm --prefix packages/tritium-web run check` with exit code 0:
generated-file checks, pinned release WASM build, strict TypeScript, 145 Node
tests, and offline archive verification. `wasm-bindgen 0.2.126` is installed
under the isolated local cache path
`/home/brianklam/.cache/tritium-release-tools/bin`; unrelated dirty EAT-O files
in the main checkout were not included.

The retained local receipt
`release/v1.1/evidence/npm-archive-fd927eab/npm-archive-receipt.json` has ID
`sha256:84349d4888e0dd92ab65668f9535d736500a7f4f0a245ded48f3b4b58de1e3f5` and
independently validates against the adjacent 627,849-byte archive
`tritium-ai-web-1.1.0-rc.2.tgz` (SHA-256
`e556029b3fc5531abf265d3bd2b556c722ae60ce3488b61c1b31bcf4939a8fbd`). Its
CycloneDX SBOM is retained in the same directory. This remains exact-revision
local package evidence, not candidate-registry admission, a cross-platform
package matrix, or physical browser/WebGPU qualification; regenerate it for a
later source revision.

### PyTorch and browser-source software checks — 2026-10-04

At code baseline `0fd7685c26b687f3e846237a23c39adfb2108d9b`, the broader local
reference suite reported:

```text
PYTHONPATH=crates/tritium-py/python pytest -q -rs \
  crates/tritium-py/tests/test_torch_api.py \
  crates/tritium-py/tests/test_autograd.py \
  crates/tritium-py/tests/test_hf_lifecycle_receipt.py
24 passed, 2 skipped
npm --prefix packages/tritium-web run check:generated
passed
```

Both skips require the installed `pytritium` wheel; no local wheel was
qualified by this run. The two TypeScript projects pass direct `tsc --noEmit`
checks as recorded above. This verifies local reference/package source
behavior only; the full wheel build/install, WebAssembly build and physical
browser WebGPU checks remain separate gates.

### Hosted CI and wheel matrix — 2026-10-04

For pushed revision `93ec04188bab40fc0b9308507f6e049c0ac9ca03`, hosted checks
completed without failures:

| Workflow | Result | Scope limit |
|---|---:|---|
| CI (`37258090450`) | 19 passed, 7 skipped | CUDA/ROCm/Metal/wgpu hardware, fuzz, performance, and real-model serving jobs were skipped |
| Wheels (`37258090477`) | 22 passed, 1 skipped | CUDA wheel job skipped; CPU wheel builds and installed-wheel/tutorial checks passed |
| CodeQL (`37258090452`) | passed | Static analysis only |
| Docs (`37258090451`) | passed | Documentation build/link checks |
| Capstone CPU smoke (`37258090459`) | passed | CPU E2E smoke; not a GPU or model-quality result |

The CI run covered CPU Linux/macOS/Windows checks, GPU-feature compilation,
WASI conformance, API stability, packaging readiness, compatibility/community
contract checks, serving contract mocks and the source-free web archive. These
hosted checks improve release confidence but do not close the skipped physical
backend, real-model, performance, or model-quality release gates.

### Hugging Face QAT/PTQ and distributed CPU integration — 2026-10-04

At code revision `d80c8a8255eac2015d26c69cf28fd83572e803bb`, local CPU tests
passed for the tiny randomly initialized Llama fixture:

```text
PYTHONPATH=crates/tritium-py/python pytest -q -rs crates/tritium-py/tests/test_huggingface_qat.py
7 passed
PYTHONPATH=crates/tritium-py/python pytest -q -rs \
  crates/tritium-py/tests/test_huggingface_distributed.py::test_two_rank_cpu_ddp_step_and_checkpoint \
  crates/tritium-py/tests/test_huggingface_distributed.py::test_two_rank_cpu_fsdp_step_and_sharded_state_resume \
  crates/tritium-py/tests/test_huggingface_distributed.py::test_accelerate_cpu_bf16_in_fresh_runtime
3 passed
```

These checks cover QAT and PTQ save/reload, tied embedding/head storage,
Trainer checkpoint resume, Accelerate state/RNG resume, and two-rank CPU/Gloo
DDP/FSDP semantics. They use a synthetic tiny model and are not Qwen quality or
performance evidence. Accelerator DDP/FSDP on two distinct physical GPUs,
CUDA-specific qualification, and the known PyTorch CPU full-state-export issue
remain open. User-owned EAT-O files were modified during these tests and were
not part of their scope.

### Exact-wheel CUDA/fp16 smoke — 2026-10-05

At pushed source revision `dc4ea62b792d159cc1785258ae9d9baca692f0c2`, the
exact Linux x86-64 CPU wheel from hosted workflow `37259200813` was installed
in an isolated virtual environment and exercised on the physical RTX 4090 by
`crates/tritium-py/tests/hf_cuda_worker.py`. The installed `tritium` import
resolved inside that virtual environment, not from the source checkout. The
worker completed five fp16 training steps on a one-layer randomly initialized
Llama fixture, observed zero host transfers in the profiled ternary operator,
and exactly restored the saved Accelerate checkpoint. The receipt is
`/mnt/2tb/tritium-release-evidence/hf-cuda-dc4ea62-20261004/cuda-training-receipt.json`
(`sha256:aa999302ab9349c89ca74fb4696998bc86e8a46c7b1496a2246e9cecaf929431`),
verified with `scripts/verify-cuda-training-receipt.py` against the exact
wheel SHA-256 `cfde3d156cc2dfcd57e92f66f6493c4e0f9f82153ac4381c3fc19211ee301cc3`.
The five measured steps took 62.55 ms; this tiny synthetic smoke is not a
training-performance claim, multi-GPU qualification, Qwen quality result, or
candidate-wide release receipt. The temporary venv and checkpoint were used
only for this run and are not release evidence.

### PTQ public `convert()` artifact seam — 2026-10-07

At pushed source revision `0ac608ab94f4c37e5dd85da7dd89245a06c8d466`, the
focused source-tree PTQ artifact suite passed:

```text
PYTHONPATH=crates/tritium-py/python pytest -q -rs crates/tritium-py/tests/test_ptq_artifacts.py
37 passed in 3.08s
```

This verifies the public `prepare`/`calibrate`/`convert()` artifact path chosen
for the PTQ test seam. It is a local CPU software test, not a real-checkpoint
quality, performance, wheel-install, or release-receipt result. The clean-tree
web package check passed generated-file validation and compiled the Rust WASM
guest, but stopped because the pinned `wasm-bindgen 0.2.126` executable was not
available on this host's `PATH`; TypeScript, package tests, and archive
verification therefore remain unverified locally for this checkout.

The current Hugging Face and ONNX paths also passed focused CPU checks:

```text
PYTHONPATH=crates/tritium-py/python pytest -q -rs crates/tritium-py/tests/test_huggingface_qat.py
7 passed in 1.62s
PYTHONPATH=crates/tritium-py/python /tmp/tritium-onnx-check/bin/python -m pytest -q -rs \
  crates/tritium-py/tests/test_module_onnx.py::test_public_facade_executes_qat_ptq_and_refinement_artifacts_in_ort
1 passed in 6.19s
```

The ONNX test used an isolated Python 3.14 environment with the CI-pinned
`onnx==1.22.0`, `onnxruntime==1.27.0`, and `onnxscript==0.7.1`, while reusing
the host's PyTorch installation. These are small-model software checks; they
do not establish flagship quality, hardware performance, or release readiness.

### Portable web package source build and archive — 2026-10-07

At clean source revision `09f2e9021fca33163f98d33bb3cee45057203c21`,
`packages/tritium-web` passed its complete local check in a detached worktree:

```text
npm run check
generated-file checks: passed
wasm32-unknown-unknown release build and wasm-bindgen: passed
strict TypeScript: passed
Node tests: 145 passed, 0 failed, 0 skipped
source-free npm archive verification: passed
```

The durable output is `/mnt/2tb/tritium-release-evidence/npm-web-09f2e902/`:
`npm-archive-receipt.json` has receipt ID
`sha256:09826a94ef97a870cdd48b5eb5efb909a0c06ec99ed877536e3b1497e70bef2d`,
and binds the 627,707-byte `tritium-ai-web-1.1.0-rc.2.tgz` with SHA-256
`7a852ad0ca43d5998c0aeb3d9686ff0ec4b0a602261e13c2599d0c808959de47`.
The SBOM is `tritium-web-node22.cdx.json`. The build used Node 24.21.0/npm
12.0.2 and the upstream checksum-verified `wasm-bindgen 0.2.126` Linux binary.
This qualifies the local source-free package path only; it is not physical
Chrome/Firefox/Safari WebGPU evidence, a browser performance result, or
candidate-registry admission.

### Candidate-source Python and release-admission regression checks — 2026-10-07

At source revision `6ac1c040d3908a39301e437088a0052812770ec9`, the complete
Python binding suite passed on the local CPU environment:

```text
PYTHONPATH=crates/tritium-py/python pytest -q -rs crates/tritium-py/tests
377 passed, 27 skipped, 15 warnings in 223.15s
python -m unittest scripts.tests.test_release_evidence_status scripts.tests.test_release_status -v
Ran 44 tests in 3.056s — OK
```

The Python skips were explicit: ONNX was not installed in the host environment,
seven checks require an installed candidate wheel, eight checks require the
CUDA-enabled Tritium extension, and the remaining checks require external
LamQuant/BLUT checkouts. They are not release passes. Separately, hosted CI run
[`37588299083`](https://github.com/Quitetall/tritium/actions/runs/37588299083)
completed successfully for this source revision; its CUDA, ROCm, wgpu, Metal,
physical-model-serving and performance jobs were skipped on hosted runners.

Existing local RC manifests are still bound to older source revisions, and the
local release CLI is not installed on this host. These software regressions do
not qualify candidate artifacts, physical backends, flagship model quality,
performance, or v1.1 release readiness.

### Current-source Linux CPU abi3 wheel — 2026-10-07

At exact source revision `671cef6698717a5da7a66607f2493d02525348f3`, a CPU
abi3 wheel was built from a clean detached worktree with the pinned
`manylinux_2_28_x86_64` image, Rust 1.98.0, and maturin 1.10.2. The durable
artifact and receipts are under
`/mnt/2tb/tritium-release-evidence/pytritium-cpu-671cef66/`.
`verify-wheel.py --install-smoke` passed; the 10,526,236-byte wheel SHA-256 is
`4082854acf38e1ffb4f7cf28178654e4ae7f06f446baef981a7b92f0415b2318`.

The exact wheel then passed the installed functional smoke on CPU under
CPython 3.14.7, PyTorch 2.11.0+cu130, Transformers 5.5.3, and safetensors
0.7.0. The receipt ID is
`sha256:6be8b85d8906e3974940ddf5c59e1cf7de3dfd8f4b55324de20b2ba0a2f66113`
(file SHA-256
`3db36d8083e4b73afd2ef80b648a230ca81b4814483c9913ac1288889fa6bdbe`). It
proves native CPU matmul, Hugging Face QAT forward/backward, optimizer update
and resume, safetensors save/reload, and tied-weight identity. The compatibility
receipt for this one Linux x86-64 CPython 3.14 cell is
`compatibility-receipt-cp314.json` (SHA-256
`6bfc30d333e1ae92fcb2d5a261e7280027271b2d4ecfa8f5e96b60aefd27aa53`). The
functional smoke used a venv with host dependencies visible; it proves exact
wheel loading and runtime behavior but is not a fully isolated dependency
installation. Other platforms/interpreters, CUDA, full package-matrix,
model-quality, and public-release gates remain open.

The wheel is now bound to an explicitly partial package-probe candidate at
`/mnt/2tb/tritium-release-evidence/pytritium-cpu-671cef66/candidate/manifest.json`
(manifest SHA-256
`6be0b70b10416dea9f917a7859bb26329711a9806ba7f01c727c6e94e605c796`). The
clean worktree's `release-status` check accepted the manifest as
`CANDIDATE_EVIDENCE_VALID`; its registry binds the functional receipt above to
the exact wheel. The durable report is
`/mnt/2tb/tritium-release-evidence/pytritium-cpu-671cef66/registry/report.json`
(SHA-256
`bce08e2fcd1a04835d80e67e4ad27fd06e9d035a691555800af2fb4c38932a3c`). It
correctly reports `LOCAL_RC_BLOCKED`: the `packages` row has only `clean-install`
and still misses `compatibility-matrix`, `crate-archive`, and `npm-archive`;
the other eleven release gates also remain missing. This one-wheel probe is not
the complete public RC and does not replace the prior candidate records.

### Current-source publishable crate archive qualification — 2026-10-07

From the same clean source revision `671cef6698717a5da7a66607f2493d02525348f3`,
the following command produced the crate archives:

```text
CARGO_TARGET_DIR=/mnt/2tb/tritium-crates-cargo-package-671cef66 cargo package --locked --workspace --no-verify --allow-dirty --target-dir /mnt/2tb/tritium-crates-cargo-package-671cef66
```

The command also packages the two workspace members marked `publish = []`; the
original output is preserved under
`/mnt/2tb/tritium-crates-cargo-package-671cef66/`. The exact 23 publishable
archives, excluding only those two non-published members, are retained under
`/mnt/2tb/tritium-crate-archives-671cef66/`.

`scripts/qualify-crate-archives.py` passed its exact inventory and source/VCS
checks, then built an isolated consumer with vendored dependencies and
`CARGO_NET_OFFLINE=true`:

```text
python scripts/qualify-crate-archives.py --archives /mnt/2tb/tritium-crate-archives-671cef66 \
  --source-revision 671cef6698717a5da7a66607f2493d02525348f3 \
  --release 1.1.0-rc.2 --run-id local-crates-671cef66-20261007 \
  --output /mnt/2tb/tritium-release-evidence/pytritium-cpu-671cef66/crate-archive-receipt.json
qualify-crate-archives: PASS: 23 crates
```

Receipt schema is `tritium.crate-archive-qualification.v1`, ID
`sha256:bbce29000f8a5a12add933e34d2bc7846596bd90d06d255f230c83270cfd05e6`,
file SHA-256
`8e75bce8c49195d156983531a71d53f91294d9748d28ba705c7832bd47b75443`. This
is a successful local offline crate-consumer check, but the receipt is not yet
registered against a candidate manifest; it does not close the package gate.

### Exact-source CI crate archive qualification — 2026-10-07

The successful CI run [37596503849](https://github.com/Quitetall/tritium/actions/runs/37596503849)
on source revision `2587652f2b7862c84b3a5f236d0c62860cfb60bb` uploaded the
`crate-candidates-2587652f2b7862c84b3a5f236d0c62860cfb60bb` artifact. It retains
the exact 23 publishable `.crate` files, their 23 CycloneDX documents, and the
offline consumer receipt. The receipt is
`sha256:ca07a50bbb5e32da291d9bd87b7d69ca7b7f6c7ae9717bd8ec8d9e192da8004d`
(`tritium.crate-archive-qualification.v1`, run
`github-37596503849-1-crate-archives`).

The receipt was independently revalidated against the downloaded archives and
the exact-source `Cargo.lock`; all 23 packages are recorded as compiled in the
offline, isolated Cargo-home consumer. All 23 SBOM roots also match their
archive SHA-256, byte count, release identity, and source revision. This is
candidate-revision package evidence, but it is not yet registered against a
complete candidate manifest and therefore does not close the aggregate package
gate.

### PTQ parallelism probe on a loaded workstation — 2026-10-07

The public `prepare`/`calibrate`/`convert()` parallelism probe was run against
source revision `44869608` and the installed `1.1.0rc2` abi3 wheel with SHA-256
`b21343f103aec89c0e843f72730dafb23dbdc49c639677e531d96f684888c94b`:

```text
PYTHONPATH=crates/tritium-py/python \
  /tmp/tritium-release-wheel-perf/bin/python -m unittest \
  scripts.tests.test_ptq_parallelism -v
serial_seconds=36.949 parallel_seconds=36.595 speedup=1.01x
FAIL: expected at least 1.5x speedup
```

The artifact identity and weighted-error assertions passed before the speedup
assertion failed. The workstation was heavily contended at the time: a
contemporaneous reading showed load average `35.90` with 32 CPUs available, and
multiple unrelated Rust builds were active. Treat this as a noisy local
performance result, not proof that parallel fitting regressed or that the
performance target passed. The test now prints CPU count, affinity, and load
average with its timings so a clean rerun can distinguish host contention from
a solver regression. The exact-source hosted tutorial run `37621206763` was
still active when this entry was recorded. It has now completed and failed the
300-second end-to-end limit at 1,088.645 seconds. Stage receipts in the job log
show conversion from 0.605 seconds after calibration to 864.038 seconds
(863.433 seconds of conversion), ONNX export in 127.408 seconds, and 224.607
seconds for the remaining measured work. Memory stayed available (at least
about 11 GiB in sampled logs) and temporary disk stayed above 80 GiB. This
confirms a CPU-time blocker rather than OOM or disk pressure. It does not isolate
a single solver hotspot within conversion; profiler-backed optimization and an
exact-candidate rerun are still required. The gate remains red and unchanged.

### Exact-source installed-wheel PTQ parallelism — 2026-10-07

On source revision `19656644ad3c26c9e4f6f3948ba08c304c6b4b24`, the exact
Linux CPU abi3 wheel passed the installed-wheel `qualify public PTQ row-fitting
parallelism` step in Actions run
[37626387409](https://github.com/Quitetall/tritium/actions/runs/37626387409)
(job `112810906163`). The test exercises public `prepare` → `calibrate` →
`convert()` and passed its assertions that serial and four-thread runs produce
the same algorithm ID, fitted-artifact digest, and weighted error, with at least
1.5× measured speedup. The pinned SmolLM2 CPU tutorial in the same workflow
finished with `1,081.175s` elapsed against the unchanged `300s` budget. Its
stage markers place public conversion at `853.208s` total elapsed, or about
`852.602s` after the `0.606s` calibration point. The prior exact-source run
`37621206763` recorded `863.433s` of conversion, so this run is roughly 1.3%
faster by wall clock on different hosted runners—not a statistically isolated
solver improvement and nowhere near sufficient for the five-minute gate. The
entire tutorial still fails the time gate. The separate general Python unit-test
job correctly skips the PTQ probe because PyTorch is absent there; the
installed-wheel lane is its authoritative execution environment. The passing
parallelism probe is a bounded row-fitting result, not a whole-model throughput
or release-performance claim.

The run's remaining measured stages were approximately `20.784s` for native
checkpoint round-trip, `43.831s` for generation, `129.405s` for ONNX export,
`26.648s` for ONNX replay, and `7.161s` from QAT reload start through resume.
Sampled available memory remained at least `12.2 GiB`, and runner temp disk
remained above `83 GiB`; this was a CPU-time failure, not OOM or disk pressure.
The exact-source workflow is
[37626387409](https://github.com/Quitetall/tritium/actions/runs/37626387409)
on revision `19656644ad3c26c9e4f6f3948ba08c304c6b4b24`.

### Public PTQ artifact-path regression — 2026-10-07

The chosen PTQ test seam is the public `prepare` → `calibrate` → `convert()`
artifact path, not a direct private solver call. On branch head
`8bef95b6fb7acc8320d77823d6ef001f4bb3a78f`, the focused source-tree test
`test_live_module_fit_consumes_bound_curvature_and_rejects_source_drift` passed
(1 passed, 36 deselected). It writes and reloads the conversion artifact,
checks fitted trits/scales and weighted error, exercises deterministic resume,
and rejects source-weight drift. This confirms the public artifact contract on
a small CPU fixture; it is not a model-level quality or performance result.

### Exact-source hosted SmolLM2 tutorial rerun — 2026-10-07

Actions run [37629720595](https://github.com/Quitetall/tritium/actions/runs/37629720595)
completed with only the pinned SmolLM2 CPU tutorial failing; the CI aggregate,
CodeQL, CPU wheels, abi3 matrix, publish readiness, docs, and source-free wheel
tutorial passed. The tutorial ran on exact source revision
`8bef95b6fb7acc8320d77823d6ef001f4bb3a78f` and finished at `690.001s` against
the unchanged `300s` budget. Stage markers show calibration at `0.441s`,
conversion at `552.095s` (about `551.654s` after calibration), native checkpoint
round-trip at `565.363s`, generation at `593.501s`, ONNX export at `665.686s`,
ONNX replay at `685.167s`, and QAT resume at `690.001s`.

Sampled memory stayed above about `12.8 GiB` and runner temp disk above `83.5
GiB`; this remains a CPU-time failure, not OOM or disk pressure. Conversion was
about 35% faster than the preceding hosted run's `852.602s`, but the code was
unchanged and hosted runner variation was not controlled. Treat this as noisy
measurement, not evidence of an optimization. The gate is still red by
`390.001s`, and profiler-backed optimization plus an exact-candidate rerun
remain required.

A second exact-source run, Actions run
[37632137262](https://github.com/Quitetall/tritium/actions/runs/37632137262),
again used revision `15bc7e25739a136637e14a5cb4a21e226491ff1c` (before the
local Python/Rust bridge edit described below). It failed the same tutorial
time gate at `1074.451s`; conversion completed at `852.649s`. Memory remained
above roughly `11.2 GiB` and runner temp disk above `81.8 GiB`, so this was
again a CPU-time failure rather than memory or disk exhaustion. The run's
conversion time differs substantially from the earlier exact-source run, so
the pair confirms that hosted CPU timing is variable; neither result alone
identifies a code regression or improvement. At the time of this run, the
local bridge edit had only been checked on a smaller deterministic fixture;
the later exact-candidate hosted run is recorded below.

The exact-candidate hosted run
[37637322050](https://github.com/Quitetall/tritium/actions/runs/37637322050)
then exercised commit `44942a8a156a3350cbbb613791ef1c09e6eb5e77`, including
the binary bridge. It completed the tutorial's functional stages but failed
the unchanged 300-second limit at `1033.296s`; conversion completed at
`805.704s`. Sampled available memory stayed above about `12.0 GiB` and runner
temp disk above `83.4 GiB`. This remains a CPU-time failure. Relative to run
`37632137262`, the observed conversion was about 47 seconds shorter, but the
hosted runs were not controlled or same-runner A/B comparisons. Do not
attribute that difference to the bridge or claim a speedup.

### Local dev-build PTQ profile — 2026-10-07

Commit `44942a8a156a3350cbbb613791ef1c09e6eb5e77` was profiled through the
public `prepare → calibrate → convert()` tutorial on the cached, pinned
`HuggingFaceTB/SmolLM2-135M-Instruct` revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. This used the checked-out Python
package and its locally built abi3 extension in
`/tmp/tritium-release-wheel-perf`, Python 3.14.7, PyTorch 2.11.0+cu130,
Transformers 5.5.3, CPU on an Intel i9-14900K host (32 logical CPUs). The
extension came from `maturin develop` without `--release`, so it used the
development profile, not the optimized wheel profile. The tutorial passed
functionally in `779.871s` with an explicit `max_seconds=1800`; this is not a
pass of the frozen 300-second hosted gate, a release-wheel timing result, or a
model-quality qualification. Its receipt SHA-256 is
`a980e19b48b46d94078222d56465cd69642b8e2106ebcd99f42e47a9657b3ff6`; the
selected dense/checkpoint byte counts were `537,919,488` / `92,192,265` (5.83x).

In this development-build trace, `convert()` took `559.044s` and the native
`fit_joint_ternary_diagonal` bridge was called 2,106 times. This identifies a
candidate area to inspect, but does not prove it is the optimized wheel's
dominant cost. Python-side timings are inclusive across the checkpointing call
tree and must not be summed. The host was concurrently loaded (1-minute load
average peaked near 70), so elapsed time is diagnostic, not a comparative
benchmark. Raw pstats remain at
`/tmp/tritium-ptq-profile-44942a8a.pstats` (SHA-256
`ad8871ac616d73ceb6f5b418f1bd255ed2efc75e73cdfcdc047e6a1cb7dc2f8b`); the
receipt and profile are local temporary evidence, not yet part of the release
evidence bundle. The same commit's hosted tutorial later failed the 300-second
gate as recorded above.

### Release-wheel grouped PTQ bridge experiment — 2026-10-07

The optimized Linux wheel from run `37637322050` was compared with a local
release-profile wheel that batches scale groups into one private native call.
Both used the public `prepare → calibrate → convert()` path, the same cached
SmolLM2-135M source and calibration receipt, `max_working_bytes=256 MiB`, and
the same `tritium.salt-v2-joint-diagonal-catq-relays-3@1` algorithm. The
baseline wheel SHA-256 is
`962f9a8d057d316be5bfe992e5c4ac271ba5c8eb62461cb5b9b462480e105b06`; the
experimental local release wheel SHA-256 is
`64fb7adf0f999ce166744dccb0495f9af6e916f629e3b68c1abf07fc5cffb398`.

Across two unprofiled local conversions per wheel, baseline times were
`105.094s` and `125.474s`; grouped-call times were `87.340s` and `91.390s`.
The observed medians were `115.284s` and `89.365s` (22.5% lower for the
grouped build). All four conversions emitted the identical artifact ID
`sha256:2beb214271ee9e1581e721f4f87937a3e12c1266ba4bc2e8bc99ab57b56059d4`.
The accompanying cProfile runs reduced calls to
`fit_joint_ternary_diagonal*` from 2,106 to 224 (89.4% fewer). Its profile is
at `/tmp/tritium-ptq-groups-convert.pstats` (SHA-256
`d7c6a02b6aaf11e7cf294efc99b481363bb82392154fa03903be9f0bb846e1aa`).
The grouped cProfile run took `88.697s` total; `_joint_additive_projection`
accounted for `78.124s` cumulative, and the native
`fit_joint_ternary_diagonal_groups` call accounted for `66.955s` self time.
Thus the measured local hot path is still native solver work, not Python/native
call count or artifact sealing. These figures are profiled local timings and
should guide optimization only; they do not explain the hosted runner's roughly
ninefold slower conversion by themselves.

### Installed-wheel SmolLM2 CPU tutorial — 2026-10-07

Built and installed the Linux CPU abi3 wheel (`pytritium 1.1.0rc2`) into an
isolated Python 3.14 environment with its optional ONNX dependencies, then ran
the pinned SmolLM2 tutorial through the public `prepare` → `calibrate` →
`convert()` path. The tutorial completed PTQ, generation, native checkpoint
round-trip, ONNX export/replay, and a QAT update plus optimizer resume in
`190.242s`, below its frozen `300s` local limit. Receipt:
`/tmp/tritium-ptq-installed-wheel-final-1791390391/receipt.json`.

The receipt records 211 selected and 61 preserved parameters, `537,919,488`
selected dense bytes versus a `92,192,265`-byte compact checkpoint (5.83× for
this small fixture), ONNX replay max absolute error `8.01e-5` against `1e-4`
tolerance, and PTQ artifact `sha256:2beb214271ee9e1581e721f4f87937a3e12c1266ba4bc2e8bc99ab57b56059d4`.
The generated sample was a short sentence; this is functional CPU smoke evidence,
not a model-quality claim. It does not qualify the hosted CI tutorial, GPU
performance, Qwen, or public release. The first attempt used an environment
missing ONNX optional dependencies and stopped at ONNX export; the successful
rerun used the existing isolated environment containing ONNX and ONNX Runtime.

This is promising local evidence for the bridge change, not release
qualification or a controlled performance claim: the baseline is the hosted
wheel, the new wheel was built locally with a different manylinux tag, and
host load varied substantially during the trials. The algorithm/configuration
was unchanged. The PTQ artifact suite passed 39/39, including exact
legacy-per-group parity and a public `convert()` two-group artifact
round-trip; the row-parallelism regression also passed. The grouped bridge is
committed as `9248f275`; its CI, docs, CodeQL, capstone, and wheel-platform
matrix passed. Its pinned tutorial had not completed when that wheel workflow
was superseded by the next candidate.

### SALT G128 joint-fit microbenchmark — 2026-10-07

A deterministic Divan benchmark now exercises the production diagonal-F64
solver configuration on one 128-weight group, separately for one, two, and
three planes. It uses the public bridge's 16-iteration cap, four deterministic
restarts, f16 scale scoring, `1e-8` ridge, `1e6` conditioning limit, and both
relay basins. Fixture construction and an output-shape/finite-objective
preflight are outside the timed loop. The source under test was solver commit
`0969789962b8be74366b3116552e987d81b5a263`; the benchmark harness SHA-256 is
`cd969158503fc0409348632534eef9ecf3f319497ddbdacc42f76318e744036d`.

On the Intel i9-14900K, Rust 1.98.0 optimized build, one Divan run measured
median fit times of `83.56 µs` (P=1), `318.1 µs` (P=2), and `780.3 µs` (P=3),
with 100 samples per case. This is a local microbenchmark baseline, not a
before/after speedup claim or hosted tutorial evidence: the host load average
was about `29.8` across 32 logical CPUs, and no matched pre-optimization run was
made. Re-run both variants under matched conditions before attributing
performance changes to the fused diagonal-statistic pass.

Command: `RUSTC_WRAPPER= cargo bench --offline -p tritium-benches --bench
salt_fit`.

### Hosted pinned SmolLM2 PTQ/QAT tutorial — 2026-10-07

Wheel workflow [37644673976](https://github.com/Quitetall/tritium/actions/runs/37644673976)
tested source commit `0969789962b8be74366b3116552e987d81b5a263`. The platform
wheel builds, clean-install checks, source-free tutorial, abi3 matrix, and matrix
admission passed, but the pinned SmolLM2 PTQ/QAT tutorial job failed its frozen
300-second wall-time limit after completing its stages in `1017.603s`. PTQ
conversion completed at `788.529s`; native checkpoint round-trip, generation,
ONNX export/replay, QAT step, and QAT resume also completed before the runner
raised the budget error. This is a functional tutorial run that fails the timing
gate, not an OOM or a model-quality result. The separately selected test seam
for future regression coverage is the public `tritium.torch.convert()` artifact
path; the existing focused public grouped-fit artifact round-trip passed locally.

The next hosted wheel workflow, run
[37652633017](https://github.com/Quitetall/tritium/actions/runs/37652633017),
tested pushed source commit `46bf27caf8e68edc9993b758ff39435e26119b00` and
failed the same frozen tutorial gate at `818.029s` against `300s`. The run
completed rather than timing out at the job level; this is still a wall-time
failure, not an OOM. The optimization that skips assignment reconstruction when
the current trit assignment is unchanged was committed as `ee55d6dc` and pushed
after that run finished. Workflow run
[37655145029](https://github.com/Quitetall/tritium/actions/runs/37655145029)
for `ee55d6dc` completed with an overall failure, although its job API lists
only three passing platform-wheel jobs and one skipped CUDA job; it contains no
SmolLM2 tutorial job and exposes no failed job log. The wheel artifacts passed,
but this run gives no timing result for the solver change. No performance
effect is claimed. The exact cause of the workflow-level failure and the absent
tutorial job are unresolved. CUDA, ROCm, Metal, wgpu, real-model serving, and
performance-regression jobs were skipped and remain unverified.

The following run,
[37655983490](https://github.com/Quitetall/tritium/actions/runs/37655983490),
did execute the tutorial on source `d2e072c62d7b3c775b3c19105d84216d10ce2dac`
and Linux CPU wheel SHA-256
`f572e4695a1d134e9ddfb903c1e086cf2bc93b409e698dbd220c9b76cc5536e4`. PTQ,
checkpoint round-trip, generation, ONNX export/replay, and QAT/resume all
completed, but the frozen 300-second tutorial budget failed at `1035.006s`.
The stage log reports PTQ conversion at `814.560s`; the preceding hosted run
`37652633017` reported conversion at `646.857s` and total time `818.029s`.
These are not a controlled A/B and do not establish that the solver change
caused the slower result. The runner reported roughly 12–14 GiB available
memory and over 80 GiB temporary disk during the run; this is not an OOM or
disk-pressure failure. The upload step was skipped after the timing failure,
so the stage log is the available hosted evidence; the wheel, abi3 matrix, and
other independent smoke artifacts were uploaded successfully.

The exact-source follow-up on commit `aec05035d992dd6fdd24628bfbd8fc80815bb6fd`
is Actions run
[37660642998](https://github.com/Quitetall/tritium/actions/runs/37660642998).
Platform wheels, installed-wheel smoke, source-free tutorial, abi3 matrix, and
matrix admission passed; the CUDA wheel was skipped. The pinned tutorial
completed its functional path but failed the unchanged 300-second budget at
`1001.291s`. Stage timings were calibration `0.620s`, conversion `778.364s`,
native checkpoint round-trip `800.979s`, generation `851.183s`, ONNX export
`965.086s`, ONNX replay `994.313s`, QAT step `999.878s`, and optimizer resume
`1001.291s`. The runner retained at least `12,398,168 KiB` available memory
and `87,492,580 KiB` temporary disk; this was neither OOM nor disk pressure.
The tutorial error was raised only after those functional stages completed.

Compared with run `37655983490`, conversion was about 36 seconds shorter and
total time about 34 seconds shorter, but hosted runs are not controlled
same-runner A/B measurements. No speedup is attributed to the assignment-skip
change. The frozen hosted timing gate remains red; local installed-wheel pass
evidence does not replace it. CUDA, ROCm, Metal, wgpu, physical performance, and
real-model serving remain unverified or skipped.

### Native PTQ row-fit allocation reduction — 2026-10-07

The native SALT V2 row fitter now borrows validated `DiagonalF64` evidence
instead of copying the 128-value group diagonal for every row, and uses an
in-place unstable sort for deterministic weighted-absolute initialization
ordering. The original index remains the unique tie-breaker, so the total order
and all quantile anchors are unchanged. The quantize crate's 235 unit tests,
public diagonal/affine bit-conformance test, and strict Clippy check passed.

The optimized local microbenchmark medians were 76.66 µs (P=1), 311.3 µs
(P=2), and 769.0 µs (P=3), 30 samples per case. Nearby runs on this busy host
varied by roughly 2×, so these numbers do not establish a speedup. This is a
software allocation reduction only; the pinned hosted 300-second tutorial
gate remains unverified for this change and must be rerun before any timing
claim.

### G128 PTQ row-parallelism benchmark — 2026-10-07

The `salt_fit` benchmark now includes a 64-row P=2 throughput case. It uses
the same deterministic G128 fixture and SALT fit configuration as the
single-group microbenchmark, with a Rayon pool created outside the timed loop.
On the Intel i9-14900K (32 logical CPUs), Rust 1.98.0, the one-worker median
was `19.77ms` per 64 rows and the four-worker median was `6.283ms` (`3.15x`);
30 samples per case. The exact solver output is checked by the existing
exhaustive and bit-conformance tests. This indicates that the row-parallel
path scales well on this local CPU. It is not a hosted-runner measurement or a
full-model conversion timing, so it does not establish that the pinned tutorial
will meet its `300s` gate.

### SALT V2 assignment/scale temporary storage reduction — 2026-10-07

The exact assignment codebook now uses fixed stack storage for its bounded
`3^P` states and an in-place deterministic sort (state ID is the unique tie
breaker). Scale canonicalization also moves each sign-corrected trit plane
through one owning vector instead of cloning it again after ordering. The
exhaustive assignment/tie oracle, scale-sign/order reconstruction test, public
solver conformance test, full quantizer unit suite (235 tests), and strict
quantizer Clippy passed.

The same local G128 benchmark measured medians of 77.46 µs (P=1), 323.1 µs
(P=2), and 797.8 µs (P=3), versus the immediately preceding run's 77.24 µs,
324.7 µs, and 811.7 µs. The differences are within the observed host noise and
do not establish a speedup. The benefit claimed here is bounded temporary
storage/allocation reduction only; the hosted 300-second tutorial has not yet
run against this source.

### Telemetry-enabled hosted SmolLM2 tutorial — 2026-10-07

Wheel workflow [37664061975](https://github.com/Quitetall/tritium/actions/runs/37664061975)
tested source commit `e35e6b25d7b53aa3971b5922fa622f62955fe751`. All functional
tutorial stages completed, but the frozen 300-second gate failed at
`828.685s`. Calibration took `0.520s`; conversion completed at `646.665s`,
native checkpoint round-trip at `665.490s`, generation at `707.171s`, ONNX
export at `799.370s`, replay at `822.754s`, and QAT optimizer resume at
`828.685s`.

The runner reported four logical CPUs. Samples retained about 13.4–14.5 GiB
available memory and 83.4–84.3 GiB temporary disk. Across the logged
`cpu.stat` samples, `nr_throttled` and `throttled_usec` stayed at zero. Load
average rose from below one to roughly five while the Python conversion ran.
This rules out OOM, disk pressure, and observed cgroup throttling as causes;
the measured failure is concentrated in native PTQ solver CPU time on the
four-CPU runner. It does not establish whether additional CPU parallelism,
per-core throughput, or both explain the gap to local timing. The frozen gate
remains red, and no model-quality or release qualification follows from the
functional completion.

### Native PTQ fit-result memory reduction baseline — 2026-10-07

Wheel workflow [37676403818](https://github.com/Quitetall/tritium/actions/runs/37676403818)
tested parent commit `369f69f3bc7d2c19db180fd2da2b6f7966a3b7ea`, before the
two follow-up bridge-memory commits. The pinned tutorial completed its full
functional path but failed the unchanged 300-second wall-time gate at
`624.605s`. Calibration took `0.428s`; PTQ conversion completed at `488.073s`,
native checkpoint round-trip at `501.302s`, generation at `529.505s`, ONNX
export at `603.480s`, replay at `619.631s`, and QAT optimizer resume at
`624.605s`.

The runner had four logical CPUs, at least about 12 GiB available memory during
the final sample, and over 83 GiB temporary disk. It was not an OOM or disk
failure. This result establishes the hosted parent baseline for commits
`ea2c7cd5` and `909add9c`; it does not qualify those changes or pass the frozen
timing gate. The candidate workflows must be inspected separately before
claiming any performance effect.

### Hosted compact-fit batching regression and bounded-batch follow-up — 2026-10-08

Wheel workflow [37678462152](https://github.com/Quitetall/tritium/actions/runs/37678462152)
tested `909add9c0a8136753eba94a80aca738836ec21ed`. All functional stages
completed, but the pinned tutorial failed the unchanged 300-second limit at
`983.764s`; PTQ conversion took `764.628s`. Checkpoint round-trip completed at
`785.224s`, generation at `827.808s`, ONNX export at `951.673s`, replay at
`976.718s`, and QAT resume at `983.764s`. The exact source-free tutorial,
platform wheels, ABI matrix, CI, docs, CodeQL, and capstone smoke passed; GPU,
ROCm, Metal, wgpu, and real-model serving lanes were skipped.

During PTQ, both the parent and candidate runners exposed four logical CPUs,
about 13.7–13.9 GiB mean available memory, similar mean load-1 (4.91 vs. 4.85),
and zero cgroup CPU-throttle events. The candidate accumulated about 2,956
cgroup CPU-seconds during conversion vs. about 1,817 for the parent. This is a
strong regression signal for the candidate execution path, but not a controlled
same-host A/B: CPU model/frequency and runner placement are not pinned, so the
source change is not yet proven to be the sole cause.

A new four-thread `salt_fit` bridge-collection benchmark compares flat full-fit
collection, flat compact collection, one-group-at-a-time compact collection,
and eight-group batches on the same deterministic 2,048-row fixture. All four
paths assert exact equality of scales, trits, and aggregate objective before
timing. The two exploratory runs were noisy and contradictory: the first
measured medians of 189.7 ms (flat full), 191.8 ms (flat compact), and 255.2 ms
(one-group compact); the second measured 207.7 ms, 292.3 ms, 210.1 ms, and
196.5 ms (eight-group batch), respectively. These do not support a stable
speedup or regression claim; local load was high and changed during the runs.

The unpushed follow-up `e4aadccc` now decodes weights per bounded group batch,
fits multiple adjacent groups in each Rayon pass, and limits retained fit
results to an 8 MiB trit-output estimate or 4,096 rows, whichever is smaller.
Local Rust tests (16), public PTQ/refinement tests (43), strict Clippy, and
exact-output benchmark preflights pass. It has not yet been measured by hosted
CI; the frozen 300-second gate remains open.

### Hosted PTQ timing confirmation and G64/P3 benchmark — 2026-10-08

Wheel workflow [37846314955](https://github.com/Quitetall/tritium/actions/runs/37846314955)
tested source `314c9f53e7df659e3819455e67abe414b2f85579`. Its pinned
SmolLM2-135M CPU tutorial completed conversion at `756.593s` and all later
functional stages (checkpoint round-trip, generation, ONNX export/replay, QAT
step and optimizer resume) at `972.244s`, then failed the unchanged 300-second
gate. The runner reported four logical CPUs, about 11.4–15.3 GiB available
memory and over 83 GiB temporary disk; observed cgroup throttling remained
zero. This is consistent with the previously identified native solver CPU
cost, not an OOM, disk-pressure, or tutorial-functionality failure. It is not
a controlled performance comparison or model-quality result.

To match the production compact recipe more closely, `salt_fit` now includes
64-column, three-plane independent row fits at one and four threads. A local
optimized-build sample on the i9-14900K measured a four-thread median of
`9.999ms` for 64 rows (`6.400 K rows/s`); the one-thread median was `24.10ms`.
A shorter initial sample measured `8.514ms`, showing material run-to-run
variation. This is an exploratory microbenchmark only: fixture values are
synthetic, there is no before/after candidate comparison, and it does not
qualify the hosted tutorial. The user's selected PTQ regression seam remains the public
`convert()` artifact path; `test_public_convert_persists_grouped_fit_artifact`
already exercises it. Next action is a controlled solver-level optimization
experiment that preserves artifact bytes on the fixed fixture, followed by a
fresh hosted tutorial rerun; do not relax its budget or reduce the model/profile
to make the gate pass.

### Exact weighted-quantile total cache — 2026-10-08

The production row fitter reuses the same ordered absolute-weight/curvature
context across all three plane-count basins, but recomputed its total curvature
for each restart quantile. `WeightedAbsOrder` now stores the total in the same
sorted summation order, eliminating those repeated sums while keeping each
prefix scan allocation-free. The quantile regression checks total and selected
value bits against the previous ordered fold, and the three-plane solver
fingerprint remains unchanged. The full quantizer suite passed (235 unit tests
plus integration tests), as did strict Clippy and formatting.

The local G64/P3 benchmark sample after the first implementation (which also
allocated a cumulative-prefix vector) measured medians of `23.10ms`/64 rows
at one thread and `6.085ms`/64 rows at four threads. The earlier same-host
candidate-free sample measured `24.10ms` and `9.999ms`, respectively. A shorter
pre-change sample had measured `8.514ms` at four threads, so host/run variation
is material. These are not a controlled before/after, and no speedup is
claimed. The final allocation-free revision measured `23.03ms` and `5.971ms`
medians under the same G64/P3 harness, again with no matched old/new run.
Full-model PTQ artifact identity and the hosted 300-second tutorial remain
unverified for this change. Next: perform
a matched old/new public `convert()` artifact comparison, then rerun the exact
hosted tutorial before attributing any runtime effect.

The exact pre-optimization wheel workflow [37849611144](https://github.com/Quitetall/tritium/actions/runs/37849611144)
then completed on source `cab6c07555c1d5e32cdde9f17b2cff1d80746e86` with the
same CPU-time failure. The pinned tutorial converted at `757.190s` and finished
all functional stages at `978.399s`, exceeding the unchanged `300.000s` limit.
Checkpoint round-trip completed at `777.949s`, generation at `820.966s`, ONNX
export at `945.624s`, replay at `971.381s`, and QAT optimizer resume at
`978.399s`. The hosted runner had four logical CPUs, roughly 13.5 GiB or more
available memory during the late stages, over 83 GiB temporary disk, and no
recorded cgroup CPU-throttle events. This is a baseline for the subsequent
solver-cache candidate, not evidence that candidate improves performance or
model quality. The next gates remain an exact public `convert()` artifact
comparison and the hosted tutorial on the candidate commit.

### PR #51 exact-head ABI3 matrix — 2026-10-08

The wheel workflow for candidate `74d57c6a5397a650b0e858515cbe59c092a5c67b`
produced `abi3-compatibility-receipt` at
[run 37852101721](https://github.com/Quitetall/tritium/actions/runs/37852101721).
The owning validator in `scripts/aggregate-wheel-smoke.py` accepted the
downloaded receipt against the exact source revision and release
`1.1.0-rc.2`: schema `tritium.abi3-matrix-qualification.v1`, `passed: true`,
16 CPython/platform cells, receipt ID
`sha256:69e0d6a0c40b53bbd72a461bfc35088079c8bc810a931c27df255d129e0290d6`.
This closes the compatibility-matrix evidence requirement for that exact
source revision, but it is not yet attached to a release-candidate manifest or
registered in the release evidence registry; the `packages` gate therefore
remains open.

### Matched public `convert()` artifact comparison — 2026-10-08

The selected regression seam is the public Python `prepare` → `calibrate` →
`convert()` path, exercised by
`test_public_convert_persists_grouped_fit_artifact`. The test passed once with
a freshly built `tritium-py` extension from baseline `c1aec7b9` and once from
candidate `1b38b8007be85166e8fa6cd1cf3f1c54fa059be5`, using the same unchanged
Python PTQ wrapper, fixture, locked dependencies, and test source. The resulting
conversion manifest, weight manifest, both plane trit payloads, and both plane
scale payloads were byte-identical (matching SHA-256 per file). This closes
artifact-identity coverage for this deterministic small CPU fixture; it does
not establish a full-model quality/runtime result or SOTA behavior.

The exact-head wheel workflow
[37853447148](https://github.com/Quitetall/tritium/actions/runs/37853447148)
was still running at the time of this check, with the pinned SmolLM2 CPU
tutorial active. Do not push a new commit until that `cancel-in-progress`
workflow finishes. Its final tutorial timing remains the next authoritative
measurement for this candidate.

The same machine's existing `salt_fit` G128 single-group microbench was run
with `CARGO_BUILD_JOBS=1 cargo bench --locked -p tritium-benches --bench
salt_fit -- joint_diagonal_g128 --min-time 1 --sample-count 10`. On the
i9-14900K, median latency was `73.78µs` for P1, `294.3µs` for P2, and `745µs`
for P3 (the bench config uses four deterministic restarts, 16 max iterations,
F16 scoring, and both relay basins). The P3 distribution had a `2.156ms`
slowest sample, so this short fixture benchmark is an algorithm-scaling signal,
not a stable performance claim or a function-level profile. `samply` could not
start a recording in this environment (`mmap failed`), and `gdb` attach was
denied by ptrace policy, so the next code optimization still needs finer
attribution before changes are chosen.

### Exact-head tutorial and ABI3 result — 2026-10-08

The workflow on source revision
`1b38b8007be85166e8fa6cd1cf3f1c54fa059be5` completed as
[run 37853447148](https://github.com/Quitetall/tritium/actions/runs/37853447148).
All wheel builds, installed-wheel functional smoke, source-free tutorial, and
the ABI3 matrix passed; the pinned SmolLM2 CPU tutorial alone failed its frozen
`300s` wall-time gate. It completed every function stage, with conversion at
`756.794s` and total tutorial time `972.680s` (`672.680s` over budget).
Checkpoint round-trip completed at `777.183s`, generation at `819.440s`, ONNX
export at `941.074s`, replay at `965.791s`, and QAT optimizer resume at
`972.680s`. The matching pre-cache baseline run `37849611144` recorded
`757.190s` conversion and `978.399s` total. This small difference across
separate hosted runners does not demonstrate a speedup; treat the cache as
artifact-preserving, not as a meaningful tutorial optimization.

The candidate runner had four logical CPUs, at least about `12.0 GiB` available
memory in late samples, more than `83 GiB` temporary disk, and zero recorded
cgroup CPU throttling. This remains a CPU-time failure, not an OOM or storage
failure. The exact-source ABI3 receipt downloaded from the same run passed the
owning `validate_receipt` check for release `1.1.0-rc.2`: schema
`tritium.abi3-matrix-qualification.v1`, 16 cells, run ID
`github-37853447148-1-abi3-matrix`, receipt ID
`sha256:c30797a66685f64c294c4309f4264d5c4d846ff3fab0ac3a9f59ee878ff76c4f`.
As before, matrix evidence alone does not close the full `packages` gate.

### Exact assignment codebook deduplication probe — 2026-10-08

The solver's exact ternary assignment codebook now removes duplicate
reconstructions after total-order sorting. Equal reconstruction values have
identical error for every weight; retaining the first (lowest state) preserves
the prior tie order while avoiding a second search for the start of a duplicate
run. The exhaustive assignment-oracle tests pass. A test-only ignored phase
profile was added to attribute G64/P3 work without changing the production API.

The public `prepare` → `calibrate` → `convert()` artifact test was rebuilt
against this exact working-tree Rust extension and passed (`1 passed`). All six
persisted files matched the previously recorded baseline/candidate comparison:
conversion manifest `503ddde4…cec57d6`, weight manifest
`976cf130…bb6765f`, plane-0 scales `c8de1782…b717c9`, plane-0 trits
`e1c9310f…326bd17`, plane-1 scales `e5693158…0a17b0`, and plane-1 trits
`9adb3b52…703ac12`. This is deterministic small-fixture artifact identity,
not full-model equivalence or quality evidence.

The test-only phase profile over 256 deterministic G64/P3 rows measured
`552.816ms` total: assignment `211.214ms`, scale solving `118.757ms`,
reconstruction `141.629ms`, and uninstrumented remainder `81.216ms`. An earlier
same-harness profile before codebook deduplication measured `565.809ms` total,
with assignment at `252.060ms`; this instrumentation is diagnostic, not a
release benchmark. Matched optimized `salt_fit` runs on this host varied across
roughly `20.2–23.2ms` (one thread) and `5.4–6.9ms` (four threads) per 64-row
fixture, with candidate samples overlapping baseline samples. Therefore no
production speedup is established. Strict quantizer tests (235 passed, 1
ignored), strict Clippy, formatting, and the public artifact test pass. The
hosted run [37856071971](https://github.com/Quitetall/tritium/actions/runs/37856071971)
has passed its wheel, install-smoke, and ABI3 jobs. The pinned SmolLM2 CPU
tutorial completed every functional stage but failed the frozen `300s` budget:
PTQ conversion completed at `757.355s`, checkpoint round-trip at `777.642s`,
generation at `819.383s`, ONNX export at `942.070s`, replay at `966.852s`, and
QAT optimizer resume at `973.779s` (`673.779s` over budget). This is effectively
unchanged from run `37853447148` (`756.794s` conversion, `972.680s` total), and
does not test the codebook-deduplication change because it was not in the tested
source revision. It reinforces that the release tutorial gate is still a
substantial CPU optimization blocker.

### Move solver receipts instead of cloning — 2026-10-08

`fit_joint_ternary_prepared` used to clone every restart receipt, including its
nested accepted-update and scale-solve vectors, into the result, then drop the
originals with the candidate states. It now moves those receipts out of the
internal fit states. Public `JointTernaryFit` fields, receipt ordering, solver
decisions, and artifact schema are unchanged; the existing bitwise-determinism
test compares full results, including receipts.

Validation on this working tree: full `tritium-quantize` suite (235 passed, 1
ignored), strict Clippy, formatting, and the freshly rebuilt public
`prepare` → `calibrate` → `convert()` artifact test all pass. The six persisted
files retain the baseline fixture hashes recorded above. One optimized G64/P3
benchmark sample measured `20.56ms`/64 rows at one thread and `9.409ms` at four
threads; the host load average was 8.58 and unrelated CPU-heavy processes were
active. This is not a controlled before/after and establishes no speedup. The
change removes redundant nested-vector copies by construction, but its runtime
impact and contribution to the frozen tutorial gate remain unqualified.

A matched short `salt_fit` comparison was then run on this host, pinned to CPU
IDs 27–30, using the same command and one sample per revision:
`taskset -c 27-30 env CARGO_BUILD_JOBS=1 cargo bench --locked -p
tritium-benches --bench salt_fit -- joint_diagonal_g64_p3_ptq_rows
--min-time 1 --sample-count 10`. At one thread, baseline `f4398109` measured
`36.49ms`/64 rows and candidate `61af063a` measured `36.95ms`; at four threads,
baseline measured `9.414ms` and candidate `9.581ms`. This is neutral to slightly
slower within a single sample per revision and establishes no runtime benefit.
The allocation reductions are supported structurally and by correctness tests,
but wall-time impact remains unproven. This microbenchmark is not a substitute
for the hosted full-model tutorial gate.

The previously active exact-head workflow has now completed as
[run 37858286157](https://github.com/Quitetall/tritium/actions/runs/37858286157)
on source `f439810937b96f487b8f0164d3009b5cd93899f5`. Wheel builds, installed
wheel smoke, the source-free tutorial, and ABI3 qualification passed; the CUDA
wheel was skipped. The pinned CPU tutorial completed all functions but failed
the unchanged `300s` gate at `919.186s`. Stage timings were calibration
`0.599s`, PTQ conversion `700.368s`, native checkpoint round-trip `721.012s`,
generation `763.559s`, ONNX export `887.330s`, ONNX replay `912.259s`, and QAT
resume `919.186s`. Compared with the earlier hosted conversion near `757s`,
this run is about 7.5% faster, but separate-run variance prevents attributing
that difference to a specific code change. The tutorial remains roughly
`619s` over budget, with ONNX export also taking about `124s`; both require
optimization without changing the frozen model or recipe. This run predates
the local receipt-move and deferred-plane-copy commits, which are now eligible
for an exact-head hosted measurement.

To narrow the next optimization target without another full-model run, the
existing ignored native phase profiler was run with
`cargo test --locked -p tritium-quantize profile_g64_p3_solver_phases --
--ignored --nocapture`. Its synthetic 256-row G64/P3 fixture reported
`526.671ms` total: assignment `209.351ms` (39.7%), scale solve `113.536ms`
(21.6%), reconstruction/objective `121.110ms` (23.0%), and other fit work
`82.675ms` (15.7%). This debug-test fixture is not a full-model or optimized
build measurement, but it makes solver assignment and reconstruction the
leading code-level targets; receipt/bridge copies are not the only plausible
cost. No quality-affecting solver iteration or basin settings were changed.

The same allocation pass also changes scale-solve candidates to carry a
three-plane sign/permutation map instead of cloned trit vectors. Candidate
reconstruction/objective is evaluated through that map with the existing fused
diagonal scoring order; the state planes are transformed in place only after
the candidate is accepted. This removes per-iteration plane copies, including
for rejected M-step candidates, without changing public receipts or canonical
plane ordering. The full Rust suite and strict Clippy pass, and the rebuilt
public `convert()` test again preserves all six artifact hashes. No fresh
runtime comparison is claimed: during the attempted local bench window, host
load exceeded 10 with unrelated multi-core work active. The optimization remains
an evidence-backed allocation reduction whose wall-time impact is unmeasured.

The preceding exact-head run
[37856071971](https://github.com/Quitetall/tritium/actions/runs/37856071971)
on source `5f0feca67a31b4c1634c868550894cdeb465bdc9` confirms the same result:
the pinned CPU tutorial completed all functions but failed `300s`, with PTQ
conversion at `757.355s` and total time `973.779s`. Wheel, installed-wheel
smoke, and ABI3 jobs passed. Because this revision predates the codebook
deduplication, it is a control only and provides no hosted performance evidence
for that change.

### Relay-basin hot-loop reuse — 2026-10-09

The release-mode test-only phase profiler was extended to separate metric
validation, weighted-order construction, deterministic starts, and relay-basin
scale initialization. On its synthetic 256-row G64/P3 fixture, baseline source
`4550a1ba` reported `83.722ms` total and `37.965ms` in relay initialization.
The candidate reuses the already-computed two edge `tanh` values for both the
relay value and its derivatives; this preserves the old f64 operation order and
outputs. Its diagnostic profile reported `65.895ms` total and `21.103ms` in
relay initialization. The test-only timing is directional, not a production
claim.

The optimized `salt_fit` benchmark was then run on pinned CPU IDs 27–30 with
separate Cargo target directories, preventing one checkout from reusing the
other checkout's compiled quantizer. Same-command medians over 64 G64/P3 rows:

| Source | One thread | Four threads |
|---|---:|---:|
| Baseline `4550a1ba` | `37.36ms` | `9.792ms` |
| Candidate | `29.65ms` | `7.859ms` |

This fixture indicates about 20% lower solver latency locally; it is not a
full-model result and the hosted 300-second gate is still authoritative. A
bitwise reference test covers both relay variants over lengths 1–128 and eight
deterministic inputs. The public `prepare` → `calibrate` → `convert()` artifact
test passes and now asserts SHA-256 identities for all six persisted output
files; all match the existing fixed-fixture values. The full quantizer suite
passes (236 passed, 1 ignored), strict Clippy and formatting pass. Exact-head
hosted qualification remains open: workflows on `4550a1ba` predate this change.

A second matched benchmark used the larger G128 fixture with separate target
directories and the same CPU pinning. Single-fit medians were `146.2µs`,
`505µs`, and `1.17ms` for baseline P1/P2/P3, versus `107µs`, `398.7µs`, and
`957µs` for the candidate. The P2 64-row batch measured `32.76ms` baseline vs
`25.82ms` candidate at one thread, and `8.22ms` vs `6.462ms` at four threads.
These synthetic local results indicate a consistent reduction, but do not
establish full-model quality, end-to-end PTQ time, or release performance.

The exact-head hosted workflow
[run 37860276563](https://github.com/Quitetall/tritium/actions/runs/37860276563)
on `4550a1bac07f3c5d610db7ad092ffa14be02d108` has completed. Wheel builds,
installed-wheel smoke, and ABI3 qualification passed; the CUDA wheel was
skipped. The pinned SmolLM2 CPU tutorial completed all stages but failed the
unchanged `300s` gate at `919.186s`: PTQ conversion was `700.368s`, native
checkpoint round-trip `721.012s`, generation `763.559s`, ONNX export
`887.330s`, ONNX replay `912.259s`, and QAT resume `919.186s` (calibration
was `0.599s`). The optimization is not yet measured by hosted end-to-end
qualification; the candidate exact-head run remains required and the tutorial
is still far over budget.

### Portable training CPU backend receipt — 2026-10-08

On clean source commit `672ad1bd1325229a13bd2a0125b77dfca73b2d66`, the V2 CPU
training backend was run on the physical i9-14900K host against the frozen
36-operation/117-case corpus. The source-free receipt was independently
reopened under `ReleaseCandidate` policy by
`training_capability_table`; it records peak resident bytes `4192` and peak
scratch bytes `132032`. Bundle BLAKE3 is
`2c9e9ffcc9720fc0518078c391ebbf6f9a8dbeba6e4f502a78be8e1c41a66d04`, its
SHA-256 is
`395eb08bb873509e0ce069fdcd785cd2e53e05273c678af59e2e7111b0585a29`, and its
size is 41,317 bytes. The durable receipt is stored at
`/mnt/2tb/tritium-release-evidence/training-backends/672ad1bd/cpu-v2/`.
This qualifies the CPU family at that exact source revision only; it is not a
seven-backend aggregate, a performance receipt, or final release qualification.
The other six target families and final-source regeneration remain open.

### Matched portable-training CPU/CUDA receipts — 2026-10-08

At clean source `cea334e0c681422e9ee72d8e0e902193c9dcb92e`, the V2 CPU and
CUDA backends independently executed and reopened against the same frozen
36-operation/117-case corpus. CPU ran on the i9-14900K; CUDA ran on the
physical RTX 4090. The paired admission command was
`cargo run --locked -p tritium-testkit --example training_capability_table --
--schema v2 <CPU_DIGEST>=<CPU_RECEIPT> <CUDA_DIGEST>=<CUDA_RECEIPT>`.
Both receipts report peak resident bytes `4192` and peak scratch bytes
`132032`.

| Backend | BLAKE3 bundle ID | SHA-256 | Bytes | Durable receipt directory |
|---|---|---|---:|---|
| CPU | `e6712fadbf08b6470b6c00c10f5ee9c15057d51886c23ff2563b871cab9f5a10` | `f326baadc08606ced3f05d1a58220c8f8bd131d7b4b41bf4a7e32002865c5b52` | 41,317 | `/mnt/2tb/tritium-release-evidence/training-backends/cea334e0/cpu-v2/` |
| CUDA | `521cbd37d2857d78813f9a70bfcf15f43c27d363dd2021b66e265e86c7592868` | `cfafb62826188b9b63dc8124f14cf53c13435814cabb238b04fc05b0d8b544d9` | 41,259 | `/mnt/2tb/tritium-release-evidence/training-backends/cea334e0/cuda-v2/` |

These are two of the seven required backend families at one exact source
revision. They do not yet form an aggregate release receipt, cover WASI/MCU,
ROCm, Metal or native wgpu, or qualify the separate performance gate. The
bundles must be regenerated if the source changes before candidate freeze.

### Exact-head SmolLM2 tutorial outcome — 2026-10-09

The completed hosted wheel workflow
[run 37861626145](https://github.com/Quitetall/tritium/actions/runs/37861626145)
tested source `8c5a994e11763503dd35d38e8a128b08dbf47266`. Wheel builds for
Linux, macOS, and Windows, installed-wheel smoke, the source-free tutorial, and
the ABI3 matrix passed. The CUDA wheel lane was skipped as configured. The
pinned SmolLM2 CPU tutorial completed its functional path but failed the frozen
`300s` budget at `749.959s`. Stage markers recorded conversion at `526.575s`,
native checkpoint round-trip at `547.678s`, generation at `591.284s`, ONNX
export at `717.628s`, ONNX replay at `742.924s`, and QAT resume at `749.959s`;
calibration took `0.610s`. The job failed only when enforcing the wall-time
budget after QAT resume. No tutorial receipt artifact was uploaded by this
failed job, so these timings are preserved from its hosted job log.

This run is about `169s` faster end-to-end than the prior recorded hosted run at
`919.186s`, but the runs are not a controlled before/after experiment and this
result remains `450s` over budget. It therefore does not attribute the
difference to a particular optimization and does not pass the release gate.
The public `prepare` → `calibrate` → `convert()` artifact-path test
`test_public_convert_persists_grouped_fit_artifact` passes locally and verifies
the persisted artifact hashes; the latest hosted failure confirms that this
software seam is functional but does not yet make full-model PTQ fast enough.

The exact-head CI and wheel runs for pushed revision
`2960a097e14dd30f98785a273c17e197c35cc4ee` then completed. CI run
[37863291103](https://github.com/Quitetall/tritium/actions/runs/37863291103)
passed its executed jobs; hardware-only and real-model lanes were skipped.
Wheel run
[37863291107](https://github.com/Quitetall/tritium/actions/runs/37863291107)
passed platform wheel builds, installed-wheel smoke, source-free tutorial, and
the ABI3 matrix; its CUDA wheel lane was skipped. The pinned SmolLM2 tutorial
completed functionally but failed the frozen `300s` gate at `527.413s`.
Recorded stages were calibration `0.469s`, conversion `369.797s`, native
checkpoint round-trip `385.945s`, generation `421.026s`, ONNX export
`500.457s`, ONNX replay `521.624s`, and QAT resume `527.413s`. No tutorial
receipt artifact was uploaded by the failed job.

Compared with the immediately preceding hosted result of `749.959s`, this is
`222.546s` faster end-to-end and `156.778s` faster at conversion. The runs are
not controlled for runner variability, and both fail the same frozen gate; the
difference is not attributed to a code change. The latest run is still
`227.413s` over budget and does not qualify the full-model tutorial. It tested
`2960a097`, before the locally committed relay-normalization buffer reuse, so a
new exact-head wheel/tutorial run is required for that source change.

### Matched portable training CPU/CUDA V3 receipts — 2026-10-08

On clean source `4256284069e0135da1f62e2853eafc96f764cc53`, the V3 CPU and CUDA
training backends independently executed against the same current
37-operation/122-case corpus. CPU ran on the physical i9-14900K host; CUDA ran
on the physical RTX 4090 (`cuda:0`). The source-free
`training_capability_table --schema v3` admission reopened both bundles
together. Both report peak resident bytes `4192` and peak scratch bytes
`132032`. The source identity is
`tritium-train@1.1.0-rc.2+source-git:4256284069e0135da1f62e2853eafc96f764cc53`;
the manifest digest is
`fda9e905f09151ae4fa55183e460bf9bc9b3dd35d77be7b225bb210da8b40fc5`, the
vector digest is
`c8df31ee8ac867d9009909f11fc9513d3b7464e403ebd88ea6437a26ab78009f`. CPU
receipt digest:
`038176fdb5deace9f37826ddccb4573f0290eafa74fccfdee702bd32d5802fe7`; CUDA
receipt digest:
`712c52a905c6cc99c8927abf8555e95da6d794faca1b4d51535af338b52c7d59`. Durable
receipts are stored under
`/mnt/2tb/tritium-release-evidence/training-backends/42562840/{cpu-v3,cuda-v3}/`.

These qualify two of seven backend families at this exact candidate source.
They are not a seven-backend aggregate, a performance receipt, or final release
qualification; the other backend families and final-source regeneration
remain open.

### Matched portable-training V2 CPU/CUDA/WASI receipts — 2026-10-08

The frozen V2 release corpus was executed and independently admitted for three
backend families at clean source `4256284069e0135da1f62e2853eafc96f764cc53`.
CPU ran on the i9-14900K, CUDA on the RTX 4090, and WASI in a Wasmtime 48.0.1
guest on `x86_64`. The WASI bundle came from exact-head CI run
[37864560574](https://github.com/Quitetall/tritium/actions/runs/37864560574)
and was reopened locally from the clean source checkout. The three-way
`training_capability_table --schema v2` admission reports 36 operations, 117
cases, peak resident bytes `4192`, and peak scratch bytes `132032` for each
family. Manifest digest:
`9093a1a7f9a3422c399943782aadf4df6b11833cf2253db0db56ff2d9dedb098`; vector
digest:
`38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7`.

| Backend | Receipt bundle | Durable receipt directory |
|---|---|---|
| CPU | `cc5b2af1ed0ca930b9deb0311db55f1dbd88dc9bd2c4ce1c1c678ef283d90973` | `/mnt/2tb/tritium-release-evidence/training-backends/42562840/cpu-v2/` |
| CUDA | `fc7f02867170e96832596a0c61b7994690d66a59a07213449bce9ee9d9d49f46` | `/mnt/2tb/tritium-release-evidence/training-backends/42562840/cuda-v2/` |
| WASI | `dda13a83c8f6c22f9daf5a4f18d80384a767eb4b18686c5d5c4b7a5ac712b0ff` | `/mnt/2tb/tritium-release-evidence/training-backends/42562840/wasi-v2/` |

These are three of seven required V2 release families at this exact source;
they do not form the aggregate qualification or the separate performance
receipt. ROCm, Metal, native wgpu, MCU, and final-source regeneration remain
open. V3 CPU/CUDA receipts above are separate extension evidence and are not
substituted into the frozen V2 release corpus.

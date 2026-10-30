# Source-free web archive and native parity — 2026-10-10

Executed clean source: `a3542beebd2e8bc518544e0b795a448b94203b24`.
Contract: ADR 0033 and plan 0050. These are package, deterministic WASM and tiny
native-reference results, not physical-browser WebGPU, model-quality, latency
or full-release qualification. No browser lane was fabricated.

## Exact hosted npm bytes

[CI run 38040527733, web-package job 114179696616](https://github.com/Quitetall/tritium/actions/runs/38040527733/job/114179696616)
verified this exact checkout and ran the complete package check. Its actual log
records 149 tests passing, zero failures and zero skips, plus offline installed
archive verification. Observed hosted runtime was Node `v22.23.3`, npm `10.9.9`.
The overall CI run subsequently completed successfully; this does not turn
skipped physical hardware/model/performance lanes into passing evidence.

Artifact `11665432234` contains the original npm archive, qualification receipt
and CycloneDX SBOM. Original ZIP SHA-256 matched GitHub metadata:
`fb2128d3dfe069b7bb710f7ba38b9d5b6a2259c773174d8a6a3e201990c60ca2`.
Only the three explicitly named ZIP members were extracted.

- Archive: `tritium-ai-web-1.1.0-rc.2.tgz`, 627941 bytes.
- Archive SHA-256: `35f0f44c618dfc7dcb3ce3a9a6e089a95f4ca99d353da1ee7ca7375603284781`.
- Original [npm receipt](evidence/web-a3542bee/npm-archive-receipt.json):
  `sha256:d8978f975a0493dd25a46134eb7710d3dbf53abdef3f550abe060aecf5f61a43`.
- Embedded guest SHA-256: `a5d242a06e6eb6d76dff95fc50af588450993c0cbb4a4e8b7eefc6eec6e26651`.
- Embedded build: `tritium-wasm@1.1.0-rc.2+source-git:a3542beebd2e8bc518544e0b795a448b94203b24`.

The unchanged Python npm validator admitted the original bytes and receipt.
`qualify-npm-compatibility.py` projected the original **hosted Node 22** receipt,
not the later local Node 24 run. Original receipt/SBOM bytes remain immutable;
no candidate registration, signature or registry publication is implied.

## Native reference and separate local replay

The shipped `produce-browser-native-reference.py` ran the frozen
`salt-ste-sgd-256-v1` scenario at clean source a3542bee on the i9-14900K CPU. It
performed the native training step, SALT V2 export and strict reload and sealed
[native receipt](evidence/web-a3542bee/native-reference.json)
`sha256:6438dcb24883b9c284431f8b081f5abcf9462a7acf732593ba73d6a8f27e81ef`.
Its 224-byte artifact SHA-256 is
`6e889858c06a7eb91133f69a948ab8356a444c677eecd9e800ec689380a6e17e`.
The source-bound browser producer separately re-admitted both native and npm
inputs before any attempted physical-browser execution. Its temporary
dependency symlink reused the existing cache and was removed after validation.

The exact hosted tarball was installed into a fresh empty consumer with
`npm install --offline --ignore-scripts --no-audit --no-fund --package-lock=false`.
On local Node `v24.21.0`, both public import paths resolved inside that
consumer's `node_modules`, with no repository-relative runtime imports. The
installed deterministic WASM executed all 36 operations and 117 cases twice;
its observed build/guest digests matched the hosted receipt. The exact corpus
inventory was 72 valid and 45 expected-invalid cases, without substitutions.

The installed package then ran prepare, forward, backward, one optimizer step,
checkpoint, export, fresh-session resume, re-export and repeated disposal.
Export bytes matched the native reference exactly; checkpoint and export bytes
survived fresh-session resume unchanged. The checkpoint SHA-256 is
`43fa2cf70f020ee94d5146cc08c3a9ae19367fb4c45d3c087c850b8e54d61189`.
Strict consumer TypeScript passed with `skipLibCheck: false` against the installed
declarations. The local [execution result](evidence/web-a3542bee/local-node24-replay.json)
is explicitly labeled WASM/Node 24, not WebGPU or Node 22 evidence.

The first authored replay harness passed the checkpoint result envelope rather
than `checkpoint.bytes` to `resume`, receiving the documented
`invalid_schema: checkpoint must not be empty` error. Correcting that caller to
the installed declaration made the complete replay pass; no product API was
changed to accommodate the erroneous call. The three Python input-validator
suites passed 11 tests in 0.093s. Those synthetic tests are distinct from the
real archive/reference executions above.

## Physical lanes and custody

This task has no attached display. The agent-workspace-linux skill requires an
isolated workspace for GUI/browser automation, but its MCP tool family was
unavailable. No host browser/profile was attached, display created through a
workaround, or headless software adapter promoted into physical evidence.
Existing older Chrome/Firefox traces remain historical, not relabeled for
this source. Fresh Chrome/Firefox lanes and physical-macOS Safari, all six fault
classes, trace admission and the complete browser aggregator remain open.

Original hosted ZIP/archive/receipt/SBOM, provenance/job log, native reference,
local replay/type harnesses and validator results are retained at
`/home/brianklam/Projects/Tritium/archive/verification/web-a3542bee-20261010/`.
Public receipt/result copies are byte-compared after transfer. The installed
consumer dependency tree and temporary dependency symlink are removed; retained
results are moved out of scratch. The reused pinned managed checkout and shared
build cache remain, and unrelated staged/dirty work is preserved. No old August
scratch/model weight, large campaign, cloud resource or published artifact was
modified.

Remaining full-release fronts are serving empirical qualification; PyTorch/HF
lifecycle/PTQ/refinement/distributed evidence; remaining physical backend and
browser/performance matrices; whole-model ONNX and complete package/tutorial/
Colab provenance; authorized recipe freeze and Qwen language/MTP quality,
physical bytes, runtime/memory and reproduction; production security/deployment;
audited model zoo/community; independent clearance, signing, human activation
and authorized publication. A future candidate must regenerate source-bound
evidence rather than reuse these identities under a new source label.

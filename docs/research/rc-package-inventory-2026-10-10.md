# Exact-source CPU package inventory — 2026-10-10

Plan 0051 packaging progress, bounded to source
`81587fe1e01779aca2f160bdd69aaf038ea790f5` and `1.1.0-rc.2`.
This is a partial local candidate, not full-release qualification.

## Retained and re-admitted

Hosted CI run `38031776767` and wheel run `38031776696` completed successfully
at the exact revision. Their downloaded artifacts comprise 23 Rust crates,
three CPU abi3 wheels and the npm archive. Each original bound CycloneDX SBOM
is retained. The shipped input generator and assembler produced an immutable
27-artifact manifest; the real shipped CLI supplied SHA-256 and raw BLAKE3
identities. Clean-source/version admission returned `CANDIDATE_EVIDENCE_VALID`
(exit 0).

The shipped validators independently re-admitted ten hosted receipt kinds:
crate archive, npm archive, ABI3 compatibility, clean install, installed QAT
tutorial, HF lifecycle, export/reload, observability, API signature and CPU
dispatcher overhead. All 16 interpreter records reproduce the original ABI3
receipt, including exact wheel byte counts/digests. The three wheels passed
full archive topology, metadata and RECORD inspection. These receipts remain
unchanged; no local approval or qualification receipt was fabricated.

The approximately 37 MiB collection is saved outside scratch at:

`/home/brianklam/Projects/Tritium/archive/verification/rc-packages-81587fe1-20261010`

It retains the package files, SBOMs, raw hosted receipts/traces, GitHub API
metadata, immutable partial registry, generated gate reports and a replay
driver. Redundant copies created for this collection were byte-compared before
removal. No August campaign data was moved or deleted.

Manifest SHA-256:
`28686b0f84135b8646a59c06685a00312ca0b50a768d61604f91be075978c280`.

Partial registry SHA-256:
`f5820bfcf93f076a203a95bb39b982a239a462e092494a0516e87c70dbb14de4`.

## Provenance and claim limits

Generated in-toto statements identify the local candidate assembler and
collection invocation. They are unsigned assembly attestations, not trusted
CI-builder attestations, release signatures or independent reproduction.
CPU wheel evidence is not CUDA/ROCm/Metal/wgpu qualification; WASI/Node evidence
is not Chrome/Firefox/Safari WebGPU training qualification. Tiny frontend
fixtures are not whole-Qwen quality or ONNX generation evidence.

With the partial registry, `release-status` returned `LOCAL_RC_BLOCKED`
(exit 1). The packages row passes for these collected files; the other 11
aggregate rows still lack required evidence. This does not establish complete
candidate inventory, `LOCAL_RC_READY`, SOTA status or public activation.

The exact-source CPU CLI build succeeded in 1m25s using the reused SSD Cargo
target, two jobs and no Rust wrapper. It was used for file identities only.
An eight-module local packaging regression command timed out after 120 seconds
(exit 124), with the test process observed blocked in filesystem I/O at
`folio_wait_bit_common`. No local regression pass is claimed from that run.
Hosted exact-source CI success is separate from this incomplete local command.

## Binding next work

1. Complete the exact-source candidate inventory with CUDA, ONNX/model and
   deployment artifacts; verify trusted provenance and signing separately.
2. Execute remaining distributed/accelerator frontend, estimator/refinement,
   native-backend and actual-browser gates on their physical targets.
3. Admit/freeze the source and scalable recipe before flagship Qwen language/MTP
   conversion, quality/task-retention, physical-byte and runtime qualification.
4. Complete whole-model ONNX, real deployment/security, four-model zoo,
   independent reproduction/signoff and human-authorized publication.

All original full-release requirements remain binding.

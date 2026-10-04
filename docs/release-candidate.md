# Local release-candidate evidence

Tritium admits unpublished release artifacts without treating their presence as
release readiness. `scripts/assemble-release-candidate.py` creates deterministic
artifact identities and SLSA provenance. `scripts/release-status` then rehashes
every byte and prints `CANDIDATE_EVIDENCE_VALID`. That status does **not** mean
`LOCAL_RC_READY`; model-zoo, browser, serving, package-matrix, signing and
second-machine gates remain separate.

## Gate status (measured 2026-09-03)

### Flagship campaign status refresh (2026-10-03)

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
the router currently marks the worker alive as soon as the thread is spawned.
Thus the first request can be accepted and wait in the queue during CUDA batch
initialization, and readiness can report ready before batch initialization has
finished. This conflicts with plan 0052's requirement that readiness remain
false during startup. The fix and a public `/readyz`/chat-startup regression
test remain open. This source-tree measurement uses the local BitNet GGUF
fixture; it is not a Qwen candidate receipt or a release qualification.

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
| `packages` | PARTIAL | `compatibility-matrix` | **Not blocked — CI produces this on every release and the release workflow now carries it into the payload.** The rc.2 `abi3-compatibility-receipt` passes `aggregate-wheel-smoke.py`'s own validator: `tritium.abi3-matrix-qualification.v1`, bound to `d16c0dda`, `passed: true`, 16 cells spanning CPython 3.9.25–3.14.7 across three platforms and three distinct wheels. Harvesting it advances the union 15→16. It does **not** by itself close the gate: crate, npm, clean-install, and compatibility receipts still must bind one exact revision before a coherent `packages` PASS. |
| `pytorch-hf` | PARTIAL | `distributed-training` | Two or more GPUs. |
| `native-backends` | PARTIAL | `backend-manifest`, `performance` | All seven trace families, in order — `FAMILIES = ("cpu", "cuda", "rocm", "metal", "wgpu", "wasi", "mcu")`. Needs AMD *and* Apple *and* an MCU board. |
| `estimators-refinement` | PARTIAL | `refinement`, `baseline-ablation` | Separate local SALT campaign runs and baseline ablations; no current receipt is registered for these kinds. |
| `flagship-qwen` | **NOT CONFIRMED RUNNING — last canonical record says stalled** | `conversion-refinement`, `quality`, `task-retention`, `runtime`, `physical-bytes` | The last canonical campaign probe found 0/506 published masters and no completion seal (see the 2026-09-29 refresh above). Its former `/mnt/4tb/tmp` workspace is no longer at the recorded path. Revalidate workspace location and calibration provenance before any resume; Stage 7 recipe freeze is a prerequisite. |
| `stage7-freeze` | NONE | `stage7-recipe-freeze` | Complete the 1.7B recipe freeze before unsealing/running the pinned Qwen flagship, as required by plan 0043. |
| `onnx` | NONE | `onnx-inference` | Whole-Qwen ONNX execution traces — downstream of the flagship artifact. |
| `browser` | NONE | `browser-conformance` | **Three** lanes, all required: `--chrome-lane`, `--firefox-lane`, `--safari-lane`. The Safari lane is gated on a macOS `os.name`, so this needs Apple hardware, not merely a browser. |
| `serving` | PARTIAL | `oci-runtime-{cpu,cuda}`, `serving-deployment-{cpu,cuda}` | Both `oci-security-*` kinds are **done** (2026-09-03). The remaining four all need an **admissible serving bundle**, which does not exist on this box: `tritium-serve` rejects the only complete-looking candidate with `InvalidAdmission("manifest package")` because its `tritium.json` carries no top-level `manifest_package_id` and is marked `complete_model: false`. Deployment additionally needs Kubernetes, a Helm chart archive, and a `--bundle-manifest`. |
| `zoo-community` | NONE | `model-zoo`, `generated-claims`, `governance-docs` | All three come from **one** `qualify-zoo-community.py` call. It requires a `--governance-review` whose `independent_from_maintainers` field must be `True` (`verify-zoo-community-receipt.py:426-429`) and a named reviewer with an `organization` — i.e. a second person. It also requires four frozen model entries, the fourth being the flagship. |
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

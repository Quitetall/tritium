# Local release-candidate evidence

Tritium admits unpublished release artifacts without treating their presence as
release readiness. `scripts/assemble-release-candidate.py` creates deterministic
artifact identities and SLSA provenance. `scripts/release-status` then rehashes
every byte and prints `CANDIDATE_EVIDENCE_VALID`. That status does **not** mean
`LOCAL_RC_READY`; model-zoo, browser, serving, package-matrix, signing and
second-machine gates remain separate.

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
  "release": "1.1.0-rc.0",
  "source_revision": "FULL_40_CHARACTER_GIT_REVISION",
  "builder": {
    "id": "https://github.com/OWNER/REPOSITORY/actions/workflows/release.yml",
    "build_type": "https://tritium.ai/build/package/v1",
    "invocation_id": "EXACT_WORKFLOW_RUN_ID"
  },
  "artifacts": [
    {
      "id": "tritium-torch-linux-cpu",
      "kind": "python-wheel",
      "path": "tritium_torch-1.1.0rc0-cp39-abi3-manylinux_2_28_x86_64.whl",
      "sbom": "tritium-torch-linux-cpu.cdx.json"
    }
  ]
}
```

## Assemble and verify

```bash
cargo build --release -p tritium-cli
python scripts/assemble-release-candidate.py \
  --inputs /tmp/tritium-v1.1-inputs.json \
  --output release/v1.1/manifest.json \
  --digest-tool target/release/tritium
scripts/release-status \
  --candidate release/v1.1/manifest.json \
  --digest-tool target/release/tritium
```

Assembly sorts artifacts by ID, streams SHA-256/BLAKE3 through shipped CLI,
writes canonical in-toto/SLSA v1 statements, fsyncs data and directories, then
strictly reloads generated candidate. It never overwrites existing manifest or
provenance. For `model-bundle` and `onnx-bundle` inputs, a missing named SBOM is
generated before provenance publication; a supplied SBOM must reproduce the
same canonical inventory exactly. Failure removes every SBOM and metadata file
created by that assembly attempt.

Verification requires:

- canonical `1.1.0-rc.N` version matching Cargo, Python, npm and compatibility
  mirrors;
- clean checkout at exact source revision;
- ordinary contained files with no symlink traversal or unmanifested payload;
- exact bytes, SHA-256 and BLAKE3 for every artifact;
- digest-bound CycloneDX/SPDX SBOM tied to exact artifact ID;
- digest-bound in-toto/SLSA v1 provenance tied to artifact SHA-256, source
  revision and builder identity.

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
  --digest-tool target/release/tritium \
  --output release/v1.1/qwen-onnx.cdx.json
```

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
  --digest-tool target/release/tritium
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
signing key, then verify it against a reviewed `allowed_signers` file:

```bash
python scripts/local-rc-signoff.py seal \
  --report release/v1.1-evidence/status.json \
  --registry release/v1.1-evidence/registry.json \
  --candidate release/v1.1/manifest.json \
  --principal release-maintainer --key /secure/release-key \
  --output release/v1.1-evidence/signoff.json
python scripts/local-rc-signoff.py verify \
  --report release/v1.1-evidence/status.json \
  --registry release/v1.1-evidence/registry.json \
  --candidate release/v1.1/manifest.json \
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

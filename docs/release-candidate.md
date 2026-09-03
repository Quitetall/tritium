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
  "release": "1.1.0-rc.1",
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
      "path": "pytritium-1.1.0rc1-cp39-abi3-manylinux_2_28_x86_64.whl",
      "sbom": "pytritium-linux-cpu.cdx.json"
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
  --archive release/v1.1/tritium-ai-web-1.1.0-rc.1.tgz \
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
  --digest-tool target/release/tritium \
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
  --release 1.1.0-rc.1 \
  --output release/v1.1/tritium-1.1.0-rc.1.tgz
python scripts/generate-deployment-sbom.py \
  --artifact release/v1.1/tritium-1.1.0-rc.1.tgz \
  --artifact-id tritium-helm \
  --kind helm-chart \
  --release 1.1.0-rc.1 \
  --source-revision "$(git rev-parse HEAD)" \
  --digest-tool target/release/tritium \
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
signing key. Sign-off re-runs canonical candidate admission, including every
artifact identity, SBOM, provenance and closed-directory check, through the
same digest tool used for release status. Then verify it against a reviewed
`allowed_signers` file:

```bash
python scripts/local-rc-signoff.py seal \
  --report release/v1.1-evidence/status.json \
  --registry release/v1.1-evidence/registry.json \
  --candidate release/v1.1/manifest.json \
  --digest-tool target/release/tritium \
  --principal release-maintainer --key /secure/release-key \
  --output release/v1.1-evidence/signoff.json
python scripts/local-rc-signoff.py verify \
  --report release/v1.1-evidence/status.json \
  --registry release/v1.1-evidence/registry.json \
  --candidate release/v1.1/manifest.json \
  --digest-tool target/release/tritium \
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

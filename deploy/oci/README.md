# Qualified OCI inputs

These files replace the former developer root `Dockerfile`. Both images use
digest-pinned builders and a shell-free, package-manager-free distroless runtime.
The build wrapper creates a clean `git archive`, vendors the locked Cargo graph,
builds with network disabled, and requests BuildKit SBOM plus SLSA provenance
attestations in the OCI archive. After strict layout verification, it emits a
CycloneDX document beside the archive. That document inventories every transport
member and closes the OCI descriptor graph; image-bound embedded attestations do
not replace exact archive-byte binding.

```bash
export TRITIUM_OCI_BUILDER_ID=https://github.com/OWNER/REPOSITORY/actions/runs/RUN_ID
scripts/build-oci-candidate cpu release/v1.1/manifest.json /absolute/empty/output
scripts/build-oci-candidate cuda release/v1.1/manifest.json /absolute/empty/output
```

The builder first admits the plan-0051 candidate manifest, requires its exact
source revision, retains the deterministic source archive, and binds both
candidate and source identities in its build receipt. `TRITIUM_OCI_BUILDER_ID`
is mandatory and must be a safe HTTPS workflow/run identity. BuildKit emits
SLSA provenance v1 in `mode=max`; admission requires its semantic BuildKit
definition, exact `SOURCE_REVISION` argument, resolved dependencies, LLB graph,
builder URI, invocation ID, and timestamps. Its SPDX predicate must contain a
real document plus at least one package. Builder and invocation identities are
also copied into the exact transport CycloneDX inventory and must match outer
candidate provenance. All outputs build in a sibling staging directory; only a
fully verified archive, receipt, checksums, source and SBOM replace the empty
destination atomically. The current contract is Linux/amd64. A release gate
must inspect the emitted OCI
archive, admit its attached attestations, load it by digest, then run it with an
arbitrary UID, `--read-only`, `--cap-drop=ALL`, `no-new-privileges`, a bounded
`/tmp` tmpfs, and a read-only model mount. The Compose files encode those runtime
controls. Launch them through `scripts/run-oci-compose`, which rejects mutable
image tags. They remain compatibility profiles—not smoke evidence—until an
external harness proves readiness, requests, drain, write rejection and receipt
parity, and until the strict schema-v3 serving loader replaces the legacy GGUF
binary path.

No image is pushed by this workflow. CUDA qualification additionally requires
an NVIDIA host and must record driver, runtime, toolkit, GPU, and image identities.

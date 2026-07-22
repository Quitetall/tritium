# Qualified OCI inputs

These files replace the former developer root `Dockerfile`. Both images use
digest-pinned builders and a shell-free, package-manager-free distroless runtime.
The build wrapper creates a clean `git archive`, vendors the locked Cargo graph,
builds with network disabled, and requests BuildKit SBOM plus SLSA provenance
attestations in the OCI archive.

```bash
scripts/build-oci-candidate cpu release/v1.1/manifest.json /absolute/empty/output
scripts/build-oci-candidate cuda release/v1.1/manifest.json /absolute/empty/output
```

The builder first admits the plan-0051 candidate manifest, requires its exact
source revision, retains the deterministic source archive, and binds both
candidate and source identities in its build receipt. The current contract is
Linux/amd64. A release gate must inspect the emitted OCI
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

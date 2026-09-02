# Security policy & threat model

Tritium is ternary-model inference and training infrastructure. This document
states the supported-version policy, trust boundaries, hardening, and private
vulnerability-reporting process for the v1.1 release line.

## Supported versions

The latest stable `1.x` minor receives correctness and security fixes. The
immediately previous stable minor receives critical/high security fixes for 90
days after the next stable minor. Release candidates receive development fixes
but may change before the stable release. Exact details are in
[SUPPORT.md](SUPPORT.md); a backend or package is supported only for cells marked
`qualified` in the generated compatibility matrix.

> **Full threat model:** the complete surface-by-surface analysis — 30 threats
> across the model-file parsers, the C ABI / FFI, the HTTP server, the compute
> kernels + backend dispatch, and the supply-chain + build pipeline, each with
> mitigations cited to source (file/line or CI lane) and residual risks — is in
> [`docs/security/threat-model.md`](docs/security/threat-model.md). This page is
> the policy + trust-model summary; that document is the detailed review.

## Trust model

The primary untrusted input is a **model file**: weights, config metadata, and
tokenizer data loaded from disk or the network. Tritium parses several such
formats, all of which must be treated as attacker-controlled bytes:

- **GGUF** (`tritium-format`: `read_gguf`) — the BitNet/LLaMA weight container.
- **safetensors** — the HF tensor container.
- **SALT** bundles + the legacy SALT-in-GGUF + sparse-plane stacks (the ternary
  quantization artifacts).
- **`.tqbin` / `.tqidx`** — the training corpus + its shuffle index.

A model file is **data, not code**: Tritium never executes code embedded in a
model, so loading a hostile file cannot, by construction, run attacker code. The
realistic threats are therefore **memory-safety** and **resource-exhaustion** bugs
in the parsers, plus **supply-chain** compromise of dependencies.

### Trust boundary

```
untrusted bytes (model file / corpus)
        │
        ▼
  tritium-format parsers  ← THE trust boundary: bounds-checked, never-panic
        │
        ▼
  validated in-memory structs → backends / runner (trusted)
```

## Hardening (what defends the boundary)

- **Never-panic parsing.** Every parser reads through a bounds-checked cursor with
  little-endian framing and returns `Result`; a malformed or truncated file yields
  a typed error, never an out-of-bounds read, panic, or `unwrap` on attacker data.
- **Continuous fuzzing.** Every untrusted-byte parser has a `cargo-fuzz` target
  (`gguf_parse`, `safetensors_parse`, `salt_bundle_parse`, `salt_gguf_parse`,
  `salt_legacy_parse`, `sparse_plane_parse`, `tqbin_parse`, `tqidx_parse`,
  `unpack_i2s`, `unpack_tq_rows`, `zero_bitmap`) run on a
  scheduled CI lane; the v0.90 gate is ≥24h cumulative fuzzing with zero open
  findings and committed corpora.
- **Memory safety.** The foundation crates are `#![forbid(unsafe_code)]` /
  `#![deny(unsafe_code)]`; the few `unsafe` blocks (SIMD intrinsic kernels, the
  `linkme` registration static) are narrowly scoped and carry `// SAFETY:` notes.
- **Resource bounds.** Size/count fields from a header are validated against the
  actual buffer length **before** allocation, so a malicious header cannot trigger
  an unbounded allocation; size arithmetic is overflow-checked.
- **Supply chain.** `cargo-deny` gates licenses, RustSec advisories, and banned /
  duplicate crates on every push; an SBOM (CycloneDX) is generated in CI;
  dependency requirements forbid wildcards, the workspace lockfile binds resolved
  versions, and higher-risk native/network dependencies use exact requirements.

## Out of scope

- **Model-output safety / alignment.** Tritium is an execution engine; it does not
  filter or align generated text.
- **Operator-selected capacity.** Tritium validates declared artifact and runtime
  ceilings, but the operator remains responsible for choosing limits appropriate
  to the host and workload.
- **The training/serving host.** Sandboxing the process (seccomp, cgroups,
  containers) is a deployment concern, not the library's.

## Reporting a vulnerability

Report suspected vulnerabilities through
[GitHub's private vulnerability reporting](https://github.com/Quitetall/tritium/security/advisories/new)
— the **Report a vulnerability** button on the Security tab. That route keeps the
report private, gives us a place to draft a fix and an advisory with you, and can
issue a CVE. Email `briankhanglam@gmail.com` only if you cannot use it.

Either way, do not use an issue, discussion, or chat, and do not send proprietary
model or dataset contents unless a safe transfer has been agreed.

Include the affected version and full source/artifact identity, impact, minimal
reproduction, required environment, and whether the issue is already public.
You should receive acknowledgement within three business days and an initial
severity/coordination assessment within seven business days. These are response
targets, not a commercial SLA. Fix and disclosure timing depends on severity,
exploitability, downstream coordination, and release readiness.

The reporter and maintainer coordinate an embargo and advisory when warranted.
Credit is offered unless the reporter prefers anonymity. Public disclosure
before a mitigation is available may be necessary for active exploitation or
material user risk, but the reason and remaining exposure will be documented.

## Supply-chain and deployment boundary

Release artifacts are admitted by digest, source revision, and SBOM gates.
Cryptographic build provenance and artifact signatures are **not yet
implemented** — do not treat a downloaded artifact as attested. Model/tokenizer licenses and authenticity remain separate
from numerical compatibility: a loadable artifact is not automatically trusted
or redistributable. Operators must verify candidate/public manifests and should
run serving images with the documented non-root, read-only, bounded-resource
posture. Compromised registries, CI identities, signing keys, model sources, and
runtime dependencies are in scope for private reporting.

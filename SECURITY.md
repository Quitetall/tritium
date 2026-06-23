# Security policy & threat model

Tritium is a from-scratch ternary-LLM inference + training library. This document
states the trust model, the threat boundary for **untrusted model files**, the
hardening that defends it, and how to report a vulnerability. It is the v0.90
security gate (ADR 0011).

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
  `salt_legacy_parse`, `sparse_plane_parse`, `tqbin_parse`, `tqidx_parse`) run on a
  scheduled CI lane; the v0.90 gate is ≥24h cumulative fuzzing with zero open
  findings and committed corpora.
- **Memory safety.** The foundation crates are `#![forbid(unsafe_code)]` /
  `#![deny(unsafe_code)]`; the few `unsafe` blocks (SIMD intrinsic kernels, the
  `linkme` registration static) are narrowly scoped and carry `// SAFETY:` notes.
- **Resource bounds.** Size/count fields from a header are validated against the
  actual buffer length **before** allocation, so a malicious header cannot trigger
  an unbounded allocation; size arithmetic is overflow-checked.
- **Supply chain.** `cargo-deny` gates licenses, RustSec advisories, and banned /
  duplicate crates on every push; an SBOM (CycloneDX) is generated in CI; external
  dependency versions are explicitly pinned (no wildcards).

## Out of scope

- **Model-output safety / alignment.** Tritium is an execution engine; it does not
  filter or align generated text.
- **Resource limits on valid-but-huge models.** A legitimately enormous model can
  exhaust memory; bounding that is the embedding application's concern (e.g. the
  `tritium-serve` operator sets process limits).
- **The training/serving host.** Sandboxing the process (seccomp, cgroups,
  containers) is a deployment concern, not the library's.

## Reporting a vulnerability

Report suspected vulnerabilities privately to the maintainer
(briankhanglam@gmail.com) rather than via a public issue. Please include a
reproduction (ideally a minimized input file) and the affected version/commit.
Pre-1.0, there is no formal embargo SLA, but reports are triaged promptly.

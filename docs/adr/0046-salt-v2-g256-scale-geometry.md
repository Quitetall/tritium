# ADR 0046 — SALT V2 G256 scale geometry

Status: **PROPOSED** (2026-10-03)

- **Deciders:** Brian Lam
- **Relates:** [ADR 0028](./0028-salt-v2-additive-ternarization.md),
  [plan 0043](../plans/0043-salt-v2-sota-campaign.md), and the SALT V2 package
  contract in `crates/tritium-format/src/salt_v2_package.rs`.

## Context

The frozen Stage-7 recipe grid includes G64, G128, and G256 scale-group
experiments. G128 packages use SALT V2 package version 1. Version 2 already
adds explicit per-tensor scale geometry for G64, but currently rejects G256.
That makes the planned G256 recipe impossible to measure through the real
format and CUDA path.

## Proposal

Extend the existing version-2 scale-geometry tag without changing package
version or the meaning of existing bytes:

| Package version | Geometry tag | Scale group |
|---|---:|---:|
| 1 | 0 | G128 |
| 2 | 0 | G128 |
| 2 | 1 | G64 |
| 2 | 2 | G256 |

All other geometry tags remain invalid. Version 1 remains G128-only and
byte-identical. A G64 or G256 package is emitted as version 2. Readers that do
not know tag 2 must reject it; they must not guess a geometry or silently
reinterpret the package.

This adds a representable research candidate, not a new default. G128 remains
the default. G64 and G256 remain experimental until the preregistered matched
quality, physical-byte, and runtime gates promote a candidate. No quality,
compression, or speed claim follows from format or kernel parity alone.

## Compatibility and rollback

Existing version-1 and version-2 tag-0/tag-1 packages retain their exact
interpretation. New version-2 tag-2 packages are intentionally unreadable by
older readers that reject unknown geometry. Producers must only select G256
when explicitly configured. If a future geometry cannot fit the remaining
version-2 tag space, it requires a new package version rather than reusing a
tag.

The implementation must test canonical version selection, package
round-tripping, rejection of a G256 tag under version 1, CPU parity, streamed
reader parity, CUDA exact/fast behavior, and the Stage-7 sanitizer matrix.
Rollback disables G256 production while preserving readers for already emitted
version-2 tag-2 packages; removing reader support requires a separate migration
decision.

## Status boundary

This proposal does not authorize promoting G256 into the frozen winning
recipe, changing byte ceilings, or starting the Qwen flagship run. Those remain
governed by ADR 0028 and plan 0043. Explicit owner acceptance is required before
calling this wire-format extension an adopted public contract.

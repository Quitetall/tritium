# Promoted decision records

This directory holds the ADRs whose decision **shipped**. It is not the full research
record — see [`docs/RESEARCH.md`](../RESEARCH.md) for the split and the reason for it.

Each entry names the change that implemented it. That citation is the promotion
criterion: an ADR without one does not belong here yet.

| ADR | decision | shipped in |
|---|---|---|
| _(none promoted yet)_ | | |

## Promoting a record

1. Confirm the decision is actually implemented in this repository — code merged, not
   merely planned or measured.
2. Copy the ADR from `Quitetall/tritium-research` into this directory, unchanged.
   Promote the campaign plan it was measured against alongside it, so the
   preregistered gate arrives with the decision.
3. Add a row above citing the implementing PR or commit.
4. If the record contains a claim later retracted, do not silently edit it — append a
   dated amendment. The point of publishing a frozen gate is that it stays frozen.

## Why this is manual

"Did this ship, and does a reader of the codebase benefit from it?" is a judgement.
It also cannot be derived from the records themselves today: of 41 ADRs, 18 carry no
status line and exactly one says `IMPLEMENTED`.

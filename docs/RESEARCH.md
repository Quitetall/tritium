# Research, ADRs, and campaign plans

Tritium's research runs in two repositories, and the split is about **signal**, not
secrecy.

| | repository | holds |
|---|---|---|
| working front | `Quitetall/tritium-research` (private) | every ADR, preregistered campaign plan, benchmark-receipt archive, and the living roadmap — including the ones that go nowhere |
| this repo | `Quitetall/tritium` (public) | the records whose decision **shipped**, promoted once the code lands |

Most research does not survive contact with measurement. This project has refuted its
own levers repeatedly — a ternary teacher turned out to be redundant, per-group
curvature allocation measured *worse* than uniform, a flash drafter was deleted after
a −2.7% end-to-end result. Those refutations are real work and they are kept, but
publishing all of them here would bury the handful of decisions a reader of this
codebase actually needs.

So the public record is not "all our research". It is **the research that became
code**.

## Promotion rule

An ADR moves from the private repo into `docs/adr/` here when its decision has been
implemented, and **the promoting pull request must cite the commit or PR that
implemented it**. No implementation, no promotion — that is the whole mechanism, and
it is deliberately manual: "did this actually ship, and does a reader benefit?" is a
judgement, not a status field.

(It could not be automated today in any case: of 41 ADRs, 18 carry no status line and
exactly one says `IMPLEMENTED`. Normalising that vocabulary is worth doing on its own
merits, but it is not what gates promotion.)

Campaign plans follow the same rule and are promoted with the ADR they served, so a
promoted decision arrives with the preregistered gate it was measured against.

## On preregistration

Preregistration is load-bearing: quality campaigns freeze their acceptance thresholds
*before* any scored run, and a promoted plan is published with the gate it committed
to — pass or fail — so a claim cannot be quietly moved after the fact.

One honest caveat. Research documents were tracked in this repository until
2026-07-30 and removed in `5b0fd3cc`. `git rm` does not rewrite history, so those
files remain readable from this repository's history. Treat the pre-2026-07-30
corpus as public. Everything after that date follows the rule above.

## What lives here regardless

- [`docs/BENCHMARKS.md`](./BENCHMARKS.md) — the reproducible benchmark ledger (every
  number beside its exact command and environment).
- [`docs/compatibility.md`](./compatibility.md) — the generated, receipt-backed
  support matrix.
- [`docs/ternary-formats.md`](./ternary-formats.md) — format documentation.
- The mdbook under `docs/book/` — the user guide.

Want a specific decision or plan? Open an issue. Records are promoted on request when
the work they describe has shipped.

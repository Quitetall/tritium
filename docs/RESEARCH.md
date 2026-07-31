# Research, ADRs, and campaign plans

Tritium's research notes, Architecture Decision Records, preregistered
campaign plans, benchmark-receipt archives, and the living roadmap are
maintained in a separate repository and **published alongside their
results**, not before.

Preregistration is load-bearing here: quality campaigns (for example the
Qwen3.6-27B flagship conversion gating v1.1 stable) freeze their acceptance
thresholds *before* any scored run, and the frozen gates are re-published
with the results — pass or fail — so a claim can never be quietly moved
after the fact.

What lives in-repo instead:

- [`docs/BENCHMARKS.md`](./BENCHMARKS.md) — the reproducible benchmark
  ledger (every number beside its exact command and environment).
- [`docs/compatibility.md`](./compatibility.md) — the generated,
  receipt-backed support matrix.
- [`docs/ternary-formats.md`](./ternary-formats.md) — format documentation.
- The mdbook under `docs/book/` — the user guide.

Questions about a specific decision or plan? Open an issue — relevant
records are published on request or with their results.

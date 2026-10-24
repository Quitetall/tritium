# tritium-salt

SALT V2 campaign orchestration: evidence stores, receipts, resumable conversion pipelines.

Plan-0043 Stage-7 evidence uses `Stage7TokenEvidencePack`: a strict,
campaign-bound, seek-backed token reader with bounded sequence windows and a
terminal same-handle integrity check.

`QteaTensorWorkPlan` binds a checked native QTEA numerical plan to the campaign
source, canonical tensor owner/index and original-storage digest. It checks codec
limits and a combined logical fit/export byte ceiling before executing, then
returns the actual complete-file identity only after a successful unpublished
write. Caller sink buffers, other model/cache/process state and physical RAM
measurements are separate scope. Source binding is not authenticated custody.
This is residual-bearing tensor work, not a pure additive-ternary artifact,
registered frontier solver, deployable model or qualification receipt. The QTEA
builtin remains unavailable until its complete admission/conformance work lands.

The bounded synthetic `qtea_conformance` example exercises the same public
work-plan writer, actual cold reader and packed CPU forward. The repository's
`scripts/verify-qtea-reference.sh` can explicitly select `tritium-salt` as its fifth
argument to compare its24 observations with the separately supplied, SHA-bound
Python reference. The default `tritium-cpu` producer remains a direct-fitter
comparison; neither producer certifies quality or performance.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

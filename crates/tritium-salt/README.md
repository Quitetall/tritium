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
qualified model runtime, deployable model or qualification receipt.

`QteaReferenceAdapter` is an installable **Experimental CPU tensor fitter** when
paired with an explicit caller `FrontierResourceEstimator`. Its catalog descriptor
matches that real implementation. Registry creation installs nothing automatically;
minimum-Registered and Certified profiles still refuse it. Availability does not
promise model capture/import/loading, GPU, quality, performance or source custody.

Its optional `admit` path consumes an existing Experimental registry plan and
portable Fit request plus the caller's independently selected plan ID. It checks
descriptor, source/input/shape and known logical resource floors before execution.
V1 requests alone do not bind machine/evidence IDs; the selected plan ID and
returned stage output do. Execution cannot create Registered/Certified status.
Receipts require explicit caller observations and
cannot underreport known raw artifact length; they are not qualification.

The bounded synthetic `qtea_conformance` example exercises the same public
adapter/registry/admitted writer, actual cold reader and packed CPU forward. Only
the resource-estimator fixture is synthetic, not the mathematical solver.
The repository's
`scripts/verify-qtea-reference.sh` can explicitly select `tritium-salt` as its fifth
argument to compare its24 observations with the separately supplied, SHA-bound
Python reference. The default `tritium-cpu` producer remains a direct-fitter
comparison; neither producer certifies quality or performance.

`QteaTensorWorkPlan::from_calibrated` consumes the common immutable
`CalibratedTensor` produced by `CalibrationCollector` or finalized-cache replay.
It borrows the actual full input Gram without another reducer, format or matrix
copy. Exact expected tensor/precision/model/owner must agree; grouped, gather and
output-Fisher inputs refuse rather than being relabelled as full input capture.
The common selected-weight normalization is preserved, not claimed as author
sequence-normalized capture reproduction. This is still Experimental; original
stored-byte authenticity, corpus completeness, model loading and physical memory
remain separate obligations. Capture and fit/export have separate logical caps.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

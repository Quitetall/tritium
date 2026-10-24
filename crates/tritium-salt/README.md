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

## Observed-source QTEA model building (experimental)

`QteaModelBuilder` joins actual original-byte observation, the common calibrated
bridge/native fitter, exact caller-selected admission and the seek-backed model
writer (proposed ADR0167). It requires an `ObservedWeightSource`, explicit
`QteaModelProfile`, campaign identity, empty private spool and positive
`QteaModelBuildLimits`. Source config/names/original shapes are admitted through
the same `QteaModelInventory` plans as cold loading, before tensor fitting.
The observer's input plan is observation only; its actions are not QTEA output
recipes or completed KeepDense fitting evidence.

For inspect-then-run selection, `inspect_owner(name, curvature, recipe, callback)`
prepares actual observed bytes and returns caller-owned data from a borrowed work
plan (proposed ADR0169). It uses the same preparation as fitting and never accesses
the spool, fits or marks an owner complete. Inspection errors are retryable; they
do not poison the cohort. Widened weights are dropped, not cached for execution.
Inspection is not SPD validation, aggregate spool availability, physical admission
or a fit receipt. Bind the selected request to the inspected work identity: fitting
rereads the source and reconstructs the work, so changed bytes/curvature/recipes
cannot reuse a stale request. Existing selected plan identity checks still apply.

Call `fit_owner` exactly once for every canonical matrix. Supply either actual
`QteaOwnerCurvature::Calibrated` Full/Matmul/input-only statistics or an explicit
`Native` bound Gram; grouped/gather/Fisher data do not become an identity fallback.
In particular, native embedding metric selection is explicit caller work, not a
claim that embedding gather was captured as matmul. The callback supplies the
existing opaque selected plan, exact request and separately selected expected ID
through `QteaSelectedAdmission`. It never creates measurements or a completed
frontier stage receipt. Tensor source bytes are checked and widened one at a time
with the shared decoder; known widened/spool refusals precede fitting.

Any failed/duplicate/unsupported `fit_owner` poisons the unpublished cohort. `finish`
requires a complete matrix inventory and empty private Read/Write/Seek output.
It preserves original non-matrix float state and asset bytes, adds bounded
`qtea-fit.json` software observations, writes from the bounded spool and checks
all actual wire/record/numerical identity bindings. Whole original-source
revalidation and final output verification precede a successful result.

`verify_qtea_model_build_record` repeats structural package/observation binding
checks with the existing identity derivations. It does not reconstruct an opaque
admission token, authenticate original/capture/producer execution, approve a
resource estimate or rerun the fitting algorithm. Record identities and estimates
are not measured resource vectors or source/quality signatures. The asset is not
a sealed production model-fit log and cannot satisfy Stage7 by itself.

The `finish` result is a private candidate stream, NOT fsync,
no-clobber directory publication or distribution permission. Late errors may
leave private spool/output bytes; the caller must discard them. Logical source,
widened/native/export, spool and observation caps exclude caller-retained
calibration/model/metadata/allocator/process RSS. No new dense model shadow,
automatic planner, GPU promise or trust promotion is introduced.

The `host-qtea` feature enables matching NN cold-runtime checks. Tests cover full
standard tied/untied and hybrid language/retained-MTP builds from F32/F16/BF16
originals, actual collector/native admitted work, exact asset/state preservation,
partial writers, source mutation and tampered records. MTP remains Unverified;
real-model capture/metric choice, CLI/Python campaign orchestration, durable
qualified release publication and independent qualification remain required.

For explicit local candidate storage, `builder.publish_candidate(&mut sink)`
uses `DirectoryTensorSink`'s shared unqualified transaction (proposed ADR0168).
Complete `finish` source/numeric/observation checks run in the private pending
file; the sink then independently checks actual framing/blob/package identity,
syncs the files and commits without overwrite. Output has `candidate.json` with
`qualification=false`, not a counterfeit sealed `evidence.jsonl`. Returned
post-commit warnings must be reported; they are not permission to roll back.
No published candidate becomes Registered/Certified, verified MTP, authenticated
source, qualified runtime or an approved public release merely by being stored.

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

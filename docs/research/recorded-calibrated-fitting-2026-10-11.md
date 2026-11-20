# Recorded calibrated fitting candidate

Implements proposed research ADR0061; not a frozen profile or release verdict.
`fit_calibrated_levels_recorded` consumes an actual immutable CalibrationCache,
validates its complete expected binding and original geometry, preflights direct
limits and the recorder, then invokes the same cache reducer and calibrated
numerical controller. The existing source-f32 digest and output semantic hash
are shared with uncalibrated recording; no numerical/hash/forward clone is added.

The new registered `tritium.quantize.calibrated_levels_fitted@1` event has a
Rust-owned CalibratedFitPlan, actual CalibrationFactors, source-f32 digest,
Rayon pool size and ordered output identities. Strict typed fold rechecks log
integrity and cache/plan/factor/selection/geometry/level correspondence. CLI
evidence verify/view uses that same fold. The old direct_levels_fitted@1 payload,
hash domains and semantics are unchanged; no old Identity/raw-Gram observation
is relabeled as K-FAC. No separate receipt file or independent wire vocabulary.

Tests compare actual cache/reducer/controller interfaces on full and ragged
grouped statistics, input-only/dense/indexed/gather factors, source/data/tokens/
recipe changes, rejection without success, existing zero-Fisher damping,
cross-field tampering with otherwise valid generic chains, unsupported versions,
unknown fields and sealed replay. They do not establish original raw BF16 model
truth, declared recipe execution, numerical quality, speed or physical memory.

Whole-model Engine/QuantPlan lifecycle and source equality, sealed calibrated
package replay, structured/lazy metric reuse, spool/mmap/stream decode,
GPU/out-of-core/full RSS, complete architecture/MTP taps, adapter/legacy/campaign
migration and all independent empirical release gates remain required.

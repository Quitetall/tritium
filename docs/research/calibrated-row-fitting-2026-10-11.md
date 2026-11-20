# Actual calibration factors in direct row fitting

Proposed ADR0060 numerical consumer on base
c7f04a78cc3fa00be35c6dc7401028562d40176f. Reuses an owned clean checkout after
its previous verifier finished; the separate live c7f04a78 broad gate is untouched.

`fit_calibrated_levels` checks the expected original tensor/model context,
fit geometry, actual source shape/finite values and aggregate existing direct
limits before numerical/output allocation. Full inputs retain cross-group
correlation. Explicit grouped statistics are assembled block-diagonally, including
ragged widths independent of weight group/basis. Actual output rows use the
existing K-FAC builder's Fisher scalar and absolute damping; the selected direct
profile retains its separately declared existing regularization. No rows are
averaged and input-only statistics never acquire fictitious output gradients.

One assembled row matrix is live, and each original row is passed to the unchanged
direct controller for all named levels. Basis transforms, fitter arithmetic,
feedback and stored scale rounding remain owned by that controller. The existing
owned tensor checks assembled plane-major trits and tied/free scale installation.
Source and calibration factors stay read-only. This retains numerical copies,
output/calibration buffers and solver state; it is not full RSS/GPU/out-of-core
admission or a performance claim.

Context equality does not independently prove caller f32 weights came from the
raw source identified by the supplied model ID. No existing v1 event is altered
or relabeled, no fit provenance/root or release verdict is invented. Registered
calibrated observations, complete engine source equality and log/package replay
remain required before describing model fits as audited calibrated ancestry.

Candidate checks exercise six profiles, two codecs, Identity/ragged and distinct
Hadamard/weight-group widths, actual per-row K-FAC and fresh original level fits,
codec roundtrip and core execution, all four input/output encodings, full/grouped
diagonal parity, expected context/aggregate bounds/invalid source rejection and
zero-Fisher/damping semantics. Synthetic fixture tokens, inputs and gradients
are actual cache bytes; these are numerical checks, not independent objective,
model-quality or physical performance proof. Guarded invocation
365caf9cf20149fda0037e59665156db ended exit101: three new tests passed and the
zero-Fisher test incorrectly expected an undamped zero matrix to be accepted.
The existing DensePsdMetric rejects ZeroMetric; build_kfac_metric maps this to
InvalidKfacMetric. Preserved that failure and original source/control snapshot.
After terminal confirmation, corrected only test expectations/helper vocabulary,
not the production metric/controller or PSD gate. The production three-file
hash index is unchanged. Corrected invocation613a0888beb14d92b290e89a6053fc53
ended terminal exit0 with all four source and base HEAD guards unchanged.
The four new numerical tests and existing calibration/cache/direct regressions
passed, as did strict all-target quantize Clippy, strict Rustdoc and workspace
formatting. Zero Fisher without positive damping remains an explicit rejection,
not an identity fallback. These are scoped software results only. Exact committed
tree broad gates and normal public push remain pending.

Full source/data/recipe/metric fitting/replay, immutable source, source/window
spooling/mmap/streaming decode/RSS; direct allocation/full f64/frontier/refinement/
GPU-out-of-core/thin frontend migration; Qwen hybrid/MTP/quality/runtime/bytes/
reproduction and all backend/browser/training/serving/RC/security/zoo/community/
independent release/human activation obligations remain open.

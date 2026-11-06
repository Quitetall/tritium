# Shared paired salience fold: software candidate

The numerical fold is now owned by tritium-quantize::engine. NN calibration is
the actual model graph adapter, not another numerical implementation. Existing
NN smooth_scales/scale_cols/divide_rows imports remain compatibility reexports.
SaliencePair applies column multiplication and inverse norm/linear output-row
division together. It preflights statistics, alpha, shapes, actual f32 scales and
finite outputs before changing either caller-owned half. It does not infer graph
eligibility from tensor names or establish a calibration/source/recipe claim.

NN's fallible try_fold preserves the supported tied-head SwiGLU graph order:
q/k/v plus attention norm; gate/up plus feed-forward norm; down plus up output
rows. Up's input multiplication precedes its output division. The tied embedding,
o_proj and final norm remain unchanged. Legacy fold delegates with its existing
infallible return shape, panicking on rejected input. Checked inputs require
positive sample/channel counts, alpha in [0,1], finite nonnegative moments,
complete compatible graph geometry and representable finite transformed values.
Malformed inputs formerly producing partial/silent results are not compatibility
targets. Algebraic cancellation is not a bit-identical logit or quality claim.

The implementation keeps the old ordered f64 RMS/geometric-mean/powf computation,
f32 scale cast and f32 multiply/divide. It does not combine scale factors or
rewrite salience into an Identity metric. The NN adapter still clones the model,
as the legacy path did; neither this nor preflight scanning establishes bounded
full-process RSS, faster conversion or out-of-core execution.

Scoped verifier1d47e2921eb740ee90f2bdbc1e59661a completed exit0: five shared
paired tests, two NN graph/oracle tests and all166 NN library tests passed;
strict all-target scoped Clippy, docs and fmt passed. The same caller fixture
then ran the actual clean pre-change production fold at
bea5a0d4f007cb9ce19e6e2d9bb2f13ef6b4c24f. Eighteen contexts (1/3 layers,
zero/uniform/nonuniform moments, alpha0/0.75/1) produced bit-identical weights
and norms. The oracle contains no fold formula or copied production pipeline.
Its temporary baseline test was removed and that checkout verified clean.
Initial missing-Debug and subsequent broken rustdoc-link failures are retained
alongside corrected source/control/log/exit, not overwritten as passing runs.

This is a numerical migration prerequisite under proposed ADR0064, not full
model-aware CLI migration. Typed original A -> transformed dense B -> fitted C
observations, executed capture context, actual source/cache binding and complete
cross-stage package admission are still required. KeepDense on B must not be
reported as preserving A; fitting caches collected on B cannot claim A's identity.
No old campaign or source artifact is migrated, deleted or retroactively admitted.

Commit-tree/canonical/hosted checks follow separately. These software fixtures
do not qualify real Qwen quality, GPU/backend execution, performance, physical
byte reduction, independent reproduction, SOTA status or public release. Full
ADR0033/P0-P7 fitting/runtime/hardware/training/serving/packaging/zoo/community
and independent/human release gates remain open.

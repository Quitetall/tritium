# Observed original-source calibration

Implements the raw-source prerequisite amendment in research ADR0064. This is
an experimental software path, not original-A salience capture, captured
statistics qualification, out-of-core/full RSS, real-model quality or release
clearance. Existing stage/C profiles and their bytes are unchanged.

ObservedWeightSource observes actual complete WeightSource contents under the
existing QuantPlan limits. Plan actions are validated but not executed; no fit,
package or evidence is fabricated. Engine::observe_source uses this same
implementation, existing dense payload binding and original-source identity
derivation. A pre-change actual production observer probe preserves twelve
F32/F16/Bf16 and asset/signed-zero/inventory contexts for byte comparison.

The adapter borrows the raw provider and retains only immutable expected
interpretations, content identities and asset lengths, not dense raw payloads.
Every returned tensor/asset must match those expected bytes before decoding.
Caller caps are checked before provider reads. Complete inventory checks reject
changed names/geometry/precision/assets; explicit revalidate rereads all raw
contents through those same checked reads. This is not a file lock or snapshot,
and its frozen source_id alone does not certify current or future mutable files.
Input Grams keep the existing explicit semantics; none is synthesized and no
calibration recipe/objective/source claim is inferred from their presence.

SourceCalibrationModel consumes this checked raw adapter. StagedCalibrationModel
keeps its explicit B-stage construction and capture semantics. Both use one
private dense loader, StandardModelPlan/NameSchema/config/optional QK admission,
dense decoder/assembler, owned immutable masters and tape projection graph.
The source adapter is not retained after construction. No identity-transform
stage, second graph, runner extraction, full raw clone or hash assertion stands
in for original A. Per-window master copies/tape allocations remain explicit;
the logical config/master/token ceilings are not process RSS admission.

Tests cross actual source/model interfaces: memory and seekable safetensors
shards, exact dtype/signed-zero/raw assets, normal guarded fitting, foreign
names/payloads/metadata/caps/Grams, externally changed safetensors payloads,
original masters outliving raw sources, existing legacy/CPU/reference loaded
execution, and invalid windows before tape mutation. These fixtures do not
qualify real Qwen or physical hardware. Existing B capture/replay/runtime tests
must also pass after sharing the implementation. No pass is claimed until the
actual frozen candidate executes its focused and broader checks.

Next: record actual original-A salience token/text execution and ordered f64
moments/counts, cold original graph replay and mandatory ancestry; complete
gather/biased/untied/hybrid/MoE/MTP graphs; actual convert/quantize/Qwen/Python/HF
and both campaign migrations; full SOTA/refinement/GPU/out-of-core/RSS/format,
native/hardware/browser/distributed training, language+MTP quality/runtime/
physical bytes/independent reproduction, production serving/security/packaging/
ONNX/Colab/zoo/docs/community/signing and explicit human release activation.

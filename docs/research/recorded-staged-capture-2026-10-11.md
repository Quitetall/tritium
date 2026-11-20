# Recorded staged capture and cold replay

Implements the recorded B capture amendment in research ADR0064. This is an
additive software interface, not owner acceptance, quality, physical performance,
SOTA, complete campaign migration or public release activation.

`StagedCalibrationModel::capture_next_token_recorded` executes the existing
token-ID capture controller and seals registered capture start/completion events.
With the tokenizer feature, `capture_next_token_text_recorded` first executes
the shared runtime tokenizer on the actual stage assets and records exact raw
text plus the actual encoded windows. Logs can contain private calibration text;
do not publish them without data authorization.

The controller derives stage PackageId, stored-byte B source, mandatory transform
reference, tokenizer/config identities and ordered raw matrix interpretations
from the admitted stage/model. It does not accept caller provenance hashes.
Its fixed profile remains next-token, sparse k1 CE, mean normalization per context,
widened f32 execution, unit sample weights and true masks. Final target labels
are retained and bound even though they are absent from model input tokens.
The same existing objective byte encoder is used for capture and typed validation;
existing cache wire data, numerical loss and identity domains are unchanged.

`CalibrationCaptureView::fold` checks integrity, strict payload/version decoding,
complete root lifecycle, ordered consumers, exact sample/window counts, tokenizer
and objective context, and source agreement. A start without completion is not
a completed view. CLI `evidence verify/view` invokes this shared known view.
These checks cannot establish actual tokenizer, graph or gradient execution.

`replay_staged_capture` takes a complete standalone verified log, actual admitted
B and explicit receiving logical limits. It checks stage/source/transform/assets
and recorded caps against receiver admission, reconstructs masters from actual B,
reruns actual text encoding where applicable, then shared forward/loss/backward
and cache construction. Every encoded window and complete ordered cache binding
must agree. No supplied graph or recorded gradients are reused. An integrity-valid
fabricated cache claim can pass structural folding and still fail actual replay.
The resulting caches use the existing transport and calibrated fitting interface.

These logical limits exclude source package, parser state, input metadata/log
serialization, tape-owned master copies, activation/gradient scratch, spool and
full process RSS. Current capture/replay is not a practical 27B out-of-core path.
Separate-process execution does not itself clear independent empirical gates.

Next: bind capture/replay provenance to actual conversion admission; capture
original-A salience rather than declared statistics; complete embedding gather
and tied-consumer estimation, biased/untied/hybrid/MoE/MTP graphs; migrate actual
CLI/quantize/Qwen/Python/HF and both campaigns. All ADR0033 fitting/refinement,
GPU/memory/format, hardware/browser/training, official-model quality/runtime/bytes/
reproduction, serving/security/packaging/zoo/docs/community/signing and explicit
human public activation requirements remain required.

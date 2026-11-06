# Public PTQ failure profiling — diagnostic candidate

Parent source: `aef8e3d3ada341cc16f1473ba4cd491a73890269`. This is private
benchmark diagnostics, not a numerical change, new public request/schema,
performance-gate amendment, causal diagnosis or release clearance.

## Retained actual failure

Hosted wheel run38171255332 at9e0 failed the real public `prepare` → `calibrate`
→ `convert` row-fitting test:1.418898601087588x against the unchanged1.5x gate.
The serial/parallel fitted trit/scale digest was identical. Native wall time
improved1.520→0.605s, while outside-native wall time rose0.062→0.510s. That
locates an extra interval, not its cause. A same-wheel local unmodified probe
passed4.64x under heavy I/O pressure and a different Python/Torch environment;
it neither reproduces nor explains the hosted failure.

Failed-step log and the exact Linux wheel (artifact11713283539) remain under
`~/Projects/Tritium/archive/verification/quantize-plan-cli-wip-20261011`.
Wheel SHA256:
`e9b5f3f09f7b22ed9c348311599d80dd48b0935f6910adfb50ded2e91176abd9`.
No workflow rerun, threshold reduction or fitted-weight change was requested.

## Diagnostic candidate

The first serial/parallel pair remains unprofiled. Its installation/RECORD,
native source, algorithm, complete native row coverage, fitted digest and MSE
checks remain, followed by the original1.5x assertion. Ambient
`TRITIUM_PTQ_DIAGNOSTIC_PROFILE` is explicitly removed from that pair's child
environment so it cannot contaminate qualification.

Only a failed speedup starts separate one-/four-thread diagnostic children.
They execute the exact same installed-wheel probe, with cProfile enabled only
around the real public convert call. No alternate fitter, conversion formula,
warmup artifact or pipeline is introduced. The diagnostic children must match
the measured pair's installation, algorithm, native work, fitted digest and
MSE before their profiles are reported. Failed or mismatched diagnostics print
only an unknown marker and exception type, not exception messages, private
paths, ambient environment or child output.

Profiles are explicitly diagnostic-only, stripped of directory paths, limited
to the top25 cumulative functions and16,000 characters per child. Profiled
times never replace measured times and cannot turn the original failure green.
The child90-second deadline and1.5x threshold are unchanged. No diagnostic
output is an evidence receipt or qualification claim.

## Validation state

The new control tests were run red before implementation:24 cases, seven
failures/two errors (including changed-identity subcases), as expected for
missing profiling/control behavior. The exact red log is preserved. One
negative assertion was subsequently made privacy-safe to avoid rendering the
entire environment on failure; no numerical expectation was relaxed.

`python -m unittest scripts.tests.test_ptq_parallelism_contract -v` then passed
all24 cases in0.192s. These include actual cProfile formatting plus mocked child
control/provenance/failure cases; they are not native runtime, hardware or
performance qualification. The pre-profile production source is recoverable
from the immutable parent and retained beside the logs.

An isolated exact-wheel exercise with native source/RECORD checks attempts one
unprofiled and two profiled real conversion calls, requiring unchanged outputs.
Its first attempt (session32365) ended1 because a child exceeded the unchanged
90-second limit. The unchanged, separately logged retry (session41369) also
ended1 with the same operational timeout and empty JSON. The second log SHA256
is `5e564fb79766938828007a303a6e437bdc90e587e061fad140796fef204c5a64`.
The terminal handle, exit file and unchanged candidate source were checked.
No actual profiler/output-custody pass or cause of either timeout is inferred;
host I/O pressure is context, not proof of causation. No further blind retry
or relaxation of the child deadline is made.
Source/controls/logs are under
`~/Projects/Tritium/archive/verification/ptq-failure-profiling-wip-20261011`.
This diagnostic candidate is not yet committed or published.

Additional control coverage rejects changed algorithm identity and malformed,
non-diagnostic or overlong profiles while retaining the original speedup failure.
The expanded 25-case suite passed in 35.341s (exec session96396, exit0), including
real cProfile formatting. This is control-software validation only; the actual
installed-wheel exercise remains incomplete after both child timeouts.

## Remaining

Complete actual profiler exercise, scoped/commit-tree gates and normal
publication; capture a hosted failure profile and reproduce/minimize the cause
before any production repair. Full model/campaign/receipt migration, Qwen
hybrid/MTP and Python/HF integration, SOTA/refinement, GPU/out-of-core/full RSS,
physical backend/browser/training/serving, official model quality/runtime/bytes/
reproduction, security/packaging/ONNX/Colab/zoo/docs/community, independent
clearance and explicit human public activation all remain binding.

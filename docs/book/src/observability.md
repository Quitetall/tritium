# Ternary observability

Ordinary loss curves do not explain why a ternary run succeeds or fails.
Tritium exposes one bounded diagnostics snapshot that can feed local analysis,
TensorBoard, Weights & Biases, or OpenTelemetry without making any of those
services part of the training core.

## Collect once, send anywhere

Collect diagnostics after the backward pass if gradient metrics are needed.
Projection is real work, so do it at a deliberate cadence and select a bounded
set of weights rather than placing a whole-model scan in every optimizer step:

```python
from tritium.torch import collect_diagnostics

loss.backward()
if global_step % 250 == 0:
    snapshot = collect_diagnostics(
        model,
        step=global_step,
        paths=("classifier.weight",),  # choose a weight below the work ceiling
        max_latent_elements=1_000_000,
        extra_metrics={
            # These were measured by the caller, not inferred by Tritium.
            "teacher_kl": teacher_kl,
            "runtime/decode_ms": decode_ms,
            "memory/resident_bytes": resident_bytes,
        },
    )

    for name, value in snapshot.scalar_metrics().items():
        print(name, value)
```

The collector recognizes latent QAT modules and inference-only
`AdditiveTernaryLinear` or `AdditiveTernaryEmbedding` graphs. Tied weights are
reported once with all aliases only when every latent consumer shares the same
estimator instance and projection training mode; conflicting tied projections
fail closed. Built-in
estimators do not change parameters, gradients, estimator state, RNG, or module
training mode during collection.
External estimators are rejected by default because `no_grad` cannot prevent a
plugin from mutating buffers or Python state. Set
`allow_external_estimators=True` only after auditing that plugin's projection
behavior.

The default preflight ceiling is one million selected latent elements. The
collector fails before projection if the selection exceeds it. Raise
`max_latent_elements` only with an explicit memory/cadence budget; `None` means
unbounded and is inappropriate for routine large-model training. Hard packed
weights use compressed counting and do not reconstruct a dense float shadow,
but their scan still consumes device time.

Each tensor record includes:

- exact `-1`, `0`, and `+1` counts for every physical plane;
- zero rate, scale range/mean/standard deviation, group size, and structure;
- estimator identity for latent state or codec identity for hard state, plus
  shape, aliases, and mode;
- reconstruction RMSE for latent weights;
- gradient norm and finiteness after backward;
- exact AbsMean/SALT clipped-gradient saturation rate where that definition
  applies;
- packed code plus deployment-scale bytes and code-plus-scale bits per weight.

The last value is a tensor payload measurement, not a complete-artifact size.
Manifests, preserved tensors, tokenizer/config assets, alignment, resident
buffers, and transient memory must come from their own receipts. Pass those
measurements through `extra_metrics`; the collector deliberately does not
pretend to infer them.

## TensorBoard

TensorBoard remains entirely optional. Tritium accepts an already-created
writer and never chooses a log directory:

```python
from torch.utils.tensorboard import SummaryWriter
from tritium.torch import log_tensorboard

with SummaryWriter("runs/experiment-17") as writer:
    log_tensorboard(snapshot, writer)
```

Scalar summaries use hierarchical names. Trit counts use the writer's raw
histogram API, so logging a 27B model does not expand three aggregate counts
back into billions of Python values.

## Weights & Biases

Authentication and run lifecycle remain owned by the application:

```python
import wandb
from tritium.torch import WandbDiagnostics

with wandb.init(project="ternary-research") as run:
    telemetry = WandbDiagnostics(run)
    telemetry.log(snapshot)
```

For repeated logging, use `WandbDiagnostics(run)`. It rejects a decreasing
explicit step, matching W&B's monotonic-step contract. Sample at a bounded
cadence rather than logging many times per second. `wandb` is imported only when
the adapter is called. Tests can inject a `histogram_factory`, which is how
Tritium validates the adapter without a login, hosted account, or network
request.

## OpenTelemetry

Create one adapter per meter and reuse it for the training loop. This creates a
single stable instrument set instead of attempting to re-register gauges at
every step:

```python
from opentelemetry import metrics
from tritium.torch import OpenTelemetryDiagnostics

meter = metrics.get_meter("my-training-job")
telemetry = OpenTelemetryDiagnostics(meter)  # aggregate metrics only

for global_step, batch in enumerate(loader):
    # forward, backward, optimizer...
    if global_step % 250 == 0:
        snapshot = collect_diagnostics(
            model,
            step=global_step,
            paths=("classifier.weight",),
        )
        telemetry.log(snapshot)
```

Selected-snapshot counts, code-plus-scale bytes/rate, and caller measurements
are the default. They are whole-model values only when the snapshot selected
the whole model. Per-tensor attributes can create thousands of time series, so they
require `OpenTelemetryDiagnostics(meter, include_tensors=True,
max_tensor_series=...)` and fail over that explicit budget. Do not put steps,
prompts, user IDs, checkpoint digests, or other unbounded values in metric names
or attributes. Put high-cardinality identities in the experiment receipt or
trace/resource metadata instead. Caller metric names are capped at 64 stable
series per snapshot.

`log_opentelemetry(snapshot, meter)` is a one-shot convenience form. Reuse
`OpenTelemetryDiagnostics` for repeated logging.

## Interpretation

Use the metrics together:

| Signal | Typical interpretation |
|---|---|
| rising saturation, flat loss | clipped STE gradients cannot move enough masters |
| rapid trit-count changes | optimizer or schedule may be causing code churn |
| scale collapse or explosion | estimator scale learning is unstable |
| finite loss, non-finite gradient flag | fail the step before optimizer state is poisoned |
| lower reconstruction RMSE, worse teacher KL | local weight fit is improving the wrong proxy |
| high zero rate, unchanged runtime | the selected kernel or layout does not exploit zeros |
| low tensor payload bytes, large artifact | preserved state, metadata, or duplicated owners dominate |

Diagnostics are observations, not release evidence by themselves. A SOTA or
support claim still requires the immutable quality, physical-accounting,
runtime, hardware, and reproduction receipts defined by the release plan.

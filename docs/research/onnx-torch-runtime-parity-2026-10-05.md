# PyTorch and ONNX Runtime numerical parity probe

**Status:** diagnostic only; not a release receipt or an accepted change to a
frozen gate.

## Why this was run

The pinned SmolLM2 CPU tutorial job for source revision
`292a89755ef6601dad311b64545e501d36e91126` stopped in its ONNX-export phase in
[CI run 37290995970](https://github.com/Quitetall/tritium/actions/runs/37290995970).
The run log ended with a runner shutdown signal before the export phase could
report its result. A local reproduction was run to distinguish an exporter
failure from a runner interruption.

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- Python: 3.14; Torch 2.11.0; Transformers 5.5.3; ONNX 1.22.0;
  ONNX Runtime 1.27.0; ONNXScript 0.7.1.
- The run used the source-tree Python package and its available native
  extension, not the exact candidate wheel used by CI. CI uses Python 3.13.
- The Qwen campaign was not started or modified.

## Observations

The full local tutorial reached PTQ conversion, native checkpoint reload, and
generation. Those phases completed at about 63 s, 69 s, and 74 s respectively.
ONNX export then completed graph translation, but the export function rejected
its ORT replay under its existing `rtol=1e-4`, `atol=1e-5` comparison:

- 157 of 344,064 logits exceeded the current comparison bound.
- Maximum absolute difference was approximately `3.96e-5`.
- The assertion occurred before the tutorial emitted `onnx-export-complete`.

A separate, minimized three-token export reproduced the assertion. The
exported graph was also replayed at the original seven-token prompt length:

| Probe | Max absolute difference | Values outside current bound | Greedy-token agreement |
|---|---:|---:|---:|
| Packed Tritium model, 3-token input | `5.05e-5` | 28 / 147,456 | 100% at tested positions |
| Packed Tritium model, 7-token input | `5.63e-5` | 161 / 344,064 | 100% at tested positions |
| Untouched dense model, 7-token input | `4.96e-5` | 575 / 344,064 | 100% at tested positions |

The packed-model greedy continuation matched for all eight tested generated
tokens. These are narrow diagnostic samples, not a model-quality or
cross-platform qualification.

Changing one variable at a time did not identify a safe exporter workaround:

- Disabling ORT graph optimizations increased the three-token failures from 28
  to 279.
- Eager attention increased them to 75; ORT intra-op thread counts 1, 4, and
  the default showed the same result.
- Torch outputs were bit-identical across 1, 2, 4, and 8 CPU threads.
- The legacy TorchScript exporter failed while tracing the Transformers 5.5.3
  causal-mask path (`IndexError`), so it is not a drop-in replacement.

The dense-model probe also exceeded the current bound. This suggests that the
observed gap is not unique to Tritium's packed weights, but it does not prove
that the two runtimes are equivalent for all inputs, devices, or architectures.

## Decision and remaining work

Do not change the public tolerance or claim the ONNX gate passes from these
results. The implementation plan describes a frozen numerical tolerance, and
the current tutorial still fails it. A contract change requires an ADR and
independent review. Before proposing a replacement criterion, extend the
measurements across prompts, sequence lengths, model outputs, and supported
runtime versions; include both logit error and downstream decision stability.
Then either make the exporter satisfy the frozen contract or seek an explicit
ADR amendment with migration and rollback.

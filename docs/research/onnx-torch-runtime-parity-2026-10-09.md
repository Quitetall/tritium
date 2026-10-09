# Follow-up: hosted ONNX/PyTorch parity diagnostics

**Status:** diagnostic evidence only. No release gate is cleared and no
tolerance or acceptance contract is changed.

## Hosted failures

Three exact-source `wheels` runs failed the pinned SmolLM2 PTQ/QAT CPU tutorial
at ONNX replay:

| Run | Source | Hosted CPU | ORT intra-op | Violations | Max absolute difference |
|---|---|---|---:|---:|---:|
| [37969073677](https://github.com/Quitetall/tritium/actions/runs/37969073677) | `92754479a2fc0dc296615c396d208eb6ea183061` | AMD EPYC 9V74, 80-Core | 4 | 3 / 344,064 | `0.0001450777` |
| [37970776971](https://github.com/Quitetall/tritium/actions/runs/37970776971) | `b9615731a21a05da03179388039f7e91b1393422` | AMD EPYC 9V45, 96-Core | 2 | 2 / 344,064 | `0.0001379251` |
| [37973919718](https://github.com/Quitetall/tritium/actions/runs/37973919718) | `85fb298a1db675998cb0c11d20349fa9d2937731` | AMD EPYC 9V45, 96-Core | 2 | 2 / 344,064 | `0.0001379251` |

Both used PyTorch 2.11.0 CPU, ONNX 1.22.0, ONNX Runtime 1.27.0, four-CPU
affinity, `ORT_DISABLE_ALL`, and the frozen `rtol=1e-4`, `atol=1e-4` bound.
The second run's two failing values were at `[0,0,34041]` (absolute error
`0.0001038582`) and `[0,0,41255]` (`0.0001020394`). The ONNX graph already
promotes its seven terminal vocabulary MatMuls to FP64 before casting logits
back to FP32.

The third run repeated the same result on the same 9V45 worker profile. Its
checkpoint, input/reference tensors, and ONNX graph digests match run 379707;
the observed-output digest also matches. This makes nondeterministic
conversion/calibration an unlikely explanation for this particular failure.

The diagnostic at run 37970776971 also showed a large intermediate in the
captured layer-11 MLP: its reference down-projection ranged from about `-372`
to `12,213`. This alone does not prove the cause of the final-logit mismatch;
the final outputs are still close and only two values exceed the bound.

## Local intervention probe

The saved graph was replayed on an Intel Core i9-14900K, against the saved
AMD-host PyTorch reference tensors. ORT thread-count results were:

| ORT intra-op threads | Max absolute difference | Values outside bound |
|---:|---:|---:|
| 1 | `0.0001379251` | 2 |
| 2 | `0.0001145601` | 0 |
| 3 | `0.0000629425` | 0 |
| 4 | `0.0000935793` | 0 |

These are cross-machine diagnostics, not AMD results. In particular, local
success at four threads does not contradict the saved AMD failure at four
threads. It demonstrates that thread-count tuning on one machine is not a
portable parity fix.

As a second, in-memory-only probe on that Intel host, promoting the captured
layer-11 down-projection MatMul to FP64 reduced maximum absolute difference to
`0.0001010895`; promoting all 278 MatMul/Gemm nodes reduced it to
`0.0000873804`. Both passed the bound on that host. This has not been validated
on AMD or other backends, and the all-matmul latency sweep was interrupted
before it produced a measurement. Neither variant is an accepted fix or a
performance claim.

## Candidate implementation probe

The regression now exercises the public `prepare → calibrate → convert →
export_onnx` path. It promotes every typed FP32 MatMul/Gemm to FP64 inputs and
accumulation, then casts the result back to FP32. The focused regression tests
pass, and the complete `test_module_onnx.py` module passes locally (26 passed,
1 skipped). These checks establish implementation behavior on the local Intel
host; they do not clear the hosted AMD parity gate. The candidate still needs a
fresh hosted replay against the pinned tutorial and a runtime-overhead
measurement before the precision strategy can be accepted.

## Decision and next experiment

Do not change the frozen tolerance and do not select an ORT thread count just
because it passes on one CPU. Keep the hosted gate red. The next useful
experiment is to test a narrowly scoped, mathematically justified precision
strategy against the exact saved graph on the hosted CPU matrix, while measuring
runtime overhead and retaining the original artifact parity requirement. If no
portable implementation meets the gate, bring evidence to an ADR rather than
silently changing the acceptance bound.

Downloaded CI diagnostics and temporary Python packages used for this analysis
are scratch only; they are not release evidence or repository artifacts.

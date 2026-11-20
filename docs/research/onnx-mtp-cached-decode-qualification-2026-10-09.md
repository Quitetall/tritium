# Whole-Qwen ONNX worker: cached MTP coverage — 2026-10-09

## Finding

At source `1a3f5f5ed7864c52d9df29685983c26f0d60cbea`, the installed-worker
`_mtp_cases` path ran two independent prefill/replay pairs. It never passed a
prefill cache to `QwenOnnxCausalLM.draft`, although that API already accepts
`past_key_values`. Cached-only MTP errors therefore could not contribute to
the retained numerical error. Shifted IDs and target hidden rows were copied
from the MTP reference without binding them back to the separately executed
language target. Empty or infinite replay caches could also compare equal.

The worker cannot currently produce real whole-Qwen qualification evidence:
the packed native model's `mtp_verified` property remains false and it exposes
no production `reference_mtp` method. These findings do not invalidate a
previously admitted flagship receipt; no such receipt was produced here.

## Correction

For each existing frozen prompt, the worker now:

1. Executes native language prefill and obtains its greedy next token.
2. Executes an independent native prefill plus that token's cached continuation,
   and derives the next sample from the continuation logits.
3. Requests both aligned transactions from the promoted MTP reference.
4. Requires exactly two reference outputs, exact `[tokens[1:], sampled_next]`
   shifted IDs, matching target token IDs/hidden width, the exact native target
   hidden rows and complete final-hidden geometry.
5. Executes observed and replay MTP prefill, retaining each returned cache.
6. Executes one-token MTP decode against those two actual caches independently.
7. Includes both executions' logits and final-hidden errors in the numerical
   maximum, verifies greedy outputs, and compares their returned states.

The four retained MTP cases are labeled `mtp-prefill-0`,
`mtp-cached-decode-0`, `mtp-prefill-1`, and `mtp-cached-decode-1`. They use the
existing `mtp` case kind, existing execution/receipt schemas and unchanged
`1e-3` tolerance. This fulfills an existing worker execution requirement, not
a new public API, wire format, policy exception or authorization row.

MTP cache admission requires nonempty CPU float32 tensors with finite values.
The shared language replay comparison likewise no longer accepts empty,
non-finite or wrong-dtype states as exact. The existing `_execute` path rejects
a trace when any resulting parity boolean is false or its error exceeds the
frozen tolerance. Production oracle promotion remains unchanged.

## Regressions and verification

Before the worker correction, the updated MTP suite reported **7 failed,
1 passed**: cached calls were missing, a decode-only `0.5` logit error measured
as zero, misaligned tokens/hidden rows did not raise, and empty/infinite caches
were accepted. A separate language replay regression reported **3 failed** for
empty, infinite and float64 caches before the shared comparison was corrected.

The retained software tests additionally cover malformed hidden geometry,
missing decode reference output, NaN/empty/wrong-dtype states, replay-only
numerical drift, replay cache drift and object-identity proof that decode
receives its own observed/replay prefill cache.

```sh
TMPDIR=/mnt/4tb/tmp PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=crates/tritium-py/python timeout 90 python -B -m pytest -q \
  crates/tritium-py/tests/test_qualify_onnx_worker.py \
  crates/tritium-py/tests/test_bundle_binding.py \
  crates/tritium-py/tests/test_torch_onnx.py -p no:cacheprovider \
  --basetemp=/mnt/4tb/tmp/tritium-mtp-decode-regression

TMPDIR=/mnt/4tb/tmp timeout 60 python -B -m unittest \
  scripts.tests.test_qualify_onnx_inference \
  scripts.tests.test_verify_onnx_inference_receipt \
  scripts.tests.test_verify_workflow_source -q
```

Results: **75 passed** in the Python suite, no skips; **11 passed** in the
producer/verifier/workflow suite. Diff checks passed. These are software
regressions using controlled references, not a 27B execution or physical
model-quality receipt. Task-created pytest scratch is removed after the run.

## Remaining release obligations

The worker's `states_exact` comparison proves ONNX replay identity, **not
native-versus-ONNX cache parity**. Final whole-Qwen cache claims still require
an immutable native state-snapshot oracle and comparison covering the hybrid
language schedule, DeltaNet convolution/recurrent state, full-attention K/V
and MTP K/V. Missing native observations cannot be substituted with copied
flags or deterministic self-replay.

The independently authorized production MTP oracle, its packed-native promotion
seam, complete fitted/refined checkpoint and final installed-worker execution
also remain open. The current fixture-only authorization is not promoted to
production by this change. No Qwen fitting or paid compute was started.

Separately, hosted runs at the preceding source confirm the CI quota correction:
the supply-chain job in CI run `37992092983` and source-free tutorial job in
wheels run `37992092691` both completed successfully. They do not qualify the
new worker revision or replace any flagship release obligation.

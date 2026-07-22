# Definitive Guide to Ternary Deep Learning

Ternary deep learning is not “cast a float tensor to two bits.” It is a model
design and conversion discipline in which selected weights are represented by
values from `{-1, 0, +1}`, scales carry magnitude, and every quality or
compression claim is tied to the physical artifact that a runtime executes.

This chapter explains Tritium's contracts and the reasoning behind them. It
does not claim that the unfinished v1.1 flagship campaign has passed. Current
research comparisons and unresolved hypotheses live in the
[mid-2026 survey](../../research-ternary-sota-mid2026.md) and the frozen
[SALT V2 campaign](../../plans/0043-salt-v2-sota-campaign.md).

## 1. Start with the representation

A single ternary plane reconstructs a matrix row as:

```text
W_hat[r, c] = alpha[r] * T[r, c]
T[r, c] in {-1, 0, +1}
alpha[r] >= 0
```

Tritium can sum one to three planes:

```text
W_hat = alpha_0 * T_0 + alpha_1 * T_1 + alpha_2 * T_2
```

The scales may be row- or group-conditioned according to the artifact schema.
Additional planes are residual corrections, not a license to call the model
“1.58-bit.” They cost bytes and runtime work.

### Logical entropy is not physical rate

`log2(3) ≈ 1.585` describes the entropy of one uniformly distributed trit. A
real model also stores scales, plane maps, indexes, manifests, padding,
preserved tensors, tokenizer/configuration assets, and possibly sparse or
entropy-coding metadata. Report all of these separately:

- selected dense source bytes;
- packed target-weight bytes;
- complete serialized artifact bytes;
- resident runtime bytes;
- transient and peak bytes during conversion, load, and execution.

Compression is a ratio only after its numerator and denominator are named. A
logical trit count cannot prove a 10x whole-model reduction.

## 2. Separate latent training state from hard deployment state

QAT needs a floating master so an optimizer can accumulate small updates. The
forward path projects that master to ternary weights; the backward path uses a
surrogate gradient. This graph is *latent* and trainable.

A hard artifact contains packed trits, scales, preserved non-target state, and
identity metadata. It contains no targeted floating master or estimator state.
Tritium makes conversion explicit because silently serializing a latent master
beside packed weights destroys both the memory claim and the deployment trust
boundary.

The same distinction applies to PTQ refinement. Scale-only and hard-PV
(PV-Tuning-style alternating continuous-scale and discrete-code optimization)
results are children of a PTQ artifact with distinct ancestry and cost. They
are not relabeled as ordinary PTQ. The [research survey](../../research-ternary-sota-mid2026.md)
places this refinement family in the broader literature.

## 3. Choose PTQ, QAT, or refinement intentionally

### Post-training quantization

Use PTQ when a dense checkpoint exists and retraining cost or data access is
limited. The minimum trustworthy workflow is:

1. **Prepare:** inspect the whole module graph, target exact weights, preserve
   ties, and freeze a versioned recipe without mutating the source unexpectedly.
2. **Calibrate:** run representative inputs, hash their content identity so an
   externally reproducible replay can be verified, and collect
   activation/curvature evidence under bounded memory.
3. **Convert:** fit ternary planes and scales, allocate residual planes under a
   physical budget, and emit a source-bound result.
4. **Validate:** strict-load the hard model and measure reconstruction, output
   divergence, held-out loss, tasks, generation, bytes, memory, and runtime.

Weight MSE is useful for debugging but is not a model-quality gate. A locally
better layer fit can make end-task loss worse. Activation-aware reconstruction,
block error feedback, and held-out evaluation exist to close that proxy gap.

### Quantization-aware training

Use QAT when training data and optimization budget are available or PTQ cannot
recover enough quality. A normal PyTorch optimizer updates floating masters;
the ternary projection is used in the forward pass. Convert only after training
and validation have finished.

QAT is not automatically better. It can overfit training or fine-tuning data,
trap saturated weights, destabilize scale learning, or hide dense state in
checkpoints. Treating a small calibration-only sample as QAT training data is a
separate methodological error. Always compare QAT against a strong PTQ baseline
at matched artifact bytes.

### Refinement

Refinement starts from an immutable PTQ parent:

- **Scale-only** freezes all trits and optimizes scales. It is cheaper, preserves
  the discrete code, and is the first refinement to try.
- **Hard-PV** alternates bounded scale fitting with discrete ternary polishing.
  It can move trits and therefore needs its own receipt, ancestry, compute cost,
  and acceptance gate.

Training and validation sets must not overlap. Any calibration reuse must be
declared and accounted for in the evaluation design. A child that fails its
frozen validation gate is retained as a failed experiment, not promoted by
changing the threshold.

## 4. Understand the estimator

The exact derivative of the discrete map that selects a trit code is zero
almost everywhere, so QAT uses a straight-through estimator (STE). A scaled
reconstruction can still have a gradient through a learned scale; Tritium's
AbsMean path deliberately detaches its derived scale, while learned-scale
estimators update their scale parameters separately. Tritium treats the
estimator as versioned algorithm state rather than an anonymous backward hook.

The core catalog includes:

- **AbsMean STE:** a stable baseline using magnitude-derived scales and a
  clipped surrogate gradient;
- **annealed STE:** keeps the forward projection hard while progressively
  sharpening its smooth `tanh` backward surrogate;
- **LSQ-style estimation:** learns scale parameters with a normalized surrogate;
- **TWN:** thresholded ternary weights with a shared magnitude scale;
- **TTQ:** separately learns positive and negative scales and therefore deploys
  as two physical planes in Tritium;
- **sparse ternary:** explicitly targets a zero rate instead of treating zeros
  as accidental;
- **SALT STE:** aligns training projection with Tritium's additive deployment
  representation.

Estimator choice changes optimization behavior, stored state, and sometimes the
deployable representation. Record its algorithm ID and schema version. An
external estimator must pass shape, device, dtype, finiteness, trit-domain,
scale, alias, serialization, and backward checks before it can participate in a
release claim.

### Scales and thresholds are model state

The estimator determines both the code-selection boundary and the magnitude of
the decoded weight. In Tritium's current PyTorch estimators, with `c` indexing a
row's quantization group:

- **AbsMean** derives `s_r = mean_c(abs(W[r,c]))`, then selects
  `T = clamp(round(W / s), -1, 1)`. Hard rounding places the code boundaries at
  `-0.5s` and `+0.5s`; its backward mask is active only where
  `abs(W / s) < 1`. The derived scale is detached and stored in `float16`, so
  quality evaluation must use the rounded deployment scale rather than only
  its higher-precision precursor.
- **TWN** sets `delta_r = threshold_factor * mean_c(abs(W[r,c]))` (the default
  factor is `0.7`), selects nonzeros only where `abs(W) > delta`, and uses the
  mean selected magnitude as the row scale. Raising the threshold creates more
  zeros but can increase reconstruction error.
- **LSQ-style estimation** keeps its learned scale positive as
  `softplus(log_scale) + epsilon`, normalizes and clamps weights to `[-1, 1]`,
  and rounds the normalized value to a trit.
- **TTQ** learns separate positive and negative magnitudes through `softplus`.
  Its threshold is the row absolute mean multiplied by a learned sigmoid ratio.
  Because Tritium represents those unequal magnitudes as two physical planes,
  a matched-rate TTQ comparison must price both planes; it is not byte-equivalent
  to one-plane TWN.

The Python estimator API currently uses full-row groups. SALT V2 recipes can
instead use fixed groups such as G128. Group geometry, positive-scale
invariants, scale encoding, and deployment rounding are therefore part of the
artifact identity, not tuning notes.

### Common STE failures

- **Saturation:** a clipped surrogate gives zero gradient outside its active
  range. Inspect the fraction of masters outside the gradient window.
- **Scale collapse or explosion:** monitor scale distributions and finite
  gradients, not only loss.
- **Code churn:** large learning rates repeatedly flip trits and prevent the
  scales from settling.
- **Premature sharpening:** an annealed estimator's forward pass is already
  hard; increasing the backward-surrogate temperature too quickly can collapse
  useful gradients before the floating model reaches a good basin.
- **Dense checkpoint shadow:** a deployment export accidentally retains the
  master or optimizer tensors.

## 5. Add planes where they buy quality

A greedy residual scheme fits the first plane, subtracts its reconstruction,
then fits another. It is simple and deterministic, but not necessarily optimal:
early choices can force expensive later corrections.

SOTA-oriented additive PTQ therefore considers:

- alternating ternary-code and conditioned-scale solves;
- activation- or curvature-weighted reconstruction;
- blockwise residual propagation rather than isolated row MSE;
- output reconstruction on representative hidden states;
- byte allocation across tensors instead of a uniform plane count;
- scale-only refinement before discrete polishing;
- matched-byte baselines and ablations for each mechanism.

The allocator must price every plane's codes, scales, alignment, and indexes.
It must also emit the chosen per-tensor map so the runtime executes the same
model that evaluation measured.

## 6. Treat zeros as a semantic and physical resource

Zeros can skip additions, improve entropy coding, and sometimes align with
structured sparsity. None of those benefits is automatic.

- Report the zero rate per plane and tensor family.
- Distinguish unstructured zeros from a hardware-supported pattern such as 2:4.
- Include sparse indexes or masks in physical bytes.
- Benchmark the actual sparse kernel; dense execution of a sparse tensor is not
  a sparse speed result.
- Measure the quality cost of forcing structure after ternarization.

Sparse-ternary research is promising, but a high zero rate alone does not prove
that a GPU sparse tensor core can consume the layout or run faster.

### Structured S34 is a separate representation

Tritium's S34 codec requires exactly one zero in each physical quartet. Its 32
legal quartet states fit in five bits. This is a trained constraint, not a
post-hoc pruning label, and Tritium has a direct runtime codec for it. S34 must
still earn separate Pareto admission through measured quality, physical bytes,
and native runtime results; codec existence alone is structural evidence.

## 7. Preserve model semantics

Whole-model conversion is more than replacing `nn.Linear`:

- tied embeddings and language heads must share one packed owner;
- biases, norms, rotary state, QK normalization, routing, convolutional modules,
  and non-target tensors need explicit dispositions;
- multi-token prediction (MTP) and speculative draft layers need independent
  coverage and quality evidence;
- tokenizer, generation configuration, source revision, and architecture must
  remain identity-bound;
- deferred vision or multimodal tensors must be listed, not silently dropped;
- a root module replacement must be rebound by the caller.

A coverage receipt accounts for every in-scope tensor or state item, including
aliases, roles, and dispositions, as converted, preserved, or rejected with a
reason. Partial coverage is not a whole-model claim.

## 8. Evaluate a model, not a tensor collection

A defensible campaign couples quality, size, and performance:

1. **Integrity:** exact source/model/tokenizer/data/recipe identities and strict
   artifact reload.
2. **Coverage:** every parameter and tied owner accounted for.
3. **Numerics:** reconstruction, logits, loss/perplexity, NaN/Inf checks, and
   native-vs-reference parity.
4. **Tasks:** frozen language, reasoning, code, instruction-following, and draft
   acceptance suites appropriate to the model.
5. **Physical accounting:** serialized, resident, transient, and peak bytes.
6. **Runtime:** prefill, decode, batch/context curves, energy where measured,
   warmup, repetitions, uncertainty, and exact device/software identity.
7. **Baselines:** dense source, simple ternary round-to-nearest (RTN), current
   mainstream low-bit, and additive ablations at matched physical bytes.
8. **Reproduction:** independent machine/run identity and immutable inputs.

“Best ternary result,” “best global low-bit Pareto point,” and “near-lossless”
are separate claims. Tritium's v1.1 campaign freezes their thresholds before
the flagship result is observed.

## 9. Debug systematically

When quality collapses, do not immediately add planes or train longer.

| Symptom | First checks |
|---|---|
| Catastrophic loss after conversion | coverage, source digest, tokenizer, tied owners, preserved state, scale geometry |
| Good weight MSE, bad perplexity | activation distribution, output reconstruction, sensitive layers, calibration representativeness |
| QAT stops improving | saturation, trit churn, scale gradients, annealing schedule, optimizer state |
| Artifact is larger than expected | preserved tensors, duplicate owners, scale/index overhead, alignment, dense shadows |
| Runtime is slower | unpack/decode cost, host transfer, dense dequantization, batch/context regime, wrong backend |
| Reload differs | schema/recipe mismatch, noncanonical files, missing aliases, source-shell drift |
| Sparse result has no speedup | kernel lacks skip support, index overhead, pattern mismatch, bandwidth not compute bound |

Change one mechanism at a time and keep failed receipts. Otherwise an apparent
improvement cannot be attributed or reproduced.

## 10. Use the narrowest workflow that answers the question

- **Can this checkpoint tolerate ternary weights?** Run bounded PTQ and model
  quality evaluation.
- **Where should extra bytes go?** Sweep additive allocations at matched complete
  artifact sizes.
- **Can cheap recovery close the gap?** Run scale-only refinement.
- **Does the discrete code need to change?** Admit hard-PV with a separate cost
  and validation set.
- **Can training recover further?** Compare QAT from the same source and report
  optimizer/checkpoint cost.
- **Is it deployable?** Strict-export, reload, and execute the exact packed
  artifact through the target runtime without a dense shadow.
- **Is it SOTA?** Reproduce current baselines under one harness and satisfy the
  predeclared quality, byte, runtime, and independence gates.

The practical rule is simple: optimize the model users will actually store and
run, and make every transformation visible in its type, identity, ancestry, and
receipt.

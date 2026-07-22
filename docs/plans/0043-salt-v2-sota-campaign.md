# 0043 — SALT V2 implementation and SOTA campaign

Status: **IN PROGRESS** (Qwen family/MTP fixture, resumable tensor-master driver,
exact physical nested allocation, selected-package structural admission, and
bounded canonical package writing implemented through 2026-07-20;
checkpoint-scale and empirical campaign gates remain open)

- **Decision:** [ADR 0028](../adr/0028-salt-v2-additive-ternarization.md)
- **Research cutoff:** 2026-07-14, inclusive
- **Claim status:** not achieved
- **Active flagship:** `Qwen/Qwen3.6-27B` revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- **Active conversion scope:** language core and bundled one-layer MTP drafter
- **Deferred scope:** vision encoder and multimodal integration
- **Experiment tracks:** PTQ and refined, reported separately
- **Spend policy:** local-first; no paid run without explicit future approval
- **Development rung:** SmolLM2-1.7B, reusing Tritium's existing model and
  campaign support

## Active campaign override — 2026-07-15

This section is the active execution order. It supersedes the original
Qwen3-8B then Qwen3-32B target ordering, which remains in the ADR and git
history; it does not mark any empirical gate complete. The original numeric
quality, rate, parity, provenance, and publication gates transfer to the 27B
flagship. A later confirmation model requires a new preregistration.

The flagship artifact is `Qwen/Qwen3.6-27B` at immutable revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. The first product scope puts every
non-vision rank-2 tensor into the additive ternary allocation domain, including
the language embedding, untied output head, and all MTP matrices. The pinned
checkpoint contains 506 such matrices with 27,318,026,240 coefficients. Its 360
non-vision non-matrix tensors remain at source precision. All 333 vision tensors
are identity-bound with disposition `ExcludedFutureVision`; they are neither
converted nor omitted from coverage. A profile that preserves any rank-2
language or MTP tensor must be separately preregistered and cannot support the
unqualified language-plus-MTP ternarization claim.

Vision remains the end-state product scope, but the vision encoder and
multimodal connector are deferred until the language and MTP paths satisfy
their architecture, identity, quality, accounting, and runtime gates.

The pinned official
[model card](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/README.md),
[configuration](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/config.json),
and [weight index](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/model.safetensors.index.json)
establish the target shape. The checkpoint uses an outer `qwen3_5` multimodal
wrapper with a nested
`text_config`. Its dense 27B text model has 64 layers, hidden size 5120, and a
padded vocabulary of 248,320. The layer schedule repeats three
linear-attention/Gated DeltaNet blocks followed by one full-attention block.
The checkpoint declares `mtp_num_hidden_layers = 1`, and the official serving
instructions include MTP deployment configurations.

Tritium now has a content-bound Qwen3.5-family reference adapter that parses the
outer/nested configuration, executes configured Gated DeltaNet and
full-attention language layers, and exact-loads the one-layer MTP module. A
small pinned-vLLM prefill/decode fixture passes the MTP gate. This is component
evidence only: the pinned 27B checkpoint has not been loaded, all 64 layers have
not been compared with the reference, and no checkpoint-scale coverage,
host/CUDA parity, allocation, or serving receipt exists. Those proofs remain
ahead of the production campaign driver on the critical path.

The active order is:

1. Add a source-identity-bound Qwen3.6 architecture adapter: nested-config
   parsing, tensor-name/shape validation, all 64 scheduled language layers,
   Gated DeltaNet state semantics, full-attention blocks, and the one-layer MTP
   module. Prove dense reference parity before accepting ternary weights.
2. Freeze and test the language-plus-MTP coverage policy. Account for all 506
   in-scope rank-2 matrices, 360 preserved non-matrix tensors, 333 deferred
   vision tensors, and every runtime allocation; fail closed on unconsumed,
   duplicate, or shape/dtype-mismatched tensors.
3. Connect the measured whole-model SALT V2 receipt to
   `PhysicalSizeReport` and the immutable campaign ledger, then land the
   resumable production driver. No geometry-only or caller-supplied byte total
   is claim evidence.
4. Run local synthetic, sliced-layer, and smallest practical end-to-end gates;
   optimize the exact/fused kernels and sharded/offloaded conversion path before
   attempting the complete 27B checkpoint.
5. Run the frozen 27B **PTQ track** and publish its artifact, quality, physical
   size, resident memory, and runtime results under a PTQ label.
6. Only after the PTQ stop gates, run the separately identified **refined
   track** (scale-only, then hard-trit/PV refinement as admitted). Never merge
   refined measurements into the PTQ result.
7. Reproduce matched baselines, complete the language-plus-MTP evidence pack,
   and only then schedule the deferred multimodal adapter and vision policy.

Steps 1–4 are prerequisites for Stage 8. Steps 5 and 6 are Stage 8 and Stage 9,
respectively; step 7 feeds Stage 11 and may only propose, not authorize, the
deferred Stage 10 scope.

Local hardware is the default for engineering, correctness, profiling, and any
campaign rung that fits. This plan grants no cloud or rental authorization. A
paid run requires a new explicit approval after its frozen recipe, estimated
cost, stop gate, and resume proof are presented.

Individual results may retain their narrow labels. The public umbrella claim
"Tritium SOTA" remains forbidden until every applicable gate in both
[ADR 0026](../adr/0026-sota-campaign.md) and
[ADR 0028](../adr/0028-salt-v2-additive-ternarization.md) is green and the
claim is generated from the evidence ledger.

## Outcome

Build and evaluate a zero-point-free SALT V2 converter and native inference
path whose stored weights are only additive ternary planes plus positive scales.
The campaign must answer, with reproducible measurements:

1. Does joint, output-aware additive ternarization beat SALT V1 and other
   ternary/additive methods at the same physical bytes?
2. Can Qwen3.6-27B reach the preregistered near-zero-divergence gate at an
   approximately 3.3–3.6 matrix-bpw point?
3. Does the direct ternary kernel produce a global quality/size/runtime Pareto
   point, rather than a quality-only result with a slow decoder?
4. How much of the PTQ gap does the separately reported refined track close at
   27B, and at what measured compute and memory cost?

“No” is a valid result. A failed gate is recorded with its artifact and stops
the dependent spend. It must not be rewritten as a qualitative SOTA claim.

## Non-goals

- No group bias, zero point, floating residual, arbitrary codebook, lattice,
  trellis, or learned decoder.
- No claim based on logical `log2(3)` bpw alone.
- No dense-dequantized inference number presented as a SALT runtime result.
- No from-scratch 32B–50B pretraining.
- No full-model multi-billion-token QAT before PTQ and short-refinement gates.
- No model-growth result conflated with same-model conversion fidelity.
- No edits to the older campaign's acceptance history to make the new method
  appear shipped.

## Frozen campaign structure

### Binding implementation profile

- The deployable representation is zero-point-free `sum s_p*T_p`, `P=1..3`.
- `CompactV1` is an exact prefix capped at 2.25 physical core-projection bpw.
  `NearLosslessV1` contains that prefix, is capped at 4.0 physical
  core-projection bpw, and requires strict BF16 non-inferiority after reloading
  the exact package.
- D2 is the mandatory aligned reference. B3 and S34 are separately compiled
  codecs over the same semantic tensor; a package chooses one codec. A challenger
  is retained only when it is non-dominated in exact serialized bytes, exact
  resident bytes, quality, and measured wall time.
- Scale groups default to 128. G64/G256 require promotion evidence. Optional
  planes use regular 256-coefficient presence macrotiles or a denser encoding,
  so allocation metadata remains at or below 0.01 bpw and no tensor is padded to
  its maximum plane count.
- The first frozen evidence pack is 512x2048 tokens: 50% C4, 25% OpenWebMath,
  and 25% StarCoderData. Any larger pack is a new provenance rung.
- Recovery is evaluated A16 first, then A8, then A4. Soft optimization occupies
  at most the first 80%; the final 20% uses exported hard trits and narrowed
  scales. CE precedes conditional cached-logit KD; PV discrete polish is last.
- The direct-model track freezes the recipe on SmolLM2-1.7B, then runs the
  pinned Qwen3.6-27B language-plus-MTP checkpoint. No confirmation model,
  model-growth track, or vision run is authorized by this plan.
- Stable orchestration lives in `tritium-salt` through `SaltV2::explain` and
  `SaltV2::reconcile`. `SaltPipeline::{start, advance, resume}` is experimental
  but must execute the identical stage sequence and durable evidence contract.

### Model ladder

| Rung | Model | Purpose | May choose hyperparameters? |
|---|---|---|---|
| 0 | deterministic synthetic matrices and the existing tiny transformer | solver, accounting, and parity oracles | implementation constants only |
| 1 | SmolLM2-135M | full-pipeline smoke and cheap negative-result discovery | no |
| 2 | SmolLM2-1.7B | choose group size, packing, restart count, and curvature variant | yes, only from the preregistered grid |
| 3 | Qwen3.6-27B language plus MTP | primary quality, size, memory, and runtime result | no; recipe frozen from rung 2 |
| 3b | Llama2-7B | literature bridge for QTIP, LLVQ, AQLM, VPTQ, and PV-Tuning | no; same frozen recipe |
| 4 | Qwen3.6 multimodal | deferred end state | not authorized; requires a new preregistration |

The 1.7B evaluation split is held out from pilot selection. Qwen3.6-27B task
results remain sealed until the recipe digest is frozen. If a Qwen-specific
correctness bug requires a recipe change, invalidate the affected run, fix it,
and rerun all methods; do not patch only SALT V2.

### Physical-rate points

All rate constraints are converted to integer byte ceilings before fitting.
The primary matrix-rate points are:

| ID | Maximum matrix bpw | Purpose |
|---|---:|---|
| `R2` | 2.25 | `CompactV1`; usually one base plane plus selectively allocated residual planes |
| `R3` | 3.50 | Primary `NearLosslessV1` operating point, below its hard 4.0-bpw ceiling |
| `R4` | 4.25 | D2 dual-plane baseline/control only; cannot publish as either stable profile |

For `N_q` quantized weights, the matrix byte ceiling is
`floor(rate * N_q / 8)`. The allocator may not exceed it by one alignment block.
Every run also reports actual whole-artifact bpw and resident bpw. Matched
baselines are constrained first by whole-artifact bytes; their matrix rate may
be reduced when they leave more tensors unquantized. A matrix-only comparison
is labeled as such and cannot establish the global claim.

Rung 2 starts at group size 128 and evaluates G64/G256 only under the promotion
rule. D2 is always retained; B3 and S34 compete for separate Pareto admission.
The plane cap is selected from `{2, 3}` and the output-aware curvature method
from `{guided-fisher, forward-kl-kronecker}`. Diagonal Fisher, input Hessian, no
rotation, and greedy SALT V1 remain ablations.

### Calibration and evaluation provenance

Create immutable `CalibrationProvenance`, `EvaluationProvenance`, and
`RecipeProvenance` records before any scored run. The initial frozen data plan
is:

- rung 1 smoke: a fixed 128-sequence prefix of the evidence pack;
- rungs 2–4: 512 sequences × 2,048 tokens, composed of 50% C4, 25%
  OpenWebMath, and 25% StarCoderData at immutable revisions;
- a larger evidence rung may be proposed only after the frozen pack's learning
  curve or confidence interval shows that calibration variance, rather than the
  representation, is the binding limitation;
- deterministic sampling seed, ordered token digest, tokenizer digest, and
  dataset revision in every cache;
- calibration, refinement, early-stop validation, and final test samples are
  disjoint by digest.

If the selected dataset cannot be redistributed, freeze its public revision and
the exact sample-index/token digests and publish a deterministic builder.

The final suite contains:

- WikiText-2 and C4 perplexity for literature continuity;
- a held-out modern-corpus perplexity set from the frozen snapshot;
- MMLU, ARC-Challenge, HellaSwag, BoolQ, GSM8K, and MATH with fixed lm-eval
  revision, shot counts, templates, stop strings, and tokenizer;
- teacher forward-KL, Jensen-Shannon divergence, top-token agreement, and logit
  cosine error on a frozen prompt set;
- prefill, time-to-first-token, batch-1 decode, batched decode, peak resident
  bytes, artifact load time, and conversion peak memory.

The primary task aggregate is the unweighted mean of the six percentage-point
scores produced by the frozen harness (accuracy or exact match as appropriate).
No post-hoc rescaling is allowed. Per-task values remain gates, so the aggregate
cannot hide one large regression.

### Baseline matrix

Every baseline is bound to a source revision, recipe, command, artifact digest,
and actual byte count. The required matrix is:

| Class | Required baselines |
|---|---|
| Upper bound | source bf16/fp16 model, evaluated by the same harness |
| Tritium | flat ternary; SALT V1 greedy at the same physical byte ceilings; SALT V2 without each major mechanism |
| Ternary/additive | PTQTP, relabeled by actual physical rate; BPDQ; PT²-LLM when reproducible on the selected architecture |
| Vector/lattice | QTIP, LLVQ, and one of AQLM or VPTQ; UniSVQ for Qwen3 |
| Mainstream | a current reproducible GPTQ/AWQ-class W4 baseline |

Published paper numbers are context only. A baseline is “reproduced” only when
its actual artifact, coverage policy, evaluation outputs, and runtime are
captured locally. Unsupported architecture is reported as such and may narrow a
claim, but is not silently scored as a loss.

## Proposed CLI and artifact contract

### Recipe

The canonical recipe is portable JSON. Fields shown below are required; unknown
fields fail closed.

```json
{
  "schema": "tritium.salt-v2.recipe.v1",
  "source_model_id": "<content id>",
  "calibration_id": "<trc1 id>",
  "evaluation_id": "<tre1 id>",
  "profile": "NearLosslessV1",
  "group_size": 128,
  "allocation_macrotile": 256,
  "min_planes": 1,
  "max_planes": 3,
  "codec": "B3",
  "matrix_byte_ceiling": 2250000000,
  "artifact_byte_ceiling": 2500000000,
  "curvature": "guided-fisher",
  "rotation": { "kind": "signed-rht", "seed": 0 },
  "solver": {
    "em_restarts": 4,
    "coordinate_sweeps": 10,
    "ridge_condition_limit": 1000000.0,
    "feedback": "block-ldlq-delta-corrected"
  },
  "refinement": { "kind": "none" },
  "seed": 0
}
```

There is intentionally no `zero_point`, `bias`, `outlier_dtype`, `codebook`, or
`decoder` field.

### Commands

The library boundary lands before a production CLI driver:

```rust
let preview = SaltV2::explain(&spec)?;
let receipt = SaltV2::reconcile(&spec, work_root, &mut driver)?;

let mut staged = SaltPipeline::start(&spec, work_root)?;
staged.advance(&mut driver)?;
let resumed = SaltPipeline::resume(&spec, work_root)?;
```

Once the real filesystem/model driver is linked, the CLI projection is:

```text
tritium salt explain --source <model> --evidence <pack> --profile <profile>

tritium salt synthesize \
  --source <hf-or-local-model> \
  --evidence <evidence-pack> \
  --recipe <recipe.json> \
  --work-dir <content-addressed-work-root> \
  --output <model.tsalt2>

tritium report salt-v2 \
  --artifact <model.tsalt2> \
  --evaluation <evaluation.json> \
  --output <report.json>

tritium report salt-v2-compare \
  --ledger <campaign-ledger.tcmp> \
  --output <comparison.json>
```

`salt synthesize` executes the existing `ConversionStage` order:
`Ingest → Calibrate → Profile → Search → Refine → Pack → Validate → Publish`.
Each completed stage writes a content-addressed output and a `StageReceipt`; a
resume verifies every upstream digest before doing work.

### Required report fields

Every tensor, model, and campaign report includes:

```text
source_model_id, recipe_id, calibration_id, evaluation_id
packing, group_size, plane_histogram, quantized_parameter_count
trit_payload_bytes, scale_bytes, allocation_map_bytes, transform_bytes
padding_bytes, tensor_header_bytes, container_bytes, unquantized_tensor_bytes
matrix_bytes, artifact_file_bytes, steady_resident_bytes, peak_resident_bytes
logical_bpw, matrix_bpw, artifact_bpw, resident_bpw
frob_error, hessian_error, block_output_error, teacher_kl
perplexities, task_metrics, divergence_metrics
prefill, ttft, decode, batch_decode, load_time, conversion_wall_time
gpu_hours, cpu_hours, peak_host_bytes, peak_device_bytes
hardware, driver, clocks, contention, source_revision, commands
```

The report builder recomputes byte totals from the opened immutable artifact.
It rejects a supplied total that differs from the file or resident allocation.

## Implementation stages

Each stage is independently mergeable and must pass fmt, clippy, unit tests,
and the relevant CPU/CUDA parity gate before the next stage depends on it.

### Stage 0 — Freeze evidence schema and physical accounting

**Work**

- Add physical-rate value types to `tritium-quantize`: checked byte ceilings,
  `PhysicalSizeReport`, and exact component counters.
- Extend `CampaignMetrics`/`MeasuredPackage` rather than creating a parallel
  ledger.
- Add report validation against actual file size and a checked resident-
  allocation receipt. Stage 6 supplies the SALT V2 CUDA allocation receipt.
- Add a fixture demonstrating the PTQTP correction:
  `P=2, G=128, direct-2bit => 4.25 bpw` before headers.

**Gate**

- Property tests cover uniform and mixed plane counts, all short final groups,
  alignment, overflow, and zero-length rejection.
- Encoded length, reported matrix bytes, opened file length, predicted resident
  geometry, and decoded tensor geometry agree exactly.
- No CLI or report labels dual-plane direct storage as 1.58 bpw.

### Stage 1 — SALT V2 format and zero-point-free API

**Work**

- Add `SaltV2Tensor`/row descriptors in `tritium-format` with non-negative fp16
  scales and hard trits only.
- Implement D2 first, B3 second, and structurally valid S34 third.
- Keep D2/B3/S34 as the direct-runtime formats. Benchmark any ANS/rANS candidate
  only as seekable outer transport, and count expanded fixed-codec bytes as
  resident weights unless a separate native entropy-stream kernel is admitted.
- Encode mixed plane counts without row-wide dense padding; charge every map,
  descriptor, and alignment byte.
- Preserve the existing SALT/TQ2 reader. SALT V2 receives a new explicit
  version/type and is never detected by ambiguous byte sniffing.

**Gate**

- Canonical round-trip is byte-identical for every admitted codec.
- Malformed radix digits, non-canonical padding, negative or non-finite scales,
  zero scales paired with nonzero trits, oversized dimensions, duplicate maps,
  and unsupported versions fail closed.
- Format structures have no zero-point or floating-residual field.
- Fuzz decode never panics or reads outside the declared payload.

### Stage 2 — Calibration, activation cache, and curvature artifacts

**Work**

- Build a sharded, content-addressed `ActivationCache` with layer/tensor shape,
  token mask, sequence boundaries, dtype, and source digest.
- Add block input Hessian accumulation and damped factorization.
- Add GuidedQuant-style output/Fisher correction and YAQA-style forward-KL
  Kronecker approximation behind separate enum variants.
- Record condition number, damping, discarded spectrum, cache bytes, and time.

**Gate**

- Synthetic linear models recover analytic Hessian/Fisher values.
- Shard order and resume do not change the artifact digest or solve result.
- Curvature is positive semidefinite within tolerance after damping; failed
  factorization is explicit, never an implicit diagonal fallback.
- On rung 1, evaluate 100 fixed-seed perturbation pairs. Output-aware curvature
  must order at least 60 pairs consistently with held-out loss and must exceed
  diagonal Fisher's correct-order count; report both counts and rank
  correlations.

### Stage 3 — Joint ternary EM and conditioned scale solve

**Work**

- Implement exact joint assignment over `3^P` states for `P=1..3`.
- Implement Hessian-weighted non-negative scale solve with adaptive ridge and
  scale-sign canonicalization.
- Add OA-EM-style deterministic multi-start initialization.
- Keep SALT V1 `residual_expand` unchanged as the ablation oracle.

**Gate**

- Exhaustive tiny-matrix tests match a brute-force global optimum where the
  complete state space is tractable.
- Every accepted E or M update is non-increasing under the active objective.
- Same inputs, recipe, hardware policy, and seed yield identical trits, scales,
  metrics, and receipt.
- Joint solve beats or ties greedy AbsMean Hessian error on every golden and
  strictly beats it on at least one non-degenerate golden.

### Stage 4 — Block feedback, delta correction, and byte allocator

**Work**

- Add deterministic group-aware ordering.
- Add BlockLDLQ/GPTQ residual propagation.
- Recompute and propagate the delta caused by every scale refit.
- Replace logical-bpw water filling with exact marginal loss per encoded byte.
- Recompute every exact prefix loss after a jointly fitted master solution
  changes; invalidate both profile allocations rather than mixing curves from
  different master fits. Cache only digest-bound curves.

**Gate**

- Feedback matches an independent small-matrix reference.
- Removing delta correction produces a measurable regression on a constructed
  scale-refit case; the corrected path matches the reference.
- Allocations never exceed the integer matrix or artifact byte ceiling.
- Higher byte ceilings never worsen the selected checkpoint's full-model
  validation metric; local proxy monotonicity alone is insufficient.

### Stage 5 — Output reconstruction and refinement

**Work**

- Add block-output reconstruction on cached activations.
- Add scale-only teacher-KL refinement with fixed trits/allocation.
- Treat master-layer fixed-trit/prefix admission and selected package-allocation
  admission as separate proofs; neither may stand in for the other.
- Add smooth warmup followed by PV hard-trit/scale alternating updates.
- Project and validate hard trits after every discrete phase and before every
  checkpoint.

**Frozen token caps**

| Rung | Scale-only maximum | Short PV maximum |
|---|---:|---:|
| 1–2 | 8M tokens | 32M tokens |
| Qwen3.6-27B | 64M tokens | 512M tokens |

The 27B row inherits the former 32B hard maximums; neither value is a 27B
measurement or a prediction of what the hybrid graph needs. The 1.7B pilot may
tighten either cap before the recipe freezes. The caps may not increase in
response to a 27B outcome; any later expansion requires a new preregistration.

Evaluate at 1/8, 1/4, 1/2, and full cap. Stop after three evaluations without
improvement in the frozen validation aggregate. The best earlier checkpoint is
retained; consuming the cap is not success.

**Gate**

- A locally lower reconstruction loss may not replace a checkpoint with worse
  held-out perplexity or teacher KL.
- Final serialized trits exactly match the trits evaluated by the last hard
  checkpoint.
- PTQ, scale-only, and short-PV receipts and reports remain separately labeled.

### Stage 6 — Native CPU/CUDA execution

**Work**

- Add the exact CPU semantic reference for D2, B3, and S34.
- Add `upload_salt_v2` and exact/fast `salt_v2_forward` entry points without
  dense materialization.
- Implement D2 add/sub/skip first. Implement B3 and S34 unpack as separate
  candidates so each rate/runtime tradeoff is visible.
- Add mixed-plane tile scheduling and fused scale accumulation.
- Implement signed RHT only after a no-rotation end-to-end baseline exists.

**Gate**

- CPU, exact CUDA, and dense mathematical reconstruction agree within the fp16
  scale/reduction oracle; exact and fast satisfy ADR 0027 policy.
- Compute Sanitizer is clean for all shapes, short groups, plane counts, and
  packings.
- No measured kernel allocates or writes a dense dequantized weight tensor.
- Reported steady and peak device allocations equal the checked resident-
  allocation receipt and its physical-size report.
- Report unpack, transform, ternary accumulation, scale epilogue, and total
  latency separately, then use total latency for claims.
- A format is dropped from the claimed frontier if its end-to-end runtime is
  dominated at equal or worse quality and bytes.

### Stage 7 — Rung 1/2 pilot and recipe freeze

Run the complete grid only on SmolLM2-1.7B, after a 135M smoke:

```text
group size:       64, 128, 256
codec:            D2, B3, S34
plane cap:        2, 3
rotation:         none, signed-RHT
curvature:        input-Hessian, guided-Fisher, forward-KL-Kronecker
solver ablation:  greedy, joint, joint+feedback, joint+feedback+output recon
rate:             R2, R3, R4
```

Use staged successive halving rather than the full Cartesian product:

1. one-layer proxy rejects incorrect or clearly dominated variants;
2. four representative layers retain the best half by output loss and bytes;
3. full 1.7B PTQ retains the non-dominated quality/bytes/runtime set;
4. only the best PTQ point per rate receives scale-only refinement;
5. only one recipe receives short PV.

**Freeze gate**

- At R3, the selected joint method closes at least 25% of SALT V1's held-out
  perplexity gap to bf16 and is no worse than SALT V1 on any frozen task.
- At some rate, output-aware curvature improves full-model held-out perplexity
  over input-Hessian-only with the same bytes.
- Physical accounting and native-kernel gates are green.
- Freeze recipe, evaluation thresholds, and digests before unsealing the pinned
  Qwen3.6-27B checkpoint.

If the freeze gate fails, publish the 1.7B negative result and stop. Do not run
the 27B campaign.

### Stage 8 — Dense parity, baselines, and Qwen3.6-27B PTQ proof

**Work**

- Prove dense host and CUDA parity against the pinned reference implementation
  for all 64 hybrid text layers before accepting any ternary projection.
- Prove the MTP token shift, concatenation order, positional alignment, and
  cache behavior against an official serving oracle; checkpoint names and
  shapes alone are insufficient.
- Produce an exact 1,199-tensor coverage receipt: 506 additive-ternary matrices,
  360 preserved non-matrix tensors, and 333 identity-bound deferred vision
  tensors, with no unknown, duplicate, or missing tensor. Recompute these counts
  from the pinned weight index and shard headers; the stated values are a golden
  for the pinned revision, not constants to accept against a changed source.
- Reproduce required baselines at R2/R3/R4 or the largest attainable artifact
  not exceeding the ceiling. If the rate gap exceeds 0.05 artifact bpw, report
  the point for context but do not use it to establish a strict head-to-head
  win.
- Run frozen SALT V2 PTQ once; repeat packaging and evaluation to prove
  determinism.
- Run any literature-continuity model only after the flagship artifact exists;
  it cannot substitute for the 27B result.

**Current narrow evidence (2026-07-17):** the MTP implementation has an exact
15-tensor loader and a compiled-authorized synthetic H32/I48/V37 fixture. The
fixture executes pinned vLLM's real `EagleProposer.set_inputs_first_pass`, target
and MTP models, Triton attention, multi-token prefill, cached decode, target
logits/argmax, MTP logits, and complete two-KV-head caches. Two fresh-cache CUDA
oracle generations were byte-identical, and the Tritium CPU reference matched
the sealed lanes under the fixed 2e-3 absolute profile. The generator rejects
all inherited Triton controls and the non-prefixed compiler/PTXAS override set,
then attests the resolved bundled compiler inputs. This proves the narrow
fixture contract only. It does not satisfy the 64-layer dense parity, exact 27B
coverage, Tritium CUDA, checkpoint-scale serving, or PTQ work items above.

**PTQ gate before refinement**

- R3 SALT V2 closes at least 50% of SALT V1's perplexity gap to bf16.
- It is non-dominated by every reproduced additive/ternary baseline in
  quality versus artifact bytes.
- It does not require dense dequantization and is no slower than SALT V1 by
  more than 10% in the claimed serving regime.

If this gate fails, spend at most the first one-eighth of the scale-only cap.
Continue scale-only only when a conservative learning-curve fit projects that
the gate is reachable within the frozen cap. Short PV stops.

The 50% gap-closure threshold is intentionally inherited and preregistered. A
different Qwen3.6 SALT V1 gap does not relax it after measurement; failure is a
valid negative result.

### Stage 9 — Qwen3.6-27B scale-only and short-PV proof

Run three seeds for the selected refinement recipe. Report all seeds and the
predeclared aggregate; do not select the best seed as the headline.

**Near-zero-divergence gate at R3**

- relative held-out perplexity increase `<= 1.0%` versus bf16;
- mean six-task accuracy decrease `<= 0.5` percentage points;
- no individual task decrease greater than `1.0` point;
- divergence thresholds frozen at rung 2 pass;
- 95% paired-bootstrap confidence intervals do not cross the failure boundary,
  or the result is reported inconclusive rather than passed.

**Additive-ternary SOTA gate**

- primary quality aggregate is strictly better than every reproduced
  additive/ternary baseline at no greater artifact bytes;
- a statistical tie is reported as a tie;
- the native kernel is measured and the artifact/recipe is publishable.

**Global Pareto gate**

- no reproduced low-bit baseline has both better quality and no larger artifact;
- no reproduced low-bit baseline has both better claimed-regime runtime and no
  worse quality at no larger resident bytes.

Near-zero conversion, additive-ternary SOTA, and global Pareto-SOTA are three
independent booleans. Passing one does not imply the others.

### Stage 10 — Deferred multimodal and confirmation work

Stage 10 is not authorized by this plan. It begins only after the complete 27B
language-plus-MTP evidence pack passes its applicable gates and a new
preregistration defines the exact vision coverage policy, dense multimodal
oracle, evaluation suite, confirmation model if any, compute ceiling, and stop
conditions. Language-plus-MTP results remain scoped as such and are never
extrapolated to multimodal behavior or another model size.

Any confirmation model belongs to its own future preregistration. It is not a
hidden substage of this deferred multimodal stage.

The former Qwen3-8B-to-32B/50B Net2Wider experiment is no longer an active SOTA
campaign stage. Any future model-growth experiment is reported separately from
same-model conversion and requires its own function-preservation oracle and
claim language.

### Stage 11 — Reproducibility package and claim generation

**Work**

- Publish immutable model artifacts, recipes, calibration/evaluation builders,
  reports, ledger, exact commands, environment capture, and known limitations.
- Generate comparison tables from the ledger. Handwritten numbers fail review.
- Add a claim generator that emits only the claim tiers whose machine-readable
  gates are true.
- Re-run the primary inference measurements on a second same-class machine or
  independent environment before using “SOTA.”

**Gate**

- A clean checkout reproduces hashes or documents the expected hardware-only
  nondeterminism boundary.
- Every table cell links to a report/receipt and every receipt binds its source
  revision and command.
- The abstract-level claim is no broader than model family, rate, coverage,
  hardware, and refinement track actually measured.

## Cost authorization and required evidence

This plan authorizes zero paid GPU-hours and zero rental spend. The earlier
8B/32B estimates do not scale reliably to Qwen3.6's hybrid 27B graph and are
superseded. Before requesting paid compute, the implementation must emit a
measured forecast from local runs rather than extrapolating a paper's model or
kernel:

| Forecast input | Required evidence |
|---|---|
| Dense reference and SALT peak memory | allocation receipts separated into host, device-resident, and transient bytes |
| Tensor fitting throughput | timed representative DeltaNet, full-attention, embedding, head, and MTP matrices |
| Activation capture and evaluation throughput | token counts, sequence geometry, cache bytes, and wall time from the frozen corpus |
| Resume overhead | injected interruption with verified content-addressed restart |
| Baseline cost | same-box measured or independently frozen recipe, never a literature runtime estimate |

A future authorization request must state the exact hardware, measured lower
and upper GPU-hour bounds, storage and egress, contingency, stop gate, and
maximum spend. PTQ is estimated first. Scale-only and short-PV forecasts are
separate and remain unauthorized until the PTQ gate passes.

### Spend controls and time reductions

- Cache and content-address teacher logits, activations, Hessians, Fisher, and
  tokenized corpora once; reuse only when all provenance digests match.
- Use one-layer and four-layer successive halving before a full-model solve.
- Quantize independent layers in parallel, but serialize any step whose error
  feedback crosses the layer boundary.
- Run PTQ first, then scale-only, then PV. Never launch all refinement modes in
  parallel before the preceding gate.
- Freeze trits during scale-only refinement; this sharply reduces optimizer
  state and search cost.
- Refine only blocks selected by validation-gradient contribution when the
  frozen ablation proves it matches full refinement. Charge its selection map.
- Evaluate learning curves at fixed fractions of the token cap and stop on
  marginal improvement per GPU-hour.
- Use fp16/bf16 activation caches where the curvature parity gate permits;
  keep accumulators at the precision required by the solver oracle.
- Reuse ADR 0027 checkpointing, resident packed storage, host offload, and
  gradient streaming. Do not replicate 27B optimizer state; require a measured
  sharded/offloaded admission receipt before refinement.
- Prefer interruptible/spot instances only after resume and checkpoint
  integrity are proven; include preemption waste in billed hours.

## Review and validation protocol

For every implementation commit:

1. Run the narrow unit/property tests for the changed stage.
2. Run `cargo fmt --all -- --check`.
3. Run clippy with warnings denied for affected crates and all targets.
4. Run relevant CPU/CUDA parity and sanitizer gates.
5. Review the commit through the repository-required DeepSeek V4 Pro commit
   reviewer, verify every cited finding in source, fix confirmed defects, and
   re-review until PASS or PASS WITH NITS.
6. Record skipped false positives in the follow-up commit message.

No paid run starts from an uncommitted or unreviewed tree. The recipe records
the exact clean revision.

## Current implementation status and gaps

The current software commits implement a bounded reference substrate, not the
empirical claim. They add the canonical zero-point-free D2/B3/S34 semantic
package, exact serialized and indexed-runtime accounting, deterministic joint
`3^P` fitting, conditioned scale solving, activation-cache reopen validation,
source-bound input/Fisher/Kronecker curvature primitives, standalone
block-feedback and delta-correction primitives, model-integrated full-inverse-
Hessian feedback with a provisional-to-full delta-corrected refit, a content-
addressed ordered `SaltV2MasterFit` separated from exact-byte allocation, an
exact equal-cost byte allocator with a scalable indexed lexicographic tie path,
mixed-plane model fitting, source/result-bound recovery and G1-growth evidence,
CPU semantic-reference execution, CUDA direct packed add-sub-skip execution,
and the resumable `tritium-salt` evidence facade. Feedback receipts bind every
natural column group, both fit inputs and reconstructions, the exact number of
nonzero deltas, and the final working/reconstruction states. Legacy SALT-GGUF
and TSLB consumers now index through `Read + Seek`, validate complete container
structure and every SALT payload, and build final packed arenas without retaining
an artifact-wide copy. The canonical SALT V2 package now has the same strict
seek-backed format seam: it validates every D2/B3/S34 plane and scale, retains
only owned tensor metadata plus the compact physical presence map, reports exact
per-tensor indexed-runtime requirements, streams named tensors in arbitrary
order with 64 KiB payload staging, and detects same-handle source mutation. The
CUDA consumer now preallocates those exact payload/scale arenas, streams the
reader through fixed 64 KiB host buffers, constructs only the compact map/rank
index, and publishes the resident handle and allocation receipt only after the
reader's terminal mutation check succeeds. Exact CUDA projection and selected-
row embedding kernels now execute D2/B3/S34 directly from those arenas, publish
caller output transactionally, and report transient plus steady allocations
without a dense weight shadow. A strict `tritium-nn` assembler binds the caller's
expected package identity before and after load, consumes every package matrix
exactly once, retains only required fp32 vectors, supports tied and untied heads,
preserves Qwen2 QKV biases and Qwen3 QK-normalization weights, and returns a
canonical whole-model weight-allocation receipt. Real-CUDA fixtures compare complete
Llama/Qwen model logits against an independent dense Hugging Face reconstruction
for every admitted codec.
The Qwen3.5-family reference seam now parses the outer and nested configuration,
content-binds canonical execution semantics and every source tensor, assembles
configured Gated DeltaNet/full-attention language schedules, and exact-loads the
15-tensor one-layer MTP graph. MTP execution is capability-gated by a private
compiled oracle ledger. Its only row is a reproducible synthetic fixture that
binds official vLLM first-pass proposer inputs, target hidden/logit/token parity,
MTP hidden/logit parity, and full prefill/decode KV caches; its receipt is
explicitly ineligible for production. The sealed v1 parser now also understands
a distinct production-checkpoint prefill/decode coverage and evidence tuple,
and parity receipts admit production only for that tuple after every lane
passes. The compiled production ledger remains deliberately empty: the pinned
Qwen3.6-27B source weights, production oracle body and implementation manifest
have not been acquired, executed and reviewed, so fixture evidence cannot mint
production MTP capability.
The reference fitter now also exposes `plan_salt_v2_tensor_master` and
`fit_salt_v2_tensor_master`: a source-bound global tensor ordinal can be
cataloged before fitting, then its canonical rate-free Pmax master is emitted
one 256-coefficient tile at a time. Two-tensor goldens prove those independent
streams and receipts are byte-identical to the corresponding outputs of the
whole-model solver. This removes model-wide master residency from the PTQ
producer boundary. `reconcile_qwen36_ptq` now joins the admitted seek-backed
Qwen shards to an exact 506-record evidence namespace, rejects extra/missing or
misordered records and mixed token streams, plans the immutable campaign while
retaining only one widened matrix and one factor record, resumes only missing
masters, and seals through the existing content-addressed store. The producer
that collects all 506 `S2KF` records from real calibration remains open.
The same driver is exposed to Python as
`tritium.salt.reconcile_qwen36_ptq_masters`; its immutable receipt is explicitly
rate-free and cannot be mistaken for a packaged model or evaluation result.
Guided-Fisher, input-Hessian, and forward-KL curvature can now retain exact
Kronecker structure instead of expanding one dense G128 matrix per output row.
The solver materializes only the active G128
`output_scalar * input_block + damping * I` work block, and bit-exact goldens
match fully expanded curvature and the emitted master payload. Evidence
residency is therefore one shared
input block per input group plus one scalar per output row. The canonical
`S2KF` record now binds source model/cache/token identities, global tensor
ordinal, tensor geometry, upstream evidence, every input block, every output
scalar, and damping behind a domain-separated checksum. Its bounded reader
rejects corruption, truncation, noncanonical encodings, forged allocation
counts, and provenance drift before the reopened record can drive an exact
tensor-master fit. The restart API consumes the source/cache/token identities
already bound by that record, so it does not reconstruct a model-sized
activation cache merely to resume pure PTQ. Checkpoint-scale evidence
collection remains open.
Pipeline ownership is process-serialized through a reserved lock namespace.
Compact packages remain exact prefixes of their near-lossless packages. ADR
0028's 2026-07-15 amendments make the ordered-master prefix curve, rather than
independent per-P refits, the binding pricing contract and keep entropy coding
outside the direct-runtime claim until it earns separate evidence.

The following work deliberately remains open and keeps this plan in progress:

- the Qwen3.5-family adapter and MTP fixture gate exist, but the exact pinned
  Qwen3.6-27B shards have not been streamed through them. There is no proof yet
  for the 64-layer 3:1 mixed schedule, the checkpoint-scale language/MTP tensor
  set, Tritium host/CUDA parity, full serving lifecycle, or practical recurrent
  and KV-cache allocations. The compiled MTP authorization ledger contains no
  production evidence row, so fixture success cannot admit Stage 8;
- the production pure-PTQ driver now connects admitted checkpoint tensors to
  reopened factorized-curvature evidence and the canonical campaign store, but
  checkpoint-scale evidence collection, block-output reconstruction,
  scale-only teacher-KL, and hard PV updates remain open;
- `fit_salt_v2_master(...) -> SaltV2MasterFit` now invokes the standalone dense
  BlockLDLQ path, then
  `allocate_and_pack_salt_v2_master(...) -> SaltV2ModelFitResult` slices exact
  prefixes without refitting. The whole-model compatibility entry point still
  owns the complete fitted master, but the independent tensor-master path now
  bounds producer memory to one source matrix, one 256-coefficient fit tile and
  solver-local state while retaining the global architecture ordinal and exact
  whole-model bytes. This is still a bounded CPU reference: its full
  inverse Hessian is dense binary64, input columns and natural groups must
  preserve row-local G128 scale geometry. Canonical per-tensor radix-3 masters are now streamed into an
  immutable CAS, semantically reopened with bounded staging, installed beneath a
  base-preserving Qwen campaign namespace, and sealed only after all 506 ordered
  language/MTP PTQ masters verify. A sealed PTQ campaign can now open a distinct,
  typed ScaleOnly child: its descriptor binds the exact parent completion, base
  workspace, campaign, ordered master set, every parent tensor-master identity,
  and a bounded-streaming projection of static geometry, per-tile admissible
  prefixes, plane order, and every hard trit. Child losses and scales may change;
  install, reopen, progress, seal, and completion reopen reject any trit/prefix or
  live-parent drift. Parent checks run immediately before CAS/completion
  publication; fixed-structure and prepublication failures publish no child object
  or slot, terminal parent rechecks fail closed, sealed namespaces cannot be
  rewritten through install, and the public PTQ opener cannot be used to escape
  the typed refined capability. Ordinary PTQ lifecycle scans skip the new fixed
  projection and its additional trit hashing, preserving their existing bytes
  and identities.
  The sealed parent can now bind one immutable Compact/NearLossless allocation
  pair. Both two-bit maps are streamed through bounded temporary storage into CAS;
  the receipt binds the parent completion, allocation policy, map/loss identities,
  selected-plane totals, and nesting. Strict reopen replays every verified map
  count against every canonical parent master and rederives all identities. A
  second typed capability now admits the exact Compact and NearLossless SALT V2
  packages. It requires exact codec, tensor order, names, shapes, tile geometry,
  selected plane counts, parent-prefix semantic tensor identities, serialized
  ledger, indexed-runtime ledger, and both byte ceilings. Exact package bytes are
  retained as CAS records; the small admission manifest is published last and
  binds both package IDs and both selected map/loss identities. Reopen restages
  the records with bounded memory, reparses the complete packages, repeats parent
  and selection validation, and performs terminal package mutation checks. Only
  this package-admitted capability exposes the v2 scale-only opener; its campaign
  catalog binds the admission ID, selection ID, and both package IDs, and its
  install/progress paths use stable package-record inode/version pins plus small
  manifest checks around child work, while open, seal, and completion reopen
  repeat full cryptographic package and parent-prefix validation.
  The format layer now also has an exact seek-backed package writer. It plans
  headers, physical ledgers, offsets, ragged embedded counts, and the canonical
  full-tile map from a lazy flat count stream, then writes one selected semantic
  tile at a time directly into disjoint payload/scale regions. D2, B3, and S34
  goldens are byte-identical to the canonical in-memory encoder. Its retained
  state is tensor metadata plus two bits per full allocation tile; it does not
  materialize a whole-model `SaltV2Package`. The selected campaign now drives
  both Compact and NearLossless writers in one verified parent-master pass,
  borrowing both prefixes from the same decoded tile. Exact outputs flow
  directly into package admission without a second caller-source copy, and the
  Compact staging file is released before NearLossless CAS publication. The
  governed fixture is byte-identical to the canonical packages and reopens with
  no temporary files. Checkpoint-scale allocation-map production remains open.
  This remains **structural fixture evidence**, not checkpoint-scale or quality
  evidence. The pure-PTQ checkpoint/evidence driver now targets these stores,
  but has not been run on the pinned checkpoint; PV/KL execution over real
  tensors remains unimplemented. A measured flagship receipt is still required
  before Stage 5 can advance. CAS master installation otherwise stages one
  exact record, fuses generic byte validation with canonical SALT semantic decoding
  through the retained file handle, requires decoder completion before
  sync/publication, and returns both generic and semantic receipts without an
  install-time reopen. Strict campaign reopen/progress/seal paths use the same
  one-pass verified visitor instead of reading each master twice. Exclusive
  campaign resume and pre-seal paths mark canonical slot roots and reclaim only
  canonical unreferenced objects, reject unknown object layouts before any unlink,
  remove recognized crash-left slot temporaries, serialize same-handle mutation,
  and never sweep a sealed namespace. Exact-length semantic failures therefore
  publish neither CAS objects nor slots, while valid crash-window orphans are
  deterministically reclaimed before the 27B campaign;
- package/runtime scale geometry is G128-only; the G64/G256 promotion ablations
  require a versioned format/runtime field before they can enter the grid;
- the equal-cost allocator is exact and scalable, including binary64 reduction
  and lexicographic ties. General unequal-cost two-budget allocation remains an
  exact bounded dynamic program and fails closed above 4096 states; a scalable
  general solver requires a new proof or a deliberately approximate campaign
  mode with a separately labeled lower bound;
- signed-RHT execution fails closed; S34 has a constrained deterministic CPU
  reference fitter but no campaign evidence; the direct exact runtime is the
  correctness baseline, and the current `fast` CUDA entry point aliases it
  rather than claiming a speedup;
- B3 and S34 have matched semantic fixtures and compact-codec microbenchmarks,
  but no ANS/rANS transport has been admitted. Any candidate must remain
  independently seekable and win after tables, indexes, checksums, padding,
  decode latency, and peak scratch are charged. Expansion before execution does
  not reduce the resident-weight claim;
- `PhysicalSizeReport::from_salt_v2_package_bytes_with_runtime_receipts(...)`
  now rederives the complete indexed layout from canonical package bytes and
  rejects wrong-count or component-disagreeing per-tensor resident runtime
  ledgers. The whole-model assembler now aggregates independently checked CUDA
  receipts in canonical package order and separately content-binds every
  preserved fp32 vector. The production campaign adapter must still feed this
  measured whole-model receipt into `PhysicalSizeReport` and the immutable
  campaign ledger; caller-provided architecture geometry alone is not claim
  evidence;
- legacy G1 widening has canonical source/result identities and deterministic
  oracle evidence, but model growth is no longer on the active flagship path;
- the eager semantic SALT V2 decoder remains as a compatibility/reference API,
  while `SaltV2PackageReader` supplies strict bounded-staging package access.
  The CUDA adapter consumes its packed-plane visitor directly into final device
  arenas and the `tritium-nn` assembler maps a complete package plus
  configuration/preserved vectors into runnable projections, embeddings, and
  heads. A strict schema-v3 loader now derives all identities from the manifest,
  verifies both physical ledgers and every language sidecar, assembles the full
  language/MTP graph from a selected seek-backed package, and is reachable from
  the Python wheel for greedy token generation without dense matrix shadows.
  Feature-gated CUDA wheels now stream those matrices into final device SALT V2
  allocations; a full synthetic graph emitted a device-resident receipt and ran
  on an RTX 4090, while CPU-only wheels reject CUDA placement. This is synthetic
  runtime evidence, not pinned-checkpoint, large-K performance, or MTP-oracle
  promotion evidence. It remains a host-orchestrated correctness path: the
  fused resident decoder does not yet accept SALT V2 projections. The standard
  SwiGLU training adapter now preserves optional QKV bias and conventional
  per-head Q/K RMSNorm through dense and packed CUDA graphs, but this does not
  implement Qwen3.6's hybrid DeltaNet/gated-attention training graph or its
  zero-centered norm semantics. Those specialized device/training integrations
  must precede a Qwen3.6 refinement campaign;
- optimized resident large-K CUDA SALT kernels, SALT V2 fused decoder/prefill
  dispatch, and Qwen3.6's device-resident hybrid recurrent/KV cache remain
  required for practical 27B serving;
- required third-party baselines have not all been integrated or reproduced
  under one byte/evaluation boundary;
- the 135M, 1.7B, optional Llama2-7B, and Qwen3.6-27B acceptance artifacts do
  not exist, and no paid campaign has been run;
- no Qwen3.6-27B dense-parity, PTQ, refined, language-plus-MTP, or multimodal
  acceptance artifact exists;
- the release registry now fail-closes conversion admission through
  `tritium.qwen36-conversion-refinement.v1`: the exact pinned language-plus-MTP
  source, 1,199-tensor coverage, 64-layer host/CUDA and MTP-oracle parity,
  separate Compact PTQ/NearLossless PTQ/NearLossless refined candidate bundles,
  refined-parent lineage, strict reload and repeat determinism are mandatory.
  This is validator infrastructure only; no checkpoint-scale receipt exists;
- held-out admission now separately requires `tritium.qwen36-quality.v1` and
  `tritium.qwen36-task-retention.v1`, exact children of that conversion receipt
  and bound to its refined candidate bundle. The quality contract recomputes
  50% PTQ gap closure and the <=1% refined perplexity/CI gate, requires a
  complete preregistered matched-byte baseline inventory, and keeps near-zero,
  additive-ternary SOTA and global-low-bit Pareto verdicts independent. The
  task contract recomputes six individual and mean accuracy deltas and rejects
  confidence bounds above 1.0/0.5 percentage points. No empirical receipt exists;
- native execution and accounting now fail-close through
  `tritium.qwen36-runtime.v1` and `tritium.qwen36-physical-bytes.v1`. Runtime
  admission requires a physical CUDA identity, direct ternary kernels, zero
  dense materialization/host transfers, all three tracks across prefill/decode
  and two context/batch regimes, >=20 iterations, <=10% PTQ slowdown versus
  SALT V1, and measured MTP acceptance plus >1x speedup. Physical admission
  rehashes all three candidate bundles, recomputes matrix/whole/resident bpw and
  dense reduction ratios, enforces 2.25/4.0 matrix caps plus <=0.01 metadata bpw,
  and requires host/device/transient peaks. These remain validator-only gates;
- no near-zero-divergence, additive-ternary SOTA, global Pareto, or cost result
  has been earned by this implementation alone.

## Definition of done

- [ ] Qwen3.6-27B dense reference parity covers all 64 mixed-schedule language
      layers and the bundled one-layer MTP drafter before SALT conversion.
- [ ] The language-plus-MTP coverage and physical ledgers consume every expected
      tensor exactly once and bind all preserved bytes and runtime allocations.
- [ ] The Qwen3.6-27B PTQ and refined tracks produce separate artifacts,
      provenance, metrics, costs, and claim booleans.
- [ ] No paid run occurs without a new explicit approval, and multimodal work
      remains deferred until the language-plus-MTP gates authorize it.
- [ ] Stage 0 physical accounting agrees with actual artifact and resident bytes.
- [ ] Stage 1 zero-point-free direct and radix-3 formats round-trip and fuzz clean.
- [ ] Stage 2 output-aware curvature artifacts are deterministic and validated.
- [ ] Stage 3 joint `3^P` solver beats/ties the greedy oracle and brute-force goldens.
- [ ] Stage 4 feedback, delta correction, and exact-byte allocation pass references.
- [ ] Stage 5 PTQ, scale-only, and short-PV tracks remain separately reproducible.
- [ ] Stage 6 native exact/fast kernels pass parity, sanitizer, and no-dense-materialization gates.
- [ ] Stage 7 freezes the recipe on 1.7B or records a terminal negative result.
- [ ] Stage 8 reproduces baselines and completes the frozen Qwen3.6-27B PTQ proof.
- [ ] Stage 9 completes the separately labeled 27B refined proof and records
      independent near-zero, additive-SOTA, and global-Pareto booleans.
- [ ] Any confirmation model or multimodal stage is separately preregistered and
      runs only when authorized by the 27B language-plus-MTP gates.
- [ ] Stage 11 publishes hashes, commands, reports, ledger, costs, and generated claim.
- [ ] The public "Tritium SOTA" claim is generated only after all applicable
      ADR 0026 and ADR 0028 gates are green.
- [ ] ADR 0028 remains Proposed until these gates, not document completion, justify a status change.

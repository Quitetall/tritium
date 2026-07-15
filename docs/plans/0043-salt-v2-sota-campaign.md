# 0043 — SALT V2 implementation and SOTA campaign

Status: **IN PROGRESS** (software/reference substrate implemented 2026-07-15;
empirical campaign gates remain open)

- **Decision:** [ADR 0028](../adr/0028-salt-v2-additive-ternarization.md)
- **Research cutoff:** 2026-07-14, inclusive
- **Claim status:** not achieved
- **Primary proof rung:** Qwen3-8B, the fixed 8–9B-class model
- **Confirmation rung:** Qwen3-32B
- **Development rung:** SmolLM2-1.7B, reusing Tritium's existing model and
  campaign support

## Outcome

Build and evaluate a zero-point-free SALT V2 converter and native inference
path whose stored weights are only additive ternary planes plus positive scales.
The campaign must answer, with reproducible measurements:

1. Does joint, output-aware additive ternarization beat SALT V1 and other
   ternary/additive methods at the same physical bytes?
2. Can Qwen3-8B reach the preregistered near-zero-divergence gate at an
   approximately 3.3–3.6 matrix-bpw point?
3. Does the direct ternary kernel produce a global quality/size/runtime Pareto
   point, rather than a quality-only result with a slow decoder?
4. If the 8B result passes, does the result reproduce on Qwen3-32B without a
   disproportionate training bill?

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
- The direct-model track is Qwen3-8B then Qwen3-32B. The capacity track widens
  Qwen3-8B's SwiGLU axes toward approximately 32B coefficients, then requires a
  distinct whole-head/hidden-width transform before the endpoint containing at
  least 50B stored ternary coefficients.
- Stable orchestration lives in `tritium-salt` through `SaltV2::explain` and
  `SaltV2::reconcile`. `SaltPipeline::{start, advance, resume}` is experimental
  but must execute the identical stage sequence and durable evidence contract.

### Model ladder

| Rung | Model | Purpose | May choose hyperparameters? |
|---|---|---|---|
| 0 | deterministic synthetic matrices and the existing tiny transformer | solver, accounting, and parity oracles | implementation constants only |
| 1 | SmolLM2-135M | full-pipeline smoke and cheap negative-result discovery | no |
| 2 | SmolLM2-1.7B | choose group size, packing, restart count, and curvature variant | yes, only from the preregistered grid |
| 3 | Qwen3-8B | primary 8–9B quality and cost result | no; recipe frozen from rung 2 |
| 3b | Llama2-7B | literature bridge for QTIP, LLVQ, AQLM, VPTQ, and PV-Tuning | no; same frozen recipe |
| 4 | Qwen3-32B | confirmation and scale result | no; only batch/sharding may change |
| G1 | Qwen3-8B widened in SwiGLU intermediate axes | function-preserving capacity point near 32B stored coefficients | only the preregistered target and seed |
| G2 | whole-head/hidden-width growth from G1 | capacity endpoint with at least 50B stored ternary coefficients | no; transform must pass exact dense-function and receipt gates |

The 1.7B evaluation split is held out from pilot selection. Qwen3-8B and
Qwen3-32B task results remain sealed until the recipe digest is frozen. If a
Qwen-specific correctness bug requires a recipe change, invalidate the affected
run, fix it, and rerun all methods; do not patch only SALT V2.

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
- Add smooth warmup followed by PV hard-trit/scale alternating updates.
- Project and validate hard trits after every discrete phase and before every
  checkpoint.

**Frozen token caps**

| Rung | Scale-only maximum | Short PV maximum |
|---|---:|---:|
| 1–2 | 8M tokens | 32M tokens |
| Qwen3-8B | 32M tokens | 256M tokens |
| Qwen3-32B | 64M tokens | 512M tokens |

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
- Freeze recipe, evaluation thresholds, and digests before unsealing Qwen3-8B.

If the freeze gate fails, publish the 1.7B negative result and stop. Do not rent
the 8B refinement campaign.

### Stage 8 — Reproduce baselines and run the Qwen3-8B PTQ proof

**Work**

- Reproduce required baselines at R2/R3/R4 or the largest attainable artifact
  not exceeding the ceiling. If the rate gap exceeds 0.05 artifact bpw, report
  the point for context but do not use it to establish a strict head-to-head
  win.
- Run frozen SALT V2 PTQ once; repeat packaging and evaluation to prove
  determinism.
- Run the same recipe on Llama2-7B for literature continuity after the primary
  artifact exists.

**PTQ gate before renting refinement**

- R3 SALT V2 closes at least 50% of SALT V1's perplexity gap to bf16.
- It is non-dominated by every reproduced additive/ternary baseline in
  quality versus artifact bytes.
- It does not require dense dequantization and is no slower than SALT V1 by
  more than 10% in the claimed serving regime.

If this gate fails, spend at most the first one-eighth of the scale-only cap.
Continue scale-only only when a conservative learning-curve fit projects that
the gate is reachable within the frozen cap. Short PV stops.

### Stage 9 — Qwen3-8B scale-only and short-PV proof

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

### Stage 10 — Qwen3-32B confirmation

This stage is authorized only if Stage 9 passes near-zero divergence or the
additive-ternary SOTA gate. Use ADR 0027's resident/host-offload and supervised
multi-GPU infrastructure. The quantization recipe is unchanged; only sharding,
batch geometry, and device count may vary. Refinement execution geometry enters
recipe provenance because it can change the artifact; inference geometry
enters evaluation provenance.

**Gate**

- Re-run PTQ before refinement and confirm the direction and approximate
  fraction of gap closure seen at 8B.
- Apply the same near-zero and SOTA definitions, not relaxed 32B thresholds.
- Run a failure-injection resume test before a paid long refinement.
- Report wall time, billed GPU-hours, peak host/device memory, checkpoint bytes,
  and retries, including failed attempts.

If the 32B result fails, the 8B claim remains scoped to 8B. It is not
extrapolated to 32B–50B.

### Stage 10b — Grow Qwen3-8B to the 32B and >=50B coefficient frontier

This track is reported separately from same-model conversion. First apply the
existing deterministic SwiGLU Net2Wider transform until the planned stored
ternary coefficient count is approximately 32B. The receipt binds the source
model, target coefficient count, old/new intermediate widths, mapping, split
weights, and seed. Dense logits before ternarization must match within the
existing function-preservation tolerance.

The >=50B endpoint may not be produced by silently stretching only the FFN.
It requires a separately reviewed whole-head/hidden-width transform that repeats
complete attention/head structures, handles RMSNorm and residual geometry, and
passes a dense end-to-end function-preservation oracle before SALT fitting.

**Gates**

- the planner counts stored ternary coefficients, plane-for-plane, rather than
  labeling an fp parameter count as ternary capacity;
- G1 reaches its declared approximately 32B coefficient target with the minimum
  valid intermediate width and a replayable receipt;
- G2 stores at least 50B ternary coefficients and passes the independent dense
  function oracle before any paid recovery;
- both points report quality versus exact package/resident bytes against direct
  Qwen3-8B and Qwen3-32B; a capacity win is not called lossless conversion;
- failure of the whole-head transform leaves G1 publishable but blocks the >=50B
  claim.

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

## Cost and rental ladder

GPU-hour estimates are planning bounds, not provider quotes. Before each rental,
replace the rate with a current written quote and record it in the campaign
plan. Dollar examples use a blended **$3–$6 per H100-equivalent GPU-hour** and
exclude tax, storage, and egress. Add a 10–20% contingency; never hide failed
or preempted hours.

| Step | Expected compute | Example rental | Hard authorization ceiling |
|---|---:|---:|---:|
| Local unit, 135M smoke, kernel work | 20–80 RTX 4090 hours | $0 when local | no cloud required |
| 1.7B pilot after local pruning | 4–16 H100-equivalent GPU-hours | $12–$96 | 24 GPU-hours |
| One frozen 8B PTQ conversion | 8–32 GPU-hours | $24–$192 | 40 GPU-hours |
| Complete 8B PTQ/baseline/ablation rung | 48–160 GPU-hours | $144–$960 | 192 GPU-hours |
| 8B scale-only plus three-seed short PV | 96–384 GPU-hours | $288–$2,304 | 512 GPU-hours |
| One 32B PTQ confirmation | 32–128 GPU-hours | $96–$768 | 160 GPU-hours |
| 32B scale-only/short-PV confirmation | 512–2,048 GPU-hours | $1,536–$12,288 | 2,048 GPU-hours |
| Broad paper campaign with all baselines and repeats | 2,000–5,000 additional GPU-hours | $6,000–$30,000 | separate approval |

The 8B proof is not expected to cost $25,000. That figure becomes plausible
only for a broad 32B paper campaign with multiple baselines, ablations, seeds,
and expensive on-demand rates. A single PTQ conversion should be tens of
GPU-hours, consistent with published 7B PTQ ranges from minutes for PTQTP/BPDQ
through approximately 8 GPU-hours for VPTQ and roughly 24 for AQLM. The short
teacher-KL phase, not ternary packing, is the main variable cost.

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
  gradient streaming. Do not replicate 32B optimizer state without the
  documented approximately 768 GB host-RAM geometry or a sharded alternative.
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
block-feedback and delta-correction primitives, an exact equal-cost byte
allocator with a scalable indexed lexicographic tie path, mixed-plane model
fitting, source/result-bound recovery and G1-growth evidence, CPU semantic-
reference execution, CUDA direct packed add-sub-skip execution, and the
resumable `tritium-salt` evidence facade. Legacy SALT-GGUF and TSLB consumers now
index through `Read + Seek`, validate complete container structure and every SALT
payload, and build final packed arenas without retaining an artifact-wide copy.
Pipeline ownership is process-serialized through a reserved lock namespace.
Compact packages remain exact prefixes of their near-lossless packages. ADR
0028's 2026-07-15 amendments make the ordered-master prefix curve, rather than
independent per-P refits, the binding pricing contract and keep entropy coding
outside the direct-runtime claim until it earns separate evidence.

The following work deliberately remains open and keeps this plan in progress:

- the production model driver does not yet connect block-output reconstruction,
  scale-only teacher-KL, or hard PV updates to real checkpoint tensors;
- the model fitter does not yet invoke the standalone block-feedback/delta path.
  The required integration seam is a durable Search-stage master artifact:
  `fit_salt_v2_master(...) -> SaltV2MasterFit`, followed by
  `allocate_and_pack_salt_v2_master(...) -> SaltV2ModelFitResult`. It must bind
  full input-column inverse-Hessian state, natural column groups, the source
  model, and a detailed feedback receipt before `feedback_applied` can be true;
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
- physical component truth still requires the opened immutable package and a
  checked runtime allocation receipt; caller-provided architecture geometry
  alone is not claim evidence;
- G1 widening now has canonical source/result identities, deterministic oracle
  evidence, transactional rollback, and a tracked-fp32-payload preflight. The
  dense oracle still constructs infallible full-model clones, so this is not a
  model-scale RSS/admission guarantee and remains a blocker before a 32B run;
- the distinct whole-head/hidden-width transform, RMSNorm/residual handling,
  and dense function-preservation oracle required for G2 do not exist;
- SALT V2 package opening is still eager and materializes semantic trit vectors;
  there is no seek-indexed SALT V2 model loader, and the legacy model adapter
  rejects Qwen QK-norm and QKV-bias checkpoints. A bounded SALT V2 consumer and
  end-to-end target-architecture integration must precede any Qwen campaign;
- resident large-K CUDA SALT kernels and a paged fp16 or int8 KV cache with a
  runtime context override remain required for practical 32B serving;
- required third-party baselines have not all been integrated or reproduced
  under one byte/evaluation boundary;
- the 135M, 1.7B, Qwen3-8B, Llama2-7B, Qwen3-32B, and growth acceptance artifacts
  do not exist, and no paid campaign has been run;
- no near-zero-divergence, additive-ternary SOTA, global Pareto, or cost result
  has been earned by this implementation alone.

## Definition of done

- [ ] Stage 0 physical accounting agrees with actual artifact and resident bytes.
- [ ] Stage 1 zero-point-free direct and radix-3 formats round-trip and fuzz clean.
- [ ] Stage 2 output-aware curvature artifacts are deterministic and validated.
- [ ] Stage 3 joint `3^P` solver beats/ties the greedy oracle and brute-force goldens.
- [ ] Stage 4 feedback, delta correction, and exact-byte allocation pass references.
- [ ] Stage 5 PTQ, scale-only, and short-PV tracks remain separately reproducible.
- [ ] Stage 6 native exact/fast kernels pass parity, sanitizer, and no-dense-materialization gates.
- [ ] Stage 7 freezes the recipe on 1.7B or records a terminal negative result.
- [ ] Stage 8 reproduces baselines and completes the frozen Qwen3-8B PTQ proof.
- [ ] Stage 9 records independent near-zero, additive-SOTA, and global-Pareto booleans.
- [ ] Stage 10 runs only when authorized by Stage 9 and scopes any 32B failure honestly.
- [ ] Stage 11 publishes hashes, commands, reports, ledger, costs, and generated claim.
- [ ] ADR 0028 remains Proposed until these gates, not document completion, justify a status change.

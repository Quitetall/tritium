# ADR 0028 — SALT V2: output-aware additive ternarization

Status: **PROPOSED** (2026-07-15)

- **Deciders:** Brian Lam
- **Research cutoff:** 2026-07-14, inclusive
- **Relates:** supersedes the fitting, allocation, and accounting decisions in
  [ADR 0001](./0001-salt-quantization.md) while preserving its additive ternary
  execution invariant; consumes the Fisher and distillation machinery in plans
  [0039](../plans/0039-fisher-sensitivity.md),
  [0040](../plans/0040-real-model-distillation.md), and
  [0042](../plans/0042-paper-and-param-scaling.md); uses the device-resident and
  host-offload substrate in [ADR 0027](./0027-device-resident-training-perf-and-scale.md);
  the implementation and evidence sequence is
  [plan 0043](../plans/0043-salt-v2-sota-campaign.md).

> **Claim boundary.** This ADR is a preregistered research decision, not evidence
> that SALT V2 is state of the art. Tritium has not yet demonstrated a
> matched-rate quality win over the strongest low-bit methods, a near-lossless
> 8–9B conversion, or a 32B result. Those claims become true only when the frozen
> gates in this ADR and plan 0043 are satisfied by published artifacts.

## Context

SALT V1 represents a group as a sum of ternary planes and allocates more planes
to sensitive groups:

```text
W_g ~= sum_p s[g,p] * T[g,p],       T[g,p,i] in {-1, 0, +1}
```

That representation is the right systems constraint. A dot product uses
add/subtract/skip operations inside each plane and applies one scale after the
plane accumulation. It does not require an arbitrary codebook lookup, lattice
decoder, or reconstructed floating-point weight matrix.

The V1 optimizer is no longer strong enough for a SOTA claim. It greedily fits
each residual plane with AbsMean, uses local reconstruction error in its
rate-distortion curve, approximates curvature with a diagonal signal, and
separates fitting from later healing. The field has since shown that four
effects matter at extreme compression:

1. **Initialization basin.** OA-EM changes AQLM's 3B-model pre-refinement
   perplexity from 352.39 to 16.82 at the same nominal rate and similar compute.
2. **Non-local curvature.** GuidedQuant and YAQA improve QTIP without changing
   its stored representation by replacing local activation MSE with end-loss or
   forward-KL-aware curvature.
3. **Joint discrete fitting and error feedback.** BPDQ, PTQTP, QuIP#, and QTIP
   jointly solve codes/scales and propagate the remaining error instead of
   independently fitting residuals.
4. **Proxy failure.** BCJR-QAT reports cases where layer reconstruction MSE
   improves while model perplexity regresses. Weight MSE is diagnostic, not the
   acceptance objective.

### Research position at the cutoff

Rates in the table are not assumed comparable unless the same evaluation and
storage boundary is stated. They establish the methods SALT V2 must reproduce
or beat under plan 0043's matched protocol.

| Method | Relevant result | Representation consequence |
|---|---|---|
| [AQLM](https://arxiv.org/abs/2401.06118) | Llama2 7B/13B/70B WikiText-2 perplexity 6.14/5.33/3.83 at about 2 bpw, versus fp 5.12/4.57/3.12 | Strong additive fitting, but arbitrary floating codewords violate native ternary execution. |
| [VPTQ](https://arxiv.org/abs/2409.17066) | Llama2 7B/13B/70B perplexity 6.13/5.32/3.93 at about 2.02 estimated bpw | Hessian initialization and GPTQ propagation transfer; centroid codebooks do not. |
| [PV-Tuning](https://arxiv.org/abs/2405.14852) | Llama2 7B/13B/70B perplexity 5.84/5.12/3.78 | Alternating continuous/discrete optimization is representation-agnostic and directly reusable. |
| [QuIP#](https://arxiv.org/abs/2402.04396) | Fine-tuned Llama2 7B/13B/70B perplexity 6.19/5.35/3.91 at 2-bit payload | RHT and BlockLDLQ transfer; E8 reconstruction does not. |
| [QTIP](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6de2e84b8da47bb2eb5e2ac96c63d2b0-Abstract-Conference.html) | Fine-tuned Llama2 perplexity 5.86/5.11/3.70; measured 188 tok/s for 7B and 23.5 tok/s for 70B on RTX 6000 Ada | This is the measured quality/runtime bar. Trellis-decoded multilevel values violate the ternary invariant. |
| [GuidedQuant](https://proceedings.mlr.press/v267/kim25d.html) | Pure-QTIP Llama2-7B perplexity 6.82 to 6.11; 70B 3.87 to 3.80 | End-loss curvature is reusable without changing the format. |
| [YAQA](https://arxiv.org/abs/2505.22988) | QTIP Llama3.1-8B perplexity 9.39 to 8.39; 70B 6.02 to 5.30 | Forward-KL real-Fisher and two-sided feedback are reusable. |
| [PTQTP](https://arxiv.org/abs/2509.16989) | Qwen3-32B perplexity 8.64 to 10.06 and Avg7 86.28 to 82.09 | Exact ternary-pair assignment and regularized scale solving transfer directly, but its dual-plane physical rate is not 1.58 bpw. |
| [BPDQ](https://arxiv.org/abs/2602.04163) | Qwen3-32B W3/G128 perplexity 9.97 versus fp 9.34; W2/G128 perplexity 12.97 versus 9.34 | Closest optimization peer: exact coordinate search, Hessian refit, ordering, propagation, and delta correction all transfer. Binary-plus-bias storage does not. |
| [LLVQ](https://arxiv.org/abs/2603.11021) | Llama2-7B at 2-bit payload: 6.83 PTQ and 5.48 after scale-only tuning, versus fp 5.11 | Spherical gain optimization transfers; Leech/Golay decoding does not. No measured end-to-end GPU throughput was reported. |
| [OA-EM](https://arxiv.org/abs/2604.08118) | Llama3.2-3B 2-bit pre-PV perplexity 352.39 to 16.82 and post-PV 12.66 to 11.53 | Output-aware multi-start initialization is mandatory evidence against greedy residual fitting. |
| [BCJR-QAT](https://arxiv.org/abs/2605.10655) | Local proxy improvements can regress full-model perplexity | End-to-end KL must be a gate, not an optional report. |
| [UniSVQ](https://arxiv.org/abs/2606.10520) / [LC-QAT](https://arxiv.org/abs/2606.10531) | Qwen3-32B 2-bit perplexity 7.61 to 9.26; Qwen3-8B LC-QAT 9.72 to 10.23 | Strong PTQ initialization plus smooth QAT transfers; quaternary codes and a dense affine decoder do not. |

No cited 2-bit or ternary PTQ result establishes zero model-level loss across
perplexity, reasoning, knowledge, and generation. Near-zero divergence is
therefore a target to falsify, not a premise.

## Decision

### Binding V2 profile, format, campaign, and facade choices

The following choices are binding for the first implementation. They narrow the
larger ablation space below and take precedence where an exploratory option in this
ADR would otherwise conflict.

- The only deployable core-weight synthesis is
  `W_hat[g] = sum_{p=1}^{P_g} s[g,p] T[g,p]`, with `P_g in 1..=3`,
  `s[g,p] >= 0`, and `T[g,p] in {-1,0,+1}`. There is no final offset, DC term,
  arbitrary codebook, INT4/FP residual, or dense projection. A training-only
  protected path is permitted only with a receipt proving it decays to exactly
  zero; use during more than 1% of scheduled steps is separately flagged.
- `CompactV1` is a successively refinable prefix capped at **2.25 physical
  core-projection bpw**. `NearLosslessV1` contains that exact prefix, is capped at
  **4.0 physical core-projection bpw**, and publishes only after strict BF16
  non-inferiority. The compact artifact is never independently re-fit from the
  near-lossless artifact.
- The semantic tensor is independent of its physical codec. `D2` is the mandatory
  aligned 2-bit reference and performance baseline; `B3` packs five radix-3 trits
  per byte; `S34` encodes an explicitly trained one-zero-per-four structure in five
  bits. One package uses one codec. B3 or S34 enters a claimed frontier only when it
  is Pareto-better in exact serialized bytes or resident bytes without losing the
  selected end-to-end wall-time gate.
- Scale groups default to **128 weights**. G64 and G256 are ablations; they replace
  G128 only if their preregistered held-out quality/physical-byte/runtime point is
  non-dominated. Plane presence is allocated at a regular 256-coefficient macrotile
  (or an encoding with no greater metadata) so two optional-plane bitmaps cost at
  most 0.0078125 bpw. Tensor-wide maximum-`P` padding is forbidden, and adaptive
  dispatch/index metadata is capped at 0.01 bpw.
- Every result reports logical bits, exact serialized core bytes, exact resident
  core bytes, all serialized and resident metadata, required runtime shadows,
  preserved tensors, exact whole-package bytes, and whole-model steady/peak
  resident bytes. A nominal rate cannot authorize a run or a claim.
- The frozen evidence pack starts with **512 sequences x 2,048 tokens**:
  50% C4, 25% OpenWebMath, and 25% StarCoderData, with immutable revisions,
  tokenizer/sample digests, boundaries, masks, and seed. Larger evidence is a
  separately identified rung, not a silent replacement.
- Weight recovery is proven at A16 first, then A8, then A4. A later activation rung
  cannot obscure whether additive weights passed. The final 20% of any refinement
  uses only hard exported trits and deployment-narrowed scales; cached teacher KL
  is enabled only after a CE-only plateau, then a PV-style exact discrete polish
  closes the run.
- Direct conversion runs on Qwen3-8B and Qwen3-32B are distinct from parameter
  growth. The growth track first widens Qwen3-8B's SwiGLU intermediate axes toward
  approximately 32B coefficients, then requires a separate whole-head/hidden-width
  transform before claiming architectural growth beyond that seam. The capacity
  endpoint stores at least **50B ternary coefficients** and is compared against the
  direct models by exact bytes and quality; it is not called a lossless 8B
  conversion.
- `tritium-salt` owns the high-level control plane. Stable entry points are
  `SaltV2::explain` and `SaltV2::reconcile` with
  `SaltProfile::{CompactV1, NearLosslessV1}`. Experimental
  `SaltPipeline::{start, advance, resume}` exposes the same ordered stages. Work is
  content-addressed, durable, resumable, and atomically published. Receipts bind
  provenance, stage digests, exact physical ledgers, hardware and GPU-hours,
  metrics/confidence intervals, and replay information. `QualityGateFailed` retains
  its evidence and can never reach Publish.

The initial implementation order is: preregistration; semantic format and facade;
joint fitter and exact-byte nested allocator; curvature and second-order feedback;
reconstruction/recovery; D2 then B3/S34 runtime candidates; A16 then A8 then A4;
Qwen3-8B; Qwen3-32B and growth; matched baselines and claim generation.

### 1. Preserve a zero-point-free additive ternary representation

SALT V2 keeps the only representation that the native kernel may execute:

```text
W_hat[g,i] = sum_{p=0}^{P_g-1} s[g,p] * T[g,p,i]
T[g,p,i] in {-1, 0, +1}
s[g,p] >= 0
P_g in {1, 2, 3}
```

There is deliberately **no group bias or zero point** `b_g`. A signed scale is
canonicalized by negating its trits. The fitting implementation may use fp32 or
fp64 accumulators, but deployment scales are fp16 unless a preregistered scale
ablation proves that another representation wins after its bytes and runtime
are counted.

The following are disallowed in a SALT V2 artifact:

- arbitrary floating-point codebooks or per-weight centroids;
- a dense affine reconstruction or neural decoder;
- lattice, trellis, or Golay decoding to multilevel floating weights;
- floating-point residual or outlier weights outside the ternary planes;
- per-group zero points or affine biases;
- dense dequantization as the measured inference path.

Norm parameters and other one-dimensional tensors may remain fp16. The
embedding and LM head policy must match each baseline and their bytes must be
included in whole-artifact accounting. A result may additionally report an
all-2D-ternary variant, but may not silently mix the two coverage policies.

### 2. Make physical bytes the optimization constraint

`log2(3)` is the information content of one trit, not the storage cost of an
artifact. SALT V2 records four distinct rates:

```text
logical_bpw   = sum_g |g| * P_g * log2(3) / N_quantized
matrix_bpw    = 8 * encoded_quantized_matrix_bytes / N_quantized
artifact_bpw  = 8 * complete_artifact_file_bytes / N_model_parameters
resident_bpw  = 8 * steady_state_model_resident_bytes / N_model_parameters
```

For uniform group size `G`, plane count `P`, fp16 scales, and no other
metadata, direct two-bit trit storage costs:

```text
R_direct(P,G) = 2P + 16P/G  bpw
```

Thus PTQTP-style two-plane storage is 4.25 bpw at G128, not 1.58 bpw.
For independently radix-3-packed planes:

```text
R_radix3(P,G) = P * ceil(G * log2(3)) / G + 16P/G  bpw
```

At G128, one plane is 1.71094 bpw and two planes are 3.42188 bpw before
headers and allocation maps. At G256, the corresponding rates are 1.64844 and
3.29688 bpw. Current TQ2_0 uses 66 bytes per 256 weights per plane, or 2.0625
bpw per plane including its fp16 scale.

For mixed `P_g`, the allocator must charge the exact encoded bytes for every
plane, its scale, plane-count or presence map, alignment, row descriptor, and
padding. The final report additionally counts container metadata, unquantized
tensors, tokenizer/configuration files when shipped in the model package, and
any runtime shadow representation. Logical bpw may be printed only beside the
physical figures.

SALT V2 supports three explicit codecs over the same semantic tensor:

- **`D2` / `direct-2bit`**: aligned two-bit trits, the mandatory correctness
  oracle and fast CUDA baseline;
- **`B3` / `radix3`**: five radix-3 trits per byte for the compact disk/capacity
  candidate, with decode overhead measured rather than assumed away;
- **`S34`**: 32 legal one-zero-per-four states in five bits, available only to
  tensors trained or recovered under that structural constraint.

D2 is always produced as the reference. A published package chooses exactly
one codec; B3 and S34 are admitted only by the Pareto rule above.

### 3. Replace greedy residual fitting with a joint output-aware solver

For each output block with input curvature `H`, the PTQ objective begins as:

```text
D_H(W, W_hat) = trace((W - W_hat) H (W - W_hat)^T)
```

The solver performs the following ordered operations:

1. **Optional signed RHT shaping.** Apply deterministic randomized Hadamard
   transforms to reduce incoherence. The seed is part of recipe identity. A
   transform ships only if its online activation cost is add/subtract based,
   included in runtime, and improves the measured Pareto frontier.
2. **End-loss curvature.** Cache the activation Hessian and an output-aware
   correction using GuidedQuant-style Fisher or YAQA-style forward-KL
   curvature. Diagonal Fisher remains a fallback and ablation, not the target
   method.
3. **OA-EM initialization.** Run at least the frozen number of deterministic
   restarts. Each coordinate E step jointly assigns all plane trits for one
   weight, conditional on the other coordinates, by exact enumeration of
   `3^P` states: 3, 9, or 27 candidates. This is an exact coordinate update,
   not a claim of a global discrete optimum. The M step solves all plane scales
   together under the Hessian metric.
4. **Conditioned scale solve.** Solve the non-negative weighted least-squares
   problem with a PTQTP-style condition-number-triggered ridge. Canonicalize
   scale signs into the trits. Reject NaN, singular, or non-monotone updates.
5. **Block feedback.** Quantize in a deterministic, group-aware order and
   propagate error with BlockLDLQ/GPTQ. After any scale refit, apply BPDQ-style
   delta correction so later columns see the actual, not stale, reconstruction.
6. **Output reconstruction.** Optimize block output error on cached
   activations. Raw weight Frobenius error remains secondary telemetry.

Every accepted coordinate or scale update must be non-increasing under the
objective active for that stage. A full-model perplexity/KL check decides
between stage checkpoints; a lower local proxy may not replace a better
full-model checkpoint.

### 4. Allocate planes by measured marginal loss per physical byte

SALT V1's allocator uses `log2(3)` as the cost of every added trit plane. SALT
V2 instead solves the discrete constrained problem:

```text
minimize    L_hat({P_g, T_g, s_g})
subject to  encoded_bytes({P_g}, packing, alignment, metadata) <= B
            P_g in {1, 2, 3}
```

The initial marginal score is the end-loss-curvature-weighted decrease from a
fully refitted `P`-plane solution divided by the exact added encoded bytes.
Candidate additions are periodically re-evaluated because jointly refitting a
group changes its later marginal gains. Runtime-aware selection reports a
second frontier constrained by measured kernel latency; it never substitutes
an estimated operation count for the byte-constrained quality frontier.

Group sizes `{64, 128, 256}` and plane caps `{2, 3}` are a preregistered pilot
grid. They are selected once on SmolLM2-1.7B, then frozen before the Qwen3-8B
primary experiment. The 8B or 32B test sets may not be used to choose them.

### 5. Separate PTQ, scale-only, and short-refinement claims

SALT V2 publishes three distinct tracks:

- **PTQ:** calibration, curvature, discrete solve, and packing only. No model
  parameter receives a gradient update.
- **PTQ+scale:** only per-plane scales are optimized against teacher outputs;
  trits and plane allocation remain fixed.
- **Short refine:** PV-style alternating hard-trit and continuous-scale updates,
  preceded by a smooth LC-style warmup and driven by teacher forward KL. The
  final artifact contains hard trits only.

Full-model QAT is not part of the first SOTA claim. It may start only if the
short-refinement rung passes its stop gate and the expected improvement per
GPU-hour justifies expansion. Results from the three tracks may not be merged
into one unlabeled number.

### 6. Co-design the kernel without weakening the format invariant

For each group and plane the measured inference kernel computes:

```text
a[g,p] = sum_i T[g,p,i] * x[g,i]       // add, subtract, or skip
y      = sum_{g,p} s[g,p] * a[g,p]     // one scale application per accumulator
```

The direct path must consume packed trits without materializing dense fp16
weights. Radix-3 unpack, rotation, scale application, scheduling of mixed plane
counts, and any sparse residual representation are included in end-to-end
timing. Dense residual planes are preferred unless a measured architecture-
specific density threshold proves sparse storage faster as well as smaller.

Exact and fast numerical policies follow ADR 0027. Exact is the conversion and
parity oracle. Fast may use different reduction order but must meet the frozen
logit, token-parity, and task gates.

## Proposed API contracts

The stable caller boundary is intentionally small. The same staged pipeline
implements both calls:

```rust
pub enum SaltProfile {
    CompactV1,
    NearLosslessV1,
}

impl SaltV2 {
    pub fn explain(spec: &SaltSpec) -> Result<SaltExplanation, SaltError>;
    pub fn reconcile(
        spec: &SaltSpec,
        work_root: impl AsRef<Path>,
        driver: &mut impl SaltDriver,
    ) -> Result<SaltReceipt, SaltError>;
}

impl SaltPipeline {
    pub fn start(spec: &SaltSpec, work_root: impl AsRef<Path>) -> Result<Self, SaltError>;
    pub fn advance(&mut self, driver: &mut impl SaltDriver)
        -> Result<AdvanceOutcome, SaltError>;
    pub fn resume(spec: &SaltSpec, work_root: impl AsRef<Path>)
        -> Result<Self, SaltError>;
}
```

The names below are the experimental model-fitting contract behind the driver.
They may be split across files, but changing their semantic boundary requires an
ADR update.

```rust
pub enum SaltV2Packing {
    D2,
    B3,
    S34,
}

pub enum SaltV2Curvature {
    DiagonalFisher,
    InputHessian,
    GuidedFisher,
    ForwardKlKronecker,
}

pub enum SaltV2Refinement {
    None,
    ScaleOnly { max_tokens: u64 },
    PvKl { warmup_tokens: u64, hard_tokens: u64 },
}

pub struct PhysicalRateTarget {
    pub max_matrix_bytes: u64,
    pub max_artifact_bytes: u64,
    pub max_resident_bytes: Option<u64>,
}

pub struct SaltV2Config {
    pub group_size: usize,
    pub min_planes: usize,
    pub max_planes: usize,
    pub packing: SaltV2Packing,
    pub curvature: SaltV2Curvature,
    pub transform_seed: Option<u64>,
    pub em_restarts: usize,
    pub coordinate_sweeps: usize,
    pub ridge_condition_limit: f64,
    pub rate: PhysicalRateTarget,
    pub refinement: SaltV2Refinement,
}

pub struct SaltV2TensorFitInput<'a> {
    pub name: &'a str,
    pub weights: &'a [f32],
    pub rows: usize,
    pub cols: usize,
    pub curvature: &'a CurvatureArtifact,
}

pub struct SaltV2ModelFitInput<'a> {
    pub tensors: &'a [SaltV2TensorFitInput<'a>],
    pub activations: &'a ActivationCache,
    pub source_model_id: ModelId,
}

pub struct SaltV2ModelFitResult {
    pub tensors: Vec<SaltV2Tensor>,
    pub metrics: SaltV2ModelFitMetrics,
    pub receipt: SaltV2ModelFitReceipt,
}

pub fn fit_salt_v2_model(
    input: SaltV2ModelFitInput<'_>,
    config: &SaltV2Config,
) -> Result<SaltV2ModelFitResult, SaltV2Error>;
```

`SaltV2Tensor` contains only dimensions, group geometry, plane counts, non-negative
scales, hard trits, transform identity, and encoding metadata. It has no zero-
point field. The model-level fit is the rate-allocation boundary: it may call a
tensor solver internally, but only the model fit can enforce matrix and artifact
byte ceilings across tensors. `SaltV2ModelFitMetrics` reports logical, matrix,
artifact-estimate, and resident-estimate rates separately, plus Frobenius,
Hessian, block-output, and teacher-KL metrics. `SaltV2ModelFitReceipt` binds
source tensor digests, calibration ID, curvature digests, recipe ID, seeds,
solver version, and output digest.

`tritium-format` adds a versioned SALT V2 tensor/package manifest with exact
byte counters and D2/B3/S34 payloads. `tritium-cuda` adds upload and matmul
entry points that consume those payloads directly. `tritium-salt` reuses the
existing `ConversionRun` state machine and interoperates with
`CalibrationProvenance`, `RecipeProvenance`, and `CampaignLedger`; its receipt
adds whole-workflow evidence without weakening those lower-level identities.

## Preregistered evidence and claim gates

### Correctness and accounting

- Every decoded trit is in `{-1,0,+1}`; scales are finite and non-negative.
- CPU decode, exact CUDA decode, dense plane reconstruction, and serialized
  round-trip agree within the frozen fp16-scale tolerance.
- The encoder's byte counters equal actual file lengths and steady-state device
  allocations. Tests cover short final groups, padding, mixed plane counts, and
  both packings.
- Re-running a recipe with identical inputs produces byte-identical artifacts
  and receipts.
- No result is labeled by logical bpw alone.

### Quality

The development model is SmolLM2-1.7B. The primary 8–9B rung is Qwen3-8B, and
the confirmation rung is Qwen3-32B. Llama2-7B is a literature bridge for QTIP,
LLVQ, AQLM, and VPTQ; it cannot replace the Qwen primary result.

The preregistered near-zero-divergence gate at the selected approximately
3.3–3.6 matrix-bpw point is all of:

- held-out perplexity increase no more than 1.0% relative to bf16;
- mean accuracy decrease no more than 0.5 percentage points on the frozen task
  suite, with no task decreasing by more than 1.0 point;
- teacher forward-KL and top-token disagreement below thresholds frozen by the
  1.7B pilot, before the 8B run;
- all comparisons use identical tokenizer, context, prompts, and scoring code.

Failure of any component means “near-zero divergence” was not achieved. A
larger ternary model beating a smaller fp model is reported separately as
quality-per-byte parameter scaling, not as lossless conversion.

### SOTA and Pareto claims

At each claimed physical budget, SALT V2 must be compared locally against the
strongest reproducible applicable baselines: SALT V1, PTQTP, BPDQ, QTIP,
LLVQ, AQLM or VPTQ, UniSVQ, and a modern W4 baseline. Published numbers may be
shown for context but cannot establish the claim when protocol or storage
boundaries differ.

A point is called **additive-ternary SOTA** only if it wins the frozen primary
quality aggregate at no greater artifact bpw than every reproduced additive or
ternary baseline. It is called **global low-bit Pareto-SOTA** only if no
reproduced low-bit method is both better in quality and no larger, and the SALT
V2 direct kernel is no slower at the claimed serving regime. Statistical ties
are ties, not wins. The primary refined experiment uses three seeds and reports
confidence intervals; deterministic PTQ reports repeated byte and runtime
reproducibility.

Runtime claims require end-to-end prefill, batch-1 and batched decode, peak
resident memory, and energy when available on the same GPU. QTIP's reported
188 tok/s for 7B and 23.5 tok/s for 70B on RTX 6000 Ada is a required external
sanity bar, not a substitute for same-box measurement.

## Alternatives considered

### Keep SALT V1 and add more healing

Rejected. Greedy AbsMean initialization and logical-rate allocation leave too
much quality to recover with expensive training and obscure whether the stored
representation or the optimizer produced the gain.

### Adopt AQLM, QTIP, LLVQ, or UniSVQ storage

Rejected for the SALT artifact. These are strong comparison targets, but their
free codebooks, multilevel lattice/trellis values, or dense decoder violate the
native add/subtract/skip contract. Their optimization mechanisms are adopted
where representation-independent.

### Add a group bias or zero point as in affine binary decomposition

Rejected. It creates an additional dense input-group reduction and scale,
weakens the zero-centered ternary invariant, and makes the format less directly
comparable to existing Tritium kernels. BPDQ remains a baseline rather than a
reason to add `b_g`.

### Call two planes “1.58-bit” because each plane is ternary

Rejected as incorrect physical accounting. Two independent trits have a
3.1699-bit information floor and occupy 4 bits with direct two-bit storage,
before scales.

### Begin with a 32B full-QAT campaign

Rejected. It spends the largest budget before proving the solver, accounting,
and quality hypothesis. The 1.7B and 8B stop gates must pass first.

## Consequences and risks

**Positive:** the representation stays native to Tritium's fastest execution
path; the optimizer incorporates the strongest representation-independent PTQ
and short-refinement ideas; physical bytes and runtime become first-class
constraints; a negative result remains scientifically useful and bounded.

**Negative:** joint Hessian fitting, multi-start EM, and block feedback are more
complex than residual AbsMean; radix-3 packing may save bytes but lose runtime;
end-loss curvature creates large activation/Hessian caches; a 1% perplexity
gate may be unattainable below approximately 3.5 bpw; the best global 2-bit
methods may remain ahead in quality.

**Research risk:** combining individually strong methods does not guarantee an
additive benefit. Plan 0043 therefore ablates each major mechanism and stops
before costly scale-up if it does not improve held-out model metrics.

## Definition of done for this ADR

This ADR may move from Proposed only after plan 0043 produces:

- a byte-exact SALT V2 format and direct CUDA execution path;
- deterministic 1.7B and 8B evidence for each ablation and physical-rate point;
- reproduced matched-rate baselines and complete provenance;
- an 8B near-zero-divergence pass or an explicit negative result;
- a 32B confirmation only if its preceding stop gates pass;
- a claim statement whose scope is mechanically derived from the gates, with
  no manual promotion from “not achieved” to “SOTA.”

## Amendment 2026-07-15 — prefix pricing and physical B3 rate

This amendment resolves two contradictions found while implementing the
byte-exact reference package. It supersedes the conflicting rate and refit
wording above; the representation and campaign gates are unchanged.

First, `R_radix3` is an information-rate lower bound, not the physical rate of
the binding B3 codec. B3 stores five trits per byte. At G128, one full plane
therefore uses 26 payload bytes plus one 2-byte scale, or exactly 1.75 bpw before
allocation-map, index, header, and alignment bytes. A full 256-coefficient tile
uses 52 payload bytes plus two scales and has the same 1.75-bpw plane rate. B3
cannot by itself establish a greater-than-10x resident-memory claim versus
FP16. S34 uses 40 payload bytes plus four scale bytes per full G128 tile, or
1.375 bpw before map/index overhead, and can cross that threshold only if its
constrained quality and direct-runtime gates pass.

Second, the exact Compact-prefix invariant takes precedence over independent
refits at each plane count. Each group has one deterministic jointly fitted
`Pmax` master solution. Its planes are ordered by the complete prefix-loss tuple,
and `D_g(P)` is the measured loss of the first `P` planes of that same master.
Compact and NearLossless allocate over those exact prefix curves, with the
NearLossless counts bounded below by the Compact counts. Updating a master fit
invalidates and recomputes the complete curve and both allocations. No
independently refitted `P` candidate may be described as a byte prefix.

The first package/runtime reference remains G128-only. G64 or G256 promotion
requires an explicit versioned scale-geometry field and parity evidence before
either geometry can be serialized as SALT V2.

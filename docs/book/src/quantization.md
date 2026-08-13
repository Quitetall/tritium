# Quantization (SALT)

`tritium-quantize` implements **SALT** — *Sensitivity-Allocated Layered Ternary*
quantization. SALT spends extra capacity **only where the model is sensitive**,
along a single accuracy↔size knob, while keeping inference multiply-free. It is
designed in ADR 0001 (see the [research repository](https://github.com/Quitetall/tritium-research)) and scheduled by the
v0.40 quantization ADR (see the [research repository](https://github.com/Quitetall/tritium-research)).

## Why not flat ternary

Flat ternary (every weight crushed to `{-1, 0, +1}` with one AbsMean scale per
tensor/channel, à la BitNet b1.58) is maximally cheap but loses accuracy on
weight groups that carry high information. SALT keeps the multiply-free kernel
but adds capacity selectively. Flat ternary is exactly SALT's `T = 1` special
case, not a rival.

## The pipeline

SALT operates per weight group `g` (granularity = a kernel tile: per output
channel or per 128-element block, so compute stays regular):

> **Implemented today vs planned.** Steps **1, 3, and 4** are implemented in
> `tritium-quantize` and drive the `tritium quantize` CLI. Steps **2** (mode
> codebook), **5** (sparse-plane *application* in the quantizer — the sparse
> storage form exists in `tritium-format`, but the quantizer currently writes
> dense planes), and **6** (STE heal — the offline quantize path has no
> `tritium-train` dependency, so no automatic heal runs there) are scheduled but
> not yet wired into the offline pipeline. See
> ADR 0006 (see the [research repository](https://github.com/Quitetall/tritium-research)).

1. **Residual ternary expansion.** Approximate the group as a sum of ternary
   planes, each fitting the previous residual:
   `W_g ≈ Σ_{p=1..T_g} s_{g,p} · t_{g,p}` with `t ∈ {-1, 0, +1}`. `T` planes ≈
   `1.585·T` effective bits, and inference cost is `T` multiply-free passes —
   the existing add/sub/skip kernel, looped.
2. **Mode codebook.** Replace the single per-plane `absmean` with a small
   non-uniform scale set from k-means/GMM over the group's residual magnitudes.
   Sharpens each plane's fit; does not change the budget.
3. **Sensitivity rank.** Per-group sensitivity `H_g` from the Hessian
   diagonal / diagonal Fisher — reusing the Hessian GPTQ already computes, so the
   signal is free.
4. **Plane allocation — rate-distortion water-filling.** Minimize
   `Σ_g H_g · err_g(T_g)` subject to a bit budget, greedily assigning the next
   plane to the group with the largest marginal loss-drop-per-bit. Low-sensitivity
   groups settle at `T = 1`; irrelevant tiles at `T = 0` (skipped).
5. **Prune to minimal.** Residual planes (`p ≥ 2`) are mostly zeros, so store them
   **sparse** (nonzeros only) and magnitude-prune.
6. **Heal.** A short STE fine-tune (via `tritium-train`) recovers residual loss —
   see [Training](./training.md).

## The single knob

The whole pipeline is driven by one knob — **average bits-per-weight** — which
trades accuracy against size along a smooth ~1.58–3 bpw curve. The CLI exposes
it directly:

```sh
tritium quantize --input model.safetensors --output model.tslb --bpw 2.0
```

`--bpw 1.585` is all-base ternary (the `T = 1` flat case); higher budgets buy
extra residual planes on the most sensitive tiles (up to `~4.75` bpw at `T = 3`).
You can preview the bpw/error tradeoff on a raw fp32 matrix without committing to
a full quantize via `tritium report salt`.

PyTorch researchers can use the same native water-filling allocator with
measured group curves:

```python
from tritium.torch import allocate_planes

allocation = allocate_planes(
    group_sizes=[128, 128],
    sensitivities=[1.0, 8.0],
    error_curves=[[10.0, 6.0, 4.0, 3.0], [10.0, 2.0, 0.5, 0.0]],
    target_bpw=3.0,
)
print(allocation.plane_counts, allocation.achieved_bpw)
```

Curves are evidence inputs, not guessed metrics. Native deterministic
tie-breaking returns counts within budget; malformed, nonfinite, or undersized
curves fail before allocation.

Generic live-module PTQ accepts same measured knob. Set
`TernaryConfig.ptq(..., target_bpw=2.0)` and `convert(...)` will measure
curvature-weighted error for one, two, and three additive planes per selected
weight, then run native water-filling. Result exposes `achieved_bpw`, and its
strict artifact records each selected weight's plane count. This first generic
integration allocates at weight granularity; Stage-7 SALT model conversion
retains finer allocation-tile maps.

## Stage-7 token evidence

Collect the frozen source rows at the three immutable Hub revisions before
building the token pack:

```sh
python scripts/collect-stage7-sampled-rows.py \
  --output-dir ./stage7-sampled-rows
```

Install its frozen producer dependencies first:

```sh
python -m pip install --requirement scripts/requirements-stage7-collection.txt
```

The collector preflights access to
all three datasets before downloading any payload, verifies every downloaded
LFS SHA-256 and size through one retained descriptor, terminally rehashes that
descriptor, and uses each frozen partition seed to rank source locators into
four disjoint lanes. It requires at least one source row per eventual sequence
as well as a conservative UTF-8 byte floor, then atomically publishes
`sampled-rows.json`, its JSONL lanes, and a separate
`tritium.stage7-row-acquisition.v1` receipt. The download ceiling is
campaign-wide. Duplicate content is excluded
deterministically and disclosed in the receipt; insufficient unique content,
existing output, incomplete sources, changed shard identities, and unauthorised
StarCoderData access fail closed. No public or synthetic dataset may silently
replace the gated lane. Authenticate with `hf auth login` only after Hugging
Face grants access to `bigcode/starcoderdata`.

The row collector is acquisition evidence, not proof that every raw-text lane
contains enough tokenizer output. `build-stage7-evidence-pack` remains the
authority: it tokenizes the published rows with the pinned model tokenizer and
fails unless all four exact 512x2048 partitions can be emitted.

Build the frozen SmolLM recipe-selection token pack from a content-bound
`tritium.stage7-sampled-rows.v1` manifest:

```sh
tritium salt build-stage7-evidence-pack \
  --model-dir ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --sampled-rows ./stage7-sampled-rows/sampled-rows.json \
  --output-dir ./stage7-token-evidence
```

The input contains four ordered partitions and three dataset lanes per
partition. Every lane binds its exact Hub commit, config, optional `data_dir`,
split, source-row index, text-field name, raw-text digest, and file identity.
For StarCoderData, config `default`, `data_dir="python"`, and source field
`content` are distinct provenance values; the sampled JSONL still transports
that verified content under its normalized `text` key. The builder uses the
snapshot tokenizer, writes exactly 512 unique 2,048-token sequences per
partition, and publishes `stage7.u32le` before its canonical `manifest.json`.
The qualifier rejects any payload other than the fixed 16 MiB geometry before
opening it and rechecks bounded bytes through one descriptor. The builder
consumes already selected rows; it does not silently download or resample
datasets. StarCoderData remains gated and requires authorized Hub access during
the separate row-collection step.

Before calibration or evaluation, reopen a bounded sequence window against both
the model tokenizer and the pack identity frozen by campaign provenance:

```sh
tritium salt inspect-stage7-evidence-pack \
  --model-dir ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --manifest ./stage7-token-evidence/manifest.json \
  --expected-pack-id sha256:<campaign-frozen-pack-id> \
  --partition calibration \
  --start-sequence 0 \
  --sequence-count 128
```

This command uses the reusable `tritium-salt` seek-backed reader. Admission
recomputes the manifest and every sequence identity, verifies frozen dataset
geometry and source-row disjointness, rejects out-of-vocabulary tokens, checks
the complete payload digest, then rechecks the retained file handle after the
selected read. Its JSON receipt identifies the exact ordered token window; it
does not claim model execution or quality.

Run the frozen 135M execution seam from the installed wheel:

```python
from pathlib import Path

from tritium.torch import run_stage7_smollm2_smoke

snapshot = (
    Path.home()
    / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots"
    / "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
)

result = run_stage7_smollm2_smoke(
    "./stage7-campaign.json",
    snapshot,
    "./stage7-smoke",
    device="cuda",
)
print(result.model.package_id, result.model.mean_loss)
```

The driver strictly resumes capture, additive PTQ fitting, allocation, native
SALT V2 packaging, and causal evaluation. It admits only the campaign-frozen
model/token prefix and emits qualifier-compatible receipts. A completed smoke
proves workflow integrity and physical package production, not 1.7B recipe
quality or Stage-7 qualification. SmolLM matrices requiring G64 use explicit
SALT V2 package-version 2 scale geometry; G128-only packages remain canonical
version 1.

## Stage-7 full recipe freeze

The full 1.7B successive-halving campaign is driven by the source-bound
orchestrator:

```sh
python scripts/run-stage7-recipe-freeze.py \
  --campaign /evidence/stage7/campaign.json \
  --model-root /models/SmolLM2-1.7B \
  --smoke-model-root /models/SmolLM2-135M \
  --source-root . \
  --runner python /path/to/measure-one-recipe.py \
  --auxiliary-runner python /path/to/measure-baselines-and-refinements.py \
  --output /evidence/stage7/trace.json
```

Both runners receive one canonical JSON request on stdin and must emit one
strict JSON object on stdout. The measurement runner supplies every frozen
recipe row; the auxiliary runner supplies BF16/SALT V1 baselines and the five
refinement records. The orchestrator binds each request to clean `HEAD`, model,
campaign, runner argv, runner executable/script digests, and evidence paths,
caches only validated immutable rows, resumes completed work, and invokes the
existing qualifier before publishing the trace. Request schemas are versioned
when this identity contract changes, so caches from older runner code cannot be
silently replayed. It never synthesizes metrics. A structural smoke or a
missing real runner cannot satisfy Stage-7.

Before candidate work, each runner must answer a capability preflight request
(`tritium.stage7-capabilities-request.v2`) with a
`tritium.stage7-capabilities.v1` declaration bound to the exact request ID and
campaign source revision. The measurement runner must advertise all three stages, D2/B3/S34,
G64/G128/G256, two/three planes, both rotations, all three curvature modes, all
six solver variants, full artifacts, and physical reports. The auxiliary runner
advertises only baseline/refinement features. Missing or stale capability
declarations fail before any measurement cache is written.

Campaign templates are pre-evidence plans. When source code changes before a
run, rebind the template to clean `HEAD` with no-replace output:

```sh
python scripts/rebind-stage7-campaign.py \
  --template /evidence/stage7/campaign-template.json \
  --source-root . \
  --run-id stage7-smollm2-17b-real-$(git rev-parse --short HEAD) \
  --output /evidence/stage7/campaign.json
```

The rebinder changes only top-level `source_revision` and `run_id`; nested
stale revisions, dirty trees, malformed templates, and existing outputs fail
closed. It creates no measurements and does not qualify a recipe freeze.

## SALT V2 Qwen master campaigns

The legacy `tritium quantize` command above is not the Qwen3.6-27B SALT V2
campaign path. Admit the pinned source first to persist its content-bound
language/MTP/vision coverage proof:

```python
from tritium.salt import admit_qwen36_source

source = admit_qwen36_source(
    "/models/Qwen3.6-27B",
    revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    work_dir="./tritium-work",
)
print(source.source_model_id, source.additive_tensors, source.mtp_tensors)
```

For release evidence, the equivalent strict producer persists one canonical
JSON receipt and the SHA-256 of the durable proof:

```sh
python scripts/admit-qwen36-source.py \
  --model-dir /models/Qwen3.6-27B \
  --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --work-dir ./tritium-work \
  --output ./tritium-work/source-admission.json
```

Source admission performs no calibration, fitting, packaging, or quality
claim. `identity_status` remains candidate-only until official payload
authentication is independently registered. Advanced users with a fully
collected canonical `S2KF` evidence directory can then resume the rate-free
master stage directly from Python:

```python
from tritium.salt import reconcile_qwen36_ptq_masters

receipt = reconcile_qwen36_ptq_masters(
    "/models/Qwen3.6-27B",
    revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    work_dir="./tritium-work",
    evidence_dir="./curvature-evidence",
)
print(receipt.campaign_id, receipt.additive_tensors)
```

This boundary admits and seek-reads the source checkpoint in Rust, requires the
exact 506-record evidence namespace and one campaign-wide token stream, widens
only one matrix at a time, resumes valid content-addressed masters, and seals a
canonical structural receipt. It does **not** return a deployable model: profile
allocation, package assembly, evaluation, and export remain governed later
stages. The high-level `tritium.torch.quantize(...)` facade now composes
`prepare` → `calibrate` → `convert` for this Qwen source path. Live
`torch.nn.Module` PTQ uses activation calibration and returns a module-scoped
conversion result; it does not silently substitute Qwen S2KF evidence.

For Transformers-backed Qwen3.6 capture, use the strict component boundary:

```python
from tritium.torch import (
    Qwen36LanguageMtpOracle,
    capture_qwen36_components,
    resolve_qwen36_components,
)

components = resolve_qwen36_components(model)  # requires model.mtp
oracle = Qwen36LanguageMtpOracle(model)
receipt = capture_qwen36_components(
    model,
    data_factory,
    model_dir="/models/Qwen3.6-27B",
    declared_revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    work_dir="./tritium-work",
    evidence_dir="./curvature-evidence",
    curvature="input-hessian",
    activation_cache_digest=cache_digest,
    token_stream_digest=token_digest,
    damping=1e-4,
    execution_model=oracle,
)
```

`Qwen36LanguageMtpOracle` keeps Transformers on its ordinary forward path and
captures final language states with a local norm hook before executing the
attached MTP graph. This avoids the `output_hidden_states=True` capture path,
which is not safe with disk-offloaded Qwen checkpoints. Its output retains
`logits`/`loss` plus finite `mtp_hidden_states` and `mtp_logits` fields for
calibration and parity diagnostics.

Resolution requires canonical `model.language_model`, `lm_head`, and retained
`mtp` modules before native evidence mutation. A Transformers graph that drops
MTP tensors fails closed; language-only diagnostics require an explicit
`require_mtp=False` resolution and cannot enter flagship capture.

### Native output-aware group fitting

Captured S2KF records can be passed to the native joint solver without
materializing one dense Hessian per output row:

```python
from tritium.torch import fit_kronecker_group

fit = fit_kronecker_group(
    linear.weight.detach().cpu(),
    "./curvature-evidence/000123.s2kf",
    planes=3,
    scale_precision="f16",
    # Fit bounded output-row windows when converting large matrices.
    row_start=0,
    row_count=None,
)
projection = fit.projection
print(fit.record_digest, fit.objective)
```

The Rust solver validates S2KF checksum, PSD group geometry, output-row
curvature, deterministic restart settings, and canonical hard decode. The
Python boundary bounds file-backed evidence before reading it and supports
bounded output-row windows. The helper is a fitting primitive, not release
admission: source-model, token stream, calibration-cache, package, and quality
receipts remain mandatory.

## Hardware constraints (load-bearing)

- **Regular compute.** Plane counts are quantized to `{1, 2, 3}` and allocated at
  tile granularity, so every tile runs a fixed number of add-only passes. One
  dense base plane everywhere; extra planes only on selected tiles.
- **Sparse-vs-dense residual.** GPU sparse-matmul overhead pays only below ~10%
  nonzero density; above that, the plane stays dense with **whole-tile skip**.
  The threshold is measured per architecture, not assumed.
- **Storage.** `tritium-format` extends `TQ2_0` with a residual sidecar — base
  plane + optional sparse planes + per-plane scales. The on-disk bundle is the
  `.tslb` SALT bundle (or a GGUF container holding the SALT rows).

SALT is an **engineering** synthesis of established techniques — residual ternary
expansion (ABC-Net, AQLM), non-uniform mode scales (Deep Compression,
SqueezeLLM), sensitivity allocation (HAWQ, SqueezeLLM), and sparse residual
planes (SpQR) — chosen so every plane is still ternary and runs on the existing
add/sub/skip kernel. There is no new hardware path. See
ADR 0001 (see the [research repository](https://github.com/Quitetall/tritium-research)) for the full derivation and the
prior-art references.

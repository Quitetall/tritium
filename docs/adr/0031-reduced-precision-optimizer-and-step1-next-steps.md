# ADR 0031 — Reduced-precision optimizer + Step-1: next steps

Status: Proposed (roadmap). Follows ADR 0029 (training throughput; Lever 5) and the Step-1 baseline.
Date: 2026-07-19.

## Context — where we are

The training-throughput campaign (ADR 0029) closed its Lever-5 numerics: the resident CUDA distillation
trainer (`DeviceTrainer`) now has two opt-in reduced-precision optimizer modes, both **validated
end-to-end on the real SmolLM2-135M** and both stable:

| Mode | Flag | 135M held-out recovery | Notes |
|---|---|---|---|
| f32 (default) | — | **960×** | byte-identical baseline |
| int8 moments | `MomentPrecision::Int8` | **939×** | block-wise int8 `m`/√`v`; was diverging, now fixed |
| bf16 master | `MasterPrecision::Bf16` | **907×** | master confined to bf16 grid + stochastic rounding |
| tf32 GEMMs (ref) | `TRITIUM_DISTILL_TF32` | 920× | prior lever, for comparison |

Both reduced-precision modes cost a **~2–5% recovery tax** (same class as tf32) for cutting
optimizer-state / master VRAM. All are opt-in; the f32 path is byte-identical. The int8 divergence was
root-caused (a block's `m`-absmax and √`v`-absmax can be dominated by *different* coordinates — an
oscillating outlier dominates the √`v` grid but not the `m` grid, so a steady neighbour's `v` rounds to
code 0 while its `m` survives, collapsing the denominator) and fixed by flooring a nonzero √`v` at int8
code 1 (commit `6fea28c`).

**What is deliberately NOT done yet:** the actual `u16` bf16-master *storage* swap. Today
`MasterPrecision::Bf16` confines the f32 master's *values* to the bf16 grid — numerically identical to a
real bf16 master, so it validates recovery — but the master is still stored as f32, so no VRAM is saved.
The VRAM realization is a separate, mechanical change deferred here because it has zero benefit at 135M
and non-trivial risk (see below).

The Step-1 experiment established the honest baseline: on the committed 8k-token Alice corpus, SALT
distillation drives ternary-135M's held-out ppl from 2205 down to a **~11–13× fp plateau** (224.8 ppl,
recovery-vs-PTQ 9453×) — a **data-bound** ceiling, not a method limit. Closing to fp needs a bigger,
recognized corpus.

## Decision — the next steps, in priority order

### 1. Standard-corpus Step-1 run (f32) — the field-comparable number (highest value)
**Corpus prep DONE (2026-07-24, commit 2e91147, compute-free).** `tools/gen_corpus.py` now reads
`.parquet` (WikiText/C4 `text` column) and takes a separate `--eval-file` for the held-out split; a
WikiText-2-raw corpus is generated locally — 500k-token train pool + 8192 held-out from the disjoint
test split, all ids in-vocab (61× the committed 8k Alice fixture). **What remains is the GPU run:** point
`TRITIUM_CORPUS` at that JSON and run the 135M distill-to-convergence on the **f32 path** with
`TRITIUM_DISTILL_CURVE` checkpointing; report held-out ppl vs fp (19.73) and vs published SmolLM2-135M.
Needs none of the reduced-precision work (int8/bf16 VRAM wins are irrelevant at 135M). Consider LR warmup +
a deterministic teacher cache (`TRITIUM_TEACHER_CACHE`) for a cleaner curve — the constant-LR/online-
teacher recipe is numerically chaotic early (the f32 control itself swings to ppl 4.4e9 by step 3).

### 2. `u16` bf16-master storage swap — VRAM realization (do when it matters at scale)
Make the persistent master actually bf16 to halve its VRAM. This is the central-path ripple deferred from
Lever 5:
- `ResidentTrainParam.master`: `DeviceTensor` (f32) → a bf16 store (`CudaSlice<u16>`), half size.
- Add one shared f32 dequant scratch (largest-leaf) on `DeviceTrainer`; before each SALT reconstruction,
  dequant the bf16 master → scratch, then run the existing `salt_quantize_forward_dev` on the scratch
  (keeps the reconstruction kernel unchanged). The `adamw_step_bf16_master` kernel (commit `0c0b8d4`,
  already gated bit-identical to `tritium_train::bf16`) updates the bf16 master directly.
- Update the consumers of `param.master.buf`: `prepare_quantized`, `pack_training_salt`/`repack`,
  `download_master`, `resident_state` (Parameter plane), and the DCP checkpoint paths — ~10 sites. Guard
  or extend the checkpoint/offload paths for the bf16 store.
- Gate: recovery must match the already-validated grid mode (numerically identical), plus a VRAM-usage
  assertion (`ResidentTrainerStats` master elements halved).
**Why deferred:** invisible at 135M (model+optimizer tiny), real risk on the shared, actively-edited
reconstruction path, and the recovery impact is *already* validated by the grid mode. Best done at
1.7B/32B where it can be validated against actual OOM pressure — and on a clean base (the parallel
inference session currently edits `backend.rs`/`train.rs`/`mod.rs`; stage with `git add -p`).

### 3. Combined int8 moments + bf16 master, at 1.7B/32B
Both modes are independent today (bf16 master runs with f32 moments). At scale, combine them for the full
~4× optimizer-state + 2× master VRAM cut, and re-measure the stacked recovery tax (expect ~5–8%). This is
gated on (2) and on GPU headroom (the 1.7B/32B masters + Adam state are the whole point of Lever 5).

### 4. int8/bf16 checkpoint serialization
The reduced-precision resident state is training-only today; `write_state`/`read_state` cover f32 (and the
CPU `Int8AdamW` state round-trips), but the *device* trainer's int8/bf16 DCP save/reload path is not
wired. Needed before any long multi-session reduced-precision campaign.

### 5. Remaining throughput levers (reference; ADR 0029)
- **Lever 6 — launch-overhead reduction (CUDA graphs / fusion):** the biggest remaining *full-step* lever.
  65×-per-GEMM ≠ 65×-per-step because the step is launch/glue-bound; Lever 6 is what actually moves the
  full-step wall-clock. Highest-leverage throughput work after Step-1.
- **Lever 3 — top-k sparse KD loss + teacher cache: CPU half DONE (2026-07-24, compute-free).** The loss
  op `topk_kd_forward`/`vjp` + `Tape::topk_kd` (commit 177f734, gradchecked + proven identical to dense
  softmax-xent) and the `TTPK` top-k teacher-cache format `topk_teacher_cache.rs` (commit 601c38a, a
  vocab/(2k)=384× byte shrink at k=64, round-trip tested). Both reviewed clean. The lm-head gradient stays
  dense (softmax normalizer) — the win is teacher-cache I/O, not backward FLOPs. **Remaining (GPU):** the
  producer that writes `TTPK` from the teacher forward's top-k, and the nn-side reader that feeds the pairs
  into `topk_kd` in the distill loop.
- **Lever 7 — 2:4 structured sparsity (ADR 0024):** payoff gated on Lever 6 + scale.

## Consequences
- The reduced-precision optimizer is *usable now* (opt-in, validated) but only *pays off* at scale; the
  135M work should run f32.
- Step-1's field-comparable number is unblocked and independent of all the above — it is the recommended
  immediate next action.
- The u16 storage swap is the only non-mechanical debt from Lever 5; it is well-scoped here and low-risk to
  pick up later.

## Verification (definition of done per step)
- Step-1 corpus run: held-out ppl curve on a recognized corpus, reported vs fp and vs published SmolLM2.
- Storage swap: recovery == grid mode within noise, `ResidentTrainerStats` master bytes halved, all
  existing `DeviceTrainer` tests green (f32 path byte-identical).
- Combined mode: stacked recovery measured at 1.7B/32B; no divergence.
- Checkpoint: reduced-precision resident state round-trips through DCP save/reload bit-exact.

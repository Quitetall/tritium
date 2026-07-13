# ADR 0027 — Device-resident training: resident optimizer, ternary compute, and scale to 32B

Status: **PRODUCTION SOFTWARE PATHS LANDED / COMPUTE AND SCALE ACCEPTANCE OPEN** (2026-07-13)

- **Deciders:** Brian Lam
- **Relates:** completes the plan-0043 GPU training path
  (`~/.claude/plans/vast-wibbling-gadget.md`); builds directly on the shipped
  device-resident engine (P2.1–P2.5d, commits `eb756b1`→`27c1669`). Consumes the
  SALT-aware training scheme of [ADR 0016](./0016-ternary-training-methods.md) and
  PEPPER execution of [ADR 0017](./0017-pepper-ternary-execution.md); feeds the
  v1.x SALT-distillation capstone ([ADR 0020](./0020-v1x-salt-distillation-capstone.md)).
  Inherits the bit-exactness reduction discipline of
  [ADR 0018](./0018-canonical-tree-reduction-order.md) /
  [ADR 0023](./0023-relaxed-reduction-tier.md).

> **This ADR is an onboarding contract.** It specifies *which features to build and
> how they fit the existing architecture* — not the line-level implementation. Each
> track below is independently shippable, has a stated interface, and a falsifiable
> gate. Build them in order (A→F); each depends only on what precedes it. Every new
> CUDA kernel is gated **device == CPU** against its `tritium-train` CPU oracle
> (`--fmad=false`, sequential reduction — the `train_grad.cu` contract), the same way
> every P2 kernel already is.

---

## Context: what exists, and the two things stopping the win from scaling

Phase 2 built a **device-resident autograd engine** that trains a real transformer
entirely in VRAM. The state of the world you are inheriting:

**The engine (`crates/tritium-cuda/`):**
- `kernels/train_grad.cu` — every training kernel, compiled `--fmad=false`: matmul
  fwd + `grad_a`/`grad_w`, `rmsnorm_train_*`, `silu_*`, `ew_mul_*`, `ew_add_*`,
  `accumulate` (`dst+=src`), `softmax_*`, `causal_mask_*`, `rope_apply`,
  `slice_cols_forward`, `copy_into_cols`, `transpose_forward`, `embed_gather_*`,
  `softmax_xent_backward`, `scale_const`.
- `src/cuda/backend.rs` — the resident launch methods: `dev_upload` / `dev_upload_i32`
  / `dev_alloc_zeros` / `dev_download`, and one `*_dev` method per kernel, each taking
  `&CudaSlice<f32>` (no per-op host copy) and guarding buffer sizes with
  `CudaSlice::len()`.
- `src/train.rs` — **`pub struct DeviceTape<'a>`**: the GPU analogue of
  `tritium_train::Tape`. Forward methods (`leaf`/`embed`/`rmsnorm`/`matmul`/`silu`/
  `mul`/`add`/`attention` + the attention-internal `rope`/`slice_cols`/`scale_const`/
  `causal_mask`/`softmax`/`transpose`/`concat`) append a device buffer to `vals` and
  record a `DevOp`; `backward` replays the `DevOp`s in reverse, accumulating grads
  on-device (reverse-topological: `out.id > input.id`, so accumulation is correct for
  residuals + the tied embedding). `xent_backward(logits, target, rows, cols, want)`
  seeds the softmax-xent grad, runs the whole backward, and **downloads only the
  `want` grad ids to host** (hides `CudaSlice`).

**The validation (all gated, all green):** per-op device==CPU gradchecks; the full
transformer block (`device_tape_transformer_block_matches_cpu_tape`, rel 2.8e-7);
the whole real SmolLM2-135M (`device_tape_trains_smollm2_matching_cpu_tape`,
tritium-nn, logits+grads rel ~2e-6); and end-to-end device SALT distillation
(`salt_distillation_device_tape_recovers_heldout`, **recovers 960× vs PTQ ≈ the CPU
path's 957×**). `common::device_forward` (tritium-nn tests) is the shared device
model builder; it returns weight-leaf ids in `fp` order so grads map back to masters.

**The SALT-STE fact you will use everywhere:** `ste::salt_quantize_vjp` is the
**identity** (`grad_out.to_vec()` — see `crates/tritium-train/src/ops/ste.rs`). So the
gradient w.r.t. the quantized weight *is* the gradient w.r.t. the latent master. The
current distillation quantizes masters on the host each step (`salt_quantize_forward`)
and only hoists that (identity) STE out of the graph.

**Measured, honest speedup (warm, steady state; `27c1669` corrected an earlier bad
baseline):** device vs the CPU tape is **scale-dependent** — 1.5× (seq=8 tiny) → 2.4×
(SwiGLU block seq=64) → 3.3–4.4× (4-layer stack) → **~4.7× (real 30-layer fwd+bwd)**;
but the **full distillation step is only ~1.7×** (seq=32).

### Why the full step is only 1.7× — the two bottlenecks (this ADR's target)

1. **Per-step host round-trips.** Each distillation step: quantize all masters on the
   **CPU** (`salt_quantize_forward`), upload the quantized weights, run device fwd+bwd,
   **download every weight gradient** (135M floats ≈ 540 MB at 135M; proportionally
   worse at 1.7B/32B), then run **AdamW on the CPU**. The model fwd+bwd is ~4.7×, but
   the CPU quantize + CPU AdamW + the full-grad download dominate the step and don't
   speed up. **Fix: keep masters, Adam state, quantize, and the optimizer resident on
   the device (Track A).**
2. **`embed_gather_backward` is `O(vocab·seq·dim)`** — one thread per `gw` element
   (`vocab·dim` threads) each scanning `seq` tokens, chosen for bit-exact ascending-`t`
   accumulation with no atomics. Fine for the correctness gate; a real hotspot at
   `vocab = 49152`. **Fix: a segmented/atomic-free faster variant that stays bit-exact
   (Track B).**

Everything above `135M` also needs **bounded activation VRAM** (Track C) and, at 32B,
**the ternary compute win + offload** (Tracks D, E). Track F is the scale-up itself.

---

## Track A — Device-resident optimizer, masters, and SALT-quantize (the perf pass)

**Goal.** Eliminate the per-step host round-trips so the *full* distillation step
approaches the model's ~4.7× (and better at scale). After this track, a step touches
the host only for the token ids and (occasionally) an eval/checkpoint download.

**Features to build:**

1. **Device `salt_quantize` kernel** — the on-device forward of
   `ste::salt_quantize_forward(wf, rows, cols, t)`: `T` ternary planes, per-256-block
   AbsMean scales, `Ŵ = Σ_p s_p·t_p` written back to f32. Gate: **device == CPU
   `salt_quantize_forward`** within 1e-4 on random weights across `T ∈ {1,2,3}`. Add a
   `salt_quantize_dev(&d_master, &mut d_quantized, rows, cols, t)` backend method.
   (Its STE backward is identity — no backward kernel needed; the device grad w.r.t.
   `d_quantized` is already the master grad.)
2. **Device `adamw` kernel** — `AdamW` step on resident buffers, bit-close to
   `tritium_train::optim::AdamW::step`: fused `m`/`v` update, bias correction, decoupled
   weight decay, `θ -= lr·(...)`. One thread per parameter. Gate: **device == CPU
   `AdamW::step`** within 1e-5 over several steps on a synthetic parameter+grad stream.
   Add `adamw_step_dev(&mut d_master, &d_grad, &mut d_m, &mut d_v, step, lr, betas, eps, wd)`.
3. **Resident training-state ownership.** A thin `DeviceTrainer` (new; `src/train.rs`)
   that owns, per trainable weight, the resident master + Adam `m`/`v` (`dev_upload`
   once at construction), and drives the loop: per step → `salt_quantize_dev` (masters
   → resident quantized) → `DeviceTape` forward (quantized leaves) → resident xent
   backward → `adamw_step_dev` (resident grads → resident masters). No per-step
   download. Provide `download_master(i) -> Vec<f32>` for eval/checkpoint.

**Architecture / where it plugs in.** `DeviceTape` stays the graph engine; do **not**
fold the optimizer into it. `DeviceTrainer` is the outer loop that reuses `DeviceTape`
for fwd+bwd but keeps the weight/optimizer buffers resident across steps and calls the
two new kernels. The `DeviceTape::backward` already returns resident grad buffers
(`Vec<CudaSlice>`); add a resident-grad accessor (or a variant of `xent_backward` that
returns the resident `grads` without downloading) so `DeviceTrainer` reads grads on
device. Keep `xent_backward` (host-out) for the gates.

**Gate (the falsifiable exit).** `salt_distillation_device_trainer_recovers_heldout`:
same held-out recovery as `salt_distillation_device_tape_recovers_heldout` (≥ the
current 960× vs PTQ, within run-to-run noise), and **mean step time strictly below the
Track-0 device-distillation step** (the ~1123ms it prints today at seq=32), with the
gap widening at larger seq. Report the new step ms + the fraction now on-device.

---

## Track B — Faster `embed_gather_backward` (bit-exact)

**Goal.** Remove the `O(vocab·seq·dim)` scan without breaking bit-exactness.

**Feature.** A tiled/segmented backward that scatters `gy[t,:]` into `gw[tokens[t],:]`
in **ascending `t`** order (the invariant that makes it bit-exact vs
`ops::embed::gather_vjp`). Options for the other agent to weigh: (a) sort token ids →
segmented reduction per unique vocab row (ascending original-`t` within a segment);
(b) one block per token accumulating into `gw` rows with a **deterministic** ordering
(not raw `atomicAdd`, which reorders and breaks bit-exactness). Whichever: the gate is
`resident_attention_ops_match_cpu`'s gather-backward assertion staying **`== 0.0`**
(bit-exact), plus a microbench showing a large speedup at `vocab=49152, dim=576`.

**Architecture.** Drop-in replacement behind `embed_gather_backward_dev` — no
DeviceTape change. Keep the current kernel as the reference oracle in the gate.

---

## Track C — Gradient checkpointing (bounded activation VRAM)

**Goal.** Today `DeviceTape::backward` holds a grad buffer for **every value id** (all
activations across all layers) — `O(model + all activations)` VRAM. That fits 135M on a
4090 but not 1.7B, and never 32B. Bound it.

**Feature.** Per-block (or per-N-ops) activation checkpointing: keep only block
boundaries resident in the forward; **recompute** each block's internal activations
during its backward, then free them. The `DevOp` tape already records enough to replay
a block's forward — add a "checkpoint boundary" marker and a recompute path in
`backward`. Target peak activation memory `O(√depth)` (standard checkpointing).

**Architecture.** Extend `DeviceTape` with checkpoint boundaries (e.g. `DeviceTape::
checkpoint()` between transformer blocks in `device_forward`). `backward` recomputes a
checkpointed segment's forward before replaying its vjps. **Gate:** grads identical
(rel < 1e-4) to non-checkpointed `DeviceTape` on the transformer-block test; measured
peak VRAM drop; the 1.7B model fits on the 4090 with checkpointing on.

---

## Track D — Fused multiply-free SALT GEMM on the device (the ternary compute win)

**Goal.** The Phase-3 lever from plan 0043 + [ADR 0017](./0017-pepper-ternary-execution.md):
compute `Ŵ·x = Σ_p s_p·(t_p·x)` **directly from the ternary planes**, multiply-free, with
**no dense-`Ŵ` materialization** (which is GBs/weight at 32B). This is both a memory win
(store planes, not dense fp) and a compute win (ternary `t_p·x` is add/sub, not multiply).

**Features.**
1. **Fused ternary matmul forward** on resident plane-packed weights. The inference
   path already has `salt_mpgemm_tiled_f32` + `CudaBackend::upload_salt` /
   `SaltResidentLinear` (`src/cuda/backend.rs`) — reuse the packing + kernel as the
   forward oracle. Add a training entry that consumes resident planes.
2. **Backward through the planes:** `dgrad` = `Σ_p s_p·(t_pᵀ·gy)` (multiply-free);
   `wgrad` stays fp against the latent master (identity STE, Track A). Gate: **device
   ternary-GEMM fwd+bwd == the dense `salt_quantize` → `matmul` path** the current
   distillation uses (bit-close), so switching to fused planes changes speed/memory but
   not the trained result.

**Architecture.** A new `DevOp::SaltMatmul { planes, scales, … }` (weights are resident
plane buffers, not dense f32 leaves) plus its fwd/bwd. This makes the *weights*
resident as planes — dovetails with Track A (masters can be stored+quantized as planes)
and Track C (planes are far smaller than dense fp). Sequence: land Track A first (dense
quantize path proven resident), then swap the dense quantize+matmul for the fused
plane path here, gated equal.

---

## Track E — CPU-offload AdamW + multi-GPU (the 32B enabler)

**Goal.** 32B latent masters + Adam state exceed a single GPU. Stream them.

**Features.**
1. **CPU-offload AdamW:** masters + `m`/`v` live in host RAM; per layer, stream the
   slice to device for `adamw_step_dev` (Track A), stream back. Overlap copy with
   compute. Gate: identical updates to the fully-resident AdamW (rel < 1e-5) on a model
   that fits both ways; peak VRAM independent of parameter count.
2. **Multi-GPU data/tensor parallel** for the fwd+bwd, reusing the NCCL wire-correctness
   already validated ([`crates/tritium-cuda/src/nccl.rs`], v0.60). Gradient all-reduce
   must be **gradient-level checked** (assert reduced grad == full-batch reference — Adam
   hides scale errors; see the distributed-parity note). This track is **fenced HW**
   (rented multi-GPU), same document-and-defer discipline as the NCCL wall.

**Architecture.** `DeviceTrainer` gains an offload policy (resident | host-offload) and,
for multi-GPU, wraps the per-step grad with a `ProcessGroup::all_reduce`. Keep
single-GPU resident as the default; offload/multi-GPU are opt-in for scale.

**Implemented seam.** The policy lives in the production `CampaignTrainer` boundary,
which dispatches to separate `DeviceTrainer` and `HostOffloadTrainer` modules. This
keeps each optimizer's state machine narrow while preserving the specified resident
default and explicit host-offload campaign policy. Distributed execution is currently
host-offload only.

---

## Track F — Scale-up validation: 1.7B → 32B, and the paper endpoint

**Goal.** Turn the engine into the paper's result.

**Steps (each a gate, not a promise):**
1. **1.7B on the 4090** with Tracks A–D: device SALT distillation recovers held-out ppl;
   report step time + VRAM. (1.7B fits with checkpointing + plane weights.)
2. **Grow-then-ternarize prototype:** function-preserving expansion (net2net/LiGO) of a
   smaller fp model → SALT-distill the grown model → show the byte-optimal Pareto point
   (more params, fewer bytes, ≥ fp quality). This is the paper's core claim; the engine
   makes it runnable.
3. **32B (rented multi-GPU, Track E):** the ADR-0020 capstone — SALT-distill a
   32B-class model to **≤1% held-out ppl vs fp**. Fenced HW.
4. **Quality-vs-bytes curve** across model sizes (135M→1.7B→32B) for the paper — the
   endpoint the whole path exists to produce.

**Architecture.** No new engine; this track *uses* A–E. The blocker was always speed +
memory at scale, which A–E remove.

---

## Reuse map (do not rebuild)

- **Bit-exactness oracles:** `crates/tritium-train/src/ops/*` (`norm`, `act`,
  `elementwise`, `softmax`, `rope`, `shape`, `embed`, `loss`, `ste`) and
  `optim::AdamW` — every device kernel is gated against these.
- **Engine:** `DeviceTape` + the `*_dev` methods + `crates/tritium-cuda/kernels/train_grad.cu`
  (add kernels here, same `--fmad=false` module — no `build.rs` change needed for it; note
  there are stale copies under `.claude/worktrees/` — the crate-root path is the live one).
- **Ternary weights on device:** `salt_mpgemm_tiled_f32`, `CudaBackend::upload_salt`,
  `SaltResidentLinear` (Track D forward oracle + packing).
- **Model builder:** `crates/tritium-nn/tests/common/mod.rs::device_forward` +
  `extract` (fp-order weight ids) + the corpus/ppl helpers.
- **Distillation harness:** `crates/tritium-nn/tests/salt_distill_heldout.rs` (the
  device gate `salt_distillation_device_tape_recovers_heldout` is the Track-A baseline).
- **Distributed:** `crates/tritium-cuda/src/nccl.rs` + `tritium_train::dist` (Track E).

## Verification discipline (applies to every track)

- **Every new kernel** ships with a `device == CPU` gate vs its oracle, in the same
  file as the existing `resident_*` tests (`crates/tritium-cuda/src/train.rs` `#[cfg(test)]`),
  before it is used. Pure `+`/`*`/`/`/`sqrt` kernels must be **bit-exact** (`== 0.0`);
  transcendentals (`exp`/`sin`/`cos`) are gated `< 1e-4`.
- **Every track** re-runs the whole-model gate
  (`device_tape_trains_smollm2_matching_cpu_tape`) and the distillation-recovery gate to
  prove no regression, and **measures before/after** (step ms, VRAM) — "no win" is a
  valid, reportable result; keep the benches as regression gates. Warm up before timing.
- **Reviews:** each commit reviewed by the code-reviewer subagent (project policy),
  fixed, re-reviewed, then pushed via the deploy key.

## Shared-tree hazard (operational, read before you commit)

The `tritium-cuda` crate is edited **concurrently** by another session (inference /
autotune tracks). Training files (`train.rs`, `train_grad.cu`) are yours; `backend.rs`
and `consts.rs` are **shared** — the other session adds inference/autotune code there.
**Stage explicit paths only; never `--amend`/`rebase`/`reset` on the shared branch**
(a rewrite has swallowed the other session's commit before). When `backend.rs` /
`consts.rs` contain both sessions' edits, stage only your hunks with `git add -p` and
verify the staged diff has no foreign lines before committing.

## Risks

- **Device AdamW / salt_quantize numerics** (Tracks A, D) — bit-close, not bit-exact,
  is acceptable *only* if the distillation still recovers within noise; gate on recovery,
  not just per-op rel.
- **embed_gather_backward determinism** (Track B) — any atomic/reordered accumulation
  breaks bit-exactness; the sorted/segmented approach must preserve ascending-`t` order.
- **Checkpointing recompute correctness** (Track C) — the recompute must reproduce the
  forward exactly (same ops, same order); gate grads against the non-checkpointed tape.
- **Scale is fenced HW** (Tracks E, F.3) — document-and-defer; do not block the 4090
  tracks (A–D, F.1–F.2) on rented multi-GPU.

## Decision

Adopt Tracks A–F as the completion of the device-resident training path. **Build order:
A (resident optimizer — unlocks the real per-step win) → B (embed hotspot) → C
(checkpointing — unlocks 1.7B) → D (fused ternary GEMM — memory+compute for 32B) → F.1–2
(1.7B + grow-then-ternarize on the 4090) → E + F.3–4 (32B on rented multi-GPU, fenced).**
Each track is done when its gate is green, benched, reviewed, and pushed.

---

## Implementation Results (updated 2026-07-13)

This section records the implemented Tracks A through E software seams and the Track F
experiment substrate. Track D's production Fast path is numerically accepted, but its
compute-win and full-step performance gates remain open. Commit `a644971` adds the
final production seams planned here: packed resident campaign state, resident gradient
streaming, immutable Exact/Fast compute policy, resident/host-offload campaign policy,
and supervised process-per-GPU NCCL orchestration. Physical-VRAM, multi-GPU, 1.7B,
32B, and paper-result gates remain empirical acceptance work on repaired or rented
hardware. Green software tests are not substitutes for those measurements.

### Landed work

- **Track A:** strict CUDA lint baseline (`a6ac90b`); resident SALT quantization
  (`4b702fa`); resident AdamW (`18dd879`); `DeviceTensor`, resident gradients, and
  `DeviceTrainer` (`20aea05`, `0171d0e`), with fail-closed poisoning if a resident
  optimizer failure could leave mixed parameter generations (`7a51fa6`). The CUDA
  and CPU SALT gates use the
  authoritative `tritium-train::ops::ste::salt_quantize_forward` oracle, whose
  AbsMean scale is **per row and per plane**. The earlier per-256-block wording in
  Track A was inaccurate and must not be used to reinterpret the landed numerical
  contract.
- **Track B:** deterministic segmented embedding gradients (`aac4f9a`), preserving
  ascending token-position accumulation and bit-exact CPU parity.
- **Training data and memory substrate:** bounded offline teacher-cache I/O
  (`fdc4f44`, with lockfile repair `bf7ab42`); lazy gradient-slot liveness
  (`fbd76b8`); streaming distributed checkpoints (`644c338`, `9926622`); resident
  NCCL collectives (`15defb3`); and sqrt-depth activation checkpointing with
  explicit fail-closed frontiers (`3769ea9`).
- **Tracks D and E:** compact training SALT planes (`8d7daef`); correctness-first
  per-parameter host Adam offload (`10c6e22`); packed `salt_embed`, `salt_matmul`,
  packed attention, activation VJP, dense identity-STE master gradients,
  tied-weight accumulation, and checkpoint replay (`6fe264c`, `1db88a8`,
  `49b36b2`). Finalized leaf gradients stream directly into either resident or host
  Adam and are released immediately (`c55ac8e`, `a644971`). The production resident
  campaign explicitly selects `DeviceTrainerWeightStorage::Packed`, which holds
  master plus Adam moments and one shared largest-leaf packing scratch instead of
  per-parameter dense quantized tensors and residuals. Its resident gradient stream
  separately avoids a full-model requested-gradient collection. Both trainers
  round-trip through bounded DCP v1 sources and sinks, enforce contiguous optimizer
  steps before mutation, and poison after any failure that can leave mixed
  generations.
  Host-offloaded Adam uses two persistent page-locked/device staging slots and a
  dedicated transfer stream. Cudarc events order next-leaf H2D, current Adam, and
  prior-leaf D2H without a host synchronization between leaves. Packed forward and
  activation-gradient contractions expose immutable `exact` and `fast` production
  policies. Exact preserves dense-order arithmetic without materializing dense
  weights; Fast selects the plane-grouped/tiled multiply-free contractions and is
  gated against a whole-model scaled `2e-4` tolerance.
- **Track F substrate:** deterministic intermediate-width Net2Wider transforms and
  quality/bytes selection (`e4b8f5b`, `f1a4e3c`). A production tied-SwiGLU
  training adapter now owns the canonical HuggingFace parameter map (`embed`, then
  per-layer `q/k/v/o/gate/up/down`), fixed norm state, and packed device graph
  builder.
  It validates the SmolLM2 135M, 360M, and 1.7B geometries and rejects QKV bias,
  QK norm, untied heads, and tensor-shape drift before allocating a campaign.
- **Distributed campaign:** `NcclProcessGroup::xent_backward_into` reduces each
  finalized resident gradient before host Adam (`afd5871`). Commit `a644971` adds a
  supervised process per explicit CUDA device, immutable rendezvous, deterministic
  rank-window partitioning, hardware/plan/step consensus, lifecycle barriers,
  per-rank evidence, timeout and peer termination, and rank-0 ownership of lock,
  checkpoint, report, and artifact publication. Linux workers arm a parent-death
  signal before `exec`, with a post-arm parent check that closes the supervisor-death
  race. The two-or-more-GPU full-batch gradient-reference and end-to-end campaign
  gates remain hardware-fenced.
- **Production campaign and evidence:** the CUDA CLI campaign (`5ded216`,
  `2f90cb4`, `81ae5c7`, `a644971`) now acquires advisory locks for the checkpoint,
  report, and artifact before model or cache I/O. Lock opening rejects symlinks,
  special files, hardlinked inodes, and inode replacement without truncating stale
  unlocked sidecars. The campaign
  binds source/config/corpus/cache/evaluation/growth/hardware identities into an
  immutable plan, resumes bounded DCP state, marks warmup timings after every
  resume, and records exact CUDA async-pool high-water plus point-in-time,
  device-wide NVML framebuffer-used samples. The campaign requires a separate
  held-out corpus and rejects any exact training-window overlap. It persists each
  source-fp and reloaded-artifact NLL accumulator's exact f64 bit pattern after
  every window; the report records both PPL values and their relative delta.
  Plan/report schema v5 binds state policy, compute policy, world topology,
  partition rule, and hardware fleet. Reports carry per-rank timing and allocator
  evidence plus per-rank and fleet host/staging byte totals. The
  `optimizer_state_device_fraction` field describes persistent master and Adam
  placement (`1.0` resident, `0.0` host-offload), not GPU utilization or step time.
  Single-GPU defaults remain `resident` plus `exact`; distributed execution requires
  explicit `host-offload` state until optimizer-state sharding is implemented.
- **Terminal artifact:** streamed training-SALT GGUF export and exact package
  hashing (`c3d3347`, `b51e56c`, `afd6cfb`, `d0ba9aa`, `a74945c`) produce a
  deterministic, self-contained artifact. Existing output is never overwritten:
  publication is an atomic same-directory no-replace link. Resume deterministically
  rebuilds the expected package through a bounded hasher. Evaluation rehashes the
  exact opened inode, validates plan/source/growth provenance, drops host optimizer
  state before dense CPU reload, and uses the strict training-SALT loader and scorer
  (`09ef4f6`).

### Local gate results and measurements

- Resident SALT matches the per-row CPU oracle for `T in {1,2,3}` within `1e-4`;
  resident AdamW matches CPU master and moment updates within `1e-5` across
  multiple steps.
- Segmented embedding gradients are bit-exact across repeated tokens, boundary
  dimensions, untouched rows, and invalid-token guards.
- Lazy gradients, checkpoint replay, and sqrt-depth logical activation bounds are
  green on branched, multi-block, and full GQA transformer-block graphs; replayed
  gradients remain within the `1e-4` contract.
- Host-offloaded Adam state matches the fully resident optimizer within `1e-5` and
  keeps both device and pinned staging bounded at `6 * largest_parameter` f32
  elements (two slots, each holding master plus two moments), with at most two
  updates in flight rather than staging proportional to model size.
- Packed-only resident Adam matches the collected-gradient resident path and keeps
  requested parameter gradients bounded by reverse-topological liveness. The focused
  resident gate proves its live peak is below the 11-element materialized collection
  while matching collected Adam updates for every bound leaf. Resident persistent
  state is `3 * dense_parameter_bytes` plus one
  largest-parameter f32 packing scratch; packed codes/scales are accounted
  separately. Repeated and skipped steps are rejected before either trainer mutates.
- Packed SALT forward, activation gradients, dense master gradients, tied
  embedding/head accumulation, repack-after-offload, and checkpoint replay are
  green against the dense SALT oracle. The `T=3`, `576 x 576` packed representation
  is about `25.5%` of the dense f32 weight bytes.
- The tiled packed contractions are bit-exact to the scalar packed oracle for
  `T in {1,2,3}`, tail shapes, zero-work launches, and `K=8193`; both remain within
  the `1e-4` dense-oracle contract. On a GPU-contended diagnostic 4090 run, tiled
  activation-gradient improved `1.42x` over scalar, while repack plus forward was
  effectively flat (`1.00x` scalar, `1.01x` dense).
- The full packed/checkpointed SmolLM2-135M path at `T=2` uses `86,866,944` weight
  bytes versus `537,919,488` dense bytes (`0.1615x`). The latest release gate's
  logical checkpoint activation peak was `741,888 / 4,019,712` elements (`0.1846x`),
  with `2,911` recomputed operations.
- The 40-step packed plus streamed-offload campaign recovers held-out quality by
  `963x` versus PTQ (fp PPL `19.729`, PTQ `2.125e6`, distilled `2207.241`), clearing
  this ADR's approximately `960x` measured target. The executable test retains a
  `950x` regression floor to keep accepted variation near the measured result; that
  floor is not the measured result. Streaming lowers requested-gradient peak to
  `116,785,152` bytes from a `537,919,488`-byte materialized collection (`0.2171x`)
  and emits all 211 parameters exactly once in a stable reverse order.
- The strict packed whole-model numerical gate is green. On SmolLM2-135M, the
  exact packed path is bitwise equal to the dense CUDA path at the checked logits,
  tied-embedding gradient, and layer-0 down gradient (`0.000e0` max-absolute
  delta for all three). Low-level forward and activation-gradient gates are also
  bitwise equal for `T in {1,2,3}` and `K in {7,257,576,8193}`. The production Fast
  policy's latest 30-layer scaled deltas were `1.332e-4` logits, `1.016e-4` tied
  embedding gradient, and `6.714e-5` layer-0 down gradient, all inside its accumulated
  `2e-4` whole-model gate.
- Campaign-focused tests cover plan immutability, path aliases, early locking,
  report/checkpoint crash ordering, exact-window holdout contamination, complete
  and partial evaluation resume, artifact determinism/corruption/loadability,
  no-replace publication, fail-closed NVML parsing and identity checks, output-lock
  contention and inode attacks, worker failure/timeout cleanup, and supervisor-death
  containment. Local validation after `a644971` recorded 112 CUDA/NCCL-feature tests
  passing with four intentional performance ignores and eight runtime-fenced NCCL
  cases excluded; the CLI unit suite passed 76 tests with five helper probes ignored,
  and all 11 CLI integration tests passed. Strict CLI, CUDA, and selected NN clippy
  gates are green under `-D warnings`; formatting is green.
- **Negative performance result:** production Fast is numerically accepted but has
  not shown a stable compute win. Two single-pass, fixed-order diagnostics, not
  warmed benchmarks, printed dense/Exact/Fast timings of `159/111/239 ms` and
  `349/115/226 ms`. Fast was slower than Exact in both, so no speedup claim is
  accepted. A contended 40-step campaign averaged `4377 ms/step`, missing the
  unchanged `1123 ms` Track-0 gate despite its `963x` recovery. Profiling-driven
  kernel work, uncontended timing, and physical overlap measurements remain
  acceptance work.

### Remaining empirical acceptance

- Profile the production Fast contractions on an uncontended, repaired CUDA runtime
  until the combined packed path beats dense materialization. Rerun the retained
  `vocab=49152, dim=576` Track-B microbenchmark, measure actual pinned-transfer and
  Adam overlap, and clear the `1123 ms` full-step gate. The correctness-proven Fast
  path is implemented; the compute-win claim is not accepted.
- Run the supervised campaign on at least two GPUs. Required evidence is reduced
  gradient equality to the concatenated full-batch reference before Adam, rank
  consensus, timeout/crash cleanup, checkpoint resume, terminal artifact equality,
  and per-rank physical memory/timing. Unit and one-GPU process-supervision tests do
  not replace this gate.
- After repairing the local NVIDIA runtime, run the cached SmolLM2-360M campaign and
  then a compatible 1.7B tied-SwiGLU artifact. Record held-out recovery, warmed step
  time, checkpoint recovery, artifact reload PPL, exact package bytes, sampled NVML,
  and allocator-pool high-water. No compatible 1.7B artifact is cached locally.
- Use the resulting fleet-memory report before renting for 32B. The current
  process-per-GPU data-parallel design replicates host optimizer state per rank, so a
  world-two 32B run needs about `768 GB` decimal (`715 GiB`) for master plus two Adam
  moments before teacher cache, pinned slots, checkpoints, and runtime overhead.
  Provide at least `1 TiB` host RAM or add optimizer-state sharding before that run.
- The campaign records evidence but does not declare the `1123 ms`, physical-fit,
  multi-GPU, `<=1%` 32B quality, or paper gates passed. Acceptance remains an explicit
  comparison against the ADR thresholds. Do not infer it from green unit, quality,
  or logical-memory gates.

### Hardware and data fences

- **Local NVIDIA runtime:** the running open kernel module is `610.43.02`, while
  installed `nvidia-utils 610.43.03-1` supplies NVML `610.43.03`; `nvidia-smi`
  currently fails with `Driver/library version mismatch`. The running kernel is
  `7.1.3-1-cachyos`; the installed `linux-cachyos` package is `7.1.3-2`. Reboot into
  the installed kernel/driver pair before collecting NVML, NCCL, performance, or
  physical-VRAM evidence. After removing a false-green skip (`98127e9`), eight
  one-GPU NCCL cases fail at communicator initialization: six `world1_*` tests,
  logical-length validation, and FSDP parity. The foreign-context test remains
  two-or-more-GPU fenced. None supplies NCCL execution evidence. No reboot was
  performed in this shared session.
- **1.7B / local 4090:** no compatible local 1.7B training artifact or recovery,
  warmed step-time, checkpoint-recovery, or physical-VRAM result has been recorded.
  The adapter and bounded packed/offload campaign support the geometry; the missing
  model artifact and repaired runtime are now the local gates.
- **New production multi-GPU path:** requires a two-or-more-GPU run. The older 2xA100
  v0.60 ProcessGroup result and the gated resident-gradient primitive do not validate
  the newly landed supervisor, partition, consensus, evidence, and publication path.
- **32B capstone:** remains rented-hardware and data fenced. Host master plus two
  Adam moments require about `384 GB` decimal per replicated rank. The current
  world-two design therefore needs about `768 GB` decimal before teacher,
  checkpoint, pinned staging, and runtime overhead. The prior `512 GiB` host target
  is insufficient; use at least `1 TiB` or shard optimizer state first.
- **Paper endpoint:** no measured 135M-to-1.7B-to-32B quality-versus-bytes curve or
  grown-model Pareto result exists yet. The committed report and ledger types are
  schemas, not experimental evidence.

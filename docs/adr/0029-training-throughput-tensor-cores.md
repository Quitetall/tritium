# ADR 0029 — Device-resident training throughput: batching, tensor cores, and the launch-bound step

Status: **PROPOSED / IN PROGRESS** (2026-07-18)

- **Deciders:** Brian Lam
- **Relates:** completes the perf-pass deferred by [ADR 0027](./0027-device-resident-training-perf-and-scale.md)
  (device-resident training engine). Consumes the SALT distillation of ADR 0016 / [0028](./0028-salt-v2-additive-ternarization.md);
  references [ADR 0024](./0024-structured-24-ternary.md) (structured 2:4 ternary — Lever 7) and
  [ADR 0023](./0023-relaxed-reduction-tier.md) (relaxed-reduction tier, rejected — the bit-exact
  reduction order is the product identity, so the tensor-core tier is gated on *recovery*, not
  bit-exactness). Feeds the v1.x capstone ([ADR 0020](./0020-v1x-salt-distillation-capstone.md)).

## Context — the real bottleneck is training throughput, not the method

SALT distillation recovers held-out ppl vs naive PTQ (135M: PTQ 2.1e6 → distilled ~2214, 957×) but
stays far from fp (fp 19.7). The distillation *objective is already correct* — full-vocab logit KD
`softmax_xent(student, teacher_probs)`. The gap is **budget**: the committed gate runs **40 steps on
a few kilotokens**. The device engine trains at **seq=32 / batch=1 = 32 tok/s**, so a real
convergence run was infeasible. This ADR removes the throughput wall so **Step 1** (135M
distill-to-convergence) can run and answer the load-bearing question: does the method approach fp,
or plateau at the small-model ternary floor (which would *support* the paper's grow-then-ternarize
thesis)?

### Recon of the f32 engine (4 parallel agents, all confirmed in code)
- Strictly **2D, single-sequence, batch=1, 32 tok/step, no grad accumulation**.
- Attention materializes the full `[seq,seq]` score matrix **per head, looped sequentially**, retained
  for backward → O(n_head·seq²) memory + many tiny launches.
- The **teacher fp forward is re-run every step** (a second full forward — ~half the step).
- **All training kernels are naive f32** (`--fmad=false`, one-thread-per-output scalar reduction,
  `compute_75`, no tiling) — **zero tensor cores**; the only tensor-core code is INT8 IMMA inference
  prefill. cuBLAS is not linked.
- Optimizer master + Adam m + v are all f32 (~21GB resident at 1.7B).

## Levers (build order) and status

**Lever 2 — Batching (DONE — commit `d06568d`, reviewed clean).**
`DeviceTape::slice_rows`/`concat_rows` (a row-block of row-major data is exactly a column-slice of the
flattened `[1, rows*cols]` view — exact reuse of the bit-exact `slice_cols`/`concat` vjps, **no new
kernel**). Batched forward: embed/norms/MLP/tied-head at `M=batch*seq`; only attention loops
per-sequence on row-blocks. Gate `device_tape_batched_block_matches_per_sequence` (fwd bit-exact,
grads <1e-4). **Measured ceiling:** pure batching caps at **2.1×** (per-seq attention = many tiny
launches); balanced **batch=4 × seq=512 = 3.9×** (1013 → ~4030 tok/s). The f32 engine floors ~4× from
parallelism alone.

**Lever 1 — tf32 tensor-core tier (DONE — commits `e08f687`, `5b0778a`, `df052d8`, `e8ab5eb`, `1b5730e`, all reviewed).**
cuBLASLt's `Matmul<f32>` uses `CUBLAS_COMPUTE_32F_FAST_TF32` — **tf32 tensor cores, fp32 accumulate,
f32 in/out, no cast kernels**. `TensorCoreGemm` (train.rs) exposes forward `Y=X·Wᵀ`, `grad_a=gY·W`,
`grad_w=gYᵀ·X`; wired into `DeviceTape` as an opt-in `.with_tensor_core(&tc)` policy (dense `Matmul`
forward + checkpoint-recompute + both backward GEMMs; `None` = unchanged f32 path; packed `SaltMatmul`
untouched). Gated `device_tape_tf32_matches_f32_whole_model` (tf32 vs f32 <6e-4 across full depth,
with a lower-bound sentinel so a silent un-wire fails). Opt-in in the resident recovery gate via
`TRITIUM_DISTILL_TF32` (default off).
  - **Per-GEMM: 65×** (3.49ms → 0.053ms; the naive f32 kernel was ~65× off optimal — tiling + tensor
    cores combined).
  - **Full-step: only 1.2–1.6×** (`bench_tf32_whole_model_step`). **This is the key finding:** once the
    GEMMs are fast, the step is **launch/glue-bound** — GEMMs are ~14% of the step; the rest is the
    naive f32 glue kernels (norm/softmax/silu/elementwise/rope), the hundreds of tiny per-head
    attention launches, and per-step weight upload. The ratio grows with seq (GEMM fraction) and would
    be higher on a resident trainer (no re-upload) and larger models.

**Lever 3 — Teacher caching (queued; top-k refined).** The ~2× win is **teacher caching** — eliminate
the redundant per-step teacher forward (infra exists: `TeacherCacheReader`, `TRITIUM_TEACHER_CACHE`).
**Top-k KD does NOT sparsify compute:** the KD logit-gradient is `softmax(z)[j] − t[j]`, and
`softmax(z)[j]` is nonzero for every vocab entry, so the dominant tied-lm-head weight-grad stays dense.
Top-k only shrinks the teacher *disk cache* (~500×) and makes precompute feasible. Truly sparsifying
the head grad needs sampled-softmax/NCE (deeper, quality risk) — not before Step 1. Add a KD
temperature knob (currently absent) as a near-free quality lever.

**Lever 5 — Optimizer VRAM (IN PROGRESS 2026-07-19; user-prioritized before the corpus run).** bf16
master + 8-bit Adam cut the resident optimizer state ~4× (moments dominate: `m`+`v` f32 ≈ 2× the
model). Invisible at 135M (model+optimizer tiny) but validated there so it is proven before scaling.
Landed so far, CPU-oracle-first then device-mirrored and gated:
- `tritium_train::Int8AdamW` (commit 98074cb, reviewed clean): block-wise int8 moments, block 256. **The
  second moment is stored in sqrt-space** (`v_q = round(√v/scale)`) — a naive linear-`v` quantizer
  underflows small values while `m` keeps its history and the step `m/(√v+eps)` explodes (the descent
  gate caught it: linear-`v` diverged to loss 4958, sqrt-space converges within 3× of f32).
- `tritium_train::bf16` stochastic-rounding master (commit 811f49b, reviewed clean): SR keeps a sub-ULP
  update alive in expectation where nearest rounding stalls (gated: unbiased over 200k draws; a
  coarse-grid weight climbs to target under SR, stalls under nearest).
- Device `adamw_step_8bit` CUDA kernel + `adamw_step_8bit_dev` (commit 85f6c58): one CUDA block per
  256-element optimizer block, bit-identical to the oracle (all correctly-rounded ops, `--fmad=false`);
  parity gate `adamw_step_8bit_matches_cpu_oracle` (5 steps, ragged tail) — params tight, codes within ±1.
- `DeviceTrainer` int8 path: `MomentPrecision::Int8` + `new_with_options`, opt-in (F32 default byte-
  identical), moments dispatched to the int8 kernel.
- `DeviceTrainer` bf16-master path (commit 7693255): `MasterPrecision::Bf16` confines the master to the
  bf16 grid with SR after each step (`sr_round_to_bf16grid`) — numerically a real bf16 master without
  swapping the storage type through the reconstruction path (the u16 VRAM-halving swap is a mechanical
  follow-up gated on this). Unit gate `device_trainer_reduced_precision_trains_like_f32` covers
  f32/int8/bf16-master all tracking f32 on a toy distill.

**Int8 divergence — ROOT-CAUSED AND FIXED (commit 6fea28c).** The int8 A/B first blew up (tracked f32 to
step 24 then exploded to ppl 3.9e25). Root cause: a block's `m` absmax and `√v` absmax can be dominated
by **different** coordinates — a huge-magnitude *oscillating* coord has `m≈0` (sign cancellation) but a
large `√v` (RMS), so it dominates the `√v` grid but not the `m` grid; a steady neighbour's `v` then rounds
to code 0 while its `m` survives, and when it goes quiet (`g≈0`, residual `m`) `vi` collapses to 0 and the
`m/(√v+eps)` step explodes. (The first repro used a *steady* dominant coord, which dominates both grids, so
it missed this; the oscillating repro reproduces the blow-up.) Fix: a nonzero `√v` never dequantizes to 0
— floor its code at 1, in both the CPU oracle and the kernel (still bit-identical). **Result: int8 135M
recovery A/B now recovers 939× (stable), vs f32's 960×** — a ~2% precision tax, matching bf16-master (907×)
and tf32 (920×). Both Lever-5 halves now work end-to-end.

**Lever 5 status.** int8 moments and bf16 master both validated end-to-end on the real 135M (939× / 907×,
stable, ~2–5% recovery tax for cutting optimizer-state / master VRAM). Remaining is mechanical: the actual
u16 bf16-master storage swap through the SALT reconstruction path (VRAM realization; numerically identical
to the validated grid mode) and, at 1.7B/32B, combining bf16 master with int8 moments. The recovery cost is
invisible at 135M and the standard-corpus run uses f32; these buy headroom at scale.

**Lever 6 — Launch-overhead reduction (queued; the biggest remaining full-step lever).** Since the
step is launch-bound, the payoff of tf32 *and* 2:4 is unlocked here: **CUDA graphs** (capture+replay the
step's launch sequence → eliminate per-launch CPU overhead) and/or **kernel fusion + batched attention**
(one GEMM across heads instead of the per-head loop). This is what converts the 65×-per-GEMM into a
large *full-step* speedup.

**Lever 7 — 2:4 structured sparsity ([ADR 0024](./0024-structured-24-ternary.md)).** User-requested. A
2:4-constrained ternary SALT quantizer + sparse-tensor-core GEMM → 2× on GEMMs + 2× weight compression;
natural fit (ternary is ~30–50% zero) and a method variant to evaluate in Step 1 (does 2:4-ternary
distill as well as dense-ternary?). **Honest sequencing:** it speeds only GEMMs (~14% of a launch-bound
step → ~7% full-step) **until Lever 6 + scale make the step GEMM-bound.** Do it after Lever 6.

## Step 1 — the decisive experiment (gated behind Levers 1–3/6)
Run 135M SALT distillation **to convergence** — real corpus (not the cycled fixture), tens of
thousands of steps, cosine LR + warmup — through the **resident `DeviceTrainer`** path (chosen: its
default `DenseQuantized` forward already runs dense `dt.matmul`, which the tf32 tier covers, so
`device_forward_resident` + `.with_tensor_core` needs **no change to the trainer**). Trace the
recovery-vs-tokens curve. **Expected + acceptable outcome:** it plateaus at ~1.3–2× fp (the
small-model ternary floor), which *validates the method and points at grow-then-ternarize* — sub-2B
ternary loses quality even with infinite tokens (BitNet parity is at 2B+). If it asymptotes near fp,
even better.

## Honest findings / non-goals
- 65×-per-GEMM ≠ 65×-per-step; the step is launch/glue-bound (Lever 6 is the real full-step lever).
- Top-k KD does not sparsify the lm-head gradient (softmax normalizer keeps it dense).
- 2:4 sparsity's payoff is gated on launch-reduction + scale, not immediate.
- The tensor-core tier is bit-exact-*relaxed* (tf32), gated on **recovery-PPL**, with the f32
  `--fmad=false` path retained as the correctness oracle (ADR 0023 precedent).
- **tf32 is a net loss at 135M/seq=32 (validated 2026-07-19, VRAM freed).** Controlled A/B on the
  real SmolLM2-135M resident recovery gate (40 steps, one variable — `TRITIUM_DISTILL_TF32` on/off,
  everything else identical): **f32 recovers 960× at 518 ms/step; tf32 recovers 920× at 488 ms/step.**
  So tf32 buys only **~6% full-step time** (the GEMMs it accelerates are ~14% of a launch-bound step)
  while **costing ~4% recovery** (reduced mantissa perturbs the SALT master updates). Verdict: keep
  the committed gate on f32 (the design already defaults there); do **not** enable tf32 for the 135M
  distill. tf32 may still pay off at 1.7B/32B where GEMMs dominate the step and 4% recovery is
  amortized against a larger speedup — but that is **unverified** and separately VRAM-gated.

## Blocker cleared
The shared RTX 4090 VRAM freed (13.5GB of 24GB now free; the parallel Q-blocked-attention python3
process exited, leaving only the user's `tritium-serve`). The tf32 recovery A/B ran (result above);
the Step-1 recovery-vs-tokens curve runs next on the f32 path.

## Verification
- Batching + tf32 gates green (`cargo test -p tritium-cuda --features cuda`), all 55 train tests pass
  (f32 path unregressed). Benches `bench_batched_throughput`, `bench_tf32_whole_model_step` are
  regression-recorded.
- **tf32 recovery A/B — DONE (2026-07-19):** `salt_distillation_device_trainer_recovers_heldout`,
  40 steps. f32 = **960×** (gate ✅), tf32 = **920×** (below the 950× gate). tf32 does *not* preserve
  recovery at this scale; the numbers land in the honest-findings note above.
- **Step 1 — recovery-vs-tokens curve DONE (2026-07-19, f32).** `TRITIUM_DISTILL_CURVE=K` traces the
  held-out (disjoint) ppl vs fp 19.73 in a single run. 768 steps (3 epochs, 8k-token Alice corpus):
  2205 → 729 → 439 → **298**, still descending. A 2304-step / 9-epoch plateau run then bottomed out:
  final **224.8 ppl (11.4× fp)**, recovery-vs-PTQ **9453×**, oscillating in a 220–290 floor. **So on
  the 8k-token corpus, SALT distillation drives ternary-135M monotonically down but plateaus at
  ~11–13× fp** — at the time read as the data's ceiling; the WikiText-2 run below shows the plateau
  was substantially an LR artifact.

- **Step 1 — FIELD-COMPARABLE RESULT (2026-07-30, WikiText-2, f32).** The first run on a recognized
  corpus (500k-token train pool + 4096 held-out from the disjoint **WikiText-2 test split**, SmolLM2
  tokenizer). Required fixing an O(seq²) eval OOM first (commit `c0dae70`, see below).
  **fp SmolLM2-135M = 23.827 ppl | ternary PTQ = 3.281e6 (catastrophic).** Two 5000-step runs:

  | recipe | final ppl | best | gap to fp | recovery vs PTQ | tail oscillation |
  |---|---|---|---|---|---|
  | constant LR 2e-3 (gate default) | 563.3 | 431.2 @2800 | 23.6× | 5824× | ±25% |
  | **warmup 200 + cosine → 1e-4** | **266.9** | **265.8** | **11.2×** | **12 292×** | **±8%** |

  **The constant-LR plateau was an optimization artifact, not a data or method limit.** The baseline
  descended 3250 → 431 by step 2800 then oscillated flat for 2200 steps; adding the (already
  implemented but unused) `LrSchedule` beat it at *every* checkpoint, broke through the 431 floor at
  step 2400, and settled at **266.9 ppl — 2.1× better final, 2.1× more recovery, 3× less oscillation**.
  Only **160k of the 500k-token pool** was consumed (⅓ epoch), so the run is still token-limited: more
  steps (and `T`>2 SALT planes) are the untested levers before any claim about the ternary floor.

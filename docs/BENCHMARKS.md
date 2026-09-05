# Tritium benchmarks — the reproducible ledger

**Every number in this file reproduces from one command.** No number is recorded without the
exact command beside it and the environment it ran in (GPU, driver, co-resident processes).
Competitor numbers carry their exact invocations; we document their repro lines, we do not wrap
their binaries. The ledger is updated ONLY by re-running the harness — dated entries, newest
first. (ADR 0026 Track R.)

## The command

```sh
tritium report compare \
  --model <path/to/model.gguf> \
  --tokens <path/to/token-ids.json> \
  --backend cuda \
  --prompt-len 512 --decode-steps 256 --warmup 16 --reps 3 --runs 5 \
  --format json
```

Emits: decode tok/s ×3 (median), prefill tok/s + ttft p50/p95 over a 512-token prompt, and an
environment capture (GPU, driver, VRAM, co-resident compute processes, date, git commit). The
JSON is the artifact; the tables below are transcriptions of it.

**`--backend` resolves against backends linked into the binary**, so build the one you intend to
measure — the flag cannot reach a backend the binary does not contain:

| hardware | build | `--backend` |
|---|---|---|
| NVIDIA | `cargo build --release -p tritium-cli --features cuda` | `cuda` |
| AMD / Intel / any Vulkan GPU | `cargo build --release -p tritium-cli --features wgpu` | `wgpu` |
| AMD with a ROCm toolkit | `cargo build --release -p tritium-cli --features rocm` | `rocm` |

`wgpu` needs only a Vulkan driver (no vendor toolkit) and is the portable lane; `rocm` needs
`hipcc` present at build time. On any backend other than `cuda` the `gpu` field is taken from the
backend's own adapter name and suffixed with the backend that produced it —
`"<adapter name> [wgpu]"`. It is deliberately **not** taken from `nvidia-smi` there: a box can
expose several adapters, so `nvidia-smi` may describe a card the run never touched. `driver` and
`vram_*` come only from `nvidia-smi` and stay `"unavailable"` off NVIDIA — disclose that rather
than filling it in. Note `roofline_4090_pct` and `baseline_4090_drop_pct`
are computed against fixed RTX-4090 reference constants by definition (hence the names); off that
box they are a distance from a named reference point, **not** a claim about the local device.

### Methodology and honest caveats

- **Decode** = single-stream greedy steps after prefill+warmup, M=1, the memory-bound regime.
  Effective weight-stream bandwidth = tok/s × weight bytes is the honest cross-model metric.
- **Prefill (pp512)** = one 512-token `forward` p50 over N runs. First-run numbers include
  one-time JIT/tune costs unless the autotune disk cache is warm — run twice, record the second.
- **Contention**: numbers move ±10% under desktop/co-resident-GPU load; the `co_resident`
  capture field discloses the box state. The order-alternated ABBA protocol (OPTIMIZATION-LOG
  round 21) is the tie-breaker for close comparisons.
- **Competitor lines** (documented, not wrapped):
  - llama.cpp: `llama-bench -m <model.gguf> -p 512 -n 128 -ngl 99`
  - bitnet.cpp (CPU): `llama-bench -m <i2s.gguf> -p 512 -n 128 -t 14`

## Ledger

### 2026-08-20 dual-graph dispatch + the stacked long-ctx headline @ 431b837f

**Box state: quiet-validated per leg** (pre-gate 3×<15% util, post <15% after
a 4 s settle, warm plain reference 20.57–20.61 s across all six legs — the
tightest plain spread this campaign; the parallel session's job bursts were
dodged leg-by-leg). RTX 4090, driver 610.57.04, CUDA 13.3, HEAD build.

**The dispatch (02c8ff84 + 431b837f):** TreeGraphs captures are now keyed
(bucket, node_blocked) and picked per replay against
`TREE_NB_MIN_PREFIX = 1536` — the measured kernel crossover (extended prefix
ladder, mirror ABBA: nb/fused 1.016/0.992 at 1536, −18/−22% at 2048, −40%
at 3968, +109% at 512). `TRITIUM_TREE_NB` is tri-state (auto default / 1
force / 0 off); the round-31 NB win needs no knob and short prefixes keep
the fused body by construction (gated: `tree_nb_dispatch` pins the variant
counts (1,0)→(1,1)→(1,1) across the threshold).

**The stacked headline (measure_tau_ctx long protocol — the round-30 §4a
harness: 6 wt103 3,520-token prefixes, 256 greedy, 1-prompt discarded
warmup (the round-30 treatment),
plain reference exact/f32 on 8125; order E C C E then G G):**

| config | e2e vs plain (2 visits) | spec wall | τ | lossless |
|---|---|---:|---:|---|
| exact/f32 forced-on | 0.871 / 0.872 | 23.60 / 23.61 s | 3.901 | **6/6 + 6/6** |
| **fast+f16 forced-on (auto-NB)** | **1.858 / 1.860** | **11.07 / 11.07 s** | 2.505 | 1/6 (RFC 0001 tier) |
| governor-on fast+f16 | 1.741 / 1.745 | 11.81 / 11.81 s | 2.681 | 1/6; 304 suppressed-plain |

Against round-30 §4a on the same protocol: exact 0.882/0.893 → 0.871/0.872
(parity-class, cross-day clock drift); **fast+f16 1.648/1.624 → 1.858/1.860
(+13.6%, pairwise +12.7/+14.5%)**; governor 1.638/1.630 → 1.741/1.745
(+6.7%). The NB capture line
fired on every fast-tier leg (engagement proven); spec walls are identical
to 0.01 s across visits (deterministic replay). τ 2.505 vs round-30's 2.553:
the NB verify's ≤1e-4 drift flips different near-ties than the fused body's
— fast-tier semantics, the verifier still owns losslessness (the exact legs
stay 12/12 token-identical to plain greedy). The governor now lands within
6% of forced-on (was ~parity in round 30's terms) while still probing —
τ 2.68 sits inside the fast floor band (2.15–2.95), so partial suppression
is the cost model working as designed.

**The long-ctx serving headline is now ≈1.86× vs plain exact/f32 (≈1.27× vs
plain+f16, the 08-08 §2b basis) with no configuration beyond the fast+f16
opt-ins** — the same-config before was 0.97–0.98× (fast+f16, 08-08
§2c-long), and the lossless exact tier's before was 0.37×.

#### Caveats

- Cross-day comparisons to round 30 carry clock-state drift (the exact
  legs bound it at −1.2 to −2.4% per visit); within-table pairs are
  same-session.
- The plain reference is the HEAD binary (plain path untouched since
  8a808098 — the diff is drafter/verify-KV management only).
- identical=1/6 on the fast legs (the 08-08 §2c-long recorded 0/6 on this
  shape, different binary): NB's drift pattern happens to match plain
  greedy on one prompt — the same documented tier trade.

**Still owed:** upstream llama.cpp Q2_0 rerun; MI300X/Metal (user-gated).

> **2026-09-04 — what the Q2_0 debt actually is.** Recorded five times in this
> ledger as "the artifacts remain absent". Investigated; it is worse than that,
> and the correction is load-bearing for anyone who picks this up.
>
> **1. The named artifact does not exist publicly.** There is no Qwen3.5-4B Q2_0
> GGUF on HuggingFace; the `-m qwen35-4b-q2_0.gguf` in the 07-30 repro line was a
> locally-produced file that is not on this box. What *is* published is the
> comparison this ledger already called the clean follow-up — Prism ML's
> Ternary-Bonsai re-quantized to Q2_0 at 1.7B/4B/8B/27B, plus matching TQ2_0 at
> 4B/8B (`darkstarinitiative/AJAN-SIMIT-Ternary-Bonsai-Q2_0-GGUF`, Apache-2.0).
> The 1.7B and 4B are pulled to the model cache.
>
> **2. Our Q2_0 reader cannot read them, and the reason is not the type id.**
> `tritium inspect` fails with `tensor offset/size outside the data section`.
> `GGML_TYPE_Q2_0 = 42` (`gguf.rs:30`) matches the files exactly, so the reader
> believes it understands them. The disagreement is the block layout: `gguf.rs:418`
> expects **18 bytes per 64 elements** (f16 scale + 16 bytes of codes, 2.25 bpw),
> and every type-42 tensor in both files measures **exactly 17.000 bytes per 64
> elements** — 2.125 bpw, which is what their model card states. Derived from
> consecutive tensor offsets: 197 tensors in the 1.7B and 253 in the 4B, one
> distinct ratio, `[17.0]`. The one-byte delta is consistent with an 8-bit scale
> where we assume f16, but that is inference; the measurement is the byte count.
>
> **3. So "Q2_0" currently names two different formats.** Ours follows llama.cpp
> PR #24448; the artifacts in the wild follow Prism ML's fork. Which becomes the
> standard is open — Q2_0-g64 was still standardizing as of the ADR 0040 intake —
> so this is *not* recorded as a bug in either implementation. What it does mean:
> the same-model "Bonsai Q2_0 vs its Tritium import" comparison is blocked on
> reader work, not on hardware, and a type-id match is not evidence of format
> agreement. Worth a loud reject rather than a bounds error when the block size
> disagrees.
>
> **4. The upstream leg needs its premise checked first.** Their model card says
> Q2_0 is first-class in Prism ML's fork and "mainline `llama.cpp` support may
> vary by version". If mainline cannot read Q2_0, the "upstream-master Q2_0
> rerun" as written is not runnable, and the honest replacement is the Bonsai
> same-model run labelled against the fork. One mainline build settles it.

### 2026-08-18 spec-cost levers @ 82f80b1b — delta re-sync, node-blocked tree attention, TB1 LUT verdict

**Box state:** every number below ran in verified-quiet windows (desktop
graphics only, ~370 MiB, checked immediately before each bench; the RTX 4090 /
610.57.04 / CUDA 13.3 box was otherwise held by a parallel session's training
jobs for most of the day — two serve-level e2e attempts were CONTAMINATED
mid-run by those jobs and are NOT recorded). Binary provenance: working tree
at the named commits; kernel benches are in-tree harnesses.

| lever | state | measured (quiet box) | scope |
|---|---|---|---|
| delta re-sync (f355b795) | **ADOPTED**, on by default | probe re-entry **6.22 ms vs 71.27 ms** full re-prefill at the ctx-4032/gap-64 shape (11.5×; masked catch-up 145.8 ms); drafts token-identical to the full path | kept multi-spec rows with gap < dpos; short-prefix rows keep the full path |
| node-blocked tree attention (69f4aec3) | **ADOPTED for long-ctx serving** via opt-in `TRITIUM_TREE_NB=1` (default off pending dual-graph dispatch) | kernel: fused→NB @ prefix 3968 **221.2 → 131.5 µs/layer f16 (−40.6%)**, 228.3 → 139.1 g (−39.1%); prefix 512 +109%; **e2e §4: +9.8%** (1.523 vs 1.3865, verify EWMA −16%) | fast tier, dense solo route, buckets ≤ 8 |
| drafter flash profile (879af54d) | **REFUTED and DELETED** (§4: e2e −2.7%, drafter +7%/step — the rmsnorm_fast precedent) | measured loss at its one motivating shape; the drafter's long-ctx step cost is NOT attention-dominated | removed; kernels (the shipped M=N pair) unaffected |
| TB1 LUT decode (848d2f36) | **REFUTED — integration not considered** | see §3 | task #58 closed with the sharper mechanism |

#### 1. Delta re-sync (drafter_catchup_bench, quiet box)

Median of 5, rotation-fair legs at the probe shape (ctx 4032, gap 64, drafter
pool N=4, longctx drafter): re-prefill **71.27 ms** [71.0–71.8], masked
catch-up **145.83 ms** (2.28 ms/step), delta re-sync **6.22 ms** [6.2–6.4].
This also recaptures the round-30 §5 quiet-box absolutes owed by that entry
(contended run was 95.1/222.6 ms — ordering unchanged). Gates:
`reverse_adopt_restores_bit_exact` (bytes bit-for-bit after a real scramble +
refusal contract), `cuda_delta_resync_matches_full_reprefill` (drafts
token-identical), `cuda_batched_spec_delta_resync_fires_and_stays_lossless`
(the branch fired 3× in situ, streams token-identical to plain greedy),
batch_serve 8/8.

#### 2. Node-blocked tree attention (tree_fused_vs_exact_pair_kernel_abba_bench, quiet box)

BitNet verify shape (n_head 20 / kv 5 / hd 128, m=8 chain tree), 200-launch
batches, mirror ABBA (pair, fused, nb, nb, fused, pair):

| prefix | exact pair | fused (shipped fast tier) | nb+combine | nb/fused |
|---|---:|---:|---:|---:|
| 512, f16 | 107.8 µs | 40.9 µs | 85.4 µs | 2.09× (LOSES) |
| 512, f32 | — | — | — | (also losing; row lost to output truncation, not re-run) |
| 3968, f32 | 933.7 µs | 228.3 µs | **139.1 µs** | **0.609** |
| 3968, f16 | 801.7 µs | 221.2 µs | **131.5 µs** | **0.594** |

The mechanism: the fused body walks the whole [0, ctx) K/V range once per
(node, head) block and is latency-bound (~2% of f32 peak); the NB kernel
stages each 32-key tile decoded into shared once for all 8 node-warps, so the
serial chains run at smem latency and the codec amortizes 8-fold. At short
prefix the (n_head × n_slice) grid underoccupies — hence opt-in, long-ctx
scoped. RFC 0001 gate: worst rel 8.3e-5 (≤1e-4) including an
empty-trailing-slices shape and a 40× magnitude-spike shape (the combine's
exp-underflow regime). Implied verify effect at 4K fast+f16: attention
~6.6 → ~3.9 ms of the 10.9 ms verify (denominator: the task-#65 session
probe's verify EWMA, quiet box — re-derive when the owed e2e lands); the serve-level A/B is owed (below).

#### 3. TB1 LUT verdict (tb1_lut_gateup_l2defeated_bench, quiet box)

Gateup shape (M=1, N=13824, K=2560), same-session interleaved, rotating
replicas (4×L2 per format), continuous rotation phase, min-of-5 + σ:

| kernel | L2-resident | L2-defeated | byte-sensitivity |
|---|---:|---:|---:|
| TQ2 mainline (2.06 b/w) | 8.48 µs | 15.13 µs | 1.78× |
| TB1 scan (1.58 b/w) | 21.58 µs (2.54×) | 25.87 µs (**1.71×**) | 1.20× |
| TB1 LUT (1.58 b/w) | 18.80 µs (2.22×) | 24.06 µs (**1.59×**) | 1.28× |

Round 16's L2-resident 2.58× reproduces (2.54×). The redesign premise —
that the per-element expansion chains (~60% of the compiled loop's
instructions) were the cost — is REFUTED by its own success: the LUT kernel
removes them (bit-exact, gated) and buys only −13%. The sharper mechanism:
TB1 is latency-bound on data-dependent serial sign addressing (uncoalesced
per-lane window byte loads + the cross-block `sign_base` carry), which is
why its byte-sensitivity is 1.2–1.3× where the dense TQ2 kernel's is 1.78× —
the 19–23% byte saving (19% at the bench's trits, 23% nominal at p=0.422)
has nothing to repay it. The docs/ternary-formats.md bar
("close most of the 2.58× gap") is decisively unmet; integration is not
considered. First-ever L2-defeated TB1 row (the round-16 verdict was
resident-only), closing the bitmap+signs half of ADR 0036's L7 A/B (the TQ1 half is the
08-08 entry's §6 serve rows).

#### 4. Serve-level e2e A/B (2026-08-19, opportunistic quiet legs)

The parallel session's training campaign held the GPU for ~14 h; the legs ran
in a validated-quiet window (per-leg gates: 3×<15% util pre, <15% post after
a 4 s settle, warm plain reference < 15 s, spec wall < 26 s; every leg passed
all four). Protocol: plain reference server (8125, default env, warm,
constant); spec server (8124) relaunched per leg with
`TRITIUM_SPEC_ADAPTIVE=0 TRITIUM_KERNEL_TIER=fast TRITIUM_KV=f16
TRITIUM_DRAFT_K=legacy` (legacy k pins bucket 8 so the NB capture engages —
verified per leg via the c5431fad capture line) + the leg's knobs; per leg
one discarded warm pass then 3 recorded 3,520-token wt103 prompts × 256
greedy. Mirror order B F N X X N F B:

| config | e2e vs plain (2 visits) | spec wall | verify EWMA ms | drafter ms/tok | τ |
|---|---|---:|---:|---:|---:|
| B baseline | 1.385 / 1.388 | 8.15 / 8.15 s | 11.76 / 11.91 | 1.97 / 1.95 | 3.547 |
| F `TRITIUM_DRAFT_ATTN=flash` (knob since DELETED) | 1.350 / 1.349 | 8.38 / 8.38 s | 11.88 / 11.91 | 2.11 / 2.10 | 3.559 |
| N `TRITIUM_TREE_NB=1` | **1.525 / 1.521** | **7.42 / 7.44 s** | **9.98 / 9.93** | 1.95 / 2.01 | 3.597 |
| X both | 1.371 / 1.331 | 8.50 / 8.51 s | 8.60 / 8.64 | 2.11 / 2.06 | 3.116 |

Verdicts. (1) **NB: +9.8% e2e** (means 1.523 vs 1.3865, unrounded), verify EWMA −16%,
spec wall −8.8% — clears the 3% bar; ADOPTED for long-ctx serving as the
opt-in (default-on still needs the per-prefix dual-graph dispatch; the
+109% short-prefix kernel loss stands). τ shifts 3.547→3.597 because the
fast tier's ≤1e-4 verify drift flips occasional near-tie acceptances —
deterministic per config, and losslessness is the verifier's as always.
(2) **Flash: a measured LOSS** — e2e −2.7%, drafter +7%/step, reproducing
the contaminated-run signal: the drafter's long-ctx step cost is NOT
attention-dominated at this shape. Per the rmsnorm_fast precedent
(measured, rejected, deleted) the profile is REMOVED in this commit; the
M=N flash kernels it reused are untouched. (3) X loses like flash, with τ 3.116 — a flash×NB
compounding interaction (near-tie verify flips change committed prefixes,
then drafter divergence compounds), moot with flash deleted. Cross-check:
per-visit reproducibility is ±0.15% on spec walls and 3 of 4 e2e pairs;
the drafter EWMAs spread ±0.7–1.6% and the X e2e pair ±1.5% — traced
entirely to the wX1 plain-reference visit (11.65 s vs every other leg's
~11.3), not the spec side (8.50/8.51). Still far tighter than any ABBA
this campaign.

Note: e2e ratios here are NOT comparable to the round-30 §4a absolutes
(different harness: fresh-spec-server legs with a discarded warm pass + a
persistently-warm plain reference + legacy-k, vs measure_tau_ctx's
cold-discard protocol) — within-table comparisons only.

#### Caveats + owed

- (resolved 2026-08-19) The previously-owed e2e A/B landed as §4; the
  contaminated-run flash signal reproduced quiet and the profile is deleted.
- The NB kernel's +109% at prefix 512 means `TRITIUM_TREE_NB=1` must not be
  set for short-ctx serving; adoption (default-on) requires the per-prefix
  dual-graph dispatch.
- TB1 numbers use the bench's ~1/3-zero trits (round-16 continuity), which
  understates the format's byte edge at BitNet's real p=0.422 — disclosed by
  the printed byte percentages; the verdict does not depend on it.

**Still owed:** upstream llama.cpp Q2_0 rerun; MI300X/Metal sessions
(user-gated).

### 2026-08-09/16 adaptive spec stack @ 16f52f3f — governor, cost-model floors, batched fast tier, longctx drafter, truncate-reconcile

**Box state:** the 2026-08-09 rows (§1–§3) were measured on the quiet box at their
own commits (4190673…d073dc8, 73bcd00; per-section disclosure in the session
records). The 2026-08-16 truncate-reconcile ABBA (§4) started and ran quiet
(desktop graphics only, ~370 MiB across 3 processes, verified at launch); one
transient 4.2 GiB compute process (PID 481169, exited before identification) was
present at the end-of-run snapshot — the final short-ctx visits may carry it,
disclosed in §4c. The §5 microbench ran CONTENDED (a game co-resident, 6.5 GiB /
~50% util) — its 2.34× verdict margin is the signal, not the absolutes. RTX 4090,
driver 610.57.04, CUDA 13.3. Binary provenance §4: clean `git archive` exports of
8a808098 (baseline) and 16f52f3f (candidate), separate target dirs, binaries
verified distinct; the plain reference server is the candidate binary at default
env for every visit (the plain path is untouched by these commits).

**Engine state — what this entry adds on top of the 08-08 sweep:**

| lever | default | mechanism | measured win | scope |
|---|---|---|---|---|
| adaptive spec governor (round 28) | ON (`TRITIUM_SPEC_ADAPTIVE=0` kills) | τ-EWMA breakeven suppression + 4-token probe / 64 committed | long-ctx solo exact **2.42×** / fast+f16 **1.22×** over non-adaptive; dormant (1.000–1.002×) short-ctx + N=4 | solo + batched spec |
| cost-model floors (round 29 L7) | ON (`TRITIUM_SPEC_COST_FLOORS=0` reverts fixed) | measured (V+k·d)/P breakevens, tier-aware; probe-resync EWMA split from steady-state d | **+4–10%** over fixed floors in the collapsed band; fast-tier floors derive 2.15–2.95 (fixed 1.5 was too low; exact truth at the 3.0 clamp) | governor floor inputs |
| batched fast tier (round 29 L6) | `exact` (opt-in `TRITIUM_KERNEL_TIER=fast`) | L3b online-softmax fused body over `TreeCtrlAddr` (6 shims, 101→107 kernels) | batched spec fast-vs-exact **+22–40%** short / **+112–171%** long ctx (N∈{2,4}, both rungs); kernel pairs −58…−79% | paged/slots verify (completes RFC 0001's reach) |
| longctx drafter (round 29 L5) | recommended `--draft-model drafter-8L768-longctx.gguf` | seq-4096 retrain (root cause: the 1024-token training window) | long-ctx τ **2.85–3.08** vs s3's ~1.1; short-ctx ≤1% cost | drafter artifact (blut-side) |
| truncate-reconcile (682a0e7a) | ON (structural, no knob) | `truncate_kv` partial-match reconcile: a rejected draft suffix rewinds the drafter's KV watermark, the accepted-prefix KV survives — replaces reset + ctx-linear re-prefill on EVERY partial accept and probe | §4: long-ctx forced-on exact **+43–45%**, fast+f16 **+83%** (paired; 0.89→**1.62–1.65× vs plain**), governor-on **+45–52%**; drafter ms/token 32→3.0–3.8 (exact), 20–22→2.2 (fast) | solo + I0 spec drafter KV (batched enrollment: §5, not adopted) |

#### 1. Governor + floors (rounds 28–29, quiet box 2026-08-09)

measure_tau protocol (§2c/2c-long shapes of the 08-08 entry), ABBA at server
launch. Long-ctx 3776 solo, adaptive-vs-forced: exact 2.42×, fast+f16 1.22×;
short-ctx and batched N=4 dormant (1.000–1.002×). Losslessness pinned by
forced-collapse==plain-greedy gates on both paths. Cost-model floors: +4–10%
over the fixed floors in the collapsed band; the probe-resync split keeps
recovery reachable (4–5 full-accept probes by tier). At these commits the
governor landed 0.87–0.90× vs plain at long ctx (residual = probe cost) — §4
re-measures that residual after the reconcile.

#### 2. Batched fast tier (round 29, quiet box 2026-08-09)

tier_fast_slots_bench harness (N∈{2,4} × {dense,paged} × both rungs):
fast-vs-exact **+22–40% short ctx / +112–171% long ctx**; kernel pairs
−58…−79%; RFC 0001 gates green (isolated ≤1.62e-5, in-situ ≤6.4e-6, τ EXACTLY
pinned, committed streams identical across tiers).

#### 3. Longctx drafter verdict (round 29 addendum, quiet box 2026-08-09)

Long-ctx τ 2.85–3.08 (fast/exact) vs s3's ~1.1 (round-29 session shape;
the 08-08 §2c-long measured s3 at 1.23–1.35) — the 1024-token training
window was the collapse. Short ctx unchanged (≤1% e2e). Honest half at that
commit: forced-on long-ctx spec still lost e2e (0.65× exact / 0.76× fast+f16)
because breakeven τ rises with context — the finding that motivated §4.

#### 4. Truncate-based drafter reconcile (2026-08-16, quiet box, ABBA @ 16f52f3f; functional commit 682a0e7a)

Harness: measure_tau_ctx (the §2c protocol, shape-parametrized; 1-prompt
discarded warmup per server launch instead of a full discarded pass —
disclosed deviation). Spec server 8124 w/ `--draft-model
drafter-8L768-longctx.gguf`, plain reference 8125 (candidate binary, default
env, exact/f32) — every ratio below is vs that same reference. Old =
8a808098, New = 16f52f3f, order O N N O / N O O N per comparison.

**§4a. Long ctx (6 wt103 prefixes of 3,520 tokens, 256 greedy each):**

| config | e2e vs plain, Old (2 visits) | e2e vs plain, New (2 visits) | τ (old → new) | drafter ms/tok (old → new) |
|---|---|---|---|---|
| forced-on, exact/f32 | 0.615 / 0.617 | **0.882 / 0.893** | 3.901 → 3.901 (bit-pinned) | 32.2–33.2 → 3.0–3.8 |
| forced-on, fast+f16 | 0.889 / 0.901 | **1.648 / 1.624** | 2.553 → 2.553 (bit-pinned) | 19.5–22.0 → 2.2 |
| governor-on, fast+f16 | 1.078 / 1.124 | **1.638 / 1.630** | 3.76 → 3.67–3.86 | 14.1–15.4 → 3.7–3.8 |

Reading. (1) Where drafts are forced, τ is IDENTICAL to three decimals across
binaries — the reconcile changes drafter cost only, never a drafted token; the
exact-tier leg stayed 6/6 token-identical to plain greedy on both binaries
(fast-tier non-identity to the exact reference is the documented RFC 0001
trade, deterministic across visits). (2) The mechanism receipt is the
draft_token EWMA: the ~ctx-linear re-prefill term (~30 ms/tok at ctx≈3.6k) is
gone, leaving the drafter's own step cost. (3) **Long-ctx spec now WINS
outright: fast+f16 forced-on is 1.62–1.65× vs plain exact/f32** — it beats the
08-08 sweep's best long-ctx config (plain + `TRITIUM_KV=f16`, ≈1.47× on the
same basis). (4) The governor now agrees: suppression fell 830 → 535–569 of
the ~1,530 tokens per pass, probe-resync EWMA 10.2 → 3.4–3.5 ms, and governor-on
e2e matches forced-on (1.63×) — the residual-probe-cost regime (§1's
0.87–0.90×) is closed at this shape.

**§4b. Governor-on baseline note:** the Old governor-on rows (1.08–1.12×)
sit above §1's 0.87–0.90× because this run composes the longctx drafter +
cost-model floors (both post-date §1's residual measurement) and a different
prompt-set/day; the paired same-session columns are the claim, not the
cross-day absolutes.

**§4c. Short ctx (12 wt103 96-token prefixes, fast+f16, governor default):**
Old 1.255 / 1.345 → New 1.341 / 1.423 vs plain. Means +6%, but per-visit τ
ranged 5.0–6.4 (adaptive-k content variance) and the transient end-of-run
process may touch the last two visits — recorded as **no regression**, not a
win. Suppressed-plain counts fell 1,677–2,092 → 1,495–1,595 (cheaper probes
un-stick suppression at short ctx too).

Gates for §4 (all green before the bench): `cuda_truncate_kv_matches_fresh_prefill`
(truncate-then-append BIT-IDENTICAL to fresh append on the surviving prefix),
`cpu_truncate_kv_matches_fresh_prefill` (host-branch twin),
`truncate_reconcile_pins` (partial-accept drafts token-equal to a fresh
drafter; the old clean-overshoot stall pinned dead), spec_lookup 3/3,
batch_serve 8/8 (release, serial). Reviews: 682a0e7a PASS WITH NITS (F1 doc
accuracy — corrected in 2d9651ed's message; F2 host-branch test — added).

#### 5. Batched enrollment leg — measured, NOT adopted (2026-08-16, contended box)

`drafter_catchup_bench` (in-tree, `--ignored`): at the probe shape (ctx 4032,
gap 64, drafter pool N=4), a masked k=1 `draft_batch` catch-up loop costs
**222.6 ms median (3.48 ms/step)** vs **95.1 ms** for the enrollment path's
reset + M=4032 re-prefill + adopt — **re-prefill wins 2.34×**, so the batched
gap-close guard stays at gap ≤ 1 and multi-slot probe re-entries keep the
enrollment re-prefill. Box was contended (game co-resident); the legs were
same-process interleaved and the margin is 2.34×, but re-run the harness on a
quiet box before citing the absolutes.

#### Caveats

- §1–§3 numbers were measured at their own commits on 2026-08-09; §4–§5 at
  16f52f3f on 2026-08-16. No cross-day absolute comparisons — every claim is a
  same-session ABBA pair.
- §4's plain reference is the CANDIDATE binary (plain path untouched by
  682a0e7a/2d9651ed/16f52f3f — the diff is drafter-KV-management only); Old
  and New spec servers are clean-archive builds of their shas.
- §4 long-ctx fast+f16 "wins outright" is vs plain exact/f32 on the same
  visit; vs plain+f16 (the 08-08 §2b basis, ≈1.47×) the margin is ≈+11–12%,
  computed across entries — re-measure same-session before citing that number.
- The 1-prompt warmup (vs the full discarded cold pass of the §2c protocol)
  trades a sharper first-prompt tail for 2× wall time; spec-vs-plain ratios
  are within-visit and share it.

**Still owed:** upstream llama.cpp Q2_0 quiet-box rerun (unchanged); quiet-box
§5 absolutes; MI300X/Metal validation sessions (user-gated).

### 2026-08-08 final sweep @ 07b9d6a — the quiet-box ledger: absolutes + the adopted opt-in tiers

**Box state: QUIET.** RTX 4090, driver 610.57.04, CUDA 13.3. Co-resident the whole
sweep: desktop graphics only, 1.5–1.75 GiB total (kwin ~161–184 MiB, Xwayland,
plasmashell, firefox ~317 MiB, ghostty, krunner, one codex-desktop electron
gpu-process 192 MiB) — **zero compute jobs**, re-verified per section
(`nvidia-smi --query-compute-apps`). Binary provenance: all Tritium binaries built
from a clean `git archive 07b9d6a` export (`cargo build --release -p tritium-cli
-p tritium-serve --features tritium-cli/cuda,tritium-serve/cuda`); the compare-JSON
`git_commit` field is empty because the export has no `.git` — the sha is pinned
here. JSON/log artifacts: session scratchpad `final-*.{json,log,txt}` (outside the
repo; receipts dirs are gitignored).

**Engine state — the adopted opt-in levers (each measured below, scope stated):**

| lever | default | opt-in | measured win (this sweep) | scope |
|---|---|---|---|---|
| kernel tier (RFC 0001 + Amdt 1, L3b) | `exact` | `TRITIUM_KERNEL_TIER=fast` | spec verify wall −18% @ short ctx, **−61% (2.57×) @ ctx≈3.5–4k**; NOT token-identical to exact greedy | solo spec verify only; paged/slots stay exact; exact stays default+CI |
| KV dtype | `f32` | `TRITIUM_KV=f16` | plain decode **+46.7% @ ctx≈3.8–4k**; batched N=4 spec **+16.2%** | decode KV read path; lossless claim per L6 gates |
| LM head (ADR 0036 L2) | `f16` | `TRITIUM_LM_HEAD=i8` | plain decode +3–6% (noisy; see §2a) | UNDRAFTED decode only (spec path refuted in round 26) |
| weights packing | `tq2` | `TRITIUM_WEIGHTS=tq1` | **−628…−640 MiB** serve peak, token-parity; costs ~24% batched-spec throughput | capacity rung, not a speed rung |

#### 1. Plain decode + pp512 + ttft (the ledger command, all defaults)

Command: `tritium report compare --model ggml-model-i2_s.gguf --tokens prompt512.json
--backend cuda --prompt-len 512 --decode-steps 256 --warmup 16 --reps 3 --runs 5
--format json` @ 07b9d6a clean build. Tier `exact`, head `f16`, KV `f32` (all
defaults). Three back-to-back bundles:

| bundle | decode tok/s (3 reps) | median | pp512 tok/s | ttft p50 / p95 ms |
|---|---|---:|---:|---|
| run 1 | 206.6 / 211.0 / 211.8 | **211.0** | **22,714.7** | 22.55 / 23.55 |
| run 2 | 194.9 / 195.5 / 172.0 | 194.9 | 20,264.4 | 24.50 / 28.22 |
| run 3 | 176.0 / 172.0 / 177.6 | 176.0 | 19,885.3 | 25.38 / 26.61 |

The monotone decline across bundles on an idle 34 °C box is the documented
sustained-clock settle (07-18 entry), not contention; the four baseline visits
inside §2a's ABBA pairs landed 189–214. **Honest quiet-box absolutes @ HEAD:
decode 176–214 tok/s (bundle medians), pp512 19.9–23.0k tok/s, ttft p50
22.6–25.4 ms.** pp512 is 1.6–1.8× the same-session llama.cpp Q4_K_M line (§5).

#### 2a. `TRITIUM_LM_HEAD=i8` — plain decode, ABBA×2

Same ledger command ± the env var, order A B B A / B A A B (8 bundles):

| config | bundle medians tok/s | mean |
|---|---|---:|
| baseline f16 head | 189.1 / 211.0 / 213.8 / 207.0 | 205.2 |
| `TRITIUM_LM_HEAD=i8` | 219.2 / 230.4 / 212.3 / 203.9 | 216.4 |

**+5.5% mean / +3.2% median-of-medians**; pairwise spread −1.5%…+15.9%. Direction
confirms round 26's same-session +7.75%, but at this effect size the box's clock
variance dominates a 4-pair ABBA — recorded as +3–6%, not a sharper number.
pp512 unchanged within noise (i8 19.8–23.5k vs base 22.1–23.0k).

#### 2b. `TRITIUM_KV=f16` — plain decode at long ctx, ABBA

Ledger command with a real 3,900-token wt103 prompt file, `--prompt-len 3776`
(prompt 3776 + 16 warmup + 256 steps = ctx 4048, the 4096 cap; 3900 overflowed —
disclosed), order A B B A:

| config | decode medians tok/s (2 visits) | pp3776 tok/s | gain |
|---|---|---:|---:|
| KV f32 (default) | 75.6 / 70.8 | 7,453 / 7,068 | — |
| `TRITIUM_KV=f16` | **108.3 / 106.5** | 7,135 / 7,110 | **+46.7%** |

The L6 +47% long-ctx claim reproduces exactly on the quiet box. Reps within each
visit were tight (±1.5%); prefill unchanged within noise.

#### 2c. `TRITIUM_KERNEL_TIER=fast` — solo spec decode (short ctx)

Harness: measure_tau.py (12 wt103 96-token prefixes, 256 greedy each) against two
clean-HEAD `tritium-serve` instances (spec 8124 w/ `--draft-model
drafter-8L768-s3.gguf`, plain 8125; both `--backend cuda --raw-tokens --eos
4294967295`); per leg the cold pass is discarded, warm recorded; leg order
E F C C F E (E=exact spec, F=fast spec, C=fast+f16 spec; the plain reference
server is always default/exact). ctx spans 96→352 per prompt — note this is a
SHORTER shape than round 27's ctx≈512 point.

| spec config (verify tier / KV) | identical to exact greedy | tok/verify (tau) | spec wall (2 legs) | e2e vs plain (exact) |
|---|---|---:|---|---:|
| exact / f32 | **12/12 + 12/12 (lossless)** | 3.575 | 10.29 s / 9.56 s | 1.151× / 1.203× |
| fast / f32 | 3/12 + 3/12 | 4.183 | 7.74 s / 8.56 s | **1.511× / 1.446×** |
| fast / f16 | 4/12 + 4/12 | 3.726 | 7.69 s / 7.80 s | 1.488× / 1.465× |

Fast-vs-exact spec wall at this shape: **+21.8%** (19.85 s vs 16.30 s summed) —
well under round 27's +57–60% at ctx≈512; shape and box differ, the ctx≈3.5k
point below is where L3b's claim lands. **Losslessness, honestly: the exact tier
is spec-lossless (24/24 prompts token-identical to plain greedy, deterministic
across legs). The fast tier is NOT token-identical to exact greedy (3/12,
deterministic) — RFC 0001 drift-tier semantics: ~1e-6 kernel drift flips argmax
at near-ties over 256-token horizons.** Fast tau 4.183 vs exact 3.575 is measured
on the divergent continuations, not a like-for-like acceptance gain. Single-stream
throughput at this shape: plain ~260–270 tok/s (96-tok prompts — shorter-ctx than
§1's 512), exact spec ~299–321, **fast+f16 spec ~394–399 tok/s**.

#### 2c-long / 4. Best-composed single-user config at long ctx

Same leg protocol, 6 wt103 prefixes of 3,520 tokens (denser doc walk `i%191==7`;
the standard walk has only 3 long-enough docs), 256 greedy each → ctx 3520→3776.
Order E C C E:

| spec config | identical | tau | spec wall | e2e vs plain (exact/f32) |
|---|---|---:|---|---:|
| exact / f32 | 6/6 + 6/6 | 1.228 | 56.73 s / 59.43 s | **0.373× / 0.369×** |
| fast / f16 | 0/6 + 0/6 | 1.354 | 23.65 s / 21.59 s | 0.979× / 0.974× |

Two honest findings. (1) **The fast tier's long-ctx verify win is real: the fast
spec wall is 2.57× faster than exact (45.24 s vs 116.16 s summed, −61%)** — the
round-27 +188–240% e2e claim direction confirmed at its ctx point (fast-vs-exact
e2e here +157–175%). (2) **Solo spec is NOT a long-ctx win with this drafter:
tau collapses to 1.23–1.35, exact spec is a 2.7× slowdown vs plain, fast+f16
only recovers to parity (0.97×).** The best single-user long-ctx config today is
therefore **plain decode + `TRITIUM_KV=f16` (+ i8 head): 106–108 tok/s vs the
70–76 plain-f32 baseline** — not spec. At short/standard ctx the best config IS
fast+f16 solo spec (≈1.47–1.51× plain, §2c).

#### 3. Batched multi-slot spec decode — the quiet-box headline recapture

Harness: measure_multi.py (Round-25 protocol: server `--batch-slots 4`, shared
draft k — the shipped default incl. the 9930062 bucket snap — N concurrent
192-token wt103 streams, drafted-vs-undrafted ABBA×2 at server-launch level,
5 samples/visit → n=20/config, p50). Clean-HEAD serve, tier exact, KV f32:

| N streams | drafted p50 agg tok/s | undrafted p50 agg tok/s | speedup | tok/verify p50 |
|---|---:|---:|---:|---:|
| 2 | **309.7** (min 254.9, max 433.4) | 164.3 | **1.88×** | 2.08 |
| 4 | **496.4** (min 428.5, max 526.6) | 307.5 | **1.61×** | 2.01 |

These are the quiet-box numbers owed since Round 25. Sample spread is ~7× tighter
than the contended T8 rows (min/max within ±15% vs 149–506). Vs Round 25's
lighter-box 2.44×/1.52×: N=2 lands lower (the undrafted baseline is much faster
on a quiet box), N=4 lands higher — the drafted-vs-undrafted ordering is stable
and the N=4 aggregate ~496 tok/s is the serve headline.

N=4 + `TRITIUM_KV=f16` (same harness, drafted-only ABBA×2 f32-vs-f16, n=20):
**544.7 vs 468.8 p50 → +16.2%** — the L6-stage-2 +12–14% claim holds (slightly
exceeded) on the quiet box.

#### 5. Competitor line (documented invocation, same session)

`llama-bench -m /home/brianklam/models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf
-p 512 -n 128 -ngl 99`, fork build 41a666dac (8943) — same binary as the 07-11
reference:

| engine | model | pp512 tok/s | tg128 tok/s |
|---|---|---:|---:|
| llama.cpp CUDA | Qwen3.5-4B Q4_K_M (2.54 GiB) | 12,765.52 ± 1,290.62 | 204.19 ± 3.23 |

Stable vs 07-11 (12,281 ± 828 / 200.1) and the T8 rerun (12,608 / 199.8).
Different models (2.4B ternary vs 4.2B Q4) — bandwidth normalization is the
comparison basis. **Still owed:** the upstream-master Q2_0 quiet-box rerun from
2026-07-30 — the upstream build (1212) and the Q2_0 gguf remain absent on this
box; not improvised.

#### 6. TQ1 capacity rung (batched-spec serve VRAM) — quick quiet-box recapture

Harness: measure_tq1.py (batched-spec serve `--batch-slots 4` + drafter,
`TRITIUM_WEIGHTS=tq2/tq1/tq1/tq2` ABBA at launch, peak per-process VRAM at 0.5 s
during a warm 4-stream sample, then a 256-token single-stream greedy parity
completion per visit):

| weights | peak serve VRAM (2 visits) | 4-stream agg tok/s | parity |
|---|---:|---:|---|
| tq2 (default) | 7,592 / 7,610 MiB | 504.0 / 501.2 | reference |
| tq1 | **6,964 / 6,970 MiB (−628…−640 MiB)** | 378.9 / 396.4 | **token-identical, all visits** |

The capacity rung reproduces quiet (T8's −672 was contended-peak). NEW fact the
quiet box finally settles (round 16's open item): **tq1 costs ~24% batched-spec
throughput** — it is a capacity/VRAM rung, not a free lever.

#### Caveats

- Decode absolutes move ~±10% with the box's sustained-clock state even quiet;
  every relative claim above is same-session ABBA.
- §2c/§2c-long single-stream tok/s are 96-tok- and 3,520-tok-prompt shapes, not
  the §1 512-prompt ledger shape — don't cross-compare the absolutes.
- The fast tier's non-identity to exact greedy (3/12 short, 0/6 long) is the
  documented RFC 0001 trade; exact remains the default and the lossless tier.
- Long-ctx spec numbers are drafter-limited (tau ≈ 1.3); a long-ctx-competent
  drafter would move §2c-long, not the kernels.



**Box state (disclosed up front): NOT quiet.** A parallel session's
`salt_distill_heldout` test (Tritium repo, PID 2609694) ran on-GPU for the whole sweep,
growing 388 MiB → 4.3 GiB and pulling 27–100% GPU util between my runs; desktop stack
~1.05 GiB (kwin/Xwayland/plasmashell/firefox/ghostty). RTX 4090, driver 610.57.04,
CUDA 13.3. Every number below carries that contention; relative pairs were measured
same-session order-alternated (ABBA) so both sides share it. **Quiet-box rerun of the
absolutes is owed.** Binary provenance: the shared working tree held uncommitted
parallel-session changes (including `tritium-cli/src/report.rs`), so all Tritium
binaries were built from a clean `git archive dde0c61` export
(`cargo build --release -p tritium-cli -p tritium-serve --features
tritium-cli/cuda,tritium-serve/cuda`), not the dirty tree.

#### 1. Plain decode + pp512 (the ledger command)

Command: `tritium report compare --model ggml-model-i2_s.gguf --tokens prompt512.json
--backend cuda --prompt-len 512 --decode-steps 256 --warmup 16 --reps 3 --runs 5
--format json` @ dde0c61 (clean-HEAD build). Tokens: first 512 ids of a wt103 doc
(BOS 128000). Run twice per protocol; both bundles recorded because the co-resident
distill was RAMPING between them (run 1 caught the lighter window).

| bundle | decode tok/s (3 reps) | median | pp512 tok/s | ttft p50 / p95 ms |
|---|---|---:|---:|---|
| run 1 (07:28 UTC, distill just starting) | 216.1 / 211.8 / 215.0 | **215.0** | **22,298.6** | 22.96 / 23.88 |
| run 2 (07:29 UTC, distill ramping) | 191.7 / 191.2 / 191.1 | 191.2 | 19,603.8 | 26.05 / 26.74 |

pp512 at HEAD is **19.6–22.3k tok/s even contended** — up from the 2026-07-18 ledger
line of 12,275 (v3-attention era) and above the same-day llama.cpp Q4_K_M reference
(12,608, below). Decode 191–215 sits inside the documented contention band around the
221–222 quiet-class entries; treat the decode absolutes as contended, not a regression.
JSON artifacts: session scratchpad `compare-head-run{1,2}.json` (receipts dir pattern
`docs/receipts-ws-*` is gitignored; artifacts live outside the repo).

#### 2. Solo speculative decode — lossless, tau, e2e

Harness: `~/blut/scripts/measure_tau.py` (12 wt103 prose prefixes, 256 greedy tokens
each) against two clean-HEAD `tritium-serve` instances: spec on 8124
(`--draft-model ~/blut/data/drafter-8L768-s3.gguf`), plain on 8125; both
`--model ggml-model-i2_s.gguf --backend cuda --raw-tokens --eos 4294967295
--max-completion-tokens 4096`. Cold pass discarded (prompt 0 paid ~22 s of JIT on both
servers); the warm pass is the record:

| metric | value |
|---|---:|
| losslessness | **12/12 prompts token-identical** (spec == plain greedy), both passes |
| tok/verify (tau) | **3.575** (3,057 committed / 855 verifies) |
| e2e wall | spec 10.02 s vs plain 11.08 s → **1.105×** |

Per-prompt spread was contention-noisy (spec 0.47–1.39 s per 256 tok); the lossless
count and tau are contention-immune, the 1.105× e2e is same-session relative.

#### 3. Batched multi-slot spec decode (with the 9930062 bucket snap)

Harness: measure_multi.py class (session scratchpad copy pointed at the clean-HEAD
`tritium-serve`), Round-25 protocol: server at `--batch-slots 4`, shared draft k
(shared draft-k, the shipped default — the historical `TRITIUM_MULTI_K` knob no longer exists; per-slot k was refuted at 9930062),
N concurrent 192-token wt103 streams, drafted-vs-undrafted ABBA×2 at the
server-launch level, 5 samples/visit → n=20 per config, p50.

| N streams | drafted p50 agg tok/s | undrafted p50 agg tok/s | speedup | tok/verify p50 |
|---|---:|---:|---:|---:|
| 2 | 189.8 (min 149.5, max 408.7) | 102.9 | **1.84×** | 2.09 |
| 4 | 276.2 (min 244.0, max 506.3) | 217.8 | **1.27×** | 2.01 |

Honest reading: the Round-25 headline (2.44×/1.52×, tok/verify 2.63) was measured on a
lighter box (5 GB idle squatter, no active compute); this session's absolutes are
~half Round-25's and the sample spread (149–506 tok/s at fixed config) shows the
distill co-resident dominating variance. These rows re-confirm drafted > undrafted at
HEAD under load; they do NOT cleanly re-measure the bucket-snap +6.6% (same-session
relative claim in 9930062) and are not comparable to Round-25's absolutes.
Quiet-box recapture owed for the headline table.

#### 4. TQ1 capacity rung (batched-spec serve)

Harness: same batched-spec server (`--batch-slots 4` + drafter), `TRITIUM_WEIGHTS=tq2`
vs `tq1`, ABBA at launch level (tq2/tq1/tq1/tq2); peak per-process VRAM sampled from
`nvidia-smi --query-compute-apps` at 0.5 s during a warm+measured 4-stream sample,
then a single-stream 256-token greedy parity completion per visit.

| weights | peak serve VRAM (2 visits) | parity (256-tok greedy) |
|---|---:|---|
| tq2 (default) | 7,648 / 7,636 MiB | reference |
| tq1 | **6,976 / 6,976 MiB (−672 MiB)** | **token-identical to tq2, all visits** |

Two new facts at HEAD: the batched-spec path now ACCEPTS `TRITIUM_WEIGHTS=tq1`
(round 15's "batch/tree reject loudly in v1" is lifted), and the capacity win holds
end-to-end in the serve process (−672 MiB ≈ the −18%-of-ternary-bytes rung from
round 15). No speed claim: the contended 4-stream aggregates spanned 259–431 tok/s
*within the tq2 pair alone*, so tq1-vs-tq2 throughput stays an open quiet-box item
(round 16's "re-bench uncontended" note stands).

#### 5. Competitor line (documented invocation, re-run same-day)

`llama-bench -m /home/brianklam/models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf -p 512
-n 128 -ngl 99`, fork build 41a666dac (8943) — the same binary as the 2026-07-11
reference line. Ran in the lightest window of the session (desktop ~1.4 GiB, distill
at 388 MiB just starting):

| engine | model | pp512 tok/s | tg128 tok/s |
|---|---|---:|---:|
| llama.cpp CUDA | Qwen3.5-4B Q4_K_M (2.54 GiB) | 12,607.84 ± 1,173.85 | 199.79 ± 3.65 |

Stable vs the 07-11 reference (12,281 ± 828 / 200.1 ± 6.5). **Still owed:** the
upstream-master Q2_0 quiet-box rerun from the 2026-07-30 entry — the upstream build
(1212) and the Q2_0 gguf are not present on this box; not improvised.

### 2026-07-30 — llama.cpp Q2_0 CUDA merged upstream (PR #25707, merged TODAY) — first same-box head-to-head

The mainstream-ternary-CUDA gap closed this morning. Same box (RTX 4090, ~5.3 GB
co-resident desktop load — CONTENDED, both engines equally), same day, ABBA
interleaved. llama.cpp = upstream master 5f55650a7 (build 1212), fresh CUDA build.
Tritium @ HEAD via the ledger decode command (256 steps, 16 warmup).

| engine | model | weights | decode tok/s (spread) | eff. weight stream |
|---|---|---|---:|---:|
| Tritium CUDA | BitNet 2B4T ternary I2_S | 1.71 GiB | 264-281 (median ~277) | ~474 GiB/s |
| llama.cpp CUDA (NEW Q2_0) | Qwen3.5-4B Q2_0 g64 | 1.42 GiB | 223-264 (mean-of-runs 248) | ~352 GiB/s |
| llama.cpp CUDA (prior ledger line) | Qwen3.5-4B TQ2_0 | 1.35 GiB | 24.0 | ~32 GiB/s |

Reading, honestly: llama.cpp's new Q2_0 CUDA path is REAL — a ~10x jump over its
TQ2_0 line, landing within ~25-35% of Tritium's bandwidth-normalized decode
efficiency on its first day. Tritium still leads (~474 vs ~352 GiB/s effective),
but "only ternary CUDA engine" is retired as a claim as of 2026-07-30. Different
models/architectures (2.4B BitNet vs 4B Qwen) — bandwidth normalization is the
comparison basis; a same-model run (Bonsai Q2_0 vs its Tritium import via the new
q2_0 reader) is the clean follow-up. pp512 for Q2_0: 12,199 +- 1,109 (llama.cpp
MMQ-class; Tritium's ledger pp512 12,275 sustained — parity band).

Repro: `llama-bench -m qwen35-4b-q2_0.gguf -p 512 -n 128 -ngl 99` (master
5f55650a7); Tritium ledger command above. Contention disclosed; quiet-box rerun
owed before any published table.



<!-- Newest first. Each entry: date, git commit, environment line, table, JSON path/attachment. -->

### 2026-07-12 — IMMA prefill live (ADR 0026 Track P steps 1-4)

Command: `tritium report compare --model ggml-model-i2_s.gguf --tokens <ids> --backend cuda
--prompt-len 512 --decode-steps 256 --warmup 16 --reps 3 --runs 5` @ git 5075683-era tree
(Track P complete through the dispatch), RTX 4090, driver 610.43.02, co-resident: kwin only.

| metric | value | vs 2026-07-11 baseline |
|---|---:|---|
| pp512 prefill | **1,969.7 tok/s** (p50 260.0 ms) | 1,068 → **+84%** |
| decode (256 steps after the 512-token prefill, ctx 512→768) | 221.5 tok/s median | shape differs from the 6-token-prompt baseline (273-276); not comparable directly |
| decode (32 steps, 64-token prompt) | 317.9 tok/s | — |

Numerics: IMMA prefill is BIT-IDENTICAL to dp4a (gate: 128,256 logits to_bits, one-shot ==
4×128-chunked == dp4a). The ≥6,000 pp512 campaign gate remains open — gap analysis in
OPTIMIZATION-LOG round 22.

### 2026-07-18 — rev-4 IMMA + v2 prefill attention (the pp512 gate falls)

Command: `tritium report compare --model ggml-model-i2_s.gguf --tokens <ids> --backend cuda
--prompt-len 512 --decode-steps 256 --warmup 16 --reps 3 --runs 5` @ b20f78f-era tree
(rev-4 IMMA JIT a3e8a49 + v2 attention 98ab046 + nit fold b20f78f), RTX 4090, driver
610.43.02, co-resident: kwin + one idle tritium-serve holding 9.1 GB VRAM (0% compute).

| metric | value | vs 2026-07-12 | vs 2026-07-11 baseline |
|---|---:|---|---|
| pp512 prefill | **9,068.8 tok/s** (p50 56.4 ms) | 1,969.7 → **4.6×** | 1,068 → **8.5×** |
| decode (256 steps after the 512-token prefill) | 222.4 tok/s median | 221.5 → unchanged | — |

Run-to-run clock variance puts pp512 at 8,400–9,100 across the session (the order-alternated
5-run ttft matrix measured p50 60.6/60.7 ms on the A/A2 pair); the compare-bundle line above
is the reproducible-command artifact. **The ADR 0026 ≥6,000 gate is PASSED** (stretch 10,000
within reach — attention is now 74% of prefill; see OPTIMIZATION-LOG round 23).

Config isolation (same command, env kill switches, order-alternated ABBA):

| config | p50 ms | pp512 tok/s |
|---|---:|---:|
| dp4a + rev-1 attention (`TRITIUM_IMMA_PREFILL=0 TRITIUM_ATTN_V2=0`) | 318.3–320.4 | ~1,600 |
| rev-4 IMMA only (`TRITIUM_ATTN_V2=0`) | 126.9–127.1 | ~4,030 |
| v2 attention only (`TRITIUM_IMMA_PREFILL=0`) | 225.6–259.0 | ~2,100 |
| both (defaults) | 60.6–60.7 | **~8,440** |

Numerics: BOTH legs are bit-identical to their predecessors by gate (rev-4 IMMA: to_bits vs
dp4a over 128,256 logits incl. one-shot == 4×128 chunked; v2 attention: to_bits vs rev-1 per
(row, head), 6 regimes incl. the ctx=3584 shared-budget boundary) — zero numerics re-gating.

Methodology caveat learned this session: an earlier run measured a STALE release binary
(pre-commit) and reported flat results — `nsys` kernel names are the cheap staleness check
(`gqa_attention_batch_v2_f32` + r4-keyed tune files must appear).

### 2026-07-18 (later) — v3 Q-blocked attention: past the 10,000 stretch

Command: same compare bundle @ v3 tree (Q-blocked attention on top of rev-4 IMMA), RTX 4090,
co-resident: kwin + the idle 9.1 GB serve.

| metric | value | note |
|---|---:|---|
| pp512 prefill | **12,274.7 tok/s** (p50 41.9 ms, 20-run bundle) | sustained-clock ABBA pairs spanned 12,103–16,893 (30.3–42.3 ms) across the box's clock states |
| v3 vs v2 attention (ABBA, same session) | **2.2–2.8×** | the trustworthy relative number; v2 measured 5,482–6,032 in the same session |
| decode | v3 on/off ABBA: 145.6/148.3 and 108.7/156.5 | statistically indistinguishable — v3 is prefill-only; today's absolute decode (~130–156) is depressed vs yesterday's 222.4 by box state, not code |

**The 10,000 stretch goal is PASSED at the slow end of the spread; the fast end (16,893)
exceeds the llama.cpp Qwen3.5-4B Q4_K_M reference (12,281).** v3 blocks 8 query rows per
(block, head): staged K feeds all 8 dot chains, each V load folds into 8 accumulators, scores
return to the global scratch (lifting v2's ctx cap — v3 has none). Bit-identical to rev-1/v2
per (row, head) by to_bits gate (staircase, BQ tails, ctx 3809 > the v2 cap, both dtypes).
nsys @ v3: attention 45% (~22.6 ms) / IMMA GEMM 43% — the profile is now balanced; further
pp512 gains need both (cp.async smem swizzle on IMMA; deeper Q-blocking or the numerics-RFC
flash rewrite on attention).

### 2026-07-11 head-to-head (reference)

From `docs/research-ternary-sota-mid2026.md` §7, Tritium @ a80e185 — the pp512 gap to
mainstream was 11.5×; at 2026-07-18 it is **1.35×**:

| engine | model | decode tok/s | pp512 tok/s |
|---|---|---:|---:|
| Tritium CUDA | BitNet 2B4T (ternary I2_S) | 301.4–302.8 | 1,068 |
| llama.cpp CUDA | Qwen3.5-4B Q4_K_M | 200.1 ± 6.5 | 12,281 ± 828 |
| llama.cpp CUDA | Qwen3.5-4B TQ2_0 | 24.0 ± 1.6 | 107 ± 5 |
| bitnet.cpp CPU (14t) | BitNet 2B4T I2_S | 23.1 ± 0.1 | ~203 |

(Quiet-box re-baseline at HEAD b701c53, 2026-07-12, 512-step decode ×3: 273.2–275.9 tok/s —
the 301–302 line above was measured at 256 steps under lighter desktop load; both are inside
the documented contention spread. The pp512 line predates the Track P work.)

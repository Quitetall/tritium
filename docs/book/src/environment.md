# Environment variables

Every knob is loud-reject: a typo'd value fails at startup (or warns and
reads as off for the diagnostic flags), never silently does the wrong thing.
Boolean knobs take `1`/`0`; unset = the stated default.

## Server configuration (`tritium-serve`)

Every CLI flag has a `TRITIUM_*` twin; precedence is
`defaults < --config JSON < TRITIUM_* < CLI flags`. The full flag list is
`tritium-serve --help`; the twins are the flag name upper-snake-cased
(`--batch-slots` ⇄ `TRITIUM_BATCH_SLOTS`, `--max-completion-tokens` ⇄
`TRITIUM_MAX_COMPLETION_TOKENS`, …). Beyond the twins:

| Variable | Purpose |
|---|---|
| `TRITIUM_CONFIG` | Default `--config` path (strict JSON overlay). |
| `TRITIUM_AUTH_TOKEN` / `TRITIUM_AUTH_TOKENS` | Bearer auth; **required** for a non-loopback `--host`. `TOKENS` is comma-separated for rotation. |

## Performance rungs (engine, read at model build)

These select kernels/precision at model-build time — one setting per
process. All are measured in [`docs/BENCHMARKS.md`](https://github.com/Quitetall/tritium/blob/main/docs/BENCHMARKS.md);
the defaults are the lossless/bit-exact configuration.

| Variable | Values (default first) | What it does |
|---|---|---|
| `TRITIUM_KERNEL_TIER` | `exact` \| `fast` | RFC 0001 numerics tier. `fast` swaps the spec-verify attention for fused/node-blocked one-pass kernels (≤1e-4 vs exact; not token-identical). The long-ctx spec headline runs `fast`. |
| `TRITIUM_KV` | `f32` \| `f16` \| `i8` \| `t2` | KV-cache dtype ladder. `f16` is the measured long-ctx win (+47% plain decode at 4K); `i8`/`t2` are capacity rungs. |
| `TRITIUM_LM_HEAD` | `f16` \| `i8` | LM-head table dtype. `i8` is +3–6% undrafted decode, ppl-gated. |
| `TRITIUM_WEIGHTS` | `tq2` \| `tq1` | Weight packing. `tq1` is a capacity rung (−18% weight VRAM, ~parity solo decode, costs batched-spec throughput). |
| `TRITIUM_TREE_NB` | `auto` \| `1` \| `0` | Node-blocked tree-verify dispatch. `auto` replays the node-blocked capture at prefix ≥ 1536 and the fused one below (dual graphs per bucket); `1`/`0` force/kill for A/B. |
| `TRITIUM_SPEC_ADAPTIVE` | `1` \| `0` \| `force` | The adaptive spec governor (suppress drafting below breakeven, probe to recover). `0` = always draft; `force` = testing aid (every verify counts collapsed). |
| `TRITIUM_SPEC_COST_FLOORS` | `1` \| `0` | Measured (V+k·d)/P breakeven floors vs the fixed fallbacks. |
| `TRITIUM_DRAFT_K` | `adaptive` \| `legacy` | Draft-length policy (acceptance-EWMA adaptive vs fixed 6). |
| `TRITIUM_DRAFT_CHAIN` | `1` \| `0` | Device-chained drafting (one host round-trip per k-token draft). |

## Tuning / kill switches (engine)

| Variable | Values | What it does |
|---|---|---|
| `TRITIUM_IMMA_TUNE` | `tune` \| `load` \| `off` | IMMA prefill tile policy (disk-cached autotune / load-only / disable shadows). |
| `TRITIUM_IMMA_PREFILL` | `1` \| `0` | IMMA tensor-core prefill kill switch. |
| `TRITIUM_IMMA_MIN_M` | integer (32) | Prefill M at/above which IMMA dispatches. |
| `TRITIUM_ATTN_V2` / `TRITIUM_ATTN_V3` | `1` \| `0` | Prefill attention generation kill switches. |
| `TRITIUM_PREFILL_CHUNK` | integer (128) | Serve-side chunked-prefill size. |
| `TRITIUM_KV_F16` | `1` \| `0` | Legacy alias for `TRITIUM_KV=f16`. |
| `TRITIUM_WGPU_ADAPTER` | name substring | wgpu adapter selection; an unmatched substring FAILS (never silently picks another GPU). |

## Diagnostics (engine; `1`/`0`, default off)

| Variable | What it does |
|---|---|
| `TRITIUM_SPEC_STATS` | Per-request spec-decode stats on stderr. |
| `TRITIUM_TREE_TRACE` | Per-phase tree-verify timing trace. |
| `TRITIUM_TREE_DEBUG` | Tree-graph capture logging. |
| `TRITIUM_TREE_EAGER` | Bypass the captured tree graph (~600 eager launches per verify — debugging only). |
| `TRITIUM_TREE_ATTN_DUMP` | Per-layer attention-output snapshots (the RFC 0001 in-situ drift seam). |

## Tooling

| Variable | What it does |
|---|---|
| `TRITIUM_MODEL_DIR` | The shared model cache (`~/.cache/tritium-models` default) — where `tritium pull` downloads and every report/bench harness looks. `TRITIUM_MODEL_CACHE` is the honored legacy alias. |
| `HF_TOKEN` | HuggingFace bearer token for `tritium pull` on gated repos. |
| `TRITIUM_CORPUS` | Evaluation corpus path for the ppl harnesses (WikiText-2 basis). |

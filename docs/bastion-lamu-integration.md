# BASTION × LAMU integration (ADR 0014 boundary, implemented surface)

Status: Tritium side **live** (2026-07-06); LAMU backend scaffolded; drafter
marginals pending (python, lucebox-hub).

## The seam

Per ADR 0014, Tritium is the **target verifier**; the drafter and the
orchestration/budget loop are external. LAMU's architecture (its ADR 0033)
talks to every backend through a port, so the boundary is HTTP:

```
LAMU orchestrator                    tritium-serve --backend cuda
────────────────────────────────    ─────────────────────────────────────────
POST /v1/tree/session                reset + batched prefill of the prompt
  {"prompt_tokens":[...]}       →      → {"pending_token": t1}
loop:
  drafter → block marginals
  build best-first tree rooted t1
  POST /v1/tree/verify               one batched tree forward, greedy accept
    {"tokens":[t1,...],               walk, KV promote, watermark += L
     "parents":[-1,...]}        →      → {"committed":[t2..t_{L+1}]}
  t1 = committed.last()
```

- **One session at a time** (the serve worker owns one model); a
  `/v1/chat/completions` request invalidates the session.
- Node 0 of every tree MUST be the current pending token (`parents[0] == -1`,
  `parents[i] < i`); duplicate sibling tokens allowed, first match wins.
- `committed` is never empty (full draft reject ≡ one plain greedy step), and
  its last element is the next pending token — the orchestrator carries no
  other state.
- Errors: 400 malformed tree, 501 backend without the CUDA resident decoder,
  429 queue full, 503 draining.

Validated end-to-end on the wire (BitNet 2B4T, RTX 4090): session pending ==
reference first token; perfect 3-draft chains commit 4 tokens per round-trip;
the committed stream is token-identical to plain greedy; malformed trees 400.
The same losslessness is gated in-repo by `cuda_tree_verify_greedy_lossless`.

## LAMU side

`TritiumBackend` (lamu-core/src/backends/tritium.rs) spawns
`tritium-serve --model <gguf> --backend cuda --port <p>`, probes `/healthz`,
and passes chat through the OpenAI surface like every other backend. The
BASTION loop is a LAMU-side driver that owns the drafter and calls the two
tree endpoints above.

## Remaining (in dependency order)

1. **Drafter marginals** (python, `lucebox-hub/dflash`): the DFlash server
   currently runs its own draft+verify loop against its own target. BASTION
   needs it to expose one forward's position-wise marginals for a block of B
   future positions, e.g. `POST /draft {"context_tokens":[...], "block":B}
   → {"topk_tokens":[[...];B], "topk_probs":[[...];B]}`. That plus the tree
   builder (best-first over Π q_k, roofline budget N from
   `bitnet_2b4t_decode_ceiling`) completes the loop.
2. **Speedup gate** (ADR 0014 Pe): end-to-end tok/s ≥ target factor vs the
   345 tok/s AR baseline once (1) exists.
3. **Sampling accept rule** (Tritium): the greedy walk is live; the
   speculative-sampling rule (accept prob min(1, p/q), residual sampling)
   needs drafter probabilities on the wire — extend `/v1/tree/verify` with
   optional per-node `q` once (1) lands, plus the K-seed distribution gate.

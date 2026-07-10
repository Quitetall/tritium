# 0037 — Qwen-family arch: QKV-bias + QK-norm + explicit head_dim  (serves: ADR 0020, reaches the 32B/Qwen targets)

## Goal
Extend the general inference engine (0035) to Qwen2/2.5 (**QKV bias**) and Qwen3 (**QK-norm** +
a **head_dim decoupled** from `n_embd/n_head`), un-gating the guards 0035 added. Gate: small
Qwen2.5 + Qwen3 token-exact vs `transformers`.

## What shipped
- `TransformerBlock` gains `q_bias/k_bias/v_bias` (Qwen2/2.5) and `q_norm/k_norm` (Qwen3 per-head
  RMSNorm), each a `Vec<f32>` (empty = skip). The block forward applies bias per output channel
  right after the q/k/v projections, then per-head RMSNorm over `head_dim` **before RoPE** — the
  Qwen3 order. Free fns `add_bias` / `qk_norm` (crates/tritium-nn/src/layers/transformer_block.rs).
- `ModelConfig.head_dim` is now an **explicit field** (Qwen3: `n_head·head_dim` may exceed
  `n_embd`). `from_gguf` derives `n_embd/n_head`; `from_hf_config` reads the explicit `head_dim`
  (default `n_embd/n_head`).
- `load_hf` **detects** `qkv_bias` (a `.q_proj.bias` weight) and `qk_norm` (a `.q_norm.weight`)
  from the loaded tensors and enables them (was: reject); `build_standard_model` fetches the
  bias/QK-norm tensors when the flags are set.

## Gate (met)
`tests/hf_inference.rs` — parametrized `assert_conforms`, three `#[ignore]`d gates, all
**16/16 greedy token-exact** vs transformers with last-row rel-err < 1e-3:
- SmolLM2-135M (Llama, 2.0e-6) — no regression.
- **Qwen2.5-0.5B (QKV-bias, 4.2e-6).**
- **Qwen3-0.6B (QK-norm + explicit head_dim, 2.3e-6).**
References: `tools/reference/{qwen25_0.5b,qwen3_0.6b}_ref.json` (gen: `tools/gen_hf_logits.py`).
22 nn lib + 5 layers integration tests green; fmt + clippy `-D --all-targets` clean.

## Notes / follow-ons
- `load_salt` (0036) still targets standard (no-bias) models; it now also detects+rejects Qwen
  bias/QK-norm bundles (the detection load_hf does) rather than silently building bias-less.
- SALT-quantized Qwen inference (ternary 2D + fp bias/QK-norm) is a later extension.
- Next: **0038** — SALT-aware distillation trainer (the core loop).

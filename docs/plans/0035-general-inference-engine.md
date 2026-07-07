# 0035 — General inference engine: config-driven standard-transformer forward  (serves: ADR 0020 step 1, the keystone)

## Goal

Make Tritium run a model it was **not** hardcoded for. Today the runner is BitNet-2B4T
only (GGUF I2_S, squared-ReLU MLP, `attn_sub_norm`/`ffn_sub_norm`, tied LM head, fixed GGUF
tensor names). This plan adds a **config-driven architecture descriptor** + a **HF-safetensors
loader** + a **generalized decoder block** (SwiGLU MLP, optional sub-norms, untied LM head)
so a standard-transformer fp model (Llama-family: SwiGLU / GQA / RoPE / RMSNorm) loads from
its `config.json` + safetensors and runs a CPU forward.

**Success criterion:** a small, ungated Llama-architecture model (**SmolLM2-135M-Instruct**)
loads via the new path and its next-token greedy argmax is **token-exact** vs a
`transformers` reference over a fixed prompt (and per-position logit rel-err < 1e-3). This is
the keystone that unblocks teacher-caching + the ternary student forward (ADR 0020 steps 2–6).

**Scope guard.** fp (dense) forward of a **standard** transformer only. Non-goals here
(explicit follow-on plans): SALT multi-plane inference loader (0036), attention QKV **bias**
(Qwen2/2.5), **QK-norm** (Qwen3), SSM/`linear_attn` (Qwen3.6), MoE. The `ArchSpec` descriptor
this plan adds carries the flags for those so later plans only flip a flag + add the small op.

**This plan is 5 commits** (one per step) — each step has its own gate + commit + review. The
executor commits each step separately and does not proceed to step N+1 until step N's gate is
green and its review is triaged.

## Preconditions

- Branch `main` at `2aa23bc` (`git log --oneline -1` shows it) or later; `git status` clean.
- Already shipped (do NOT rebuild): the BitNet runner (`ModelRunner`, `TransformerBlock`,
  `Relu2Mlp`, `Projection`), the ops (`rmsnorm`, `rope_apply`, `gqa_attention`), the
  `tritium_format::SafeTensors` parser (BTreeMap-backed, `tensor_f32` widens bf16/f16→f32),
  and the sharded/mmap resolver pattern in `crates/tritium-cli/src/report.rs`
  (`resolve_shards`/`shards_from_index`) — reuse its logic, do not import the CLI.
- `crates/tritium-nn/tests/salt_accuracy.rs` shows the manual BitNet build-from-safetensors
  pattern (`proj`, `build_weights`) — the loader in Step 4 generalizes it.

## Steps

### Step 1 — `ArchSpec` descriptor + `ModelConfig::from_hf_config`

- **Files:** `crates/tritium-nn/src/config.rs`.
- **Add** (verbatim contract — these are the arch-variation axes):

```rust
/// The feed-forward activation/shape family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpKind {
    /// BitNet: `down(ffn_sub_norm(relu(gate(x))² ⊙ up(x)))`.
    Relu2,
    /// Llama/Qwen: `down(silu(gate(x)) ⊙ up(x))` (no sub-norm).
    SwiGlu,
}

/// Architecture-variation axes beyond the shared llama-family dims in [`ModelConfig`].
/// Defaults describe **BitNet** so existing GGUF loads are unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchSpec {
    pub mlp: MlpKind,
    /// BitNet applies `attn_sub_norm` to the attention output before `o_proj`.
    pub attn_sub_norm: bool,
    /// BitNet applies `ffn_sub_norm` inside the MLP (implied by `MlpKind::Relu2`).
    pub ffn_sub_norm: bool,
    /// Qwen3: per-head RMSNorm on Q and K after projection. (Descriptor only; the op
    /// lands in a later plan — assert `false` on the load path until then.)
    pub qk_norm: bool,
    /// Qwen2/2.5: additive bias on q/k/v projections. (Descriptor only for now.)
    pub qkv_bias: bool,
    /// `false` ⇒ a separate `lm_head.weight`; `true` ⇒ tie to the token embedding.
    pub tied_embeddings: bool,
}

impl ArchSpec {
    /// BitNet-2B4T defaults (what the GGUF path assumes today).
    pub fn bitnet() -> Self {
        Self { mlp: MlpKind::Relu2, attn_sub_norm: true, ffn_sub_norm: true,
                qk_norm: false, qkv_bias: false, tied_embeddings: true }
    }
}
```

- **Add** `ModelConfig::from_hf_config(json: &serde_json::Value) -> Result<(ModelConfig, ArchSpec), NnError>`:
  read a HuggingFace `config.json`. Map: `hidden_size→n_embd`, `num_hidden_layers→n_layers`,
  `num_attention_heads→n_head`, `num_key_value_heads→n_head_kv` (default to `n_head` if absent),
  `intermediate_size→n_ff`, `max_position_embeddings→n_ctx`, `rope_theta→rope_theta`
  (default 10000), `rms_norm_eps→rms_eps`, `architectures[0]`/`model_type→arch`. Build `ArchSpec`:
  `hidden_act=="silu" ⇒ MlpKind::SwiGlu` else `Relu2`; `attn_sub_norm=ffn_sub_norm=false` for a
  standard model; `tied_embeddings = json["tie_word_embeddings"].as_bool().unwrap_or(false)`;
  `qk_norm`/`qkv_bias` from `false` defaults (flags for later). Add `NnError::MissingConfig(String)`
  for absent required keys. `serde_json` is already a `tritium-nn` dep? If not, add
  `serde_json = { workspace = true }` to `crates/tritium-nn/Cargo.toml`.
- **Test (TDD):** a unit test in `config.rs` feeding a minimal SmolLM2-style JSON literal
  (`{"model_type":"llama","hidden_size":576,"num_hidden_layers":30,"num_attention_heads":9,
  "num_key_value_heads":3,"intermediate_size":1536,"rope_theta":100000.0,"rms_norm_eps":1e-5,
  "hidden_act":"silu","tie_word_embeddings":true,"max_position_embeddings":8192}`) →
  asserts `n_embd==576`, `gqa_group()==3`, `MlpKind::SwiGlu`, `tied_embeddings==true`,
  `attn_sub_norm==false`.
- **Command:** `cargo test -p tritium-nn --lib config:: 2>&1 | tail -8`
- **Expected (PASS):** the new `from_hf_config_*` test(s) `ok`; existing `derived_dims` still `ok`;
  `test result: ok.`
- **If you see** a serde_json missing-crate error: add the dep (above) and re-run. A field
  default mismatch: re-read the map above; `num_key_value_heads` defaults to `n_head`.
- **Paste:** full output of the test command.

### Step 2 — SwiGLU MLP + `Mlp` dispatch

- **Files:** `crates/tritium-nn/src/layers/mlp.rs`, `crates/tritium-nn/src/layers/mod.rs`.
- **Add** `SwiGluMlp { gate: Projection, up: Projection, down: Projection }` with
  `forward(&self, backend, x, m, out)` computing `out = down( silu(gate(x)) ⊙ up(x) )`,
  `silu(z)=z*sigmoid(z)` (no sub-norm). Mirror `Relu2Mlp::forward`'s shape contract and scratch
  discipline; reuse `Projection::forward`. Add an enum the block holds:

```rust
/// The block's feed-forward, dispatched by architecture.
#[allow(missing_debug_implementations)]
pub enum Mlp {
    Relu2(Relu2Mlp),
    SwiGlu(SwiGluMlp),
}
impl Mlp {
    pub fn forward(&self, backend: &dyn TernaryBackend, x: &[f32], m: usize, out: &mut [f32]) -> Result<(), NnError> {
        match self { Mlp::Relu2(m2) => m2.forward(backend, x, m, out), Mlp::SwiGlu(sg) => sg.forward(backend, x, m, out) }
    }
}
```

- **Test (TDD):** a unit test with tiny hand-set dense weights (n_embd=2, n_ff=3, m=1): compute
  `silu(gate·x) ⊙ up·x` then `down·` by hand (fp64), assert Tritium matches within 1e-5.
- **Command:** `cargo test -p tritium-nn --lib mlp:: 2>&1 | tail -8`
- **Expected (PASS):** new `swiglu_*` test `ok`; existing `Relu2Mlp` tests still `ok`.
- **If you see** a silu sign error (negative-input branch): `silu(z)=z/(1+e^{-z})`; check the
  hand reference uses the same.
- **Paste:** full output.

### Step 3 — Generalize `TransformerBlock` + untied LM head in `ModelWeights`

- **Files:** `crates/tritium-nn/src/layers/transformer_block.rs`, `crates/tritium-nn/src/model/weights.rs`,
  `crates/tritium-nn/src/model/runner.rs`.
- **Edit** `TransformerBlock.mlp: Relu2Mlp` → `pub mlp: Mlp`. In its `forward`, `attn_sub_norm`
  is already `Vec` (empty ⇒ skip) — keep that. Update the BitNet build sites (GGUF loader,
  `salt_accuracy.rs`) to wrap `Relu2Mlp` as `Mlp::Relu2(..)`.
- **Add** to `ModelWeights`: `pub lm_head: Option<Projection>` (None ⇒ tie to `token_embd`).
- **Edit** the runner's LM-head step (`runner.rs` ~line 321, "Tied LM head"): if
  `self.weights.lm_head` is `Some(head)`, `logits = head.forward(last_norm)`; else the existing
  tied dot-product against `token_embd`. Keep the tied path byte-identical for BitNet.
- **Command:** `cargo build -p tritium-nn 2>&1 | tail -5 && cargo test -p tritium-nn --lib 2>&1 | tail -6`
- **Expected (PASS):** builds; all existing `tritium-nn` lib tests still `ok` (no regression —
  the tied/Relu2 paths are unchanged).
- **If you see** a match-arm/exhaustiveness error at a BitNet build site: wrap its
  `Relu2Mlp` in `Mlp::Relu2(..)`.
- **Paste:** full output.
- **Commit note:** this step + Step 2 may be one commit if cleaner; the plan allows it.

### Step 4 — HF-safetensors config-driven loader

- **Files:** `crates/tritium-nn/src/model/weights.rs` (add `load_hf`), a new
  `crates/tritium-nn/src/model/hf.rs` if it keeps `weights.rs` tidy.
- **Add** `ModelWeights::load_hf(dir: &Path) -> Result<(ModelConfig, ArchSpec, ModelWeights), NnError>`:
  1. Read `dir/config.json` → `from_hf_config` (Step 1).
  2. Resolve safetensors shards (reuse the resolver *logic* from `report.rs`: prefer
     `model.safetensors.index.json` → `weight_map`, else `model.safetensors`, else sorted
     `*.safetensors`). mmap each; build a `name → (shard, SafeTensors view)` lookup, or read
     eagerly for a small model. (A small model fits RAM; keep it simple — eager `tensor_f32`.)
  3. Build `ModelWeights` from the **standard Llama/Qwen tensor-name schema** below, each 2D
     weight wrapped `Projection::Dense(DenseLinear::new(w, n_out, k_in)?)`.

  | Role | Tensor name | Shape |
  |---|---|---|
  | token embedding | `model.embed_tokens.weight` | `[vocab, n_embd]` |
  | q / k / v / o proj | `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` | GQA widths |
  | mlp gate/up/down | `model.layers.{i}.mlp.{gate,up,down}_proj.weight` | `[n_ff,n_embd]` / `[n_embd,n_ff]` |
  | pre-attn norm | `model.layers.{i}.input_layernorm.weight` | `[n_embd]` |
  | pre-mlp norm | `model.layers.{i}.post_attention_layernorm.weight` | `[n_embd]` |
  | final norm | `model.norm.weight` | `[n_embd]` |
  | lm head (untied) | `lm_head.weight` (absent ⇒ tied) | `[vocab, n_embd]` |

  Build `TransformerBlock` with `attn_sub_norm = vec![]` (standard), `mlp = Mlp::SwiGlu(..)`
  (per `ArchSpec`), `ffn_norm = post_attention_layernorm`. Set `ModelWeights.lm_head =
  spec.tied_embeddings.then(|| None).unwrap_or_else(|| lm_head present)` — i.e. `None` when tied,
  else `Some(Projection::Dense(lm_head))`. Assert `!spec.qk_norm && !spec.qkv_bias` here with a
  clear `NnError` ("arch needs QK-norm/QKV-bias — not yet supported, plan 0037").
- **Add** `ModelRunner::from_hf(dir, backend) -> Result<Self, NnError>` (thin: `load_hf` +
  existing `from_weights`).
- **Command:** `cargo build -p tritium-nn 2>&1 | tail -5`
- **Expected (PASS):** builds clean.
- **If you see** a shape assertion at load: the schema widths — q is `n_head·head_dim`, k/v are
  `n_head_kv·head_dim`; verify against `config.json`.
- **Paste:** full output + `cargo clippy -p tritium-nn -- -D warnings 2>&1 | tail -3`.

### Step 5 — HF conformance gate (the success criterion)

- **Files:** `crates/tritium-nn/tests/hf_inference.rs` (new, `#[ignore]`d like `salt_accuracy.rs`);
  extend `tools/gen_reference.py` (or add `tools/gen_hf_logits.py`).
- **Fixture:** `SmolLM2-135M-Instruct` (ungated, Apache-2.0, Llama-arch: SwiGLU/GQA/RoPE/RMSNorm,
  tied embeddings, no biases, no QK-norm). Download to
  `~/.cache/tritium-models/smollm2-135m/` (`config.json`, `model.safetensors`, tokenizer):
  `huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct --local-dir ~/.cache/tritium-models/smollm2-135m` (or curl the three raw files). The test **skips cleanly** if absent.
- **Reference:** a python script that loads the model in `transformers`, runs a fixed prompt
  (token ids, e.g. the first 16 of a canned string), and writes `smollm2_ref.json` =
  `{prompt_ids, next_argmax_per_pos, logits_last_row}` (top-k logits + full last-row for the
  rel-err check).
- **Test:** load via `ModelRunner::from_hf`, teacher-force the `prompt_ids`, and assert:
  (a) per-position greedy argmax == `next_argmax_per_pos` (token-exact), and
  (b) last-position logit **rel-err < 1e-3** vs `logits_last_row`.
- **Command:**
  `cargo test -p tritium-nn --release --test hf_inference -- --ignored --nocapture 2>&1 | tail -20`
- **Expected (PASS):** prints the per-position match count `N/N` and `logit rel-err = <…> (< 1e-3)`;
  `test result: ok.` (or a clean `skipping: … absent` if the model isn't downloaded — in which
  case download it first; the gate is not satisfied by a skip).
- **If you see** argmax mismatch at position 0 only: check RoPE θ (SmolLM2 uses a large θ — read
  it from `config.json`, don't assume 10000) and that RMSNorm ε matches. Mismatch everywhere:
  the SwiGLU vs Relu2 dispatch (Step 2) or a q/k/v width swap. A constant logit scale offset:
  tied-head path (SmolLM2 ties) — confirm `lm_head==None` routes to the `token_embd` dot-product.
- **Paste:** full output of the test command.

## Gate

`hf_inference` runs on the real SmolLM2-135M and reports **token-exact greedy** over the prompt
+ **last-row logit rel-err < 1e-3** vs `transformers`. Plus: `cargo test -p tritium-nn` (default,
non-ignored) fully green (no BitNet regression), `cargo clippy -p tritium-nn -- -D warnings`
clean, `cargo fmt --check` clean.

## Commit

One commit per step (5 total), each message ending with the footer. Step-5 (the gate) message:

```
feat(nn): config-driven general inference — standard-transformer fp forward (ADR 0020 keystone)

Runner is no longer BitNet-only. Adds ArchSpec + ModelConfig::from_hf_config, a SwiGLU MLP
(Mlp dispatch), untied-LM-head support, and a config-driven HF-safetensors loader
(ModelWeights::load_hf / ModelRunner::from_hf) using the standard llama/qwen tensor-name schema.

Gate: SmolLM2-135M-Instruct loads from config.json + safetensors and runs a CPU forward that is
greedy-token-exact vs transformers over a 16-token prompt (last-row logit rel-err < 1e-3). No
BitNet regression (tied/Relu2 paths byte-identical). Descriptor carries qk_norm/qkv_bias/SSM
flags off (later plans 0036/0037).

Unblocks ADR 0020 steps 2-6 (SALT loader, teacher-caching, ternary student forward).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0166rZwZDrh6awfgZzLS3Ez4
```

## Review

After each step's commit, review with the project's **code-review skill / `code-reviewer`
subagent** (NOT lamu/local-llm, per policy). Paste the verdict + every finding verbatim; the
strong model triages before the next step. Focus the Step-4/5 review on: the tensor-name schema
(width/shape correctness, tied-vs-untied), RoPE/RMSNorm parameter sourcing from config.json, and
no-regression on the BitNet path.

## Done criterion

All 5 steps committed; `hf_inference` green on real SmolLM2-135M (token-exact + rel-err < 1e-3);
default `tritium-nn` tests + clippy `-D` + fmt clean; `main` clean. Flip the ROADMAP 0035 row to
`done` and write plan **0036 — SALT multi-plane inference loader** (run what `quantize` emits;
ADR 0020 step 2).

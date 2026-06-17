//! `ModelRunner`: load a GGUF model and generate tokens.
//!
//! The top of the inference spine. [`load`](ModelRunner::load) reads the config +
//! weights and picks a backend; [`forward`](ModelRunner::forward) runs one prefill
//! / decode step (embedding → blocks → final norm → LM head → logits);
//! [`generate`](ModelRunner::generate) loops `forward` + greedy sampling until
//! `eos` or a length cap.
//!
//! The LM head is **tied** to the token embedding (BitNet 2B4T sets
//! `tie_word_embeddings = true` and ships no separate `output.weight`), so
//! unembedding is a dense fp32 matmul against `token_embd`.

use tritium_spec::TernaryBackend;

use crate::config::ModelConfig;
use crate::error::NnError;
use crate::kv_cache::KvCache;
use crate::layers::BlockDump;
use crate::model::weights::ModelWeights;
use crate::ops::{rmsnorm, sample_greedy};
use tritium_format::GgufFile;

/// Per-stage activations for the fidelity ladder, captured by
/// [`ModelRunner::forward_dump`]. All hidden-state vectors are `[seq, n_embd]`
/// row-major; logits are `[vocab]` for the last position.
#[derive(Debug, Default, Clone)]
pub struct ForwardDump {
    /// `hidden_states[0]` — the raw token embedding (no norm).
    pub embedding: Vec<f32>,
    /// Layer-0 `input_layernorm` output (`[seq, n_embd]`).
    pub layer0_attn_norm: Vec<f32>,
    /// Layer-0 attention output after `attn_sub_norm` + `o_proj`, pre-residual.
    pub layer0_attn_out: Vec<f32>,
    /// `hidden_states[i+1]` for every layer `i` — the block output (post both
    /// residuals). Length `n_layers`, each `[seq, n_embd]`.
    pub hidden_states: Vec<Vec<f32>>,
    /// Final `output_norm` output (`[seq, n_embd]`).
    pub final_norm: Vec<f32>,
    /// Logits at the last position (`[vocab]`).
    pub logits: Vec<f32>,
}

/// A loaded model plus its per-layer KV caches and execution backend.
#[allow(missing_debug_implementations)]
pub struct ModelRunner {
    /// Model dimensions.
    pub config: ModelConfig,
    /// Loaded weights (embeddings, blocks, final norm, tied LM head).
    pub weights: ModelWeights,
    /// One KV cache per transformer block; length `config.n_layers`.
    pub kv: Vec<KvCache>,
    /// The execution backend for ternary GEMMs.
    pub backend: Box<dyn TernaryBackend>,
    /// (cuda) The device-resident decode fast path (v0.3.1, ADR 0013), built lazily
    /// on the first non-dump forward when `backend` is a CUDA backend. Keeps the
    /// residual stream + KV in VRAM across all layers, replacing ~210 round-trips per
    /// token with ~1. The host path stays the golden oracle (used for `forward_dump`
    /// and on non-CUDA backends).
    #[cfg(feature = "cuda")]
    resident: Option<tritium_cuda::CudaDecodeModel>,
    /// (cuda) Whether we have already tried (and possibly failed) to build `resident`,
    /// so a non-CUDA backend probes the downcast exactly once.
    #[cfg(feature = "cuda")]
    resident_probed: bool,
}

impl ModelRunner {
    /// Load a runner from a parsed GGUF `file` (+ its raw `bytes`) onto `backend`.
    ///
    /// Reads [`ModelConfig::from_gguf`], loads [`ModelWeights`], and allocates one
    /// [`KvCache`] per layer sized to the context length.
    ///
    /// # Errors
    /// [`NnError::MissingMetadata`] / [`NnError::MissingTensor`] /
    /// [`NnError::UnsupportedTensorType`] on a malformed file; [`NnError::Backend`]
    /// on an upload failure.
    pub fn load(
        file: &GgufFile,
        bytes: &[u8],
        backend: Box<dyn TernaryBackend>,
    ) -> Result<Self, NnError> {
        let config = ModelConfig::from_gguf(file)?;
        let weights = ModelWeights::load(file, bytes, &config, backend.as_ref())?;

        let head_dim = config.head_dim() as usize;
        let n_head_kv = config.n_head_kv as usize;
        let max_ctx = config.n_ctx as usize;
        let kv = (0..config.n_layers)
            .map(|_| KvCache::new(max_ctx, n_head_kv, head_dim))
            .collect();

        Ok(Self {
            config,
            weights,
            kv,
            backend,
            #[cfg(feature = "cuda")]
            resident: None,
            #[cfg(feature = "cuda")]
            resident_probed: false,
        })
    }

    /// Convenience: load from a GGUF byte buffer using the runtime registry's
    /// `"cpu"` backend.
    ///
    /// # Errors
    /// [`NnError::Backend`] if the GGUF cannot be parsed, [`NnError::BackendUnavailable`]
    /// if no CPU backend is registered, or any error from [`load`](Self::load).
    pub fn load_cpu(bytes: &[u8]) -> Result<Self, NnError> {
        let file = tritium_format::read_gguf(bytes).map_err(|e| NnError::Backend(e.to_string()))?;
        let backend = cpu_backend()?;
        Self::load(&file, bytes, backend)
    }

    /// Reset every layer's KV cache (start a fresh sequence). Also resets the
    /// device-resident decoder's KV watermark when present.
    pub fn reset(&mut self) {
        for c in &mut self.kv {
            c.reset();
        }
        #[cfg(feature = "cuda")]
        if let Some(r) = self.resident.as_mut() {
            r.reset();
        }
    }

    /// Run one prefill / decode step over `tokens` at absolute positions
    /// `positions`, returning the next-token logits `[vocab]` for the last token.
    ///
    /// # Errors
    /// [`NnError::Shape`] on inconsistent lengths, or [`NnError::Backend`] on a
    /// backend failure.
    pub fn forward(&mut self, tokens: &[u32], positions: &[usize]) -> Result<Vec<f32>, NnError> {
        self.forward_inner(tokens, positions, None)
    }

    /// Like [`forward`](Self::forward), but captures per-stage activations into
    /// `dump` for the fidelity ladder.
    ///
    /// # Errors
    /// Same as [`forward`](Self::forward).
    pub fn forward_dump(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        dump: &mut ForwardDump,
    ) -> Result<Vec<f32>, NnError> {
        self.forward_inner(tokens, positions, Some(dump))
    }

    fn forward_inner(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        mut dump: Option<&mut ForwardDump>,
    ) -> Result<Vec<f32>, NnError> {
        let n_embd = self.config.n_embd as usize;
        let seq = tokens.len();
        if seq == 0 || positions.len() != seq {
            return Err(NnError::Shape {
                expected: seq,
                got: positions.len(),
            });
        }

        // Device-resident fast path (v0.3.1): when the backend is CUDA and we are not
        // capturing per-stage activations, run the whole forward on the GPU (residual
        // stream + KV stay in VRAM). The dump path keeps the host orchestration so the
        // fidelity ladder can still inspect each stage.
        #[cfg(feature = "cuda")]
        if dump.is_none() {
            if let Some(logits) = self.forward_resident(tokens, positions)? {
                return Ok(logits);
            }
        }

        // Embedding gather: hidden = token_embd[token] for each token.
        let mut hidden = vec![0.0f32; seq * n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            let row = tok as usize;
            let src = self
                .weights
                .token_embd
                .get(row * n_embd..row * n_embd + n_embd)
                .ok_or_else(|| NnError::MissingTensor(format!("token_embd row {row}")))?;
            hidden[t * n_embd..t * n_embd + n_embd].copy_from_slice(src);
        }
        if let Some(d) = dump.as_deref_mut() {
            d.embedding = hidden.clone();
            d.hidden_states.clear();
        }

        // Per-layer block forward.
        let mut next = vec![0.0f32; seq * n_embd];
        let n_layers = self.weights.layers.len();
        for li in 0..n_layers {
            // Borrow the block and its KV cache disjointly.
            let block = &self.weights.layers[li];
            let kv = &mut self.kv[li];
            if li == 0 && dump.is_some() {
                let mut bd = BlockDump::default();
                block.forward_dump(
                    self.backend.as_ref(),
                    &hidden,
                    positions,
                    kv,
                    &self.config,
                    &mut next,
                    &mut bd,
                )?;
                if let Some(d) = dump.as_deref_mut() {
                    d.layer0_attn_norm = bd.attn_norm_out;
                    d.layer0_attn_out = bd.attn_out;
                }
            } else {
                block.forward(
                    self.backend.as_ref(),
                    &hidden,
                    positions,
                    kv,
                    &self.config,
                    &mut next,
                )?;
            }
            std::mem::swap(&mut hidden, &mut next);
            if let Some(d) = dump.as_deref_mut() {
                d.hidden_states.push(hidden.clone());
            }
        }

        // Final RMSNorm over the last token only (we only need its logits), but
        // dump the full sequence when capturing.
        let last = seq - 1;
        let mut final_full = if dump.is_some() {
            vec![0.0f32; seq * n_embd]
        } else {
            Vec::new()
        };
        let mut last_norm = vec![0.0f32; n_embd];
        for t in 0..seq {
            let src = &hidden[t * n_embd..t * n_embd + n_embd];
            if t == last {
                rmsnorm(
                    src,
                    &self.weights.output_norm,
                    self.config.rms_eps,
                    &mut last_norm,
                )?;
            }
            if dump.is_some() {
                let dst = &mut final_full[t * n_embd..t * n_embd + n_embd];
                rmsnorm(src, &self.weights.output_norm, self.config.rms_eps, dst)?;
            }
        }
        if let Some(d) = dump.as_deref_mut() {
            d.final_norm = final_full;
        }

        // Tied LM head: logits[v] = <last_norm, token_embd[v]>.
        let vocab = self.weights.vocab;
        let mut logits = vec![0.0f32; vocab];
        let embd = &self.weights.token_embd;
        for (v, slot) in logits.iter_mut().enumerate() {
            let row = &embd[v * n_embd..v * n_embd + n_embd];
            let mut acc = 0.0f32;
            for k in 0..n_embd {
                acc += last_norm[k] * row[k];
            }
            *slot = acc;
        }
        if let Some(d) = dump {
            d.logits = logits.clone();
        }

        Ok(logits)
    }

    /// (cuda) Run the forward through the device-resident decoder if the backend is
    /// CUDA, returning `Some(last-token logits)`; `None` means the backend has no
    /// resident path and the caller should fall back to the host orchestration.
    ///
    /// Each of the `seq` tokens is driven through one device `step` (so a multi-token
    /// prefill is processed as a sequential causal decode — numerically identical to
    /// the batched host prefill, since each token's reductions are unchanged). Only
    /// the last token's logits are returned, matching [`forward`](Self::forward).
    #[cfg(feature = "cuda")]
    fn forward_resident(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Option<Vec<f32>>, NnError> {
        if !self.ensure_resident()? {
            return Ok(None);
        }
        let model = self
            .resident
            .as_mut()
            .expect("ensure_resident returned true so resident is built");
        let mut logits = Vec::new();
        for (&tok, &pos) in tokens.iter().zip(positions.iter()) {
            logits = model
                .step(tok, pos)
                .map_err(|e| NnError::Backend(e.to_string()))?;
        }
        Ok(Some(logits))
    }

    /// (cuda) Build the device-resident decoder on first use. Returns `true` if a
    /// resident decoder is available (already built or built now), `false` if the
    /// backend is not a CUDA backend (probed once, then cached).
    #[cfg(feature = "cuda")]
    fn ensure_resident(&mut self) -> Result<bool, NnError> {
        if self.resident.is_some() {
            return Ok(true);
        }
        if self.resident_probed {
            return Ok(false);
        }
        self.resident_probed = true;
        // Recover the concrete `CudaBackend` from the `dyn TernaryBackend` (the
        // defaulted `as_any` hook returns `None` for every non-CUDA backend).
        let Some(cuda) = self
            .backend
            .as_any()
            .and_then(|a| a.downcast_ref::<tritium_cuda::CudaBackend>())
        else {
            return Ok(false);
        };
        let spec = Self::build_decode_spec(&self.weights, &self.config);
        let model = cuda
            .build_decode_model(&spec)
            .map_err(|e| NnError::Backend(e.to_string()))?;
        self.resident = Some(model);
        Ok(true)
    }

    /// (cuda) Assemble the borrowed [`tritium_cuda::DecodeModelSpec`] from the loaded
    /// weights + config. The borrow is consumed inside `build_decode_model`; the
    /// resident model owns shared (`Arc`) device handles afterwards.
    #[cfg(feature = "cuda")]
    fn build_decode_spec<'a>(
        weights: &'a ModelWeights,
        config: &ModelConfig,
    ) -> tritium_cuda::DecodeModelSpec<'a> {
        use crate::layers::TernaryLinear;
        use tritium_cuda::{DecodeLayerSpec, DecodeLinearSpec, DecodeModelSpec};

        fn lin(tl: &TernaryLinear) -> DecodeLinearSpec<'_> {
            DecodeLinearSpec {
                weights: &*tl.weights,
                scales: &tl.scales,
            }
        }

        let layers = weights
            .layers
            .iter()
            .map(|b| DecodeLayerSpec {
                attn_norm: &b.attn_norm,
                attn_sub_norm: &b.attn_sub_norm,
                ffn_norm: &b.ffn_norm,
                ffn_sub_norm: &b.mlp.ffn_sub_norm,
                q: lin(&b.q_proj),
                k: lin(&b.k_proj),
                v: lin(&b.v_proj),
                o: lin(&b.o_proj),
                gate: lin(&b.mlp.gate),
                up: lin(&b.mlp.up),
                down: lin(&b.mlp.down),
            })
            .collect();

        DecodeModelSpec {
            token_embd: &weights.token_embd,
            output_norm: &weights.output_norm,
            layers,
            n_embd: config.n_embd as usize,
            n_head: config.n_head as usize,
            n_head_kv: config.n_head_kv as usize,
            head_dim: config.head_dim() as usize,
            n_ff: config.n_ff as usize,
            vocab: weights.vocab,
            max_ctx: config.n_ctx as usize,
            rope_theta: config.rope_theta,
            rms_eps: config.rms_eps,
        }
    }

    /// Greedily generate up to `max_new` tokens continuing `prompt` (token IDs),
    /// returning the generated IDs (not including the prompt). Stops early at the
    /// supplied `eos` token. Resets the KV caches first.
    ///
    /// # Errors
    /// [`NnError::Shape`] / [`NnError::Backend`] propagated from
    /// [`forward`](Self::forward).
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: u32,
    ) -> Result<Vec<u32>, NnError> {
        self.reset();
        if prompt.is_empty() {
            return Ok(Vec::new());
        }

        // Prefill the whole prompt at positions 0..len.
        let positions: Vec<usize> = (0..prompt.len()).collect();
        let mut logits = self.forward(prompt, &positions)?;
        let mut out = Vec::with_capacity(max_new);

        for pos in (prompt.len()..).take(max_new) {
            let next = sample_greedy(&logits).ok_or(NnError::Shape {
                expected: 1,
                got: 0,
            })?;
            if next == eos {
                break;
            }
            out.push(next);
            logits = self.forward(&[next], &[pos])?;
        }
        Ok(out)
    }
}

/// Construct a fresh CPU backend trait object via the linked `tritium-cpu` crate.
///
/// The runtime [`Registry`] only hands out borrows, so for an owned
/// `Box<dyn TernaryBackend>` we go through the registered init by name. We expose
/// this as a tiny helper so the test/CLI lanes can `load_cpu` without naming the
/// concrete backend type.
fn cpu_backend() -> Result<Box<dyn TernaryBackend>, NnError> {
    for entry in tritium_runtime::BACKENDS {
        if entry.name == "cpu" {
            return (entry.init)().map_err(|e| NnError::Backend(e.to_string()));
        }
    }
    Err(NnError::BackendUnavailable)
}

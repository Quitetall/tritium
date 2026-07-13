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

use std::path::Path;

use tritium_spec::TernaryBackend;

use crate::config::ModelConfig;
use crate::error::NnError;
#[cfg(feature = "cuda")]
use crate::error::ResidentOpError;
use crate::kv_cache::KvCache;
use crate::layers::{BlockDump, BlockScratch};
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

    /// Build a runner from already-constructed [`ModelWeights`] — e.g. an in-memory
    /// SALT / fp quantization for the accuracy harness — allocating one [`KvCache`]
    /// per layer. The model starts on the host forward path; the device-resident
    /// decoder is built lazily and is skipped for any model carrying a dense
    /// ([`Projection::Dense`](crate::layers::Projection)) projection.
    #[must_use]
    pub fn from_weights(
        config: ModelConfig,
        weights: ModelWeights,
        backend: Box<dyn TernaryBackend>,
    ) -> Self {
        let head_dim = config.head_dim() as usize;
        let n_head_kv = config.n_head_kv as usize;
        let max_ctx = config.n_ctx as usize;
        let kv = (0..config.n_layers)
            .map(|_| KvCache::new(max_ctx, n_head_kv, head_dim))
            .collect();
        Self {
            config,
            weights,
            kv,
            backend,
            #[cfg(feature = "cuda")]
            resident: None,
            #[cfg(feature = "cuda")]
            resident_probed: false,
        }
    }

    /// Load a **standard-transformer fp** model (Llama/SmolLM2/…) from a HuggingFace
    /// directory (`config.json` + safetensors) onto `backend` — the general (non-BitNet)
    /// inference path. See [`ModelWeights::load_hf`].
    ///
    /// # Errors
    /// Propagates [`ModelWeights::load_hf`] errors (bad config, missing/unreadable tensor,
    /// unsupported arch, shape mismatch).
    pub fn from_hf(dir: &Path, backend: Box<dyn TernaryBackend>) -> Result<Self, NnError> {
        let (config, _spec, weights) = ModelWeights::load_hf(dir)?;
        Ok(Self::from_weights(config, weights, backend))
    }

    /// Load a **SALT-quantized** model: ternary 2D weights from `bundle` (dequant-to-dense),
    /// norms + `config.json` from `model_dir`. See [`ModelWeights::load_salt`].
    ///
    /// # Errors
    /// Propagates [`ModelWeights::load_salt`] errors.
    pub fn from_salt(
        model_dir: &Path,
        bundle: &Path,
        backend: Box<dyn TernaryBackend>,
    ) -> Result<Self, NnError> {
        let (config, weights) = ModelWeights::load_salt(model_dir, bundle)?;
        Ok(Self::from_weights(config, weights, backend))
    }

    /// Load a self-contained tied-SwiGLU training-SALT GGUF artifact.
    ///
    /// This reference-evaluation path dequantizes the artifact's private SALT
    /// tensors into exact-fp dense projections and requires all config, norms,
    /// weights, and provenance to live in the supplied bytes. It never falls back
    /// to a model directory.
    ///
    /// # Errors
    /// Propagates [`ModelWeights::load_training_salt_gguf`] validation errors.
    pub fn from_training_salt_gguf(
        bytes: &[u8],
        backend: Box<dyn TernaryBackend>,
    ) -> Result<Self, NnError> {
        let (config, weights) = ModelWeights::load_training_salt_gguf(bytes)?;
        Ok(Self::from_weights(config, weights, backend))
    }

    /// (cuda) Borrow the lazily-built device-resident decoder, building it first if
    /// needed (returns `None` on a non-CUDA backend). TEST access — production
    /// callers (tritium-serve) go through the typed facade below
    /// ([`tree_verify_greedy`](Self::tree_verify_greedy), [`new_batch`](Self::new_batch), …),
    /// which is the supported API.
    ///
    /// # Errors
    /// [`NnError::Backend`] if building the resident decoder fails.
    #[cfg(feature = "cuda")]
    #[doc(hidden)]
    pub fn resident_cuda(&mut self) -> Result<Option<&mut tritium_cuda::CudaDecodeModel>, NnError> {
        if !self.ensure_resident()? {
            return Ok(None);
        }
        Ok(self.resident.as_mut())
    }

    /// (cuda) Borrow the resident decoder for a facade op, folding the two failure
    /// modes into [`ResidentOpError`].
    #[cfg(feature = "cuda")]
    fn resident_for_op(&mut self) -> Result<&mut tritium_cuda::CudaDecodeModel, ResidentOpError> {
        match self.ensure_resident() {
            Err(e) => Err(ResidentOpError::Build(e.to_string())),
            Ok(false) => Err(ResidentOpError::Unavailable),
            Ok(true) => self.resident.as_mut().ok_or(ResidentOpError::Unavailable),
        }
    }

    /// (cuda) Probe for the CUDA device-resident decoder, building it on the
    /// first call: `Ok(true)` = present, `Ok(false)` = non-CUDA backend,
    /// `Err(Build)` = a CUDA backend whose decoder build FAILED — callers that
    /// answer clients should surface that as an internal error, not "feature
    /// unsupported".
    ///
    /// # Errors
    /// [`ResidentOpError::Build`] when the lazy decoder build fails.
    #[cfg(feature = "cuda")]
    pub fn try_resident_decoder(&mut self) -> Result<bool, ResidentOpError> {
        match self.ensure_resident() {
            Err(e) => Err(ResidentOpError::Build(e.to_string())),
            Ok(ok) => Ok(ok && self.resident.is_some()),
        }
    }

    /// (cuda) Whether this runner has (or can build) the CUDA device-resident
    /// decoder — the DISPATCH probe for spec-decode lookup and continuous
    /// batching, where a build failure just means "fall back to the plain
    /// path" (it reads as `false`). Client-facing feature gates should use
    /// [`try_resident_decoder`](Self::try_resident_decoder) instead so build
    /// failures classify as errors, not absence.
    #[cfg(feature = "cuda")]
    pub fn has_resident_decoder(&mut self) -> bool {
        self.try_resident_decoder().unwrap_or(false)
    }

    /// (cuda) BASTION tree-verify, greedy rule: forward the token tree
    /// (`parents[i] < i`, root parent `-1`), walk greedy acceptance on the
    /// device and commit the accepted path's KV. Returns the committed tokens
    /// (always ≥ 1: the root's successor). Lossless vs plain greedy decode.
    ///
    /// # Errors
    /// [`ResidentOpError`] — `Unavailable` on a non-CUDA backend, `Op` with the
    /// device/validation error otherwise.
    #[cfg(feature = "cuda")]
    pub fn tree_verify_greedy(
        &mut self,
        tokens: &[u32],
        parents: &[i32],
    ) -> Result<Vec<u32>, ResidentOpError> {
        self.resident_for_op()?
            .tree_verify_greedy(tokens, parents)
            .map_err(ResidentOpError::Op)
    }

    /// (cuda) BASTION tree-verify, logits form: forward the token tree and
    /// return the flat `[tokens.len() × vocab]` logits for a host-side accept
    /// rule (sampling). Pair with [`tree_commit`](Self::tree_commit).
    ///
    /// # Errors
    /// [`ResidentOpError`] — see [`tree_verify_greedy`](Self::tree_verify_greedy).
    #[cfg(feature = "cuda")]
    pub fn tree_verify_logits(
        &mut self,
        tokens: &[u32],
        parents: &[i32],
    ) -> Result<Vec<f32>, ResidentOpError> {
        self.resident_for_op()?
            .tree_verify_logits(tokens, parents)
            .map_err(ResidentOpError::Op)
    }

    /// (cuda) Commit the accepted `path` (tree-node indices from
    /// [`tree_verify_logits`](Self::tree_verify_logits)) into the KV cache.
    ///
    /// # Errors
    /// [`ResidentOpError`] — `Op(InvalidInput)` when no tree is pending or the
    /// path is malformed.
    #[cfg(feature = "cuda")]
    pub fn tree_commit(&mut self, path: &[usize]) -> Result<(), ResidentOpError> {
        self.resident_for_op()?
            .tree_commit(path)
            .map_err(ResidentOpError::Op)
    }

    /// (cuda) Allocate continuous-batching state for `n` concurrent sequences
    /// (per-layer `[n, max_ctx, kv_width]` KV arenas + M=N scratch).
    ///
    /// # Errors
    /// [`ResidentOpError`] — `Op` with the stringified device error on an
    /// allocation failure (the CUDA path reports these as
    /// `BackendError::Backend`, not `OutOfMemory`).
    #[cfg(feature = "cuda")]
    pub fn new_batch(&mut self, n: usize) -> Result<tritium_cuda::BatchKv, ResidentOpError> {
        self.resident_for_op()?
            .new_batch(n)
            .map_err(ResidentOpError::Op)
    }

    /// (cuda) Paged-KV batch pool (ADR 0025): page pools of `pool_pages`
    /// pages shared by all `n` slots through per-slot page tables — KV VRAM
    /// scales with the pool, not `n × max_ctx`. Callers must
    /// [`tritium_cuda::BatchKv::reserve_pages`] before stepping/adopting into
    /// a slot and [`tritium_cuda::BatchKv::release_pages`] at retirement.
    ///
    /// # Errors
    /// [`ResidentOpError`] — `Op` with the device error.
    #[cfg(feature = "cuda")]
    pub fn new_batch_paged(
        &mut self,
        n: usize,
        pool_pages: usize,
    ) -> Result<tritium_cuda::BatchKv, ResidentOpError> {
        self.resident_for_op()?
            .new_batch_paged(n, pool_pages)
            .map_err(ResidentOpError::Op)
    }

    /// (cuda) Continuous-batching admission: copy this runner's single-sequence
    /// KV rows `[0, len)` into batch slot `row` (prefill the prompt through the
    /// single-sequence path first, then adopt + [`BatchKv::set_position`]).
    ///
    /// # Errors
    /// [`ResidentOpError`] — `Op(InvalidInput)` on a bad row/len or a non-f32 KV rung.
    #[cfg(feature = "cuda")]
    pub fn adopt_into_batch_row(
        &mut self,
        batch: &mut tritium_cuda::BatchKv,
        row: usize,
        len: usize,
    ) -> Result<(), ResidentOpError> {
        self.resident_for_op()?
            .copy_kv_into_batch_row(batch, row, len)
            .map_err(ResidentOpError::Op)
    }

    /// (cuda) One lockstep M=N decode step through the captured batch graph:
    /// feed each slot's token, return per-slot logits.
    ///
    /// # Errors
    /// [`ResidentOpError`] — `Op` with the device error.
    #[cfg(feature = "cuda")]
    pub fn decode_batch_graph(
        &mut self,
        batch: &mut tritium_cuda::BatchKv,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, ResidentOpError> {
        self.resident_for_op()?
            .decode_batch_graph(batch, tokens)
            .map_err(ResidentOpError::Op)
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

    /// Drop the device-resident decoder (if built) so the next forward rebuilds it from
    /// the *current* weights. Call after mutating a layer's weights in place (e.g. a QAT
    /// [`replace_weights`](crate::layers::TernaryLinear::replace_weights) swap, plan
    /// 0010) — the resident decoder holds its own device copies and would otherwise serve
    /// stale weights. Without the `cuda` feature this is a no-op; with it on, it clears the
    /// resident slot + re-probe flag (on a non-CUDA backend that slot was never built, so
    /// the next forward just re-probes once — harmless).
    pub fn invalidate_resident(&mut self) {
        #[cfg(feature = "cuda")]
        {
            self.resident = None;
            self.resident_probed = false;
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
        if dump.is_none()
            && let Some(logits) = self.forward_resident(tokens, positions)?
        {
            return Ok(logits);
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

        // Per-layer block forward. Pre-allocate scratch once, reuse across layers.
        let mut next = vec![0.0f32; seq * n_embd];
        let n_head = self.config.n_head as usize;
        let n_head_kv = self.config.n_head_kv as usize;
        let head_dim = self.config.head_dim() as usize;
        let q_width = n_head * head_dim;
        let kv_width = n_head_kv * head_dim;
        let mut scratch = BlockScratch::new(seq, n_embd, q_width, kv_width);
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
                block.forward_with_scratch(
                    self.backend.as_ref(),
                    &hidden,
                    positions,
                    kv,
                    &self.config,
                    &mut next,
                    &mut scratch,
                )?;
            }
            std::mem::swap(&mut hidden, &mut next);
            if let Some(d) = dump.as_deref_mut() {
                d.hidden_states.push(hidden.clone());
            }
        }

        // Final RMSNorm: compute only the last token's norm (we only need its
        // logits for the LM head). The dump path computes the full sequence.
        let last = seq - 1;
        let mut last_norm = vec![0.0f32; n_embd];
        if let Some(d) = dump.as_deref_mut() {
            let mut final_full = vec![0.0f32; seq * n_embd];
            for t in 0..seq {
                let src = &hidden[t * n_embd..t * n_embd + n_embd];
                let dst = &mut final_full[t * n_embd..t * n_embd + n_embd];
                rmsnorm(src, &self.weights.output_norm, self.config.rms_eps, dst)?;
            }
            last_norm.copy_from_slice(&final_full[last * n_embd..last * n_embd + n_embd]);
            d.final_norm = final_full;
        } else {
            let src = &hidden[last * n_embd..last * n_embd + n_embd];
            rmsnorm(
                src,
                &self.weights.output_norm,
                self.config.rms_eps,
                &mut last_norm,
            )?;
        }

        // LM head. Untied ⇒ a dedicated `lm_head` projection; tied ⇒ the dot-product
        // against the token embedding (BitNet). Both map `last_norm` ([n_embd]) → logits.
        let logits = if let Some(head) = &self.weights.lm_head {
            let mut logits = vec![0.0f32; head.n_out()];
            head.forward(self.backend.as_ref(), &last_norm, 1, &mut logits)?;
            logits
        } else {
            // Tied LM head: logits[v] = <last_norm, token_embd[v]>.
            // Parallelized with rayon: 128K × 2560 dot products.
            use rayon::prelude::*;
            let vocab = self.weights.vocab;
            let mut logits = vec![0.0f32; vocab];
            let embd = &self.weights.token_embd;
            logits
                .par_chunks_mut(1024)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base = chunk_idx * 1024;
                    for (i, slot) in chunk.iter_mut().enumerate() {
                        let v = base + i;
                        let row = &embd[v * n_embd..v * n_embd + n_embd];
                        let mut acc = 0.0f32;
                        for k in 0..n_embd {
                            acc += last_norm[k] * row[k];
                        }
                        *slot = acc;
                    }
                });
            logits
        };
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
    ///
    /// Cost note: sequential prefill is O(seq) device forwards rather than one batched
    /// pass, so a long prompt prefills more slowly than the host's batched GEMMs. v0.3.1
    /// targets the *decode* gate (where this path is the win); a batched device prefill
    /// is the deferred IMMA prefill work (ADR 0013, follow-up). For the short prompts the
    /// decode gate uses this is immaterial.
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
        // v0.3.6: a multi-token forward (the prompt) is a single **batched M=P prefill** —
        // one device-resident forward over all tokens — instead of P sequential decode
        // steps (the TTFT cliff). A single token (decode) replays the M=1 CUDA graph. Both
        // are bit-identical to the per-token loop (the batch kernels share the M=1 order).
        let logits = if tokens.len() > 1 {
            model
                .prefill(tokens, positions)
                .map_err(|e| NnError::Backend(e.to_string()))?
        } else {
            model
                .step_graph(tokens[0], positions[0])
                .map_err(|e| NnError::Backend(e.to_string()))?
        };
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
        // defaulted `as_concrete` hook returns `None` for every non-CUDA backend).
        let Some(cuda) = self
            .backend
            .as_concrete()
            .and_then(|a| a.downcast_ref::<tritium_cuda::CudaBackend>())
        else {
            return Ok(false);
        };
        // A model with any dense (SALT / fp) projection cannot use the TQ2_0-only
        // resident decoder; it runs the host forward instead.
        let Some(spec) = Self::build_decode_spec(&self.weights, &self.config) else {
            return Ok(false);
        };
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
    ) -> Option<tritium_cuda::DecodeModelSpec<'a>> {
        use crate::layers::Projection;
        use tritium_cuda::{DecodeLayerSpec, DecodeLinearSpec, DecodeModelSpec};

        // The device-resident decoder is TQ2_0-only: every projection must be
        // ternary. A model carrying any dense (SALT / fp) projection returns `None`
        // here and runs the host-orchestrated forward instead.
        fn lin(p: &Projection) -> Option<DecodeLinearSpec<'_>> {
            let tl = p.as_ternary()?;
            Some(DecodeLinearSpec {
                weights: &*tl.weights,
                scales: &tl.scales,
            })
        }

        let layers = weights
            .layers
            .iter()
            .map(|b| {
                // The resident decoder is BitNet-only; a non-Relu2 MLP ⇒ no resident build
                // (fall back to the host path).
                let mlp = b.mlp.as_relu2()?;
                Some(DecodeLayerSpec {
                    attn_norm: &b.attn_norm,
                    attn_sub_norm: &b.attn_sub_norm,
                    ffn_norm: &b.ffn_norm,
                    ffn_sub_norm: &mlp.ffn_sub_norm,
                    q: lin(&b.q_proj)?,
                    k: lin(&b.k_proj)?,
                    v: lin(&b.v_proj)?,
                    o: lin(&b.o_proj)?,
                    gate: lin(&mlp.gate)?,
                    up: lin(&mlp.up)?,
                    down: lin(&mlp.down)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;

        Some(DecodeModelSpec {
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
        })
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

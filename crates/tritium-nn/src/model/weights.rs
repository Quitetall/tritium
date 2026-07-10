//! Loaded model weights, read from a GGUF file.
//!
//! Ternary weights take the I2_S → internal path ([`tritium_format`]) and are
//! uploaded to the backend as [`TernaryLinear`]; norms, the token embedding, and
//! the LM head are widened to host-side fp32 ([`crate::tensor`]). The loader maps
//! GGUF tensor names (`token_embd.weight`, `blk.N.*`, `output_norm.weight`) to
//! these fields.
//!
//! # BitNet 2B4T specifics (WF-4)
//!
//! - I2_S ternary tensors (`ggml_type == 36`) are sized as `n_elements/4 + 32`
//!   (the GGUF reader leaves `n_bytes == 0` for type-36) and decoded via
//!   [`tritium_format::unpack_i2s_tensor`] into `[N, K]` row-major trits + a
//!   per-tensor f32 scale. ggml stores a linear weight `[N_out, K_in]` with dims
//!   `[K_in, N_out]` (fastest-first), so `N_out = dims[1]`, `K_in = dims[0]`.
//! - `token_embd.weight` is F16 `[vocab, n_embd]`; widened to fp32. The model
//!   **ties** the LM head to it (`tie_word_embeddings = true`, no `output.weight`
//!   tensor in the GGUF), so the runner unembeds with this same matrix.
//! - Norms (`attn_norm`, `ffn_norm`, `attn_sub_norm`, `ffn_sub_norm`,
//!   `output_norm`) are F32 (`ggml_type == 0`) in this checkpoint; F16 is also
//!   accepted for portability.

use half::f16;
use tritium_core::Trit;
use tritium_format::{
    GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GgufFile, QK_K, TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES,
    TensorInfo, unpack_i2s_tensor, unpack_tq1_0_row, unpack_tq2_0_row,
};
use tritium_spec::TernaryBackend;

use crate::config::{ArchSpec, ModelConfig};
use crate::error::NnError;
use crate::layers::{Mlp, Projection, Relu2Mlp, TernaryLinear, TransformerBlock};
use crate::tensor::f16_bytes_to_f32;

/// The weights for one decoder layer, ready to run.
///
/// A thin alias around [`TransformerBlock`] today; kept as a distinct loader-side
/// type so the GGUF-name → block-field mapping has a home if it grows.
pub type LayerWeights = TransformerBlock;

/// ggml type-ids the loader consumes for dense (non-ternary) tensors.
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_I2_S: u32 = tritium_format::GGML_TYPE_I2_S;

/// All weights for a model: embeddings, per-layer blocks, final norm, LM head.
#[allow(missing_debug_implementations)]
pub struct ModelWeights {
    /// Token embedding table, fp32, `[vocab, n_embd]` row-major. Also used as the
    /// (tied) LM head: `logits = h · token_embd[v]` for each vocab row `v`.
    pub token_embd: Vec<f32>,
    /// Vocabulary size (rows of `token_embd`).
    pub vocab: usize,
    /// Hidden size (columns of `token_embd`).
    pub n_embd: usize,
    /// Per-layer transformer blocks, length `n_layers`.
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm weight before the LM head; length `n_embd`.
    pub output_norm: Vec<f32>,
    /// Untied LM head projection (`n_embd → vocab`). `None` ⇒ the head is **tied** to
    /// `token_embd` (BitNet and models with `tie_word_embeddings`).
    pub lm_head: Option<Projection>,
}

impl ModelWeights {
    /// Load all weights from a parsed GGUF `file` per `config`, uploading ternary
    /// tensors to `backend`. `bytes` is the full GGUF byte buffer (the reader does
    /// not retain payloads; the loader locates each with
    /// [`GgufFile::tensor_data_offset`] + [`TensorInfo::offset`]).
    ///
    /// # Errors
    /// - [`NnError::MissingTensor`] if a required tensor is absent.
    /// - [`NnError::UnsupportedTensorType`] if a tensor uses an unexpected ggml
    ///   type-id.
    /// - [`NnError::Backend`] if a weight upload fails, or [`NnError::Shape`] on a
    ///   malformed tensor.
    pub fn load(
        file: &GgufFile,
        bytes: &[u8],
        config: &ModelConfig,
        backend: &dyn TernaryBackend,
    ) -> Result<Self, NnError> {
        // GGUF-specific integrity check the generic builder (vocab =
        // len/n_embd) can't express: the file's declared embedding dims must
        // agree with the CONFIG's n_embd. Dims-vs-payload consistency is
        // already enforced by the reader (n_bytes is computed FROM dims and
        // bounds-checked), so element_count suffices — no decode needed
        // (review: the old full F16->f32 decode here was information-free,
        // ~657MB read + ~1.3GB transient per load).
        let n_embd = config.n_embd as usize;
        let embd_info = require(file, "token_embd.weight")?;
        let vocab = *embd_info
            .dims
            .last()
            .ok_or_else(|| NnError::MissingTensor("token_embd.weight (no dims)".to_owned()))?
            as usize;
        let embd_len = embd_info
            .element_count()
            .map_err(|e| NnError::Backend(format!("token_embd.weight dims: {e}")))?
            as usize;
        if embd_len != vocab * n_embd {
            return Err(NnError::Shape {
                expected: vocab * n_embd,
                got: embd_len,
            });
        }

        // P2e: one config-driven skeleton for every loading path — the GGUF
        // dialect supplies the name schema, `load_dense` the norms/embedding,
        // and `load_ternary` (backend upload) the projections.
        crate::model::hf::build_standard_model(
            config,
            &ArchSpec::bitnet(),
            crate::model::hf::NameSchema::Gguf,
            |name| load_dense(file, bytes, name),
            // Shape hints unused: load_ternary derives [N, K] from the
            // file's own dims (pre-existing behavior). TODO(non-BitNet GGUF):
            // check them against the config-derived n_out/k_in so a
            // config/file head_dim disagreement fails at load, not runtime.
            |name, _n_out, _k_in| {
                Ok(Projection::Ternary(load_ternary(file, bytes, backend, name)?))
            },
        )
    }
}

/// Look up a tensor or return [`NnError::MissingTensor`].
fn require<'a>(file: &'a GgufFile, name: &str) -> Result<&'a TensorInfo, NnError> {
    file.tensor(name)
        .ok_or_else(|| NnError::MissingTensor(name.to_owned()))
}

/// Borrow a tensor's payload bytes from the full GGUF buffer.
///
/// For I2_S (`n_bytes == 0` from the reader) the payload is sized as
/// `n_elements/4 + 32`; otherwise `n_bytes` from the reader is authoritative.
fn payload<'a>(file: &GgufFile, bytes: &'a [u8], info: &TensorInfo) -> Result<&'a [u8], NnError> {
    let start = (file.tensor_data_offset + info.offset) as usize;
    let n_elements = info
        .element_count()
        .map_err(|e| NnError::Backend(e.to_string()))? as usize;
    let len = if info.ggml_type == GGML_TYPE_I2_S {
        n_elements / 4 + 32
    } else {
        info.n_bytes as usize
    };
    bytes
        .get(start..start + len)
        .ok_or_else(|| NnError::MissingTensor(format!("{} (payload out of bounds)", info.name)))
}

/// Load a dense (F32 or F16) tensor as fp32, in ggml memory order.
fn load_dense(file: &GgufFile, bytes: &[u8], name: &str) -> Result<Vec<f32>, NnError> {
    let info = require(file, name)?;
    let p = payload(file, bytes, info)?;
    match info.ggml_type {
        GGML_TYPE_F32 => Ok(p
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        GGML_TYPE_F16 => Ok(f16_bytes_to_f32(p)),
        other => Err(NnError::UnsupportedTensorType(other)),
    }
}

/// Load a ternary tensor (I2_S, TQ1_0 or TQ2_0) into a [`TernaryLinear`].
///
/// ggml dims are `[K_in, N_out]` (fastest-first); the decoded trits are `[N, K]`
/// row-major and the per-tensor f32 scale becomes the linear's weight scale.
///
/// TQ1_0 / TQ2_0 carry a scale PER 256-BLOCK while the ternary stack is
/// per-tensor-scaled: pure ternary tensors round-trip exactly because every
/// block scale is either the tensor scale (block has a nonzero trit) or zero
/// (all-zero block — its trits decode to 0 regardless, and are forced to the
/// zero trit here so `trit × tensor_scale` reproduces it). Genuinely
/// non-uniform block scales are rejected loudly rather than silently
/// mis-scaled. This makes TQ1_0 files (18% smaller ternary payloads) a
/// first-class interchange format: every backend sees the identical trits +
/// scale it would have seen from the equivalent I2_S/TQ2_0 file.
fn load_ternary(
    file: &GgufFile,
    bytes: &[u8],
    backend: &dyn TernaryBackend,
    name: &str,
) -> Result<TernaryLinear, NnError> {
    let info = require(file, name)?;
    if info.dims.len() != 2 {
        return Err(NnError::Shape {
            expected: 2,
            got: info.dims.len(),
        });
    }
    let k_in = info.dims[0] as usize;
    let n_out = info.dims[1] as usize;
    let n_elements = n_out * k_in;

    let p = payload(file, bytes, info)?;
    let (trits, scale) = match info.ggml_type {
        GGML_TYPE_I2_S => {
            let mut trits = vec![Trit::ZERO; n_elements];
            let scale = unpack_i2s_tensor(p, n_elements, &mut trits)
                .map_err(|e| NnError::Backend(e.to_string()))?;
            (trits, scale)
        }
        t @ (GGML_TYPE_TQ1_0 | GGML_TYPE_TQ2_0) => {
            let block_bytes = if t == GGML_TYPE_TQ1_0 {
                TQ1_0_BLOCK_BYTES
            } else {
                TQ2_0_BLOCK_BYTES
            };
            let nb = k_in.div_ceil(QK_K);
            let row_bytes = nb * block_bytes;
            if p.len() < n_out * row_bytes {
                return Err(NnError::Backend(format!(
                    "{}: payload {} B < {} rows × {row_bytes} B",
                    info.name,
                    p.len(),
                    n_out
                )));
            }
            let mut trits = vec![Trit::ZERO; n_elements];
            let mut scales = vec![f16::ZERO; nb];
            let mut tensor_scale: Option<f32> = None;
            for r in 0..n_out {
                let row = &p[r * row_bytes..(r + 1) * row_bytes];
                let out = &mut trits[r * k_in..(r + 1) * k_in];
                if t == GGML_TYPE_TQ1_0 {
                    unpack_tq1_0_row(row, out, &mut scales)
                } else {
                    unpack_tq2_0_row(row, out, &mut scales)
                }
                .map_err(|e| NnError::Backend(e.to_string()))?;
                for (bi, &d) in scales.iter().enumerate() {
                    let dv = f32::from(d);
                    if dv == 0.0 {
                        // All-zero block: its decoded values are 0 at any
                        // scale; force the trits so trit × tensor_scale
                        // reproduces them exactly.
                        let start = bi * QK_K;
                        let end = (start + QK_K).min(k_in);
                        out[start..end].fill(Trit::ZERO);
                    } else {
                        match tensor_scale {
                            None => tensor_scale = Some(dv),
                            Some(sv) if sv == dv => {}
                            Some(sv) => {
                                return Err(NnError::Backend(format!(
                                    "{}: non-uniform block scales ({sv} vs {dv} at row {r} \
                                     block {bi}) — not a pure ternary tensor; refusing to \
                                     mis-scale it through the per-tensor ternary path",
                                    info.name
                                )));
                            }
                        }
                    }
                }
            }
            // `tritium repack` preserves the source I2_S f32 scale in
            // metadata (TQ block scales are f16 and real BitNet scales are
            // not f16-representable); prefer it so a repacked file loads
            // BIT-IDENTICALLY to its source. The f16 block scale must agree
            // with it to f16 precision — a mismatch means the metadata
            // belongs to a different tensor generation.
            let scale = match file
                .metadata
                .get(&format!("tritium.i2s_scale.{}", info.name))
                .and_then(tritium_format::GgufValue::as_f32)
            {
                Some(exact) => {
                    if let Some(sv) = tensor_scale {
                        let expect = f32::from(f16::from_f32(exact));
                        if sv != expect {
                            return Err(NnError::Backend(format!(
                                "{}: tritium.i2s_scale metadata ({exact}, f16 {expect}) \
                                 disagrees with the file's block scale ({sv}) — stale \
                                 metadata from a different tensor generation; delete the \
                                 key or re-run `tritium repack` from the source file",
                                info.name
                            )));
                        }
                    }
                    exact
                }
                None => tensor_scale.unwrap_or(0.0),
            };
            (trits, scale)
        }
        other => return Err(NnError::UnsupportedTensorType(other)),
    };

    TernaryLinear::new(backend, &trits, n_out, k_in, scale)
}

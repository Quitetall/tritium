//! Loaded model weights, read from a GGUF file.
//!
//! Ternary weights take the I2_S → internal path ([`tritium_format`]) and are
//! uploaded to the backend as [`TernaryLinear`]. Native GGUF norms, token embedding,
//! and LM head are widened to host-side fp32 ([`crate::tensor`]); SALT loading retains
//! its 2D token table and projections packed. The loader maps
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
    GGML_TYPE_Q2_0, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GgufFile, QK_K, TQ1_0_BLOCK_BYTES,
    TQ2_0_BLOCK_BYTES, TensorInfo, unpack_i2s_tensor, unpack_tq1_0_row, unpack_tq2_0_row,
};
use tritium_spec::TernaryBackend;

use crate::config::{ArchSpec, ModelConfig};
use crate::error::NnError;
use crate::layers::{Projection, Q2Linear, TernaryLinear, TokenEmbedding, TransformerBlock};
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
    /// Dense or packed token table, `[vocab, n_embd]`. Also used as the tied LM
    /// head: `logits = h · token_embd[v]` for each vocabulary row `v`.
    pub token_embd: TokenEmbedding,
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
        //
        // Untied ternary head (ADR 0032 L1): BitNet ties by default, but a GGUF
        // carrying an explicit `output.weight` is UNTIED — load that tensor as
        // a separate (ternary) lm_head instead of dotting against token_embd.
        // The tensor's presence is the source of truth, not the arch default.
        let mut arch = ArchSpec::bitnet();
        if file.tensor("output.weight").is_some() {
            arch.tied_embeddings = false;
        }
        crate::model::hf::build_standard_model(
            config,
            &arch,
            crate::model::hf::NameSchema::Gguf,
            |name, _expected_len| load_dense(file, bytes, name),
            // Shape hints unused: load_ternary derives [N, K] from the
            // file's own dims (pre-existing behavior). TODO(non-BitNet GGUF):
            // check them against the config-derived n_out/k_in so a
            // config/file head_dim disagreement fails at load, not runtime.
            |name, _n_out, _k_in| load_projection(file, bytes, backend, name),
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

/// Load an explicit GGUF ternary projection without byte sniffing.
fn load_projection(
    file: &GgufFile,
    bytes: &[u8],
    backend: &dyn TernaryBackend,
    name: &str,
) -> Result<Projection, NnError> {
    let info = require(file, name)?;
    if info.ggml_type != GGML_TYPE_Q2_0 {
        return Ok(Projection::Ternary(load_ternary(
            file, bytes, backend, name,
        )?));
    }
    if info.dims.len() != 2 {
        return Err(NnError::Shape {
            expected: 2,
            got: info.dims.len(),
        });
    }
    let k_in = usize::try_from(info.dims[0])
        .map_err(|_| NnError::Backend(format!("{name}: Q2_0 K exceeds usize")))?;
    let n_out = usize::try_from(info.dims[1])
        .map_err(|_| NnError::Backend(format!("{name}: Q2_0 N exceeds usize")))?;
    let source = payload(file, bytes, info)?;
    let mut packed = Vec::new();
    packed.try_reserve_exact(source.len()).map_err(|error| {
        NnError::Backend(format!(
            "allocate Q2_0 payload for {} bytes: {error}",
            source.len()
        ))
    })?;
    packed.extend_from_slice(source);
    let exact_scale = file
        .metadata
        .get(&format!("tritium.i2s_scale.{name}"))
        .and_then(tritium_format::GgufValue::as_f32);
    Ok(Projection::Q2(Q2Linear::new_with_uniform_scale_override(
        packed,
        n_out,
        k_in,
        exact_scale,
    )?))
}

/// Load an I2_S, TQ1_0 or TQ2_0 tensor into a [`TernaryLinear`].
///
/// ggml dims are `[K_in, N_out]` (fastest-first); the decoded trits are `[N, K]`
/// row-major and the per-tensor f32 scale becomes the linear's weight scale.
///
/// TQ1_0 / TQ2_0 carry one scale per 256-element block. The device ternary
/// stack is per-output-row-scaled: every nonzero block in a row must therefore
/// share one scale, while zero-scale blocks are forced to zero trits. Standard
/// Q2_0 uses [`Q2Linear`] instead so its per-64-group scales remain exact.
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
            let block_bytes = match t {
                GGML_TYPE_TQ1_0 => TQ1_0_BLOCK_BYTES,
                GGML_TYPE_TQ2_0 => TQ2_0_BLOCK_BYTES,
                _ => unreachable!("match arm admits only TQ1_0/TQ2_0"),
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
            // ADR 0032 T2a: block scales must be uniform WITHIN a row, but
            // rows may differ (a per-row-scaled ternary head). A tensor whose
            // rows all agree stays on the per-tensor path (bit-identical to
            // the old loader); differing rows load through
            // `TernaryLinear::with_row_scales` — the GEMM contract is per-row
            // (`weight_scale[n]`) either way.
            let mut row_scales: Vec<f32> = Vec::with_capacity(n_out);
            let mut rows_uniform = true;
            for r in 0..n_out {
                let row = &p[r * row_bytes..(r + 1) * row_bytes];
                let out = &mut trits[r * k_in..(r + 1) * k_in];
                match t {
                    GGML_TYPE_TQ1_0 => unpack_tq1_0_row(row, out, &mut scales),
                    GGML_TYPE_TQ2_0 => unpack_tq2_0_row(row, out, &mut scales),
                    _ => unreachable!("match arm admits only TQ1_0/TQ2_0"),
                }
                .map_err(|e| NnError::Backend(e.to_string()))?;
                let mut row_scale: Option<f32> = None;
                for (bi, &d) in scales.iter().enumerate() {
                    let dv = f32::from(d);
                    if dv == 0.0 {
                        // All-zero block: its decoded values are 0 at any
                        // scale; force the trits so trit × scale reproduces
                        // them exactly.
                        let start = bi * QK_K;
                        let end = (start + QK_K).min(k_in);
                        out[start..end].fill(Trit::ZERO);
                    } else {
                        match row_scale {
                            None => row_scale = Some(dv),
                            Some(sv) if sv == dv => {}
                            Some(sv) => {
                                return Err(NnError::Backend(format!(
                                    "{}: non-uniform block scales ({sv} vs {dv} within row \
                                     {r} block {bi}) — not a pure ternary row; refusing to \
                                     mis-scale it through the ternary path",
                                    info.name
                                )));
                            }
                        }
                    }
                }
                // An all-zero row inherits the tensor scale (its values are 0
                // at any scale); record 0.0 for now and patch after the loop.
                let rs = row_scale.unwrap_or(0.0);
                match (tensor_scale, row_scale) {
                    (None, Some(dv)) => tensor_scale = Some(dv),
                    (Some(sv), Some(dv)) if sv != dv => rows_uniform = false,
                    _ => {}
                }
                row_scales.push(rs);
            }
            // `tritium repack` preserves the source I2_S f32 scale in
            // metadata (TQ block scales are f16 and real BitNet scales are
            // not f16-representable); prefer it so a repacked file loads
            // BIT-IDENTICALLY to its source. The f16 block scale must agree
            // with it to f16 precision — a mismatch means the metadata
            // belongs to a different tensor generation.
            if !rows_uniform {
                // Per-row-scaled tensor: all-zero rows inherit any nonzero
                // scale (their decode is 0 regardless; a nonzero placeholder
                // avoids a degenerate 0-scale row in the GEMM contract).
                let fill = tensor_scale.unwrap_or(0.0);
                for rs in &mut row_scales {
                    if *rs == 0.0 {
                        *rs = fill;
                    }
                }
                return TernaryLinear::with_row_scales(backend, &trits, n_out, k_in, row_scales);
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal single-tensor GGUF v3 blob: one TQ2_0 tensor `name`,
    /// `n_out` rows × `k_in` cols, per-ROW scales (uniform within each row).
    fn gguf_with_tq2(name: &str, n_out: usize, k_in: usize, row_scales: &[f32]) -> Vec<u8> {
        use tritium_format::pack_tq2_0_row;
        assert_eq!(k_in % QK_K, 0, "test rows are whole blocks");
        let nb = k_in / QK_K;
        let row_bytes = nb * TQ2_0_BLOCK_BYTES;
        let mut data = vec![0u8; n_out * row_bytes];
        for (r, &sc) in row_scales.iter().enumerate() {
            // Deterministic trits: row r = repeating [+1, -1, 0, ...] shifted.
            let trits: Vec<Trit> = (0..k_in)
                .map(|i| Trit::from_i8((((i + r) % 3) as i8) - 1).unwrap())
                .collect();
            let scales = vec![f16::from_f32(sc); nb];
            pack_tq2_0_row(
                &trits,
                &scales,
                &mut data[r * row_bytes..(r + 1) * row_bytes],
            )
            .expect("pack row");
        }
        let mut b = Vec::new();
        let s = |b: &mut Vec<u8>, t: &str| {
            b.extend((t.len() as u64).to_le_bytes());
            b.extend(t.as_bytes());
        };
        b.extend(0x46554747u32.to_le_bytes()); // GGUF
        b.extend(3u32.to_le_bytes());
        b.extend(1u64.to_le_bytes()); // 1 tensor
        b.extend(0u64.to_le_bytes()); // 0 kv
        s(&mut b, name);
        b.extend(2u32.to_le_bytes()); // ndims
        b.extend((k_in as u64).to_le_bytes()); // fastest-first
        b.extend((n_out as u64).to_le_bytes());
        b.extend(GGML_TYPE_TQ2_0.to_le_bytes());
        b.extend(0u64.to_le_bytes()); // offset
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend(&data);
        b
    }

    /// Minimal single-tensor GGUF v3 blob carrying standard llama.cpp Q2_0.
    fn gguf_with_q2(name: &str, n_out: usize, k_in: usize, row_scales: &[f32]) -> Vec<u8> {
        use tritium_format::{GGML_TYPE_Q2_0, Q2_0_BLOCK_BYTES, Q2_0_GROUP_SIZE, pack_q2_0_row};
        assert_eq!(k_in % Q2_0_GROUP_SIZE, 0, "test rows are whole Q2_0 blocks");
        assert_eq!(row_scales.len(), n_out);
        let nb = k_in / Q2_0_GROUP_SIZE;
        let row_bytes = nb * Q2_0_BLOCK_BYTES;
        let mut data = vec![0u8; n_out * row_bytes];
        for (r, &scale) in row_scales.iter().enumerate() {
            let trits: Vec<Trit> = (0..k_in)
                .map(|i| Trit::from_i8((((i + r) % 3) as i8) - 1).unwrap())
                .collect();
            let scales = vec![f16::from_f32(scale); nb];
            pack_q2_0_row(
                &trits,
                &scales,
                &mut data[r * row_bytes..(r + 1) * row_bytes],
            )
            .expect("pack Q2_0 row");
        }
        let mut b = Vec::new();
        let s = |b: &mut Vec<u8>, t: &str| {
            b.extend((t.len() as u64).to_le_bytes());
            b.extend(t.as_bytes());
        };
        b.extend(0x46554747u32.to_le_bytes());
        b.extend(3u32.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        s(&mut b, name);
        b.extend(2u32.to_le_bytes());
        b.extend((k_in as u64).to_le_bytes());
        b.extend((n_out as u64).to_le_bytes());
        b.extend(GGML_TYPE_Q2_0.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend(&data);
        b
    }

    fn load(name: &str, blob: &[u8]) -> TernaryLinear {
        let file = tritium_format::read_gguf(blob).expect("parse gguf");
        let backend = tritium_cpu::CpuBackend::new();
        load_ternary(&file, blob, &backend, name).expect("load ternary")
    }

    /// ADR 0032 T2a: rows with DIFFERING (row-uniform) scales load as a
    /// per-row-scaled TernaryLinear; a uniform tensor keeps the old
    /// replicated-scalar behavior (bit-identical back-compat).
    #[test]
    fn tq2_row_scales_load_per_row_and_uniform_stays_scalar() {
        let rows = [1.0f32, 2.0, 0.5];
        let lin = load(
            "output.weight",
            &gguf_with_tq2("output.weight", 3, 256, &rows),
        );
        assert_eq!(lin.n_out, 3);
        assert_eq!(
            lin.scales,
            rows.to_vec(),
            "per-row scales must survive load"
        );

        let uni = load(
            "output.weight",
            &gguf_with_tq2("output.weight", 3, 256, &[0.75; 3]),
        );
        assert_eq!(
            uni.scales,
            vec![0.75f32; 3],
            "uniform stays replicated scalar"
        );
    }

    /// Within-row non-uniformity must still refuse loudly (the original
    /// mis-scale guard, now scoped per row).
    #[test]
    fn tq2_within_row_nonuniform_refuses() {
        use tritium_format::pack_tq2_0_row;
        let (n_out, k_in) = (1usize, 512usize);
        let nb = k_in / QK_K;
        let row_bytes = nb * TQ2_0_BLOCK_BYTES;
        let mut data = vec![0u8; row_bytes];
        let trits: Vec<Trit> = (0..k_in)
            .map(|i| Trit::from_i8(((i % 3) as i8) - 1).unwrap())
            .collect();
        // Two blocks in ONE row with different scales.
        let scales = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        pack_tq2_0_row(&trits, &scales, &mut data).expect("pack row");
        let mut blob = gguf_with_tq2("output.weight", n_out, k_in, &[1.0]);
        let dlen = data.len();
        let start = blob.len() - dlen;
        blob[start..].copy_from_slice(&data);
        let file = tritium_format::read_gguf(&blob).expect("parse gguf");
        let backend = tritium_cpu::CpuBackend::new();
        let err = load_ternary(&file, &blob, &backend, "output.weight");
        assert!(err.is_err(), "within-row non-uniform scales must refuse");
    }

    #[test]
    fn q2_row_scales_load_and_forward_matches_tq2_sibling() {
        let backend = tritium_cpu::CpuBackend::new();
        let row_scales = [1.0f32, 2.0, 0.5];
        let q2_blob = gguf_with_q2("output.weight", 3, 256, &row_scales);
        let tq2_blob = gguf_with_tq2("output.weight", 3, 256, &row_scales);
        let q2_file = tritium_format::read_gguf(&q2_blob).expect("parse Q2_0 GGUF");
        let tq2_file = tritium_format::read_gguf(&tq2_blob).expect("parse TQ2_0 GGUF");
        let q2 = load_projection(&q2_file, &q2_blob, &backend, "output.weight")
            .expect("load Q2_0 projection");
        let tq2 = load_projection(&tq2_file, &tq2_blob, &backend, "output.weight")
            .expect("load TQ2_0 projection");

        assert!(matches!(q2, Projection::Q2(_)));
        assert!(matches!(tq2, Projection::Ternary(_)));
        let act: Vec<f32> = (0..2 * 256)
            .map(|i| ((i % 29) as f32 - 14.0) / 7.0)
            .collect();
        let mut q2_out = vec![0.0f32; 2 * 3];
        let mut tq2_out = vec![0.0f32; 2 * 3];
        q2.forward(&backend, &act, 2, &mut q2_out)
            .expect("Q2_0 forward");
        tq2.forward(&backend, &act, 2, &mut tq2_out)
            .expect("TQ2_0 forward");
        assert_eq!(q2_out, tq2_out, "normalized siblings must be bit-identical");
    }

    #[test]
    fn q2_within_row_scales_remain_exact_without_dense_shadow() {
        use tritium_format::{Q2_0_BLOCK_BYTES, Q2_0_GROUP_SIZE, pack_q2_0_row};

        let (n_out, k_in) = (1usize, 256usize);
        let nb = k_in / Q2_0_GROUP_SIZE;
        let row_bytes = nb * Q2_0_BLOCK_BYTES;
        let trits: Vec<Trit> = (0..k_in)
            .map(|i| Trit::from_i8(((i % 3) as i8) - 1).unwrap())
            .collect();
        let scales = [1.0f32, 1.0, 2.0, 1.0].map(f16::from_f32);
        let mut data = vec![0u8; row_bytes];
        pack_q2_0_row(&trits, &scales, &mut data).expect("pack nonuniform Q2_0 row");
        let mut blob = gguf_with_q2("output.weight", n_out, k_in, &[1.0]);
        let start = blob.len() - data.len();
        blob[start..].copy_from_slice(&data);
        let file = tritium_format::read_gguf(&blob).expect("parse Q2_0 GGUF");
        let backend = tritium_cpu::CpuBackend::new();
        let projection = load_projection(&file, &blob, &backend, "output.weight")
            .expect("load nonuniform Q2_0 projection");
        let Projection::Q2(q2) = &projection else {
            panic!("standard Q2_0 must stay packed");
        };
        assert_eq!(q2.packed_bytes(), row_bytes);

        let act: Vec<f32> = (0..k_in).map(|i| ((i % 17) as f32 - 8.0) / 4.0).collect();
        let mut got = [0.0f32; 1];
        projection
            .forward(&backend, &act, 1, &mut got)
            .expect("Q2_0 forward");
        let weights: Vec<f32> = trits
            .iter()
            .enumerate()
            .map(|(index, trit)| f32::from(scales[index / Q2_0_GROUP_SIZE]) * f32::from(trit.get()))
            .collect();
        let dense = crate::layers::DenseLinear::new(weights, 1, k_in).expect("dense oracle");
        let mut want = [0.0f32; 1];
        dense
            .forward(&act, 1, &mut want)
            .expect("dense Q2_0 oracle");
        assert_eq!(got, want, "per-group Q2_0 scales must remain exact");
    }
}

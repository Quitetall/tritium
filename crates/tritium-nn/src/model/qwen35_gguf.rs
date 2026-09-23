//! Qwen3.5-family language models from llama.cpp's `qwen35` GGUF — including PrismML's
//! Hadamard-folded ternary checkpoints (Ternary Bonsai 2 27B, a ternary Qwen3.8-27B).
//!
//! The hybrid graph itself is Tritium's existing [`Qwen35TextRunner`]. This module is only a
//! [`Qwen35HfTensorSource`]: it answers the Hugging Face tensor names that
//! [`super::qwen35_hf::load_language_weights`] asks for, by reading the GGUF tensor that
//! llama.cpp's converter produced and **undoing every transform the converter applied**. Getting
//! any one of these wrong still loads and runs; it just computes a different function.
//!
//! What the converter (`conversion/qwen.py`, `Qwen3NextModel` + `_LinearAttentionVReorderBase`)
//! does to a Qwen3.5 checkpoint, and what this reader does back:
//!
//! | HF tensor | GGUF | converter | undone here |
//! |---|---|---|---|
//! | `*_layernorm`, `norm`, `q_norm`, `k_norm` | `*_norm` | `w + 1` | `g − 1` (exact: Sterbenz) |
//! | `linear_attn.norm` | `ssm_norm` | none | none |
//! | `A_log` | `ssm_a` | `−exp(A_log)`, V heads tiled | `ln(−a)`, heads regrouped |
//! | `dt_bias` | `ssm_dt.bias` | V heads tiled | regrouped |
//! | `conv1d` | `ssm_conv1d` | squeezed, V channels tiled | V channels regrouped |
//! | `in_proj_qkv` | `attn_qkv` | V rows tiled | V rows regrouped |
//! | `in_proj_z` | `attn_gate` | rows tiled | regrouped |
//! | `in_proj_a` / `in_proj_b` | `ssm_alpha` / `ssm_beta` | rows tiled | regrouped |
//! | `out_proj` | `ssm_out` | kept grouped when Hadamard-folded | none |
//!
//! "Tiled" is llama.cpp's broadcast order for the 48 DeltaNet value heads over 16 key heads:
//! `[v0 of every k-head, v1 of every k-head, …]`, where Hugging Face groups each key head's value
//! heads together. Every regrouping is a permutation of whole rows (or scalars), so it is exact on
//! packed ternary rows, which are independent per output channel.
//!
//! **The basis.** A PrismML checkpoint declares `prism.hadamard.*`: every folded weight stores
//! `W·Bᵀ` with `B = H₁₀₂₄·D`, `D` an explicit ±1 vector per input width. Each such projection is
//! given its [`SignedBlockHadamard`], so it rotates its own input; the embedding table is declared
//! inverse-after-lookup and un-rotates the rows it gathers. `ssm_out` is folded in HF (grouped)
//! order — llama.cpp permutes its tiled activation back before rotating — and this runner's
//! activations are already grouped, so it needs no permutation at all.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use half::bf16;
use tritium_format::{GgufError, GgufFile, GgufValue, TensorInfo, read_gguf_prefix};
use tritium_format::{
    PQ2_0_BLOCK_BYTES, PQ2_0_GROUP_SIZE, Q2_0_BLOCK_BYTES, split_pq2_0_into_q2_0,
};
use tritium_spec::TernaryBackend;

use super::qwen35::Qwen35TextRunner;
use super::qwen35_hf::{Qwen35HfTensorSource, load_language_weights};
use crate::error::NnError;
use crate::layers::{DenseLinear, Projection, Q2Linear, SignedBlockHadamard, TokenEmbedding};
use crate::qwen35_config::{Qwen35CheckpointConfig, Qwen35TextConfig};

/// ggml type ids this reader accepts.
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_BF16: u32 = 30;
/// PrismML's private group-128 Q2_0 (`block_pq2_0`), ggml type 142.
const GGML_TYPE_PQ2_0: u32 = 142;

/// A Qwen3.5-family language model loaded from a `qwen35` GGUF.
#[allow(missing_debug_implementations)]
pub struct Qwen35GgufLanguageModel {
    config: Qwen35CheckpointConfig,
    runner: Qwen35TextRunner,
    receipt: Qwen35GgufLoadReceipt,
}

/// What a GGUF load consumed and what it had to reconstruct.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Qwen35GgufLoadReceipt {
    /// Projections stored as group-128 ternary (PQ2_0).
    pub ternary_projections: usize,
    /// Of those, projections carrying a folded Hadamard basis.
    pub folded_projections: usize,
    /// Whether the token table is a folded ternary table un-rotated at gather.
    pub folded_embedding: bool,
    /// Hadamard block width, when the checkpoint declares one.
    pub hadamard_block: Option<usize>,
}

impl Qwen35GgufLanguageModel {
    /// Load a Qwen3.5-family language model from `gguf` and the base model's `config.json`.
    ///
    /// The GGUF's own architecture keys are checked against `config_json` field by field and any
    /// disagreement fails the load: the HF config is what the runner is built from, so it has to
    /// describe the same network the tensors belong to.
    ///
    /// # Errors
    /// [`NnError::MissingConfig`] for a bad config or a config/GGUF mismatch,
    /// [`NnError::MissingTensor`] for an absent or mistyped tensor, and [`NnError::Backend`] for a
    /// malformed file or basis.
    pub fn load(
        gguf: &Path,
        config_json: &str,
        backend: Box<dyn TernaryBackend>,
    ) -> Result<Self, NnError> {
        let config = Qwen35CheckpointConfig::from_hf_config(config_json)?;
        let source = Qwen35GgufSource::open(gguf, &config.text)?;
        let weights = load_language_weights(&source, &config.text)?;
        let receipt = source.receipt();
        let runner = Qwen35TextRunner::new(&config.text, weights, backend)?;
        Ok(Self {
            config,
            runner,
            receipt,
        })
    }

    /// Validated model configuration.
    #[must_use]
    pub const fn config(&self) -> &Qwen35CheckpointConfig {
        &self.config
    }

    /// The hybrid language runner.
    #[must_use]
    pub const fn runner(&self) -> &Qwen35TextRunner {
        &self.runner
    }

    /// What the load consumed.
    #[must_use]
    pub const fn receipt(&self) -> &Qwen35GgufLoadReceipt {
        &self.receipt
    }
}

/// Geometry of the DeltaNet value-head reorder.
#[derive(Clone, Copy, Debug)]
struct HeadTiling {
    key_heads: usize,
    values_per_key: usize,
}

impl HeadTiling {
    /// Regroup `units` head-blocks from llama.cpp's tiled order back to HF's grouped order.
    ///
    /// Each head owns `unit_len` consecutive elements (rows × row length, or 1 for a per-head
    /// scalar). Grouped head `k·r + j` (key head `k`, value `j`) sits at tiled position `j·K + k`.
    fn regroup<T: Copy>(self, tiled: &[T], unit_len: usize) -> Vec<T> {
        let heads = self.key_heads * self.values_per_key;
        debug_assert_eq!(tiled.len(), heads * unit_len);
        let mut grouped = Vec::with_capacity(tiled.len());
        for key in 0..self.key_heads {
            for value in 0..self.values_per_key {
                let source = value * self.key_heads + key;
                grouped.extend_from_slice(&tiled[source * unit_len..(source + 1) * unit_len]);
            }
        }
        grouped
    }
}

struct Qwen35GgufSource {
    /// Payloads are read on demand at `tensor_data_offset + offset`; only the header is held.
    handle: RefCell<File>,
    file: GgufFile,
    tensors: BTreeMap<String, TensorInfo>,
    tiling: HeadTiling,
    value_head_dim: usize,
    key_width: usize,
    /// Signed block Hadamard per input width, when the checkpoint declares one.
    bases: BTreeMap<usize, Arc<SignedBlockHadamard>>,
    hadamard_block: Option<usize>,
    folded: BTreeSet<String>,
    inverse: BTreeSet<String>,
    ternary_count: Cell<usize>,
    folded_count: Cell<usize>,
    folded_embedding: Cell<bool>,
}

impl Qwen35GgufSource {
    fn open(path: &Path, text: &Qwen35TextConfig) -> Result<Self, NnError> {
        let io = |error: std::io::Error| NnError::Backend(format!("{}: {error}", path.display()));
        let mut handle = File::open(path).map_err(io)?;
        let file_len = handle.metadata().map_err(io)?.len();
        // The header (tokenizer vocab included) is tens of megabytes at most; read a prefix and
        // grow it only if the header runs past it.
        let mut prefix_len = 64u64 << 20;
        let file = loop {
            let len = prefix_len.min(file_len);
            let mut prefix = vec![
                0u8;
                usize::try_from(len).map_err(|_| {
                    NnError::Backend("GGUF header prefix does not fit in memory".into())
                })?
            ];
            handle.seek(SeekFrom::Start(0)).map_err(io)?;
            handle.read_exact(&mut prefix).map_err(io)?;
            match read_gguf_prefix(&prefix, file_len) {
                Ok(file) => break file,
                Err(GgufError::Truncated) if len < file_len => prefix_len *= 4,
                Err(error) => return Err(NnError::Backend(format!("GGUF: {error}"))),
            }
        };
        let tensors = file
            .tensors
            .iter()
            .map(|info| (info.name.clone(), info.clone()))
            .collect();
        let dn = &text.delta_net;
        let tiling = HeadTiling {
            key_heads: dn.num_key_heads as usize,
            values_per_key: (dn.num_value_heads / dn.num_key_heads) as usize,
        };
        let mut source = Self {
            handle: RefCell::new(handle),
            file,
            tensors,
            tiling,
            value_head_dim: dn.value_head_dim as usize,
            key_width: (dn.num_key_heads * dn.key_head_dim) as usize,
            bases: BTreeMap::new(),
            hadamard_block: None,
            folded: BTreeSet::new(),
            inverse: BTreeSet::new(),
            ternary_count: 0.into(),
            folded_count: 0.into(),
            folded_embedding: false.into(),
        };
        source.check_architecture(text)?;
        source.read_basis()?;
        Ok(source)
    }

    fn receipt(&self) -> Qwen35GgufLoadReceipt {
        Qwen35GgufLoadReceipt {
            ternary_projections: self.ternary_count.get(),
            folded_projections: self.folded_count.get(),
            folded_embedding: self.folded_embedding.get(),
            hadamard_block: self.hadamard_block,
        }
    }

    fn meta(&self, key: &str) -> Option<&GgufValue> {
        self.file.get_metadata(key)
    }

    fn meta_u64(&self, key: &str) -> Result<u64, NnError> {
        self.meta(key)
            .and_then(GgufValue::as_u64)
            .ok_or_else(|| NnError::MissingConfig(format!("GGUF key {key} absent or not integral")))
    }

    /// Every architecture key the GGUF carries must agree with the HF config.
    fn check_architecture(&self, text: &Qwen35TextConfig) -> Result<(), NnError> {
        let arch = self
            .meta("general.architecture")
            .and_then(GgufValue::as_str);
        if arch != Some("qwen35") {
            return Err(NnError::MissingConfig(format!(
                "GGUF architecture {arch:?} is not qwen35"
            )));
        }
        let fa = &text.full_attention;
        let dn = &text.delta_net;
        let checks: [(&str, u64); 12] = [
            ("qwen35.block_count", text.layer_types.len() as u64),
            ("qwen35.embedding_length", u64::from(text.hidden_size)),
            (
                "qwen35.feed_forward_length",
                u64::from(text.intermediate_size),
            ),
            ("qwen35.attention.head_count", u64::from(fa.num_heads)),
            (
                "qwen35.attention.head_count_kv",
                u64::from(fa.num_key_value_heads),
            ),
            ("qwen35.attention.key_length", u64::from(fa.head_dim)),
            ("qwen35.ssm.conv_kernel", u64::from(dn.conv_kernel_dim)),
            ("qwen35.ssm.state_size", u64::from(dn.key_head_dim)),
            ("qwen35.ssm.group_count", u64::from(dn.num_key_heads)),
            ("qwen35.ssm.time_step_rank", u64::from(dn.num_value_heads)),
            (
                "qwen35.ssm.inner_size",
                u64::from(dn.num_value_heads) * u64::from(dn.value_head_dim),
            ),
            (
                "qwen35.rope.dimension_count",
                u64::from(text.rope.rotary_dim),
            ),
        ];
        for (key, want) in checks {
            let got = self.meta_u64(key)?;
            if got != want {
                return Err(NnError::MissingConfig(format!(
                    "GGUF {key} = {got} but config.json implies {want}"
                )));
            }
        }
        if self.meta_u64("qwen35.full_attention_interval")? == 0 {
            return Err(NnError::MissingConfig(
                "GGUF full_attention_interval is zero".into(),
            ));
        }
        Ok(())
    }

    /// Read `prism.hadamard.*`, accepting exactly the transform this runner implements.
    fn read_basis(&mut self) -> Result<(), NnError> {
        let Some(version) = self.meta("prism.hadamard.version") else {
            return Ok(());
        };
        if version.as_u64() != Some(1) {
            return Err(NnError::Backend(format!(
                "unsupported prism.hadamard.version {version:?}"
            )));
        }
        let string = |key: &str| -> Result<String, NnError> {
            self.meta(key)
                .and_then(GgufValue::as_str)
                .map(str::to_owned)
                .ok_or_else(|| NnError::Backend(format!("{key} absent or not a string")))
        };
        let expect = |key: &str, want: &str| -> Result<(), NnError> {
            let got = string(key)?;
            if got == want {
                Ok(())
            } else {
                Err(NnError::Backend(format!("{key} = {got:?}, need {want:?}")))
            }
        };
        expect(
            "prism.hadamard.transform",
            "normalized-sylvester-walsh-hadamard",
        )?;
        expect("prism.hadamard.axis", "input-last-dimension")?;
        expect("prism.hadamard.sign_mode", "explicit")?;
        let block = usize::try_from(self.meta_u64("prism.hadamard.block_size")?)
            .map_err(|_| NnError::Backend("Hadamard block does not fit usize".into()))?;
        let ints = |key: &str| -> Result<Vec<i64>, NnError> {
            self.meta(key)
                .and_then(GgufValue::as_array)
                .ok_or_else(|| NnError::Backend(format!("{key} absent or not an array")))?
                .iter()
                .map(|value| {
                    value
                        .as_i64()
                        .ok_or_else(|| NnError::Backend(format!("{key} has a non-integer entry")))
                })
                .collect()
        };
        let widths = ints("prism.hadamard.sign_widths")?;
        let values = ints("prism.hadamard.sign_values")?;
        let mut offset = 0usize;
        for width in widths {
            let width = usize::try_from(width)
                .map_err(|_| NnError::Backend(format!("negative Hadamard width {width}")))?;
            let signs = values
                .get(offset..offset + width)
                .ok_or_else(|| NnError::Backend("Hadamard sign values are short".into()))?
                .iter()
                .map(|v| *v as f32)
                .collect();
            offset += width;
            self.bases
                .insert(width, Arc::new(SignedBlockHadamard::new(block, signs)?));
        }
        if offset != values.len() {
            return Err(NnError::Backend(
                "Hadamard sign values are longer than their widths".into(),
            ));
        }
        let names = |key: &str| -> Result<BTreeSet<String>, NnError> {
            match self.meta(key) {
                None => Ok(BTreeSet::new()),
                Some(value) => value
                    .as_array()
                    .ok_or_else(|| NnError::Backend(format!("{key} is not an array")))?
                    .iter()
                    .map(|v| {
                        v.as_str().map(str::to_owned).ok_or_else(|| {
                            NnError::Backend(format!("{key} has a non-string entry"))
                        })
                    })
                    .collect(),
            }
        };
        let folded = names("prism.hadamard.weight_names")?;
        let inverse = names("prism.hadamard.inverse_weight_names")?;
        self.folded = folded;
        self.inverse = inverse;
        if self.inverse.iter().any(|name| name != "token_embd.weight") {
            return Err(NnError::Backend(
                "only token_embd.weight may be declared inverse-after-lookup".into(),
            ));
        }
        self.hadamard_block = Some(block);
        Ok(())
    }

    fn info(&self, name: &str) -> Result<&TensorInfo, NnError> {
        self.tensors
            .get(name)
            .ok_or_else(|| NnError::MissingTensor(format!("GGUF tensor {name}")))
    }

    /// The first `len` payload bytes of a tensor, read from the file.
    fn bytes(&self, info: &TensorInfo, len: usize) -> Result<Vec<u8>, NnError> {
        let io = |error: std::io::Error| NnError::Backend(format!("read {}: {error}", info.name));
        let start = self
            .file
            .tensor_data_offset
            .checked_add(info.offset)
            .ok_or_else(|| NnError::Backend(format!("{} offset overflows", info.name)))?;
        let mut out = vec![0u8; len];
        let mut handle = self.handle.borrow_mut();
        handle.seek(SeekFrom::Start(start)).map_err(io)?;
        handle.read_exact(&mut out).map_err(io)?;
        Ok(out)
    }

    /// `rows × cols` in HF orientation: ggml stores `ne0 = cols` fastest, `ne1 = rows`.
    fn check_matrix(&self, info: &TensorInfo, rows: usize, cols: usize) -> Result<(), NnError> {
        if info.dims != [cols as u64, rows as u64] {
            return Err(NnError::MissingTensor(format!(
                "{} has ggml dims {:?}, expected [{cols}, {rows}]",
                info.name, info.dims
            )));
        }
        Ok(())
    }

    fn f32_values(&self, info: &TensorInfo) -> Result<Vec<f32>, NnError> {
        let n = usize::try_from(
            info.element_count()
                .map_err(|e| NnError::Backend(e.to_string()))?,
        )
        .map_err(|_| NnError::Backend("element count overflows".into()))?;
        match info.ggml_type {
            GGML_TYPE_F32 => Ok(self
                .bytes(info, n * 4)?
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            GGML_TYPE_BF16 => Ok(self
                .bytes(info, n * 2)?
                .chunks(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect()),
            other => Err(NnError::MissingTensor(format!(
                "{} has ggml type {other}, expected F32 or BF16",
                info.name
            ))),
        }
    }

    /// Map an HF language tensor name to its GGUF name.
    fn gguf_name(hf: &str) -> Option<String> {
        if hf == "model.language_model.embed_tokens.weight" {
            return Some("token_embd.weight".into());
        }
        if hf == "model.language_model.norm.weight" {
            return Some("output_norm.weight".into());
        }
        if hf == "lm_head.weight" {
            return Some("output.weight".into());
        }
        let rest = hf.strip_prefix("model.language_model.layers.")?;
        let (layer, suffix) = rest.split_once('.')?;
        let gguf = match suffix {
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "post_attention_norm.weight",
            "linear_attn.in_proj_qkv.weight" => "attn_qkv.weight",
            "linear_attn.in_proj_z.weight" => "attn_gate.weight",
            "linear_attn.in_proj_a.weight" => "ssm_alpha.weight",
            "linear_attn.in_proj_b.weight" => "ssm_beta.weight",
            "linear_attn.out_proj.weight" => "ssm_out.weight",
            "linear_attn.conv1d.weight" => "ssm_conv1d.weight",
            "linear_attn.norm.weight" => "ssm_norm.weight",
            "linear_attn.dt_bias" => "ssm_dt.bias",
            "linear_attn.A_log" => "ssm_a",
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            _ => return None,
        };
        Some(format!("blk.{layer}.{gguf}"))
    }

    fn resolve(&self, hf: &str) -> Result<(String, &TensorInfo), NnError> {
        let name = Self::gguf_name(hf)
            .ok_or_else(|| NnError::MissingTensor(format!("no GGUF mapping for {hf}")))?;
        let info = self.info(&name)?;
        Ok((name, info))
    }

    /// Group-128 ternary rows as a `Q2Linear`, V heads regrouped, basis attached.
    fn ternary(&self, hf: &str, rows: usize, cols: usize) -> Result<Q2Linear, NnError> {
        let (name, info) = self.resolve(hf)?;
        if info.ggml_type != GGML_TYPE_PQ2_0 {
            return Err(NnError::MissingTensor(format!(
                "{name} has ggml type {}, expected PQ2_0 (142)",
                info.ggml_type
            )));
        }
        self.check_matrix(info, rows, cols)?;
        if !cols.is_multiple_of(PQ2_0_GROUP_SIZE) {
            return Err(NnError::Shape {
                expected: PQ2_0_GROUP_SIZE,
                got: cols,
            });
        }
        let src_row = cols / PQ2_0_GROUP_SIZE * PQ2_0_BLOCK_BYTES;
        let src = self.bytes(info, rows * src_row)?;
        let dst_row = cols / PQ2_0_GROUP_SIZE * 2 * Q2_0_BLOCK_BYTES;
        let mut packed = vec![0u8; rows * dst_row];
        split_pq2_0_into_q2_0(&src, &mut packed)
            .map_err(|error| NnError::Backend(format!("{name}: {error}")))?;
        let packed = self.regroup_rows(hf, packed, dst_row)?;
        let mut linear = Q2Linear::new(packed, rows, cols)?;
        self.ternary_count.set(self.ternary_count.get() + 1);
        if self.folded.contains(&name) || self.inverse.contains(&name) {
            let basis = self.bases.get(&cols).ok_or_else(|| {
                NnError::Backend(format!(
                    "{name} is folded but no sign vector has width {cols}"
                ))
            })?;
            linear = linear.with_basis(Arc::clone(basis))?;
            if self.folded.contains(&name) {
                self.folded_count.set(self.folded_count.get() + 1);
            }
        }
        Ok(linear)
    }

    /// Undo llama.cpp's V-head tiling on the rows of a DeltaNet input projection.
    fn regroup_rows<T: Copy>(
        &self,
        hf: &str,
        data: Vec<T>,
        row_len: usize,
    ) -> Result<Vec<T>, NnError> {
        let head_rows = self.value_head_dim;
        if hf.ends_with("linear_attn.in_proj_qkv.weight") {
            let split = self.key_width * 2 * row_len;
            let mut out = data[..split].to_vec();
            out.extend(self.tiling.regroup(&data[split..], head_rows * row_len));
            Ok(out)
        } else if hf.ends_with("linear_attn.in_proj_z.weight") {
            Ok(self.tiling.regroup(&data, head_rows * row_len))
        } else if hf.ends_with("linear_attn.in_proj_a.weight")
            || hf.ends_with("linear_attn.in_proj_b.weight")
        {
            Ok(self.tiling.regroup(&data, row_len))
        } else {
            Ok(data)
        }
    }
}

impl Qwen35HfTensorSource for Qwen35GgufSource {
    fn tensor_f32_exact(&self, hf: &str, expected: &[usize]) -> Result<Vec<f32>, NnError> {
        let (name, info) = self.resolve(hf)?;
        let values = self.f32_values(info)?;
        let want: usize = expected.iter().product();
        if values.len() != want {
            return Err(NnError::MissingTensor(format!(
                "{name} has {} values, {hf} expects {expected:?}",
                values.len()
            )));
        }
        let suffix = hf.rsplit_once("layers.").map_or(hf, |(_, rest)| rest);
        Ok(if hf.ends_with("linear_attn.A_log") {
            // ssm_a = −exp(A_log), heads tiled.
            let a_log = values
                .iter()
                .map(|a| {
                    if *a < 0.0 {
                        Ok((-*a).ln())
                    } else {
                        Err(NnError::Backend(format!(
                            "{name} has a non-negative entry {a}"
                        )))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.tiling.regroup(&a_log, 1)
        } else if hf.ends_with("linear_attn.dt_bias") {
            self.tiling.regroup(&values, 1)
        } else if hf.ends_with("linear_attn.conv1d.weight") {
            // [channels][kernel], channel-major like HF's [channels, 1, kernel]; only the V
            // channels were tiled.
            let kernel = *expected.last().unwrap_or(&1);
            let split = self.key_width * 2 * kernel;
            let mut out = values[..split].to_vec();
            out.extend(
                self.tiling
                    .regroup(&values[split..], self.value_head_dim * kernel),
            );
            out
        } else if suffix.ends_with("linear_attn.norm.weight") {
            values
        } else if hf.ends_with("norm.weight") {
            // The converter stored 1 + w; the zero-centered norm here adds the 1 itself.
            values.iter().map(|g| g - 1.0).collect()
        } else {
            values
        })
    }

    fn projection_exact(&self, hf: &str, rows: usize, cols: usize) -> Result<Projection, NnError> {
        let (name, info) = self.resolve(hf)?;
        if info.ggml_type == GGML_TYPE_PQ2_0 {
            return Ok(Projection::Q2(self.ternary(hf, rows, cols)?));
        }
        self.check_matrix(info, rows, cols)?;
        if self.folded.contains(&name) {
            return Err(NnError::Backend(format!(
                "{name} is declared Hadamard-folded but is not ternary; a dense folded path is not implemented"
            )));
        }
        let values = self.regroup_rows(hf, self.f32_values(info)?, cols)?;
        // The only dense projections in a ternary checkpoint are the DeltaNet alpha/beta heads.
        // The runner requires one activation arithmetic across a model, and the ternary weights
        // around them take A8 activations, so these do too. llama.cpp runs them in BF16; the
        // difference is visible in the parity test's log-probability gap, not hidden.
        Ok(Projection::Dense(DenseLinear::new(values, rows, cols)?))
    }

    fn token_embedding_exact(
        &self,
        hf: &str,
        rows: usize,
        cols: usize,
    ) -> Result<TokenEmbedding, NnError> {
        let (name, info) = self.resolve(hf)?;
        if info.ggml_type != GGML_TYPE_PQ2_0 {
            self.check_matrix(info, rows, cols)?;
            return TokenEmbedding::from_dense(self.f32_values(info)?, rows, cols);
        }
        let table = self.ternary(hf, rows, cols)?;
        self.folded_embedding.set(self.inverse.contains(&name));
        TokenEmbedding::from_q2(Arc::new(table))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The converter's `_reorder_v_heads`, transcribed, so regroup is tested against the thing it
    /// inverts rather than against itself.
    fn converter_tile(
        grouped: &[u32],
        key_heads: usize,
        values_per_key: usize,
        unit: usize,
    ) -> Vec<u32> {
        let mut tiled = Vec::with_capacity(grouped.len());
        for value in 0..values_per_key {
            for key in 0..key_heads {
                let head = key * values_per_key + value;
                tiled.extend_from_slice(&grouped[head * unit..(head + 1) * unit]);
            }
        }
        tiled
    }

    #[test]
    fn regroup_inverts_the_converters_tiling() {
        for (key_heads, values_per_key, unit) in [(16, 3, 1), (16, 3, 4), (2, 5, 3), (4, 1, 2)] {
            let grouped: Vec<u32> = (0..(key_heads * values_per_key * unit) as u32).collect();
            let tiled = converter_tile(&grouped, key_heads, values_per_key, unit);
            let tiling = HeadTiling {
                key_heads,
                values_per_key,
            };
            assert_eq!(
                tiling.regroup(&tiled, unit),
                grouped,
                "K={key_heads} r={values_per_key}"
            );
        }
    }

    #[test]
    fn every_language_tensor_name_maps() {
        for (hf, gguf) in [
            (
                "model.language_model.embed_tokens.weight",
                "token_embd.weight",
            ),
            ("lm_head.weight", "output.weight"),
            (
                "model.language_model.layers.7.linear_attn.A_log",
                "blk.7.ssm_a",
            ),
            (
                "model.language_model.layers.3.self_attn.q_proj.weight",
                "blk.3.attn_q.weight",
            ),
            (
                "model.language_model.layers.63.mlp.down_proj.weight",
                "blk.63.ffn_down.weight",
            ),
        ] {
            assert_eq!(Qwen35GgufSource::gguf_name(hf).as_deref(), Some(gguf));
        }
        assert_eq!(Qwen35GgufSource::gguf_name("model.visual.blocks.0.x"), None);
    }
}

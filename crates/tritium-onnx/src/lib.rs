//! # tritium-onnx — Tritium's ternary operators for ONNX Runtime.
//!
//! Two layers, so the always-on CI stays green without the onnxruntime native
//! library:
//!
//! - **Layer 1 (always on, no external deps).** [`ternary_mpgemm_kernel`] and
//!   [`ternary_embedding_kernel`] consume TQ2_0 / TQ1_0 weights directly. Their
//!   conformance tests pull neither `ort` nor `onnxruntime`.
//!
//! - **Layer 2 (feature `onnx`, pulls `ort`).** Packed projection/embedding
//!   operators retain `com.tritium` opset 1. Experimental cache attention uses
//!   [`TritiumKvAttentionOp`] and projected recurrent
//!   [`TritiumQwenDeltaNetOp`] cores at opset 2. Enabling the feature fetches
//!   ONNX Runtime.
//!
//! With feature `model`, [`encode_causal_lm`] serializes packed decoder layers
//! using opset-1 canonical TQ or opset-2 additive SALT V2 projection/embedding
//! nodes plus standard opset-21 RMSNorm (including optional Q/K norm),
//! full-head RoPE, GQA attention,
//! SwiGLU, residual, and K/V-cache glue.
//!
//! ## Feature gate
//!
//! ```text
//! cargo build -p tritium-onnx                  # lean: Layer 1 only, zero ort
//! cargo test  -p tritium-onnx                  # Layer 1 bit-exact conformance gate
//! cargo test  -p tritium-onnx --features model # deterministic protobuf export
//! cargo test  -p tritium-onnx --features onnx  # + Layer 2 ort custom-op registration
//! ```
#![deny(missing_docs)]

use half::f16;
use tritium_core::{GemmShape, TernaryFormat, Trit, reference_mpgemm};
use tritium_format::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks,
    salt_v2::{S34_TRITS_PER_GROUP, SaltV2Codec},
    salt_v2_package::{
        SALT_V2_ALLOCATION_TILE_SIZE, SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES,
        SALT_V2_MAX_PLANES, SALT_V2_SCALE_GROUP_SIZE, unpack_salt_v2_plane_into,
    },
    unpack_tq1_0_row, unpack_tq2_0_row,
};

/// Stable Tritium ONNX custom-operator domain.
pub const ONNX_DOMAIN: &str = "com.tritium";

/// Stable opset-1 packed ternary matrix multiplication node name.
pub const ONNX_OP_NAME: &str = "TritiumTernaryMpGemm";

/// Stable opset-1 packed ternary embedding node name.
pub const ONNX_EMBEDDING_OP_NAME: &str = "TritiumTernaryEmbedding";

/// Experimental opset-2 cache-aware grouped-query attention node name.
pub const ONNX_KV_ATTENTION_OP_NAME: &str = "TritiumKvAttention";

/// Experimental opset-2 projected Qwen Gated DeltaNet recurrent-core node.
pub const ONNX_QWEN_DELTANET_OP_NAME: &str = "TritiumQwenDeltaNet";

/// Stable additive SALT V2 packed matrix multiplication node name.
pub const ONNX_SALT_V2_OP_NAME: &str = "TritiumSaltV2MpGemm";

/// Experimental opset-2 additive SALT V2 selected-row embedding node name.
pub const ONNX_SALT_V2_EMBEDDING_OP_NAME: &str = "TritiumSaltV2Embedding";

/// Node-attribute name for the contraction/embedding dimension `K`.
pub const ATTR_K: &str = "K";

/// Node-attribute name for the packing format (`0` = TQ2_0, `1` = TQ1_0).
pub const ATTR_FORMAT: &str = "format";

/// Additive SALT V2 matrix output-row count attribute.
pub const ATTR_ROWS: &str = "rows";

/// Additive SALT V2 codec attribute (`0` = D2, `1` = B3, `2` = S34).
pub const ATTR_CODEC: &str = "codec";

/// Additive SALT V2 terminal partial allocation-map bits.
pub const ATTR_TERMINAL_MAP_VALUE: &str = "terminal_map_value";

/// Cache-aware attention query-head-count attribute.
pub const ATTR_N_HEAD: &str = "n_head";

/// Cache-aware attention key/value-head-count attribute.
pub const ATTR_N_KV_HEAD: &str = "n_kv_head";

/// Cache-aware attention head-width attribute.
pub const ATTR_HEAD_DIM: &str = "head_dim";

/// Cache-aware attention prefix-length attribute.
pub const ATTR_PAST_TOKENS: &str = "past_tokens";

/// DeltaNet depthwise convolution kernel width attribute.
pub const ATTR_CONV_KERNEL_DIM: &str = "conv_kernel_dim";

/// DeltaNet query/key head-count attribute.
pub const ATTR_NUM_KEY_HEADS: &str = "num_key_heads";

/// DeltaNet value/recurrent head-count attribute.
pub const ATTR_NUM_VALUE_HEADS: &str = "num_value_heads";

/// DeltaNet per-head query/key width attribute.
pub const ATTR_KEY_HEAD_DIM: &str = "key_head_dim";

/// DeltaNet per-head value width attribute.
pub const ATTR_VALUE_HEAD_DIM: &str = "value_head_dim";

/// Backend-neutral validated geometry for Qwen Gated DeltaNet recurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenDeltaNetGeometry {
    conv_kernel_dim: usize,
    num_key_heads: usize,
    num_value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
}

/// Derived Qwen Gated DeltaNet tensor widths and state sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenDeltaNetDimensions {
    key_width: usize,
    value_width: usize,
    conv_width: usize,
    conv_state_len: usize,
    recurrent_state_len: usize,
}

/// Invalid or overflowing Qwen Gated DeltaNet geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenDeltaNetGeometryError(&'static str);

impl core::fmt::Display for QwenDeltaNetGeometryError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for QwenDeltaNetGeometryError {}

impl QwenDeltaNetGeometry {
    /// Validate and bind one DeltaNet geometry.
    ///
    /// # Errors
    /// Returns [`QwenDeltaNetGeometryError`] for zero dimensions, invalid head
    /// grouping, or derived-width overflow.
    pub fn new(
        conv_kernel_dim: usize,
        num_key_heads: usize,
        num_value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Self, QwenDeltaNetGeometryError> {
        let geometry = Self {
            conv_kernel_dim,
            num_key_heads,
            num_value_heads,
            key_head_dim,
            value_head_dim,
        };
        geometry.dimensions()?;
        Ok(geometry)
    }

    /// Depthwise causal-convolution history width.
    #[must_use]
    pub const fn conv_kernel_dim(self) -> usize {
        self.conv_kernel_dim
    }

    /// Query/key head count.
    #[must_use]
    pub const fn num_key_heads(self) -> usize {
        self.num_key_heads
    }

    /// Value/recurrent head count.
    #[must_use]
    pub const fn num_value_heads(self) -> usize {
        self.num_value_heads
    }

    /// Query/key lanes per head.
    #[must_use]
    pub const fn key_head_dim(self) -> usize {
        self.key_head_dim
    }

    /// Value lanes per head.
    #[must_use]
    pub const fn value_head_dim(self) -> usize {
        self.value_head_dim
    }

    /// Derive every projected width and explicit-state element count.
    ///
    /// # Errors
    /// Returns [`QwenDeltaNetGeometryError`] for invalid grouping or overflow.
    pub fn dimensions(self) -> Result<QwenDeltaNetDimensions, QwenDeltaNetGeometryError> {
        if self.conv_kernel_dim == 0
            || self.num_key_heads == 0
            || self.num_value_heads == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
        {
            return Err(QwenDeltaNetGeometryError(
                "DeltaNet geometry dimensions must be nonzero",
            ));
        }
        if !self.num_value_heads.is_multiple_of(self.num_key_heads) {
            return Err(QwenDeltaNetGeometryError(
                "DeltaNet num_value_heads must be divisible by num_key_heads",
            ));
        }
        let key_width = self
            .num_key_heads
            .checked_mul(self.key_head_dim)
            .ok_or(QwenDeltaNetGeometryError("DeltaNet key width overflow"))?;
        let value_width = self
            .num_value_heads
            .checked_mul(self.value_head_dim)
            .ok_or(QwenDeltaNetGeometryError("DeltaNet value width overflow"))?;
        let conv_width = key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
            .ok_or(QwenDeltaNetGeometryError(
                "DeltaNet convolution width overflow",
            ))?;
        let conv_state_len =
            conv_width
                .checked_mul(self.conv_kernel_dim)
                .ok_or(QwenDeltaNetGeometryError(
                    "DeltaNet convolution state overflow",
                ))?;
        let recurrent_state_len = self
            .num_value_heads
            .checked_mul(self.key_head_dim)
            .and_then(|len| len.checked_mul(self.value_head_dim))
            .ok_or(QwenDeltaNetGeometryError(
                "DeltaNet recurrent state overflow",
            ))?;
        Ok(QwenDeltaNetDimensions {
            key_width,
            value_width,
            conv_width,
            conv_state_len,
            recurrent_state_len,
        })
    }
}

impl QwenDeltaNetDimensions {
    /// Globally split query width.
    #[must_use]
    pub const fn key_width(self) -> usize {
        self.key_width
    }

    /// Recurrent value/output width.
    #[must_use]
    pub const fn value_width(self) -> usize {
        self.value_width
    }

    /// Packed Q/K/V convolution width.
    #[must_use]
    pub const fn conv_width(self) -> usize {
        self.conv_width
    }

    /// Explicit convolution-state element count.
    #[must_use]
    pub const fn conv_state_len(self) -> usize {
        self.conv_state_len
    }

    /// Explicit recurrent-state element count.
    #[must_use]
    pub const fn recurrent_state_len(self) -> usize {
        self.recurrent_state_len
    }
}

#[cfg(any(feature = "model", feature = "onnx"))]
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub(crate) enum QwenDeltaNetInput {
    RawQkv,
    Z,
    BetaLogits,
    DecayLogits,
    ConvWeight,
    NormWeight,
    DtBias,
    ALog,
    ConvState,
    RecurrentState,
    Epsilon,
}

#[cfg(any(feature = "model", feature = "onnx"))]
impl QwenDeltaNetInput {
    pub(crate) const ALL: [Self; 11] = [
        Self::RawQkv,
        Self::Z,
        Self::BetaLogits,
        Self::DecayLogits,
        Self::ConvWeight,
        Self::NormWeight,
        Self::DtBias,
        Self::ALog,
        Self::ConvState,
        Self::RecurrentState,
        Self::Epsilon,
    ];

    #[cfg(feature = "onnx")]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::RawQkv => "raw_qkv",
            Self::Z => "z",
            Self::BetaLogits => "beta_logits",
            Self::DecayLogits => "decay_logits",
            Self::ConvWeight => "conv_weight",
            Self::NormWeight => "norm_weight",
            Self::DtBias => "dt_bias",
            Self::ALog => "a_log",
            Self::ConvState => "conv_state",
            Self::RecurrentState => "recurrent_state",
            Self::Epsilon => "epsilon",
        }
    }
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub(crate) enum QwenDeltaNetOutputSlot {
    NormalizedCore,
    ConvState,
    RecurrentState,
}

#[cfg(feature = "onnx")]
impl QwenDeltaNetOutputSlot {
    pub(crate) const ALL: [Self; 3] = [Self::NormalizedCore, Self::ConvState, Self::RecurrentState];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::NormalizedCore => "normalized_core",
            Self::ConvState => "next_conv",
            Self::RecurrentState => "next_recurrent",
        }
    }
}

/// Errors from the always-on ternary mpGEMM kernel.
///
/// A small, dependency-free enum (no `thiserror`) implementing
/// [`core::fmt::Display`] + [`std::error::Error`], so the kernel reports a
/// precise reason for every rejection (unsupported format, bad packed length,
/// shape disagreement) without dragging an error framework into the lean build.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OnnxTernaryError {
    /// The ternary `format` is not one this kernel packs/unpacks (only TQ2_0 and
    /// TQ1_0 are supported).
    UnsupportedFormat(TernaryFormat),
    /// The packed weight byte length does not match the `[N, K]` shape in this
    /// `format`. `n` is derived from `scales.len()`.
    PackedLenMismatch {
        /// Byte length the `[N, K]` shape + format requires.
        expected: usize,
        /// Byte length actually supplied.
        got: usize,
    },
    /// The activation length disagrees with `m * k`.
    ActivationLenMismatch {
        /// Length `m * k` requires.
        expected: usize,
        /// Activation length actually supplied.
        got: usize,
    },
    /// `tritium-format` rejected a packed weight row while unpacking.
    Unpack(String),
    /// [`tritium_core::reference_mpgemm`] rejected the call (a shape mismatch the
    /// upfront checks did not catch).
    Kernel(String),
    /// A dimension or derived byte/element count overflowed `usize`.
    ShapeOverflow(&'static str),
    /// A per-output/per-row scale is negative or non-finite.
    InvalidScale {
        /// Index of the invalid scale.
        index: usize,
    },
    /// A token tensor's shape does not contain exactly the supplied token count.
    TokenShapeMismatch {
        /// Element count implied by the token tensor shape.
        expected: usize,
        /// Number of supplied token IDs.
        got: usize,
    },
    /// A token ID is negative or outside the embedding vocabulary.
    TokenOutOfRange {
        /// Flat position in the input token tensor.
        position: usize,
        /// Rejected token ID.
        token: i64,
        /// Number of rows in the embedding table.
        vocab: usize,
    },
    /// Cache-aware attention geometry is zero, inconsistent, or not valid GQA.
    InvalidAttentionGeometry(String),
    /// One attention input has the wrong flat element count.
    AttentionInputLength {
        /// Stable input name (`q`, `k_cache`, or `v_cache`).
        input: &'static str,
        /// Required flat element count.
        expected: usize,
        /// Supplied flat element count.
        got: usize,
    },
    /// One attention input contains a NaN or infinity.
    InvalidAttentionValue {
        /// Stable input name.
        input: &'static str,
        /// Flat rejected element index.
        index: usize,
    },
    /// A SALT V2 indexed arena or its canonical metadata is malformed.
    InvalidSaltV2(String),
    /// An additive SALT V2 activation is NaN or infinite.
    InvalidActivationValue {
        /// Flat rejected activation index.
        index: usize,
    },
}

impl core::fmt::Display for OnnxTernaryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OnnxTernaryError::UnsupportedFormat(fmt) => {
                write!(f, "tritium-onnx: unsupported ternary format {fmt:?}")
            }
            OnnxTernaryError::PackedLenMismatch { expected, got } => {
                write!(
                    f,
                    "tritium-onnx: packed weights len {got} != expected {expected}"
                )
            }
            OnnxTernaryError::ActivationLenMismatch { expected, got } => {
                write!(
                    f,
                    "tritium-onnx: activation len {got} != expected {expected} (m*k)"
                )
            }
            OnnxTernaryError::Unpack(msg) => write!(f, "tritium-onnx: unpack: {msg}"),
            OnnxTernaryError::Kernel(msg) => write!(f, "tritium-onnx: mpgemm: {msg}"),
            OnnxTernaryError::ShapeOverflow(what) => {
                write!(f, "tritium-onnx: {what} overflows addressable memory")
            }
            OnnxTernaryError::InvalidScale { index } => write!(
                f,
                "tritium-onnx: scale {index} must be finite and nonnegative"
            ),
            OnnxTernaryError::TokenShapeMismatch { expected, got } => write!(
                f,
                "tritium-onnx: token shape contains {expected} elements, got {got} token IDs"
            ),
            OnnxTernaryError::TokenOutOfRange {
                position,
                token,
                vocab,
            } => write!(
                f,
                "tritium-onnx: token {token} at flat position {position} is outside vocabulary 0..{vocab}"
            ),
            OnnxTernaryError::InvalidAttentionGeometry(reason) => {
                write!(f, "tritium-onnx: invalid KV attention geometry: {reason}")
            }
            OnnxTernaryError::AttentionInputLength {
                input,
                expected,
                got,
            } => write!(
                f,
                "tritium-onnx: KV attention {input} has {got} elements, expected {expected}"
            ),
            OnnxTernaryError::InvalidAttentionValue { input, index } => write!(
                f,
                "tritium-onnx: KV attention {input}[{index}] must be finite"
            ),
            OnnxTernaryError::InvalidSaltV2(reason) => {
                write!(f, "tritium-onnx: invalid SALT V2 operand: {reason}")
            }
            OnnxTernaryError::InvalidActivationValue { index } => write!(
                f,
                "tritium-onnx: SALT V2 activation[{index}] must be finite"
            ),
        }
    }
}

impl std::error::Error for OnnxTernaryError {}

/// Packed bytes per block for a format this kernel supports.
fn block_bytes(format: TernaryFormat) -> Result<usize, OnnxTernaryError> {
    match format {
        TernaryFormat::Tq2_0 => Ok(TQ2_0_BLOCK_BYTES),
        TernaryFormat::Tq1_0 => Ok(TQ1_0_BLOCK_BYTES),
        other => Err(OnnxTernaryError::UnsupportedFormat(other)),
    }
}

fn packed_layout(
    rows: usize,
    k: usize,
    format: TernaryFormat,
) -> Result<(usize, usize), OnnxTernaryError> {
    let row_bytes = num_blocks(k)
        .checked_mul(block_bytes(format)?)
        .ok_or(OnnxTernaryError::ShapeOverflow("packed row byte count"))?;
    let packed_bytes = rows
        .checked_mul(row_bytes)
        .ok_or(OnnxTernaryError::ShapeOverflow("packed tensor byte count"))?;
    Ok((row_bytes, packed_bytes))
}

fn validate_scales(scales: &[f32]) -> Result<(), OnnxTernaryError> {
    if let Some((index, _)) = scales
        .iter()
        .enumerate()
        .find(|(_, scale)| !scale.is_finite() || **scale < 0.0)
    {
        return Err(OnnxTernaryError::InvalidScale { index });
    }
    Ok(())
}

fn unpack_row(
    packed: &[u8],
    k: usize,
    format: TernaryFormat,
    trits: &mut [Trit],
    scratch: &mut [f16],
) -> Result<(), OnnxTernaryError> {
    let result = match format {
        TernaryFormat::Tq2_0 => unpack_tq2_0_row(packed, trits, scratch),
        TernaryFormat::Tq1_0 => unpack_tq1_0_row(packed, trits, scratch),
        other => return Err(OnnxTernaryError::UnsupportedFormat(other)),
    };
    result.map_err(|error| OnnxTernaryError::Unpack(error.to_string()))?;
    if let Some((block, scale)) = scratch
        .iter()
        .copied()
        .enumerate()
        .find(|(_, scale)| *scale != f16::ONE)
    {
        return Err(OnnxTernaryError::Unpack(format!(
            "block {block} carries internal scale {scale:?}; Tritium ONNX requires unit block scales"
        )));
    }
    if trits.len() != k {
        return Err(OnnxTernaryError::Kernel(
            "internal unpacked-row length mismatch".to_owned(),
        ));
    }
    Ok(())
}

/// Unpack `[N, K]` `packed` ternary weights into a flat `Vec<Trit>`, validating
/// the byte length against `n`, `k`, and `format` (the same pattern as the
/// candle / wasm backends). The per-block scales are discarded (the packer fixes
/// them to 1.0; the per-channel scales are applied in the contraction).
fn unpack_weights(
    packed: &[u8],
    n: usize,
    k: usize,
    format: TernaryFormat,
) -> Result<Vec<Trit>, OnnxTernaryError> {
    let nb = num_blocks(k);
    let (row_bytes, expected) = packed_layout(n, k, format)?;
    if packed.len() != expected {
        return Err(OnnxTernaryError::PackedLenMismatch {
            expected,
            got: packed.len(),
        });
    }
    let trit_count = n.checked_mul(k).ok_or(OnnxTernaryError::ShapeOverflow(
        "unpacked weight element count",
    ))?;
    let mut trits = vec![Trit::ZERO; trit_count];
    let mut scratch = vec![f16::ONE; nb];
    for ni in 0..n {
        let row = &packed[ni * row_bytes..(ni + 1) * row_bytes];
        let trow = &mut trits[ni * k..ni * k + k];
        unpack_row(row, k, format, trow, &mut scratch).map_err(|error| match error {
            OnnxTernaryError::Unpack(message) => {
                OnnxTernaryError::Unpack(format!("row {ni}: {message}"))
            }
            other => other,
        })?;
    }
    Ok(trits)
}

/// Run Tritium's ternary mpGEMM: `out[m, n] = scale[n] * Σ_k act[m, k] * w[n, k]`.
///
/// This is the always-on, dependency-free kernel (Layer 1). It unpacks the
/// `[N, K]` `packed` ternary weights in `format` and runs
/// [`tritium_core::reference_mpgemm`], so its output is **bit-exact** with the
/// reference every Tritium backend is graded against — the conformance gate runs
/// it with no `ort`/`onnxruntime` dependency.
///
/// - `act`    — `[M, K]` row-major f32 activations (`act.len() == m * k`).
/// - `packed` — `[N, K]` ternary weights packed in `format`, output-major. `N`
///   is taken from `scales.len()`.
/// - `scales` — `[N]` per-output-channel scales.
/// - `m`, `k` — the activation `M` and `K`.
///
/// # Errors
/// [`OnnxTernaryError`] if `act.len() != m * k`, the packed length is wrong for
/// `[N = scales.len(), K = k]` in `format`, the format is unsupported, or the
/// kernel itself rejects the shapes.
pub fn ternary_mpgemm_kernel(
    act: &[f32],
    packed: &[u8],
    scales: &[f32],
    m: usize,
    k: usize,
    format: TernaryFormat,
) -> Result<Vec<f32>, OnnxTernaryError> {
    let expected_activation = m
        .checked_mul(k)
        .ok_or(OnnxTernaryError::ShapeOverflow("activation element count"))?;
    if act.len() != expected_activation {
        return Err(OnnxTernaryError::ActivationLenMismatch {
            expected: expected_activation,
            got: act.len(),
        });
    }
    validate_scales(scales)?;
    let n = scales.len();
    let trits = unpack_weights(packed, n, k, format)?;
    let output_len = m.checked_mul(n).ok_or(OnnxTernaryError::ShapeOverflow(
        "mpGEMM output element count",
    ))?;
    let mut out = vec![0f32; output_len];
    reference_mpgemm(act, &trits, scales, GemmShape { m, n, k }, &mut out)
        .map_err(|e| OnnxTernaryError::Kernel(e.to_string()))?;
    Ok(out)
}

/// One descriptor-free indexed SALT V2 matrix operand.
///
/// The arenas are the production runtime layout: codec payloads, f16 group-128
/// scales, a two-bit plane-count map, and one u32 rank prefix per 256 tiles.
/// The terminal partial map byte is carried by `terminal_map_value` and is not
/// duplicated in `allocation_map`.
#[derive(Debug, Clone, Copy)]
pub struct SaltV2PackedMatrix<'a> {
    /// Output rows.
    pub rows: usize,
    /// Input/contraction columns.
    pub columns: usize,
    /// Physical plane codec.
    pub codec: SaltV2Codec,
    /// Concatenated canonical plane payloads.
    pub payload: &'a [u8],
    /// Concatenated nonnegative finite f16 group scales.
    pub scales: &'a [f16],
    /// Complete bytes of the two-bit-per-tile plane-count map.
    pub allocation_map: &'a [u8],
    /// Cumulative plane ranks at 256-tile boundaries.
    pub rank_prefixes: &'a [u32],
    /// Remaining zero-padded plane-count map bits.
    pub terminal_map_value: u32,
}

impl<'a> SaltV2PackedMatrix<'a> {
    fn logical_coefficients(self) -> Result<usize, OnnxTernaryError> {
        if self.rows == 0 || self.columns == 0 {
            return Err(OnnxTernaryError::InvalidSaltV2(
                "rows and columns must be positive".into(),
            ));
        }
        self.rows
            .checked_mul(self.columns)
            .ok_or(OnnxTernaryError::ShapeOverflow("SALT V2 coefficient count"))
    }

    fn tile_count(self) -> Result<usize, OnnxTernaryError> {
        Ok(self
            .logical_coefficients()?
            .div_ceil(SALT_V2_ALLOCATION_TILE_SIZE))
    }

    fn tile_plane_count(self, tile: usize) -> Result<usize, OnnxTernaryError> {
        let tile_count = self.tile_count()?;
        if tile >= tile_count {
            return Err(OnnxTernaryError::InvalidSaltV2(
                "tile index exceeds matrix geometry".into(),
            ));
        }
        let bit = tile.checked_mul(2).ok_or(OnnxTernaryError::ShapeOverflow(
            "SALT V2 allocation-map bit offset",
        ))?;
        let stored_bits = self.allocation_map.len() * u8::BITS as usize;
        let code = if bit < stored_bits {
            (self.allocation_map[bit / 8] >> (bit % 8)) & 0b11
        } else {
            ((self.terminal_map_value >> (bit - stored_bits)) & 0b11) as u8
        };
        let count = usize::from(code) + 1;
        if count > SALT_V2_MAX_PLANES {
            return Err(OnnxTernaryError::InvalidSaltV2(format!(
                "tile {tile} encodes {count} planes"
            )));
        }
        Ok(count)
    }

    fn plane_rank_before(self, tile: usize) -> Result<usize, OnnxTernaryError> {
        let block = tile / SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES;
        let block_start = block * SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES;
        let mut rank =
            if block == 0 {
                0
            } else {
                usize::try_from(*self.rank_prefixes.get(block - 1).ok_or_else(|| {
                    OnnxTernaryError::InvalidSaltV2("rank prefix is absent".into())
                })?)
                .map_err(|_| OnnxTernaryError::InvalidSaltV2("rank prefix exceeds usize".into()))?
            };
        for current in block_start..tile {
            rank = rank
                .checked_add(self.tile_plane_count(current)?)
                .ok_or(OnnxTernaryError::ShapeOverflow("SALT V2 plane rank"))?;
        }
        Ok(rank)
    }

    fn payload_bytes(self, logical_len: usize) -> Result<usize, OnnxTernaryError> {
        let stored = if self.codec == SaltV2Codec::S34 {
            logical_len.div_ceil(S34_TRITS_PER_GROUP) * S34_TRITS_PER_GROUP
        } else {
            logical_len
        };
        self.codec
            .ledger(stored)
            .map(|ledger| ledger.physical_bytes)
            .map_err(|error| OnnxTernaryError::InvalidSaltV2(error.to_string()))
    }

    fn plane(
        self,
        tile: usize,
        plane: usize,
        logical_len: usize,
    ) -> Result<(&'a [u8], &'a [f16]), OnnxTernaryError> {
        let rank = self.plane_rank_before(tile)?;
        let payload_stride = self.payload_bytes(logical_len)?;
        let full_payload_stride = self.payload_bytes(SALT_V2_ALLOCATION_TILE_SIZE)?;
        let scale_stride = logical_len.div_ceil(SALT_V2_SCALE_GROUP_SIZE);
        let full_scale_stride = SALT_V2_ALLOCATION_TILE_SIZE.div_ceil(SALT_V2_SCALE_GROUP_SIZE);
        let payload_start = rank
            .checked_mul(full_payload_stride)
            .and_then(|offset| offset.checked_add(plane.checked_mul(payload_stride)?))
            .ok_or(OnnxTernaryError::ShapeOverflow("SALT V2 payload offset"))?;
        let scale_start = rank
            .checked_mul(full_scale_stride)
            .and_then(|offset| offset.checked_add(plane.checked_mul(scale_stride)?))
            .ok_or(OnnxTernaryError::ShapeOverflow("SALT V2 scale offset"))?;
        let payload = self
            .payload
            .get(payload_start..payload_start + payload_stride)
            .ok_or_else(|| OnnxTernaryError::InvalidSaltV2("plane payload is absent".into()))?;
        let scales = self
            .scales
            .get(scale_start..scale_start + scale_stride)
            .ok_or_else(|| OnnxTernaryError::InvalidSaltV2("plane scales are absent".into()))?;
        Ok((payload, scales))
    }

    fn validate_structure(self) -> Result<(), OnnxTernaryError> {
        let logical = self.logical_coefficients()?;
        let tiles = self.tile_count()?;
        let expected_map = tiles.checked_mul(2).ok_or(OnnxTernaryError::ShapeOverflow(
            "SALT V2 allocation-map bit count",
        ))? / 8;
        let expected_ranks = tiles.saturating_sub(1) / SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES;
        if self.allocation_map.len() != expected_map || self.rank_prefixes.len() != expected_ranks {
            return Err(OnnxTernaryError::InvalidSaltV2(format!(
                "metadata lengths differ from geometry: map {}/{expected_map}, ranks {}/{expected_ranks}",
                self.allocation_map.len(),
                self.rank_prefixes.len()
            )));
        }
        let terminal_bits = (tiles * 2) % 8;
        if terminal_bits == 0 {
            if self.terminal_map_value != 0 {
                return Err(OnnxTernaryError::InvalidSaltV2(
                    "terminal allocation-map padding is nonzero".into(),
                ));
            }
        } else if self.terminal_map_value >> terminal_bits != 0 {
            return Err(OnnxTernaryError::InvalidSaltV2(
                "terminal allocation-map padding is nonzero".into(),
            ));
        }
        let mut payload_bytes = 0usize;
        let mut scale_count = 0usize;
        let mut planes_before = 0usize;
        for tile in 0..tiles {
            if tile > 0 && tile % SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES == 0 {
                let declared =
                    usize::try_from(self.rank_prefixes[tile / 256 - 1]).map_err(|_| {
                        OnnxTernaryError::InvalidSaltV2("rank prefix exceeds usize".into())
                    })?;
                if declared != planes_before {
                    return Err(OnnxTernaryError::InvalidSaltV2(format!(
                        "rank prefix before tile {tile} is {declared}, expected {planes_before}"
                    )));
                }
            }
            let start = tile * SALT_V2_ALLOCATION_TILE_SIZE;
            let len = (logical - start).min(SALT_V2_ALLOCATION_TILE_SIZE);
            let planes = self.tile_plane_count(tile)?;
            payload_bytes = payload_bytes
                .checked_add(self.payload_bytes(len)?.checked_mul(planes).ok_or(
                    OnnxTernaryError::ShapeOverflow("SALT V2 tile payload bytes"),
                )?)
                .ok_or(OnnxTernaryError::ShapeOverflow("SALT V2 payload bytes"))?;
            scale_count = scale_count
                .checked_add(
                    len.div_ceil(SALT_V2_SCALE_GROUP_SIZE)
                        .checked_mul(planes)
                        .ok_or(OnnxTernaryError::ShapeOverflow("SALT V2 tile scales"))?,
                )
                .ok_or(OnnxTernaryError::ShapeOverflow("SALT V2 scale count"))?;
            planes_before += planes;
        }
        if self.payload.len() != payload_bytes || self.scales.len() != scale_count {
            return Err(OnnxTernaryError::InvalidSaltV2(format!(
                "arena lengths differ from geometry: payload {}/{payload_bytes}, scales {}/{scale_count}",
                self.payload.len(),
                self.scales.len()
            )));
        }
        if let Some((index, _)) = self
            .scales
            .iter()
            .enumerate()
            .find(|(_, scale)| !scale.is_finite() || scale.to_bits() & 0x8000 != 0)
        {
            return Err(OnnxTernaryError::InvalidSaltV2(format!(
                "scale {index} must be finite with a clear sign bit"
            )));
        }
        Ok(())
    }

    /// Validate complete indexed geometry, canonical metadata, arena lengths,
    /// every f16 scale, every codec payload, and zero-scale group semantics.
    ///
    /// # Errors
    /// Returns [`OnnxTernaryError`] for any malformed or noncanonical operand.
    pub fn validate(self) -> Result<(), OnnxTernaryError> {
        self.validate_structure()?;
        let logical = self.logical_coefficients()?;
        let mut decoded = Vec::new();
        decoded
            .try_reserve_exact(SALT_V2_ALLOCATION_TILE_SIZE)
            .map_err(|_| {
                OnnxTernaryError::InvalidSaltV2("decode scratch allocation failed".into())
            })?;
        for tile in 0..self.tile_count()? {
            let start = tile * SALT_V2_ALLOCATION_TILE_SIZE;
            let logical_len = (logical - start).min(SALT_V2_ALLOCATION_TILE_SIZE);
            for plane in 0..self.tile_plane_count(tile)? {
                let (packed, scales) = self.plane(tile, plane, logical_len)?;
                unpack_salt_v2_plane_into(self.codec, packed, logical_len, &mut decoded)
                    .map_err(|error| OnnxTernaryError::InvalidSaltV2(error.to_string()))?;
                for (group, scale) in scales.iter().enumerate() {
                    if scale.to_bits() != 0 {
                        continue;
                    }
                    let group_start = group * SALT_V2_SCALE_GROUP_SIZE;
                    let group_end = (group_start + SALT_V2_SCALE_GROUP_SIZE).min(logical_len);
                    if decoded[group_start..group_end]
                        .iter()
                        .any(|&trit| trit != Trit::ZERO)
                    {
                        return Err(OnnxTernaryError::InvalidSaltV2(format!(
                            "tile {tile} plane {plane} group {group} has zero scale with nonzero trits"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Execute additive SALT V2 matrix multiplication without a dense weight shadow.
///
/// Reduction order is row, group-128, plane, then column, matching Tritium's
/// host and CUDA SALT V2 execution contract.
///
/// # Errors
/// Returns [`OnnxTernaryError`] for malformed geometry or arenas, noncanonical
/// codec data, non-finite operands/arithmetic, or a derived-size overflow.
pub fn salt_v2_mpgemm_kernel(
    activation: &[f32],
    m: usize,
    matrix: SaltV2PackedMatrix<'_>,
) -> Result<Vec<f32>, OnnxTernaryError> {
    matrix.validate()?;
    let expected = m
        .checked_mul(matrix.columns)
        .ok_or(OnnxTernaryError::ShapeOverflow(
            "SALT V2 activation element count",
        ))?;
    if activation.len() != expected {
        return Err(OnnxTernaryError::ActivationLenMismatch {
            expected,
            got: activation.len(),
        });
    }
    if let Some((index, _)) = activation
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(OnnxTernaryError::InvalidActivationValue { index });
    }
    let output_len = m
        .checked_mul(matrix.rows)
        .ok_or(OnnxTernaryError::ShapeOverflow(
            "SALT V2 output element count",
        ))?;
    let mut output = vec![0.0f32; output_len];
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(SALT_V2_ALLOCATION_TILE_SIZE)
        .map_err(|_| OnnxTernaryError::InvalidSaltV2("decode scratch allocation failed".into()))?;
    for batch in 0..m {
        let input = &activation[batch * matrix.columns..(batch + 1) * matrix.columns];
        for row in 0..matrix.rows {
            let row_start = row * matrix.columns;
            let row_end = row_start + matrix.columns;
            let mut coefficient = row_start;
            let mut accumulator = 0.0f32;
            while coefficient < row_end {
                let tile = coefficient / SALT_V2_ALLOCATION_TILE_SIZE;
                let tile_start = tile * SALT_V2_ALLOCATION_TILE_SIZE;
                let local_start = coefficient - tile_start;
                let logical_len =
                    (matrix.logical_coefficients()? - tile_start).min(SALT_V2_ALLOCATION_TILE_SIZE);
                let group = local_start / SALT_V2_SCALE_GROUP_SIZE;
                let group_end = ((group + 1) * SALT_V2_SCALE_GROUP_SIZE).min(logical_len);
                let segment_len = (group_end - local_start).min(row_end - coefficient);
                let segment_end = coefficient + segment_len;
                let planes = matrix.tile_plane_count(tile)?;
                for plane in 0..planes {
                    let (packed, scales) = matrix.plane(tile, plane, logical_len)?;
                    unpack_salt_v2_plane_into(matrix.codec, packed, logical_len, &mut decoded)
                        .map_err(|error| OnnxTernaryError::InvalidSaltV2(error.to_string()))?;
                    let scale = scales[group].to_f32();
                    let mut group_sum = 0.0f32;
                    for current in coefficient..segment_end {
                        let column = current - row_start;
                        group_sum += decoded[current - tile_start].to_f32() * input[column];
                    }
                    accumulator += group_sum * scale;
                    if !accumulator.is_finite() {
                        return Err(OnnxTernaryError::InvalidSaltV2(
                            "contraction produced a non-finite value".into(),
                        ));
                    }
                }
                coefficient = segment_end;
            }
            output[batch * matrix.rows + row] = accumulator;
        }
    }
    Ok(output)
}

/// Gather selected rows from an additive SALT V2 embedding table.
///
/// Only requested rows are reconstructed into output storage. Packed arenas
/// remain canonical and no vocabulary-sized dense shadow is materialized.
/// Returned flat storage has logical shape `token_shape + [matrix.columns]`.
///
/// # Errors
/// Returns [`OnnxTernaryError`] for malformed tensor shape, out-of-range token
/// IDs, invalid SALT V2 arenas or codec bytes, or derived-size overflow.
pub fn salt_v2_embedding_kernel(
    tokens: &[i64],
    token_shape: &[usize],
    matrix: SaltV2PackedMatrix<'_>,
) -> Result<Vec<f32>, OnnxTernaryError> {
    matrix.validate()?;
    salt_v2_embedding_kernel_admitted(tokens, token_shape, matrix)
}

pub(crate) fn salt_v2_embedding_kernel_admitted(
    tokens: &[i64],
    token_shape: &[usize],
    matrix: SaltV2PackedMatrix<'_>,
) -> Result<Vec<f32>, OnnxTernaryError> {
    matrix.validate_structure()?;
    let token_count = token_shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or(OnnxTernaryError::ShapeOverflow("token element count"))
    })?;
    if tokens.len() != token_count {
        return Err(OnnxTernaryError::TokenShapeMismatch {
            expected: token_count,
            got: tokens.len(),
        });
    }
    let output_len =
        token_count
            .checked_mul(matrix.columns)
            .ok_or(OnnxTernaryError::ShapeOverflow(
                "SALT V2 embedding output element count",
            ))?;
    let mut output = vec![0.0f32; output_len];
    let logical = matrix.logical_coefficients()?;
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(SALT_V2_ALLOCATION_TILE_SIZE)
        .map_err(|_| OnnxTernaryError::InvalidSaltV2("decode scratch allocation failed".into()))?;
    for (position, &token) in tokens.iter().enumerate() {
        let row = usize::try_from(token)
            .ok()
            .filter(|&row| row < matrix.rows)
            .ok_or(OnnxTernaryError::TokenOutOfRange {
                position,
                token,
                vocab: matrix.rows,
            })?;
        let row_start = row
            .checked_mul(matrix.columns)
            .ok_or(OnnxTernaryError::ShapeOverflow(
                "SALT V2 embedding row offset",
            ))?;
        let row_end = row_start + matrix.columns;
        let destination = &mut output[position * matrix.columns..(position + 1) * matrix.columns];
        let mut coefficient = row_start;
        while coefficient < row_end {
            let tile = coefficient / SALT_V2_ALLOCATION_TILE_SIZE;
            let tile_start = tile * SALT_V2_ALLOCATION_TILE_SIZE;
            let local_start = coefficient - tile_start;
            let logical_len = (logical - tile_start).min(SALT_V2_ALLOCATION_TILE_SIZE);
            let segment_len = (logical_len - local_start).min(row_end - coefficient);
            for plane in 0..matrix.tile_plane_count(tile)? {
                let (packed, scales) = matrix.plane(tile, plane, logical_len)?;
                unpack_salt_v2_plane_into(matrix.codec, packed, logical_len, &mut decoded)
                    .map_err(|error| OnnxTernaryError::InvalidSaltV2(error.to_string()))?;
                for local in local_start..local_start + segment_len {
                    destination[coefficient - row_start + local - local_start] +=
                        decoded[local].to_f32() * scales[local / SALT_V2_SCALE_GROUP_SIZE].to_f32();
                }
            }
            coefficient += segment_len;
        }
    }
    Ok(output)
}

/// Gather scaled rows from a packed ternary embedding table without
/// materializing a dense vocabulary-sized shadow.
///
/// `packed` stores `[vocab, K]` output-major TQ2_0/TQ1_0 rows and `scales`
/// stores one finite nonnegative scale per vocabulary row. `token_shape` may be
/// scalar or any-rank; the returned flat buffer has logical shape
/// `token_shape + [K]`.
///
/// # Errors
/// [`OnnxTernaryError`] if the token shape/count, a token ID, a scale, packed
/// byte length, or a derived size is invalid.
pub fn ternary_embedding_kernel(
    tokens: &[i64],
    token_shape: &[usize],
    packed: &[u8],
    scales: &[f32],
    k: usize,
    format: TernaryFormat,
) -> Result<Vec<f32>, OnnxTernaryError> {
    let token_count = token_shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or(OnnxTernaryError::ShapeOverflow("token element count"))
    })?;
    if tokens.len() != token_count {
        return Err(OnnxTernaryError::TokenShapeMismatch {
            expected: token_count,
            got: tokens.len(),
        });
    }
    validate_scales(scales)?;
    let vocab = scales.len();
    let (row_bytes, expected_packed) = packed_layout(vocab, k, format)?;
    if packed.len() != expected_packed {
        return Err(OnnxTernaryError::PackedLenMismatch {
            expected: expected_packed,
            got: packed.len(),
        });
    }
    let output_len = token_count
        .checked_mul(k)
        .ok_or(OnnxTernaryError::ShapeOverflow(
            "embedding output element count",
        ))?;
    let mut output = vec![0.0f32; output_len];
    let mut trits = vec![Trit::ZERO; k];
    let mut scratch = vec![f16::ONE; num_blocks(k)];
    for (position, &token) in tokens.iter().enumerate() {
        let row = usize::try_from(token)
            .ok()
            .filter(|&row| row < vocab)
            .ok_or(OnnxTernaryError::TokenOutOfRange {
                position,
                token,
                vocab,
            })?;
        let start = row * row_bytes;
        unpack_row(
            &packed[start..start + row_bytes],
            k,
            format,
            &mut trits,
            &mut scratch,
        )
        .map_err(|error| match error {
            OnnxTernaryError::Unpack(message) => {
                OnnxTernaryError::Unpack(format!("row {row}: {message}"))
            }
            other => other,
        })?;
        let scale = scales[row];
        let out = &mut output[position * k..(position + 1) * k];
        for (value, &trit) in out.iter_mut().zip(&trits) {
            *value = scale * trit.to_f32();
        }
    }
    Ok(output)
}

/// Cache-aware causal grouped-query attention semantic oracle.
///
/// `q` has shape `[query_tokens, n_head, head_dim]`; `k_cache` and `v_cache`
/// have shape `[past_tokens + query_tokens, n_kv_head, head_dim]`. Query row
/// `i` may attend through cache row `past_tokens + i`. Returned flat buffer has
/// same shape as `q`.
///
/// # Errors
/// [`OnnxTernaryError`] if geometry is empty/invalid, dimensions overflow, any
/// flat input length disagrees with declared geometry, or an input is non-finite.
#[allow(clippy::too_many_arguments)]
pub fn kv_attention_kernel(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    query_tokens: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    past_tokens: usize,
) -> Result<Vec<f32>, OnnxTernaryError> {
    if query_tokens == 0 || n_head == 0 || n_kv_head == 0 || head_dim == 0 {
        return Err(OnnxTernaryError::InvalidAttentionGeometry(
            "query_tokens, n_head, n_kv_head, and head_dim must be positive".to_owned(),
        ));
    }
    if !n_head.is_multiple_of(n_kv_head) {
        return Err(OnnxTernaryError::InvalidAttentionGeometry(format!(
            "n_head {n_head} is not divisible by n_kv_head {n_kv_head}"
        )));
    }
    let total_tokens = past_tokens
        .checked_add(query_tokens)
        .ok_or(OnnxTernaryError::ShapeOverflow("KV cache token count"))?;
    let q_len = query_tokens
        .checked_mul(n_head)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or(OnnxTernaryError::ShapeOverflow(
            "KV attention query elements",
        ))?;
    let kv_len = total_tokens
        .checked_mul(n_kv_head)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or(OnnxTernaryError::ShapeOverflow(
            "KV attention cache elements",
        ))?;
    for (input, got, expected) in [
        ("q", q.len(), q_len),
        ("k_cache", k_cache.len(), kv_len),
        ("v_cache", v_cache.len(), kv_len),
    ] {
        if got != expected {
            return Err(OnnxTernaryError::AttentionInputLength {
                input,
                expected,
                got,
            });
        }
    }
    for (input, values) in [("q", q), ("k_cache", k_cache), ("v_cache", v_cache)] {
        if let Some((index, _)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(OnnxTernaryError::InvalidAttentionValue { input, index });
        }
    }

    let group_size = n_head / n_kv_head;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut output = vec![0.0; q_len];
    let mut probabilities = vec![0.0_f64; total_tokens];
    for query in 0..query_tokens {
        let visible = past_tokens + query + 1;
        for head in 0..n_head {
            let kv_head = head / group_size;
            let mut maximum = f64::NEG_INFINITY;
            for key in 0..visible {
                let mut score = 0.0_f64;
                for lane in 0..head_dim {
                    score += f64::from(q[(query * n_head + head) * head_dim + lane])
                        * f64::from(k_cache[(key * n_kv_head + kv_head) * head_dim + lane]);
                }
                probabilities[key] = score * scale;
                maximum = maximum.max(probabilities[key]);
            }
            let mut sum = 0.0;
            for probability in &mut probabilities[..visible] {
                *probability = (*probability - maximum).exp();
                sum += *probability;
            }
            for lane in 0..head_dim {
                let mut value = 0.0_f64;
                for key in 0..visible {
                    value += probabilities[key] / sum
                        * f64::from(v_cache[(key * n_kv_head + kv_head) * head_dim + lane]);
                }
                output[(query * n_head + head) * head_dim + lane] = value as f32;
            }
        }
    }
    Ok(output)
}

#[cfg(feature = "onnx")]
pub use onnx_op::{
    QwenDeltaNetInputs, QwenDeltaNetOutput, QwenDeltaNetTensor, TritiumKvAttentionKernel,
    TritiumKvAttentionOp, TritiumQwenDeltaNetKernel, TritiumQwenDeltaNetOp,
    TritiumSaltV2EmbeddingKernel, TritiumSaltV2EmbeddingOp, TritiumSaltV2MpGemmKernel,
    TritiumSaltV2MpGemmOp, TritiumTernaryEmbeddingKernel, TritiumTernaryEmbeddingOp,
    TritiumTernaryMpGemmKernel, TritiumTernaryMpGemmOp, tritium_operator_domain,
};

#[cfg(feature = "onnx")]
mod onnx_op;

#[cfg(feature = "model")]
pub use model::{
    AdmittedExternalCausalLmDigests, AdmittedExternalQwen35BundleDigests, BitNetConfig,
    BitNetGgufTensorProvider, CausalActivation, CausalLmDecoderLayer, CausalLmModel,
    CausalQueryProjection, ExternalOnnxModel, ExternalQwen35Bundle, ExternalQwen35BundleFiles,
    MappedBitNet, MappedQwen35, MappedSmolLm2, OnnxArtifactIdentityV2, OnnxModelError,
    OnnxPackedMatrix, PackedMatrixStorage, PackedTernaryMatrix, Qwen35Config, Qwen35LayerType,
    Qwen35MtpDecoder, Qwen35MtpModel, Qwen35OnnxAncestryV1, Qwen35PackedTensorProvider,
    Qwen35TensorProvider, QwenCausalLmDecoderLayer, QwenCausalLmModel, QwenDeltaNetDecoderLayer,
    QwenDeltaNetLayerModel, QwenFullAttentionDecoderLayer, RotaryEmbedding, RotaryMode,
    SeekExternalQwen35Bundle, SmolLm2Config, SmolLm2TensorProvider, TiedEmbeddingHeadModel,
    TiedEmbeddingHeadModelV2, UnsupportedGraphDiagnostic, UnsupportedGraphItemKind,
    VerifiedExternalCausalLmModel, VerifiedExternalOnnxModel, VerifiedExternalOnnxModelV2,
    VerifiedExternalQwen35Bundle, VerifiedOnnxArtifactIdentityV2, VerifiedQwen35OnnxAncestryV1,
    diagnose_unsupported_graph, encode_causal_lm,
    encode_dynamic_external_qwen35_bundle_to_file_with_ancestry,
    encode_dynamic_external_qwen35_bundle_with_ancestry, encode_dynamic_qwen_causal_lm,
    encode_dynamic_qwen35_mtp, encode_external_causal_lm, encode_external_qwen_causal_lm,
    encode_external_qwen35_bundle, encode_external_qwen35_bundle_to_file_with_ancestry,
    encode_external_qwen35_bundle_with_ancestry, encode_external_qwen35_mtp,
    encode_external_tied_embedding_head, encode_external_tied_embedding_head_v2,
    encode_qwen_causal_lm, encode_qwen_deltanet_layer, encode_qwen35_mtp,
    encode_tied_embedding_head, encode_tied_embedding_head_v2, map_bitnet_gguf_causal_lm,
    map_qwen35_causal_lm, map_qwen35_packed_causal_lm, map_qwen36_27b_causal_lm,
    map_qwen36_27b_packed_causal_lm, map_smollm2_causal_lm, verify_external_causal_lm,
    verify_external_qwen35_bundle, verify_external_qwen35_bundle_from_file,
    verify_external_tied_embedding_head, verify_external_tied_embedding_head_v2,
};

#[cfg(feature = "model")]
mod model;

#[cfg(feature = "qwen-package")]
mod qwen_package;

#[cfg(feature = "qwen-package")]
pub use qwen_package::{
    Qwen35PackageMatrixSpec, Qwen35PackagePreservedSpec, Qwen35PackageSourceSpec,
    Qwen35SaltV2PackageSource,
};

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_format::{
        pack_tq1_0_row, pack_tq2_0_row,
        salt_v2::SaltV2Codec,
        salt_v2_package::{
            SALT_V2_ALLOCATION_TILE_SIZE, SALT_V2_SCALE_GROUP_SIZE, pack_salt_v2_plane,
        },
    };
    use tritium_testkit::{ConformanceVector, FROZEN_COUNT, FROZEN_SEED, generate_vectors};

    /// Pack a vector's `[N, K]` `i8` weights exactly as the conformance harness
    /// does: per row, unit (1.0) block scales, concatenated output-major.
    fn pack(v: &ConformanceVector, format: TernaryFormat) -> Vec<u8> {
        let nb = num_blocks(v.k);
        let unit = vec![f16::ONE; nb];
        let row_bytes = nb * block_bytes(format).unwrap();
        let mut packed = vec![0u8; v.n * row_bytes];
        for ni in 0..v.n {
            let trits: Vec<Trit> = v.weights[ni * v.k..ni * v.k + v.k]
                .iter()
                .map(|&w| Trit::from_i8(w).unwrap())
                .collect();
            let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
            match format {
                TernaryFormat::Tq2_0 => pack_tq2_0_row(&trits, &unit, out).unwrap(),
                TernaryFormat::Tq1_0 => pack_tq1_0_row(&trits, &unit, out).unwrap(),
                other => panic!("cannot pack {other:?}"),
            };
        }
        packed
    }

    fn pack_rows(rows: &[Vec<Trit>], format: TernaryFormat) -> Vec<u8> {
        let k = rows.first().map_or(0, Vec::len);
        assert!(rows.iter().all(|row| row.len() == k));
        let nb = num_blocks(k);
        let unit = vec![f16::ONE; nb];
        let row_bytes = nb * block_bytes(format).unwrap();
        let mut packed = vec![0u8; rows.len() * row_bytes];
        for (row, output) in rows.iter().zip(packed.chunks_exact_mut(row_bytes)) {
            match format {
                TernaryFormat::Tq2_0 => pack_tq2_0_row(row, &unit, output).unwrap(),
                TernaryFormat::Tq1_0 => pack_tq1_0_row(row, &unit, output).unwrap(),
                other => panic!("cannot pack {other:?}"),
            }
        }
        packed
    }

    fn vectors() -> Vec<ConformanceVector> {
        generate_vectors(FROZEN_SEED, FROZEN_COUNT)
    }

    struct SaltFixture {
        payload: Vec<u8>,
        scales: Vec<f16>,
        allocation_map: Vec<u8>,
        rank_prefixes: Vec<u32>,
        terminal_map_value: u32,
        dense: Vec<f32>,
    }

    fn salt_fixture(codec: SaltV2Codec, rows: usize, columns: usize) -> SaltFixture {
        let logical = rows * columns;
        let mut payload = Vec::new();
        let mut scales = Vec::new();
        let mut dense = vec![0.0; logical];
        let tile_count = logical.div_ceil(SALT_V2_ALLOCATION_TILE_SIZE);
        let mut allocation_map = vec![0u8; tile_count * 2 / 8];
        let mut rank_prefixes = Vec::new();
        let mut terminal_map_value = 0u32;
        let mut planes_before = 0usize;
        for (tile_index, start) in (0..logical)
            .step_by(SALT_V2_ALLOCATION_TILE_SIZE)
            .enumerate()
        {
            let len = (logical - start).min(SALT_V2_ALLOCATION_TILE_SIZE);
            let plane_count = 1 + tile_index % 3;
            if tile_index > 0 && tile_index % SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES == 0 {
                rank_prefixes.push(u32::try_from(planes_before).unwrap());
            }
            let bit = tile_index * 2;
            let stored_bits = allocation_map.len() * 8;
            if bit < stored_bits {
                allocation_map[bit / 8] |= ((plane_count - 1) as u8) << (bit % 8);
            } else {
                terminal_map_value |= ((plane_count - 1) as u32) << (bit - stored_bits);
            }
            for plane_index in 0..plane_count {
                let trits: Vec<Trit> = (0..len)
                    .map(|index| {
                        if codec == SaltV2Codec::S34 {
                            match index % 4 {
                                0 => Trit::ZERO,
                                1 | 2 => Trit::POS,
                                _ => Trit::NEG,
                            }
                        } else {
                            match (index + plane_index + tile_index) % 3 {
                                0 => Trit::NEG,
                                1 => Trit::ZERO,
                                _ => Trit::POS,
                            }
                        }
                    })
                    .collect();
                payload.extend(pack_salt_v2_plane(codec, &trits).unwrap());
                let group_count = len.div_ceil(SALT_V2_SCALE_GROUP_SIZE);
                for group in 0..group_count {
                    let scale = f16::from_f32(0.125 * (1 + plane_index + group) as f32);
                    scales.push(scale);
                    let begin = group * SALT_V2_SCALE_GROUP_SIZE;
                    let end = (begin + SALT_V2_SCALE_GROUP_SIZE).min(len);
                    for local in begin..end {
                        dense[start + local] += trits[local].to_f32() * scale.to_f32();
                    }
                }
            }
            planes_before += plane_count;
        }
        SaltFixture {
            payload,
            scales,
            allocation_map,
            rank_prefixes,
            terminal_map_value,
            dense,
        }
    }

    /// The always-on kernel reproduces the frozen conformance set. Because it
    /// runs the SAME `reference_mpgemm` over pack->unpack round-tripped trits, it
    /// is bit-exact; grading it exactly (rather than within a tolerance) proves
    /// the pack/unpack/readback plumbing has no off-by-one or layout regression.
    #[test]
    fn kernel_matches_reference_on_frozen_set() {
        // generate_vectors appends the fixed boundary set on top of the
        // FROZEN_COUNT random vectors, so the total exceeds FROZEN_COUNT.
        let vs = vectors();
        let total = vs.len();
        assert!(total > FROZEN_COUNT, "boundary vectors must be included");
        let mut checked = 0usize;
        for v in vs {
            let format = v.format;
            let packed = pack(&v, format);
            let got =
                ternary_mpgemm_kernel(&v.activation, &packed, &v.scales, v.m, v.k, format).unwrap();
            assert_eq!(
                got, v.expected,
                "vector {}: kernel must be bit-exact with the reference",
                v.id
            );
            checked += 1;
        }
        assert_eq!(checked, total, "every frozen vector exercised");
    }

    #[test]
    fn salt_v2_kernel_matches_independent_additive_oracle_for_every_codec() {
        let (m, rows, columns) = (2, 2, 300);
        let activation: Vec<f32> = (0..m * columns)
            .map(|index| (index as f32 % 17.0 - 8.0) * 0.0625)
            .collect();
        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            let fixture = salt_fixture(codec, rows, columns);
            let matrix = SaltV2PackedMatrix {
                rows,
                columns,
                codec,
                payload: &fixture.payload,
                scales: &fixture.scales,
                allocation_map: &fixture.allocation_map,
                rank_prefixes: &fixture.rank_prefixes,
                terminal_map_value: fixture.terminal_map_value,
            };
            let got = salt_v2_mpgemm_kernel(&activation, m, matrix).unwrap();
            let expected: Vec<f32> = (0..m)
                .flat_map(|batch| {
                    let fixture = &fixture;
                    let activation = &activation;
                    (0..rows).map(move |row| {
                        (0..columns).fold(0.0, |sum, column| {
                            sum + activation[batch * columns + column]
                                * fixture.dense[row * columns + column]
                        })
                    })
                })
                .collect();
            for (actual, reference) in got.iter().zip(expected) {
                assert!((actual - reference).abs() < 2.0e-5, "{codec:?}");
            }
        }
    }

    #[test]
    fn salt_v2_kernel_rejects_noncanonical_or_incomplete_arenas() {
        let fixture = salt_fixture(SaltV2Codec::B3, 2, 300);
        let valid = SaltV2PackedMatrix {
            rows: 2,
            columns: 300,
            codec: SaltV2Codec::B3,
            payload: &fixture.payload,
            scales: &fixture.scales,
            allocation_map: &fixture.allocation_map,
            rank_prefixes: &fixture.rank_prefixes,
            terminal_map_value: fixture.terminal_map_value,
        };
        assert!(salt_v2_mpgemm_kernel(&[0.0; 599], 2, valid).is_err());
        let truncated = SaltV2PackedMatrix {
            payload: &fixture.payload[..fixture.payload.len() - 1],
            ..valid
        };
        assert!(salt_v2_mpgemm_kernel(&[0.0; 600], 2, truncated).is_err());
        let trailing = SaltV2PackedMatrix {
            terminal_map_value: fixture.terminal_map_value | (1 << 6),
            ..valid
        };
        assert!(salt_v2_mpgemm_kernel(&[0.0; 600], 2, trailing).is_err());
        let mut bad_scales = fixture.scales.clone();
        bad_scales[0] = f16::NAN;
        let nonfinite = SaltV2PackedMatrix {
            scales: &bad_scales,
            ..valid
        };
        assert!(salt_v2_mpgemm_kernel(&[0.0; 600], 2, nonfinite).is_err());
        let mut negative_zero_scales = fixture.scales.clone();
        negative_zero_scales[0] = f16::from_bits(0x8000);
        let negative_zero = SaltV2PackedMatrix {
            scales: &negative_zero_scales,
            ..valid
        };
        assert!(negative_zero.validate().is_err());
        let mut zero_scales = fixture.scales.clone();
        zero_scales[0] = f16::ZERO;
        let zero_nonzero_group = SaltV2PackedMatrix {
            scales: &zero_scales,
            ..valid
        };
        assert!(zero_nonzero_group.validate().is_err());

        let mut malformed_payload = fixture.payload.clone();
        let malformed_payload = (0..malformed_payload.len())
            .rev()
            .find_map(|index| {
                let original = malformed_payload[index];
                (0..u8::BITS).find_map(|bit| {
                    malformed_payload[index] = original ^ (1 << bit);
                    let candidate = SaltV2PackedMatrix {
                        payload: &malformed_payload,
                        ..valid
                    };
                    candidate
                        .validate()
                        .is_err()
                        .then(|| malformed_payload.clone())
                })
            })
            .expect("B3 fixture must expose canonical padding");
        let malformed = SaltV2PackedMatrix {
            payload: &malformed_payload,
            ..valid
        };
        assert!(salt_v2_embedding_kernel(&[0], &[1], malformed).is_err());
    }

    #[test]
    fn salt_v2_kernel_crosses_compact_map_and_rank_prefix_boundary() {
        let columns = 257 * SALT_V2_ALLOCATION_TILE_SIZE;
        let fixture = salt_fixture(SaltV2Codec::D2, 1, columns);
        assert_eq!(fixture.allocation_map.len(), 64);
        assert_eq!(fixture.rank_prefixes.len(), 1);
        let matrix = SaltV2PackedMatrix {
            rows: 1,
            columns,
            codec: SaltV2Codec::D2,
            payload: &fixture.payload,
            scales: &fixture.scales,
            allocation_map: &fixture.allocation_map,
            rank_prefixes: &fixture.rank_prefixes,
            terminal_map_value: fixture.terminal_map_value,
        };
        let activation = vec![1.0; columns];
        let output = salt_v2_mpgemm_kernel(&activation, 1, matrix).unwrap();
        assert!((output[0] - fixture.dense.iter().sum::<f32>()).abs() < 2.0e-4);

        let mut bad_rank = fixture.rank_prefixes.clone();
        bad_rank[0] += 1;
        let corrupted = SaltV2PackedMatrix {
            rank_prefixes: &bad_rank,
            ..matrix
        };
        assert!(salt_v2_mpgemm_kernel(&activation, 1, corrupted).is_err());
    }

    #[test]
    fn salt_v2_embedding_gathers_selected_additive_rows_for_every_codec() {
        let (rows, columns) = (3, 300);
        let tokens = [2, 0, 2, 1];
        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            let fixture = salt_fixture(codec, rows, columns);
            let matrix = SaltV2PackedMatrix {
                rows,
                columns,
                codec,
                payload: &fixture.payload,
                scales: &fixture.scales,
                allocation_map: &fixture.allocation_map,
                rank_prefixes: &fixture.rank_prefixes,
                terminal_map_value: fixture.terminal_map_value,
            };
            let actual = salt_v2_embedding_kernel(&tokens, &[2, 2], matrix).unwrap();
            let expected: Vec<f32> = tokens
                .iter()
                .flat_map(|&row| {
                    fixture.dense[row as usize * columns..(row as usize + 1) * columns]
                        .iter()
                        .copied()
                })
                .collect();
            assert_eq!(actual, expected, "{codec:?}");
        }
    }

    #[test]
    fn salt_v2_embedding_rejects_shape_and_token_errors() {
        let fixture = salt_fixture(SaltV2Codec::B3, 2, 8);
        let matrix = SaltV2PackedMatrix {
            rows: 2,
            columns: 8,
            codec: SaltV2Codec::B3,
            payload: &fixture.payload,
            scales: &fixture.scales,
            allocation_map: &fixture.allocation_map,
            rank_prefixes: &fixture.rank_prefixes,
            terminal_map_value: fixture.terminal_map_value,
        };
        assert!(salt_v2_embedding_kernel(&[0], &[2], matrix).is_err());
        assert!(salt_v2_embedding_kernel(&[-1], &[1], matrix).is_err());
        assert!(salt_v2_embedding_kernel(&[2], &[1], matrix).is_err());
    }

    #[test]
    fn rejects_k_mismatch() {
        // act is [M=2, K=4] but we tell the kernel K=8 -> act.len() 8 != 2*8.
        let act = vec![1.0f32; 2 * 4];
        let r = ternary_mpgemm_kernel(&act, &[0u8; 64], &[1.0, 1.0], 2, 8, TernaryFormat::Tq2_0);
        assert!(
            matches!(r, Err(OnnxTernaryError::ActivationLenMismatch { .. })),
            "K mismatch must error, got {r:?}"
        );
    }

    #[test]
    fn rejects_packed_len_mismatch() {
        // act is a valid [M=2, K=32]; the packed buffer is empty for a real
        // [N=2, K=32] shape.
        let act = vec![1.0f32; 2 * 32];
        let r = ternary_mpgemm_kernel(&act, &[], &[1.0, 1.0], 2, 32, TernaryFormat::Tq2_0);
        assert!(
            matches!(r, Err(OnnxTernaryError::PackedLenMismatch { .. })),
            "packed length mismatch must error, got {r:?}"
        );
    }

    #[test]
    fn rejects_unsupported_format() {
        let act = vec![1.0f32; 256];
        // I2sInt8 is a GPU-only packing this kernel does not consume.
        let r = ternary_mpgemm_kernel(&act, &[], &[1.0], 1, 256, TernaryFormat::I2sInt8);
        assert!(
            matches!(r, Err(OnnxTernaryError::UnsupportedFormat(_))),
            "unsupported format must error, got {r:?}"
        );
    }

    #[test]
    fn embedding_gathers_only_selected_scaled_rows() {
        let k = 260;
        let rows = vec![
            (0..k)
                .map(|column| match column % 3 {
                    0 => Trit::NEG,
                    1 => Trit::ZERO,
                    _ => Trit::POS,
                })
                .collect::<Vec<_>>(),
            vec![Trit::POS; k],
            vec![Trit::NEG; k],
        ];
        let scales = [0.5, 2.0, 1.25];
        let tokens = [2, 0, 2, 1];
        for format in [TernaryFormat::Tq2_0, TernaryFormat::Tq1_0] {
            let packed = pack_rows(&rows, format);
            let got =
                ternary_embedding_kernel(&tokens, &[2, 2], &packed, &scales, k, format).unwrap();
            let expected: Vec<f32> = tokens
                .iter()
                .flat_map(|&token| {
                    rows[token as usize]
                        .iter()
                        .map(move |trit| scales[token as usize] * trit.to_f32())
                })
                .collect();
            assert_eq!(got, expected, "selected-row gather must match for {format}");
        }
    }

    #[test]
    fn kv_attention_prompt_and_cached_decode_match_worked_oracle() {
        let prompt =
            kv_attention_kernel(&[1.0, 2.0], &[1.0, 1.0], &[10.0, 20.0], 2, 1, 1, 1, 0).unwrap();
        assert_eq!(prompt, vec![10.0, 15.0]);

        let decode =
            kv_attention_kernel(&[3.0], &[1.0, 1.0, 1.0], &[10.0, 20.0, 40.0], 1, 1, 1, 1, 2)
                .unwrap();
        assert!((decode[0] - 70.0 / 3.0).abs() < 1e-5);

        let overflow_resistant =
            kv_attention_kernel(&[f32::MAX], &[f32::MAX], &[7.0], 1, 1, 1, 1, 0).unwrap();
        assert_eq!(overflow_resistant, vec![7.0]);
        assert!(kv_attention_kernel(&[f32::NAN], &[1.0], &[1.0], 1, 1, 1, 1, 0).is_err());
    }

    #[test]
    fn embedding_rejects_shape_token_and_scale_errors() {
        let k = 256;
        let rows = vec![vec![Trit::ZERO; k]];
        let packed = pack_rows(&rows, TernaryFormat::Tq2_0);
        let shape = ternary_embedding_kernel(&[0], &[2], &packed, &[1.0], k, TernaryFormat::Tq2_0);
        assert!(matches!(
            shape,
            Err(OnnxTernaryError::TokenShapeMismatch { .. })
        ));
        let token = ternary_embedding_kernel(&[-1], &[1], &packed, &[1.0], k, TernaryFormat::Tq2_0);
        assert!(matches!(
            token,
            Err(OnnxTernaryError::TokenOutOfRange { .. })
        ));
        let scale =
            ternary_embedding_kernel(&[0], &[1], &packed, &[f32::NAN], k, TernaryFormat::Tq2_0);
        assert!(matches!(scale, Err(OnnxTernaryError::InvalidScale { .. })));
    }

    #[test]
    fn kernels_reject_non_unit_internal_block_scales() {
        let k = 256;
        let rows = vec![vec![Trit::POS; k]];
        let mut packed = pack_rows(&rows, TernaryFormat::Tq2_0);
        let scale_offset = TQ2_0_BLOCK_BYTES - core::mem::size_of::<f16>();
        packed[scale_offset..].copy_from_slice(&f16::ZERO.to_le_bytes());
        let result = ternary_embedding_kernel(&[0], &[1], &packed, &[1.0], k, TernaryFormat::Tq2_0);
        assert!(matches!(result, Err(OnnxTernaryError::Unpack(_))));
    }

    #[test]
    fn error_messages_are_distinct_and_nonempty() {
        let e1 = OnnxTernaryError::UnsupportedFormat(TernaryFormat::I2sInt8);
        let e2 = OnnxTernaryError::PackedLenMismatch {
            expected: 66,
            got: 0,
        };
        let e3 = OnnxTernaryError::ActivationLenMismatch {
            expected: 8,
            got: 4,
        };
        let s1 = e1.to_string();
        let s2 = e2.to_string();
        let s3 = e3.to_string();
        assert!(!s1.is_empty() && !s2.is_empty() && !s3.is_empty());
        assert_ne!(s1, s2);
        assert_ne!(s2, s3);
    }
}

//! Precision lattice and canonical ternary packing schemes.

/// Element precisions Tritium reasons about across backends.
///
/// Ternary weights pair with higher-precision *activations* (BitNet runs
/// W1.58**A8**), and modern GPUs add fp8/fp4 paths (Hopper/Blackwell), so the
/// dtype set spans the full mixed-precision space, not just ternary.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum DType {
    /// `{-1, 0, +1}` weights. On-disk/in-VRAM width depends on [`TernaryFormat`].
    Ternary,
    /// Signed 4-bit (GPTQ-Marlin / ExLlamaV2 weight path).
    I4,
    /// Signed 8-bit (common activation path for ternary matmul).
    I8,
    /// Unsigned 8-bit.
    U8,
    /// fp8 E4M3 (Hopper+ activations).
    F8E4M3,
    /// fp8 E5M2.
    F8E5M2,
    /// fp4 E2M1 element (MXFP4).
    F4E2M1,
    /// IEEE half.
    F16,
    /// bfloat16.
    BF16,
    /// IEEE single.
    F32,
}

impl DType {
    /// Nominal storage width in bits. Ternary returns its information-theoretic
    /// floor (`log2 3`); real packed width is format-dependent — see
    /// [`TernaryFormat::bits_per_weight`].
    pub const fn ideal_bits(self) -> f32 {
        match self {
            DType::Ternary => crate::TERNARY_IDEAL_BITS,
            DType::I4 | DType::F4E2M1 => 4.0,
            DType::I8 | DType::U8 | DType::F8E4M3 | DType::F8E5M2 => 8.0,
            DType::F16 | DType::BF16 => 16.0,
            DType::F32 => 32.0,
        }
    }

    /// True for the ternary weight type.
    pub const fn is_ternary(self) -> bool {
        matches!(self, DType::Ternary)
    }

    /// True for floating-point element types.
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            DType::F8E4M3 | DType::F8E5M2 | DType::F4E2M1 | DType::F16 | DType::BF16 | DType::F32
        )
    }
}

/// Canonical packing schemes for ternary weights — the shared vocabulary across
/// crates. The byte-exact layout, pack, and unpack live in `tritium-format`;
/// this enum only fixes the *names*, *block size*, and *effective width* so every
/// crate agrees on them.
///
/// Both follow ggml's 256-element super-block with one fp16 block scale.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum TernaryFormat {
    /// llama.cpp **TQ1_0** — 5 trits per byte (`3^5 = 243 < 256`). 1.6875 bpw incl.
    /// block scale. Most compact; unpack needs base-3 division — CPU/edge oriented.
    Tq1_0,
    /// llama.cpp **TQ2_0** — 2 bits per trit, 4 per byte. 2.0625 bpw incl. block
    /// scale. Cheap shift/mask unpack; matches BitNet's int8 packing — GPU oriented.
    Tq2_0,
    /// **I2sInt8** — the IMMA (int8 tensor-core) GPU layout, derived from an I2_S
    /// checkpoint at load (`tritium_format::convert_i2s_to_int8`). Weights stay
    /// **2-bit packed in VRAM** (≈2.0 bpw, no per-block scale — the I2_S source
    /// carries a single per-tensor `f32` scale), interleaved so the IMMA kernel can
    /// unpack them to int8 operands for `mma.m16n8k32` in shared memory. Added in
    /// v0.30 (ADR 0005) for the compute-bound prefill path; the CPU/reference
    /// backends do not consume it and return
    /// [`UnsupportedFormat`](crate::TernaryFormat) — it is a GPU-only packing.
    I2sInt8,
}

impl TernaryFormat {
    /// Effective bits per weight, block scale amortized over the block.
    pub const fn bits_per_weight(self) -> f32 {
        match self {
            TernaryFormat::Tq1_0 => 1.6875,
            TernaryFormat::Tq2_0 => 2.0625,
            // 2-bit packed quants, one f32 scale per *tensor* (not per block), so the
            // scale's per-weight cost is negligible — effectively 2.0 bpw in VRAM.
            TernaryFormat::I2sInt8 => 2.0,
        }
    }

    /// Weights per quantization block (ggml `QK_K`). One fp16 scale per block.
    pub const fn block_size(self) -> usize {
        256
    }

    /// Whether unpack is a cheap shift/mask (`true`, GPU-friendly) or base-3
    /// division (`false`, CPU/edge). Both `Tq2_0` and `I2sInt8` are 2-bit aligned.
    pub const fn is_bit_aligned(self) -> bool {
        matches!(self, TernaryFormat::Tq2_0 | TernaryFormat::I2sInt8)
    }
}

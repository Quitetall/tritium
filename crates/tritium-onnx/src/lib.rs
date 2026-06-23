//! # tritium-onnx — Tritium's ternary mpGEMM for ONNX Runtime.
//!
//! Two layers, so the always-on CI stays green without the onnxruntime native
//! library:
//!
//! - **Layer 1 (always on, no external deps).** [`ternary_mpgemm_kernel`] is a
//!   plain function: it unpacks `[N, K]` packed ternary weights (TQ2_0 / TQ1_0,
//!   like `tritium-candle` / `tritium-wasm`) and runs
//!   [`tritium_core::reference_mpgemm`] to produce the `[M, N]` f32 output. Its
//!   conformance test is **bit-exact** with the frozen vector set and pulls
//!   neither `ort` nor `onnxruntime` — it is the default-feature gate.
//!
//! - **Layer 2 (feature `onnx`, pulls `ort`).** [`TritiumTernaryMpGemmOp`]
//!   implements the `ort` 2.x custom-operator traits, registering an ONNX node
//!   `"TritiumTernaryMpGemm"` whose kernel calls Layer 1. Enabling the feature
//!   adds `ort` with the `download-binaries` feature so a build with network
//!   fetches a prebuilt onnxruntime — no system library required.
//!
//! ## Feature gate
//!
//! ```text
//! cargo build -p tritium-onnx                  # lean: Layer 1 only, zero ort
//! cargo test  -p tritium-onnx                  # Layer 1 bit-exact conformance gate
//! cargo test  -p tritium-onnx --features onnx  # + Layer 2 ort custom-op registration
//! ```
#![deny(missing_docs)]

use half::f16;
use tritium_core::{GemmShape, TernaryFormat, Trit, reference_mpgemm};
use tritium_format::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
};

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
    let row_bytes = nb * block_bytes(format)?;
    let expected = n * row_bytes;
    if packed.len() != expected {
        return Err(OnnxTernaryError::PackedLenMismatch {
            expected,
            got: packed.len(),
        });
    }
    let mut trits = vec![Trit::ZERO; n * k];
    let mut scratch = vec![f16::ONE; nb];
    for ni in 0..n {
        let row = &packed[ni * row_bytes..(ni + 1) * row_bytes];
        let trow = &mut trits[ni * k..ni * k + k];
        let res = match format {
            TernaryFormat::Tq2_0 => unpack_tq2_0_row(row, trow, &mut scratch),
            TernaryFormat::Tq1_0 => unpack_tq1_0_row(row, trow, &mut scratch),
            other => return Err(OnnxTernaryError::UnsupportedFormat(other)),
        };
        res.map_err(|e| OnnxTernaryError::Unpack(format!("row {ni}: {e}")))?;
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
    if act.len() != m * k {
        return Err(OnnxTernaryError::ActivationLenMismatch {
            expected: m * k,
            got: act.len(),
        });
    }
    let n = scales.len();
    let trits = unpack_weights(packed, n, k, format)?;
    let mut out = vec![0f32; m * n];
    reference_mpgemm(act, &trits, scales, GemmShape { m, n, k }, &mut out)
        .map_err(|e| OnnxTernaryError::Kernel(e.to_string()))?;
    Ok(out)
}

#[cfg(feature = "onnx")]
pub use onnx_op::{
    ATTR_FORMAT, ATTR_K, ONNX_DOMAIN, ONNX_OP_NAME, TritiumTernaryMpGemmKernel,
    TritiumTernaryMpGemmOp, tritium_operator_domain,
};

#[cfg(feature = "onnx")]
mod onnx_op;

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_format::{pack_tq1_0_row, pack_tq2_0_row};
    use tritium_testkit::{ConformanceVector, FROZEN_COUNT, FROZEN_SEED, generate_vectors};

    fn parse_format(tag: &str) -> TernaryFormat {
        match tag {
            "tq2_0" => TernaryFormat::Tq2_0,
            "tq1_0" => TernaryFormat::Tq1_0,
            other => panic!("unexpected format tag {other}"),
        }
    }

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

    fn vectors() -> Vec<ConformanceVector> {
        generate_vectors(FROZEN_SEED, FROZEN_COUNT)
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
            let format = parse_format(&v.format);
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

//! # tritium-burn — Tritium's ternary mpGEMM as a burn op.
//!
//! Exposes [`ternary_mpgemm`], a backend-generic op that runs Tritium's ternary
//! (BitNet b1.58) matrix multiply on a burn [`Tensor`]: an `[M, K]` f32
//! activation tensor times `[N, K]` packed ternary weights (TQ2_0 / TQ1_0) with
//! `[N]` per-output-channel scales, producing `[M, N]` f32. The kernel is
//! [`tritium_core::reference_mpgemm`] itself, so a burn BitNet layer is
//! **bit-exact** with the reference every Tritium backend is graded against.
//!
//! ## Host round-trip — works on any [`Backend`], in f32
//!
//! burn has no stable custom-kernel ABI that is portable across its backends, so
//! the op is a host round-trip rather than a backend-native kernel: read the
//! `[M, K]` f32 data out of the burn tensor, unpack the ternary weights exactly
//! like the candle / wasm backends, run `reference_mpgemm` into a `Vec<f32>`, and
//! build the `[M, N]` result (pinned to `DType::F32`) with [`Tensor::from_data`]
//! on `act`'s device. This works on any burn backend — CPU NdArray, wgpu, cuda —
//! at the cost of a device→host→device copy, and is **bit-exact** with the
//! reference. It mirrors candle's `CustomOp1::cpu_fwd`, which is also a host
//! computation.
//!
//! The op computes in **f32**: the activation tensor must have `DType::F32` (the
//! default float dtype for these backends); a half-precision (`f16`/`bf16`)
//! activation returns [`BurnTernaryError::TensorRead`] rather than running at a
//! lower precision. The `[M, N]` result is always `f32`.
//!
//! ## Feature gate
//!
//! The op (and the heavy `burn-tensor` / `burn-ndarray` dependencies) live behind
//! the **`burn`** feature, off by default — so the default workspace build and
//! `cargo test --workspace` stay burn-free. Enable it to build the op:
//!
//! ```text
//! cargo test -p tritium-burn --features burn
//! ```
#![deny(missing_docs)]

#[cfg(feature = "burn")]
pub use burn_op::{BurnTernaryError, ternary_mpgemm};

#[cfg(feature = "burn")]
mod burn_op {
    use core::fmt;

    use burn_tensor::{DType, Tensor, TensorData, backend::Backend};
    use half::f16;
    use tritium_core::{GemmShape, TernaryFormat, Trit, reference_mpgemm};
    use tritium_format::{
        TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
    };

    /// Error returned by [`ternary_mpgemm`]. burn has no `Msg`-style framework
    /// error to thread through, so the op carries its own small enum (mirrors
    /// candle's `Error::Msg` paths). Implements `std::error::Error`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[non_exhaustive]
    pub enum BurnTernaryError {
        /// The activation tensor's `K` (its second dim) disagreed with the `k`
        /// passed for the weights.
        KMismatch {
            /// `K` read from the activation tensor's `[M, K]` shape.
            activation_k: usize,
            /// `k` declared for the `[N, K]` weights.
            weight_k: usize,
        },
        /// The packed weight buffer was the wrong length for `[N, K]` in the
        /// given format (`N = scales.len()`).
        PackedLenMismatch {
            /// Bytes expected: `N * num_blocks(K) * block_bytes(format)`.
            expected: usize,
            /// Bytes actually supplied.
            got: usize,
        },
        /// The format is not one this op packs/unpacks (only TQ2_0 / TQ1_0).
        UnsupportedFormat(TernaryFormat),
        /// Reading the f32 elements back out of the burn tensor failed.
        TensorRead(String),
        /// `tritium-format` rejected a packed weight row while unpacking.
        Unpack(String),
        /// The reference kernel rejected the assembled buffers (shape mismatch).
        Kernel(String),
    }

    impl fmt::Display for BurnTernaryError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                BurnTernaryError::KMismatch {
                    activation_k,
                    weight_k,
                } => write!(
                    f,
                    "tritium-burn: activation K={activation_k} != weight K={weight_k}"
                ),
                BurnTernaryError::PackedLenMismatch { expected, got } => write!(
                    f,
                    "tritium-burn: packed weights len {got} != expected {expected}"
                ),
                BurnTernaryError::UnsupportedFormat(fmtm) => {
                    write!(f, "tritium-burn: unsupported ternary format {fmtm:?}")
                }
                BurnTernaryError::TensorRead(m) => write!(f, "tritium-burn: tensor read: {m}"),
                BurnTernaryError::Unpack(m) => write!(f, "tritium-burn: unpack: {m}"),
                BurnTernaryError::Kernel(m) => write!(f, "tritium-burn: mpgemm: {m}"),
            }
        }
    }

    impl std::error::Error for BurnTernaryError {}

    /// Packed bytes per block for a supported ternary format.
    fn block_bytes(format: TernaryFormat) -> Result<usize, BurnTernaryError> {
        match format {
            TernaryFormat::Tq2_0 => Ok(TQ2_0_BLOCK_BYTES),
            TernaryFormat::Tq1_0 => Ok(TQ1_0_BLOCK_BYTES),
            other => Err(BurnTernaryError::UnsupportedFormat(other)),
        }
    }

    /// Unpack the `[N, K]` `packed` weights into a flat `Vec<Trit>`, validating
    /// the byte length against `n`, `k`, and `format`.
    fn unpack(
        packed: &[u8],
        n: usize,
        k: usize,
        format: TernaryFormat,
    ) -> Result<Vec<Trit>, BurnTernaryError> {
        let nb = num_blocks(k);
        let row_bytes = nb * block_bytes(format)?;
        let expected = n * row_bytes;
        if packed.len() != expected {
            return Err(BurnTernaryError::PackedLenMismatch {
                expected,
                got: packed.len(),
            });
        }
        let mut trits = vec![Trit::ZERO; n * k];
        // Per-block scale scratch: the packer fixed these to 1.0; the per-channel
        // scales are applied in the contraction.
        let mut scratch = vec![f16::ONE; nb];
        for ni in 0..n {
            let row = &packed[ni * row_bytes..(ni + 1) * row_bytes];
            let trow = &mut trits[ni * k..ni * k + k];
            let res = match format {
                TernaryFormat::Tq2_0 => unpack_tq2_0_row(row, trow, &mut scratch),
                TernaryFormat::Tq1_0 => unpack_tq1_0_row(row, trow, &mut scratch),
                other => return Err(BurnTernaryError::UnsupportedFormat(other)),
            };
            res.map_err(|e| BurnTernaryError::Unpack(format!("row {ni}: {e}")))?;
        }
        Ok(trits)
    }

    /// Run Tritium's ternary mpGEMM on `act` (an `[M, K]` **f32** [`Tensor`] on any
    /// burn [`Backend`]) against `[N, K]` `packed` ternary weights in `format`
    /// with `[N]` `scales`, returning the `[M, N]` f32 result on the same device
    /// as `act`. `N` (the output-channel count) is taken from `scales.len()`.
    ///
    /// The implementation is a host round-trip (read `act` to host, unpack
    /// weights, run [`reference_mpgemm`], rebuild the result tensor pinned to
    /// `DType::F32`), so it is correct on any burn backend without a
    /// backend-native kernel — see the crate docs.
    ///
    /// # Errors
    /// [`BurnTernaryError`] if reading `act` fails (including a deferred kernel
    /// error surfaced by a lazy backend, or a non-`f32` activation dtype), `act`'s
    /// `K` disagrees with `k`, the packed length is wrong for `[N = scales.len(),
    /// K]` in `format`, the format is unsupported, or the kernel rejects the
    /// assembled buffers. Does not panic on a native backend (a read failure is
    /// returned as [`BurnTernaryError::TensorRead`]).
    pub fn ternary_mpgemm<B: Backend>(
        act: Tensor<B, 2>,
        packed: &[u8],
        scales: &[f32],
        k: usize,
        format: TernaryFormat,
    ) -> Result<Tensor<B, 2>, BurnTernaryError> {
        // N (output channels) is exactly the number of per-channel scales — derive
        // it rather than taking a second loose `usize` the caller could swap with k.
        let n = scales.len();
        let [m, act_k] = act.dims();
        if act_k != k {
            return Err(BurnTernaryError::KMismatch {
                activation_k: act_k,
                weight_k: k,
            });
        }
        // Validate the packed length BEFORE the (potentially expensive) tensor
        // read so a length bug is reported cheaply.
        let trits = unpack(packed, n, k, format)?;

        let device = act.device();
        // Fallible read: on a lazy backend (wgpu/cuda) a deferred kernel error
        // surfaces here; `try_into_data` returns it instead of `into_data`'s
        // panic, so the documented "read fails -> TensorRead" path holds.
        let data: TensorData = act
            .try_into_data()
            .map_err(|e| BurnTernaryError::TensorRead(format!("{e:?}")))?;
        // f32-only: a non-f32 (e.g. f16/bf16) tensor returns TensorRead here
        // rather than running at reduced precision.
        let acts: Vec<f32> = data
            .to_vec::<f32>()
            .map_err(|e| BurnTernaryError::TensorRead(format!("{e:?}")))?;
        // `dims()` already guaranteed [M, K]; `to_vec` yields M*K row-major f32.
        debug_assert_eq!(acts.len(), m * k);

        let mut out = vec![0f32; m * n];
        reference_mpgemm(&acts, &trits, scales, GemmShape { m, n, k }, &mut out)
            .map_err(|e| BurnTernaryError::Kernel(format!("{e}")))?;

        // Pin the result to f32 (via the (&device, DType) creation options) so it
        // is not silently downcast on a backend whose default float dtype is half.
        Ok(Tensor::<B, 2>::from_data(
            TensorData::new(out, [m, n]),
            (&device, DType::F32),
        ))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use burn_ndarray::NdArray;
        use tritium_format::{pack_tq1_0_row, pack_tq2_0_row};
        use tritium_testkit::{ConformanceVector, FROZEN_COUNT, FROZEN_SEED, generate_vectors};

        /// The conformance backend: burn's CPU NdArray backend.
        type B = NdArray;

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

        /// The burn op reproduces the frozen conformance set. Because it runs the
        /// same `reference_mpgemm` over pack→unpack-round-tripped trits, it is
        /// bit-exact; the gate proves the burn Tensor <-> slice plumbing (shape,
        /// data readback, rebuild) is correct.
        #[test]
        fn burn_op_matches_reference_on_frozen_set() {
            // generate_vectors appends the fixed boundary set on top of the
            // FROZEN_COUNT random vectors, so the total exceeds FROZEN_COUNT.
            let vs = vectors();
            let total = vs.len();
            assert!(total > FROZEN_COUNT, "boundary vectors must be included");
            let mut checked = 0usize;
            let device = Default::default();
            for v in vs {
                let format = v.format;
                let packed = pack(&v, format);
                let act = Tensor::<B, 2>::from_data(
                    TensorData::new(v.activation.clone(), [v.m, v.k]),
                    &device,
                );
                let out = ternary_mpgemm(act, &packed, &v.scales, v.k, format).unwrap();
                assert_eq!(out.dims(), [v.m, v.n], "vector {}: output shape", v.id);
                let got = out.into_data().to_vec::<f32>().unwrap();
                // The op runs the SAME reference_mpgemm over pack->unpack
                // round-tripped trits in the same order, so it is bit-exact with
                // the frozen `expected`; grade it exactly (a tolerance here would
                // mask a real Tensor<->slice plumbing regression).
                assert_eq!(
                    got, v.expected,
                    "vector {}: burn op must be bit-exact with the reference",
                    v.id
                );
                checked += 1;
            }
            assert_eq!(checked, total, "every frozen vector exercised");
        }

        #[test]
        fn rejects_k_mismatch() {
            let device = Default::default();
            let act =
                Tensor::<B, 2>::from_data(TensorData::new(vec![1.0f32; 2 * 4], [2, 4]), &device);
            // N=2 (scales), weight K=8 != activation K=4
            let r = ternary_mpgemm(act, &[0u8; 64], &[1.0, 1.0], 8, TernaryFormat::Tq2_0);
            assert!(
                matches!(r, Err(BurnTernaryError::KMismatch { .. })),
                "K mismatch must error: {r:?}"
            );
        }

        #[test]
        fn rejects_packed_len_mismatch() {
            let device = Default::default();
            let act =
                Tensor::<B, 2>::from_data(TensorData::new(vec![1.0f32; 2 * 32], [2, 32]), &device);
            // N=2 (scales), empty packed buffer for a real [N=2, K=32] shape
            let r = ternary_mpgemm(act, &[], &[1.0, 1.0], 32, TernaryFormat::Tq2_0);
            assert!(
                matches!(r, Err(BurnTernaryError::PackedLenMismatch { .. })),
                "packed length mismatch must error: {r:?}"
            );
        }

        #[test]
        fn rejects_unsupported_format() {
            let device = Default::default();
            let act =
                Tensor::<B, 2>::from_data(TensorData::new(vec![1.0f32; 2 * 32], [2, 32]), &device);
            // I2sInt8 is a GPU-only packing this host op does not consume.
            let r = ternary_mpgemm(act, &[], &[1.0, 1.0], 32, TernaryFormat::I2sInt8);
            assert!(
                matches!(r, Err(BurnTernaryError::UnsupportedFormat(_))),
                "unsupported format must error: {r:?}"
            );
        }

        #[test]
        fn error_is_std_error_and_displays() {
            // Confirm the enum satisfies the std::error::Error contract and its
            // Display is non-empty (the crate ships its own error, no thiserror).
            fn assert_std_error<E: std::error::Error>(_: &E) {}
            let e = BurnTernaryError::KMismatch {
                activation_k: 4,
                weight_k: 8,
            };
            assert_std_error(&e);
            assert!(!e.to_string().is_empty());
        }
    }
}

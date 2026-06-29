//! # tritium-candle — Tritium's ternary mpGEMM as a candle op.
//!
//! Exposes [`ternary_mpgemm`], a [`candle_core::CustomOp1`] that runs Tritium's
//! ternary (BitNet b1.58) matrix multiply on a candle [`Tensor`]: an `[M, K]`
//! f32 activation tensor times `[N, K]` packed ternary weights (TQ2_0 / TQ1_0)
//! with `[N]` per-output-channel scales, producing `[M, N]` f32. The kernel is
//! [`tritium_core::reference_mpgemm`] itself, so a candle BitNet layer is
//! **bit-exact** with the reference every Tritium backend is graded against.
//!
//! This lets a candle model graph use Tritium ternary weights as a drop-in
//! `CustomOp1` while the rest of the network stays in candle.
//!
//! ## Feature gate
//!
//! The op (and the heavy `candle-core` dependency) live behind the **`candle`**
//! feature, off by default — so the default workspace build and
//! `cargo test --workspace` stay candle-free. Enable it to build the op:
//!
//! ```text
//! cargo test -p tritium-candle --features candle
//! ```
#![deny(missing_docs)]

#[cfg(feature = "candle")]
pub use candle_op::{TernaryMpGemm, ternary_mpgemm};

#[cfg(feature = "candle")]
mod candle_op {
    use candle_core::{CpuStorage, CustomOp1, Error as CandleError, Layout, Result, Shape, Tensor};
    use half::f16;
    use tritium_core::{GemmShape, TernaryFormat, Trit, reference_mpgemm};
    use tritium_format::{
        TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
    };

    /// Packed bytes per block for a supported ternary format.
    fn block_bytes(format: TernaryFormat) -> Result<usize> {
        match format {
            TernaryFormat::Tq2_0 => Ok(TQ2_0_BLOCK_BYTES),
            TernaryFormat::Tq1_0 => Ok(TQ1_0_BLOCK_BYTES),
            other => Err(CandleError::Msg(format!(
                "tritium-candle: unsupported ternary format {other:?}"
            ))),
        }
    }

    /// A candle [`CustomOp1`] computing `out[M,N] = scale[n] * Σ_k act[m,k] * w[n,k]`
    /// for packed ternary weights. Captures the `[N, K]` packed weights, the `[N]`
    /// scales, the weight shape, and the packing format; the single tensor
    /// argument is the `[M, K]` activations.
    #[derive(Debug)]
    pub struct TernaryMpGemm<'a> {
        packed: &'a [u8],
        scales: &'a [f32],
        n: usize,
        k: usize,
        format: TernaryFormat,
    }

    impl TernaryMpGemm<'_> {
        /// Unpack the `[N, K]` packed weights into a flat `Vec<Trit>`, validating
        /// the byte length against the captured shape + format.
        fn unpack(&self) -> Result<Vec<Trit>> {
            let nb = num_blocks(self.k);
            let row_bytes = nb * block_bytes(self.format)?;
            let expected = self.n * row_bytes;
            if self.packed.len() != expected {
                return Err(CandleError::Msg(format!(
                    "tritium-candle: packed weights len {} != expected {expected} for [N={}, K={}] {:?}",
                    self.packed.len(),
                    self.n,
                    self.k,
                    self.format
                )));
            }
            let mut trits = vec![Trit::ZERO; self.n * self.k];
            // Per-block scale scratch: the packer fixed these to 1.0; the
            // per-channel scales are applied in the contraction.
            let mut scratch = vec![f16::ONE; nb];
            for ni in 0..self.n {
                let row = &self.packed[ni * row_bytes..(ni + 1) * row_bytes];
                let trow = &mut trits[ni * self.k..ni * self.k + self.k];
                let res = match self.format {
                    TernaryFormat::Tq2_0 => unpack_tq2_0_row(row, trow, &mut scratch),
                    TernaryFormat::Tq1_0 => unpack_tq1_0_row(row, trow, &mut scratch),
                    other => {
                        return Err(CandleError::Msg(format!(
                            "tritium-candle: unsupported format {other:?}"
                        )));
                    }
                };
                res.map_err(|e| CandleError::Msg(format!("tritium-candle: unpack row {ni}: {e}")))?;
            }
            Ok(trits)
        }
    }

    impl CustomOp1 for TernaryMpGemm<'_> {
        fn name(&self) -> &'static str {
            "tritium-ternary-mpgemm"
        }

        fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
            let (m, k) = layout.shape().dims2()?;
            if k != self.k {
                return Err(CandleError::Msg(format!(
                    "tritium-candle: activation K={k} != weight K={}",
                    self.k
                )));
            }
            let all = storage.as_slice::<f32>()?;
            // The reference kernel reads a contiguous [M, K] row-major slice.
            let acts = match layout.contiguous_offsets() {
                Some((start, end)) => &all[start..end],
                None => {
                    return Err(CandleError::Msg(
                        "tritium-candle: activations must be contiguous (call .contiguous() first)"
                            .to_string(),
                    ));
                }
            };
            let trits = self.unpack()?;
            let mut out = vec![0f32; m * self.n];
            reference_mpgemm(
                acts,
                &trits,
                self.scales,
                GemmShape { m, n: self.n, k },
                &mut out,
            )
            .map_err(|e| CandleError::Msg(format!("tritium-candle: mpgemm: {e}")))?;
            Ok((CpuStorage::F32(out), Shape::from((m, self.n))))
        }
    }

    /// Run Tritium's ternary mpGEMM on `act` (an `[M, K]` f32 [`Tensor`]) against
    /// `[N, K]` `packed` ternary weights in `format` with `[N]` `scales`, returning
    /// the `[M, N]` f32 result on the same device as `act`. `N` (the output-channel
    /// count) is taken from `scales.len()`.
    ///
    /// The op borrows `packed`/`scales`, so a candle module that owns its weight
    /// bytes calls this once per forward rather than storing the op.
    ///
    /// # Errors
    /// A [`candle_core::Error`] if `act` is not a 2-D f32 tensor, is non-contiguous,
    /// its `K` disagrees with `k`, the packed length is wrong for
    /// `[N = scales.len(), K]` in `format`, or the format is unsupported.
    pub fn ternary_mpgemm(
        act: &Tensor,
        packed: &[u8],
        scales: &[f32],
        k: usize,
        format: TernaryFormat,
    ) -> Result<Tensor> {
        // N (output channels) is exactly the number of per-channel scales — derive
        // it rather than taking a second loose `usize` the caller could swap with k.
        let op = TernaryMpGemm {
            packed,
            scales,
            n: scales.len(),
            k,
            format,
        };
        act.apply_op1_no_bwd(&op)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use candle_core::{Device, Tensor};
        use tritium_format::{pack_tq1_0_row, pack_tq2_0_row};
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

        fn vectors() -> Vec<ConformanceVector> {
            generate_vectors(FROZEN_SEED, FROZEN_COUNT)
        }

        /// The candle op reproduces the frozen conformance set. Because it runs the
        /// same `reference_mpgemm` over pack→unpack-round-tripped trits, it is
        /// bit-exact; the gate proves the candle Tensor <-> slice plumbing (layout,
        /// shape, readback) is correct.
        #[test]
        fn candle_op_matches_reference_on_frozen_set() {
            // generate_vectors appends the fixed boundary set on top of the
            // FROZEN_COUNT random vectors, so the total exceeds FROZEN_COUNT.
            let vs = vectors();
            let total = vs.len();
            assert!(total > FROZEN_COUNT, "boundary vectors must be included");
            let mut checked = 0usize;
            for v in vs {
                let format = v.format;
                let packed = pack(&v, format);
                let act = Tensor::from_vec(v.activation.clone(), (v.m, v.k), &Device::Cpu).unwrap();
                let out = ternary_mpgemm(&act, &packed, &v.scales, v.k, format).unwrap();
                assert_eq!(out.dims(), [v.m, v.n], "vector {}: output shape", v.id);
                let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                // The op runs the SAME reference_mpgemm over pack->unpack
                // round-tripped trits in the same order, so it is bit-exact with
                // the frozen `expected`; grade it exactly (a tolerance here would
                // mask a real Tensor<->slice plumbing regression).
                assert_eq!(
                    got, v.expected,
                    "vector {}: candle op must be bit-exact with the reference",
                    v.id
                );
                checked += 1;
            }
            assert_eq!(checked, total, "every frozen vector exercised");
        }

        #[test]
        fn rejects_k_mismatch() {
            let act = Tensor::from_vec(vec![1.0f32; 2 * 4], (2, 4), &Device::Cpu).unwrap();
            // N=2 (scales), weight K=8 != activation K=4
            let r = ternary_mpgemm(&act, &[0u8; 64], &[1.0, 1.0], 8, TernaryFormat::Tq2_0);
            assert!(r.is_err(), "K mismatch must error");
        }

        #[test]
        fn rejects_packed_len_mismatch() {
            let act = Tensor::from_vec(vec![1.0f32; 2 * 32], (2, 32), &Device::Cpu).unwrap();
            // N=2 (scales), empty packed buffer for a real [N=2, K=32] shape
            let r = ternary_mpgemm(&act, &[], &[1.0, 1.0], 32, TernaryFormat::Tq2_0);
            assert!(r.is_err(), "packed length mismatch must error");
        }

        #[test]
        fn rejects_non_contiguous_activation() {
            // A transposed view is non-contiguous; the op must reject it (the
            // reference kernel reads a contiguous [M, K] slice) rather than
            // silently mis-reading strided data.
            let base = Tensor::from_vec(vec![1.0f32; 4 * 32], (32, 4), &Device::Cpu).unwrap();
            let act = base.t().unwrap(); // [4, 32], non-contiguous
            assert!(!act.is_contiguous());
            let nb = num_blocks(32);
            let packed = vec![0u8; 2 * nb * block_bytes(TernaryFormat::Tq2_0).unwrap()];
            let r = ternary_mpgemm(&act, &packed, &[1.0, 1.0], 32, TernaryFormat::Tq2_0);
            assert!(r.is_err(), "non-contiguous activation must error");
        }
    }
}

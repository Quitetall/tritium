//! # tritium-cpu
//!
//! The x86-64 CPU execution backend. A runtime-dispatched AVX2 kernel with a
//! scalar fallback (the scalar path delegates to
//! [`tritium_core::reference_mpgemm`], so it is correct by construction). The
//! backend self-registers with `tritium-runtime` through the `BACKENDS` `linkme`
//! distributed slice, so linking this crate into a binary makes a `"cpu"` backend
//! appear in the runtime registry with no central edit.
//!
//! ## Path
//!
//! 1. [`CpuBackend::upload_weights`] validates the packed byte length against the
//!    `[N, K]` shape and the format's block size, then stores the bytes verbatim
//!    in a [`CpuBuffer`].
//! 2. [`CpuBackend::mpgemm`] downcasts the buffer, unpacks the `[N, K]` weights to
//!    `Vec<Trit>` with the matching `tritium-format` row unpacker, and runs the
//!    contraction `out[m, n] = scale[n] · Σ_k act[m, k] · trit[n, k]`.
//!
//! The packed block scale is fixed to `1.0` by the host packer, so the unpacked
//! trits are the raw weights; the per-output-channel `scales` are the only thing
//! applied to the contraction (matching the testkit's packing-vs-scaling split).
//!
//! ## Kernels
//!
//! Two kernels back [`CpuBackend::mpgemm`], chosen at runtime: an AVX2 intrinsic
//! kernel when the host advertises `avx2`, and the scalar reference otherwise. The
//! AVX2 kernel folds its SIMD-decoded contributions sequentially in `f32`,
//! reproducing the reference accumulation bit-for-bit — comfortably inside the
//! `1e-4` relative tolerance of ADR 0002 (in fact exact). The independent `M` rows
//! are spread across `rayon`'s thread pool without changing per-row arithmetic, so
//! results are deterministic regardless of thread count.
//
// `linkme`'s `distributed_slice` expands to a static with a custom
// `#[link_section]`, and the AVX2 kernel needs hand-written `unsafe` for its
// intrinsics. We therefore `deny` (not `forbid`) unsafe code at the crate level
// and grant `#[allow(unsafe_code)]` narrowly on the registration static; the
// kernel module carries its own scoped `unsafe` with `// SAFETY:` notes.
#![deny(unsafe_code)]

use core::any::Any;

use tritium_core::{GemmShape, TernaryFormat, Trit};
use tritium_format::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
};
use tritium_runtime::BackendEntry;
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, TernaryBackend};

// The kernel module holds the only hand-written `unsafe` in the crate: the AVX2
// intrinsic kernel (its `unsafe fn` plus the raw-pointer loads/stores). The
// crate-level `deny(unsafe_code)` is relaxed for exactly this module; each unsafe
// block inside carries a `// SAFETY:` note.
#[allow(unsafe_code)]
mod kernel;

// SIMD kernel variants for v0.30 (AVX-512 / VNNI, ARM NEON, T-MAC LUT). Skeleton
// module tree (ADR 0005); WF-C implements them and wires the selection into
// `kernel::dispatch_mpgemm` behind the existing `is_x86_feature_detected!` /
// `target_arch` dispatch, gated by the cross-ISA conformance parity gate.
mod simd;

/// Owned host-memory buffer of packed weight bytes.
///
/// [`CpuBackend::upload_weights`] copies the caller's packed bytes into one of
/// these; [`CpuBackend::mpgemm`] downcasts the [`DeviceBuffer`] back to it and
/// unpacks on the fly. Storing the packed (not unpacked) bytes keeps the upload
/// cheap and the buffer the same size as the on-disk weight.
#[derive(Debug, Clone)]
pub struct CpuBuffer {
    bytes: Vec<u8>,
}

impl CpuBuffer {
    /// Wrap already-packed bytes in a buffer.
    #[must_use]
    fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// The packed bytes this buffer holds.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl DeviceBuffer for CpuBuffer {
    fn len_bytes(&self) -> usize {
        self.bytes.len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Packed bytes per block for a format this backend supports.
///
/// # Errors
/// [`BackendError::UnsupportedFormat`] for any format the CPU backend does not
/// pack (everything but TQ2_0 / TQ1_0 today).
fn block_bytes(format: TernaryFormat) -> Result<usize, BackendError> {
    match format {
        TernaryFormat::Tq2_0 => Ok(TQ2_0_BLOCK_BYTES),
        TernaryFormat::Tq1_0 => Ok(TQ1_0_BLOCK_BYTES),
        other => Err(BackendError::UnsupportedFormat(other)),
    }
}

/// The x86-64 CPU backend.
///
/// Stateless: every call re-derives what it needs from its arguments, so a single
/// `CpuBackend` is shared freely across threads (`Send + Sync`).
#[derive(Debug, Default, Clone)]
pub struct CpuBackend;

impl CpuBackend {
    /// Construct the CPU backend.
    #[must_use]
    pub fn new() -> Self {
        CpuBackend
    }

    /// Unpack the `[N, K]` packed weights in `buf` to a flat `Vec<Trit>` (one
    /// `N·K` row-major buffer), validating the byte length against `shape`.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] if the byte length disagrees with the shape
    /// and format; [`BackendError::Backend`] if `tritium-format` rejects a row.
    fn unpack_weights(
        buf: &CpuBuffer,
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Vec<Trit>, BackendError> {
        let GemmShape { n, k, .. } = shape;
        let nb = num_blocks(k);
        let row_bytes = nb * block_bytes(format)?;
        let expected = n * row_bytes;
        if buf.bytes.len() != expected {
            return Err(BackendError::ShapeMismatch {
                expected,
                got: buf.bytes.len(),
            });
        }

        let mut trits = vec![Trit::ZERO; n * k];
        // Per-block scale scratch: the host packer fixed these to 1.0 and the
        // per-channel scales are applied in `mpgemm`, so the unpacked block scales
        // are discarded.
        let mut scratch = vec![half::f16::ONE; nb];
        for ni in 0..n {
            let row = &buf.bytes[ni * row_bytes..(ni + 1) * row_bytes];
            let trits_row = &mut trits[ni * k..ni * k + k];
            let res = match format {
                TernaryFormat::Tq2_0 => unpack_tq2_0_row(row, trits_row, &mut scratch),
                TernaryFormat::Tq1_0 => unpack_tq1_0_row(row, trits_row, &mut scratch),
                other => return Err(BackendError::UnsupportedFormat(other)),
            };
            res.map_err(|e| BackendError::Backend(format!("unpack row {ni}: {e}")))?;
        }
        Ok(trits)
    }
}

impl TernaryBackend for CpuBackend {
    fn device_id(&self) -> &str {
        "cpu"
    }

    fn capabilities(&self) -> DeviceCaps {
        let mut features: Vec<String> = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                features.push("avx2".to_owned());
            }
            if is_x86_feature_detected!("fma") {
                features.push("fma".to_owned());
            }
        }
        DeviceCaps::new("cpu", host_arch_name()).with_features(features)
    }

    fn upload_weights(
        &self,
        packed: &[u8],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let GemmShape { n, k, .. } = shape;
        let nb = num_blocks(k);
        let row_bytes = nb * block_bytes(format)?;
        let expected = n * row_bytes;
        if packed.len() != expected {
            return Err(BackendError::InvalidInput(format!(
                "packed len {} != expected {expected} for shape {shape:?} format {format:?}",
                packed.len()
            )));
        }
        Ok(Box::new(CpuBuffer::new(packed.to_vec())))
    }

    fn mpgemm(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        scales: &[f32],
        shape: GemmShape,
        format: TernaryFormat,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let buf = weights
            .as_any()
            .downcast_ref::<CpuBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a CpuBuffer".to_owned()))?;

        let GemmShape { m, n, k } = shape;
        // Validate operand lengths against the shape before any kernel runs.
        if act.len() != m * k {
            return Err(BackendError::ShapeMismatch {
                expected: m * k,
                got: act.len(),
            });
        }
        if scales.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: scales.len(),
            });
        }
        if out.len() != m * n {
            return Err(BackendError::ShapeMismatch {
                expected: m * n,
                got: out.len(),
            });
        }

        // Unpack the [N, K] weights; this also validates the packed byte length.
        let trits = Self::unpack_weights(buf, shape, format)?;

        // WF-C dispatch hook: `dispatch_mpgemm` picks AVX2-vs-scalar today; the
        // v0.30 `simd::{avx512,neon,lut}` paths slot in here behind feature
        // detection (results must stay bit-parity with the scalar reference).
        kernel::dispatch_mpgemm(act, &trits, scales, shape, out)
    }
}

/// A human-readable host arch name with the SIMD tier appended when known.
fn host_arch_name() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            "x86_64 (avx2)".to_owned()
        } else {
            "x86_64".to_owned()
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::env::consts::ARCH.to_owned()
    }
}

/// `init` constructor for the runtime registry. The CPU backend is always
/// available, so this never fails.
///
/// # Errors
/// Never returns `Err`; the signature matches [`BackendEntry::init`].
fn init_cpu() -> Result<Box<dyn TernaryBackend>, BackendError> {
    Ok(Box::new(CpuBackend::new()))
}

// `linkme`'s `distributed_slice` expands to a static with a custom
// `#[link_section]`, which the `unsafe_code` lint flags. Grant a narrowly-scoped
// allow on exactly this registration static (the crate otherwise `deny`s unsafe
// outside the kernel module).
#[allow(unsafe_code)]
#[linkme::distributed_slice(tritium_runtime::BACKENDS)]
static CPU: BackendEntry = BackendEntry {
    name: "cpu",
    init: init_cpu,
};

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use tritium_format::{pack_tq1_0_row, pack_tq2_0_row};
    use tritium_testkit::{Tolerance, generate_vectors, run_conformance};

    /// Pack an `[N, K]` trit matrix into the format's row layout, block scale
    /// fixed to `1.0` (the testkit convention), ready for `upload_weights`.
    fn pack(trits: &[Trit], shape: GemmShape, format: TernaryFormat) -> Vec<u8> {
        let GemmShape { n, k, .. } = shape;
        let nb = num_blocks(k);
        let unit = vec![f16::ONE; nb];
        let row_bytes = nb * block_bytes(format).unwrap();
        let mut packed = vec![0u8; n * row_bytes];
        for ni in 0..n {
            let row = &trits[ni * k..ni * k + k];
            let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
            match format {
                TernaryFormat::Tq2_0 => pack_tq2_0_row(row, &unit, out).unwrap(),
                TernaryFormat::Tq1_0 => pack_tq1_0_row(row, &unit, out).unwrap(),
                _ => unreachable!("only TQ2_0/TQ1_0 are packed in tests"),
            }
        }
        packed
    }

    /// Run one mpGEMM end-to-end through the backend (upload + compute).
    fn run(
        trits: &[Trit],
        act: &[f32],
        scales: &[f32],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Vec<f32> {
        let backend = CpuBackend::new();
        let packed = pack(trits, shape, format);
        let buf = backend.upload_weights(&packed, shape, format).unwrap();
        let mut out = vec![0.0f32; shape.m * shape.n];
        backend
            .mpgemm(act, buf.as_ref(), scales, shape, format, &mut out)
            .unwrap();
        out
    }

    fn trits(vals: &[i8]) -> Vec<Trit> {
        vals.iter().map(|&v| Trit::from_i8(v).unwrap()).collect()
    }

    /// Relative-tolerance assertion matching ADR 0002 (`1e-4`).
    fn close(got: f32, want: f32) -> bool {
        (got - want).abs() <= 1e-4 * want.abs().max(1.0)
    }

    // ---- the conformance gate ------------------------------------------------

    #[test]
    fn conformance_zero_failures() {
        let vectors = generate_vectors(0xC0FFEE, 64);
        let report = run_conformance(&CpuBackend::new(), &vectors, Tolerance::default());
        assert!(
            report.is_ok(),
            "{} conformance failures: {:?}",
            report.failed.len(),
            report.failed
        );
        assert_eq!(report.passed, vectors.len(), "all vectors must pass");
    }

    #[test]
    fn conformance_zero_failures_other_seeds() {
        for seed in [1u64, 7, 42, 0xDEAD_BEEF] {
            let vectors = generate_vectors(seed, 24);
            let report = run_conformance(&CpuBackend::new(), &vectors, Tolerance::default());
            assert!(
                report.is_ok(),
                "seed {seed}: {} failures: {:?}",
                report.failed.len(),
                report.failed
            );
        }
    }

    // ---- device_id / capabilities -------------------------------------------

    #[test]
    fn device_id_is_cpu() {
        assert_eq!(CpuBackend::new().device_id(), "cpu");
    }

    #[test]
    fn capabilities_report_cpu_backend() {
        let caps = CpuBackend::new().capabilities();
        assert_eq!(caps.backend, "cpu");
        assert!(
            !caps.device_name.is_empty(),
            "device name should be set: {:?}",
            caps.device_name
        );
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            assert!(caps.has_feature("avx2"), "avx2 host must advertise avx2");
        }
    }

    // ---- registration --------------------------------------------------------

    #[test]
    fn registers_with_runtime() {
        // Force this crate's linkme static to be referenced so the entry is kept.
        let _ = &CPU;
        let backend = init_cpu().expect("cpu init never fails");
        assert_eq!(backend.device_id(), "cpu");
    }

    // ---- upload_weights validation ------------------------------------------

    #[test]
    fn upload_rejects_wrong_length() {
        let backend = CpuBackend::new();
        let shape = GemmShape::new(1, 2, 256);
        // One byte short of the expected packed size.
        let nb = num_blocks(256);
        let short = vec![0u8; 2 * nb * TQ2_0_BLOCK_BYTES - 1];
        // `Box<dyn DeviceBuffer>` is not `Debug`, so `unwrap_err` is unavailable;
        // match on the result instead.
        match backend.upload_weights(&short, shape, TernaryFormat::Tq2_0) {
            Err(BackendError::InvalidInput(_)) => {}
            other => panic!(
                "expected InvalidInput, got {:?}",
                other.map(|_| "ok-buffer")
            ),
        }
    }

    #[test]
    fn upload_accepts_correct_length() {
        let backend = CpuBackend::new();
        let shape = GemmShape::new(4, 3, 256);
        let w = vec![Trit::ZERO; 3 * 256];
        let packed = pack(&w, shape, TernaryFormat::Tq2_0);
        let buf = backend
            .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
            .unwrap();
        assert_eq!(buf.len_bytes(), packed.len());
    }

    #[test]
    fn mpgemm_rejects_foreign_buffer() {
        // A DeviceBuffer that is not a CpuBuffer must be rejected.
        #[derive(Debug)]
        struct Foreign;
        impl DeviceBuffer for Foreign {
            fn len_bytes(&self) -> usize {
                0
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
        let backend = CpuBackend::new();
        let shape = GemmShape::new(1, 1, 256);
        let act = vec![0.0f32; 256];
        let scales = vec![1.0f32; 1];
        let mut out = vec![0.0f32; 1];
        let err = backend
            .mpgemm(
                &act,
                &Foreign,
                &scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut out,
            )
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn mpgemm_rejects_bad_operand_lengths() {
        let backend = CpuBackend::new();
        let shape = GemmShape::new(2, 1, 256);
        let w = vec![Trit::ZERO; 256];
        let packed = pack(&w, shape, TernaryFormat::Tq2_0);
        let buf = backend
            .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
            .unwrap();
        // act too short (should be 2*256).
        let act = vec![0.0f32; 256];
        let scales = vec![1.0f32; 1];
        let mut out = vec![0.0f32; 2];
        let err = backend
            .mpgemm(
                &act,
                buf.as_ref(),
                &scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut out,
            )
            .unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch { .. }), "{err:?}");
    }

    // ---- boundary cases ------------------------------------------------------

    #[test]
    fn boundary_all_zero_weights_give_zero() {
        let shape = GemmShape::new(4, 3, 256);
        let w = vec![Trit::ZERO; 3 * 256];
        let act: Vec<f32> = (0..4 * 256).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let scales = vec![1.5f32; 3];
        let out = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        assert!(
            out.iter().all(|&x| x == 0.0),
            "all-zero weights -> zero out"
        );
    }

    #[test]
    fn boundary_m_equals_one() {
        let shape = GemmShape::new(1, 5, 256);
        let w: Vec<Trit> = (0..5 * 256)
            .map(|i| Trit::from_sign((i % 3) as i8 - 1))
            .collect();
        let act: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.0).collect();
        let scales = vec![1.0f32; 5];
        let out = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        let mut want = vec![0.0f32; 5];
        tritium_core::reference_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();
        for (g, e) in out.iter().zip(&want) {
            assert!(close(*g, *e));
        }
    }

    #[test]
    fn boundary_m_large() {
        let shape = GemmShape::new(64, 2, 256);
        let w: Vec<Trit> = (0..2 * 256)
            .map(|i| Trit::from_sign((i % 3) as i8 - 1))
            .collect();
        let act: Vec<f32> = (0..64 * 256)
            .map(|i| ((i * 31 % 200) as f32) / 50.0 - 2.0)
            .collect();
        let scales = vec![0.7f32, 1.3];
        let out = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        let mut want = vec![0.0f32; 64 * 2];
        tritium_core::reference_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();
        for (g, e) in out.iter().zip(&want) {
            assert!(close(*g, *e));
        }
    }

    #[test]
    fn boundary_n_equals_one() {
        let shape = GemmShape::new(3, 1, 256);
        let w: Vec<Trit> = (0..256)
            .map(|i| Trit::from_sign((i % 3) as i8 - 1))
            .collect();
        let act: Vec<f32> = (0..3 * 256).map(|i| (i as f32 % 5.0) - 2.0).collect();
        let scales = vec![2.0f32];
        let out = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        let mut want = vec![0.0f32; 3];
        tritium_core::reference_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();
        for (g, e) in out.iter().zip(&want) {
            assert!(close(*g, *e));
        }
    }

    #[test]
    fn boundary_all_pos_and_all_neg() {
        let shape = GemmShape::new(2, 2, 256);
        let act: Vec<f32> = (0..2 * 256).map(|i| (i as f32 % 11.0) - 5.0).collect();
        let scales = vec![1.0f32, 1.0];

        // All +1: each output is the row-sum of the activation.
        let allpos = vec![Trit::POS; 2 * 256];
        let out_pos = run(&allpos, &act, &scales, shape, TernaryFormat::Tq2_0);
        for (mi, _) in (0..2).enumerate() {
            let row_sum: f32 = act[mi * 256..mi * 256 + 256].iter().sum();
            for ni in 0..2 {
                let g = out_pos[mi * 2 + ni];
                assert!(close(g, row_sum));
            }
        }

        // All -1: each output is the negated row-sum.
        let allneg = vec![Trit::NEG; 2 * 256];
        let out_neg = run(&allneg, &act, &scales, shape, TernaryFormat::Tq1_0);
        for (mi, _) in (0..2).enumerate() {
            let row_sum: f32 = act[mi * 256..mi * 256 + 256].iter().sum();
            for ni in 0..2 {
                let g = out_neg[mi * 2 + ni];
                assert!(close(g, -row_sum));
            }
        }
    }

    #[test]
    fn nan_inf_activation_propagates_without_ub() {
        // A NaN/Inf activation hitting a non-zero trit must propagate, not crash.
        let shape = GemmShape::new(1, 1, 256);
        let w = vec![Trit::POS; 256];
        let mut act = vec![1.0f32; 256];
        act[0] = f32::NAN;
        let scales = vec![1.0f32];
        let out = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        assert!(
            out[0].is_nan(),
            "NaN activation must propagate to the output"
        );

        let mut act_inf = vec![1.0f32; 256];
        act_inf[5] = f32::INFINITY;
        let out_inf = run(&w, &act_inf, &scales, shape, TernaryFormat::Tq2_0);
        assert!(
            out_inf[0].is_infinite(),
            "Inf activation must propagate to the output"
        );
    }

    // ---- AVX2 vs scalar parity ----------------------------------------------

    #[test]
    fn avx2_vs_scalar_parity() {
        #[cfg(target_arch = "x86_64")]
        {
            if !is_x86_feature_detected!("avx2") {
                eprintln!("avx2 not detected on this host — skipping AVX2/scalar parity test");
                return;
            }
            let mut s = 0xBADC_0FFE_u64;
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            };
            for trial in 0..24 {
                let m = 1 + (next() % 8) as usize;
                let n = 1 + (next() % 8) as usize;
                // Mix ragged Ks (with an AVX2 scalar tail) and large block-aligned
                // Ks (256/512) where the reference's own f32 cancellation is worst.
                let k = match trial % 3 {
                    0 => 17 + (next() % 100) as usize, // ragged tail
                    1 => 256,                          // one block
                    _ => 512,                          // two blocks
                };
                let shape = GemmShape::new(m, n, k);
                // Large-magnitude activations make partial sums cancel hard, so the
                // reference's f32 order is the only one within tolerance — and the
                // kernel reproduces it bit-for-bit.
                let act: Vec<f32> = (0..m * k)
                    .map(|_| (next() % 20000) as f32 / 100.0 - 100.0)
                    .collect();
                let w: Vec<Trit> = (0..n * k)
                    .map(|_| Trit::from_sign((next() % 3) as i8 - 1))
                    .collect();
                let scales: Vec<f32> = (0..n)
                    .map(|_| (next() % 200) as f32 / 100.0 + 0.1)
                    .collect();

                let mut scal = vec![0.0f32; m * n];
                kernel::scalar_mpgemm(&act, &w, &scales, shape, &mut scal).unwrap();
                let mut avx = vec![0.0f32; m * n];
                // SAFETY: avx2 availability was just confirmed by the guard above.
                #[allow(unsafe_code)]
                unsafe {
                    kernel::avx2_mpgemm(&act, &w, &scales, shape, &mut avx).unwrap();
                }
                // The kernel reproduces the reference's exact f32 accumulation
                // order, so parity is bit-exact, not merely within 1e-4.
                for (a, e) in avx.iter().zip(&scal) {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "trial {trial} shape {shape:?}: avx2 {a} vs scalar {e}"
                    );
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            eprintln!("not x86_64 — AVX2/scalar parity test is a no-op");
        }
    }

    // ---- determinism ---------------------------------------------------------

    #[test]
    fn determinism_same_input_same_bytes() {
        let shape = GemmShape::new(8, 4, 512);
        let w: Vec<Trit> = (0..4 * 512)
            .map(|i| Trit::from_sign((i % 3) as i8 - 1))
            .collect();
        let act: Vec<f32> = (0..8 * 512)
            .map(|i| ((i * 17 % 300) as f32) / 30.0 - 5.0)
            .collect();
        let scales: Vec<f32> = (0..4).map(|i| 0.5 + i as f32 * 0.25).collect();

        let a = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        let b = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        // Compare the exact bit patterns, not just within tolerance.
        let ab: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
        let bb: Vec<u32> = b.iter().map(|x| x.to_bits()).collect();
        assert_eq!(ab, bb, "same input must produce byte-identical output");
    }

    #[test]
    fn parallel_matches_reference_and_is_deterministic() {
        // M=64, N=32, K=512 splits into multiple rayon chunks (per-row op count
        // forces several tasks), so this exercises the parallel dispatch path. The
        // per-row accumulation order is unchanged, so the result must still be
        // bit-identical run-to-run and within tolerance of the reference.
        let shape = GemmShape::new(64, 32, 512);
        let w: Vec<Trit> = (0..32 * 512)
            .map(|i| Trit::from_sign((i % 3) as i8 - 1))
            .collect();
        let act: Vec<f32> = (0..64 * 512)
            .map(|i| ((i * 13 % 4000) as f32) / 20.0 - 100.0)
            .collect();
        let scales: Vec<f32> = (0..32).map(|i| 0.3 + (i % 7) as f32 * 0.2).collect();

        let a = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        let b = run(&w, &act, &scales, shape, TernaryFormat::Tq2_0);
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "parallel run must be deterministic"
        );

        let mut want = vec![0.0f32; 64 * 32];
        tritium_core::reference_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();
        for (g, e) in a.iter().zip(&want) {
            assert!(close(*g, *e), "parallel vs reference: {g} vs {e}");
        }
    }

    // ---- format round-trip through both packers -----------------------------

    #[test]
    fn both_formats_agree_with_reference() {
        let shape = GemmShape::new(2, 3, 256);
        let w = trits(&(0..3 * 256).map(|i| (i % 3) as i8 - 1).collect::<Vec<_>>());
        let act: Vec<f32> = (0..2 * 256).map(|i| (i as f32 % 9.0) - 4.0).collect();
        let scales = vec![1.1f32, 0.9, 1.0];

        let mut want = vec![0.0f32; 2 * 3];
        tritium_core::reference_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();

        for format in [TernaryFormat::Tq2_0, TernaryFormat::Tq1_0] {
            let got = run(&w, &act, &scales, shape, format);
            for (g, e) in got.iter().zip(&want) {
                assert!(close(*g, *e), "format {format:?}: {g} vs {e}");
            }
        }
    }
}

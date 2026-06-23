//! # tritium-wasm — scalar [`TernaryBackend`] for `wasm32-wasip1`.
//!
//! The kernel is [`tritium_core::reference_mpgemm`] itself, run over weights
//! unpacked with `tritium-format`'s row unpackers. It is therefore **bit-exact**
//! with the reference every other backend is graded against, so it passes the
//! conformance set trivially — the gate then really proves the wasm toolchain,
//! WASI, and IEEE-754 `f32` semantics execute it correctly inside the guest.
//!
//! ## Why not reuse `tritium-cpu`
//!
//! `tritium-cpu`'s scalar contraction is `pub(crate)` (unreachable from here), and
//! `tritium-cpu` does not build for `wasm32`: it depends on `tritium-runtime`
//! (whose `linkme::distributed_slice` is *unimplemented on wasm* —
//! `distributed_slice is not implemented for this platform`) and on `rayon`. So
//! this crate depends only on the wasm-clean `tritium-core` / `tritium-spec` /
//! `tritium-format`, and reuses the shared numeric truth `reference_mpgemm` with
//! zero duplication.
//!
//! ## No self-registration
//!
//! There is no `linkme` `BACKENDS` registration here (it does not compile on
//! wasm); the backend is wired via the explicit [`init_wasm`] constructor — the
//! wasm analogue of a `BackendEntry::init`.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

use core::any::Any;

use tritium_core::{GemmShape, TernaryFormat, Trit, reference_mpgemm};
use tritium_format::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
};
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, TernaryBackend};

/// Owned host-memory buffer of packed weight bytes (the wasm linear-memory copy).
#[derive(Debug, Clone)]
pub struct WasmBuffer {
    bytes: Vec<u8>,
}

impl DeviceBuffer for WasmBuffer {
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
/// [`BackendError::UnsupportedFormat`] for any format other than TQ2_0 / TQ1_0.
fn block_bytes(format: TernaryFormat) -> Result<usize, BackendError> {
    match format {
        TernaryFormat::Tq2_0 => Ok(TQ2_0_BLOCK_BYTES),
        TernaryFormat::Tq1_0 => Ok(TQ1_0_BLOCK_BYTES),
        other => Err(BackendError::UnsupportedFormat(other)),
    }
}

/// The scalar WebAssembly backend. Stateless; shared freely (`Send + Sync`).
#[derive(Debug, Default, Clone)]
pub struct WasmBackend;

impl WasmBackend {
    /// Construct the wasm backend. Always available, never fails.
    #[must_use]
    pub fn new() -> Self {
        WasmBackend
    }

    /// Unpack the `[N, K]` packed weights in `buf` to a flat `Vec<Trit>`,
    /// validating the byte length against `shape` and `format`.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] if the byte length disagrees with the shape
    /// and format; [`BackendError::Backend`] if `tritium-format` rejects a row.
    fn unpack_weights(
        buf: &WasmBuffer,
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
        // per-channel scales are applied in `mpgemm`, so they are discarded.
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

impl TernaryBackend for WasmBackend {
    fn device_id(&self) -> &str {
        "wasm"
    }

    fn capabilities(&self) -> DeviceCaps {
        // Pure scalar: no SIMD / fp8 / IMMA. The fused W1.58A8 path degrades to
        // the trait-default host quant (see the fused-fallback contract).
        DeviceCaps::new("wasm", "wasm32 (scalar)")
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
        Ok(Box::new(WasmBuffer {
            bytes: packed.to_vec(),
        }))
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
            .downcast_ref::<WasmBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a WasmBuffer".to_owned()))?;
        let trits = Self::unpack_weights(buf, shape, format)?;
        reference_mpgemm(act, &trits, scales, shape, out).map_err(BackendError::Core)
    }
}

/// Explicit `init` constructor — the wasm analogue of `BackendEntry::init`, since
/// `linkme` self-registration is unavailable on wasm32.
///
/// # Errors
/// Never returns `Err`; the scalar backend is always available.
pub fn init_wasm() -> Result<Box<dyn TernaryBackend>, BackendError> {
    Ok(Box::new(WasmBackend::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_testkit::{
        FROZEN_COUNT, FROZEN_SEED, Tolerance, generate_vectors, run_conformance,
        run_fused_fallback_contract,
    };

    // The frozen conformance set is exactly `generate_vectors(FROZEN_SEED,
    // FROZEN_COUNT)` (pinned by testkit's `frozen_set_matches_pinned_generator`
    // drift gate). We regenerate it in-memory rather than `frozen_vectors()`
    // (which reads the committed JSONL) so the SAME test runs under wasmtime,
    // where the WASI sandbox does not expose the testkit crate's vector file.
    fn vectors() -> Vec<tritium_testkit::ConformanceVector> {
        generate_vectors(FROZEN_SEED, FROZEN_COUNT)
    }

    /// The wasm backend reproduces the frozen conformance set with zero failures.
    /// Because the kernel *is* `reference_mpgemm`, this is bit-exact.
    #[test]
    fn conformance_zero_failures() {
        let vectors = vectors();
        let report = run_conformance(&WasmBackend::new(), &vectors, Tolerance::default());
        assert!(
            report.is_ok(),
            "{} failures: {:?}",
            report.failed.len(),
            report.failed
        );
        assert_eq!(report.passed, vectors.len(), "all vectors must pass");
    }

    /// Capability fallback: wasm advertises no fp8/IMMA yet must serve the fused
    /// path via the host-default degrade (no panic, correct output).
    #[test]
    fn fused_path_degrades() {
        let caps = WasmBackend::new().capabilities();
        assert!(!caps.supports_fp8 && !caps.supports_imma);
        let vectors = vectors();
        let report =
            run_fused_fallback_contract(&WasmBackend::new(), &vectors, Tolerance::default());
        assert!(
            report.is_ok(),
            "{} fused failures: {:?}",
            report.failed.len(),
            report.failed
        );
        assert_eq!(report.passed, vectors.len());
    }

    #[test]
    fn device_id_is_wasm() {
        assert_eq!(WasmBackend::new().device_id(), "wasm");
        assert_eq!(init_wasm().unwrap().device_id(), "wasm");
    }
}

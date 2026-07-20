//! # tritium-wasm — scalar [`TernaryBackend`] for WebAssembly.
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
#[cfg(any(test, target_arch = "wasm32"))]
use tritium_format::salt_v2_package::{read_salt_v2_package, write_salt_v2_package};
use tritium_format::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
};
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, MpGemm, TernaryBackend};

mod portable;
mod request;
pub use portable::WasmTrainBackendV1;
pub use request::tritium_execute_portable_request_json;

#[cfg(any(test, target_arch = "wasm32"))]
fn admit_canonical_salt_v2_package(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let decoded = read_salt_v2_package(bytes).map_err(|error| error.to_string())?;
    let encoded = write_salt_v2_package(&decoded.package).map_err(|error| error.to_string())?;
    if encoded.bytes != bytes {
        return Err("SALT V2 package is valid but not byte-canonical".to_owned());
    }
    Ok(encoded.bytes)
}

/// Strictly decode and byte-canonically reload one SALT V2 package without a
/// JSON-expanded byte array.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn tritium_admit_salt_v2_package(bytes: &[u8]) -> Result<Vec<u8>, wasm_bindgen::JsError> {
    admit_canonical_salt_v2_package(bytes).map_err(|error| wasm_bindgen::JsError::new(&error))
}

#[cfg(target_arch = "wasm32")]
fn portable_training_conformance_report()
-> Result<tritium_testkit::TrainingConformanceReport, String> {
    use tritium_spec::TrainingVectorSetV1;
    use tritium_testkit::run_training_conformance;

    let vectors = TrainingVectorSetV1::parse_json(TrainingVectorSetV1::canonical_json())
        .map_err(|error| error.to_string())?;
    let backend =
        WasmTrainBackendV1::new("wasm32-unknown-unknown").map_err(|error| error.to_string())?;
    Ok(run_training_conformance(&backend, &vectors))
}

#[cfg(target_arch = "wasm32")]
fn digest_hex(bytes: [u8; 32]) -> String {
    let mut result = String::with_capacity(64);
    for byte in bytes {
        use core::fmt::Write;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

#[cfg(target_arch = "wasm32")]
fn hash_field(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

#[cfg(target_arch = "wasm32")]
fn portable_training_report_digest() -> Result<String, String> {
    let report = portable_training_conformance_report()?;
    let expected_cases = tritium_spec::TrainingVectorSetV1::parse_json(
        tritium_spec::TrainingVectorSetV1::canonical_json(),
    )
    .map_err(|error| error.to_string())?
    .cases()
    .len();
    if !report.failed.is_empty() || report.passed.len() != expected_cases {
        return Err(format!(
            "portable conformance incomplete: passed={}, failed={}, expected={expected_cases}",
            report.passed.len(),
            report.failed.len(),
        ));
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"tritium.portable-wasm-conformance-report.v1\0");
    hasher.update(&tritium_spec::TrainingOpManifestV1::digest());
    hasher.update(&tritium_spec::TrainingVectorSetV1::digest());
    hasher.update(&(report.passed.len() as u64).to_le_bytes());
    hasher.update(&(report.failed.len() as u64).to_le_bytes());
    for pass in report.passed {
        hash_field(&mut hasher, pass.case_id.as_bytes());
        match pass.receipt {
            Some(receipt) => {
                hasher.update(&[1]);
                hash_field(&mut hasher, receipt.backend_id.as_bytes());
                hash_field(&mut hasher, receipt.backend_build.as_bytes());
                hash_field(
                    &mut hasher,
                    receipt.physical_device.as_deref().unwrap_or("").as_bytes(),
                );
                hasher.update(&receipt.manifest_digest);
                hasher.update(&receipt.vector_digest.unwrap_or([0; 32]));
                hash_field(&mut hasher, receipt.operation.as_bytes());
                hash_field(&mut hasher, format!("{:?}", receipt.execution).as_bytes());
                hash_field(&mut hasher, format!("{:?}", receipt.dtype).as_bytes());
                hasher.update(&receipt.limits.max_rank.to_le_bytes());
                hasher.update(&receipt.limits.max_elements.to_le_bytes());
                hasher.update(&receipt.limits.max_bytes.to_le_bytes());
                hasher.update(&receipt.input_digest);
                hasher.update(&receipt.output_digest);
                hasher.update(&receipt.peak_resident_bytes.to_le_bytes());
                hasher.update(&receipt.scratch_bytes.to_le_bytes());
                hasher.update(&receipt.host_transfers.to_le_bytes());
                hasher.update(&[u8::from(receipt.device_resident)]);
            }
            None => {
                hasher.update(&[0]);
            }
        }
    }
    for failure in report.failed {
        hash_field(&mut hasher, failure.case_id.as_bytes());
        hash_field(&mut hasher, format!("{:?}", failure.reason).as_bytes());
    }
    Ok(digest_hex(*hasher.finalize().as_bytes()))
}

/// Execute the complete canonical portable-training corpus inside the guest.
///
/// Zero means all cases passed. A nonzero value is one plus the number of
/// failed cases, capped to `u32::MAX`; `u32::MAX` also represents setup failure.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn tritium_portable_conformance_status() -> u32 {
    match portable_training_conformance_report() {
        Ok(report)
            if report.failed.is_empty()
                && report.passed.len() == tritium_portable_conformance_case_count() as usize =>
        {
            0
        }
        Ok(report) => u32::try_from(report.failed.len())
            .unwrap_or(u32::MAX - 1)
            .saturating_add(1),
        Err(_) => u32::MAX,
    }
}

/// Number of canonical cases the bundled guest must execute.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn tritium_portable_conformance_case_count() -> u32 {
    use tritium_spec::TrainingVectorSetV1;

    TrainingVectorSetV1::parse_json(TrainingVectorSetV1::canonical_json()).map_or(0, |vectors| {
        u32::try_from(vectors.cases().len()).unwrap_or(0)
    })
}

/// Number of frozen manifest operations advertised by the guest.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn tritium_portable_operation_count() -> u32 {
    u32::try_from(tritium_spec::TrainingOpManifestV1::operations().len()).unwrap_or(0)
}

/// Combined caller-buffer ceiling enforced by the portable guest.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn tritium_portable_max_caller_bytes() -> u32 {
    u32::try_from(WasmTrainBackendV1::max_caller_bytes()).unwrap_or(u32::MAX)
}

/// Linker-enforced maximum WebAssembly linear memory in bytes.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn tritium_portable_max_linear_memory_bytes() -> u32 {
    192 * 1024 * 1024
}

/// Source-bound build identity embedded in portable-training receipts.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn tritium_portable_build_id() -> String {
    format!(
        "{}@{}+{}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("TRITIUM_SOURCE_ID")
    )
}

/// Guest-embedded digest of the exact frozen operation manifest.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn tritium_portable_manifest_digest() -> String {
    digest_hex(tritium_spec::TrainingOpManifestV1::digest())
}

/// Guest-embedded digest of the exact canonical semantic-vector corpus.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn tritium_portable_vector_digest() -> String {
    digest_hex(tritium_spec::TrainingVectorSetV1::digest())
}

/// Digest of ordered case identities and every normalized execution receipt.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn tritium_portable_report_digest() -> String {
    portable_training_report_digest().unwrap_or_default()
}

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

    fn mpgemm(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
        let MpGemm {
            act,
            weights,
            scales,
            shape,
            format,
            out,
        } = p;
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
    use tritium_spec::{TrainBackendV1, TrainingVectorSetV1};
    use tritium_testkit::{
        FROZEN_COUNT, FROZEN_SEED, Tolerance, generate_vectors, run_conformance,
        run_fused_fallback_contract, run_training_conformance,
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

    #[test]
    fn direct_salt_admission_is_byte_identical_and_rejects_corruption() {
        use half::f16;
        use tritium_format::salt_v2::SaltV2Codec;
        use tritium_format::salt_v2_package::{
            SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile,
        };

        let plane =
            SaltV2Plane::new(vec![-1, 0, 1], vec![f16::from_f32(0.5)]).expect("valid plane");
        let tile = SaltV2Tile::new(vec![plane]).expect("valid tile");
        let tensor = SaltV2Tensor::new("weight", vec![1, 3], vec![tile]).expect("valid tensor");
        let package = SaltV2Package::new(SaltV2Codec::B3, vec![tensor]).expect("valid package");
        let bytes = write_salt_v2_package(&package)
            .expect("encode package")
            .bytes;
        assert_eq!(admit_canonical_salt_v2_package(&bytes).unwrap(), bytes);

        let mut corrupt = bytes;
        corrupt[0] ^= u8::MAX;
        assert!(admit_canonical_salt_v2_package(&corrupt).is_err());
    }

    #[test]
    fn portable_training_manifest_is_complete() {
        let vectors = TrainingVectorSetV1::parse_json(include_bytes!(
            "../../../spec/training/v1/vectors/v1.json"
        ))
        .expect("parse canonical training vectors");
        let physical_device = if cfg!(target_arch = "wasm32") {
            option_env!("TRITIUM_WASM_PHYSICAL_DEVICE").unwrap_or("wasmtime:unversioned")
        } else {
            "wasm32:structural-host"
        };
        let backend = WasmTrainBackendV1::new(physical_device).expect("valid device identity");
        let report = run_training_conformance(&backend, &vectors);
        assert!(
            report.is_ok(),
            "{} WASM portable-training failures: {:?}",
            report.failed.len(),
            report.failed
        );
        assert_eq!(report.passed.len(), vectors.cases().len());
        assert_eq!(backend.capabilities().supported_operations.len(), 35);
        assert!(
            report
                .passed
                .iter()
                .filter_map(|case| case.receipt.as_ref())
                .all(|receipt| receipt.physical_device.as_deref() == Some(physical_device))
        );
    }
}

//! GPU host side for the CUDA backend. Compiled only with `--features cuda`.
//!
//! This module owns a [`cudarc`] device handle, loads the PTX emitted by
//! `build.rs`, and drives the addition-only TQ2_0 mpGEMM kernel. It maps every
//! `cudarc` driver error to a [`BackendError`] so the backend never panics on a
//! device failure, and reports allocation failures as
//! [`BackendError::OutOfMemory`].
//!
//! The crate-level `#![deny(unsafe_code)]` stands; the only `unsafe` here is the
//! kernel launch (the driver's `launch` is an `unsafe fn`), behind a narrowly
//! scoped `#[allow(unsafe_code)]` with a `SAFETY:` justification — exactly the
//! pattern `tritium-runtime` uses for its `distributed_slice` statics.

use core::any::Any;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

use tritium_core::{GemmShape, TernaryFormat};
use tritium_format::{TQ2_0_BLOCK_BYTES, num_blocks};
use tritium_runtime::BackendEntry;
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, TernaryBackend};

/// Module name the PTX is registered under in the device.
const MODULE_NAME: &str = "tq2_0_add";
/// Kernel entry point — must match the `extern "C"` symbol in the `.cu` file.
const KERNEL_NAME: &str = "tq2_0_add_mpgemm";
/// CUDA threads per block for the 1-D launch grid.
const THREADS_PER_BLOCK: u32 = 256;

/// The PTX produced by `build.rs` (`nvcc -ptx`). Embedded at compile time so the
/// backend needs no PTX file on disk at runtime.
const TQ2_0_ADD_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/tq2_0_add.ptx"));

/// Map a `cudarc` driver error to a [`BackendError`]. Allocation failures surface
/// as [`BackendError::OutOfMemory`]; everything else is stringified into
/// [`BackendError::Backend`] so the device error text survives.
fn driver_err(context: &str, err: &DriverError) -> BackendError {
    BackendError::Backend(format!("{context}: {err}"))
}

/// Device-resident packed TQ2_0 weights for one matmul operand.
///
/// Wraps a [`CudaSlice<u8>`] (the htod copy of the host-packed bytes) plus the
/// `[N, K]` geometry and the per-row packed byte stride, so `mpgemm` can validate
/// and launch without re-deriving them.
#[derive(Debug)]
pub struct CudaBuffer {
    /// Device allocation holding the packed TQ2_0 bytes, `[N * row_bytes]`.
    device: CudaSlice<u8>,
    /// Output channels (`N`).
    n: usize,
    /// Contraction dimension (`K`).
    k: usize,
    /// Packed bytes per weight row (`num_blocks(k) * TQ2_0_BLOCK_BYTES`).
    row_bytes: usize,
    /// Total bytes uploaded (`device.len()`), cached for [`DeviceBuffer::len_bytes`].
    bytes: usize,
}

impl DeviceBuffer for CudaBuffer {
    fn len_bytes(&self) -> usize {
        self.bytes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A CUDA execution backend bound to a single device ordinal.
///
/// Construct with [`CudaBackend::new`]; it opens the device, loads the PTX module,
/// and caches a friendly `device_id` like `"cuda:0"`. Cheap to clone the handle —
/// the underlying [`CudaDevice`] is reference-counted by `cudarc`.
#[derive(Debug)]
pub struct CudaBackend {
    /// Reference-counted device + loaded module handle.
    device: Arc<CudaDevice>,
    /// Backend identifier, e.g. `"cuda:0"`.
    device_id: String,
    /// Human-readable device name reported by the driver, e.g. `"NVIDIA H100"`.
    device_name: String,
}

impl CudaBackend {
    /// Open CUDA device `ordinal`, load the TQ2_0 add kernel, and return a backend.
    ///
    /// # Errors
    /// [`BackendError::Backend`] if the device cannot be opened or the PTX module
    /// fails to load (no driver, no GPU, malformed PTX, …).
    pub fn new(ordinal: usize) -> Result<Self, BackendError> {
        let device = CudaDevice::new(ordinal).map_err(|e| driver_err("open cuda device", &e))?;

        device
            .load_ptx(Ptx::from_src(TQ2_0_ADD_PTX), MODULE_NAME, &[KERNEL_NAME])
            .map_err(|e| driver_err("load tq2_0_add ptx", &e))?;

        let device_name = device
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".to_owned());

        Ok(Self {
            device,
            device_id: format!("cuda:{ordinal}"),
            device_name,
        })
    }

    /// Packed bytes per weight row for `k` trits in TQ2_0.
    fn row_bytes(k: usize) -> usize {
        num_blocks(k) * TQ2_0_BLOCK_BYTES
    }
}

impl TernaryBackend for CudaBackend {
    fn device_id(&self) -> &str {
        &self.device_id
    }

    fn capabilities(&self) -> DeviceCaps {
        // total_memory is left at 0 (unknown): the cudarc safe API does not expose
        // a portable free/total query here, and the contract permits 0.
        DeviceCaps::new("cuda", self.device_name.clone()).with_features(vec!["tq2_0".to_owned()])
    }

    fn upload_weights(
        &self,
        packed: &[u8],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        if format != TernaryFormat::Tq2_0 {
            return Err(BackendError::UnsupportedFormat(format));
        }
        let GemmShape { n, k, .. } = shape;
        let row_bytes = Self::row_bytes(k);
        let expected = n * row_bytes;
        if packed.len() != expected {
            return Err(BackendError::InvalidInput(format!(
                "packed len {} != expected {expected} for shape {shape:?} (tq2_0)",
                packed.len()
            )));
        }

        // htod copy of the packed bytes. A driver OOM here is reported as such.
        let device = self.device.htod_sync_copy(packed).map_err(|e| {
            if is_oom(&e) {
                BackendError::OutOfMemory {
                    requested: expected,
                }
            } else {
                driver_err("upload weights (htod)", &e)
            }
        })?;

        Ok(Box::new(CudaBuffer {
            device,
            n,
            k,
            row_bytes,
            bytes: packed.len(),
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
        if format != TernaryFormat::Tq2_0 {
            return Err(BackendError::UnsupportedFormat(format));
        }
        let buf = weights
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a CudaBuffer".into()))?;

        let GemmShape { m, n, k } = shape;
        if buf.n != n || buf.k != k {
            return Err(BackendError::ShapeMismatch {
                expected: buf.n * buf.k,
                got: n * k,
            });
        }
        // Host-side length checks mirror reference_mpgemm so a mismatch is a typed
        // error, never an out-of-bounds device read.
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
        if m == 0 || n == 0 {
            // Nothing to compute; out is already correctly sized (and empty).
            return Ok(());
        }

        let func = self
            .device
            .get_func(MODULE_NAME, KERNEL_NAME)
            .ok_or_else(|| BackendError::Backend(format!("kernel {KERNEL_NAME} not loaded")))?;

        // Upload activations + scales; allocate the output on device.
        let d_act = self
            .device
            .htod_sync_copy(act)
            .map_err(|e| alloc_or_backend("upload act (htod)", &e, act.len() * 4))?;
        let d_scales = self
            .device
            .htod_sync_copy(scales)
            .map_err(|e| alloc_or_backend("upload scales (htod)", &e, scales.len() * 4))?;
        let mut d_out = self
            .device
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| alloc_or_backend("alloc out", &e, m * n * 4))?;

        let total = (m * n) as u32;
        let grid = total.div_ceil(THREADS_PER_BLOCK);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        // Scalars are passed by value as i32 (matching the kernel signature).
        let params = (
            &d_act,
            &buf.device,
            &d_scales,
            &mut d_out,
            m as i32,
            n as i32,
            k as i32,
            buf.row_bytes as i32,
        );

        // SAFETY: `LaunchAsync::launch` is `unsafe` because the kernel signature
        // is not type-checked against `params` by the compiler. We uphold the
        // contract manually: the param tuple's order and types exactly match the
        // `extern "C"` kernel (`const float*`, `const unsigned char*`,
        // `const float*`, `float*`, then four `int`s), the device buffers were all
        // allocated above (or in `upload_weights`) with sizes validated against
        // `shape`, and the launch grid covers exactly `m*n` threads so no thread
        // indexes past any buffer. No host memory aliased by the kernel is mutated
        // concurrently.
        #[allow(unsafe_code)]
        unsafe {
            func.launch(cfg, params)
                .map_err(|e| driver_err("launch tq2_0_add", &e))?;
        }

        // dtoh copy of the result. `dtoh_sync_copy_into` synchronizes the stream,
        // so the kernel has completed before we read back.
        self.device
            .dtoh_sync_copy_into(&d_out, out)
            .map_err(|e| driver_err("download out (dtoh)", &e))?;

        Ok(())
    }
}

/// Heuristic: did this driver error come from an allocation running out of memory?
fn is_oom(err: &DriverError) -> bool {
    // `DriverError`'s Display includes the CUDA status string; the out-of-memory
    // status renders as "out of memory". This keeps us off the unstable numeric
    // status value while still classifying the common case.
    format!("{err}")
        .to_ascii_lowercase()
        .contains("out of memory")
}

/// Classify an allocation/copy failure as OOM (with the requested byte count) or a
/// generic backend error.
fn alloc_or_backend(context: &str, err: &DriverError, requested: usize) -> BackendError {
    if is_oom(err) {
        BackendError::OutOfMemory { requested }
    } else {
        driver_err(context, err)
    }
}

/// Construct the backend on device 0 for the runtime registry.
///
/// Returns `Err` (which the registry logs and skips) when no CUDA device is
/// available — the expected case on cpu-only machines that still link this crate.
fn init_cuda() -> Result<Box<dyn TernaryBackend>, BackendError> {
    Ok(Box::new(CudaBackend::new(0)?))
}

// Self-register into the runtime's distributed slice, but only with the `cuda`
// feature. `linkme`'s `distributed_slice` expands to a `#[link_section]` static
// that trips the `unsafe_code` lint, hence the scoped allow (same pattern as
// `tritium-runtime`'s own registrations).
#[allow(unsafe_code)]
#[linkme::distributed_slice(tritium_runtime::BACKENDS)]
static CUDA: BackendEntry = BackendEntry {
    name: "cuda",
    init: init_cuda,
};

#[cfg(test)]
mod tests {
    //! GPU conformance test. Runs only with `--features cuda` AND a working CUDA
    //! device, so it is exercised on the Wave D GPU CI lane, never on cpu-only
    //! lanes. When no device is present the test self-skips (constructing the
    //! backend returns `Err`) rather than failing.
    //!
    //! `run_conformance` itself packs each vector's trits to TQ2_0 (block scale
    //! 1.0), uploads via `upload_weights`, runs `mpgemm` with the per-channel
    //! scales, and grades against `reference_mpgemm` — so the test only has to
    //! supply the TQ2_0 vectors this kernel supports.

    use super::*;
    use tritium_testkit::{Tolerance, generate_vectors, run_conformance};

    #[test]
    fn cuda_matches_reference_within_tolerance() {
        // Skip cleanly when no GPU is present (cpu-only dev box / wrong CI lane).
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping cuda conformance: no device ({e})");
                return;
            }
        };

        // This kernel only handles TQ2_0; the generator alternates formats.
        let tq2: Vec<_> = generate_vectors(0xC0FFEE, 16)
            .into_iter()
            .filter(|v| v.format == "tq2_0")
            .collect();
        assert!(!tq2.is_empty(), "expected some tq2_0 conformance vectors");

        let report = run_conformance(&backend, &tq2, Tolerance::default());
        assert!(
            report.is_ok(),
            "{} cuda conformance cases failed: {:?}",
            report.failed.len(),
            report.failed
        );
    }

    #[test]
    fn rejects_tq1_0_format() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => return, // no device: nothing to assert about format handling
        };
        let shape = GemmShape { m: 1, n: 1, k: 256 };
        let err = backend
            .upload_weights(&[0u8; 54], shape, TernaryFormat::Tq1_0)
            .unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedFormat(_)));
    }
}

//! GPU host side for the CUDA backend. Compiled only with `--features cuda`.
//!
//! This module owns a [`cudarc`] context + default stream, loads the PTX emitted
//! by `build.rs`, and drives the addition-only TQ2_0 mpGEMM kernel. It maps every
//! `cudarc` driver error to a [`BackendError`] so the backend never panics on a
//! device failure, and reports allocation failures as
//! [`BackendError::OutOfMemory`].
//!
//! ## cudarc 0.19 API
//!
//! Ported from the 0.13 device API to the 0.19 context/stream API:
//! - [`cudarc::driver::CudaContext::new`] returns an `Arc<CudaContext>`; memory
//!   and launches go through its [`default_stream`](cudarc::driver::CudaContext::default_stream).
//! - PTX is loaded with [`CudaContext::load_module`] (taking a
//!   [`cudarc::nvrtc::Ptx`] built from our pre-compiled string) and the kernel is
//!   fetched with [`CudaModule::load_function`].
//! - Host↔device copies use the stream's `clone_htod` / `clone_dtoh` /
//!   `memcpy_dtoh`, and launches use the `launch_builder(...).arg(...).launch(cfg)`
//!   builder. cudarc's `fallback-dynamic-loading` feature dlopen's `libcuda` at
//!   runtime, so there is no build-time CUDA-toolkit-version pin (which is what
//!   lets this crate build against CUDA 13.3).
//!
//! The crate-level `#![deny(unsafe_code)]` stands; the only `unsafe` here is the
//! kernel launch (`launch_builder(...).launch` is an `unsafe fn`), behind a
//! narrowly scoped `#[allow(unsafe_code)]` with a `SAFETY:` justification — exactly
//! the pattern `tritium-runtime` uses for its `distributed_slice` statics.

use core::any::Any;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use tritium_core::{GemmShape, TernaryFormat};
use tritium_format::{TQ2_0_BLOCK_BYTES, num_blocks};
use tritium_runtime::BackendEntry;
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, TernaryBackend};

/// Kernel entry point — must match the `extern "C"` symbol in the `.cu` file.
/// (cudarc 0.19 keys modules by the returned [`CudaModule`] handle, not by a
/// registered module name, so only the function symbol is needed.)
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
///
/// Internal to the crate: it crosses the [`TernaryBackend`] boundary only as a
/// `Box<dyn DeviceBuffer>`, downcast back here via [`core::any::Any`].
#[derive(Debug)]
pub(crate) struct CudaBuffer {
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
/// Construct with [`CudaBackend::new`]; it opens the context, loads the PTX module,
/// resolves the kernel, and caches a friendly `device_id` like `"cuda:0"`. The
/// underlying [`CudaContext`], [`CudaStream`], and [`CudaModule`] are all
/// reference-counted (`Arc`) by `cudarc`.
#[derive(Debug)]
pub struct CudaBackend {
    /// The context's default stream — all memory ops and launches go through it.
    /// The stream holds its own `Arc<CudaContext>`, so the context stays alive for
    /// as long as the backend does without a separate field.
    stream: Arc<CudaStream>,
    /// Loaded PTX module (kept alive so `func` stays valid).
    _module: Arc<CudaModule>,
    /// The resolved `tq2_0_add_mpgemm` kernel.
    func: CudaFunction,
    /// Backend identifier, e.g. `"cuda:0"`.
    device_id: String,
    /// Human-readable device name reported by the driver, e.g. `"NVIDIA H100"`.
    device_name: String,
}

impl CudaBackend {
    /// Open CUDA device `ordinal`, load the TQ2_0 add kernel, and return a backend.
    ///
    /// # Errors
    /// [`BackendError::Backend`] if the device cannot be opened, the PTX module
    /// fails to load, or the kernel symbol is missing (no driver, no GPU, malformed
    /// PTX, …).
    pub fn new(ordinal: usize) -> Result<Self, BackendError> {
        let ctx = CudaContext::new(ordinal).map_err(|e| driver_err("open cuda device", &e))?;
        let stream = ctx.default_stream();

        let module = ctx
            .load_module(Ptx::from_src(TQ2_0_ADD_PTX))
            .map_err(|e| driver_err("load tq2_0_add ptx", &e))?;
        let func = module
            .load_function(KERNEL_NAME)
            .map_err(|e| driver_err("resolve tq2_0_add kernel", &e))?;

        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".to_owned());

        Ok(Self {
            stream,
            _module: module,
            func,
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
        // total_memory is left at its default (unknown): the contract permits 0 and
        // the runtime does not rely on the figure here.
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
        let device = self.stream.clone_htod(packed).map_err(|e| {
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

        // Upload activations + scales; allocate the output on device.
        let d_act = self
            .stream
            .clone_htod(act)
            .map_err(|e| alloc_or_backend("upload act (htod)", &e, act.len() * 4))?;
        let d_scales = self
            .stream
            .clone_htod(scales)
            .map_err(|e| alloc_or_backend("upload scales (htod)", &e, scales.len() * 4))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| alloc_or_backend("alloc out", &e, m * n * 4))?;

        let total = (m * n) as u32;
        let grid = total.div_ceil(THREADS_PER_BLOCK);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        // Scalars are passed by value as i32 (matching the kernel signature). They
        // are bound to locals so they outlive the launch builder's borrows.
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let row_bytes_i = buf.row_bytes as i32;

        let mut launch = self.stream.launch_builder(&self.func);
        launch
            .arg(&d_act)
            .arg(&buf.device)
            .arg(&d_scales)
            .arg(&mut d_out)
            .arg(&m_i)
            .arg(&n_i)
            .arg(&k_i)
            .arg(&row_bytes_i);

        // SAFETY: `LaunchArgs::launch` is `unsafe` because the kernel signature is
        // not type-checked against the pushed args by the compiler. We uphold the
        // contract manually: the args were pushed in exactly the order and types of
        // the `extern "C"` kernel (`const float*`, `const unsigned char*`,
        // `const float*`, `float*`, then four `int`s), the device buffers were all
        // allocated above (or in `upload_weights`) with sizes validated against
        // `shape`, only `d_out` is passed mutably (matching the kernel's single
        // `float* out` output), and the launch grid covers exactly `m*n` threads so
        // no thread indexes past any buffer. All host scalars outlive this call.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch tq2_0_add", &e))?;
        }

        // dtoh copy of the result into the caller's buffer. `memcpy_dtoh` on the
        // default stream is ordered after the launch and synchronizes the stream,
        // so the kernel has completed before the bytes land in `out`.
        self.stream
            .memcpy_dtoh(&d_out, out)
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
    //! GPU conformance + CPU↔CUDA parity tests. Run only with `--features cuda` AND
    //! a working CUDA device, so they are exercised on the Wave D GPU CI lane, never
    //! on cpu-only lanes. When no device is present the tests self-skip
    //! (constructing the backend returns `Err`) rather than failing.
    //!
    //! `run_conformance` itself packs each vector's trits to TQ2_0 (block scale
    //! 1.0), uploads via `upload_weights`, runs `mpgemm` with the per-channel
    //! scales, and grades against `reference_mpgemm` — so the test only has to
    //! supply the TQ2_0 vectors this kernel supports.

    use super::*;
    use tritium_cpu::CpuBackend;
    use tritium_testkit::{ConformanceVector, Tolerance, generate_vectors, run_conformance};

    /// The full conformance set this kernel is responsible for: every TQ2_0 vector
    /// from the committed generator (the kernel does not handle TQ1_0).
    fn tq2_vectors() -> Vec<ConformanceVector> {
        let v: Vec<_> = generate_vectors(0xC0FFEE, 16)
            .into_iter()
            .filter(|v| v.format == "tq2_0")
            .collect();
        assert!(!v.is_empty(), "expected some tq2_0 conformance vectors");
        v
    }

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

        let tq2 = tq2_vectors();
        let report = run_conformance(&backend, &tq2, Tolerance::default());
        assert!(
            report.is_ok(),
            "{} cuda conformance cases failed: {:?}",
            report.failed.len(),
            report.failed
        );
    }

    /// ADR 0002 U2: CPU↔CUDA parity. The *same* committed TQ2_0 vectors run through
    /// both [`CpuBackend`] and [`CudaBackend`]; every output element must agree
    /// within `1e-4` relative. This is the load-bearing cross-backend gate — it
    /// catches a backend that is internally self-consistent (passes conformance)
    /// but disagrees with the other backend on shared inputs.
    #[test]
    fn cuda_matches_cpu_within_tolerance() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping cpu<->cuda parity: no device ({e})");
                return;
            }
        };
        let cpu = CpuBackend::new();
        let tol = Tolerance::default();

        // Run both backends over the identical TQ2_0 vector set.
        let cpu_report = run_conformance(&cpu, &tq2_vectors(), tol);
        assert!(
            cpu_report.is_ok(),
            "cpu backend failed its own conformance, parity is moot: {:?}",
            cpu_report.failed
        );

        // Replay each vector through both backends and compare outputs directly,
        // rather than only against the shared reference, so any CPU/CUDA divergence
        // surfaces even within the reference tolerance band.
        for v in tq2_vectors() {
            let shape = GemmShape::new(v.m, v.n, v.k);
            let trits: Vec<_> = v
                .weights
                .iter()
                .map(|&w| tritium_core::Trit::from_i8(w).expect("vector weight in {-1,0,1}"))
                .collect();
            let packed = pack_tq2_0(&trits, shape);

            let cpu_out = run_backend(&cpu, &packed, &v.activation, &v.scales, shape);
            let cuda_out = run_backend(&cuda, &packed, &v.activation, &v.scales, shape);

            assert_eq!(
                cpu_out.len(),
                cuda_out.len(),
                "{}: output len mismatch",
                v.id
            );
            for (i, (&c, &g)) in cpu_out.iter().zip(&cuda_out).enumerate() {
                assert!(
                    tol.accepts(g, c),
                    "{}: cpu/cuda disagree at [{i}]: cpu={c} cuda={g}",
                    v.id
                );
            }
        }
    }

    /// Pack an `[N, K]` trit matrix to TQ2_0 rows, block scale fixed to `1.0` (the
    /// testkit convention), ready for `upload_weights`.
    fn pack_tq2_0(trits: &[tritium_core::Trit], shape: GemmShape) -> Vec<u8> {
        use tritium_format::pack_tq2_0_row;
        let GemmShape { n, k, .. } = shape;
        let nb = num_blocks(k);
        let unit = vec![half::f16::ONE; nb];
        let row_bytes = nb * TQ2_0_BLOCK_BYTES;
        let mut packed = vec![0u8; n * row_bytes];
        for ni in 0..n {
            let row = &trits[ni * k..ni * k + k];
            let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
            pack_tq2_0_row(row, &unit, out).expect("pack tq2_0 row");
        }
        packed
    }

    /// Upload weights + run one TQ2_0 mpGEMM through any backend, returning `[M, N]`.
    fn run_backend<B: TernaryBackend>(
        backend: &B,
        packed: &[u8],
        act: &[f32],
        scales: &[f32],
        shape: GemmShape,
    ) -> Vec<f32> {
        let buf = backend
            .upload_weights(packed, shape, TernaryFormat::Tq2_0)
            .expect("upload weights");
        let mut out = vec![0.0f32; shape.m * shape.n];
        backend
            .mpgemm(
                act,
                buf.as_ref(),
                scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut out,
            )
            .expect("mpgemm");
        out
    }

    #[test]
    fn rejects_tq1_0_format() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => return, // no device: nothing to assert about format handling
        };
        let shape = GemmShape { m: 1, n: 1, k: 256 };
        // The format gate runs before any length check, so the bytes need not be a
        // valid TQ1_0 length. `Box<dyn DeviceBuffer>` is not `Debug`, so `unwrap_err`
        // is unavailable — match on the result instead (same idiom as tritium-cpu).
        match backend.upload_weights(&[0u8; 66], shape, TernaryFormat::Tq1_0) {
            Err(BackendError::UnsupportedFormat(_)) => {}
            other => panic!(
                "expected UnsupportedFormat, got {:?}",
                other.map(|_| "ok-buffer")
            ),
        }
    }
}

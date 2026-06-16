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
use tritium_format::{
    IMMA_K, IMMA_N, IMMA_WTILE_BYTES, TQ2_0_BLOCK_BYTES, num_blocks,
};
use tritium_runtime::BackendEntry;
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, TernaryBackend};

/// Kernel entry point — must match the `extern "C"` symbol in the `.cu` file.
/// (cudarc 0.19 keys modules by the returned [`CudaModule`] handle, not by a
/// registered module name, so only the function symbol is needed.)
const KERNEL_NAME: &str = "tq2_0_add_mpgemm";
/// The decode-oriented tiled add-only kernel (v0.30 WF-A): one warp per output,
/// one block per row with the activation row staged in shared memory.
const KERNEL_NAME_TILED: &str = "tq2_0_add_mpgemm_tiled";
/// The IMMA int8 tensor-core prefill kernel (v0.30 WF-A part 2): one warp per
/// 16×8 output tile, `mma.m16n8k32` int32 accumulate, ternary weights in the
/// [`TernaryFormat::I2sInt8`] tile interleave.
const KERNEL_NAME_IMMA: &str = "tq2_0_imma_mpgemm";
/// On-device per-token int8 absmax activation quant (W1.58A8), the first step of
/// the fused `mpgemm_with_act_quant` override.
const KERNEL_NAME_ACT_QUANT: &str = "act_quant_int8_per_token";
/// Threads per block for `act_quant_int8_per_token` — must match the kernel's
/// `ACT_QUANT_THREADS` (its shared reduction is sized for this, a power of two).
const ACT_QUANT_THREADS: u32 = 256;
/// CUDA threads per block for the 1-D launch grid (simple kernel).
const THREADS_PER_BLOCK: u32 = 256;
/// Warps per block for the tiled kernel — each warp computes one output column,
/// so a block covers this many `N` at once (8 warps = 256 threads).
const WARPS_PER_BLOCK: u32 = 8;
/// Largest `K` the tiled kernel accepts: it stages `K` f32 activations in shared
/// memory (`K * 4` bytes = 32 KiB at the cap), comfortably under the 48 KiB
/// default dynamic-shared budget and covering every BitNet shape (max K = 6912).
const TILED_K_MAX: usize = 8_192;
/// Largest `M` routed to the tiled (decode) kernel. Above this the problem is
/// prefill-shaped and the one-thread-per-output kernel is the better default
/// until the IMMA tensor-core kernel lands (WF-A part 2).
const TILED_M_MAX: usize = 64;

/// The PTX produced by `build.rs` (`nvcc -ptx`). Embedded at compile time so the
/// backend needs no PTX file on disk at runtime.
const TQ2_0_ADD_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/tq2_0_add.ptx"));
/// The IMMA prefill kernel + the on-device act-quant kernel, compiled by `build.rs`
/// to a SECOND PTX target at `compute_80` (the `mma.m16n8k32` int8 shape needs
/// sm_80+, above the add kernel's sm_75 floor). Embedded the same way.
const TQ2_0_IMMA_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/tq2_0_imma.ptx"));

/// Map a `cudarc` driver error to a [`BackendError`]. Allocation failures surface
/// as [`BackendError::OutOfMemory`]; everything else is stringified into
/// [`BackendError::Backend`] so the device error text survives.
fn driver_err(context: &str, err: &DriverError) -> BackendError {
    BackendError::Backend(format!("{context}: {err}"))
}

/// Device-resident packed ternary weights for one matmul operand.
///
/// Wraps a [`CudaSlice<u8>`] (the htod copy of the host-packed bytes) plus the
/// `[N, K]` geometry, the packing [`TernaryFormat`], and a format-specific stride
/// ([`Stride`]), so `mpgemm` / `mpgemm_with_act_quant` can validate and launch
/// without re-deriving them.
///
/// Internal to the crate: it crosses the [`TernaryBackend`] boundary only as a
/// `Box<dyn DeviceBuffer>`, downcast back here via [`core::any::Any`].
#[derive(Debug)]
pub(crate) struct CudaBuffer {
    /// Device allocation holding the packed bytes (TQ2_0 rows or the I2sInt8 tiles).
    device: CudaSlice<u8>,
    /// Output channels (`N`), unpadded.
    n: usize,
    /// Contraction dimension (`K`), unpadded.
    k: usize,
    /// The packing this buffer holds — gates which kernel may consume it.
    format: TernaryFormat,
    /// Format-specific addressing stride (TQ2_0 per-row bytes vs IMMA k-tile count).
    stride: Stride,
    /// Total bytes uploaded (`device.len()`), cached for [`DeviceBuffer::len_bytes`].
    bytes: usize,
}

/// Format-specific addressing metadata cached alongside a [`CudaBuffer`].
#[derive(Debug, Clone, Copy)]
enum Stride {
    /// TQ2_0: packed bytes per weight row (`num_blocks(k) * TQ2_0_BLOCK_BYTES`).
    Tq2_0 { row_bytes: usize },
    /// I2sInt8: the packed K-tile count (`ceil(k / IMMA_K)`), the kernel's
    /// `num_ktiles` launch argument and the per-n-tile k-tile stride.
    I2sInt8 { num_ktiles: usize },
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
    /// Loaded add-only PTX module (kept alive so `func`/`func_tiled` stay valid).
    _module: Arc<CudaModule>,
    /// Loaded IMMA PTX module (kept alive so `func_imma`/`func_act_quant` stay
    /// valid). A separate `compute_80` image, distinct from `_module`'s `compute_75`.
    _imma_module: Arc<CudaModule>,
    /// The resolved `tq2_0_add_mpgemm` kernel (one thread per output).
    func: CudaFunction,
    /// The resolved `tq2_0_add_mpgemm_tiled` kernel (warp per output, shared-mem
    /// staged activations) — the decode path.
    func_tiled: CudaFunction,
    /// The resolved `tq2_0_imma_mpgemm` kernel (IMMA int8 tensor-core prefill).
    func_imma: CudaFunction,
    /// The resolved `act_quant_int8_per_token` kernel (on-device W1.58A8 quant).
    func_act_quant: CudaFunction,
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
        let func_tiled = module
            .load_function(KERNEL_NAME_TILED)
            .map_err(|e| driver_err("resolve tq2_0_add_tiled kernel", &e))?;

        // The IMMA kernels live in their own compute_80 PTX (the add kernel's
        // compute_75 PTX cannot assemble `mma.m16n8k32`). Loading it JITs to the
        // present device, which the GPU lane guarantees is sm_80+ (the 4090 is
        // sm_89). A pre-sm_80 device would fail here, which is the correct error.
        let imma_module = ctx
            .load_module(Ptx::from_src(TQ2_0_IMMA_PTX))
            .map_err(|e| driver_err("load tq2_0_imma ptx", &e))?;
        let func_imma = imma_module
            .load_function(KERNEL_NAME_IMMA)
            .map_err(|e| driver_err("resolve tq2_0_imma kernel", &e))?;
        let func_act_quant = imma_module
            .load_function(KERNEL_NAME_ACT_QUANT)
            .map_err(|e| driver_err("resolve act_quant kernel", &e))?;

        let device_name = ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".to_owned());

        Ok(Self {
            stream,
            _module: module,
            _imma_module: imma_module,
            func,
            func_tiled,
            func_imma,
            func_act_quant,
            device_id: format!("cuda:{ordinal}"),
            device_name,
        })
    }

    /// Packed bytes per weight row for `k` trits in TQ2_0.
    fn row_bytes(k: usize) -> usize {
        num_blocks(k) * TQ2_0_BLOCK_BYTES
    }

    /// Pick the add-only kernel for this problem shape. The tiled (decode) kernel
    /// wins for small `M` and is bounded by its shared-memory activation stage
    /// (`K * 4` ≤ 48 KiB); everything else uses the one-thread-per-output kernel.
    fn select_add_kernel(m: usize, k: usize) -> AddKernel {
        if m > 0 && m <= TILED_M_MAX && k <= TILED_K_MAX {
            AddKernel::Tiled
        } else {
            AddKernel::Simple
        }
    }

    /// Run one TQ2_0 add-only mpGEMM through the chosen kernel. Shared by the
    /// public [`TernaryBackend::mpgemm`] (which auto-selects) and the tests (which
    /// force each kernel so both stay gated against the reference).
    ///
    /// # Errors
    /// Validation [`BackendError::ShapeMismatch`] / [`BackendError::UnsupportedFormat`]
    /// as documented on [`TernaryBackend::mpgemm`]; device failures via the cudarc
    /// error mapping.
    fn mpgemm_kernel(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        scales: &[f32],
        shape: GemmShape,
        format: TernaryFormat,
        out: &mut [f32],
        kernel: AddKernel,
    ) -> Result<(), BackendError> {
        if format != TernaryFormat::Tq2_0 {
            return Err(BackendError::UnsupportedFormat(format));
        }
        let buf = weights
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a CudaBuffer".into()))?;
        // The add path consumes TQ2_0 rows; an I2sInt8 buffer would mis-address.
        let Stride::Tq2_0 { row_bytes } = buf.stride else {
            return Err(BackendError::UnsupportedFormat(buf.format));
        };

        let GemmShape { m, n, k } = shape;
        if buf.n != n || buf.k != k {
            return Err(BackendError::ShapeMismatch {
                expected: buf.n * buf.k,
                got: n * k,
            });
        }
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
            return Ok(());
        }

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

        // Kernel-specific launch geometry. Both kernels take the identical argument
        // list, so only the function handle and the grid/shared config differ.
        let (func, cfg) = match kernel {
            AddKernel::Simple => {
                let total = (m * n) as u32;
                let grid = total.div_ceil(THREADS_PER_BLOCK);
                (
                    &self.func,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (THREADS_PER_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    },
                )
            }
            AddKernel::Tiled => {
                // `select_add_kernel` only routes K within the shared budget here;
                // assert it for direct callers (the tests) so an oversized K fails
                // loudly rather than as a cryptic CUDA shared-mem launch error.
                debug_assert!(
                    k <= TILED_K_MAX,
                    "tiled kernel K={k} exceeds the {TILED_K_MAX} shared-mem cap"
                );
                // One warp per output column → a block covers WARPS_PER_BLOCK of N;
                // one block-row per M. Shared memory stages this row's K acts.
                let grid_n = (n as u32).div_ceil(WARPS_PER_BLOCK);
                (
                    &self.func_tiled,
                    LaunchConfig {
                        grid_dim: (grid_n, m as u32, 1),
                        block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
                        shared_mem_bytes: (k * 4) as u32,
                    },
                )
            }
        };

        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let row_bytes_i = row_bytes as i32;

        let mut launch = self.stream.launch_builder(func);
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
        // not type-checked against the pushed args. Both `tq2_0_add_mpgemm` and
        // `tq2_0_add_mpgemm_tiled` declare the identical parameter list
        // (`const float*`, `const unsigned char*`, `const float*`, `float*`, then
        // four `int`s), pushed here in that exact order/type. Only `d_out` is
        // mutable (the single `float* out`). Device buffers were sized against
        // `shape` above / in `upload_weights`; the tiled grid covers `M` rows ×
        // `ceil(N / WARPS_PER_BLOCK)` warp-columns with bounds checks (`mi >= m`,
        // `ni >= n`) inside the kernel, and the shared request is `K * 4` bytes,
        // matching the kernel's `extern __shared__ float[K]`. All host scalars
        // outlive the launch.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|e| driver_err("launch tq2_0_add", &e))?;
        }

        self.stream
            .memcpy_dtoh(&d_out, out)
            .map_err(|e| driver_err("download out (dtoh)", &e))?;

        Ok(())
    }

    /// The fused W1.58A8 path on the IMMA int8 tensor cores: quantize `act` to
    /// per-token int8 **on device**, contract against the [`TernaryFormat::I2sInt8`]
    /// weights with `mma.m16n8k32`, and fold both the per-token activation scale and
    /// the per-channel `weight_scales` into the `f32` output — all without an extra
    /// host pass or an H2D round-trip of the quantized activations.
    ///
    /// Validation mirrors the spec default's: `act`/`out`/`weight_scales` lengths vs
    /// `shape`, the buffer's `[N, K]` geometry, and that the buffer is an I2sInt8
    /// packing. Empty `M`/`N` is a no-op.
    ///
    /// # Errors
    /// [`BackendError::ShapeMismatch`] / [`BackendError::UnsupportedFormat`] on bad
    /// inputs; device failures via the cudarc error mapping.
    fn imma_with_act_quant(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        weight_scales: &[f32],
        shape: GemmShape,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let buf = weights
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a CudaBuffer".into()))?;
        let Stride::I2sInt8 { num_ktiles } = buf.stride else {
            return Err(BackendError::UnsupportedFormat(buf.format));
        };

        let GemmShape { m, n, k } = shape;
        if buf.n != n || buf.k != k {
            return Err(BackendError::ShapeMismatch {
                expected: buf.n * buf.k,
                got: n * k,
            });
        }
        if act.len() != m * k {
            return Err(BackendError::ShapeMismatch {
                expected: m * k,
                got: act.len(),
            });
        }
        if weight_scales.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: weight_scales.len(),
            });
        }
        if out.len() != m * n {
            return Err(BackendError::ShapeMismatch {
                expected: m * n,
                got: out.len(),
            });
        }
        if m == 0 || n == 0 {
            return Ok(());
        }

        // --- Upload activations; alloc the on-device int8 quant + per-token scale. ---
        let d_act = self
            .stream
            .clone_htod(act)
            .map_err(|e| alloc_or_backend("upload act (htod)", &e, act.len() * 4))?;
        let mut d_qact = self
            .stream
            .alloc_zeros::<i8>(m * k)
            .map_err(|e| alloc_or_backend("alloc qact", &e, m * k))?;
        let mut d_act_scale = self
            .stream
            .alloc_zeros::<f32>(m)
            .map_err(|e| alloc_or_backend("alloc act_scale", &e, m * 4))?;
        let d_wscale = self
            .stream
            .clone_htod(weight_scales)
            .map_err(|e| alloc_or_backend("upload weight_scales (htod)", &e, n * 4))?;
        let mut d_out = self
            .stream
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| alloc_or_backend("alloc out", &e, m * n * 4))?;

        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let num_ktiles_i = num_ktiles as i32;

        // --- Step 1: per-token int8 absmax quant (one block per row). ---
        {
            let cfg = LaunchConfig {
                grid_dim: (m as u32, 1, 1),
                block_dim: (ACT_QUANT_THREADS, 1, 1),
                shared_mem_bytes: 0, // the kernel's reduction buffer is static __shared__
            };
            let mut launch = self.stream.launch_builder(&self.func_act_quant);
            launch
                .arg(&d_act)
                .arg(&mut d_qact)
                .arg(&mut d_act_scale)
                .arg(&m_i)
                .arg(&k_i);
            // SAFETY: `launch` is `unsafe` — args are not type-checked against the
            // kernel. `act_quant_int8_per_token(const float*, signed char*, float*,
            // int, int)` is fed exactly: `d_act` (f32 [M,K], read), `d_qact` (i8
            // [M,K], written), `d_act_scale` (f32 [M], written), then `m`, `k`. The
            // grid is `M` blocks of `ACT_QUANT_THREADS` (matching the kernel's
            // `ACT_QUANT_THREADS` static shared reduction). All buffers were sized
            // against `shape`; host scalars outlive the launch.
            #[allow(unsafe_code)]
            unsafe {
                launch
                    .launch(cfg)
                    .map_err(|e| driver_err("launch act_quant", &e))?;
            }
        }

        // --- Step 2: IMMA int8 contraction, folding both scales into `out`. ---
        {
            let grid_n = (n as u32).div_ceil(IMMA_N as u32);
            let grid_m = (m as u32).div_ceil(IMMA_M as u32);
            let cfg = LaunchConfig {
                grid_dim: (grid_n, grid_m, 1),
                block_dim: (WARP_SIZE, 1, 1), // one warp per 16×8 output tile
                shared_mem_bytes: 0,          // the kernel's staging is static __shared__
            };
            let mut launch = self.stream.launch_builder(&self.func_imma);
            launch
                .arg(&d_qact)
                .arg(&buf.device)
                .arg(&d_act_scale)
                .arg(&d_wscale)
                .arg(&mut d_out)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .arg(&num_ktiles_i);
            // SAFETY: `launch` is `unsafe`. `tq2_0_imma_mpgemm(const signed char*,
            // const unsigned char*, const float*, const float*, float*, int, int,
            // int, int)` is fed in that exact order/type: `d_qact` (i8 [M,K]),
            // `buf.device` (the I2sInt8 tile bytes, validated to the packed length in
            // `upload_weights`), `d_act_scale` (f32 [M]), `d_wscale` (f32 [N]),
            // `d_out` (f32 [M,N], the only mutable operand), then `m,n,k,num_ktiles`.
            // The grid covers `ceil(M/16)·ceil(N/8)` tiles; the kernel bounds-checks
            // every global (`gm<m`, `gn<n`, `gk<k`) and addresses the weight tiles
            // with `num_ktiles`, the stride cached in the buffer. Host scalars
            // outlive the launch.
            #[allow(unsafe_code)]
            unsafe {
                launch
                    .launch(cfg)
                    .map_err(|e| driver_err("launch tq2_0_imma", &e))?;
            }
        }

        self.stream
            .memcpy_dtoh(&d_out, out)
            .map_err(|e| driver_err("download out (dtoh)", &e))?;

        Ok(())
    }
}

/// Threads in a warp — the IMMA kernel launches exactly one warp per output tile.
const WARP_SIZE: u32 = 32;

/// The IMMA `mma.m16n8k32` output-tile M dimension (16 rows per warp tile). The N
/// tile dimension is [`tritium_format::IMMA_N`] (8).
const IMMA_M: usize = 16;

/// Which add-only kernel a launch should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AddKernel {
    /// One thread per output element — the v0.10 kernel; general fallback.
    Simple,
    /// One warp per output, shared-mem staged activations — the decode path.
    Tiled,
}

impl TernaryBackend for CudaBackend {
    fn device_id(&self) -> &str {
        &self.device_id
    }

    fn capabilities(&self) -> DeviceCaps {
        // total_memory is left at its default (unknown): the contract permits 0 and
        // the runtime does not rely on the figure here. Both packings this backend
        // consumes are advertised so the runtime can route either kernel here.
        DeviceCaps::new("cuda", self.device_name.clone())
            .with_features(vec!["tq2_0".to_owned(), "i2s_int8".to_owned()])
    }

    fn upload_weights(
        &self,
        packed: &[u8],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let GemmShape { n, k, .. } = shape;

        // Each format has its own packed byte length and addressing stride; TQ1_0
        // is not a GPU packing here.
        let (expected, stride) = match format {
            TernaryFormat::Tq2_0 => {
                let row_bytes = Self::row_bytes(k);
                (n * row_bytes, Stride::Tq2_0 { row_bytes })
            }
            TernaryFormat::I2sInt8 => {
                // The IMMA tile interleave (see `tritium_format::convert_i2s_to_int8`):
                // `ceil(N/IMMA_N) · ceil(K/IMMA_K) · IMMA_WTILE_BYTES` bytes.
                let num_ntiles = n.div_ceil(IMMA_N);
                let num_ktiles = k.div_ceil(IMMA_K);
                (
                    num_ntiles * num_ktiles * IMMA_WTILE_BYTES,
                    Stride::I2sInt8 { num_ktiles },
                )
            }
            other => return Err(BackendError::UnsupportedFormat(other)),
        };

        if packed.len() != expected {
            return Err(BackendError::InvalidInput(format!(
                "packed len {} != expected {expected} for shape {shape:?} ({format:?})",
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
            format,
            stride,
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
        // Auto-select the add-only kernel by shape (decode → tiled), then run it.
        // All validation + the launch live in `mpgemm_kernel`.
        let kernel = Self::select_add_kernel(shape.m, shape.k);
        self.mpgemm_kernel(act, weights, scales, shape, format, out, kernel)
    }

    /// CUDA override of the fused W1.58A8 path. For an [`TernaryFormat::I2sInt8`]
    /// buffer this drives the on-device quant + IMMA int8 kernel
    /// ([`imma_with_act_quant`](CudaBackend::imma_with_act_quant)), dropping the
    /// host quant pass + the H2D round-trip the spec default would do. For a TQ2_0
    /// buffer (no int8 tensor-core path) it defers to the spec default — host
    /// per-token quant → [`mpgemm`](TernaryBackend::mpgemm) (add-only kernel) → fold
    /// — so both packings are served and the override stays within the `mpgemm`
    /// tolerance of that default (ADR 0005's "fused == host-A8" gate).
    ///
    /// # Errors
    /// As [`imma_with_act_quant`](CudaBackend::imma_with_act_quant) /
    /// [`mpgemm`](TernaryBackend::mpgemm) document, plus
    /// [`BackendError::InvalidInput`] if the buffer is not a CUDA buffer.
    fn mpgemm_with_act_quant(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        weight_scales: &[f32],
        shape: GemmShape,
        format: TernaryFormat,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let buf = weights
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a CudaBuffer".into()))?;
        match buf.format {
            TernaryFormat::I2sInt8 => {
                self.imma_with_act_quant(act, weights, weight_scales, shape, out)
            }
            // TQ2_0 (and anything else uploadable) → the host-quant default, which
            // delegates to the add-only `mpgemm`. `format` must agree with the
            // buffer; the default's `mpgemm` re-checks it.
            _ => self.default_mpgemm_with_act_quant(act, weights, weight_scales, shape, format, out),
        }
    }
}

/// Free function holding the spec's default `mpgemm_with_act_quant` body, so the
/// CUDA override can fall back to it for non-IMMA (TQ2_0) buffers without losing
/// the on-device IMMA path for I2sInt8. It is the literal host-A8 reference:
/// per-token int8 absmax quant on the host, then [`TernaryBackend::mpgemm`], then
/// fold the per-token scale — identical to [`tritium_spec`]'s provided default and
/// to the v0.20 caller-side quant.
impl CudaBackend {
    fn default_mpgemm_with_act_quant(
        &self,
        act: &[f32],
        weights: &dyn DeviceBuffer,
        weight_scales: &[f32],
        shape: GemmShape,
        format: TernaryFormat,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let GemmShape { m, n, k } = shape;
        if act.len() != m * k {
            return Err(BackendError::ShapeMismatch {
                expected: m * k,
                got: act.len(),
            });
        }
        if out.len() != m * n {
            return Err(BackendError::ShapeMismatch {
                expected: m * n,
                got: out.len(),
            });
        }
        if weight_scales.len() != n {
            return Err(BackendError::ShapeMismatch {
                expected: n,
                got: weight_scales.len(),
            });
        }

        // Per-token int8 absmax quant kept in f32 (the f32 `mpgemm` consumes it
        // directly), plus the per-token dequant multiplier. Matches the spec default
        // / `tritium-nn::ops::act_quant` (round-half-to-even, clamp [-128, 127]).
        let mut q = vec![0.0_f32; m * k];
        let mut act_scale = vec![0.0_f32; m];
        quantize_act_int8_host(act, m, k, &mut q, &mut act_scale);

        self.mpgemm(&q, weights, weight_scales, shape, format, out)?;

        // Fold the per-token activation scale: out[m,n] *= act_scale[m].
        for (row, &s) in out.chunks_exact_mut(n).zip(act_scale.iter()) {
            for v in row {
                *v *= s;
            }
        }
        Ok(())
    }
}

/// Symmetric int8 activation-quant positive cap (`Qp`). Matches
/// `tritium_spec`'s `A8_QB` / `tritium-nn::ops::act_quant::QB` and the device
/// `act_quant_int8_per_token` kernel.
const A8_QB: f32 = 127.0;

/// Host per-token int8 absmax quant — the fallback used by
/// [`CudaBackend::default_mpgemm_with_act_quant`] for TQ2_0 buffers. A verbatim copy
/// of `tritium_spec::quantize_act_int8` (which cannot be imported — it is private to
/// that crate). The fused-vs-default gate keeps the two from drifting: the IMMA
/// device quant, this host copy, and the spec copy all reproduce the same int8
/// values, pinned by `act_quant` parity in the spec's own `act_quant_golden` test.
fn quantize_act_int8_host(act: &[f32], m: usize, k: usize, q_out: &mut [f32], scale_out: &mut [f32]) {
    for r in 0..m {
        let row = &act[r * k..r * k + k];
        let mut gamma = 0.0_f32;
        for &v in row {
            let a = v.abs();
            if a > gamma {
                gamma = a;
            }
        }
        let out_row = &mut q_out[r * k..r * k + k];
        if gamma == 0.0 {
            for q in out_row.iter_mut() {
                *q = 0.0;
            }
            scale_out[r] = 0.0;
            continue;
        }
        let s = A8_QB / gamma;
        for (q, &v) in out_row.iter_mut().zip(row) {
            *q = (v * s).round_ties_even().clamp(-128.0, A8_QB);
        }
        scale_out[r] = gamma / A8_QB;
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

    /// Upload weights + run one TQ2_0 mpGEMM through a *forced* add kernel, so a
    /// test can gate each path independently of the shape-based auto-selection.
    fn run_kernel(
        cuda: &CudaBackend,
        packed: &[u8],
        act: &[f32],
        scales: &[f32],
        shape: GemmShape,
        kernel: AddKernel,
    ) -> Vec<f32> {
        let buf = cuda
            .upload_weights(packed, shape, TernaryFormat::Tq2_0)
            .expect("upload weights");
        let mut out = vec![0.0f32; shape.m * shape.n];
        cuda.mpgemm_kernel(
            act,
            buf.as_ref(),
            scales,
            shape,
            TernaryFormat::Tq2_0,
            &mut out,
            kernel,
        )
        .expect("mpgemm_kernel");
        out
    }

    /// Both add kernels must match the CPU reference (within tolerance) on the full
    /// committed TQ2_0 conformance set. This gates the new tiled kernel directly,
    /// and re-gates the simple kernel, regardless of which one auto-selection picks.
    #[test]
    fn both_add_kernels_match_reference() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping both-kernel gate: no device ({e})");
                return;
            }
        };
        let cpu = CpuBackend::new();
        let tol = Tolerance::default();

        for v in tq2_vectors() {
            let shape = GemmShape::new(v.m, v.n, v.k);
            let trits: Vec<_> = v
                .weights
                .iter()
                .map(|&w| tritium_core::Trit::from_i8(w).expect("weight in {-1,0,1}"))
                .collect();
            let packed = pack_tq2_0(&trits, shape);
            let cpu_out = run_backend(&cpu, &packed, &v.activation, &v.scales, shape);

            let simple = run_kernel(
                &cuda,
                &packed,
                &v.activation,
                &v.scales,
                shape,
                AddKernel::Simple,
            );
            for (i, (&g, &c)) in simple.iter().zip(&cpu_out).enumerate() {
                assert!(tol.accepts(g, c), "{}: simple vs cpu [{i}] {g} {c}", v.id);
            }

            // The tiled kernel only accepts K within its shared-memory budget.
            if v.k <= TILED_K_MAX {
                let tiled = run_kernel(
                    &cuda,
                    &packed,
                    &v.activation,
                    &v.scales,
                    shape,
                    AddKernel::Tiled,
                );
                for (i, (&g, &c)) in tiled.iter().zip(&cpu_out).enumerate() {
                    assert!(tol.accepts(g, c), "{}: tiled vs cpu [{i}] {g} {c}", v.id);
                }
            }
        }
    }

    /// The tiled kernel must be correct on boundary shapes: tail `K` (not a 256
    /// multiple, so a partial final TQ2_0 block), partial warps (`N` not a multiple
    /// of `WARPS_PER_BLOCK`), partial grids (`M`/`N` of 1), and `K` at the cap.
    #[test]
    fn tiled_handles_tail_shapes() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping tiled tail-shape gate: no device ({e})");
                return;
            }
        };
        let cpu = CpuBackend::new();
        let tol = Tolerance::default();

        // (M, N, K) — tail K, partial warps/blocks, single rows/cols, K at the cap.
        let shapes = [
            (1usize, 1usize, 1usize),
            (1, 7, 300),
            (5, 130, 257),
            (64, 3, 2560),
            (3, 33, 6912),
            (1, 1, TILED_K_MAX),
        ];

        for (m, n, k) in shapes {
            assert!(k <= TILED_K_MAX, "test shape K exceeds the tiled cap");
            let shape = GemmShape::new(m, n, k);

            // Deterministic ternary weights, activations, and per-channel scales.
            let trits: Vec<_> = (0..n * k)
                .map(|i| tritium_core::Trit::from_i8(((i % 3) as i8) - 1).unwrap())
                .collect();
            let act: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
            let scales: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.25).collect();

            let packed = pack_tq2_0(&trits, shape);
            let cpu_out = run_backend(&cpu, &packed, &act, &scales, shape);
            let tiled = run_kernel(&cuda, &packed, &act, &scales, shape, AddKernel::Tiled);

            assert_eq!(tiled.len(), cpu_out.len(), "shape {shape:?}: len");
            for (i, (&g, &c)) in tiled.iter().zip(&cpu_out).enumerate() {
                assert!(
                    tol.accepts(g, c),
                    "shape {shape:?}: tiled vs cpu [{i}] tiled={g} cpu={c}"
                );
            }
        }
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

    // ---- IMMA int8 tensor-core path (v0.30 WF-A part 2) ------------------------
    //
    // Tolerance: the conformance default (`relative = 1e-4`, ADR 0002). The IMMA
    // kernel contracts in **int32**, which is *exact* for int8×ternary (no overflow
    // for any BitNet K — see `kernels/tq2_0_imma.cu`), so the only float rounding is
    // the single per-output `act_scale·weight_scale·acc`. The 1e-4 band is therefore
    // the *reference's* own f32-accumulate rounding, not a defect of this kernel —
    // no widened reduction bar is needed (cf. the tiled add-only kernel, which sums
    // in double to stay inside the band; the IMMA integer accumulate is exact).

    /// Build an I2_S tensor payload (`N·K/4` quant bytes + one trailing `f32` scale)
    /// from an `[N, K]` row-major trit matrix, inverting the 32-byte block striping
    /// (`code = trit + 1`, element `pos` of a 128-block at byte `pos%32`, shift
    /// `6 - 2*(pos/32)`). `n*k` must be a multiple of 128 (the conformance shapes
    /// all are: K ∈ {256, 512}).
    fn build_i2s_payload(trits: &[i8], scale: f32) -> Vec<u8> {
        let n_elements = trits.len();
        assert!(
            n_elements.is_multiple_of(128),
            "i2s payload needs 128-multiple elems"
        );
        let mut quants = vec![0u8; n_elements / 4];
        for (global, &t) in trits.iter().enumerate() {
            let block = global / 128;
            let pos = global % 128;
            let group = pos / 32;
            let gp = pos % 32;
            let code = (t + 1) as u8; // {-1,0,1} -> {0,1,2}
            quants[block * 32 + gp] |= code << (6 - 2 * group);
        }
        let mut payload = quants;
        payload.extend_from_slice(&scale.to_le_bytes());
        payload
    }

    /// Pack an `[N, K]` trit matrix into the IMMA `I2sInt8` layout by routing it
    /// through the *real* converter (`build_i2s_payload` → `convert_i2s_to_int8`),
    /// so the test exercises exactly the bytes the kernel will see in production.
    /// Returns the packed bytes (block scale folded into the per-tensor `scale`,
    /// which the test keeps separate as the per-channel scale, so pass `scale = 1`).
    fn pack_i2s_int8(trits: &[i8], shape: GemmShape) -> Vec<u8> {
        let GemmShape { n, k, .. } = shape;
        let payload = build_i2s_payload(trits, 1.0);
        let w = tritium_format::convert_i2s_to_int8(&payload, GemmShape { m: 0, n, k })
            .expect("convert i2s -> int8");
        w.bytes
    }

    /// IMMA == reference within tolerance over the conformance set. The vectors'
    /// weights are converted to `I2sInt8`, uploaded, and run through the fused
    /// `mpgemm_with_act_quant` (which routes I2sInt8 → on-device quant + IMMA). The
    /// reference is `mpgemm_with_act_quant`'s contract on the *same f32 activations*:
    /// `out[m,n] = act_scale[m]·weight_scale[n]·Σ q[m,k]·w[n,k]`, which the testkit
    /// CPU path computes via the spec default — so this gates IMMA == host-A8 == ref
    /// in one shot.
    #[test]
    fn imma_matches_reference_within_tolerance() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping imma conformance: no device ({e})");
                return;
            }
        };
        let cpu = CpuBackend::new();
        let tol = Tolerance::default();

        for v in tq2_vectors() {
            let shape = GemmShape::new(v.m, v.n, v.k);

            // Reference: the host-A8 default path on the CPU backend over the SAME
            // f32 activations + per-channel weight scales.
            let cpu_buf = {
                let trits: Vec<_> = v
                    .weights
                    .iter()
                    .map(|&w| tritium_core::Trit::from_i8(w).expect("weight in {-1,0,1}"))
                    .collect();
                let packed = pack_tq2_0(&trits, shape);
                cpu.upload_weights(&packed, shape, TernaryFormat::Tq2_0)
                    .expect("cpu upload")
            };
            let mut ref_out = vec![0.0f32; shape.m * shape.n];
            cpu.mpgemm_with_act_quant(
                &v.activation,
                cpu_buf.as_ref(),
                &v.scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut ref_out,
            )
            .expect("cpu host-A8 reference");

            // IMMA: upload the I2sInt8 weights, run the fused override (on-device
            // quant + tensor-core contraction).
            let imma_bytes = pack_i2s_int8(&v.weights, shape);
            let imma_buf = cuda
                .upload_weights(&imma_bytes, shape, TernaryFormat::I2sInt8)
                .expect("imma upload");
            let mut imma_out = vec![0.0f32; shape.m * shape.n];
            cuda.mpgemm_with_act_quant(
                &v.activation,
                imma_buf.as_ref(),
                &v.scales,
                shape,
                TernaryFormat::I2sInt8,
                &mut imma_out,
            )
            .expect("imma fused mpgemm");

            assert_eq!(imma_out.len(), ref_out.len(), "{}: len", v.id);
            for (i, (&g, &c)) in imma_out.iter().zip(&ref_out).enumerate() {
                assert!(
                    tol.accepts(g, c),
                    "{}: imma vs host-A8 ref [{i}] imma={g} ref={c}",
                    v.id
                );
            }
        }
    }

    /// The CUDA fused override (IMMA) == the spec host-A8 default == the v0.20
    /// caller-side quant, all within tolerance — the "fused == host-A8" gate of ADR
    /// 0005. Three independently-derived results over the same inputs:
    ///   1. `cuda.mpgemm_with_act_quant` on an I2sInt8 buffer → on-device quant + IMMA.
    ///   2. The spec *default* `mpgemm_with_act_quant` (host quant → `mpgemm`) run on
    ///      the CPU backend (a TQ2_0 buffer).
    ///   3. The v0.20 caller-side quant: quantize on the host, then call plain
    ///      `mpgemm` and fold the per-token scale by hand.
    #[test]
    fn imma_fused_equals_host_a8_and_caller_quant() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping fused parity: no device ({e})");
                return;
            }
        };
        let cpu = CpuBackend::new();
        let tol = Tolerance::default();

        for v in tq2_vectors() {
            let shape = GemmShape::new(v.m, v.n, v.k);
            let GemmShape { m, n, k } = shape;
            let trits: Vec<_> = v
                .weights
                .iter()
                .map(|&w| tritium_core::Trit::from_i8(w).expect("weight in {-1,0,1}"))
                .collect();
            let tq2 = pack_tq2_0(&trits, shape);

            // (1) CUDA fused override on I2sInt8.
            let imma_bytes = pack_i2s_int8(&v.weights, shape);
            let imma_buf = cuda
                .upload_weights(&imma_bytes, shape, TernaryFormat::I2sInt8)
                .expect("imma upload");
            let mut fused = vec![0.0f32; m * n];
            cuda.mpgemm_with_act_quant(
                &v.activation,
                imma_buf.as_ref(),
                &v.scales,
                shape,
                TernaryFormat::I2sInt8,
                &mut fused,
            )
            .expect("cuda fused");

            // (2) Spec host-A8 default on the CPU backend (TQ2_0).
            let cpu_buf = cpu
                .upload_weights(&tq2, shape, TernaryFormat::Tq2_0)
                .expect("cpu upload");
            let mut host_a8 = vec![0.0f32; m * n];
            cpu.mpgemm_with_act_quant(
                &v.activation,
                cpu_buf.as_ref(),
                &v.scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut host_a8,
            )
            .expect("cpu host-A8");

            // (3) v0.20 caller-side quant: host quant → plain `mpgemm` → fold.
            let mut q = vec![0.0f32; m * k];
            let mut act_scale = vec![0.0f32; m];
            quantize_act_int8_host(&v.activation, m, k, &mut q, &mut act_scale);
            let mut caller = vec![0.0f32; m * n];
            cpu.mpgemm(
                &q,
                cpu_buf.as_ref(),
                &v.scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut caller,
            )
            .expect("cpu plain mpgemm");
            for (row, &s) in caller.chunks_exact_mut(n).zip(act_scale.iter()) {
                for x in row {
                    *x *= s;
                }
            }

            for i in 0..m * n {
                assert!(
                    tol.accepts(fused[i], host_a8[i]),
                    "{}: fused vs host-A8 [{i}] {} {}",
                    v.id,
                    fused[i],
                    host_a8[i]
                );
                assert!(
                    tol.accepts(fused[i], caller[i]),
                    "{}: fused vs caller-quant [{i}] {} {}",
                    v.id,
                    fused[i],
                    caller[i]
                );
            }
        }
    }

    /// IMMA tail/boundary shapes: M not a multiple of 16, N not a multiple of 8, and
    /// single rows/cols — the padding in the I2sInt8 tiles and the kernel's global
    /// bounds checks must keep every covered output correct. K stays a 256-multiple
    /// (the I2_S converter needs a 128-multiple element count); the M/N tails are the
    /// interesting axes for the 16×8 tile.
    #[test]
    fn imma_handles_tail_shapes() {
        let cuda = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping imma tail shapes: no device ({e})");
                return;
            }
        };
        let cpu = CpuBackend::new();
        let tol = Tolerance::default();

        // (M, N, K): single row/col, partial 16-row tile, partial 8-col tile.
        let shapes = [
            (1usize, 1usize, 256usize),
            (1, 8, 256),
            (3, 5, 256),
            (16, 8, 512),
            (17, 9, 256),
            (33, 13, 512),
        ];
        for (m, n, k) in shapes {
            let shape = GemmShape::new(m, n, k);
            // Deterministic ternary weights, activations, per-channel scales.
            let raw: Vec<i8> = (0..n * k).map(|i| ((i % 3) as i8) - 1).collect();
            let act: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
            let scales: Vec<f32> = (0..n).map(|j| 1.0 + (j % 4) as f32 * 0.5).collect();

            // Reference: host-A8 default on the CPU backend.
            let trits: Vec<_> = raw
                .iter()
                .map(|&w| tritium_core::Trit::from_i8(w).unwrap())
                .collect();
            let cpu_buf = cpu
                .upload_weights(&pack_tq2_0(&trits, shape), shape, TernaryFormat::Tq2_0)
                .expect("cpu upload");
            let mut ref_out = vec![0.0f32; m * n];
            cpu.mpgemm_with_act_quant(
                &act,
                cpu_buf.as_ref(),
                &scales,
                shape,
                TernaryFormat::Tq2_0,
                &mut ref_out,
            )
            .expect("cpu host-A8");

            let imma_buf = cuda
                .upload_weights(&pack_i2s_int8(&raw, shape), shape, TernaryFormat::I2sInt8)
                .expect("imma upload");
            let mut imma_out = vec![0.0f32; m * n];
            cuda.mpgemm_with_act_quant(
                &act,
                imma_buf.as_ref(),
                &scales,
                shape,
                TernaryFormat::I2sInt8,
                &mut imma_out,
            )
            .expect("imma fused");

            for (i, (&g, &c)) in imma_out.iter().zip(&ref_out).enumerate() {
                assert!(
                    tol.accepts(g, c),
                    "shape {shape:?}: imma vs ref [{i}] imma={g} ref={c}"
                );
            }
        }
    }
}

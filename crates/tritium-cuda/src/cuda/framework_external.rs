//! Validated CUDA adapter for framework-owned tensor storage and streams.
//!
//! PyTorch retains allocation and lifetime ownership. Every public safe call
//! validates context, allocation range, alignment, and geometry before the
//! narrow raw-driver launch. No synchronization or host transfer occurs.

use super::graph_raw::{pp, raw_launch};
use super::*;

const THREADS: u32 = 256;

/// Framework tensor scalar type accepted by external ternary Linear kernels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalLinearScalar {
    F32,
    F16,
}

/// Shared MxNxK geometry for external ternary Linear launches.
#[derive(Clone, Copy, Debug)]
pub struct ExternalLinearGeometry {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub row_bytes: usize,
}

/// Framework-owned buffers used to pack one dense master weight.
#[derive(Clone, Copy, Debug)]
pub struct ExternalLinearPack {
    pub stream: usize,
    pub master: usize,
    pub scales: usize,
    pub packed: usize,
    pub n: usize,
    pub k: usize,
    pub row_bytes: usize,
    pub scalar: ExternalLinearScalar,
}

/// Framework-owned buffers used by one projected ternary forward.
#[derive(Clone, Copy, Debug)]
pub struct ExternalLinearForward {
    pub stream: usize,
    pub input: usize,
    pub packed: usize,
    pub scales: usize,
    pub bias: Option<usize>,
    pub output: usize,
    pub geometry: ExternalLinearGeometry,
    pub scalar: ExternalLinearScalar,
}

/// Framework-owned buffers used by one first-order projected VJP.
#[derive(Clone, Copy, Debug)]
pub struct ExternalLinearBackward {
    pub stream: usize,
    pub grad_output: usize,
    pub input: usize,
    pub master: usize,
    pub packed: usize,
    pub scales: usize,
    pub grad_input: usize,
    pub grad_master: usize,
    pub grad_bias: Option<usize>,
    pub geometry: ExternalLinearGeometry,
    pub scalar: ExternalLinearScalar,
    pub master_scalar: ExternalLinearScalar,
}

#[derive(Debug)]
pub(super) struct ExternalCudaKernels {
    context: Arc<CudaContext>,
    module: sys::CUmodule,
    pack: sys::CUfunction,
    pack_f16: sys::CUfunction,
    forward: sys::CUfunction,
    forward_f16: sys::CUfunction,
    forward_tiled: sys::CUfunction,
    forward_tiled_f16: sys::CUfunction,
    grad_input: sys::CUfunction,
    grad_input_f16: sys::CUfunction,
    grad_master: sys::CUfunction,
    grad_master_f16: sys::CUfunction,
    grad_master_autocast: sys::CUfunction,
    grad_bias: sys::CUfunction,
    grad_bias_f16: sys::CUfunction,
}

#[cfg(test)]
impl ExternalCudaKernels {
    pub(super) fn forward_for_test(&self) -> sys::CUfunction {
        self.forward
    }
    pub(super) fn forward_f16_for_test(&self) -> sys::CUfunction {
        self.forward_f16
    }
    pub(super) fn forward_tiled_for_test(&self) -> sys::CUfunction {
        self.forward_tiled
    }
    pub(super) fn forward_tiled_f16_for_test(&self) -> sys::CUfunction {
        self.forward_tiled_f16
    }
}

// SAFETY: handles are immutable process-valid driver handles. CUDA Driver API
// permits launches from multiple host threads; each call supplies and validates
// its own stream. Module unload happens only after owning backend is dropped.
#[allow(unsafe_code)]
unsafe impl Send for ExternalCudaKernels {}
// SAFETY: same invariant; shared access mutates no Rust memory.
#[allow(unsafe_code)]
unsafe impl Sync for ExternalCudaKernels {}

impl ExternalCudaKernels {
    pub(super) fn load(ctx: &Arc<CudaContext>) -> Result<Self, BackendError> {
        ctx.bind_to_thread()
            .map_err(|error| driver_err("external kernels bind", &error))?;
        let ptx = CString::new(TRAIN_GRAD_PTX)
            .map_err(|_| BackendError::InvalidInput("training PTX has an interior NUL".into()))?;
        // SAFETY: `ptx` is a live NUL-terminated PTX image. Returned module is
        // owned by this value and unloaded exactly once in Drop.
        #[allow(unsafe_code)]
        let module = unsafe { result::module::load_data(ptx.as_ptr().cast()) }
            .map_err(|error| driver_err("external module load_data", &error))?;
        let get = |name: &str| -> Result<sys::CUfunction, BackendError> {
            let name = CString::new(name).map_err(|_| {
                BackendError::InvalidInput("kernel name has an interior NUL".into())
            })?;
            // SAFETY: module stays live; names are frozen extern-C PTX symbols.
            #[allow(unsafe_code)]
            unsafe { result::module::get_function(module, name) }
                .map_err(|error| driver_err("external module get_function", &error))
        };
        let loaded = (|| {
            Ok(Self {
                context: Arc::clone(ctx),
                module,
                pack: get(KERNEL_NAME_EXTERNAL_PACK)?,
                pack_f16: get(KERNEL_NAME_EXTERNAL_PACK_F16)?,
                forward: get(KERNEL_NAME_EXTERNAL_FORWARD)?,
                forward_f16: get(KERNEL_NAME_EXTERNAL_FORWARD_F16)?,
                forward_tiled: get(KERNEL_NAME_EXTERNAL_FORWARD_TILED)?,
                forward_tiled_f16: get(KERNEL_NAME_EXTERNAL_FORWARD_TILED_F16)?,
                grad_input: get(KERNEL_NAME_PROJECTED_GRAD_A)?,
                grad_input_f16: get(KERNEL_NAME_PROJECTED_GRAD_A_F16)?,
                grad_master: get(KERNEL_NAME_EXTERNAL_GRAD_MASTER)?,
                grad_master_f16: get(KERNEL_NAME_EXTERNAL_GRAD_MASTER_F16)?,
                grad_master_autocast: get(KERNEL_NAME_EXTERNAL_GRAD_MASTER_AUTOCAST)?,
                grad_bias: get(KERNEL_NAME_BIAS_BWD)?,
                grad_bias_f16: get(KERNEL_NAME_BIAS_BWD_F16)?,
            })
        })();
        if loaded.is_err() {
            // SAFETY: ownership has not escaped; unload failed partial module.
            #[allow(unsafe_code)]
            unsafe {
                let _ = result::module::unload(module);
            }
        }
        loaded
    }
}

impl Drop for ExternalCudaKernels {
    fn drop(&mut self) {
        if !self.module.is_null() {
            let _context = CurrentContextRestore::bind(&self.context).ok();
            // SAFETY: module was loaded by `load`, is owned here, and no launch
            // can outlive the backend borrow required to start it.
            #[allow(unsafe_code)]
            unsafe {
                let _ = result::module::unload(self.module);
            }
        }
    }
}

impl ExternalLinearGeometry {
    fn validate(self) -> Result<(i32, i32, i32, i32), BackendError> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(BackendError::InvalidInput(
                "external ternary Linear dimensions must be non-zero".into(),
            ));
        }
        let expected_row_bytes = num_blocks(self.k)
            .checked_mul(TQ2_0_BLOCK_BYTES)
            .ok_or_else(|| BackendError::InvalidInput("packed row bytes overflow".into()))?;
        if self.row_bytes != expected_row_bytes {
            return Err(BackendError::ShapeMismatch {
                expected: expected_row_bytes,
                got: self.row_bytes,
            });
        }
        let to_i32 = |value: usize, label: &str| {
            i32::try_from(value)
                .map_err(|_| BackendError::InvalidInput(format!("{label} does not fit i32")))
        };
        Ok((
            to_i32(self.m, "M")?,
            to_i32(self.n, "N")?,
            to_i32(self.k, "K")?,
            to_i32(self.row_bytes, "packed row bytes")?,
        ))
    }

    fn elements(self) -> Result<(usize, usize, usize), BackendError> {
        let mk = self
            .m
            .checked_mul(self.k)
            .ok_or_else(|| BackendError::InvalidInput("M*K overflows".into()))?;
        let mn = self
            .m
            .checked_mul(self.n)
            .ok_or_else(|| BackendError::InvalidInput("M*N overflows".into()))?;
        let nk = self
            .n
            .checked_mul(self.k)
            .ok_or_else(|| BackendError::InvalidInput("N*K overflows".into()))?;
        Ok((mk, mn, nk))
    }
}

fn bytes_f32(elements: usize, label: &str) -> Result<usize, BackendError> {
    elements
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| BackendError::InvalidInput(format!("{label} byte count overflows")))
}

impl ExternalLinearScalar {
    fn bytes(self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
            Self::F16 => size_of::<u16>(),
        }
    }

    fn select(self, f32_kernel: sys::CUfunction, f16_kernel: sys::CUfunction) -> sys::CUfunction {
        match self {
            Self::F32 => f32_kernel,
            Self::F16 => f16_kernel,
        }
    }

    fn select_grad_master(
        self,
        master: Self,
        f32_kernel: sys::CUfunction,
        f16_kernel: sys::CUfunction,
        autocast_kernel: sys::CUfunction,
    ) -> Result<sys::CUfunction, BackendError> {
        match (self, master) {
            (Self::F32, Self::F32) => Ok(f32_kernel),
            (Self::F16, Self::F16) => Ok(f16_kernel),
            (Self::F16, Self::F32) => Ok(autocast_kernel),
            (Self::F32, Self::F16) => Err(BackendError::InvalidInput(
                "fp32 activations with fp16 master weights are unsupported".into(),
            )),
        }
    }
}

fn bytes_scalar(
    elements: usize,
    scalar: ExternalLinearScalar,
    label: &str,
) -> Result<usize, BackendError> {
    elements
        .checked_mul(scalar.bytes())
        .ok_or_else(|| BackendError::InvalidInput(format!("{label} byte count overflows")))
}

fn grid(elements: usize) -> Result<(u32, u32, u32), BackendError> {
    let blocks = elements
        .checked_add(THREADS as usize - 1)
        .ok_or_else(|| BackendError::InvalidInput("launch size overflows".into()))?
        / THREADS as usize;
    Ok((
        u32::try_from(blocks)
            .map_err(|_| BackendError::InvalidInput("launch grid does not fit u32".into()))?,
        1,
        1,
    ))
}

impl CudaBackend {
    fn bind_external_context(&self) -> Result<CurrentContextRestore, BackendError> {
        CurrentContextRestore::bind(self.stream.context())
    }

    /// Test-only accessors so the in-crate parity gate can launch BOTH forward kernels on
    /// byte-identical inputs. `external_linear_forward` itself validates that pointers belong to the
    /// FRAMEWORK's context, so it cannot be driven from backend-owned buffers.
    #[cfg(test)]
    pub(super) fn external_kernels_for_test(&self) -> Result<&ExternalCudaKernels, BackendError> {
        self.external_kernels()
    }

    fn external_kernels(&self) -> Result<&ExternalCudaKernels, BackendError> {
        if let Some(kernels) = self.external_kernels.get() {
            return Ok(kernels);
        }
        let candidate = ExternalCudaKernels::load(self.stream.context())?;
        // Another caller may win this race. Its module becomes authoritative;
        // dropping our candidate safely unloads only the losing duplicate.
        let _ = self.external_kernels.set(candidate);
        self.external_kernels.get().ok_or_else(|| {
            BackendError::Backend("external CUDA kernel cache failed to initialize".into())
        })
    }

    fn validated_external_stream(&self, stream: usize) -> Result<sys::CUstream, BackendError> {
        let ctx = self.stream.context();
        let raw = stream as sys::CUstream;
        if raw.is_null() {
            // Legacy default stream belongs to context current on this thread.
            return Ok(raw);
        }
        let mut stream_ctx = std::ptr::null_mut();
        // SAFETY: `raw` is not dereferenced by Rust; driver validates handle and
        // writes one CUcontext into live storage.
        #[allow(unsafe_code)]
        unsafe { sys::cuStreamGetCtx(raw, &mut stream_ctx).result() }
            .map_err(|error| driver_err("external stream context query", &error))?;
        if stream_ctx != ctx.cu_ctx() {
            return Err(BackendError::InvalidInput(
                "external CUDA stream belongs to another context".into(),
            ));
        }
        Ok(raw)
    }

    fn validate_external_span(
        &self,
        pointer: usize,
        bytes: usize,
        alignment: usize,
        label: &str,
    ) -> Result<sys::CUdeviceptr, BackendError> {
        if pointer == 0 || bytes == 0 || !pointer.is_multiple_of(alignment) {
            return Err(BackendError::InvalidInput(format!(
                "{label} has null, empty, or misaligned CUDA storage"
            )));
        }
        let ptr = pointer as sys::CUdeviceptr;
        let mut pointer_ctx = std::ptr::null_mut();
        let mut range_start: sys::CUdeviceptr = 0;
        let mut range_size = 0usize;
        // SAFETY: output variables have attribute-matching types and remain
        // live. Driver validates `ptr`; no pointed-to tensor memory is read.
        #[allow(unsafe_code)]
        unsafe {
            sys::cuPointerGetAttribute(
                (&mut pointer_ctx as *mut sys::CUcontext).cast(),
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
                ptr,
            )
            .result()
            .and_then(|()| {
                sys::cuPointerGetAttribute(
                    (&mut range_start as *mut sys::CUdeviceptr).cast(),
                    sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                    ptr,
                )
                .result()
            })
            .and_then(|()| {
                sys::cuPointerGetAttribute(
                    (&mut range_size as *mut usize).cast(),
                    sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_RANGE_SIZE,
                    ptr,
                )
                .result()
            })
        }
        .map_err(|error| driver_err(&format!("{label} pointer query"), &error))?;
        if pointer_ctx != self.stream.context().cu_ctx() {
            return Err(BackendError::InvalidInput(format!(
                "{label} belongs to another CUDA context"
            )));
        }
        let allocation_end = (range_start as usize)
            .checked_add(range_size)
            .ok_or_else(|| BackendError::InvalidInput(format!("{label} allocation overflows")))?;
        let span_end = pointer
            .checked_add(bytes)
            .ok_or_else(|| BackendError::InvalidInput(format!("{label} span overflows")))?;
        if pointer < range_start as usize || span_end > allocation_end {
            return Err(BackendError::InvalidInput(format!(
                "{label} declared span exceeds CUDA allocation"
            )));
        }
        Ok(ptr)
    }

    /// Pack fp32 master weights into framework-owned TQ2_0 storage on caller stream.
    ///
    /// # Safety
    ///
    /// Every pointer must retain a live allocation of the declared size until
    /// stream work completes. Read/write aliases must obey kernel contracts.
    /// Caller must also keep this backend alive until completion.
    #[allow(unsafe_code)]
    pub unsafe fn external_linear_pack(
        &self,
        request: ExternalLinearPack,
    ) -> Result<(), BackendError> {
        let _context = self.bind_external_context()?;
        let kernels = self.external_kernels()?;
        let geometry = ExternalLinearGeometry {
            m: 1,
            n: request.n,
            k: request.k,
            row_bytes: request.row_bytes,
        };
        let (_, n, k, row_bytes) = geometry.validate()?;
        let (_, _, nk) = geometry.elements()?;
        let stream = self.validated_external_stream(request.stream)?;
        let master = self.validate_external_span(
            request.master,
            bytes_scalar(nk, request.scalar, "master")?,
            request.scalar.bytes(),
            "master",
        )?;
        let scales = self.validate_external_span(
            request.scales,
            bytes_f32(geometry.n, "scales")?,
            align_of::<f32>(),
            "scales",
        )?;
        let packed_bytes = geometry
            .n
            .checked_mul(geometry.row_bytes)
            .ok_or_else(|| BackendError::InvalidInput("packed byte count overflows".into()))?;
        let packed =
            self.validate_external_span(request.packed, packed_bytes, 1, "packed weights")?;
        let mut params = [
            pp(&master),
            pp(&scales),
            pp(&packed),
            pp(&n),
            pp(&k),
            pp(&row_bytes),
        ];
        raw_launch(
            request.scalar.select(kernels.pack, kernels.pack_f16),
            grid(geometry.n)?,
            (THREADS, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// Run projected fp32 ternary Linear directly into framework-owned output.
    ///
    /// # Safety
    ///
    /// Every pointer must retain a live allocation of the declared size until
    /// stream work completes. Read/write aliases must obey kernel contracts.
    /// Caller must also keep this backend alive until completion.
    #[allow(unsafe_code)]
    pub unsafe fn external_linear_forward(
        &self,
        request: ExternalLinearForward,
    ) -> Result<(), BackendError> {
        let _context = self.bind_external_context()?;
        let kernels = self.external_kernels()?;
        let geometry = request.geometry;
        let (m, n, k, row_bytes) = geometry.validate()?;
        let (mk, mn, _) = geometry.elements()?;
        let stream = self.validated_external_stream(request.stream)?;
        let input = self.validate_external_span(
            request.input,
            bytes_scalar(mk, request.scalar, "input")?,
            request.scalar.bytes(),
            "input",
        )?;
        let packed_bytes = geometry
            .n
            .checked_mul(geometry.row_bytes)
            .ok_or_else(|| BackendError::InvalidInput("packed byte count overflows".into()))?;
        let packed =
            self.validate_external_span(request.packed, packed_bytes, 1, "packed weights")?;
        let scales = self.validate_external_span(
            request.scales,
            bytes_f32(geometry.n, "scales")?,
            align_of::<f32>(),
            "scales",
        )?;
        let bias = match request.bias {
            Some(pointer) => self.validate_external_span(
                pointer,
                bytes_scalar(geometry.n, request.scalar, "bias")?,
                request.scalar.bytes(),
                "bias",
            )?,
            None => 0,
        };
        let output = self.validate_external_span(
            request.output,
            bytes_scalar(mn, request.scalar, "output")?,
            request.scalar.bytes(),
            "output",
        )?;
        let mut params = [
            pp(&input),
            pp(&packed),
            pp(&scales),
            pp(&bias),
            pp(&output),
            pp(&m),
            pp(&n),
            pp(&k),
            pp(&row_bytes),
        ];

        // Prefer the TILED kernel. The untiled `tq2_projected_linear_forward` assigns one thread per
        // output element and loops `k` inside it, so every one of the M rows re-reads the whole
        // weight matrix and adjacent threads read addresses `row_bytes` apart. Both defects are
        // invisible at M=1 and dominate past ~16: measured externally on a 4090 at 2048x2048, 5.4x
        // slower than torch fp16 at M=1 but 830x at M=2048, growing linearly in M.
        //
        // The tiled kernel stages the activation row in shared memory once per block and gives each
        // warp one output column, so weight bytes coalesce and the row is reused N times.
        //
        // Two conditions gate it, and both fall back rather than fail:
        //   * f32 only -- there is no f16 tiled variant yet, so half precision keeps the old path;
        //   * `k * 4` bytes of dynamic shared memory must fit the 48 KiB default block limit.
        let shared_bytes = geometry
            .k
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| BackendError::InvalidInput("shared-memory size overflows".into()))?;
        const MAX_DYNAMIC_SHARED: usize = 48 * 1024;
        // Both scalars tile now. fp16 matters most in practice: PyTorch autocast hands this op
        // fp16 activations, and a bf16 autocast is ALSO cast to fp16 by
        // `_ternary_linear_cuda_autocast`, so an f32-only fast path is unreachable from any
        // mixed-precision training loop -- exactly the case that needs M > 1.
        let mut tiled = shared_bytes <= MAX_DYNAMIC_SHARED;
        // Test-only: force the legacy untiled kernel so the parity gate can run BOTH paths on
        // byte-identical inputs. Without this the old path becomes unreachable for f32 and the
        // kernel this one replaces could rot silently.
        if cfg!(test) && std::env::var_os("TRITIUM_EXTERNAL_FORCE_UNTILED").is_some() {
            tiled = false;
        }
        if tiled {
            const WARPS_PER_BLOCK: u32 = 8;
            let grid_n = u32::try_from(geometry.n)
                .map_err(|_| BackendError::InvalidInput("N does not fit u32".into()))?
                .div_ceil(WARPS_PER_BLOCK);
            let grid_m = u32::try_from(geometry.m)
                .map_err(|_| BackendError::InvalidInput("M does not fit u32".into()))?;
            return raw_launch(
                request
                    .scalar
                    .select(kernels.forward_tiled, kernels.forward_tiled_f16),
                (grid_n, grid_m, 1),
                (WARPS_PER_BLOCK * 32, 1, 1),
                u32::try_from(shared_bytes).map_err(|_| {
                    BackendError::InvalidInput("shared bytes do not fit u32".into())
                })?,
                stream,
                &mut params,
            );
        }

        raw_launch(
            request.scalar.select(kernels.forward, kernels.forward_f16),
            grid(mn)?,
            (THREADS, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// Run first-order activation, dense-master STE, and optional bias VJPs.
    ///
    /// # Safety
    ///
    /// Every pointer must retain a live allocation of the declared size until
    /// stream work completes. Read/write aliases must obey kernel contracts.
    /// Caller must also keep this backend alive until completion.
    #[allow(unsafe_code)]
    pub unsafe fn external_linear_backward(
        &self,
        request: ExternalLinearBackward,
    ) -> Result<(), BackendError> {
        let _context = self.bind_external_context()?;
        let kernels = self.external_kernels()?;
        let geometry = request.geometry;
        let (m, n, k, row_bytes) = geometry.validate()?;
        let (mk, mn, nk) = geometry.elements()?;
        let stream = self.validated_external_stream(request.stream)?;
        let span = |pointer, elements, label| {
            self.validate_external_span(
                pointer,
                bytes_scalar(elements, request.scalar, label)?,
                request.scalar.bytes(),
                label,
            )
        };
        let grad_output = span(request.grad_output, mn, "grad_output")?;
        let input = span(request.input, mk, "input")?;
        let master = self.validate_external_span(
            request.master,
            bytes_scalar(nk, request.master_scalar, "master")?,
            request.master_scalar.bytes(),
            "master",
        )?;
        let packed_bytes = geometry
            .n
            .checked_mul(geometry.row_bytes)
            .ok_or_else(|| BackendError::InvalidInput("packed byte count overflows".into()))?;
        let packed =
            self.validate_external_span(request.packed, packed_bytes, 1, "packed weights")?;
        let scales = self.validate_external_span(
            request.scales,
            bytes_f32(geometry.n, "scales")?,
            align_of::<f32>(),
            "scales",
        )?;
        let grad_input = span(request.grad_input, mk, "grad_input")?;
        let grad_master = self.validate_external_span(
            request.grad_master,
            bytes_scalar(nk, request.master_scalar, "grad_master")?,
            request.master_scalar.bytes(),
            "grad_master",
        )?;
        let grad_bias = request
            .grad_bias
            .map(|pointer| span(pointer, geometry.n, "grad_bias"))
            .transpose()?;

        let mut input_params = [
            pp(&grad_output),
            pp(&packed),
            pp(&scales),
            pp(&grad_input),
            pp(&m),
            pp(&n),
            pp(&k),
            pp(&row_bytes),
        ];
        raw_launch(
            request
                .scalar
                .select(kernels.grad_input, kernels.grad_input_f16),
            grid(mk)?,
            (THREADS, 1, 1),
            0,
            stream,
            &mut input_params,
        )?;

        let mut master_params = [
            pp(&grad_output),
            pp(&input),
            pp(&master),
            pp(&scales),
            pp(&grad_master),
            pp(&m),
            pp(&n),
            pp(&k),
        ];
        raw_launch(
            request.scalar.select_grad_master(
                request.master_scalar,
                kernels.grad_master,
                kernels.grad_master_f16,
                kernels.grad_master_autocast,
            )?,
            grid(nk)?,
            (THREADS, 1, 1),
            0,
            stream,
            &mut master_params,
        )?;

        if let Some(grad_bias) = grad_bias {
            let mut bias_params = [pp(&grad_output), pp(&grad_bias), pp(&m), pp(&n)];
            raw_launch(
                request
                    .scalar
                    .select(kernels.grad_bias, kernels.grad_bias_f16),
                grid(geometry.n)?,
                (THREADS, 1, 1),
                0,
                stream,
                &mut bias_params,
            )?;
        }
        Ok(())
    }
}

pub(super) struct CurrentContextRestore {
    previous: Option<sys::CUcontext>,
}

impl CurrentContextRestore {
    pub(super) fn capture() -> Result<Self, BackendError> {
        let previous = result::ctx::get_current()
            .map_err(|error| driver_err("query current CUDA context", &error))?;
        Ok(Self { previous })
    }

    fn bind(ctx: &Arc<CudaContext>) -> Result<Self, BackendError> {
        let restore = Self::capture()?;
        ctx.bind_to_thread()
            .map_err(|error| driver_err("bind external CUDA context", &error))?;
        Ok(restore)
    }
}

impl Drop for CurrentContextRestore {
    fn drop(&mut self) {
        let previous = self.previous.unwrap_or(std::ptr::null_mut());
        // SAFETY: `previous` was returned by `cuCtxGetCurrent` on this thread
        // moments ago. Restoring it neither transfers nor destroys ownership.
        #[allow(unsafe_code)]
        unsafe {
            let _ = result::ctx::set_current(previous);
        }
    }
}

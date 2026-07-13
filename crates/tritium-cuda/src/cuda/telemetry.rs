//! Synchronized CUDA device-memory and async-pool evidence.

use core::ffi::{c_char, c_void};

use cudarc::driver::{CudaContext, result, sys};
use tritium_spec::BackendError;

use super::{CudaBackend, driver_err};

/// Stable CUDA device identity recorded with memory and performance evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaDeviceIdentity {
    /// CUDA ordinal used to open the backend.
    pub ordinal: usize,
    /// Driver-reported device name.
    pub device_name: String,
    /// PCI address in CUDA's canonical `domain:bus:device.function` form.
    pub pci_bus_id: String,
    /// CUDA Driver API version returned by `cuDriverGetVersion`.
    pub cuda_driver_version: u32,
}

/// One synchronized CUDA memory observation.
///
/// `device_used_sample_bytes` is the instantaneous `total - free` result from
/// `cuMemGetInfo`; it is not a device-wide high-water mark. The pool high-water
/// fields are exact for allocations made through the current CUDA async pool
/// since the tracker last reset them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaMemorySnapshot {
    /// Device memory reported free to the current CUDA context.
    pub device_free_bytes: u64,
    /// Device memory reported available to the current CUDA context.
    pub device_total_bytes: u64,
    /// Instantaneous `device_total_bytes - device_free_bytes` sample.
    pub device_used_sample_bytes: u64,
    /// Bytes currently in use by applications through the current async pool.
    pub pool_used_current_bytes: u64,
    /// Maximum pool bytes in use since the last synchronized reset.
    pub pool_used_high_water_bytes: u64,
    /// Backing bytes currently reserved by the current async pool.
    pub pool_reserved_current_bytes: u64,
    /// Maximum pool backing bytes reserved since the last synchronized reset.
    pub pool_reserved_high_water_bytes: u64,
}

/// Scoped observer for one backend's current CUDA async memory pool.
///
/// Construction resets the pool high-water marks after synchronizing the whole
/// CUDA context. Sampling also synchronizes the whole context and fails if code
/// in the process switches the device's current pool, which would otherwise make
/// subsequent observations silently incomplete.
#[derive(Debug)]
pub struct CudaMemoryTelemetry<'backend> {
    backend: &'backend CudaBackend,
    pool: sys::CUmemoryPool,
}

impl CudaBackend {
    /// Start synchronized memory telemetry for this backend's current async pool.
    ///
    /// The CUDA device must support stream-ordered allocation. The returned
    /// tracker has already reset both used and reserved pool high-water marks.
    pub fn start_memory_telemetry(
        &self,
    ) -> Result<(CudaDeviceIdentity, CudaMemoryTelemetry<'_>), BackendError> {
        let context = self.stream.context();
        if !context.has_async_alloc() {
            return Err(BackendError::Backend(
                "CUDA device does not support asynchronous memory pools; exact allocator high-water telemetry is unavailable"
                    .into(),
            ));
        }
        if self.cuda_version == 0 {
            return Err(BackendError::Backend(
                "CUDA driver version is unavailable; refusing unversioned memory telemetry".into(),
            ));
        }

        let identity = CudaDeviceIdentity {
            ordinal: context.ordinal(),
            device_name: context.name().map_err(|error| {
                driver_err("query CUDA device name for memory telemetry", &error)
            })?,
            pci_bus_id: query_pci_bus_id(context)?,
            cuda_driver_version: self.cuda_version,
        };
        let telemetry = CudaMemoryTelemetry {
            backend: self,
            pool: current_memory_pool(context)?,
        };
        telemetry.reset_synchronized()?;
        Ok((identity, telemetry))
    }
}

impl CudaMemoryTelemetry<'_> {
    /// Synchronize every stream and reset both async-pool high-water counters.
    ///
    /// The returned snapshot is taken after the reset. CUDA defines a reset while
    /// allocations remain live to leave each high-water value at least as large
    /// as its corresponding current value.
    pub fn reset_synchronized(&self) -> Result<CudaMemorySnapshot, BackendError> {
        let context = self.backend.stream.context();
        context
            .synchronize()
            .map_err(|error| driver_err("synchronize CUDA memory telemetry reset", &error))?;
        self.require_current_pool(context)?;
        reset_pool_high_water(
            self.pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
        )?;
        reset_pool_high_water(
            self.pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
        )?;
        self.sample_after_synchronize(context)
    }

    /// Synchronize every stream in the backend context and observe memory state.
    pub fn sample_synchronized(&self) -> Result<CudaMemorySnapshot, BackendError> {
        let context = self.backend.stream.context();
        context
            .synchronize()
            .map_err(|error| driver_err("synchronize CUDA memory telemetry sample", &error))?;
        self.sample_after_synchronize(context)
    }

    fn sample_after_synchronize(
        &self,
        context: &CudaContext,
    ) -> Result<CudaMemorySnapshot, BackendError> {
        self.require_current_pool(context)?;
        let (free, total) = context
            .mem_get_info()
            .map_err(|error| driver_err("query CUDA free/total memory", &error))?;
        let device_free_bytes = u64::try_from(free)
            .map_err(|_| BackendError::Backend("CUDA free-memory value exceeds u64".into()))?;
        let device_total_bytes = u64::try_from(total)
            .map_err(|_| BackendError::Backend("CUDA total-memory value exceeds u64".into()))?;
        let device_used_sample_bytes = device_total_bytes
            .checked_sub(device_free_bytes)
            .ok_or_else(|| {
                BackendError::Backend(format!(
                    "CUDA reported free memory {device_free_bytes} above total memory {device_total_bytes}"
                ))
            })?;

        let snapshot = CudaMemorySnapshot {
            device_free_bytes,
            device_total_bytes,
            device_used_sample_bytes,
            pool_used_current_bytes: pool_attribute(
                self.pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            )?,
            pool_used_high_water_bytes: pool_attribute(
                self.pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
            )?,
            pool_reserved_current_bytes: pool_attribute(
                self.pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            )?,
            pool_reserved_high_water_bytes: pool_attribute(
                self.pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            )?,
        };
        validate_snapshot(snapshot)?;
        Ok(snapshot)
    }

    fn require_current_pool(&self, context: &CudaContext) -> Result<(), BackendError> {
        let current = current_memory_pool(context)?;
        if current != self.pool {
            return Err(BackendError::Backend(
                "CUDA device current memory pool changed while telemetry was active; refusing incomplete high-water evidence"
                    .into(),
            ));
        }
        Ok(())
    }
}

fn validate_snapshot(snapshot: CudaMemorySnapshot) -> Result<(), BackendError> {
    if snapshot.pool_used_high_water_bytes < snapshot.pool_used_current_bytes {
        return Err(BackendError::Backend(
            "CUDA pool used-memory high-water mark is below current usage".into(),
        ));
    }
    if snapshot.pool_reserved_high_water_bytes < snapshot.pool_reserved_current_bytes {
        return Err(BackendError::Backend(
            "CUDA pool reserved-memory high-water mark is below current reservation".into(),
        ));
    }
    if snapshot.pool_reserved_current_bytes < snapshot.pool_used_current_bytes
        || snapshot.pool_reserved_high_water_bytes < snapshot.pool_used_high_water_bytes
    {
        return Err(BackendError::Backend(
            "CUDA pool backing-memory accounting is below used-memory accounting".into(),
        ));
    }
    Ok(())
}

fn query_pci_bus_id(context: &CudaContext) -> Result<String, BackendError> {
    context
        .bind_to_thread()
        .map_err(|error| driver_err("bind CUDA context for PCI identity", &error))?;
    let mut bytes = [0_u8; 32];
    #[allow(unsafe_code)]
    // SAFETY: `bytes` is a live writable buffer of the supplied length and
    // `cu_device` is owned by `context`. CUDA writes at most `bytes.len()` bytes.
    let result = unsafe {
        sys::cuDeviceGetPCIBusId(
            bytes.as_mut_ptr().cast::<c_char>(),
            bytes.len() as i32,
            context.cu_device(),
        )
    };
    result
        .result()
        .map_err(|error| driver_err("query CUDA PCI bus id", &error))?;
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .ok_or_else(|| BackendError::Backend("CUDA PCI bus id was not NUL-terminated".into()))?;
    let bus_id = core::str::from_utf8(&bytes[..end])
        .map_err(|error| BackendError::Backend(format!("CUDA PCI bus id is not UTF-8: {error}")))?;
    if bus_id.is_empty() {
        return Err(BackendError::Backend(
            "CUDA returned an empty PCI bus id".into(),
        ));
    }
    Ok(bus_id.to_owned())
}

fn current_memory_pool(context: &CudaContext) -> Result<sys::CUmemoryPool, BackendError> {
    context
        .bind_to_thread()
        .map_err(|error| driver_err("bind CUDA context for memory-pool query", &error))?;
    #[allow(unsafe_code)]
    // SAFETY: `cu_device` is the live device retained by `context`; cudarc returns
    // the driver-owned current pool handle without transferring ownership.
    let pool = unsafe { result::device::get_mem_pool(context.cu_device()) }
        .map_err(|error| driver_err("query CUDA current memory pool", &error))?;
    if pool.is_null() {
        return Err(BackendError::Backend(
            "CUDA returned a null current memory pool".into(),
        ));
    }
    Ok(pool)
}

fn pool_attribute(
    pool: sys::CUmemoryPool,
    attribute: sys::CUmemPool_attribute,
) -> Result<u64, BackendError> {
    let mut value = 0_u64;
    #[allow(unsafe_code)]
    // SAFETY: `pool` is a live driver-owned pool checked by the scoped tracker;
    // all queried attributes in this module require a writable `cuuint64_t`.
    unsafe {
        result::mem_pool::get_attribute(pool, attribute, (&mut value as *mut u64).cast::<c_void>())
    }
    .map_err(|error| driver_err("query CUDA memory-pool attribute", &error))?;
    Ok(value)
}

fn reset_pool_high_water(
    pool: sys::CUmemoryPool,
    attribute: sys::CUmemPool_attribute,
) -> Result<(), BackendError> {
    let mut zero = 0_u64;
    #[allow(unsafe_code)]
    // SAFETY: `pool` is a live driver-owned pool checked by the scoped tracker;
    // CUDA specifies a writable `cuuint64_t` containing zero to reset either
    // high-water attribute.
    unsafe {
        result::mem_pool::set_attribute(pool, attribute, (&mut zero as *mut u64).cast::<c_void>())
    }
    .map_err(|error| driver_err("reset CUDA memory-pool high-water attribute", &error))
}

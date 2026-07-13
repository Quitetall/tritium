#![cfg(feature = "cuda")]

use tritium_cuda::train::DeviceTensor;
use tritium_cuda::{CudaBackend, CudaMemoryTelemetry};
use tritium_spec::BackendError;

static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct CurrentPoolGuard {
    device: cudarc::driver::sys::CUdevice,
    original: cudarc::driver::sys::CUmemoryPool,
    alternate: cudarc::driver::sys::CUmemoryPool,
    active: bool,
}

impl CurrentPoolGuard {
    fn restore(mut self) {
        use cudarc::driver::result;

        // SAFETY: both pool handles came from CUDA for `device`; restore the
        // retained original before destroying the unused alternate exactly once.
        unsafe {
            result::device::set_mem_pool(self.device, self.original)
                .expect("restore original CUDA pool");
            result::mem_pool::destroy(self.alternate).expect("destroy alternate CUDA pool");
        }
        self.active = false;
    }
}

impl Drop for CurrentPoolGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        // SAFETY: best-effort panic cleanup uses the same live handles retained
        // by this guard. Errors cannot be reported safely from `Drop`.
        unsafe {
            let _ = cudarc::driver::result::device::set_mem_pool(self.device, self.original);
            let _ = cudarc::driver::result::mem_pool::destroy(self.alternate);
        }
    }
}

fn backend_or_skip(test: &str) -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(backend) => Some(backend),
        Err(BackendError::Backend(message)) if message.starts_with("open cuda device:") => {
            eprintln!("skipping {test}: no CUDA device (backend error: {message})");
            None
        }
        Err(error) => panic!("initialize CUDA backend for {test}: {error}"),
    }
}

fn telemetry_or_skip<'a>(backend: &'a CudaBackend, test: &str) -> Option<CudaMemoryTelemetry<'a>> {
    match backend.start_memory_telemetry() {
        Ok((_identity, telemetry)) => Some(telemetry),
        Err(error) if error.to_string().contains("asynchronous memory pools") => {
            eprintln!("skipping {test}: CUDA memory pools are unavailable ({error})");
            None
        }
        Err(error) => panic!("start CUDA memory telemetry: {error}"),
    }
}

#[test]
fn public_memory_telemetry_reports_identity_and_consistent_baseline() {
    let _guard = GPU_TEST_LOCK.lock().expect("lock CUDA telemetry tests");
    let Some(backend) =
        backend_or_skip("public_memory_telemetry_reports_identity_and_consistent_baseline")
    else {
        return;
    };
    let (identity, telemetry) = match backend.start_memory_telemetry() {
        Ok(result) => result,
        Err(error) if error.to_string().contains("asynchronous memory pools") => {
            eprintln!(
                "skipping public_memory_telemetry_reports_identity_and_consistent_baseline: \
                 CUDA memory pools are unavailable ({error})"
            );
            return;
        }
        Err(error) => panic!("start CUDA memory telemetry: {error}"),
    };

    assert_eq!(identity.ordinal, 0);
    assert!(!identity.device_name.is_empty());
    assert!(!identity.pci_bus_id.is_empty());
    assert_ne!(identity.cuda_driver_version, 0);

    let snapshot = telemetry
        .sample_synchronized()
        .expect("sample CUDA memory telemetry");
    assert_eq!(
        snapshot.device_used_sample_bytes,
        snapshot.device_total_bytes - snapshot.device_free_bytes
    );
    assert!(snapshot.pool_used_high_water_bytes >= snapshot.pool_used_current_bytes);
    assert!(snapshot.pool_reserved_high_water_bytes >= snapshot.pool_reserved_current_bytes);
    assert!(snapshot.pool_reserved_current_bytes >= snapshot.pool_used_current_bytes);
}

#[test]
fn public_memory_telemetry_preserves_allocator_high_water_after_free() {
    let _guard = GPU_TEST_LOCK.lock().expect("lock CUDA telemetry tests");
    let test = "public_memory_telemetry_preserves_allocator_high_water_after_free";
    let Some(backend) = backend_or_skip(test) else {
        return;
    };
    let Some(telemetry) = telemetry_or_skip(&backend, test) else {
        return;
    };
    let before = telemetry
        .reset_synchronized()
        .expect("reset CUDA memory high-water marks");

    const ELEMENTS: usize = 1024 * 1024;
    let tensor = DeviceTensor::upload(&backend, &vec![0.0; ELEMENTS])
        .expect("allocate public device tensor");
    let live = telemetry
        .sample_synchronized()
        .expect("sample live CUDA allocation");
    let requested_bytes = ELEMENTS * size_of::<f32>();
    assert!(
        live.pool_used_current_bytes
            >= before.pool_used_current_bytes + u64::try_from(requested_bytes).unwrap()
    );
    assert!(live.pool_used_high_water_bytes >= live.pool_used_current_bytes);
    assert!(live.pool_reserved_high_water_bytes >= live.pool_used_high_water_bytes);
    let reset_while_live = telemetry
        .reset_synchronized()
        .expect("reset CUDA high-water marks while allocation remains live");
    assert_eq!(
        reset_while_live.pool_used_current_bytes,
        live.pool_used_current_bytes
    );
    assert!(
        reset_while_live.pool_used_high_water_bytes >= reset_while_live.pool_used_current_bytes
    );

    drop(tensor);
    let freed = telemetry
        .sample_synchronized()
        .expect("sample freed CUDA allocation");
    assert!(freed.pool_used_current_bytes < live.pool_used_current_bytes);
    assert!(freed.pool_used_high_water_bytes >= reset_while_live.pool_used_high_water_bytes);
    assert!(
        freed.pool_reserved_high_water_bytes >= reset_while_live.pool_reserved_high_water_bytes
    );
}

#[test]
fn public_memory_telemetry_fails_if_the_process_switches_current_pool() {
    use cudarc::driver::{CudaContext, result, sys};

    let _guard = GPU_TEST_LOCK.lock().expect("lock CUDA telemetry tests");
    let test = "public_memory_telemetry_fails_if_the_process_switches_current_pool";
    let Some(backend) = backend_or_skip(test) else {
        return;
    };
    let Some(telemetry) = telemetry_or_skip(&backend, test) else {
        return;
    };
    let context = CudaContext::new(0).expect("retain CUDA primary context");
    let device = context.cu_device();
    let properties = sys::CUmemPoolProps {
        allocType: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        handleTypes: sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
        location: sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            __bindgen_anon_1: sys::CUmemLocation_st__bindgen_ty_1 { id: 0 },
        },
        win32SecurityAttributes: core::ptr::null_mut(),
        maxSize: 0,
        usage: 0,
        reserved: [0; 54],
    };

    // SAFETY: `device` is retained by `context`; the properties describe a
    // device-local pinned allocation pool and remain live for the call.
    let guard = unsafe {
        let original = result::device::get_mem_pool(device).expect("query original CUDA pool");
        let alternate = result::mem_pool::create(&properties).expect("create alternate CUDA pool");
        result::device::set_mem_pool(device, alternate).expect("switch current CUDA pool");
        CurrentPoolGuard {
            device,
            original,
            alternate,
            active: true,
        }
    };

    let sample = telemetry.sample_synchronized();
    guard.restore();

    let error = sample.expect_err("pool drift must invalidate scoped telemetry");
    assert!(error.to_string().contains("current memory pool changed"));
}

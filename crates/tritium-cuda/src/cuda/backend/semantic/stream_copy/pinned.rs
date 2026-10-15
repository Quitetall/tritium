//! Two-slot copy lifetime machinery, independent of automatic event tracking.
use super::*;
use crate::cuda::backend::PinnedPrefix;
use cudarc::driver::result::DriverError;
use cudarc::driver::{CudaEvent, DeviceRepr, PinnedHostSlice, ValidAsZeroBits};

pub(super) struct Window<T: DeviceRepr + ValidAsZeroBits> {
    data: PinnedHostSlice<T>,
    pointer: *mut T,
    completion: CudaEvent,
    stream: Arc<CudaStream>,
    pending: bool,
    recorded: bool,
}
impl<T: DeviceRepr + ValidAsZeroBits> Window<T> {
    #[allow(unsafe_code)]
    pub(super) fn new(backend: &CudaBackend, capacity: usize) -> Result<Self, BackendError> {
        if capacity == 0 {
            return Err(invalid("pinned window must be nonempty"));
        }
        // Captured graphs would retain host pointers beyond this upload owner.
        if backend
            .stream
            .capture_status()
            .map_err(|e| driver_err("inspect upload capture status", &e))?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(invalid(
                "pinned tensor upload does not support graph capture",
            ));
        }
        let requested = capacity
            .checked_mul(core::mem::size_of::<T>())
            .ok_or_else(|| invalid("pinned window size overflow"))?;
        // SAFETY: allocation is opaque/uninitialized; initialize every element
        // through the raw pointer before making any typed slice. Zero is valid T.
        let mut data =
            unsafe { backend.stream.context().alloc_pinned::<T>(capacity) }.map_err(|e| {
                alloc_or_backend("allocate bounded pinned upload window", &e, requested)
            })?;
        let pointer = data
            .as_mut_ptr()
            .map_err(|e| driver_err("initialize pinned upload pointer", &e))?;
        // SAFETY: unique allocation spans capacity T elements; all are initialized.
        unsafe { pointer.write_bytes(0, capacity) };
        let completion = backend
            .stream
            .context()
            .new_event(None)
            .map_err(|e| driver_err("create pinned upload completion", &e))?;
        Ok(Self {
            data,
            pointer,
            completion,
            stream: backend.stream.clone(),
            pending: false,
            recorded: false,
        })
    }
    pub(super) fn wait(&mut self) -> Result<(), BackendError> {
        self.complete()
            .map_err(|e| driver_err("complete pinned upload window", &e))
    }
    fn complete(&mut self) -> Result<(), DriverError> {
        if self.pending {
            let result = if self.recorded {
                self.completion.synchronize()
            } else {
                self.stream.synchronize()
            };
            result?;
            self.pending = false;
            self.recorded = false;
        }
        Ok(())
    }
    #[allow(unsafe_code)]
    pub(super) fn mutable(&mut self) -> Result<&mut [T], BackendError> {
        self.wait()?;
        // SAFETY: fully initialized owned storage; wait completed every DMA
        // read. Exclusive self borrow forbids another copy while this slice lives.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.pointer, self.data.len()) })
    }
    pub(super) fn copy(
        &mut self,
        device: &mut CudaSlice<T>,
        at: usize,
        count: usize,
    ) -> Result<(), BackendError> {
        if self.pending {
            return Err(invalid("pinned upload window reused before completion"));
        }
        let end = at
            .checked_add(count)
            .ok_or_else(|| invalid("pinned copy extent overflow"))?;
        if end > device.len() || count > self.data.len() {
            return Err(invalid("pinned copy exceeds declared extent"));
        }
        if self
            .stream
            .capture_status()
            .map_err(|e| driver_err("inspect pinned copy capture status", &e))?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(invalid("pinned tensor copy does not support graph capture"));
        }
        // Mark before enqueue: an enqueue/event-record failure must drain the
        // stream in Drop, not trust a stale completion event from the last use.
        self.pending = true;
        self.recorded = false;
        let prefix = PinnedPrefix::new(&mut self.data, count)?;
        self.stream
            .memcpy_htod(&prefix, &mut device.slice_mut(at..end))
            .map_err(|e| driver_err("enqueue bounded pinned upload", &e))?;
        self.completion
            .record(&self.stream)
            .map_err(|e| driver_err("record bounded pinned upload completion", &e))?;
        self.recorded = true;
        Ok(())
    }
}
impl<T: DeviceRepr + ValidAsZeroBits> Drop for Window<T> {
    fn drop(&mut self) {
        if self.pending
            && let Err(error) = self.complete()
        {
            // Cleanup cannot return an error. Retain memory through a fallback
            // drain and record its result in the owning context error channel.
            self.stream.context().record_err::<()>(Err(error));
            self.stream.context().record_err(self.stream.synchronize());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[allow(unsafe_code)]
    fn physical_pinned_windows_reuse_and_drop_without_automatic_tracking() {
        for tracking in [true, false] {
            let backend = CudaBackend::new(0).expect("pinned window test requires physical CUDA");
            if !tracking {
                // SAFETY: isolated context, one stream; explicit events and a
                // terminal drain cover every allocation/copy/drop in this test.
                unsafe { backend.stream.context().disable_event_tracking() };
            }
            let mut device = backend.stream.alloc_zeros::<u8>(37 * 257).unwrap();
            let mut expected = vec![0; device.len()];
            {
                let mut windows = [
                    Window::new(&backend, 257).unwrap(),
                    Window::new(&backend, 257).unwrap(),
                ];
                for block in 0..37 {
                    let active = block % 2;
                    let count = if block == 36 { 11 } else { 257 };
                    let bytes = windows[active].mutable().unwrap();
                    for (i, b) in bytes[..count].iter_mut().enumerate() {
                        *b = ((block * 19 + i) % 251) as u8;
                    }
                    expected[block * 257..block * 257 + count].copy_from_slice(&bytes[..count]);
                    windows[active]
                        .copy(&mut device, block * 257, count)
                        .unwrap();
                    assert!(
                        windows[active]
                            .copy(&mut device, block * 257, count)
                            .is_err(),
                        "queued slot may not be reused before wait"
                    );
                }
                // Drop rather than finish/wait exercises queued-source custody.
            }
            backend.stream.synchronize().unwrap();
            assert_eq!(backend.stream.clone_dtoh(&device).unwrap(), expected);
            backend.stream.context().check_err().unwrap();
        }
    }
    #[test]
    fn physical_pinned_window_refuses_graph_capture_and_bad_extent() {
        let backend = CudaBackend::new(0).expect("pinned capture test requires physical CUDA");
        backend
            .stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .unwrap();
        assert!(Window::<u8>::new(&backend, 32).is_err());
        backend.stream.end_capture(cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH).unwrap();
        let mut device = backend.stream.alloc_zeros::<u8>(32).unwrap();
        let mut window = Window::<u8>::new(&backend, 16).unwrap();
        assert!(window.copy(&mut device, usize::MAX, 1).is_err());
        assert!(window.copy(&mut device, 0, 17).is_err());
        assert!(window.copy(&mut device, 31, 2).is_err());
        assert!(!window.pending);
        backend.stream.context().check_err().unwrap();
    }
}

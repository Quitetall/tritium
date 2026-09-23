//! Device-resident Gated DeltaNet recurrent state and its per-token step.
//!
//! 48 of Qwen3.6's 64 layers are `linear_attention`. Its state update ran on the
//! host, where an nsys profile of decode measured
//! `Qwen35DeltaNet::recurrent_forward` at 31.8% of every CPU sample and roughly
//! 145 ms of each 270 ms token, against 107 ms of actual GPU kernel. The state
//! is 48 heads x 128 x 128 f32 per layer -- 3 MiB -- and `stage_forward` copied
//! all of it host-side on every token before touching it, which is the 7.6% the
//! same profile attributes to `__memcpy_avx_unaligned_erms`.
//!
//! Keeping the state on the device removes both: the copy becomes a
//! device-to-device transfer, and commit becomes a pointer swap.

use super::*;

/// Narrow a geometry value to the kernel ABI's `u32`.
fn to_u32(value: usize, field: &str) -> Result<u32, BackendError> {
    u32::try_from(value)
        .map_err(|_| BackendError::InvalidInput(format!("{field} exceeds the u32 kernel ABI")))
}

/// One layer's Gated DeltaNet recurrence, resident on the device.
///
/// Mirrors the host cache's double buffer: a committed state and a staging copy
/// the step mutates, so an aborted transaction leaves the committed state
/// untouched and a commit is a swap rather than a copy.
pub struct DeltaNetResidentState {
    current: CudaSlice<f32>,
    staging: CudaSlice<f32>,
    value_heads: u32,
    key_head_dim: u32,
    value_head_dim: u32,
    group_size: u32,
}

impl core::fmt::Debug for DeltaNetResidentState {
    /// Geometry only: the device allocations have no meaningful debug form, and
    /// the state is megabytes per layer.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("DeltaNetResidentState")
            .field("value_heads", &self.value_heads)
            .field("key_head_dim", &self.key_head_dim)
            .field("value_head_dim", &self.value_head_dim)
            .field("group_size", &self.group_size)
            .finish_non_exhaustive()
    }
}

impl DeltaNetResidentState {
    /// Elements in one state buffer.
    #[must_use]
    pub const fn elements(&self) -> usize {
        (self.value_heads * self.key_head_dim * self.value_head_dim) as usize
    }

    /// Value heads carried by this state.
    #[must_use]
    pub const fn value_heads(&self) -> usize {
        self.value_heads as usize
    }

    /// Publish staging as committed. A swap; allocates nothing and cannot fail,
    /// which is what lets the host cache commit without holding a backend.
    pub fn commit(&mut self) {
        core::mem::swap(&mut self.current, &mut self.staging);
    }
}

impl CudaBackend {
    /// Allocate one layer's zeroed DeltaNet state on the device.
    ///
    /// # Errors
    /// Rejects a zero or oversized geometry, a `value_heads` that is not a whole
    /// multiple of the key-head group, or a device allocation failure.
    pub fn new_deltanet_state(
        &self,
        value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        group_size: usize,
    ) -> Result<DeltaNetResidentState, BackendError> {
        if value_heads == 0 || key_head_dim == 0 || value_head_dim == 0 || group_size == 0 {
            return Err(BackendError::InvalidInput(
                "DeltaNet state geometry must be non-zero".into(),
            ));
        }
        if !value_heads.is_multiple_of(group_size) {
            return Err(BackendError::InvalidInput(
                "DeltaNet value heads must be a whole number of key-head groups".into(),
            ));
        }
        // One thread owns one value lane, so the lane count is the block size.
        if value_head_dim > 1024 {
            return Err(BackendError::InvalidInput(
                "DeltaNet value head dim exceeds the one-lane-per-thread block limit".into(),
            ));
        }
        let elements = value_heads
            .checked_mul(key_head_dim)
            .and_then(|value| value.checked_mul(value_head_dim))
            .ok_or_else(|| {
                BackendError::InvalidInput("DeltaNet state element count overflows".into())
            })?;
        let bytes = elements * core::mem::size_of::<f32>();
        let current = self
            .stream
            .alloc_zeros::<f32>(elements)
            .map_err(|error| alloc_or_backend("allocate DeltaNet state", &error, bytes))?;
        let staging = self
            .stream
            .alloc_zeros::<f32>(elements)
            .map_err(|error| alloc_or_backend("allocate DeltaNet staging state", &error, bytes))?;
        Ok(DeltaNetResidentState {
            current,
            staging,
            value_heads: to_u32(value_heads, "DeltaNet value heads")?,
            key_head_dim: to_u32(key_head_dim, "DeltaNet key head dim")?,
            value_head_dim: to_u32(value_head_dim, "DeltaNet value head dim")?,
            group_size: to_u32(group_size, "DeltaNet group size")?,
        })
    }

    /// Copy the committed state into staging, beginning a transaction.
    ///
    /// # Errors
    /// Returns [`BackendError::Backend`] on a driver failure.
    pub fn deltanet_stage(&self, state: &mut DeltaNetResidentState) -> Result<(), BackendError> {
        let elements = state.elements();
        let source = state.current.slice(..elements);
        self.stream
            .memcpy_dtod(&source, &mut state.staging)
            .map_err(|error| driver_err("stage DeltaNet state", &error))
    }

    /// Clear the committed and staged state without releasing capacity.
    ///
    /// # Errors
    /// Returns [`BackendError::Backend`] on a driver failure.
    pub fn deltanet_reset(&self, state: &mut DeltaNetResidentState) -> Result<(), BackendError> {
        self.stream
            .memset_zeros(&mut state.current)
            .map_err(|error| driver_err("reset DeltaNet state", &error))?;
        self.stream
            .memset_zeros(&mut state.staging)
            .map_err(|error| driver_err("reset DeltaNet staging state", &error))
    }

    /// Read the staged state back to the host, for parity checks and diagnostics.
    ///
    /// # Errors
    /// Rejects a wrong output length, or returns a driver failure.
    pub fn deltanet_read_staged(
        &self,
        state: &DeltaNetResidentState,
        output: &mut [f32],
    ) -> Result<(), BackendError> {
        let elements = state.elements();
        if output.len() != elements {
            return Err(BackendError::ShapeMismatch {
                expected: elements,
                got: output.len(),
            });
        }
        let view = state.staging.slice(..elements);
        self.stream
            .memcpy_dtoh(&view, output)
            .map_err(|error| driver_err("read staged DeltaNet state", &error))
    }

    /// Overwrite the staged state from the host, for parity checks and diagnostics.
    ///
    /// # Errors
    /// Rejects a wrong input length, or returns a driver failure.
    pub fn deltanet_write_staged(
        &self,
        state: &mut DeltaNetResidentState,
        values: &[f32],
    ) -> Result<(), BackendError> {
        let elements = state.elements();
        if values.len() != elements {
            return Err(BackendError::ShapeMismatch {
                expected: elements,
                got: values.len(),
            });
        }
        self.stream
            .memcpy_htod(values, &mut state.staging)
            .map_err(|error| driver_err("write staged DeltaNet state", &error))
    }

    /// Advance the staged state by one token, for every head at once.
    ///
    /// `kk` and `qq` are the key and query already multiplied by their L2
    /// inverses (and, for `qq`, the query scale), and `beta` and `decay` are the
    /// gate values. Every one of those needs a transcendental, and `expf` is not
    /// required to agree with `f32::exp`, so they are computed host-side and the
    /// kernel performs multiply/add only. That is what lets the device state stay
    /// bit-identical to the host reduction.
    ///
    /// # Errors
    /// Rejects any operand whose length disagrees with the state geometry, or
    /// returns a driver failure.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_recurrent_step(
        &self,
        state: &mut DeltaNetResidentState,
        kk: &[f32],
        qq: &[f32],
        value: &[f32],
        beta: &[f32],
        decay: &[f32],
        core_out: &mut [f32],
    ) -> Result<(), BackendError> {
        let value_heads = state.value_heads as usize;
        let key_head_dim = state.key_head_dim as usize;
        let value_head_dim = state.value_head_dim as usize;
        let key_heads = value_heads / state.group_size as usize;
        let projected_len = key_heads * key_head_dim;
        let lane_len = value_heads * value_head_dim;
        for (name, got, expected) in [
            ("kk", kk.len(), projected_len),
            ("qq", qq.len(), projected_len),
            ("value", value.len(), lane_len),
            ("beta", beta.len(), value_heads),
            ("decay", decay.len(), value_heads),
            ("core", core_out.len(), lane_len),
        ] {
            if got != expected {
                let _ = name;
                return Err(BackendError::ShapeMismatch { expected, got });
            }
        }

        let d_kk = self
            .stream
            .clone_htod(kk)
            .map_err(|error| driver_err("upload DeltaNet key projection", &error))?;
        let d_qq = self
            .stream
            .clone_htod(qq)
            .map_err(|error| driver_err("upload DeltaNet query projection", &error))?;
        let d_value = self
            .stream
            .clone_htod(value)
            .map_err(|error| driver_err("upload DeltaNet value", &error))?;
        let d_beta = self
            .stream
            .clone_htod(beta)
            .map_err(|error| driver_err("upload DeltaNet beta", &error))?;
        let d_decay = self
            .stream
            .clone_htod(decay)
            .map_err(|error| driver_err("upload DeltaNet decay", &error))?;
        let mut d_core = self.stream.alloc_zeros::<f32>(lane_len).map_err(|error| {
            alloc_or_backend(
                "allocate DeltaNet core output",
                &error,
                lane_len * core::mem::size_of::<f32>(),
            )
        })?;

        let shared_mem_bytes = to_u32(
            2 * key_head_dim * core::mem::size_of::<f32>(),
            "DeltaNet shared bytes",
        )?;
        let cfg = LaunchConfig {
            grid_dim: (state.value_heads, 1, 1),
            block_dim: (state.value_head_dim, 1, 1),
            shared_mem_bytes,
        };
        let mut launch = self.stream.launch_builder(&self.func_deltanet_step);
        launch
            .arg(&mut state.staging)
            .arg(&d_kk)
            .arg(&d_qq)
            .arg(&d_value)
            .arg(&d_beta)
            .arg(&d_decay)
            .arg(&mut d_core)
            .arg(&state.key_head_dim)
            .arg(&state.value_head_dim)
            .arg(&state.group_size);
        // SAFETY: every operand length is checked against the state geometry
        // above, the grid is one block per head and one thread per value lane,
        // and each thread writes only its own column of its own head.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|error| driver_err("launch DeltaNet recurrent step", &error))?;
        }
        let view = d_core.slice(..lane_len);
        self.stream
            .memcpy_dtoh(&view, core_out)
            .map_err(|error| driver_err("download DeltaNet core", &error))
    }
}

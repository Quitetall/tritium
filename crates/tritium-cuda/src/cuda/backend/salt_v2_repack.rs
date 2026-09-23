//! Load-time repack of a resident SALT V2 B3 tensor into the "D0X" layout: plane 0
//! of every tile dense and row-major, the remaining planes in a per-row list.
//!
//! The ragged B3 layout makes a GEMV do per-row rank and tile-table setup, and
//! defeats reusing a lane's activations across rows (lanes diverge on plane
//! counts). Every tile has a plane 0, so that part is uniform; see the D0X block in
//! `kernels/salt_v2.cu`. The bytes are the B3 bytes unchanged -- only their order
//! differs -- so a repacked tensor holds exactly the original weights.

use super::salt_v2_runtime::salt_v2_stream_dispatch;
use super::*;
use tritium_format::salt_v2_package::SALT_V2_ALLOCATION_TILE_SIZE;

/// Rows per block of `salt_v2_d0x_f32` (its `kD0xRows`).
pub(super) const D0X_ROWS: usize = 4;
/// Warps per block of `salt_v2_d0x_f32` (its `kD0xWarps`).
const D0X_WARPS: u32 = 8;
/// `u64` words per `D0xTensor` descriptor.
pub(super) const D0X_DESCRIPTOR_WORDS: usize = 9;
/// A plane goes dense when at least this share of tiles carries it; the rest of
/// its tiles are zero-padded. `TRITIUM_D0X_DENSE_COVERAGE` overrides, for tuning.
const D0X_DENSE_COVERAGE: f64 = 0.75;

/// A SALT V2 B3 tensor repacked for the D0X GEMV.
pub(crate) struct SaltV2D0x {
    dense_payload: CudaSlice<u32>,
    dense_scales: CudaSlice<u32>,
    row_ptr: CudaSlice<u32>,
    extra_tile: CudaSlice<u8>,
    extra_payload: CudaSlice<u32>,
    extra_scales: CudaSlice<u32>,
    pub(super) rows: usize,
    /// Planes stored dense (1-3).
    pub(super) dense_planes: u32,
}

/// A fused D0X descriptor for `tensor`, writing rows to `output`.
pub(super) fn d0x_descriptor(
    tensor: &SaltV2D0x,
    stream: &CudaStream,
    output: sys::CUdeviceptr,
    first_row: u32,
) -> Result<[u64; D0X_DESCRIPTOR_WORDS], BackendError> {
    let rows = u32::try_from(tensor.rows)
        .map_err(|_| BackendError::InvalidInput("D0X rows exceed the u32 kernel ABI".into()))?;
    Ok([
        crate::cuda::graph_raw::dptr(&tensor.dense_payload, stream),
        crate::cuda::graph_raw::dptr(&tensor.dense_scales, stream),
        crate::cuda::graph_raw::dptr(&tensor.row_ptr, stream),
        crate::cuda::graph_raw::dptr(&tensor.extra_tile, stream),
        crate::cuda::graph_raw::dptr(&tensor.extra_payload, stream),
        crate::cuda::graph_raw::dptr(&tensor.extra_scales, stream),
        output,
        u64::from(rows) | (u64::from(first_row) << 32),
        u64::from(tensor.dense_planes),
    ])
}

/// Warps per row group for a D0X GEMV over `total_rows` rows: enough that the
/// launch has ~8K warps (a 5120-row projection at one warp per four rows has only
/// 1280, 10 per SM), and no more, since every extra warp per group adds a
/// shared-memory reduction. `TRITIUM_D0X_GROUP_WARPS` overrides, for tuning.
pub(super) fn d0x_group_warps(total_rows: u32) -> u32 {
    if let Some(warps) = std::env::var("TRITIUM_D0X_GROUP_WARPS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|warps| [1, 2, 4, 8].contains(warps))
    {
        return warps;
    }
    let groups = total_rows.div_ceil(D0X_ROWS as u32).max(1);
    let mut warps = 1;
    while warps < D0X_WARPS && groups * warps < 8192 {
        warps *= 2;
    }
    warps
}

/// Launch the fused D0X GEMV (m = 1) over `tensor_count` descriptors whose rows
/// are all multiples of [`D0X_ROWS`].
///
/// # Errors
/// Returns a driver failure.
pub(super) fn launch_salt_v2_d0x_on(
    stream: &CudaStream,
    function: &CudaFunction,
    descriptors: &CudaSlice<u64>,
    tensor_count: u32,
    total_rows: u32,
    k: u32,
    input: &CudaSlice<f32>,
) -> Result<(), BackendError> {
    let group_warps = d0x_group_warps(total_rows);
    let groups_per_block = D0X_WARPS / group_warps;
    let cfg = LaunchConfig {
        grid_dim: (
            total_rows
                .div_ceil(D0X_ROWS as u32)
                .div_ceil(groups_per_block),
            1,
            1,
        ),
        block_dim: (D0X_WARPS * 32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(function);
    launch
        .arg(input)
        .arg(descriptors)
        .arg(&tensor_count)
        .arg(&total_rows)
        .arg(&k)
        .arg(&group_warps);
    // SAFETY: every descriptor was built from a repack this backend owns and an
    // output buffer sized for its rows; each warp writes its own rows once.
    #[allow(unsafe_code)]
    unsafe {
        launch
            .launch(cfg)
            .map(|_| ())
            .map_err(|error| driver_err("launch SALT V2 D0X forward", &error))
    }
}

impl CudaBackend {
    /// Repack a resident B3 tensor into the D0X layout on the device.
    ///
    /// # Errors
    /// Rejects a tensor the row-stream kernels cannot serve, one whose rows are not
    /// a multiple of the GEMV's rows per warp, or malformed allocation metadata;
    /// or returns a driver failure.
    pub(crate) fn repack_salt_v2_d0x(
        &self,
        tensor: &SaltV2ResidentTensor,
    ) -> Result<SaltV2D0x, BackendError> {
        self.validate_salt_v2_resident_context(tensor)?;
        salt_v2_stream_dispatch(tensor.columns, tensor.scale_group_size, tensor.codec_tag)
            .ok_or_else(|| {
                BackendError::InvalidInput("D0X repack needs a row-stream B3 tensor".into())
            })?;
        if !tensor.rows.is_multiple_of(D0X_ROWS) {
            return Err(BackendError::InvalidInput(format!(
                "D0X repack needs rows in multiples of {D0X_ROWS}, got {}",
                tensor.rows
            )));
        }
        let narrow = |value: usize, name: &str| {
            u32::try_from(value).map_err(|_| {
                BackendError::InvalidInput(format!("{name} exceeds the u32 kernel ABI"))
            })
        };
        let rows = narrow(tensor.rows, "SALT V2 rows")?;
        let k = narrow(tensor.columns, "SALT V2 columns")?;
        let tile_count = narrow(tensor.tile_count, "SALT V2 tile count")?;
        let index_metadata = tensor.index_metadata.as_ref().unwrap_or(&tensor.payload);
        let tiles = tensor.rows * tensor.columns / SALT_V2_ALLOCATION_TILE_SIZE;
        let alloc = |len: usize, what: &str| {
            self.stream
                .alloc_zeros::<u32>(len.max(1))
                .map_err(|error| alloc_or_backend(what, &error, len * 4))
        };
        let cfg = LaunchConfig {
            grid_dim: (rows.div_ceil(SALT_V2_STREAM_WARPS), 1, 1),
            block_dim: (SALT_V2_STREAM_WARPS * 32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut two_or_more = alloc(tensor.rows, "allocate D0X plane counts")?;
        let mut three = alloc(tensor.rows, "allocate D0X plane counts")?;
        let mut bad = alloc(1, "allocate D0X flag")?;
        let mut launch = self.stream.launch_builder(&self.func_salt_v2_repack_count);
        launch
            .arg(index_metadata)
            .arg(&mut two_or_more)
            .arg(&mut three)
            .arg(&mut bad)
            .arg(&rows)
            .arg(&k)
            .arg(&tile_count)
            .arg(&tensor.allocation_map_bytes)
            .arg(&tensor.terminal_map_value);
        // SAFETY: validated resident handle; one slot per row in each count, one flag.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|error| driver_err("launch D0X count", &error))?;
        }
        let read = |slice: &CudaSlice<u32>, what: &str| {
            self.stream
                .clone_dtoh(slice)
                .map_err(|error| driver_err(what, &error))
        };
        let two_host = read(&two_or_more, "read D0X plane counts")?;
        let three_host = read(&three, "read D0X plane counts")?;
        let bad_host = read(&bad, "read D0X flag")?;
        if bad_host[0] != 0 {
            return Err(BackendError::InvalidInput(format!(
                "D0X repack: {} rows have malformed allocation metadata",
                bad_host[0]
            )));
        }
        // Planes 2 and 3 go dense when enough tiles carry them to pay for padding.
        let threshold = std::env::var("TRITIUM_D0X_DENSE_COVERAGE")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(D0X_DENSE_COVERAGE);
        let total = |counts: &[u32]| counts.iter().map(|&c| u64::from(c)).sum::<u64>() as f64;
        let coverage_two = total(&two_host) / tiles as f64;
        let coverage_three = total(&three_host) / tiles as f64;
        let dense_planes: u32 = if coverage_two < threshold {
            1
        } else if coverage_three < threshold {
            2
        } else {
            3
        };
        let mut row_ptr_host = Vec::with_capacity(tensor.rows + 1);
        let mut running = 0u32;
        row_ptr_host.push(0);
        for (two, three) in two_host.iter().zip(&three_host) {
            let extras = match dense_planes {
                1 => two + three,
                2 => *three,
                _ => 0,
            };
            running = running
                .checked_add(extras)
                .ok_or_else(|| BackendError::InvalidInput("D0X extras overflow u32".into()))?;
            row_ptr_host.push(running);
        }
        let extras = running as usize;
        let row_ptr = self
            .stream
            .clone_htod(&row_ptr_host)
            .map_err(|error| driver_err("upload D0X row pointers", &error))?;
        let dense = dense_planes as usize;
        let mut dense_payload = alloc(dense * tiles * 13, "allocate D0X dense payload")?;
        let mut dense_scales = alloc(dense * tiles, "allocate D0X dense scales")?;
        let mut extra_tile = self
            .stream
            .alloc_zeros::<u8>(extras.max(1))
            .map_err(|error| alloc_or_backend("allocate D0X extra tiles", &error, extras))?;
        let mut extra_payload = alloc(extras * 13, "allocate D0X extra payload")?;
        let mut extra_scales = alloc(extras, "allocate D0X extra scales")?;
        let mut launch = self.stream.launch_builder(&self.func_salt_v2_repack_write);
        launch
            .arg(&tensor.payload)
            .arg(&tensor.scales)
            .arg(index_metadata)
            .arg(&row_ptr)
            .arg(&mut dense_payload)
            .arg(&mut dense_scales)
            .arg(&mut extra_tile)
            .arg(&mut extra_payload)
            .arg(&mut extra_scales)
            .arg(&rows)
            .arg(&k)
            .arg(&tensor.allocation_map_bytes)
            .arg(&tensor.rank_prefix_count)
            .arg(&tensor.terminal_map_value)
            .arg(&dense_planes);
        // SAFETY: the outputs were sized from the counts pass 1 took over the same
        // metadata, and each plane-tile is written to exactly one slot.
        #[allow(unsafe_code)]
        unsafe {
            launch
                .launch(cfg)
                .map_err(|error| driver_err("launch D0X write", &error))?;
        }
        Ok(SaltV2D0x {
            dense_payload,
            dense_scales,
            row_ptr,
            extra_tile,
            extra_payload,
            extra_scales,
            rows: tensor.rows,
            dense_planes,
        })
    }

    /// Repack `tensor` and run the D0X GEMV on one activation row, for parity gates.
    ///
    /// # Errors
    /// As [`Self::repack_salt_v2_d0x`], a wrong activation length, or a driver
    /// failure.
    pub fn salt_v2_forward_d0x_probe(
        &self,
        tensor: &SaltV2ResidentTensor,
        activation: &[f32],
    ) -> Result<Vec<f32>, BackendError> {
        if activation.len() != tensor.columns {
            return Err(BackendError::ShapeMismatch {
                expected: tensor.columns,
                got: activation.len(),
            });
        }
        let repacked = self.repack_salt_v2_d0x(tensor)?;
        let input = self
            .stream
            .clone_htod(activation)
            .map_err(|error| driver_err("upload D0X probe input", &error))?;
        let output = self
            .stream
            .alloc_zeros::<f32>(tensor.rows)
            .map_err(|error| alloc_or_backend("allocate D0X probe output", &error, tensor.rows))?;
        let words = d0x_descriptor(
            &repacked,
            &self.stream,
            crate::cuda::graph_raw::dptr(&output, &self.stream),
            0,
        )?;
        let descriptors = self
            .stream
            .clone_htod(&words)
            .map_err(|error| driver_err("upload D0X probe descriptor", &error))?;
        launch_salt_v2_d0x_on(
            &self.stream,
            &self.func_salt_v2_d0x,
            &descriptors,
            1,
            u32::try_from(tensor.rows)
                .map_err(|_| BackendError::InvalidInput("rows exceed u32".into()))?,
            u32::try_from(tensor.columns)
                .map_err(|_| BackendError::InvalidInput("columns exceed u32".into()))?,
            &input,
        )?;
        self.stream
            .clone_dtoh(&output)
            .map_err(|error| driver_err("read D0X probe output", &error))
    }
}

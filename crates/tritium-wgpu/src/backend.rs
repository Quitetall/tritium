//! The wgpu (Vulkan) ternary mpGEMM backend: host-unpack weights, upload to a
//! storage buffer, run the WGSL kernel (`mpgemm.wgsl`), read back.

use core::any::Any;

use pollster::FutureExt as _;
use wgpu::util::DeviceExt as _;

use tritium_core::{GemmShape, TernaryFormat, Trit};
use tritium_format::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
};
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, MpGemm, TernaryBackend};

/// Workgroup size for the 1-D flattened-output dispatch (must match the WGSL
/// `@workgroup_size`).
const WG_SIZE: u32 = 64;

/// `[M, N, K]` dims + the 2-D dispatch x-extent, passed to the shader as a 16-byte
/// std140 uniform.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    m: u32,
    n: u32,
    k: u32,
    /// `workgroups_x * WG_SIZE`: the shader flattens `(gid.x, gid.y)` to a linear
    /// output index as `gid.y * lane_stride + gid.x`.
    lane_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PointwiseParams {
    len: u32,
    operation: u32,
    scalar: f32,
    padding: u32,
}

// std140 uniform structs round up to a 16-byte multiple; pin both the size AND
// the field offsets so a field reorder (which preserves size) can't silently land
// `n` where the shader reads `k`.
const _: () = assert!(core::mem::size_of::<Dims>().is_multiple_of(16));
const _: () = assert!(core::mem::offset_of!(Dims, m) == 0);
const _: () = assert!(core::mem::offset_of!(Dims, n) == 4);
const _: () = assert!(core::mem::offset_of!(Dims, k) == 8);
const _: () = assert!(core::mem::offset_of!(Dims, lane_stride) == 12);

/// Device buffer: weights host-unpacked + widened to `i32`, resident in a Vulkan
/// storage buffer, plus the `[N, K]` dims and the original packed byte count.
#[derive(Debug)]
pub struct WgpuBuffer {
    weights: wgpu::Buffer, // array<i32> [N*K], {-1,0,1}, stride 4, std430
    n: usize,
    k: usize,
    bytes: usize, // original packed byte count, for len_bytes()
}

impl DeviceBuffer for WgpuBuffer {
    fn len_bytes(&self) -> usize {
        self.bytes
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Cross-platform GPU backend: WGSL ternary mpGEMM over wgpu (Vulkan).
///
/// The device, queue, and compiled pipeline are built once in [`WgpuBackend::new`]
/// and shared across calls. `upload_weights`/`mpgemm` are synchronous (the trait
/// is sync); the only async work — adapter/device acquisition and buffer mapping —
/// is driven to completion with `pollster::block_on` / `device.poll`.
#[derive(Debug)]
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pointwise_pipeline: wgpu::ComputePipeline,
    pointwise_bind_group_layout: wgpu::BindGroupLayout,
    device_name: String,
}

/// Packed bytes per block for a format this backend supports.
fn block_bytes(format: TernaryFormat) -> Result<usize, BackendError> {
    match format {
        TernaryFormat::Tq2_0 => Ok(TQ2_0_BLOCK_BYTES),
        TernaryFormat::Tq1_0 => Ok(TQ1_0_BLOCK_BYTES),
        other => Err(BackendError::UnsupportedFormat(other)),
    }
}

/// Select a Vulkan adapter, preferring the discrete NVIDIA GPU.
///
/// `PowerPreference` alone is unreliable on Linux/Mesa, so we enumerate and pick
/// explicitly: honor `TRITIUM_WGPU_ADAPTER` (substring of the adapter name), then
/// prefer NVIDIA discrete (vendor `0x10DE`), then any discrete GPU, then the first.
fn pick_adapter(instance: &wgpu::Instance) -> Result<wgpu::Adapter, BackendError> {
    // `wgpu::Adapter` is not `Clone`, so select an index then move it out.
    let mut adapters = instance.enumerate_adapters(wgpu::Backends::VULKAN);
    if adapters.is_empty() {
        return Err(BackendError::Backend("no Vulkan adapter".into()));
    }
    for a in &adapters {
        let i = a.get_info();
        eprintln!(
            "tritium-wgpu adapter: name={:?} type={:?} backend={:?} vendor=0x{:04x}",
            i.name, i.device_type, i.backend, i.vendor
        );
    }
    const NVIDIA: u32 = 0x10DE;
    let want = std::env::var("TRITIUM_WGPU_ADAPTER").ok();
    let idx = want
        .as_ref()
        .and_then(|w| adapters.iter().position(|a| a.get_info().name.contains(w)))
        .or_else(|| {
            adapters.iter().position(|a| {
                let i = a.get_info();
                i.device_type == wgpu::DeviceType::DiscreteGpu && i.vendor == NVIDIA
            })
        })
        .or_else(|| {
            adapters
                .iter()
                .position(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        })
        .unwrap_or(0);
    Ok(adapters.swap_remove(idx))
}

impl WgpuBackend {
    /// Acquire a Vulkan adapter + device and compile the mpGEMM pipeline.
    ///
    /// # Errors
    /// [`BackendError::Backend`] if no Vulkan adapter is present or device
    /// creation fails — the registry logs and skips it, and the conformance test
    /// self-skips.
    pub fn new() -> Result<Self, BackendError> {
        async fn init() -> Result<WgpuBackend, BackendError> {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN,
                ..Default::default()
            });
            let adapter = pick_adapter(&instance)?;
            let info = adapter.get_info();
            let device_name = format!("{} ({:?})", info.name, info.device_type);

            // Request the adapter's real limits (not the conservative 128 MiB /
            // 65535-workgroup defaults), so production-scale GEMMs are not capped
            // far below the device's true capacity (e.g. the 4090's 24 GB).
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("tritium-wgpu"),
                        required_features: wgpu::Features::empty(),
                        required_limits: adapter.limits(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
                .map_err(|e| BackendError::Backend(format!("request_device: {e}")))?;

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mpgemm"),
                source: wgpu::ShaderSource::Wgsl(include_str!("mpgemm.wgsl").into()),
            });

            let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            let ro = wgpu::BufferBindingType::Storage { read_only: true };
            let rw = wgpu::BufferBindingType::Storage { read_only: false };
            let uni = wgpu::BufferBindingType::Uniform;
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mpgemm-bgl"),
                    entries: &[
                        entry(0, uni),
                        entry(1, ro),
                        entry(2, ro),
                        entry(3, ro),
                        entry(4, rw),
                    ],
                });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mpgemm-pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mpgemm-pipe"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            let pointwise_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-pointwise"),
                source: wgpu::ShaderSource::Wgsl(include_str!("pointwise.wgsl").into()),
            });
            let pointwise_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("portable-pointwise-bgl"),
                    entries: &[entry(0, uni), entry(1, ro), entry(2, ro), entry(3, rw)],
                });
            let pointwise_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("portable-pointwise-pl"),
                bind_group_layouts: &[&pointwise_bind_group_layout],
                push_constant_ranges: &[],
            });
            let pointwise_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("portable-pointwise-pipe"),
                    layout: Some(&pointwise_layout),
                    module: &pointwise_shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

            Ok(WgpuBackend {
                device,
                queue,
                pipeline,
                bind_group_layout,
                pointwise_pipeline,
                pointwise_bind_group_layout,
                device_name,
            })
        }
        init().block_on()
    }

    /// Physical adapter identity used to bind portable-training receipts.
    pub(crate) fn physical_device(&self) -> &str {
        &self.device_name
    }

    /// Run one frozen portable pointwise opcode on resident storage buffers.
    pub(crate) fn pointwise(
        &self,
        left: &[f32],
        right: &[f32],
        operation: u32,
        scalar: f32,
    ) -> Result<Vec<f32>, BackendError> {
        if left.len() != right.len() || left.len() > u32::MAX as usize {
            return Err(BackendError::ShapeMismatch {
                expected: left.len(),
                got: right.len(),
            });
        }
        if left.is_empty() {
            return Ok(Vec::new());
        }
        let params = PointwiseParams {
            len: left.len() as u32,
            operation,
            scalar,
            padding: 0,
        };
        let usage = wgpu::BufferUsages::STORAGE;
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-pointwise-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let left_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-pointwise-left"),
                contents: bytemuck::cast_slice(left),
                usage,
            });
        let right_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-pointwise-right"),
                contents: bytemuck::cast_slice(right),
                usage,
            });
        let bytes = core::mem::size_of_val(left) as u64;
        let result_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-pointwise-result"),
            size: bytes,
            usage: usage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-pointwise-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portable-pointwise-bg"),
            layout: &self.pointwise_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: left_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: right_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result_buf.as_entire_binding(),
                },
            ],
        });
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("portable-pointwise-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("portable-pointwise-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pointwise_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((left.len() as u32).div_ceil(WG_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&result_buf, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        if let Some(error) = self.device.pop_error_scope().block_on() {
            return Err(BackendError::Backend(format!(
                "wgpu pointwise device error: {error}"
            )));
        }
        rx.recv()
            .map_err(|error| BackendError::Backend(format!("map channel: {error}")))?
            .map_err(|error| BackendError::Backend(format!("buffer map: {error}")))?;
        let mut result = vec![0.0_f32; left.len()];
        {
            let data = slice.get_mapped_range();
            result.copy_from_slice(bytemuck::cast_slice(&data));
        }
        staging.unmap();
        Ok(result)
    }
}

impl TernaryBackend for WgpuBackend {
    fn device_id(&self) -> &str {
        "wgpu"
    }

    fn capabilities(&self) -> DeviceCaps {
        // No fp8/IMMA; the fused W1.58A8 path degrades via the trait default.
        DeviceCaps::new("wgpu", self.device_name.clone()).with_features(vec!["vulkan".to_owned()])
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

        // Host-unpack both formats to a flat trit buffer (block scale fixed to
        // 1.0 by the packer → discarded), then widen to i32 for std430.
        let mut trits = vec![Trit::ZERO; n * k];
        let mut scratch = vec![half::f16::ONE; nb];
        for ni in 0..n {
            let row = &packed[ni * row_bytes..(ni + 1) * row_bytes];
            let trits_row = &mut trits[ni * k..ni * k + k];
            let res = match format {
                TernaryFormat::Tq2_0 => unpack_tq2_0_row(row, trits_row, &mut scratch),
                TernaryFormat::Tq1_0 => unpack_tq1_0_row(row, trits_row, &mut scratch),
                other => return Err(BackendError::UnsupportedFormat(other)),
            };
            res.map_err(|e| BackendError::Backend(format!("unpack row {ni}: {e}")))?;
        }
        let widened: Vec<i32> = trits.iter().map(|t| i32::from(t.get())).collect();

        let weights = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("weights"),
                contents: bytemuck::cast_slice(&widened),
                usage: wgpu::BufferUsages::STORAGE,
            });

        Ok(Box::new(WgpuBuffer {
            weights,
            n,
            k,
            bytes: packed.len(),
        }))
    }

    fn mpgemm(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
        let MpGemm {
            act,
            weights,
            scales,
            shape,
            format: _format,
            out,
        } = p;
        let buf = weights
            .as_any()
            .downcast_ref::<WgpuBuffer>()
            .ok_or_else(|| BackendError::InvalidInput("buffer is not a WgpuBuffer".to_owned()))?;
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

        // Zero-dim shapes: the reference returns all-zeros (empty output, or an
        // empty K-contraction → every out is scale·0 = 0). Mirror it without
        // touching the GPU — a zero-size buffer binding is a wgpu validation error.
        if m * n == 0 || k == 0 {
            out.fill(0.0);
            return Ok(());
        }
        // The shader indexes in u32 (M*N flattened, K loop). Reject shapes that
        // would silently wrap the cast rather than dispatching a truncated grid.
        // (`m * n` already fits usize — `out.len() == m * n` passed above.)
        if m > u32::MAX as usize
            || n > u32::MAX as usize
            || k > u32::MAX as usize
            || m * n > u32::MAX as usize
        {
            return Err(BackendError::InvalidInput(format!(
                "shape {shape:?} exceeds the wgpu kernel's u32 index range"
            )));
        }

        // 2-D dispatch grid: split the ceil(M*N / 64) workgroups across x and y so
        // we never exceed the device's max-workgroups-per-dimension (65535 even on
        // the 4090 — a real Vulkan limit, not the wgpu default). The shader
        // reconstructs the linear index via `gid.y * lane_stride + gid.x`.
        let max_dim = self.device.limits().max_compute_workgroups_per_dimension;
        let total_groups = ((m * n) as u32).div_ceil(WG_SIZE).max(1);
        let (gx, gy) = if total_groups <= max_dim {
            (total_groups, 1)
        } else {
            (max_dim, total_groups.div_ceil(max_dim))
        };
        let dims = Dims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lane_stride: gx * WG_SIZE,
        };

        // Capture device-side validation errors (oversized binding/dispatch, etc.)
        // for the whole GPU sequence, so they surface as BackendError below rather
        // than firing wgpu's default uncaptured-error panic handler.
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let act_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("act"),
                contents: bytemuck::cast_slice(act),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let scales_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scales"),
                contents: bytemuck::cast_slice(scales),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_bytes = (m as u64) * (n as u64) * core::mem::size_of::<f32>() as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mpgemm-bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dims_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: act_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf.weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scales_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mpgemm-enc"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mpgemm-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        // Map + block until the GPU work + copy complete, then read back.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();

        // Surface any device-side validation error as a BackendError instead of
        // the default uncaptured-error PANIC (the trait forbids panicking on bad
        // input). Checked before reading the staging buffer, which would be stale.
        if let Some(err) = self.device.pop_error_scope().block_on() {
            return Err(BackendError::Backend(format!("wgpu device error: {err}")));
        }

        rx.recv()
            .map_err(|e| BackendError::Backend(format!("map channel: {e}")))?
            .map_err(|e| BackendError::Backend(format!("buffer map: {e}")))?;
        {
            let data = slice.get_mapped_range();
            out.copy_from_slice(bytemuck::cast_slice(&data));
        }
        staging.unmap();
        Ok(())
    }
}

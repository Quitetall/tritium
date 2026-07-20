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
    auxiliary: u32,
    secondary: u32,
    tertiary: u32,
    padding_0: u32,
    padding_1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConcatParams {
    rows: u32,
    part_count: u32,
    total_columns: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EmbeddingParams {
    vocab: u32,
    width: u32,
    sequence: u32,
    operation: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SaltParams {
    rows: u32,
    cols: u32,
    planes: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Frozen scalar configuration for one FSQ dispatch.
pub(crate) struct FsqParams {
    total: u32,
    len: u32,
    bound: u32,
    estimator: u32,
    execution: u32,
    alpha: f32,
    seed_low: u32,
    seed_high: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeParams {
    n_token: u32,
    n_head: u32,
    head_dim: u32,
    inverse: u32,
    theta: f32,
    padding_0: f32,
    padding_1: f32,
    padding_2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxXentParams {
    rows: u32,
    cols: u32,
    execution: u32,
    padding: u32,
    gradient_scale: f32,
    padding_1: f32,
    padding_2: f32,
    padding_3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Validated grouped-convolution geometry for one native dispatch.
pub(crate) struct ConvParams {
    pub(crate) batch: u32,
    pub(crate) c_in: u32,
    pub(crate) c_out: u32,
    pub(crate) input_h: u32,
    pub(crate) input_w: u32,
    pub(crate) kernel_h: u32,
    pub(crate) kernel_w: u32,
    pub(crate) stride_h: u32,
    pub(crate) stride_w: u32,
    pub(crate) dilation_h: u32,
    pub(crate) dilation_w: u32,
    pub(crate) pad_top: u32,
    pub(crate) pad_left: u32,
    pub(crate) groups: u32,
    pub(crate) output_h: u32,
    pub(crate) output_w: u32,
    pub(crate) execution: u32,
    pub(crate) pad_bottom: u32,
    pub(crate) pad_right: u32,
    pub(crate) padding: u32,
}

/// Host-visible grouped-convolution outputs.
pub(crate) struct ConvOutput {
    pub(crate) result: Vec<f32>,
    pub(crate) grad_weight: Vec<f32>,
    pub(crate) grad_scale: Vec<f32>,
}

impl FsqParams {
    /// Build packed FSQ shader parameters after adapter validation.
    pub(crate) fn new(
        total: u32,
        len: u32,
        bound: u32,
        estimator: u32,
        execution: u32,
        alpha: f32,
        seed: u64,
    ) -> Self {
        Self {
            total,
            len,
            bound,
            estimator,
            execution,
            alpha,
            seed_low: seed as u32,
            seed_high: (seed >> 32) as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Precomputed scalar state for one portable AdamW dispatch.
pub(crate) struct AdamWParams {
    len: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    correction1: f32,
    correction2: f32,
    shrink: f32,
    padding_3: f32,
}

/// Validated AdamW scalars, including host-computed bias correction.
pub(crate) struct AdamWScalars {
    pub(crate) learning_rate: f32,
    pub(crate) beta1: f32,
    pub(crate) beta2: f32,
    pub(crate) epsilon: f32,
    pub(crate) correction1: f32,
    pub(crate) correction2: f32,
    pub(crate) shrink: f32,
}

impl AdamWParams {
    /// Build parameters after host-side validation and bias correction.
    pub(crate) fn new(len: u32, scalars: AdamWScalars) -> Self {
        Self {
            len,
            padding_0: 0,
            padding_1: 0,
            padding_2: 0,
            learning_rate: scalars.learning_rate,
            beta1: scalars.beta1,
            beta2: scalars.beta2,
            epsilon: scalars.epsilon,
            correction1: scalars.correction1,
            correction2: scalars.correction2,
            shrink: scalars.shrink,
            padding_3: 0.0,
        }
    }
}

type AdamWOutput = (Vec<f32>, Vec<f32>, Vec<f32>);
type MuonOutput = (Vec<f32>, Vec<f32>);

/// Host-visible state produced after one native int8 AdamW step.
pub(crate) struct Int8AdamWOutput {
    pub(crate) parameter: Vec<f32>,
    pub(crate) moment1_q8: Vec<u8>,
    pub(crate) moment2_q8: Vec<u8>,
    pub(crate) moment1_scale: Vec<f32>,
    pub(crate) moment2_scale: Vec<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
/// Validated dimensions and scalars for one portable Muon step.
pub(crate) struct MuonParams {
    len: u32,
    rows: u32,
    cols: u32,
    steps: u32,
    momentum_decay: f32,
    scale: f32,
    shrink: f32,
    padding: f32,
}

impl MuonParams {
    /// Build packed shader parameters.
    pub(crate) fn new(
        rows: u32,
        cols: u32,
        steps: u32,
        momentum_decay: f32,
        scale: f32,
        shrink: f32,
    ) -> Self {
        Self {
            len: rows * cols,
            rows,
            cols,
            steps,
            momentum_decay,
            scale,
            shrink,
            padding: 0.0,
        }
    }
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
    concat_pipeline: wgpu::ComputePipeline,
    concat_bind_group_layout: wgpu::BindGroupLayout,
    embedding_pipeline: wgpu::ComputePipeline,
    embedding_bind_group_layout: wgpu::BindGroupLayout,
    adamw_pipeline: wgpu::ComputePipeline,
    adamw_terms_pipeline: wgpu::ComputePipeline,
    adamw_variance_pipeline: wgpu::ComputePipeline,
    adamw_finish_pipeline: wgpu::ComputePipeline,
    cautious_adamw_pipelines: [wgpu::ComputePipeline; 4],
    adamw_bind_group_layout: wgpu::BindGroupLayout,
    int8_adamw_pipelines: [wgpu::ComputePipeline; 8],
    int8_adamw_bind_group_layout: wgpu::BindGroupLayout,
    muon_pipeline: wgpu::ComputePipeline,
    muon_bind_group_layout: wgpu::BindGroupLayout,
    salt_pipeline: wgpu::ComputePipeline,
    salt_bind_group_layout: wgpu::BindGroupLayout,
    fsq_pipeline: wgpu::ComputePipeline,
    fsq_bind_group_layout: wgpu::BindGroupLayout,
    rope_pipeline: wgpu::ComputePipeline,
    rope_bind_group_layout: wgpu::BindGroupLayout,
    softmax_xent_pipeline: wgpu::ComputePipeline,
    softmax_xent_bind_group_layout: wgpu::BindGroupLayout,
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
                    entries: &[
                        entry(0, uni),
                        entry(1, ro),
                        entry(2, ro),
                        entry(3, ro),
                        entry(4, rw),
                    ],
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
            let concat_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-concat"),
                source: wgpu::ShaderSource::Wgsl(include_str!("concat.wgsl").into()),
            });
            let concat_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("portable-concat-bgl"),
                    entries: &[
                        entry(0, uni),
                        entry(1, ro),
                        entry(2, ro),
                        entry(3, ro),
                        entry(4, rw),
                    ],
                });
            let concat_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("portable-concat-pl"),
                bind_group_layouts: &[&concat_bind_group_layout],
                push_constant_ranges: &[],
            });
            let concat_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("portable-concat-pipe"),
                    layout: Some(&concat_layout),
                    module: &concat_shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
            let embedding_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-embedding"),
                source: wgpu::ShaderSource::Wgsl(include_str!("embedding.wgsl").into()),
            });
            let embedding_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("portable-embedding-bgl"),
                    entries: &[
                        entry(0, uni),
                        entry(1, ro),
                        entry(2, ro),
                        entry(3, ro),
                        entry(4, rw),
                    ],
                });
            let embedding_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("portable-embedding-pl"),
                bind_group_layouts: &[&embedding_bind_group_layout],
                push_constant_ranges: &[],
            });
            let embedding_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("portable-embedding-pipe"),
                    layout: Some(&embedding_layout),
                    module: &embedding_shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
            let adamw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-adamw"),
                source: wgpu::ShaderSource::Wgsl(include_str!("adamw.wgsl").into()),
            });
            let adamw_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("portable-adamw-bgl"),
                    entries: &[
                        entry(0, uni),
                        entry(1, ro),
                        entry(2, ro),
                        entry(3, ro),
                        entry(4, ro),
                        entry(5, rw),
                        entry(6, rw),
                        entry(7, rw),
                        entry(8, rw),
                        entry(9, rw),
                        entry(10, rw),
                    ],
                });
            let adamw_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("portable-adamw-pl"),
                bind_group_layouts: &[&adamw_bind_group_layout],
                push_constant_ranges: &[],
            });
            let adamw_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("portable-adamw-pipe"),
                layout: Some(&adamw_layout),
                module: &adamw_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            let adamw_finish_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-adamw-finish"),
                source: wgpu::ShaderSource::Wgsl(include_str!("adamw_finish.wgsl").into()),
            });
            let adamw_terms_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-adamw-terms"),
                source: wgpu::ShaderSource::Wgsl(include_str!("adamw_terms.wgsl").into()),
            });
            let adamw_terms_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("portable-adamw-terms-pipe"),
                    layout: Some(&adamw_layout),
                    module: &adamw_terms_shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
            let adamw_variance_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-adamw-variance"),
                source: wgpu::ShaderSource::Wgsl(include_str!("adamw_variance.wgsl").into()),
            });
            let adamw_variance_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("portable-adamw-variance-pipe"),
                    layout: Some(&adamw_layout),
                    module: &adamw_variance_shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
            let cautious_shader = |label: &'static str, source: &'static str| {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                })
            };
            let cautious_adamw_pipelines = [
                (
                    "portable-cautious-adamw-mask",
                    include_str!("cautious_adamw_mask.wgsl"),
                ),
                (
                    "portable-cautious-adamw-lr",
                    include_str!("cautious_adamw_lr.wgsl"),
                ),
                (
                    "portable-cautious-adamw-rescale",
                    include_str!("cautious_adamw_rescale.wgsl"),
                ),
                (
                    "portable-cautious-adamw-finish",
                    include_str!("cautious_adamw_finish.wgsl"),
                ),
            ]
            .map(|(label, source)| {
                let shader = cautious_shader(label, source);
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&adamw_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            });
            let adamw_finish_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("portable-adamw-finish-pipe"),
                    layout: Some(&adamw_layout),
                    module: &adamw_finish_shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
            let int8_adamw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-int8-adamw"),
                source: wgpu::ShaderSource::Wgsl(include_str!("int8_adamw.wgsl").into()),
            });
            let int8_adamw_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("portable-int8-adamw-bgl"),
                    entries: &[
                        entry(0, uni),
                        entry(1, rw),
                        entry(2, ro),
                        entry(3, rw),
                        entry(4, rw),
                        entry(5, rw),
                        entry(6, rw),
                        entry(7, rw),
                        entry(8, rw),
                    ],
                });
            let int8_adamw_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("portable-int8-adamw-pl"),
                    bind_group_layouts: &[&int8_adamw_bind_group_layout],
                    push_constant_ranges: &[],
                });
            let int8_adamw_pipelines = [
                "dequantize",
                "square_variance",
                "products",
                "finish_products",
                "finish_variance",
                "update_parameter",
                "reduce_scales",
                "quantize",
            ]
            .map(|entry_point| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: Some(&int8_adamw_layout),
                    module: &int8_adamw_shader,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            });
            let muon_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-muon"),
                source: wgpu::ShaderSource::Wgsl(include_str!("muon.wgsl").into()),
            });
            let muon_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("portable-muon-bgl"),
                    entries: &[
                        entry(0, uni),
                        entry(1, rw),
                        entry(2, ro),
                        entry(3, rw),
                        entry(4, rw),
                    ],
                });
            let muon_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("portable-muon-pl"),
                bind_group_layouts: &[&muon_bind_group_layout],
                push_constant_ranges: &[],
            });
            let muon_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("portable-muon-pipe"),
                layout: Some(&muon_layout),
                module: &muon_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            let salt_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-salt"),
                source: wgpu::ShaderSource::Wgsl(include_str!("salt.wgsl").into()),
            });
            let salt_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("portable-salt-bgl"),
                    entries: &[entry(0, uni), entry(1, ro), entry(2, rw), entry(3, rw)],
                });
            let salt_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("portable-salt-pl"),
                bind_group_layouts: &[&salt_bind_group_layout],
                push_constant_ranges: &[],
            });
            let salt_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("portable-salt-pipe"),
                layout: Some(&salt_layout),
                module: &salt_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            let fsq_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-fsq"),
                source: wgpu::ShaderSource::Wgsl(include_str!("fsq.wgsl").into()),
            });
            let fsq_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("portable-fsq-bgl"),
                    entries: &[
                        entry(0, uni),
                        entry(1, ro),
                        entry(2, ro),
                        entry(3, ro),
                        entry(4, rw),
                    ],
                });
            let fsq_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("portable-fsq-pl"),
                bind_group_layouts: &[&fsq_bind_group_layout],
                push_constant_ranges: &[],
            });
            let fsq_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("portable-fsq-pipe"),
                layout: Some(&fsq_layout),
                module: &fsq_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            let rope_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-rope"),
                source: wgpu::ShaderSource::Wgsl(include_str!("rope.wgsl").into()),
            });
            let rope_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("portable-rope-bgl"),
                    entries: &[entry(0, uni), entry(1, ro), entry(2, ro), entry(3, rw)],
                });
            let rope_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("portable-rope-pl"),
                bind_group_layouts: &[&rope_bind_group_layout],
                push_constant_ranges: &[],
            });
            let rope_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("portable-rope-pipe"),
                layout: Some(&rope_layout),
                module: &rope_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            let softmax_xent_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-softmax-xent"),
                source: wgpu::ShaderSource::Wgsl(include_str!("softmax_xent.wgsl").into()),
            });
            let softmax_xent_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("portable-softmax-xent-bgl"),
                    entries: &[entry(0, uni), entry(1, ro), entry(2, ro), entry(3, rw)],
                });
            let softmax_xent_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("portable-softmax-xent-pl"),
                    bind_group_layouts: &[&softmax_xent_bind_group_layout],
                    push_constant_ranges: &[],
                });
            let softmax_xent_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("portable-softmax-xent-pipe"),
                    layout: Some(&softmax_xent_layout),
                    module: &softmax_xent_shader,
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
                concat_pipeline,
                concat_bind_group_layout,
                embedding_pipeline,
                embedding_bind_group_layout,
                adamw_pipeline,
                adamw_terms_pipeline,
                adamw_variance_pipeline,
                adamw_finish_pipeline,
                cautious_adamw_pipelines,
                adamw_bind_group_layout,
                int8_adamw_pipelines,
                int8_adamw_bind_group_layout,
                muon_pipeline,
                muon_bind_group_layout,
                salt_pipeline,
                salt_bind_group_layout,
                fsq_pipeline,
                fsq_bind_group_layout,
                rope_pipeline,
                rope_bind_group_layout,
                softmax_xent_pipeline,
                softmax_xent_bind_group_layout,
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
        extra: &[f32],
        operation: u32,
        scalar: f32,
        auxiliary: u32,
    ) -> Result<Vec<f32>, BackendError> {
        self.pointwise_sized(
            left,
            right,
            extra,
            operation,
            scalar,
            auxiliary,
            0,
            0,
            left.len(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pointwise_sized(
        &self,
        left: &[f32],
        right: &[f32],
        extra: &[f32],
        operation: u32,
        scalar: f32,
        auxiliary: u32,
        secondary: u32,
        tertiary: u32,
        output_len: usize,
    ) -> Result<Vec<f32>, BackendError> {
        if output_len > u32::MAX as usize {
            return Err(BackendError::InvalidInput(
                "pointwise output exceeds u32".into(),
            ));
        }
        if output_len == 0 {
            return Ok(Vec::new());
        }
        let params = PointwiseParams {
            len: output_len as u32,
            operation,
            scalar,
            auxiliary,
            secondary,
            tertiary,
            padding_0: 0,
            padding_1: 0,
        };
        let usage = wgpu::BufferUsages::STORAGE;
        let dummy = [0.0_f32];
        let left_binding = if left.is_empty() { &dummy } else { left };
        let right_binding = if right.is_empty() { &dummy } else { right };
        let extra_binding = if extra.is_empty() { &dummy } else { extra };
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
                contents: bytemuck::cast_slice(left_binding),
                usage,
            });
        let right_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-pointwise-right"),
                contents: bytemuck::cast_slice(right_binding),
                usage,
            });
        let extra_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-pointwise-extra"),
                contents: bytemuck::cast_slice(extra_binding),
                usage,
            });
        let bytes = (output_len * core::mem::size_of::<f32>()) as u64;
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
                    resource: extra_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
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
            pass.dispatch_workgroups((output_len as u32).div_ceil(WG_SIZE), 1, 1);
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
        let mut result = vec![0.0_f32; output_len];
        {
            let data = slice.get_mapped_range();
            result.copy_from_slice(bytemuck::cast_slice(&data));
        }
        staging.unmap();
        Ok(result)
    }

    pub(crate) fn concat_cols(
        &self,
        parts: &[&[f32]],
        rows: usize,
        lengths: &[usize],
    ) -> Result<Vec<f32>, BackendError> {
        if parts.len() != lengths.len() || parts.is_empty() {
            return Err(BackendError::InvalidInput(
                "concat requires one length per nonempty part list".into(),
            ));
        }
        let total_columns = lengths.iter().try_fold(0_usize, |total, &length| {
            total.checked_add(length).ok_or_else(|| {
                BackendError::InvalidInput("concat column total overflows usize".into())
            })
        })?;
        let output_len = rows.checked_mul(total_columns).ok_or_else(|| {
            BackendError::InvalidInput("concat output size overflows usize".into())
        })?;
        if output_len == 0 {
            return Ok(Vec::new());
        }
        if rows > u32::MAX as usize
            || total_columns > u32::MAX as usize
            || parts.len() > u32::MAX as usize
        {
            return Err(BackendError::InvalidInput(
                "concat geometry exceeds u32".into(),
            ));
        }
        let mut values = Vec::new();
        let mut offsets = Vec::with_capacity(parts.len());
        let mut lengths_u32 = Vec::with_capacity(parts.len());
        for (part, &width) in parts.iter().zip(lengths) {
            let expected_part = rows.checked_mul(width).ok_or_else(|| {
                BackendError::InvalidInput("concat part size overflows usize".into())
            })?;
            if part.len() != expected_part || width > u32::MAX as usize {
                return Err(BackendError::ShapeMismatch {
                    expected: expected_part,
                    got: part.len(),
                });
            }
            offsets.push(u32::try_from(values.len()).map_err(|_| {
                BackendError::InvalidInput("concat staging offset exceeds u32".into())
            })?);
            lengths_u32.push(width as u32);
            values.extend_from_slice(part);
            if values.len() > u32::MAX as usize {
                return Err(BackendError::InvalidInput(
                    "concat flattened input exceeds u32".into(),
                ));
            }
        }
        let params = ConcatParams {
            rows: rows as u32,
            part_count: parts.len() as u32,
            total_columns: total_columns as u32,
            padding: 0,
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-concat-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let values_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-concat-values"),
                contents: bytemuck::cast_slice(&values),
                usage: storage,
            });
        let lengths_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-concat-lengths"),
                contents: bytemuck::cast_slice(&lengths_u32),
                usage: storage,
            });
        let offsets_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-concat-offsets"),
                contents: bytemuck::cast_slice(&offsets),
                usage: storage,
            });
        let bytes = (output_len * core::mem::size_of::<f32>()) as u64;
        let result_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-concat-result"),
            size: bytes,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-concat-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portable-concat-bg"),
            layout: &self.concat_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: values_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lengths_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: result_buf.as_entire_binding(),
                },
            ],
        });
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("portable-concat-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("portable-concat-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.concat_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((output_len as u32).div_ceil(WG_SIZE), 1, 1);
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
                "wgpu concat device error: {error}"
            )));
        }
        rx.recv()
            .map_err(|error| BackendError::Backend(format!("map channel: {error}")))?
            .map_err(|error| BackendError::Backend(format!("buffer map: {error}")))?;
        let mut result = vec![0.0_f32; output_len];
        {
            let data = slice.get_mapped_range();
            result.copy_from_slice(bytemuck::cast_slice(&data));
        }
        staging.unmap();
        Ok(result)
    }

    pub(crate) fn embedding(
        &self,
        weight: &[f32],
        tokens: &[u32],
        gradient: &[f32],
        vocab: usize,
        width: usize,
        backward: bool,
    ) -> Result<Vec<f32>, BackendError> {
        if vocab > u32::MAX as usize
            || width > u32::MAX as usize
            || tokens.len() > u32::MAX as usize
        {
            return Err(BackendError::InvalidInput(
                "embedding geometry exceeds u32".into(),
            ));
        }
        let output_len = if backward {
            vocab.checked_mul(width)
        } else {
            tokens.len().checked_mul(width)
        }
        .ok_or_else(|| BackendError::InvalidInput("embedding output overflows usize".into()))?;
        if output_len == 0 {
            return Ok(Vec::new());
        }
        let params = EmbeddingParams {
            vocab: vocab as u32,
            width: width as u32,
            sequence: tokens.len() as u32,
            operation: u32::from(backward),
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let dummy_f32 = [0.0_f32];
        let dummy_u32 = [0_u32];
        let gradient_binding = if gradient.is_empty() {
            &dummy_f32
        } else {
            gradient
        };
        let token_binding = if tokens.is_empty() {
            &dummy_u32
        } else {
            tokens
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-embedding-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let weight_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-embedding-weight"),
                contents: bytemuck::cast_slice(weight),
                usage: storage,
            });
        let tokens_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-embedding-tokens"),
                contents: bytemuck::cast_slice(token_binding),
                usage: storage,
            });
        let gradient_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-embedding-gradient"),
                contents: bytemuck::cast_slice(gradient_binding),
                usage: storage,
            });
        let bytes = (output_len * core::mem::size_of::<f32>()) as u64;
        let result_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-embedding-result"),
            size: bytes,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-embedding-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portable-embedding-bg"),
            layout: &self.embedding_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tokens_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gradient_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: result_buf.as_entire_binding(),
                },
            ],
        });
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("portable-embedding-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("portable-embedding-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.embedding_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((output_len as u32).div_ceil(WG_SIZE), 1, 1);
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
                "wgpu embedding device error: {error}"
            )));
        }
        rx.recv()
            .map_err(|error| BackendError::Backend(format!("map channel: {error}")))?
            .map_err(|error| BackendError::Backend(format!("buffer map: {error}")))?;
        let mut result = vec![0.0_f32; output_len];
        {
            let data = slice.get_mapped_range();
            result.copy_from_slice(bytemuck::cast_slice(&data));
        }
        staging.unmap();
        Ok(result)
    }

    /// Execute one AdamW update with all state resident through dispatch.
    pub(crate) fn adamw(
        &self,
        parameter: &[f32],
        gradient: &[f32],
        moment1: &[f32],
        moment2: &[f32],
        params: AdamWParams,
        cautious: bool,
    ) -> Result<AdamWOutput, BackendError> {
        let len = parameter.len();
        if len == 0
            || gradient.len() != len
            || moment1.len() != len
            || moment2.len() != len
            || params.len as usize != len
        {
            return Err(BackendError::ShapeMismatch {
                expected: len,
                got: gradient.len(),
            });
        }
        let storage = wgpu::BufferUsages::STORAGE;
        let input = |label, values: &[f32]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(values),
                    usage: storage,
                })
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-adamw-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let parameter_buf = input("portable-adamw-parameter", parameter);
        let gradient_buf = input("portable-adamw-gradient", gradient);
        let moment1_buf = input("portable-adamw-moment1", moment1);
        let moment2_buf = input("portable-adamw-moment2", moment2);
        let bytes = core::mem::size_of_val(parameter) as u64;
        let output = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: storage | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let staging = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let updated_parameter = output("portable-adamw-updated-parameter");
        let updated_moment1 = output("portable-adamw-updated-moment1");
        let updated_moment2 = output("portable-adamw-updated-moment2");
        let scratch1 = output("portable-adamw-scratch1");
        let scratch2 = output("portable-adamw-scratch2");
        let aligned = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-cautious-adamw-aligned"),
                contents: bytemuck::bytes_of(&0_u32),
                usage: storage,
            });
        let parameter_staging = staging("portable-adamw-parameter-staging");
        let moment1_staging = staging("portable-adamw-moment1-staging");
        let moment2_staging = staging("portable-adamw-moment2-staging");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portable-adamw-bg"),
            layout: &self.adamw_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parameter_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gradient_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: moment1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: moment2_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: updated_parameter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: updated_moment1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: updated_moment2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: scratch1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: scratch2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: aligned.as_entire_binding(),
                },
            ],
        });
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("portable-adamw-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("portable-adamw-moments-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.adamw_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((len as u32).div_ceil(WG_SIZE), 1, 1);
        }
        for (label, pipeline) in [
            ("portable-adamw-terms-pass", &self.adamw_terms_pipeline),
            (
                "portable-adamw-variance-pass",
                &self.adamw_variance_pipeline,
            ),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((len as u32).div_ceil(WG_SIZE), 1, 1);
        }
        if cautious {
            for (index, pipeline) in self.cautious_adamw_pipelines.iter().enumerate() {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(match index {
                        0 => "portable-cautious-adamw-mask-pass",
                        1 => "portable-cautious-adamw-lr-pass",
                        2 => "portable-cautious-adamw-rescale-pass",
                        _ => "portable-cautious-adamw-finish-pass",
                    }),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((len as u32).div_ceil(WG_SIZE), 1, 1);
            }
        } else {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("portable-adamw-finish-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.adamw_finish_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((len as u32).div_ceil(WG_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&updated_parameter, 0, &parameter_staging, 0, bytes);
        encoder.copy_buffer_to_buffer(&updated_moment1, 0, &moment1_staging, 0, bytes);
        encoder.copy_buffer_to_buffer(&updated_moment2, 0, &moment2_staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let parameter_slice = parameter_staging.slice(..);
        let moment1_slice = moment1_staging.slice(..);
        let moment2_slice = moment2_staging.slice(..);
        let (parameter_tx, parameter_rx) = std::sync::mpsc::channel();
        let (moment1_tx, moment1_rx) = std::sync::mpsc::channel();
        let (moment2_tx, moment2_rx) = std::sync::mpsc::channel();
        parameter_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = parameter_tx.send(result);
        });
        moment1_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = moment1_tx.send(result);
        });
        moment2_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = moment2_tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        if let Some(error) = self.device.pop_error_scope().block_on() {
            return Err(BackendError::Backend(format!(
                "wgpu AdamW device error: {error}"
            )));
        }
        for received in [parameter_rx.recv(), moment1_rx.recv(), moment2_rx.recv()] {
            received
                .map_err(|error| BackendError::Backend(format!("map channel: {error}")))?
                .map_err(|error| BackendError::Backend(format!("buffer map: {error}")))?;
        }
        let read = |slice: &wgpu::BufferSlice<'_>| {
            let data = slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            result
        };
        let parameter = read(&parameter_slice);
        let moment1 = read(&moment1_slice);
        let moment2 = read(&moment2_slice);
        parameter_staging.unmap();
        moment1_staging.unmap();
        moment2_staging.unmap();
        Ok((parameter, moment1, moment2))
    }

    /// Execute one block-wise int8 AdamW step without host-side tensor math.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn int8_adamw(
        &self,
        parameter: &[f32],
        gradient: &[f32],
        moment1_q8: &[u8],
        moment2_q8: &[u8],
        moment1_scale: &[f32],
        moment2_scale: &[f32],
        params: AdamWParams,
    ) -> Result<Int8AdamWOutput, BackendError> {
        let len = parameter.len();
        let blocks = len.div_ceil(256);
        if len == 0
            || gradient.len() != len
            || moment1_q8.len() != len
            || moment2_q8.len() != len
            || moment1_scale.len() != blocks
            || moment2_scale.len() != blocks
            || params.len as usize != len
        {
            return Err(BackendError::ShapeMismatch {
                expected: len,
                got: gradient.len(),
            });
        }
        let storage = wgpu::BufferUsages::STORAGE;
        let input = |label, contents: &[u8], copy_src| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: storage
                        | if copy_src {
                            wgpu::BufferUsages::COPY_SRC
                        } else {
                            wgpu::BufferUsages::empty()
                        },
                })
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-int8-adamw-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let parameter_buf = input(
            "portable-int8-adamw-parameter",
            bytemuck::cast_slice(parameter),
            true,
        );
        let gradient_buf = input(
            "portable-int8-adamw-gradient",
            bytemuck::cast_slice(gradient),
            false,
        );
        let moment1_codes: Vec<u32> = moment1_q8.iter().map(|&value| u32::from(value)).collect();
        let moment2_codes: Vec<u32> = moment2_q8.iter().map(|&value| u32::from(value)).collect();
        let moment1_buf = input(
            "portable-int8-adamw-moment1",
            bytemuck::cast_slice(&moment1_codes),
            true,
        );
        let moment2_buf = input(
            "portable-int8-adamw-moment2",
            bytemuck::cast_slice(&moment2_codes),
            true,
        );
        let moment1_scale_buf = input(
            "portable-int8-adamw-moment1-scale",
            bytemuck::cast_slice(moment1_scale),
            true,
        );
        let moment2_scale_buf = input(
            "portable-int8-adamw-moment2-scale",
            bytemuck::cast_slice(moment2_scale),
            true,
        );
        let tensor_bytes = core::mem::size_of_val(parameter) as u64;
        let scratch = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: tensor_bytes,
                usage: storage,
                mapped_at_creation: false,
            })
        };
        let scratch1 = scratch("portable-int8-adamw-scratch1");
        let scratch2 = scratch("portable-int8-adamw-scratch2");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portable-int8-adamw-bg"),
            layout: &self.int8_adamw_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parameter_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gradient_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: moment1_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: moment2_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: moment1_scale_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: moment2_scale_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: scratch1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: scratch2.as_entire_binding(),
                },
            ],
        });
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("portable-int8-adamw-encoder"),
            });
        for (index, pipeline) in self.int8_adamw_pipelines.iter().enumerate() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("portable-int8-adamw-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = if index == 6 {
                blocks as u32
            } else {
                (len as u32).div_ceil(WG_SIZE)
            };
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        let staging = |label, size| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let scale_bytes = core::mem::size_of_val(moment1_scale) as u64;
        let parameter_staging = staging("portable-int8-adamw-parameter-staging", tensor_bytes);
        let moment1_staging = staging("portable-int8-adamw-moment1-staging", tensor_bytes);
        let moment2_staging = staging("portable-int8-adamw-moment2-staging", tensor_bytes);
        let moment1_scale_staging =
            staging("portable-int8-adamw-moment1-scale-staging", scale_bytes);
        let moment2_scale_staging =
            staging("portable-int8-adamw-moment2-scale-staging", scale_bytes);
        for (source, destination, size) in [
            (&parameter_buf, &parameter_staging, tensor_bytes),
            (&moment1_buf, &moment1_staging, tensor_bytes),
            (&moment2_buf, &moment2_staging, tensor_bytes),
            (&moment1_scale_buf, &moment1_scale_staging, scale_bytes),
            (&moment2_scale_buf, &moment2_scale_staging, scale_bytes),
        ] {
            encoder.copy_buffer_to_buffer(source, 0, destination, 0, size);
        }
        self.queue.submit(Some(encoder.finish()));
        let staging_buffers = [
            &parameter_staging,
            &moment1_staging,
            &moment2_staging,
            &moment1_scale_staging,
            &moment2_scale_staging,
        ];
        let receivers: Vec<_> = staging_buffers
            .iter()
            .map(|buffer| {
                let (tx, rx) = std::sync::mpsc::channel();
                buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        let _ = tx.send(result);
                    });
                rx
            })
            .collect();
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        if let Some(error) = self.device.pop_error_scope().block_on() {
            return Err(BackendError::Backend(format!(
                "wgpu int8 AdamW device error: {error}"
            )));
        }
        for receiver in receivers {
            receiver
                .recv()
                .map_err(|error| BackendError::Backend(format!("map channel: {error}")))?
                .map_err(|error| BackendError::Backend(format!("buffer map: {error}")))?;
        }
        let read = |buffer: &wgpu::Buffer| {
            let data = buffer.slice(..).get_mapped_range();
            let bytes = data.to_vec();
            drop(data);
            bytes
        };
        let parameter = bytemuck::cast_slice(&read(&parameter_staging)).to_vec();
        let moment1_words: Vec<u32> = bytemuck::cast_slice(&read(&moment1_staging)).to_vec();
        let moment2_words: Vec<u32> = bytemuck::cast_slice(&read(&moment2_staging)).to_vec();
        let moment1_scale = bytemuck::cast_slice(&read(&moment1_scale_staging)).to_vec();
        let moment2_scale = bytemuck::cast_slice(&read(&moment2_scale_staging)).to_vec();
        for buffer in staging_buffers {
            buffer.unmap();
        }
        Ok(Int8AdamWOutput {
            parameter,
            moment1_q8: moment1_words.into_iter().map(|value| value as u8).collect(),
            moment2_q8: moment2_words.into_iter().map(|value| value as u8).collect(),
            moment1_scale,
            moment2_scale,
        })
    }

    /// Execute one Muon step with momentum and Newton-Schulz workspace on device.
    pub(crate) fn muon(
        &self,
        parameter: &[f32],
        gradient: &[f32],
        momentum: &[f32],
        params: MuonParams,
    ) -> Result<MuonOutput, BackendError> {
        let len = parameter.len();
        if len == 0 || gradient.len() != len || momentum.len() != len || params.len as usize != len
        {
            return Err(BackendError::ShapeMismatch {
                expected: len,
                got: gradient.len(),
            });
        }
        let r = params.rows.min(params.cols) as usize;
        let square = r
            .checked_mul(r)
            .ok_or_else(|| BackendError::InvalidInput("Muon workspace overflow".to_owned()))?;
        let workspace_len = len
            .checked_mul(3)
            .and_then(|value| {
                square
                    .checked_mul(3)
                    .and_then(|extra| value.checked_add(extra))
            })
            .and_then(|value| value.checked_add(2))
            .ok_or_else(|| BackendError::InvalidInput("Muon workspace overflow".to_owned()))?;
        let storage = wgpu::BufferUsages::STORAGE;
        let input = |label, values: &[f32], copy_src| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(values),
                    usage: storage
                        | if copy_src {
                            wgpu::BufferUsages::COPY_SRC
                        } else {
                            wgpu::BufferUsages::empty()
                        },
                })
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-muon-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let parameter_buf = input("portable-muon-parameter", parameter, true);
        let gradient_buf = input("portable-muon-gradient", gradient, false);
        let momentum_buf = input("portable-muon-momentum", momentum, true);
        let workspace = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-muon-workspace"),
            size: (workspace_len * core::mem::size_of::<f32>()) as u64,
            usage: storage,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portable-muon-bg"),
            layout: &self.muon_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parameter_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gradient_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: momentum_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: workspace.as_entire_binding(),
                },
            ],
        });
        let bytes = core::mem::size_of_val(parameter) as u64;
        let staging = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let parameter_staging = staging("portable-muon-parameter-staging");
        let momentum_staging = staging("portable-muon-momentum-staging");
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("portable-muon-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("portable-muon-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.muon_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&parameter_buf, 0, &parameter_staging, 0, bytes);
        encoder.copy_buffer_to_buffer(&momentum_buf, 0, &momentum_staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let parameter_slice = parameter_staging.slice(..);
        let momentum_slice = momentum_staging.slice(..);
        let (parameter_tx, parameter_rx) = std::sync::mpsc::channel();
        let (momentum_tx, momentum_rx) = std::sync::mpsc::channel();
        parameter_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = parameter_tx.send(result);
        });
        momentum_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = momentum_tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        if let Some(error) = self.device.pop_error_scope().block_on() {
            return Err(BackendError::Backend(format!(
                "wgpu Muon device error: {error}"
            )));
        }
        for received in [parameter_rx.recv(), momentum_rx.recv()] {
            received
                .map_err(|error| BackendError::Backend(format!("map channel: {error}")))?
                .map_err(|error| BackendError::Backend(format!("buffer map: {error}")))?;
        }
        let read = |slice: &wgpu::BufferSlice<'_>| {
            let data = slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            result
        };
        let parameter = read(&parameter_slice);
        let momentum = read(&momentum_slice);
        parameter_staging.unmap();
        momentum_staging.unmap();
        Ok((parameter, momentum))
    }

    /// Execute grouped Conv1d/Conv2d forward or VJP on device.
    pub(crate) fn convolution(
        &self,
        x: &[f32],
        weight: &[f32],
        scale: &[f32],
        grad_output: &[f32],
        params: ConvParams,
    ) -> Result<ConvOutput, BackendError> {
        let backward = params.execution != 0;
        let result_len = if backward {
            params.batch as usize
                * params.c_in as usize
                * params.input_h as usize
                * params.input_w as usize
        } else {
            params.batch as usize
                * params.c_out as usize
                * params.output_h as usize
                * params.output_w as usize
        };
        let weight_len = weight.len();
        let scale_len = scale.len();
        let storage = wgpu::BufferUsages::STORAGE;
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-conv-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let input = |label, values: &[f32]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(values),
                    usage: storage,
                })
        };
        let x_buf = input("portable-conv-x", x);
        let weight_buf = input("portable-conv-weight", weight);
        let scale_buf = input("portable-conv-scale", scale);
        let grad_output_buf = input("portable-conv-grad-output", grad_output);
        let output = |label, len: usize| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len.max(1) * core::mem::size_of::<f32>()) as u64,
                usage: storage | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let result = output("portable-conv-result", result_len);
        let grad_weight = output("portable-conv-grad-weight", weight_len);
        let grad_scale = output("portable-conv-grad-scale", scale_len);

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("portable-conv"),
                source: wgpu::ShaderSource::Wgsl(include_str!("conv.wgsl").into()),
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
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("portable-conv-bgl"),
                entries: &[
                    entry(0, wgpu::BufferBindingType::Uniform),
                    entry(1, ro),
                    entry(2, ro),
                    entry(3, ro),
                    entry(4, ro),
                    entry(5, rw),
                    entry(6, rw),
                    entry(7, rw),
                ],
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("portable-conv-pl"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("portable-conv-pipe"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portable-conv-bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scale_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grad_output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grad_weight.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: grad_scale.as_entire_binding(),
                },
            ],
        });

        let staging = |label, len: usize| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len.max(1) * core::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let result_staging = staging("portable-conv-result-staging", result_len);
        let weight_staging = staging("portable-conv-weight-staging", weight_len);
        let scale_staging = staging("portable-conv-scale-staging", scale_len);
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("portable-conv-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("portable-conv-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &result,
            0,
            &result_staging,
            0,
            (result_len * core::mem::size_of::<f32>()) as u64,
        );
        if backward {
            encoder.copy_buffer_to_buffer(
                &grad_weight,
                0,
                &weight_staging,
                0,
                core::mem::size_of_val(weight) as u64,
            );
            encoder.copy_buffer_to_buffer(
                &grad_scale,
                0,
                &scale_staging,
                0,
                core::mem::size_of_val(scale) as u64,
            );
        }
        self.queue.submit(Some(encoder.finish()));
        let readback = |buffer: &wgpu::Buffer, len: usize| -> Result<Vec<f32>, BackendError> {
            let slice = buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |mapped| {
                let _ = tx.send(mapped);
            });
            self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
            rx.recv()
                .map_err(|error| BackendError::Backend(format!("map channel: {error}")))?
                .map_err(|error| BackendError::Backend(format!("buffer map: {error}")))?;
            let data = slice.get_mapped_range();
            let values = bytemuck::cast_slice(&data)[..len].to_vec();
            drop(data);
            buffer.unmap();
            Ok(values)
        };
        if let Some(error) = self.device.pop_error_scope().block_on() {
            return Err(BackendError::Backend(format!(
                "wgpu convolution device error: {error}"
            )));
        }
        let result_values = readback(&result_staging, result_len)?;
        let (weight_values, scale_values) = if backward {
            (
                readback(&weight_staging, weight_len)?,
                readback(&scale_staging, scale_len)?,
            )
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(ConvOutput {
            result: result_values,
            grad_weight: weight_values,
            grad_scale: scale_values,
        })
    }

    /// Execute greedy multi-plane SALT reconstruction on device.
    pub(crate) fn salt(
        &self,
        weight: &[f32],
        rows: u32,
        cols: u32,
        planes: u32,
    ) -> Result<Vec<f32>, BackendError> {
        let expected = (rows as usize)
            .checked_mul(cols as usize)
            .ok_or_else(|| BackendError::InvalidInput("SALT shape overflow".to_owned()))?;
        if weight.len() != expected || rows == 0 || cols == 0 || planes == 0 {
            return Err(BackendError::ShapeMismatch {
                expected,
                got: weight.len(),
            });
        }
        let storage = wgpu::BufferUsages::STORAGE;
        let params = SaltParams {
            rows,
            cols,
            planes,
            padding: 0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-salt-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let weight_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-salt-weight"),
                contents: bytemuck::cast_slice(weight),
                usage: storage,
            });
        let residual = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-salt-residual"),
            size: u64::from(cols) * core::mem::size_of::<f32>() as u64,
            usage: storage,
            mapped_at_creation: false,
        });
        let bytes = core::mem::size_of_val(weight) as u64;
        let result = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-salt-result"),
            size: bytes,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-salt-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portable-salt-bg"),
            layout: &self.salt_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: residual.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result.as_entire_binding(),
                },
            ],
        });
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("portable-salt-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("portable-salt-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.salt_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&result, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |mapped| {
            let _ = tx.send(mapped);
        });
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        if let Some(error) = self.device.pop_error_scope().block_on() {
            return Err(BackendError::Backend(format!(
                "wgpu SALT device error: {error}"
            )));
        }
        rx.recv()
            .map_err(|error| BackendError::Backend(format!("map channel: {error}")))?
            .map_err(|error| BackendError::Backend(format!("buffer map: {error}")))?;
        let data = slice.get_mapped_range();
        let values = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(values)
    }

    /// Execute FSQ forward or VJP, including seeded stochastic rounding, on device.
    pub(crate) fn fsq(
        &self,
        x: &[f32],
        upstream: &[f32],
        levels: &[u32],
        params: FsqParams,
    ) -> Result<Vec<f32>, BackendError> {
        if x.is_empty() || upstream.len() != x.len() || params.total as usize != x.len() {
            return Err(BackendError::ShapeMismatch {
                expected: x.len(),
                got: upstream.len(),
            });
        }
        let storage = wgpu::BufferUsages::STORAGE;
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-fsq-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let input = |label, contents: &[u8]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: storage,
                })
        };
        let x_buf = input("portable-fsq-x", bytemuck::cast_slice(x));
        let levels_buf = input("portable-fsq-levels", bytemuck::cast_slice(levels));
        let upstream_buf = input("portable-fsq-upstream", bytemuck::cast_slice(upstream));
        let bytes = core::mem::size_of_val(x) as u64;
        let result = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-fsq-result"),
            size: bytes,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-fsq-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portable-fsq-bg"),
            layout: &self.fsq_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: levels_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: upstream_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: result.as_entire_binding(),
                },
            ],
        });
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("portable-fsq-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("portable-fsq-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fsq_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((x.len() as u32).div_ceil(WG_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&result, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |mapped| {
            let _ = tx.send(mapped);
        });
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        if let Some(error) = self.device.pop_error_scope().block_on() {
            return Err(BackendError::Backend(format!(
                "wgpu FSQ device error: {error}"
            )));
        }
        rx.recv()
            .map_err(|error| BackendError::Backend(format!("map channel: {error}")))?
            .map_err(|error| BackendError::Backend(format!("buffer map: {error}")))?;
        let data = slice.get_mapped_range();
        let values = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(values)
    }

    /// Apply NeoX half-rotated RoPE or its inverse on device.
    pub(crate) fn rope(
        &self,
        x: &[f32],
        positions: &[u32],
        n_head: u32,
        head_dim: u32,
        theta: f32,
        inverse: bool,
    ) -> Result<Vec<f32>, BackendError> {
        let expected = positions
            .len()
            .checked_mul(n_head as usize)
            .and_then(|value| value.checked_mul(head_dim as usize))
            .ok_or_else(|| BackendError::InvalidInput("RoPE shape overflow".to_owned()))?;
        if x.len() != expected || positions.is_empty() || n_head == 0 || head_dim == 0 {
            return Err(BackendError::ShapeMismatch {
                expected,
                got: x.len(),
            });
        }
        let params = RopeParams {
            n_token: positions.len() as u32,
            n_head,
            head_dim,
            inverse: u32::from(inverse),
            theta,
            padding_0: 0.0,
            padding_1: 0.0,
            padding_2: 0.0,
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-rope-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let x_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-rope-x"),
                contents: bytemuck::cast_slice(x),
                usage: storage,
            });
        let positions_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-rope-positions"),
                contents: bytemuck::cast_slice(positions),
                usage: storage,
            });
        let bytes = core::mem::size_of_val(x) as u64;
        let result = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-rope-result"),
            size: bytes,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-rope-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portable-rope-bg"),
            layout: &self.rope_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: positions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result.as_entire_binding(),
                },
            ],
        });
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("portable-rope-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("portable-rope-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rope_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let pairs = (positions.len() as u32) * n_head * (head_dim / 2);
            pass.dispatch_workgroups(pairs.div_ceil(WG_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&result, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |mapped| {
            let _ = tx.send(mapped);
        });
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        if let Some(error) = self.device.pop_error_scope().block_on() {
            return Err(BackendError::Backend(format!(
                "wgpu RoPE device error: {error}"
            )));
        }
        rx.recv()
            .map_err(|error| BackendError::Backend(format!("map channel: {error}")))?
            .map_err(|error| BackendError::Backend(format!("buffer map: {error}")))?;
        let data = slice.get_mapped_range();
        let values = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(values)
    }

    /// Execute mean softmax cross-entropy forward or logits VJP on device.
    pub(crate) fn softmax_xent(
        &self,
        logits: &[f32],
        target: &[f32],
        rows: u32,
        cols: u32,
        gradient_scale: f32,
        backward: bool,
    ) -> Result<Vec<f32>, BackendError> {
        let elements = (rows as usize)
            .checked_mul(cols as usize)
            .ok_or_else(|| BackendError::InvalidInput("softmax xent shape overflow".to_owned()))?;
        if logits.len() != elements || target.len() != elements || elements == 0 {
            return Err(BackendError::ShapeMismatch {
                expected: elements,
                got: logits.len(),
            });
        }
        let params = SoftmaxXentParams {
            rows,
            cols,
            execution: u32::from(backward),
            padding: 0,
            gradient_scale,
            padding_1: 0.0,
            padding_2: 0.0,
            padding_3: 0.0,
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("portable-softmax-xent-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let input = |label, values: &[f32]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(values),
                    usage: storage,
                })
        };
        let logits_buf = input("portable-softmax-xent-logits", logits);
        let target_buf = input("portable-softmax-xent-target", target);
        let output_len = if backward { elements } else { 1 };
        let bytes = (output_len * core::mem::size_of::<f32>()) as u64;
        let result = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-softmax-xent-result"),
            size: bytes,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("portable-softmax-xent-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portable-softmax-xent-bg"),
            layout: &self.softmax_xent_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: logits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: target_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result.as_entire_binding(),
                },
            ],
        });
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("portable-softmax-xent-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("portable-softmax-xent-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.softmax_xent_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&result, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |mapped| {
            let _ = tx.send(mapped);
        });
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        if let Some(error) = self.device.pop_error_scope().block_on() {
            return Err(BackendError::Backend(format!(
                "wgpu softmax xent device error: {error}"
            )));
        }
        rx.recv()
            .map_err(|error| BackendError::Backend(format!("map channel: {error}")))?
            .map_err(|error| BackendError::Backend(format!("buffer map: {error}")))?;
        let data = slice.get_mapped_range();
        let values = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(values)
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

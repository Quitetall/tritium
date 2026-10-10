//! Explicit dense/additive tensor execution without a retained host shadow.

use super::{Dims, WG_SIZE, WgpuBackend};
use core::any::Any;
use pollster::FutureExt as _;
use std::sync::{Arc, Mutex};
use tritium_core::{apply_basis, apply_inverse_basis};
use tritium_format::AdditiveTensor;
use tritium_spec::{
    BackendError, Basis, DeviceBuffer, TensorCaps, TensorExecution, TensorMatmul, TensorView,
    admitted_execution_group,
};
use wgpu::util::DeviceExt as _;

struct SemanticTensor {
    rows: usize,
    cols: usize,
    basis: Basis,
    values: Option<wgpu::Buffer>,
    bytes: usize,
    // Exact backend-instance identity, not merely the same adapter name.
    owner: Arc<()>,
}

impl DeviceBuffer for SemanticTensor {
    fn len_bytes(&self) -> usize {
        self.bytes
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(super) struct SemanticExecutor {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    owner: Arc<()>,
    // Serialize this executor's error scopes and readback on a shared device.
    gpu_calls: Mutex<()>,
}

impl core::fmt::Debug for SemanticExecutor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SemanticExecutor").finish_non_exhaustive()
    }
}

fn invalid(message: &str) -> BackendError {
    BackendError::InvalidInput(message.into())
}

fn product(a: usize, b: usize) -> Result<usize, BackendError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("wgpu semantic geometry overflows"))
}

fn buffer_bytes(backend: &WgpuBackend, count: usize) -> Result<usize, BackendError> {
    let bytes = product(count, core::mem::size_of::<f32>())?;
    let limits = backend.device.limits();
    if count > u32::MAX as usize
        || bytes as u64 > limits.max_buffer_size
        || bytes as u64 > u64::from(limits.max_storage_buffer_binding_size)
    {
        return Err(invalid(
            "wgpu semantic payload exceeds device binding/index limits",
        ));
    }
    Ok(bytes)
}

fn readback(backend: &WgpuBackend, buffer: &wgpu::Buffer) -> Result<Vec<f32>, BackendError> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    backend.device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|e| BackendError::Backend(format!("wgpu semantic map channel: {e}")))?
        .map_err(|e| BackendError::Backend(format!("wgpu semantic buffer map: {e}")))?;
    let values = {
        let data = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, f32>(&data).to_vec()
    };
    buffer.unmap();
    Ok(values)
}

impl SemanticExecutor {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let entries = (0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 3,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("semantic-tensor-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("semantic-tensor-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("semantic-dense-matmul"),
            source: wgpu::ShaderSource::Wgsl(include_str!("semantic.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("semantic-dense-matmul"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout,
            owner: Arc::new(()),
            gpu_calls: Mutex::new(()),
        }
    }

    pub(super) fn caps(
        backend: &WgpuBackend,
        view: TensorView<'_>,
    ) -> Result<Option<TensorCaps>, BackendError> {
        let (rows, cols, execution) = match view {
            TensorView::Dense { rows, cols, .. } => {
                view.decoded_payload_bytes()?;
                (rows, cols, TensorExecution::Native)
            }
            TensorView::Additive(additive) => {
                let layout = additive.layout();
                if !admitted_execution_group(layout.law, layout.group) {
                    return Ok(None);
                }
                AdditiveTensor::validate_view(additive)
                    .map_err(|e| invalid(&format!("wgpu additive capability input: {e:?}")))?;
                (
                    usize::try_from(layout.rows).map_err(|_| invalid("row count overflows"))?,
                    usize::try_from(layout.cols).map_err(|_| invalid("column count overflows"))?,
                    TensorExecution::Emulated,
                )
            }
        };
        if rows != 0 && cols != 0 && (rows > u32::MAX as usize || cols > u32::MAX as usize) {
            return Err(invalid("wgpu semantic dimensions exceed u32"));
        }
        let bytes = buffer_bytes(backend, product(rows, cols)?)?;
        Ok(Some(TensorCaps {
            execution,
            payload_bytes: bytes as u64,
        }))
    }

    fn checked_gpu<R>(
        &self,
        backend: &WgpuBackend,
        requested: usize,
        action: impl FnOnce() -> Result<R, BackendError>,
    ) -> Result<R, BackendError> {
        let _guard = self
            .gpu_calls
            .lock()
            .map_err(|_| BackendError::Backend("wgpu semantic executor lock poisoned".into()))?;
        backend
            .device
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        backend
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let result = action();
        let validation = backend.device.pop_error_scope().block_on();
        let allocation = backend.device.pop_error_scope().block_on();
        if let Some(error) = validation {
            return Err(BackendError::Backend(format!(
                "wgpu semantic validation: {error}"
            )));
        }
        if allocation.is_some() {
            return Err(BackendError::OutOfMemory { requested });
        }
        result
    }

    pub(super) fn upload(
        &self,
        backend: &WgpuBackend,
        view: TensorView<'_>,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let caps = Self::caps(backend, view)?.ok_or(BackendError::UnsupportedTensor)?;
        let mut decoded = Vec::new();
        let (rows, cols, basis, values) = match view {
            TensorView::Dense { rows, cols, values } => (rows, cols, Basis::Identity, values),
            TensorView::Additive(additive) => {
                let layout = additive.layout();
                let rows = layout.rows as usize;
                let cols = layout.cols as usize;
                decoded
                    .try_reserve_exact(product(rows, cols)?)
                    .map_err(|_| BackendError::OutOfMemory {
                        requested: caps.payload_bytes as usize,
                    })?;
                decoded.resize(rows * cols, 0.);
                for (row, values) in decoded.chunks_exact_mut(cols).enumerate() {
                    additive
                        .dequant_row_into(row, values)
                        .map_err(|e| invalid(&format!("wgpu additive expansion: {e:?}")))?;
                }
                if decoded.iter().any(|value| !value.is_finite()) {
                    return Err(invalid(
                        "wgpu additive expansion produced nonfinite weights",
                    ));
                }
                (rows, cols, layout.basis, decoded.as_slice())
            }
        };
        let values = if values.is_empty() {
            None
        } else {
            Some(self.checked_gpu(backend, caps.payload_bytes as usize, || {
                Ok(backend
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("semantic-tensor-weights"),
                        contents: bytemuck::cast_slice(values),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    }))
            })?)
        };
        Ok(Box::new(SemanticTensor {
            rows,
            cols,
            basis,
            values,
            bytes: caps.payload_bytes as usize,
            owner: Arc::clone(&self.owner),
        }))
    }

    fn tensor<'a>(&self, tensor: &'a dyn DeviceBuffer) -> Result<&'a SemanticTensor, BackendError> {
        let tensor = tensor
            .as_any()
            .downcast_ref::<SemanticTensor>()
            .ok_or_else(|| invalid("not a wgpu semantic tensor"))?;
        if !Arc::ptr_eq(&self.owner, &tensor.owner) {
            return Err(invalid("wgpu semantic tensor belongs to another backend"));
        }
        Ok(tensor)
    }

    pub(super) fn matmul(
        &self,
        backend: &WgpuBackend,
        p: TensorMatmul<'_>,
    ) -> Result<(), BackendError> {
        let tensor = self.tensor(p.tensor)?;
        let inputs = product(p.batch, tensor.cols)?;
        let outputs = product(p.batch, tensor.rows)?;
        if p.act.len() != inputs || p.transformed_act.len() != inputs || p.out.len() != outputs {
            return Err(invalid("wgpu semantic matmul shape mismatch"));
        }
        let empty = outputs == 0 || tensor.cols == 0;
        let mut grid = (0, 0);
        let mut output_bytes = 0;
        if !empty {
            buffer_bytes(backend, inputs)?;
            output_bytes = buffer_bytes(backend, outputs)?;
            if p.batch > u32::MAX as usize || outputs > (u32::MAX - (WG_SIZE - 1)) as usize {
                return Err(invalid("wgpu semantic dispatch indices exceed u32"));
            }
            let max = backend
                .device
                .limits()
                .max_compute_workgroups_per_dimension
                .min(u32::MAX / WG_SIZE);
            let groups = (outputs as u32).div_ceil(WG_SIZE);
            grid = (groups.min(max), groups.div_ceil(max));
            if grid.1 > max
                || u64::from(grid.0) * u64::from(grid.1) * u64::from(WG_SIZE)
                    > u64::from(u32::MAX) + 1
            {
                return Err(invalid("wgpu semantic dispatch exceeds device grid limits"));
            }
        }
        // All malformed geometry is rejected before touching caller buffers.
        p.transformed_act.copy_from_slice(p.act);
        if tensor.cols != 0 {
            for activation in p.transformed_act.chunks_exact_mut(tensor.cols) {
                apply_basis(activation, tensor.basis)
                    .map_err(|e| invalid(&format!("wgpu activation basis: {e:?}")))?;
            }
        }
        if empty {
            p.out.fill(0.);
            return Ok(());
        }
        let weights = tensor
            .values
            .as_ref()
            .ok_or_else(|| invalid("wgpu semantic payload absent"))?;
        let result = self.checked_gpu(backend, output_bytes, || {
            let dims = Dims {
                m: p.batch as u32,
                n: tensor.rows as u32,
                k: tensor.cols as u32,
                lane_stride: grid.0 * WG_SIZE,
            };
            let dims = backend
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("semantic-dims"),
                    contents: bytemuck::bytes_of(&dims),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let act = backend
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("semantic-activations"),
                    contents: bytemuck::cast_slice(p.transformed_act),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let output = backend.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("semantic-output"),
                size: output_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = backend.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("semantic-output-readback"),
                size: output_bytes as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let buffers = [&dims, &act, weights, &output];
            let entries = buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>();
            let bindings = backend
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("semantic-matmul-bindings"),
                    layout: &self.layout,
                    entries: &entries,
                });
            let mut encoder =
                backend
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("semantic-matmul"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("semantic-matmul"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups(grid.0, grid.1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_bytes as u64);
            backend.queue.submit(Some(encoder.finish()));
            readback(backend, &staging)
        })?;
        p.out.copy_from_slice(&result);
        Ok(())
    }

    pub(super) fn embed(
        &self,
        backend: &WgpuBackend,
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let tensor = self.tensor(tensor)?;
        let count = product(ids.len(), tensor.cols)?;
        if out.len() != count || ids.iter().any(|&row| row >= tensor.rows) {
            return Err(invalid("wgpu semantic gather shape or row ID is invalid"));
        }
        if out.is_empty() {
            return Ok(());
        }
        let bytes = buffer_bytes(backend, count)?;
        let weights = tensor
            .values
            .as_ref()
            .ok_or_else(|| invalid("wgpu semantic payload absent"))?;
        let mut result = self.checked_gpu(backend, bytes, || {
            let staging = backend.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("semantic-gather-readback"),
                size: bytes as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder =
                backend
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("semantic-gather"),
                    });
            let row_bytes = (tensor.cols * core::mem::size_of::<f32>()) as u64;
            for (index, &row) in ids.iter().enumerate() {
                encoder.copy_buffer_to_buffer(
                    weights,
                    row as u64 * row_bytes,
                    &staging,
                    index as u64 * row_bytes,
                    row_bytes,
                );
            }
            backend.queue.submit(Some(encoder.finish()));
            readback(backend, &staging)
        })?;
        for row in result.chunks_exact_mut(tensor.cols) {
            apply_inverse_basis(row, tensor.basis)
                .map_err(|e| invalid(&format!("wgpu inverse embedding basis: {e:?}")))?;
        }
        out.copy_from_slice(&result);
        Ok(())
    }
}

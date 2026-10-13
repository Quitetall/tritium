//! Opt-in device-f32 region; real shared weights, one reusable bounded workspace.
use super::*;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use tritium_spec::TensorSwiGlu;

#[derive(Debug)]
pub(crate) struct Workspace {
    shape: (usize, usize, usize),
    input: CudaSlice<f32>,
    rotated: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    output: CudaSlice<f32>,
    ones: CudaSlice<f32>,
    host_output: Vec<f32>,
}

impl Workspace {
    fn new(backend: &CudaBackend, m: usize, k: usize, f: usize) -> Result<Self, BackendError> {
        let input = product(m, k)?;
        let hidden = product(m, f)?;
        let alloc = |n| {
            backend
                .stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| alloc_or_backend("SwiGLU workspace", &e, n * 4))
        };
        let mut ones = Vec::new();
        let count = k.max(f);
        ones.try_reserve_exact(count)
            .map_err(|_| BackendError::OutOfMemory {
                requested: count * 4,
            })?;
        ones.resize(count, 1.);
        let ones = backend
            .stream
            .clone_htod(&ones)
            .map_err(|e| alloc_or_backend("SwiGLU unit scales", &e, count * 4))?;
        let mut host_output = Vec::new();
        host_output
            .try_reserve_exact(input)
            .map_err(|_| BackendError::OutOfMemory {
                requested: input * 4,
            })?;
        host_output.resize(input, 0.);
        Ok(Self {
            shape: (m, k, f),
            input: alloc(input)?,
            rotated: alloc(input)?,
            gate: alloc(hidden)?,
            up: alloc(hidden)?,
            output: alloc(input)?,
            ones,
            host_output,
        })
    }
}

fn project(
    backend: &CudaBackend,
    tensor: &CudaTensor,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    ones: &CudaSlice<f32>,
    batch: usize,
) -> Result<(), BackendError> {
    if let Some(packed) = &tensor.packed {
        packed.launch(backend, input, output, batch as u32)
    } else {
        backend.matmul_forward_dev(
            input,
            tensor
                .values
                .as_ref()
                .ok_or_else(|| invalid("SwiGLU dense payload missing"))?,
            ones,
            GemmShape {
                m: batch,
                n: tensor.rows,
                k: tensor.cols,
            },
            output,
        )
    }
}

pub(crate) fn forward(backend: &CudaBackend, p: TensorSwiGlu<'_>) -> Result<bool, BackendError> {
    if !backend.semantic_swiglu_enabled {
        return Ok(false);
    }
    let gate = CudaTensor::checked(backend, p.gate)?;
    let up = CudaTensor::checked(backend, p.up)?;
    let down = CudaTensor::checked(backend, p.down)?;
    let (k, f, m) = (gate.cols, gate.rows, p.batch);
    if k == 0
        || f == 0
        || up.cols != k
        || up.rows != f
        || down.cols != f
        || down.rows != k
        || p.act.len() != product(m, k)?
        || p.out.len() != product(m, k)?
    {
        return Err(invalid("SwiGLU tensor/activation geometry mismatch"));
    }
    if m == 0 {
        return Ok(true);
    }
    CudaBackend::check_grad_launch_bounds(m, f, k)?;
    CudaBackend::check_grad_launch_bounds(m, k, f)?;
    let mut slot = backend
        .semantic_swiglu_workspace
        .lock()
        .map_err(|_| invalid("SwiGLU workspace mutex poisoned"))?;
    if slot.as_ref().is_none_or(|w| w.shape != (m, k, f)) {
        let replacement = Workspace::new(backend, m, k, f)?;
        *slot = Some(replacement);
    }
    let w = slot
        .as_mut()
        .ok_or_else(|| invalid("SwiGLU workspace absent"))?;
    backend
        .stream
        .memcpy_htod(p.act, &mut w.input)
        .map_err(|e| driver_err("SwiGLU input upload", &e))?;
    let mut last_basis = None;
    for (tensor, output) in [(gate, &mut w.gate), (up, &mut w.up)] {
        let input = if tensor.basis == Basis::Identity {
            &w.input
        } else {
            if last_basis != Some(tensor.basis) {
                backend
                    .stream
                    .memcpy_dtod(&w.input, &mut w.rotated)
                    .map_err(|e| driver_err("SwiGLU basis copy", &e))?;
                super::basis::apply(backend, &mut w.rotated, k, tensor.basis, false)?;
                last_basis = Some(tensor.basis);
            }
            &w.rotated
        };
        project(backend, tensor, input, output, &w.ones, m)?;
    }
    let count = u32::try_from(w.gate.len()).map_err(|_| invalid("SwiGLU hidden index overflow"))?;
    let mut launch = backend.stream.launch_builder(&backend.func_semantic_swiglu);
    launch.arg(&mut w.gate).arg(&w.up).arg(&count);
    #[allow(unsafe_code)]
    // SAFETY: validated shape bounds one thread per disjoint gate/up lane.
    unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
        .map_err(|e| driver_err("SwiGLU device activation", &e))?;
    drop(launch);
    super::basis::apply(backend, &mut w.gate, f, down.basis, false)?;
    project(backend, down, &w.gate, &mut w.output, &w.ones, m)?;
    backend
        .stream
        .memcpy_dtoh(&w.output, &mut w.host_output)
        .map_err(|e| driver_err("SwiGLU staged readback", &e))?;
    p.out.copy_from_slice(&w.host_output);
    Ok(true)
}

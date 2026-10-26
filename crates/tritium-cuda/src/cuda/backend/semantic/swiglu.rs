//! Opt-in device-f32 region; real shared weights, geometry-keyed private scratch.
//! Full workspace/model payload admission remains a separate contract.
use super::*;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use tritium_spec::TensorSwiGlu;

pub(crate) struct Workspace {
    shape: (usize, usize, usize),
    input: CudaSlice<f32>,
    rotated: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    output: CudaSlice<f32>,
    ones: CudaSlice<f32>,
    host_output: Vec<f32>,
    invalid: CudaSlice<u32>,
}

impl core::fmt::Debug for Workspace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SwiGluWorkspace")
            .field("shape", &self.shape)
            .field(
                "host_output_capacity_elements",
                &self.host_output.capacity(),
            )
            .field("status_words", &self.invalid.len())
            .finish_non_exhaustive()
    }
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
            invalid: backend
                .stream
                .alloc_zeros::<u32>(1)
                .map_err(|e| alloc_or_backend("SwiGLU private status", &e, 4))?,
        })
    }
}

fn check_finite(
    backend: &CudaBackend,
    values: &CudaSlice<f32>,
    status: &mut CudaSlice<u32>,
) -> Result<(), BackendError> {
    let count = u32::try_from(values.len())
        .map_err(|_| invalid("SwiGLU activation exceeds u32 index ABI"))?;
    if count == 0 {
        return Ok(());
    }
    let mut launch = backend
        .stream
        .launch_builder(&backend.func_semantic_activation_check_finite);
    launch.arg(values).arg(status).arg(&count);
    // SAFETY: checked u32 length and same-context workspace. One reader per
    // value; private status uses atomicOr and is reset on the owning stream.
    #[allow(unsafe_code)]
    unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
        .map_err(|e| driver_err("SwiGLU transformed activation finite check", &e))?;
    Ok(())
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
        packed.launch(
            backend,
            input,
            output,
            batch as u32,
            product(batch, tensor.rows)?,
        )
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
    if p.act.iter().any(|v| !v.is_finite()) {
        return Err(invalid("SwiGLU input contains a non-finite value"));
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
        .memset_zeros(&mut w.invalid)
        .map_err(|e| driver_err("reset SwiGLU private status", &e))?;
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
                check_finite(backend, &w.rotated, &mut w.invalid)?;
                last_basis = Some(tensor.basis);
            }
            &w.rotated
        };
        project(backend, tensor, input, output, &w.ones, m)?;
    }
    let count = u32::try_from(w.gate.len()).map_err(|_| invalid("SwiGLU hidden index overflow"))?;
    let mut launch = backend.stream.launch_builder(&backend.func_semantic_swiglu);
    launch
        .arg(&mut w.gate)
        .arg(&w.up)
        .arg(&mut w.invalid)
        .arg(&count);
    #[allow(unsafe_code)]
    // SAFETY: validated shape bounds one thread per disjoint gate/up lane.
    unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
        .map_err(|e| driver_err("SwiGLU device activation", &e))?;
    drop(launch);
    super::basis::apply(backend, &mut w.gate, f, down.basis, false)?;
    if down.basis != Basis::Identity {
        check_finite(backend, &w.gate, &mut w.invalid)?;
    }
    project(backend, down, &w.gate, &mut w.output, &w.ones, m)?;
    backend
        .stream
        .memcpy_dtoh(&w.output, &mut w.host_output)
        .map_err(|e| driver_err("SwiGLU staged readback", &e))?;
    let mut invalid_values = [0u32];
    backend
        .stream
        .memcpy_dtoh(&w.invalid, &mut invalid_values)
        .map_err(|e| driver_err("SwiGLU private status readback", &e))?;
    backend
        .stream
        .synchronize()
        .map_err(|e| driver_err("SwiGLU readback synchronization", &e))?;
    if invalid_values[0] != 0 || w.host_output.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "SwiGLU computed a non-finite intermediate or output",
        ));
    }
    p.out.copy_from_slice(&w.host_output);
    Ok(true)
}

//! Opt-in device-f32 region; real shared weights, geometry-keyed private scratch.
//! Full workspace/model payload admission remains a separate contract.
use super::*;
use crate::SwiGluWorkspaceKind;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use tritium_spec::TensorSwiGlu;

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
    backend.with_swiglu_workspace(
        SwiGluWorkspaceKind::SharedTensor,
        m,
        k,
        f,
        None,
        |w, _, _| {
            let ones = w
                .ones
                .as_ref()
                .ok_or_else(|| invalid("shared SwiGLU unit scales absent"))?;
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
                        let rotated = w
                            .rotated
                            .as_mut()
                            .ok_or_else(|| invalid("shared SwiGLU rotation scratch absent"))?;
                        backend
                            .stream
                            .memcpy_dtod(&w.input, rotated)
                            .map_err(|e| driver_err("SwiGLU basis copy", &e))?;
                        super::basis::apply(backend, rotated, k, tensor.basis, false)?;
                        check_finite(backend, rotated, &mut w.invalid)?;
                        last_basis = Some(tensor.basis);
                    }
                    w.rotated
                        .as_ref()
                        .ok_or_else(|| invalid("shared SwiGLU rotation scratch absent"))?
                };
                project(backend, tensor, input, output, ones, m)?;
            }
            let count =
                u32::try_from(w.gate.len()).map_err(|_| invalid("SwiGLU hidden index overflow"))?;
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
            project(backend, down, &w.gate, &mut w.output, ones, m)?;
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
        },
    )
}

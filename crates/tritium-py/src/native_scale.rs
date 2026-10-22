//! Explicit numerical native scale training bridge; not child publication.
use crate::{
    model_conversion::NativeModelConversion,
    model_level::NativePackedMatrix,
    torch_native::{checked_product, cpu_f32, export_array_f64, tensor_shape},
};
use dlpark::legacy::Dlpack;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyBytes},
};
use std::sync::Mutex;
use tritium_runtime::TensorPayloadBudget;
use tritium_schema::{NativeRefinementPlan, PlaneCounts, TensorStorage};
use tritium_spec::AdditiveMetadata;
use tritium_train::pv_tuning::{NativeCpuScaleSession, NativeScaleRefinementSession};

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

#[pyfunction]
pub(crate) fn _validate_native_scale_plan(text: &str) -> PyResult<()> {
    if text.len() > 16 * 1024 * 1024 {
        return Err(error("native training plan exceeds byte limit"));
    }
    let plan: NativeRefinementPlan = serde_json::from_str(text).map_err(error)?;
    plan.validate().map_err(|e| error(format!("{e:?}")))?;
    if !matches!(
        plan.recipe,
        tritium_schema::NativeRecoveryRecipe::ScaleOnly { .. }
    ) {
        return Err(error("native Python training requires ScaleOnly"));
    }
    Ok(())
}
fn zeros(len: usize) -> PyResult<Vec<f64>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(error)?;
    out.resize(len, 0.0);
    Ok(out)
}
fn admit_output(matrix: &NativePackedMatrix, indices: usize) -> PyResult<usize> {
    let (_, cols, max) = matrix.dimensions();
    let count = tritium_cpu::CpuBackend::new()
        .additive_stored_scale_count(&*matrix.shared_tensor())
        .map_err(error)?;
    if count
        .checked_mul(8)
        .and_then(|n| n.checked_add(cols.checked_mul(4)?))
        .and_then(|n| n.checked_add(indices.checked_mul(size_of::<usize>())?))
        .is_none_or(|n| n > max)
    {
        return Err(error(
            "native scale adjoint output/scratch exceeds byte limit",
        ));
    }
    Ok(count)
}
pub(crate) fn matmul_vjp(
    matrix: &NativePackedMatrix,
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let x: Dlpack = input.extract()?;
    let dy: Dlpack = gradient.extract()?;
    let xs = tensor_shape(&x, "activation")?;
    let ys = tensor_shape(&dy, "cotangent")?;
    let (rows, cols, _) = matrix.dimensions();
    if xs.last() != Some(&cols)
        || ys.last() != Some(&rows)
        || xs[..xs.len() - 1] != ys[..ys.len() - 1]
    {
        return Err(error("native scale adjoint shape mismatch"));
    }
    let batch = checked_product(&xs[..xs.len() - 1], "batch")?;
    let count = admit_output(matrix, 0)?;
    let x = cpu_f32(&x, "activation")?;
    let dy = cpu_f32(&dy, "cotangent")?;
    let handle = matrix.shared_tensor();
    let values = py.detach(|| {
        let mut out = zeros(count)?;
        let mut row = vec![0.0; cols];
        tritium_cpu::CpuBackend::new()
            .accumulate_additive_matmul_scale_vjp(&*handle, x, dy, batch, &mut row, &mut out)
            .map_err(error)?;
        Ok::<_, PyErr>(out)
    })?;
    export_array_f64(py, &[count], values, "native scale cotangent")
}
pub(crate) fn embed_vjp(
    matrix: &NativePackedMatrix,
    py: Python<'_>,
    indices: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let ids: Dlpack = indices.extract()?;
    let dy: Dlpack = gradient.extract()?;
    let shape = tensor_shape(&ids, "indices")?;
    let grad = tensor_shape(&dy, "cotangent")?;
    let (rows, cols, _) = matrix.dimensions();
    if grad.last() != Some(&cols) || grad[..grad.len() - 1] != shape {
        return Err(error("native embedding scale adjoint shape mismatch"));
    }
    let raw = ids.cpu_data_slice::<i64>().map_err(error)?;
    let count = admit_output(matrix, raw.len())?;
    if raw.iter().any(|&i| i < 0 || i as u64 >= rows as u64) {
        return Err(error("native embedding index out of range"));
    }
    let dy = cpu_f32(&dy, "cotangent")?;
    let handle = matrix.shared_tensor();
    let values = py.detach(|| {
        let ids: Vec<_> = raw.iter().map(|&i| i as usize).collect();
        let mut out = zeros(count)?;
        let mut row = vec![0.0; cols];
        tritium_cpu::CpuBackend::new()
            .accumulate_additive_embed_scale_vjp(&*handle, &ids, dy, &mut row, &mut out)
            .map_err(error)?;
        Ok::<_, PyErr>(out)
    })?;
    export_array_f64(py, &[count], values, "native embedding scale cotangent")
}

#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativeScaleTraining {
    state: Mutex<NativeCpuScaleSession>,
    specs: Vec<NativePackedMatrix>,
}
impl NativeScaleTraining {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, NativeCpuScaleSession>> {
        self.state
            .lock()
            .map_err(|_| error("native training state lock poisoned"))
    }
}

pub(crate) fn begin(
    py: Python<'_>,
    result: &NativeModelConversion,
    plan_json: &str,
    matrices: Vec<(String, Py<NativePackedMatrix>)>,
    max_snapshot_bytes: u64,
    max_parent_bytes: u64,
    checkpoint: Option<&[u8]>,
) -> PyResult<NativeScaleTraining> {
    if plan_json.len() > 16 * 1024 * 1024 || max_parent_bytes == 0 || max_snapshot_bytes == 0 {
        return Err(error("native training plan/byte limits invalid"));
    }
    _validate_native_scale_plan(plan_json)?;
    let plan: NativeRefinementPlan = serde_json::from_str(plan_json).map_err(error)?;
    plan.validate().map_err(|e| error(format!("{e:?}")))?;
    if checkpoint.is_some_and(|b| b.len() as u64 > plan.limits.max_checkpoint_bytes) {
        return Err(error("native checkpoint exceeds byte limit"));
    }
    if matrices.iter().map(|(name, _)| name).ne(plan.owners.iter()) {
        return Err(error("native training owner names/order mismatch"));
    }
    let specs: Vec<_> = matrices
        .iter()
        .map(|(_, m)| m.borrow(py).with_tensor(m.borrow(py).shared_tensor()))
        .collect();
    let roots: Vec<_> = plan
        .owners
        .iter()
        .cloned()
        .zip(specs.iter().map(NativePackedMatrix::shared_tensor))
        .collect();
    let driver = py.detach(|| {
        let (mut reader, parent) = result.refinement_parent(&plan.parent.level)?;
        if parent != plan.parent {
            return Err(error("native training parent tuple differs"));
        }
        plan.validate_parent(reader.manifest(), &parent)
            .map_err(|e| error(format!("{e:?}")))?;
        let mut retained = 0u64;
        let mut largest = 0u64;
        for name in &plan.owners {
            let meta = reader
                .tensor_meta(&plan.parent.level, name)
                .map_err(error)?;
            let TensorStorage::Additive { layout, planes } = &meta.storage else {
                return Err(error("native training owner is not additive"));
            };
            let m = match planes {
                PlaneCounts::Uniform(p) => AdditiveMetadata::uniform(*layout, *p),
                PlaneCounts::PerTile(c) => AdditiveMetadata::per_tile(*layout, c),
                _ => return Err(error("unsupported plane allocation")),
            }
            .map_err(error)?;
            let mut bytes = m.auxiliary_bytes().map_err(error)? as u64;
            for p in 0..m.plane_count() {
                bytes = bytes
                    .checked_add(m.plane_trit_count(p).map_err(error)? as u64)
                    .ok_or_else(|| error("native state size overflow"))?;
            }
            bytes = bytes
                .checked_add(
                    (m.scale_count() as u64)
                        .checked_mul(12)
                        .ok_or_else(|| error("native state size overflow"))?,
                )
                .ok_or_else(|| error("native state size overflow"))?;
            retained = retained
                .checked_add(bytes)
                .ok_or_else(|| error("native state size overflow"))?;
            largest = largest.max(
                reader
                    .tensor_byte_len(&plan.parent.level, name)
                    .map_err(error)?,
            );
        }
        if retained > plan.limits.max_state_bytes
            || retained
                .checked_add(largest)
                .is_none_or(|n| n > max_parent_bytes)
        {
            return Err(error("native decoded parent/state exceeds byte limit"));
        }
        let mut parents = Vec::new();
        for name in &plan.owners {
            let meta = reader
                .tensor_meta(&plan.parent.level, name)
                .map_err(error)?
                .clone();
            let len = usize::try_from(
                reader
                    .tensor_byte_len(&plan.parent.level, name)
                    .map_err(error)?,
            )
            .map_err(error)?;
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(len).map_err(error)?;
            reader
                .visit_tensor(&plan.parent.level, name, |chunk| {
                    bytes.extend_from_slice(chunk);
                    Ok::<_, PyErr>(())
                })
                .map_err(error)?;
            parents.push(
                tritium_format::trit_model::decode_additive_payload(&meta, &bytes)
                    .map_err(error)?,
            );
        }
        reader.verify().map_err(error)?;
        let state = if let Some(bytes) = checkpoint {
            NativeScaleRefinementSession::resume(plan, parents, bytes)
        } else {
            NativeScaleRefinementSession::new(plan, parents)
        }
        .map_err(error)?;
        NativeCpuScaleSession::bind(state, roots, TensorPayloadBudget::new(max_snapshot_bytes))
            .map_err(error)
    })?;
    Ok(NativeScaleTraining {
        state: Mutex::new(driver),
        specs,
    })
}

#[pymethods]
impl NativeScaleTraining {
    #[getter]
    fn completed_updates(&self) -> PyResult<u64> {
        Ok(self.lock()?.session().completed_updates())
    }
    fn frame(
        &self,
        py: Python<'_>,
        owner: &str,
        revision: u64,
        include_scales: bool,
    ) -> PyResult<(NativePackedMatrix, Option<Py<PyAny>>)> {
        let (matrix, scales) = py.detach(|| {
            let state = self.lock()?;
            if state.session().completed_updates() != revision {
                return Err(error("stale native training revision"));
            }
            let i = state
                .session()
                .plan()
                .owners
                .iter()
                .position(|n| n == owner)
                .ok_or_else(|| error("native owner absent"))?;
            let handle = state
                .handles()
                .nth(i)
                .ok_or_else(|| error("native handle absent"))?
                .1;
            let matrix = self.specs[i].with_tensor(handle.clone());
            let scales = if include_scales {
                let source = state.session().weights().nth(i).expect("bound owner");
                if source
                    .scales()
                    .len()
                    .checked_mul(8)
                    .is_none_or(|n| n > self.specs[i].dimensions().2)
                {
                    return Err(error("native scale leaf exceeds output byte limit"));
                }
                let mut values = zeros(source.scales().len())?;
                for (value, &scale) in values.iter_mut().zip(source.scales()) {
                    *value = f64::from(scale);
                }
                Some(values)
            } else {
                None
            };
            Ok::<_, PyErr>((matrix, scales))
        })?;
        let scales = scales
            .map(|s| export_array_f64(py, &[s.len()], s, "native stored scales"))
            .transpose()?;
        Ok((matrix, scales))
    }
    fn step(
        &self,
        py: Python<'_>,
        revision: u64,
        window: Vec<u8>,
        gradients: Vec<Option<Dlpack>>,
    ) -> PyResult<usize> {
        let mut values = Vec::new();
        let mut any = false;
        {
            let state = self.lock()?;
            if state.session().completed_updates() != revision {
                return Err(error("stale native training revision"));
            }
            let expected = state
                .session()
                .plan()
                .training
                .get(revision as usize)
                .ok_or_else(|| error("native training schedule exhausted"))?;
            if window.as_slice() != expected.content {
                return Err(error("native training window differs"));
            }
            if gradients.len() != state.handles().len() {
                return Err(error("native gradient owner count differs"));
            }
            let count = state
                .session()
                .weights()
                .try_fold(0u64, |n, w| n.checked_add(w.scales().len() as u64))
                .ok_or_else(|| error("native gradient size overflow"))?;
            if count
                .checked_mul(20)
                .is_none_or(|n| n > state.session().plan().limits.max_workspace_bytes)
            {
                return Err(error(
                    "native gradient transfer/update exceeds workspace byte limit",
                ));
            }
            for (gradient, weight) in gradients.iter().zip(state.session().weights()) {
                if let Some(g) = gradient {
                    if tensor_shape(g, "scale gradient")? != [weight.scales().len()] {
                        return Err(error("native scale gradient shape differs"));
                    }
                    g.cpu_data_slice::<f64>().map_err(error)?;
                }
            }
            for (gradient, weight) in gradients.iter().zip(state.session().weights()) {
                let count = weight.scales().len();
                let v = if let Some(g) = gradient {
                    any = true;
                    if tensor_shape(g, "scale gradient")? != [count] {
                        return Err(error("native scale gradient shape differs"));
                    }
                    g.cpu_data_slice::<f64>()
                        .map_err(error)?
                        .iter()
                        .map(|&v| v as f32)
                        .collect()
                } else {
                    vec![0.0; count]
                };
                values.push(v);
            }
        }
        if !any {
            return Err(error("native training has no owner gradients"));
        }
        py.detach(|| {
            let mut state = self.lock()?;
            if state.session().completed_updates() != revision {
                return Err(error("stale native training revision"));
            }
            state
                .step(&values.iter().map(Vec::as_slice).collect::<Vec<_>>())
                .map_err(error)
        })
    }
    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.lock()?.session().checkpoint_bytes().map_err(error))?;
        Ok(PyBytes::new(py, &bytes))
    }
}

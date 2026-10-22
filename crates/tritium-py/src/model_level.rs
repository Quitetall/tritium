//! Native level receiver over the common reader, stream upload and CPU semantics.
use crate::torch_native::{checked_product, cpu_f32, export_array, tensor_shape};
use dlpark::legacy::Dlpack;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyBytes},
};
use std::{io::Cursor, sync::Mutex};
use tritium_evidence::{EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION};
use tritium_format::{TritPackageReadLimits, trit_model::TritModelReader};
use tritium_runtime::{
    TensorPayloadBudget, TensorUploadContext, TensorUploadResources, TritTensorSource,
    upload_tensor_stream_logged_with_resources,
};
use tritium_schema::{TensorStorage, TensorUploadPolicy};
use tritium_spec::{DeviceBuffer, TensorMatmul, TernaryBackend};

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}
fn zeros(count: usize, maximum: usize) -> PyResult<Vec<f32>> {
    let bytes = count
        .checked_mul(4)
        .ok_or_else(|| error("output byte overflow"))?;
    if bytes > maximum {
        return Err(error("native output/scratch exceeds byte limit"));
    }
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(error)?;
    values.resize(count, 0.0);
    Ok(values)
}

/// Native-only immutable packed CPU matrix. No latent/dense master is retained.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativePackedMatrix {
    tensor: Box<dyn DeviceBuffer>,
    #[pyo3(get)]
    rows: usize,
    #[pyo3(get)]
    columns: usize,
    #[pyo3(get)]
    payload_bytes: usize,
    #[pyo3(get)]
    storage_json: String,
    #[pyo3(get)]
    max_output_bytes: usize,
}
#[pymethods]
impl NativePackedMatrix {
    /// Float32 CPU DLPack input/output; basis and scale law stay in the backend.
    fn matmul(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let input: Dlpack = input.extract()?;
        let mut shape = tensor_shape(&input, "input")?;
        if shape.last() != Some(&self.columns) {
            return Err(error("native matrix input width differs"));
        }
        let batch = checked_product(&shape[..shape.len() - 1], "batch")?;
        let values = cpu_f32(&input, "input")?;
        let out_count = batch
            .checked_mul(self.rows)
            .ok_or_else(|| error("output shape overflow"))?;
        let scratch_count = batch
            .checked_mul(self.columns)
            .ok_or_else(|| error("scratch shape overflow"))?;
        // One combined logical bound, not two independently oversized allocations.
        if out_count
            .checked_add(scratch_count)
            .and_then(|n| n.checked_mul(4))
            .is_none_or(|n| n > self.max_output_bytes)
        {
            return Err(error("native output/scratch exceeds byte limit"));
        }
        let output = py.detach(|| {
            let mut output = zeros(out_count, self.max_output_bytes)?;
            let mut scratch = zeros(scratch_count, self.max_output_bytes)?;
            tritium_cpu::CpuBackend::new()
                .matmul(TensorMatmul {
                    act: values,
                    tensor: &*self.tensor,
                    batch,
                    transformed_act: &mut scratch,
                    out: &mut output,
                })
                .map_err(error)?;
            Ok::<_, PyErr>(output)
        })?;
        *shape
            .last_mut()
            .ok_or_else(|| error("input rank is zero"))? = self.rows;
        export_array(py, &shape, output, "native matrix output")
    }
    /// Float32 CPU input adjoint; no dense or decoded-trit master is created.
    fn input_vjp(&self, py: Python<'_>, gradient: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let input: Dlpack = gradient.extract()?;
        let mut shape = tensor_shape(&input, "output gradient")?;
        if shape.last() != Some(&self.rows) {
            return Err(error("native output gradient width differs"));
        }
        let batch = checked_product(&shape[..shape.len() - 1], "batch")?;
        let values = cpu_f32(&input, "output gradient")?;
        let count = batch
            .checked_mul(self.columns)
            .ok_or_else(|| error("input adjoint shape overflow"))?;
        if count
            .checked_add(self.columns)
            .and_then(|n| n.checked_mul(4))
            .is_none_or(|n| n > self.max_output_bytes)
        {
            return Err(error(
                "native input adjoint output/scratch exceeds byte limit",
            ));
        }
        let output = py.detach(|| {
            let mut output = zeros(count, self.max_output_bytes)?;
            let mut row = zeros(self.columns, self.max_output_bytes)?;
            tritium_cpu::CpuBackend::new()
                .additive_matmul_input_vjp(&*self.tensor, values, batch, &mut row, &mut output)
                .map_err(error)?;
            Ok::<_, PyErr>(output)
        })?;
        *shape
            .last_mut()
            .ok_or_else(|| error("output gradient rank is zero"))? = self.columns;
        export_array(py, &shape, output, "native input adjoint")
    }
    /// Int64 CPU DLPack indices; only requested embedding rows are materialized.
    fn gather(&self, py: Python<'_>, indices: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let input: Dlpack = indices.extract()?;
        let mut shape = tensor_shape(&input, "indices")?;
        let values = input.cpu_data_slice::<i64>().map_err(error)?;
        let count = values
            .len()
            .checked_mul(self.columns)
            .ok_or_else(|| error("gather shape overflow"))?;
        if count
            .checked_mul(4)
            .and_then(|n| {
                values
                    .len()
                    .checked_mul(size_of::<usize>())
                    .and_then(|i| n.checked_add(i))
            })
            .is_none_or(|n| n > self.max_output_bytes)
        {
            return Err(error("native gather exceeds byte limit"));
        }
        let output = py.detach(|| {
            let mut indices = Vec::new();
            indices.try_reserve_exact(values.len()).map_err(error)?;
            for &index in values {
                let index =
                    usize::try_from(index).map_err(|_| error("embedding index is negative"))?;
                if index >= self.rows {
                    return Err(error("embedding index is out of range"));
                }
                indices.push(index);
            }
            let mut output = zeros(count, self.max_output_bytes)?;
            tritium_cpu::CpuBackend::new()
                .embed_rows(&*self.tensor, &indices, &mut output)
                .map_err(error)?;
            Ok::<_, PyErr>(output)
        })?;
        shape.push(self.columns);
        export_array(py, &shape, output, "native embedding output")
    }
}

struct State {
    reader: TritModelReader<Cursor<Vec<u8>>>,
    recorder: Option<EvidenceRecorder>,
    sealed: Option<String>,
    ordinal: u64,
}
/// Frozen byte-image level session; mutable reader state is private and locked.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativeModelLevel {
    state: Mutex<State>,
    level: String,
    payload: TensorPayloadBudget,
    workspace: TensorPayloadBudget,
    max_output_bytes: usize,
}
pub(crate) fn open(
    bytes: Vec<u8>,
    level: String,
    run: String,
    payload: u64,
    workspace: u64,
    max_output_bytes: usize,
) -> PyResult<NativeModelLevel> {
    if max_output_bytes == 0 {
        return Err(error("max_output_bytes must be positive"));
    }
    let limits = TritPackageReadLimits {
        max_source_bytes: bytes.len() as u64,
        max_manifest_bytes: 16 * 1024 * 1024,
        max_directory_bytes: 64 * 1024 * 1024,
    };
    let reader = TritModelReader::open(Cursor::new(bytes), limits).map_err(error)?;
    reader.model_id(&level).map_err(error)?;
    let recorder = EvidenceRecorder::new(run, [TENSOR_UPLOADED_REGISTRATION]).map_err(error)?;
    Ok(NativeModelLevel {
        state: Mutex::new(State {
            reader,
            recorder: Some(recorder),
            sealed: None,
            ordinal: 0,
        }),
        level,
        payload: TensorPayloadBudget::new(payload),
        workspace: TensorPayloadBudget::new(workspace),
        max_output_bytes,
    })
}
impl NativeModelLevel {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, State>> {
        self.state
            .lock()
            .map_err(|_| error("native model level lock poisoned"))
    }
}
#[pymethods]
impl NativeModelLevel {
    /// Actual typed storage metadata; never a Python format parser.
    fn descriptors(&self) -> PyResult<String> {
        let state = self.lock()?;
        let level = state
            .reader
            .manifest()
            .levels
            .iter()
            .find(|l| l.name == self.level)
            .ok_or_else(|| error("level absent"))?;
        serde_json::to_string(&level.tensors).map_err(error)
    }
    #[getter]
    fn owned_payload_bytes(&self) -> u64 {
        self.payload.used_bytes()
    }
    #[pyo3(signature=(name, *, max_bytes=67_108_864))]
    fn asset<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if max_bytes == 0 {
            return Err(error("max_bytes must be positive"));
        }
        let output = py.detach(|| {
            let mut state = self.lock()?;
            let len = usize::try_from(state.reader.asset_byte_len(name).map_err(error)?)
                .map_err(error)?;
            if len > max_bytes {
                return Err(error("asset exceeds byte limit"));
            }
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(len).map_err(error)?;
            state
                .reader
                .visit_asset(name, |chunk| {
                    bytes.extend_from_slice(chunk);
                    Ok::<_, PyErr>(())
                })
                .map_err(error)?;
            Ok::<_, PyErr>(bytes)
        })?;
        Ok(PyBytes::new(py, &output))
    }
    #[pyo3(signature=(name, *, max_bytes=67_108_864))]
    fn preserved_tensor<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        max_bytes: usize,
    ) -> PyResult<(String, Vec<u64>, Bound<'py, PyBytes>)> {
        if max_bytes == 0 {
            return Err(error("max_bytes must be positive"));
        }
        let (precision, shape, output) = py.detach(|| {
            let mut state = self.lock()?;
            let meta = state.reader.tensor_meta(&self.level, name).map_err(error)?;
            let TensorStorage::Dense { shape, precision } = &meta.storage else {
                return Err(error("tensor is not dense-preserved"));
            };
            let shape = shape.clone();
            let precision = format!("{precision:?}");
            let len = usize::try_from(
                state
                    .reader
                    .tensor_byte_len(&self.level, name)
                    .map_err(error)?,
            )
            .map_err(error)?;
            if len > max_bytes {
                return Err(error("preserved tensor exceeds byte limit"));
            }
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(len).map_err(error)?;
            state
                .reader
                .visit_tensor(&self.level, name, |chunk| {
                    bytes.extend_from_slice(chunk);
                    Ok::<_, PyErr>(())
                })
                .map_err(error)?;
            Ok((precision, shape, bytes))
        })?;
        Ok((precision, shape, PyBytes::new(py, &output)))
    }
    fn packed_tensor(&self, py: Python<'_>, name: &str) -> PyResult<NativePackedMatrix> {
        py.detach(|| {
            let mut state = self.lock()?;
            let meta = state.reader.tensor_meta(&self.level, name).map_err(error)?;
            let TensorStorage::Additive { layout, .. } = meta.storage else {
                return Err(error("tensor is not additive"));
            };
            let rows = usize::try_from(layout.rows).map_err(error)?;
            let columns = usize::try_from(layout.cols).map_err(error)?;
            let storage_json = serde_json::to_string(&meta.storage).map_err(error)?;
            let ordinal = state.ordinal;
            let State {
                reader, recorder, ..
            } = &mut *state;
            let recorder = recorder
                .as_mut()
                .ok_or_else(|| error("upload session already sealed"))?;
            let source =
                TritTensorSource::new(reader.tensor_stream(&self.level, name).map_err(error)?)
                    .map_err(error)?;
            let (tensor, _) = upload_tensor_stream_logged_with_resources(
                &tritium_cpu::CpuBackend::new(),
                &source,
                TensorUploadPolicy {
                    native_only: true,
                    max_payload_bytes: Some(self.payload.limit_bytes()),
                },
                name,
                TensorUploadContext {
                    recorder,
                    span: &format!("upload/{ordinal}"),
                    parent: None,
                    logical_time: ordinal,
                },
                TensorUploadResources {
                    payload: &self.payload,
                    host_workspace: &self.workspace,
                    device_workspace: &self.workspace,
                },
            )
            .map_err(error)?;
            let payload_bytes = tensor.len_bytes();
            state.ordinal = ordinal
                .checked_add(1)
                .ok_or_else(|| error("upload ordinal overflow"))?;
            Ok(NativePackedMatrix {
                tensor,
                rows,
                columns,
                payload_bytes,
                storage_json,
                max_output_bytes: self.max_output_bytes,
            })
        })
    }
    /// Verify complete source and seal successful upload observations, not qualification.
    fn finish(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            let mut state = self.lock()?;
            state.reader.verify().map_err(error)?;
            if let Some(log) = &state.sealed {
                return Ok(log.clone());
            }
            let recorder = state
                .recorder
                .take()
                .ok_or_else(|| error("upload session not available"))?;
            let log = recorder.seal().map_err(error)?.jsonl().to_owned();
            state.sealed = Some(log.clone());
            Ok(log)
        })
    }
}

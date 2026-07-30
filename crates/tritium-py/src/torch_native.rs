//! Zero-copy PyTorch CPU adapter for the dispatcher-visible ternary Linear.
//!
//! Inputs arrive through DLPack and remain owned by PyTorch. Outputs are
//! Rust-owned ndarrays exported through a single-use DLPack capsule. Packed
//! TQ2_0 weights are cached by Python parameter identity, storage identity,
//! shape, and PyTorch's mutation version. No dense weight shadow is retained.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use dlpark::Builder;
use dlpark::legacy::Dlpack;
use half::f16;
use ndarray::{ArrayD, IxDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyWeakrefMethods, PyWeakrefReference};
use tritium_core::{GemmShape, TernaryFormat, Trit};
use tritium_format::{TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row};
use tritium_spec::{DeviceBuffer, MpGemm, MpGemmProjectedVjp, TernaryBackend};

use crate::cpu_backend;
const CACHE_CAPACITY: usize = 4096;
type LinearBackwardCapsules = (Py<PyAny>, Py<PyAny>, Option<Py<PyAny>>);

#[derive(Debug)]
struct LinearGeometry {
    input_shape: Vec<usize>,
    master_shape: Vec<usize>,
    m: usize,
    n: usize,
    k: usize,
}

struct PackedLinear {
    weights: Box<dyn DeviceBuffer>,
    scales: Vec<f32>,
    n: usize,
    k: usize,
}

struct CacheEntry {
    owner: Py<PyWeakrefReference>,
    parameter_version: u64,
    storage_identity: u64,
    data_ptr: usize,
    byte_offset: u64,
    packed: Arc<PackedLinear>,
}

#[derive(Default)]
struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    invalidations: AtomicU64,
}

static CACHE: OnceLock<Mutex<VecDeque<CacheEntry>>> = OnceLock::new();
static CACHE_STATS: CacheStats = CacheStats {
    hits: AtomicU64::new(0),
    misses: AtomicU64::new(0),
    invalidations: AtomicU64::new(0),
};

fn cache() -> &'static Mutex<VecDeque<CacheEntry>> {
    CACHE.get_or_init(|| Mutex::new(VecDeque::new()))
}

fn cache_lock() -> PyResult<std::sync::MutexGuard<'static, VecDeque<CacheEntry>>> {
    cache()
        .lock()
        .map_err(|_| PyRuntimeError::new_err("ternary packed-weight cache lock is poisoned"))
}

fn tensor_shape(tensor: &Dlpack, label: &str) -> PyResult<Vec<usize>> {
    tensor
        .shape()
        .map_err(|error| {
            PyValueError::new_err(format!("{label} DLPack shape is invalid: {error}"))
        })?
        .iter()
        .enumerate()
        .map(|(axis, &dimension)| {
            usize::try_from(dimension).map_err(|_| {
                PyValueError::new_err(format!(
                    "{label} dimension {axis} does not fit the host address space"
                ))
            })
        })
        .collect()
}

fn cpu_f32<'a>(tensor: &'a Dlpack, label: &str) -> PyResult<&'a [f32]> {
    tensor.cpu_data_slice::<f32>().map_err(|error| {
        PyValueError::new_err(format!(
            "{label} must be a compact CPU float32 DLPack tensor: {error}"
        ))
    })
}

fn checked_product(dimensions: &[usize], label: &str) -> PyResult<usize> {
    dimensions.iter().try_fold(1usize, |product, &dimension| {
        product.checked_mul(dimension).ok_or_else(|| {
            PyValueError::new_err(format!(
                "{label} element count overflows the host address space"
            ))
        })
    })
}

fn linear_geometry(input: &Dlpack, master: &Dlpack) -> PyResult<LinearGeometry> {
    let input_shape = tensor_shape(input, "input")?;
    let master_shape = tensor_shape(master, "master")?;
    if input_shape.is_empty() {
        return Err(PyValueError::new_err(
            "ternary Linear input must have at least one dimension",
        ));
    }
    if master_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "ternary Linear master must have rank 2",
        ));
    }
    let n = master_shape[0];
    let k = master_shape[1];
    if input_shape[input_shape.len() - 1] != k {
        return Err(PyValueError::new_err(
            "ternary Linear input width does not match master weight",
        ));
    }
    let m = checked_product(&input_shape[..input_shape.len() - 1], "input batch")?;
    Ok(LinearGeometry {
        input_shape,
        master_shape,
        m,
        n,
        k,
    })
}

fn projection_scale_values(scales: Option<&Dlpack>, n: usize) -> PyResult<Option<&[f32]>> {
    match scales {
        Some(tensor) => {
            let shape = tensor_shape(tensor, "projection scales")?;
            if shape.as_slice() != [n] {
                return Err(PyValueError::new_err(
                    "projection scale shape does not match output width",
                ));
            }
            Ok(Some(cpu_f32(tensor, "projection scales")?))
        }
        None => Ok(None),
    }
}

fn export_array(
    py: Python<'_>,
    shape: &[usize],
    values: Vec<f32>,
    label: &str,
) -> PyResult<Py<PyAny>> {
    let array = ArrayD::from_shape_vec(IxDyn(shape), values)
        .map_err(|error| PyRuntimeError::new_err(format!("cannot shape {label}: {error}")))?;
    let dlpack: Dlpack = Builder::from(Box::new(array))
        .try_build()
        .map_err(|error| PyRuntimeError::new_err(format!("cannot export {label}: {error}")))?;
    Ok(dlpack.into_pyobject(py)?.unbind())
}

fn build_packed(
    backend: &dyn TernaryBackend,
    master: &[f32],
    scales: &[f32],
    n: usize,
    k: usize,
) -> PyResult<Arc<PackedLinear>> {
    let expected = n
        .checked_mul(k)
        .ok_or_else(|| PyValueError::new_err("master element count overflows"))?;
    if master.len() != expected {
        return Err(PyValueError::new_err(format!(
            "master has {} elements; expected {expected}",
            master.len()
        )));
    }
    if n == 0 || k == 0 {
        return Err(PyValueError::new_err(
            "native ternary Linear requires non-empty weight dimensions",
        ));
    }
    if scales.len() != n {
        return Err(PyValueError::new_err(format!(
            "projection scales have {} elements; expected {n}",
            scales.len()
        )));
    }

    let blocks = num_blocks(k);
    let row_bytes = blocks
        .checked_mul(TQ2_0_BLOCK_BYTES)
        .ok_or_else(|| PyValueError::new_err("packed row byte count overflows"))?;
    let packed_len = n
        .checked_mul(row_bytes)
        .ok_or_else(|| PyValueError::new_err("packed weight byte count overflows"))?;
    let mut packed = vec![0u8; packed_len];
    let block_scales = vec![f16::ONE; blocks];
    let mut row_trits = vec![Trit::ZERO; k];

    for ((row, packed_row), &scale) in master
        .chunks_exact(k)
        .zip(packed.chunks_exact_mut(row_bytes))
        .zip(scales)
    {
        if scale == 0.0 {
            row_trits.fill(Trit::ZERO);
        } else {
            let denominator = scale.max(f32::MIN_POSITIVE);
            for (trit, &value) in row_trits.iter_mut().zip(row) {
                let quantized = (value / denominator).round_ties_even().clamp(-1.0, 1.0) as i8;
                *trit = Trit::from_i8(quantized)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            }
        }
        pack_tq2_0_row(&row_trits, &block_scales, packed_row)
            .map_err(|error| PyRuntimeError::new_err(format!("cannot pack master row: {error}")))?;
    }

    let shape = GemmShape::new(0, n, k);
    let weights = backend
        .upload_weights(&packed, shape, TernaryFormat::Tq2_0)
        .map_err(|error| PyRuntimeError::new_err(format!("cannot cache packed weight: {error}")))?;
    Ok(Arc::new(PackedLinear {
        weights,
        scales: scales.to_vec(),
        n,
        k,
    }))
}

#[allow(clippy::too_many_arguments)]
fn cached_packed(
    py: Python<'_>,
    master_owner: &Bound<'_, PyAny>,
    master_values: &[f32],
    projection_scales: Option<&[f32]>,
    parameter_version: u64,
    storage_identity: u64,
    data_ptr: usize,
    byte_offset: u64,
    n: usize,
    k: usize,
    backend: &dyn TernaryBackend,
) -> PyResult<Option<Arc<PackedLinear>>> {
    let mut entries = cache_lock()?;
    entries.retain(|entry| entry.owner.bind(py).upgrade().is_some());
    if let Some(position) = entries.iter().position(|entry| {
        entry
            .owner
            .bind(py)
            .upgrade()
            .is_some_and(|owner| owner.is(master_owner))
    }) {
        let Some(entry) = entries.remove(position) else {
            return Err(PyRuntimeError::new_err(
                "ternary packed-weight cache changed during lookup",
            ));
        };
        if entry.parameter_version == parameter_version
            && entry.storage_identity == storage_identity
            && entry.data_ptr == data_ptr
            && entry.byte_offset == byte_offset
            && entry.packed.n == n
            && entry.packed.k == k
        {
            let packed = Arc::clone(&entry.packed);
            entries.push_front(entry);
            CACHE_STATS.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(packed));
        }
        CACHE_STATS.invalidations.fetch_add(1, Ordering::Relaxed);
    }

    let Some(projection_scales) = projection_scales else {
        return Ok(None);
    };
    CACHE_STATS.misses.fetch_add(1, Ordering::Relaxed);
    let packed = build_packed(backend, master_values, projection_scales, n, k)?;
    entries.push_front(CacheEntry {
        owner: PyWeakrefReference::new(master_owner)?.unbind(),
        parameter_version,
        storage_identity,
        data_ptr,
        byte_offset,
        packed: Arc::clone(&packed),
    });
    entries.truncate(CACHE_CAPACITY);
    Ok(Some(packed))
}

/// Run one compact CPU float32 ternary Linear and return a single-use DLPack capsule.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _ternary_linear_cpu_dlpack(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    master: &Bound<'_, PyAny>,
    bias: Option<&Bound<'_, PyAny>>,
    projection_scales: Option<&Bound<'_, PyAny>>,
    master_owner: &Bound<'_, PyAny>,
    parameter_version: u64,
    storage_identity: u64,
) -> PyResult<Option<Py<PyAny>>> {
    let input_tensor: Dlpack = input.extract()?;
    let master_tensor: Dlpack = master.extract()?;
    let bias_tensor: Option<Dlpack> = bias.map(|tensor| tensor.extract::<Dlpack>()).transpose()?;
    let scales_tensor: Option<Dlpack> = projection_scales
        .map(|tensor| tensor.extract::<Dlpack>())
        .transpose()?;

    let LinearGeometry {
        input_shape,
        master_shape: _,
        m,
        n,
        k,
    } = linear_geometry(&input_tensor, &master_tensor)?;
    let input_values = cpu_f32(&input_tensor, "input")?;
    let master_values = cpu_f32(&master_tensor, "master")?;
    let scale_values = projection_scale_values(scales_tensor.as_ref(), n)?;
    if input_values.iter().any(|value| !value.is_finite())
        || (scale_values.is_some() && master_values.iter().any(|value| !value.is_finite()))
    {
        return Ok(None);
    }
    let bias_values = match bias_tensor.as_ref() {
        Some(tensor) => {
            let shape = tensor_shape(tensor, "bias")?;
            if shape.as_slice() != [n] {
                return Err(PyValueError::new_err(
                    "ternary Linear bias shape does not match output width",
                ));
            }
            Some(cpu_f32(tensor, "bias")?)
        }
        None => None,
    };

    let backend = cpu_backend().map_err(PyRuntimeError::new_err)?;
    let tensor = master_tensor.tensor();
    let Some(packed) = cached_packed(
        py,
        master_owner,
        master_values,
        scale_values,
        parameter_version,
        storage_identity,
        tensor.data_ptr() as usize,
        tensor.byte_offset,
        n,
        k,
        backend.as_ref(),
    )?
    else {
        return Ok(None);
    };
    let out_len = m
        .checked_mul(n)
        .ok_or_else(|| PyValueError::new_err("ternary Linear output element count overflows"))?;

    let mut output = vec![0.0f32; out_len];
    py.detach(|| {
        backend
            .mpgemm(MpGemm {
                act: input_values,
                weights: packed.weights.as_ref(),
                scales: &packed.scales,
                shape: GemmShape::new(m, n, k),
                format: TernaryFormat::Tq2_0,
                out: &mut output,
            })
            .map_err(|error| {
                PyRuntimeError::new_err(format!("native ternary Linear failed: {error}"))
            })?;
        if let Some(values) = bias_values {
            for row in output.chunks_exact_mut(n) {
                for (value, &row_bias) in row.iter_mut().zip(values) {
                    *value += row_bias;
                }
            }
        }
        Ok::<(), PyErr>(())
    })?;

    let mut output_shape = input_shape;
    let output_rank = output_shape.len();
    output_shape[output_rank - 1] = n;
    Ok(Some(export_array(
        py,
        &output_shape,
        output,
        "native output",
    )?))
}

/// Run one compact CPU float32 ternary Linear VJP from cached packed weights.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _ternary_linear_backward_cpu_dlpack(
    py: Python<'_>,
    grad_output: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
    master: &Bound<'_, PyAny>,
    projection_scales: Option<&Bound<'_, PyAny>>,
    master_owner: &Bound<'_, PyAny>,
    parameter_version: u64,
    storage_identity: u64,
    has_bias: bool,
) -> PyResult<Option<LinearBackwardCapsules>> {
    let grad_output_tensor: Dlpack = grad_output.extract()?;
    let input_tensor: Dlpack = input.extract()?;
    let master_tensor: Dlpack = master.extract()?;
    let scales_tensor: Option<Dlpack> = projection_scales
        .map(|tensor| tensor.extract::<Dlpack>())
        .transpose()?;

    let LinearGeometry {
        input_shape,
        master_shape,
        m,
        n,
        k,
    } = linear_geometry(&input_tensor, &master_tensor)?;
    let grad_output_shape = tensor_shape(&grad_output_tensor, "grad_output")?;
    let mut expected_grad_shape = input_shape.clone();
    let output_rank = expected_grad_shape.len();
    expected_grad_shape[output_rank - 1] = n;
    if grad_output_shape != expected_grad_shape {
        return Err(PyValueError::new_err(
            "ternary Linear grad_output shape does not match forward output",
        ));
    }
    let grad_output_values = cpu_f32(&grad_output_tensor, "grad_output")?;
    let input_values = cpu_f32(&input_tensor, "input")?;
    let master_values = cpu_f32(&master_tensor, "master")?;
    let scale_values = projection_scale_values(scales_tensor.as_ref(), n)?;
    if scale_values.is_some() && master_values.iter().any(|value| !value.is_finite()) {
        return Ok(None);
    }

    let backend = cpu_backend().map_err(PyRuntimeError::new_err)?;
    let tensor = master_tensor.tensor();
    let Some(packed) = cached_packed(
        py,
        master_owner,
        master_values,
        scale_values,
        parameter_version,
        storage_identity,
        tensor.data_ptr() as usize,
        tensor.byte_offset,
        n,
        k,
        backend.as_ref(),
    )?
    else {
        return Ok(None);
    };

    let mut grad_input = vec![0.0f32; m * k];
    let mut grad_master = vec![0.0f32; n * k];
    let mut grad_bias = has_bias.then(|| vec![0.0f32; n]);
    py.detach(|| {
        backend
            .mpgemm_projected_vjp(MpGemmProjectedVjp {
                act: input_values,
                weights: packed.weights.as_ref(),
                scales: &packed.scales,
                grad_output: grad_output_values,
                shape: GemmShape::new(m, n, k),
                format: TernaryFormat::Tq2_0,
                grad_act: &mut grad_input,
                grad_projected_weight: &mut grad_master,
                grad_bias: grad_bias.as_deref_mut(),
            })
            .map_err(|error| {
                PyRuntimeError::new_err(format!("native ternary Linear backward failed: {error}"))
            })
    })?;

    for ((master_row, grad_row), &scale) in master_values
        .chunks_exact(k)
        .zip(grad_master.chunks_exact_mut(k))
        .zip(&packed.scales)
    {
        let denominator = scale.max(f32::MIN_POSITIVE);
        for (&master_value, grad_value) in master_row.iter().zip(grad_row) {
            if !(scale > 0.0 && (master_value / denominator).abs() < 1.0) {
                *grad_value = 0.0;
            }
        }
    }

    let grad_input = export_array(py, &input_shape, grad_input, "native grad_input")?;
    let grad_master = export_array(py, &master_shape, grad_master, "native grad_master")?;
    let grad_bias = grad_bias
        .map(|values| export_array(py, &[n], values, "native grad_bias"))
        .transpose()?;
    Ok(Some((grad_input, grad_master, grad_bias)))
}

/// Reset native packed-weight cache and deterministic diagnostics counters.
#[pyfunction]
pub(crate) fn _ternary_linear_cache_clear() -> PyResult<()> {
    cache_lock()?.clear();
    CACHE_STATS.hits.store(0, Ordering::Relaxed);
    CACHE_STATS.misses.store(0, Ordering::Relaxed);
    CACHE_STATS.invalidations.store(0, Ordering::Relaxed);
    Ok(())
}

/// Return bounded-cache diagnostics used by conformance and performance gates.
#[pyfunction]
pub(crate) fn _ternary_linear_cache_info(py: Python<'_>) -> PyResult<BTreeMap<&'static str, u64>> {
    let mut cache = cache_lock()?;
    cache.retain(|entry| entry.owner.bind(py).upgrade().is_some());
    let entries = u64::try_from(cache.len())
        .map_err(|_| PyRuntimeError::new_err("ternary cache entry count does not fit u64"))?;
    Ok(BTreeMap::from([
        ("capacity", CACHE_CAPACITY as u64),
        ("entries", entries),
        ("hits", CACHE_STATS.hits.load(Ordering::Relaxed)),
        (
            "invalidations",
            CACHE_STATS.invalidations.load(Ordering::Relaxed),
        ),
        ("misses", CACHE_STATS.misses.load(Ordering::Relaxed)),
    ]))
}

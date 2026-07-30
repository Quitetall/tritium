//! # tritium (Python bindings)
//!
//! PyO3 bindings exposing a minimal, safe Python surface over the Tritium
//! inference spine:
//!
//! - [`Model::load`] — load a ternary GGUF model, selecting a compute backend via
//!   [`tritium_runtime`]'s registry (CPU today; the loader is backend-generic).
//! - [`Model::generate`] — greedily decode tokens, **releasing the GIL** during the
//!   Rust compute so other Python threads run concurrently.
//! - [`ternary_matmul`] — a low-level helper that runs one ternary mpGEMM
//!   (`out = scale · act · Wᵀ`) for tests and experimentation.
//!
//! ## Error handling (no panics across FFI)
//!
//! Every fallible Rust path returns a `Result`; the boundary converts each error
//! into a Python exception ([`PyValueError`] / [`PyRuntimeError`]) rather than
//! unwinding. A wrong dtype, a wrong shape, or a malformed model therefore raises a
//! catchable Python exception — never a segfault or an abort. The crate carries no
//! hand-written `unsafe`; the GIL release uses the safe [`Python::detach`].
//!
//! ## Threading
//!
//! [`Model`] wraps its [`ModelRunner`] in a [`Mutex`] so the `#[pymethods]`
//! `&self` receivers can mutate the runner. Generation releases the GIL and takes
//! the lock, so concurrent `generate` calls from multiple Python threads serialize
//! on the model (correct, deadlock-free) while *other* Python work proceeds freely
//! during the compute.
#![forbid(unsafe_code)]
#![allow(unreachable_pub)] // pyo3's `#[pymethods]` expansion emits `pub` items.

mod hf_assets;
mod kronecker;
mod module_package;
mod onnx;
mod ops;
mod qwen;
mod salt;
mod torch_native;

use std::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use tritium_core::{GemmShape, TernaryFormat, Trit};
use tritium_nn::{ModelRunner, TernaryLinear};
use tritium_spec::TernaryBackend;

// Linked so its `linkme` backend self-registration into `tritium_runtime::BACKENDS`
// is present in the extension module — the registry that `Model::load` and
// `ternary_matmul` consult would otherwise find no `cpu` backend.
use tritium_cpu as _;

/// A loaded ternary model that can greedily generate tokens.
///
/// Construct one with [`Model::load`]; call [`Model::generate`] to decode. The
/// underlying [`ModelRunner`] is held behind a [`Mutex`] so generation can take a
/// `&self` receiver yet still mutate the KV cache.
#[pyclass(module = "tritium")]
struct Model {
    runner: Mutex<ModelRunner>,
}

#[pymethods]
impl Model {
    /// Load a model from a GGUF file at `gguf_path`, selecting the CPU backend via
    /// the runtime registry.
    ///
    /// Raises `ValueError` if the path cannot be read, and `RuntimeError` if the
    /// bytes are not a valid model (bad GGUF, missing weights, no backend).
    #[staticmethod]
    fn load(py: Python<'_>, gguf_path: &str) -> PyResult<Self> {
        let path = gguf_path.to_owned();
        // Reading the file and building the runner is pure Rust compute + I/O; let
        // other Python threads run while it happens.
        let runner = py.detach(move || -> Result<ModelRunner, String> {
            let bytes = std::fs::read(&path).map_err(|e| format!("cannot read `{path}`: {e}"))?;
            ModelRunner::load_cpu(&bytes).map_err(|e| format!("cannot load `{path}`: {e}"))
        });
        match runner {
            Ok(runner) => Ok(Self {
                runner: Mutex::new(runner),
            }),
            // A read failure is a usage error (ValueError); a parse/weights/backend
            // failure is a runtime condition (RuntimeError).
            Err(msg) if msg.contains("cannot read") => Err(PyValueError::new_err(msg)),
            Err(msg) => Err(PyRuntimeError::new_err(msg)),
        }
    }

    /// Greedily generate up to `max_new_tokens` continuing `token_ids`, returning
    /// the newly generated token IDs (the prompt is not included).
    ///
    /// The Rust compute runs with the GIL **released** ([`Python::detach`]),
    /// so other Python threads execute concurrently. `greedy` is accepted for API
    /// stability; v0.20 only implements greedy decoding, so `greedy=False` decodes
    /// greedily as well.
    ///
    /// Raises `ValueError` if any token ID is negative or exceeds `u32`, and
    /// `RuntimeError` if generation fails inside the model.
    #[pyo3(signature = (token_ids, max_new_tokens, greedy = true, eos = 128_001))]
    fn generate(
        &self,
        py: Python<'_>,
        token_ids: Vec<i64>,
        max_new_tokens: usize,
        greedy: bool,
        eos: u32,
    ) -> PyResult<Vec<u32>> {
        let _ = greedy; // only greedy decoding exists in v0.20; flag kept for API.

        // Validate + narrow the prompt before touching the model, mapping any bad
        // value to a Python exception rather than panicking on the cast.
        let prompt: Vec<u32> = token_ids
            .into_iter()
            .map(|v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("token id {v} out of range for u32"))
                })
            })
            .collect::<PyResult<_>>()?;

        // Release the GIL for the compute. Lock the runner inside the closure so the
        // borrow does not cross the FFI boundary. A poisoned lock (a prior panic in
        // another thread) becomes a clean RuntimeError instead of a re-panic.
        let result = py.detach(move || -> Result<Vec<u32>, String> {
            let mut runner = self
                .runner
                .lock()
                .map_err(|_| "model mutex poisoned by an earlier failure".to_owned())?;
            runner
                .generate(&prompt, max_new_tokens, eos)
                .map_err(|e| e.to_string())
        });
        result.map_err(PyRuntimeError::new_err)
    }

    fn __repr__(&self) -> String {
        match self.runner.lock() {
            Ok(r) => format!(
                "Model(arch={:?}, n_layers={}, n_embd={})",
                r.config.arch, r.config.n_layers, r.config.n_embd
            ),
            Err(_) => "Model(<poisoned>)".to_owned(),
        }
    }
}

/// Run one ternary mpGEMM: `out[m, n] = scale · Σ_k act[m, k] · w[n, k]`.
///
/// - `activations` is a list of `m` rows, each a list of `k` floats (`[M, K]`).
/// - `weights` is a list of `n` rows, each a list of `k` ints in `{-1, 0, 1}`
///   (`[N, K]`); any other integer raises `ValueError`.
/// - `scale` is the single per-tensor weight scale.
/// - `device` selects an explicitly compiled backend; CPU is the default.
///
/// Returns the `[M, N]` result as a list of `m` rows of `n` floats. This is the
/// low-level primitive used by tests and wheel qualification to exercise the
/// same backend-generic `TernaryLinear` path the model uses.
///
/// Raises `ValueError` on ragged/empty input, a non-ternary weight, or a
/// dtype/shape mismatch, and `RuntimeError` if the backend GEMM fails — never a
/// panic.
#[pyfunction]
#[pyo3(signature = (activations, weights, scale, device = "cpu"))]
fn ternary_matmul(
    py: Python<'_>,
    activations: Vec<Vec<f32>>,
    weights: Vec<Vec<i64>>,
    scale: f32,
    device: &str,
) -> PyResult<Vec<Vec<f32>>> {
    validate_device(device)?;
    // Shape validation up front so every error is a Python exception, not a panic.
    if activations.is_empty() {
        return Err(PyValueError::new_err(
            "activations must have at least one row",
        ));
    }
    if weights.is_empty() {
        return Err(PyValueError::new_err("weights must have at least one row"));
    }
    let m = activations.len();
    let k = activations[0].len();
    let n = weights.len();
    if k == 0 {
        return Err(PyValueError::new_err("activation rows must be non-empty"));
    }
    if activations.iter().any(|r| r.len() != k) {
        return Err(PyValueError::new_err(
            "all activation rows must have the same length K",
        ));
    }
    if weights.iter().any(|r| r.len() != k) {
        return Err(PyValueError::new_err(format!(
            "every weight row must have length K = {k} (the activation width)"
        )));
    }

    // Flatten weights into ternary trits, rejecting any value outside {-1, 0, 1}.
    let mut trits = Vec::with_capacity(n * k);
    for row in &weights {
        for &v in row {
            let t = match v {
                -1 => Trit::NEG,
                0 => Trit::ZERO,
                1 => Trit::POS,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "weight value {other} is not ternary (must be -1, 0, or 1)"
                    )));
                }
            };
            trits.push(t);
        }
    }
    let act: Vec<f32> = activations.into_iter().flatten().collect();

    // Compute with the GIL released. The CPU backend comes from the registry; the
    // `TernaryLinear` path packs, uploads, and runs the same mpGEMM the model uses.
    let device = device.to_owned();
    let flat = py.detach(move || -> Result<Vec<f32>, String> {
        let backend = backend_for_device(&device)?;
        let linear =
            TernaryLinear::new(backend.as_ref(), &trits, n, k, scale).map_err(|e| e.to_string())?;
        let mut out = vec![0.0f32; m * n];
        linear
            .forward(backend.as_ref(), &act, m, &mut out)
            .map_err(|e| e.to_string())?;
        let _ = GemmShape::new(m, n, k); // documents the geometry; shape-checked above.
        let _ = TernaryFormat::Tq2_0; // the packing the linear layer uses internally.
        Ok(out)
    });

    let flat = flat.map_err(PyRuntimeError::new_err)?;
    // Reshape the flat [M*N] buffer into M rows of N.
    Ok(flat.chunks_exact(n).map(<[f32]>::to_vec).collect())
}

#[pyfunction]
fn compiled_backends() -> Vec<&'static str> {
    #[cfg(feature = "cuda")]
    return vec!["cpu", "cuda"];
    #[cfg(not(feature = "cuda"))]
    vec!["cpu"]
}

/// Construct a fresh CPU backend trait object from the runtime registry.
///
/// Returns a stringified error if no `cpu` backend registered (so the caller can
/// surface it as a Python exception).
fn cpu_backend() -> Result<Box<dyn TernaryBackend>, String> {
    for entry in tritium_runtime::BACKENDS {
        if entry.name == "cpu" {
            return (entry.init)().map_err(|e| e.to_string());
        }
    }
    Err("no `cpu` backend registered in the runtime".to_owned())
}

fn validate_device(device: &str) -> PyResult<()> {
    if device == "cpu" {
        return Ok(());
    }
    if device == "cuda" || device.starts_with("cuda:") {
        if let Some(ordinal) = device.strip_prefix("cuda:") {
            ordinal.parse::<usize>().map_err(|_| {
                PyValueError::new_err("CUDA device must be `cuda` or `cuda:<ordinal>`")
            })?;
        }
        #[cfg(feature = "cuda")]
        return Ok(());
        #[cfg(not(feature = "cuda"))]
        return Err(PyValueError::new_err(
            "this Tritium wheel was not compiled with CUDA support",
        ));
    }
    Err(PyValueError::new_err(
        "device must be `cpu`, `cuda`, or `cuda:<ordinal>`",
    ))
}

fn backend_for_device(device: &str) -> Result<Box<dyn TernaryBackend>, String> {
    if device == "cpu" {
        return cpu_backend();
    }
    #[cfg(feature = "cuda")]
    {
        let ordinal = device
            .strip_prefix("cuda:")
            .map_or(0, |value| value.parse::<usize>().unwrap_or(0));
        tritium_cuda::CudaBackend::new(ordinal)
            .map(|backend| Box::new(backend) as Box<dyn TernaryBackend>)
            .map_err(|error| error.to_string())
    }
    #[cfg(not(feature = "cuda"))]
    Err("CUDA backend is unavailable in this build".to_owned())
}

/// The compiled `tritium._tritium` extension module.
///
/// Exposes [`Model`], the [`ternary_matmul`] helper, and the autograd-op forward/vjp functions
/// (`conv1d_*`, `fsq_*`, `ste_*`) that the pure-Python `tritium.autograd` layer wraps in
/// `torch.autograd.Function`s. Building it links `tritium-cpu`, whose `linkme` self-registration makes
/// the CPU backend visible to the runtime registry that [`Model::load`] / [`ternary_matmul`] consult.
#[pymodule]
fn _tritium(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<qwen::QwenModel>()?;
    m.add_class::<qwen::QwenLoadReceipt>()?;
    m.add_class::<qwen::QwenReferenceLanguageOutput>()?;
    m.add_class::<onnx::QwenOnnxBundleReceipt>()?;
    m.add_class::<onnx::QwenOnnxModel>()?;
    m.add_class::<onnx::QwenOnnxOperatorCounts>()?;
    m.add_class::<onnx::QwenOnnxLanguageOutput>()?;
    m.add_class::<onnx::QwenOnnxMtpOutput>()?;
    m.add_class::<module_package::ModuleSaltV2Receipt>()?;
    m.add_class::<kronecker::KroneckerEvidenceBuilder>()?;
    m.add_class::<kronecker::KroneckerEvidenceReceipt>()?;
    m.add_class::<kronecker::Qwen36KroneckerCaptureTask>()?;
    m.add_class::<kronecker::Qwen36KroneckerCaptureReceipt>()?;
    m.add_class::<kronecker::Qwen36KroneckerCaptureSession>()?;
    kronecker::register_exceptions(m)?;
    m.add_function(wrap_pyfunction!(ternary_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(compiled_backends, m)?)?;
    m.add_function(wrap_pyfunction!(onnx::verify_qwen35_onnx_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(onnx::stage_qwen35_onnx_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(onnx::export_qwen35_onnx_bundle, m)?)?;
    // Autograd-op primitives (ADR 0030): forward/vjp for ternary Conv1d, FSQ, and STE.
    m.add_function(wrap_pyfunction!(ops::conv1d_forward, m)?)?;
    m.add_function(wrap_pyfunction!(ops::conv1d_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(ops::fsq_forward, m)?)?;
    m.add_function(wrap_pyfunction!(ops::fsq_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ste_absmean_scale, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ste_quantize_forward, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ste_quantize_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(ops::lsq_forward, m)?)?;
    m.add_function(wrap_pyfunction!(ops::lsq_vjp, m)?)?;
    m.add_function(wrap_pyfunction!(
        torch_native::_ternary_linear_cpu_dlpack,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        torch_native::_ternary_linear_cache_clear,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        torch_native::_ternary_linear_cache_info,
        m
    )?)?;
    m.add_class::<salt::Qwen36PtqMasterReceipt>()?;
    m.add_class::<salt::Qwen36PtqPackageReceipt>()?;
    m.add_function(wrap_pyfunction!(salt::reconcile_qwen36_ptq_masters, m)?)?;
    m.add_function(wrap_pyfunction!(salt::reconcile_qwen36_ptq_packages, m)?)?;
    m.add_function(wrap_pyfunction!(salt::verify_salt_v2_package, m)?)?;
    m.add_function(wrap_pyfunction!(salt::verify_preserved_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(salt::verify_hf_asset, m)?)?;
    m.add_function(wrap_pyfunction!(
        module_package::pack_module_conversion_salt_v2,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(salt::inspect_qwen36_ptq_evidence, m)?)?;
    m.add_function(wrap_pyfunction!(salt::publish_directory_noreplace, m)?)?;
    m.add(
        "__doc__",
        "Tritium: ternary-model inference + autograd ops from Python.",
    )?;
    Ok(())
}

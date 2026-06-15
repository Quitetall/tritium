//! The generic conformance runner: replay vectors against any backend.

use half::f16;
use tritium_core::{GemmShape, TernaryFormat, Trit};
use tritium_format::{num_blocks, pack_tq1_0_row, pack_tq2_0_row};
use tritium_spec::TernaryBackend;

use crate::vector::{ConformanceVector, Tolerance};

/// A single vector that a backend failed, with enough detail to debug it.
#[derive(Clone, Debug)]
pub struct FailedCase {
    /// The failing vector's [`ConformanceVector::id`].
    pub id: String,
    /// Why it failed.
    pub reason: FailureReason,
}

/// The category of a conformance failure.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum FailureReason {
    /// The backend returned an error from `upload_weights` or `mpgemm`.
    BackendError(String),
    /// The vector itself was malformed (e.g. a weight outside `{-1,0,+1}`, or a
    /// buffer length disagreeing with its declared shape).
    MalformedVector(String),
    /// The format string was not one this harness packs.
    UnknownFormat(String),
    /// The output disagreed with the reference beyond tolerance, at `index`.
    Mismatch {
        /// Flat `[M, N]` index of the first offending element.
        index: usize,
        /// What the reference produced.
        expected: f32,
        /// What the backend produced.
        got: f32,
    },
}

impl std::fmt::Display for FailureReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FailureReason::BackendError(m) => write!(f, "backend error: {m}"),
            FailureReason::MalformedVector(m) => write!(f, "malformed vector: {m}"),
            FailureReason::UnknownFormat(m) => write!(f, "unknown format: {m}"),
            FailureReason::Mismatch {
                index,
                expected,
                got,
            } => write!(f, "mismatch at [{index}]: expected {expected}, got {got}"),
        }
    }
}

/// The outcome of running a backend over a vector set.
#[derive(Clone, Debug)]
pub struct Report {
    /// Number of vectors the backend reproduced within tolerance.
    pub passed: usize,
    /// Every vector that failed, with its reason.
    pub failed: Vec<FailedCase>,
}

impl Report {
    /// `true` if no vector failed.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.failed.is_empty()
    }

    /// Total vectors run (`passed + failed`).
    #[must_use]
    pub fn total(&self) -> usize {
        self.passed + self.failed.len()
    }
}

/// Parse the format tag a [`ConformanceVector`] carries.
fn parse_format(tag: &str) -> Option<TernaryFormat> {
    match tag {
        "tq2_0" => Some(TernaryFormat::Tq2_0),
        "tq1_0" => Some(TernaryFormat::Tq1_0),
        _ => None,
    }
}

/// Convert a vector's raw `i8` weights into validated [`Trit`]s.
///
/// Returns the offending value on the first out-of-range element.
fn to_trits(weights: &[i8]) -> Result<Vec<Trit>, i8> {
    let mut out = Vec::with_capacity(weights.len());
    for &w in weights {
        match Trit::from_i8(w) {
            Ok(t) => out.push(t),
            Err(_) => return Err(w),
        }
    }
    Ok(out)
}

/// Pack one vector's `[N, K]` weights into the format's row layout.
///
/// **Packing scale vs mpGEMM scale are deliberately distinct.** The block scale
/// stored by the packer is fixed to `1.0` here, so unpacking yields the original
/// trits unscaled; the per-output-channel multipliers in
/// [`ConformanceVector::scales`] are passed separately to
/// [`TernaryBackend::mpgemm`] and are the *only* thing that scales the
/// contraction. This matches the reference contract
/// `out[m,n] = scales[n] · Σ_k act[m,k] · w[n,k]`: if the packing scale also
/// scaled the trits, the output would be multiplied twice. A real model would
/// fold AbsMean into one of the two; the harness keeps them orthogonal so a
/// backend that double-applies, or that ignores the packing scale, is caught.
///
/// Each of the `N` rows is packed independently into `num_blocks(K)` blocks; the
/// per-row packed bytes are concatenated output-major to match the `[N, K]`
/// weight layout `upload_weights` expects.
fn pack_weights(
    trits: &[Trit],
    shape: GemmShape,
    format: TernaryFormat,
) -> Result<Vec<u8>, String> {
    let GemmShape { n, k, .. } = shape;
    let nb = num_blocks(k);
    let unit_scales = vec![f16::ONE; nb];
    let block_bytes = block_bytes(format)?;
    let row_bytes = nb * block_bytes;
    let mut packed = vec![0u8; n * row_bytes];

    for ni in 0..n {
        let row = &trits[ni * k..ni * k + k];
        let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
        let res = match format {
            TernaryFormat::Tq2_0 => pack_tq2_0_row(row, &unit_scales, out),
            TernaryFormat::Tq1_0 => pack_tq1_0_row(row, &unit_scales, out),
            other => return Err(format!("harness cannot pack format {other:?}")),
        };
        res.map_err(|e| format!("packing row {ni}: {e}"))?;
    }
    Ok(packed)
}

/// Packed bytes per block for a format the harness supports.
fn block_bytes(format: TernaryFormat) -> Result<usize, String> {
    match format {
        TernaryFormat::Tq2_0 => Ok(tritium_format::TQ2_0_BLOCK_BYTES),
        TernaryFormat::Tq1_0 => Ok(tritium_format::TQ1_0_BLOCK_BYTES),
        other => Err(format!("harness cannot pack format {other:?}")),
    }
}

/// Run one vector against a backend, returning `Ok(())` on pass or the reason it
/// failed.
fn run_one<B: TernaryBackend>(
    backend: &B,
    v: &ConformanceVector,
    tol: Tolerance,
) -> Result<(), FailureReason> {
    let format =
        parse_format(&v.format).ok_or_else(|| FailureReason::UnknownFormat(v.format.clone()))?;
    let shape = GemmShape::new(v.m, v.n, v.k);

    // Validate the vector's own internal consistency before touching the backend.
    if !shape.buffers_fit(v.activation.len(), v.weights.len(), v.expected.len()) {
        return Err(FailureReason::MalformedVector(format!(
            "buffer lengths {{act:{}, w:{}, out:{}}} disagree with shape {:?}",
            v.activation.len(),
            v.weights.len(),
            v.expected.len(),
            shape
        )));
    }
    if v.scales.len() != v.n {
        return Err(FailureReason::MalformedVector(format!(
            "scales len {} != N {}",
            v.scales.len(),
            v.n
        )));
    }

    let trits = to_trits(&v.weights).map_err(|bad| {
        FailureReason::MalformedVector(format!("weight {bad} outside {{-1,0,1}}"))
    })?;
    let packed = pack_weights(&trits, shape, format).map_err(FailureReason::MalformedVector)?;

    let buffer = backend
        .upload_weights(&packed, shape, format)
        .map_err(|e| FailureReason::BackendError(e.to_string()))?;

    let mut out = vec![0.0f32; v.m * v.n];
    backend
        .mpgemm(
            &v.activation,
            buffer.as_ref(),
            &v.scales,
            shape,
            format,
            &mut out,
        )
        .map_err(|e| FailureReason::BackendError(e.to_string()))?;

    for (i, (&got, &want)) in out.iter().zip(&v.expected).enumerate() {
        if !tol.accepts(got, want) {
            return Err(FailureReason::Mismatch {
                index: i,
                expected: want,
                got,
            });
        }
    }
    Ok(())
}

/// Replay every vector against `backend`, grading the output with `tol`.
///
/// For each vector the runner converts its `i8` weights to [`Trit`]s, packs each
/// `[N, K]` row with the format's host-side packer (block scale fixed to `1.0` —
/// see [`pack_weights`]), uploads the bytes via
/// [`TernaryBackend::upload_weights`], runs [`TernaryBackend::mpgemm`] with the
/// vector's per-channel `scales`, and compares the result to
/// [`ConformanceVector::expected`] under `tol`. A backend is conformant iff the
/// returned [`Report`] has no failures.
///
/// This never panics: a malformed vector, an unknown format, a backend error, or
/// a numeric mismatch all become entries in [`Report::failed`].
///
/// ```
/// use tritium_testkit::{generate_vectors, run_conformance, Tolerance};
/// # use tritium_testkit::reference_backend_for_doctest as backend;
/// let vectors = generate_vectors(7, 4);
/// let report = run_conformance(&backend(), &vectors, Tolerance::default());
/// assert!(report.is_ok());
/// ```
pub fn run_conformance<B: TernaryBackend>(
    backend: &B,
    vectors: &[ConformanceVector],
    tol: Tolerance,
) -> Report {
    let mut passed = 0;
    let mut failed = Vec::new();
    for v in vectors {
        match run_one(backend, v, tol) {
            Ok(()) => passed += 1,
            Err(reason) => failed.push(FailedCase {
                id: v.id.clone(),
                reason,
            }),
        }
    }
    Report { passed, failed }
}

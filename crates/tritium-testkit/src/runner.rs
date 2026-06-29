//! The generic conformance runner: replay vectors against any backend.

use half::f16;
use tritium_core::{GemmShape, TernaryFormat, Trit};
use tritium_format::{num_blocks, pack_tq1_0_row, pack_tq2_0_row};
use tritium_spec::{MpGemm, TernaryBackend};

use crate::reference_backend::ReferenceBackend;
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

    /// Panic with the failing ids and reasons unless every vector passed.
    ///
    /// The assertion-style companion to [`is_ok`](Self::is_ok): use it in a test
    /// to turn a non-conformant [`Report`] into a readable failure.
    ///
    /// # Panics
    /// Panics if [`failed`](Self::failed) is non-empty.
    pub fn assert_conformant(&self) {
        assert!(self.is_ok(), "{self}");
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "conformance: {}/{} passed", self.passed, self.total())?;
        if self.failed.is_empty() {
            return Ok(());
        }
        write!(f, "; {} failed:", self.failed.len())?;
        for case in &self.failed {
            write!(f, "\n  - {}: {}", case.id, case.reason)?;
        }
        Ok(())
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
    let format = v.format;
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
        .mpgemm(MpGemm {
            act: &v.activation,
            weights: buffer.as_ref(),
            scales: &v.scales,
            shape,
            format,
            out: &mut out,
        })
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

/// Per-row activation scale `γ_m / 127` (`γ_m = max_k |act[m,k]|`) — the same
/// per-token absmax the fused W1.58A8 path applies. Used as the tolerance floor
/// in [`run_one_fused`]: the fused output is the unscaled mpGEMM product scaled
/// by this factor, so lifting the plain `max(1, |P|)` floor through it
/// (`max(s_m, |want|)`) keeps the grade sound w.r.t. the mpGEMM contract even
/// when `γ_m > 127` (scale ≥ 1).
fn act_row_scales(act: &[f32], m: usize, k: usize) -> Vec<f32> {
    (0..m)
        .map(|r| {
            let gamma = act[r * k..r * k + k]
                .iter()
                .fold(0.0_f32, |acc, &v| acc.max(v.abs()));
            gamma / 127.0
        })
        .collect()
}

/// Run one vector through the **fused** path, grading against the host-default
/// reference.
fn run_one_fused<B: TernaryBackend>(
    backend: &B,
    reference: &ReferenceBackend,
    v: &ConformanceVector,
    tol: Tolerance,
) -> Result<(), FailureReason> {
    let format = v.format;
    let shape = GemmShape::new(v.m, v.n, v.k);

    // Same internal-consistency checks as the plain runner.
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

    // Reference fused output: the spec's host-default W1.58A8 path
    // (`ReferenceBackend` does not override `mpgemm_with_act_quant`), so this is
    // the canonical "host-A8" answer independent of `v.expected` (which is the
    // *plain* mpGEMM reference).
    let ref_buf = reference
        .upload_weights(&packed, shape, format)
        .map_err(|e| FailureReason::BackendError(format!("reference upload: {e}")))?;
    let mut want = vec![0.0f32; v.m * v.n];
    reference
        .mpgemm_with_act_quant(MpGemm {
            act: &v.activation,
            weights: ref_buf.as_ref(),
            scales: &v.scales,
            shape,
            format,
            out: &mut want,
        })
        .map_err(|e| FailureReason::BackendError(format!("reference fused: {e}")))?;

    // Subject backend fused output. An error here is itself a contract failure:
    // a backend lacking fp8/IMMA must *degrade* to the host path and return a
    // result, never refuse.
    let buffer = backend
        .upload_weights(&packed, shape, format)
        .map_err(|e| FailureReason::BackendError(e.to_string()))?;
    let mut got = vec![0.0f32; v.m * v.n];
    backend
        .mpgemm_with_act_quant(MpGemm {
            act: &v.activation,
            weights: buffer.as_ref(),
            scales: &v.scales,
            shape,
            format,
            out: &mut got,
        })
        .map_err(|e| FailureReason::BackendError(e.to_string()))?;

    // Grade with a per-row floor of `act_scale[m]` rather than the fixed 1.0:
    // the fused output is the mpGEMM product scaled per token, so this keeps the
    // tolerance equivalent to the mpGEMM contract for any activation magnitude.
    let row_scales = act_row_scales(&v.activation, v.m, v.k);
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        if !tol.accepts_with_floor(g, w, row_scales[i / v.n]) {
            return Err(FailureReason::Mismatch {
                index: i,
                expected: w,
                got: g,
            });
        }
    }
    Ok(())
}

/// Run the **fused W1.58A8 fallback contract** against a backend.
///
/// Where [`run_conformance`] exercises the plain [`TernaryBackend::mpgemm`], this
/// exercises [`TernaryBackend::mpgemm_with_act_quant`] — the fused activation-quant
/// path a BitNet linear layer actually calls. The contract every backend must
/// honour is *graceful degradation*: a device that advertises no fp8 / IMMA
/// acceleration (`capabilities().supports_fp8 == false`, no `i2s_int8` feature)
/// must still **serve** the fused path by falling back to the host quant —
/// returning `Ok` with a correct result, never an error, never a panic. The
/// trait's default impl provides exactly this fallback; pinning it here means a
/// new backend (wasm, wgpu, …) cannot regress it, and an accelerated override is
/// held to the same numbers (the "fused == host-A8" gate of ADR 0005).
///
/// Each vector's inputs are reused: `activation` is quantized by the fused path,
/// `weights` are packed + uploaded, and `scales` serve as the per-channel
/// `weight_scales`. The graded reference is the host-default fused output obtained
/// by running the same call on [`ReferenceBackend`], so this contract is
/// independent of [`ConformanceVector::expected`] (the plain-mpGEMM reference). A
/// backend that errors out of the fused call, whose buffer is rejected, or that
/// diverges beyond `tol` becomes a [`FailedCase`]; the function itself never
/// panics.
///
/// ```
/// use tritium_testkit::{generate_vectors, run_fused_fallback_contract, Tolerance};
/// # use tritium_testkit::reference_backend_for_doctest as backend;
/// let vectors = generate_vectors(7, 4);
/// let report = run_fused_fallback_contract(&backend(), &vectors, Tolerance::default());
/// assert!(report.is_ok());
/// ```
pub fn run_fused_fallback_contract<B: TernaryBackend>(
    backend: &B,
    vectors: &[ConformanceVector],
    tol: Tolerance,
) -> Report {
    let reference = ReferenceBackend::new();
    let mut passed = 0;
    let mut failed = Vec::new();
    for v in vectors {
        match run_one_fused(backend, &reference, v, tol) {
            Ok(()) => passed += 1,
            Err(reason) => failed.push(FailedCase {
                id: v.id.clone(),
                reason,
            }),
        }
    }
    Report { passed, failed }
}

#[cfg(test)]
mod fallback_tests {
    use core::any::Any;

    use tritium_core::{GemmShape, TernaryFormat};
    use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, MpGemm, TernaryBackend};

    use super::{FailureReason, run_fused_fallback_contract};
    use crate::generate_vectors;
    use crate::reference_backend::ReferenceBackend;
    use crate::vector::Tolerance;

    /// The reference backend degrades for free (no fused override) and is correct.
    #[test]
    fn fused_fallback_passes_for_reference_backend() {
        let vectors = generate_vectors(0x5EED, 8);
        let report =
            run_fused_fallback_contract(&ReferenceBackend::new(), &vectors, Tolerance::default());
        assert!(
            report.is_ok(),
            "reference backend must serve the fused path: {:?}",
            report.failed
        );
        assert_eq!(report.passed, vectors.len());
    }

    /// A zero-byte buffer that uploads fine but carries nothing — enough for a
    /// backend whose fused path refuses before touching it.
    struct NoopBuf;
    impl DeviceBuffer for NoopBuf {
        fn len_bytes(&self) -> usize {
            0
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// A backend that advertises no fp8 yet **errors** out of the fused path
    /// instead of degrading — the exact failure mode the contract must catch.
    struct RefusesToDegrade;
    impl TernaryBackend for RefusesToDegrade {
        fn device_id(&self) -> &str {
            "refuses"
        }
        fn capabilities(&self) -> DeviceCaps {
            DeviceCaps::new("refuses", "errors instead of falling back")
        }
        fn upload_weights(
            &self,
            _packed: &[u8],
            _shape: GemmShape,
            _format: TernaryFormat,
        ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
            Ok(Box::new(NoopBuf))
        }
        fn mpgemm(&self, _p: MpGemm<'_>) -> Result<(), BackendError> {
            Err(BackendError::Backend("no mpgemm".into()))
        }
        fn mpgemm_with_act_quant(&self, _p: MpGemm<'_>) -> Result<(), BackendError> {
            Err(BackendError::Backend(
                "refuses fused; does not degrade".into(),
            ))
        }
    }

    /// Teeth: a backend that refuses to degrade fails every vector with a
    /// `BackendError` — proving the contract is not a no-op.
    #[test]
    fn fused_fallback_flags_a_backend_that_refuses_to_degrade() {
        let vectors = generate_vectors(0x5EED, 4);
        let report = run_fused_fallback_contract(&RefusesToDegrade, &vectors, Tolerance::default());
        assert!(
            !report.is_ok(),
            "a backend that errors instead of degrading must fail the contract"
        );
        assert_eq!(report.failed.len(), vectors.len());
        assert!(matches!(
            report.failed[0].reason,
            FailureReason::BackendError(_)
        ));
    }

    /// A backend that returns `Ok` from the fused path but writes grossly wrong
    /// numbers — degrades without erroring, yet is incorrect.
    struct WrongFused;
    impl TernaryBackend for WrongFused {
        fn device_id(&self) -> &str {
            "wrong"
        }
        fn capabilities(&self) -> DeviceCaps {
            DeviceCaps::new("wrong", "returns Ok with bogus output")
        }
        fn upload_weights(
            &self,
            _packed: &[u8],
            _shape: GemmShape,
            _format: TernaryFormat,
        ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
            Ok(Box::new(NoopBuf))
        }
        fn mpgemm(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
            p.out.fill(1.0e30);
            Ok(())
        }
        fn mpgemm_with_act_quant(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
            p.out.fill(1.0e30);
            Ok(())
        }
    }

    /// Teeth: a backend that degrades (returns `Ok`) but computes wrong output is
    /// flagged with `Mismatch` — exercising the numeric-divergence path, not just
    /// the error path. (Bounded reference outputs can never reach `1e30`.)
    #[test]
    fn fused_fallback_flags_a_backend_that_degrades_but_is_wrong() {
        let vectors = generate_vectors(0x5EED, 4);
        let report = run_fused_fallback_contract(&WrongFused, &vectors, Tolerance::default());
        assert!(
            !report.is_ok(),
            "a backend returning bogus fused output must fail the contract"
        );
        assert_eq!(report.failed.len(), vectors.len());
        assert!(matches!(
            report.failed[0].reason,
            FailureReason::Mismatch { .. }
        ));
    }
}

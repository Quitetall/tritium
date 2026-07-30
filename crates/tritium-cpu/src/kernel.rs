//! The two ternary mpGEMM kernels and their runtime dispatch.
//!
//! Both kernels compute the identical contraction
//!
//! ```text
//! out[m, n] = scale[n] · Σ_k act[m, k] · trit[n, k]
//! ```
//!
//! over already-unpacked [`Trit`]s (`{-1, 0, +1}`), expressed in the reference's
//! **add / subtract / skip** form — never a literal multiply by the trit.
//!
//! - [`scalar_mpgemm`] delegates to [`tritium_core::reference_mpgemm`], so it is
//!   correct by construction and serves as both the fallback and the ground truth
//!   the AVX2 path is graded against.
//! - [`avx2_mpgemm`] is an `#[target_feature(enable = "avx2")]` kernel using
//!   `core::arch::x86_64` intrinsics. It vectorises the trit decode and the
//!   per-element add/sub/skip into a signed contribution vector, then folds that
//!   contribution **sequentially in f32 in k-order** — reproducing the reference's
//!   single-accumulator rounding bit-for-bit (any reordered or re-widened
//!   reduction drifts by ~1e-4 at K=512, the conformance floor for near-zero
//!   outputs). It therefore matches the scalar/reference output exactly.
//!
//! [`dispatch_mpgemm`] picks the AVX2 path at runtime when the host advertises
//! `avx2`, and the scalar path otherwise, parallelising over the independent `M`
//! rows with `rayon`.

use rayon::prelude::*;
use tritium_core::{GemmShape, Trit, reference_mpgemm};
use tritium_format::{QK_K, TQ2_0_BLOCK_BYTES};

use tritium_spec::BackendError;

/// Run the ternary mpGEMM with the best kernel the host supports, parallelising
/// over the independent `M` rows.
///
/// `weights` are the already-unpacked `[N, K]` trits (output-major). The shape
/// and buffer lengths are validated up front; on mismatch the scalar reference's
/// typed error is surfaced as [`BackendError::Core`].
///
/// The `M` rows of a mpGEMM are mutually independent (each reads its own `act`
/// row and writes its own `out` row, sharing only the read-only weights/scales),
/// so the work splits cleanly across `rayon`'s thread pool. Each task runs the
/// same single-threaded kernel on a disjoint chunk of rows, so the numeric result
/// is identical to the serial path regardless of thread count — the per-row
/// accumulation order is unchanged.
///
/// # Errors
/// [`BackendError::Core`] if a buffer length disagrees with `shape`.
pub(crate) fn dispatch_mpgemm(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), BackendError> {
    let GemmShape { m, n, k } = shape;
    // Validate here so the per-chunk shapes below are exact and the kernels never
    // index out of bounds.
    if act.len() != m * k || weights.len() != n * k || scales.len() != n || out.len() != m * n {
        // Reuse the reference's typed `ShapeMismatch` for the precise lengths.
        return reference_mpgemm(act, weights, scales, shape, out).map_err(BackendError::Core);
    }
    if m == 0 || n == 0 {
        return Ok(());
    }

    // Split the M rows into chunks and run one kernel invocation per chunk in
    // parallel. `par_chunks_mut(k|n)` yields disjoint, non-overlapping slices, so
    // there is no aliasing between tasks.
    let chunk_rows = row_chunk(m, n, k);
    out.par_chunks_mut(chunk_rows * n)
        .zip(act.par_chunks(chunk_rows * k))
        .try_for_each(|(out_chunk, act_chunk)| {
            let rows = out_chunk.len() / n;
            let chunk_shape = GemmShape::new(rows, n, k);
            run_chunk(act_chunk, weights, scales, chunk_shape, out_chunk)
        })
}

/// VJP for `Y = A · (scale * trit)ᵀ`, retaining no dense weight shadow.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_mpgemm_projected_vjp(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    grad_output: &[f32],
    shape: GemmShape,
    grad_act: &mut [f32],
    grad_projected_weight: &mut [f32],
    grad_bias: Option<&mut [f32]>,
) -> Result<(), BackendError> {
    let GemmShape { m, n, k } = shape;
    let mk = m
        .checked_mul(k)
        .ok_or_else(|| BackendError::InvalidInput("M*K element count overflows".to_owned()))?;
    let nk = n
        .checked_mul(k)
        .ok_or_else(|| BackendError::InvalidInput("N*K element count overflows".to_owned()))?;
    let mn = m
        .checked_mul(n)
        .ok_or_else(|| BackendError::InvalidInput("M*N element count overflows".to_owned()))?;
    let lengths = [
        (act.len(), mk),
        (weights.len(), nk),
        (scales.len(), n),
        (grad_output.len(), mn),
        (grad_act.len(), mk),
        (grad_projected_weight.len(), nk),
    ];
    if let Some(&(got, expected)) = lengths.iter().find(|(got, expected)| got != expected) {
        return Err(BackendError::ShapeMismatch { expected, got });
    }
    if let Some(values) = grad_bias.as_ref()
        && values.len() != n
    {
        return Err(BackendError::ShapeMismatch {
            expected: n,
            got: values.len(),
        });
    }

    grad_act.fill(0.0);
    grad_projected_weight.fill(0.0);
    if m == 0 || n == 0 {
        if let Some(values) = grad_bias {
            values.fill(0.0);
        }
        return Ok(());
    }

    if k > 0 {
        #[cfg(target_arch = "x86_64")]
        let used_simd = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            avx2_projected_vjp(
                act,
                weights,
                scales,
                grad_output,
                shape,
                grad_act,
                grad_projected_weight,
            );
            true
        } else {
            false
        };
        #[cfg(not(target_arch = "x86_64"))]
        let used_simd = false;

        if !used_simd {
            grad_act
                .par_chunks_mut(k)
                .enumerate()
                .for_each(|(mi, grad_row)| {
                    for ni in 0..n {
                        let upstream = grad_output[mi * n + ni];
                        let weight_row = &weights[ni * k..(ni + 1) * k];
                        for (grad_value, &trit) in grad_row.iter_mut().zip(weight_row) {
                            let projected = trit.to_f32() * scales[ni];
                            *grad_value += upstream * projected;
                        }
                    }
                });

            grad_projected_weight
                .par_chunks_mut(k)
                .enumerate()
                .for_each(|(ni, grad_row)| {
                    for mi in 0..m {
                        let upstream = grad_output[mi * n + ni];
                        let act_row = &act[mi * k..(mi + 1) * k];
                        for (grad_value, &act_value) in grad_row.iter_mut().zip(act_row) {
                            *grad_value += upstream * act_value;
                        }
                    }
                });
        }
    }

    if let Some(values) = grad_bias {
        values.par_iter_mut().enumerate().for_each(|(ni, value)| {
            let mut acc = 0.0f32;
            for mi in 0..m {
                acc += grad_output[mi * n + ni];
            }
            *value = acc;
        });
    }
    Ok(())
}

/// TQ2_0 projected VJP decoded directly from packed bytes.
///
/// Caller supplies bytes whose TQ2_0 codes were validated when their
/// [`crate::CpuBuffer`] was uploaded. Shape lengths are revalidated here before
/// writing any output, without allocating an `N*K` dense trit shadow.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_tq2_projected_vjp(
    act: &[f32],
    packed_weights: &[u8],
    scales: &[f32],
    grad_output: &[f32],
    shape: GemmShape,
    grad_act: &mut [f32],
    grad_projected_weight: &mut [f32],
    grad_bias: Option<&mut [f32]>,
) -> Result<(), BackendError> {
    let GemmShape { m, n, k } = shape;
    let blocks = k.div_ceil(QK_K);
    let row_bytes = blocks.checked_mul(TQ2_0_BLOCK_BYTES).ok_or_else(|| {
        BackendError::InvalidInput("TQ2_0 packed row byte count overflows".to_owned())
    })?;
    let packed_len = n.checked_mul(row_bytes).ok_or_else(|| {
        BackendError::InvalidInput("TQ2_0 packed weight byte count overflows".to_owned())
    })?;
    if packed_weights.len() != packed_len {
        return Err(BackendError::ShapeMismatch {
            expected: packed_len,
            got: packed_weights.len(),
        });
    }

    let mk = m
        .checked_mul(k)
        .ok_or_else(|| BackendError::InvalidInput("M*K element count overflows".to_owned()))?;
    let nk = n
        .checked_mul(k)
        .ok_or_else(|| BackendError::InvalidInput("N*K element count overflows".to_owned()))?;
    let mn = m
        .checked_mul(n)
        .ok_or_else(|| BackendError::InvalidInput("M*N element count overflows".to_owned()))?;
    let lengths = [
        (act.len(), mk),
        (scales.len(), n),
        (grad_output.len(), mn),
        (grad_act.len(), mk),
        (grad_projected_weight.len(), nk),
    ];
    if let Some(&(got, expected)) = lengths.iter().find(|(got, expected)| got != expected) {
        return Err(BackendError::ShapeMismatch { expected, got });
    }
    if let Some(values) = grad_bias.as_ref()
        && values.len() != n
    {
        return Err(BackendError::ShapeMismatch {
            expected: n,
            got: values.len(),
        });
    }

    grad_act.fill(0.0);
    grad_projected_weight.fill(0.0);
    if m == 0 || n == 0 {
        if let Some(values) = grad_bias {
            values.fill(0.0);
        }
        return Ok(());
    }

    if k > 0 {
        #[cfg(target_arch = "x86_64")]
        let used_simd = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            avx2_tq2_projected_vjp(
                act,
                packed_weights,
                scales,
                grad_output,
                shape,
                row_bytes,
                grad_act,
                grad_projected_weight,
            );
            true
        } else {
            false
        };
        #[cfg(not(target_arch = "x86_64"))]
        let used_simd = false;

        if !used_simd {
            grad_act
                .par_chunks_mut(k)
                .enumerate()
                .for_each(|(mi, grad_row)| {
                    for ni in 0..n {
                        let upstream = grad_output[mi * n + ni];
                        let row = &packed_weights[ni * row_bytes..(ni + 1) * row_bytes];
                        for (ki, grad_value) in grad_row.iter_mut().enumerate() {
                            let trit = decode_tq2_trit(row, ki);
                            *grad_value += upstream * (trit * scales[ni]);
                        }
                    }
                });

            grad_projected_weight
                .par_chunks_mut(k)
                .enumerate()
                .for_each(|(ni, grad_row)| {
                    for mi in 0..m {
                        let upstream = grad_output[mi * n + ni];
                        let act_row = &act[mi * k..(mi + 1) * k];
                        for (grad_value, &act_value) in grad_row.iter_mut().zip(act_row) {
                            *grad_value += upstream * act_value;
                        }
                    }
                });
        }
    }

    if let Some(values) = grad_bias {
        values.par_iter_mut().enumerate().for_each(|(ni, value)| {
            let mut acc = 0.0f32;
            for mi in 0..m {
                acc += grad_output[mi * n + ni];
            }
            *value = acc;
        });
    }
    Ok(())
}

#[inline]
fn decode_tq2_trit(row: &[u8], ki: usize) -> f32 {
    let block = ki / QK_K;
    let element = ki % QK_K;
    let chunk = element >> 7;
    let lane = (element & 127) >> 5;
    let byte = element & 31;
    let packed = row[block * TQ2_0_BLOCK_BYTES + chunk * 32 + byte];
    f32::from((packed >> (2 * lane)) & 3) - 1.0
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn avx2_tq2_projected_vjp(
    act: &[f32],
    packed_weights: &[u8],
    scales: &[f32],
    grad_output: &[f32],
    shape: GemmShape,
    row_bytes: usize,
    grad_act: &mut [f32],
    grad_projected_weight: &mut [f32],
) {
    let GemmShape { m, n, k } = shape;
    grad_act
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(mi, grad_row)| {
            // SAFETY: runtime dispatch checked AVX2+FMA. Packed payload and
            // output lengths were validated before parallel work began.
            unsafe {
                avx2_tq2_grad_act_row(
                    grad_row,
                    packed_weights,
                    scales,
                    &grad_output[mi * n..(mi + 1) * n],
                    n,
                    k,
                    row_bytes,
                );
            }
        });
    grad_projected_weight
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(ni, grad_row)| {
            // SAFETY: same AVX2+FMA dispatch and validated slices.
            unsafe {
                avx2_grad_projected_weight_row(grad_row, act, grad_output, ni, m, n, k);
            }
        });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_tq2_grad_act_row(
    grad_row: &mut [f32],
    packed_weights: &[u8],
    scales: &[f32],
    upstream: &[f32],
    n: usize,
    k: usize,
    row_bytes: usize,
) {
    use core::arch::x86_64::{
        _mm_and_si128, _mm_loadl_epi64, _mm_set1_epi8, _mm_srli_epi16, _mm_sub_epi8,
        _mm256_cvtepi8_epi32, _mm256_cvtepi32_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_set1_ps, _mm256_storeu_ps,
    };

    const LANES: usize = 8;
    let code_mask = _mm_set1_epi8(3);
    let one = _mm_set1_epi8(1);
    for ni in 0..n {
        let upstream_vector = _mm256_set1_ps(upstream[ni]);
        let scale_vector = _mm256_set1_ps(scales[ni]);
        let row = &packed_weights[ni * row_bytes..(ni + 1) * row_bytes];
        for block_index in 0..k.div_ceil(QK_K) {
            let block_offset = block_index * TQ2_0_BLOCK_BYTES;
            let block_k = block_index * QK_K;
            for chunk in 0..2 {
                for lane in 0..4 {
                    let group_k = block_k + chunk * 128 + lane * 32;
                    if group_k >= k {
                        continue;
                    }
                    let group_len = (k - group_k).min(32);
                    let simd_len = group_len - group_len % LANES;
                    let mut byte_index = 0;
                    while byte_index < simd_len {
                        let packed_offset = block_offset + chunk * 32 + byte_index;
                        // SAFETY: one TQ2_0 block has 64 qs bytes; byte_index
                        // advances through at most 32 bytes in this chunk.
                        let bytes =
                            unsafe { _mm_loadl_epi64(row.as_ptr().add(packed_offset).cast()) };
                        let shifted = match lane {
                            0 => bytes,
                            1 => _mm_srli_epi16::<2>(bytes),
                            2 => _mm_srli_epi16::<4>(bytes),
                            3 => _mm_srli_epi16::<6>(bytes),
                            _ => unreachable!(),
                        };
                        let codes = _mm_and_si128(shifted, code_mask);
                        let trits = _mm_sub_epi8(codes, one);
                        let trits = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(trits));
                        let projected = _mm256_mul_ps(trits, scale_vector);
                        let grad_offset = group_k + byte_index;
                        // SAFETY: group_len is clipped to logical K.
                        let prior = unsafe { _mm256_loadu_ps(grad_row.as_ptr().add(grad_offset)) };
                        let next = _mm256_fmadd_ps(upstream_vector, projected, prior);
                        // SAFETY: same clipped bound covers output store.
                        unsafe {
                            _mm256_storeu_ps(grad_row.as_mut_ptr().add(grad_offset), next);
                        }
                        byte_index += LANES;
                    }
                    for tail in simd_len..group_len {
                        let ki = group_k + tail;
                        let trit = decode_tq2_trit(row, ki);
                        grad_row[ki] += upstream[ni] * (trit * scales[ni]);
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_projected_vjp(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    grad_output: &[f32],
    shape: GemmShape,
    grad_act: &mut [f32],
    grad_projected_weight: &mut [f32],
) {
    let GemmShape { m, n, k } = shape;
    grad_act
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(mi, grad_row)| {
            // SAFETY: runtime dispatch checked AVX2+FMA; validated slices cover
            // this complete output row and all shared inputs.
            unsafe {
                avx2_grad_act_row(
                    grad_row,
                    weights,
                    scales,
                    &grad_output[mi * n..(mi + 1) * n],
                    n,
                    k,
                );
            }
        });
    grad_projected_weight
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(ni, grad_row)| {
            // SAFETY: same AVX2+FMA dispatch; validated slices cover every
            // strided upstream value and activation row.
            unsafe {
                avx2_grad_projected_weight_row(grad_row, act, grad_output, ni, m, n, k);
            }
        });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_grad_act_row(
    grad_row: &mut [f32],
    weights: &[Trit],
    scales: &[f32],
    upstream: &[f32],
    n: usize,
    k: usize,
) {
    use core::arch::x86_64::{
        _mm_loadl_epi64, _mm256_cvtepi8_epi32, _mm256_cvtepi32_ps, _mm256_fmadd_ps,
        _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
    };

    const LANES: usize = 8;
    let simd_k = k - k % LANES;
    for ni in 0..n {
        let upstream_vector = _mm256_set1_ps(upstream[ni]);
        let scale_vector = _mm256_set1_ps(scales[ni]);
        let weight_row = &weights[ni * k..(ni + 1) * k];
        let mut ki = 0;
        while ki < simd_k {
            // SAFETY: loop keeps `ki + LANES <= k`; Trit is repr-transparent i8.
            let trits = unsafe { _mm_loadl_epi64(weight_row.as_ptr().add(ki).cast()) };
            let trits = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(trits));
            let projected = _mm256_mul_ps(trits, scale_vector);
            // SAFETY: grad row has k elements; same loop bound covers load/store.
            let prior = unsafe { _mm256_loadu_ps(grad_row.as_ptr().add(ki)) };
            let next = _mm256_fmadd_ps(upstream_vector, projected, prior);
            // SAFETY: `ki + LANES <= k`; store stays within grad row.
            unsafe { _mm256_storeu_ps(grad_row.as_mut_ptr().add(ki), next) };
            ki += LANES;
        }
        for ki in simd_k..k {
            grad_row[ki] += upstream[ni] * (weight_row[ki].to_f32() * scales[ni]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_grad_projected_weight_row(
    grad_row: &mut [f32],
    act: &[f32],
    grad_output: &[f32],
    ni: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    use core::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

    const LANES: usize = 8;
    let simd_k = k - k % LANES;
    for mi in 0..m {
        let upstream = grad_output[mi * n + ni];
        let upstream_vector = _mm256_set1_ps(upstream);
        let act_row = &act[mi * k..(mi + 1) * k];
        let mut ki = 0;
        while ki < simd_k {
            // SAFETY: loop keeps `ki + LANES <= k` for all three row slices.
            let activation = unsafe { _mm256_loadu_ps(act_row.as_ptr().add(ki)) };
            // SAFETY: same bound keeps read within grad row.
            let prior = unsafe { _mm256_loadu_ps(grad_row.as_ptr().add(ki)) };
            let next = _mm256_fmadd_ps(upstream_vector, activation, prior);
            // SAFETY: same bound keeps write within grad row.
            unsafe { _mm256_storeu_ps(grad_row.as_mut_ptr().add(ki), next) };
            ki += LANES;
        }
        for ki in simd_k..k {
            grad_row[ki] += upstream * act_row[ki];
        }
    }
}

/// Rows per parallel task. A coarse heuristic: keep at least a few thousand
/// add/sub/skip ops per task so the rayon overhead is amortised, and never split
/// finer than one row.
fn row_chunk(m: usize, n: usize, k: usize) -> usize {
    // Target ~64K inner ops per chunk; clamp to [1, m].
    const TARGET_OPS: usize = 1 << 16;
    let per_row = n.saturating_mul(k).max(1);
    (TARGET_OPS / per_row).clamp(1, m)
}

/// Run the best available kernel on one chunk of `M` rows (serial within the
/// chunk; the chunk itself is one rayon task).
fn run_chunk(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), BackendError> {
    #[cfg(target_arch = "x86_64")]
    let k = shape.k;
    #[cfg(target_arch = "x86_64")]
    {
        // A8 fast path (v1.x): when every activation is an integer in [-128, 127]
        // — the W1.58A8 protocol `TernaryLinear` always feeds this backend — the
        // whole contraction is EXACT integer arithmetic (|Σ| ≤ 128·k < 2²⁴ for
        // k ≤ 65536, so even the reference's f32 fold never rounds), which makes
        // an int8 `maddubs` kernel **bit-identical** to the sequential reference
        // fold — measured ~2x on the end-to-end BitNet CPU decode (the kernel
        // itself is larger; other ops share the budget). It also preempts the
        // AVX-512 kernel below: both are bit-identical, and 32 int8 elems/op
        // beats 16 f32 lanes on paper, though that routing preference is
        // unmeasured on an AVX-512 host (this build box is AVX2). Arbitrary-float
        // callers fall through to the bit-exact f32 kernels unchanged; the
        // detection scan is O(m·k), noise next to the O(m·n·k) GEMM.
        if is_x86_feature_detected!("avx2") && k <= A8_EXACT_K_MAX && act_is_a8_integer(act) {
            // AVX-VNNI (Alder/Raptor Lake +): one vpdpbusd where the AVX2 path
            // needs maddubs+madd+add — bit-identical (exact integer arithmetic
            // either way), ~1/3 the port-0/1 µops in the inner loop.
            if is_x86_feature_detected!("avxvnni") {
                // SAFETY: `avxvnni`+`avx2` just confirmed; lengths re-validated inside.
                return unsafe { avx2vnni_mpgemm_a8(act, weights, scales, shape, out) };
            }
            // SAFETY: `avx2` was just confirmed; buffer lengths were validated by
            // `dispatch_mpgemm` and are re-validated inside before any intrinsic.
            return unsafe { avx2_mpgemm_a8(act, weights, scales, shape, out) };
        }
        // Prefer the widest kernel the host can execute. AVX-512 (with the BW +
        // VL subsets, needed for the byte-lane trit compares that emit a mask from
        // a 128-bit operand) is selected first where present; otherwise AVX2;
        // otherwise scalar. The AVX-512 kernel compiles on every x86-64 target but
        // only runs on an AVX-512 host, so on the AVX2 build box this branch is
        // inert and the AVX2 path below runs. All three share the reference's
        // k-order f32 fold, so the choice is bit-neutral.
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: `avx512_mpgemm` requires the `avx512f` + `avx512bw` +
            // `avx512vl` target features, all just confirmed present by the runtime
            // checks above. All buffer lengths were validated by `dispatch_mpgemm`
            // and are re-validated inside the call before any intrinsic touches
            // memory.
            return unsafe { crate::simd::avx512::avx512_mpgemm(act, weights, scales, shape, out) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2_mpgemm` requires the `avx2` target feature, which the
            // runtime check immediately above has just confirmed is present on
            // this CPU. All buffer lengths were validated by `dispatch_mpgemm` and
            // are re-validated inside the call before any intrinsic touches memory.
            return unsafe { avx2_mpgemm(act, weights, scales, shape, out) };
        }
    }
    // Baseline NEON is mandatory on aarch64; select it unconditionally there.
    // NEON shares the x86 kernels' per-element k-order f32 fold, so its output is
    // bit-identical to the scalar reference (and to the AVX2/AVX-512 paths on
    // x86). Written as the function tail (not an early `return`) so the trailing
    // scalar fallback is only compiled where no SIMD kernel applies — no
    // unreachable-code warning.
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `neon_mpgemm` requires the `neon` target feature. On aarch64,
        // baseline NEON (`Advanced SIMD`) is part of the mandatory architecture —
        // it is always present, so the precondition holds unconditionally on this
        // `#[cfg(target_arch = "aarch64")]` path. All buffer lengths were
        // validated by `dispatch_mpgemm` and are re-validated inside the call.
        unsafe { crate::simd::neon::neon_mpgemm(act, weights, scales, shape, out) }
    }
    // Universal terminal fallback: the bit-exact scalar reference. Reached on
    // x86-64 with neither AVX-512 nor AVX2 (very old CPUs / restricted CPUID), or
    // any architecture with no SIMD kernel. Kept deliberately bit-exact (not the
    // tolerance-only T-MAC LUT) so cross-host output stays byte-reproducible for
    // golden-file / deterministic-checkpoint callers; the LUT
    // (`crate::simd::lut::lut_mpgemm`) is implemented and tested but its
    // production wiring is the per-ISA SIMD gather, deferred to a later step.
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_mpgemm(act, weights, scales, shape, out)
    }
}

/// The scalar ground-truth kernel: delegate to [`tritium_core::reference_mpgemm`].
///
/// # Errors
/// [`BackendError::Core`] if a buffer length disagrees with `shape`.
pub(crate) fn scalar_mpgemm(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), BackendError> {
    reference_mpgemm(act, weights, scales, shape, out).map_err(BackendError::Core)
}

/// AVX2 ternary mpGEMM. Same contraction as [`scalar_mpgemm`], vectorised over
/// the `K` dimension eight `f32` lanes at a time.
///
/// For each output it vectorises the trit decode and the per-element
/// add/sub/skip — masking the activation to `+a` on `+1` lanes, `-a` on `-1`
/// lanes, `0` on `0` lanes (no multiply by the trit) — into a signed contribution
/// vector, then folds that vector **sequentially in f32 in k-order**. That fold
/// reproduces the reference's single-`f32`-accumulator rounding bit-for-bit:
/// `acc + 0.0 == acc` (skip) and `acc + (-a) == acc - a` (sub) are exact in IEEE
/// round-to-nearest, so the result is identical to [`scalar_mpgemm`], not merely
/// within tolerance. (A reordered or f64-widened SIMD reduction would be *more*
/// accurate than the reference yet drift ~1e-4 from it at K=512 — past the
/// conformance floor for near-zero outputs.) The `K` tail that does not fill a
/// lane is folded with the same scalar add/sub/skip loop, in k-order.
///
/// # Safety
/// The caller must ensure the `avx2` target feature is available on the host
/// (checked via `is_x86_feature_detected!("avx2")`). Calling this on a CPU
/// without AVX2 is undefined behaviour.
///
/// # Errors
/// [`BackendError::Core`] if a buffer length disagrees with `shape`; in that case
/// no intrinsic runs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_mpgemm(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), BackendError> {
    use core::arch::x86_64::{
        _mm256_blendv_ps, _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_cvtepi8_epi32,
        _mm256_loadu_ps, _mm256_set1_epi32, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };

    let GemmShape { m, n, k } = shape;
    // Validate up front so no intrinsic ever reads out of bounds. This mirrors
    // the reference's checks; on mismatch we return before touching SIMD.
    if act.len() != m * k || weights.len() != n * k || scales.len() != n || out.len() != m * n {
        // Defer to the scalar reference purely for its typed error; it will
        // detect the same inconsistency and return `ShapeMismatch`.
        return scalar_mpgemm(act, weights, scales, shape, out);
    }

    // Eight f32 lanes per AVX2 register.
    const LANES: usize = 8;
    let k_simd = k - (k % LANES);

    // Trits are `#[repr(transparent)]` over i8, so a `&[Trit]` is bit-identical
    // to a `&[i8]` whose elements are all in `{-1, 0, 1}`. Reinterpret for the
    // byte loads the intrinsics need.
    // SAFETY: `Trit` is `#[repr(transparent)]` over `i8` and has the same size
    // and alignment; the slice length is unchanged, so the new slice covers
    // exactly the same valid, initialised, immutably-borrowed bytes. Every value
    // is in `{-1,0,1}`, a subset of valid `i8`.
    let weights_i8: &[i8] =
        unsafe { core::slice::from_raw_parts(weights.as_ptr().cast::<i8>(), weights.len()) };

    // The reference walks `k` in one sequential **f32** accumulator. Over K up to
    // 512 that single accumulator's own rounding (partial sums reach ~150 before
    // cancelling down to ~0.2) is itself ~1e-4 — right at the conformance floor.
    // Any SIMD reduction that reorders or re-widens those adds produces a
    // *different* ~1e-4 rounding and drifts past the tolerance. So this kernel
    // uses SIMD only to decode the trits and form the signed per-element
    // contribution (`+a` / `-a` / `0`, the add/sub/skip), then accumulates that
    // contribution **sequentially in f32 in k-order**, reproducing the reference's
    // accumulation bit-for-bit. `acc + 0.0 == acc`, `acc + (-a) == acc - a`, so the
    // signed-contribution fold is identical to the reference's branchy add/sub/skip.
    //
    // Inside a `#[target_feature(enable = "avx2")]` function the AVX2 intrinsics
    // are callable without an `unsafe` block; `unsafe` is confined to the
    // intrinsics that dereference a raw pointer.
    let mut signed_buf = vec![0.0f32; k_simd];
    for mi in 0..m {
        let arow = &act[mi * k..mi * k + k];
        for ni in 0..n {
            let wrow = &weights_i8[ni * k..ni * k + k];
            let zero = _mm256_setzero_ps();

            // Vectorised pass: fill `signed_buf[ki]` with the add/sub/skip value.
            let mut ki = 0;
            while ki < k_simd {
                // SAFETY: `ki + LANES <= k_simd <= k = arow.len()`, so the eight
                // f32 starting at `arow.as_ptr().add(ki)` are in bounds and
                // initialised. `loadu` permits any alignment.
                let a = unsafe { _mm256_loadu_ps(arow.as_ptr().add(ki)) };

                // Widen 8 packed i8 trits to 8 i32 lanes.
                // SAFETY: `ki + LANES <= k = wrow.len()`, so the 8 bytes at
                // `wrow.as_ptr().add(ki)` are in bounds and initialised.
                // `_mm_loadl_epi64` reads exactly those 8 bytes; the pointer cast
                // is between integer element types of compatible size with no
                // alignment requirement (loadl is an unaligned 64-bit load).
                let lo = unsafe {
                    core::arch::x86_64::_mm_loadl_epi64(
                        wrow.as_ptr().add(ki).cast::<core::arch::x86_64::__m128i>(),
                    )
                };
                let t = _mm256_cvtepi8_epi32(lo);

                // pos_mask = (trit == 1), neg_mask = (trit == -1), as i32 lanes,
                // reinterpreted as float masks for blendv.
                let pos_maskf = _mm256_castsi256_ps(_mm256_cmpeq_epi32(t, _mm256_set1_epi32(1)));
                let neg_maskf = _mm256_castsi256_ps(_mm256_cmpeq_epi32(t, _mm256_set1_epi32(-1)));

                // Signed contribution per lane: `+a` where the trit is +1, `-a`
                // where it is -1, `0` where it is 0 (the skip). Two blends, no
                // multiply: select `+a` on the +1 lanes, then overwrite the -1
                // lanes with `-a`. The two masks are mutually exclusive.
                let neg_a = _mm256_sub_ps(zero, a);
                let plus = _mm256_blendv_ps(zero, a, pos_maskf);
                let signed = _mm256_blendv_ps(plus, neg_a, neg_maskf);

                // SAFETY: `ki + LANES <= k_simd = signed_buf.len()`, so the 8-wide
                // store is in bounds; `storeu` has no alignment requirement.
                unsafe { _mm256_storeu_ps(signed_buf.as_mut_ptr().add(ki), signed) };

                ki += LANES;
            }

            // Sequential f32 fold over the signed contributions, in k-order —
            // bit-identical to the reference's accumulator.
            let mut acc = 0.0f32;
            for &s in &signed_buf {
                acc += s;
            }
            // Tail in the same k-order, add/sub/skip in f32.
            for kt in k_simd..k {
                match wrow[kt] {
                    1 => acc += arow[kt],
                    -1 => acc -= arow[kt],
                    _ => {}
                }
            }

            out[mi * n + ni] = acc * scales[ni];
        }
    }

    Ok(())
}

/// Largest `K` the A8 integer fast path accepts: `128·k < 2²⁴` keeps every
/// partial sum (and the reference's f32 fold of the same integers) exact, so
/// the reordered int32 accumulation is bit-identical to the sequential fold.
#[cfg(target_arch = "x86_64")]
const A8_EXACT_K_MAX: usize = 65_536;

/// True iff every activation is an integer-valued f32 in [-128, 127] — the
/// shape `ops::quantize_activation_int8` always produces. NaN/inf fail the
/// `v == v.trunc()` check.
#[cfg(target_arch = "x86_64")]
fn act_is_a8_integer(act: &[f32]) -> bool {
    act.iter()
        .all(|&v| v == v.trunc() && (-128.0..=127.0).contains(&v))
}

/// AVX2 int8 ternary mpGEMM for A8-quantized (integer-valued) activations.
///
/// Per row, activations convert to `i8` once (exact — see [`act_is_a8_integer`]).
/// Ternary weights load as `i8 ∈ {-1,0,+1}` and shift to unsigned codes
/// `w + 1 ∈ {0,1,2}` so `_mm256_maddubs_epi16(codes_u8, act_i8)` (unsigned ×
/// signed) applies; the identity `Σ (w+1)·a = Σ w·a + Σ a` recovers the signed
/// sum by subtracting the row's precomputed `Σ a`. `maddubs` pair-sums cannot
/// saturate here (|code·a| ≤ 2·128, pair ≤ 512 ≪ 32767); `madd` widens to
/// exact i32 lanes. The final `(Σ w·a) as f32 · scales[n]` is the identical
/// last multiply the reference performs on the identical exact integer.
///
/// # Safety
/// Caller must ensure `avx2` is available. Buffer lengths are re-validated
/// before any intrinsic runs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_mpgemm_a8(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), BackendError> {
    use core::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm256_add_epi8,
        _mm256_add_epi32, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_setzero_si256,
    };

    let GemmShape { m, n, k } = shape;
    if act.len() != m * k || weights.len() != n * k || scales.len() != n || out.len() != m * n {
        return scalar_mpgemm(act, weights, scales, shape, out);
    }

    // SAFETY: `Trit` is `#[repr(transparent)]` over `i8` (see `avx2_mpgemm`).
    let weights_i8: &[i8] =
        unsafe { core::slice::from_raw_parts(weights.as_ptr().cast::<i8>(), weights.len()) };

    const LANES: usize = 32; // i8 lanes per AVX2 register
    let k_simd = k - (k % LANES);
    let ones16 = _mm256_set1_epi16(1);
    let one8 = _mm256_set1_epi8(1);

    let mut q8 = vec![0i8; k];
    for mi in 0..m {
        let arow = &act[mi * k..mi * k + k];
        // Exact f32 → i8 (integers in [-128, 127] by the caller's precondition),
        // plus the row sum for the code-shift identity.
        let mut row_sum: i64 = 0;
        for (dst, &v) in q8.iter_mut().zip(arow) {
            let q = v as i32 as i8;
            *dst = q;
            row_sum += i64::from(q);
        }

        for ni in 0..n {
            let wrow = &weights_i8[ni * k..ni * k + k];
            let mut acc = _mm256_setzero_si256();
            let mut ki = 0;
            while ki < k_simd {
                // SAFETY: `ki + 32 <= k_simd <= wrow.len()`; loadu is unaligned.
                let w = unsafe { _mm256_loadu_si256(wrow.as_ptr().add(ki).cast::<__m256i>()) };
                // SAFETY: `ki + 32 <= k_simd <= q8.len()`; loadu is unaligned.
                let a = unsafe { _mm256_loadu_si256(q8.as_ptr().add(ki).cast::<__m256i>()) };
                // codes = w + 1 ∈ {0,1,2} as u8 (no wrap: w ≥ -1).
                let codes = _mm256_add_epi8(w, one8);
                // u8×i8 pair-sum to i16 (exact, ≤ 512), widen to i32 (exact).
                let pairs = _mm256_maddubs_epi16(codes, a);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, ones16));
                ki += LANES;
            }
            // Horizontal i32 sum of `acc` (order-free: integer adds are exact).
            let lo = _mm256_extracti128_si256::<0>(acc);
            let hi = _mm256_extracti128_si256::<1>(acc);
            let s128 = _mm_add_epi32(lo, hi);
            let s64 = _mm_add_epi32(s128, _mm_shuffle_epi32::<0b00_01_10_11>(s128));
            let s32 = _mm_add_epi32(s64, _mm_shuffle_epi32::<0b01_00_11_10>(s64));
            let mut total: i64 = i64::from(_mm_cvtsi128_si32(s32));
            // Scalar tail in the same exact integer arithmetic.
            for kt in k_simd..k {
                total += i64::from(i32::from(wrow[kt]) + 1) * i64::from(q8[kt]);
            }
            let signed = total - row_sum; // Σ (w+1)·a − Σ a = Σ w·a, exact
            out[mi * n + ni] = signed as f32 * scales[ni];
        }
    }
    Ok(())
}

/// AVX-VNNI variant of [`avx2_mpgemm_a8`]: `vpdpbusd` fuses the
/// `maddubs → madd → add` triple into one instruction (u8×i8 products summed
/// 4-wide straight into the i32 accumulator — every 4-group |sum| ≤ 1024, so
/// the arithmetic is exact and the result is BIT-IDENTICAL to the AVX2 path
/// and the scalar reference; only the instruction count changes).
///
/// # Safety
/// Caller must ensure `avxvnni` (and `avx2`) are available and that `act`
/// holds integer-valued f32 in [-128, 127] (checked by [`act_is_a8_integer`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
pub(crate) unsafe fn avx2vnni_mpgemm_a8(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), BackendError> {
    use core::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm256_add_epi8,
        _mm256_dpbusd_avx_epi32, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_set1_epi8,
        _mm256_setzero_si256,
    };

    let GemmShape { m, n, k } = shape;
    if act.len() != m * k || weights.len() != n * k || scales.len() != n || out.len() != m * n {
        return scalar_mpgemm(act, weights, scales, shape, out);
    }

    // SAFETY: `Trit` is `#[repr(transparent)]` over `i8` (see `avx2_mpgemm`).
    let weights_i8: &[i8] =
        unsafe { core::slice::from_raw_parts(weights.as_ptr().cast::<i8>(), weights.len()) };

    const LANES: usize = 32;
    let k_simd = k - (k % LANES);
    let one8 = _mm256_set1_epi8(1);

    let mut q8 = vec![0i8; k];
    for mi in 0..m {
        let arow = &act[mi * k..mi * k + k];
        let mut row_sum: i64 = 0;
        for (dst, &v) in q8.iter_mut().zip(arow) {
            let q = v as i32 as i8;
            *dst = q;
            row_sum += i64::from(q);
        }

        for ni in 0..n {
            let wrow = &weights_i8[ni * k..ni * k + k];
            let mut acc = _mm256_setzero_si256();
            let mut ki = 0;
            while ki < k_simd {
                // SAFETY: `ki + 32 <= k_simd <= wrow.len()`; loadu is unaligned.
                let w = unsafe { _mm256_loadu_si256(wrow.as_ptr().add(ki).cast::<__m256i>()) };
                // SAFETY: `ki + 32 <= k_simd <= q8.len()`; loadu is unaligned.
                let a = unsafe { _mm256_loadu_si256(q8.as_ptr().add(ki).cast::<__m256i>()) };
                let codes = _mm256_add_epi8(w, one8);
                // u8×i8, 4-wide sum into i32 lanes — one instruction, exact.
                acc = _mm256_dpbusd_avx_epi32(acc, codes, a);
                ki += LANES;
            }
            let lo = _mm256_extracti128_si256::<0>(acc);
            let hi = _mm256_extracti128_si256::<1>(acc);
            let s128 = _mm_add_epi32(lo, hi);
            let s64 = _mm_add_epi32(s128, _mm_shuffle_epi32::<0b00_01_10_11>(s128));
            let s32 = _mm_add_epi32(s64, _mm_shuffle_epi32::<0b01_00_11_10>(s64));
            let mut total: i64 = i64::from(_mm_cvtsi128_si32(s32));
            for kt in k_simd..k {
                total += i64::from(i32::from(wrow[kt]) + 1) * i64::from(q8[kt]);
            }
            let signed = total - row_sum;
            out[mi * n + ni] = signed as f32 * scales[ni];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_core::Trit;

    fn trits(vals: &[i8]) -> Vec<Trit> {
        vals.iter().map(|&v| Trit::from_i8(v).unwrap()).collect()
    }

    #[test]
    fn scalar_matches_reference_hand_case() {
        // 1 row, 1 output, k=3: [+1,-1,+1] · [10,3,5] = 12, scaled by 2 -> 24.
        let act = [10.0, 3.0, 5.0];
        let w = trits(&[1, -1, 1]);
        let mut out = [0.0; 1];
        scalar_mpgemm(&act, &w, &[2.0], GemmShape::new(1, 1, 3), &mut out).unwrap();
        assert_eq!(out[0], 24.0);
    }

    /// Micro-benchmark (run explicitly: `cargo test -p tritium-cpu --release
    /// -- --ignored a8_paths_bench --nocapture`): AVX2 vs AVX-VNNI A8 GEMM at
    /// a BitNet decode shape.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "micro-benchmark, run with --ignored"]
    fn a8_paths_bench() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let (m, n, k) = (1usize, 2560usize, 2560usize);
        let shape = GemmShape::new(m, n, k);
        let act: Vec<f32> = (0..m * k)
            .map(|i| ((i * 37 + 11) % 255) as f32 - 127.0)
            .collect();
        let w: Vec<Trit> = (0..n * k)
            .map(|i| Trit::from_i8(((i * 31 + 7) % 3) as i8 - 1).unwrap())
            .collect();
        let scales = vec![0.01f32; n];
        let mut out = vec![0.0f32; m * n];
        let iters = 200;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            // SAFETY: avx2 confirmed above.
            unsafe { avx2_mpgemm_a8(&act, &w, &scales, shape, &mut out).unwrap() };
        }
        let t_avx2 = t0.elapsed();
        eprintln!("avx2   : {:?}/iter", t_avx2 / iters);
        if is_x86_feature_detected!("avxvnni") {
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                // SAFETY: avxvnni confirmed.
                unsafe { avx2vnni_mpgemm_a8(&act, &w, &scales, shape, &mut out).unwrap() };
            }
            let t_vnni = t0.elapsed();
            eprintln!(
                "avxvnni: {:?}/iter  ({:.2}x)",
                t_vnni / iters,
                t_avx2.as_secs_f64() / t_vnni.as_secs_f64()
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn a8_fast_path_bit_matches_scalar_reference() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("avx2 not detected — skipping A8 kernel test");
            return;
        }
        let mut s = 0xDEAD_BEEF_1234_5678u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // Integer-valued activations exactly as quantize_activation_int8 emits,
        // including the extremes; ragged tails and the BitNet widths.
        for &(m, n, k) in &[
            (1usize, 5usize, 19usize),
            (2, 4, 512),
            (1, 8, 2560),
            (3, 3, 33),
        ] {
            let act: Vec<f32> = (0..m * k)
                .map(|_| ((next() % 256) as i64 - 128) as f32)
                .collect();
            assert!(act_is_a8_integer(&act));
            let w = trits(
                &(0..n * k)
                    .map(|_| (next() % 3) as i8 - 1)
                    .collect::<Vec<_>>(),
            );
            let scales: Vec<f32> = (0..n).map(|_| (next() % 50) as f32 / 25.0 + 0.1).collect();
            let shape = GemmShape::new(m, n, k);
            let mut want = vec![0.0f32; m * n];
            scalar_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();
            let mut got = vec![0.0f32; m * n];
            // SAFETY: avx2 confirmed above.
            unsafe { avx2_mpgemm_a8(&act, &w, &scales, shape, &mut got).unwrap() };
            for (g, e) in got.iter().zip(&want) {
                assert_eq!(g.to_bits(), e.to_bits(), "got {g}, want {e} ({shape:?})");
            }
            if is_x86_feature_detected!("avxvnni") {
                let mut got = vec![0.0f32; m * n];
                // SAFETY: avxvnni confirmed above.
                unsafe { avx2vnni_mpgemm_a8(&act, &w, &scales, shape, &mut got).unwrap() };
                for (g, e) in got.iter().zip(&want) {
                    assert_eq!(
                        g.to_bits(),
                        e.to_bits(),
                        "vnni: got {g}, want {e} ({shape:?})"
                    );
                }
            }
        }
        // Non-integer / out-of-range activations must fail detection (the
        // dispatch keeps them on the bit-exact f32 kernels).
        assert!(!act_is_a8_integer(&[1.0, 2.5]));
        assert!(!act_is_a8_integer(&[1.0, 200.0]));
        assert!(!act_is_a8_integer(&[1.0, f32::NAN]));
        assert!(!act_is_a8_integer(&[-129.0]));
        assert!(act_is_a8_integer(&[-128.0, 127.0, 0.0, -0.0]));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("avx2 not detected on this host — skipping AVX2 kernel unit test");
            return;
        }
        let mut s = 0x1234_5678u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // Cover a ragged tail (16 + 3) and the block-aligned K=512 where the
        // reference's f32 cancellation is worst — the kernel must reproduce the
        // reference accumulation bit-for-bit, not merely within 1e-4.
        for &(m, n, k) in &[(3usize, 5usize, 19usize), (2, 4, 512)] {
            let act: Vec<f32> = (0..m * k)
                .map(|_| (next() % 20000) as f32 / 100.0 - 100.0)
                .collect();
            let w = trits(
                &(0..n * k)
                    .map(|_| (next() % 3) as i8 - 1)
                    .collect::<Vec<_>>(),
            );
            let scales: Vec<f32> = (0..n).map(|_| (next() % 50) as f32 / 25.0 + 0.1).collect();
            let shape = GemmShape::new(m, n, k);

            let mut want = vec![0.0f32; m * n];
            scalar_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();
            let mut got = vec![0.0f32; m * n];
            // SAFETY: avx2 was just confirmed available by the guard above.
            unsafe { avx2_mpgemm(&act, &w, &scales, shape, &mut got).unwrap() };

            for (g, e) in got.iter().zip(&want) {
                assert_eq!(
                    g.to_bits(),
                    e.to_bits(),
                    "got {g}, want {e} (shape {shape:?})"
                );
            }
        }
    }
}

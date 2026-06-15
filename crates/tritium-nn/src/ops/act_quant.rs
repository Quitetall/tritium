//! Activation quantization for BitNet's W1.58**A8** path.
//!
//! BitNet quantizes activations to int8 per token (absmax) before every ternary
//! linear; the `γ_x / Qb` scale (with `Qb = 128`) folds into the ternary GEMM
//! output. To match the reference greedy tokens we replicate this quant exactly
//! rather than running fp16 activations (see the A8 risk in the plan — the
//! rounding mode must match `BitLinear`). The ternary `mpgemm` stays f32-in, so
//! this returns the *dequantized* activations it consumes plus the per-token
//! scale.

use crate::error::NnError;

/// Per-token int8 absmax activation quant (`Qb = 128`).
///
/// Block size for the absmax scale, `Qb`. The int8 range is `[-Qb, Qb-1]`; the
/// per-token scale is `gamma_x / Qb` where `gamma_x = max(|act_row|)`.
pub const QB: f32 = 128.0;

/// Quantize a `[rows, cols]` activation tensor to int8 per token (absmax), then
/// dequantize back to `f32` for the ternary GEMM.
///
/// Writes into `out_q` the quantized-then-dequantized `f32` activations (the
/// values the ternary GEMM consumes) and into `out_scale` the per-token scale
/// `gamma_x / Qb` (one per row).
///
/// # Errors
/// [`NnError::Shape`] if `act.len()`/`out_q.len()` ≠ `rows * cols` or
/// `out_scale.len()` ≠ `rows`.
pub fn quantize_activation_int8(
    act: &[f32],
    rows: usize,
    cols: usize,
    out_q: &mut [f32],
    out_scale: &mut [f32],
) -> Result<(), NnError> {
    let _ = (act, rows, cols, out_q, out_scale);
    todo!("WF-1/WF-2: per-token int8 absmax activation quant, rounding mode matched to BitLinear")
}

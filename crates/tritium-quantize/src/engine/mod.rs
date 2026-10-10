//! Shared production numerical kernels for conversion and training (ADR 0044 D11).
//!
//! These kernels preserve the legacy F32Accum64 ladder semantics. They are the
//! numerical implementation, not yet the full typed QuantPlan/source/sink engine.
//! The f64 feedback oracle remains an explicitly separate reference profile.

mod activation_aware;
mod ladder;

pub use activation_aware::{
    ActivationAwareConfig, ScaleRefitMode, auto_decay, fit_tensor,
    fit_tensor_with_scale_refit_mode, fit_to_dense,
};
pub use ladder::{fit_ladder_group, ladder_quantize_at, ladder_step};

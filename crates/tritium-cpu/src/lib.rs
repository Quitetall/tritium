//! # tritium-cpu
//!
//! CPU execution backend. A runtime-dispatched AVX2 kernel with a scalar fallback
//! (the scalar path delegates to [`tritium_core::reference_mpgemm`], so it is
//! correct by construction). Self-registers with `tritium-runtime`. Implementation
//! in progress (v0.10 Wave C).

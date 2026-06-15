//! # tritium-testkit
//!
//! Reference conformance vectors (generated from [`tritium_core::reference_mpgemm`])
//! and a generic runner that replays them against any [`tritium_spec::TernaryBackend`]
//! implementation, asserting the tolerance. This is what makes cross-backend
//! correctness structural. Implementation in progress (v0.10 Wave B).
#![forbid(unsafe_code)]

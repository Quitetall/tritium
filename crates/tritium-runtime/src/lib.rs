//! # tritium-runtime
//!
//! Backend registry and dispatch. Backends self-register through a `linkme`
//! distributed slice, so adding one needs no central edit here. Implementation in
//! progress (v0.10 Wave B).
#![forbid(unsafe_code)]

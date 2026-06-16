//! Model assembly: GGUF weight loading, the tokenizer contract, and the runner
//! that ties config + weights + ops into token generation.
//!
//! This is the top of the inference spine; the heavy integration (full forward,
//! the fidelity ladder, the acceptance gate) lands in WF-4 as documented stubs.

mod runner;
mod tokenizer;
mod weights;

pub use runner::{ForwardDump, ModelRunner};
pub use tokenizer::Tokenizer;
pub use weights::{LayerWeights, ModelWeights};

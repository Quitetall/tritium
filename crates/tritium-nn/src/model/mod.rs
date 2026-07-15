//! Model assembly: GGUF weight loading, the tokenizer contract, and the runner
//! that ties config + weights + ops into token generation.
//!
//! This is the top of the inference spine; the heavy integration (full forward,
//! the fidelity ladder, the acceptance gate) lands in WF-4 as documented stubs.

#[cfg(feature = "tokenizer")]
mod bpe_tokenizer;
mod hf;
mod hf_shards;
mod qwen35;
mod qwen35_hf;
mod runner;
#[cfg(feature = "cuda")]
mod salt_v2;
mod tokenizer;
mod training_salt;
mod weights;

#[cfg(feature = "tokenizer")]
pub use bpe_tokenizer::GgufBpeTokenizer;
pub use qwen35::{
    Qwen35TextCache, Qwen35TextLayerWeights, Qwen35TextMixerWeights, Qwen35TextOutput,
    Qwen35TextRunner, Qwen35TextWeights,
};
pub use qwen35_hf::{Qwen35HfLanguageModel, Qwen35HfLanguageReceipt};
pub use runner::{ForwardDump, ModelRunner};
#[cfg(feature = "cuda")]
pub use salt_v2::{
    SaltV2LoadedTensorReceipt, SaltV2ModelAllocationReceipt, SaltV2PreservedTensorReceipt,
};
pub use tokenizer::Tokenizer;
pub use training_salt::{
    TRAINING_SALT_COMPLETED_STEP_KEY, TRAINING_SALT_FORMAT_KEY, TRAINING_SALT_FORMAT_VALUE,
    TRAINING_SALT_GROWTH_RECEIPT_KEY, TRAINING_SALT_HF_CONFIG_KEY,
    TRAINING_SALT_INITIAL_STUDENT_DIGEST_KEY, TRAINING_SALT_PLAN_FINGERPRINT_KEY,
    TRAINING_SALT_PLANES_KEY, TRAINING_SALT_SOURCE_MODEL_DIGEST_KEY,
    TRAINING_SALT_UNTIED_FORMAT_VALUE, TrainingSaltArtifactMetadata, TrainingSaltGrowthReceipt,
    parse_training_salt_artifact_metadata,
};
pub use weights::{LayerWeights, ModelWeights};

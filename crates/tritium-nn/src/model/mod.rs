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
mod qwen35_hf_source;
mod qwen35_mtp;
mod qwen35_mtp_oracle;
mod qwen35_salt_v2;
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
pub use qwen35_hf::{
    Qwen35HfLanguageModel, Qwen35HfLanguageMtpModel, Qwen35HfLanguageMtpReceipt,
    Qwen35HfLanguageReceipt, Qwen35MtpPromotionError, Qwen35TensorSchemaEntry,
    Qwen35TensorSchemaRole, Qwen35VerifiedHfLanguageMtpModel, qwen35_language_mtp_tensor_schema,
};
pub use qwen35_hf_source::{
    QWEN35_HF_SOURCE_ARCHITECTURE, Qwen35ContentVerifiedHfSource, Qwen35HfSource,
    Qwen35HfSourceIdentity, Qwen35HfTensorMetadata, Qwen35TensorStreamError,
    qwen36_27b_canonical_source_config,
};
pub use qwen35_mtp::{
    QWEN35_MTP_UNVERIFIED_REASON, QWEN35_MTP_VLLM_ORACLE_REVISION, QWEN35_MTP_VLLM_SOURCE_SHA256,
    Qwen35MtpCache, Qwen35MtpInputPlan, Qwen35MtpLayerWeights, Qwen35MtpOracleCoverageProfile,
    Qwen35MtpOracleEvidenceClass, Qwen35MtpOutput, Qwen35MtpParityReceipt, Qwen35MtpRunner,
    Qwen35MtpStatus, Qwen35MtpWeights, UnverifiedQwen35Mtp,
};
pub use qwen35_salt_v2::{
    Qwen35SaltV2BundleAdmission, Qwen35SaltV2LanguageMtpModel, Qwen35SaltV2LoadReceipt,
};
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

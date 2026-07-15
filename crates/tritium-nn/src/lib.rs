//! # tritium-nn
//!
//! The inference layer that turns the ternary mpGEMM primitive into a running
//! model. It sits above [`tritium_spec::TernaryBackend`]: ternary linear layers
//! call `backend.mpgemm`, while norms, softmax, RoPE, and sampling run in fp32.
//!
//! v0.20 scope (see ADR 0004): RMSNorm, RoPE, GQA attention, KV cache, sampling,
//! a transformer block, and a [`ModelConfig`]-driven model runner that loads
//! **BitNet b1.58 2B4T** from GGUF and generates tokens. This module is the
//! foundation (config + the simplest ops); the remaining ops + runner land in the
//! per-op and integration waves.
//!
//! No `unsafe` here — the numerics are plain Rust; the backend owns the SIMD/GPU.
#![forbid(unsafe_code)]

mod config;
mod error;
mod evaluation;
mod kv_cache;
mod layers;
mod model;
mod ops;
mod qwen35_config;
mod salt_v2_growth;
mod teacher_cache;
mod tensor;
mod training;

pub use config::{ArchSpec, MlpKind, ModelConfig};
pub use error::NnError;
#[cfg(feature = "cuda")]
pub use error::ResidentOpError;
pub use evaluation::{TeacherForcedPerplexity, teacher_forced_perplexity_windows};
pub use kv_cache::KvCache;
pub use layers::{
    BlockDump, BlockScratch, DenseLinear, Mlp, Projection, Relu2Mlp, SaltLinear, SwiGluMlp,
    TernaryLinear, TokenEmbedding, TransformerBlock,
};
#[cfg(feature = "tokenizer")]
pub use model::GgufBpeTokenizer;
pub use model::{ForwardDump, LayerWeights, ModelRunner, ModelWeights, Tokenizer};
#[cfg(feature = "cuda")]
pub use model::{
    SaltV2LoadedTensorReceipt, SaltV2ModelAllocationReceipt, SaltV2PreservedTensorReceipt,
};
pub use model::{
    TRAINING_SALT_COMPLETED_STEP_KEY, TRAINING_SALT_FORMAT_KEY, TRAINING_SALT_FORMAT_VALUE,
    TRAINING_SALT_GROWTH_RECEIPT_KEY, TRAINING_SALT_HF_CONFIG_KEY,
    TRAINING_SALT_INITIAL_STUDENT_DIGEST_KEY, TRAINING_SALT_PLAN_FINGERPRINT_KEY,
    TRAINING_SALT_PLANES_KEY, TRAINING_SALT_SOURCE_MODEL_DIGEST_KEY,
    TRAINING_SALT_UNTIED_FORMAT_VALUE, TrainingSaltArtifactMetadata, TrainingSaltGrowthReceipt,
    parse_training_salt_artifact_metadata,
};
pub use ops::{
    QB, gqa_attention, quantize_activation_int8, rmsnorm, rope_apply, sample_categorical,
    sample_greedy, sample_top_k, sample_top_p, softmax_rows, truncated_top_k, truncated_top_p,
};
pub use qwen35_config::{
    QWEN36_27B_REPOSITORY, QWEN36_27B_REVISION, Qwen35CheckpointConfig, Qwen35DeltaNetConfig,
    Qwen35Dtype, Qwen35FullAttentionConfig, Qwen35LayerType, Qwen35MtpConfig,
    Qwen35NormWeightSemantics, Qwen35OutputGate, Qwen35RopeConfig, Qwen35RopeType,
    Qwen35TextConfig, Qwen35VisionScope,
};
pub use salt_v2_growth::{
    AppliedIntermediateGrowthReceipt, DENSE_GROWTH_ORACLE_ALGORITHM_V1,
    DENSE_GROWTH_ORACLE_TOLERANCE, FixedEmbeddingPolicy, GrowthFunctionPreservationEvidence,
    GrowthPlanError, GrowthReceiptDigest, GrowthResultModelId, GrowthSourceModelId, GrowthTarget,
    GrowthTrackedFp32PayloadEstimate, IntermediateGrowthPlan, MAX_ADDITIVE_PLANES,
    MAX_STAGE1_RECEIPT_WIDTH, PlaneWeightHistogram, ProjectionCoefficientLedger,
    ProjectionGeometry, ProjectionPlaneCounts, Stage2Requirement,
    WHOLE_HEAD_AND_HIDDEN_COEFFICIENT_THRESHOLD,
};
pub use teacher_cache::{
    TeacherCacheError, TeacherCacheReader, TeacherCacheWriter, hash_teacher_corpus,
    hash_teacher_weights,
};
pub use tensor::f16_bytes_to_f32;
#[cfg(feature = "cuda")]
pub use training::{
    PackedTrainingForward, ResidentTrainingForward, packed_device_forward, resident_device_forward,
};
pub use training::{
    SwiGluTrainingArchitecture, SwiGluTrainingModel, TiedSwiGluTrainingArchitecture,
    TiedSwiGluTrainingModel, TrainingAdapterError, TrainingParameter,
    semantic_training_model_digest,
};

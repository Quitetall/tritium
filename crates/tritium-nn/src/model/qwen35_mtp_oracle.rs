//! Compiled-allowlist loading for sealed Qwen3.5-family MTP oracle traces.
//!
//! This module is deliberately crate-private. An artifact is not decoded into
//! owned numeric vectors until its exact body ID and scalar authorization tuple
//! match the compiled ledger for the content-derived source model.

use core::mem::size_of;

use tritium_format::ModelId;

use super::qwen35_hf_source::Qwen35HfSourceIdentity;
use crate::error::NnError;
use crate::qwen35_config::Qwen35CheckpointConfig;

const ORACLE_MAGIC: &[u8; 8] = b"TRQ35MO\0";
const ORACLE_VERSION: u16 = 1;
const NUMERIC_PROFILE_FP32_STORAGE_TF32_ATTN_ABS_2E_NEG3: u16 = 1;
const COVERAGE_PROFILE_FIXTURE_PREFILL_DECODE: u16 = 1;
const COVERAGE_PROFILE_PRODUCTION_CHECKPOINT_PREFILL_DECODE: u16 = 2;
const EVIDENCE_CLASS_FIXTURE: u16 = 1;
const EVIDENCE_CLASS_PRODUCTION: u16 = 2;
const NUMERIC_ABSOLUTE_TOLERANCE: f32 = 2.0e-3;
const MAX_ORACLE_BODY_BYTES: usize = 16 * 1024 * 1024;
const ORACLE_HEADER_BYTES: usize = 136;
const ORACLE_BODY_ID_CONTEXT: &str = "tritium qwen3.5 mtp oracle body v1";

// Exact fixture authorization frozen from the independently reproduced pinned
// vLLM CUDA oracle. This row is fixture evidence only and cannot satisfy a
// production Qwen3.6 campaign gate.
const FIXTURE_SOURCE_MODEL_ID: [u8; 32] = [
    0xe7, 0x9e, 0xea, 0xcd, 0x41, 0x6d, 0x7c, 0xe9, 0xd4, 0xf2, 0x23, 0x95, 0x5b, 0x18, 0x9b, 0xcc,
    0x90, 0xeb, 0xb4, 0x18, 0x82, 0x61, 0x9f, 0x4c, 0x4b, 0x3b, 0x88, 0x04, 0x65, 0x12, 0x6d, 0x92,
];
const FIXTURE_BODY_ID: [u8; 32] = [
    0x49, 0x10, 0xae, 0x29, 0x4f, 0xc2, 0xb2, 0xaf, 0x75, 0xf2, 0xec, 0xb8, 0xbc, 0x58, 0x23, 0x7c,
    0xfe, 0x0a, 0xe1, 0x2a, 0x25, 0x92, 0xb2, 0x08, 0x38, 0x57, 0x77, 0x3c, 0x20, 0x5c, 0x08, 0xc2,
];
const FIXTURE_ORACLE_MANIFEST_ID: [u8; 32] = [
    0x3b, 0x7a, 0x2c, 0x3b, 0xfd, 0x4f, 0x8f, 0xc4, 0x5e, 0xc1, 0x78, 0x72, 0x43, 0x00, 0x48, 0xb8,
    0x4f, 0x33, 0x77, 0x28, 0xb3, 0x14, 0x5e, 0xba, 0xa4, 0x60, 0xf5, 0x64, 0x42, 0x03, 0x15, 0x7d,
];

#[repr(u16)]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35MtpOracleCoverageProfile {
    /// Two-transaction fixture: multi-token prefill followed by one-token decode.
    FixturePrefillDecode = 1,
    /// Pinned checkpoint: multi-token prefill followed by one-token decode.
    ProductionCheckpointPrefillDecode = 2,
}

#[repr(u16)]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35MtpOracleEvidenceClass {
    /// Small deterministic fixture evidence, not checkpoint-scale qualification.
    Fixture = 1,
    /// Exact pinned production checkpoint and independently recorded runtime.
    ProductionCheckpoint = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OracleAuthorization {
    source_model_id: [u8; 32],
    body_id: [u8; 32],
    oracle_manifest_id: [u8; 32],
    numeric_profile: u16,
    coverage_profile: u16,
    evidence_class: u16,
}

// Keep this ledger private: callers may present artifacts, but they cannot add
// authorization rows at runtime or manufacture a promoted trace in safe code.
static ORACLE_AUTHORIZATION_LEDGER: &[OracleAuthorization] = &[OracleAuthorization {
    source_model_id: FIXTURE_SOURCE_MODEL_ID,
    body_id: FIXTURE_BODY_ID,
    oracle_manifest_id: FIXTURE_ORACLE_MANIFEST_ID,
    numeric_profile: NUMERIC_PROFILE_FP32_STORAGE_TF32_ATTN_ABS_2E_NEG3,
    coverage_profile: COVERAGE_PROFILE_FIXTURE_PREFILL_DECODE,
    evidence_class: EVIDENCE_CLASS_FIXTURE,
}];

/// One authorized target/MTP transaction decoded from a sealed oracle body.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct AuthorizedQwen35MtpStep {
    token_ids: Vec<u32>,
    sampled_next: u32,
    expected_target_hidden: Vec<f32>,
    expected_target_logits: Vec<f32>,
    expected_mtp_hidden: Vec<f32>,
    expected_logits: Vec<f32>,
    expected_cache_keys: Vec<f32>,
    expected_cache_values: Vec<f32>,
}

impl AuthorizedQwen35MtpStep {
    /// Target-model token transaction in original order.
    pub(super) fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    /// Independently sampled next token used by the MTP input shift.
    pub(super) const fn sampled_next(&self) -> u32 {
        self.sampled_next
    }

    /// Expected post-final-normalization target hidden rows.
    pub(super) fn expected_target_hidden(&self) -> &[f32] {
        &self.expected_target_hidden
    }

    /// Expected last-position target-model vocabulary logits.
    pub(super) fn expected_target_logits(&self) -> &[f32] {
        &self.expected_target_logits
    }

    /// Expected final-normalized MTP hidden rows.
    pub(super) fn expected_mtp_hidden(&self) -> &[f32] {
        &self.expected_mtp_hidden
    }

    /// Expected last-position MTP vocabulary logits.
    pub(super) fn expected_logits(&self) -> &[f32] {
        &self.expected_logits
    }

    /// Expected complete flattened MTP key cache after this transaction.
    pub(super) fn expected_cache_keys(&self) -> &[f32] {
        &self.expected_cache_keys
    }

    /// Expected complete flattened MTP value cache after this transaction.
    pub(super) fn expected_cache_values(&self) -> &[f32] {
        &self.expected_cache_values
    }
}

/// Opaque trace capability produced only after exact compiled authorization.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct AuthorizedQwen35MtpTrace {
    body_id: [u8; 32],
    oracle_manifest_id: [u8; 32],
    source_model_id: ModelId,
    tolerance: f32,
    coverage_profile: Qwen35MtpOracleCoverageProfile,
    evidence_class: Qwen35MtpOracleEvidenceClass,
    max_context: usize,
    hidden_size: usize,
    vocab_size: usize,
    kv_width: usize,
    steps: Vec<AuthorizedQwen35MtpStep>,
}

impl AuthorizedQwen35MtpTrace {
    /// Domain-separated digest of the exact authorized body bytes.
    pub(super) const fn body_id(&self) -> [u8; 32] {
        self.body_id
    }

    /// Manifest identity assigned by the independent oracle producer.
    pub(super) const fn oracle_manifest_id(&self) -> [u8; 32] {
        self.oracle_manifest_id
    }

    /// Content-derived source model exercised by every transaction.
    pub(super) const fn source_model_id(&self) -> ModelId {
        self.source_model_id
    }

    /// Fixed absolute comparison tolerance selected by numeric profile v1.
    pub(super) const fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Coverage claim carried by this authorized artifact.
    pub(super) const fn coverage_profile(&self) -> Qwen35MtpOracleCoverageProfile {
        self.coverage_profile
    }

    /// Evidence strength carried by this authorized artifact.
    pub(super) const fn evidence_class(&self) -> Qwen35MtpOracleEvidenceClass {
        self.evidence_class
    }

    /// Trace-local cache capacity.
    pub(super) const fn max_context(&self) -> usize {
        self.max_context
    }

    /// Residual width bound into the artifact header.
    pub(super) const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Vocabulary width bound into the artifact header.
    pub(super) const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Flattened key/value width per cached position.
    pub(super) const fn kv_width(&self) -> usize {
        self.kv_width
    }

    /// Ordered prefill/decode transactions.
    pub(super) fn steps(&self) -> &[AuthorizedQwen35MtpStep] {
        &self.steps
    }
}

/// Authenticate and strictly decode a Qwen3.5-family MTP oracle body.
///
/// Authorization uses the actual content-derived source identity, not the ID
/// asserted by the artifact. The body and scalar header must match one private
/// compiled ledger row before any token or floating-point vector is parsed.
pub(super) fn load_authorized_qwen35_mtp_oracle(
    body: &[u8],
    config: &Qwen35CheckpointConfig,
    identity: &Qwen35HfSourceIdentity,
) -> Result<AuthorizedQwen35MtpTrace, NnError> {
    load_authorized_qwen35_mtp_oracle_parts(
        body,
        config,
        identity.model_id(),
        *identity.manifest().config_digest(),
    )
}

fn load_authorized_qwen35_mtp_oracle_parts(
    body: &[u8],
    config: &Qwen35CheckpointConfig,
    actual_source_model_id: ModelId,
    actual_config_digest: [u8; 32],
) -> Result<AuthorizedQwen35MtpTrace, NnError> {
    enforce_body_bound(body)?;
    let body_id = oracle_body_id(body);
    let mut cursor = OracleCursor::new(body);
    let header = OracleHeader::parse(&mut cursor)?;

    authorize_header(&header, actual_source_model_id, body_id)?;
    validate_source_binding(
        &header,
        config,
        actual_source_model_id,
        actual_config_digest,
    )?;
    decode_authorized_vectors(&mut cursor, header, body_id, actual_source_model_id)
}

fn enforce_body_bound(body: &[u8]) -> Result<(), NnError> {
    if body.len() > MAX_ORACLE_BODY_BYTES {
        return Err(invalid_oracle(format!(
            "body is {} bytes, limit is {MAX_ORACLE_BODY_BYTES}",
            body.len()
        )));
    }
    if body.len() < ORACLE_HEADER_BYTES {
        return Err(invalid_oracle("body is truncated before the fixed header"));
    }
    Ok(())
}

fn oracle_body_id(body: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(ORACLE_BODY_ID_CONTEXT);
    hasher.update(body);
    *hasher.finalize().as_bytes()
}

fn authorize_header(
    header: &OracleHeader,
    actual_source_model_id: ModelId,
    body_id: [u8; 32],
) -> Result<(), NnError> {
    let authorized = ORACLE_AUTHORIZATION_LEDGER.iter().any(|row| {
        row.source_model_id == *actual_source_model_id.as_bytes()
            && row.body_id == body_id
            && row.oracle_manifest_id == header.oracle_manifest_id
            && row.numeric_profile == header.numeric_profile
            && row.coverage_profile == header.coverage_profile
            && row.evidence_class == header.evidence_class
    });
    if !authorized {
        return Err(invalid_oracle(
            "body/source/manifest/profile tuple is absent from the compiled allowlist",
        ));
    }
    Ok(())
}

fn validate_source_binding(
    header: &OracleHeader,
    config: &Qwen35CheckpointConfig,
    actual_source_model_id: ModelId,
    actual_config_digest: [u8; 32],
) -> Result<(), NnError> {
    if header.source_model_id != *actual_source_model_id.as_bytes() {
        return Err(invalid_oracle(
            "asserted source model ID does not match the loaded source identity",
        ));
    }
    if header.source_config_digest != actual_config_digest {
        return Err(invalid_oracle(
            "source config digest does not match the loaded semantic manifest",
        ));
    }
    if config.text.mtp.num_hidden_layers != 1 || config.text.mtp.dedicated_embeddings {
        return Err(invalid_oracle(
            "loaded configuration does not have the shared-embedding one-layer MTP geometry",
        ));
    }
    if !config.text.use_cache {
        return Err(invalid_oracle(
            "loaded configuration does not enable autoregressive cache semantics",
        ));
    }

    let expected_kv_width = config
        .text
        .full_attention
        .num_key_value_heads
        .checked_mul(config.text.full_attention.head_dim)
        .ok_or_else(|| invalid_oracle("loaded key/value geometry overflows u32"))?;
    if header.hidden_size != config.text.hidden_size
        || header.vocab_size != config.text.vocab_size
        || header.kv_width != expected_kv_width
    {
        return Err(invalid_oracle(
            "artifact hidden/vocabulary/key-value geometry does not match the loaded config",
        ));
    }
    if header.max_context == 0 || header.max_context > config.text.max_position_embeddings {
        return Err(invalid_oracle(
            "artifact context is zero or exceeds the loaded position limit",
        ));
    }
    Ok(())
}

fn decode_authorized_vectors(
    cursor: &mut OracleCursor<'_>,
    header: OracleHeader,
    body_id: [u8; 32],
    source_model_id: ModelId,
) -> Result<AuthorizedQwen35MtpTrace, NnError> {
    let (coverage_profile, evidence_class) = match (header.coverage_profile, header.evidence_class)
    {
        (COVERAGE_PROFILE_FIXTURE_PREFILL_DECODE, EVIDENCE_CLASS_FIXTURE) => (
            Qwen35MtpOracleCoverageProfile::FixturePrefillDecode,
            Qwen35MtpOracleEvidenceClass::Fixture,
        ),
        (COVERAGE_PROFILE_PRODUCTION_CHECKPOINT_PREFILL_DECODE, EVIDENCE_CLASS_PRODUCTION) => (
            Qwen35MtpOracleCoverageProfile::ProductionCheckpointPrefillDecode,
            Qwen35MtpOracleEvidenceClass::ProductionCheckpoint,
        ),
        _ => return Err(invalid_oracle("unsupported oracle coverage/evidence tuple")),
    };
    if header.step_count != 2 {
        return Err(invalid_oracle(
            "prefill/decode coverage requires exactly two transactions",
        ));
    }

    let max_context = usize_from_u32(header.max_context, "max_context")?;
    let hidden_size = usize_from_u32(header.hidden_size, "hidden_size")?;
    let vocab_size = usize_from_u32(header.vocab_size, "vocab_size")?;
    let kv_width = usize_from_u32(header.kv_width, "kv_width")?;
    let step_count = usize_from_u32(header.step_count, "step_count")?;
    let mut steps = try_vec_with_capacity(step_count, "oracle steps")?;
    let mut cumulative_tokens = 0usize;
    let mut previous_sampled_next = None;

    for step_index in 0..step_count {
        let token_count = usize_from_u32(cursor.u32("step token count")?, "token_count")?;
        if (step_index == 0 && token_count < 2) || (step_index == 1 && token_count != 1) {
            return Err(invalid_oracle(
                "prefill/decode coverage requires a multi-token prefill then one-token decode",
            ));
        }
        cumulative_tokens = cumulative_tokens
            .checked_add(token_count)
            .ok_or_else(|| invalid_oracle("cumulative token count overflows usize"))?;
        if cumulative_tokens > max_context {
            return Err(invalid_oracle(
                "fixture transactions exceed the artifact context limit",
            ));
        }

        let token_ids = cursor.u32_vector(token_count, "step token IDs")?;
        if token_ids.iter().any(|&token| token as usize >= vocab_size) {
            return Err(invalid_oracle("step token ID exceeds the vocabulary"));
        }
        if let Some(expected_continuation) = previous_sampled_next
            && token_ids.as_slice() != [expected_continuation]
        {
            return Err(invalid_oracle(
                "cached-decode token does not continue the prior sampled token",
            ));
        }
        let sampled_next = cursor.u32("sampled next token")?;
        if sampled_next as usize >= vocab_size {
            return Err(invalid_oracle("sampled next token exceeds the vocabulary"));
        }
        previous_sampled_next = Some(sampled_next);

        let hidden_values = token_count
            .checked_mul(hidden_size)
            .ok_or_else(|| invalid_oracle("hidden-state vector length overflows usize"))?;
        let cache_values = cumulative_tokens
            .checked_mul(kv_width)
            .ok_or_else(|| invalid_oracle("cache vector length overflows usize"))?;
        let expected_target_hidden = cursor.f32_vector(hidden_values, "target hidden states")?;
        let expected_target_logits = cursor.f32_vector(vocab_size, "target logits")?;
        if first_argmax(&expected_target_logits) != Some(sampled_next as usize) {
            return Err(invalid_oracle(
                "sampled next token is not the greedy target-logit selection",
            ));
        }
        let expected_mtp_hidden = cursor.f32_vector(hidden_values, "MTP hidden states")?;
        let expected_logits = cursor.f32_vector(vocab_size, "MTP logits")?;
        let expected_cache_keys = cursor.f32_vector(cache_values, "MTP cache keys")?;
        let expected_cache_values = cursor.f32_vector(cache_values, "MTP cache values")?;
        steps.push(AuthorizedQwen35MtpStep {
            token_ids,
            sampled_next,
            expected_target_hidden,
            expected_target_logits,
            expected_mtp_hidden,
            expected_logits,
            expected_cache_keys,
            expected_cache_values,
        });
    }

    if cursor.remaining() != 0 {
        return Err(invalid_oracle("body has trailing bytes"));
    }

    Ok(AuthorizedQwen35MtpTrace {
        body_id,
        oracle_manifest_id: header.oracle_manifest_id,
        source_model_id,
        tolerance: NUMERIC_ABSOLUTE_TOLERANCE,
        coverage_profile,
        evidence_class,
        max_context,
        hidden_size,
        vocab_size,
        kv_width,
        steps,
    })
}

pub(super) fn first_argmax(values: &[f32]) -> Option<usize> {
    let (&first, tail) = values.split_first()?;
    if !first.is_finite() {
        return None;
    }
    let mut best_index = 0usize;
    let mut best = first;
    for (offset, &value) in tail.iter().enumerate() {
        if !value.is_finite() {
            return None;
        }
        if value > best {
            best = value;
            best_index = offset + 1;
        }
    }
    Some(best_index)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OracleHeader {
    numeric_profile: u16,
    coverage_profile: u16,
    evidence_class: u16,
    oracle_manifest_id: [u8; 32],
    source_model_id: [u8; 32],
    source_config_digest: [u8; 32],
    max_context: u32,
    hidden_size: u32,
    vocab_size: u32,
    kv_width: u32,
    step_count: u32,
}

impl OracleHeader {
    fn parse(cursor: &mut OracleCursor<'_>) -> Result<Self, NnError> {
        if cursor.digest8("magic")? != *ORACLE_MAGIC {
            return Err(invalid_oracle("bad magic"));
        }
        if cursor.u16("version")? != ORACLE_VERSION {
            return Err(invalid_oracle("unsupported body version"));
        }
        if cursor.u16("reserved")? != 0 {
            return Err(invalid_oracle("reserved header field is non-zero"));
        }
        let numeric_profile = cursor.u16("numeric profile")?;
        let coverage_profile = cursor.u16("coverage profile")?;
        let evidence_class = cursor.u16("evidence class")?;
        if cursor.u16("reserved2")? != 0 {
            return Err(invalid_oracle("reserved2 header field is non-zero"));
        }
        if numeric_profile != NUMERIC_PROFILE_FP32_STORAGE_TF32_ATTN_ABS_2E_NEG3
            || !matches!(
                (coverage_profile, evidence_class),
                (
                    COVERAGE_PROFILE_FIXTURE_PREFILL_DECODE,
                    EVIDENCE_CLASS_FIXTURE
                ) | (
                    COVERAGE_PROFILE_PRODUCTION_CHECKPOINT_PREFILL_DECODE,
                    EVIDENCE_CLASS_PRODUCTION
                )
            )
        {
            return Err(invalid_oracle("unsupported oracle profile tuple"));
        }
        Ok(Self {
            numeric_profile,
            coverage_profile,
            evidence_class,
            oracle_manifest_id: cursor.digest32("oracle manifest ID")?,
            source_model_id: cursor.digest32("source model ID")?,
            source_config_digest: cursor.digest32("source config digest")?,
            max_context: cursor.u32("max context")?,
            hidden_size: cursor.u32("hidden size")?,
            vocab_size: cursor.u32("vocabulary size")?,
            kv_width: cursor.u32("key/value width")?,
            step_count: cursor.u32("step count")?,
        })
    }
}

#[derive(Debug)]
struct OracleCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> OracleCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, count: usize, field: &str) -> Result<&'a [u8], NnError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| invalid_oracle(format!("{field} byte range overflows usize")))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| invalid_oracle(format!("body is truncated in {field}")))?;
        self.offset = end;
        Ok(value)
    }

    fn u16(&mut self, field: &str) -> Result<u16, NnError> {
        let bytes: [u8; 2] = self
            .take(2, field)?
            .try_into()
            .map_err(|_| invalid_oracle(format!("body is truncated in {field}")))?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self, field: &str) -> Result<u32, NnError> {
        let bytes: [u8; 4] = self
            .take(4, field)?
            .try_into()
            .map_err(|_| invalid_oracle(format!("body is truncated in {field}")))?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn digest8(&mut self, field: &str) -> Result<[u8; 8], NnError> {
        self.take(8, field)?
            .try_into()
            .map_err(|_| invalid_oracle(format!("body is truncated in {field}")))
    }

    fn digest32(&mut self, field: &str) -> Result<[u8; 32], NnError> {
        self.take(32, field)?
            .try_into()
            .map_err(|_| invalid_oracle(format!("body is truncated in {field}")))
    }

    fn u32_vector(&mut self, count: usize, field: &str) -> Result<Vec<u32>, NnError> {
        let byte_count = count
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| invalid_oracle(format!("{field} byte count overflows usize")))?;
        let bytes = self.take(byte_count, field)?;
        let mut output = try_vec_with_capacity(count, field)?;
        for chunk in bytes.chunks_exact(size_of::<u32>()) {
            let lane: [u8; 4] = chunk
                .try_into()
                .map_err(|_| invalid_oracle(format!("body is truncated in {field}")))?;
            output.push(u32::from_le_bytes(lane));
        }
        Ok(output)
    }

    fn f32_vector(&mut self, count: usize, field: &str) -> Result<Vec<f32>, NnError> {
        let byte_count = count
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| invalid_oracle(format!("{field} byte count overflows usize")))?;
        let bytes = self.take(byte_count, field)?;
        let mut output = try_vec_with_capacity(count, field)?;
        for chunk in bytes.chunks_exact(size_of::<f32>()) {
            let lane: [u8; 4] = chunk
                .try_into()
                .map_err(|_| invalid_oracle(format!("body is truncated in {field}")))?;
            let value = f32::from_le_bytes(lane);
            if !value.is_finite() {
                return Err(invalid_oracle(format!(
                    "{field} contains a non-finite value"
                )));
            }
            output.push(value);
        }
        Ok(output)
    }
}

fn usize_from_u32(value: u32, field: &str) -> Result<usize, NnError> {
    usize::try_from(value)
        .map_err(|_| invalid_oracle(format!("{field} exceeds the host address width")))
}

fn try_vec_with_capacity<T>(count: usize, field: &str) -> Result<Vec<T>, NnError> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|error| {
        NnError::ResourceExhausted(format!(
            "allocate {count} values for Qwen3.5 MTP oracle {field}: {error}"
        ))
    })?;
    Ok(values)
}

fn invalid_oracle(message: impl Into<String>) -> NnError {
    NnError::InvalidArtifact(format!("Qwen3.5 MTP oracle artifact: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const TEST_CONFIG_DIGEST: [u8; 32] = [0x44; 32];
    const STEP_ONE_TARGET_OFFSET: usize = ORACLE_HEADER_BYTES + 4 + (2 * 4) + 4;
    const STEP_ONE_TARGET_LOGITS_OFFSET: usize = STEP_ONE_TARGET_OFFSET + (8 * 4);

    fn test_config() -> Qwen35CheckpointConfig {
        let value = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "language_model_only": false,
            "model_type": "qwen3_5",
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attn_output_gate": true,
                "dtype": "bfloat16",
                "full_attention_interval": 2,
                "head_dim": 2,
                "hidden_act": "silu",
                "hidden_size": 4,
                "intermediate_size": 6,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 2,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 2,
                "linear_value_head_dim": 2,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 32,
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "num_key_value_heads": 1,
                "output_gate_type": "swish",
                "partial_rotary_factor": 1.0,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [1, 0, 0],
                    "partial_rotary_factor": 1.0,
                    "rope_theta": 1000000,
                    "rope_type": "default"
                },
                "tie_word_embeddings": false,
                "use_cache": true,
                "vocab_size": 7
            },
            "tie_word_embeddings": false,
            "vision_config": {"model_type": "qwen3_5"}
        });
        Qwen35CheckpointConfig::from_hf_config(&value.to_string()).expect("valid tiny config")
    }

    fn push_u16(body: &mut Vec<u8>, value: u16) {
        body.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(body: &mut Vec<u8>, value: u32) {
        body.extend_from_slice(&value.to_le_bytes());
    }

    fn push_f32s(body: &mut Vec<u8>, count: usize, offset: f32) {
        for index in 0..count {
            body.extend_from_slice(&(offset + index as f32 / 64.0).to_le_bytes());
        }
    }

    fn push_logits(body: &mut Vec<u8>, count: usize, winner: usize, offset: f32) {
        for index in 0..count {
            let value = if index == winner {
                offset + 1.0
            } else {
                offset
            };
            body.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn body_with_decode_token(decode_token: u32) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(ORACLE_MAGIC);
        push_u16(&mut body, ORACLE_VERSION);
        push_u16(&mut body, 0);
        push_u16(
            &mut body,
            NUMERIC_PROFILE_FP32_STORAGE_TF32_ATTN_ABS_2E_NEG3,
        );
        push_u16(&mut body, COVERAGE_PROFILE_FIXTURE_PREFILL_DECODE);
        push_u16(&mut body, EVIDENCE_CLASS_FIXTURE);
        push_u16(&mut body, 0);
        body.extend_from_slice(&FIXTURE_ORACLE_MANIFEST_ID);
        body.extend_from_slice(&FIXTURE_SOURCE_MODEL_ID);
        body.extend_from_slice(&TEST_CONFIG_DIGEST);
        push_u32(&mut body, 8);
        push_u32(&mut body, 4);
        push_u32(&mut body, 7);
        push_u32(&mut body, 2);
        push_u32(&mut body, 2);
        assert_eq!(body.len(), ORACLE_HEADER_BYTES);

        push_u32(&mut body, 2);
        push_u32(&mut body, 1);
        push_u32(&mut body, 2);
        push_u32(&mut body, 3);
        push_f32s(&mut body, 8, 0.0);
        push_logits(&mut body, 7, 3, 0.0);
        push_f32s(&mut body, 8, 1.0);
        push_f32s(&mut body, 7, 2.0);
        push_f32s(&mut body, 4, 3.0);
        push_f32s(&mut body, 4, 4.0);

        push_u32(&mut body, 1);
        push_u32(&mut body, decode_token);
        push_u32(&mut body, 4);
        push_f32s(&mut body, 4, 5.0);
        push_logits(&mut body, 7, 4, 5.0);
        push_f32s(&mut body, 4, 6.0);
        push_f32s(&mut body, 7, 7.0);
        push_f32s(&mut body, 6, 8.0);
        push_f32s(&mut body, 6, 9.0);
        body
    }

    fn valid_body() -> Vec<u8> {
        body_with_decode_token(3)
    }

    fn production_profile_body() -> Vec<u8> {
        let mut body = valid_body();
        body[14..16]
            .copy_from_slice(&COVERAGE_PROFILE_PRODUCTION_CHECKPOINT_PREFILL_DECODE.to_le_bytes());
        body[16..18].copy_from_slice(&EVIDENCE_CLASS_PRODUCTION.to_le_bytes());
        body
    }

    fn decode_for_parser_test(body: &[u8]) -> Result<AuthorizedQwen35MtpTrace, NnError> {
        enforce_body_bound(body)?;
        let mut cursor = OracleCursor::new(body);
        let header = OracleHeader::parse(&mut cursor)?;
        let source_model_id = ModelId::from_digest(FIXTURE_SOURCE_MODEL_ID);
        validate_source_binding(&header, &test_config(), source_model_id, TEST_CONFIG_DIGEST)?;
        decode_authorized_vectors(&mut cursor, header, oracle_body_id(body), source_model_id)
    }

    #[test]
    fn strict_parser_decodes_fixture_contract() {
        let trace = decode_for_parser_test(&valid_body()).expect("decode valid body");
        assert_eq!(trace.source_model_id().as_bytes(), &FIXTURE_SOURCE_MODEL_ID);
        assert_eq!(trace.oracle_manifest_id(), FIXTURE_ORACLE_MANIFEST_ID);
        assert_eq!(trace.tolerance(), NUMERIC_ABSOLUTE_TOLERANCE);
        assert_eq!(
            trace.coverage_profile(),
            Qwen35MtpOracleCoverageProfile::FixturePrefillDecode
        );
        assert_eq!(
            trace.evidence_class(),
            Qwen35MtpOracleEvidenceClass::Fixture
        );
        assert_eq!(trace.max_context(), 8);
        assert_eq!(trace.hidden_size(), 4);
        assert_eq!(trace.vocab_size(), 7);
        assert_eq!(trace.kv_width(), 2);
        assert_eq!(trace.steps().len(), 2);
        assert_eq!(trace.steps()[0].token_ids(), [1, 2]);
        assert_eq!(trace.steps()[0].sampled_next(), 3);
        assert_eq!(trace.steps()[0].expected_target_hidden().len(), 8);
        assert_eq!(trace.steps()[0].expected_target_logits().len(), 7);
        assert_eq!(trace.steps()[0].expected_mtp_hidden().len(), 8);
        assert_eq!(trace.steps()[0].expected_logits().len(), 7);
        assert_eq!(trace.steps()[1].expected_cache_keys().len(), 6);
        assert_eq!(trace.steps()[1].expected_cache_values().len(), 6);
        assert_ne!(trace.body_id(), [0; 32]);
    }

    #[test]
    fn strict_parser_understands_production_profile_without_authorizing_it() {
        let body = production_profile_body();
        let trace = decode_for_parser_test(&body).expect("decode production wire profile");
        assert_eq!(
            trace.coverage_profile(),
            Qwen35MtpOracleCoverageProfile::ProductionCheckpointPrefillDecode
        );
        assert_eq!(
            trace.evidence_class(),
            Qwen35MtpOracleEvidenceClass::ProductionCheckpoint
        );
        let error = load_authorized_qwen35_mtp_oracle_parts(
            &body,
            &test_config(),
            ModelId::from_digest(FIXTURE_SOURCE_MODEL_ID),
            TEST_CONFIG_DIGEST,
        )
        .expect_err("production profile without compiled row must not authorize");
        assert!(error.to_string().contains("compiled allowlist"));
    }

    #[test]
    fn compiled_allowlist_rows_are_unique() {
        for (index, row) in ORACLE_AUTHORIZATION_LEDGER.iter().enumerate() {
            assert!(
                ORACLE_AUTHORIZATION_LEDGER[index + 1..]
                    .iter()
                    .all(|candidate| candidate != row),
                "duplicate oracle authorization row at index {index}"
            );
        }
    }

    #[test]
    fn compiled_allowlist_has_no_production_evidence_row() {
        assert!(
            ORACLE_AUTHORIZATION_LEDGER
                .iter()
                .all(|row| row.evidence_class != EVIDENCE_CLASS_PRODUCTION)
        );
    }

    #[test]
    fn authorization_rejects_unlisted_body_before_hostile_vector_lengths() {
        let mut body = valid_body();
        body.truncate(ORACLE_HEADER_BYTES);
        push_u32(&mut body, u32::MAX);
        let error = load_authorized_qwen35_mtp_oracle_parts(
            &body,
            &test_config(),
            ModelId::from_digest(FIXTURE_SOURCE_MODEL_ID),
            TEST_CONFIG_DIGEST,
        )
        .expect_err("placeholder body must not authorize");
        assert!(error.to_string().contains("compiled allowlist"));
    }

    #[test]
    fn parser_rejects_hostile_declared_token_length_without_allocating() {
        let mut body = valid_body();
        body.truncate(ORACLE_HEADER_BYTES);
        push_u32(&mut body, u32::MAX);
        let error = decode_for_parser_test(&body).expect_err("reject hostile token count");
        assert!(error.to_string().contains("context limit"));
    }

    #[test]
    fn parser_rejects_trailing_bytes() {
        let mut body = valid_body();
        body.push(0);
        let error = decode_for_parser_test(&body).expect_err("reject trailing byte");
        assert!(error.to_string().contains("trailing bytes"));
    }

    #[test]
    fn parser_rejects_noncontiguous_cached_decode() {
        let body = body_with_decode_token(4);
        let error = decode_for_parser_test(&body).expect_err("reject unrelated decode token");
        assert!(error.to_string().contains("prior sampled token"));
    }

    #[test]
    fn parser_rejects_nonfinite_numeric_lane() {
        let mut body = valid_body();
        body[STEP_ONE_TARGET_OFFSET..STEP_ONE_TARGET_OFFSET + 4]
            .copy_from_slice(&f32::NAN.to_le_bytes());
        let error = decode_for_parser_test(&body).expect_err("reject NaN");
        assert!(error.to_string().contains("non-finite"));
    }

    #[test]
    fn parser_rejects_sampled_token_that_is_not_target_argmax() {
        let mut body = valid_body();
        let lane_five = STEP_ONE_TARGET_LOGITS_OFFSET + (5 * 4);
        body[lane_five..lane_five + 4].copy_from_slice(&2.0_f32.to_le_bytes());
        let error = decode_for_parser_test(&body).expect_err("reject divergent greedy token");
        assert!(error.to_string().contains("greedy target-logit selection"));
    }

    #[test]
    fn first_argmax_matches_torch_first_maximum_tie_break() {
        assert_eq!(first_argmax(&[1.0, 3.0, 3.0, 2.0]), Some(1));
        assert_eq!(first_argmax(&[]), None);
        assert_eq!(first_argmax(&[0.0, f32::NAN]), None);
    }

    #[test]
    fn parser_rejects_source_config_digest_mismatch() {
        let body = valid_body();
        let mut cursor = OracleCursor::new(&body);
        let header = OracleHeader::parse(&mut cursor).expect("parse fixed header");
        let error = validate_source_binding(
            &header,
            &test_config(),
            ModelId::from_digest(FIXTURE_SOURCE_MODEL_ID),
            [0x55; 32],
        )
        .expect_err("reject mismatched source config digest");
        assert!(error.to_string().contains("config digest"));
    }

    #[test]
    fn parser_rejects_loaded_geometry_mismatch() {
        let body = valid_body();
        let mut cursor = OracleCursor::new(&body);
        let header = OracleHeader::parse(&mut cursor).expect("parse fixed header");
        let mut config = test_config();
        config.text.hidden_size = 8;
        let error = validate_source_binding(
            &header,
            &config,
            ModelId::from_digest(FIXTURE_SOURCE_MODEL_ID),
            TEST_CONFIG_DIGEST,
        )
        .expect_err("reject mismatched loaded geometry");
        assert!(error.to_string().contains("geometry"));
    }

    #[test]
    fn parser_rejects_artifacts_over_sixteen_mib() {
        let body = vec![0; MAX_ORACLE_BODY_BYTES + 1];
        let error = decode_for_parser_test(&body).expect_err("reject oversized body");
        assert!(error.to_string().contains("limit"));
    }
}

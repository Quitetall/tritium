//! Strict production-artifact admission and bounded startup self-test receipts.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_nn::Qwen35SaltV2LoadReceipt;

use crate::generator::{GenRequest, Generator, Sampling, Step};
use crate::qwen_generator::QwenGenerator;

const IDENTITY_HEX_CHARS: usize = 64;
const REVISION_HEX_CHARS: usize = 40;
const MAX_LABEL_BYTES: usize = 256;

/// Authenticated schema-v3 artifact state before execution self-test.
#[derive(Clone, Debug)]
pub struct AdmittedArtifactV1 {
    server_source_revision: String,
    server_build_id: String,
    model_source_revision: String,
    manifest_package_id: String,
    salt_package_id: String,
    preserved_package_id: String,
    config_package_id: String,
    profile: String,
    codec: String,
    backend_policy: String,
    effective_backend: String,
    physical_device_id: String,
    loaded_bundle_bytes: u64,
    resident_bytes: u64,
}

/// Opaque capability binding one executable generator to its admitted artifact.
///
/// Its fields are private: only a strict loader adapter inside Tritium can
/// create a value that a production router accepts.
pub struct AdmittedGeneratorV1 {
    generator: Box<dyn Generator>,
    artifact: AdmittedArtifactV1,
}

impl fmt::Debug for AdmittedGeneratorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AdmittedGeneratorV1")
            .field("artifact", &self.artifact)
            .finish_non_exhaustive()
    }
}

impl AdmittedArtifactV1 {
    /// Bind startup state to one strict schema-v3 Qwen load receipt.
    pub fn from_qwen36_salt_v3(
        receipt: &Qwen35SaltV2LoadReceipt,
        server_source_revision: &str,
        server_build_id: &str,
        backend_policy: &str,
        effective_backend: &str,
        physical_device_id: &str,
    ) -> Result<Self, StartupError> {
        if backend_policy != effective_backend {
            return Err(StartupError::InvalidAdmission(
                "effective backend differs from fail-closed backend policy",
            ));
        }
        let codec = match receipt.codec() {
            SaltV2Codec::D2 => "d2",
            SaltV2Codec::B3 => "b3",
            SaltV2Codec::S34 => "s34",
            _ => return Err(StartupError::InvalidAdmission("unsupported SALT V2 codec")),
        };
        for (label, value) in [
            ("manifest package", receipt.manifest_package_id()),
            ("SALT package", receipt.package_id()),
            ("preserved package", receipt.preserved_package_id()),
            ("config package", receipt.config_package_id()),
        ] {
            validate_hex(value, IDENTITY_HEX_CHARS, label)?;
        }
        validate_hex(
            server_source_revision,
            REVISION_HEX_CHARS,
            "server source revision",
        )?;
        validate_hex(
            receipt.source_revision(),
            REVISION_HEX_CHARS,
            "model source revision",
        )?;
        for (label, value) in [
            ("server build id", server_build_id),
            ("profile", receipt.profile()),
            ("backend policy", backend_policy),
            ("effective backend", effective_backend),
            ("physical device id", physical_device_id),
        ] {
            validate_label(value, label)?;
        }
        if receipt.loaded_bundle_bytes() == 0 || receipt.resident_bytes() == 0 {
            return Err(StartupError::InvalidAdmission(
                "artifact byte ledgers must be nonzero",
            ));
        }
        Ok(Self {
            server_source_revision: server_source_revision.to_owned(),
            server_build_id: server_build_id.to_owned(),
            model_source_revision: receipt.source_revision().to_owned(),
            manifest_package_id: receipt.manifest_package_id().to_owned(),
            salt_package_id: receipt.package_id().to_owned(),
            preserved_package_id: receipt.preserved_package_id().to_owned(),
            config_package_id: receipt.config_package_id().to_owned(),
            profile: receipt.profile().to_owned(),
            codec: codec.to_owned(),
            backend_policy: backend_policy.to_owned(),
            effective_backend: effective_backend.to_owned(),
            physical_device_id: physical_device_id.to_owned(),
            loaded_bundle_bytes: receipt.loaded_bundle_bytes(),
            resident_bytes: receipt.resident_bytes(),
        })
    }
}

/// Bind one strict Qwen bundle and its exact load receipt into an opaque
/// production-serving capability.
pub fn admit_qwen36_salt_v3(
    model: tritium_nn::Qwen35SaltV2LanguageMtpModel,
    eos: u32,
    server_source_revision: &str,
    server_build_id: &str,
    backend_policy: &str,
    effective_backend: &str,
    physical_device_id: &str,
) -> Result<AdmittedGeneratorV1, StartupError> {
    let artifact = AdmittedArtifactV1::from_qwen36_salt_v3(
        model.receipt(),
        server_source_revision,
        server_build_id,
        backend_policy,
        effective_backend,
        physical_device_id,
    )?;
    Ok(AdmittedGeneratorV1 {
        generator: Box::new(QwenGenerator::new(model, eos)),
        artifact,
    })
}

/// Immutable evidence returned by a successful production startup.
#[derive(Clone, Debug, Serialize)]
pub struct StartupReceiptV1 {
    /// Receipt schema version.
    pub schema_version: u32,
    /// Authenticated artifact kind.
    pub artifact_kind: &'static str,
    /// Tritium server source revision.
    pub server_source_revision: String,
    /// Tritium build identity.
    pub server_build_id: String,
    /// Pinned upstream model source revision.
    pub model_source_revision: String,
    /// Exact schema-v3 manifest bytes.
    pub manifest_package_id: String,
    /// Exact selected SALT profile package.
    pub salt_package_id: String,
    /// Exact preserved-tensor package.
    pub preserved_package_id: String,
    /// Exact execution configuration package.
    pub config_package_id: String,
    /// Selected refinement/profile discriminant.
    pub profile: String,
    /// Physical ternary codec.
    pub codec: String,
    /// Requested backend policy.
    pub backend_policy: String,
    /// Effective backend; must equal policy.
    pub effective_backend: String,
    /// Bounded physical device identity.
    pub physical_device_id: String,
    /// Exact bytes consumed from bundle.
    pub loaded_bundle_bytes: u64,
    /// Tracked steady model bytes.
    pub resident_bytes: u64,
    /// Deterministic one-token startup self-test identity.
    pub self_test_digest: String,
}

/// One-way production readiness handle. Revocation cannot silently re-enable.
#[derive(Clone, Debug)]
pub struct ProductionReadiness {
    serving: Arc<AtomicBool>,
    receipt: Arc<StartupReceiptV1>,
}

impl ProductionReadiness {
    fn new(receipt: StartupReceiptV1) -> Self {
        Self {
            serving: Arc::new(AtomicBool::new(true)),
            receipt: Arc::new(receipt),
        }
    }

    /// Permanently revoke readiness before drain or after device/artifact failure.
    pub fn revoke(&self) {
        self.serving.store(false, Ordering::SeqCst);
    }

    /// Whether startup remains admitted and unrevoked.
    #[must_use]
    pub fn is_serving(&self) -> bool {
        self.serving.load(Ordering::SeqCst)
    }

    /// Immutable startup receipt.
    #[must_use]
    pub fn receipt(&self) -> &StartupReceiptV1 {
        self.receipt.as_ref()
    }
}

/// Startup admission or self-test failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StartupError {
    /// Authenticated load receipt has unusable identity or policy fields.
    InvalidAdmission(&'static str),
    /// Generator geometry cannot execute bounded self-test.
    InvalidGenerator(&'static str),
    /// Backend failed startup inference.
    SelfTest(String),
}

impl fmt::Display for StartupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAdmission(message) | Self::InvalidGenerator(message) => {
                formatter.write_str(message)
            }
            Self::SelfTest(message) => write!(formatter, "startup self-test failed: {message}"),
        }
    }
}

impl std::error::Error for StartupError {}

/// Execute one deterministic token before worker spawn and seal readiness.
pub fn prepare_production_generator(
    admitted: AdmittedGeneratorV1,
) -> Result<(Box<dyn Generator>, ProductionReadiness), StartupError> {
    let AdmittedGeneratorV1 {
        mut generator,
        artifact,
    } = admitted;
    let n_ctx = generator.n_ctx();
    let vocab = generator.vocab();
    if n_ctx < 2 || vocab == 0 {
        return Err(StartupError::InvalidGenerator(
            "generator cannot execute one-token startup self-test",
        ));
    }
    let request = GenRequest {
        prompt_tokens: vec![0],
        max_new: 1,
        sampling: Sampling::Greedy,
        stop_eos: false,
        logprobs: None,
    };
    let mut steps = Vec::with_capacity(1);
    generator
        .generate(&request, &mut |step| {
            steps.push(step);
            steps.len() < 2
        })
        .map_err(|error| StartupError::SelfTest(error.to_string()))?;
    if steps.len() != 1 || steps[0].token as usize >= vocab {
        return Err(StartupError::InvalidGenerator(
            "generator returned invalid startup self-test steps",
        ));
    }
    let self_test_digest = self_test_digest(n_ctx, vocab, &steps[0]);
    let receipt = StartupReceiptV1 {
        schema_version: 1,
        artifact_kind: "tritium-qwen36-salt-v3",
        server_source_revision: artifact.server_source_revision,
        server_build_id: artifact.server_build_id,
        model_source_revision: artifact.model_source_revision,
        manifest_package_id: artifact.manifest_package_id,
        salt_package_id: artifact.salt_package_id,
        preserved_package_id: artifact.preserved_package_id,
        config_package_id: artifact.config_package_id,
        profile: artifact.profile,
        codec: artifact.codec,
        backend_policy: artifact.backend_policy,
        effective_backend: artifact.effective_backend,
        physical_device_id: artifact.physical_device_id,
        loaded_bundle_bytes: artifact.loaded_bundle_bytes,
        resident_bytes: artifact.resident_bytes,
        self_test_digest,
    };
    Ok((generator, ProductionReadiness::new(receipt)))
}

fn self_test_digest(n_ctx: usize, vocab: usize, step: &Step) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"tritium startup self-test v1\0");
    hasher.update(&(n_ctx as u64).to_le_bytes());
    hasher.update(&(vocab as u64).to_le_bytes());
    hasher.update(&step.token.to_le_bytes());
    hasher.update(&[u8::from(step.finished)]);
    hasher.update(&[match step.finish_reason {
        None => 0,
        Some(crate::generator::FinishReason::Stop) => 1,
        Some(crate::generator::FinishReason::Length) => 2,
    }]);
    hasher.finalize().to_hex().to_string()
}

fn validate_hex(value: &str, bytes: usize, label: &'static str) -> Result<(), StartupError> {
    if value.len() != bytes
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(StartupError::InvalidAdmission(label));
    }
    Ok(())
}

fn validate_label(value: &str, label: &'static str) -> Result<(), StartupError> {
    if value.is_empty()
        || value.len() > MAX_LABEL_BYTES
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/' | b'+')
        })
    {
        return Err(StartupError::InvalidAdmission(label));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::{FinishReason, MockGenerator};

    fn artifact() -> AdmittedArtifactV1 {
        AdmittedArtifactV1 {
            server_source_revision: "a".repeat(40),
            server_build_id: "tritium-serve@1.1.0-rc.0".into(),
            model_source_revision: "b".repeat(40),
            manifest_package_id: "c".repeat(64),
            salt_package_id: "d".repeat(64),
            preserved_package_id: "e".repeat(64),
            config_package_id: "f".repeat(64),
            profile: "compact-v1".into(),
            codec: "b3".into(),
            backend_policy: "cpu".into(),
            effective_backend: "cpu".into(),
            physical_device_id: "cpu:x86_64".into(),
            loaded_bundle_bytes: 100,
            resident_bytes: 80,
        }
    }

    #[test]
    fn self_test_is_deterministic_and_readiness_is_one_way() {
        let generator = MockGenerator::new(vec![7]);
        let admitted = AdmittedGeneratorV1 {
            generator: Box::new(generator),
            artifact: artifact(),
        };
        let (_, readiness) = prepare_production_generator(admitted).unwrap();
        assert_eq!(readiness.receipt().self_test_digest.len(), 64);
        assert!(readiness.is_serving());
        readiness.revoke();
        assert!(!readiness.is_serving());
    }

    #[test]
    fn self_test_rejects_empty_or_out_of_vocab_output() {
        let empty = MockGenerator::new(Vec::new());
        assert!(matches!(
            prepare_production_generator(AdmittedGeneratorV1 {
                generator: Box::new(empty),
                artifact: artifact(),
            }),
            Err(StartupError::InvalidGenerator(_))
        ));
        let mut invalid = MockGenerator::new(vec![99]);
        invalid.vocab = 8;
        invalid.end_reason = FinishReason::Length;
        assert!(matches!(
            prepare_production_generator(AdmittedGeneratorV1 {
                generator: Box::new(invalid),
                artifact: artifact(),
            }),
            Err(StartupError::InvalidGenerator(_))
        ));
    }
}

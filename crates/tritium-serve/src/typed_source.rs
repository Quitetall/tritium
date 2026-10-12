//! Explicit retained-file candidate loading; no production startup authority.
use serde::Deserialize;
use std::{fs::File, io::Read, path::Path};

const MAX_LOAD_JSON: u64 = 16 * 1024;
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TypedConfig {
    path: String,
    level: String,
    max_source_bytes: u64,
    max_manifest_bytes: usize,
    max_directory_bytes: usize,
    max_config_bytes: u64,
    max_tensor_bytes: u64,
    max_decoded_tensor_bytes: u64,
    max_payload_bytes: u64,
    native_only: bool,
}
impl TypedConfig {
    pub(crate) fn read(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() > MAX_LOAD_JSON {
            return Err("--typed load JSON must be a regular file of at most 16KiB".into());
        }
        let mut bytes = Vec::new();
        file.take(MAX_LOAD_JSON + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_LOAD_JSON {
            return Err("--typed load JSON grew beyond 16KiB".into());
        }
        let config: Self = serde_json::from_slice(&bytes)?;
        config.validate()?;
        Ok(config)
    }
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.path.len() > 4096 || !Path::new(&self.path).is_absolute() {
            return Err("--typed requires an absolute model path of at most 4096 bytes".into());
        }
        if self.level.is_empty() || self.level.len() > 256 {
            return Err("--typed requires an explicit nonempty level of at most 256 bytes".into());
        }
        if [
            self.max_source_bytes,
            self.max_manifest_bytes as u64,
            self.max_directory_bytes as u64,
            self.max_config_bytes,
            self.max_tensor_bytes,
            self.max_decoded_tensor_bytes,
            self.max_payload_bytes,
        ]
        .contains(&0)
        {
            return Err("--typed source/read/decode/payload limits must all be positive".into());
        }
        Ok(())
    }
    pub(crate) fn load(
        self,
        backend: Box<dyn tritium_spec::TernaryBackend>,
    ) -> Result<tritium_nn::LoadedModel, Box<dyn std::error::Error>> {
        self.validate()?;
        let budget = tritium_runtime::TensorPayloadBudget::new(self.max_payload_bytes);
        let mut recorder = tritium_evidence::EvidenceRecorder::new(
            "serve-typed-load",
            [
                tritium_evidence::TENSOR_UPLOADED_REGISTRATION,
                tritium_evidence::MODEL_LOADED_REGISTRATION,
            ],
        )?;
        let policy = tritium_schema::TensorUploadPolicy {
            native_only: self.native_only,
            max_payload_bytes: Some(self.max_payload_bytes),
        };
        let loaded = tritium_nn::load(
            tritium_nn::ModelSource::TritFile {
                path: Path::new(&self.path),
                level: &self.level,
                receiving: tritium_format::TritPackageReadLimits {
                    max_source_bytes: self.max_source_bytes,
                    max_manifest_bytes: self.max_manifest_bytes,
                    max_directory_bytes: self.max_directory_bytes,
                },
                limits: tritium_nn::Qwen35TritReadLimits {
                    max_config_bytes: self.max_config_bytes,
                    max_tensor_bytes: self.max_tensor_bytes,
                    max_decoded_tensor_bytes: self.max_decoded_tensor_bytes,
                },
            },
            backend,
            policy,
            &budget,
            tritium_runtime::TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        )?;
        eprintln!(
            "WARNING: typed candidate, not production-qualified: {}",
            serde_json::to_string(loaded.receipt.observation())?
        );
        Ok(loaded)
    }
}

//! Shared bounded typed load request; no production/release authority.
use serde::Deserialize;
use std::{fs::File, io::Read, path::Path};
const MAX_LOAD_JSON: u64 = 16 * 1024;
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Execution {
    Host {},
    ResidentDeviceF32 { max_device_payload_bytes: usize },
}
impl Default for Execution {
    fn default() -> Self {
        Self::Host {}
    }
}
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamConfig {
    max_context: usize,
    max_segment_tokens: usize,
    max_host_cache_payload_bytes: u64,
}
impl StreamConfig {
    fn limits(self) -> super::StreamLimits {
        super::StreamLimits {
            max_context: self.max_context,
            max_segment_tokens: self.max_segment_tokens,
            max_host_cache_payload_bytes: self.max_host_cache_payload_bytes,
        }
    }
}
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamedResources {
    max_source_workspace_bytes: u64,
    max_host_workspace_bytes: u64,
    max_device_workspace_bytes: u64,
    max_retained_host_bytes: u64,
}
/// One retained-file load configuration shared by CLI and serving.
/// Bounded read and validation do not initialize a backend or grant qualification.
/// Optional streamed_resources requires encoded matrix upload; omission preserves
/// compatibility. All fields are private and loading revalidates the request.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedLoadConfig {
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
    #[serde(default)]
    execution: Execution,
    stream_limits: Option<StreamConfig>,
    streamed_resources: Option<StreamedResources>,
}
impl TypedLoadConfig {
    /// Read a regular load JSON of at most16KiB, before backend initialization.
    /// # Errors
    /// I/O, malformed/unknown JSON fields, invalid paths/levels/limits/selections.
    pub fn read(path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
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
    /// Actual configured model path; not a discovery hint.
    #[must_use]
    pub fn path(&self) -> &Path {
        Path::new(&self.path)
    }
    /// Explicit stream capacity, if selected; not physical/model qualification.
    #[must_use]
    pub fn stream_limits(&self) -> Option<super::StreamLimits> {
        self.stream_limits.map(StreamConfig::limits)
    }
    /// Require encoded/resource-admitted loading before creating a backend.
    /// # Errors
    /// Invalid configuration or absent streamed_resources/stream_limits selection.
    pub fn require_encoded_streaming(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.validate()?;
        if self.streamed_resources.is_none() {
            return Err("--typed generation requires explicit streamed_resources and stream_limits; no decoded fallback".into());
        }
        Ok(())
    }
    fn validate(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
        if let Some(s) = self.stream_limits
            && (s.max_context == 0
                || s.max_segment_tokens == 0
                || s.max_host_cache_payload_bytes == 0
                || s.max_segment_tokens > s.max_context)
        {
            return Err(
                "--typed stream limits must be positive and segment must fit context".into(),
            );
        }
        if let Execution::ResidentDeviceF32 {
            max_device_payload_bytes,
        } = self.execution
            && (max_device_payload_bytes == 0 || self.stream_limits.is_none())
        {
            return Err("--typed resident execution requires positive device payload and explicit stream_limits".into());
        }
        if let Some(r) = self.streamed_resources {
            if self.stream_limits.is_none() {
                return Err("--typed streamed_resources requires explicit stream_limits".into());
            }
            if let Execution::ResidentDeviceF32 {
                max_device_payload_bytes,
            } = self.execution
                && max_device_payload_bytes as u64 > self.max_payload_bytes
            {
                return Err("--typed resident envelope exceeds max_payload_bytes".into());
            }
            if [
                r.max_source_workspace_bytes,
                r.max_host_workspace_bytes,
                r.max_retained_host_bytes,
            ]
            .contains(&0)
            {
                return Err("--typed streamed source/host workspace and retained-host limits must be positive".into());
            }
        }
        Ok(())
    }
    /// Load using the common retained-source seam and caller-owned evidence.
    /// Logical resource guards follow the returned Runner. Does not initialize a
    /// backend, bind listeners, verify MTP or authorize production/public release.
    /// # Errors
    /// Invalid selections, source, planning, budgets, upload and graph failures.
    pub fn load(
        self,
        mut backend: Box<dyn tritium_spec::TernaryBackend>,
        context: tritium_runtime::TensorUploadContext<'_>,
    ) -> Result<super::LoadedModel, Box<dyn std::error::Error + Send + Sync>> {
        self.validate()?;
        match self.execution {
            Execution::Host {} if backend.resident_decoder_requested() => {
                return Err("--typed host request cannot accept resident-selected backend".into());
            }
            Execution::Host {} => {}
            Execution::ResidentDeviceF32 {
                max_device_payload_bytes,
            } => {
                backend
                    .request_resident_decoder(max_device_payload_bytes)
                    .map_err(|e| {
                        format!(
                            "--typed resident execution selection on {}: {e}",
                            backend.device_id()
                        )
                    })?;
                if !backend.resident_decoder_requested() {
                    return Err(
                        "--typed backend did not retain mandatory resident selection".into(),
                    );
                }
            }
        }
        let budget = tritium_runtime::TensorPayloadBudget::new(self.max_payload_bytes);
        let policy = tritium_schema::TensorUploadPolicy {
            native_only: self.native_only,
            max_payload_bytes: Some(self.max_payload_bytes),
        };
        let source = super::ModelSource::TritFile {
            path: self.path(),
            level: &self.level,
            receiving: tritium_format::TritPackageReadLimits {
                max_source_bytes: self.max_source_bytes,
                max_manifest_bytes: self.max_manifest_bytes,
                max_directory_bytes: self.max_directory_bytes,
            },
            limits: super::Qwen35TritReadLimits {
                max_config_bytes: self.max_config_bytes,
                max_tensor_bytes: self.max_tensor_bytes,
                max_decoded_tensor_bytes: self.max_decoded_tensor_bytes,
            },
        };
        let loaded = match (self.streamed_resources, self.stream_limits) {
            (Some(r), Some(stream)) => {
                let source_workspace =
                    tritium_runtime::TensorPayloadBudget::new(r.max_source_workspace_bytes);
                let host_workspace =
                    tritium_runtime::TensorPayloadBudget::new(r.max_host_workspace_bytes);
                let device_workspace =
                    tritium_runtime::TensorPayloadBudget::new(r.max_device_workspace_bytes);
                let retained_host =
                    tritium_runtime::TensorPayloadBudget::new(r.max_retained_host_bytes);
                super::load_streamed_with_resources(
                    source,
                    backend,
                    policy,
                    super::ModelLoadResources {
                        payload: &budget,
                        source_workspace: &source_workspace,
                        host_workspace: &host_workspace,
                        device_workspace: &device_workspace,
                        retained_host: &retained_host,
                    },
                    stream.limits(),
                    context,
                )?
            }
            (Some(_), None) => {
                return Err("--typed streamed_resources requires explicit stream_limits".into());
            }
            (None, Some(stream)) => {
                super::load_with_limits(source, backend, policy, &budget, stream.limits(), context)?
            }
            (None, None) => super::load(source, backend, policy, &budget, context)?,
        };
        let expected = match self.execution {
            Execution::Host {} => tritium_spec::DecoderExecution::Host,
            Execution::ResidentDeviceF32 { .. } => {
                tritium_spec::DecoderExecution::ResidentDeviceF32
            }
        };
        if loaded.runner.execution() != expected {
            return Err("--typed loaded execution disagrees with explicit selection".into());
        }
        if self.streamed_resources.is_some() && loaded.runner.resource_admission().is_none() {
            return Err("--typed streamed load did not retain logical resource admission".into());
        }
        Ok(loaded)
    }
}

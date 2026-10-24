//! Shared explicit host-QTEA command request. Logical caps, not qualification.
use super::load_config::{StreamConfig, read_config};
use serde::Deserialize;
use std::{error::Error, path::Path};
use tritium_format::{
    TritPackageReadLimits,
    qtea::{QteaLimits, model::QteaModelReadLimits},
};

/// Bounded regular-file request shared by CLI and serving (proposed ADR0165).
///
/// Fields are private; reading and loading both validate. The declared source
/// digest is an assertion, not official-checkpoint authentication. Weight,
/// staging, metadata and cache caps are separate logical domains, not RSS.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QteaLoadConfig {
    profile: String,
    path: String,
    expected_source_model: [u8; 32],
    max_source_bytes: u64,
    max_manifest_bytes: usize,
    max_directory_bytes: usize,
    max_tensors: usize,
    max_assets: usize,
    max_asset_bytes: usize,
    max_config_bytes: usize,
    max_vector_elements: usize,
    max_tensor_bytes: usize,
    max_validation_bytes: usize,
    max_weight_bytes: usize,
    max_staging_bytes: usize,
    #[serde(default, deserialize_with = "present_preserved_cap")]
    max_preserved_tensor_elements: Option<usize>,
    stream_limits: StreamConfig,
}
fn present_preserved_cap<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<usize>, D::Error> {
    let value = usize::deserialize(deserializer)?;
    if value == 0 {
        return Err(serde::de::Error::custom(
            "hybrid preserved tensor cap must be positive",
        ));
    }
    Ok(Some(value))
}
impl QteaLoadConfig {
    /// Read at most16KiB of regular JSON, without opening model/device state.
    /// # Errors
    /// I/O, unsupported/ambiguous JSON, profile, path, digest or positive caps.
    pub fn read(path: &Path) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let config: Self = read_config(path, "--qtea")?;
        config.validate()?;
        Ok(config)
    }
    /// Explicit model path, not a discovery or custody hint.
    #[must_use]
    pub fn path(&self) -> &Path {
        Path::new(&self.path)
    }
    /// Mandatory bounded context/cache selection; not physical-memory admission.
    #[must_use]
    pub fn stream_limits(&self) -> super::StreamLimits {
        self.stream_limits.limits()
    }
    fn validate(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        match (self.profile.as_str(), self.max_preserved_tensor_elements) {
            ("tritium.qtea-host-load.v1", None) => {}
            ("tritium.qtea-hybrid-host-load.v1", Some(cap)) if cap > 0 => {}
            _ => return Err("--qtea unsupported load profile or preserved-cap selection".into()),
        }
        if !self.path().is_absolute()
            || self.path.len() > 4096
            || self.path.chars().any(char::is_control)
        {
            return Err(
                "--qtea requires an absolute model path of at most 4096 bytes without controls"
                    .into(),
            );
        }
        if self.expected_source_model == [0; 32] {
            return Err("--qtea requires a nonzero declared source-model digest".into());
        }
        if self.max_source_bytes == 0
            || [
                self.max_manifest_bytes,
                self.max_directory_bytes,
                self.max_tensors,
                self.max_assets,
                self.max_asset_bytes,
                self.max_config_bytes,
                self.max_vector_elements,
                self.max_tensor_bytes,
                self.max_validation_bytes,
                self.max_weight_bytes,
                self.max_staging_bytes,
            ]
            .contains(&0)
        {
            return Err("--qtea source/read/weight/staging limits must all be positive".into());
        }
        let stream = self.stream_limits();
        if stream.max_context == 0
            || stream.max_segment_tokens == 0
            || stream.max_host_cache_payload_bytes == 0
            || stream.max_segment_tokens > stream.max_context
        {
            return Err(
                "--qtea stream limits must be positive and segment must fit context".into(),
            );
        }
        Ok(())
    }
    /// Open one regular model and use the existing source-bound scalar-host loader.
    /// No backend selector, GPU initialization, upload or dense fallback exists.
    /// # Errors
    /// Invalid context/configuration/source, model/text admission or construction.
    pub fn load(
        self,
        context: tritium_runtime::TensorUploadContext<'_>,
    ) -> Result<super::LoadedModel, Box<dyn Error + Send + Sync>> {
        self.validate()?;
        context.recorder.check_emit::<tritium_schema::ModelLoaded>(
            context.span,
            context.parent,
            context.logical_time,
        )?;
        let (file, metadata) = super::retained_plan::open_regular_source(self.path())?;
        if metadata.len() > self.max_source_bytes {
            return Err("--qtea model exceeds max_source_bytes".into());
        }
        let limits = super::HostQteaModelLimits {
            receiving: TritPackageReadLimits {
                max_source_bytes: self.max_source_bytes,
                max_manifest_bytes: self.max_manifest_bytes,
                max_directory_bytes: self.max_directory_bytes,
            },
            model: QteaModelReadLimits {
                max_tensors: self.max_tensors,
                max_assets: self.max_assets,
                max_asset_bytes: self.max_asset_bytes,
                max_config_bytes: self.max_config_bytes,
                max_vector_elements: self.max_vector_elements,
                tensor: QteaLimits {
                    max_tensor_bytes: self.max_tensor_bytes,
                    max_validation_bytes: self.max_validation_bytes,
                },
            },
            max_weight_bytes: self.max_weight_bytes,
            max_staging_bytes: self.max_staging_bytes,
        };
        if let Some(max_preserved_tensor_elements) = self.max_preserved_tensor_elements {
            return Ok(super::LoadedModel::load_qtea_hybrid(
                file,
                self.expected_source_model,
                super::HostQteaHybridModelLimits {
                    base: limits,
                    max_preserved_tensor_elements,
                },
                self.stream_limits(),
                context,
            )?);
        }
        Ok(super::LoadedModel::load_qtea(
            file,
            self.expected_source_model,
            limits,
            self.stream_limits(),
            context,
        )?)
    }
}

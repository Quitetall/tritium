//! Shared explicit QTEA execution request. Logical caps, not qualification.
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
    #[serde(default, deserialize_with = "present_cuda")]
    cuda: Option<CudaOptions>,
    #[serde(default, deserialize_with = "present_preserved_cap")]
    max_preserved_tensor_elements: Option<usize>,
    stream_limits: StreamConfig,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CudaOptions {
    device: usize,
    max_device_weight_bytes: u64,
    max_device_staging_bytes: u64,
}
fn present_cuda<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<CudaOptions>, D::Error> {
    CudaOptions::deserialize(d).map(Some)
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
    /// Required explicit command backend, derived from the admitted profile.
    /// CUDA is host-orchestrated packed matrix execution, not a resident graph.
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        if self.cuda.is_some() { "cuda" } else { "cpu" }
    }
    fn validate(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        match (
            self.profile.as_str(),
            self.max_preserved_tensor_elements,
            self.cuda.is_some(),
        ) {
            ("tritium.qtea-host-load.v1", None, false) => {}
            ("tritium.qtea-hybrid-host-load.v1", Some(cap), false) if cap > 0 => {}
            ("tritium.qtea-cuda-ordered-host-load.v1", None, true) => {}
            ("tritium.qtea-hybrid-cuda-ordered-host-load.v1", Some(cap), true) if cap > 0 => {}
            ("tritium.qtea-cuda-swiglu-host-load.v1", None, true) => {}
            ("tritium.qtea-hybrid-cuda-swiglu-host-load.v1", Some(cap), true) if cap > 0 => {}
            ("tritium.qtea-cuda-swiglu-reuse-host-load.v1", None, true) => {}
            ("tritium.qtea-hybrid-cuda-swiglu-reuse-host-load.v1", Some(cap), true) if cap > 0 => {}
            _ => return Err("--qtea unsupported load profile or preserved-cap selection".into()),
        }
        if let Some(cuda) = &self.cuda {
            if i32::try_from(cuda.device).is_err()
                || cuda.max_device_weight_bytes == 0
                || cuda.max_device_staging_bytes == 0
            {
                return Err("--qtea invalid CUDA device or nonpositive device caps".into());
            }
            #[cfg(not(feature = "cuda"))]
            return Err("--qtea CUDA profile requires build feature cuda; no host fallback".into());
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
    /// Open one regular model with its explicit admitted execution profile.
    /// Host defaults stay unchanged. CUDA requests never fall back to host/dense.
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
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            let backend = tritium_cuda::CudaBackend::new(cuda.device)?;
            let backend = if matches!(
                self.profile.as_str(),
                "tritium.qtea-cuda-swiglu-host-load.v1"
                    | "tritium.qtea-hybrid-cuda-swiglu-host-load.v1"
                    | "tritium.qtea-cuda-swiglu-reuse-host-load.v1"
                    | "tritium.qtea-hybrid-cuda-swiglu-reuse-host-load.v1"
            ) {
                backend.with_device_swiglu()
            } else {
                backend
            };
            let backend = Box::new(
                if matches!(
                    self.profile.as_str(),
                    "tritium.qtea-cuda-swiglu-reuse-host-load.v1"
                        | "tritium.qtea-hybrid-cuda-swiglu-reuse-host-load.v1"
                ) {
                    let device =
                        tritium_runtime::TensorPayloadBudget::new(cuda.max_device_staging_bytes);
                    let host = tritium_runtime::TensorPayloadBudget::new(u64::try_from(
                        self.max_staging_bytes,
                    )?);
                    backend.with_swiglu_workspace_resources(
                        tritium_cuda::SwiGluWorkspaceResources::new(&device, &host)?,
                    )?
                } else {
                    backend
                },
            );
            let gpu = super::CudaQteaModelLimits {
                host: limits,
                max_device_weight_bytes: cuda.max_device_weight_bytes,
                max_device_staging_bytes: cuda.max_device_staging_bytes,
            };
            return Ok(
                if let Some(max_preserved) = self.max_preserved_tensor_elements {
                    super::LoadedModel::load_qtea_hybrid_cuda(
                        file,
                        self.expected_source_model,
                        gpu,
                        max_preserved,
                        self.stream_limits(),
                        backend,
                        context,
                    )?
                } else {
                    super::LoadedModel::load_qtea_cuda(
                        file,
                        self.expected_source_model,
                        gpu,
                        self.stream_limits(),
                        backend,
                        context,
                    )?
                },
            );
        }
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

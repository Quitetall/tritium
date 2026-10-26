//! Explicit experimental QTEA execution selection, not production admission.
use std::{error::Error, path::Path};

pub(crate) struct QteaConfig {
    #[cfg(feature = "host-qtea")]
    config: tritium_nn::QteaLoadConfig,
}
impl QteaConfig {
    pub(crate) fn backend_name(&self) -> &'static str {
        #[cfg(feature = "host-qtea")]
        {
            self.config.backend_name()
        }
        #[cfg(not(feature = "host-qtea"))]
        {
            "unavailable"
        }
    }
    pub(crate) fn read(path: &Path) -> Result<Self, Box<dyn Error>> {
        #[cfg(feature = "host-qtea")]
        {
            Ok(Self {
                config: tritium_nn::QteaLoadConfig::read(path)
                    .map_err(|e| -> Box<dyn Error> { e })?,
            })
        }
        #[cfg(not(feature = "host-qtea"))]
        {
            let _ = path;
            Err("--qtea requires build feature host-qtea".into())
        }
    }
    pub(crate) fn load(self) -> Result<tritium_nn::LoadedModel, Box<dyn Error>> {
        #[cfg(feature = "host-qtea")]
        {
            let mut recorder = tritium_evidence::EvidenceRecorder::new(
                "serve-qtea-load",
                [tritium_evidence::MODEL_LOADED_REGISTRATION],
            )?;
            let model = self
                .config
                .load(tritium_runtime::TensorUploadContext {
                    recorder: &mut recorder,
                    span: "load",
                    parent: None,
                    logical_time: 0,
                })
                .map_err(|e| -> Box<dyn Error> { e })?;
            if !model.tokenizer.has_source_chat_template() {
                return Err("--qtea serving requires a packaged source chat template; no inferred formatter".into());
            }
            eprintln!(
                "WARNING: QTEA candidate with explicit execution profile, not production-qualified: {}",
                serde_json::to_string(model.receipt.observation())?
            );
            eprintln!(
                "QTEA logical cache admission (not physical memory): {:?}",
                model.runner.cache_admission()
            );
            recorder.seal()?;
            Ok(model)
        }
        #[cfg(not(feature = "host-qtea"))]
        {
            Err("--qtea requires build feature host-qtea".into())
        }
    }
}

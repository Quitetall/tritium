//! Serving diagnostics around the shared bounded retained-file load request.
use std::path::Path;
#[cfg(test)]
#[path = "typed_source_tests.rs"]
mod tests;
pub(crate) struct TypedConfig(tritium_nn::TypedLoadConfig);
impl TypedConfig {
    pub(crate) fn read(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self(
            tritium_nn::TypedLoadConfig::read(path)
                .map_err(|e| -> Box<dyn std::error::Error> { e })?,
        ))
    }
    pub(crate) fn load(
        self,
        backend: Box<dyn tritium_spec::TernaryBackend>,
    ) -> Result<tritium_nn::LoadedModel, Box<dyn std::error::Error>> {
        let mut recorder = tritium_evidence::EvidenceRecorder::new(
            "serve-typed-load",
            [
                tritium_evidence::TENSOR_UPLOADED_REGISTRATION,
                tritium_evidence::MODEL_LOADED_REGISTRATION,
            ],
        )?;
        let context = tritium_runtime::TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        };
        let loaded = self
            .0
            .load(backend, context)
            .map_err(|e| -> Box<dyn std::error::Error> { e })?;
        eprintln!(
            "WARNING: typed candidate, not production-qualified: {}",
            serde_json::to_string(loaded.receipt.observation())?
        );
        eprintln!(
            "typed runtime diagnostics (not qualification): execution={:?}, context={}, segment={}, resident_payload={:?}",
            loaded.runner.execution(),
            loaded.runner.max_context(),
            loaded
                .runner
                .cache_admission()
                .map_or(loaded.runner.max_context(), |a| a
                    .limits()
                    .max_segment_tokens),
            loaded.runner.resident_payload_bytes()
        );
        if let Some(admission) = loaded.runner.resource_admission() {
            eprintln!(
                "typed encoded-stream logical resource admission (not physical fit/qualification): {admission:?}"
            );
        }
        Ok(loaded)
    }
}

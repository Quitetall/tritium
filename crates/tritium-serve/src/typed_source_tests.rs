use super::TypedConfig;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tritium_spec::{
    BackendError, DeviceBuffer, DeviceCaps, GemmShape, MpGemm, TensorCaps, TensorMatmul,
    TensorMetadata, TensorUploadPlan, TensorView, TernaryBackend,
};

#[path = "../tests/portable_fixture/mod.rs"]
mod fixture;

struct NoStreaming {
    cpu: tritium_cpu::CpuBackend,
    uploads: Arc<AtomicUsize>,
}
impl TernaryBackend for NoStreaming {
    fn device_id(&self) -> &str {
        "decoded-only-test"
    }
    fn capabilities(&self) -> DeviceCaps {
        self.cpu.capabilities()
    }
    fn tensor_caps(&self, v: TensorView<'_>) -> Result<Option<TensorCaps>, BackendError> {
        self.cpu.tensor_caps(v)
    }
    fn tensor_upload_plan(
        &self,
        v: TensorMetadata<'_>,
    ) -> Result<Option<TensorUploadPlan>, BackendError> {
        self.cpu.tensor_upload_plan(v)
    }
    fn upload_tensor(&self, v: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        self.cpu.upload_tensor(v)
    }
    fn matmul(&self, p: TensorMatmul<'_>) -> Result<(), BackendError> {
        self.cpu.matmul(p)
    }
    fn embed_rows(
        &self,
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        self.cpu.embed_rows(tensor, ids, out)
    }
    fn upload_weights(
        &self,
        bytes: &[u8],
        shape: GemmShape,
        format: tritium_core::TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.cpu.upload_weights(bytes, shape, format)
    }
    fn mpgemm(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
        self.cpu.mpgemm(p)
    }
    // Deliberately retain default unknown streaming support. A compatibility
    // fallback would succeed, upload matrices and be caught by this test.
}

struct Scratch(std::path::PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn typed_streaming_decline_never_calls_decoded_upload_or_invents_admission() {
    let root = Scratch(std::env::temp_dir().join(format!("typed-decline-{}", std::process::id())));
    std::fs::create_dir(&root.0).unwrap();
    let model = root.0.join("model.trit");
    let config_path = root.0.join("load.json");
    for hybrid in [false, true] {
        for additive in [false, true] {
            std::fs::write(&model, fixture::package(hybrid, additive)).unwrap();
            let config = serde_json::json!({
                "path":model, "level":"test", "max_source_bytes":1_000_000,
                "max_manifest_bytes":128_000,"max_directory_bytes":128_000,
                "max_config_bytes":16_000,"max_tensor_bytes":16_000,
                "max_decoded_tensor_bytes":16_000,"max_payload_bytes":1_000_000,
                "native_only":false,
                "stream_limits":{"max_context":16,"max_segment_tokens":2,"max_host_cache_payload_bytes":1_000_000},
                "streamed_resources":{"max_source_workspace_bytes":65_536,"max_host_workspace_bytes":1_000_000,"max_device_workspace_bytes":0,"max_retained_host_bytes":1_000_000}
            });
            std::fs::write(&config_path, config.to_string()).unwrap();
            let uploads = Arc::new(AtomicUsize::new(0));
            let backend = || {
                Box::new(NoStreaming {
                    cpu: tritium_cpu::CpuBackend::new(),
                    uploads: uploads.clone(),
                })
            };
            let error = TypedConfig::read(&config_path)
                .unwrap()
                .load(backend())
                .err()
                .expect("streaming must refuse the decoded-only backend")
                .to_string();
            assert!(
                error.contains("backend streaming plan is unknown"),
                "{error}"
            );
            assert_eq!(uploads.load(Ordering::SeqCst), 0);
            let mut compatibility = config;
            compatibility
                .as_object_mut()
                .unwrap()
                .remove("streamed_resources");
            std::fs::write(&config_path, compatibility.to_string()).unwrap();
            let loaded = TypedConfig::read(&config_path)
                .unwrap()
                .load(backend())
                .unwrap();
            assert!(loaded.runner.resource_admission().is_none());
            assert!(uploads.load(Ordering::SeqCst) > 0);
            drop(loaded);
        }
    }
}

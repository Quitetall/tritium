//! Structural NN dispatch tests; fixture outputs are not numerical qualification.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use tritium_evidence::{EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION};
use tritium_nn::TensorLoader;
use tritium_runtime::{TensorPayloadBudget, TensorUploadContext};
use tritium_spec::{TensorCaps, TensorMatmul, TensorSwiGlu, TensorUploadPolicy, TensorView};

#[derive(Debug, Clone, Copy)]
enum Mode {
    Decline,
    Execute,
    Fail,
}

#[derive(Debug)]
struct RegionProbe {
    cpu: tritium_cpu::CpuBackend,
    mode: Mode,
    regions: AtomicUsize,
    matrices: AtomicUsize,
}

impl RegionProbe {
    fn new(mode: Mode) -> Self {
        Self {
            cpu: tritium_cpu::CpuBackend::new(),
            mode,
            regions: AtomicUsize::new(0),
            matrices: AtomicUsize::new(0),
        }
    }
}

impl TernaryBackend for RegionProbe {
    fn device_id(&self) -> &str {
        "swiglu-structural-probe"
    }
    fn capabilities(&self) -> DeviceCaps {
        self.cpu.capabilities()
    }
    fn upload_weights(
        &self,
        packed: &[u8],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.cpu.upload_weights(packed, shape, format)
    }
    fn mpgemm(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
        self.cpu.mpgemm(p)
    }
    fn tensor_caps(&self, view: TensorView<'_>) -> Result<Option<TensorCaps>, BackendError> {
        self.cpu.tensor_caps(view)
    }
    fn upload_tensor(&self, view: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.cpu.upload_tensor(view)
    }
    fn matmul(&self, p: TensorMatmul<'_>) -> Result<(), BackendError> {
        self.matrices.fetch_add(1, Ordering::SeqCst);
        self.cpu.matmul(p)
    }
    fn try_tensor_swiglu(&self, p: TensorSwiGlu<'_>) -> Result<bool, BackendError> {
        self.regions.fetch_add(1, Ordering::SeqCst);
        assert_eq!(p.act.len(), p.batch * 2);
        assert_eq!(p.out.len(), p.batch * 2);
        assert_eq!(p.gate.len_bytes(), 24);
        assert_eq!(p.up.len_bytes(), 24);
        assert_eq!(p.down.len_bytes(), 24);
        match self.mode {
            Mode::Decline => Ok(false),
            Mode::Execute => {
                for row in p.out.as_chunks_mut::<2>().0 {
                    row.copy_from_slice(&[-3., 5.]);
                }
                Ok(true)
            }
            Mode::Fail => {
                p.out.fill(1234.);
                Err(BackendError::Backend("region injected failure".into()))
            }
        }
    }
}

fn typed(backend: &dyn TernaryBackend) -> SwiGluMlp {
    let budget = TensorPayloadBudget::new(72);
    let mut recorder =
        EvidenceRecorder::new("swiglu-probe", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let mut loader = TensorLoader::new(
        backend,
        TensorUploadPolicy::default(),
        &budget,
        TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    );
    let mut projection = |name, rows, cols| {
        Projection::Tensor(
            loader
                .upload(
                    name,
                    TensorView::Dense {
                        rows,
                        cols,
                        values: &[0.25; 6],
                    },
                )
                .unwrap(),
        )
    };
    SwiGluMlp::new(
        projection("gate", 3, 2),
        projection("up", 3, 2),
        projection("down", 2, 3),
    )
    .unwrap()
}

#[test]
fn actual_nn_layer_dispatches_one_region_without_individual_matrices() {
    let backend = RegionProbe::new(Mode::Execute);
    let mlp = typed(&backend);
    let mut out = [19.; 4];
    mlp.forward(&backend, &[1., -2., 0.5, 0.75], 2, &mut out)
        .unwrap();
    assert_eq!(out, [-3., 5., -3., 5.]);
    assert_eq!(backend.regions.load(Ordering::SeqCst), 1);
    assert_eq!(backend.matrices.load(Ordering::SeqCst), 0);
}

#[test]
fn unsupported_region_retains_actual_host_path() {
    let backend = RegionProbe::new(Mode::Decline);
    let mlp = typed(&backend);
    let input = [1., -2., 0.5, 0.75];
    let mut expected = [0.; 4];
    let mut out = [19.; 4];
    mlp.forward(&backend.cpu, &input, 2, &mut expected).unwrap();
    mlp.forward(&backend, &input, 2, &mut out).unwrap();
    assert_eq!(out, expected);
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(out.iter().any(|v| *v != 0.));
    assert_eq!(backend.regions.load(Ordering::SeqCst), 1);
    assert_eq!(backend.matrices.load(Ordering::SeqCst), 3);
}

#[test]
fn region_error_never_falls_back_or_publishes_partial_output() {
    let backend = RegionProbe::new(Mode::Fail);
    let mlp = typed(&backend);
    let mut out = [17., 19.];
    let error = mlp.forward(&backend, &[1., -2.], 1, &mut out).unwrap_err();
    assert!(
        matches!(error, NnError::Backend(message) if message.contains("region injected failure"))
    );
    assert_eq!(out, [17., 19.]);
    assert_eq!(backend.regions.load(Ordering::SeqCst), 1);
    assert_eq!(backend.matrices.load(Ordering::SeqCst), 0);
}

#[test]
fn invalid_buffers_fail_before_region_and_mixed_projection_keeps_host_path() {
    let backend = RegionProbe::new(Mode::Execute);
    let mut mlp = typed(&backend);
    let mut out = [17., 19.];
    for (input, batch) in [(&[1.][..], 1), (&[][..], usize::MAX)] {
        assert!(mlp.forward(&backend, input, batch, &mut out).is_err());
        assert_eq!(out, [17., 19.]);
    }
    assert!(mlp.forward(&backend, &[1., 2.], 1, &mut out[..1]).is_err());
    assert_eq!(backend.regions.load(Ordering::SeqCst), 0);
    mlp.up = dense_exact(3, 2);
    mlp.forward(&backend, &[1., -2.], 1, &mut out).unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
    assert_eq!(backend.regions.load(Ordering::SeqCst), 0);
    assert_eq!(backend.matrices.load(Ordering::SeqCst), 2);
}

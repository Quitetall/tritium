use tritium_core::{GemmShape, TernaryFormat, Trit};
use tritium_nn::{
    DenseLinear, NnError, Projection, ProjectionActivationMode, SwiGluMlp, TernaryLinear,
};
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, MpGemm, TernaryBackend};

fn dense_exact(n_out: usize, k_in: usize) -> Projection {
    Projection::Dense(DenseLinear::new_exact(vec![0.25; n_out * k_in], n_out, k_in).unwrap())
}

fn dense_a8(n_out: usize, k_in: usize) -> Projection {
    Projection::Dense(DenseLinear::new(vec![0.25; n_out * k_in], n_out, k_in).unwrap())
}

#[test]
fn binding_rejects_mixed_activation_modes() {
    let error = SwiGluMlp::new(dense_exact(3, 2), dense_a8(3, 2), dense_exact(2, 3))
        .err()
        .expect("mixed activation modes must fail");
    assert!(
        matches!(error, NnError::MissingConfig(ref message) if message.contains("one activation arithmetic mode")),
        "{error:?}"
    );
}

#[test]
fn binding_rejects_bad_projection_geometry() {
    let bad_up = SwiGluMlp::new(dense_exact(3, 2), dense_exact(4, 2), dense_exact(2, 3))
        .err()
        .expect("up width mismatch must fail");
    assert_eq!(
        bad_up,
        NnError::Shape {
            expected: 3,
            got: 4,
        }
    );

    let bad_down = SwiGluMlp::new(dense_exact(3, 2), dense_exact(3, 2), dense_exact(2, 4))
        .err()
        .expect("down input mismatch must fail");
    assert_eq!(
        bad_down,
        NnError::Shape {
            expected: 3,
            got: 4,
        }
    );
}

#[test]
fn binding_exposes_shared_activation_mode() {
    let exact = SwiGluMlp::new(dense_exact(3, 2), dense_exact(3, 2), dense_exact(2, 3)).unwrap();
    assert_eq!(
        exact.activation_mode().unwrap(),
        ProjectionActivationMode::F32
    );

    let a8 = SwiGluMlp::new(dense_a8(3, 2), dense_a8(3, 2), dense_a8(2, 3)).unwrap();
    assert_eq!(a8.activation_mode().unwrap(), ProjectionActivationMode::A8);
}

#[test]
fn binding_rejects_nonfinite_gate_parameters() {
    let mut gate = dense_exact(3, 2);
    if let Projection::Dense(linear) = &mut gate {
        linear.weights[1] = f32::NAN;
    }
    let error = SwiGluMlp::new(gate, dense_exact(3, 2), dense_exact(2, 3))
        .err()
        .expect("non-finite gate projection must fail");
    assert!(matches!(error, NnError::Backend(message) if message.contains("non-finite")));
}

#[test]
fn activation_mode_keeps_parameter_scans_off_the_forward_hot_path() {
    let mut mlp = SwiGluMlp::new(dense_exact(3, 2), dense_exact(3, 2), dense_exact(2, 3))
        .expect("valid SwiGLU");
    if let Projection::Dense(linear) = &mut mlp.gate {
        linear.weights[1] = f32::NAN;
    }

    assert_eq!(
        mlp.activation_mode().unwrap(),
        ProjectionActivationMode::F32
    );
}

#[test]
fn binding_rejects_mutated_retained_parameter_geometry() {
    let mut dense_gate = dense_exact(3, 2);
    if let Projection::Dense(linear) = &mut dense_gate {
        linear.weights.pop();
    }
    assert!(matches!(
        SwiGluMlp::new(dense_gate, dense_exact(3, 2), dense_exact(2, 3)),
        Err(NnError::Shape {
            expected: 6,
            got: 5
        })
    ));

    let cpu = tritium_cpu::CpuBackend::new();
    let mut ternary_gate =
        Projection::Ternary(TernaryLinear::new(&cpu, &[Trit::ZERO; 6], 3, 2, 1.0).unwrap());
    if let Projection::Ternary(linear) = &mut ternary_gate {
        linear.scales.pop();
    }
    assert!(matches!(
        SwiGluMlp::new(ternary_gate, dense_a8(3, 2), dense_a8(2, 3)),
        Err(NnError::Shape {
            expected: 3,
            got: 2
        })
    ));
}

#[test]
fn forward_rejects_extent_overflow_without_publishing_output() {
    let backend = tritium_cpu::CpuBackend::new();
    let mlp = SwiGluMlp::new(dense_exact(3, 2), dense_exact(3, 2), dense_exact(2, 3)).unwrap();
    let mut output = [17.0, 19.0];
    let error = mlp
        .forward(&backend, &[], usize::MAX, &mut output)
        .expect_err("input extent overflow must fail");
    assert_eq!(
        error,
        NnError::Shape {
            expected: usize::MAX,
            got: 0,
        }
    );
    assert_eq!(output, [17.0, 19.0]);
}

#[derive(Debug, Default)]
struct MutateThenFailBackend {
    cpu: tritium_cpu::CpuBackend,
}

impl TernaryBackend for MutateThenFailBackend {
    fn device_id(&self) -> &str {
        "mutate-then-fail"
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

    fn mpgemm(&self, parameters: MpGemm<'_>) -> Result<(), BackendError> {
        parameters.out.fill(1234.0);
        Err(BackendError::Backend(
            "intentional mutate-then-fail backend".to_owned(),
        ))
    }
}

#[test]
fn down_failure_does_not_publish_partial_output() {
    let cpu = tritium_cpu::CpuBackend::new();
    let down = Projection::Ternary(TernaryLinear::new(&cpu, &[Trit::ZERO; 6], 2, 3, 1.0).unwrap());
    let mlp = SwiGluMlp::new(dense_a8(3, 2), dense_a8(3, 2), down).unwrap();
    let mut output = [17.0, 19.0];
    let error = mlp
        .forward(
            &MutateThenFailBackend::default(),
            &[1.0, -2.0],
            1,
            &mut output,
        )
        .expect_err("down backend failure must propagate");
    assert!(matches!(error, NnError::Backend(message) if message.contains("mutate-then-fail")));
    assert_eq!(output, [17.0, 19.0]);
}

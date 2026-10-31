use std::io::{self, Cursor, Write};
use tritium_format::qtea::{QteaError, QteaLimits, QteaTensor};
use tritium_quantize::qtea_reference::{
    ArithmeticError, TensorFit, TensorFitConfig, TensorFitPlan,
};
use tritium_quantize::{CurvatureSourceId, InputGram, InputGramAccumulator};
use tritium_salt::{
    BuiltinSolverStatus, ContentId, QteaTensorOrigin, QteaTensorWorkError, QteaTensorWorkPlan,
    SolverFamily, builtin_solver_capabilities,
};

struct Fixture {
    rows: usize,
    cols: usize,
    weights: Vec<f32>,
    gram: InputGram,
}
impl Fixture {
    fn new(rows: usize, cols: usize, source: u8, scale: f32) -> Self {
        let weights = (0..rows)
            .flat_map(|r| {
                (0..cols).map(move |c| {
                    (((r + 1) * (c + 3)) % 17) as f32 * 0.35 - 2.8
                        + c as f32 * 0.19
                        + r as f32 * 0.07
                })
            })
            .collect();
        let mut x = vec![0.; cols * cols];
        for c in 0..cols {
            x[c * cols + c] = scale + c as f32 * 0.05;
        }
        let source = CurvatureSourceId::new([source; 32], [2; 32], [3; 32]).unwrap();
        let mut builder = InputGramAccumulator::new_bound(cols, source, 0).unwrap();
        builder.accumulate_batch(&x, cols, None, None).unwrap();
        Self {
            rows,
            cols,
            weights,
            gram: builder.finish().unwrap(),
        }
    }
    fn plan(&self, config: TensorFitConfig) -> TensorFitPlan<'_> {
        TensorFit::plan(
            &self.weights,
            self.rows,
            self.cols,
            &self.gram,
            config,
            TensorFit::required_work_bytes(self.rows, self.cols).unwrap(),
        )
        .unwrap()
    }
}
fn config(group_size: usize, salient_ratio: f64) -> TensorFitConfig {
    TensorFitConfig {
        group_size,
        salient_ratio,
        ..TensorFitConfig::default()
    }
}
fn limits() -> QteaLimits {
    QteaLimits {
        max_tensor_bytes: 1 << 20,
        max_validation_bytes: 4096,
    }
}
fn origin() -> QteaTensorOrigin<'static> {
    QteaTensorOrigin {
        source_id: ContentId::of_bytes(b"declared campaign artifact, not numerical model ID"),
        owner: "model.layers.0.weight",
        tensor_index: 7,
        source_tensor_digest: [9; 32],
    }
}

#[test]
fn real_work_writer_is_byte_identical_and_cold_runtime_owns_execution() {
    for (r, c, g) in [(4, 8, 2), (8, 12, 4), (20, 6, 2)] {
        let f = Fixture::new(r, c, 1, 1.);
        for ratio in [0., 0.25, 1.] {
            let numeric = f.plan(config(g, ratio));
            let direct = numeric
                .clone()
                .fit()
                .unwrap()
                .to_qtea_tensor(limits())
                .unwrap();
            let plan = QteaTensorWorkPlan::new(numeric, origin(), limits(), usize::MAX).unwrap();
            let id = plan.input_id();
            let ledger = plan.byte_ledger();
            let metadata = plan.metadata();
            let mut bytes = Vec::new();
            let result = plan.fit_to(&mut bytes).unwrap();
            assert_eq!(bytes, direct.as_bytes());
            assert_eq!(result.input_id(), id);
            assert_eq!(result.tensor_id(), ContentId::of_bytes(&bytes));
            assert_eq!(result.metadata(), metadata);
            assert_eq!(result.byte_ledger(), ledger);
            assert_eq!(result.wire_digest(), direct.view().content_digest());
            assert_ne!(result.tensor_id().as_bytes(), &result.wire_digest());
            let cold = QteaTensor::read_from(&mut Cursor::new(bytes), limits()).unwrap();
            drop(direct);
            let x = vec![0.25; c];
            let mut actual = vec![0.; r];
            let mut staging = vec![0.; r];
            tritium_cpu::qtea::matmul_into(cold.view(), &x, 1, &mut actual, &mut staging).unwrap();
            assert!(actual.iter().all(|v| v.is_finite()));
        }
    }
    let qtea = builtin_solver_capabilities()
        .unwrap()
        .into_iter()
        .find(|c| c.family() == SolverFamily::QteaSalientResidual)
        .unwrap();
    assert!(matches!(
        qtea.status(),
        BuiltinSolverStatus::Available {
            trust: tritium_salt::SolverTrust::Experimental
        }
    ));
    assert_eq!(
        qtea.descriptor(),
        Some(
            tritium_salt::QteaReferenceAdapter::new()
                .unwrap()
                .descriptor()
        )
    );
}

#[test]
fn owner_storage_and_actual_numerical_inputs_are_distinct_bound_namespaces() {
    let f = Fixture::new(4, 4, 1, 1.);
    let numeric = f.plan(config(2, 0.25));
    let make = |o| QteaTensorWorkPlan::new(numeric.clone(), o, limits(), usize::MAX).unwrap();
    let base = make(origin());
    let mut identity = b"tritium qtea tensor work input v1\0".to_vec();
    identity.extend_from_slice(origin().source_id.as_bytes());
    identity.extend_from_slice(&(origin().owner.len() as u64).to_le_bytes());
    identity.extend_from_slice(origin().owner.as_bytes());
    identity.extend_from_slice(&origin().tensor_index.to_le_bytes());
    identity.extend_from_slice(&origin().source_tensor_digest);
    identity.extend_from_slice(&numeric.input_digest());
    assert_eq!(base.input_id(), ContentId::of_bytes(&identity));
    for o in [
        QteaTensorOrigin {
            source_id: ContentId::of_bytes(b"other source"),
            ..origin()
        },
        QteaTensorOrigin {
            owner: "model.layers.0.alias",
            ..origin()
        },
        QteaTensorOrigin {
            tensor_index: 8,
            ..origin()
        },
        QteaTensorOrigin {
            source_tensor_digest: [8; 32],
            ..origin()
        },
    ] {
        assert_ne!(base.input_id(), make(o).input_id());
    }
    let extra_caps = QteaTensorWorkPlan::new(
        numeric,
        origin(),
        QteaLimits {
            max_tensor_bytes: limits().max_tensor_bytes + 1,
            ..limits()
        },
        usize::MAX,
    )
    .unwrap();
    assert_eq!(base.input_id(), extra_caps.input_id());
    for alt in [Fixture::new(4, 4, 2, 1.), Fixture::new(4, 4, 1, 2.)] {
        assert_ne!(
            base.input_id(),
            QteaTensorWorkPlan::new(alt.plan(config(2, 0.25)), origin(), limits(), usize::MAX)
                .unwrap()
                .input_id()
        );
    }
    let mut changed = Fixture::new(4, 4, 1, 1.);
    changed.weights[0] = f32::from_bits(changed.weights[0].to_bits() ^ 1);
    assert_ne!(
        base.input_id(),
        QteaTensorWorkPlan::new(
            changed.plan(config(2, 0.25)),
            origin(),
            limits(),
            usize::MAX
        )
        .unwrap()
        .input_id()
    );
    assert_ne!(
        base.input_id(),
        QteaTensorWorkPlan::new(f.plan(config(2, 1.)), origin(), limits(), usize::MAX)
            .unwrap()
            .input_id()
    );
    let text = format!("{base:?}");
    assert!(!text.contains("weights: ["));
    assert!(!text.contains("InputGram"));
}

#[test]
fn complete_logical_scope_caps_and_invalid_origins_refuse_before_execution() {
    let f = Fixture::new(4, 4, 1, 1.);
    let numeric = f.plan(config(2, 0.25));
    let make = |o, l, cap| QteaTensorWorkPlan::new(numeric.clone(), o, l, cap);
    let plan = make(origin(), limits(), usize::MAX).unwrap();
    let m = plan.memory();
    assert_eq!(m.borrowed_input_bytes, 4 * 4 * 4 + 4 * 4 * 8);
    assert_eq!(m.native_fit_bytes, numeric.required_work_bytes());
    assert_eq!(
        m.total_logical_bytes,
        m.borrowed_input_bytes + m.native_fit_bytes + limits().max_tensor_bytes + 4
    );
    assert!(make(origin(), limits(), m.total_logical_bytes).is_ok());
    assert_eq!(
        make(origin(), limits(), m.total_logical_bytes - 1).unwrap_err(),
        QteaTensorWorkError::LogicalLimit {
            required: m.total_logical_bytes,
            limit: m.total_logical_bytes - 1
        }
    );
    assert_eq!(
        make(
            origin(),
            QteaLimits {
                max_tensor_bytes: usize::MAX,
                ..limits()
            },
            usize::MAX
        )
        .unwrap_err(),
        QteaTensorWorkError::SizeOverflow
    );
    assert_eq!(
        make(
            origin(),
            QteaLimits {
                max_tensor_bytes: plan.byte_ledger().total_bytes - 1,
                ..limits()
            },
            usize::MAX
        )
        .unwrap_err(),
        QteaTensorWorkError::Format(QteaError::TensorLimit)
    );
    assert_eq!(
        make(
            origin(),
            QteaLimits {
                max_validation_bytes: 3,
                ..limits()
            },
            usize::MAX
        )
        .unwrap_err(),
        QteaTensorWorkError::Format(QteaError::ValidationLimit)
    );
    let too_long = "x".repeat(4097);
    for o in [
        QteaTensorOrigin {
            source_id: ContentId::from_digest([0; 32]),
            ..origin()
        },
        QteaTensorOrigin {
            source_tensor_digest: [0; 32],
            ..origin()
        },
        QteaTensorOrigin {
            owner: "",
            ..origin()
        },
        QteaTensorOrigin {
            owner: "bad\nowner",
            ..origin()
        },
        QteaTensorOrigin {
            owner: &too_long,
            ..origin()
        },
    ] {
        assert_eq!(
            QteaTensorWorkPlan::new(numeric.clone(), o, limits(), usize::MAX).unwrap_err(),
            QteaTensorWorkError::InvalidOrigin
        );
    }
}

struct Sink {
    bytes: Vec<u8>,
    fail_after: Option<usize>,
    interrupted: bool,
    calls: usize,
}
impl Write for Sink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.calls += 1;
        if self.interrupted {
            self.interrupted = false;
            return Err(io::ErrorKind::Interrupted.into());
        }
        if self
            .fail_after
            .is_some_and(|limit| self.bytes.len() >= limit)
        {
            return Err(io::ErrorKind::BrokenPipe.into());
        }
        let count = bytes
            .len()
            .min(7)
            .min(self.fail_after.map_or(usize::MAX, |n| n - self.bytes.len()));
        self.bytes.extend_from_slice(&bytes[..count]);
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn partial_failure_returns_no_result_and_interrupted_writer_retries_exact_bytes() {
    let f = Fixture::new(4, 4, 1, 1.);
    let make = || {
        QteaTensorWorkPlan::new(f.plan(config(2, 0.25)), origin(), limits(), usize::MAX).unwrap()
    };
    let mut partial = Sink {
        bytes: vec![],
        fail_after: Some(7),
        interrupted: false,
        calls: 0,
    };
    assert_eq!(
        make().fit_to(&mut partial).unwrap_err(),
        QteaTensorWorkError::Format(QteaError::Io(io::ErrorKind::BrokenPipe))
    );
    assert_eq!(partial.bytes.len(), 7);
    let mut retried = Sink {
        bytes: vec![],
        fail_after: None,
        interrupted: true,
        calls: 0,
    };
    let result = make().fit_to(&mut retried).unwrap();
    assert_eq!(result.tensor_id(), ContentId::of_bytes(&retried.bytes));
    assert_eq!(retried.bytes.len(), result.byte_ledger().total_bytes);
    assert!(retried.calls > 1);
    assert_eq!(partial.bytes, retried.bytes[..7]);
}

#[test]
fn actual_numerical_failure_never_touches_writer() {
    let mut f = Fixture::new(4, 4, 1, 1.);
    let source = CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap();
    let mut builder = InputGramAccumulator::new_bound(4, source, 0).unwrap();
    builder.accumulate_batch(&[1.; 4], 1, None, None).unwrap();
    f.gram = builder.finish().unwrap();
    let plan = QteaTensorWorkPlan::new(
        f.plan(TensorFitConfig {
            damping: 0.,
            ..config(2, 0.25)
        }),
        origin(),
        limits(),
        usize::MAX,
    )
    .unwrap();
    let mut sink = Sink {
        bytes: vec![],
        fail_after: None,
        interrupted: false,
        calls: 0,
    };
    assert_eq!(
        plan.fit_to(&mut sink).unwrap_err(),
        QteaTensorWorkError::Fit(ArithmeticError::InvalidGram)
    );
    assert_eq!(sink.calls, 0);
    assert!(sink.bytes.is_empty());
}

#[test]
fn a_zero_progress_writer_is_not_a_successful_result() {
    struct Zero;
    impl Write for Zero {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Ok(0)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let f = Fixture::new(4, 4, 1, 1.);
    let plan =
        QteaTensorWorkPlan::new(f.plan(config(2, 0.25)), origin(), limits(), usize::MAX).unwrap();
    assert_eq!(
        plan.fit_to(&mut Zero).unwrap_err(),
        QteaTensorWorkError::Format(QteaError::Io(io::ErrorKind::WriteZero))
    );
}

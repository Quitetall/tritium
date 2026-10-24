//! Common observed calibration to real QTEA work; synthetic context is not custody.
use tritium_format::qtea::QteaLimits;
use tritium_quantize::engine::{
    CalibratedTensor, CalibrationBatch, CalibrationCacheBuilder, CalibrationCollector,
    CalibrationInput, CalibrationOutputBatch,
};
use tritium_quantize::qtea_reference::{TensorFit, TensorFitConfig};
use tritium_quantize::{CurvatureSourceId, InputGramAccumulator};
use tritium_salt::{
    ContentId, QteaCalibratedTensorInput, QteaCalibrationError, QteaTensorOrigin,
    QteaTensorWorkError, QteaTensorWorkPlan,
};
use tritium_schema::*;

const X: [f32; 16] = [
    1., 0., 0., 0., 0.25, 1., 0., 0., 0., 0.5, 1., 0., 0., 0., 0.75, 1.,
];
const W: [f64; 4] = [1., 2., 0., 3.];
const M: [bool; 4] = [true, false, true, true];
fn original() -> QuantSourceTensor {
    QuantSourceTensor {
        name: "model.weight".into(),
        shape: vec![4, 4],
        precision: ScalePrecision::F32,
    }
}
fn source() -> CalibrationSource {
    CalibrationSource {
        model: FitInputDigest([1; 32]),
        cache: FitInputDigest([2; 32]),
        tokens: FitInputDigest([3; 32]),
    }
}
fn plan(
    partition: CalibrationPartition,
    consumer: CalibrationConsumer,
    output: CalibrationOutput,
) -> CalibrationPlan {
    let mut p = CalibrationPlan {
        tensor: original(),
        source: source(),
        partition,
        consumer,
        output,
        sample_start: 17,
        damping: 0.,
        max_numeric_bytes: u64::MAX,
    };
    // Grouped collection additionally admits batch-slicing scratch at accumulate.
    p.max_numeric_bytes = CalibrationCollector::required_numeric_bytes(&p).unwrap()
        + (X.len() * std::mem::size_of::<f32>()) as u64;
    p
}
fn batch(start: usize, end: usize) -> CalibrationBatch<'static> {
    CalibrationBatch {
        samples: end - start,
        input: CalibrationInput::Activations(&X[start * 4..end * 4]),
        output: CalibrationOutputBatch::None,
        weights: Some(&W[start..end]),
        mask: Some(&M[start..end]),
    }
}
fn capture(split: bool) -> CalibratedTensor {
    let mut c = CalibrationCollector::new(plan(
        CalibrationPartition::Full,
        CalibrationConsumer::Matmul,
        CalibrationOutput::None,
    ))
    .unwrap();
    if split {
        c.accumulate(batch(0, 1)).unwrap();
        c.accumulate(batch(1, 4)).unwrap();
    } else {
        c.accumulate(batch(0, 4)).unwrap();
    }
    c.finish().unwrap()
}
fn origin() -> QteaTensorOrigin<'static> {
    QteaTensorOrigin {
        source_id: ContentId::of_bytes(b"different campaign namespace"),
        owner: "model.weight",
        tensor_index: 3,
        source_tensor_digest: [9; 32],
    }
}
fn input<'a>(
    o: &'a QuantSourceTensor,
    w: &'a [f32],
    c: &'a CalibratedTensor,
) -> QteaCalibratedTensorInput<'a> {
    QteaCalibratedTensorInput {
        original: o,
        weights: w,
        calibration: c,
        source_model: OriginalSourceId::from_bytes([1; 32]),
        origin: origin(),
    }
}
fn config() -> TensorFitConfig {
    TensorFitConfig {
        group_size: 2,
        ..Default::default()
    }
}
fn limits() -> QteaLimits {
    QteaLimits {
        max_tensor_bytes: 4096,
        max_validation_bytes: 4,
    }
}
fn make<'a>(
    i: QteaCalibratedTensorInput<'a>,
) -> Result<QteaTensorWorkPlan<'a>, QteaCalibrationError> {
    QteaTensorWorkPlan::from_calibrated(
        i,
        config(),
        limits(),
        TensorFit::required_work_bytes(4, 4).unwrap(),
        usize::MAX,
    )
}

#[test]
fn actual_full_weighted_masked_collector_and_bridge_preserve_native_bytes() {
    let o = original();
    let w = [-1., 1., -2., 2.].repeat(4);
    let c = capture(false);
    let split = capture(true);
    assert_eq!(c.factors(), split.factors());
    assert_eq!(c.inputs(), split.inputs());
    let base = plan(
        CalibrationPartition::Full,
        CalibrationConsumer::Matmul,
        CalibrationOutput::None,
    );
    let mut left = CalibrationCollector::new(base.clone()).unwrap();
    left.accumulate(batch(0, 2)).unwrap();
    let mut tail = base.clone();
    tail.sample_start += 2;
    let mut right = CalibrationCollector::new(tail).unwrap();
    right.accumulate(batch(2, 4)).unwrap();
    let mut merged = CalibrationCollector::new(base).unwrap();
    merged.merge_shard(&right).unwrap();
    merged.merge_shard(&left).unwrap();
    let merged = merged.finish().unwrap();
    assert_eq!(c.factors(), merged.factors());
    let mut direct = InputGramAccumulator::new_bound(
        4,
        CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap(),
        17,
    )
    .unwrap();
    direct.accumulate_batch(&X, 4, Some(&W), Some(&M)).unwrap();
    let gram = direct.finish().unwrap();
    assert_eq!(&gram, &c.inputs()[0]);
    assert_eq!(gram.sample_count(), 4);
    assert_eq!(gram.selected_count(), 3);
    assert_eq!(gram.total_weight(), 4.);
    let p = make(input(&o, &w, &c)).unwrap();
    let q = make(input(&o, &w, &split)).unwrap();
    let raw = QteaTensorWorkPlan::new(
        TensorFit::plan(&w, 4, 4, &gram, config(), usize::MAX).unwrap(),
        origin(),
        limits(),
        usize::MAX,
    )
    .unwrap();
    assert_eq!(p.input_id(), q.input_id());
    assert_eq!(p.input_id(), raw.input_id());
    assert_eq!(
        p.input_id(),
        make(input(&o, &w, &merged)).unwrap().input_id()
    );
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut d = Vec::new();
    let pa = p.fit_to(&mut a).unwrap();
    q.fit_to(&mut b).unwrap();
    raw.fit_to(&mut d).unwrap();
    assert_eq!(a, b);
    assert_eq!(a, d);
    assert_eq!(pa.tensor_id(), ContentId::of_bytes(&a));
    assert_ne!(origin().source_id.as_bytes(), &source().model.0); // Do not conflate namespaces.
}

#[test]
fn context_name_precision_shape_source_and_owner_mismatches_refuse() {
    let o = original();
    let w = [-1., 1., -2., 2.].repeat(4);
    let c = capture(false);
    for case in 0..5 {
        let mut expected = o.clone();
        if case == 0 {
            expected.name = "alias.weight".into();
        }
        if case == 1 {
            expected.precision = ScalePrecision::Bf16;
        }
        if case == 2 {
            expected.shape = vec![8, 4];
        }
        let mut i = input(&expected, &w, &c);
        if case == 3 {
            i.source_model = OriginalSourceId::from_bytes([8; 32]);
        }
        if case == 4 {
            i.origin.owner = "alias.weight";
        }
        assert_eq!(make(i).unwrap_err(), QteaCalibrationError::ContextMismatch);
    }
    let mut changed = w.clone();
    changed[0] = -1.01;
    // Metadata equality does not authenticate original bytes; actual f32 inputs are hashed.
    assert_ne!(
        make(input(&o, &w, &c)).unwrap().input_id(),
        make(input(&o, &changed, &c)).unwrap().input_id()
    );
    let safe = format!("{:?}", input(&o, &w, &c));
    assert!(!safe.contains("values:"));
    assert!(!safe.contains("weights: ["));
}

#[test]
fn declared_grouped_gather_and_output_statistics_cannot_be_relabelled_full_input() {
    let o = original();
    let w = [-1., 1., -2., 2.].repeat(4);
    for width in [2, 4, 8] {
        let mut c = CalibrationCollector::new(plan(
            CalibrationPartition::Groups { columns: width },
            CalibrationConsumer::Matmul,
            CalibrationOutput::None,
        ))
        .unwrap();
        c.accumulate(batch(0, 4)).unwrap();
        let c = c.finish().unwrap();
        assert_eq!(
            make(input(&o, &w, &c)).unwrap_err(),
            QteaCalibrationError::RequiresInputOnlyFullGram
        );
    }
    let y = [1.; 16];
    let mut dense = CalibrationCollector::new(plan(
        CalibrationPartition::Full,
        CalibrationConsumer::Matmul,
        CalibrationOutput::DenseFisher,
    ))
    .unwrap();
    let mut b = batch(0, 4);
    b.output = CalibrationOutputBatch::Dense(&y);
    dense.accumulate(b).unwrap();
    let c = dense.finish().unwrap();
    assert_eq!(
        make(input(&o, &w, &c)).unwrap_err(),
        QteaCalibrationError::RequiresInputOnlyFullGram
    );
    let indices = [0, 1, 2, 3];
    let factors = [1.; 4];
    let mut gather = CalibrationCollector::new(plan(
        CalibrationPartition::Full,
        CalibrationConsumer::GatherIdentity,
        CalibrationOutput::IndexedFisher,
    ))
    .unwrap();
    gather
        .accumulate(CalibrationBatch {
            samples: 4,
            input: CalibrationInput::GatherIdentity,
            output: CalibrationOutputBatch::Indexed {
                indices: &indices,
                factors: &factors,
            },
            weights: None,
            mask: None,
        })
        .unwrap();
    let c = gather.finish().unwrap();
    assert_eq!(
        make(input(&o, &w, &c)).unwrap_err(),
        QteaCalibrationError::RequiresInputOnlyFullGram
    );
}

#[test]
fn existing_native_codec_and_combined_limits_still_apply() {
    let o = original();
    let w = [-1., 1., -2., 2.].repeat(4);
    let c = capture(false);
    let cap = TensorFit::required_work_bytes(4, 4).unwrap();
    assert!(matches!(
        QteaTensorWorkPlan::from_calibrated(
            input(&o, &w, &c),
            config(),
            limits(),
            cap - 1,
            usize::MAX
        ),
        Err(QteaCalibrationError::Work(QteaTensorWorkError::Fit(_)))
    ));
    assert!(matches!(
        QteaTensorWorkPlan::from_calibrated(
            input(&o, &w, &c),
            config(),
            QteaLimits {
                max_tensor_bytes: 1,
                ..limits()
            },
            cap,
            usize::MAX
        ),
        Err(QteaCalibrationError::Work(QteaTensorWorkError::Format(_)))
    ));
    assert!(matches!(
        QteaTensorWorkPlan::from_calibrated(input(&o, &w, &c), config(), limits(), cap, 0),
        Err(QteaCalibrationError::Work(
            QteaTensorWorkError::LogicalLimit { .. }
        ))
    ));
}

#[test]
fn finalized_existing_cache_replays_directly_into_the_same_qtea_bridge() {
    let o = original();
    let w = [-1., 1., -2., 2.].repeat(4);
    let request = CalibrationCacheRequest {
        tensor: o.clone(),
        model: OriginalSourceId::from_bytes([1; 32]),
        consumer: CalibrationConsumer::Matmul,
        output: CalibrationOutput::None,
        sample_start: 17,
        limits: CalibrationCacheLimits {
            max_payload_bytes: 1_000_000,
            max_samples: 4,
            max_windows: 1,
        },
    };
    let mut cache = CalibrationCacheBuilder::new(
        request,
        b"synthetic tokenizer",
        b"declared synthetic objective",
    )
    .unwrap();
    cache.capture(&[10, 11, 12, 13], batch(0, 4)).unwrap();
    let cache = cache.finish().unwrap();
    let a = cache
        .collect(CalibrationPartition::Full, 0., 1_000_000)
        .unwrap();
    let b = cache
        .collect_chunked(CalibrationPartition::Full, 0., 1_000_000, 1)
        .unwrap();
    assert_eq!(a.factors(), b.factors());
    assert_eq!(a.factors().plan.source.cache.0, *cache.id().as_bytes());
    let pa = make(input(&o, &w, &a)).unwrap();
    let pb = make(input(&o, &w, &b)).unwrap();
    assert_eq!(pa.input_id(), pb.input_id());
    let mut bytes = Vec::new();
    pa.fit_to(&mut bytes).unwrap();
    assert!(!bytes.is_empty());
}

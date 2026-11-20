//! Physical transactional KV/GQA checks, not model quality or speed qualification.
#![cfg(feature = "cuda")]

use tritium_cuda::CudaBackend;
use tritium_nn::gqa_attention;
use tritium_spec::{
    AttentionGeometry, AttentionLimits, AttentionSegment, BackendError, ResidentAttention,
    TernaryBackend,
};

fn cuda() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(backend) => Some(backend),
        Err(error) if std::env::var("TRITIUM_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("required physical CUDA backend unavailable: {error}")
        }
        Err(error) => {
            eprintln!("SKIP resident attention physical check: {error}");
            None
        }
    }
}

fn limits(geometry: AttentionGeometry) -> AttentionLimits {
    let mut limits = AttentionLimits {
        max_context: 13,
        max_segment_tokens: 5,
        max_device_payload_bytes: usize::MAX,
    };
    limits.max_device_payload_bytes = geometry.payload_bytes(limits).unwrap();
    limits
}

fn ramp(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 13 + seed * 7) % 43) as f32 / 17. - 1.1)
        .collect()
}

fn compare(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    assert!(expected.iter().all(|v| v.is_finite()));
    assert!(
        expected.iter().any(|v| v.abs() > 1e-3),
        "constant-zero oracle"
    );
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        let tolerance = 3e-5 + expected.abs() * 3e-5;
        assert!(
            got.is_finite() && (got - expected).abs() <= tolerance,
            "lane {index}: device={got}, host={expected}, bound={tolerance}"
        );
    }
}

struct Inputs {
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    tokens: usize,
    scale: f32,
}
impl Inputs {
    fn new(geometry: AttentionGeometry, tokens: usize, seed: usize) -> Self {
        let (q, kv) = geometry.widths().unwrap();
        Self {
            query: ramp(tokens * q, seed),
            key: ramp(tokens * kv, seed + 2),
            value: ramp(tokens * kv, seed + 5),
            tokens,
            scale: 1. / (geometry.head_dim as f32).sqrt(),
        }
    }
    fn stage(
        &self,
        resident: &mut dyn ResidentAttention,
        output: &mut [f32],
    ) -> Result<(), BackendError> {
        resident.stage(AttentionSegment {
            query: &self.query,
            key: &self.key,
            value: &self.value,
            tokens: self.tokens,
            scale: self.scale,
            output,
        })
    }
    fn oracle(
        &self,
        geometry: AttentionGeometry,
        prefix_key: &[f32],
        prefix_value: &[f32],
    ) -> Vec<f32> {
        let mut key = prefix_key.to_vec();
        key.extend_from_slice(&self.key);
        let mut value = prefix_value.to_vec();
        value.extend_from_slice(&self.value);
        let kv_width = geometry.widths().unwrap().1;
        let offset = prefix_key.len() / kv_width;
        let mut output = vec![f32::NAN; self.query.len()];
        gqa_attention(
            &self.query,
            &key,
            &value,
            self.tokens,
            offset + self.tokens,
            geometry.query_heads,
            geometry.kv_heads,
            geometry.head_dim,
            self.scale,
            offset,
            &mut output,
        )
        .unwrap();
        output
    }
}

#[test]
fn prefill_cached_segments_abort_replace_reset_and_factory_lifetime() {
    let Some(backend) = cuda() else {
        return;
    };
    let geometries = [
        (1, 1, 1),
        (6, 2, 7),
        (8, 2, 32),
        (4, 1, 64),
        (3, 3, 128),
        (4, 2, 256),
        (2, 1, 1024),
    ];
    let mut executors: Vec<_> = geometries
        .into_iter()
        .map(|(query_heads, kv_heads, head_dim)| {
            let geometry = AttentionGeometry {
                query_heads,
                kv_heads,
                head_dim,
            };
            backend
                .resident_attention(geometry, limits(geometry))
                .unwrap()
                .unwrap()
        })
        .collect();
    // Returned resources, functions and context remain owned without a backend.
    drop(backend);
    let mut comparisons = 0;
    for resident in &mut executors {
        let geometry = resident.geometry();
        assert_eq!(resident.limits(), limits(geometry));
        assert_eq!(
            resident.payload_bytes(),
            limits(geometry).max_device_payload_bytes
        );
        let mut prefix_key = Vec::new();
        let mut prefix_value = Vec::new();
        for (iteration, tokens) in [5, 3, 1, 4].into_iter().enumerate() {
            let base = resident.len();
            let input = Inputs::new(geometry, tokens, iteration + 1);
            let expected = input.oracle(geometry, &prefix_key, &prefix_value);
            let mut output = vec![f32::NAN; expected.len()];
            input.stage(resident.as_mut(), &mut output).unwrap();
            compare(&output, &expected);
            comparisons += 1;
            assert_eq!(resident.len(), base);
            assert_eq!(resident.staged_len(), Some(base + tokens));

            // Model-level cancellation needs no device/prefix copy. A subsequent
            // unrelated stage may overwrite the inaccessible tail, not the prefix.
            resident.abort();
            resident.commit();
            assert_eq!(resident.len(), base);
            let abandoned = Inputs::new(geometry, tokens, iteration + 40);
            abandoned.stage(resident.as_mut(), &mut output).unwrap();
            input.stage(resident.as_mut(), &mut output).unwrap();
            compare(&output, &expected);
            comparisons += 1;

            // A rejected replacement must invalidate that pending transaction.
            let mut malformed = Inputs::new(geometry, tokens, iteration + 80);
            malformed.key.pop();
            output.fill(-1234.);
            assert!(malformed.stage(resident.as_mut(), &mut output).is_err());
            assert!(output.iter().all(|&v| v == -1234.));
            assert_eq!(resident.staged_len(), None);
            resident.commit();
            assert_eq!(resident.len(), base);
            input.stage(resident.as_mut(), &mut output).unwrap();
            compare(&output, &expected);
            comparisons += 1;
            resident.commit();
            assert_eq!(resident.len(), base + tokens);
            assert_eq!(resident.staged_len(), None);
            prefix_key.extend_from_slice(&input.key);
            prefix_value.extend_from_slice(&input.value);
        }
        let input = Inputs::new(geometry, 1, 100);
        let mut output = vec![-1234.; input.query.len()];
        assert!(input.stage(resident.as_mut(), &mut output).is_err());
        assert!(output.iter().all(|&v| v == -1234.));
        assert_eq!(resident.len(), 13);
        resident.reset();
        assert!(resident.is_empty());
        assert_eq!(resident.staged_len(), None);
        let expected = input.oracle(geometry, &[], &[]);
        input.stage(resident.as_mut(), &mut output).unwrap();
        compare(&output, &expected);
        comparisons += 1;
        resident.commit();
        resident.reset();
    }
    assert_eq!(comparisons, 91);
    eprintln!("physical host-oracle comparisons={comparisons}, geometries=7");
}

#[test]
fn malformed_budget_capacity_nonfinite_and_arithmetic_failure_preserve_prefix_and_output() {
    let Some(backend) = cuda() else {
        return;
    };
    let geometry = AttentionGeometry {
        query_heads: 6,
        kv_heads: 2,
        head_dim: 7,
    };
    let admitted = limits(geometry);
    assert!(matches!(
        backend.resident_attention(
            geometry,
            AttentionLimits {
                max_device_payload_bytes: admitted.max_device_payload_bytes - 1,
                ..admitted
            }
        ),
        Err(BackendError::TensorPayloadBudgetExceeded { .. })
    ));
    let unsupported = AttentionGeometry {
        head_dim: 1025,
        ..geometry
    };
    assert!(
        backend
            .resident_attention(unsupported, limits(unsupported))
            .unwrap()
            .is_none()
    );
    assert!(
        tritium_cpu::CpuBackend::new()
            .resident_attention(geometry, admitted)
            .unwrap()
            .is_none()
    );
    let mut resident = backend
        .resident_attention(geometry, admitted)
        .unwrap()
        .unwrap();
    let prefix = Inputs::new(geometry, 3, 1);
    let mut output = vec![0.; prefix.query.len()];
    prefix.stage(resident.as_mut(), &mut output).unwrap();
    resident.commit();
    for case in 0..9 {
        let mut input = Inputs::new(geometry, 1, 5);
        match case {
            0 => {
                input.query.pop();
            }
            1 => {
                input.value.pop();
            }
            2 => {
                input.query[0] = f32::NAN;
            }
            3 => {
                input.key[0] = f32::INFINITY;
            }
            4 => {
                input.value[0] = f32::NEG_INFINITY;
            }
            5 => {
                input.scale = f32::NAN;
            }
            6 => {
                input = Inputs::new(geometry, 6, 5);
            }
            7 => {
                input = Inputs::new(geometry, 0, 5);
            }
            8 => {
                input.query.fill(f32::MAX);
                input.key.fill(f32::MAX);
            }
            _ => unreachable!(),
        }
        let valid = Inputs::new(geometry, 1, 5);
        let mut pending_output = vec![0.; valid.query.len()];
        valid.stage(resident.as_mut(), &mut pending_output).unwrap();
        let mut output = vec![-1234.; input.query.len()];
        assert!(
            input.stage(resident.as_mut(), &mut output).is_err(),
            "case={case}"
        );
        assert!(output.iter().all(|&v| v == -1234.), "case={case}");
        assert_eq!(resident.len(), 3);
        assert_eq!(resident.staged_len(), None);
        resident.commit();
        assert_eq!(resident.len(), 3);
        let expected = valid.oracle(geometry, &prefix.key, &prefix.value);
        valid.stage(resident.as_mut(), &mut pending_output).unwrap();
        compare(&pending_output, &expected);
        resident.abort();
    }
    let valid = Inputs::new(geometry, 1, 5);
    let mut short_output = vec![-1234.; valid.query.len() - 1];
    assert!(valid.stage(resident.as_mut(), &mut short_output).is_err());
    assert!(short_output.iter().all(|&v| v == -1234.));
    assert_eq!(resident.len(), 3);
}

#[test]
fn online_softmax_long_context_zero_negative_and_peaked_scale_match_host() {
    let Some(backend) = cuda() else {
        return;
    };
    let geometry = AttentionGeometry {
        query_heads: 6,
        kv_heads: 2,
        head_dim: 7,
    };
    let mut comparisons = 0;
    for context in [1, 33, 257] {
        for scale in [0., -0.37, 4.] {
            let mut limits = AttentionLimits {
                max_context: context,
                max_segment_tokens: context.min(17),
                max_device_payload_bytes: usize::MAX,
            };
            limits.max_device_payload_bytes = geometry.payload_bytes(limits).unwrap();
            let mut resident = backend
                .resident_attention(geometry, limits)
                .unwrap()
                .unwrap();
            let mut prefix_key = Vec::new();
            let mut prefix_value = Vec::new();
            while resident.len() < context {
                let tokens = (context - resident.len()).min(limits.max_segment_tokens);
                let mut input = Inputs::new(geometry, tokens, resident.len() + 3);
                input.scale = scale;
                let expected = input.oracle(geometry, &prefix_key, &prefix_value);
                let mut output = vec![f32::NAN; expected.len()];
                input.stage(resident.as_mut(), &mut output).unwrap();
                compare(&output, &expected);
                comparisons += 1;
                resident.commit();
                prefix_key.extend_from_slice(&input.key);
                prefix_value.extend_from_slice(&input.value);
            }
            assert_eq!(resident.len(), context);
        }
    }
    assert_eq!(comparisons, 57);
    eprintln!("physical online-softmax comparisons={comparisons}, max_context=257");
}

//! Physical opt-in region checks, not full-model quality or speed qualification.
#![cfg(feature = "cuda")]

use tritium_core::{PerTilePlan, Trit};
use tritium_cuda::CudaBackend;
use tritium_evidence::{EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION};
use tritium_format::AdditiveTensor;
use tritium_nn::{Projection, SwiGluMlp, TensorLoader};
use tritium_runtime::{TensorPayloadBudget, TensorUploadContext};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
};
use tritium_spec::{DeviceBuffer, TensorSwiGlu, TensorUploadPolicy, TensorView, TernaryBackend};

enum Weight {
    Dense {
        rows: usize,
        cols: usize,
        values: Vec<f32>,
    },
    Additive(Box<AdditiveTensor>),
}

impl Weight {
    fn view(&self) -> TensorView<'_> {
        match self {
            Self::Dense { rows, cols, values } => TensorView::Dense {
                rows: *rows,
                cols: *cols,
                values,
            },
            Self::Additive(tensor) => TensorView::Additive(tensor.view()),
        }
    }
}

fn dense(rows: usize, cols: usize, seed: usize) -> Weight {
    Weight::Dense {
        rows,
        cols,
        values: (0..rows * cols)
            .map(|i| ((i + seed) % 19) as f32 / 64. - 0.125)
            .collect(),
    }
}

#[test]
#[ignore = "explicit physical CUDA nonfinite shared-region regression; no silent skip"]
fn physical_shared_region_refuses_nan_without_publishing_output() {
    let backend = CudaBackend::new(0)
        .expect("physical CUDA required")
        .with_device_swiglu();
    let tensor = backend
        .upload_tensor(TensorView::Dense {
            rows: 1,
            cols: 1,
            values: &[1.],
        })
        .unwrap();
    let mut out = [19.];
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30] {
        let result = backend.try_tensor_swiglu(TensorSwiGlu {
            act: &[value],
            gate: &*tensor,
            up: &*tensor,
            down: &*tensor,
            batch: 1,
            out: &mut out,
        });
        assert!(
            result.is_err(),
            "nonfinite region must refuse, got {result:?}; output_bits={:?}",
            out.map(f32::to_bits)
        );
        assert_eq!(out, [19.]);
    }
    assert!(
        backend
            .try_tensor_swiglu(TensorSwiGlu {
                act: &[1.],
                gate: &*tensor,
                up: &*tensor,
                down: &*tensor,
                batch: 1,
                out: &mut out
            })
            .unwrap()
    );
    assert!(out[0].is_finite() && out[0] > 0.7 && out[0] < 0.8);
    let diagnostic = format!("{backend:?}");
    assert!(diagnostic.contains("host_output_capacity_elements"));
    assert!(
        !diagnostic.contains("host_output: ["),
        "backend diagnostics must not dump private activations"
    );
    out.fill(19.);
    let ordinary = CudaBackend::new(0).expect("physical CUDA required");
    assert!(
        !ordinary
            .try_tensor_swiglu(TensorSwiGlu {
                act: &[f32::NAN],
                gate: &*tensor,
                up: &*tensor,
                down: &*tensor,
                batch: 1,
                out: &mut out
            })
            .unwrap()
    );
    assert_eq!(out, [19.]);
    let mlp = mlp(&backend, &[dense(1, 1, 1), dense(1, 1, 2), dense(1, 1, 3)]);
    assert!(mlp.forward(&backend, &[f32::NAN], 1, &mut out).is_err());
    assert_eq!(out, [19.]);
}

#[test]
#[ignore = "explicit physical CUDA intermediate overflow masked by a pruned down projection"]
fn physical_shared_region_refuses_overflow_even_when_down_is_pruned() {
    let backend = CudaBackend::new(0)
        .expect("physical CUDA required")
        .with_device_swiglu();
    let values = vec![1.; 32 * 32];
    let gate = backend
        .upload_tensor(TensorView::Dense {
            rows: 32,
            cols: 32,
            values: &values,
        })
        .unwrap();
    let pruned = additive(
        AdditiveLayout {
            rows: 32,
            cols: 32,
            tile: 256,
            group: 32,
            max_planes: 2,
            allocation: PlaneAllocation::PerTile,
            codec: PlaneCodec::B3,
            law: ADMITTED_LAWS[0].law,
            basis: Basis::Identity,
            transport: Transport::Raw,
        },
        true,
        true,
    );
    let down = backend.upload_tensor(pruned.view()).unwrap();
    let mut output = [19.; 32];
    let result = backend.try_tensor_swiglu(TensorSwiGlu {
        act: &[1e20; 32],
        gate: &*gate,
        up: &*gate,
        down: &*down,
        batch: 1,
        out: &mut output,
    });
    assert!(
        result.is_err(),
        "overflow must refuse even if pruned down masks it: {result:?}, finite_output={}",
        output.iter().all(|x| x.is_finite())
    );
    assert_eq!(output, [19.; 32]);
    assert!(
        backend
            .try_tensor_swiglu(TensorSwiGlu {
                act: &[0.; 32],
                gate: &*gate,
                up: &*gate,
                down: &*down,
                batch: 1,
                out: &mut output
            })
            .unwrap()
    );
    assert_eq!(output, [0.; 32], "valid retry must reset the error flag");
}

#[test]
#[ignore = "explicit physical CUDA transformed overflow cannot be masked by pruned matrices"]
fn physical_shared_region_refuses_masked_basis_overflow_and_recovers() {
    let backend = CudaBackend::new(0)
        .expect("physical CUDA required")
        .with_device_swiglu();
    let layout = AdditiveLayout {
        rows: 4,
        cols: 4,
        tile: 256,
        group: 32,
        max_planes: 2,
        allocation: PlaneAllocation::PerTile,
        codec: PlaneCodec::B3,
        law: ADMITTED_LAWS[0].law,
        basis: Basis::Hadamard { block: 4 },
        transport: Transport::Raw,
    };
    let pruned = additive(layout, true, true);
    for down_basis in [false, true] {
        let values = vec![if down_basis { 2.5e18 } else { 1. }; 16];
        let plain = backend
            .upload_tensor(TensorView::Dense {
                rows: 4,
                cols: 4,
                values: &values,
            })
            .unwrap();
        let rotated = backend.upload_tensor(pruned.view()).unwrap();
        let (gate, down) = if down_basis {
            (&*plain, &*rotated)
        } else {
            (&*rotated, &*plain)
        };
        let input = [if down_basis { 1. } else { f32::MAX }; 4];
        let mut output = [19.; 4];
        assert!(
            backend
                .try_tensor_swiglu(TensorSwiGlu {
                    act: &input,
                    gate,
                    up: gate,
                    down,
                    batch: 1,
                    out: &mut output
                })
                .is_err()
        );
        assert_eq!(output, [19.; 4]);
        assert!(
            backend
                .try_tensor_swiglu(TensorSwiGlu {
                    act: &[0.; 4],
                    gate,
                    up: gate,
                    down,
                    batch: 1,
                    out: &mut output
                })
                .unwrap()
        );
        assert_eq!(output, [0.; 4]);
    }
}

#[test]
#[ignore = "explicit physical CUDA output-only overflow refuses without poisoning a retry"]
fn physical_shared_region_refuses_nonfinite_final_output_and_recovers() {
    let backend = CudaBackend::new(0)
        .expect("physical CUDA required")
        .with_device_swiglu();
    let gate = backend
        .upload_tensor(TensorView::Dense {
            rows: 1,
            cols: 1,
            values: &[1.],
        })
        .unwrap();
    let down = backend
        .upload_tensor(TensorView::Dense {
            rows: 1,
            cols: 1,
            values: &[f32::MAX],
        })
        .unwrap();
    let mut output = [19.];
    assert!(
        backend
            .try_tensor_swiglu(TensorSwiGlu {
                act: &[2.],
                gate: &*gate,
                up: &*gate,
                down: &*down,
                batch: 1,
                out: &mut output
            })
            .is_err()
    );
    assert_eq!(output, [19.]);
    assert!(
        backend
            .try_tensor_swiglu(TensorSwiGlu {
                act: &[0.],
                gate: &*gate,
                up: &*gate,
                down: &*down,
                batch: 1,
                out: &mut output
            })
            .unwrap()
    );
    assert_eq!(output, [0.]);
}

fn additive(mut layout: AdditiveLayout, compact: bool, pruned: bool) -> Weight {
    let planes = layout.max_planes;
    layout.allocation = PlaneAllocation::PerTile;
    let counts: Vec<_> = (0..layout.rows as usize * (layout.cols as usize).div_ceil(256))
        .map(|i| {
            if pruned {
                0
            } else if compact {
                (i % (planes as usize + 1)) as u8
            } else {
                planes
            }
        })
        .collect();
    let plan = PerTilePlan::new(layout, &counts).unwrap();
    let trits = [Trit::POS, Trit::NEG, Trit::ZERO, Trit::POS]
        .into_iter()
        .cycle()
        .take(plan.trit_count())
        .collect();
    let scales = [0.03125, 0.0625, 0.125]
        .into_iter()
        .cycle()
        .take(plan.scale_count())
        .collect();
    Weight::Additive(Box::new(if compact {
        AdditiveTensor::new_per_tile(layout, counts, trits, scales).unwrap()
    } else {
        layout.allocation = PlaneAllocation::Uniform;
        AdditiveTensor::new(layout, planes, trits, scales).unwrap()
    }))
}

fn mlp(backend: &dyn TernaryBackend, sources: &[Weight; 3]) -> SwiGluMlp {
    let budget = TensorPayloadBudget::new(u64::MAX);
    let mut recorder =
        EvidenceRecorder::new("swiglu-physical", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let mut loader = TensorLoader::new(
        backend,
        TensorUploadPolicy {
            native_only: true,
            max_payload_bytes: None,
        },
        &budget,
        TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    );
    let mut upload = |i: usize| {
        Projection::Tensor(
            loader
                .upload(["gate", "up", "down"][i], sources[i].view())
                .unwrap(),
        )
    };
    SwiGluMlp::new(upload(0), upload(1), upload(2)).unwrap()
}

fn compare(host: &CudaBackend, device: &CudaBackend, sources: [Weight; 3], k: usize) -> usize {
    let gate_layout = match sources[0].view() {
        TensorView::Additive(tensor) => Some(tensor.layout()),
        _ => None,
    };
    let expected_layer = mlp(host, &sources);
    let layer = mlp(device, &sources);
    drop(sources); // Actual admitted handles, not retained host-source fixtures.
    for batch in [0, 1, 3, 1] {
        let input: Vec<_> = (0..batch * k)
            .map(|i| ((i % 23) as f32 - 11.) / 16.)
            .collect();
        let mut expected = vec![99.; input.len()];
        let mut actual = expected.clone();
        expected_layer
            .forward(host, &input, batch, &mut expected)
            .unwrap();
        layer.forward(device, &input, batch, &mut actual).unwrap();
        for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                a.is_finite() && b.is_finite(),
                "nonfinite lane {i}, batch {batch}"
            );
            assert!(
                (a - b).abs() <= 1e-4 + 1e-4 * b.abs(),
                "lane {i}, batch {batch}: {a} vs {b}"
            );
        }
        // A cancellation residual below the absolute tolerance need not remain
        // nonzero under the explicitly different device exponential.
        if batch > 0 && expected.iter().any(|v| v.abs() > 1e-4) {
            assert!(
                actual.iter().any(|v| *v != 0.),
                "nonzero reference became zero: k={k}, batch={batch}, gate={gate_layout:?}, expected_max={}, first_expected={:?}",
                expected
                    .iter()
                    .fold(0_f32, |max, value| max.max(value.abs())),
                &expected[..expected.len().min(8)]
            );
        }
    }
    4
}

#[test]
fn actual_typed_nn_swiglu_matches_unfused_cuda_on_dense_packed_rotated_and_compact_weights() {
    let host = CudaBackend::new(0).expect("SwiGLU physical parity requires CUDA");
    let device = CudaBackend::new(0)
        .expect("SwiGLU physical parity requires CUDA")
        .with_device_swiglu();
    assert!(!host.device_id().contains(":swiglu-device-f32"));
    assert!(device.device_id().ends_with(":swiglu-device-f32"));
    let mut forwards = 0;
    for (k, f) in [(4, 8), (132, 260)] {
        forwards += compare(
            &host,
            &device,
            [dense(f, k, 1), dense(f, k, 5), dense(k, f, 9)],
            k,
        );
        for admitted in ADMITTED_LAWS {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                for basis_case in 0..3 {
                    for compact in [false, true] {
                        let source = |rows, cols, index| {
                            let basis = match (basis_case, index) {
                                (0, _) => Basis::Identity,
                                (1, _) => Basis::Hadamard { block: 4 },
                                (2, 1) => Basis::Hadamard { block: 2 },
                                (2, _) => Basis::SignedRht {
                                    block: 4,
                                    seed: u64::MAX,
                                    domain: index as u64,
                                },
                                _ => unreachable!(),
                            };
                            additive(
                                AdditiveLayout {
                                    rows,
                                    cols,
                                    tile: 256,
                                    group: 32,
                                    max_planes: admitted.max_planes,
                                    allocation: PlaneAllocation::Uniform,
                                    codec,
                                    law: admitted.law,
                                    basis,
                                    transport: Transport::Raw,
                                },
                                compact,
                                false,
                            )
                        };
                        let sources = [
                            source(f as u64, k as u64, 0),
                            source(f as u64, k as u64, 1),
                            source(k as u64, f as u64, 2),
                        ];
                        forwards += compare(&host, &device, sources, k);
                    }
                }
            }
        }
        let layout = AdditiveLayout {
            rows: f as u64,
            cols: k as u64,
            tile: 256,
            group: 32,
            max_planes: 3,
            allocation: PlaneAllocation::PerTile,
            codec: PlaneCodec::B3,
            law: ADMITTED_LAWS[0].law,
            basis: Basis::SignedRht {
                block: 4,
                seed: 7,
                domain: 9,
            },
            transport: Transport::Raw,
        };
        forwards += compare(
            &host,
            &device,
            [
                additive(layout, true, false),
                dense(f, k, 2),
                dense(k, f, 7),
            ],
            k,
        );
        forwards += compare(
            &host,
            &device,
            [additive(layout, true, true), dense(f, k, 2), dense(k, f, 7)],
            k,
        );
        let down = AdditiveLayout {
            rows: k as u64,
            cols: f as u64,
            ..layout
        };
        forwards += compare(
            &host,
            &device,
            [dense(f, k, 2), dense(f, k, 7), additive(down, true, true)],
            k,
        );
    }
    assert_eq!(forwards, 608);
    eprintln!(
        "actual NN SwiGLU physical forward comparisons={forwards}, fixture tolerance abs=1e-4 rel=1e-4; not model qualification"
    );
}

#[test]
fn near_cancelling_rotated_region() {
    let host = CudaBackend::new(0).expect("cancellation probe requires CUDA");
    let device = CudaBackend::new(0)
        .expect("cancellation probe requires CUDA")
        .with_device_swiglu();
    let base = AdditiveLayout {
        rows: 8,
        cols: 4,
        tile: 256,
        group: 32,
        max_planes: 3,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law: ADMITTED_LAWS[0].law,
        basis: Basis::SignedRht {
            block: 4,
            seed: u64::MAX,
            domain: 0,
        },
        transport: Transport::Raw,
    };
    let mut sources = [
        additive(base, false, false),
        additive(
            AdditiveLayout {
                basis: Basis::Hadamard { block: 2 },
                ..base
            },
            false,
            false,
        ),
        additive(
            AdditiveLayout {
                rows: 4,
                cols: 8,
                basis: Basis::SignedRht {
                    block: 4,
                    seed: u64::MAX,
                    domain: 2,
                },
                ..base
            },
            false,
            false,
        ),
    ];
    let expected_layer = mlp(&host, &sources);
    let layer = mlp(&device, &sources);
    let input = [-7. / 16., -6. / 16., -5. / 16., -4. / 16.];
    let mut expected = [0.; 4];
    let mut actual = [0.; 4];
    expected_layer
        .forward(&host, &input, 1, &mut expected)
        .unwrap();
    layer.forward(&device, &input, 1, &mut actual).unwrap();
    // Change only down to expose four hidden lanes through the production MLP.
    sources[2] = Weight::Dense {
        rows: 4,
        cols: 8,
        values: (0..32)
            .map(|i| if i / 8 == i % 8 { 1. } else { 0. })
            .collect(),
    };
    let mut hidden_host = [0.; 4];
    let mut hidden_device = [0.; 4];
    mlp(&host, &sources)
        .forward(&host, &input, 1, &mut hidden_host)
        .unwrap();
    mlp(&device, &sources)
        .forward(&device, &input, 1, &mut hidden_device)
        .unwrap();
    assert!(hidden_host.iter().any(|v| v.abs() > 1e-6));
    for (&a, &b) in hidden_device.iter().zip(&hidden_host) {
        assert!((a - b).abs() < 1e-7);
    }
    // On this machine the host residual is 2.0372681e-10 and device is zero.
    // This remains an explicitly bounded near-zero case, not exact parity.
    for (&a, &b) in actual.iter().zip(&expected) {
        assert!(a.is_finite() && b.is_finite());
        assert!(a.abs() < 1e-8 && b.abs() < 1e-8);
        assert!((a - b).abs() < 1e-8);
    }
}

#[test]
fn physical_region_execution_default_decline_and_failures_are_observable() {
    let ordinary = CudaBackend::new(0).expect("SwiGLU ownership test requires CUDA");
    let device = CudaBackend::new(0)
        .expect("SwiGLU ownership test requires CUDA")
        .with_device_swiglu()
        .with_device_swiglu();
    assert_eq!(device.device_id().matches(":swiglu-device-f32").count(), 1);
    let gate = device.upload_tensor(dense(8, 4, 1).view()).unwrap();
    let up = device.upload_tensor(dense(8, 4, 2).view()).unwrap();
    let down = device.upload_tensor(dense(4, 8, 3).view()).unwrap();
    let wrong = ordinary.upload_tensor(dense(8, 4, 4).view()).unwrap();
    let malformed = device.upload_tensor(dense(7, 4, 4).view()).unwrap();
    let act = [0.5, -0.25, 1., -2.];
    let mut out = [19.; 4];
    fn request<'a>(
        act: &'a [f32],
        gate: &'a dyn DeviceBuffer,
        up: &'a dyn DeviceBuffer,
        down: &'a dyn DeviceBuffer,
        out: &'a mut [f32],
    ) -> TensorSwiGlu<'a> {
        TensorSwiGlu {
            act,
            gate,
            up,
            down,
            batch: 1,
            out,
        }
    }
    assert!(
        !ordinary
            .try_tensor_swiglu(request(&act, &*gate, &*up, &*down, &mut out))
            .unwrap()
    );
    assert_eq!(out, [19.; 4]);
    for invalid_gate in [&*wrong, &*malformed] {
        assert!(
            device
                .try_tensor_swiglu(request(&act, invalid_gate, &*up, &*down, &mut out))
                .is_err()
        );
        assert_eq!(out, [19.; 4]);
    }
    for batch in [2, usize::MAX] {
        assert!(
            device
                .try_tensor_swiglu(TensorSwiGlu {
                    act: &act,
                    gate: &*gate,
                    up: &*up,
                    down: &*down,
                    batch,
                    out: &mut out
                })
                .is_err()
        );
        assert_eq!(out, [19.; 4]);
    }
    assert!(
        device
            .try_tensor_swiglu(request(&act, &*gate, &*up, &*down, &mut out[..3]))
            .is_err()
    );
    assert_eq!(out, [19.; 4]);
    assert!(
        device
            .try_tensor_swiglu(request(&act, &*gate, &*up, &*down, &mut out))
            .unwrap(),
        "probe must execute, not compare two host fallbacks"
    );
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(out.iter().any(|v| *v != 19. && *v != 0.));
    assert!(
        device
            .try_tensor_swiglu(TensorSwiGlu {
                act: &[],
                gate: &*gate,
                up: &*up,
                down: &*down,
                batch: 0,
                out: &mut []
            })
            .unwrap()
    );
}

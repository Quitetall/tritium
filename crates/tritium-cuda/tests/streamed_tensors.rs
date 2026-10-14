//! Physical encoded-source execution; missing GPU is a hard failure, not a skip.
#![cfg(feature = "cuda")]
use tritium_cuda::CudaBackend;
use tritium_evidence::{EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION};
use tritium_runtime::{
    TensorPayloadBudget, TensorUploadContext, TensorUploadResources,
    upload_tensor_stream_logged_with_resources,
};
use tritium_spec::*;
struct Source<'a> {
    metadata: TensorMetadata<'a>,
    encoding: TensorStreamEncoding,
    bytes: &'a [u8],
    fail: bool,
}

#[test]
fn physical_bounded_stream_crosses_payload_and_native_suffix_windows() {
    use tritium_core::{PerTilePlan, Trit};
    use tritium_format::AdditiveTensor;
    use tritium_schema::{ADMITTED_LAWS, AdditiveLayout, PlaneAllocation, PlaneCodec, Transport};
    let cuda = CudaBackend::new(0).expect("bounded stream test requires physical CUDA");
    let layout = AdditiveLayout {
        rows: 10_000,
        cols: 32,
        tile: 256,
        group: 32,
        max_planes: 2,
        allocation: PlaneAllocation::PerTile,
        codec: PlaneCodec::B3,
        law: ADMITTED_LAWS[0].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let counts = vec![2; layout.rows as usize];
    let plan = PerTilePlan::new(layout, &counts).unwrap();
    let trits = (0..plan.trit_count())
        .map(|i| [Trit::ZERO, Trit::NEG, Trit::POS, Trit::NEG][i % 4])
        .collect();
    let scales = vec![0.5; plan.scale_count()];
    let tensor = AdditiveTensor::new_per_tile(layout, counts, trits, scales).unwrap();
    let raw = tensor.encode_payload().unwrap();
    assert!(raw.len() > 65536);
    let source = Source {
        metadata: TensorMetadata::Additive(AdditiveMetadata::from_view(tensor.view())),
        encoding: TensorStreamEncoding::CanonicalRawAdditive,
        bytes: &raw,
        fail: false,
    };
    let stream_plan = cuda
        .tensor_stream_plan(source.metadata(), source.encoding())
        .unwrap()
        .unwrap();
    assert_eq!(stream_plan.upload.host_workspace_bytes, 131072);
    assert_eq!(stream_plan.upload.device_workspace_bytes, 4);
    assert!(stream_plan.upload.caps.payload_bytes > stream_plan.upload.host_workspace_bytes);
    let device = cuda.upload_tensor_stream(&source).unwrap();
    let ids = [0, 4999, 9999];
    let mut actual = vec![0.; ids.len() * 32];
    cuda.embed_rows(&*device, &ids, &mut actual).unwrap();
    let mut expected = vec![0.; actual.len()];
    for (i, &row) in ids.iter().enumerate() {
        tensor
            .view()
            .dequant_row_into(row, &mut expected[i * 32..(i + 1) * 32])
            .unwrap();
    }
    assert_eq!(actual, expected);
    for precision in [
        tritium_schema::ScalePrecision::F16,
        tritium_schema::ScalePrecision::Bf16,
        tritium_schema::ScalePrecision::F32,
    ] {
        let values = (0..1024 * 128)
            .map(|i| ((i % 9) as f32 - 4.) / 8.)
            .collect::<Vec<_>>();
        let raw = values
            .iter()
            .flat_map(|&v| match precision {
                tritium_schema::ScalePrecision::F16 => {
                    half::f16::from_f32(v).to_bits().to_le_bytes().to_vec()
                }
                tritium_schema::ScalePrecision::Bf16 => {
                    half::bf16::from_f32(v).to_bits().to_le_bytes().to_vec()
                }
                _ => v.to_le_bytes().to_vec(),
            })
            .collect::<Vec<_>>();
        let source = Source {
            metadata: TensorMetadata::Dense {
                rows: 1024,
                cols: 128,
            },
            encoding: TensorStreamEncoding::DenseLittleEndian(precision),
            bytes: &raw,
            fail: false,
        };
        let plan = cuda
            .tensor_stream_plan(source.metadata(), source.encoding())
            .unwrap()
            .unwrap();
        assert_eq!(plan.upload.host_workspace_bytes, 196608);
        assert!(plan.upload.caps.payload_bytes > plan.upload.host_workspace_bytes);
        let device = cuda.upload_tensor_stream(&source).unwrap();
        let ids = [0, 511, 1023];
        let mut actual = vec![0.; 384];
        cuda.embed_rows(&*device, &ids, &mut actual).unwrap();
        let expected = ids
            .into_iter()
            .flat_map(|r| values[r * 128..(r + 1) * 128].iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}

#[test]
fn physical_private_device_check_accepts_finite_cancellation_and_refuses_overflow() {
    use tritium_core::Trit;
    use tritium_format::AdditiveTensor;
    use tritium_schema::{
        AdditiveLayout, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw,
        ScalePrecision, Transport,
    };
    let cuda = CudaBackend::new(0).expect("finite stream test requires physical CUDA");
    for compact in [false, true] {
        for cancellation in [false, true] {
            let layout = AdditiveLayout {
                rows: 2,
                cols: 4,
                tile: 256,
                group: 32,
                max_planes: 3,
                allocation: if compact {
                    PlaneAllocation::PerTile
                } else {
                    PlaneAllocation::Uniform
                },
                codec: PlaneCodec::D2,
                law: ScaleLaw {
                    anchor: ScaleAnchor::Group,
                    relation: PlaneRelation::Free,
                    precision: ScalePrecision::F32,
                },
                basis: Basis::Identity,
                transport: Transport::Raw,
            };
            let trits = if cancellation {
                [Trit::POS; 8]
                    .into_iter()
                    .chain([Trit::NEG; 8])
                    .chain([Trit::ZERO; 8])
                    .collect()
            } else {
                vec![Trit::POS; 24]
            };
            let tensor = if compact {
                AdditiveTensor::new_per_tile(layout, vec![3, 3], trits, vec![f32::MAX; 6]).unwrap()
            } else {
                AdditiveTensor::new(layout, 3, trits, vec![f32::MAX; 6]).unwrap()
            };
            let raw = tensor.encode_payload().unwrap();
            let source = Source {
                metadata: TensorMetadata::Additive(AdditiveMetadata::from_view(tensor.view())),
                encoding: TensorStreamEncoding::CanonicalRawAdditive,
                bytes: &raw,
                fail: false,
            };
            let result = cuda.upload_tensor_stream(&source);
            if cancellation {
                let device = result.unwrap();
                let mut actual = [1.; 8];
                cuda.embed_rows(&*device, &[0, 1], &mut actual).unwrap();
                assert_eq!(actual, [0.; 8]);
            } else {
                assert!(result.is_err());
            }
        }
    }
}

#[test]
fn physical_direct_stream_refuses_swallowed_visitor_failure() {
    struct Swallow;
    impl TensorStreamSource for Swallow {
        fn metadata(&self) -> TensorMetadata<'_> {
            TensorMetadata::Dense { rows: 1, cols: 1 }
        }
        fn encoding(&self) -> TensorStreamEncoding {
            TensorStreamEncoding::DenseLittleEndian(tritium_schema::ScalePrecision::F32)
        }
        fn encoded_bytes(&self) -> u64 {
            4
        }
        fn visit(
            &self,
            visitor: &mut dyn FnMut(&[u8]) -> Result<(), BackendError>,
        ) -> Result<(), BackendError> {
            let _ = visitor(&vec![0; 65537]);
            let _ = visitor(&1.0f32.to_le_bytes());
            Ok(())
        }
    }
    let cuda = CudaBackend::new(0).expect("poisoned stream test requires physical CUDA");
    assert!(cuda.upload_tensor_stream(&Swallow).is_err());
}
impl TensorStreamSource for Source<'_> {
    fn metadata(&self) -> TensorMetadata<'_> {
        self.metadata
    }
    fn encoding(&self) -> TensorStreamEncoding {
        self.encoding
    }
    fn encoded_bytes(&self) -> u64 {
        self.bytes.len() as u64
    }
    fn visit(
        &self,
        visitor: &mut dyn FnMut(&[u8]) -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        for chunk in self.bytes.chunks(7) {
            visitor(chunk)?;
        }
        if self.fail {
            return Err(BackendError::Backend(
                "terminal fixture source fault".into(),
            ));
        }
        Ok(())
    }
}
struct StreamBackend<'a>(&'a CudaBackend);
impl TernaryBackend for StreamBackend<'_> {
    fn device_id(&self) -> &str {
        self.0.device_id()
    }
    fn physical_device_id(&self) -> &str {
        self.0.physical_device_id()
    }
    fn capabilities(&self) -> DeviceCaps {
        self.0.capabilities()
    }
    fn tensor_caps(&self, tensor: TensorView<'_>) -> Result<Option<TensorCaps>, BackendError> {
        self.0.tensor_caps(tensor)
    }
    fn upload_tensor(&self, tensor: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        // Only the small frozen test fixture owns decoded source arrays. The
        // production CUDA upload here crosses the mandatory encoded stream seam.
        let metadata = TensorMetadata::from_view(tensor)?;
        let (encoding, bytes) = match tensor {
            TensorView::Additive(view) => (
                TensorStreamEncoding::CanonicalRawAdditive,
                tritium_format::AdditiveTensor::from_view(view)
                    .and_then(|t| t.encode_payload())
                    .map_err(|e| BackendError::InvalidInput(format!("fixture: {e:?}")))?,
            ),
            TensorView::Dense { values, .. } => (
                TensorStreamEncoding::DenseLittleEndian(tritium_schema::ScalePrecision::F32),
                values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        };
        self.0.upload_tensor_stream(&Source {
            metadata,
            encoding,
            bytes: &bytes,
            fail: false,
        })
    }
    fn matmul(&self, p: TensorMatmul<'_>) -> Result<(), BackendError> {
        self.0.matmul(p)
    }
    fn embed_rows(
        &self,
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        self.0.embed_rows(tensor, ids, out)
    }
    fn upload_weights(
        &self,
        p: &[u8],
        s: GemmShape,
        f: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.0.upload_weights(p, s, f)
    }
    fn mpgemm(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
        self.0.mpgemm(p)
    }
}
#[test]
fn physical_encoded_streams_execute_the_unchanged_uniform_and_compact_reference_vectors() {
    let cuda = CudaBackend::new(0).expect("streamed tensor conformance requires physical CUDA");
    let backend = StreamBackend(&cuda);
    let uniform =
        tritium_testkit::assert_additive_tensor_conformance(&backend, TensorExecution::Native);
    let compact =
        tritium_testkit::assert_candidate_per_tile_conformance(&backend, TensorExecution::Native);
    assert_eq!(uniform, 180);
    assert!(compact > uniform);
    tritium_testkit::assert_additive_upload_rejections(&backend);
    eprintln!(
        "CUDA_STREAM_PHYSICAL device={} uniform={uniform} compact={compact}",
        cuda.physical_device_id()
    );
}
#[test]
fn physical_encoded_resources_and_source_faults_release_without_observation() {
    let cuda = CudaBackend::new(0).expect("streamed resource checks require physical CUDA");
    let tensor = tritium_testkit::additive_upload_fixture();
    let bytes = tensor.encode_payload().unwrap();
    let metadata = TensorMetadata::Additive(AdditiveMetadata::from_view(tensor.view()));
    let encoding = TensorStreamEncoding::CanonicalRawAdditive;
    let plan = cuda
        .tensor_stream_plan(metadata, encoding)
        .unwrap()
        .unwrap();
    for case in 0..5 {
        let payload =
            TensorPayloadBudget::new(plan.upload.caps.payload_bytes - u64::from(case == 0));
        let host =
            TensorPayloadBudget::new(plan.upload.host_workspace_bytes - u64::from(case == 1));
        let device =
            TensorPayloadBudget::new(plan.upload.device_workspace_bytes - u64::from(case == 4));
        let source = Source {
            metadata,
            encoding,
            bytes: &bytes,
            fail: case == 2,
        };
        let mut recorder =
            EvidenceRecorder::new("cuda-stream-guards", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
        let result = upload_tensor_stream_logged_with_resources(
            &cuda,
            &source,
            TensorUploadPolicy::default(),
            "fixture",
            TensorUploadContext {
                recorder: &mut recorder,
                span: "upload",
                parent: None,
                logical_time: 0,
            },
            TensorUploadResources {
                payload: &payload,
                host_workspace: &host,
                device_workspace: &device,
            },
        );
        if case == 3 {
            let (handle, caps) = result.unwrap();
            assert_eq!(caps, plan.upload.caps);
            assert_eq!(payload.used_bytes(), caps.payload_bytes);
            assert_eq!(recorder.events().len(), 1);
            drop(handle);
        } else {
            assert!(result.is_err());
            assert!(recorder.events().is_empty());
        }
        assert_eq!(payload.used_bytes(), 0);
        assert_eq!(host.used_bytes(), 0);
        assert_eq!(device.used_bytes(), 0);
    }
}

#[test]
fn physical_dense_stream_precision_splits_match_actual_gather_and_matmul() {
    let cuda = CudaBackend::new(0).expect("dense stream checks require physical CUDA");
    let values = [1., 2., 3., -1., 0.5, 2.];
    for precision in [
        tritium_schema::ScalePrecision::F16,
        tritium_schema::ScalePrecision::Bf16,
        tritium_schema::ScalePrecision::F32,
    ] {
        let bytes: Vec<_> = values
            .iter()
            .flat_map(|&v| match precision {
                tritium_schema::ScalePrecision::F16 => {
                    half::f16::from_f32(v).to_bits().to_le_bytes().to_vec()
                }
                tritium_schema::ScalePrecision::Bf16 => {
                    half::bf16::from_f32(v).to_bits().to_le_bytes().to_vec()
                }
                _ => v.to_le_bytes().to_vec(),
            })
            .collect();
        let source = Source {
            metadata: TensorMetadata::Dense { rows: 2, cols: 3 },
            encoding: TensorStreamEncoding::DenseLittleEndian(precision),
            bytes: &bytes,
            fail: false,
        };
        let handle = cuda.upload_tensor_stream(&source).unwrap();
        let mut rows = [0.; 6];
        cuda.embed_rows(&*handle, &[0, 1], &mut rows).unwrap();
        assert_eq!(rows, values);
        let mut out = [0.; 4];
        let mut scratch = [0.; 6];
        cuda.matmul(TensorMatmul {
            act: &[2., -1., 4., 1., 2., 3.],
            tensor: &*handle,
            batch: 2,
            transformed_act: &mut scratch,
            out: &mut out,
        })
        .unwrap();
        assert_eq!(out, [12., 5.5, 14., 6.]);
    }
}

//! Strict carrier checks, not pretrained model/source qualification.
use std::{collections::BTreeMap, io::Cursor};
use tritium_core::DType;
use tritium_format::{
    qtea::{QteaLimits, model::*},
    *,
};
use tritium_schema::{BlobId, ScalePrecision};
fn id(bytes: &[u8]) -> BlobId {
    BlobId::from_bytes(*blake3::hash(bytes).as_bytes())
}
fn limits() -> QteaModelReadLimits {
    QteaModelReadLimits {
        max_tensors: 8,
        max_assets: 8,
        max_asset_bytes: 1024,
        max_config_bytes: 1024,
        max_vector_elements: 8,
        tensor: QteaLimits {
            max_tensor_bytes: 4096,
            max_validation_bytes: 4096,
        },
    }
}
fn receiving() -> TritPackageReadLimits {
    TritPackageReadLimits {
        max_source_bytes: 1 << 20,
        max_manifest_bytes: 16384,
        max_directory_bytes: 16384,
    }
}
struct Fixture {
    manifest: QteaModelManifest,
    blobs: Vec<(TritBlobKind, Vec<u8>)>,
}
impl Fixture {
    fn new() -> Self {
        let hex = include_str!("fixtures/qtea_v1.hex")
            .split_whitespace()
            .collect::<String>();
        let tensor = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect::<Vec<_>>();
        let config = b"{}".to_vec();
        let token = b"[]".to_vec();
        let vector = [0x3c00_u16; 4]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        Self {
            manifest: QteaModelManifest {
                profile: QTEA_MODEL_PROFILE.into(),
                version: 1,
                source_model: [3; 32],
                config: id(&config),
                assets: BTreeMap::from([("tokenizer.json".into(), id(&token))]),
                tensors: BTreeMap::from([
                    (
                        "matrix".into(),
                        QteaModelTensor::Qtea {
                            blob: id(&tensor),
                            rows: 4,
                            columns: 2,
                        },
                    ),
                    (
                        "norm".into(),
                        QteaModelTensor::Vector {
                            blob: id(&vector),
                            length: 4,
                            precision: ScalePrecision::F16,
                        },
                    ),
                ]),
            },
            blobs: vec![
                (TritBlobKind::Asset("config.json".into()), config),
                (TritBlobKind::Asset("tokenizer.json".into()), token),
                (TritBlobKind::QteaTensor, tensor),
                (TritBlobKind::DenseTensor(DType::F16), vector),
            ],
        }
    }
    fn blob_refs(&self) -> Vec<TritBlob<'_>> {
        self.blobs
            .iter()
            .map(|(kind, bytes)| TritBlob {
                kind: kind.clone(),
                bytes,
            })
            .collect()
    }
    fn manifest(&self) -> Vec<u8> {
        encode_qtea_model_manifest(&self.manifest, 16384).unwrap()
    }
    fn bytes(&self) -> Vec<u8> {
        write_trit_package_bounded(&self.manifest(), &self.blob_refs(), 1 << 20).unwrap()
    }
}
fn open(
    bytes: Vec<u8>,
    caps: QteaModelReadLimits,
) -> Result<QteaModelReader<Cursor<Vec<u8>>>, QteaModelError> {
    QteaModelReader::open(Cursor::new(bytes), receiving(), caps, [3; 32])
}

#[test]
fn named_model_uses_existing_envelope_and_spool_without_relabelling() {
    let fixture = Fixture::new();
    let bytes = fixture.bytes();
    let outer = read_trit_package(&bytes).unwrap();
    assert!(
        outer
            .blobs()
            .iter()
            .any(|blob| blob.kind == TritBlobKind::QteaTensor)
    );
    assert!(tritium_format::trit_model::read_trit_model(&bytes).is_err());
    let mut spool = Vec::new();
    let mut blobs = Vec::new();
    for (kind, bytes) in &fixture.blobs {
        let offset = spool.len() as u64;
        spool.extend_from_slice(bytes);
        blobs.push(TritSpoolBlob {
            kind: kind.clone(),
            id: id(bytes),
            offset,
            length: bytes.len() as u64,
        });
    }
    let mut emitted = Vec::new();
    let written = write_trit_package_from_spool(
        &fixture.manifest(),
        &blobs,
        &mut Cursor::new(spool),
        1 << 20,
        &mut emitted,
    )
    .unwrap();
    assert_eq!(emitted, bytes);
    assert_eq!(written, outer.package_id());
    let mut reader = open(emitted, limits()).unwrap();
    assert_eq!(reader.package_id(), written);
    assert_eq!(reader.config().unwrap(), b"{}");
    assert_eq!(reader.asset("tokenizer.json").unwrap(), b"[]");
    assert_eq!(
        reader.qtea("matrix").unwrap().as_bytes(),
        fixture.blobs[2].1
    );
    assert_eq!(reader.vector("norm").unwrap(), vec![1.; 4]);
    assert!(reader.qtea("norm").is_err());
    assert!(reader.vector("matrix").is_err());
    reader.verify().unwrap();
}

#[test]
fn invalid_versions_missing_extra_and_wrong_kind_payloads_refuse() {
    for case in 0..6 {
        let mut f = Fixture::new();
        match case {
            0 => f.manifest.version = 2,
            1 => f.manifest.profile = "other".into(),
            2 => {
                f.manifest.config = BlobId::from_bytes([9; 32]);
            }
            3 => {
                f.blobs[2].0 = TritBlobKind::AdditiveTensor;
            }
            4 => {
                f.blobs
                    .push((TritBlobKind::Asset("unused".into()), vec![42]));
            }
            _ => {
                f.manifest
                    .assets
                    .insert("config.json".into(), f.manifest.config);
            }
        }
        if let Ok(manifest) = encode_qtea_model_manifest(&f.manifest, 16384) {
            let bytes = write_trit_package_bounded(&manifest, &f.blob_refs(), 1 << 20).unwrap();
            assert!(open(bytes, limits()).is_err());
        }
    }
}

#[test]
fn receiver_caps_and_actual_wire_source_geometry_are_separate() {
    let f = Fixture::new();
    for case in 0..6 {
        let mut caps = limits();
        match case {
            0 => caps.max_tensors = 1,
            1 => caps.max_assets = 0,
            2 => caps.max_asset_bytes = 1,
            3 => caps.max_config_bytes = 1,
            4 => caps.max_vector_elements = 3,
            _ => caps.tensor.max_tensor_bytes = 10,
        };
        assert!(open(f.bytes(), caps).is_err());
    }
    let mut f = Fixture::new();
    let QteaModelTensor::Qtea { rows, .. } = f.manifest.tensors.get_mut("matrix").unwrap() else {
        unreachable!()
    };
    *rows = 8;
    let mut reader = open(f.bytes(), limits()).unwrap();
    assert!(reader.qtea("matrix").is_err());
    f = Fixture::new();
    f.manifest.source_model = [7; 32];
    let mut reader =
        QteaModelReader::open(Cursor::new(f.bytes()), receiving(), limits(), [7; 32]).unwrap();
    assert!(reader.qtea("matrix").is_err());
    assert!(open(f.bytes(), limits()).is_err());
    let mut receiving = receiving();
    receiving.max_source_bytes = 1;
    assert!(QteaModelReader::open(Cursor::new(f.bytes()), receiving, limits(), [7; 32]).is_err());
}

#[test]
fn nonfinite_preserved_vectors_and_payload_mutation_refuse() {
    let mut f = Fixture::new();
    f.blobs[3].1[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    let blob = id(&f.blobs[3].1);
    let QteaModelTensor::Vector { blob: old, .. } = f.manifest.tensors.get_mut("norm").unwrap()
    else {
        unreachable!()
    };
    *old = blob;
    let mut reader = open(f.bytes(), limits()).unwrap();
    assert!(reader.vector("norm").is_err());
    let mut bytes = Fixture::new().bytes();
    let n = bytes.len();
    bytes[n - 1] ^= 1;
    assert!(open(bytes, limits()).is_err());
}

#[test]
fn unknown_and_duplicate_manifest_fields_are_not_accepted() {
    use ciborium::value::{CanonicalValue, Value};
    fn canonical(value: &mut Value) {
        match value {
            Value::Map(entries) => {
                for (k, v) in entries.iter_mut() {
                    canonical(k);
                    canonical(v);
                }
                entries.sort_by(|a, b| {
                    CanonicalValue::from(a.0.clone()).cmp(&CanonicalValue::from(b.0.clone()))
                });
            }
            Value::Array(items) => {
                for item in items {
                    canonical(item);
                }
            }
            _ => {}
        }
    }
    let f = Fixture::new();
    let mut value: Value = ciborium::de::from_reader(f.manifest().as_slice()).unwrap();
    let Value::Map(entries) = &mut value else {
        unreachable!()
    };
    entries.push((Value::Text("surprise".into()), Value::Null));
    canonical(&mut value);
    let mut manifest = Vec::new();
    ciborium::ser::into_writer(&value, &mut manifest).unwrap();
    let bytes = write_trit_package_bounded(&manifest, &f.blob_refs(), 1 << 20).unwrap();
    assert!(read_trit_package(&bytes).is_ok());
    assert!(open(bytes, limits()).is_err());
    let Value::Map(entries) = &mut value else {
        unreachable!()
    };
    entries.retain(|(key, _)| key != &Value::Text("surprise".into()));
    entries.push((Value::Text("version".into()), Value::Integer(1.into())));
    canonical(&mut value);
    let mut manifest = Vec::new();
    ciborium::ser::into_writer(&value, &mut manifest).unwrap();
    assert!(write_trit_package_bounded(&manifest, &f.blob_refs(), 1 << 20).is_err());
}

#[test]
fn preserved_precision_bits_are_widened_without_changing_matrix_roles() {
    for precision in [
        ScalePrecision::F16,
        ScalePrecision::Bf16,
        ScalePrecision::F32,
    ] {
        let mut f = Fixture::new();
        let expected = [1., -0.5, 0.25, 2.];
        let (dtype, bytes) = match precision {
            ScalePrecision::F16 => (
                DType::F16,
                expected
                    .iter()
                    .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            ScalePrecision::Bf16 => (
                DType::BF16,
                expected
                    .iter()
                    .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
                    .collect(),
            ),
            ScalePrecision::F32 => (
                DType::F32,
                expected.iter().flat_map(|v| f32::to_le_bytes(*v)).collect(),
            ),
            _ => unreachable!(),
        };
        let blob = id(&bytes);
        f.blobs[3] = (TritBlobKind::DenseTensor(dtype), bytes);
        f.manifest.tensors.insert(
            "norm".into(),
            QteaModelTensor::Vector {
                blob,
                length: 4,
                precision,
            },
        );
        let mut reader = open(f.bytes(), limits()).unwrap();
        assert_eq!(reader.vector("norm").unwrap(), expected);
        assert!(reader.qtea("norm").is_err());
    }
}

#[test]
fn post_admission_read_and_seek_failures_never_yield_tensor() {
    use std::{
        cell::Cell,
        io::{self, Read, Seek, SeekFrom},
        rc::Rc,
    };
    struct Fault {
        source: Cursor<Vec<u8>>,
        fail: Rc<Cell<bool>>,
        seek: bool,
    }
    impl Read for Fault {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            if self.fail.get() && !self.seek {
                return Err(io::Error::other("injected read"));
            }
            self.source.read(b)
        }
    }
    impl Seek for Fault {
        fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
            if self.fail.get() && self.seek {
                return Err(io::Error::other("injected seek"));
            }
            self.source.seek(p)
        }
    }
    for seek in [false, true] {
        let fail = Rc::new(Cell::new(false));
        let source = Fault {
            source: Cursor::new(Fixture::new().bytes()),
            fail: Rc::clone(&fail),
            seek,
        };
        let mut reader = QteaModelReader::open(source, receiving(), limits(), [3; 32]).unwrap();
        fail.set(true);
        assert!(reader.qtea("matrix").is_err());
        assert!(reader.verify().is_err());
    }
}

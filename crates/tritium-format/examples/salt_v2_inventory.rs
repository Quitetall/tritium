//! Read-only migration input inventory through the existing strict reader.
//! This diagnostic JSON is not a canonical receipt or release qualification.
//! Usage: salt_v2_inventory PATH EXPECTED_LEGACY_PACKAGE_ID
use std::error::Error;
use std::fs::File;
use std::io::{self, Read, Seek, Write};
use std::time::Instant;

use serde::Serialize;
use tritium_format::salt_v2_package::{SaltV2PackageReader, SaltV2Transform};

#[derive(Serialize)]
#[serde(tag = "kind")]
enum Transform {
    None,
    SignedRht {
        seed: u64,
        domain: u64,
        block: Option<u64>,
    },
}

impl TryFrom<SaltV2Transform> for Transform {
    type Error = io::Error;

    fn try_from(value: SaltV2Transform) -> Result<Self, Self::Error> {
        Ok(match value {
            SaltV2Transform::None => Self::None,
            SaltV2Transform::SignedRht { seed, domain } => Self::SignedRht {
                seed,
                domain,
                block: None,
            },
            _ => return Err(io::Error::other("unsupported source transform")),
        })
    }
}

#[derive(Serialize)]
struct Tensor<'a> {
    name: &'a str,
    dimensions: &'a [u64],
    logical_coefficients: usize,
    allocation_tiles: usize,
    present_planes: usize,
    plane_count_histogram: [u64; 4],
    source_scale_group: usize,
    transform: Transform,
    encoded_coefficient_bytes: u64,
    encoded_scale_bytes: u64,
    legacy_semantic_content_digest: [u8; 32],
}

#[derive(Serialize)]
struct Inventory<'a> {
    scope: &'static str,
    package_id: String,
    codec: String,
    source_verified_unchanged: bool,
    tensors: Vec<Tensor<'a>>,
}

fn validate_expected(value: &str) -> Result<(), io::Error> {
    if value.len() != 69
        || !value.starts_with("trp1_")
        || !value.as_bytes()[5..]
            .iter()
            .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f'))
    {
        return Err(io::Error::other(
            "expected exact lowercase legacy PackageId",
        ));
    }
    Ok(())
}

fn inventory<R: Read + Seek>(
    reader: &mut SaltV2PackageReader<R>,
    expected: &str,
    destination: &mut impl Write,
) -> Result<(), Box<dyn Error>> {
    validate_expected(expected)?;
    let package_id = reader.package_id().to_string();
    if package_id != expected {
        return Err(io::Error::other("legacy package identity mismatch").into());
    }
    // Metadata/count observations cannot authorize an output from changed bytes.
    // Do this once per whole package, not once per tensor.
    reader.verify_unchanged()?;
    let mut tensors = Vec::new();
    tensors.try_reserve_exact(reader.len())?;
    for name in reader.tensor_names() {
        let info = reader
            .tensor_info(name)
            .ok_or_else(|| io::Error::other("indexed tensor disappeared"))?;
        let mut histogram = [0u64; 4];
        for count in reader.tensor_plane_counts(name)? {
            let entry = histogram
                .get_mut(count)
                .ok_or_else(|| io::Error::other("invalid legacy plane count"))?;
            *entry = entry
                .checked_add(1)
                .ok_or_else(|| io::Error::other("plane count overflow"))?;
        }
        let semantic = reader
            .semantic_tensor(name)
            .ok_or_else(|| io::Error::other("indexed semantic tensor disappeared"))?;
        tensors.push(Tensor {
            name,
            dimensions: info.dims(),
            logical_coefficients: info.logical_coefficients(),
            allocation_tiles: info.tile_count(),
            present_planes: info.present_planes(),
            plane_count_histogram: histogram,
            source_scale_group: info.scale_group_size(),
            transform: info.transform().try_into()?,
            encoded_coefficient_bytes: info.encoded_payload_bytes(),
            encoded_scale_bytes: info.encoded_scale_bytes(),
            legacy_semantic_content_digest: *semantic.content_digest(),
        });
    }
    serde_json::to_writer(
        &mut *destination,
        &Inventory {
            scope: "strict-legacy-source-inventory-only; not migration or release clearance",
            package_id,
            codec: format!("{:?}", reader.codec()),
            source_verified_unchanged: true,
            tensors,
        },
    )?;
    destination.write_all(b"\n")?;
    Ok(())
}

fn inspect<R: Read + Seek>(
    source: R,
    expected: &str,
    destination: &mut impl Write,
) -> Result<(), Box<dyn Error>> {
    validate_expected(expected)?;
    let started = Instant::now();
    let mut reader = SaltV2PackageReader::new_strict(source)?;
    eprintln!(
        "Strict reader validated {} tensors in {:.3}s; terminal verification follows",
        reader.len(),
        started.elapsed().as_secs_f64()
    );
    inventory(&mut reader, expected, destination)?;
    eprintln!(
        "Strict inventory complete in {:.3}s; no model migrated or qualified",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let path = arguments
        .next()
        .ok_or_else(|| io::Error::other("expected package path"))?;
    let expected = arguments
        .next()
        .and_then(|s| s.into_string().ok())
        .ok_or_else(|| io::Error::other("expected legacy PackageId"))?;
    if arguments.next().is_some() {
        return Err(io::Error::other("expected exactly PATH PACKAGE_ID").into());
    }
    validate_expected(&expected)?;
    eprintln!("Strict legacy source validation starting: {path:?}");
    inspect(File::open(path)?, &expected, &mut io::stdout().lock())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use std::cell::RefCell;
    use std::io::{Cursor, SeekFrom};
    use std::rc::Rc;
    use tritium_format::salt_v2::SaltV2Codec;
    use tritium_format::salt_v2_package::{
        SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, write_salt_v2_package,
    };

    fn fixture(codec: SaltV2Codec) -> (Vec<u8>, String) {
        let tensor = SaltV2Tensor::new_with_transform(
            "weight",
            vec![2, 256],
            SaltV2Transform::SignedRht { seed: 7, domain: 9 },
            [1, 3]
                .into_iter()
                .map(|count| {
                    SaltV2Tile::new(
                        (0..count)
                            .map(|_| {
                                SaltV2Plane::new(
                                    (0..256).map(|i| [0, 1, -1, 1][i % 4]).collect(),
                                    vec![f16::ONE; 2],
                                )
                                .unwrap()
                            })
                            .collect(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        let unused = SaltV2Tensor::new(
            "zz_unselected",
            vec![256],
            vec![
                SaltV2Tile::new(vec![
                    SaltV2Plane::new(
                        (0..256).map(|i| [0, -1, 1, -1][i % 4]).collect(),
                        vec![f16::ONE; 2],
                    )
                    .unwrap(),
                ])
                .unwrap(),
            ],
        )
        .unwrap();
        let encoded =
            write_salt_v2_package(&SaltV2Package::new(codec, vec![tensor, unused]).unwrap())
                .unwrap();
        let expected = {
            let reader = SaltV2PackageReader::new_strict(Cursor::new(&encoded.bytes)).unwrap();
            reader.package_id().to_string()
        };
        (encoded.bytes, expected)
    }

    #[test]
    fn strict_codecs_report_actual_metadata_and_unresolved_transform() {
        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            let (bytes, expected) = fixture(codec);
            let mut output = Vec::new();
            inspect(Cursor::new(bytes), &expected, &mut output).unwrap();
            let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(report["package_id"], expected);
            assert_eq!(report["source_verified_unchanged"], true);
            assert_eq!(report["tensors"].as_array().unwrap().len(), 2);
            let tensor = &report["tensors"][0];
            assert_eq!(tensor["dimensions"], serde_json::json!([2, 256]));
            assert_eq!(
                tensor["plane_count_histogram"],
                serde_json::json!([0, 1, 0, 1])
            );
            assert_eq!(tensor["transform"]["kind"], "SignedRht");
            assert_eq!(tensor["transform"]["seed"], 7);
            assert_eq!(tensor["transform"]["domain"], 9);
            assert!(tensor["transform"]["block"].is_null());
            assert_eq!(report["tensors"][1]["transform"]["kind"], "None");
        }
    }

    #[test]
    fn wrong_and_invalid_expected_ids_emit_nothing() {
        let (bytes, _) = fixture(SaltV2Codec::B3);
        let mut reader = SaltV2PackageReader::new_strict(Cursor::new(bytes)).unwrap();
        for expected in ["trp1_bad".to_owned(), format!("trp1_{}", "0".repeat(64))] {
            let mut output = Vec::new();
            assert!(inventory(&mut reader, &expected, &mut output).is_err());
            assert!(output.is_empty());
        }
    }

    #[test]
    fn malformed_unselected_data_and_truncated_sources_emit_nothing() {
        let (bytes, expected) = fixture(SaltV2Codec::B3);
        let mut reader = SaltV2PackageReader::new_strict(Cursor::new(&bytes)).unwrap();
        let mut packed = Vec::new();
        reader
            .visit_packed_tensor("zz_unselected", |plane| {
                packed.extend_from_slice(plane.packed_bytes());
            })
            .unwrap();
        let start = bytes
            .windows(packed.len())
            .rposition(|w| w == packed)
            .unwrap();
        let mut corrupt = bytes.clone();
        corrupt[start] = 255; // An invalid B3 code in the actual second tensor.
        assert!(SaltV2PackageReader::new_strict(Cursor::new(&corrupt)).is_err());
        for source in [&corrupt[..], &bytes[..bytes.len() - 1]] {
            let mut output = Vec::new();
            assert!(inspect(Cursor::new(source), &expected, &mut output).is_err());
            assert!(output.is_empty());
        }
    }

    struct Shared {
        bytes: Rc<RefCell<Vec<u8>>>,
        offset: u64,
    }
    impl Read for Shared {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            let start = self.offset as usize;
            let bytes = self.bytes.borrow();
            let count = bytes.len().saturating_sub(start).min(output.len());
            output[..count].copy_from_slice(&bytes[start..start + count]);
            self.offset = (start + count) as u64;
            Ok(count)
        }
    }
    impl Seek for Shared {
        fn seek(&mut self, seek: SeekFrom) -> io::Result<u64> {
            let next = match seek {
                SeekFrom::Start(next) => i128::from(next),
                SeekFrom::End(delta) => self.bytes.borrow().len() as i128 + i128::from(delta),
                SeekFrom::Current(delta) => i128::from(self.offset) + i128::from(delta),
            };
            self.offset = u64::try_from(next).map_err(io::Error::other)?;
            Ok(self.offset)
        }
    }

    #[test]
    fn same_length_source_mutation_emits_no_success_inventory() {
        let (bytes, expected) = fixture(SaltV2Codec::B3);
        let shared = Rc::new(RefCell::new(bytes));
        let mut reader = SaltV2PackageReader::new_strict(Shared {
            bytes: Rc::clone(&shared),
            offset: 0,
        })
        .unwrap();
        shared.borrow_mut()[0] ^= 1;
        let mut output = Vec::new();
        assert!(inventory(&mut reader, &expected, &mut output).is_err());
        assert!(output.is_empty());
    }
}

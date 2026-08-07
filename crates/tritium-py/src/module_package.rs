//! Strict, bounded conversion-artifact to SALT V2 package bridge.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use half::f16;
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SALT_V2_ALLOCATION_TILE_SIZE, SALT_V2_SCALE_GROUP_SIZE, SALT_V2_SCALE_GROUP_SIZE_64,
    SaltV2PackageReader, SaltV2PackageStreamPlan, SaltV2PackageStreamWriter, SaltV2Plane,
    SaltV2StreamTensorSpec, SaltV2Transform, unpack_salt_v2_plane,
};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[cfg(windows)]
use std::os::windows::fs::MetadataExt;

const MANIFEST_FILE: &str = "conversion.json";
const MAX_JSON_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptReference {
    file: String,
    digest: String,
    bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConversionManifest {
    schema_version: u64,
    artifact_kind: String,
    source_model_digest: String,
    evidence_id: String,
    algorithm_id: String,
    recipe_id: String,
    artifact_id: String,
    config: Value,
    coverage: CoverageReceipt,
    weight_receipts: Vec<ReceiptReference>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoverageReceipt {
    schema_version: u64,
    entries: Vec<CoverageEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoverageEntry {
    path: String,
    aliases: Vec<String>,
    disposition: String,
    #[serde(rename = "reason")]
    _reason: String,
    numel: u64,
    #[serde(rename = "logical_bytes")]
    _logical_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlaneReceipt {
    trits_file: String,
    trits_digest: String,
    trits_bytes: u64,
    scales_file: String,
    scales_digest: String,
    scales_bytes: u64,
    scales_shape: Vec<u64>,
    group_size: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightReceipt {
    schema_version: u64,
    recipe_id: String,
    path: String,
    aliases: Vec<String>,
    shape: Vec<u64>,
    weighted_mse: f64,
    fit_chunk_rows: u64,
    max_working_bytes: u64,
    planes: Vec<PlaneReceipt>,
}

#[derive(Debug)]
struct AdmittedPlane {
    trits_path: PathBuf,
    scales_path: PathBuf,
    trits_digest: String,
    scales_digest: String,
    trits_bytes: u64,
    scales_bytes: u64,
}

#[derive(Debug)]
struct AdmittedWeight {
    name: String,
    aliases: Vec<String>,
    rows: usize,
    columns: usize,
    scale_group_size: usize,
    planes: Vec<AdmittedPlane>,
}

#[derive(Debug)]
struct OpenedPlane {
    trits: File,
    scales: File,
}

/// Exact identity and physical ledger of one generic streamed SALT V2 package.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct ModuleSaltV2Receipt {
    package_id: String,
    packing: String,
    serialized_bytes: u64,
    resident_bytes: u64,
    tensors: u64,
}

#[pymethods]
impl ModuleSaltV2Receipt {
    #[getter]
    fn package_id(&self) -> &str {
        &self.package_id
    }

    #[getter]
    fn packing(&self) -> &str {
        &self.packing
    }

    #[getter]
    const fn serialized_bytes(&self) -> u64 {
        self.serialized_bytes
    }

    #[getter]
    const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    #[getter]
    const fn tensors(&self) -> u64 {
        self.tensors
    }
}

#[pyfunction]
#[pyo3(signature = (conversion_dir, output_file, *, packing = "b3", max_payload_bytes = 8_589_934_592))]
pub(crate) fn pack_module_conversion_salt_v2(
    py: Python<'_>,
    conversion_dir: &str,
    output_file: &str,
    packing: &str,
    max_payload_bytes: u64,
) -> PyResult<ModuleSaltV2Receipt> {
    if conversion_dir.is_empty() || output_file.is_empty() || max_payload_bytes == 0 {
        return Err(PyValueError::new_err(
            "conversion paths must be non-empty and max_payload_bytes must be positive",
        ));
    }
    let codec = match packing {
        "d2" => SaltV2Codec::D2,
        "b3" => SaltV2Codec::B3,
        "s34" => SaltV2Codec::S34,
        _ => {
            return Err(PyValueError::new_err(
                "packing must be 'd2', 'b3', or 's34'",
            ));
        }
    };
    let conversion_dir = PathBuf::from(conversion_dir);
    let output_file = PathBuf::from(output_file);
    let packing = packing.to_owned();
    py.detach(move || {
        pack(
            &conversion_dir,
            &output_file,
            codec,
            &packing,
            max_payload_bytes,
        )
    })
    .map_err(PyValueError::new_err)
}

fn pack(
    directory: &Path,
    output: &Path,
    codec: SaltV2Codec,
    packing: &str,
    max_payload_bytes: u64,
) -> Result<ModuleSaltV2Receipt, String> {
    let metadata = fs::symlink_metadata(directory)
        .map_err(|error| format!("inspect conversion directory failed: {:?}", error.kind()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err("conversion artifact must be an ordinary directory".to_owned());
    }
    if fs::symlink_metadata(output).is_ok() {
        return Err("SALT V2 output file already exists".to_owned());
    }
    let parent = output
        .parent()
        .ok_or_else(|| "SALT V2 output requires a parent directory".to_owned())?;
    let parent_metadata = fs::symlink_metadata(parent)
        .map_err(|error| format!("inspect output parent failed: {:?}", error.kind()))?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err("SALT V2 output parent must be an ordinary directory".to_owned());
    }

    let manifest_path = directory.join(MANIFEST_FILE);
    let manifest_bytes = read_regular(&manifest_path, MAX_JSON_BYTES)?;
    let manifest: ConversionManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("parse conversion manifest failed: {error}"))?;
    validate_manifest(&manifest, &manifest_bytes)?;

    let mut expected_files = BTreeSet::from([MANIFEST_FILE.to_owned()]);
    let mut weights = Vec::new();
    for (index, reference) in manifest.weight_receipts.iter().enumerate() {
        let expected_name = format!("weight-{index:05}.json");
        if reference.file != expected_name {
            return Err("conversion weight receipts are out of canonical order".to_owned());
        }
        expected_files.insert(expected_name.clone());
        let receipt_path = directory.join(&expected_name);
        let receipt_bytes = read_regular(&receipt_path, MAX_JSON_BYTES)?;
        verify_bytes(&receipt_bytes, &reference.digest, reference.bytes)?;
        let receipt: WeightReceipt = serde_json::from_slice(&receipt_bytes)
            .map_err(|error| format!("parse conversion weight receipt failed: {error}"))?;
        let (weight, files) = admit_weight(
            directory,
            index,
            receipt,
            &manifest.recipe_id,
            max_payload_bytes,
        )?;
        expected_files.extend(files);
        weights.push(weight);
    }
    if weights.is_empty() {
        return Err("conversion artifact has no fitted weights".to_owned());
    }
    validate_coverage(&manifest.coverage, &weights)?;
    let actual_files = fs::read_dir(directory)
        .map_err(|error| format!("read conversion directory failed: {:?}", error.kind()))?
        .map(|entry| {
            entry
                .map_err(|error| format!("read conversion entry failed: {:?}", error.kind()))
                .and_then(|entry| {
                    entry
                        .file_name()
                        .into_string()
                        .map_err(|_| "conversion filename is not UTF-8".to_owned())
                })
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if actual_files != expected_files {
        return Err("conversion directory contains unknown files".to_owned());
    }

    let specs = weights
        .iter()
        .map(|weight| {
            SaltV2StreamTensorSpec::new_with_layout(
                weight.name.clone(),
                vec![weight.rows as u64, weight.columns as u64],
                SaltV2Transform::None,
                weight.scale_group_size,
            )
            .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let plane_counts = weights.iter().flat_map(|weight| {
        let tiles = weight
            .rows
            .saturating_mul(weight.columns)
            .div_ceil(SALT_V2_ALLOCATION_TILE_SIZE);
        std::iter::repeat_n(weight.planes.len() as u8, tiles)
    });
    let plan = SaltV2PackageStreamPlan::new(codec, specs, plane_counts)
        .map_err(|error| error.to_string())?;
    let result = (|| {
        let file = File::create_new(output)
            .map_err(|error| format!("create SALT V2 output failed: {:?}", error.kind()))?;
        let mut writer =
            SaltV2PackageStreamWriter::new(file, plan).map_err(|error| error.to_string())?;
        for weight in &weights {
            stream_weight(weight, &mut writer)?;
        }
        let (file, ledger) = writer.finish().map_err(|error| error.to_string())?;
        file.sync_all()
            .map_err(|error| format!("sync SALT V2 output failed: {:?}", error.kind()))?;
        let mut reader = SaltV2PackageReader::new_strict(
            File::open(output)
                .map_err(|error| format!("reopen SALT V2 output failed: {:?}", error.kind()))?,
        )
        .map_err(|error| error.to_string())?;
        let resident = reader
            .indexed_runtime_ledger()
            .map_err(|error| error.to_string())?
            .steady_resident_bytes();
        if reader.ledger() != ledger {
            return Err("reopened SALT V2 physical ledger changed".to_owned());
        }
        verify_package_semantics(&mut reader, codec, &weights)?;
        Ok(ModuleSaltV2Receipt {
            package_id: reader.package_id().to_string(),
            packing: packing.to_owned(),
            serialized_bytes: ledger.total_bytes,
            resident_bytes: resident,
            tensors: weights.len() as u64,
        })
    })();
    if result.is_err() {
        let _ = fs::remove_file(output);
    }
    result
}

fn validate_manifest(manifest: &ConversionManifest, bytes: &[u8]) -> Result<(), String> {
    if manifest.schema_version != 2
        || manifest.artifact_kind != "tritium.module-additive-ptq-v2"
        || !is_sha256(&manifest.source_model_digest)
        || !is_sha256(&manifest.evidence_id)
        || manifest.algorithm_id.is_empty()
        || !is_sha256(&manifest.recipe_id)
        || !is_sha256(&manifest.artifact_id)
    {
        return Err("unsupported or incomplete conversion manifest".to_owned());
    }
    let mut identity: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("parse conversion identity failed: {error}"))?;
    identity
        .as_object_mut()
        .ok_or_else(|| "conversion manifest must be an object".to_owned())?
        .remove("artifact_id");
    let canonical = serde_json::to_vec(&identity)
        .map_err(|error| format!("encode conversion identity failed: {error}"))?;
    verify_bytes(&canonical, &manifest.artifact_id, canonical.len() as u64)?;
    let _ = &manifest.config;
    Ok(())
}

fn admit_weight(
    directory: &Path,
    index: usize,
    receipt: WeightReceipt,
    recipe_id: &str,
    maximum: u64,
) -> Result<(AdmittedWeight, BTreeSet<String>), String> {
    if receipt.schema_version != 2
        || receipt.recipe_id != recipe_id
        || receipt.path.is_empty()
        || receipt.aliases.first() != Some(&receipt.path)
        || receipt.aliases.iter().any(String::is_empty)
        || receipt.shape.len() != 2
        || receipt.shape.contains(&0)
        || !receipt.weighted_mse.is_finite()
        || receipt.weighted_mse < 0.0
        || receipt.fit_chunk_rows == 0
        || receipt.fit_chunk_rows > receipt.shape[0]
        || receipt.max_working_bytes == 0
        || !(1..=3).contains(&receipt.planes.len())
    {
        return Err("conversion weight receipt is invalid".to_owned());
    }
    let rows = usize::try_from(receipt.shape[0])
        .map_err(|_| "conversion row count exceeds platform".to_owned())?;
    let columns = usize::try_from(receipt.shape[1])
        .map_err(|_| "conversion column count exceeds platform".to_owned())?;
    let scale_group_size = if columns.is_multiple_of(SALT_V2_SCALE_GROUP_SIZE) {
        SALT_V2_SCALE_GROUP_SIZE
    } else if columns.is_multiple_of(SALT_V2_SCALE_GROUP_SIZE_64) {
        SALT_V2_SCALE_GROUP_SIZE_64
    } else {
        return Err(format!(
            "conversion weight `{}` has {} columns; SALT V2 export requires G64 alignment",
            receipt.path, columns
        ));
    };
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| "conversion weight geometry overflowed".to_owned())?;
    let mut planes = Vec::new();
    let mut files = BTreeSet::new();
    for (plane_index, plane) in receipt.planes.into_iter().enumerate() {
        let trits_name = format!("weight-{index:05}-plane-{plane_index}.trits.i8");
        let scales_name = format!("weight-{index:05}-plane-{plane_index}.scales.f16le");
        if plane.trits_file != trits_name
            || plane.scales_file != scales_name
            || plane.trits_bytes != elements as u64
            || plane.scales_bytes != rows as u64 * 2
            || plane.scales_shape != [rows as u64, 1]
            || plane.group_size != columns as u64
        {
            return Err("conversion plane geometry or byte ledger is invalid".to_owned());
        }
        if plane.trits_bytes > maximum || plane.scales_bytes > maximum {
            return Err("conversion payload exceeds byte ceiling".to_owned());
        }
        let trits_path = directory.join(&trits_name);
        let scales_path = directory.join(&scales_name);
        drop(open_verified(
            &trits_path,
            &plane.trits_digest,
            plane.trits_bytes,
        )?);
        drop(open_verified(
            &scales_path,
            &plane.scales_digest,
            plane.scales_bytes,
        )?);
        files.insert(trits_name);
        files.insert(scales_name);
        planes.push(AdmittedPlane {
            trits_path,
            scales_path,
            trits_digest: plane.trits_digest,
            scales_digest: plane.scales_digest,
            trits_bytes: plane.trits_bytes,
            scales_bytes: plane.scales_bytes,
        });
    }
    Ok((
        AdmittedWeight {
            name: receipt.path,
            aliases: receipt.aliases,
            rows,
            columns,
            scale_group_size,
            planes,
        },
        files,
    ))
}

fn validate_coverage(coverage: &CoverageReceipt, weights: &[AdmittedWeight]) -> Result<(), String> {
    if coverage.schema_version != 1 {
        return Err("unsupported conversion coverage schema".to_owned());
    }
    let selected = coverage
        .entries
        .iter()
        .filter(|entry| entry.disposition == "selected")
        .map(|entry| (entry.path.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    let fitted = weights
        .iter()
        .map(|weight| (weight.name.as_str(), weight))
        .collect::<BTreeMap<_, _>>();
    if selected.len()
        != coverage
            .entries
            .iter()
            .filter(|entry| entry.disposition == "selected")
            .count()
        || fitted.len() != weights.len()
        || selected.keys().ne(fitted.keys())
        || selected.iter().any(|(path, entry)| {
            fitted.get(path).is_none_or(|weight| {
                entry.aliases != weight.aliases
                    || entry.numel != (weight.rows as u64).saturating_mul(weight.columns as u64)
            })
        })
    {
        return Err("conversion weights differ from selected coverage".to_owned());
    }
    Ok(())
}

fn verify_package_semantics(
    reader: &mut SaltV2PackageReader<File>,
    codec: SaltV2Codec,
    weights: &[AdmittedWeight],
) -> Result<(), String> {
    for weight in weights {
        let mut opened = open_weight(weight)?;
        let mut comparison = Ok(());
        reader
            .visit_packed_tensor(&weight.name, |packed| {
                if comparison.is_err() {
                    return;
                }
                comparison = (|| {
                    if packed.plane_count() != weight.planes.len() {
                        return Err("packed plane count differs from conversion".to_owned());
                    }
                    let start = packed
                        .tile_index()
                        .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
                        .ok_or_else(|| "packed tile offset overflowed".to_owned())?;
                    let opened_plane = opened
                        .get_mut(packed.plane_index())
                        .ok_or_else(|| "packed plane index exceeds conversion".to_owned())?;
                    let source =
                        read_source_plane(weight, opened_plane, start, packed.logical_len())?;
                    let decoded =
                        unpack_salt_v2_plane(codec, packed.packed_bytes(), packed.logical_len())
                            .map_err(|error| error.to_string())?;
                    let scale_bits = source
                        .scales()
                        .iter()
                        .map(|scale| scale.to_bits())
                        .collect::<Vec<_>>();
                    let packed_scale_bits = packed
                        .scales()
                        .iter()
                        .map(|scale| scale.to_bits())
                        .collect::<Vec<_>>();
                    if decoded != source.trits() || scale_bits != packed_scale_bits {
                        return Err("packed SALT V2 semantics differ from conversion".to_owned());
                    }
                    Ok(())
                })();
            })
            .map_err(|error| error.to_string())?;
        comparison?;
        verify_opened_weight(weight, &mut opened)?;
    }
    Ok(())
}

fn stream_weight(
    weight: &AdmittedWeight,
    writer: &mut SaltV2PackageStreamWriter<File>,
) -> Result<(), String> {
    let mut opened = open_weight(weight)?;
    let elements = weight
        .rows
        .checked_mul(weight.columns)
        .ok_or_else(|| "conversion weight geometry overflowed".to_owned())?;
    for start in (0..elements).step_by(SALT_V2_ALLOCATION_TILE_SIZE) {
        let logical_len = (elements - start).min(SALT_V2_ALLOCATION_TILE_SIZE);
        let mut tile_planes = Vec::new();
        for plane in &mut opened {
            tile_planes.push(read_source_plane(weight, plane, start, logical_len)?);
        }
        writer
            .push_planes(&tile_planes)
            .map_err(|error| error.to_string())?;
    }
    verify_opened_weight(weight, &mut opened)?;
    Ok(())
}

fn read_regular(path: &Path, maximum: u64) -> Result<Vec<u8>, String> {
    let before = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect conversion file failed: {:?}", error.kind()))?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() > maximum {
        return Err("conversion file must be a bounded ordinary file".to_owned());
    }
    let mut file = File::open(path)
        .map_err(|error| format!("open conversion file failed: {:?}", error.kind()))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("inspect opened conversion file failed: {:?}", error.kind()))?;
    let current = fs::symlink_metadata(path)
        .map_err(|error| format!("reinspect conversion file failed: {:?}", error.kind()))?;
    if current.file_type().is_symlink()
        || !current.is_file()
        || !same_file_identity(&opened, &current)
    {
        return Err("conversion file changed while opening".to_owned());
    }
    let capacity = usize::try_from(opened.len())
        .map_err(|_| "conversion file length exceeds platform".to_owned())?;
    let mut bytes = Vec::with_capacity(capacity);
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("read conversion file failed: {:?}", error.kind()))?;
    if bytes.len() as u64 != opened.len() {
        return Err("conversion file changed while reading".to_owned());
    }
    Ok(bytes)
}

fn open_verified(path: &Path, digest: &str, bytes: u64) -> Result<File, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect conversion payload failed: {:?}", error.kind()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != bytes {
        return Err("conversion payload identity mismatch".to_owned());
    }
    let mut file = File::open(path)
        .map_err(|error| format!("open conversion payload failed: {:?}", error.kind()))?;
    let opened = file.metadata().map_err(|error| {
        format!(
            "inspect opened conversion payload failed: {:?}",
            error.kind()
        )
    })?;
    let current = fs::symlink_metadata(path)
        .map_err(|error| format!("reinspect conversion payload failed: {:?}", error.kind()))?;
    if current.file_type().is_symlink()
        || !current.is_file()
        || !same_file_identity(&opened, &current)
    {
        return Err("conversion payload changed while opening".to_owned());
    }
    let actual = hash_reader(&mut file)?;
    if actual != digest {
        return Err("conversion payload identity mismatch".to_owned());
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind conversion payload failed: {:?}", error.kind()))?;
    Ok(file)
}

fn open_weight(weight: &AdmittedWeight) -> Result<Vec<OpenedPlane>, String> {
    weight
        .planes
        .iter()
        .map(|plane| {
            Ok(OpenedPlane {
                trits: open_verified(&plane.trits_path, &plane.trits_digest, plane.trits_bytes)?,
                scales: open_verified(
                    &plane.scales_path,
                    &plane.scales_digest,
                    plane.scales_bytes,
                )?,
            })
        })
        .collect()
}

fn read_source_plane(
    weight: &AdmittedWeight,
    plane: &mut OpenedPlane,
    start: usize,
    logical_len: usize,
) -> Result<SaltV2Plane, String> {
    plane
        .trits
        .seek(SeekFrom::Start(start as u64))
        .map_err(|error| format!("seek conversion trits failed: {:?}", error.kind()))?;
    let mut trits = vec![0u8; logical_len];
    plane
        .trits
        .read_exact(&mut trits)
        .map_err(|error| format!("read conversion trits failed: {:?}", error.kind()))?;
    let trits = trits.into_iter().map(|value| value as i8).collect();
    let groups = logical_len.div_ceil(weight.scale_group_size);
    let mut scales = Vec::with_capacity(groups);
    for group in 0..groups {
        let coefficient = start + group * weight.scale_group_size;
        let row = coefficient / weight.columns;
        plane
            .scales
            .seek(SeekFrom::Start((row * 2) as u64))
            .map_err(|error| format!("seek conversion scales failed: {:?}", error.kind()))?;
        let mut bits = [0u8; 2];
        plane
            .scales
            .read_exact(&mut bits)
            .map_err(|error| format!("read conversion scales failed: {:?}", error.kind()))?;
        scales.push(f16::from_bits(u16::from_le_bytes(bits)));
    }
    SaltV2Plane::new_with_scale_group_size(trits, scales, weight.scale_group_size)
        .map_err(|error| error.to_string())
}

fn verify_opened_weight(weight: &AdmittedWeight, opened: &mut [OpenedPlane]) -> Result<(), String> {
    for (source, plane) in weight.planes.iter().zip(opened) {
        for (file, path, expected) in [
            (&mut plane.trits, &source.trits_path, &source.trits_digest),
            (
                &mut plane.scales,
                &source.scales_path,
                &source.scales_digest,
            ),
        ] {
            let metadata = fs::symlink_metadata(path).map_err(|error| {
                format!("reinspect conversion payload failed: {:?}", error.kind())
            })?;
            let opened = file
                .metadata()
                .map_err(|error| format!("inspect opened payload failed: {:?}", error.kind()))?;
            file.seek(SeekFrom::Start(0))
                .map_err(|error| format!("rewind conversion payload failed: {:?}", error.kind()))?;
            if metadata.file_type().is_symlink()
                || !metadata.is_file()
                || metadata.len() != opened.len()
                || !same_file_identity(&opened, &metadata)
                || hash_reader(file)? != *expected
            {
                return Err("conversion payload changed during package write".to_owned());
            }
        }
    }
    Ok(())
}

fn verify_bytes(bytes: &[u8], expected_digest: &str, expected_bytes: u64) -> Result<(), String> {
    if !is_sha256(expected_digest)
        || bytes.len() as u64 != expected_bytes
        || sha256(bytes) != expected_digest
    {
        return Err("conversion receipt identity mismatch".to_owned());
    }
    Ok(())
}

#[cfg(unix)]
fn same_file_identity(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.dev() == right.dev() && left.ino() == right.ino() && left.len() == right.len()
}

#[cfg(windows)]
fn same_file_identity(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    // volume_serial_number()/file_index() are the true identity pair but sit
    // behind the unstable windows_by_handle feature; the stable size,
    // timestamp, and attribute fields still catch a file swapped between open
    // and re-inspection.
    left.file_size() == right.file_size()
        && left.creation_time() == right.creation_time()
        && left.last_write_time() == right.last_write_time()
        && left.file_attributes() == right.file_attributes()
}

#[cfg(not(any(unix, windows)))]
fn same_file_identity(_left: &fs::Metadata, _right: &fs::Metadata) -> bool {
    false
}

fn is_sha256(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn hash_reader(reader: &mut File) -> Result<String, String> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("hash conversion payload failed: {:?}", error.kind()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(prefixed_digest(hasher.finalize().as_slice()))
}

fn sha256(bytes: &[u8]) -> String {
    prefixed_digest(Sha256::digest(bytes).as_slice())
}

fn prefixed_digest(bytes: &[u8]) -> String {
    let mut value = String::from("sha256:");
    for byte in bytes {
        let _ = write!(value, "{byte:02x}");
    }
    value
}

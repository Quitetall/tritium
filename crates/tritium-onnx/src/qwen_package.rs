//! Authenticated file-backed SALT V2 and preserved-tensor source for Qwen export.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Cursor, Read, Seek};

use tritium_format::salt_v2_package::{SaltV2PackageReader, SaltV2Transform};
use tritium_format::{PackageId, SafeTensorsReader};
use tritium_nn::HostSaltV2Linear;

use crate::{OnnxModelError, Qwen35PackedTensorProvider, SaltV2PackedMatrix};

/// Exact admitted shape of one packed Qwen matrix.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35PackageMatrixSpec<'a> {
    /// Canonical checkpoint tensor name.
    pub name: &'a str,
    /// Output rows.
    pub rows: usize,
    /// Input columns.
    pub columns: usize,
}

/// Exact admitted shape of one preserved BF16 Qwen tensor.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35PackagePreservedSpec<'a> {
    /// Canonical checkpoint tensor name.
    pub name: &'a str,
    /// Exact checkpoint rank and dimensions.
    pub shape: &'a [usize],
}

/// Authenticated physical ledgers and exact tensor schema for source admission.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35PackageSourceSpec<'a> {
    /// Complete, non-overlapping matrix schema.
    pub matrices: &'a [Qwen35PackageMatrixSpec<'a>],
    /// Complete, non-overlapping preserved-tensor schema.
    pub preserved: &'a [Qwen35PackagePreservedSpec<'a>],
    /// Exact serialized SALT V2 bytes from the authenticated manifest.
    pub package_bytes: u64,
    /// Caller policy ceiling for serialized SALT V2 bytes.
    pub max_package_bytes: u64,
    /// Exact serialized safetensors bytes from the authenticated manifest.
    pub preserved_bytes: u64,
    /// Caller policy ceiling for the authenticated preserved snapshot.
    pub max_preserved_snapshot_bytes: u64,
    /// Exact indexed SALT steady bytes from the authenticated profile ledger.
    pub salt_resident_bytes: u64,
    /// Maximum indexed SALT steady bytes admitted by the selected profile.
    pub max_salt_resident_bytes: u64,
    /// Exact widened preserved bytes derived from the admitted tensor schema.
    pub preserved_fp32_bytes: u64,
    /// Maximum widened fp32 bytes admitted for preserved tensors.
    pub max_preserved_fp32_bytes: u64,
}

/// Owned, content-bound Qwen matrix and preserved-tensor arenas.
///
/// The source keeps additive matrices packed and widens only the explicitly
/// preserved floating-point tensors. It is intended to outlive the mapped ONNX
/// graph views borrowed from it.
#[derive(Debug)]
pub struct Qwen35SaltV2PackageSource {
    names: Vec<String>,
    matrices: BTreeMap<String, HostSaltV2Linear>,
    vectors: BTreeMap<String, Vec<f32>>,
    package_id: String,
    preserved_package_id: String,
}

impl Qwen35SaltV2PackageSource {
    /// Open already-authorized regular file handles and materialize final packed arenas.
    ///
    /// `package_file` and `preserved_file` must be independently opened without
    /// following symlinks. Their exact transport identities must come from an
    /// authenticated schema-v3 manifest, never from the candidate files.
    ///
    /// # Errors
    /// Returns [`OnnxModelError`] for non-regular handles, identity mismatch,
    /// malformed packages, duplicate coverage, non-BF16 preserved tensors,
    /// source mutation, or failed decoding.
    pub fn from_files(
        package_file: File,
        preserved_file: File,
        admitted_package_id: &str,
        admitted_preserved_package_id: &str,
        spec: Qwen35PackageSourceSpec<'_>,
    ) -> Result<Self, OnnxModelError> {
        validate_source_spec(spec)?;
        require_regular_size(&package_file, "SALT V2 package", spec.package_bytes)?;
        require_regular_size(
            &preserved_file,
            "preserved safetensors",
            spec.preserved_bytes,
        )?;

        let mut package = SaltV2PackageReader::new_strict(package_file)
            .map_err(|error| invalid(format!("open SALT V2 package: {error}")))?;
        let package_id = package.package_id().to_string();
        if package_id != admitted_package_id {
            return Err(identity("SALT V2 package identity differs from admission"));
        }
        let matrix_names = package
            .tensor_names()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        require_exact_names(
            "SALT V2 package",
            &matrix_names,
            spec.matrices.iter().map(|tensor| tensor.name),
        )?;
        let mut salt_resident_bytes = 0_u64;
        for tensor in spec.matrices {
            let info = package
                .tensor_info(tensor.name)
                .ok_or_else(|| invalid(format!("missing packed matrix `{}`", tensor.name)))?;
            let expected_dims = [
                u64::try_from(tensor.rows)
                    .map_err(|_| invalid(format!("matrix `{}` rows exceed u64", tensor.name)))?,
                u64::try_from(tensor.columns)
                    .map_err(|_| invalid(format!("matrix `{}` columns exceed u64", tensor.name)))?,
            ];
            if info.dims() != expected_dims || info.transform() != SaltV2Transform::None {
                return Err(invalid(format!(
                    "packed matrix `{}` shape or transform differs from admission",
                    tensor.name
                )));
            }
            salt_resident_bytes = salt_resident_bytes
                .checked_add(info.runtime_ledger().steady_resident_bytes())
                .ok_or_else(|| invalid("aggregate SALT resident bytes overflow"))?;
        }
        if salt_resident_bytes != spec.salt_resident_bytes
            || salt_resident_bytes > spec.max_salt_resident_bytes
        {
            return Err(invalid(format!(
                "aggregate SALT resident bytes {salt_resident_bytes} differ from admitted {} or exceed policy {}",
                spec.salt_resident_bytes, spec.max_salt_resident_bytes
            )));
        }
        let mut matrices = BTreeMap::new();
        for tensor in spec.matrices {
            let matrix =
                HostSaltV2Linear::from_reader(&mut package, tensor.name).map_err(|error| {
                    invalid(format!("read SALT V2 matrix `{}`: {error}", tensor.name))
                })?;
            if matrices.insert(tensor.name.to_owned(), matrix).is_some() {
                return Err(invalid(format!(
                    "duplicate matrix tensor `{}`",
                    tensor.name
                )));
            }
        }
        package
            .verify_unchanged()
            .map_err(|error| identity(format!("SALT V2 package changed while reading: {error}")))?;

        let preserved_snapshot = read_exact_snapshot(preserved_file, spec.preserved_bytes)?;
        let preserved_package_id = PackageId::from_package_bytes(&preserved_snapshot).to_string();
        if preserved_package_id != admitted_preserved_package_id {
            return Err(identity(
                "preserved safetensors identity differs from admission",
            ));
        }
        let mut preserved = SafeTensorsReader::new(Cursor::new(preserved_snapshot))
            .map_err(|error| invalid(format!("open preserved safetensors: {error}")))?;
        let vector_names = preserved.names().map(str::to_owned).collect::<Vec<_>>();
        require_exact_names(
            "preserved safetensors",
            &vector_names,
            spec.preserved.iter().map(|tensor| tensor.name),
        )?;
        let preserved_fp32_bytes = spec.preserved.iter().try_fold(0_u64, |total, tensor| {
            let elements = tensor.shape.iter().try_fold(1_u64, |product, &dimension| {
                product.checked_mul(u64::try_from(dimension).ok()?)
            })?;
            total.checked_add(elements.checked_mul(4)?)
        });
        let Some(preserved_fp32_bytes) = preserved_fp32_bytes else {
            return Err(invalid("aggregate preserved fp32 bytes overflow"));
        };
        if preserved_fp32_bytes != spec.preserved_fp32_bytes
            || preserved_fp32_bytes > spec.max_preserved_fp32_bytes
        {
            return Err(invalid(format!(
                "aggregate preserved fp32 bytes {preserved_fp32_bytes} differ from admitted {} or exceed policy {}",
                spec.preserved_fp32_bytes, spec.max_preserved_fp32_bytes
            )));
        }
        let mut vectors = BTreeMap::new();
        for tensor in spec.preserved {
            if matrices.contains_key(tensor.name) {
                return Err(invalid(format!(
                    "tensor `{}` occurs in both packed and preserved sources",
                    tensor.name
                )));
            }
            if preserved.dtype(tensor.name) != Some("BF16")
                || preserved.shape(tensor.name) != Some(tensor.shape)
            {
                return Err(invalid(format!(
                    "preserved tensor `{}` dtype or shape differs from admission",
                    tensor.name
                )));
            }
            let values = preserved.tensor_f32(tensor.name).map_err(|error| {
                invalid(format!("read preserved tensor `{}`: {error}", tensor.name))
            })?;
            if vectors.insert(tensor.name.to_owned(), values).is_some() {
                return Err(invalid(format!(
                    "duplicate preserved tensor `{}`",
                    tensor.name
                )));
            }
        }

        let mut names = matrix_names;
        names.extend(vector_names);
        names.sort_unstable();
        Ok(Self {
            names,
            matrices,
            vectors,
            package_id,
            preserved_package_id,
        })
    }

    /// Exact admitted SALT V2 transport identity.
    #[must_use]
    pub fn package_id(&self) -> &str {
        &self.package_id
    }

    /// Exact admitted preserved-safetensors transport identity.
    #[must_use]
    pub fn preserved_package_id(&self) -> &str {
        &self.preserved_package_id
    }
}

impl<'a> Qwen35PackedTensorProvider<'a, SaltV2PackedMatrix<'a>> for Qwen35SaltV2PackageSource {
    fn tensor_names(&'a self) -> Result<&'a [String], OnnxModelError> {
        Ok(&self.names)
    }

    fn matrix(&'a self, name: &str) -> Result<SaltV2PackedMatrix<'a>, OnnxModelError> {
        let matrix = self
            .matrices
            .get(name)
            .ok_or_else(|| invalid(format!("missing packed matrix `{name}`")))?;
        Ok(SaltV2PackedMatrix {
            rows: matrix.rows(),
            columns: matrix.columns(),
            codec: matrix.codec(),
            payload: matrix.payload(),
            scales: matrix.scales(),
            allocation_map: matrix.allocation_map(),
            rank_prefixes: matrix.rank_prefixes(),
            terminal_map_value: matrix.terminal_map_value(),
        })
    }

    fn vector(&'a self, name: &str) -> Result<&'a [f32], OnnxModelError> {
        self.vectors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid(format!("missing preserved tensor `{name}`")))
    }
}

fn read_exact_snapshot(mut file: File, expected_bytes: u64) -> Result<Box<[u8]>, OnnxModelError> {
    file.rewind()
        .map_err(|error| invalid(format!("seek preserved safetensors: {error}")))?;
    let expected = usize::try_from(expected_bytes)
        .map_err(|_| invalid("preserved safetensors size exceeds host usize"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(expected)
        .map_err(|_| invalid(format!("could not allocate {expected} snapshot bytes")))?;
    file.take(expected_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| invalid(format!("read preserved safetensors snapshot: {error}")))?;
    if bytes.len() != expected {
        return Err(identity(format!(
            "preserved safetensors changed size while reading: {}/{expected}",
            bytes.len()
        )));
    }
    Ok(bytes.into_boxed_slice())
}

fn require_regular_size(
    file: &File,
    label: &str,
    expected_bytes: u64,
) -> Result<(), OnnxModelError> {
    let metadata = file
        .metadata()
        .map_err(|error| invalid(format!("inspect {label}: {error}")))?;
    if !metadata.is_file() {
        return Err(invalid(format!("{label} handle must be a regular file")));
    }
    if metadata.len() != expected_bytes {
        return Err(identity(format!(
            "{label} size {} differs from admitted {expected_bytes}",
            metadata.len()
        )));
    }
    Ok(())
}

fn require_exact_names<'a>(
    label: &str,
    actual: &[String],
    expected: impl Iterator<Item = &'a str>,
) -> Result<(), OnnxModelError> {
    let actual = actual.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(invalid(format!(
            "{label} tensor names differ from exact admission"
        )));
    }
    Ok(())
}

fn validate_source_spec(spec: Qwen35PackageSourceSpec<'_>) -> Result<(), OnnxModelError> {
    if spec.matrices.is_empty()
        || spec.preserved.is_empty()
        || spec.package_bytes == 0
        || spec.max_package_bytes == 0
        || spec.preserved_bytes == 0
        || spec.max_preserved_snapshot_bytes == 0
        || spec.salt_resident_bytes == 0
        || spec.max_salt_resident_bytes == 0
        || spec.preserved_fp32_bytes == 0
        || spec.max_preserved_fp32_bytes == 0
    {
        return Err(invalid(
            "Qwen package source admission ledgers must be nonzero",
        ));
    }
    if spec.package_bytes > spec.max_package_bytes
        || spec.preserved_bytes > spec.max_preserved_snapshot_bytes
    {
        return Err(invalid(
            "Qwen serialized source bytes exceed caller policy ceilings",
        ));
    }
    let mut names = BTreeSet::new();
    for tensor in spec.matrices {
        if tensor.name.is_empty()
            || tensor.rows == 0
            || tensor.columns == 0
            || !names.insert(tensor.name)
        {
            return Err(invalid(
                "Qwen matrix admission contains an invalid duplicate",
            ));
        }
    }
    for tensor in spec.preserved {
        if tensor.name.is_empty()
            || tensor.shape.is_empty()
            || tensor.shape.contains(&0)
            || !names.insert(tensor.name)
        {
            return Err(invalid(
                "Qwen preserved admission contains an invalid duplicate",
            ));
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> OnnxModelError {
    OnnxModelError::InvalidModel(message.into())
}

fn identity(message: impl Into<String>) -> OnnxModelError {
    OnnxModelError::ExternalDataMismatch(message.into())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use half::{bf16, f16};
    use tritium_format::PackageId;
    use tritium_format::salt_v2::SaltV2Codec;
    use tritium_format::salt_v2_package::{
        SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, write_salt_v2_package,
    };

    use super::*;

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    struct TestDir(std::path::PathBuf);

    impl TestDir {
        fn new() -> Self {
            let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "tritium-onnx-qwen-source-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture() -> (
        TestDir,
        std::path::PathBuf,
        std::path::PathBuf,
        String,
        String,
    ) {
        let dir = TestDir::new();
        let package_path = dir.0.join("profile.tsalt2");
        let preserved_path = dir.0.join("preserved.safetensors");

        let mut trits = vec![0_i8; 256];
        trits[0] = 1;
        let plane = SaltV2Plane::new(trits, vec![f16::ONE; 2]).unwrap();
        let tensor = SaltV2Tensor::new(
            "matrix.weight",
            vec![1, 256],
            vec![SaltV2Tile::new(vec![plane]).unwrap()],
        )
        .unwrap();
        let encoded =
            write_salt_v2_package(&SaltV2Package::new(SaltV2Codec::B3, vec![tensor]).unwrap())
                .unwrap();
        fs::write(&package_path, &encoded.bytes).unwrap();

        let header = br#"{"norm.weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        let mut preserved = Vec::new();
        preserved.extend_from_slice(&(header.len() as u64).to_le_bytes());
        preserved.extend_from_slice(header);
        preserved.extend_from_slice(&bf16::from_f32(1.0).to_bits().to_le_bytes());
        preserved.extend_from_slice(&bf16::from_f32(-0.5).to_bits().to_le_bytes());
        fs::write(&preserved_path, &preserved).unwrap();

        let package_id = PackageId::from_package_bytes(&encoded.bytes).to_string();
        let preserved_id = PackageId::from_package_bytes(&preserved).to_string();
        (dir, package_path, preserved_path, package_id, preserved_id)
    }

    fn salt_resident_bytes(path: &std::path::Path) -> u64 {
        SaltV2PackageReader::new_strict(File::open(path).unwrap())
            .unwrap()
            .indexed_runtime_ledger()
            .unwrap()
            .steady_resident_bytes()
    }

    #[test]
    fn source_preserves_additive_arenas_and_exact_bf16_vectors() {
        let (_dir, package_path, preserved_path, package_id, preserved_id) = fixture();
        let matrices = [Qwen35PackageMatrixSpec {
            name: "matrix.weight",
            rows: 1,
            columns: 256,
        }];
        let shape = [2];
        let preserved = [Qwen35PackagePreservedSpec {
            name: "norm.weight",
            shape: &shape,
        }];
        let spec = Qwen35PackageSourceSpec {
            matrices: &matrices,
            preserved: &preserved,
            package_bytes: fs::metadata(&package_path).unwrap().len(),
            max_package_bytes: 4096,
            preserved_bytes: fs::metadata(&preserved_path).unwrap().len(),
            max_preserved_snapshot_bytes: 4096,
            salt_resident_bytes: salt_resident_bytes(&package_path),
            max_salt_resident_bytes: 1024,
            preserved_fp32_bytes: 8,
            max_preserved_fp32_bytes: 8,
        };
        let source = Qwen35SaltV2PackageSource::from_files(
            File::open(package_path).unwrap(),
            File::open(preserved_path).unwrap(),
            &package_id,
            &preserved_id,
            spec,
        )
        .unwrap();

        assert_eq!(source.package_id(), package_id);
        assert_eq!(source.preserved_package_id(), preserved_id);
        assert_eq!(
            source.tensor_names().unwrap(),
            ["matrix.weight", "norm.weight"]
        );
        let matrix = source.matrix("matrix.weight").unwrap();
        matrix.validate().unwrap();
        assert_eq!((matrix.rows, matrix.columns), (1, 256));
        assert_eq!(source.vector("norm.weight").unwrap(), [1.0, -0.5]);
    }

    #[test]
    fn source_rejects_unadmitted_transport_identity() {
        let (_dir, package_path, preserved_path, package_id, preserved_id) = fixture();
        let matrices = [Qwen35PackageMatrixSpec {
            name: "matrix.weight",
            rows: 1,
            columns: 256,
        }];
        let shape = [2];
        let preserved = [Qwen35PackagePreservedSpec {
            name: "norm.weight",
            shape: &shape,
        }];
        let spec = Qwen35PackageSourceSpec {
            matrices: &matrices,
            preserved: &preserved,
            package_bytes: fs::metadata(&package_path).unwrap().len(),
            max_package_bytes: 4096,
            preserved_bytes: fs::metadata(&preserved_path).unwrap().len(),
            max_preserved_snapshot_bytes: 4096,
            salt_resident_bytes: salt_resident_bytes(&package_path),
            max_salt_resident_bytes: 1024,
            preserved_fp32_bytes: 8,
            max_preserved_fp32_bytes: 8,
        };
        let error = Qwen35SaltV2PackageSource::from_files(
            File::open(&package_path).unwrap(),
            File::open(&preserved_path).unwrap(),
            "trp1_wrong",
            &preserved_id,
            spec,
        )
        .unwrap_err();
        assert!(
            matches!(error, OnnxModelError::ExternalDataMismatch(reason) if reason.contains("package identity"))
        );

        let error = Qwen35SaltV2PackageSource::from_files(
            File::open(package_path).unwrap(),
            File::open(preserved_path).unwrap(),
            &package_id,
            "trp1_wrong",
            spec,
        )
        .unwrap_err();
        assert!(
            matches!(error, OnnxModelError::ExternalDataMismatch(reason) if reason.contains("preserved safetensors identity"))
        );
    }

    #[test]
    fn source_rejects_shape_and_budget_drift_before_publication() {
        let (_dir, package_path, preserved_path, package_id, preserved_id) = fixture();
        let matrices = [Qwen35PackageMatrixSpec {
            name: "matrix.weight",
            rows: 1,
            columns: 256,
        }];
        let wrong_shape = [1, 2];
        let preserved = [Qwen35PackagePreservedSpec {
            name: "norm.weight",
            shape: &wrong_shape,
        }];
        let package_bytes = fs::metadata(&package_path).unwrap().len();
        let preserved_bytes = fs::metadata(&preserved_path).unwrap().len();
        let spec = Qwen35PackageSourceSpec {
            matrices: &matrices,
            preserved: &preserved,
            package_bytes,
            max_package_bytes: 4096,
            preserved_bytes,
            max_preserved_snapshot_bytes: 4096,
            salt_resident_bytes: salt_resident_bytes(&package_path),
            max_salt_resident_bytes: 1024,
            preserved_fp32_bytes: 8,
            max_preserved_fp32_bytes: 8,
        };
        let error = Qwen35SaltV2PackageSource::from_files(
            File::open(&package_path).unwrap(),
            File::open(&preserved_path).unwrap(),
            &package_id,
            &preserved_id,
            spec,
        )
        .unwrap_err();
        assert!(
            matches!(error, OnnxModelError::InvalidModel(reason) if reason.contains("dtype or shape"))
        );

        let shape = [2];
        let preserved = [Qwen35PackagePreservedSpec {
            name: "norm.weight",
            shape: &shape,
        }];
        let constrained = Qwen35PackageSourceSpec {
            preserved: &preserved,
            max_salt_resident_bytes: 1,
            ..spec
        };
        let error = Qwen35SaltV2PackageSource::from_files(
            File::open(&package_path).unwrap(),
            File::open(&preserved_path).unwrap(),
            &package_id,
            &preserved_id,
            constrained,
        )
        .unwrap_err();
        assert!(
            matches!(error, OnnxModelError::InvalidModel(reason) if reason.contains("aggregate SALT resident bytes"))
        );

        let oversized_snapshot = Qwen35PackageSourceSpec {
            max_preserved_snapshot_bytes: 1,
            ..constrained
        };
        let error = Qwen35SaltV2PackageSource::from_files(
            File::open(package_path).unwrap(),
            File::open(preserved_path).unwrap(),
            &package_id,
            &preserved_id,
            oversized_snapshot,
        )
        .unwrap_err();
        assert!(
            matches!(error, OnnxModelError::InvalidModel(reason) if reason.contains("policy ceilings"))
        );
    }
}

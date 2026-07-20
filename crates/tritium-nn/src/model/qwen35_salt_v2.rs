//! Seek-backed Qwen3.5-family assembly from a SALT V2 matrix package and exact
//! BF16 preserved-tensor companion.

use core::mem::size_of;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use tritium_format::{
    PackageHasher, SafeTensorsReader,
    salt_v2::SaltV2Codec,
    salt_v2_package::{SaltV2PackageReader, SaltV2Transform},
};
use tritium_spec::TernaryBackend;

use crate::NnError;
use crate::QWEN36_27B_REVISION;
use crate::layers::{HostSaltV2Linear, Projection, TokenEmbedding};
use crate::model::qwen35_hf::{
    Qwen35HfTensorSource, TensorRole, TensorSpec, language_schema, load_language_weights,
    load_mtp_weights, mtp_schema,
};
use crate::model::{Qwen35TextRunner, UnverifiedQwen35Mtp};
use crate::qwen35_config::Qwen35CheckpointConfig;

const CONFIG_FILE: &str = "config.json";
const PRESERVED_FILE: &str = "preserved.safetensors";
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;

/// Exact coverage and physical identity consumed by a packed Qwen load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35SaltV2LoadReceipt {
    config_package_id: String,
    package_id: String,
    preserved_package_id: String,
    codec: SaltV2Codec,
    matrix_tensors: usize,
    preserved_tensors: usize,
    serialized_bytes: u64,
    preserved_serialized_bytes: u64,
    salt_resident_bytes: u64,
    preserved_fp32_bytes: u64,
    resident_bytes: u64,
}

impl Qwen35SaltV2LoadReceipt {
    /// Exact-byte identity of the pinned execution configuration.
    #[must_use]
    pub fn config_package_id(&self) -> &str {
        &self.config_package_id
    }

    #[must_use]
    pub fn package_id(&self) -> &str {
        &self.package_id
    }

    /// Exact-byte identity of the preserved safetensors companion.
    #[must_use]
    pub fn preserved_package_id(&self) -> &str {
        &self.preserved_package_id
    }

    #[must_use]
    pub const fn codec(&self) -> SaltV2Codec {
        self.codec
    }

    #[must_use]
    pub const fn matrix_tensors(&self) -> usize {
        self.matrix_tensors
    }

    #[must_use]
    pub const fn preserved_tensors(&self) -> usize {
        self.preserved_tensors
    }

    #[must_use]
    pub const fn serialized_bytes(&self) -> u64 {
        self.serialized_bytes
    }

    /// Exact serialized size of the preserved safetensors companion.
    #[must_use]
    pub const fn preserved_serialized_bytes(&self) -> u64 {
        self.preserved_serialized_bytes
    }

    /// Descriptor-free SALT payload, scale, map, and rank-prefix bytes.
    #[must_use]
    pub const fn salt_resident_bytes(&self) -> u64 {
        self.salt_resident_bytes
    }

    /// Logical fp32 bytes retained after exact BF16 widening.
    #[must_use]
    pub const fn preserved_fp32_bytes(&self) -> u64 {
        self.preserved_fp32_bytes
    }

    /// Total tracked steady weight bytes across SALT and preserved fp32 tensors.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

/// Packed language runner plus structurally loaded MTP graph.
///
/// Language execution is available through [`Self::runner`]. MTP deliberately
/// remains unverified until it passes the pinned serving-oracle promotion gate.
#[allow(missing_debug_implementations)]
pub struct Qwen35SaltV2LanguageMtpModel {
    config: Qwen35CheckpointConfig,
    runner: Qwen35TextRunner,
    mtp: UnverifiedQwen35Mtp,
    receipt: Qwen35SaltV2LoadReceipt,
}

impl Qwen35SaltV2LanguageMtpModel {
    /// Load one profile package from an exported bundle directory.
    ///
    /// The caller selects the canonical profile filename (for example
    /// `compact.tsalt2`). Configuration, matrix, and preserved coverage are
    /// validated exactly before model assembly. Vision tensors remain outside
    /// this language-plus-MTP artifact.
    ///
    /// # Errors
    /// Returns [`NnError`] for non-regular files, malformed configuration or
    /// formats, incomplete/overlapping tensor coverage, wrong shapes or dtypes,
    /// transformed matrices, source mutation, allocation failure, or backend
    /// construction failure.
    pub fn load_bundle_profile(
        bundle_dir: &Path,
        profile_file: &str,
        source_revision: &str,
        expected_config_package_id: &str,
        expected_package_id: &str,
        expected_preserved_package_id: &str,
        backend: Box<dyn TernaryBackend>,
    ) -> Result<Self, NnError> {
        if !matches!(profile_file, "compact.tsalt2" | "near-lossless.tsalt2")
            || source_revision != QWEN36_27B_REVISION
            || expected_config_package_id.is_empty()
            || expected_package_id.is_empty()
            || expected_preserved_package_id.is_empty()
        {
            return Err(NnError::InvalidArtifact(
                "SALT V2 profile filename and package identities must be canonical".into(),
            ));
        }
        let config_bytes = read_bounded_regular(&bundle_dir.join(CONFIG_FILE), MAX_CONFIG_BYTES)?;
        let config_package_id = hash_bytes(&config_bytes);
        if config_package_id != expected_config_package_id {
            return Err(NnError::Provenance(
                "config.json identity differs from bundle manifest".into(),
            ));
        }
        let config_text = std::str::from_utf8(&config_bytes)
            .map_err(|_| NnError::MissingConfig("config.json is not UTF-8".into()))?;
        let config = Qwen35CheckpointConfig::from_hf_config(config_text)?;
        config.validate_pinned_qwen36_27b(source_revision)?;
        let package_file = open_regular(&bundle_dir.join(profile_file), "SALT V2 profile")?;
        let preserved_path = bundle_dir.join(PRESERVED_FILE);
        let preserved_file = open_regular(&preserved_path, "preserved tensors")?;
        let preserved_serialized_bytes = preserved_file
            .metadata()
            .map_err(|error| {
                NnError::InvalidArtifact(format!(
                    "inspect opened preserved tensors {}: {error}",
                    preserved_path.display()
                ))
            })?
            .len();
        let mut preserved_identity_file = preserved_file.try_clone().map_err(|error| {
            NnError::InvalidArtifact(format!(
                "clone preserved tensor handle {}: {error}",
                preserved_path.display()
            ))
        })?;
        let preserved_package_id = hash_file(&mut preserved_identity_file, &preserved_path)?;
        if preserved_package_id != expected_preserved_package_id {
            return Err(NnError::Provenance(
                "preserved tensor identity differs from bundle manifest".into(),
            ));
        }
        let package = SaltV2PackageReader::new_strict(package_file)
            .map_err(|error| NnError::InvalidArtifact(format!("open SALT V2 profile: {error}")))?;
        if package.package_id().to_string() != expected_package_id {
            return Err(NnError::Provenance(
                "SALT V2 profile identity differs from bundle manifest".into(),
            ));
        }
        let preserved = SafeTensorsReader::new(preserved_file).map_err(|error| {
            NnError::InvalidArtifact(format!("open preserved safetensors: {error}"))
        })?;
        let source = SaltV2QwenSource {
            package: RefCell::new(package),
            preserved: RefCell::new(preserved),
        };
        let schema = combined_schema(&config)?;
        let (matrix_tensors, preserved_tensors, preserved_fp32_bytes) =
            validate_coverage(&source, &schema)?;
        let package_ledger = source.package.borrow().ledger();
        let runtime_ledger = source
            .package
            .borrow()
            .indexed_runtime_ledger()
            .map_err(|error| NnError::InvalidArtifact(error.to_string()))?;
        let codec = source.package.borrow().codec();
        let package_id = source.package.borrow().package_id().to_string();

        let language = load_language_weights(&source, &config.text)?;
        let mtp_weights = load_mtp_weights(&source, &config.text)?;
        source
            .package
            .borrow_mut()
            .verify_unchanged()
            .map_err(|error| NnError::Provenance(format!("SALT V2 profile changed: {error}")))?;
        if hash_file(&mut preserved_identity_file, &preserved_path)? != preserved_package_id {
            return Err(NnError::Provenance(
                "preserved tensor companion changed during model load".into(),
            ));
        }
        let runner = Qwen35TextRunner::new(&config.text, language, backend)?;
        let mtp = UnverifiedQwen35Mtp::new(&runner, mtp_weights)?;
        let salt_resident_bytes = runtime_ledger.steady_resident_bytes();
        let resident_bytes = salt_resident_bytes
            .checked_add(preserved_fp32_bytes)
            .ok_or_else(|| {
                NnError::ResourceExhausted("tracked Qwen weight bytes overflow".into())
            })?;
        Ok(Self {
            config,
            runner,
            mtp,
            receipt: Qwen35SaltV2LoadReceipt {
                config_package_id,
                package_id,
                preserved_package_id,
                codec,
                matrix_tensors,
                preserved_tensors,
                serialized_bytes: package_ledger.total_bytes,
                preserved_serialized_bytes,
                salt_resident_bytes,
                preserved_fp32_bytes,
                resident_bytes,
            },
        })
    }

    #[must_use]
    pub const fn config(&self) -> &Qwen35CheckpointConfig {
        &self.config
    }

    #[must_use]
    pub const fn runner(&self) -> &Qwen35TextRunner {
        &self.runner
    }

    #[must_use]
    pub const fn mtp(&self) -> &UnverifiedQwen35Mtp {
        &self.mtp
    }

    #[must_use]
    pub const fn receipt(&self) -> &Qwen35SaltV2LoadReceipt {
        &self.receipt
    }
}

struct SaltV2QwenSource {
    package: RefCell<SaltV2PackageReader<File>>,
    preserved: RefCell<SafeTensorsReader<File>>,
}

impl Qwen35HfTensorSource for SaltV2QwenSource {
    fn tensor_f32_exact(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>, NnError> {
        let mut preserved = self
            .preserved
            .try_borrow_mut()
            .map_err(|_| NnError::Backend("reentrant preserved tensor read".into()))?;
        let shape = preserved
            .shape(name)
            .ok_or_else(|| NnError::MissingTensor(name.to_owned()))?;
        if shape != expected {
            return Err(NnError::Shape {
                expected: expected.iter().copied().product(),
                got: shape.iter().copied().product(),
            });
        }
        if preserved.dtype(name) != Some("BF16") {
            return Err(NnError::InvalidArtifact(format!(
                "preserved tensor `{name}` is not exact BF16"
            )));
        }
        preserved
            .tensor_f32(name)
            .map_err(|error| NnError::InvalidArtifact(format!("read `{name}`: {error}")))
    }

    fn projection_exact(
        &self,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Projection, NnError> {
        let mut package = self
            .package
            .try_borrow_mut()
            .map_err(|_| NnError::Backend("reentrant SALT V2 matrix read".into()))?;
        let matrix = HostSaltV2Linear::from_reader(&mut package, name)?;
        require_matrix_geometry(name, &matrix, rows, columns)?;
        Ok(Projection::HostSaltV2(Arc::new(matrix)))
    }

    fn token_embedding_exact(
        &self,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<TokenEmbedding, NnError> {
        let mut package = self
            .package
            .try_borrow_mut()
            .map_err(|_| NnError::Backend("reentrant SALT V2 token-table read".into()))?;
        let matrix = Arc::new(HostSaltV2Linear::from_reader(&mut package, name)?);
        require_matrix_geometry(name, &matrix, rows, columns)?;
        TokenEmbedding::from_host_salt_v2(matrix)
    }
}

fn require_matrix_geometry(
    name: &str,
    matrix: &HostSaltV2Linear,
    rows: usize,
    columns: usize,
) -> Result<(), NnError> {
    if matrix.rows() != rows || matrix.columns() != columns {
        return Err(NnError::InvalidArtifact(format!(
            "SALT V2 matrix `{name}` has geometry {}x{}, expected {rows}x{columns}",
            matrix.rows(),
            matrix.columns()
        )));
    }
    Ok(())
}

fn combined_schema(
    config: &Qwen35CheckpointConfig,
) -> Result<BTreeMap<String, TensorSpec>, NnError> {
    let mut schema = language_schema(&config.text)?;
    for (name, spec) in mtp_schema(&config.text)? {
        if schema.insert(name.clone(), spec).is_some() {
            return Err(NnError::InvalidArtifact(format!(
                "duplicate language/MTP schema tensor `{name}`"
            )));
        }
    }
    Ok(schema)
}

fn validate_coverage(
    source: &SaltV2QwenSource,
    schema: &BTreeMap<String, TensorSpec>,
) -> Result<(usize, usize, u64), NnError> {
    let package = source.package.borrow();
    let preserved = source.preserved.borrow();
    let matrix_names = package.tensor_names().collect::<BTreeSet<_>>();
    let preserved_names = preserved.names().collect::<BTreeSet<_>>();
    let mut preserved_fp32_bytes = 0u64;
    for (name, spec) in schema {
        match spec.role {
            TensorRole::Matrix => {
                if preserved_names.contains(name.as_str()) || !matrix_names.contains(name.as_str())
                {
                    return Err(NnError::InvalidArtifact(format!(
                        "matrix `{name}` must occur exactly once in the SALT V2 profile"
                    )));
                }
                let info = package
                    .tensor_info(name)
                    .ok_or_else(|| NnError::MissingTensor(name.clone()))?;
                let expected = spec
                    .shape
                    .iter()
                    .copied()
                    .map(|dim| u64::try_from(dim).unwrap_or(u64::MAX))
                    .collect::<Vec<_>>();
                if info.dims() != expected || info.transform() != SaltV2Transform::None {
                    return Err(NnError::InvalidArtifact(format!(
                        "matrix `{name}` shape or transform differs from Qwen schema"
                    )));
                }
            }
            TensorRole::Preserved => {
                if matrix_names.contains(name.as_str())
                    || !preserved_names.contains(name.as_str())
                    || preserved.dtype(name) != Some("BF16")
                    || preserved.shape(name) != Some(spec.shape.as_slice())
                {
                    return Err(NnError::InvalidArtifact(format!(
                        "preserved tensor `{name}` must occur exactly once as exact BF16"
                    )));
                }
                let elements = spec.shape.iter().try_fold(1u64, |product, dimension| {
                    product.checked_mul(u64::try_from(*dimension).ok()?)
                });
                let bytes = elements
                    .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
                    .ok_or_else(|| {
                        NnError::ResourceExhausted(format!(
                            "preserved tensor `{name}` fp32 byte count overflows"
                        ))
                    })?;
                preserved_fp32_bytes =
                    preserved_fp32_bytes.checked_add(bytes).ok_or_else(|| {
                        NnError::ResourceExhausted("preserved Qwen fp32 bytes overflow".into())
                    })?;
            }
        }
    }
    if let Some(extra) = matrix_names.iter().find(|name| {
        !schema
            .get(**name)
            .is_some_and(|spec| spec.role == TensorRole::Matrix)
    }) {
        return Err(NnError::InvalidArtifact(format!(
            "unexpected SALT V2 matrix `{extra}`"
        )));
    }
    if let Some(extra) = preserved_names.iter().find(|name| {
        !schema
            .get(**name)
            .is_some_and(|spec| spec.role == TensorRole::Preserved)
    }) {
        return Err(NnError::InvalidArtifact(format!(
            "unexpected preserved tensor `{extra}`"
        )));
    }
    Ok((
        matrix_names.len(),
        preserved_names.len(),
        preserved_fp32_bytes,
    ))
}

fn hash_file(file: &mut File, path: &Path) -> Result<String, NnError> {
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        NnError::InvalidArtifact(format!("seek {} for identity: {error}", path.display()))
    })?;
    let mut hasher = PackageHasher::new();
    let mut scratch = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut scratch).map_err(|error| {
            NnError::InvalidArtifact(format!("hash {}: {error}", path.display()))
        })?;
        if count == 0 {
            break;
        }
        hasher.update(&scratch[..count]);
    }
    Ok(hasher.finalize().to_string())
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = PackageHasher::new();
    hasher.update(bytes);
    hasher.finalize().to_string()
}

fn open_regular(path: &Path, description: &str) -> Result<File, NnError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        NnError::InvalidArtifact(format!("inspect {description} {}: {error}", path.display()))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(NnError::InvalidArtifact(format!(
            "{description} must be an ordinary non-symlink file"
        )));
    }
    File::open(path).map_err(|error| {
        NnError::InvalidArtifact(format!("open {description} {}: {error}", path.display()))
    })
}

fn read_bounded_regular(path: &Path, max_bytes: u64) -> Result<Vec<u8>, NnError> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| NnError::MissingConfig(format!("inspect {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > max_bytes {
        return Err(NnError::MissingConfig(
            "config.json must be an ordinary file no larger than 1 MiB".into(),
        ));
    }
    fs::read(path)
        .map_err(|error| NnError::MissingConfig(format!("read {}: {error}", path.display())))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use half::f16;
    use tritium_format::{
        PackageId,
        salt_v2_package::{
            SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, write_salt_v2_package,
        },
    };

    use super::*;

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestFiles {
        directory: std::path::PathBuf,
    }

    impl TestFiles {
        fn new() -> Self {
            let directory = std::env::temp_dir().join(format!(
                "tritium-qwen-salt-source-{}-{}",
                std::process::id(),
                TEST_COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&directory).unwrap();
            Self { directory }
        }

        fn source(&self) -> SaltV2QwenSource {
            let plane = SaltV2Plane::new(vec![1, 0, -1, 1], vec![f16::from_f32(0.5)]).unwrap();
            let tensor = SaltV2Tensor::new(
                "matrix",
                vec![2, 2],
                vec![SaltV2Tile::new(vec![plane]).unwrap()],
            )
            .unwrap();
            let package = SaltV2Package::new(SaltV2Codec::B3, vec![tensor]).unwrap();
            let encoded = write_salt_v2_package(&package).unwrap();
            let package_path = self.directory.join("profile.tsalt2");
            fs::write(&package_path, encoded.bytes).unwrap();
            let preserved_path = self.directory.join("preserved.safetensors");
            fs::write(&preserved_path, safetensors()).unwrap();
            SaltV2QwenSource {
                package: RefCell::new(
                    SaltV2PackageReader::new_strict(File::open(package_path).unwrap()).unwrap(),
                ),
                preserved: RefCell::new(
                    SafeTensorsReader::new(File::open(preserved_path).unwrap()).unwrap(),
                ),
            }
        }
    }

    impl Drop for TestFiles {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.directory).unwrap();
        }
    }

    fn safetensors() -> Vec<u8> {
        let mut header = br#"{"__metadata__":{"format":"pt"},"norm":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&0x3f80_u16.to_le_bytes());
        bytes.extend_from_slice(&0xc000_u16.to_le_bytes());
        bytes
    }

    fn schema() -> BTreeMap<String, TensorSpec> {
        BTreeMap::from([
            (
                "matrix".to_owned(),
                TensorSpec {
                    shape: vec![2, 2],
                    role: TensorRole::Matrix,
                },
            ),
            (
                "norm".to_owned(),
                TensorSpec {
                    shape: vec![2],
                    role: TensorRole::Preserved,
                },
            ),
        ])
    }

    #[test]
    fn split_source_enforces_coverage_and_builds_packed_values() {
        let files = TestFiles::new();
        let source = files.source();
        assert_eq!(validate_coverage(&source, &schema()).unwrap(), (1, 1, 8));

        let projection = source.projection_exact("matrix", 2, 2).unwrap();
        assert!(matches!(projection, Projection::HostSaltV2(_)));
        let embedding = source.token_embedding_exact("matrix", 2, 2).unwrap();
        assert!(embedding.is_packed_salt());
        assert_eq!(source.tensor_f32_exact("norm", &[2]).unwrap(), [1.0, -2.0]);
        source.package.borrow_mut().verify_unchanged().unwrap();
    }

    #[test]
    fn split_source_rejects_role_and_shape_drift_before_assembly() {
        let files = TestFiles::new();
        let source = files.source();
        let mut wrong_role = schema();
        wrong_role.get_mut("matrix").unwrap().role = TensorRole::Preserved;
        assert!(matches!(
            validate_coverage(&source, &wrong_role),
            Err(NnError::InvalidArtifact(_))
        ));

        let mut wrong_shape = schema();
        wrong_shape.get_mut("matrix").unwrap().shape = vec![1, 4];
        assert!(matches!(
            validate_coverage(&source, &wrong_shape),
            Err(NnError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn preserved_reader_rejects_shape_and_dtype_substitution() {
        let files = TestFiles::new();
        let source = files.source();
        assert!(matches!(
            source.tensor_f32_exact("norm", &[1, 2]),
            Err(NnError::Shape { .. })
        ));
        assert!(matches!(
            source.tensor_f32_exact("missing", &[2]),
            Err(NnError::MissingTensor(_))
        ));
    }

    #[test]
    fn complete_bundle_loads_language_and_mtp_without_dense_matrix_shadows() {
        let files = TestFiles::new();
        let config_json = family_config_json();
        fs::write(files.directory.join(CONFIG_FILE), &config_json).unwrap();
        let config = Qwen35CheckpointConfig::from_hf_config(&config_json).unwrap();
        let schema = combined_schema(&config).unwrap();

        let mut matrices = Vec::new();
        let mut preserved = BTreeMap::new();
        for (name, spec) in &schema {
            match spec.role {
                TensorRole::Matrix => matrices.push(zero_matrix(name, &spec.shape)),
                TensorRole::Preserved => {
                    preserved.insert(name.clone(), spec.shape.clone());
                }
            }
        }
        let matrix_count = matrices.len();
        let encoded =
            write_salt_v2_package(&SaltV2Package::new(SaltV2Codec::B3, matrices).unwrap()).unwrap();
        let profile_id = PackageId::from_package_bytes(&encoded.bytes).to_string();
        fs::write(files.directory.join("compact.tsalt2"), encoded.bytes).unwrap();
        let preserved_bytes = zero_bf16_safetensors(&preserved);
        let preserved_id = PackageId::from_package_bytes(&preserved_bytes).to_string();
        fs::write(files.directory.join(PRESERVED_FILE), preserved_bytes).unwrap();

        let package = SaltV2PackageReader::new_strict(
            File::open(files.directory.join("compact.tsalt2")).unwrap(),
        )
        .unwrap();
        assert_eq!(package.package_id().to_string(), profile_id);
        let preserved_reader =
            SafeTensorsReader::new(File::open(files.directory.join(PRESERVED_FILE)).unwrap())
                .unwrap();
        let source = SaltV2QwenSource {
            package: RefCell::new(package),
            preserved: RefCell::new(preserved_reader),
        };
        let (loaded_matrices, loaded_preserved, preserved_fp32_bytes) =
            validate_coverage(&source, &schema).unwrap();
        assert_eq!(loaded_matrices, matrix_count);
        assert_eq!(loaded_preserved, schema.len() - matrix_count);
        assert!(preserved_fp32_bytes > 0);
        assert_eq!(
            hash_bytes(&fs::read(files.directory.join(PRESERVED_FILE)).unwrap()),
            preserved_id
        );

        let language = load_language_weights(&source, &config.text).unwrap();
        let mtp_weights = load_mtp_weights(&source, &config.text).unwrap();
        source.package.borrow_mut().verify_unchanged().unwrap();
        let runner = Qwen35TextRunner::new(
            &config.text,
            language,
            Box::new(tritium_cpu::CpuBackend::new()),
        )
        .unwrap();
        let _mtp = UnverifiedQwen35Mtp::new(&runner, mtp_weights).unwrap();
        let mut cache = runner.new_cache(4).unwrap();
        let output = runner.forward(&[1, 2], &mut cache).unwrap();
        assert_eq!(output.last_logits(), &[0.0; 7]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn bundle_loader_rejects_parent_traversal_before_opening_files() {
        let files = TestFiles::new();
        assert!(matches!(
            Qwen35SaltV2LanguageMtpModel::load_bundle_profile(
                &files.directory,
                "..",
                QWEN36_27B_REVISION,
                "trp1_not-used",
                "trp1_not-used",
                "trp1_not-used",
                Box::new(tritium_cpu::CpuBackend::new()),
            ),
            Err(NnError::InvalidArtifact(_))
        ));
    }

    fn zero_matrix(name: &str, shape: &[usize]) -> SaltV2Tensor {
        let coefficients = shape.iter().product::<usize>();
        let mut tiles = Vec::new();
        let mut remaining = coefficients;
        while remaining != 0 {
            let logical_len = remaining.min(256);
            let plane = SaltV2Plane::new(
                vec![0; logical_len],
                vec![f16::ZERO; logical_len.div_ceil(128)],
            )
            .unwrap();
            tiles.push(SaltV2Tile::new(vec![plane]).unwrap());
            remaining -= logical_len;
        }
        SaltV2Tensor::new(
            name,
            shape.iter().map(|dimension| *dimension as u64).collect(),
            tiles,
        )
        .unwrap()
    }

    fn zero_bf16_safetensors(tensors: &BTreeMap<String, Vec<usize>>) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        let mut offset = 0usize;
        for (name, shape) in tensors {
            let bytes = shape.iter().product::<usize>() * size_of::<u16>();
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut encoded_header = serde_json::to_vec(&header).unwrap();
        while !encoded_header.len().is_multiple_of(8) {
            encoded_header.push(b' ');
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(encoded_header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&encoded_header);
        bytes.resize(bytes.len() + offset, 0);
        bytes
    }

    fn family_config_json() -> String {
        serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "language_model_only": false,
            "model_type": "qwen3_5",
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attn_output_gate": true,
                "dtype": "bfloat16",
                "full_attention_interval": 2,
                "head_dim": 4,
                "hidden_act": "silu",
                "hidden_size": 4,
                "intermediate_size": 6,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 2,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 2,
                "linear_value_head_dim": 2,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 32,
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "num_key_value_heads": 1,
                "output_gate_type": "swish",
                "partial_rotary_factor": 0.5,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [1, 0, 0],
                    "partial_rotary_factor": 0.5,
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                },
                "tie_word_embeddings": false,
                "use_cache": true,
                "vocab_size": 7
            },
            "tie_word_embeddings": false,
            "vision_config": {"model_type": "qwen3_5"}
        })
        .to_string()
    }
}

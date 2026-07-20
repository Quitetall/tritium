//! Python boundary for production SALT V2 campaign and package orchestration.

use std::{
    collections::BTreeSet,
    fmt::Write as _,
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use tritium_format::{PackageHasher, salt_v2::SaltV2Codec, salt_v2_package::SaltV2PackageReader};
use tritium_quantize::{PhysicalBytes, SaltV2Config, SaltV2Curvature, SaltV2Packing};
use tritium_salt::{
    ContentId, Qwen36AdmittedSource, Qwen36CompleteWorkspaceReceipt,
    Qwen36PreservedSafetensorsReceipt, Qwen36PtqEvidenceDirectory, Qwen36PtqPackageLimits,
    Qwen36PtqPackagesReceipt, Qwen36TensorWorkStore,
};

const COMPACT_PACKAGE_FILE: &str = "compact.tsalt2";
const NEAR_LOSSLESS_PACKAGE_FILE: &str = "near-lossless.tsalt2";
const PRESERVED_TENSORS_FILE: &str = "preserved.safetensors";
const BUNDLE_MANIFEST_FILE: &str = "tritium.json";
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 1024 * 1024;
static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Immutable receipt for one sealed Qwen3.6 rate-free PTQ master campaign.
///
/// This is not a deployable model receipt: physical profile allocation, final
/// package assembly, and model export occur after this master-campaign gate.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct Qwen36PtqMasterReceipt {
    completion_id: String,
    base_workspace_id: String,
    campaign_id: String,
    master_set_id: String,
    source_model_id: String,
    source_identity_status: String,
    official_payload_authenticated: bool,
    additive_tensors: u64,
    additive_coefficients: u64,
    preserved_tensors: u64,
    preserved_payload_bytes: u64,
    completion_bytes: u64,
}

impl From<Qwen36CompleteWorkspaceReceipt> for Qwen36PtqMasterReceipt {
    fn from(receipt: Qwen36CompleteWorkspaceReceipt) -> Self {
        let summary = receipt.summary();
        let identity = receipt.identity_status();
        Self {
            completion_id: receipt.completion_id().to_string(),
            base_workspace_id: receipt.base_workspace_id().to_string(),
            campaign_id: receipt.campaign_id().to_string(),
            master_set_id: hex_digest(&receipt.master_set_id()),
            source_model_id: hex_digest(receipt.source_model_id().as_bytes()),
            source_identity_status: identity.as_str().to_owned(),
            official_payload_authenticated: identity.official_payload_authenticated(),
            additive_tensors: summary.additive_present(),
            additive_coefficients: receipt.additive_coefficients(),
            preserved_tensors: summary.preserved_tensors(),
            preserved_payload_bytes: summary.preserved_payload_bytes(),
            completion_bytes: receipt.completion_bytes(),
        }
    }
}

#[pymethods]
impl Qwen36PtqMasterReceipt {
    /// Content identity of exact canonical completion-seal bytes.
    #[getter]
    fn completion_id(&self) -> &str {
        &self.completion_id
    }

    /// Content identity of the immutable exact-BF16 base workspace.
    #[getter]
    fn base_workspace_id(&self) -> &str {
        &self.base_workspace_id
    }

    /// Base-bound additive campaign identity.
    #[getter]
    fn campaign_id(&self) -> &str {
        &self.campaign_id
    }

    /// Hex digest over all ordered canonical tensor masters.
    #[getter]
    fn master_set_id(&self) -> &str {
        &self.master_set_id
    }

    /// Hex semantic identity of the admitted source model.
    #[getter]
    fn source_model_id(&self) -> &str {
        &self.source_model_id
    }

    /// Stable source-authentication status label.
    #[getter]
    fn source_identity_status(&self) -> &str {
        &self.source_identity_status
    }

    /// Whether the exact payload was matched to an independently audited official identity.
    #[getter]
    fn official_payload_authenticated(&self) -> bool {
        self.official_payload_authenticated
    }

    /// Canonical additive tensor masters sealed by this campaign.
    #[getter]
    fn additive_tensors(&self) -> u64 {
        self.additive_tensors
    }

    /// Exact source coefficients represented by the additive master set.
    #[getter]
    fn additive_coefficients(&self) -> u64 {
        self.additive_coefficients
    }

    /// Exact-BF16 language/MTP tensors retained outside the additive set.
    #[getter]
    fn preserved_tensors(&self) -> u64 {
        self.preserved_tensors
    }

    /// Exact raw BF16 payload bytes retained by the base workspace.
    #[getter]
    fn preserved_payload_bytes(&self) -> u64 {
        self.preserved_payload_bytes
    }

    /// Canonical completion-seal bytes, excluding referenced master objects.
    #[getter]
    fn completion_bytes(&self) -> u64 {
        self.completion_bytes
    }

    fn __repr__(&self) -> String {
        format!(
            "Qwen36PtqMasterReceipt(additive_tensors={}, additive_coefficients={}, campaign_id='{}')",
            self.additive_tensors, self.additive_coefficients, self.campaign_id
        )
    }
}

/// Immutable receipt for atomically exported SALT profiles and preserved tensors.
///
/// The package pair contains quantized language/MTP matrices and the safetensors
/// companion contains every exact preserved BF16 language/MTP tensor. Model
/// configuration and tokenizer assets remain a later governed export step.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct Qwen36PtqPackageReceipt {
    artifact_dir: String,
    completion_id: String,
    campaign_id: String,
    admission_id: String,
    selection_id: String,
    source_model_id: String,
    source_identity_status: String,
    official_payload_authenticated: bool,
    compact_package_id: String,
    compact_serialized_bytes: u64,
    compact_resident_bytes: u64,
    near_lossless_package_id: String,
    near_lossless_serialized_bytes: u64,
    near_lossless_resident_bytes: u64,
    preserved_package_id: String,
    preserved_tensor_count: u64,
    preserved_header_bytes: u64,
    preserved_payload_bytes: u64,
    preserved_total_bytes: u64,
}

impl Qwen36PtqPackageReceipt {
    fn from_native(
        artifact_dir: &Path,
        receipt: &Qwen36PtqPackagesReceipt,
        preserved: Qwen36PreservedSafetensorsReceipt,
    ) -> Self {
        let completion = receipt.completion();
        let admission = receipt.admission();
        let identity = completion.identity_status();
        let compact = admission.compact();
        let near = admission.near_lossless();
        Self {
            artifact_dir: artifact_dir.to_string_lossy().into_owned(),
            completion_id: completion.completion_id().to_string(),
            campaign_id: completion.campaign_id().to_string(),
            admission_id: admission.admission_id().to_string(),
            selection_id: admission.selection_id().to_string(),
            source_model_id: hex_digest(completion.source_model_id().as_bytes()),
            source_identity_status: identity.as_str().to_owned(),
            official_payload_authenticated: identity.official_payload_authenticated(),
            compact_package_id: compact.package_id().to_string(),
            compact_serialized_bytes: compact.physical_bytes().serialized,
            compact_resident_bytes: compact.physical_bytes().resident,
            near_lossless_package_id: near.package_id().to_string(),
            near_lossless_serialized_bytes: near.physical_bytes().serialized,
            near_lossless_resident_bytes: near.physical_bytes().resident,
            preserved_package_id: preserved.package_id().to_string(),
            preserved_tensor_count: preserved.tensor_count(),
            preserved_header_bytes: preserved.header_bytes(),
            preserved_payload_bytes: preserved.payload_bytes(),
            preserved_total_bytes: preserved.total_bytes(),
        }
    }

    fn manifest_bytes(&self, packing: &str) -> Result<Vec<u8>, String> {
        let value = serde_json::json!({
            "schema_version": 2,
            "artifact_kind": "qwen3.6-language-mtp-salt-v2-model-weights",
            "complete_model": false,
            "packing": packing,
            "completion_id": self.completion_id,
            "campaign_id": self.campaign_id,
            "admission_id": self.admission_id,
            "selection_id": self.selection_id,
            "source_model_id": self.source_model_id,
            "source_identity_status": self.source_identity_status,
            "official_payload_authenticated": self.official_payload_authenticated,
            "preserved": {
                "file": PRESERVED_TENSORS_FILE,
                "package_id": self.preserved_package_id,
                "tensors": self.preserved_tensor_count,
                "payload_bytes": self.preserved_payload_bytes,
                "serialized_bytes": self.preserved_total_bytes,
            },
            "profiles": {
                "compact-v1": {
                    "file": COMPACT_PACKAGE_FILE,
                    "package_id": self.compact_package_id,
                    "serialized_bytes": self.compact_serialized_bytes,
                    "resident_bytes": self.compact_resident_bytes,
                },
                "near-lossless-v1": {
                    "file": NEAR_LOSSLESS_PACKAGE_FILE,
                    "package_id": self.near_lossless_package_id,
                    "serialized_bytes": self.near_lossless_serialized_bytes,
                    "resident_bytes": self.near_lossless_resident_bytes,
                },
            },
        });
        let mut bytes = serde_json::to_vec_pretty(&value).map_err(|error| error.to_string())?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

#[pymethods]
impl Qwen36PtqPackageReceipt {
    /// Atomically published directory containing both packages and its manifest.
    #[getter]
    fn artifact_dir(&self) -> &str {
        &self.artifact_dir
    }

    #[getter]
    fn completion_id(&self) -> &str {
        &self.completion_id
    }

    #[getter]
    fn campaign_id(&self) -> &str {
        &self.campaign_id
    }

    #[getter]
    fn admission_id(&self) -> &str {
        &self.admission_id
    }

    #[getter]
    fn selection_id(&self) -> &str {
        &self.selection_id
    }

    #[getter]
    fn source_model_id(&self) -> &str {
        &self.source_model_id
    }

    #[getter]
    fn source_identity_status(&self) -> &str {
        &self.source_identity_status
    }

    #[getter]
    fn official_payload_authenticated(&self) -> bool {
        self.official_payload_authenticated
    }

    #[getter]
    fn compact_path(&self) -> PathBuf {
        Path::new(&self.artifact_dir).join(COMPACT_PACKAGE_FILE)
    }

    #[getter]
    fn compact_package_id(&self) -> &str {
        &self.compact_package_id
    }

    #[getter]
    fn compact_serialized_bytes(&self) -> u64 {
        self.compact_serialized_bytes
    }

    #[getter]
    fn compact_resident_bytes(&self) -> u64 {
        self.compact_resident_bytes
    }

    #[getter]
    fn near_lossless_path(&self) -> PathBuf {
        Path::new(&self.artifact_dir).join(NEAR_LOSSLESS_PACKAGE_FILE)
    }

    #[getter]
    fn near_lossless_package_id(&self) -> &str {
        &self.near_lossless_package_id
    }

    #[getter]
    fn near_lossless_serialized_bytes(&self) -> u64 {
        self.near_lossless_serialized_bytes
    }

    #[getter]
    fn near_lossless_resident_bytes(&self) -> u64 {
        self.near_lossless_resident_bytes
    }

    #[getter]
    fn preserved_path(&self) -> PathBuf {
        Path::new(&self.artifact_dir).join(PRESERVED_TENSORS_FILE)
    }

    #[getter]
    fn preserved_package_id(&self) -> &str {
        &self.preserved_package_id
    }

    #[getter]
    fn preserved_tensor_count(&self) -> u64 {
        self.preserved_tensor_count
    }

    #[getter]
    fn preserved_header_bytes(&self) -> u64 {
        self.preserved_header_bytes
    }

    #[getter]
    fn preserved_payload_bytes(&self) -> u64 {
        self.preserved_payload_bytes
    }

    #[getter]
    fn preserved_total_bytes(&self) -> u64 {
        self.preserved_total_bytes
    }

    fn __repr__(&self) -> String {
        format!(
            "Qwen36PtqPackageReceipt(artifact_dir='{}', admission_id='{}')",
            self.artifact_dir, self.admission_id
        )
    }
}

/// Reconcile the pinned Qwen3.6 checkpoint into a sealed rate-free PTQ master campaign.
///
/// `evidence_dir` must contain exactly `000000.s2kf` through `000505.s2kf`.
/// Source preflight, widening, fitting, store validation, and filesystem I/O run
/// with the GIL released. The result is structural master evidence, not a final
/// allocated/exported model.
#[pyfunction]
#[pyo3(signature = (model_dir, declared_revision, work_dir, evidence_dir, *, packing = "b3", max_evidence_bytes = 67_108_864))]
pub(crate) fn reconcile_qwen36_ptq_masters(
    py: Python<'_>,
    model_dir: &str,
    declared_revision: &str,
    work_dir: &str,
    evidence_dir: &str,
    packing: &str,
    max_evidence_bytes: u64,
) -> PyResult<Qwen36PtqMasterReceipt> {
    for (field, value) in [
        ("model_dir", model_dir),
        ("declared_revision", declared_revision),
        ("work_dir", work_dir),
        ("evidence_dir", evidence_dir),
    ] {
        if value.is_empty() {
            return Err(PyValueError::new_err(format!("{field} must not be empty")));
        }
    }
    if declared_revision != tritium_nn::QWEN36_27B_REVISION {
        return Err(PyValueError::new_err(format!(
            "declared_revision must equal the pinned Qwen3.6 revision {}",
            tritium_nn::QWEN36_27B_REVISION
        )));
    }
    if max_evidence_bytes == 0 {
        return Err(PyValueError::new_err("max_evidence_bytes must be positive"));
    }
    let packing = match packing {
        "d2" => SaltV2Packing::D2,
        "b3" => SaltV2Packing::B3,
        "s34" => SaltV2Packing::S34,
        _ => {
            return Err(PyValueError::new_err(
                "packing must be one of 'd2', 'b3', or 's34'",
            ));
        }
    };
    let model_dir = model_dir.to_owned();
    let declared_revision = declared_revision.to_owned();
    let work_dir = work_dir.to_owned();
    let evidence_dir = evidence_dir.to_owned();

    py.detach(move || {
        let evidence = Qwen36PtqEvidenceDirectory::open_bounded(&evidence_dir, max_evidence_bytes)
            .map_err(|error| error.to_string())?;
        let curvature = evidence
            .reopen(0)
            .map_err(|error| error.to_string())?
            .kind();
        let admitted =
            Qwen36AdmittedSource::open(model_dir.as_ref(), &declared_revision, work_dir.as_ref())
                .map_err(|error| error.to_string())?;
        let config = SaltV2Config {
            packing,
            curvature,
            ..SaltV2Config::default()
        };
        tritium_salt::reconcile_qwen36_ptq(&admitted, &evidence, &config)
            .map(Qwen36PtqMasterReceipt::from)
            .map_err(|error| error.to_string())
    })
    .map_err(PyRuntimeError::new_err)
}

/// Reconcile and atomically publish both exact Qwen3.6 matrix-package profiles.
#[pyfunction]
#[pyo3(signature = (
    model_dir,
    declared_revision,
    work_dir,
    evidence_dir,
    output_dir,
    *,
    compact_max_bytes,
    compact_max_resident_bytes,
    near_lossless_max_bytes,
    near_lossless_max_resident_bytes,
    packing = "b3",
    max_evidence_bytes = 67_108_864
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconcile_qwen36_ptq_packages(
    py: Python<'_>,
    model_dir: &str,
    declared_revision: &str,
    work_dir: &str,
    evidence_dir: &str,
    output_dir: &str,
    compact_max_bytes: u64,
    compact_max_resident_bytes: u64,
    near_lossless_max_bytes: u64,
    near_lossless_max_resident_bytes: u64,
    packing: &str,
    max_evidence_bytes: u64,
) -> PyResult<Qwen36PtqPackageReceipt> {
    for (field, value) in [
        ("model_dir", model_dir),
        ("declared_revision", declared_revision),
        ("work_dir", work_dir),
        ("evidence_dir", evidence_dir),
        ("output_dir", output_dir),
    ] {
        if value.is_empty() {
            return Err(PyValueError::new_err(format!("{field} must not be empty")));
        }
    }
    if declared_revision != tritium_nn::QWEN36_27B_REVISION {
        return Err(PyValueError::new_err(format!(
            "declared_revision must equal the pinned Qwen3.6 revision {}",
            tritium_nn::QWEN36_27B_REVISION
        )));
    }
    if max_evidence_bytes == 0 {
        return Err(PyValueError::new_err("max_evidence_bytes must be positive"));
    }
    let limits = Qwen36PtqPackageLimits::new(
        PhysicalBytes {
            serialized: compact_max_bytes,
            resident: compact_max_resident_bytes,
        },
        PhysicalBytes {
            serialized: near_lossless_max_bytes,
            resident: near_lossless_max_resident_bytes,
        },
    );
    if compact_max_bytes == 0
        || compact_max_resident_bytes == 0
        || near_lossless_max_bytes == 0
        || near_lossless_max_resident_bytes == 0
        || compact_max_bytes > near_lossless_max_bytes
        || compact_max_resident_bytes > near_lossless_max_resident_bytes
    {
        return Err(PyValueError::new_err(
            "package ceilings must be positive and componentwise nested",
        ));
    }
    let packing_value = parse_packing(packing)?;
    let packing_label = packing.to_owned();
    let model_dir = PathBuf::from(model_dir);
    let declared_revision = declared_revision.to_owned();
    let work_dir = PathBuf::from(work_dir);
    let evidence_dir = PathBuf::from(evidence_dir);
    let output_dir = PathBuf::from(output_dir);

    py.detach(move || {
        validate_output_location(&output_dir, [&model_dir, &work_dir, &evidence_dir])?;
        let evidence = Qwen36PtqEvidenceDirectory::open_bounded(&evidence_dir, max_evidence_bytes)
            .map_err(|error| error.to_string())?;
        let curvature = evidence
            .reopen(0)
            .map_err(|error| error.to_string())?
            .kind();
        let admitted = Qwen36AdmittedSource::open(&model_dir, &declared_revision, &work_dir)
            .map_err(|error| error.to_string())?;
        let config = SaltV2Config {
            packing: packing_value,
            curvature,
            ..SaltV2Config::default()
        };
        publish_package_directory(
            &output_dir,
            |compact, near, preserved_output| {
                let native = tritium_salt::reconcile_qwen36_ptq_packages(
                    &admitted, &evidence, &config, limits, compact, near,
                )
                .map_err(|error| error.to_string())?;
                let workspace =
                    Qwen36TensorWorkStore::open(&admitted).map_err(|error| error.to_string())?;
                let preserved = workspace
                    .try_write_preserved_safetensors(64 * 1024, |chunk| {
                        preserved_output.write_all(chunk)
                    })
                    .map_err(|error| error.to_string())?;
                let receipt = Qwen36PtqPackageReceipt::from_native(&output_dir, &native, preserved);
                let manifest = receipt.manifest_bytes(&packing_label)?;
                Ok((receipt, manifest))
            },
            |staging, receipt| validate_staged_packages(staging, receipt, packing_value),
        )
    })
    .map_err(PyRuntimeError::new_err)
}

fn parse_packing(packing: &str) -> PyResult<SaltV2Packing> {
    match packing {
        "d2" => Ok(SaltV2Packing::D2),
        "b3" => Ok(SaltV2Packing::B3),
        "s34" => Ok(SaltV2Packing::S34),
        _ => Err(PyValueError::new_err(
            "packing must be one of 'd2', 'b3', or 's34'",
        )),
    }
}

fn validate_output_location<'a>(
    output: &Path,
    protected: impl IntoIterator<Item = &'a PathBuf>,
) -> Result<(), String> {
    let name = output
        .file_name()
        .ok_or_else(|| "output_dir must name a new directory".to_owned())?;
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent)
        .map_err(|error| format!("canonicalize output parent failed: {:?}", error.kind()))?;
    let output = parent.join(name);
    match fs::symlink_metadata(&output) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!("inspect output_dir failed: {:?}", error.kind()));
        }
        Ok(_) => return Err("output_dir already exists".to_owned()),
    }
    for path in protected {
        if let Ok(protected) = fs::canonicalize(path)
            && output.starts_with(&protected)
        {
            return Err("output_dir must be outside source, work, and evidence trees".to_owned());
        }
    }
    Ok(())
}

fn publish_package_directory<R>(
    output: &Path,
    produce: impl FnOnce(&mut File, &mut File, &mut File) -> Result<(R, Vec<u8>), String>,
    validate: impl FnOnce(&Path, &R) -> Result<(), String>,
) -> Result<R, String> {
    let name = output
        .file_name()
        .ok_or_else(|| "output_dir must name a new directory".to_owned())?;
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent)
        .map_err(|error| format!("canonicalize output parent failed: {:?}", error.kind()))?;
    let output = parent.join(name);
    if fs::symlink_metadata(&output).is_ok() {
        return Err("output_dir already exists".to_owned());
    }
    let staging = create_staging_directory(&parent)?;
    let result = (|| {
        let mut compact = File::create(staging.join(COMPACT_PACKAGE_FILE))
            .map_err(|error| format!("create compact output failed: {:?}", error.kind()))?;
        let mut near = File::create(staging.join(NEAR_LOSSLESS_PACKAGE_FILE))
            .map_err(|error| format!("create near-lossless output failed: {:?}", error.kind()))?;
        let mut preserved = File::create(staging.join(PRESERVED_TENSORS_FILE))
            .map_err(|error| format!("create preserved output failed: {:?}", error.kind()))?;
        let (receipt, manifest) = produce(&mut compact, &mut near, &mut preserved)?;
        compact
            .sync_all()
            .map_err(|error| format!("sync compact output failed: {:?}", error.kind()))?;
        near.sync_all()
            .map_err(|error| format!("sync near-lossless output failed: {:?}", error.kind()))?;
        preserved
            .sync_all()
            .map_err(|error| format!("sync preserved output failed: {:?}", error.kind()))?;
        let mut manifest_file = File::create(staging.join(BUNDLE_MANIFEST_FILE))
            .map_err(|error| format!("create bundle manifest failed: {:?}", error.kind()))?;
        manifest_file
            .write_all(&manifest)
            .map_err(|error| format!("write bundle manifest failed: {:?}", error.kind()))?;
        manifest_file
            .sync_all()
            .map_err(|error| format!("sync bundle manifest failed: {:?}", error.kind()))?;
        validate(&staging, &receipt)?;
        File::open(&staging)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("sync staging directory failed: {:?}", error.kind()))?;
        rename_directory_noreplace(&staging, &output)
            .map_err(|error| format!("publish output directory failed: {:?}", error.kind()))?;
        File::open(&parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("sync output parent failed: {:?}", error.kind()))?;
        Ok(receipt)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

fn validate_staged_packages(
    staging: &Path,
    receipt: &Qwen36PtqPackageReceipt,
    packing: SaltV2Packing,
) -> Result<(), String> {
    for (file, package_id, serialized, resident) in [
        (
            COMPACT_PACKAGE_FILE,
            receipt.compact_package_id.as_str(),
            receipt.compact_serialized_bytes,
            receipt.compact_resident_bytes,
        ),
        (
            NEAR_LOSSLESS_PACKAGE_FILE,
            receipt.near_lossless_package_id.as_str(),
            receipt.near_lossless_serialized_bytes,
            receipt.near_lossless_resident_bytes,
        ),
    ] {
        let input = File::open(staging.join(file))
            .map_err(|error| format!("reopen staged package failed: {:?}", error.kind()))?;
        let reader = SaltV2PackageReader::new_strict(input).map_err(|error| error.to_string())?;
        let indexed = reader
            .indexed_runtime_ledger()
            .map_err(|error| error.to_string())?;
        if reader.package_id().to_string() != package_id
            || reader.codec() != packing_codec(packing)
            || reader.ledger().total_bytes != serialized
            || indexed.steady_resident_bytes() != resident
        {
            return Err(format!("staged {file} identity or physical ledger changed"));
        }
    }
    validate_staged_preserved(staging, receipt)?;
    Ok(())
}

fn validate_staged_preserved(
    staging: &Path,
    receipt: &Qwen36PtqPackageReceipt,
) -> Result<(), String> {
    let path = staging.join(PRESERVED_TENSORS_FILE);
    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| format!("inspect staged preserved output failed: {:?}", error.kind()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("staged preserved output must be an ordinary file".to_owned());
    }
    if metadata.len() != receipt.preserved_total_bytes {
        return Err("staged preserved output length changed".to_owned());
    }
    let mut file = File::open(&path)
        .map_err(|error| format!("reopen staged preserved output failed: {:?}", error.kind()))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("inspect opened preserved output failed: {:?}", error.kind()))?;
    let mut hasher = PackageHasher::new();
    let mut bytes = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut bytes)
            .map_err(|error| format!("read staged preserved output failed: {:?}", error.kind()))?;
        if count == 0 {
            break;
        }
        hasher.update(&bytes[..count]);
    }
    let final_metadata = file
        .metadata()
        .map_err(|error| format!("reinspect preserved output failed: {:?}", error.kind()))?;
    if opened.len() != final_metadata.len()
        || final_metadata.len() != receipt.preserved_total_bytes
        || hasher.finalize().to_string() != receipt.preserved_package_id
    {
        return Err("staged preserved output identity changed".to_owned());
    }
    Ok(())
}

fn packing_codec(packing: SaltV2Packing) -> SaltV2Codec {
    match packing {
        SaltV2Packing::D2 => SaltV2Codec::D2,
        SaltV2Packing::B3 => SaltV2Codec::B3,
        SaltV2Packing::S34 => SaltV2Codec::S34,
    }
}

fn create_staging_directory(parent: &Path) -> Result<PathBuf, String> {
    for _ in 0..64 {
        let nonce = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(".tritium-stage-{}-{nonce}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!(
                    "create staging directory failed: {:?}",
                    error.kind()
                ));
            }
        }
    }
    Err("could not allocate a unique staging directory".to_owned())
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple",
    target_os = "redox"
))]
fn rename_directory_noreplace(source: &Path, target: &Path) -> io::Result<()> {
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        source,
        rustix::fs::CWD,
        target,
        rustix::fs::RenameFlags::NOREPLACE,
    )
    .map_err(Into::into)
}

#[cfg(all(
    unix,
    not(any(
        target_os = "linux",
        target_os = "android",
        target_vendor = "apple",
        target_os = "redox"
    ))
))]
fn rename_directory_noreplace(_source: &Path, _target: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace directory publication is unsupported on this Unix target",
    ))
}

#[cfg(windows)]
fn rename_directory_noreplace(source: &Path, target: &Path) -> io::Result<()> {
    fs::rename(source, target)
}

/// Atomically expose one staged directory without replacing an existing path.
#[pyfunction]
pub(crate) fn publish_directory_noreplace(source: &str, target: &str) -> PyResult<()> {
    let source = Path::new(source);
    let target = Path::new(target);
    let metadata = fs::symlink_metadata(source).map_err(|error| {
        PyRuntimeError::new_err(format!(
            "inspect staging directory failed: {:?}",
            error.kind()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(PyValueError::new_err(
            "staging path must be an ordinary directory",
        ));
    }
    let source_parent = fs::canonicalize(source.parent().unwrap_or_else(|| Path::new(".")))
        .map_err(|error| {
            PyRuntimeError::new_err(format!("inspect staging parent failed: {:?}", error.kind()))
        })?;
    let target_parent = fs::canonicalize(target.parent().unwrap_or_else(|| Path::new(".")))
        .map_err(|error| {
            PyRuntimeError::new_err(format!("inspect target parent failed: {:?}", error.kind()))
        })?;
    if source_parent != target_parent {
        return Err(PyValueError::new_err(
            "staging and target directories must share one parent",
        ));
    }
    rename_directory_noreplace(source, target).map_err(|error| {
        if error.kind() == io::ErrorKind::AlreadyExists {
            PyRuntimeError::new_err("output directory already exists")
        } else {
            PyRuntimeError::new_err(format!(
                "publish output directory failed: {:?}",
                error.kind()
            ))
        }
    })
}

/// Strictly reopen one exact package and return its measured identity and bytes.
#[pyfunction]
pub(crate) fn verify_salt_v2_package(
    py: Python<'_>,
    path: &str,
    expected_package_id: &str,
    expected_serialized_bytes: u64,
    expected_resident_bytes: u64,
) -> PyResult<(String, String, u64, u64)> {
    if path.is_empty() || expected_package_id.is_empty() {
        return Err(PyValueError::new_err(
            "package path and expected identity must not be empty",
        ));
    }
    let path = path.to_owned();
    let expected_package_id = expected_package_id.to_owned();
    py.detach(move || {
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| format!("inspect package failed: {:?}", error.kind()))?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err("package path must be an ordinary non-symlink file".to_owned());
        }
        if metadata.len() != expected_serialized_bytes {
            return Err(format!(
                "package length {} differs from manifest {}",
                metadata.len(),
                expected_serialized_bytes
            ));
        }
        let file = File::open(&path)
            .map_err(|error| format!("open package failed: {:?}", error.kind()))?;
        let reader =
            SaltV2PackageReader::new_strict(file).map_err(|error| error.to_string())?;
        let actual_id = reader.package_id().to_string();
        if actual_id != expected_package_id {
            return Err(format!(
                "package identity {actual_id} differs from manifest {expected_package_id}"
            ));
        }
        let serialized = reader.ledger().total_bytes;
        let resident = reader
            .indexed_runtime_ledger()
            .map_err(|error| error.to_string())?
            .steady_resident_bytes();
        if serialized != expected_serialized_bytes || resident != expected_resident_bytes {
            return Err(format!(
                "package physical ledger ({serialized}, {resident}) differs from manifest ({expected_serialized_bytes}, {expected_resident_bytes})"
            ));
        }
        let codec = match reader.codec() {
            SaltV2Codec::D2 => "d2",
            SaltV2Codec::B3 => "b3",
            SaltV2Codec::S34 => "s34",
            _ => "unknown",
        };
        Ok((actual_id, codec.to_owned(), serialized, resident))
    })
    .map_err(PyValueError::new_err)
}

/// Strictly reopen one preserved BF16 safetensors companion.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_preserved_safetensors(
    py: Python<'_>,
    path: &str,
    expected_package_id: &str,
    expected_tensor_count: u64,
    expected_payload_bytes: u64,
    expected_total_bytes: u64,
) -> PyResult<(String, u64, u64, u64)> {
    if path.is_empty() || expected_package_id.is_empty() {
        return Err(PyValueError::new_err(
            "preserved path and expected identity must not be empty",
        ));
    }
    let path = path.to_owned();
    let expected_package_id = expected_package_id.to_owned();
    py.detach(move || {
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| format!("inspect preserved file failed: {:?}", error.kind()))?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err("preserved path must be an ordinary non-symlink file".to_owned());
        }
        if metadata.len() != expected_total_bytes {
            return Err("preserved file length differs from manifest".to_owned());
        }
        let mut file = File::open(&path)
            .map_err(|error| format!("open preserved file failed: {:?}", error.kind()))?;
        let mut prefix = [0_u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|error| format!("read preserved header length failed: {:?}", error.kind()))?;
        let header_len = u64::from_le_bytes(prefix);
        let expected_header_bytes = expected_total_bytes
            .checked_sub(expected_payload_bytes)
            .ok_or_else(|| "preserved payload exceeds total byte ledger".to_owned())?;
        let header_bytes = header_len
            .checked_add(8)
            .filter(|bytes| {
                *bytes == expected_header_bytes
                    && header_len != 0
                    && header_len.is_multiple_of(8)
                    && header_len <= MAX_SAFETENSORS_HEADER_BYTES
            })
            .ok_or_else(|| "preserved header ledger differs from manifest or bound".to_owned())?;
        let header_len_usize = usize::try_from(header_len)
            .map_err(|_| "preserved header length exceeds platform bounds".to_owned())?;
        let mut header = Vec::new();
        header
            .try_reserve_exact(header_len_usize)
            .map_err(|_| "allocate preserved header failed".to_owned())?;
        header.resize(header_len_usize, 0);
        file.read_exact(&mut header)
            .map_err(|error| format!("read preserved header failed: {:?}", error.kind()))?;
        let value: serde_json::Value = serde_json::from_slice(&header)
            .map_err(|error| format!("parse preserved header failed: {error}"))?;
        let object = value
            .as_object()
            .ok_or_else(|| "preserved header must be a JSON object".to_owned())?;
        object
            .get("__metadata__")
            .and_then(serde_json::Value::as_object)
            .filter(|metadata| {
                metadata.len() == 1
                    && metadata.get("format").and_then(serde_json::Value::as_str) == Some("pt")
            })
            .ok_or_else(|| "preserved metadata is invalid".to_owned())?;
        let mut tensor_count = 0_u64;
        let mut offset = 0_u64;
        for (name, tensor) in object {
            if name == "__metadata__" {
                continue;
            }
            let tensor = tensor
                .as_object()
                .filter(|tensor| tensor.len() == 3)
                .ok_or_else(|| "preserved tensor descriptor fields are invalid".to_owned())?;
            if tensor.get("dtype").and_then(serde_json::Value::as_str) != Some("BF16") {
                return Err("preserved tensor dtype or shape is invalid".to_owned());
            }
            let shape = tensor
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| "preserved tensor shape is invalid".to_owned())?;
            let coefficients = shape.iter().try_fold(1_u64, |product, dimension| {
                product
                    .checked_mul(
                        dimension
                            .as_u64()
                            .ok_or_else(|| "preserved tensor dimension is invalid".to_owned())?,
                    )
                    .ok_or_else(|| "preserved tensor shape overflows".to_owned())
            })?;
            let offsets = tensor
                .get("data_offsets")
                .and_then(serde_json::Value::as_array)
                .filter(|offsets| offsets.len() == 2)
                .ok_or_else(|| "preserved tensor offsets are invalid".to_owned())?;
            let start = offsets[0]
                .as_u64()
                .ok_or_else(|| "preserved tensor start offset is invalid".to_owned())?;
            let end = offsets[1]
                .as_u64()
                .filter(|end| *end >= start)
                .ok_or_else(|| "preserved tensor end offset is invalid".to_owned())?;
            if start != offset {
                return Err("preserved tensor offsets are not contiguous".to_owned());
            }
            if end - start
                != coefficients
                    .checked_mul(2)
                    .ok_or_else(|| "preserved tensor byte length overflows".to_owned())?
            {
                return Err("preserved tensor shape and byte range differ".to_owned());
            }
            offset = end;
            tensor_count = tensor_count
                .checked_add(1)
                .ok_or_else(|| "preserved tensor count overflow".to_owned())?;
        }
        if tensor_count != expected_tensor_count || offset != expected_payload_bytes {
            return Err("preserved tensor or payload ledger differs from manifest".to_owned());
        }
        if header_bytes
            .checked_add(expected_payload_bytes)
            .filter(|bytes| *bytes == expected_total_bytes)
            .is_none()
        {
            return Err("preserved total byte ledger differs from manifest".to_owned());
        }
        let mut hasher = PackageHasher::new();
        hasher.update(&prefix);
        hasher.update(&header);
        let mut payload_read = 0_u64;
        let mut chunk = [0_u8; 64 * 1024];
        loop {
            let count = file
                .read(&mut chunk)
                .map_err(|error| format!("read preserved payload failed: {:?}", error.kind()))?;
            if count == 0 {
                break;
            }
            hasher.update(&chunk[..count]);
            payload_read = payload_read
                .checked_add(count as u64)
                .ok_or_else(|| "preserved payload length overflow".to_owned())?;
        }
        let actual_id = hasher.finalize().to_string();
        if payload_read != expected_payload_bytes || actual_id != expected_package_id {
            return Err("preserved payload length or identity differs from manifest".to_owned());
        }
        Ok((actual_id, tensor_count, payload_read, expected_total_bytes))
    })
    .map_err(PyValueError::new_err)
}

/// Validate and content-bind the complete pinned 506-record evidence namespace.
#[pyfunction]
#[pyo3(signature = (evidence_dir, *, max_evidence_bytes = 67_108_864))]
pub(crate) fn inspect_qwen36_ptq_evidence(
    py: Python<'_>,
    evidence_dir: &str,
    max_evidence_bytes: u64,
) -> PyResult<(String, String, u64, String, String, String)> {
    if evidence_dir.is_empty() {
        return Err(PyValueError::new_err("evidence_dir must not be empty"));
    }
    if max_evidence_bytes == 0 {
        return Err(PyValueError::new_err("max_evidence_bytes must be positive"));
    }
    let evidence_dir = evidence_dir.to_owned();
    py.detach(move || {
        let evidence = Qwen36PtqEvidenceDirectory::open_bounded(&evidence_dir, max_evidence_bytes)
            .map_err(|error| error.to_string())?;
        const RECORD_COUNT: u64 = 506;
        evidence
            .validate_complete(RECORD_COUNT)
            .map_err(|error| error.to_string())?;
        let first = evidence.reopen(0).map_err(|error| error.to_string())?;
        let expected_kind = first.kind();
        let expected_source = first.source_id();
        let mut names = BTreeSet::new();
        for ordinal in 0..RECORD_COUNT {
            let record = evidence
                .reopen(ordinal)
                .map_err(|error| error.to_string())?;
            if record.tensor_index() != ordinal {
                return Err(format!(
                    "evidence record {ordinal} declares tensor ordinal {}",
                    record.tensor_index()
                ));
            }
            if record.kind() != expected_kind || record.source_id() != expected_source {
                return Err(format!(
                    "evidence record {ordinal} changes campaign curvature provenance"
                ));
            }
            if !names.insert(record.tensor_name().to_owned()) {
                return Err(format!(
                    "evidence record {ordinal} duplicates a tensor name"
                ));
            }
        }
        let curvature = curvature_label(expected_kind);
        let evidence_id = ContentId::from_path(&evidence_dir).map_err(|error| error.to_string())?;
        Ok((
            evidence_id.to_string(),
            curvature.to_owned(),
            RECORD_COUNT,
            hex_digest(&expected_source.source_model_digest()),
            hex_digest(&expected_source.activation_cache_digest()),
            hex_digest(&expected_source.token_stream_digest()),
        ))
    })
    .map_err(PyValueError::new_err)
}

fn curvature_label(curvature: SaltV2Curvature) -> &'static str {
    match curvature {
        SaltV2Curvature::DiagonalFisher => "diagonal-fisher",
        SaltV2Curvature::InputHessian => "input-hessian",
        SaltV2Curvature::GuidedFisher => "guided-fisher",
        SaltV2Curvature::ForwardKlKronecker => "forward-kl-kronecker",
    }
}

fn hex_digest(digest: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write};

    use super::{hex_digest, publish_package_directory, rename_directory_noreplace};

    #[test]
    fn digest_hex_is_fixed_width_and_lowercase() {
        let mut digest = [0_u8; 32];
        digest[0] = 0xab;
        digest[31] = 0xcd;
        let encoded = hex_digest(&digest);
        assert_eq!(encoded.len(), 64);
        assert_eq!(&encoded[..2], "ab");
        assert_eq!(&encoded[62..], "cd");
    }

    #[test]
    fn package_directory_publication_is_atomic_and_refuses_replacement() {
        let parent = std::env::temp_dir().join(format!(
            "tritium-py-package-publish-{}-{}",
            std::process::id(),
            super::STAGING_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir(&parent).unwrap();
        let output = parent.join("artifact");
        let receipt = publish_package_directory(
            &output,
            |compact, near, preserved| {
                compact.write_all(b"compact").unwrap();
                near.write_all(b"near").unwrap();
                preserved.write_all(b"preserved").unwrap();
                Ok((17_u8, b"{}\n".to_vec()))
            },
            |_, _| Ok(()),
        )
        .unwrap();
        assert_eq!(receipt, 17);
        assert_eq!(fs::read(output.join("compact.tsalt2")).unwrap(), b"compact");
        assert_eq!(
            fs::read(output.join("near-lossless.tsalt2")).unwrap(),
            b"near"
        );
        assert_eq!(
            fs::read(output.join("preserved.safetensors")).unwrap(),
            b"preserved"
        );
        assert_eq!(fs::read(output.join("tritium.json")).unwrap(), b"{}\n");
        assert!(
            publish_package_directory(&output, |_, _, _| Ok(((), Vec::new())), |_, _| Ok(()))
                .unwrap_err()
                .contains("already exists")
        );
        fs::remove_dir_all(parent).unwrap();
    }

    #[test]
    fn failed_package_publication_removes_staging_and_publishes_nothing() {
        let parent = std::env::temp_dir().join(format!(
            "tritium-py-package-fail-{}-{}",
            std::process::id(),
            super::STAGING_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir(&parent).unwrap();
        let output = parent.join("artifact");
        let error = publish_package_directory::<()>(
            &output,
            |compact, _, _| {
                compact.write_all(b"partial").unwrap();
                Err("producer stopped".to_owned())
            },
            |_, _| Ok(()),
        )
        .unwrap_err();
        assert_eq!(error, "producer stopped");
        assert!(!output.exists());
        assert_eq!(fs::read_dir(&parent).unwrap().count(), 0);
        fs::remove_dir_all(parent).unwrap();
    }

    #[test]
    fn failed_staged_validation_publishes_nothing() {
        let parent = std::env::temp_dir().join(format!(
            "tritium-py-package-invalid-{}-{}",
            std::process::id(),
            super::STAGING_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir(&parent).unwrap();
        let output = parent.join("artifact");
        let error = publish_package_directory(
            &output,
            |compact, near, preserved| {
                compact.write_all(b"compact").unwrap();
                near.write_all(b"near").unwrap();
                preserved.write_all(b"preserved").unwrap();
                Ok(((), b"{}\n".to_vec()))
            },
            |_, _| Err("staged package changed".to_owned()),
        )
        .unwrap_err();
        assert_eq!(error, "staged package changed");
        assert!(!output.exists());
        assert_eq!(fs::read_dir(&parent).unwrap().count(), 0);
        fs::remove_dir_all(parent).unwrap();
    }

    #[test]
    fn atomic_directory_publication_never_replaces_a_racing_target() {
        let parent = std::env::temp_dir().join(format!(
            "tritium-py-package-race-{}-{}",
            std::process::id(),
            super::STAGING_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let source = parent.join("staging");
        let target = parent.join("artifact");
        fs::create_dir_all(&source).unwrap();
        fs::create_dir(&target).unwrap();
        fs::write(source.join("source"), b"source").unwrap();
        fs::write(target.join("target"), b"target").unwrap();

        let error = rename_directory_noreplace(&source, &target).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read(source.join("source")).unwrap(), b"source");
        assert_eq!(fs::read(target.join("target")).unwrap(), b"target");
        fs::remove_dir_all(parent).unwrap();
    }
}

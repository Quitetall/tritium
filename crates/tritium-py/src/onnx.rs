//! Strict native admission and atomic publication for Qwen language-plus-MTP ONNX bundles.

use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde_json::json;
use tritium_onnx::{
    AdmittedExternalQwen35BundleDigests, ExternalQwen35BundleFiles, VerifiedExternalQwen35Bundle,
    verify_external_qwen35_bundle,
};

const LANGUAGE_FILE: &str = "language.onnx";
const MTP_FILE: &str = "mtp.onnx";
const WEIGHTS_FILE: &str = "weights.bin";
const MANIFEST_FILE: &str = "tritium-onnx-manifest.json";
static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Native receipt proving both Qwen graphs and their shared external data were admitted.
#[pyclass(module = "tritium", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct QwenOnnxBundleReceipt {
    language_blake3: String,
    mtp_blake3: String,
    weights_blake3: String,
    weights_bytes: u64,
    language_tokens: u64,
    language_past_tokens: u64,
    language_layers: u64,
    mtp_tokens: u64,
    mtp_past_tokens: u64,
    mtp_layers: u64,
    source_model_id: String,
    tokenizer_id: String,
    recipe_id: String,
    package_id: String,
    converted_coverage_id: String,
    deferred_coverage_id: String,
    conversion_mode: Option<String>,
    completion_id: Option<String>,
    campaign_id: Option<String>,
    admission_id: Option<String>,
    selection_id: Option<String>,
}

#[pymethods]
impl QwenOnnxBundleReceipt {
    #[getter]
    fn language_blake3(&self) -> &str {
        &self.language_blake3
    }
    #[getter]
    fn mtp_blake3(&self) -> &str {
        &self.mtp_blake3
    }
    #[getter]
    fn weights_blake3(&self) -> &str {
        &self.weights_blake3
    }
    #[getter]
    const fn weights_bytes(&self) -> u64 {
        self.weights_bytes
    }
    #[getter]
    const fn language_tokens(&self) -> u64 {
        self.language_tokens
    }
    #[getter]
    const fn language_past_tokens(&self) -> u64 {
        self.language_past_tokens
    }
    #[getter]
    const fn language_layers(&self) -> u64 {
        self.language_layers
    }
    #[getter]
    const fn mtp_tokens(&self) -> u64 {
        self.mtp_tokens
    }
    #[getter]
    const fn mtp_past_tokens(&self) -> u64 {
        self.mtp_past_tokens
    }
    #[getter]
    const fn mtp_layers(&self) -> u64 {
        self.mtp_layers
    }
    #[getter]
    fn source_model_id(&self) -> &str {
        &self.source_model_id
    }
    #[getter]
    fn tokenizer_id(&self) -> &str {
        &self.tokenizer_id
    }
    #[getter]
    fn recipe_id(&self) -> &str {
        &self.recipe_id
    }
    #[getter]
    fn package_id(&self) -> &str {
        &self.package_id
    }
    #[getter]
    fn converted_coverage_id(&self) -> &str {
        &self.converted_coverage_id
    }
    #[getter]
    fn deferred_coverage_id(&self) -> &str {
        &self.deferred_coverage_id
    }
    #[getter]
    fn conversion_mode(&self) -> Option<&str> {
        self.conversion_mode.as_deref()
    }
    #[getter]
    fn completion_id(&self) -> Option<&str> {
        self.completion_id.as_deref()
    }
    #[getter]
    fn campaign_id(&self) -> Option<&str> {
        self.campaign_id.as_deref()
    }
    #[getter]
    fn admission_id(&self) -> Option<&str> {
        self.admission_id.as_deref()
    }
    #[getter]
    fn selection_id(&self) -> Option<&str> {
        self.selection_id.as_deref()
    }
}

/// Verify an existing three-file Qwen ONNX bundle against independently admitted digests.
#[pyfunction]
#[pyo3(signature = (language_path, mtp_path, weights_path, *, language_blake3, mtp_blake3, weights_blake3, max_graph_bytes = 268_435_456, max_weights_bytes = 68_719_476_736))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_qwen35_onnx_bundle(
    py: Python<'_>,
    language_path: &str,
    mtp_path: &str,
    weights_path: &str,
    language_blake3: &str,
    mtp_blake3: &str,
    weights_blake3: &str,
    max_graph_bytes: u64,
    max_weights_bytes: u64,
) -> PyResult<QwenOnnxBundleReceipt> {
    let inputs = Inputs::parse(
        language_path,
        mtp_path,
        weights_path,
        language_blake3,
        mtp_blake3,
        weights_blake3,
        max_graph_bytes,
        max_weights_bytes,
    )?;
    py.detach(move || verify_paths(&inputs))
        .map_err(PyValueError::new_err)
}

/// Verify, durably stage, and atomically publish a Qwen ONNX bundle without replacement.
#[pyfunction]
#[pyo3(signature = (language_path, mtp_path, weights_path, output_dir, *, language_blake3, mtp_blake3, weights_blake3, max_graph_bytes = 268_435_456, max_weights_bytes = 68_719_476_736))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_qwen35_onnx_bundle(
    py: Python<'_>,
    language_path: &str,
    mtp_path: &str,
    weights_path: &str,
    output_dir: &str,
    language_blake3: &str,
    mtp_blake3: &str,
    weights_blake3: &str,
    max_graph_bytes: u64,
    max_weights_bytes: u64,
) -> PyResult<QwenOnnxBundleReceipt> {
    let inputs = Inputs::parse(
        language_path,
        mtp_path,
        weights_path,
        language_blake3,
        mtp_blake3,
        weights_blake3,
        max_graph_bytes,
        max_weights_bytes,
    )?;
    if output_dir.is_empty() {
        return Err(PyValueError::new_err("output_dir must not be empty"));
    }
    let output = PathBuf::from(output_dir);
    py.detach(move || stage_paths(&inputs, &output))
        .map_err(PyRuntimeError::new_err)
}

struct Inputs {
    language: PathBuf,
    mtp: PathBuf,
    weights: PathBuf,
    admitted: AdmittedExternalQwen35BundleDigests,
    max_graph_bytes: u64,
    max_weights_bytes: u64,
}

impl Inputs {
    #[allow(clippy::too_many_arguments)]
    fn parse(
        language: &str,
        mtp: &str,
        weights: &str,
        language_digest: &str,
        mtp_digest: &str,
        weights_digest: &str,
        max_graph_bytes: u64,
        max_weights_bytes: u64,
    ) -> PyResult<Self> {
        if language.is_empty() || mtp.is_empty() || weights.is_empty() {
            return Err(PyValueError::new_err("ONNX input paths must not be empty"));
        }
        if max_graph_bytes == 0 || max_weights_bytes == 0 {
            return Err(PyValueError::new_err("ONNX byte ceilings must be positive"));
        }
        Ok(Self {
            language: language.into(),
            mtp: mtp.into(),
            weights: weights.into(),
            admitted: AdmittedExternalQwen35BundleDigests {
                language_model_blake3: parse_digest(language_digest, "language_blake3")?,
                mtp_model_blake3: parse_digest(mtp_digest, "mtp_blake3")?,
                weights_blake3: parse_digest(weights_digest, "weights_blake3")?,
            },
            max_graph_bytes,
            max_weights_bytes,
        })
    }
}

fn verify_paths(inputs: &Inputs) -> Result<QwenOnnxBundleReceipt, String> {
    let language =
        read_regular_bounded(&inputs.language, inputs.max_graph_bytes, "language graph")?;
    let mtp = read_regular_bounded(&inputs.mtp, inputs.max_graph_bytes, "MTP graph")?;
    let weights = read_regular_bounded(
        &inputs.weights,
        inputs.max_weights_bytes,
        "external weights",
    )?;
    let verified = verify_external_qwen35_bundle(
        ExternalQwen35BundleFiles {
            language_model_bytes: &language,
            mtp_model_bytes: &mtp,
            weights_bytes: &weights,
        },
        inputs.admitted,
    )
    .map_err(|error| error.to_string())?;
    receipt(&verified, inputs.admitted)
}

fn stage_paths(inputs: &Inputs, output: &Path) -> Result<QwenOnnxBundleReceipt, String> {
    let name = output
        .file_name()
        .ok_or_else(|| "output_dir must name a new directory".to_owned())?;
    let parent = fs::canonicalize(output.parent().unwrap_or_else(|| Path::new(".")))
        .map_err(|error| format!("canonicalize output parent failed: {:?}", error.kind()))?;
    let output = parent.join(name);
    if fs::symlink_metadata(&output).is_ok() {
        return Err("output directory already exists".to_owned());
    }
    let staging = create_staging_directory(&parent)?;
    let result = (|| {
        copy_sync(
            &inputs.language,
            &staging.join(LANGUAGE_FILE),
            inputs.max_graph_bytes,
            "language graph",
        )?;
        copy_sync(
            &inputs.mtp,
            &staging.join(MTP_FILE),
            inputs.max_graph_bytes,
            "MTP graph",
        )?;
        copy_sync(
            &inputs.weights,
            &staging.join(WEIGHTS_FILE),
            inputs.max_weights_bytes,
            "external weights",
        )?;
        let staged = Inputs {
            language: staging.join(LANGUAGE_FILE),
            mtp: staging.join(MTP_FILE),
            weights: staging.join(WEIGHTS_FILE),
            admitted: inputs.admitted,
            max_graph_bytes: inputs.max_graph_bytes,
            max_weights_bytes: inputs.max_weights_bytes,
        };
        let receipt = verify_paths(&staged)?;
        let manifest = serde_json::to_vec_pretty(&json!({
            "schema": "tritium-qwen35-onnx-bundle-v1",
            "language": {"file": LANGUAGE_FILE, "blake3": receipt.language_blake3},
            "mtp": {"file": MTP_FILE, "blake3": receipt.mtp_blake3},
            "weights": {"file": WEIGHTS_FILE, "blake3": receipt.weights_blake3, "bytes": receipt.weights_bytes},
            "identity": {"source_model_id": receipt.source_model_id, "tokenizer_id": receipt.tokenizer_id, "recipe_id": receipt.recipe_id, "package_id": receipt.package_id, "converted_coverage_id": receipt.converted_coverage_id, "deferred_coverage_id": receipt.deferred_coverage_id},
            "conversion": {"mode": receipt.conversion_mode, "completion_id": receipt.completion_id, "campaign_id": receipt.campaign_id, "admission_id": receipt.admission_id, "selection_id": receipt.selection_id}
        })).map_err(|error| format!("encode ONNX manifest failed: {error}"))?;
        let mut file = File::create_new(staging.join(MANIFEST_FILE))
            .map_err(|error| format!("create ONNX manifest failed: {:?}", error.kind()))?;
        file.write_all(&manifest)
            .map_err(|error| format!("write ONNX manifest failed: {:?}", error.kind()))?;
        file.sync_all()
            .map_err(|error| format!("sync ONNX manifest failed: {:?}", error.kind()))?;
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

fn receipt(
    verified: &VerifiedExternalQwen35Bundle,
    admitted: AdmittedExternalQwen35BundleDigests,
) -> Result<QwenOnnxBundleReceipt, String> {
    let identity = &verified.language.identity;
    let ancestry = verified.conversion_ancestry.as_ref();
    Ok(QwenOnnxBundleReceipt {
        language_blake3: hex_digest(admitted.language_model_blake3),
        mtp_blake3: hex_digest(admitted.mtp_model_blake3),
        weights_blake3: hex_digest(admitted.weights_blake3),
        weights_bytes: u64::try_from(verified.language.weights_bytes)
            .map_err(|_| "weights byte count exceeds u64")?,
        language_tokens: verified.language.tokens as u64,
        language_past_tokens: verified.language.past_tokens as u64,
        language_layers: verified.language.layers as u64,
        mtp_tokens: verified.mtp.tokens as u64,
        mtp_past_tokens: verified.mtp.past_tokens as u64,
        mtp_layers: verified.mtp.layers as u64,
        source_model_id: identity.source_model_id.clone(),
        tokenizer_id: identity.tokenizer_id.clone(),
        recipe_id: identity.recipe_id.clone(),
        package_id: identity.package_id.clone(),
        converted_coverage_id: identity.converted_coverage_id.clone(),
        deferred_coverage_id: identity.deferred_coverage_id.clone(),
        conversion_mode: ancestry.map(|value| value.conversion_mode.clone()),
        completion_id: ancestry.map(|value| value.completion_id.clone()),
        campaign_id: ancestry.map(|value| value.campaign_id.clone()),
        admission_id: ancestry.map(|value| value.admission_id.clone()),
        selection_id: ancestry.map(|value| value.selection_id.clone()),
    })
}

fn parse_digest(value: &str, label: &str) -> PyResult<[u8; 32]> {
    if value.len() != 64 {
        return Err(PyValueError::new_err(format!(
            "{label} must be 64 lowercase hexadecimal characters"
        )));
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(pair).expect("hexadecimal bytes are ASCII");
        digest[index] = u8::from_str_radix(text, 16).map_err(|_| {
            PyValueError::new_err(format!(
                "{label} must be 64 lowercase hexadecimal characters"
            ))
        })?;
        if pair.iter().any(u8::is_ascii_uppercase) {
            return Err(PyValueError::new_err(format!(
                "{label} must be 64 lowercase hexadecimal characters"
            )));
        }
    }
    Ok(digest)
}

fn read_regular_bounded(path: &Path, max_bytes: u64, label: &str) -> Result<Vec<u8>, String> {
    let before = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {label} failed: {:?}", error.kind()))?;
    if before.file_type().is_symlink()
        || !before.is_file()
        || before.len() == 0
        || before.len() > max_bytes
    {
        return Err(format!(
            "{label} must be a non-empty ordinary file no larger than {max_bytes} bytes"
        ));
    }
    let mut file =
        File::open(path).map_err(|error| format!("open {label} failed: {:?}", error.kind()))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("inspect opened {label} failed: {:?}", error.kind()))?;
    if !opened.is_file() || !same_file(&before, &opened) {
        return Err(format!("{label} changed before open"));
    }
    let capacity =
        usize::try_from(opened.len()).map_err(|_| format!("{label} exceeds platform bounds"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| format!("allocate {label} failed"))?;
    Read::by_ref(&mut file)
        .take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {label} failed: {:?}", error.kind()))?;
    let after = fs::symlink_metadata(path)
        .map_err(|error| format!("reinspect {label} failed: {:?}", error.kind()))?;
    let final_opened = file
        .metadata()
        .map_err(|error| format!("reinspect opened {label} failed: {:?}", error.kind()))?;
    if bytes.len() as u64 != opened.len()
        || !same_file(&before, &after)
        || !same_file(&opened, &final_opened)
    {
        return Err(format!("{label} changed while reading"));
    }
    Ok(bytes)
}

fn copy_sync(source: &Path, destination: &Path, max_bytes: u64, label: &str) -> Result<(), String> {
    let bytes = read_regular_bounded(source, max_bytes, label)?;
    let mut output = File::create_new(destination)
        .map_err(|error| format!("create staged {label} failed: {:?}", error.kind()))?;
    output
        .write_all(&bytes)
        .map_err(|error| format!("write staged {label} failed: {:?}", error.kind()))?;
    output
        .sync_all()
        .map_err(|error| format!("sync staged {label} failed: {:?}", error.kind()))
}

fn create_staging_directory(parent: &Path) -> Result<PathBuf, String> {
    for _ in 0..64 {
        let nonce = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".tritium-onnx-stage-{}-{nonce}",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
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
fn rename_directory_noreplace(_: &Path, _: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace directory publication is unsupported on this Unix target",
    ))
}
#[cfg(windows)]
fn rename_directory_noreplace(source: &Path, target: &Path) -> io::Result<()> {
    fs::rename(source, target)
}

#[cfg(unix)]
fn same_file(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev() && left.ino() == right.ino()
}
#[cfg(windows)]
fn same_file(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    left.volume_serial_number() == right.volume_serial_number()
        && left.file_index() == right.file_index()
}

fn hex_digest(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_parser_is_strict() {
        assert_eq!(
            parse_digest(&"ab".repeat(32), "digest").unwrap(),
            [0xab; 32]
        );
        assert!(parse_digest(&"AB".repeat(32), "digest").is_err());
        assert!(parse_digest("00", "digest").is_err());
        assert!(parse_digest(&"gg".repeat(32), "digest").is_err());
    }

    #[test]
    fn failed_stage_never_publishes_partial_directory() {
        let root = std::env::temp_dir().join(format!(
            "tritium-py-onnx-{}-{}",
            std::process::id(),
            STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        for name in [LANGUAGE_FILE, MTP_FILE, WEIGHTS_FILE] {
            fs::write(root.join(name), b"invalid").unwrap();
        }
        let output = root.join("published");
        let inputs = Inputs {
            language: root.join(LANGUAGE_FILE),
            mtp: root.join(MTP_FILE),
            weights: root.join(WEIGHTS_FILE),
            admitted: AdmittedExternalQwen35BundleDigests {
                language_model_blake3: [0; 32],
                mtp_model_blake3: [0; 32],
                weights_blake3: [0; 32],
            },
            max_graph_bytes: 1024,
            max_weights_bytes: 1024,
        };
        assert!(stage_paths(&inputs, &output).is_err());
        assert!(!output.exists());
        assert!(fs::read_dir(&root).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".tritium-onnx-stage-")
        }));
        fs::remove_dir_all(root).unwrap();
    }
}

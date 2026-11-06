//! Native authorization for the audited Qwen3.6 source-identity receipts.
//!
//! This capability joins candidate admission to the pinned official payload
//! identity before a fitting campaign is allowed to create or mutate state.

use std::{fmt, fs, io::Read, path::Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{Qwen36CampaignPreflight, Qwen36SourceProof};

const ADMISSION_SCHEMA: &str = "tritium.qwen36-source-admission.v1";
const IDENTITY_SCHEMA: &str = "tritium.qwen36-official-source-identity.v1";
const MANIFEST_SCHEMA: &str = "tritium.qwen36-official-source-manifest.v1";
const REVISION: &str = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9";
const SOURCE_MODEL_ID: &str = "126eb094f936c87bf7aeff60e57dadf5351ff082a48b8d63c7553919029cd3ca";
const MANIFEST_CONTENT_ID: &str =
    "tsc1_9553bf20975ed88ab3a673522930f9b585ae2e205959ea3dd00ee79c9587c0ba";
const SOURCE_PROOF_ID: &str =
    "tsc1_7e0c191fefc020e74bb0ea1da33d11f69a517a231970d6c9174ee66494e52aa1";
const OFFICIAL_MANIFEST_SHA256: &str =
    "7911b682b615162590074c15baa429ff23c64b7c1d66bd2e134ef6fa3a2a3a3f";
const SOURCE_BYTES: u64 = 55_562_855_904;
const SNAPSHOT_BYTES: u64 = 55_586_107_940;
const SNAPSHOT_FILES: usize = 29;
const MAX_ADMISSION_BYTES: u64 = 4 * 1024 * 1024;
const MAX_IDENTITY_BYTES: u64 = 8 * 1024 * 1024;
const MAX_PROOF_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug)]
pub enum Qwen36SourceIdentityError {
    Read(&'static str),
    File(&'static str),
    Invalid(&'static str),
}
impl fmt::Display for Qwen36SourceIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(label) => write!(f, "cannot read {label}"),
            Self::File(label) => write!(f, "{label} is not a bounded ordinary file"),
            Self::Invalid(label) => write!(
                f,
                "{label} receipt is malformed or contradicts the pinned identity"
            ),
        }
    }
}
impl std::error::Error for Qwen36SourceIdentityError {}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AdmissionReceipt {
    schema: String,
    result: String,
    receipt: AdmissionFields,
    proof_sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AdmissionFields {
    proof_id: String,
    manifest_content_id: String,
    source_model_id: String,
    repository: String,
    revision: String,
    identity_status: String,
    official_payload_authenticated: bool,
    proof_bytes: u64,
    payload_bytes: u64,
    work_dir: String,
    proof_path: String,
    total_tensors: u64,
    total_coefficients: u64,
    language_tensors: u64,
    language_coefficients: u64,
    mtp_tensors: u64,
    mtp_coefficients: u64,
    vision_tensors: u64,
    vision_coefficients: u64,
    additive_tensors: u64,
    additive_coefficients: u64,
    preserved_tensors: u64,
    preserved_coefficients: u64,
    excluded_vision_tensors: u64,
    excluded_vision_coefficients: u64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct IdentityReceipt {
    schema: String,
    result: String,
    repository: String,
    revision: String,
    source_model_id: String,
    manifest_content_id: String,
    source_proof_id: String,
    source_admission_receipt_id: String,
    official_manifest_sha256: String,
    hub_api_response_sha256: String,
    verified_file_count: u64,
    verified_total_bytes: u64,
    files: Vec<IdentityFile>,
    receipt_id: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct IdentityFile {
    name: String,
    size: u64,
    digest: String,
    algorithm: String,
}

/// Validated authorization bound to the measured preflight identity.
#[derive(Clone, Debug)]
pub struct Qwen36SourceIdentityAuthorization {
    admission_receipt_id: String,
    official_identity_receipt_id: String,
}

impl Qwen36SourceIdentityAuthorization {
    /// Validate both receipts and the admitted proof, before creating campaign state.
    pub fn open(
        admission_path: &Path,
        identity_path: &Path,
    ) -> Result<Self, Qwen36SourceIdentityError> {
        let admission: AdmissionReceipt =
            read_json(admission_path, MAX_ADMISSION_BYTES, "source-admission")?;
        let identity: IdentityReceipt =
            read_json(identity_path, MAX_IDENTITY_BYTES, "official identity")?;
        let a = &admission.receipt;
        if admission.schema != ADMISSION_SCHEMA
            || admission.result != "pass"
            || a.repository != "Qwen/Qwen3.6-27B"
            || a.revision != REVISION
            || a.identity_status != "measured-awaiting-official-registration"
            || a.official_payload_authenticated
            || a.source_model_id != SOURCE_MODEL_ID
            || a.manifest_content_id != MANIFEST_CONTENT_ID
            || a.proof_id != SOURCE_PROOF_ID
            || a.payload_bytes != SOURCE_BYTES
            || a.proof_bytes == 0
            || !sums_to(
                &[a.language_tensors, a.mtp_tensors, a.vision_tensors],
                a.total_tensors,
            )
            || !sums_to(
                &[
                    a.language_coefficients,
                    a.mtp_coefficients,
                    a.vision_coefficients,
                ],
                a.total_coefficients,
            )
            || !sums_equal(
                &[a.language_tensors, a.mtp_tensors],
                &[a.additive_tensors, a.preserved_tensors],
            )
            || !sums_equal(
                &[a.language_coefficients, a.mtp_coefficients],
                &[a.additive_coefficients, a.preserved_coefficients],
            )
            || !sums_to(
                &[
                    a.additive_tensors,
                    a.preserved_tensors,
                    a.excluded_vision_tensors,
                ],
                a.total_tensors,
            )
            || !sums_to(
                &[
                    a.additive_coefficients,
                    a.preserved_coefficients,
                    a.excluded_vision_coefficients,
                ],
                a.total_coefficients,
            )
        {
            return Err(Qwen36SourceIdentityError::Invalid("source-admission"));
        }
        let proof_path = Path::new(&a.proof_path);
        let proof_bytes = read_bounded(proof_path, MAX_PROOF_BYTES, "source proof")?;
        if hex(&Sha256::digest(&proof_bytes)) != admission.proof_sha256
            || proof_bytes.len() as u64 != a.proof_bytes
        {
            return Err(Qwen36SourceIdentityError::Invalid("source proof"));
        }
        let proof = Qwen36SourceProof::from_canonical_bytes(&proof_bytes)
            .map_err(|_| Qwen36SourceIdentityError::Invalid("source proof"))?;
        if proof.proof_id().map(|id| id.to_string()).ok().as_deref() != Some(SOURCE_PROOF_ID)
            || hex(proof.source_model_id().as_bytes()) != SOURCE_MODEL_ID
        {
            return Err(Qwen36SourceIdentityError::Invalid("source proof"));
        }
        let admission_value = serde_json::to_value(&admission)
            .map_err(|_| Qwen36SourceIdentityError::Invalid("source-admission"))?;
        let admission_id = format!(
            "sha256:{}",
            hex(&Sha256::digest(canonical(&admission_value)?))
        );
        if identity.schema != IDENTITY_SCHEMA
            || identity.result != "pass"
            || identity.repository != "Qwen/Qwen3.6-27B"
            || identity.revision != REVISION
            || identity.source_model_id != SOURCE_MODEL_ID
            || identity.manifest_content_id != MANIFEST_CONTENT_ID
            || identity.source_proof_id != SOURCE_PROOF_ID
            || identity.source_admission_receipt_id != admission_id
            || identity.official_manifest_sha256 != OFFICIAL_MANIFEST_SHA256
            || identity.verified_file_count != SNAPSHOT_FILES as u64
            || identity.verified_total_bytes != SNAPSHOT_BYTES
            || identity.files.len() != SNAPSHOT_FILES
            || !valid_hex(&identity.hub_api_response_sha256, 64)
        {
            return Err(Qwen36SourceIdentityError::Invalid("official identity"));
        }
        let files_value = serde_json::to_value(&identity.files)
            .map_err(|_| Qwen36SourceIdentityError::Invalid("official identity"))?;
        let manifest = serde_json::json!({"schema": MANIFEST_SCHEMA, "repository": identity.repository,
            "revision": identity.revision, "files": files_value});
        if hex(&Sha256::digest(canonical(&manifest)?)) != OFFICIAL_MANIFEST_SHA256
            || identity.files.iter().any(|f| {
                !valid_hex(&f.digest, if f.algorithm == "sha256" { 64 } else { 40 })
                    || !matches!(f.algorithm.as_str(), "sha256" | "git-sha1")
            })
            || identity
                .files
                .windows(2)
                .any(|pair| pair[0].name >= pair[1].name)
            || identity
                .files
                .iter()
                .try_fold(0_u64, |sum, f| sum.checked_add(f.size))
                != Some(SNAPSHOT_BYTES)
        {
            return Err(Qwen36SourceIdentityError::Invalid("official inventory"));
        }
        let receipt_id = identity.receipt_id.clone();
        let mut identity_value = serde_json::to_value(&identity)
            .map_err(|_| Qwen36SourceIdentityError::Invalid("official identity"))?;
        identity_value
            .as_object_mut()
            .ok_or(Qwen36SourceIdentityError::Invalid("official identity"))?
            .remove("receipt_id");
        if receipt_id
            != format!(
                "sha256:{}",
                hex(&Sha256::digest(canonical(&identity_value)?))
            )
        {
            return Err(Qwen36SourceIdentityError::Invalid(
                "official identity digest",
            ));
        }
        Ok(Self {
            admission_receipt_id: admission_id,
            official_identity_receipt_id: receipt_id,
        })
    }

    /// Bind authorization to the exact semantic model measured by preflight.
    pub fn bind(
        &self,
        preflight: &Qwen36CampaignPreflight,
    ) -> Result<(), Qwen36SourceIdentityError> {
        if hex(preflight.receipt().source_model_id().as_bytes()) != SOURCE_MODEL_ID {
            return Err(Qwen36SourceIdentityError::Invalid(
                "measured source identity",
            ));
        }
        Ok(())
    }

    /// Receipt identities persisted into downstream campaign evidence.
    pub fn receipt_ids(&self) -> (&str, &str) {
        (
            &self.admission_receipt_id,
            &self.official_identity_receipt_id,
        )
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(
    path: &Path,
    limit: u64,
    label: &'static str,
) -> Result<T, Qwen36SourceIdentityError> {
    let bytes = read_bounded(path, limit, label)?;
    serde_json::from_slice(&bytes).map_err(|_| Qwen36SourceIdentityError::Invalid(label))
}
fn read_bounded(
    path: &Path,
    limit: u64,
    label: &'static str,
) -> Result<Vec<u8>, Qwen36SourceIdentityError> {
    let meta = fs::symlink_metadata(path).map_err(|_| Qwen36SourceIdentityError::Read(label))?;
    if !meta.file_type().is_file() || meta.len() == 0 || meta.len() > limit {
        return Err(Qwen36SourceIdentityError::File(label));
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    let mut file = options
        .open(path)
        .map_err(|_| Qwen36SourceIdentityError::Read(label))?;
    let opened = file
        .metadata()
        .map_err(|_| Qwen36SourceIdentityError::Read(label))?;
    if !opened.is_file() || opened.len() == 0 || opened.len() > limit {
        return Err(Qwen36SourceIdentityError::File(label));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if meta.dev() != opened.dev() || meta.ino() != opened.ino() {
            return Err(Qwen36SourceIdentityError::File(label));
        }
    }

    let read_limit = limit
        .checked_add(1)
        .ok_or(Qwen36SourceIdentityError::File(label))?;
    let mut bytes = Vec::with_capacity(opened.len() as usize);
    (&mut file)
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|_| Qwen36SourceIdentityError::Read(label))?;
    if bytes.is_empty() || bytes.len() as u64 > limit {
        return Err(Qwen36SourceIdentityError::File(label));
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let after =
            fs::symlink_metadata(path).map_err(|_| Qwen36SourceIdentityError::Read(label))?;
        if !after.file_type().is_file()
            || after.dev() != opened.dev()
            || after.ino() != opened.ino()
        {
            return Err(Qwen36SourceIdentityError::File(label));
        }
    }
    Ok(bytes)
}
fn canonical(value: &Value) -> Result<Vec<u8>, Qwen36SourceIdentityError> {
    serde_json::to_vec(value).map_err(|_| Qwen36SourceIdentityError::Invalid("canonical JSON"))
}
fn hex(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(H[(b >> 4) as usize] as char);
        out.push(H[(b & 15) as usize] as char);
    }
    out
}
fn valid_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn sums_to(values: &[u64], expected: u64) -> bool {
    values
        .iter()
        .try_fold(0_u64, |sum, value| sum.checked_add(*value))
        == Some(expected)
}
fn sums_equal(left: &[u64], right: &[u64]) -> bool {
    left.iter()
        .try_fold(0_u64, |sum, value| sum.checked_add(*value))
        == right
            .iter()
            .try_fold(0_u64, |sum, value| sum.checked_add(*value))
}

#[cfg(test)]
mod tests {
    use super::{Qwen36SourceIdentityAuthorization, read_bounded};
    use std::{env, fs, path::PathBuf};

    #[cfg(unix)]
    #[test]
    fn bounded_reader_rejects_symlinked_receipt_input() {
        use std::{
            os::unix::fs::symlink,
            time::{SystemTime, UNIX_EPOCH},
        };

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after Unix epoch")
            .as_nanos();
        let directory = env::temp_dir().join(format!(
            "tritium-qwen36-source-identity-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&directory).expect("create isolated fixture directory");
        let target = directory.join("target.json");
        let link = directory.join("receipt.json");
        fs::write(&target, b"{}\n").expect("write regular fixture");
        symlink(&target, &link).expect("create symlink fixture");

        let error = read_bounded(&link, 64, "source-admission")
            .expect_err("receipt symlinks must be rejected");
        assert!(error.to_string().contains("not a bounded ordinary file"));

        fs::remove_dir_all(directory).expect("remove isolated fixture directory");
    }

    /// Offline receipt/proof check for the retained official Qwen snapshot.
    /// Run with TRITIUM_QWEN36_ADMISSION_RECEIPT and
    /// TRITIUM_QWEN36_IDENTITY_RECEIPT set to the exact pair under evaluation.
    #[test]
    #[ignore = "requires the retained local Qwen proof and receipt pair"]
    fn validates_retained_official_identity_pair() {
        let admission = PathBuf::from(
            env::var_os("TRITIUM_QWEN36_ADMISSION_RECEIPT")
                .expect("set TRITIUM_QWEN36_ADMISSION_RECEIPT"),
        );
        let identity = PathBuf::from(
            env::var_os("TRITIUM_QWEN36_IDENTITY_RECEIPT")
                .expect("set TRITIUM_QWEN36_IDENTITY_RECEIPT"),
        );
        let authorization = Qwen36SourceIdentityAuthorization::open(&admission, &identity)
            .expect("retained Qwen receipts and source proof must validate");
        let identity_value: serde_json::Value =
            serde_json::from_slice(&fs::read(&identity).expect("read validated identity receipt"))
                .expect("validated identity receipt is strict JSON");
        let receipt_ids = authorization.receipt_ids();
        assert_eq!(
            receipt_ids.0,
            identity_value["source_admission_receipt_id"]
                .as_str()
                .expect("identity binds admission receipt")
        );
        assert_eq!(
            receipt_ids.1,
            identity_value["receipt_id"]
                .as_str()
                .expect("identity exposes its receipt ID")
        );
    }

    /// Guard against a newer registry receipt being accidentally paired with
    /// the official identity receipt for its older admission parent.
    #[test]
    #[ignore = "requires retained matching and mismatching Qwen receipt files"]
    fn rejects_official_identity_with_a_different_admission_parent() {
        let admission = PathBuf::from(
            env::var_os("TRITIUM_QWEN36_MISMATCHED_ADMISSION_RECEIPT")
                .expect("set TRITIUM_QWEN36_MISMATCHED_ADMISSION_RECEIPT"),
        );
        let identity = PathBuf::from(
            env::var_os("TRITIUM_QWEN36_IDENTITY_RECEIPT")
                .expect("set TRITIUM_QWEN36_IDENTITY_RECEIPT"),
        );
        let error = Qwen36SourceIdentityAuthorization::open(&admission, &identity)
            .expect_err("mismatched parent must fail closed");
        assert!(error.to_string().contains("official identity"));
    }
}

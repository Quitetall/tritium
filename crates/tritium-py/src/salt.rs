//! Python boundary for production SALT V2 master-campaign orchestration.

use std::fmt::Write as _;

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use tritium_quantize::{SaltV2Config, SaltV2Packing};
use tritium_salt::{
    Qwen36AdmittedSource, Qwen36CompleteWorkspaceReceipt, Qwen36PtqEvidenceDirectory,
};

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

fn hex_digest(digest: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::hex_digest;

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
}

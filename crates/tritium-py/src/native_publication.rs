//! Common native publication request admission; typed publishers own the commit.
use pyo3::{PyResult, exceptions::PyValueError};
use std::path::PathBuf;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PublicationLimits {
    pub(crate) max_spool_bytes: u64,
    pub(crate) max_working_bytes: u64,
    pub(crate) max_evidence_bytes: u64,
    pub(crate) max_companion_bytes: u64,
    pub(crate) max_package_bytes: u64,
    pub(crate) max_manifest_bytes: usize,
    pub(crate) max_directory_bytes: usize,
    pub(crate) max_parent_bytes: u64,
    pub(crate) max_replay_working_bytes: u64,
    pub(crate) max_journal_bytes: u64,
}
pub(crate) struct Request {
    pub(crate) validation: Vec<tritium_schema::NativeRefinementValidation>,
    pub(crate) limits: PublicationLimits,
    pub(crate) output: PathBuf,
    pub(crate) output_text: String,
}
fn error(v: impl std::fmt::Display) -> pyo3::PyErr {
    PyValueError::new_err(v.to_string())
}
impl Request {
    pub(crate) fn new(
        output_dir: &str,
        validation_json: &str,
        limits_json: &str,
    ) -> PyResult<Self> {
        if validation_json.len() > 16 * 1024 * 1024 || limits_json.len() > 16384 {
            return Err(error("native publication request exceeds byte limit"));
        }
        let validation = serde_json::from_str(validation_json).map_err(error)?;
        let limits: PublicationLimits = serde_json::from_str(limits_json).map_err(error)?;
        if [
            limits.max_spool_bytes,
            limits.max_working_bytes,
            limits.max_evidence_bytes,
            limits.max_companion_bytes,
            limits.max_package_bytes,
            limits.max_parent_bytes,
            limits.max_replay_working_bytes,
            limits.max_journal_bytes,
        ]
        .contains(&0)
            || limits.max_manifest_bytes == 0
            || limits.max_directory_bytes == 0
        {
            return Err(error("native publication byte limits must be positive"));
        }
        let output = crate::model_refinement::path(output_dir)?;
        let leaf = output
            .file_name()
            .ok_or_else(|| error("child output basename absent"))?
            .to_owned();
        let output = output
            .parent()
            .ok_or_else(|| error("child output parent absent"))?
            .canonicalize()
            .map_err(error)?
            .join(leaf);
        let output_text = output
            .to_str()
            .ok_or_else(|| error("child output path is not UTF-8"))?
            .to_owned();
        Ok(Self {
            validation,
            limits,
            output,
            output_text,
        })
    }
}

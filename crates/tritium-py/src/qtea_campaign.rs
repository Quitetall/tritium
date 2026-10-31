//! GIL-releasing thin receiver over the shared Rust file campaign.
use pyo3::{exceptions::PyValueError, prelude::*};
use std::path::PathBuf;
use tritium_salt::{
    QteaCampaignFiles, QteaCampaignInspection, QteaFileCampaignRequest, QteaFileSourceRequest,
    inspect_qtea_file_campaign, prepare_qtea_file_source, run_qtea_file_campaign_to_directory,
};
fn error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
#[pyfunction]
#[pyo3(signature=(request_json,files_json,*,max_json_bytes=16_777_216))]
pub(crate) fn _prepare_qtea_file_source(
    py: Python<'_>,
    request_json: &str,
    files_json: &str,
    max_json_bytes: usize,
) -> PyResult<crate::model_calibration::NativeSourceObservation> {
    let request = QteaFileSourceRequest::from_slice(request_json.as_bytes(), max_json_bytes)
        .map_err(error)?;
    let (files, _) = file_inputs(files_json, max_json_bytes)?;
    py.detach(move || {
        prepare_qtea_file_source(&request, &files)
            .map(crate::model_calibration::NativeSourceObservation::from_qtea)
            .map_err(error)
    })
}
fn inputs(
    request: &str,
    files: &str,
    max: usize,
) -> PyResult<(QteaFileCampaignRequest, QteaCampaignFiles, PathBuf)> {
    let request = QteaFileCampaignRequest::from_slice(request.as_bytes(), max).map_err(error)?;
    let (files, cwd) = file_inputs(files, max)?;
    Ok((request, files, cwd))
}
fn file_inputs(files: &str, max: usize) -> PyResult<(QteaCampaignFiles, PathBuf)> {
    if max == 0 || files.len() > max {
        return Err(error("file inventory JSON exceeds cap"));
    }
    let files: QteaCampaignFiles = serde_json::from_str(files).map_err(error)?;
    let cwd = std::env::current_dir().map_err(error)?;
    let files = QteaCampaignFiles {
        source_shards: files
            .source_shards
            .into_iter()
            .map(|p| cwd.join(p))
            .collect(),
        assets: files
            .assets
            .into_iter()
            .map(|(n, p)| (n, cwd.join(p)))
            .collect(),
        curvature: files
            .curvature
            .into_iter()
            .map(|(n, p)| (n, cwd.join(p)))
            .collect(),
    };
    Ok((files, cwd))
}
#[pyfunction]
#[pyo3(signature=(request_json,files_json,*,max_json_bytes=16_777_216,max_report_bytes=16_777_216))]
pub(crate) fn _inspect_qtea_file_campaign(
    py: Python<'_>,
    request_json: &str,
    files_json: &str,
    max_json_bytes: usize,
    max_report_bytes: usize,
) -> PyResult<String> {
    let (r, f, _) = inputs(request_json, files_json, max_json_bytes)?;
    if max_report_bytes == 0 {
        return Err(error("report cap must be positive"));
    }
    py.detach(move || {
        let report = inspect_qtea_file_campaign(&r, &f).map_err(error)?;
        String::from_utf8(report.to_json(max_report_bytes).map_err(error)?).map_err(error)
    })
}
#[pyfunction]
#[pyo3(signature=(request_json,files_json,inspection_json,spool_path,output_dir,*,max_json_bytes=16_777_216,max_report_bytes=16_777_216))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _run_qtea_file_campaign(
    py: Python<'_>,
    request_json: &str,
    files_json: &str,
    inspection_json: &str,
    spool_path: &str,
    output_dir: &str,
    max_json_bytes: usize,
    max_report_bytes: usize,
) -> PyResult<String> {
    let (r, f, cwd) = inputs(request_json, files_json, max_json_bytes)?;
    let selected = QteaCampaignInspection::from_slice(inspection_json.as_bytes(), max_json_bytes)
        .map_err(error)?;
    if max_report_bytes == 0 || spool_path.is_empty() || output_dir.is_empty() {
        return Err(error(
            "positive report cap and explicit spool/output paths required",
        ));
    }
    let spool = cwd.join(spool_path);
    let out = cwd.join(output_dir);
    py.detach(move || {
        let result =
            run_qtea_file_campaign_to_directory(&r, &f, &selected, &spool, &out).map_err(error)?;
        let bytes = result.to_json(max_report_bytes).map_err(|e| {
            error(format!(
                "candidate committed; failed to encode observations (not rollback): {e}"
            ))
        })?;
        String::from_utf8(bytes).map_err(error)
    })
}

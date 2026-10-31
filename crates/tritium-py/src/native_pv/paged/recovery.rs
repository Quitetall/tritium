//! Same TNPB bytes, separately scoped paged admission and actual-parent replay.
use super::*;
use crate::native_scale::recovery::ReadLimits;
use std::fs::File;
use tritium_quantize::engine::open_original_file;
use tritium_schema::{BlobId, NativePvResumeManifest};
fn cap(v: ReadLimits) -> PyResult<NativePvPagedResumeBundleLimits> {
    v.common()?;
    Ok(NativePvPagedResumeBundleLimits {
        max_bundle_bytes: v.max_bundle_bytes,
        max_manifest_bytes: v.max_manifest_bytes,
        max_checkpoint_bytes: v.max_checkpoint_bytes,
        max_working_bytes: v.max_working_bytes,
    })
}
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativePvPagedResumeBundle {
    pub(super) root: PathBuf,
    id: BlobId,
    bytes: u64,
    manifest: NativePvResumeManifest,
    limits: ReadLimits,
    warnings: Vec<String>,
}
impl NativePvPagedResumeBundle {
    pub(super) fn reopen(&self) -> PyResult<NativePvPagedResumeBundleReader<File>> {
        let r = NativePvPagedResumeBundleReader::open(
            open_original_file(&self.root.join("resume.tnpb")).map_err(error)?,
            self.id,
            cap(self.limits)?,
        )
        .map_err(error)?;
        if r.manifest() != &self.manifest || r.bytes() != self.bytes {
            return Err(error("paged bundle observations differ"));
        }
        Ok(r)
    }
    fn from_reader(
        root: PathBuf,
        r: NativePvPagedResumeBundleReader<File>,
        limits: ReadLimits,
        warnings: Vec<String>,
    ) -> Self {
        Self {
            root,
            id: r.id(),
            bytes: r.bytes(),
            manifest: r.manifest().clone(),
            limits,
            warnings,
        }
    }
}
#[pymethods]
impl NativePvPagedResumeBundle {
    #[staticmethod]
    #[pyo3(signature=(output_dir,*,bundle_id,max_bundle_bytes,max_manifest_bytes,max_checkpoint_bytes,max_working_bytes))]
    fn open(
        py: Python<'_>,
        output_dir: &str,
        bundle_id: &str,
        max_bundle_bytes: u64,
        max_manifest_bytes: usize,
        max_checkpoint_bytes: u64,
        max_working_bytes: u64,
    ) -> PyResult<Self> {
        let limits = ReadLimits {
            max_bundle_bytes,
            max_manifest_bytes,
            max_checkpoint_bytes,
            max_working_bytes,
        };
        let caps = cap(limits)?;
        let id = crate::native_scale::recovery::blob(bundle_id)?;
        let root = crate::model_refinement::path(output_dir)?
            .canonicalize()
            .map_err(error)?;
        py.detach(|| {
            let r = NativePvPagedResumeBundleReader::open(
                open_original_file(&root.join("resume.tnpb")).map_err(error)?,
                id,
                caps,
            )
            .map_err(error)?;
            Ok(Self::from_reader(root, r, limits, Vec::new()))
        })
    }
    #[getter]
    fn output_dir(&self) -> PyResult<String> {
        self.root
            .to_str()
            .map(str::to_owned)
            .ok_or_else(|| error("paged bundle path is not UTF-8"))
    }
    #[getter]
    fn bundle_id(&self) -> String {
        crate::model_conversion::hex(self.id.as_bytes())
    }
    #[getter]
    fn bundle_bytes(&self) -> u64 {
        self.bytes
    }
    #[getter]
    fn completed_updates(&self) -> u64 {
        self.manifest.point.completed_updates
    }
    #[getter]
    fn plan_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.manifest.plan).map_err(error)
    }
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    #[getter]
    fn admission(&self) -> &'static str {
        "byte-consistent-unqualified"
    }
    #[getter]
    fn quality_qualified(&self) -> bool {
        false
    }
    #[getter]
    fn objective_authenticated(&self) -> bool {
        false
    }
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.reopen().map(drop))
    }
}
pub(super) fn save(
    training: &NativePvPagedTraining,
    py: Python<'_>,
    output_dir: &str,
    limits_json: &str,
) -> PyResult<NativePvPagedResumeBundle> {
    if limits_json.len() > 16384 {
        return Err(error("paged resume limits exceed byte limit"));
    }
    let limits: ReadLimits = serde_json::from_str(limits_json).map_err(error)?;
    let caps = cap(limits)?;
    let output = crate::model_refinement::new_output(output_dir)?;
    if output.to_str().is_none() {
        return Err(error("paged bundle path is not UTF-8"));
    }
    for root in [
        &training.directory,
        &training.parent.root().canonicalize().map_err(error)?,
    ] {
        if overlap(&output, root) {
            return Err(error(
                "paged bundle output cannot overlap parent or workspace",
            ));
        }
    }
    py.detach(|| {
        let mut lock = training.lock()?;
        let s = NativePvPagedTraining::runtime(&mut lock)?;
        let Recording::Writing(writer) = &mut s.recording else {
            return Err(error("paged resume save requires a live writer"));
        };
        let saved = s
            .driver
            .save_resume_bundle(writer, &output, caps)
            .map_err(error)?;
        Ok(NativePvPagedResumeBundle::from_reader(
            output,
            saved.bundle,
            limits,
            saved.warnings.iter().map(|x| format!("{x:?}")).collect(),
        ))
    })
}

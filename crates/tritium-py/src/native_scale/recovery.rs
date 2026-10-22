//! Python descriptors delegate byte admission/publication/replay to the common module.
use super::*;
use std::sync::Arc;
use tritium_quantize::engine::open_original_file;
use tritium_schema::{BlobId, NativeScaleResumeManifest};
use tritium_spec::DeviceBuffer;
use tritium_train::pv_tuning::{
    NativeScaleResumeBundleLimits as BundleLimits, NativeScaleResumeBundleReader as BundleReader,
    publish_native_scale_resume_bundle,
};

#[derive(Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadLimits {
    max_bundle_bytes: u64,
    max_manifest_bytes: usize,
    max_checkpoint_bytes: u64,
    max_working_bytes: u64,
}
impl ReadLimits {
    fn common(self) -> PyResult<BundleLimits> {
        if self.max_bundle_bytes == 0
            || self.max_manifest_bytes == 0
            || self.max_checkpoint_bytes == 0
            || self.max_working_bytes == 0
        {
            return Err(error("native resume bundle byte limits must be positive"));
        }
        Ok(BundleLimits {
            max_bundle_bytes: self.max_bundle_bytes,
            max_manifest_bytes: self.max_manifest_bytes,
            max_checkpoint_bytes: self.max_checkpoint_bytes,
            max_working_bytes: self.max_working_bytes,
        })
    }
}

/// Frozen observed descriptor; opening is byte consistency, not actual-parent replay.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativeScaleResumeBundle {
    root: PathBuf,
    id: BlobId,
    bytes: u64,
    manifest: NativeScaleResumeManifest,
    limits: ReadLimits,
    warnings: Vec<String>,
}
impl NativeScaleResumeBundle {
    fn from_reader(
        root: PathBuf,
        reader: BundleReader<File>,
        limits: ReadLimits,
        warnings: Vec<String>,
    ) -> Self {
        Self {
            root,
            id: reader.id(),
            bytes: reader.bytes(),
            manifest: reader.manifest().clone(),
            limits,
            warnings,
        }
    }
    fn reopen(&self) -> PyResult<BundleReader<File>> {
        let reader = BundleReader::open(
            open_original_file(&self.root.join("resume.tnsb")).map_err(error)?,
            self.id,
            self.limits.common()?,
        )
        .map_err(error)?;
        if reader.manifest() != &self.manifest || reader.bytes() != self.bytes {
            return Err(error("native resume bundle observations differ"));
        }
        Ok(reader)
    }
}
fn blob(value: &str) -> PyResult<BlobId> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(error(
            "native resume bundle_id requires 64 lowercase hex characters",
        ));
    }
    let mut out = [0; 32];
    for (dst, pair) in out.iter_mut().zip(value.as_bytes().as_chunks::<2>().0) {
        let nibble = |b: u8| {
            if b.is_ascii_digit() {
                b - b'0'
            } else {
                b - b'a' + 10
            }
        };
        *dst = nibble(pair[0]) * 16 + nibble(pair[1]);
    }
    Ok(BlobId::from_bytes(out))
}
#[pymethods]
impl NativeScaleResumeBundle {
    #[staticmethod]
    #[pyo3(signature=(output_dir,*,bundle_id,max_bundle_bytes,max_manifest_bytes,max_checkpoint_bytes,max_working_bytes))]
    #[allow(clippy::too_many_arguments)]
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
        let common = limits.common()?;
        let id = blob(bundle_id)?;
        let root = crate::model_refinement::path(output_dir)?
            .canonicalize()
            .map_err(error)?;
        py.detach(|| {
            let reader = BundleReader::open(
                open_original_file(&root.join("resume.tnsb")).map_err(error)?,
                id,
                common,
            )
            .map_err(error)?;
            Ok(Self::from_reader(root, reader, limits, Vec::new()))
        })
    }
    #[getter]
    fn output_dir(&self) -> PyResult<String> {
        self.root
            .to_str()
            .map(str::to_owned)
            .ok_or_else(|| error("native resume path is not UTF-8"))
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
        py.detach(|| self.reopen().map(|_| ()))
    }
}

pub(super) fn save(
    training: &NativeScaleTraining,
    py: Python<'_>,
    output_dir: &str,
    limits_json: &str,
) -> PyResult<NativeScaleResumeBundle> {
    if limits_json.len() > 16384 {
        return Err(error("native resume limits exceed byte limit"));
    }
    let limits: ReadLimits = serde_json::from_str(limits_json).map_err(error)?;
    let common = limits.common()?;
    let output = crate::model_refinement::new_output(output_dir)?;
    // Validate UTF-8/result metadata before the common commit point.
    if output.to_str().is_none() {
        return Err(error("native resume path is not UTF-8"));
    }
    py.detach(|| {
        let mut state = training.lock()?;
        let TrainingState {
            driver, recording, ..
        } = &mut *state;
        let Recording::Writing { writer, directory } = recording else {
            return Err(error("native resume save requires a live recorded writer"));
        };
        let parent = training.parent.root().canonicalize().map_err(error)?;
        for source in [directory.as_path(), parent.as_path()] {
            if output.starts_with(source) || source.starts_with(&output) {
                return Err(error(
                    "native resume output cannot overlap parent or recording work",
                ));
            }
        }
        let saved = publish_native_scale_resume_bundle(driver.session(), writer, &output, common)
            .map_err(error)?;
        let warnings = saved
            .warnings
            .into_iter()
            .map(|v| format!("{v:?}"))
            .collect();
        Ok(NativeScaleResumeBundle::from_reader(
            output,
            saved.bundle,
            limits,
            warnings,
        ))
    })
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordingRequest {
    path: String,
    max_journal_bytes: u64,
    max_working_bytes: u64,
}
pub(super) struct ResumeRecording {
    bundle: BundleReader<File>,
    path: PathBuf,
    directory: PathBuf,
    limits: NativeScaleJournalLimits,
}
impl ResumeRecording {
    pub(super) fn new(
        bundle: &NativeScaleResumeBundle,
        text: &str,
        plan: &NativeRefinementPlan,
        parent: &NativeModelConversion,
    ) -> PyResult<Self> {
        if text.len() > 16384 {
            return Err(error("native resume recording request exceeds byte limit"));
        }
        let request: RecordingRequest = serde_json::from_str(text).map_err(error)?;
        if request.max_journal_bytes == 0 || request.max_working_bytes == 0 {
            return Err(error("native resume journal byte limits must be positive"));
        }
        let reader = bundle.reopen()?;
        if &reader.manifest().plan != plan {
            return Err(error("native resume frozen plan differs"));
        }
        let path = crate::model_refinement::new_output(&request.path)?;
        let directory = path
            .parent()
            .ok_or_else(|| error("native resume journal directory absent"))?
            .to_owned();
        let parent = parent.root().canonicalize().map_err(error)?;
        for source in [parent.as_path(), bundle.root.as_path()] {
            if directory.starts_with(source) {
                return Err(error("native resume work cannot overlap parent or bundle"));
            }
        }
        Ok(Self {
            bundle: reader,
            path,
            directory,
            limits: NativeScaleJournalLimits {
                max_journal_bytes: request.max_journal_bytes,
                max_working_bytes: request.max_working_bytes,
            },
        })
    }
    pub(super) fn checkpoint_bytes(&self) -> u64 {
        self.bundle.manifest().point.checkpoint_bytes
    }
    pub(super) fn bind(
        self,
        parents: Vec<tritium_format::AdditiveTensor>,
        roots: Vec<(String, Arc<dyn DeviceBuffer>)>,
        budget: TensorPayloadBudget,
    ) -> PyResult<(NativeCpuScaleSession, Recording)> {
        let Self {
            bundle,
            path,
            directory,
            limits,
        } = self;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(error)?;
        let result = (|| {
            let (session, writer) = bundle.resume(parents, file, limits).map_err(error)?;
            let driver = NativeCpuScaleSession::bind(session, roots, budget).map_err(error)?;
            Ok((
                driver,
                Recording::Writing {
                    writer: Box::new(writer),
                    directory,
                },
            ))
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&path);
        }
        result
    }
}

//! Distinct unqualified child admission; never a completed PTQ conversion.
use crate::{model_conversion::hex, model_level::NativeModelLevel};
use pyo3::{
    exceptions::{PyFileExistsError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};
use tritium_evidence::{NativeScaleChildRecordView, VerifiedLog, verify_jsonl};
use tritium_format::trit_model::TritModelReader;
use tritium_format::{TritPackageReadLimits, TritPackageReader, TritSpoolBlob};
use tritium_quantize::engine::{
    DirectoryTensorSink, QuantEngineError, TritModelCompanion, TritModelCompanions, TritModelSpool,
    open_original_file, read_original_file_bounded,
};
use tritium_train::pv_tuning::{
    NATIVE_SCALE_CHECKPOINT_FILE, NATIVE_SCALE_JOURNAL_FILE, NativeScaleChildVerification,
    NativeScaleJournalLimits, NativeScaleParentReadLimits, PublishedNativeScaleChild,
    verify_native_scale_child,
};

fn error(v: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(v.to_string())
}
pub(crate) fn path(value: &str) -> PyResult<PathBuf> {
    if value.is_empty() {
        return Err(error("nonempty output directory required"));
    }
    Ok(std::env::current_dir().map_err(error)?.join(value))
}
pub(crate) fn new_output(value: &str) -> PyResult<PathBuf> {
    let output = path(value)?;
    let leaf = output
        .file_name()
        .ok_or_else(|| error("output basename absent"))?
        .to_owned();
    Ok(output
        .parent()
        .ok_or_else(|| error("output parent absent"))?
        .canonicalize()
        .map_err(error)?
        .join(leaf))
}
#[derive(Clone, Copy)]
pub(crate) struct ReadLimits {
    pub package: u64,
    pub evidence: u64,
    pub checkpoint: u64,
    pub journal: u64,
}
impl ReadLimits {
    fn check(self) -> PyResult<Self> {
        if [self.package, self.evidence, self.checkpoint, self.journal].contains(&0) {
            Err(error("native refinement byte limits must be positive"))
        } else {
            Ok(self)
        }
    }
    fn receiving(self) -> TritPackageReadLimits {
        TritPackageReadLimits {
            max_source_bytes: self.package,
            max_manifest_bytes: usize::try_from(self.package).unwrap_or(usize::MAX),
            max_directory_bytes: usize::try_from(self.package).unwrap_or(usize::MAX),
        }
    }
}

fn hash_file(
    root: &Path,
    name: &str,
    maximum: u64,
    expected: u64,
    id: &tritium_schema::BlobId,
) -> PyResult<()> {
    if expected > maximum {
        return Err(error("native refinement companion exceeds byte limit"));
    }
    let mut file = open_original_file(&root.join(name)).map_err(error)?;
    if file.metadata().map_err(error)?.len() != expected {
        return Err(error("native refinement companion length differs"));
    }
    let mut buffer = [0; 8192];
    let mut remaining = expected;
    let mut hash = blake3::Hasher::new();
    while remaining > 0 {
        let n = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..n]).map_err(error)?;
        hash.update(&buffer[..n]);
        remaining -= n as u64;
    }
    if file.read(&mut [0]).map_err(error)? != 0 || hash.finalize().as_bytes() != id.as_bytes() {
        return Err(error("native refinement companion identity differs"));
    }
    Ok(())
}
fn open_recorded(
    root: &Path,
    limits: ReadLimits,
) -> PyResult<(
    TritModelReader<File>,
    VerifiedLog,
    NativeScaleChildRecordView,
)> {
    limits.check()?;
    let text =
        read_original_file_bounded(&root.join("evidence.jsonl"), limits.evidence).map_err(error)?;
    let log = verify_jsonl(std::str::from_utf8(&text).map_err(error)?).map_err(error)?;
    let file = open_original_file(&root.join("model.trit")).map_err(error)?;
    let package_bytes = file.metadata().map_err(error)?.len();
    let mut model = TritModelReader::open(file, limits.receiving()).map_err(error)?;
    let view = NativeScaleChildRecordView::from_child(&log, &mut model).map_err(error)?;
    if package_bytes > view.started().plan.limits.max_package_bytes {
        return Err(error("native refinement exceeds declared package limit"));
    }
    hash_file(
        root,
        NATIVE_SCALE_CHECKPOINT_FILE,
        limits
            .checkpoint
            .min(view.started().plan.limits.max_checkpoint_bytes),
        view.replay().checkpoint_bytes,
        &view.replay().checkpoint,
    )?;
    hash_file(
        root,
        NATIVE_SCALE_JOURNAL_FILE,
        limits.journal,
        view.replay().journal_bytes,
        &view.replay().journal,
    )?;
    model.verify().map_err(error)?;
    Ok((model, log, view))
}

#[pyfunction]
#[pyo3(signature=(output_dir,*,max_package_bytes=8_589_934_592,max_evidence_bytes=16_777_216))]
pub(crate) fn _open_native_model(
    py: Python<'_>,
    output_dir: String,
    max_package_bytes: u64,
    max_evidence_bytes: u64,
) -> PyResult<Py<PyAny>> {
    let limits = ReadLimits {
        package: max_package_bytes,
        evidence: max_evidence_bytes,
        checkpoint: 536_870_912,
        journal: 2_147_483_648,
    }
    .check()?;
    let refined = py.detach(|| {
        let root = path(&output_dir)?;
        let reader = TritModelReader::open(
            open_original_file(&root.join("model.trit")).map_err(error)?,
            limits.receiving(),
        )
        .map_err(error)?;
        let refined = reader
            .manifest()
            .levels
            .iter()
            .any(|level| matches!(level.fit, tritium_schema::LevelFit::Refined { .. }));
        Ok::<_, PyErr>(refined)
    })?;
    if refined {
        Py::new(
            py,
            py.detach(|| NativeModelRefinement::open_inner(output_dir, limits))?,
        )
        .map(Py::into_any)
    } else {
        Py::new(
            py,
            crate::model_conversion::NativeModelConversion::open_inner(
                py,
                output_dir,
                max_package_bytes,
                max_evidence_bytes,
            )?,
        )
        .map(Py::into_any)
    }
}

/// Record-consistent complete child; no original-parent, objective or quality approval.
#[derive(Clone)]
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativeModelRefinement {
    #[pyo3(get)]
    output_dir: String,
    #[pyo3(get)]
    source_id: String,
    #[pyo3(get)]
    package_id: String,
    #[pyo3(get)]
    model_ids: BTreeMap<String, String>,
    #[pyo3(get)]
    evidence_root: String,
    #[pyo3(get)]
    run_id: String,
    #[pyo3(get)]
    parent_level: String,
    #[pyo3(get)]
    parent_package_id: String,
    #[pyo3(get)]
    publication_warnings: Vec<String>,
    limits: ReadLimits,
}
impl NativeModelRefinement {
    pub(crate) fn add_warning(&mut self, message: String) {
        self.publication_warnings.push(message);
    }
    pub(crate) fn from_published(
        output_dir: String,
        child: PublishedNativeScaleChild,
        limits: ReadLimits,
    ) -> Self {
        let parent = &child.receipt.refinement().started().plan.parent;
        let level = child
            .receipt
            .refinement()
            .started()
            .plan
            .child_level
            .clone();
        Self {
            output_dir,
            source_id: hex(parent.source.as_bytes()),
            package_id: hex(child.model.package_id().as_bytes()),
            model_ids: BTreeMap::from([(level, hex(child.model_id.as_bytes()))]),
            evidence_root: child.evidence.log().root_digest.clone(),
            run_id: child.evidence.run().into(),
            parent_level: parent.level.clone(),
            parent_package_id: hex(parent.package.as_bytes()),
            publication_warnings: child.warnings.iter().map(|w| format!("{w:?}")).collect(),
            limits,
        }
    }
    fn open_inner(output_dir: String, limits: ReadLimits) -> PyResult<Self> {
        let root = path(&output_dir)?;
        if std::fs::symlink_metadata(&root)
            .map_err(error)?
            .file_type()
            .is_symlink()
        {
            return Err(error("native refinement requires an ordinary directory"));
        }
        let root = root.canonicalize().map_err(error)?;
        let (model, log, view) = open_recorded(&root, limits)?;
        let parent = &view.started().plan.parent;
        let level = &view.started().plan.child_level;
        Ok(Self {
            output_dir: root
                .to_str()
                .ok_or_else(|| error("path is not UTF-8"))?
                .into(),
            source_id: hex(parent.source.as_bytes()),
            package_id: hex(model.package_id().as_bytes()),
            model_ids: BTreeMap::from([(
                level.clone(),
                hex(model.model_id(level).map_err(error)?.as_bytes()),
            )]),
            evidence_root: log.root_digest,
            run_id: view.reference().run.clone(),
            parent_level: parent.level.clone(),
            parent_package_id: hex(parent.package.as_bytes()),
            publication_warnings: Vec::new(),
            limits,
        })
    }
    fn verified(
        &self,
        limits: ReadLimits,
    ) -> PyResult<(
        TritModelReader<File>,
        VerifiedLog,
        NativeScaleChildRecordView,
    )> {
        let values = open_recorded(Path::new(&self.output_dir), limits)?;
        if hex(values.0.package_id().as_bytes()) != self.package_id
            || values.1.root_digest != self.evidence_root
        {
            return Err(error("native refinement package/evidence identity changed"));
        }
        Ok(values)
    }
}

#[pymethods]
impl NativeModelRefinement {
    // Operational cleanup is not an alteration of immutable child evidence.
    fn _with_cleanup_warning(&self, message: &str) -> Self {
        let mut result = self.clone();
        result.add_warning(format!(
            "NativeRunCleanup: {}",
            message.chars().take(1024).collect::<String>()
        ));
        result
    }
    #[getter]
    fn admission(&self) -> &'static str {
        "record-consistent-unqualified"
    }
    #[getter]
    fn quality_qualified(&self) -> bool {
        false
    }
    #[getter]
    fn objective_authenticated(&self) -> bool {
        false
    }
    #[staticmethod]
    #[pyo3(signature=(output_dir,*,max_package_bytes=8_589_934_592,max_evidence_bytes=16_777_216,max_checkpoint_bytes=536_870_912,max_journal_bytes=2_147_483_648))]
    fn open(
        py: Python<'_>,
        output_dir: String,
        max_package_bytes: u64,
        max_evidence_bytes: u64,
        max_checkpoint_bytes: u64,
        max_journal_bytes: u64,
    ) -> PyResult<Self> {
        py.detach(|| {
            Self::open_inner(
                output_dir,
                ReadLimits {
                    package: max_package_bytes,
                    evidence: max_evidence_bytes,
                    checkpoint: max_checkpoint_bytes,
                    journal: max_journal_bytes,
                },
            )
        })
    }
    #[pyo3(signature=(*,max_package_bytes=None,max_evidence_bytes=None))]
    fn verify(
        &self,
        py: Python<'_>,
        max_package_bytes: Option<u64>,
        max_evidence_bytes: Option<u64>,
    ) -> PyResult<()> {
        let limits = ReadLimits {
            package: self
                .limits
                .package
                .min(max_package_bytes.unwrap_or(self.limits.package)),
            evidence: self
                .limits
                .evidence
                .min(max_evidence_bytes.unwrap_or(self.limits.evidence)),
            ..self.limits
        };
        py.detach(|| self.verified(limits).map(|_| ()))
    }
    #[pyo3(signature=(parent,*,max_parent_bytes,max_replay_working_bytes))]
    fn verify_with_parent(
        &self,
        py: Python<'_>,
        parent: &crate::model_conversion::NativeModelConversion,
        max_parent_bytes: u64,
        max_replay_working_bytes: u64,
    ) -> PyResult<String> {
        if max_parent_bytes == 0 || max_replay_working_bytes == 0 {
            return Err(error("native parent/replay limits must be positive"));
        }
        py.detach(||{
            let(mut child,log,view)=self.verified(self.limits)?;
            let(mut reader,parent_view)=parent.refinement_parent_view(&view.started().plan.parent.level)?;
            let checkpoint=read_original_file_bounded(&Path::new(&self.output_dir).join(NATIVE_SCALE_CHECKPOINT_FILE),self.limits.checkpoint.min(view.started().plan.limits.max_checkpoint_bytes).min(max_parent_bytes)).map_err(error)?;
            let mut journal=open_original_file(&Path::new(&self.output_dir).join(NATIVE_SCALE_JOURNAL_FILE)).map_err(error)?;
            let verified=verify_native_scale_child(NativeScaleChildVerification{parent:&mut reader,parent_evidence:&parent_view,child:&mut child,log:&log,plan:&view.started().plan,journal:&mut journal,checkpoint:&checkpoint},
                NativeScaleParentReadLimits{max_working_bytes:max_parent_bytes,extra_payload_bytes:0},NativeScaleJournalLimits{max_journal_bytes:self.limits.journal,max_working_bytes:max_replay_working_bytes}).map_err(error)?;
            self.verified(self.limits)?;
            Ok(serde_json::json!({"scope":"parent-backed-numerical-replay","package_id":hex(verified.package().as_bytes()),"journal_id":hex(verified.replay().summary().journal().as_bytes()),
                "objective_authenticated":false,"quality_qualified":false}).to_string())
        })
    }
    #[pyo3(signature=(output_dir,*,max_package_bytes=None,max_evidence_bytes=None))]
    fn export(
        &self,
        py: Python<'_>,
        output_dir: String,
        max_package_bytes: Option<u64>,
        max_evidence_bytes: Option<u64>,
    ) -> PyResult<Self> {
        let output = new_output(&output_dir)?;
        if output.starts_with(Path::new(&self.output_dir).canonicalize().map_err(error)?) {
            return Err(error("native export cannot overlap source artifact"));
        }
        let output_text = output
            .to_str()
            .ok_or_else(|| error("output path is not UTF-8"))?
            .to_owned();
        let limits = ReadLimits {
            package: self
                .limits
                .package
                .min(max_package_bytes.unwrap_or(self.limits.package)),
            evidence: self
                .limits
                .evidence
                .min(max_evidence_bytes.unwrap_or(self.limits.evidence)),
            ..self.limits
        };
        py.detach(|| {
            let (mut original, log, view) = self.verified(limits)?;
            let root = Path::new(&self.output_dir);
            let text = read_original_file_bounded(&root.join("evidence.jsonl"), limits.evidence)
                .map_err(error)?;
            let text = std::str::from_utf8(&text).map_err(error)?;
            if verify_jsonl(text).map_err(error)?.root_digest != log.root_digest {
                return Err(error("native refinement original log changed"));
            }
            let package = TritPackageReader::open(
                open_original_file(&root.join("model.trit")).map_err(error)?,
                limits.receiving(),
            )
            .map_err(error)?;
            if hex(package.package_id().as_bytes()) != self.package_id {
                return Err(error("native refinement source changed"));
            }
            let blobs: Vec<_> = package
                .blobs()
                .iter()
                .map(|b| TritSpoolBlob {
                    kind: b.kind.clone(),
                    id: b.id,
                    offset: b.offset,
                    length: b.length,
                })
                .collect();
            let manifest = original.manifest().clone();
            let mut spool = open_original_file(&root.join("model.trit")).map_err(error)?;
            let mut checkpoint =
                open_original_file(&root.join(NATIVE_SCALE_CHECKPOINT_FILE)).map_err(error)?;
            let mut journal =
                open_original_file(&root.join(NATIVE_SCALE_JOURNAL_FILE)).map_err(error)?;
            let maximum = view
                .replay()
                .checkpoint_bytes
                .checked_add(view.replay().journal_bytes)
                .ok_or_else(|| error("companion byte overflow"))?;
            let companions = TritModelCompanions {
                files: vec![
                    TritModelCompanion {
                        name: NATIVE_SCALE_CHECKPOINT_FILE,
                        source: &mut checkpoint,
                        bytes: view.replay().checkpoint_bytes,
                        content: view.replay().checkpoint,
                    },
                    TritModelCompanion {
                        name: NATIVE_SCALE_JOURNAL_FILE,
                        source: &mut journal,
                        bytes: view.replay().journal_bytes,
                        content: view.replay().journal,
                    },
                ],
                max_bytes: maximum,
            };
            let (_, (), warnings) = DirectoryTensorSink::new(&output)
                .publish_original_spooled_model_with_companions(
                    TritModelSpool {
                        manifest: &manifest,
                        blobs: &blobs,
                        source: &mut spool,
                        limits: limits.receiving(),
                    },
                    text,
                    companions,
                    |candidate| {
                        if hex(candidate.package_id().as_bytes()) != self.package_id {
                            return Err(error("native refinement export identity differs"));
                        }
                        original.verify().map_err(error)?;
                        Ok(())
                    },
                )
                .map_err(|e| match e {
                    tritium_quantize::engine::TritModelPublicationError::Publication(
                        QuantEngineError::DestinationExists,
                    ) => PyFileExistsError::new_err("output directory already exists"),
                    other => error(other),
                })?;
            let mut result = self.clone();
            result.output_dir = output_text;
            result.limits = limits;
            result
                .publication_warnings
                .extend(warnings.iter().map(|w| format!("Export: {w:?}")));
            Ok(result)
        })
    }
    #[pyo3(signature=(level,load_run_id,*,max_payload_bytes=2_147_483_648,max_workspace_bytes=67_108_864,max_output_bytes=67_108_864,max_package_bytes=None,max_evidence_bytes=None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "matches existing explicit level receiver ceilings"
    )]
    fn open_level(
        &self,
        py: Python<'_>,
        level: String,
        load_run_id: String,
        max_payload_bytes: u64,
        max_workspace_bytes: u64,
        max_output_bytes: usize,
        max_package_bytes: Option<u64>,
        max_evidence_bytes: Option<u64>,
    ) -> PyResult<NativeModelLevel> {
        let limits = ReadLimits {
            package: self
                .limits
                .package
                .min(max_package_bytes.unwrap_or(self.limits.package)),
            evidence: self
                .limits
                .evidence
                .min(max_evidence_bytes.unwrap_or(self.limits.evidence)),
            ..self.limits
        };
        py.detach(|| {
            self.verified(limits)?;
            let bytes = read_original_file_bounded(
                &Path::new(&self.output_dir).join("model.trit"),
                limits.package,
            )
            .map_err(error)?;
            let text = read_original_file_bounded(
                &Path::new(&self.output_dir).join("evidence.jsonl"),
                limits.evidence,
            )
            .map_err(error)?;
            let log = verify_jsonl(std::str::from_utf8(&text).map_err(error)?).map_err(error)?;
            let mut reader = TritModelReader::open(std::io::Cursor::new(bytes), limits.receiving())
                .map_err(error)?;
            NativeScaleChildRecordView::from_child(&log, &mut reader).map_err(error)?;
            if hex(reader.package_id().as_bytes()) != self.package_id
                || log.root_digest != self.evidence_root
            {
                return Err(error("native refinement snapshot changed"));
            }
            self.verified(limits)?;
            crate::model_level::open_reader(
                reader,
                level,
                load_run_id,
                max_payload_bytes,
                max_workspace_bytes,
                max_output_bytes,
            )
        })
    }
    #[pyo3(signature=(name,*,max_bytes=67_108_864))]
    fn asset<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| {
            let (mut model, _, _) = self.verified(self.limits)?;
            let n = model.asset_byte_len(name).map_err(error)?;
            if max_bytes == 0 || n > max_bytes as u64 {
                return Err(error("asset exceeds byte limit"));
            }
            let mut out = Vec::new();
            out.try_reserve_exact(n as usize).map_err(error)?;
            model
                .visit_asset(name, |b| {
                    out.extend_from_slice(b);
                    Ok::<_, PyErr>(())
                })
                .map_err(error)?;
            model.verify().map_err(error)?;
            Ok(out)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}

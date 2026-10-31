//! Distinct abi3 owner-paged audit; resident API/cap semantics remain unchanged.
use super::*;
use crate::native_parent::NativeTrainingParent;
use std::io;
use tritium_train::pv_tuning::{
    NativePvPagedChildVerification, NativePvPagedChildVerificationLimits, NativePvPagedLimits,
    NativePvPagedReplayLimits, NativeRefinementOwnerReadLimits, verify_native_pv_paged_child,
};

/// Explicit disjoint logical budgets. They exclude metadata, caller buffers,
/// allocator/OS cache and physical disk/RAM/VRAM qualification.
#[derive(Clone, Copy)]
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativePvPagedVerificationLimits {
    #[pyo3(get)]
    pub max_owner_read_bytes: u64,
    #[pyo3(get)]
    pub max_owner_resident_bytes: u64,
    #[pyo3(get)]
    pub max_encoded_owner_bytes: u64,
    #[pyo3(get)]
    pub max_workspace_bytes: u64,
    #[pyo3(get)]
    pub max_replay_working_bytes: u64,
}
#[pymethods]
impl NativePvPagedVerificationLimits {
    #[new]
    #[pyo3(signature=(*,max_owner_read_bytes,max_owner_resident_bytes,max_encoded_owner_bytes,max_workspace_bytes,max_replay_working_bytes=16_384))]
    fn new(
        max_owner_read_bytes: u64,
        max_owner_resident_bytes: u64,
        max_encoded_owner_bytes: u64,
        max_workspace_bytes: u64,
        max_replay_working_bytes: u64,
    ) -> PyResult<Self> {
        let value = Self {
            max_owner_read_bytes,
            max_owner_resident_bytes,
            max_encoded_owner_bytes,
            max_workspace_bytes,
            max_replay_working_bytes,
        };
        value.check()?;
        Ok(value)
    }
}
impl NativePvPagedVerificationLimits {
    fn check(self) -> PyResult<()> {
        if [
            self.max_owner_read_bytes,
            self.max_owner_resident_bytes,
            self.max_encoded_owner_bytes,
            self.max_workspace_bytes,
        ]
        .contains(&0)
            || self.max_replay_working_bytes < 16384
        {
            return Err(error(
                "paged native verification limits must be positive; replay scratch requires16384bytes",
            ));
        }
        Ok(())
    }
    fn native(self, journal: u64) -> NativePvPagedChildVerificationLimits {
        NativePvPagedChildVerificationLimits {
            owners: NativeRefinementOwnerReadLimits {
                max_working_bytes: self.max_owner_read_bytes,
                extra_payload_bytes: 0,
            },
            replay: NativePvPagedReplayLimits {
                max_journal_bytes: journal,
                max_working_bytes: self.max_replay_working_bytes,
                paging: NativePvPagedLimits {
                    max_resident_payload_bytes: self.max_owner_resident_bytes,
                },
            },
            max_encoded_owner_bytes: self.max_encoded_owner_bytes,
        }
    }
}
fn workspace_error(e: io::Error) -> PyErr {
    if e.kind() == io::ErrorKind::AlreadyExists {
        PyFileExistsError::new_err(e.to_string())
    } else {
        error(e)
    }
}
pub(super) fn verify(
    child_descriptor: &NativeModelRefinement,
    parent: NativeTrainingParent,
    workspace_dir: PathBuf,
    limits: NativePvPagedVerificationLimits,
) -> PyResult<String> {
    limits.check()?;
    let (mut child, log, record) = child_descriptor.verified(child_descriptor.limits)?;
    if !matches!(&record, Record::Pv(_)) {
        return Err(error("paged native verification requires alternating-pv"));
    }
    let plan = &record.started().plan;
    let (mut original, parent_view) = parent.refinement_parent_view(&plan.parent.level)?;
    let parent_binding = parent_view.binding().clone();
    let work = new_output(
        workspace_dir
            .to_str()
            .ok_or_else(|| error("workspace path is not UTF-8"))?,
    )?;
    for root in [child_descriptor.root(), parent.root()] {
        let metadata = std::fs::symlink_metadata(root).map_err(error)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(error(
                "paged verification requires ordinary source directories",
            ));
        }
        let root = root.canonicalize().map_err(error)?;
        if work.starts_with(&root) || root.starts_with(&work) {
            return Err(error(
                "paged verification workspace cannot overlap parent or child",
            ));
        }
    }
    let binding = record.replay();
    let root = child_descriptor.root();
    let mut checkpoint = open_original_file(&root.join(record.names().0)).map_err(error)?;
    let mut journal = open_original_file(&root.join(record.names().1)).map_err(error)?;
    let mut workspace = workspace::Workspace::new(
        work.clone(),
        binding.checkpoint_bytes,
        limits.max_workspace_bytes,
        plan.training.len() as u64,
    )
    .map_err(workspace_error)?;
    let result = (|| {
        let proof = verify_native_pv_paged_child(
            NativePvPagedChildVerification {
                parent: &mut original,
                parent_evidence: &parent_view,
                child: &mut child,
                log: &log,
                plan,
                journal: &mut journal,
                checkpoint: &mut checkpoint,
            },
            limits.native(child_descriptor.limits.journal),
            |i| workspace.create(i),
        )
        .map_err(error)?;
        let ids = (proof.package(), proof.replay().summary().journal());
        child_descriptor.verified(child_descriptor.limits)?;
        let (mut fresh, fresh_view) = parent.refinement_parent_view(&plan.parent.level)?;
        if fresh_view.binding() != &parent_binding {
            return Err(error("paged verification parent identity changed"));
        }
        fresh.verify().map_err(error)?;
        Ok::<_, PyErr>(ids)
    })();
    let cleanup = workspace.finish();
    let (package, journal) = match result {
        Ok(ids) => ids,
        Err(e) => {
            if cleanup.warnings.is_empty() {
                return Err(e);
            }
            return Err(error(format!(
                "{e}; private workspace {}: {}",
                work.display(),
                cleanup.warnings.join("; ")
            )));
        }
    };
    Ok(serde_json::json!({"scope":"parent-backed-numerical-replay","storage":"owner-paged",
        "package_id":hex(package.as_bytes()),"journal_id":hex(journal.as_bytes()),
        "lineage_scope":"immediate-parent-only","parent_admission":parent.admission(),
        "objective_authenticated":false,"quality_qualified":false,
        "workspace_cleaned":cleanup.cleaned,"workspace_peak_reserved_bytes":cleanup.peak_reserved_bytes,
        "workspace_peak_files":cleanup.peak_files,"workspace_cleanup_warnings":cleanup.warnings}).to_string())
}

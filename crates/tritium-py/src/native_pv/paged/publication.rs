//! Paged actual-parent replay and the existing common child commit, not a second writer.
use super::*;
use serde::Deserialize;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Limits {
    max_spool_bytes: u64,
    max_working_bytes: u64,
    max_evidence_bytes: u64,
    max_companion_bytes: u64,
    max_package_bytes: u64,
    max_manifest_bytes: usize,
    max_directory_bytes: usize,
}
pub(super) fn publish(
    training: &NativePvPagedTraining,
    py: Python<'_>,
    output_dir: &str,
    run_id: String,
    machine: &str,
    validation_json: &str,
    limits_json: &str,
) -> PyResult<crate::model_refinement::NativeModelRefinement> {
    if validation_json.len() > 16 * 1024 * 1024 || limits_json.len() > 16384 {
        return Err(error("paged publication request exceeds byte limit"));
    }
    let validation: Vec<tritium_schema::NativeRefinementValidation> =
        serde_json::from_str(validation_json).map_err(error)?;
    let limits: Limits = serde_json::from_str(limits_json).map_err(error)?;
    if [
        limits.max_spool_bytes,
        limits.max_working_bytes,
        limits.max_evidence_bytes,
        limits.max_companion_bytes,
        limits.max_package_bytes,
    ]
    .contains(&0)
        || limits.max_manifest_bytes == 0
        || limits.max_directory_bytes == 0
    {
        return Err(error("paged publication limits must be positive"));
    }
    let output = crate::model_refinement::new_output(output_dir)?;
    let output_text = output
        .to_str()
        .ok_or_else(|| error("paged child path is not UTF-8"))?
        .to_owned();
    for root in [
        &training.directory,
        &training.parent.root().canonicalize().map_err(error)?,
    ] {
        if overlap(&output, root) {
            return Err(error(
                "paged child output cannot overlap parent or workspace",
            ));
        }
    }
    py.detach(|| {
        let mut lock = training.lock()?;
        let s = NativePvPagedTraining::runtime(&mut lock)?;
        let plan = s.driver.session().plan().clone();
        if s.driver.session().completed_updates() != plan.training.len() as u64 {
            return Err(error("paged recording schedule incomplete"));
        }
        // Admit fallible recorder before sealing; no successful commit can become failure below.
        let evidence = tritium_evidence::EvidenceRecorder::new(
            run_id,
            [
                tritium_evidence::NATIVE_REFINEMENT_STARTED_REGISTRATION,
                tritium_evidence::NATIVE_REFINEMENT_CHECKPOINTED_REGISTRATION,
                tritium_evidence::NATIVE_REFINEMENT_COMPLETED_REGISTRATION,
                tritium_evidence::NATIVE_PV_REPLAY_OBSERVED_REGISTRATION,
            ],
        )
        .map_err(error)?;
        if matches!(s.recording, Recording::Writing(_)) {
            let Recording::Writing(writer) = std::mem::replace(&mut s.recording, Recording::Failed)
            else {
                unreachable!()
            };
            let (file, summary) = s.driver.finish_recording(*writer).map_err(error)?;
            s.recording = Recording::Sealed(file, summary);
        }
        let Recording::Sealed(journal, summary) = &mut s.recording else {
            return Err(error("paged recording failed"));
        };
        let cp = s.driver.session().size_ledger().logical.checkpoint_bytes;
        let (replayed, proof) = source(&training.parent, &plan, training.limits, |parents| {
            s.driver
                .replay_recorded(
                    &mut *journal,
                    summary.journal(),
                    |i, n| parents.load(i, n).map_err(error),
                    |_| training.workspace.create_blob(cp, 8192).map_err(error),
                    training.limits.replay(),
                )
                .map_err(error)
        })?;
        drop(replayed);
        let (mut parent, view) = training.parent.refinement_parent_view(&plan.parent.level)?;
        let mut spool = training
            .workspace
            .create_blob(
                limits.max_spool_bytes,
                usize::try_from(limits.max_working_bytes).map_err(error)?,
            )
            .map_err(error)?;
        let result = s
            .driver
            .publish_child(
                NativePvChildSource {
                    parent: &mut parent,
                    parent_evidence: &view,
                    replay: &proof,
                    journal,
                    machine,
                    validation: &validation,
                },
                &mut spool,
                &mut tritium_quantize::engine::DirectoryTensorSink::new(&output),
                NativePvPagedChildLimits {
                    max_spool_bytes: limits.max_spool_bytes,
                    max_working_bytes: limits.max_working_bytes,
                    max_evidence_bytes: limits.max_evidence_bytes,
                    max_companion_bytes: limits.max_companion_bytes,
                    owners: training.limits.owner(),
                    receiving: tritium_format::TritPackageReadLimits {
                        max_source_bytes: limits.max_package_bytes,
                        max_manifest_bytes: limits.max_manifest_bytes,
                        max_directory_bytes: limits.max_directory_bytes,
                    },
                },
                evidence,
            )
            .map_err(error)?;
        Ok(
            crate::model_refinement::NativeModelRefinement::from_published_paged_pv(
                output_text,
                result,
                crate::model_refinement::ReadLimits {
                    package: limits.max_package_bytes,
                    evidence: limits.max_evidence_bytes,
                    checkpoint: plan.limits.max_checkpoint_bytes,
                    journal: training.limits.max_journal_bytes,
                },
            ),
        )
    })
}

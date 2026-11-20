//! One evidence/companion/no-clobber commit path after private preparation.
use super::*;
use crate::pv_tuning::journal_io::{Shared, hash_exact};
use std::cell::RefCell;

#[derive(Clone, Copy)]
pub(in crate::pv_tuning::native_scale) struct OutputLimits {
    pub max_evidence_bytes: u64,
    pub max_companion_bytes: u64,
    pub receiving: TritPackageReadLimits,
}
pub(in crate::pv_tuning::native_scale) struct Finish<'a, P, J, C> {
    pub state: Publication<'a>,
    pub source: Source<'a, P, J>,
    pub checkpoint: &'a mut C,
    pub limits: OutputLimits,
    pub recorder: EvidenceRecorder,
    pub execution: NativeRefinementExecution,
    pub staged: staging::Staged,
}
pub(in crate::pv_tuning::native_scale) fn begin<R: Read + Seek, J>(
    state: Publication<'_>,
    source: &Source<'_, R, J>,
    recorder: &EvidenceRecorder,
) -> Result<NativeRefinementExecution> {
    let plan = state.plan();
    if state.completed() != plan.training.len() as u64 {
        return Err(NativeScaleChildError::Binding(
            "native child schedule incomplete",
        ));
    }
    plan.validate_parent(source.parent.manifest(), source.parent_evidence.binding())
        .map_err(|_| NativeScaleChildError::Binding("native child parent/request mismatch"))?;
    let execution = NativeRefinementExecution {
        backend: state.backend().into(),
        device: format!(
            "cpu:{}:{}:{}",
            std::env::consts::ARCH,
            std::env::consts::OS,
            source.machine
        ),
        source_revision: env!("TRITIUM_SOURCE_ID").into(),
    };
    execution
        .validate()
        .map_err(|_| NativeScaleChildError::Binding("native child execution label invalid"))?;
    if source.machine.is_empty()
        || source.validation.len() != plan.validation.len()
        || source
            .validation
            .iter()
            .zip(&plan.validation)
            .any(|(a, b)| {
                a.content != b.content || !a.parent_loss.is_finite() || !a.child_loss.is_finite()
            })
    {
        return Err(NativeScaleChildError::Binding(
            "native child validation observations differ",
        ));
    }
    if !recorder.events().is_empty() {
        return Err(NativeScaleChildError::Binding(
            "native child requires fresh recorder",
        ));
    }
    recorder.check_emit::<NativeRefinementStarted>("refinement", None, 0)?;
    recorder.check_emit::<NativeRefinementCheckpointed>("refinement", None, 2)?;
    recorder.check_emit::<NativeRefinementCompleted>("refinement", None, 3)?;
    state.check_emit(recorder)?;
    Ok(execution)
}
pub(in crate::pv_tuning::native_scale) fn check_companion<R: Read + Seek>(
    source: &mut R,
    bytes: u64,
    id: BlobId,
) -> io::Result<()> {
    let bad = || {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "native child retained companion identity/length/seek differs",
        )
    };
    if source.seek(SeekFrom::End(0))? != bytes || source.seek(SeekFrom::Start(0))? != 0 {
        return Err(bad());
    }
    let mut scratch = [0; 8192];
    if &hash_exact(source, bytes, &mut scratch)? != id.as_bytes()
        || source.read(&mut [0])? != 0
        || source.seek(SeekFrom::End(0))? != bytes
    {
        return Err(bad());
    }
    Ok(())
}
pub(in crate::pv_tuning::native_scale) fn finish<
    R: Read + Seek,
    J: Read + Seek,
    C: Read + Seek,
    S: Read + Write + Seek,
>(
    request: Finish<'_, R, J, C>,
    spool: &mut S,
    destination: &mut DirectoryTensorSink,
) -> Result<Published> {
    let Finish {
        state,
        source,
        checkpoint,
        limits,
        mut recorder,
        execution,
        staged,
    } = request;
    let plan = state.plan();
    let summary = state.binding();
    recorder.emit(
        "refinement",
        None,
        0,
        &NativeRefinementStarted {
            plan: plan.clone(),
            execution,
        },
    )?;
    state.emit(&mut recorder)?;
    recorder.emit(
        "refinement",
        None,
        2,
        &NativeRefinementCheckpointed {
            plan: state.id(),
            completed_updates: state.completed(),
            state: native_refinement_state_id(state.id(), state.completed(), summary.checkpoint),
            checkpoint: summary.checkpoint,
            checkpoint_bytes: summary.checkpoint_bytes,
            state_payload_bytes: state.footprint().0,
            owners: staged.selected,
        },
    )?;
    recorder.emit(
        "refinement",
        None,
        3,
        &NativeRefinementCompleted {
            plan: state.id(),
            checkpoint: summary.checkpoint,
            tensors: staged.outputs,
            validation: source.validation.to_vec(),
        },
    )?;
    let evidence = recorder.seal()?;
    if evidence.jsonl().len() as u64 > limits.max_evidence_bytes {
        return Err(NativeScaleChildError::Limit(
            "native child evidence byte limit exceeded",
        ));
    }
    let mut views = state.fold(evidence.log(), source.parent, source.parent_evidence)?;
    if views.len() != 1 {
        return Err(NativeScaleChildError::Binding(
            "native child complete receipt absent",
        ));
    }
    let receipt = views.remove(0);
    if checkpoint.seek(SeekFrom::Start(0))? != 0 || source.journal.seek(SeekFrom::Start(0))? != 0 {
        return Err(NativeScaleChildError::Binding(
            "native child companion seek differs",
        ));
    }
    let checkpoint = RefCell::new(checkpoint);
    let journal = RefCell::new(source.journal);
    let mut cp = Shared(&checkpoint);
    let mut j = Shared(&journal);
    let companions = TritModelCompanions {
        files: vec![
            TritModelCompanion {
                name: state.companions().0,
                source: &mut cp,
                bytes: summary.checkpoint_bytes,
                content: summary.checkpoint,
            },
            TritModelCompanion {
                name: state.companions().1,
                source: &mut j,
                bytes: summary.bytes,
                content: summary.journal,
            },
        ],
        max_bytes: limits.max_companion_bytes,
    };
    let mut receiving = limits.receiving;
    receiving.max_source_bytes = receiving
        .max_source_bytes
        .min(plan.limits.max_package_bytes);
    let (model, model_id, warnings) = destination.publish_spooled_model_with_companions(
        TritModelSpool {
            manifest: receipt.manifest(),
            blobs: &staged.blobs,
            source: spool,
            limits: receiving,
        },
        &evidence,
        companions,
        |candidate| {
            source
                .parent_evidence
                .verify(source.parent)
                .map_err(NativeScaleChildGuardError::Parent)?;
            check_companion(
                &mut Shared(&checkpoint),
                summary.checkpoint_bytes,
                summary.checkpoint,
            )
            .map_err(NativeScaleChildGuardError::Companion)?;
            check_companion(&mut Shared(&journal), summary.bytes, summary.journal)
                .map_err(NativeScaleChildGuardError::Companion)?;
            source
                .parent_evidence
                .verify(source.parent)
                .map_err(NativeScaleChildGuardError::Parent)?;
            candidate
                .model_id(&plan.child_level)
                .map_err(NativeScaleChildGuardError::Child)
        },
    )?;
    Ok(Published {
        model,
        evidence,
        receipt,
        model_id,
        warnings,
    })
}

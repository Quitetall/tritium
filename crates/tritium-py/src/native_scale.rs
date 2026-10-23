//! Explicit native scale training/recording bridge over common Rust publication.
use crate::{
    model_level::NativePackedMatrix,
    native_parent::NativeTrainingParent,
    torch_native::{checked_product, cpu_f32, export_array_f64, tensor_shape},
};
use dlpark::legacy::Dlpack;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyBytes},
};
use std::{
    fs::{File, OpenOptions},
    ops::{Deref, DerefMut},
    path::PathBuf,
    sync::Mutex,
};
use tritium_runtime::TensorPayloadBudget;
use tritium_schema::NativeRefinementPlan;
use tritium_train::pv_tuning::{
    NativeCpuScaleSession, NativeScaleChildLimits, NativeScaleChildSource,
    NativeScaleJournalLimits, NativeScaleJournalSummary, NativeScaleJournalWriter,
    NativeScaleParentReadLimits, NativeScaleRefinementSession, decode_native_scale_parents,
    publish_native_scale_child, replay_native_scale_journal,
};
pub(crate) mod recovery;
pub(crate) use recovery::NativeScaleResumeBundle;

enum Recording {
    Off,
    Writing {
        writer: Box<NativeScaleJournalWriter<File>>,
        directory: PathBuf,
    },
    Sealed {
        file: File,
        summary: NativeScaleJournalSummary,
        directory: PathBuf,
    },
    Failed,
}
struct TrainingState {
    driver: NativeCpuScaleSession,
    recording: Recording,
    publication_ordinal: u64,
}
impl Deref for TrainingState {
    type Target = NativeCpuScaleSession;
    fn deref(&self) -> &Self::Target {
        &self.driver
    }
}
impl DerefMut for TrainingState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.driver
    }
}
impl TrainingState {
    fn update(&mut self, g: &[&[f32]]) -> PyResult<usize> {
        match &mut self.recording {
            Recording::Off => self.driver.step(g).map_err(error),
            Recording::Writing { writer, .. } => {
                self.driver.step_recorded(g, writer).map_err(error)
            }
            Recording::Sealed { .. } => Err(error("native recording is sealed")),
            Recording::Failed => Err(error("native recording is failed or closed")),
        }
    }
    fn seal(&mut self) -> PyResult<()> {
        if self.driver.session().completed_updates()
            != self.driver.session().plan().training.len() as u64
        {
            return Err(error("native recording schedule incomplete"));
        }
        if matches!(self.recording, Recording::Writing { .. }) {
            let Recording::Writing { writer, directory } =
                std::mem::replace(&mut self.recording, Recording::Failed)
            else {
                unreachable!()
            };
            let (file, summary) = (*writer).finish(self.driver.session()).map_err(error)?;
            self.recording = Recording::Sealed {
                file,
                summary,
                directory,
            };
        }
        if !matches!(self.recording, Recording::Sealed { .. }) {
            return Err(error("complete native recording required"));
        }
        Ok(())
    }
}

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

#[pyfunction]
pub(crate) fn _validate_native_scale_plan(text: &str) -> PyResult<()> {
    if text.len() > 16 * 1024 * 1024 {
        return Err(error("native training plan exceeds byte limit"));
    }
    let plan: NativeRefinementPlan = serde_json::from_str(text).map_err(error)?;
    plan.validate().map_err(|e| error(format!("{e:?}")))?;
    if !matches!(
        plan.recipe,
        tritium_schema::NativeRecoveryRecipe::ScaleOnly { .. }
    ) {
        return Err(error("native Python training requires ScaleOnly"));
    }
    Ok(())
}
pub(crate) fn zeros(len: usize) -> PyResult<Vec<f64>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(error)?;
    out.resize(len, 0.0);
    Ok(out)
}
pub(crate) fn gradient_count(matrix: &NativePackedMatrix, pv: bool) -> PyResult<usize> {
    let cpu = tritium_cpu::CpuBackend::new();
    let handle = matrix.shared_tensor();
    let scales = cpu.additive_stored_scale_count(&*handle).map_err(error)?;
    let weights = if pv {
        cpu.additive_present_weight_count(&*handle).map_err(error)?
    } else {
        0
    };
    scales
        .checked_add(weights)
        .ok_or_else(|| error("native gradient count overflow"))
}
fn admit_output(matrix: &NativePackedMatrix, indices: usize, pv: bool) -> PyResult<usize> {
    let (_, cols, max) = matrix.dimensions();
    let count = gradient_count(matrix, pv)?;
    if count
        .checked_mul(8)
        .and_then(|n| n.checked_add(cols.checked_mul(4)?))
        .and_then(|n| n.checked_add(indices.checked_mul(size_of::<usize>())?))
        .is_none_or(|n| n > max)
    {
        return Err(error(
            "native scale adjoint output/scratch exceeds byte limit",
        ));
    }
    Ok(count)
}
pub(crate) fn matmul_vjp(
    matrix: &NativePackedMatrix,
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    matmul_adjoints(matrix, py, input, gradient, false)
}
pub(crate) fn matmul_adjoints(
    matrix: &NativePackedMatrix,
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    pv: bool,
) -> PyResult<Py<PyAny>> {
    let x: Dlpack = input.extract()?;
    let dy: Dlpack = gradient.extract()?;
    let xs = tensor_shape(&x, "activation")?;
    let ys = tensor_shape(&dy, "cotangent")?;
    let (rows, cols, _) = matrix.dimensions();
    if xs.last() != Some(&cols)
        || ys.last() != Some(&rows)
        || xs[..xs.len() - 1] != ys[..ys.len() - 1]
    {
        return Err(error("native scale adjoint shape mismatch"));
    }
    let batch = checked_product(&xs[..xs.len() - 1], "batch")?;
    let count = admit_output(matrix, 0, pv)?;
    let scales = gradient_count(matrix, false)?;
    let x = cpu_f32(&x, "activation")?;
    let dy = cpu_f32(&dy, "cotangent")?;
    let handle = matrix.shared_tensor();
    let values = py.detach(|| {
        let mut out = zeros(count)?;
        let mut row = vec![0.0; cols];
        tritium_cpu::CpuBackend::new()
            .accumulate_additive_matmul_scale_vjp(
                &*handle,
                x,
                dy,
                batch,
                &mut row,
                &mut out[..scales],
            )
            .map_err(error)?;
        if pv {
            tritium_cpu::CpuBackend::new()
                .accumulate_additive_matmul_weight_vjp(
                    &*handle,
                    x,
                    dy,
                    batch,
                    &mut row,
                    &mut out[scales..],
                )
                .map_err(error)?;
        }
        Ok::<_, PyErr>(out)
    })?;
    export_array_f64(py, &[count], values, "native scale cotangent")
}
pub(crate) fn embed_vjp(
    matrix: &NativePackedMatrix,
    py: Python<'_>,
    indices: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    embed_adjoints(matrix, py, indices, gradient, false)
}
pub(crate) fn embed_adjoints(
    matrix: &NativePackedMatrix,
    py: Python<'_>,
    indices: &Bound<'_, PyAny>,
    gradient: &Bound<'_, PyAny>,
    pv: bool,
) -> PyResult<Py<PyAny>> {
    let ids: Dlpack = indices.extract()?;
    let dy: Dlpack = gradient.extract()?;
    let shape = tensor_shape(&ids, "indices")?;
    let grad = tensor_shape(&dy, "cotangent")?;
    let (rows, cols, _) = matrix.dimensions();
    if grad.last() != Some(&cols) || grad[..grad.len() - 1] != shape {
        return Err(error("native embedding scale adjoint shape mismatch"));
    }
    let raw = ids.cpu_data_slice::<i64>().map_err(error)?;
    let count = admit_output(matrix, raw.len(), pv)?;
    let scales = gradient_count(matrix, false)?;
    if raw.iter().any(|&i| i < 0 || i as u64 >= rows as u64) {
        return Err(error("native embedding index out of range"));
    }
    let dy = cpu_f32(&dy, "cotangent")?;
    let handle = matrix.shared_tensor();
    let values = py.detach(|| {
        let ids: Vec<_> = raw.iter().map(|&i| i as usize).collect();
        let mut out = zeros(count)?;
        let mut row = vec![0.0; cols];
        tritium_cpu::CpuBackend::new()
            .accumulate_additive_embed_scale_vjp(&*handle, &ids, dy, &mut row, &mut out[..scales])
            .map_err(error)?;
        if pv {
            tritium_cpu::CpuBackend::new()
                .accumulate_additive_embed_weight_vjp(
                    &*handle,
                    &ids,
                    dy,
                    &mut row,
                    &mut out[scales..],
                )
                .map_err(error)?;
        }
        Ok::<_, PyErr>(out)
    })?;
    export_array_f64(py, &[count], values, "native embedding scale cotangent")
}

#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativeScaleTraining {
    state: Mutex<TrainingState>,
    specs: Vec<NativePackedMatrix>,
    parent: NativeTrainingParent,
}
impl NativeScaleTraining {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, TrainingState>> {
        self.state
            .lock()
            .map_err(|_| error("native training state lock poisoned"))
    }
}

pub(crate) struct BeginOptions<'a> {
    pub plan_json: &'a str,
    pub max_snapshot_bytes: u64,
    pub max_parent_bytes: u64,
    pub checkpoint: Option<&'a [u8]>,
    pub recorded_resume: Option<(&'a NativeScaleResumeBundle, &'a str)>,
}

pub(crate) fn begin(
    py: Python<'_>,
    result: NativeTrainingParent,
    matrices: Vec<(String, Py<NativePackedMatrix>)>,
    options: BeginOptions<'_>,
) -> PyResult<NativeScaleTraining> {
    let BeginOptions {
        plan_json,
        max_snapshot_bytes,
        max_parent_bytes,
        checkpoint,
        recorded_resume,
    } = options;
    if checkpoint.is_some() && recorded_resume.is_some() {
        return Err(error(
            "checkpoint and recorded resume are mutually exclusive",
        ));
    }
    if plan_json.len() > 16 * 1024 * 1024 || max_parent_bytes == 0 || max_snapshot_bytes == 0 {
        return Err(error("native training plan/byte limits invalid"));
    }
    _validate_native_scale_plan(plan_json)?;
    let plan: NativeRefinementPlan = serde_json::from_str(plan_json).map_err(error)?;
    plan.validate().map_err(|e| error(format!("{e:?}")))?;
    if checkpoint.is_some_and(|b| b.len() as u64 > plan.limits.max_checkpoint_bytes) {
        return Err(error("native checkpoint exceeds byte limit"));
    }
    if matrices.iter().map(|(name, _)| name).ne(plan.owners.iter()) {
        return Err(error("native training owner names/order mismatch"));
    }
    let specs: Vec<_> = matrices
        .iter()
        .map(|(_, m)| m.borrow(py).with_tensor(m.borrow(py).shared_tensor()))
        .collect();
    let roots: Vec<_> = plan
        .owners
        .iter()
        .cloned()
        .zip(specs.iter().map(NativePackedMatrix::shared_tensor))
        .collect();
    let (driver, recording) = py.detach(|| {
        let resumed = recorded_resume
            .map(|(bundle, text)| recovery::ResumeRecording::new(bundle, text, &plan, &result))
            .transpose()?;
        let (mut reader, parent) = result.refinement_parent_view(&plan.parent.level)?;
        if parent.binding() != &plan.parent {
            return Err(error("native training parent tuple differs"));
        }
        let parents = decode_native_scale_parents(
            &mut reader,
            &parent,
            &plan,
            NativeScaleParentReadLimits {
                max_working_bytes: max_parent_bytes,
                extra_payload_bytes: resumed
                    .as_ref()
                    .map_or(0, recovery::ResumeRecording::checkpoint_bytes),
            },
        )
        .map_err(error)?;
        if let Some(resumed) = resumed {
            return resumed.bind(parents, roots, TensorPayloadBudget::new(max_snapshot_bytes));
        }
        let state = if let Some(bytes) = checkpoint {
            NativeScaleRefinementSession::resume(plan, parents, bytes)
        } else {
            NativeScaleRefinementSession::new(plan, parents)
        }
        .map_err(error)?;
        let driver =
            NativeCpuScaleSession::bind(state, roots, TensorPayloadBudget::new(max_snapshot_bytes))
                .map_err(error)?;
        Ok((driver, Recording::Off))
    })?;
    Ok(NativeScaleTraining {
        state: Mutex::new(TrainingState {
            driver,
            recording,
            publication_ordinal: 0,
        }),
        specs,
        parent: result,
    })
}

#[pymethods]
impl NativeScaleTraining {
    fn save_resume(
        &self,
        py: Python<'_>,
        output_dir: &str,
        limits_json: &str,
    ) -> PyResult<NativeScaleResumeBundle> {
        recovery::save(self, py, output_dir, limits_json)
    }
    fn start_recording(
        &self,
        py: Python<'_>,
        path: String,
        max_journal_bytes: u64,
        max_working_bytes: u64,
    ) -> PyResult<()> {
        py.detach(|| {
            let mut state = self.lock()?;
            if !matches!(state.recording, Recording::Off)
                || state.session().completed_updates() != 0
            {
                return Err(error(
                    "native recording requires a fresh zero-origin session",
                ));
            }
            let ledger = state.session().journal_size_ledger().map_err(error)?;
            if max_journal_bytes == 0
                || max_working_bytes == 0
                || ledger.journal_bytes > max_journal_bytes
                || ledger.writer_working_bytes > max_working_bytes
            {
                return Err(error("native journal byte limit exceeded"));
            }
            let path = crate::model_refinement::path(&path)?;
            let directory = path
                .parent()
                .ok_or_else(|| error("journal parent absent"))?
                .canonicalize()
                .map_err(error)?;
            if directory.starts_with(self.parent.root().canonicalize().map_err(error)?) {
                return Err(error("journal work cannot overlap parent artifact"));
            }
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(error)?;
            // Creation/header failure must never turn selected recording into unrecorded updates.
            state.recording = Recording::Failed;
            let writer = NativeScaleJournalWriter::new(
                file,
                state.session(),
                NativeScaleJournalLimits {
                    max_journal_bytes,
                    max_working_bytes,
                },
            )
            .map_err(error)?;
            state.recording = Recording::Writing {
                writer: Box::new(writer),
                directory,
            };
            Ok(())
        })
    }
    fn close_recording(&self) -> PyResult<()> {
        self.lock()?.recording = Recording::Failed;
        Ok(())
    }
    fn publish(
        &self,
        py: Python<'_>,
        output_dir: String,
        run_id: String,
        machine: String,
        validation_json: &str,
        limits_json: &str,
    ) -> PyResult<crate::model_refinement::NativeModelRefinement> {
        let crate::native_publication::Request {
            validation,
            limits,
            output,
            output_text,
        } = crate::native_publication::Request::new(&output_dir, validation_json, limits_json)?;
        py.detach(|| {
            let mut state = self.lock()?;
            let checkpoint_bytes = state.session().size_ledger().checkpoint_bytes;
            if checkpoint_bytes > limits.max_parent_bytes
                || checkpoint_bytes > limits.max_working_bytes
            {
                return Err(error("native checkpoint exceeds publication working limit"));
            }
            let predicted = state.session().journal_size_ledger().map_err(error)?;
            if checkpoint_bytes
                .checked_add(predicted.journal_bytes)
                .is_none_or(|n| n > limits.max_companion_bytes)
            {
                return Err(error("native companions exceed publication byte limit"));
            }
            state.seal()?;
            let forecast = state.session().journal_size_ledger().map_err(error)?;
            if forecast.journal_bytes > limits.max_journal_bytes
                || forecast.replay_working_bytes > limits.max_replay_working_bytes
            {
                return Err(error("native journal/replay byte limit exceeded"));
            }
            let ordinal = state.publication_ordinal;
            state.publication_ordinal = ordinal
                .checked_add(1)
                .ok_or_else(|| error("native publication ordinal exhausted"))?;
            let Recording::Sealed {
                file,
                summary,
                directory,
            } = &state.recording
            else {
                unreachable!()
            };
            if output.starts_with(directory)
                || output.starts_with(self.parent.root().canonicalize().map_err(error)?)
            {
                return Err(error("child output cannot overlap parent or journal work"));
            }
            let mut journal = file.try_clone().map_err(error)?;
            let summary = summary.clone();
            let plan = state.session().plan();
            let checkpoint = state.session().checkpoint_bytes().map_err(error)?;
            let (mut parent, parent_view) =
                self.parent.refinement_parent_view(&plan.parent.level)?;
            let parents = decode_native_scale_parents(
                &mut parent,
                &parent_view,
                plan,
                NativeScaleParentReadLimits {
                    max_working_bytes: limits.max_parent_bytes,
                    extra_payload_bytes: checkpoint.len() as u64,
                },
            )
            .map_err(error)?;
            let replay = replay_native_scale_journal(
                plan.clone(),
                parents,
                &mut journal,
                summary.journal(),
                &checkpoint,
                NativeScaleJournalLimits {
                    max_journal_bytes: limits.max_journal_bytes,
                    max_working_bytes: limits.max_replay_working_bytes,
                },
            )
            .map_err(error)?;
            let evidence = tritium_evidence::EvidenceRecorder::new(
                run_id,
                [
                    tritium_evidence::NATIVE_REFINEMENT_STARTED_REGISTRATION,
                    tritium_evidence::NATIVE_REFINEMENT_CHECKPOINTED_REGISTRATION,
                    tritium_evidence::NATIVE_REFINEMENT_COMPLETED_REGISTRATION,
                    tritium_evidence::NATIVE_SCALE_REPLAY_OBSERVED_REGISTRATION,
                ],
            )
            .map_err(error)?;
            // Spool belongs to the already-owned recording work directory, never output/source.
            let spool_path = directory.join(format!("publication-{ordinal}.spool"));
            let mut spool = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&spool_path)
                .map_err(error)?;
            let receiving = tritium_format::TritPackageReadLimits {
                max_source_bytes: limits.max_package_bytes,
                max_manifest_bytes: limits.max_manifest_bytes,
                max_directory_bytes: limits.max_directory_bytes,
            };
            let result = publish_native_scale_child(
                state.session(),
                NativeScaleChildSource {
                    parent: &mut parent,
                    parent_evidence: &parent_view,
                    replay: &replay,
                    journal: &mut journal,
                    machine: &machine,
                    validation: &validation,
                },
                &mut spool,
                &mut tritium_quantize::engine::DirectoryTensorSink::new(&output),
                NativeScaleChildLimits {
                    max_spool_bytes: limits.max_spool_bytes,
                    max_working_bytes: limits.max_working_bytes,
                    max_evidence_bytes: limits.max_evidence_bytes,
                    max_companion_bytes: limits.max_companion_bytes,
                    receiving,
                },
                evidence,
            );
            drop(spool);
            let cleanup = std::fs::remove_file(&spool_path);
            let child = result.map_err(error)?;
            let mut result = crate::model_refinement::NativeModelRefinement::from_published(
                output_text,
                child,
                crate::model_refinement::ReadLimits {
                    package: limits.max_package_bytes,
                    evidence: limits.max_evidence_bytes,
                    checkpoint: plan.limits.max_checkpoint_bytes,
                    journal: limits.max_journal_bytes,
                },
            );
            if let Err(cause) = cleanup {
                result.add_warning(format!("SpoolCleanup: {cause}"));
            }
            Ok(result)
        })
    }
    #[getter]
    fn completed_updates(&self) -> PyResult<u64> {
        Ok(self.lock()?.session().completed_updates())
    }
    fn frame(
        &self,
        py: Python<'_>,
        owner: &str,
        revision: u64,
        include_scales: bool,
    ) -> PyResult<(NativePackedMatrix, Option<Py<PyAny>>)> {
        let (matrix, scales) = py.detach(|| {
            let state = self.lock()?;
            if state.session().completed_updates() != revision {
                return Err(error("stale native training revision"));
            }
            let i = state
                .session()
                .plan()
                .owners
                .iter()
                .position(|n| n == owner)
                .ok_or_else(|| error("native owner absent"))?;
            let handle = state
                .handles()
                .nth(i)
                .ok_or_else(|| error("native handle absent"))?
                .1;
            let matrix = self.specs[i].with_tensor(handle.clone());
            let scales = if include_scales {
                let source = state.session().weights().nth(i).expect("bound owner");
                if source
                    .scales()
                    .len()
                    .checked_mul(8)
                    .is_none_or(|n| n > self.specs[i].dimensions().2)
                {
                    return Err(error("native scale leaf exceeds output byte limit"));
                }
                let mut values = zeros(source.scales().len())?;
                for (value, &scale) in values.iter_mut().zip(source.scales()) {
                    *value = f64::from(scale);
                }
                Some(values)
            } else {
                None
            };
            Ok::<_, PyErr>((matrix, scales))
        })?;
        let scales = scales
            .map(|s| export_array_f64(py, &[s.len()], s, "native stored scales"))
            .transpose()?;
        Ok((matrix, scales))
    }
    fn step(
        &self,
        py: Python<'_>,
        revision: u64,
        window: Vec<u8>,
        gradients: Vec<Option<Dlpack>>,
    ) -> PyResult<usize> {
        let mut values = Vec::new();
        let mut any = false;
        {
            let state = self.lock()?;
            if state.session().completed_updates() != revision {
                return Err(error("stale native training revision"));
            }
            let expected = state
                .session()
                .plan()
                .training
                .get(revision as usize)
                .ok_or_else(|| error("native training schedule exhausted"))?;
            if window.as_slice() != expected.content {
                return Err(error("native training window differs"));
            }
            if gradients.len() != state.handles().len() {
                return Err(error("native gradient owner count differs"));
            }
            let count = state
                .session()
                .weights()
                .try_fold(0u64, |n, w| n.checked_add(w.scales().len() as u64))
                .ok_or_else(|| error("native gradient size overflow"))?;
            if count
                .checked_mul(20)
                .is_none_or(|n| n > state.session().plan().limits.max_workspace_bytes)
            {
                return Err(error(
                    "native gradient transfer/update exceeds workspace byte limit",
                ));
            }
            for (gradient, weight) in gradients.iter().zip(state.session().weights()) {
                if let Some(g) = gradient {
                    if tensor_shape(g, "scale gradient")? != [weight.scales().len()] {
                        return Err(error("native scale gradient shape differs"));
                    }
                    g.cpu_data_slice::<f64>().map_err(error)?;
                }
            }
            for (gradient, weight) in gradients.iter().zip(state.session().weights()) {
                let count = weight.scales().len();
                let v = if let Some(g) = gradient {
                    any = true;
                    if tensor_shape(g, "scale gradient")? != [count] {
                        return Err(error("native scale gradient shape differs"));
                    }
                    g.cpu_data_slice::<f64>()
                        .map_err(error)?
                        .iter()
                        .map(|&v| v as f32)
                        .collect()
                } else {
                    vec![0.0; count]
                };
                values.push(v);
            }
        }
        if !any {
            return Err(error("native training has no owner gradients"));
        }
        py.detach(|| {
            let mut state = self.lock()?;
            if state.session().completed_updates() != revision {
                return Err(error("stale native training revision"));
            }
            state.update(&values.iter().map(Vec::as_slice).collect::<Vec<_>>())
        })
    }
    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.lock()?.session().checkpoint_bytes().map_err(error))?;
        Ok(PyBytes::new(py, &bytes))
    }
}

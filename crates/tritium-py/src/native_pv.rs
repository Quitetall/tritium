//! Distinct eager P/V adapter over the existing common optimizer and publisher.
use crate::{
    model_level::NativePackedMatrix,
    native_parent::NativeTrainingParent,
    native_scale::zeros,
    torch_native::{export_array_f64, tensor_shape},
};
use dlpark::legacy::Dlpack;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyBytes},
};
use std::{
    fs::{File, OpenOptions},
    path::PathBuf,
    sync::Mutex,
};
use tritium_runtime::TensorPayloadBudget;
use tritium_schema::{NativeRecoveryRecipe, NativeRefinementPlan};
use tritium_train::pv_tuning::{
    NativeCpuPvSession, NativePvChildLimits, NativePvChildSource, NativePvJournalLimits,
    NativePvJournalSummary, NativePvJournalWriter, NativePvParentReadLimits,
    NativePvRefinementSession, decode_native_pv_parents, publish_native_pv_child,
    replay_native_pv_journal,
};
mod recovery;
pub(crate) use recovery::NativePvResumeBundle;

fn error(v: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(v.to_string())
}
fn parse(text: &str) -> PyResult<NativeRefinementPlan> {
    if text.len() > 16 * 1024 * 1024 {
        return Err(error("native training plan exceeds byte limit"));
    }
    let plan: NativeRefinementPlan = serde_json::from_str(text).map_err(error)?;
    plan.validate().map_err(|e| error(format!("{e:?}")))?;
    if !matches!(plan.recipe, NativeRecoveryRecipe::AlternatingPv { .. }) {
        return Err(error("native P/V training requires AlternatingPv"));
    }
    Ok(plan)
}
#[pyfunction]
pub(crate) fn _validate_native_pv_plan(text: &str) -> PyResult<()> {
    parse(text).map(drop)
}
fn count(weight: &tritium_format::AdditiveTensor) -> PyResult<usize> {
    let active = if weight.plane_count() == 0 {
        0
    } else {
        weight
            .view()
            .plane_trits(0)
            .map_err(|e| error(format!("{e:?}")))?
            .len()
    };
    weight
        .scales()
        .len()
        .checked_add(active)
        .ok_or_else(|| error("native PV gradient size overflow"))
}
enum Recording {
    Off,
    Writing {
        writer: Box<NativePvJournalWriter<File>>,
        directory: PathBuf,
    },
    Sealed {
        file: File,
        summary: NativePvJournalSummary,
        directory: PathBuf,
    },
    Failed,
}
struct State {
    driver: NativeCpuPvSession,
    recording: Recording,
    publication_ordinal: u64,
}
impl State {
    fn update(&mut self, gradients: &[Vec<f32>]) -> PyResult<(usize, usize)> {
        let mut scales = Vec::new();
        let mut weights = Vec::new();
        scales.try_reserve_exact(gradients.len()).map_err(error)?;
        weights.try_reserve_exact(gradients.len()).map_err(error)?;
        for (g, w) in gradients.iter().zip(self.driver.session().weights()) {
            let (s, a) = g.split_at(w.scales().len());
            scales.push(s);
            weights.push(a);
        }
        let stats = match &mut self.recording {
            Recording::Off => self.driver.step(&scales, &weights).map_err(error)?,
            Recording::Writing { writer, .. } => self
                .driver
                .step_recorded(&scales, &weights, writer)
                .map_err(error)?,
            Recording::Sealed { .. } => return Err(error("native recording is sealed")),
            Recording::Failed => return Err(error("native recording is failed or closed")),
        };
        Ok((
            stats.changed_scales,
            stats.owners.iter().map(|o| o.changed_units).sum(),
        ))
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
            return Err(error("complete native P/V recording required"));
        }
        Ok(())
    }
}

/// First-order eager CPU P/V transport/controller, never a qualification receipt.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativePvTraining {
    state: Mutex<State>,
    specs: Vec<NativePackedMatrix>,
    parent: NativeTrainingParent,
}
impl NativePvTraining {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, State>> {
        self.state
            .lock()
            .map_err(|_| error("native P/V training state lock poisoned"))
    }
}
pub(crate) struct BeginOptions<'a> {
    pub plan_json: &'a str,
    pub max_snapshot_bytes: u64,
    pub max_parent_bytes: u64,
    pub checkpoint: Option<&'a [u8]>,
    pub recorded_resume: Option<(&'a NativePvResumeBundle, &'a str)>,
}
pub(crate) fn begin(
    py: Python<'_>,
    parent: NativeTrainingParent,
    matrices: Vec<(String, Py<NativePackedMatrix>)>,
    options: BeginOptions<'_>,
) -> PyResult<NativePvTraining> {
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
    let plan = parse(plan_json)?;
    if max_snapshot_bytes == 0 || max_parent_bytes == 0 {
        return Err(error("native training byte limits invalid"));
    }
    if checkpoint.is_some_and(|b| b.len() as u64 > plan.limits.max_checkpoint_bytes) {
        return Err(error("native checkpoint exceeds byte limit"));
    }
    if matrices.iter().map(|(n, _)| n).ne(plan.owners.iter()) {
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
            .map(|(bundle, text)| recovery::ResumeRecording::new(bundle, text, &plan, &parent))
            .transpose()?;
        let (mut reader, view) = parent.refinement_parent_view(&plan.parent.level)?;
        if view.binding() != &plan.parent {
            return Err(error("native training parent tuple differs"));
        }
        let parents = decode_native_pv_parents(
            &mut reader,
            &view,
            &plan,
            NativePvParentReadLimits {
                max_working_bytes: max_parent_bytes,
                extra_payload_bytes: resumed.as_ref().map_or(
                    checkpoint.map_or(0, |b| b.len() as u64),
                    recovery::ResumeRecording::checkpoint_bytes,
                ),
            },
        )
        .map_err(error)?;
        if let Some(resumed) = resumed {
            return resumed.bind(parents, roots, TensorPayloadBudget::new(max_snapshot_bytes));
        }
        let state = if let Some(bytes) = checkpoint {
            NativePvRefinementSession::resume(plan, parents, bytes)
        } else {
            NativePvRefinementSession::new(plan, parents)
        }
        .map_err(error)?;
        let budget = TensorPayloadBudget::new(max_snapshot_bytes);
        let driver = bind_current(state, roots, budget, checkpoint.is_some())?;
        Ok((driver, Recording::Off))
    })?;
    Ok(NativePvTraining {
        state: Mutex::new(State {
            driver,
            recording,
            publication_ordinal: 0,
        }),
        specs,
        parent,
    })
}

fn bind_current(
    state: NativePvRefinementSession,
    roots: Vec<(String, std::sync::Arc<dyn tritium_spec::DeviceBuffer>)>,
    budget: TensorPayloadBudget,
    refresh: bool,
) -> PyResult<NativeCpuPvSession> {
    let roots = if refresh {
        let mut current = Vec::new();
        current.try_reserve_exact(roots.len()).map_err(error)?;
        for ((name, root), w) in roots.into_iter().zip(state.weights()) {
            current.push((
                name,
                tritium_cpu::CpuBackend::new()
                    .snapshot_additive_weights(&root, w.view(), &budget)
                    .map_err(error)?,
            ));
        }
        current
    } else {
        roots
    };
    NativeCpuPvSession::bind(state, roots, budget).map_err(error)
}

#[pymethods]
impl NativePvTraining {
    fn save_resume(
        &self,
        py: Python<'_>,
        output_dir: &str,
        limits_json: &str,
    ) -> PyResult<NativePvResumeBundle> {
        recovery::save(self, py, output_dir, limits_json)
    }
    #[getter]
    fn completed_updates(&self) -> PyResult<u64> {
        Ok(self.lock()?.driver.session().completed_updates())
    }
    fn close_recording(&self) -> PyResult<()> {
        self.lock()?.recording = Recording::Failed;
        Ok(())
    }
    fn start_recording(
        &self,
        py: Python<'_>,
        path: &str,
        max_journal_bytes: u64,
        max_working_bytes: u64,
    ) -> PyResult<()> {
        py.detach(|| {
            let mut state = self.lock()?;
            if !matches!(state.recording, Recording::Off)
                || state.driver.session().completed_updates() != 0
            {
                return Err(error(
                    "native recording requires a fresh zero-origin session",
                ));
            }
            let ledger = state
                .driver
                .session()
                .journal_size_ledger()
                .map_err(error)?;
            if max_journal_bytes == 0
                || max_working_bytes == 0
                || ledger.journal_bytes > max_journal_bytes
                || ledger.writer_working_bytes > max_working_bytes
            {
                return Err(error("native journal byte limit exceeded"));
            }
            let path = crate::model_refinement::path(path)?;
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
                .open(path)
                .map_err(error)?;
            state.recording = Recording::Failed;
            let writer = NativePvJournalWriter::new(
                file,
                state.driver.session(),
                NativePvJournalLimits {
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
    fn frame(
        &self,
        py: Python<'_>,
        owner: &str,
        revision: u64,
        include_scales: bool,
    ) -> PyResult<(NativePackedMatrix, Option<Py<PyAny>>)> {
        let (matrix, leaf) = py.detach(|| {
            let state = self.lock()?;
            let session = state.driver.session();
            if session.completed_updates() != revision {
                return Err(error("stale native training revision"));
            }
            let index = session
                .plan()
                .owners
                .iter()
                .position(|n| n == owner)
                .ok_or_else(|| error("native owner absent"))?;
            let handle = state.driver.handles().nth(index).expect("bound owner").1;
            let matrix = self.specs[index].with_tensor(handle.clone());
            let leaf = if include_scales {
                let weight = session.weights().nth(index).expect("bound owner");
                let count = count(weight)?;
                if count
                    .checked_mul(8)
                    .is_none_or(|n| n > matrix.dimensions().2)
                {
                    return Err(error("native P/V gradient leaf exceeds output byte limit"));
                }
                let mut values = zeros(count)?;
                for (dst, &value) in values.iter_mut().zip(weight.scales()) {
                    *dst = f64::from(value);
                }
                Some(values)
            } else {
                None
            };
            Ok::<_, PyErr>((matrix, leaf))
        })?;
        let leaf = leaf
            .map(|v| export_array_f64(py, &[v.len()], v, "native P/V gradient leaf"))
            .transpose()?;
        Ok((matrix, leaf))
    }
    fn step(
        &self,
        py: Python<'_>,
        revision: u64,
        window: Vec<u8>,
        gradients: Vec<Option<Dlpack>>,
    ) -> PyResult<(usize, usize)> {
        let mut values = Vec::new();
        let mut any = false;
        {
            let state = self.lock()?;
            let session = state.driver.session();
            if session.completed_updates() != revision {
                return Err(error("stale native training revision"));
            }
            let expected = session
                .plan()
                .training
                .get(revision as usize)
                .ok_or_else(|| error("native training schedule exhausted"))?;
            if window.as_slice() != expected.content {
                return Err(error("native training window differs"));
            }
            if gradients.len() != state.driver.handles().len() {
                return Err(error("native gradient owner count differs"));
            }
            let total = session.weights().try_fold(0u64, |n, w| {
                n.checked_add(count(w)? as u64)
                    .ok_or_else(|| error("native gradient size overflow"))
            })?;
            if total
                .checked_mul(4)
                .and_then(|n| n.checked_add(session.size_ledger().update_workspace_bytes))
                .is_none_or(|n| n > session.plan().limits.max_workspace_bytes)
            {
                return Err(error(
                    "native gradient transfer/update exceeds workspace byte limit",
                ));
            }
            for (g, w) in gradients.iter().zip(session.weights()) {
                if let Some(g) = g {
                    if tensor_shape(g, "P/V gradient")? != [count(w)?] {
                        return Err(error("native P/V gradient shape differs"));
                    }
                    if g.cpu_data_slice::<f64>()
                        .map_err(error)?
                        .iter()
                        .any(|&v| !v.is_finite() || !(v as f32).is_finite())
                    {
                        return Err(error("native P/V gradient is non-finite or overflows f32"));
                    }
                }
            }
            values.try_reserve_exact(gradients.len()).map_err(error)?;
            for (g, w) in gradients.iter().zip(session.weights()) {
                let mut v = Vec::new();
                v.try_reserve_exact(count(w)?).map_err(error)?;
                if let Some(g) = g {
                    any = true;
                    v.extend(
                        g.cpu_data_slice::<f64>()
                            .map_err(error)?
                            .iter()
                            .map(|&v| v as f32),
                    );
                } else {
                    v.resize(count(w)?, 0.0);
                }
                values.push(v);
            }
        }
        if !any {
            return Err(error("native training has no owner gradients"));
        }
        py.detach(|| {
            let mut state = self.lock()?;
            if state.driver.session().completed_updates() != revision {
                return Err(error("stale native training revision"));
            }
            state.update(&values)
        })
    }
    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| {
            self.lock()?
                .driver
                .session()
                .checkpoint_bytes()
                .map_err(error)
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn publish(
        &self,
        py: Python<'_>,
        output_dir: &str,
        run_id: String,
        machine: &str,
        validation_json: &str,
        limits_json: &str,
    ) -> PyResult<crate::model_refinement::NativeModelRefinement> {
        let crate::native_publication::Request {
            validation,
            limits,
            output,
            output_text,
        } = crate::native_publication::Request::new(output_dir, validation_json, limits_json)?;
        py.detach(|| {
            let mut state = self.lock()?;
            let session = state.driver.session();
            let cp_size = session.size_ledger().checkpoint_bytes;
            let ledger = session.journal_size_ledger().map_err(error)?;
            if cp_size > limits.max_parent_bytes
                || cp_size > limits.max_working_bytes
                || cp_size
                    .checked_add(ledger.journal_bytes)
                    .is_none_or(|n| n > limits.max_companion_bytes)
            {
                return Err(error(
                    "native checkpoint/companions exceed publication byte limit",
                ));
            }
            if ledger.journal_bytes > limits.max_journal_bytes
                || ledger.replay_working_bytes > limits.max_replay_working_bytes
            {
                return Err(error("native journal/replay byte limit exceeded"));
            }
            state.seal()?;
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
            let session = state.driver.session();
            let plan = session.plan();
            let checkpoint = session.checkpoint_bytes().map_err(error)?;
            let (mut parent, parent_view) =
                self.parent.refinement_parent_view(&plan.parent.level)?;
            let parents = decode_native_pv_parents(
                &mut parent,
                &parent_view,
                plan,
                NativePvParentReadLimits {
                    max_working_bytes: limits.max_parent_bytes,
                    extra_payload_bytes: checkpoint.len() as u64,
                },
            )
            .map_err(error)?;
            let replay = replay_native_pv_journal(
                plan.clone(),
                parents,
                &mut journal,
                summary.journal(),
                &checkpoint,
                NativePvJournalLimits {
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
                    tritium_evidence::NATIVE_PV_REPLAY_OBSERVED_REGISTRATION,
                ],
            )
            .map_err(error)?;
            let spool_path = directory.join(format!("publication-{ordinal}.spool"));
            let mut spool = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&spool_path)
                .map_err(error)?;
            let result = publish_native_pv_child(
                session,
                NativePvChildSource {
                    parent: &mut parent,
                    parent_evidence: &parent_view,
                    replay: &replay,
                    journal: &mut journal,
                    machine,
                    validation: &validation,
                },
                &mut spool,
                &mut tritium_quantize::engine::DirectoryTensorSink::new(&output),
                NativePvChildLimits {
                    max_spool_bytes: limits.max_spool_bytes,
                    max_working_bytes: limits.max_working_bytes,
                    max_evidence_bytes: limits.max_evidence_bytes,
                    max_companion_bytes: limits.max_companion_bytes,
                    receiving: tritium_format::TritPackageReadLimits {
                        max_source_bytes: limits.max_package_bytes,
                        max_manifest_bytes: limits.max_manifest_bytes,
                        max_directory_bytes: limits.max_directory_bytes,
                    },
                },
                evidence,
            );
            drop(spool);
            let cleanup = std::fs::remove_file(&spool_path);
            let child = result.map_err(error)?;
            let mut result = crate::model_refinement::NativeModelRefinement::from_published_pv(
                output_text,
                child,
                crate::model_refinement::ReadLimits {
                    package: limits.max_package_bytes,
                    evidence: limits.max_evidence_bytes,
                    checkpoint: plan.limits.max_checkpoint_bytes,
                    journal: limits.max_journal_bytes,
                },
            );
            if let Err(e) = cleanup {
                result.add_warning(format!("SpoolCleanup: {e}"));
            }
            Ok(result)
        })
    }
}

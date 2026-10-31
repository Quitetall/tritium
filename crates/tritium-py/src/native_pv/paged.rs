//! Recorded owner-paged CPU transport; no resident fallback or checkpoint PyBytes.
use super::{error, parse};
use crate::{
    model_level::NativePackedMatrix,
    native_parent::NativeTrainingParent,
    native_workspace::{Checkpoint, Workspace},
    torch_native::{export_array_f64, tensor_shape},
};
use dlpark::legacy::Dlpack;
use pyo3::{prelude::*, types::PyAny};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tritium_cpu::CpuBackend;
use tritium_runtime::TensorPayloadBudget;
use tritium_schema::NativeRefinementPlan;
use tritium_train::pv_tuning::*;
mod gradients;
mod publication;
mod recovery;
pub(crate) use recovery::NativePvPagedResumeBundle;
fn strict_u64(value: &Bound<'_, PyAny>) -> PyResult<u64> {
    if !value.is_exact_instance_of::<pyo3::types::PyInt>() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "paged limit requires an exact integer",
        ));
    }
    value.extract()
}

/// Private on-disk adjoints, one owner contribution plus bounded I/O scratch.
#[derive(Clone, Copy)]
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativePvPagedGradientLimits {
    #[pyo3(get)]
    pub max_spool_bytes: u64,
    #[pyo3(get)]
    pub max_owner_working_bytes: u64,
    #[pyo3(get)]
    pub max_contributions: u64,
}
#[pymethods]
impl NativePvPagedGradientLimits {
    #[new]
    #[pyo3(signature=(*,max_spool_bytes,max_owner_working_bytes,max_contributions))]
    fn new(
        #[pyo3(from_py_with=strict_u64)] max_spool_bytes: u64,
        #[pyo3(from_py_with=strict_u64)] max_owner_working_bytes: u64,
        #[pyo3(from_py_with=strict_u64)] max_contributions: u64,
    ) -> PyResult<Self> {
        let v = Self {
            max_spool_bytes,
            max_owner_working_bytes,
            max_contributions,
        };
        v.check()?;
        Ok(v)
    }
}
impl NativePvPagedGradientLimits {
    fn check(self) -> PyResult<()> {
        if self.max_spool_bytes < 8
            || self.max_owner_working_bytes < gradients::HEADROOM
            || self.max_contributions == 0
        {
            return Err(error(
                "capture limits require spool>=8, owner working>=16384 and positive contributions",
            ));
        }
        Ok(())
    }
}

/// Disjoint logical caps; excludes caller Torch buffers, metadata, OS and RSS.
#[derive(Clone, Copy)]
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativePvPagedTrainingLimits {
    #[pyo3(get)]
    pub max_owner_read_bytes: u64,
    #[pyo3(get)]
    pub max_owner_resident_bytes: u64,
    #[pyo3(get)]
    pub max_gradient_working_bytes: u64,
    #[pyo3(get)]
    pub max_workspace_bytes: u64,
    #[pyo3(get)]
    pub max_journal_bytes: u64,
    #[pyo3(get)]
    pub max_frame_bytes: u64,
    #[pyo3(get)]
    pub max_recording_working_bytes: u64,
}
#[pymethods]
impl NativePvPagedTrainingLimits {
    #[new]
    #[pyo3(signature=(*,max_owner_read_bytes,max_owner_resident_bytes,max_gradient_working_bytes,max_workspace_bytes,max_journal_bytes,max_frame_bytes,max_recording_working_bytes=16_384))]
    fn new(
        #[pyo3(from_py_with=strict_u64)] max_owner_read_bytes: u64,
        #[pyo3(from_py_with=strict_u64)] max_owner_resident_bytes: u64,
        #[pyo3(from_py_with=strict_u64)] max_gradient_working_bytes: u64,
        #[pyo3(from_py_with=strict_u64)] max_workspace_bytes: u64,
        #[pyo3(from_py_with=strict_u64)] max_journal_bytes: u64,
        #[pyo3(from_py_with=strict_u64)] max_frame_bytes: u64,
        #[pyo3(from_py_with=strict_u64)] max_recording_working_bytes: u64,
    ) -> PyResult<Self> {
        let v = Self {
            max_owner_read_bytes,
            max_owner_resident_bytes,
            max_gradient_working_bytes,
            max_workspace_bytes,
            max_journal_bytes,
            max_frame_bytes,
            max_recording_working_bytes,
        };
        v.check()?;
        Ok(v)
    }
}
impl NativePvPagedTrainingLimits {
    fn check(self) -> PyResult<()> {
        if [
            self.max_owner_read_bytes,
            self.max_owner_resident_bytes,
            self.max_gradient_working_bytes,
            self.max_workspace_bytes,
            self.max_journal_bytes,
            self.max_frame_bytes,
        ]
        .contains(&0)
            || self.max_recording_working_bytes < 16384
        {
            return Err(error(
                "paged training limits must be positive; recording scratch requires16384bytes",
            ));
        }
        Ok(())
    }
    fn owner(self) -> NativeRefinementOwnerReadLimits {
        NativeRefinementOwnerReadLimits {
            max_working_bytes: self.max_owner_read_bytes,
            extra_payload_bytes: 0,
        }
    }
    fn paging(self) -> NativePvPagedLimits {
        NativePvPagedLimits {
            max_resident_payload_bytes: self.max_owner_resident_bytes,
        }
    }
    fn recording(self) -> NativePvPagedRecordingLimits {
        NativePvPagedRecordingLimits {
            max_journal_bytes: self.max_journal_bytes,
            max_frame_bytes: self.max_frame_bytes,
            max_working_bytes: self.max_recording_working_bytes,
        }
    }
    fn replay(self) -> NativePvPagedReplayLimits {
        NativePvPagedReplayLimits {
            max_journal_bytes: self.max_journal_bytes,
            max_working_bytes: self.max_recording_working_bytes,
            paging: self.paging(),
        }
    }
}
fn overlap(a: &Path, b: &Path) -> bool {
    a.starts_with(b) || b.starts_with(a)
}
fn source<T>(
    parent: &NativeTrainingParent,
    plan: &NativeRefinementPlan,
    limits: NativePvPagedTrainingLimits,
    f: impl FnOnce(&mut NativeRefinementOwnerSource<'_, std::fs::File>) -> PyResult<T>,
) -> PyResult<T> {
    let (mut reader, view) = parent.refinement_parent_view(&plan.parent.level)?;
    if view.binding() != &plan.parent {
        return Err(error("paged training parent tuple differs"));
    }
    NativeRefinementOwnerSource::with(&mut reader, &view, plan, limits.owner(), f).map_err(error)
}
enum Recording {
    Writing(Box<NativePvPagedJournalWriter<Checkpoint>>),
    Sealed(Checkpoint, NativePvJournalSummary),
    Failed,
}
struct Runtime {
    driver: NativeCpuPvPagedSession<Checkpoint>,
    recording: Recording,
    specs: Vec<NativePackedMatrix>,
    capture_layout: Option<gradients::Layout>,
    pending: Option<gradients::Window<Checkpoint>>,
}
/// Single-controller eager CPU training with seek-backed optimizer state.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativePvPagedTraining {
    // Drop state/files before workspace; no recursive cleanup even on unwind.
    state: Mutex<Option<Runtime>>,
    workspace: Workspace,
    directory: PathBuf,
    parent: NativeTrainingParent,
    limits: NativePvPagedTrainingLimits,
    capture_limits: Option<NativePvPagedGradientLimits>,
}
impl NativePvPagedTraining {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Option<Runtime>>> {
        self.state
            .lock()
            .map_err(|_| error("paged training state lock poisoned"))
    }
    fn runtime(v: &mut Option<Runtime>) -> PyResult<&mut Runtime> {
        v.as_mut().ok_or_else(|| error("paged training is closed"))
    }
    fn pending<'a>(&self, s: &'a mut Runtime) -> PyResult<&'a mut gradients::Window<Checkpoint>> {
        let caps = self
            .capture_limits
            .ok_or_else(|| error("gradient capture is not enabled"))?;
        self.workspace.check().map_err(error)?;
        if s.pending.is_none() {
            let layout = s.capture_layout.as_ref().expect("capture layout").clone();
            let file = self
                .workspace
                .create_blob(layout.bytes, 8192)
                .map_err(error)?;
            s.pending =
                Some(gradients::Window::new(file, layout, caps.max_contributions).map_err(error)?);
        }
        Ok(s.pending.as_mut().expect("admitted pending gradients"))
    }
    fn capture_adjoint(
        &self,
        py: Python<'_>,
        owner: &str,
        revision: u64,
        matrix: &NativePackedMatrix,
        input: crate::native_scale::spans::Input<'_>,
    ) -> PyResult<()> {
        if self.capture_limits.is_none() {
            return Err(error("gradient capture is not enabled"));
        }
        let handle = matrix.shared_tensor();
        let dimensions = matrix.dimensions();
        py.detach(|| {
            let mut lock = self.lock()?;
            let s = Self::runtime(&mut lock)?;
            let result = (|| {
                if s.driver.session().completed_updates() != revision {
                    return Err(error("stale capture revision"));
                }
                if revision >= s.driver.session().plan().training.len() as u64
                    || !matches!(s.recording, Recording::Writing(_))
                {
                    return Err(error("capture schedule exhausted or recording sealed"));
                }
                let i = s
                    .driver
                    .session()
                    .plan()
                    .owners
                    .iter()
                    .position(|n| n == owner)
                    .ok_or_else(|| error("capture owner absent"))?;
                let current = s.driver.handles().nth(i).expect("bound owner").1;
                if !Arc::ptr_eq(current, &handle) {
                    return Err(error("foreign or stale capture handle"));
                }
                let mut program =
                    crate::native_scale::spans::Program::new(input, handle, dimensions)?;
                self.pending(s)?
                    .capture_from(i, program.count(), program.maximum_values, |start, out| {
                        program.fill(start, out)
                    })
                    .map_err(error)
            })();
            if result.is_err()
                && let Some(pending) = s.pending.as_mut()
            {
                pending.poisoned = true;
            }
            result
        })
    }

    fn update(
        &self,
        s: &mut Runtime,
        revision: u64,
        window: &[u8],
        mut gradient: impl FnMut(
            usize,
            &tritium_format::AdditiveTensor,
        ) -> PyResult<(Vec<f32>, Vec<f32>)>,
    ) -> PyResult<(usize, usize)> {
        let plan = s.driver.session().plan().clone();
        if s.driver.session().completed_updates() != revision {
            return Err(error("stale paged training revision"));
        }
        if plan
            .training
            .get(revision as usize)
            .is_none_or(|w| window != w.content)
        {
            return Err(error("paged training window differs or schedule exhausted"));
        }
        let Recording::Writing(journal) = &mut s.recording else {
            return Err(error("paged recording is sealed or failed"));
        };
        let ledger = s.driver.session().recording_size_ledger().map_err(error)?;
        let checkpoint = self
            .workspace
            .create_blob(
                s.driver.session().size_ledger().logical.checkpoint_bytes,
                8192,
            )
            .map_err(error)?;
        let frame = self
            .workspace
            .create_blob(ledger.frame_bytes, 8192)
            .map_err(error)?;
        let ready = source(&self.parent, &plan, self.limits, |parents| {
            s.driver
                .stage_recorded(
                    (checkpoint, frame),
                    |i, n| parents.load(i, n).map_err(error),
                    |i, _, _, w| gradient(i, w),
                    self.limits.recording(),
                )
                .map_err(error)
        })?;
        let (next, stats) = ready.commit(&mut s.driver, journal).map_err(error)?;
        s.driver = next;
        Ok((
            stats.changed_scales,
            stats.owners.iter().map(|o| o.changed_units).sum(),
        ))
    }
}

#[pyfunction]
#[pyo3(signature=(parent,plan_json,matrices,workspace_dir,*,limits,max_snapshot_bytes,resume=None,gradient_capture=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn _begin_paged_pv_training(
    py: Python<'_>,
    parent: &Bound<'_, PyAny>,
    plan_json: &str,
    matrices: Vec<(String, Py<NativePackedMatrix>)>,
    workspace_dir: PathBuf,
    limits: PyRef<'_, NativePvPagedTrainingLimits>,
    max_snapshot_bytes: u64,
    resume: Option<PyRef<'_, NativePvPagedResumeBundle>>,
    gradient_capture: Option<PyRef<'_, NativePvPagedGradientLimits>>,
) -> PyResult<NativePvPagedTraining> {
    let parent = NativeTrainingParent::extract(parent)?;
    let plan = parse(plan_json)?;
    let limits = *limits;
    limits.check()?;
    let capture_limits = gradient_capture.as_deref().copied();
    if let Some(l) = capture_limits {
        l.check()?;
    }
    if max_snapshot_bytes == 0 || matrices.iter().map(|x| &x.0).ne(plan.owners.iter()) {
        return Err(error("paged training owner inventory/snapshot cap differs"));
    }
    let specs: Vec<_> = matrices
        .iter()
        .map(|(_, m)| {
            let m = m.borrow(py);
            m.with_tensor(m.shared_tensor())
        })
        .collect();
    let roots: Vec<_> = plan
        .owners
        .iter()
        .cloned()
        .zip(specs.iter().map(NativePackedMatrix::shared_tensor))
        .collect();
    let directory = crate::model_refinement::new_output(
        workspace_dir
            .to_str()
            .ok_or_else(|| error("workspace path is not UTF-8"))?,
    )?;
    if overlap(&directory, &parent.root().canonicalize().map_err(error)?) {
        return Err(error("paged workspace cannot overlap parent"));
    }
    let resume = resume.as_deref();
    if let Some(b) = resume
        && overlap(&directory, &b.root)
    {
        return Err(error("paged workspace cannot overlap bundle"));
    }
    py.detach(|| {
        let workspace = Workspace::storage(directory.clone(), limits.max_workspace_bytes, 6)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    pyo3::exceptions::PyFileExistsError::new_err(e.to_string())
                } else {
                    error(e)
                }
            })?;
        let journal = workspace
            .create_blob(limits.max_journal_bytes, 8192)
            .map_err(error)?;
        let (mut state, journal) = if let Some(b) = resume {
            let bundle = b.reopen()?;
            if bundle.manifest().plan != plan {
                return Err(error("paged resume frozen plan differs"));
            }
            source(&parent, &plan, limits, |s| {
                bundle
                    .resume(
                        journal,
                        |i, n| s.load(i, n).map_err(error),
                        |_| {
                            workspace
                                .create_blob(plan.limits.max_checkpoint_bytes, 8192)
                                .map_err(error)
                        },
                        limits.replay(),
                    )
                    .map_err(error)
            })?
        } else {
            let mut state = source(&parent, &plan, limits, |s| {
                NativePvPagedSession::initialize(
                    plan.clone(),
                    limits.paging(),
                    |i, n| s.load(i, n),
                    workspace
                        .create_blob(plan.limits.max_checkpoint_bytes, 8192)
                        .map_err(error)?,
                )
                .map_err(error)
            })?;
            let journal = NativePvPagedJournalWriter::new(journal, &mut state, limits.recording())
                .map_err(error)?;
            (state, journal)
        };
        let budget = TensorPayloadBudget::new(max_snapshot_bytes);
        let roots = if resume.is_some() {
            source(&parent, &plan, limits, |s| {
                state
                    .visit_weights(
                        |i, n| s.load(i, n).map_err(error),
                        |i, n, w| {
                            let h = CpuBackend::new()
                                .snapshot_additive_weights(&roots[i].1, w.view(), &budget)
                                .map_err(error)?;
                            Ok((n.to_owned(), h))
                        },
                    )
                    .map_err(error)
            })?
        } else {
            roots
        };
        let driver = source(&parent, &plan, limits, |s| {
            NativeCpuPvPagedSession::bind(state, roots, budget, |i, n| s.load(i, n)).map_err(error)
        })?;
        let capture_layout = if let Some(caps) = capture_limits {
            let cpu = CpuBackend::new();
            let counts = driver
                .handles()
                .map(|(_, h)| {
                    let s = cpu.additive_stored_scale_count(&**h).map_err(error)?;
                    let w = cpu.additive_present_weight_count(&**h).map_err(error)?;
                    if s.checked_add(w)
                        .and_then(|n| (n as u64).checked_mul(4))
                        .is_none_or(|n| n > limits.max_gradient_working_bytes)
                    {
                        return Err(error("capture step gradient working cap exceeded"));
                    }
                    Ok((s, w))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Some(
                gradients::Layout::new(&counts, caps.max_spool_bytes, caps.max_owner_working_bytes)
                    .map_err(error)?,
            )
        } else {
            None
        };
        Ok(NativePvPagedTraining {
            state: Mutex::new(Some(Runtime {
                driver,
                recording: Recording::Writing(Box::new(journal)),
                specs,
                capture_layout,
                pending: None,
            })),
            workspace,
            directory,
            parent,
            limits,
            capture_limits,
        })
    })
}

#[pymethods]
impl NativePvPagedTraining {
    #[getter]
    fn completed_updates(&self) -> PyResult<u64> {
        Ok(Self::runtime(&mut *self.lock()?)?
            .driver
            .session()
            .completed_updates())
    }
    #[getter]
    fn workspace_report(&self) -> String {
        let mut r = self.workspace.report();
        if self.state.is_poisoned() {
            r.warnings
                .push("RuntimeCleanup: training lock poisoned; no state admission".into());
        }
        let lock = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pending = lock.as_ref().and_then(|s| s.pending.as_ref());
        serde_json::json!({"storage":"owner-paged","reserved_bytes":r.reserved_bytes,"files":r.files,
            "peak_reserved_bytes":r.peak_reserved_bytes,"peak_files":r.peak_files,"cleaned":r.cleaned,"cleanup_warnings":r.warnings,
            "gradient_capture_enabled":self.capture_limits.is_some(),"pending_gradient_bytes":pending.map_or(0,|p|p.bytes()),
            "pending_contributions":pending.map_or(0,|p|p.contributions),"pending_gradient_poisoned":pending.is_some_and(|p|p.poisoned),
            "quality_qualified":false,"physical_memory_qualified":false}).to_string()
    }
    fn frame(
        &self,
        py: Python<'_>,
        owner: &str,
        revision: u64,
        include_scales: bool,
    ) -> PyResult<(NativePackedMatrix, Option<Py<PyAny>>)> {
        let (matrix, leaf) = py.detach(|| {
            let mut lock = self.lock()?;
            let s = Self::runtime(&mut lock)?;
            if s.driver.session().completed_updates() != revision {
                return Err(error("stale paged training revision"));
            }
            let i = s
                .driver
                .session()
                .plan()
                .owners
                .iter()
                .position(|n| n == owner)
                .ok_or_else(|| error("paged owner absent"))?;
            let h = s.driver.handles().nth(i).expect("bound owner").1;
            let matrix = s.specs[i].with_tensor(h.clone());
            let leaf = if include_scales {
                let cpu = CpuBackend::new();
                let count = cpu
                    .additive_stored_scale_count(&**h)
                    .map_err(error)?
                    .checked_add(cpu.additive_present_weight_count(&**h).map_err(error)?)
                    .ok_or_else(|| error("paged leaf size overflow"))?;
                if self.capture_limits.is_none()
                    && count
                        .checked_mul(8)
                        .is_none_or(|n| n > matrix.dimensions().2)
                {
                    return Err(error("paged gradient leaf exceeds output cap"));
                }
                // Values are private transport only; forward uses immutable packed h.
                if self.capture_limits.is_some() && matrix.dimensions().2 < 8 {
                    return Err(error("capture scalar token exceeds output cap"));
                }
                let count = if self.capture_limits.is_some() {
                    self.pending(s)?;
                    1
                } else {
                    count
                };
                Some(crate::native_scale::zeros(count)?)
            } else {
                None
            };
            Ok::<_, PyErr>((matrix, leaf))
        })?;
        let leaf = leaf
            .map(|v| export_array_f64(py, &[v.len()], v, "paged P/V gradient leaf"))
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
        if self.capture_limits.is_some() {
            return Err(error(
                "capture mode requires step_captured, not caller gradients",
            ));
        }
        let values = gradients
            .iter()
            .map(|g| {
                g.as_ref()
                    .map(|g| {
                        let shape = tensor_shape(g, "paged P/V gradient")?;
                        let v = g.cpu_data_slice::<f64>().map_err(error)?;
                        if shape != [v.len()]
                            || v.iter().any(|&x| !x.is_finite() || !(x as f32).is_finite())
                        {
                            return Err(error("paged gradient shape/nonfinite/overflow"));
                        }
                        Ok(v)
                    })
                    .transpose()
            })
            .collect::<PyResult<Vec<_>>>()?;
        py.detach(|| {
            let mut lock = self.lock()?;
            let s = Self::runtime(&mut lock)?;
            let plan = s.driver.session().plan().clone();
            if s.driver.session().completed_updates() != revision {
                return Err(error("stale paged training revision"));
            }
            if plan
                .training
                .get(revision as usize)
                .is_none_or(|w| window.as_slice() != w.content)
            {
                return Err(error("paged training window differs or schedule exhausted"));
            }
            if values.len() != plan.owners.len() || values.iter().all(Option::is_none) {
                return Err(error("paged gradient owner inventory/empty gradients"));
            }
            for ((_, h), v) in s.driver.handles().zip(&values) {
                let cpu = CpuBackend::new();
                let count = cpu
                    .additive_stored_scale_count(&**h)
                    .map_err(error)?
                    .checked_add(cpu.additive_present_weight_count(&**h).map_err(error)?)
                    .ok_or_else(|| error("paged gradient size overflow"))?;
                if v.is_some_and(|v| v.len() != count)
                    || (count as u64)
                        .checked_mul(4)
                        .is_none_or(|n| n > self.limits.max_gradient_working_bytes)
                {
                    return Err(error("paged gradient shape/working cap exceeded"));
                }
            }
            self.update(s, revision, &window, |i, w| {
                let split = w.scales().len();
                let count = super::count(w)?;
                let mut scales = Vec::new();
                let mut weights = Vec::new();
                scales.try_reserve_exact(split).map_err(error)?;
                weights.try_reserve_exact(count - split).map_err(error)?;
                if let Some(v) = values[i] {
                    scales.extend(v[..split].iter().map(|&x| x as f32));
                    weights.extend(v[split..].iter().map(|&x| x as f32));
                } else {
                    scales.resize(split, 0.);
                    weights.resize(count - split, 0.);
                }
                Ok((scales, weights))
            })
        })
    }
    fn admit_capture(&self, owner: &str, revision: u64) -> PyResult<()> {
        let mut lock = self.lock()?;
        let s = Self::runtime(&mut lock)?;
        if s.driver.session().completed_updates() != revision {
            return Err(error("stale capture revision"));
        }
        if revision >= s.driver.session().plan().training.len() as u64
            || !matches!(s.recording, Recording::Writing(_))
        {
            return Err(error("capture schedule exhausted or recording sealed"));
        }
        let i = s
            .driver
            .session()
            .plan()
            .owners
            .iter()
            .position(|n| n == owner)
            .ok_or_else(|| error("capture owner absent"))?;
        self.pending(s)?.admit(i).map_err(error)
    }
    fn capture_gradient(
        &self,
        py: Python<'_>,
        owner: &str,
        revision: u64,
        values: Dlpack,
    ) -> PyResult<()> {
        let shape = tensor_shape(&values, "captured gradient")?;
        let values = values.cpu_data_slice::<f64>().map_err(error)?;
        if shape != [values.len()] {
            return Err(error("capture gradient must be a vector"));
        }
        py.detach(|| {
            self.admit_capture(owner, revision)?;
            let mut lock = self.lock()?;
            let s = Self::runtime(&mut lock)?;
            // Recheck after admission; direct callers cannot cross a committed step.
            if s.driver.session().completed_updates() != revision {
                return Err(error("stale capture revision"));
            }
            let i = s
                .driver
                .session()
                .plan()
                .owners
                .iter()
                .position(|n| n == owner)
                .ok_or_else(|| error("capture owner absent"))?;
            self.pending(s)?.capture(i, values).map_err(error)
        })
    }
    fn capture_matmul_gradient(
        &self,
        py: Python<'_>,
        owner: &str,
        revision: u64,
        matrix: &NativePackedMatrix,
        input: Dlpack,
        gradient: Dlpack,
    ) -> PyResult<()> {
        let input = crate::native_scale::spans::Input::matmul(matrix, &input, &gradient)?;
        self.capture_adjoint(py, owner, revision, matrix, input)
    }
    fn capture_embed_gradient(
        &self,
        py: Python<'_>,
        owner: &str,
        revision: u64,
        matrix: &NativePackedMatrix,
        indices: Dlpack,
        gradient: Dlpack,
    ) -> PyResult<()> {
        let input = crate::native_scale::spans::Input::embed(matrix, &indices, &gradient)?;
        self.capture_adjoint(py, owner, revision, matrix, input)
    }
    fn poison_capture(&self) {
        if let Ok(mut lock) = self.state.lock()
            && let Some(s) = lock.as_mut()
            && let Some(p) = s.pending.as_mut()
        {
            p.poisoned = true;
        }
    }
    fn clear_capture(&self) -> PyResult<()> {
        let mut lock = self.lock()?;
        let s = Self::runtime(&mut lock)?;
        if self.capture_limits.is_none() {
            return Err(error("gradient capture is not enabled"));
        }
        drop(s.pending.take());
        // Admit fresh storage before exposing the next backward window.
        self.pending(s)?;
        Ok(())
    }
    fn step_captured(
        &self,
        py: Python<'_>,
        revision: u64,
        window: Vec<u8>,
    ) -> PyResult<(usize, usize)> {
        if self.capture_limits.is_none() {
            return Err(error("gradient capture is not enabled"));
        }
        py.detach(|| {
            let mut lock = self.lock()?;
            let s = Self::runtime(&mut lock)?;
            let mut pending = s
                .pending
                .take()
                .ok_or_else(|| error("no captured gradients"))?;
            if pending.poisoned || pending.contributions == 0 {
                s.pending = Some(pending);
                return Err(error(
                    "captured window empty or poisoned; zero_grad required",
                ));
            }
            let result = self.update(s, revision, &window, |i, w| {
                let scales = w.scales().len();
                let count = super::count(w)?;
                pending.gradients(i, scales, count - scales).map_err(error)
            });
            // Only committed state consumes observations. Uncertain spool reads
            // remain poisoned; source-only refusal preserves valid pending data.
            if result.is_err() {
                s.pending = Some(pending);
            }
            result
        })
    }
    fn checkpoint(&self) -> PyResult<()> {
        Err(error(
            "paged training does not materialize checkpoint bytes; use save_resume",
        ))
    }
    fn save_resume(
        &self,
        py: Python<'_>,
        output_dir: &str,
        limits_json: &str,
    ) -> PyResult<NativePvPagedResumeBundle> {
        recovery::save(self, py, output_dir, limits_json)
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
        publication::publish(
            self,
            py,
            output_dir,
            run_id,
            machine,
            validation_json,
            limits_json,
        )
    }
    fn close_recording(&self) -> PyResult<String> {
        // Retirement does not admit or inspect poisoned numerical state. Always
        // release owned files; ordinary operations still refuse a poisoned lock.
        drop(
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take(),
        );
        let _ = self.workspace.finish();
        Ok(self.workspace_report())
    }
}

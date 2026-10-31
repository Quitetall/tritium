//! Shared logical admission and one capacity-only cache for two real adapters.
use super::*;
use tritium_runtime::{TensorPayloadBudget, TensorStorageReservation};

/// Buffer family, not a weight or numerical-policy admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SwiGluWorkspaceKind {
    SharedTensor,
    PackedQtea,
}

/// Pure logical footprint; not actual capacities, weight admission or physical VRAM/RSS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SwiGluWorkspacePlan {
    /// Buffer family requested, not a tensor/weight grant.
    pub kind: SwiGluWorkspaceKind,
    /// Active activation rows.
    pub batch: usize,
    /// Input/output feature width.
    pub width: usize,
    /// Gate/up feature width.
    pub hidden: usize,
    /// Requested CUDA buffer bytes, including unit scales/rotation if needed and status.
    pub device_bytes: u64,
    /// Conservative host construction envelope including temporary unit scales.
    pub host_construction_bytes: u64,
    /// Minimum retained output capacity plus the four-byte host status allowance.
    pub host_retained_bytes: u64,
}
fn invalid(message: &str) -> BackendError {
    BackendError::InvalidInput(message.into())
}
fn elements(a: usize, b: usize) -> Result<usize, BackendError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("SwiGLU workspace element overflow"))
}
fn bytes(n: usize) -> Result<u64, BackendError> {
    u64::try_from(elements(n, 4)?).map_err(|_| invalid("SwiGLU workspace bytes exceed u64"))
}
fn add(a: u64, b: u64) -> Result<u64, BackendError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("SwiGLU workspace byte sum overflow"))
}
impl SwiGluWorkspacePlan {
    /// Plan buffer extents without a device, allocation or weight validation.
    /// # Errors
    /// Zero dimensions, extent overflow or unsupported index ABI.
    pub fn new(
        kind: SwiGluWorkspaceKind,
        batch: usize,
        width: usize,
        hidden: usize,
    ) -> Result<Self, BackendError> {
        if width == 0 || hidden == 0 {
            return Err(invalid("SwiGLU workspace dimensions must be positive"));
        }
        let input = elements(batch, width)?;
        let mid = elements(batch, hidden)?;
        u32::try_from(input).map_err(|_| invalid("SwiGLU workspace input exceeds u32 ABI"))?;
        u32::try_from(mid).map_err(|_| invalid("SwiGLU workspace hidden exceeds u32 ABI"))?;
        if kind == SwiGluWorkspaceKind::SharedTensor {
            CudaBackend::check_grad_launch_bounds(batch, hidden, width)?;
            CudaBackend::check_grad_launch_bounds(batch, width, hidden)?;
        }
        let mut plan = Self {
            kind,
            batch,
            width,
            hidden,
            device_bytes: 0,
            host_construction_bytes: 0,
            host_retained_bytes: 0,
        };
        if batch == 0 {
            return Ok(plan);
        }
        let shared = kind == SwiGluWorkspaceKind::SharedTensor;
        let input_bytes = bytes(input)?;
        let mid_bytes = bytes(mid)?;
        let scales = if shared { bytes(width.max(hidden))? } else { 0 };
        plan.device_bytes = add(
            add(
                input_bytes
                    .checked_mul(if shared { 3 } else { 2 })
                    .ok_or_else(|| invalid("SwiGLU input byte sum overflow"))?,
                mid_bytes
                    .checked_mul(2)
                    .ok_or_else(|| invalid("SwiGLU hidden byte sum overflow"))?,
            )?,
            add(scales, 4)?,
        )?;
        plan.host_retained_bytes = add(input_bytes, 4)?;
        plan.host_construction_bytes = add(plan.host_retained_bytes, scales)?;
        Ok(plan)
    }
}

/// Explicit shared logical budgets. Clones coordinate; independent ledgers do not.
/// May share a parent/domain ledger with matrices or other models. Not physical fit.
#[derive(Clone, Debug)]
pub struct SwiGluWorkspaceResources {
    device: TensorPayloadBudget,
    host: TensorPayloadBudget,
}
impl SwiGluWorkspaceResources {
    /// Clone positive ledgers without reserving or enabling numerical execution.
    /// # Errors
    /// Zero device/host limits.
    pub fn new(
        device: &TensorPayloadBudget,
        host: &TensorPayloadBudget,
    ) -> Result<Self, BackendError> {
        if device.limit_bytes() == 0 || host.limit_bytes() == 0 {
            return Err(invalid("SwiGLU workspace budgets must be positive"));
        }
        Ok(Self {
            device: device.clone(),
            host: host.clone(),
        })
    }
}

/// Descriptive current cache state, not an execution/physical-memory receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SwiGluWorkspaceSnapshot {
    /// Current capacity geometry and nominal requests.
    pub plan: SwiGluWorkspacePlan,
    /// Whether explicit lifetime reservations cover this cache.
    pub admitted: bool,
    /// This workspace's held device charge, not total shared-ledger usage.
    pub device_reserved_bytes: u64,
    /// This workspace's held host charge, not total shared-ledger usage.
    pub host_reserved_bytes: u64,
    /// Saturating counts of successful cache construction/selections, not successful inference.
    pub allocations: u64,
    /// Existing-capacity selections, including selections which later refuse.
    pub reuse_selections: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn footprint_plans_and_positive_resources_require_no_device() {
        let shared = SwiGluWorkspacePlan::new(SwiGluWorkspaceKind::SharedTensor, 1, 4, 8).unwrap();
        assert_eq!(
            (
                shared.device_bytes,
                shared.host_construction_bytes,
                shared.host_retained_bytes
            ),
            (148, 52, 20)
        );
        let qtea = SwiGluWorkspacePlan::new(SwiGluWorkspaceKind::PackedQtea, 1, 4, 8).unwrap();
        assert_eq!(
            (
                qtea.device_bytes,
                qtea.host_construction_bytes,
                qtea.host_retained_bytes
            ),
            (100, 20, 20)
        );
        let empty = SwiGluWorkspacePlan::new(SwiGluWorkspaceKind::SharedTensor, 0, 4, 8).unwrap();
        assert_eq!(
            (
                empty.device_bytes,
                empty.host_construction_bytes,
                empty.host_retained_bytes
            ),
            (0, 0, 0)
        );
        assert!(SwiGluWorkspacePlan::new(SwiGluWorkspaceKind::PackedQtea, 1, 0, 8).is_err());
        assert!(
            SwiGluWorkspacePlan::new(SwiGluWorkspaceKind::PackedQtea, usize::MAX, 4, 8).is_err()
        );
        assert!(
            SwiGluWorkspacePlan::new(SwiGluWorkspaceKind::SharedTensor, u32::MAX as usize, 1, 1)
                .is_err()
        );
        let d = TensorPayloadBudget::new(150);
        let h = TensorPayloadBudget::new(52);
        let z = TensorPayloadBudget::new(0);
        assert!(SwiGluWorkspaceResources::new(&z, &h).is_err());
        assert!(SwiGluWorkspaceResources::new(&d, &z).is_err());
        let _resources = SwiGluWorkspaceResources::new(&d, &h).unwrap();
        assert_eq!((d.used_bytes(), h.used_bytes()), (0, 0));
    }
}
#[derive(Debug, Default)]
pub(super) struct Cache {
    workspace: Option<Workspace>,
    allocations: u64,
    reuse_selections: u64,
}
struct Reservations {
    device: TensorStorageReservation,
    host: TensorStorageReservation,
}
pub(super) struct Workspace {
    pub(super) plan: SwiGluWorkspacePlan,
    pub(super) input: CudaSlice<f32>,
    pub(super) rotated: Option<CudaSlice<f32>>,
    pub(super) gate: CudaSlice<f32>,
    pub(super) up: CudaSlice<f32>,
    pub(super) output: CudaSlice<f32>,
    pub(super) ones: Option<CudaSlice<f32>>,
    pub(super) host_output: Vec<f32>,
    pub(super) invalid: CudaSlice<u32>,
    // All owned buffers drop before their charges release.
    reservations: Option<Reservations>,
}
impl core::fmt::Debug for Workspace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SwiGluWorkspace")
            .field("plan", &self.plan)
            .field(
                "host_output_capacity_elements",
                &self.host_output.capacity(),
            )
            .field("admitted", &self.reservations.is_some())
            .finish_non_exhaustive()
    }
}
fn host_values(n: usize, value: f32) -> Result<Vec<f32>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(n)
        .map_err(|_| BackendError::OutOfMemory {
            requested: n.saturating_mul(4),
        })?;
    values.resize(n, value);
    Ok(values)
}
impl Workspace {
    fn new(
        backend: &CudaBackend,
        plan: SwiGluWorkspacePlan,
        resources: Option<&SwiGluWorkspaceResources>,
        call_limits: Option<QteaCudaLimits>,
    ) -> Result<Self, BackendError> {
        let mut reservations = resources
            .map(|r| -> Result<_, BackendError> {
                let device = r.device.reserve_storage(plan.device_bytes)?;
                let host = r.host.reserve_storage(plan.host_construction_bytes)?;
                Ok(Reservations { device, host })
            })
            .transpose()?;
        let input = elements(plan.batch, plan.width)?;
        let hidden = elements(plan.batch, plan.hidden)?;
        let host_output = host_values(input, 0.)?;
        let shared = plan.kind == SwiGluWorkspaceKind::SharedTensor;
        let host_ones = if shared {
            Some(host_values(plan.width.max(plan.hidden), 1.)?)
        } else {
            None
        };
        let retained = add(bytes(host_output.capacity())?, 4)?;
        let construction = add(
            retained,
            host_ones
                .as_ref()
                .map(|v| bytes(v.capacity()))
                .transpose()?
                .unwrap_or(0),
        )?;
        if let Some(limits) = call_limits
            && retained
                > u64::try_from(limits.max_host_staging_bytes)
                    .map_err(|_| invalid("host staging cap exceeds u64"))?
        {
            return Err(invalid(
                "SwiGLU actual host capacity exceeds per-call limit",
            ));
        }
        if let Some(guards) = &mut reservations {
            guards.host.resize(construction)?;
        }
        let alloc = |n| {
            backend.stream.alloc_zeros::<f32>(n).map_err(|e| {
                alloc_or_backend("SwiGLU workspace allocation", &e, n.saturating_mul(4))
            })
        };
        let device_input = alloc(input)?;
        let rotated = if shared { Some(alloc(input)?) } else { None };
        let gate = alloc(hidden)?;
        let up = alloc(hidden)?;
        let output = alloc(input)?;
        let ones = host_ones
            .as_ref()
            .map(|v| {
                backend
                    .stream
                    .clone_htod(v)
                    .map_err(|e| alloc_or_backend("SwiGLU workspace unit scales", &e, v.len() * 4))
            })
            .transpose()?;
        let invalid = backend
            .stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| alloc_or_backend("SwiGLU workspace status", &e, 4))?;
        drop(host_ones);
        if let Some(guards) = &mut reservations {
            guards.host.resize(retained)?;
        }
        Ok(Self {
            plan,
            input: device_input,
            rotated,
            gate,
            up,
            output,
            ones,
            host_output,
            invalid,
            reservations,
        })
    }
    fn check_call(&self, limits: Option<QteaCudaLimits>) -> Result<(), BackendError> {
        if let Some(limits) = limits
            && (self.plan.device_bytes > limits.max_device_staging_bytes
                || add(bytes(self.host_output.capacity())?, 4)?
                    > u64::try_from(limits.max_host_staging_bytes)
                        .map_err(|_| invalid("host staging cap exceeds u64"))?)
        {
            return Err(invalid("cached SwiGLU workspace exceeds per-call cap"));
        }
        Ok(())
    }
}

impl CudaBackend {
    /// Opt into one budgeted shared/QTEA capacity cache. Configure before loading.
    /// Clears obsolete private scratch, not weights/model state, and does not
    /// enable device arithmetic. Independent ledgers do not coordinate.
    /// # Errors
    /// Poisoned cache. Resource positivity is checked by the resources constructor.
    pub fn with_swiglu_workspace_resources(
        mut self,
        resources: SwiGluWorkspaceResources,
    ) -> Result<Self, BackendError> {
        drop(
            self.swiglu_workspace
                .get_mut()
                .map_err(|_| invalid("SwiGLU cache poisoned"))?
                .workspace
                .take(),
        );
        self.swiglu_workspace_resources = Some(resources);
        if !self.device_id.contains(":swiglu-workspace-bounded") {
            self.device_id.push_str(":swiglu-workspace-bounded");
        }
        Ok(self)
    }
    /// Whether the explicit logical workspace policy is selected, not execution evidence.
    pub fn swiglu_workspace_admission_enabled(&self) -> bool {
        self.swiglu_workspace_resources.is_some()
    }
    /// Read metadata only; no synchronization or activation readback/qualification.
    /// # Errors
    /// Poisoned cache.
    pub fn swiglu_workspace_snapshot(
        &self,
    ) -> Result<Option<SwiGluWorkspaceSnapshot>, BackendError> {
        let cache = self
            .swiglu_workspace
            .lock()
            .map_err(|_| invalid("SwiGLU cache poisoned"))?;
        cache
            .workspace
            .as_ref()
            .map(|w| -> Result<_, BackendError> {
                Ok(SwiGluWorkspaceSnapshot {
                    plan: w.plan,
                    admitted: w.reservations.is_some(),
                    device_reserved_bytes: w
                        .reservations
                        .as_ref()
                        .map(|g| g.device.reserved_bytes())
                        .unwrap_or(0),
                    host_reserved_bytes: w
                        .reservations
                        .as_ref()
                        .map(|g| g.host.reserved_bytes())
                        .unwrap_or(0),
                    allocations: cache.allocations,
                    reuse_selections: cache.reuse_selections,
                })
            })
            .transpose()
    }
    /// Release only this backend's idle capacity cache; never reset a device/model.
    /// CUDA/host allocator pools may retain physical storage after logical release.
    /// # Errors
    /// Poisoned cache.
    pub fn release_swiglu_workspace(&self) -> Result<(), BackendError> {
        drop(
            self.swiglu_workspace
                .lock()
                .map_err(|_| invalid("SwiGLU cache poisoned"))?
                .workspace
                .take(),
        );
        Ok(())
    }
    pub(super) fn with_swiglu_workspace<R>(
        &self,
        kind: SwiGluWorkspaceKind,
        batch: usize,
        width: usize,
        hidden: usize,
        limits: Option<QteaCudaLimits>,
        call: impl FnOnce(&mut Workspace, bool, bool) -> Result<R, BackendError>,
    ) -> Result<R, BackendError> {
        let plan = SwiGluWorkspacePlan::new(kind, batch, width, hidden)?;
        if batch == 0 {
            return Err(invalid("empty SwiGLU call needs no workspace"));
        }
        if let Some(limits) = limits
            && (limits.max_device_staging_bytes == 0
                || limits.max_host_staging_bytes == 0
                || plan.device_bytes > limits.max_device_staging_bytes
                || plan.host_retained_bytes
                    > u64::try_from(limits.max_host_staging_bytes)
                        .map_err(|_| invalid("host cap exceeds u64"))?)
        {
            return Err(invalid("SwiGLU requested workspace exceeds per-call cap"));
        }
        let resources = self.swiglu_workspace_resources.as_ref();
        if resources.is_none() && kind == SwiGluWorkspaceKind::PackedQtea {
            let mut local = Workspace::new(self, plan, None, limits)?;
            return call(&mut local, false, false);
        }
        let mut cache = self
            .swiglu_workspace
            .lock()
            .map_err(|_| invalid("SwiGLU cache poisoned"))?;
        let reused = cache.workspace.as_ref().is_some_and(|w| w.plan == plan);
        if !reused {
            if resources.is_some() {
                // Admitted replacement never overlaps obsolete functional scratch.
                drop(cache.workspace.take());
            }
            // Legacy shared execution keeps its old transactional replacement
            // policy when admission was not selected.
            let replacement = Workspace::new(self, plan, resources, limits)?;
            cache.workspace = Some(replacement);
            cache.allocations = cache.allocations.saturating_add(1);
        } else {
            cache.reuse_selections = cache.reuse_selections.saturating_add(1);
        }
        let w = cache
            .workspace
            .as_mut()
            .ok_or_else(|| invalid("SwiGLU cache absent"))?;
        w.check_call(limits)?;
        call(w, reused, true)
    }
}

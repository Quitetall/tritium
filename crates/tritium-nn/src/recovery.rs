//! Deterministic resident-CUDA execution for short SALT recovery runs.
//!
//! HESTIA supplies a smooth soft phase. Packed SALT with identity STE supplies
//! the hard training tail. Exact discrete PV polishing remains a separate
//! stage; this module deliberately does not relabel hard STE as PV.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use tritium_cuda::CudaBackend;
use tritium_cuda::train::{
    CheckpointPolicy, DeviceBackwardStats, DevicePackedSaltWeight, DeviceTape, DeviceTensor,
    DeviceTrainParam, DeviceTrainer, DeviceTrainerWeightStorage, GradientLeafBinding,
};
use tritium_train::{
    AdamW, RecoveryPhase, RecoverySchedule, TemperatureError, TemperatureSchedule,
};

use crate::training::{
    TiedSwiGluTrainingModel, TrainingAdapterError, hestia_device_forward, packed_device_forward,
};

const PLAN_SCHEMA: &str = "tritium.device-recovery-plan.v1";
const PLAN_FILE: &str = "recovery-plan.json";
const MAX_PLAN_BYTES: u64 = 1024 * 1024;
static PLAN_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Estimator executed by one recovery step.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeviceRecoveryEstimator {
    /// Smooth HESTIA expectation over `{-1, 0, +1}`.
    Hestia,
    /// Packed hard forward with identity straight-through gradient.
    PackedSte,
}

/// Recovery phase executed by one step.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeviceRecoveryPhase {
    /// Differentiable first 80% of the schedule.
    Soft,
    /// Packed hard final 20% of the schedule.
    Hard,
}

impl From<RecoveryPhase> for DeviceRecoveryPhase {
    fn from(value: RecoveryPhase) -> Self {
        match value {
            RecoveryPhase::Soft => Self::Soft,
            RecoveryPhase::Hard => Self::Hard,
        }
    }
}

#[derive(Clone, Debug)]
enum SoftEstimator {
    Hestia(TemperatureSchedule),
    PackedSte,
}

/// Immutable execution policy for a resident short-recovery session.
#[derive(Clone, Debug)]
pub struct DeviceRecoveryConfig {
    schedule: RecoverySchedule,
    soft_estimator: SoftEstimator,
    salt_planes: usize,
    optimizer: AdamW,
}

impl DeviceRecoveryConfig {
    /// Build HESTIA-soft then packed-hard recovery.
    ///
    /// # Errors
    /// Rejects a temperature policy bound to another recovery schedule or an
    /// invalid packed-plane/optimizer configuration.
    pub fn hestia(
        schedule: RecoverySchedule,
        temperature: TemperatureSchedule,
        salt_planes: usize,
        optimizer: AdamW,
    ) -> Result<Self, DeviceRecoveryError> {
        if temperature.recovery_schedule() != schedule {
            return Err(DeviceRecoveryError::InvalidInput(
                "HESTIA temperature policy belongs to a different recovery schedule".into(),
            ));
        }
        let config = Self {
            schedule,
            soft_estimator: SoftEstimator::Hestia(temperature),
            salt_planes,
            optimizer,
        };
        config.validate()?;
        Ok(config)
    }

    /// Build packed-STE soft and hard recovery for the matched control run.
    ///
    /// # Errors
    /// Rejects an invalid packed-plane/optimizer configuration.
    pub fn packed_ste(
        schedule: RecoverySchedule,
        salt_planes: usize,
        optimizer: AdamW,
    ) -> Result<Self, DeviceRecoveryError> {
        let config = Self {
            schedule,
            soft_estimator: SoftEstimator::PackedSte,
            salt_planes,
            optimizer,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), DeviceRecoveryError> {
        if !(1..=3).contains(&self.salt_planes) {
            return Err(DeviceRecoveryError::InvalidInput(
                "device recovery SALT plane count must be in 1..=3".into(),
            ));
        }
        let optimizer = self.optimizer;
        if !optimizer.lr.is_finite() || optimizer.lr <= 0.0 {
            return Err(DeviceRecoveryError::InvalidInput(
                "device recovery learning rate must be finite and positive".into(),
            ));
        }
        if !optimizer.beta1.is_finite() || !(0.0..1.0).contains(&optimizer.beta1) {
            return Err(DeviceRecoveryError::InvalidInput(
                "device recovery beta1 must be finite and in [0, 1)".into(),
            ));
        }
        if !optimizer.beta2.is_finite() || !(0.0..1.0).contains(&optimizer.beta2) {
            return Err(DeviceRecoveryError::InvalidInput(
                "device recovery beta2 must be finite and in [0, 1)".into(),
            ));
        }
        if !optimizer.eps.is_finite() || optimizer.eps <= 0.0 {
            return Err(DeviceRecoveryError::InvalidInput(
                "device recovery epsilon must be finite and positive".into(),
            ));
        }
        if !optimizer.weight_decay.is_finite() || optimizer.weight_decay < 0.0 {
            return Err(DeviceRecoveryError::InvalidInput(
                "device recovery weight decay must be finite and nonnegative".into(),
            ));
        }
        Ok(())
    }
}

/// Deterministic evidence for one completed optimizer step.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceRecoveryStepReceipt {
    zero_based_step: u64,
    optimizer_step: u64,
    phase: DeviceRecoveryPhase,
    estimator: DeviceRecoveryEstimator,
    temperature_digest: Option<[u8; 32]>,
    materialized_gradient_elements: usize,
    peak_live_gradient_elements: usize,
    peak_live_activation_elements: usize,
    naive_activation_elements: usize,
}

impl DeviceRecoveryStepReceipt {
    /// Zero-based recovery-schedule step.
    #[must_use]
    pub const fn zero_based_step(&self) -> u64 {
        self.zero_based_step
    }

    /// One-based AdamW step committed into the checkpointable trainer.
    #[must_use]
    pub const fn optimizer_step(&self) -> u64 {
        self.optimizer_step
    }

    /// Soft or hard schedule phase.
    #[must_use]
    pub const fn phase(&self) -> DeviceRecoveryPhase {
        self.phase
    }

    /// Estimator used by this exact step.
    #[must_use]
    pub const fn estimator(&self) -> DeviceRecoveryEstimator {
        self.estimator
    }

    /// Digest of ordered effective f32 HESTIA temperatures, when applicable.
    #[must_use]
    pub const fn temperature_digest(&self) -> Option<[u8; 32]> {
        self.temperature_digest
    }

    /// Full requested-gradient collection size for this model.
    #[must_use]
    pub const fn materialized_gradient_elements(&self) -> usize {
        self.materialized_gradient_elements
    }

    /// Maximum simultaneously live requested-gradient elements.
    #[must_use]
    pub const fn peak_live_gradient_elements(&self) -> usize {
        self.peak_live_gradient_elements
    }

    /// Peak checkpointed activation residency observed during backward.
    #[must_use]
    pub const fn peak_live_activation_elements(&self) -> usize {
        self.peak_live_activation_elements
    }

    /// Keep-all activation baseline for the same graph.
    #[must_use]
    pub const fn naive_activation_elements(&self) -> usize {
        self.naive_activation_elements
    }
}

/// Failure from resident recovery construction, execution, or durable resume.
#[derive(Debug)]
#[non_exhaustive]
pub enum DeviceRecoveryError {
    /// Invalid model, schedule, optimizer, tensor, or lifecycle input.
    InvalidInput(String),
    /// CUDA or model-graph failure.
    Backend(String),
    /// Durable checkpoint failure.
    Checkpoint(String),
    /// Immutable plan-sidecar failure.
    Plan(String),
    /// Filesystem failure.
    Io(String),
}

impl core::fmt::Display for DeviceRecoveryError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput(reason) => {
                write!(formatter, "invalid device recovery input: {reason}")
            }
            Self::Backend(reason) => write!(formatter, "device recovery backend error: {reason}"),
            Self::Checkpoint(reason) => {
                write!(formatter, "device recovery checkpoint error: {reason}")
            }
            Self::Plan(reason) => write!(formatter, "device recovery plan error: {reason}"),
            Self::Io(reason) => write!(formatter, "device recovery I/O error: {reason}"),
        }
    }
}

impl std::error::Error for DeviceRecoveryError {}

impl From<tritium_spec::BackendError> for DeviceRecoveryError {
    fn from(error: tritium_spec::BackendError) -> Self {
        Self::Backend(error.to_string())
    }
}

impl From<TrainingAdapterError> for DeviceRecoveryError {
    fn from(error: TrainingAdapterError) -> Self {
        Self::Backend(error.to_string())
    }
}

impl From<tritium_train::DcpError> for DeviceRecoveryError {
    fn from(error: tritium_train::DcpError) -> Self {
        Self::Checkpoint(error.to_string())
    }
}

impl From<TemperatureError> for DeviceRecoveryError {
    fn from(error: TemperatureError) -> Self {
        Self::InvalidInput(error.to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TemperaturePlan {
    tau_initial_f64_bits: u64,
    tau_floor_f64_bits: u64,
    sensitivity_alpha_f64_bits: u64,
    sensitivity_f64_bits: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdamPlan {
    lr_f32_bits: u32,
    beta1_f32_bits: u32,
    beta2_f32_bits: u32,
    eps_f32_bits: u32,
    weight_decay_f32_bits: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanIdentity {
    schema: String,
    model_digest: String,
    parameter_count: usize,
    total_steps: u64,
    hard_start_step: u64,
    soft_estimator: DeviceRecoveryEstimator,
    salt_planes: usize,
    optimizer: AdamPlan,
    temperature: Option<TemperaturePlan>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanSidecar {
    plan_id: String,
    #[serde(flatten)]
    identity: PlanIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlanBindingMode {
    CreateOrVerify,
    VerifyExisting,
}

impl PlanSidecar {
    fn new(model: &TiedSwiGluTrainingModel, config: &DeviceRecoveryConfig) -> Self {
        let temperature = match &config.soft_estimator {
            SoftEstimator::Hestia(schedule) => Some(TemperaturePlan {
                tau_initial_f64_bits: schedule.tau_initial().to_bits(),
                tau_floor_f64_bits: schedule.tau_floor().to_bits(),
                sensitivity_alpha_f64_bits: schedule.sensitivity_alpha().to_bits(),
                sensitivity_f64_bits: schedule
                    .sensitivity_scores()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect(),
            }),
            SoftEstimator::PackedSte => None,
        };
        let identity = PlanIdentity {
            schema: PLAN_SCHEMA.into(),
            model_digest: recovery_model_digest(model),
            parameter_count: model.parameters().len(),
            total_steps: config.schedule.total_steps(),
            hard_start_step: config.schedule.hard_start_step(),
            soft_estimator: match config.soft_estimator {
                SoftEstimator::Hestia(_) => DeviceRecoveryEstimator::Hestia,
                SoftEstimator::PackedSte => DeviceRecoveryEstimator::PackedSte,
            },
            salt_planes: config.salt_planes,
            optimizer: AdamPlan {
                lr_f32_bits: config.optimizer.lr.to_bits(),
                beta1_f32_bits: config.optimizer.beta1.to_bits(),
                beta2_f32_bits: config.optimizer.beta2.to_bits(),
                eps_f32_bits: config.optimizer.eps.to_bits(),
                weight_decay_f32_bits: config.optimizer.weight_decay.to_bits(),
            },
            temperature,
        };
        let encoded = serde_json::to_vec(&identity)
            .expect("serializing finite recovery plan identity cannot fail");
        let plan_id = blake3::hash(&encoded).to_hex().to_string();
        Self { plan_id, identity }
    }

    fn canonical_bytes(&self) -> Result<Vec<u8>, DeviceRecoveryError> {
        let mut bytes = serde_json::to_vec(self)
            .map_err(|error| DeviceRecoveryError::Plan(error.to_string()))?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

/// Resident CUDA session whose optimizer state is exactly resumable through DCP.
pub struct DeviceRecoverySession<'backend, 'model> {
    backend: &'backend CudaBackend,
    model: &'model TiedSwiGluTrainingModel,
    config: DeviceRecoveryConfig,
    plan: PlanSidecar,
    trainer: DeviceTrainer<'backend>,
    packed: Vec<DevicePackedSaltWeight>,
    poisoned: bool,
}

impl core::fmt::Debug for DeviceRecoverySession<'_, '_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("DeviceRecoverySession")
            .field("plan_id", &self.plan.plan_id)
            .field("completed_step", &self.trainer.completed_step())
            .field("parameter_count", &self.trainer.len())
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl<'backend, 'model> DeviceRecoverySession<'backend, 'model> {
    /// Upload model masters, allocate packed SALT plus AdamW state, and freeze
    /// the exact execution plan.
    ///
    /// # Errors
    /// Rejects invalid tensor inventory, temperature coverage, or CUDA state.
    pub fn new(
        backend: &'backend CudaBackend,
        model: &'model TiedSwiGluTrainingModel,
        config: DeviceRecoveryConfig,
    ) -> Result<Self, DeviceRecoveryError> {
        config.validate()?;
        if model.parameters().is_empty() {
            return Err(DeviceRecoveryError::InvalidInput(
                "device recovery model has no trainable parameters".into(),
            ));
        }
        if let SoftEstimator::Hestia(schedule) = &config.soft_estimator
            && schedule.sensitivity_scores().len() != model.parameters().len()
        {
            return Err(DeviceRecoveryError::InvalidInput(format!(
                "HESTIA sensitivity profile has {} tensors, expected {}",
                schedule.sensitivity_scores().len(),
                model.parameters().len()
            )));
        }
        let specs: Vec<_> = model
            .parameters()
            .iter()
            .map(|parameter| DeviceTrainParam {
                master: &parameter.master,
                rows: parameter.rows,
                cols: parameter.cols,
                salt_planes: config.salt_planes,
                optimizer: config.optimizer,
            })
            .collect();
        let mut trainer = DeviceTrainer::new_with_weight_storage(
            backend,
            &specs,
            DeviceTrainerWeightStorage::Packed,
        )?;
        let packed = (0..trainer.len())
            .map(|index| {
                trainer
                    .packed_weight(index)
                    .map_err(DeviceRecoveryError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let plan = PlanSidecar::new(model, &config);
        Ok(Self {
            backend,
            model,
            config,
            plan,
            trainer,
            packed,
            poisoned: false,
        })
    }

    /// Immutable plan identity bound beside every saved DCP checkpoint.
    #[must_use]
    pub fn plan_id(&self) -> &str {
        &self.plan.plan_id
    }

    /// Completed one-based optimizer step represented by current state.
    #[must_use]
    pub fn completed_step(&self) -> u64 {
        self.trainer.completed_step()
    }

    /// Download one latent master for validation or terminal artifact export.
    pub fn download_master(&self, index: usize) -> Result<Vec<f32>, DeviceRecoveryError> {
        self.ensure_usable()?;
        self.trainer
            .download_master(index)
            .map_err(DeviceRecoveryError::from)
    }

    /// Execute one deterministic recovery step and refresh every packed handle.
    ///
    /// HESTIA requires a full requested-gradient collection because checkpoint
    /// replay still borrows every resident master. Packed STE uses bounded
    /// reverse-topological gradient streaming directly into resident AdamW.
    ///
    /// # Errors
    /// Rejects completed/poisoned sessions, invalid inputs, graph failures, or
    /// any optimizer/repack failure.
    pub fn step(
        &mut self,
        tokens: &[i32],
        target: &DeviceTensor,
    ) -> Result<DeviceRecoveryStepReceipt, DeviceRecoveryError> {
        self.ensure_usable()?;
        let zero_based_step = self.trainer.completed_step();
        let phase = self
            .config
            .schedule
            .phase_at(zero_based_step)
            .map_err(|error| DeviceRecoveryError::InvalidInput(error.to_string()))?;
        let optimizer_step = zero_based_step.checked_add(1).ok_or_else(|| {
            DeviceRecoveryError::InvalidInput("device recovery step counter overflowed".into())
        })?;
        let (estimator, temperature_digest, materialized, peak_gradient, backward) =
            match (&self.config.soft_estimator, phase) {
                (SoftEstimator::Hestia(schedule), RecoveryPhase::Soft) => {
                    let temperatures = self.temperatures(schedule, zero_based_step)?;
                    let temperature_digest = Some(hash_temperatures(&temperatures));
                    let backward =
                        self.run_hestia(tokens, target, &temperatures, optimizer_step)?;
                    let materialized = self.trainer.resident_stats().parameter_elements;
                    (
                        DeviceRecoveryEstimator::Hestia,
                        temperature_digest,
                        materialized,
                        materialized,
                        backward,
                    )
                }
                _ => {
                    let stream = self.run_packed_ste(tokens, target, optimizer_step)?;
                    (
                        DeviceRecoveryEstimator::PackedSte,
                        None,
                        stream.materialized_collection_elements,
                        stream.peak_live_requested_gradient_elements,
                        stream.backward_stats,
                    )
                }
            };
        self.repack_all()?;
        Ok(DeviceRecoveryStepReceipt {
            zero_based_step,
            optimizer_step,
            phase: phase.into(),
            estimator,
            temperature_digest,
            materialized_gradient_elements: materialized,
            peak_live_gradient_elements: peak_gradient,
            peak_live_activation_elements: backward.peak_live_activation_elements,
            naive_activation_elements: backward.naive_activation_elements,
        })
    }

    /// Save bounded-memory AdamW/master state after binding immutable plan bytes.
    ///
    /// # Errors
    /// Rejects a poisoned session, plan mismatch, filesystem failure, or DCP
    /// save failure.
    pub fn save_checkpoint(
        &mut self,
        checkpoint_dir: impl AsRef<Path>,
        shard_count: usize,
    ) -> Result<(), DeviceRecoveryError> {
        self.ensure_usable()?;
        let checkpoint_dir = checkpoint_dir.as_ref();
        self.bind_plan(checkpoint_dir, PlanBindingMode::CreateOrVerify)?;
        tritium_train::dcp::save_from(checkpoint_dir, &mut self.trainer, shard_count)?;
        Ok(())
    }

    /// Load a DCP checkpoint only after exact plan-sidecar admission, then
    /// rebuild every packed handle from restored masters.
    ///
    /// # Errors
    /// Rejects missing/mutated plans, invalid DCP state, a checkpoint beyond the
    /// schedule, or repack failure.
    pub fn resume_checkpoint(
        &mut self,
        checkpoint_dir: impl AsRef<Path>,
    ) -> Result<(), DeviceRecoveryError> {
        self.ensure_usable()?;
        let checkpoint_dir = checkpoint_dir.as_ref();
        self.bind_plan(checkpoint_dir, PlanBindingMode::VerifyExisting)?;
        tritium_train::dcp::load_into(checkpoint_dir, &mut self.trainer)?;
        if self.trainer.completed_step() > self.config.schedule.total_steps() {
            self.poisoned = true;
            return Err(DeviceRecoveryError::Checkpoint(
                "checkpoint step exceeds recovery schedule".into(),
            ));
        }
        self.repack_all()
    }

    fn ensure_usable(&self) -> Result<(), DeviceRecoveryError> {
        if self.poisoned {
            Err(DeviceRecoveryError::InvalidInput(
                "device recovery session is poisoned".into(),
            ))
        } else {
            Ok(())
        }
    }

    fn temperatures(
        &self,
        schedule: &TemperatureSchedule,
        step: u64,
    ) -> Result<Vec<f32>, DeviceRecoveryError> {
        let mut temperatures = Vec::new();
        temperatures
            .try_reserve_exact(self.model.parameters().len())
            .map_err(|_| {
                DeviceRecoveryError::InvalidInput("HESTIA temperature allocation failed".into())
            })?;
        for index in 0..self.model.parameters().len() {
            let value = schedule.tau_at(index, step)? as f32;
            if !value.is_finite() || value < tritium_train::ops::hestia::MIN_DIFFERENTIABLE_TAU {
                return Err(DeviceRecoveryError::InvalidInput(format!(
                    "HESTIA temperature for parameter {index} is not representable as differentiable f32"
                )));
            }
            temperatures.push(value);
        }
        Ok(temperatures)
    }

    fn run_hestia(
        &mut self,
        tokens: &[i32],
        target: &DeviceTensor,
        temperatures: &[f32],
        optimizer_step: u64,
    ) -> Result<DeviceBackwardStats, DeviceRecoveryError> {
        let masters = (0..self.trainer.len())
            .map(|index| {
                self.trainer
                    .master_tensor(index)
                    .map_err(DeviceRecoveryError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let arch = self.model.architecture();
        let mut tape = DeviceTape::new_with_checkpoint_policy(
            self.backend,
            arch.vocab.max(arch.n_ff).max(tokens.len()),
            CheckpointPolicy::SqrtDepth(arch.n_layers),
        )?;
        let forward = hestia_device_forward(
            &mut tape,
            self.model,
            &masters,
            &self.packed,
            temperatures,
            tokens,
        )?;
        let gradients = tape.xent_backward_device(
            forward.logits,
            target,
            tokens.len(),
            arch.vocab,
            &forward.master_leaves,
        )?;
        let backward = gradients.backward_stats();
        drop(masters);
        self.trainer.step(gradients, optimizer_step)?;
        Ok(backward)
    }

    fn run_packed_ste(
        &mut self,
        tokens: &[i32],
        target: &DeviceTensor,
        optimizer_step: u64,
    ) -> Result<tritium_cuda::train::GradientStreamReport, DeviceRecoveryError> {
        let arch = self.model.architecture();
        let mut tape = DeviceTape::new_with_checkpoint_policy(
            self.backend,
            arch.vocab.max(arch.n_ff).max(tokens.len()),
            CheckpointPolicy::SqrtDepth(arch.n_layers),
        )?;
        let forward = packed_device_forward(&mut tape, self.model, &self.packed, tokens)?;
        let bindings: Vec<_> = forward
            .master_leaves
            .iter()
            .enumerate()
            .map(|(parameter_index, &leaf_id)| GradientLeafBinding {
                leaf_id,
                parameter_index,
            })
            .collect();
        tape.xent_backward_into_resident(
            forward.logits,
            target,
            tokens.len(),
            arch.vocab,
            &bindings,
            &mut self.trainer,
            optimizer_step,
        )
        .map_err(DeviceRecoveryError::from)
    }

    fn repack_all(&mut self) -> Result<(), DeviceRecoveryError> {
        for index in 0..self.packed.len() {
            if let Err(error) = self
                .trainer
                .repack_packed_weight(index, &mut self.packed[index])
            {
                for packed in &mut self.packed {
                    packed.mark_stale();
                }
                self.poisoned = true;
                return Err(DeviceRecoveryError::from(error));
            }
        }
        Ok(())
    }

    fn bind_plan(
        &self,
        checkpoint_dir: &Path,
        mode: PlanBindingMode,
    ) -> Result<(), DeviceRecoveryError> {
        match std::fs::symlink_metadata(checkpoint_dir) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err(DeviceRecoveryError::Plan(
                    "checkpoint path must be an ordinary directory".into(),
                ));
            }
            Ok(_) => {}
            Err(error)
                if error.kind() == std::io::ErrorKind::NotFound
                    && mode == PlanBindingMode::CreateOrVerify =>
            {
                std::fs::create_dir_all(checkpoint_dir)
                    .map_err(|error| recovery_io(checkpoint_dir, error))?;
                let metadata = std::fs::symlink_metadata(checkpoint_dir)
                    .map_err(|error| recovery_io(checkpoint_dir, error))?;
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(DeviceRecoveryError::Plan(
                        "checkpoint path must be an ordinary directory".into(),
                    ));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(DeviceRecoveryError::Plan(
                    "checkpoint directory or recovery plan is missing".into(),
                ));
            }
            Err(error) => return Err(recovery_io(checkpoint_dir, error)),
        }
        let path = checkpoint_dir.join(PLAN_FILE);
        let expected = self.plan.canonical_bytes()?;
        match std::fs::symlink_metadata(&path) {
            Ok(metadata) => verify_plan_file(&path, &metadata, &expected),
            Err(error)
                if error.kind() == std::io::ErrorKind::NotFound
                    && mode == PlanBindingMode::CreateOrVerify =>
            {
                persist_plan_no_clobber(&path, &expected)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(DeviceRecoveryError::Plan("recovery plan is missing".into()))
            }
            Err(error) => Err(recovery_io(&path, error)),
        }
    }
}

pub(crate) fn recovery_model_digest(model: &TiedSwiGluTrainingModel) -> String {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium-device-recovery-model-v1");
    let arch = model.architecture();
    for value in [
        arch.n_layers,
        arch.n_embd,
        arch.n_head,
        arch.n_head_kv,
        arch.head_dim,
        arch.n_ff,
        arch.n_ctx,
        arch.vocab,
    ] {
        hash.update(&(value as u64).to_le_bytes());
    }
    hash.update(&arch.rope_theta.to_bits().to_le_bytes());
    hash.update(&arch.rms_eps.to_bits().to_le_bytes());
    hash.update(&[u8::from(model.is_lm_head_tied())]);
    for parameter in model.parameters() {
        hash_len_bytes(&mut hash, parameter.name.as_bytes());
        hash.update(&(parameter.rows as u64).to_le_bytes());
        hash.update(&(parameter.cols as u64).to_le_bytes());
        hash_f32s(&mut hash, &parameter.master);
    }
    for norms in [&arch.attn_norms, &arch.ffn_norms] {
        for norm in norms {
            hash_f32s(&mut hash, norm);
        }
    }
    for constants in &arch.attention_constants {
        for values in [
            &constants.q_bias,
            &constants.k_bias,
            &constants.v_bias,
            &constants.q_norm,
            &constants.k_norm,
        ] {
            hash_f32s(&mut hash, values);
        }
    }
    hash_f32s(&mut hash, &arch.output_norm);
    hash.finalize().to_hex().to_string()
}

fn hash_len_bytes(hash: &mut blake3::Hasher, bytes: &[u8]) {
    hash.update(&(bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn hash_f32s(hash: &mut blake3::Hasher, values: &[f32]) {
    hash.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(&value.to_bits().to_le_bytes());
    }
}

fn hash_temperatures(temperatures: &[f32]) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium-device-recovery-temperatures-v1");
    hash_f32s(&mut hash, temperatures);
    *hash.finalize().as_bytes()
}

fn recovery_io(path: &Path, error: std::io::Error) -> DeviceRecoveryError {
    DeviceRecoveryError::Io(format!("{}: {error}", path.display()))
}

fn verify_plan_file(
    path: &Path,
    metadata: &std::fs::Metadata,
    expected: &[u8],
) -> Result<(), DeviceRecoveryError> {
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > MAX_PLAN_BYTES {
        return Err(DeviceRecoveryError::Plan(
            "recovery plan must be a bounded ordinary file".into(),
        ));
    }
    let observed = std::fs::read(path).map_err(|error| recovery_io(path, error))?;
    if observed != expected {
        return Err(DeviceRecoveryError::Plan(
            "recovery plan differs from current model or configuration".into(),
        ));
    }
    Ok(())
}

fn persist_plan_no_clobber(path: &Path, bytes: &[u8]) -> Result<(), DeviceRecoveryError> {
    let parent = path
        .parent()
        .ok_or_else(|| DeviceRecoveryError::Io("recovery plan path has no parent".into()))?;
    let sequence = PLAN_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{PLAN_FILE}.{}.{}.tmp",
        std::process::id(),
        sequence
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| recovery_io(&temporary, error))?;
        file.write_all(bytes)
            .map_err(|error| recovery_io(&temporary, error))?;
        file.sync_all()
            .map_err(|error| recovery_io(&temporary, error))?;
        drop(file);
        match std::fs::hard_link(&temporary, path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let metadata =
                    std::fs::symlink_metadata(path).map_err(|error| recovery_io(path, error))?;
                verify_plan_file(path, &metadata, bytes)?;
            }
            Err(error) => return Err(recovery_io(path, error)),
        }
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| recovery_io(parent, error))?;
        std::fs::remove_file(&temporary).map_err(|error| recovery_io(&temporary, error))?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| recovery_io(parent, error))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

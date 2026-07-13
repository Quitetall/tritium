//! Process-per-GPU campaign world construction.
//!
//! This module owns only distributed process topology and rendezvous. Campaign
//! policy remains in `campaign`: workers receive immutable rank/device/NCCL
//! coordinates through inherited environment variables, and exactly one rank
//! owns filesystem lock/report/checkpoint publication.

use std::collections::HashSet;
use std::env;
use std::ffi::OsString;
use std::fmt;
use std::io;
use std::num::NonZeroUsize;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tritium_cuda::{CudaBackend, NcclId, NcclProcessGroup};
use tritium_train::dist::DistError;

const PROTOCOL_VERSION: &str = "1";
const ENV_PROTOCOL: &str = "TRITIUM_CAMPAIGN_WORLD_PROTOCOL";
const ENV_JOB_NONCE: &str = "TRITIUM_CAMPAIGN_WORLD_JOB_NONCE";
const ENV_RANK: &str = "TRITIUM_CAMPAIGN_WORLD_RANK";
const ENV_WORLD_SIZE: &str = "TRITIUM_CAMPAIGN_WORLD_SIZE";
const ENV_DEVICE: &str = "TRITIUM_CAMPAIGN_WORLD_DEVICE";
const ENV_LOCK_OWNER: &str = "TRITIUM_CAMPAIGN_WORLD_LOCK_OWNER";
const ENV_NCCL_ID: &str = "TRITIUM_CAMPAIGN_WORLD_NCCL_ID";
const WORKER_ENV: [&str; 7] = [
    ENV_PROTOCOL,
    ENV_JOB_NONCE,
    ENV_RANK,
    ENV_WORLD_SIZE,
    ENV_DEVICE,
    ENV_LOCK_OWNER,
    ENV_NCCL_ID,
];

#[derive(Debug)]
pub(crate) enum WorldError {
    InvalidConfig(String),
    InvalidEnvironment(String),
    Io {
        action: String,
        source: io::Error,
    },
    Cleanup {
        trigger: String,
        rank: usize,
        source: io::Error,
    },
    Nccl(DistError),
    NestedSupervisor,
    WorkerFailed {
        rank: usize,
        status: ExitStatus,
    },
    TimedOut {
        timeout: Duration,
    },
}

impl fmt::Display for WorldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid distributed campaign: {message}"),
            Self::InvalidEnvironment(message) => {
                write!(f, "invalid distributed worker environment: {message}")
            }
            Self::Io { action, source } => write!(f, "{action}: {source}"),
            Self::Cleanup {
                trigger,
                rank,
                source,
            } => write!(
                f,
                "{trigger}; additionally failed to reap worker rank {rank}: {source}"
            ),
            Self::Nccl(error) => write!(f, "NCCL rendezvous: {error}"),
            Self::NestedSupervisor => {
                write!(f, "a distributed worker cannot supervise another world")
            }
            Self::WorkerFailed { rank, status } => {
                write!(f, "distributed worker rank {rank} exited with {status}")
            }
            Self::TimedOut { timeout } => {
                write!(f, "distributed campaign exceeded timeout {timeout:?}")
            }
        }
    }
}

impl std::error::Error for WorldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } | Self::Cleanup { source, .. } => Some(source),
            Self::Nccl(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Rank(usize);

impl Rank {
    pub(crate) fn get(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkerRole {
    LockOwner,
    Peer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WorkerSlot {
    rank: Rank,
    device_ordinal: usize,
    role: WorkerRole,
}

impl WorkerSlot {
    pub(crate) fn rank(self) -> Rank {
        self.rank
    }

    pub(crate) fn device_ordinal(self) -> usize {
        self.device_ordinal
    }

    pub(crate) fn role(self) -> WorkerRole {
        self.role
    }

    pub(crate) fn owns_campaign_lock(self) -> bool {
        self.role == WorkerRole::LockOwner
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeviceFleet {
    devices: Vec<usize>,
    lock_owner: Rank,
}

impl DeviceFleet {
    pub(crate) fn new(devices: Vec<usize>, lock_owner_rank: usize) -> Result<Self, WorldError> {
        if devices.is_empty() {
            return Err(WorldError::InvalidConfig(
                "device fleet must contain at least one CUDA ordinal".into(),
            ));
        }
        if devices.len() > i32::MAX as usize {
            return Err(WorldError::InvalidConfig(
                "device fleet exceeds NCCL's signed rank count".into(),
            ));
        }
        let unique = devices.iter().copied().collect::<HashSet<_>>();
        if unique.len() != devices.len() {
            return Err(WorldError::InvalidConfig(
                "device ordinals must be unique (one process per GPU)".into(),
            ));
        }
        if lock_owner_rank >= devices.len() {
            return Err(WorldError::InvalidConfig(format!(
                "lock-owner rank {lock_owner_rank} is outside world size {}",
                devices.len()
            )));
        }
        Ok(Self {
            devices,
            lock_owner: Rank(lock_owner_rank),
        })
    }

    pub(crate) fn world_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.devices.len()).expect("validated non-empty fleet")
    }

    pub(crate) fn lock_owner(&self) -> Rank {
        self.lock_owner
    }

    pub(crate) fn slot(&self, rank: usize) -> Result<WorkerSlot, WorldError> {
        let device_ordinal = self.devices.get(rank).copied().ok_or_else(|| {
            WorldError::InvalidConfig(format!(
                "rank {rank} is outside world size {}",
                self.devices.len()
            ))
        })?;
        let rank = Rank(rank);
        Ok(WorkerSlot {
            rank,
            device_ordinal,
            role: if rank == self.lock_owner {
                WorkerRole::LockOwner
            } else {
                WorkerRole::Peer
            },
        })
    }

    fn slots(&self) -> impl Iterator<Item = WorkerSlot> + '_ {
        self.devices
            .iter()
            .enumerate()
            .map(|(rank, &device_ordinal)| {
                let rank = Rank(rank);
                WorkerSlot {
                    rank,
                    device_ordinal,
                    role: if rank == self.lock_owner {
                        WorkerRole::LockOwner
                    } else {
                        WorkerRole::Peer
                    },
                }
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DistributedConfig {
    fleet: DeviceFleet,
    timeout: Duration,
}

impl DistributedConfig {
    pub(crate) fn new(fleet: DeviceFleet, timeout: Duration) -> Result<Self, WorldError> {
        if timeout.is_zero() {
            return Err(WorldError::InvalidConfig(
                "worker timeout must be greater than zero".into(),
            ));
        }
        Ok(Self { fleet, timeout })
    }

    pub(crate) fn fleet(&self) -> &DeviceFleet {
        &self.fleet
    }

    pub(crate) fn timeout(&self) -> Duration {
        self.timeout
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowPartition {
    windows: NonZeroUsize,
    world_size: NonZeroUsize,
}

impl WindowPartition {
    pub(crate) fn new(windows: usize, world_size: usize) -> Result<Self, WorldError> {
        let windows = NonZeroUsize::new(windows).ok_or_else(|| {
            WorldError::InvalidConfig("training corpus has no complete windows".into())
        })?;
        let world_size = NonZeroUsize::new(world_size)
            .ok_or_else(|| WorldError::InvalidConfig("world size must be non-zero".into()))?;
        if windows < world_size {
            return Err(WorldError::InvalidConfig(format!(
                "{} corpus windows cannot supply {} distinct rank inputs per step",
                windows.get(),
                world_size.get()
            )));
        }
        Ok(Self {
            windows,
            world_size,
        })
    }

    pub(crate) fn window_for_step(
        &self,
        rank: Rank,
        optimizer_step: u64,
    ) -> Result<usize, WorldError> {
        if rank.get() >= self.world_size.get() {
            return Err(WorldError::InvalidConfig(format!(
                "rank {} is outside partition world size {}",
                rank.get(),
                self.world_size.get()
            )));
        }
        let zero_step = optimizer_step
            .checked_sub(1)
            .ok_or_else(|| WorldError::InvalidConfig("optimizer steps are one-indexed".into()))?;
        let global = u128::from(zero_step) * self.world_size.get() as u128 + rank.get() as u128;
        Ok((global % self.windows.get() as u128) as usize)
    }

    #[cfg(test)]
    pub(crate) fn windows(self) -> NonZeroUsize {
        self.windows
    }

    #[cfg(test)]
    pub(crate) fn world_size(self) -> NonZeroUsize {
        self.world_size
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct JobNonce(String);

impl JobNonce {
    fn new() -> Result<Self, WorldError> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let elapsed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| {
                WorldError::InvalidConfig(format!("system clock precedes Unix epoch: {error}"))
            })?;
        let counter = NEXT.fetch_add(1, Ordering::Relaxed);
        Self::parse(format!(
            "{:x}-{:x}-{:x}",
            std::process::id(),
            elapsed.as_nanos(),
            counter
        ))
    }

    fn parse(value: String) -> Result<Self, WorldError> {
        if value.is_empty() || value.len() > 96 {
            return Err(WorldError::InvalidEnvironment(
                "job nonce must contain 1..=96 characters".into(),
            ));
        }
        if !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() || byte == b'-')
        {
            return Err(WorldError::InvalidEnvironment(
                "job nonce must contain only hexadecimal digits and '-'".into(),
            ));
        }
        Ok(Self(value))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, PartialEq, Eq)]
struct NcclWireId([u8; 128]);

impl fmt::Debug for NcclWireId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NcclWireId").finish_non_exhaustive()
    }
}

impl NcclWireId {
    fn from_nccl(id: &NcclId) -> Self {
        Self(id.bytes().map(|byte| byte as u8))
    }

    fn to_nccl(&self) -> NcclId {
        NcclId::from_bytes(self.0.map(|byte| byte as core::ffi::c_char))
    }

    fn encode(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = String::with_capacity(self.0.len() * 2);
        for &byte in &self.0 {
            encoded.push(HEX[usize::from(byte >> 4)] as char);
            encoded.push(HEX[usize::from(byte & 0x0f)] as char);
        }
        encoded
    }

    fn decode(encoded: &str) -> Result<Self, WorldError> {
        if encoded.len() != 256 {
            return Err(WorldError::InvalidEnvironment(format!(
                "NCCL id must be 256 hexadecimal characters, got {}",
                encoded.len()
            )));
        }
        let bytes = encoded.as_bytes();
        let mut decoded = [0u8; 128];
        for (index, byte) in decoded.iter_mut().enumerate() {
            let offset = index * 2;
            let high = decode_hex(bytes[offset]).ok_or_else(|| {
                WorldError::InvalidEnvironment(format!(
                    "NCCL id contains non-hexadecimal byte at offset {offset}"
                ))
            })?;
            let low = decode_hex(bytes[offset + 1]).ok_or_else(|| {
                WorldError::InvalidEnvironment(format!(
                    "NCCL id contains non-hexadecimal byte at offset {}",
                    offset + 1
                ))
            })?;
            *byte = (high << 4) | low;
        }
        Ok(Self(decoded))
    }
}

fn decode_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Immutable coordinates inherited by one campaign worker process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WorkerRendezvous {
    job_nonce: JobNonce,
    slot: WorkerSlot,
    world_size: NonZeroUsize,
    lock_owner: Rank,
    nccl_id: NcclWireId,
}

impl WorkerRendezvous {
    fn new(
        job_nonce: JobNonce,
        slot: WorkerSlot,
        world_size: NonZeroUsize,
        lock_owner: Rank,
        nccl_id: NcclWireId,
    ) -> Result<Self, WorldError> {
        if slot.rank().get() >= world_size.get() {
            return Err(WorldError::InvalidEnvironment(format!(
                "worker rank {} is outside world size {}",
                slot.rank().get(),
                world_size.get()
            )));
        }
        if lock_owner.get() >= world_size.get() {
            return Err(WorldError::InvalidEnvironment(format!(
                "lock-owner rank {} is outside world size {}",
                lock_owner.get(),
                world_size.get()
            )));
        }
        let expected_role = if slot.rank() == lock_owner {
            WorkerRole::LockOwner
        } else {
            WorkerRole::Peer
        };
        if slot.role() != expected_role {
            return Err(WorldError::InvalidEnvironment(
                "worker role disagrees with lock-owner rank".into(),
            ));
        }
        Ok(Self {
            job_nonce,
            slot,
            world_size,
            lock_owner,
            nccl_id,
        })
    }

    /// Decode the all-or-none inherited worker environment.
    pub(crate) fn from_env() -> Result<Option<Self>, WorldError> {
        Self::decode_from(|name| env::var_os(name))
    }

    fn decode_from(read: impl FnMut(&str) -> Option<OsString>) -> Result<Option<Self>, WorldError> {
        let values = WORKER_ENV.map(read);
        let present = values.iter().filter(|value| value.is_some()).count();
        if present == 0 {
            return Ok(None);
        }
        if present != WORKER_ENV.len() {
            let missing = WORKER_ENV
                .iter()
                .zip(&values)
                .filter_map(|(name, value)| value.is_none().then_some(*name))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(WorldError::InvalidEnvironment(format!(
                "partial worker rendezvous; missing {missing}"
            )));
        }
        let value = |index: usize| -> Result<&str, WorldError> {
            values[index]
                .as_ref()
                .and_then(|raw| raw.to_str())
                .ok_or_else(|| {
                    WorldError::InvalidEnvironment(format!(
                        "{} is not valid UTF-8",
                        WORKER_ENV[index]
                    ))
                })
        };
        if value(0)? != PROTOCOL_VERSION {
            return Err(WorldError::InvalidEnvironment(format!(
                "unsupported worker protocol {}",
                value(0)?
            )));
        }
        let job_nonce = JobNonce::parse(value(1)?.to_owned())?;
        let rank = parse_usize(ENV_RANK, value(2)?)?;
        let world_size = NonZeroUsize::new(parse_usize(ENV_WORLD_SIZE, value(3)?)?)
            .ok_or_else(|| WorldError::InvalidEnvironment("world size must be non-zero".into()))?;
        let device_ordinal = parse_usize(ENV_DEVICE, value(4)?)?;
        let lock_owner = Rank(parse_usize(ENV_LOCK_OWNER, value(5)?)?);
        let role = if Rank(rank) == lock_owner {
            WorkerRole::LockOwner
        } else {
            WorkerRole::Peer
        };
        Self::new(
            job_nonce,
            WorkerSlot {
                rank: Rank(rank),
                device_ordinal,
                role,
            },
            world_size,
            lock_owner,
            NcclWireId::decode(value(6)?)?,
        )
        .map(Some)
    }

    fn apply_to(&self, command: &mut Command) {
        for (name, value) in self.encoded_environment() {
            command.env(name, value);
        }
    }

    fn encoded_environment(&self) -> [(&'static str, OsString); 7] {
        [
            (ENV_PROTOCOL, OsString::from(PROTOCOL_VERSION)),
            (ENV_JOB_NONCE, OsString::from(self.job_nonce.as_str())),
            (ENV_RANK, OsString::from(self.slot.rank().get().to_string())),
            (
                ENV_WORLD_SIZE,
                OsString::from(self.world_size.get().to_string()),
            ),
            (
                ENV_DEVICE,
                OsString::from(self.slot.device_ordinal().to_string()),
            ),
            (
                ENV_LOCK_OWNER,
                OsString::from(self.lock_owner.get().to_string()),
            ),
            (ENV_NCCL_ID, OsString::from(self.nccl_id.encode())),
        ]
    }

    #[cfg(test)]
    pub(crate) fn job_nonce(&self) -> &JobNonce {
        &self.job_nonce
    }

    pub(crate) fn slot(&self) -> WorkerSlot {
        self.slot
    }

    pub(crate) fn world_size(&self) -> NonZeroUsize {
        self.world_size
    }

    pub(crate) fn lock_owner(&self) -> Rank {
        self.lock_owner
    }

    pub(crate) fn owns_campaign_lock(&self) -> bool {
        self.slot.owns_campaign_lock()
    }

    pub(crate) fn join(&self, backend: &CudaBackend) -> Result<NcclProcessGroup, WorldError> {
        NcclProcessGroup::init_on_backend(
            backend,
            self.slot.rank().get(),
            self.world_size.get(),
            &self.nccl_id.to_nccl(),
        )
        .map_err(WorldError::Nccl)
    }
}

fn parse_usize(name: &str, value: &str) -> Result<usize, WorldError> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(WorldError::InvalidEnvironment(format!(
            "{name} must be an unsigned decimal integer"
        )));
    }
    value.parse::<usize>().map_err(|error| {
        WorldError::InvalidEnvironment(format!("{name} is outside usize: {error}"))
    })
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct WorkerExit {
    rank: Rank,
    status: ExitStatus,
}

impl WorkerExit {
    pub(crate) fn rank(self) -> Rank {
        self.rank
    }

    pub(crate) fn status(self) -> ExitStatus {
        self.status
    }
}

#[derive(Debug)]
pub(crate) struct SupervisorReport {
    job_nonce: JobNonce,
    elapsed: Duration,
    workers: Vec<WorkerExit>,
}

impl SupervisorReport {
    pub(crate) fn job_nonce(&self) -> &JobNonce {
        &self.job_nonce
    }

    pub(crate) fn elapsed(&self) -> Duration {
        self.elapsed
    }

    pub(crate) fn workers(&self) -> &[WorkerExit] {
        &self.workers
    }
}

struct LiveWorker {
    rank: Rank,
    child: std::process::Child,
    status: Option<ExitStatus>,
}

impl LiveWorker {
    fn terminate_and_reap(&mut self) -> Result<(), io::Error> {
        if self.status.is_some() {
            return Ok(());
        }
        if let Ok(Some(status)) = self.child.try_wait() {
            self.status = Some(status);
            return Ok(());
        }
        let _ = self.child.kill();
        let status = self.child.wait()?;
        self.status = Some(status);
        Ok(())
    }
}

struct LiveWorld(Vec<LiveWorker>);

impl LiveWorld {
    fn terminate_and_reap_all(&mut self) -> Result<(), (Rank, io::Error)> {
        let mut first_error = None;
        for worker in &mut self.0 {
            if let Err(error) = worker.terminate_and_reap()
                && first_error.is_none()
            {
                first_error = Some((worker.rank, error));
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn abort<T>(&mut self, trigger: WorldError) -> Result<T, WorldError> {
        match self.terminate_and_reap_all() {
            Ok(()) => Err(trigger),
            Err((rank, source)) => Err(WorldError::Cleanup {
                trigger: trigger.to_string(),
                rank: rank.get(),
                source,
            }),
        }
    }
}

impl Drop for LiveWorld {
    fn drop(&mut self) {
        let _ = self.terminate_and_reap_all();
    }
}

#[cfg(target_os = "linux")]
fn configure_worker_parent_death(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    let supervisor_pid = std::process::id() as libc::pid_t;
    // SAFETY: this closure runs after fork and before exec. It calls only the
    // async-signal-safe prctl/getppid syscalls and constructs errors from integer
    // errno values. The parent check closes the fork/prctl race: if the supervisor
    // died before PR_SET_PDEATHSIG was armed, the worker refuses to exec; if it dies
    // after arming, the kernel delivers SIGKILL.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                return Err(io::Error::last_os_error());
            }
            if libc::getppid() != supervisor_pid {
                return Err(io::Error::from_raw_os_error(libc::ECHILD));
            }
            Ok(())
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn configure_worker_parent_death(_command: &mut Command) {
    // PR_SET_PDEATHSIG is Linux-only. Other targets retain the explicit
    // terminate-and-reap error/Drop paths above but do not claim hard
    // containment if the supervisor itself is killed.
}

/// Spawn one copy of the current executable per fleet slot and supervise it.
///
/// `worker_args` are arguments after argv[0], normally
/// `std::env::args_os().skip(1)`. A worker must call [`WorkerRendezvous::from_env`]
/// before attempting to supervise so it follows the worker path instead of
/// recursively creating another world.
///
/// On spawn error, non-zero worker exit, polling error, or timeout, every live
/// peer is killed and reaped before this function returns.
pub(crate) fn supervise_current_exe(
    config: &DistributedConfig,
    worker_args: &[OsString],
) -> Result<SupervisorReport, WorldError> {
    if WorkerRendezvous::from_env()?.is_some() {
        return Err(WorldError::NestedSupervisor);
    }
    let executable = env::current_exe().map_err(|source| WorldError::Io {
        action: "resolve current executable".into(),
        source,
    })?;
    let id = NcclId::new().map_err(WorldError::Nccl)?;
    supervise_executable(config, worker_args, &executable, NcclWireId::from_nccl(&id))
}

fn supervise_executable(
    config: &DistributedConfig,
    worker_args: &[OsString],
    executable: &std::path::Path,
    nccl_id: NcclWireId,
) -> Result<SupervisorReport, WorldError> {
    let job_nonce = JobNonce::new()?;
    let mut world = LiveWorld(Vec::with_capacity(config.fleet().world_size().get()));
    for slot in config.fleet().slots() {
        let rendezvous = WorkerRendezvous::new(
            job_nonce.clone(),
            slot,
            config.fleet().world_size(),
            config.fleet().lock_owner(),
            nccl_id.clone(),
        )?;
        let mut command = Command::new(executable);
        command
            .args(worker_args)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        for name in WORKER_ENV {
            command.env_remove(name);
        }
        rendezvous.apply_to(&mut command);
        configure_worker_parent_death(&mut command);
        let child = match command.spawn() {
            Ok(child) => child,
            Err(source) => {
                let error = WorldError::Io {
                    action: format!("spawn distributed worker rank {}", slot.rank().get()),
                    source,
                };
                return world.abort(error);
            }
        };
        world.0.push(LiveWorker {
            rank: slot.rank(),
            child,
            status: None,
        });
    }

    let started = Instant::now();
    loop {
        let mut all_finished = true;
        let mut failure = None;
        let mut poll_error = None;
        for worker in &mut world.0 {
            if worker.status.is_some() {
                continue;
            }
            match worker.child.try_wait() {
                Ok(Some(status)) => {
                    worker.status = Some(status);
                    if !status.success() && failure.is_none() {
                        failure = Some((worker.rank, status));
                    }
                }
                Ok(None) => all_finished = false,
                Err(source) => {
                    poll_error = Some((worker.rank, source));
                    break;
                }
            }
        }
        if let Some((rank, status)) = failure {
            return world.abort(WorldError::WorkerFailed {
                rank: rank.get(),
                status,
            });
        }
        if let Some((rank, source)) = poll_error {
            return world.abort(WorldError::Io {
                action: format!("poll distributed worker rank {}", rank.get()),
                source,
            });
        }
        if all_finished {
            let workers = world
                .0
                .iter()
                .map(|worker| {
                    worker
                        .status
                        .map(|status| WorkerExit {
                            rank: worker.rank,
                            status,
                        })
                        .ok_or_else(|| {
                            WorldError::InvalidConfig(
                                "supervisor marked an unfinished worker complete".into(),
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(SupervisorReport {
                job_nonce,
                elapsed: started.elapsed(),
                workers,
            });
        }
        if started.elapsed() >= config.timeout() {
            return world.abort(WorldError::TimedOut {
                timeout: config.timeout(),
            });
        }
        let remaining = config.timeout().saturating_sub(started.elapsed());
        std::thread::sleep(remaining.min(Duration::from_millis(10)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[cfg(target_os = "linux")]
    const PARENT_DEATH_PID_FILE: &str = "TRITIUM_PARENT_DEATH_PID_FILE";

    #[cfg(target_os = "linux")]
    struct ChildKillGuard(std::process::Child);

    #[cfg(target_os = "linux")]
    impl Drop for ChildKillGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    #[cfg(target_os = "linux")]
    struct PidKillGuard(Option<u32>);

    #[cfg(target_os = "linux")]
    impl Drop for PidKillGuard {
        fn drop(&mut self) {
            if let Some(pid) = self.0 {
                let _ = Command::new("kill")
                    .args([OsString::from("-KILL"), OsString::from(pid.to_string())])
                    .status();
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn linux_process_is_alive(pid: u32) -> bool {
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
            return false;
        };
        stat.rsplit_once(") ")
            .and_then(|(_, fields)| fields.as_bytes().first())
            .is_some_and(|&state| state != b'Z')
    }

    #[test]
    fn fleet_rejects_empty_duplicate_and_out_of_range_owner() {
        assert!(DeviceFleet::new(vec![], 0).is_err());
        assert!(DeviceFleet::new(vec![2, 2], 0).is_err());
        assert!(DeviceFleet::new(vec![2, 5], 2).is_err());
    }

    #[test]
    fn fleet_assigns_one_explicit_lock_owner() {
        let fleet = DeviceFleet::new(vec![5, 2, 9], 1).expect("fleet");
        assert_eq!(fleet.world_size().get(), 3);
        assert_eq!(fleet.lock_owner().get(), 1);
        assert_eq!(fleet.slot(0).expect("rank 0").device_ordinal(), 5);
        assert_eq!(fleet.slot(1).expect("rank 1").role(), WorkerRole::LockOwner);
        assert!(fleet.slot(1).expect("rank 1").owns_campaign_lock());
        assert_eq!(fleet.slot(2).expect("rank 2").role(), WorkerRole::Peer);
        assert!(fleet.slot(3).is_err());
    }

    #[test]
    fn partition_assigns_a_disjoint_world_sized_window_group_per_step() {
        let partition = WindowPartition::new(10, 3).expect("partition");
        let step1 = [
            partition.window_for_step(Rank(0), 1).unwrap(),
            partition.window_for_step(Rank(1), 1).unwrap(),
            partition.window_for_step(Rank(2), 1).unwrap(),
        ];
        let step2 = [
            partition.window_for_step(Rank(0), 2).unwrap(),
            partition.window_for_step(Rank(1), 2).unwrap(),
            partition.window_for_step(Rank(2), 2).unwrap(),
        ];
        let step4 = [
            partition.window_for_step(Rank(0), 4).unwrap(),
            partition.window_for_step(Rank(1), 4).unwrap(),
            partition.window_for_step(Rank(2), 4).unwrap(),
        ];
        assert_eq!(step1, [0, 1, 2]);
        assert_eq!(step2, [3, 4, 5]);
        assert_eq!(step4, [9, 0, 1]);
        assert_eq!(partition.windows().get(), 10);
        assert_eq!(partition.world_size().get(), 3);
        assert!(partition.window_for_step(Rank(3), 1).is_err());
        assert!(partition.window_for_step(Rank(0), 0).is_err());
    }

    #[test]
    fn config_rejects_zero_timeout() {
        let fleet = DeviceFleet::new(vec![0], 0).expect("fleet");
        assert!(DistributedConfig::new(fleet.clone(), Duration::ZERO).is_err());
        let config = DistributedConfig::new(fleet, Duration::from_secs(7)).expect("config");
        assert_eq!(config.timeout(), Duration::from_secs(7));
        assert_eq!(config.fleet().world_size().get(), 1);
    }

    #[test]
    fn worker_rendezvous_environment_round_trips_all_coordinates() {
        let fleet = DeviceFleet::new(vec![7, 3], 1).expect("fleet");
        let expected = WorkerRendezvous::new(
            JobNonce::parse("abc-123".into()).expect("nonce"),
            fleet.slot(1).expect("slot"),
            fleet.world_size(),
            fleet.lock_owner(),
            NcclWireId(std::array::from_fn(|index| index as u8)),
        )
        .expect("rendezvous");
        let encoded = expected
            .encoded_environment()
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect::<HashMap<_, _>>();

        let decoded = WorkerRendezvous::decode_from(|key| encoded.get(key).cloned())
            .expect("decode")
            .expect("worker");

        assert_eq!(decoded, expected);
        assert_eq!(decoded.job_nonce().as_str(), "abc-123");
        assert_eq!(decoded.slot().rank().get(), 1);
        assert_eq!(decoded.slot().device_ordinal(), 3);
        assert_eq!(decoded.world_size().get(), 2);
        assert_eq!(decoded.lock_owner().get(), 1);
        assert!(decoded.owns_campaign_lock());
        assert_eq!(decoded.nccl_id.encode().len(), 256);
        assert_eq!(
            NcclWireId::decode(&decoded.nccl_id.encode()).unwrap(),
            decoded.nccl_id
        );
    }

    #[test]
    fn worker_rendezvous_is_absent_or_complete_and_valid() {
        assert!(WorkerRendezvous::decode_from(|_| None).unwrap().is_none());

        let mut partial = HashMap::new();
        partial.insert(ENV_PROTOCOL, OsString::from(PROTOCOL_VERSION));
        let error = WorkerRendezvous::decode_from(|key| partial.get(key).cloned()).unwrap_err();
        assert!(error.to_string().contains("partial worker rendezvous"));

        let wire = NcclWireId([0x5a; 128]).encode();
        let values = [
            (ENV_PROTOCOL, PROTOCOL_VERSION.to_owned()),
            (ENV_JOB_NONCE, "beef-7".to_owned()),
            (ENV_RANK, "2".to_owned()),
            (ENV_WORLD_SIZE, "2".to_owned()),
            (ENV_DEVICE, "0".to_owned()),
            (ENV_LOCK_OWNER, "0".to_owned()),
            (ENV_NCCL_ID, wire),
        ]
        .into_iter()
        .map(|(key, value)| (key, OsString::from(value)))
        .collect::<HashMap<_, _>>();
        let error = WorkerRendezvous::decode_from(|key| values.get(key).cloned()).unwrap_err();
        assert!(error.to_string().contains("outside world size"));
    }

    fn probe_args(name: &str) -> Vec<OsString> {
        vec![
            OsString::from(name),
            OsString::from("--ignored"),
            OsString::from("--test-threads=1"),
        ]
    }

    fn supervise_probe(name: &str, timeout: Duration) -> Result<SupervisorReport, WorldError> {
        let config =
            DistributedConfig::new(DeviceFleet::new(vec![19, 23], 0).expect("fleet"), timeout)
                .expect("config");
        let executable = env::current_exe().expect("current test executable");
        supervise_executable(
            &config,
            &probe_args(name),
            &executable,
            NcclWireId([0x3c; 128]),
        )
    }

    #[test]
    fn supervisor_returns_only_after_every_worker_succeeds() {
        let report = supervise_probe("supervisor_success_probe", Duration::from_secs(5))
            .expect("successful world");
        assert_eq!(report.workers().len(), 2);
        assert!(
            report
                .workers()
                .iter()
                .all(|worker| worker.status().success())
        );
        assert_eq!(report.workers()[0].rank().get(), 0);
        assert_eq!(report.workers()[1].rank().get(), 1);
        assert!(!report.job_nonce().as_str().is_empty());
        assert!(report.elapsed() < Duration::from_secs(5));

        let _supervisor_api: fn(
            &DistributedConfig,
            &[OsString],
        ) -> Result<SupervisorReport, WorldError> = supervise_current_exe;
        let _join_api: fn(&WorkerRendezvous, &CudaBackend) -> Result<NcclProcessGroup, WorldError> =
            WorkerRendezvous::join;
    }

    #[test]
    fn supervisor_kills_and_reaps_peers_after_worker_failure() {
        let started = Instant::now();
        let error = supervise_probe("supervisor_failure_probe", Duration::from_secs(20))
            .expect_err("rank zero must fail");
        assert!(matches!(error, WorldError::WorkerFailed { rank: 0, .. }));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn supervisor_kills_and_reaps_all_workers_after_timeout() {
        let started = Instant::now();
        let timeout = Duration::from_millis(100);
        let error = supervise_probe("supervisor_timeout_probe", timeout)
            .expect_err("workers must time out");
        assert!(matches!(error, WorldError::TimedOut { timeout: got } if got == timeout));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn worker_cannot_survive_supervisor_death() {
        let pid_file = env::temp_dir().join(format!(
            "tritium-parent-death-{}-{}.pid",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time after Unix epoch")
                .as_nanos()
        ));
        let executable = env::current_exe().expect("current test executable");
        let child = Command::new(executable)
            .args(probe_args("supervisor_parent_death_probe"))
            .env(PARENT_DEATH_PID_FILE, &pid_file)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn supervisor probe");
        let mut supervisor = ChildKillGuard(child);
        let deadline = Instant::now() + Duration::from_secs(5);
        let worker_pid = loop {
            if let Ok(contents) = std::fs::read_to_string(&pid_file) {
                break contents.trim().parse::<u32>().expect("worker pid");
            }
            assert!(
                supervisor.0.try_wait().expect("poll supervisor").is_none(),
                "supervisor probe exited before publishing its worker pid"
            );
            assert!(
                Instant::now() < deadline,
                "worker did not publish its pid before the deadline"
            );
            std::thread::sleep(Duration::from_millis(10));
        };
        let mut worker = PidKillGuard(Some(worker_pid));
        assert!(linux_process_is_alive(worker_pid));

        supervisor.0.kill().expect("kill supervisor probe");
        supervisor.0.wait().expect("reap supervisor probe");
        let deadline = Instant::now() + Duration::from_secs(5);
        while linux_process_is_alive(worker_pid) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !linux_process_is_alive(worker_pid),
            "worker {worker_pid} survived its supervisor"
        );
        worker.0 = None;
        let _ = std::fs::remove_file(pid_file);
    }

    #[test]
    #[ignore = "spawned only by supervisor_returns_only_after_every_worker_succeeds"]
    fn supervisor_success_probe() {
        let worker = WorkerRendezvous::from_env()
            .expect("valid worker environment")
            .expect("worker environment present");
        assert_eq!(worker.world_size().get(), 2);
        assert_eq!(
            worker.slot().device_ordinal(),
            [19, 23][worker.slot().rank().get()]
        );
        assert_eq!(worker.owns_campaign_lock(), worker.slot().rank().get() == 0);
        assert_eq!(worker.nccl_id, NcclWireId([0x3c; 128]));
    }

    #[test]
    #[ignore = "spawned only by supervisor_kills_and_reaps_peers_after_worker_failure"]
    fn supervisor_failure_probe() {
        let worker = WorkerRendezvous::from_env()
            .expect("valid worker environment")
            .expect("worker environment present");
        if worker.slot().rank().get() == 0 {
            std::process::exit(17);
        }
        std::thread::sleep(Duration::from_secs(30));
    }

    #[test]
    #[ignore = "spawned only by supervisor_kills_and_reaps_all_workers_after_timeout"]
    fn supervisor_timeout_probe() {
        WorkerRendezvous::from_env()
            .expect("valid worker environment")
            .expect("worker environment present");
        std::thread::sleep(Duration::from_secs(30));
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "spawned only by worker_cannot_survive_supervisor_death"]
    fn supervisor_parent_death_probe() {
        let config = DistributedConfig::new(
            DeviceFleet::new(vec![19], 0).expect("fleet"),
            Duration::from_secs(30),
        )
        .expect("config");
        let executable = env::current_exe().expect("current test executable");
        supervise_executable(
            &config,
            &probe_args("supervisor_parent_death_worker_probe"),
            &executable,
            NcclWireId([0x3c; 128]),
        )
        .expect("supervise parent-death worker");
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "spawned only by worker_cannot_survive_supervisor_death"]
    fn supervisor_parent_death_worker_probe() {
        WorkerRendezvous::from_env()
            .expect("valid worker environment")
            .expect("worker environment present");
        let pid_file = env::var_os(PARENT_DEATH_PID_FILE).expect("worker pid-file path");
        std::fs::write(pid_file, std::process::id().to_string()).expect("publish worker pid");
        std::thread::sleep(Duration::from_secs(30));
    }
}

//! Campaign-owned execution admission over exact selected Qwen packages.

use core::{convert::Infallible, fmt};
use std::{error::Error, path::Path};

use tritium_format::{ModelId, PackageId};
use tritium_nn::{
    NnError, Qwen35ExecutionOutputBatch, Qwen35ExecutionVisitError, Qwen35SaltV2LanguageMtpModel,
    Qwen35UntrustedRuntimeTranscript,
};
use tritium_quantize::SaltV2Profile;

use crate::{ContentId, Qwen36PreservedSafetensorsError};

use super::{Qwen36PackageAdmissionError, Qwen36PackageAdmittedCampaignStore};

const RECEIPT_MAGIC: [u8; 8] = *b"TSQ36EX\0";
const RECEIPT_VERSION: u16 = 1;
const RECEIPT_CHECKSUM_CONTEXT: &str = "tritium qwen3.6 admitted execution receipt checksum v1";
const FINAL_LOGITS_COVERAGE: u8 = 1;
const BLOCK_OUTPUT_COVERAGE: u8 = 2;
const MAX_IDENTITY_BYTES: usize = 4096;
const MAX_RECEIPT_BYTES: usize = 64 * 1024;
const PRESERVED_CHUNK_BYTES: usize = 64 * 1024;

/// Built-in backend implementation selected by a sealed SALT execution session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen36ExecutionBackend {
    /// Tritium's built-in reference CPU backend.
    Cpu,
    /// Tritium's built-in CUDA backend at one process-local ordinal.
    Cuda {
        /// CUDA ordinal passed directly to the built-in backend constructor.
        ordinal: u32,
    },
}

impl Qwen36ExecutionBackend {
    const fn tag(self) -> u8 {
        match self {
            Self::Cpu => 1,
            Self::Cuda { .. } => 2,
        }
    }

    const fn ordinal(self) -> u32 {
        match self {
            Self::Cpu => 0,
            Self::Cuda { ordinal } => ordinal,
        }
    }
}

/// Failure before a sealed built-in-backend execution session exists.
#[derive(Debug)]
pub enum Qwen36ExecutionSessionOpenError {
    /// Package admission, parent lineage, or durable CAS state changed.
    Admission(Qwen36PackageAdmissionError),
    /// Preserved workspace bytes could not be reconstructed and authenticated.
    Workspace(crate::Qwen36TensorWorkError),
    /// Bundle validation or built-in backend/model construction failed.
    Runtime(NnError),
}

impl fmt::Display for Qwen36ExecutionSessionOpenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => write!(formatter, "open Qwen execution session: {error}"),
            Self::Workspace(error) => write!(formatter, "open Qwen execution session: {error}"),
            Self::Runtime(error) => write!(formatter, "open Qwen execution session: {error}"),
        }
    }
}

impl Error for Qwen36ExecutionSessionOpenError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Admission(error) => Some(error),
            Self::Workspace(error) => Some(error),
            Self::Runtime(error) => Some(error),
        }
    }
}

/// Failure while a sealed SALT session streams runtime-produced logits.
#[derive(Debug)]
pub enum Qwen36ExecutionVisitError<E> {
    /// Package admission or its durable parent lineage changed.
    Admission(Qwen36PackageAdmissionError),
    /// Model execution, canonical evidence, or provenance validation failed.
    Runtime(NnError),
    /// Caller observer rejected one runtime-produced batch.
    Observer(E),
}

impl<E: fmt::Display> fmt::Display for Qwen36ExecutionVisitError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => write!(formatter, "admitted Qwen execution: {error}"),
            Self::Runtime(error) => write!(formatter, "admitted Qwen execution: {error}"),
            Self::Observer(error) => write!(formatter, "admitted Qwen observer: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for Qwen36ExecutionVisitError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Admission(error) => Some(error),
            Self::Runtime(error) => Some(error),
            Self::Observer(error) => Some(error),
        }
    }
}

/// Failure while reopening a sealed execution session for durable replay.
#[derive(Debug)]
pub enum Qwen36ExecutionReplayError<E> {
    /// A fresh built-in-backend session could not be reconstructed.
    Open(Qwen36ExecutionSessionOpenError),
    /// Fresh execution, observation, or expected-byte comparison failed.
    Execute(Qwen36ExecutionVisitError<E>),
}

impl<E: fmt::Display> fmt::Display for Qwen36ExecutionReplayError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open(error) => write!(formatter, "reopen admitted Qwen execution: {error}"),
            Self::Execute(error) => write!(formatter, "replay admitted Qwen execution: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for Qwen36ExecutionReplayError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Open(error) => Some(error),
            Self::Execute(error) => Some(error),
        }
    }
}

/// Campaign-admissible receipt minted only by a sealed built-in-backend session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36AdmittedExecutionReceipt {
    receipt_id: ContentId,
    completion_id: ContentId,
    campaign_id: ContentId,
    admission_id: ContentId,
    selection_id: ContentId,
    source_model_id: ModelId,
    master_set_id: [u8; 32],
    profile: SaltV2Profile,
    package_id: PackageId,
    preserved_package_id: PackageId,
    manifest_package_id: String,
    config_package_id: String,
    backend: Qwen36ExecutionBackend,
    backend_id: String,
    physical_device_id: String,
    backend_caps_digest: [u8; 32],
    transcript_content_id: ContentId,
    transcript_id: [u8; 32],
    token_stream_digest: [u8; 32],
    final_logits_digest: [u8; 32],
    scope_coverage: u8,
    batch_count: u64,
    token_count: u64,
    logit_count: u64,
}

impl Qwen36AdmittedExecutionReceipt {
    /// Content identity of exact canonical admitted-receipt bytes.
    #[must_use]
    pub const fn receipt_id(&self) -> ContentId {
        self.receipt_id
    }

    /// Exact complete tensor-master campaign executed by this receipt.
    #[must_use]
    pub const fn completion_id(&self) -> ContentId {
        self.completion_id
    }

    /// Exact additive campaign executed by this receipt.
    #[must_use]
    pub const fn campaign_id(&self) -> ContentId {
        self.campaign_id
    }

    /// Aggregate identity of every ordered canonical tensor master.
    #[must_use]
    pub const fn master_set_id(&self) -> &[u8; 32] {
        &self.master_set_id
    }

    /// Exact package-admission capability that authorized execution.
    #[must_use]
    pub const fn package_admission_id(&self) -> ContentId {
        self.admission_id
    }

    /// Exact selected allocation from which the runtime package was materialized.
    #[must_use]
    pub const fn selection_id(&self) -> ContentId {
        self.selection_id
    }

    /// Source-model semantic identity inherited by the admitted campaign.
    #[must_use]
    pub const fn source_model_id(&self) -> ModelId {
        self.source_model_id
    }

    /// Selected runtime profile.
    #[must_use]
    pub const fn profile(&self) -> SaltV2Profile {
        self.profile
    }

    /// Exact selected SALT package executed by the runtime.
    #[must_use]
    pub const fn package_id(&self) -> PackageId {
        self.package_id
    }

    /// Exact preserved BF16 companion executed by the runtime.
    #[must_use]
    pub const fn preserved_package_id(&self) -> PackageId {
        self.preserved_package_id
    }

    /// Exact strict bundle manifest loaded by the runtime.
    #[must_use]
    pub fn manifest_package_id(&self) -> &str {
        &self.manifest_package_id
    }

    /// Exact Hugging Face configuration loaded by the runtime.
    #[must_use]
    pub fn config_package_id(&self) -> &str {
        &self.config_package_id
    }

    /// Built-in implementation constructed inside the sealed session.
    #[must_use]
    pub const fn backend(&self) -> Qwen36ExecutionBackend {
        self.backend
    }

    /// Logical identity reported by the internally constructed backend.
    #[must_use]
    pub fn backend_id(&self) -> &str {
        &self.backend_id
    }

    /// Driver-reported physical identity from the sealed built-in backend.
    #[must_use]
    pub fn physical_device_id(&self) -> &str {
        &self.physical_device_id
    }

    /// Canonical capability digest from the internally constructed backend.
    #[must_use]
    pub const fn backend_caps_digest(&self) -> &[u8; 32] {
        &self.backend_caps_digest
    }

    /// Exact token stream including batch boundaries and order.
    #[must_use]
    pub const fn token_stream_digest(&self) -> &[u8; 32] {
        &self.token_stream_digest
    }

    /// Runtime-produced final-logit stream identity.
    #[must_use]
    pub const fn final_logits_digest(&self) -> &[u8; 32] {
        &self.final_logits_digest
    }

    /// Whether all admitted batches carry final-position logits.
    #[must_use]
    pub const fn has_final_logits(&self) -> bool {
        self.scope_coverage & FINAL_LOGITS_COVERAGE != 0
    }

    /// Whether block/window outputs are admitted by this receipt.
    #[must_use]
    pub const fn has_block_outputs(&self) -> bool {
        self.scope_coverage & BLOCK_OUTPUT_COVERAGE != 0
    }

    /// Number of fresh-cache token batches executed.
    #[must_use]
    pub const fn batch_count(&self) -> u64 {
        self.batch_count
    }

    /// Total input tokens executed.
    #[must_use]
    pub const fn token_count(&self) -> u64 {
        self.token_count
    }

    /// Total final-logit values emitted.
    #[must_use]
    pub const fn logit_count(&self) -> u64 {
        self.logit_count
    }

    /// Encode canonical `TSQ36EX` version-1 admitted evidence.
    ///
    /// # Errors
    /// Returns [`NnError`] for invalid identities, overflow, or bounded
    /// allocation failure.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, NnError> {
        let string_bytes = self
            .identity_strings()
            .iter()
            .try_fold(0usize, |total, value| {
                total
                    .checked_add(2)
                    .and_then(|total| total.checked_add(value.len()))
                    .ok_or_else(|| {
                        NnError::ResourceExhausted(
                            "Qwen admitted execution receipt length overflow".to_owned(),
                        )
                    })
            })?;
        let capacity = (8usize + 2 + 1 + 1 + 4 + 1 + 7)
            .checked_add(13 * 32)
            .and_then(|bytes| bytes.checked_add(3 * 8))
            .and_then(|bytes| bytes.checked_add(string_bytes))
            .and_then(|bytes| bytes.checked_add(32))
            .ok_or_else(|| {
                NnError::ResourceExhausted(
                    "Qwen admitted execution receipt length overflow".to_owned(),
                )
            })?;
        if capacity > MAX_RECEIPT_BYTES {
            return Err(NnError::ResourceExhausted(
                "Qwen admitted execution receipt exceeds canonical bound".to_owned(),
            ));
        }
        let mut output = Vec::new();
        output.try_reserve_exact(capacity).map_err(|_| {
            NnError::ResourceExhausted("allocate Qwen admitted execution receipt".to_owned())
        })?;
        output.extend_from_slice(&RECEIPT_MAGIC);
        output.extend_from_slice(&RECEIPT_VERSION.to_le_bytes());
        output.push(profile_tag(self.profile));
        output.push(self.backend.tag());
        output.extend_from_slice(&self.backend.ordinal().to_le_bytes());
        output.push(self.scope_coverage);
        output.extend_from_slice(&[0; 7]);
        for digest in self.bound_digests() {
            output.extend_from_slice(&digest);
        }
        for count in [self.batch_count, self.token_count, self.logit_count] {
            output.extend_from_slice(&count.to_le_bytes());
        }
        for value in self.identity_strings() {
            encode_string(&mut output, value)?;
        }
        let mut checksum = blake3::Hasher::new_derive_key(RECEIPT_CHECKSUM_CONTEXT);
        checksum.update(&output);
        output.extend_from_slice(checksum.finalize().as_bytes());
        debug_assert_eq!(output.len(), capacity);
        Ok(output)
    }

    fn from_transcript(
        authority: &ExecutionAuthority,
        transcript: &Qwen35UntrustedRuntimeTranscript,
    ) -> Result<Self, NnError> {
        let transcript_bytes = transcript.canonical_bytes()?;
        let mut receipt = Self {
            receipt_id: ContentId::from_digest([0; 32]),
            completion_id: authority.completion_id,
            campaign_id: authority.campaign_id,
            admission_id: authority.admission_id,
            selection_id: authority.selection_id,
            source_model_id: authority.source_model_id,
            master_set_id: authority.master_set_id,
            profile: authority.profile,
            package_id: authority.package_id,
            preserved_package_id: authority.preserved_package_id,
            manifest_package_id: try_owned(transcript.manifest_package_id())?,
            config_package_id: try_owned(transcript.config_package_id())?,
            backend: authority.backend,
            backend_id: try_owned(transcript.claimed_backend_id())?,
            physical_device_id: try_owned(transcript.claimed_physical_device_id())?,
            backend_caps_digest: *transcript.claimed_backend_caps_digest(),
            transcript_content_id: ContentId::of_bytes(&transcript_bytes),
            transcript_id: *transcript.transcript_id(),
            token_stream_digest: *transcript.token_stream_digest(),
            final_logits_digest: *transcript.final_logits_digest(),
            scope_coverage: FINAL_LOGITS_COVERAGE,
            batch_count: transcript.batch_count(),
            token_count: transcript.token_count(),
            logit_count: transcript.logit_count(),
        };
        let canonical = receipt.canonical_bytes()?;
        receipt.receipt_id = ContentId::of_bytes(&canonical);
        Ok(receipt)
    }

    fn identity_strings(&self) -> [&str; 4] {
        [
            &self.manifest_package_id,
            &self.config_package_id,
            &self.backend_id,
            &self.physical_device_id,
        ]
    }

    fn bound_digests(&self) -> [[u8; 32]; 13] {
        [
            *self.completion_id.as_bytes(),
            *self.campaign_id.as_bytes(),
            *self.admission_id.as_bytes(),
            *self.selection_id.as_bytes(),
            *self.source_model_id.as_bytes(),
            self.master_set_id,
            *self.package_id.as_bytes(),
            *self.preserved_package_id.as_bytes(),
            *self.transcript_content_id.as_bytes(),
            self.transcript_id,
            self.backend_caps_digest,
            self.token_stream_digest,
            self.final_logits_digest,
        ]
    }
}

/// Sealed model plus live package-admission capability.
pub struct Qwen36AdmittedExecutionSession<'admission, 'allocated, 'parent, 'store, 'source> {
    admission: &'admission Qwen36PackageAdmittedCampaignStore<'allocated, 'parent, 'store, 'source>,
    model: Qwen35SaltV2LanguageMtpModel,
    authority: ExecutionAuthority,
}

impl fmt::Debug for Qwen36AdmittedExecutionSession<'_, '_, '_, '_, '_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Qwen36AdmittedExecutionSession")
            .field("completion_id", &self.authority.completion_id)
            .field("admission_id", &self.authority.admission_id)
            .field("profile", &self.authority.profile)
            .field("backend", &self.authority.backend)
            .finish_non_exhaustive()
    }
}

impl<'admission, 'allocated, 'parent, 'store, 'source>
    Qwen36AdmittedExecutionSession<'admission, 'allocated, 'parent, 'store, 'source>
{
    /// Execute exact token batches and mint campaign-admitted final-logit evidence.
    ///
    /// # Errors
    /// Fails closed if admission state changes, runtime execution fails, backend
    /// or artifact identity differs from the sealed authority, or the observer
    /// rejects a batch. No receipt is returned on any failure.
    pub fn try_visit_final_logits<'batch, I, E>(
        &self,
        batches: I,
        observer: impl FnMut(Qwen35ExecutionOutputBatch<'_>) -> Result<(), E>,
    ) -> Result<Qwen36AdmittedExecutionReceipt, Qwen36ExecutionVisitError<E>>
    where
        I: IntoIterator<Item = &'batch [u32]>,
    {
        self.admission
            .verify_current()
            .map_err(Qwen36ExecutionVisitError::Admission)?;
        let transcript = self
            .model
            .try_visit_untrusted_final_logits(batches, observer)
            .map_err(map_execution_error)?;
        validate_transcript(&self.authority, &transcript)
            .map_err(Qwen36ExecutionVisitError::Runtime)?;
        self.admission
            .verify_current()
            .map_err(Qwen36ExecutionVisitError::Admission)?;
        Qwen36AdmittedExecutionReceipt::from_transcript(&self.authority, &transcript)
            .map_err(Qwen36ExecutionVisitError::Runtime)
    }

    fn execute_and_compare<'batch, I, E>(
        &self,
        batches: I,
        expected_canonical: &[u8],
        observer: impl FnMut(Qwen35ExecutionOutputBatch<'_>) -> Result<(), E>,
    ) -> Result<Qwen36AdmittedExecutionReceipt, Qwen36ExecutionVisitError<E>>
    where
        I: IntoIterator<Item = &'batch [u32]>,
    {
        let receipt = self.try_visit_final_logits(batches, observer)?;
        let actual = receipt
            .canonical_bytes()
            .map_err(Qwen36ExecutionVisitError::Runtime)?;
        if actual != expected_canonical {
            return Err(Qwen36ExecutionVisitError::Runtime(NnError::Provenance(
                "admitted Qwen execution differs from fresh sealed execution".to_owned(),
            )));
        }
        Ok(receipt)
    }
}

impl<'allocated, 'parent, 'store, 'source>
    Qwen36PackageAdmittedCampaignStore<'allocated, 'parent, 'store, 'source>
{
    /// Construct Tritium's built-in CPU backend and seal it to this admission.
    ///
    /// The caller supplies neither a backend nor a transcript. Exact package,
    /// preserved-source, completion, master-set, selection, and source identities
    /// are revalidated before the session exists.
    ///
    /// # Errors
    /// Returns [`Qwen36ExecutionSessionOpenError`] for changed campaign state,
    /// preserved-source reconstruction failure, bundle mismatch, or model load.
    pub fn open_cpu_execution_session<'admission>(
        &'admission self,
        bundle_dir: &Path,
        profile: SaltV2Profile,
    ) -> Result<
        Qwen36AdmittedExecutionSession<'admission, 'allocated, 'parent, 'store, 'source>,
        Qwen36ExecutionSessionOpenError,
    > {
        self.open_execution_session(profile, Qwen36ExecutionBackend::Cpu, || {
            Qwen35SaltV2LanguageMtpModel::load_bundle_profile(
                bundle_dir,
                profile_name(profile),
                Box::new(tritium_cpu::CpuBackend::new()),
            )
        })
    }

    /// Reopen a fresh built-in CPU session, re-execute, and require exact evidence.
    ///
    /// This method does not reuse a previously loaded model. Bundle deletion,
    /// replacement, config drift, package substitution, and campaign mutation are
    /// revalidated before any durable replay can succeed.
    ///
    /// # Errors
    /// Returns [`Qwen36ExecutionReplayError::Open`] when fresh session construction
    /// fails, or [`Qwen36ExecutionReplayError::Execute`] for execution, observer,
    /// or canonical-byte mismatch.
    pub fn reexecute_cpu_final_logits<'batch, I, E>(
        &self,
        bundle_dir: &Path,
        profile: SaltV2Profile,
        batches: I,
        expected_canonical: &[u8],
        observer: impl FnMut(Qwen35ExecutionOutputBatch<'_>) -> Result<(), E>,
    ) -> Result<Qwen36AdmittedExecutionReceipt, Qwen36ExecutionReplayError<E>>
    where
        I: IntoIterator<Item = &'batch [u32]>,
    {
        self.open_cpu_execution_session(bundle_dir, profile)
            .map_err(Qwen36ExecutionReplayError::Open)?
            .execute_and_compare(batches, expected_canonical, observer)
            .map_err(Qwen36ExecutionReplayError::Execute)
    }

    /// Construct Tritium's built-in CUDA backend and seal it to this admission.
    ///
    /// # Errors
    /// Returns [`Qwen36ExecutionSessionOpenError`] for backend construction,
    /// changed campaign state, bundle mismatch, or model load failure.
    #[cfg(feature = "cuda")]
    pub fn open_cuda_execution_session<'admission>(
        &'admission self,
        bundle_dir: &Path,
        profile: SaltV2Profile,
        ordinal: u32,
    ) -> Result<
        Qwen36AdmittedExecutionSession<'admission, 'allocated, 'parent, 'store, 'source>,
        Qwen36ExecutionSessionOpenError,
    > {
        self.open_execution_session(profile, Qwen36ExecutionBackend::Cuda { ordinal }, || {
            let ordinal = usize::try_from(ordinal)
                .map_err(|_| NnError::Backend("CUDA ordinal exceeds usize".to_owned()))?;
            let backend = tritium_cuda::CudaBackend::new(ordinal).map_err(NnError::from)?;
            Qwen35SaltV2LanguageMtpModel::load_bundle_profile(
                bundle_dir,
                profile_name(profile),
                Box::new(backend),
            )
        })
    }

    /// Reopen a fresh built-in CUDA session, re-execute, and require exact evidence.
    ///
    /// # Errors
    /// Returns [`Qwen36ExecutionReplayError::Open`] when backend/session
    /// reconstruction fails, or [`Qwen36ExecutionReplayError::Execute`] for
    /// execution, observer, or canonical-byte mismatch.
    #[cfg(feature = "cuda")]
    pub fn reexecute_cuda_final_logits<'batch, I, E>(
        &self,
        bundle_dir: &Path,
        profile: SaltV2Profile,
        ordinal: u32,
        batches: I,
        expected_canonical: &[u8],
        observer: impl FnMut(Qwen35ExecutionOutputBatch<'_>) -> Result<(), E>,
    ) -> Result<Qwen36AdmittedExecutionReceipt, Qwen36ExecutionReplayError<E>>
    where
        I: IntoIterator<Item = &'batch [u32]>,
    {
        self.open_cuda_execution_session(bundle_dir, profile, ordinal)
            .map_err(Qwen36ExecutionReplayError::Open)?
            .execute_and_compare(batches, expected_canonical, observer)
            .map_err(Qwen36ExecutionReplayError::Execute)
    }

    #[cfg(test)]
    pub(crate) fn open_cpu_execution_session_test_fixture<'admission>(
        &'admission self,
        bundle_dir: &Path,
        profile: SaltV2Profile,
    ) -> Result<
        Qwen36AdmittedExecutionSession<'admission, 'allocated, 'parent, 'store, 'source>,
        Qwen36ExecutionSessionOpenError,
    > {
        self.open_execution_session(profile, Qwen36ExecutionBackend::Cpu, || {
            Qwen35SaltV2LanguageMtpModel::load_bundle_profile_test_fixture(
                bundle_dir,
                profile_name(profile),
                Box::new(tritium_cpu::CpuBackend::new()),
            )
        })
    }

    #[cfg(test)]
    pub(crate) fn reexecute_cpu_final_logits_test_fixture<'batch, I, E>(
        &self,
        bundle_dir: &Path,
        profile: SaltV2Profile,
        batches: I,
        expected_canonical: &[u8],
        observer: impl FnMut(Qwen35ExecutionOutputBatch<'_>) -> Result<(), E>,
    ) -> Result<Qwen36AdmittedExecutionReceipt, Qwen36ExecutionReplayError<E>>
    where
        I: IntoIterator<Item = &'batch [u32]>,
    {
        self.open_cpu_execution_session_test_fixture(bundle_dir, profile)
            .map_err(Qwen36ExecutionReplayError::Open)?
            .execute_and_compare(batches, expected_canonical, observer)
            .map_err(Qwen36ExecutionReplayError::Execute)
    }

    fn open_execution_session<'admission>(
        &'admission self,
        profile: SaltV2Profile,
        backend: Qwen36ExecutionBackend,
        load_model: impl FnOnce() -> Result<Qwen35SaltV2LanguageMtpModel, NnError>,
    ) -> Result<
        Qwen36AdmittedExecutionSession<'admission, 'allocated, 'parent, 'store, 'source>,
        Qwen36ExecutionSessionOpenError,
    > {
        self.verify_current()
            .map_err(Qwen36ExecutionSessionOpenError::Admission)?;
        let preserved = self
            .allocated
            .parent
            .base
            .try_write_preserved_safetensors(PRESERVED_CHUNK_BYTES, |_| Ok::<_, Infallible>(()))
            .map_err(|error| match error {
                Qwen36PreservedSafetensorsError::Workspace(error) => {
                    Qwen36ExecutionSessionOpenError::Workspace(error)
                }
                Qwen36PreservedSafetensorsError::Sink(error) => match error {},
            })?;
        let authority = execution_authority(self, profile, backend, preserved.package_id());
        let model = load_model().map_err(Qwen36ExecutionSessionOpenError::Runtime)?;
        validate_loaded_model(&authority, &model)
            .map_err(Qwen36ExecutionSessionOpenError::Runtime)?;
        self.verify_current()
            .map_err(Qwen36ExecutionSessionOpenError::Admission)?;
        Ok(Qwen36AdmittedExecutionSession {
            admission: self,
            model,
            authority,
        })
    }
}

#[derive(Clone, Debug)]
struct ExecutionAuthority {
    completion_id: ContentId,
    campaign_id: ContentId,
    admission_id: ContentId,
    selection_id: ContentId,
    source_model_id: ModelId,
    master_set_id: [u8; 32],
    profile: SaltV2Profile,
    package_id: PackageId,
    preserved_package_id: PackageId,
    identity_status: &'static str,
    official_payload_authenticated: bool,
    backend: Qwen36ExecutionBackend,
}

fn execution_authority(
    admission: &Qwen36PackageAdmittedCampaignStore<'_, '_, '_, '_>,
    profile: SaltV2Profile,
    backend: Qwen36ExecutionBackend,
    preserved_package_id: PackageId,
) -> ExecutionAuthority {
    let completion = &admission.allocated.parent_completion;
    let selected = match profile {
        SaltV2Profile::CompactV1 => admission.receipt.compact(),
        SaltV2Profile::NearLosslessV1 => admission.receipt.near_lossless(),
    };
    ExecutionAuthority {
        completion_id: completion.completion_id(),
        campaign_id: completion.campaign_id(),
        admission_id: admission.receipt.admission_id(),
        selection_id: admission.receipt.selection_id(),
        source_model_id: completion.source_model_id(),
        master_set_id: completion.master_set_id(),
        profile,
        package_id: selected.package_id(),
        preserved_package_id,
        identity_status: completion.identity_status().as_str(),
        official_payload_authenticated: completion
            .identity_status()
            .official_payload_authenticated(),
        backend,
    }
}

fn validate_loaded_model(
    authority: &ExecutionAuthority,
    model: &Qwen35SaltV2LanguageMtpModel,
) -> Result<(), NnError> {
    let load = model.receipt();
    if load.profile() != profile_name(authority.profile)
        || load.package_id() != authority.package_id.to_string()
        || load.preserved_package_id() != authority.preserved_package_id.to_string()
        || load.declared_completion_id() != authority.completion_id.to_string()
        || load.declared_campaign_id() != authority.campaign_id.to_string()
        || load.declared_admission_id() != authority.admission_id.to_string()
        || load.declared_selection_id() != authority.selection_id.to_string()
        || load.declared_source_model_id() != authority.source_model_id.to_string()
        || load.declared_source_identity_status() != authority.identity_status
        || load.declared_official_payload_authenticated()
            != authority.official_payload_authenticated
    {
        return Err(NnError::Provenance(
            "Qwen bundle differs from authoritative SALT campaign lineage".to_owned(),
        ));
    }
    Ok(())
}

fn validate_transcript(
    authority: &ExecutionAuthority,
    transcript: &Qwen35UntrustedRuntimeTranscript,
) -> Result<(), NnError> {
    let expected_backend = match authority.backend {
        Qwen36ExecutionBackend::Cpu => "cpu".to_owned(),
        Qwen36ExecutionBackend::Cuda { ordinal } => format!("cuda:{ordinal}"),
    };
    if !transcript.backend_claims_are_untrusted()
        || transcript.profile() != profile_name(authority.profile)
        || transcript.package_id() != authority.package_id.to_string()
        || transcript.preserved_package_id() != authority.preserved_package_id.to_string()
        || transcript.claimed_backend_id() != expected_backend
        || !transcript.has_final_logits()
        || transcript.has_block_outputs()
    {
        return Err(NnError::Provenance(
            "Qwen runtime transcript differs from sealed execution authority".to_owned(),
        ));
    }
    Ok(())
}

fn map_execution_error<E>(error: Qwen35ExecutionVisitError<E>) -> Qwen36ExecutionVisitError<E> {
    match error {
        Qwen35ExecutionVisitError::Runtime(error) => Qwen36ExecutionVisitError::Runtime(error),
        Qwen35ExecutionVisitError::Observer(error) => Qwen36ExecutionVisitError::Observer(error),
    }
}

const fn profile_name(profile: SaltV2Profile) -> &'static str {
    match profile {
        SaltV2Profile::CompactV1 => "compact-v1",
        SaltV2Profile::NearLosslessV1 => "near-lossless-v1",
    }
}

const fn profile_tag(profile: SaltV2Profile) -> u8 {
    match profile {
        SaltV2Profile::CompactV1 => 1,
        SaltV2Profile::NearLosslessV1 => 2,
    }
}

fn try_owned(value: &str) -> Result<String, NnError> {
    validate_identity(value)?;
    let mut output = String::new();
    output
        .try_reserve_exact(value.len())
        .map_err(|_| NnError::ResourceExhausted("allocate Qwen execution identity".to_owned()))?;
    output.push_str(value);
    Ok(output)
}

fn validate_identity(value: &str) -> Result<(), NnError> {
    if value.is_empty() || value.len() > MAX_IDENTITY_BYTES || value.contains('\0') {
        return Err(NnError::Provenance(
            "Qwen execution identity is empty, oversized, or contains NUL".to_owned(),
        ));
    }
    Ok(())
}

fn encode_string(output: &mut Vec<u8>, value: &str) -> Result<(), NnError> {
    validate_identity(value)?;
    let length = u16::try_from(value.len()).map_err(|_| {
        NnError::ResourceExhausted("Qwen execution identity exceeds u16".to_owned())
    })?;
    output.extend_from_slice(&length.to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_receipt_layout_is_frozen() {
        let receipt = Qwen36AdmittedExecutionReceipt {
            receipt_id: ContentId::from_digest([0; 32]),
            completion_id: ContentId::from_digest([1; 32]),
            campaign_id: ContentId::from_digest([2; 32]),
            admission_id: ContentId::from_digest([3; 32]),
            selection_id: ContentId::from_digest([4; 32]),
            source_model_id: ModelId::from_digest([5; 32]),
            master_set_id: [6; 32],
            profile: SaltV2Profile::NearLosslessV1,
            package_id: PackageId::from_digest([7; 32]),
            preserved_package_id: PackageId::from_digest([8; 32]),
            manifest_package_id: "manifest".to_owned(),
            config_package_id: "config".to_owned(),
            backend: Qwen36ExecutionBackend::Cuda { ordinal: 9 },
            backend_id: "cuda:9".to_owned(),
            physical_device_id: "cuda:9:GPU-fixture".to_owned(),
            backend_caps_digest: [10; 32],
            transcript_content_id: ContentId::from_digest([11; 32]),
            transcript_id: [12; 32],
            token_stream_digest: [13; 32],
            final_logits_digest: [14; 32],
            scope_coverage: FINAL_LOGITS_COVERAGE,
            batch_count: 15,
            token_count: 16,
            logit_count: 17,
        };
        let canonical = receipt.canonical_bytes().expect("encode frozen receipt");
        assert_eq!(&canonical[..8], &RECEIPT_MAGIC);
        assert_eq!(&canonical[8..10], &RECEIPT_VERSION.to_le_bytes());
        assert_eq!(canonical[10], 2);
        assert_eq!(canonical[11], 2);
        assert_eq!(&canonical[12..16], &9_u32.to_le_bytes());
        assert_eq!(canonical[16], FINAL_LOGITS_COVERAGE);
        assert_eq!(&canonical[17..24], &[0; 7]);
        assert_eq!(canonical.len(), 542);
        assert_eq!(
            ContentId::of_bytes(&canonical).to_string(),
            "tsc1_28e9d809f0b8a02b96da9eef37f6484e8d1344765b519a4182b11061d4b661c6"
        );
    }
}

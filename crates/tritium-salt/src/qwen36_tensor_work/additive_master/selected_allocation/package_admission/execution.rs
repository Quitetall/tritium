//! Campaign-owned execution admission over exact selected Qwen packages.

mod output_binding;
mod refined_candidate;

pub use output_binding::{
    Qwen36FinalLogitsOutputBindingError, Qwen36FinalLogitsOutputBindingReceipt,
    Qwen36OutputScopeBindingReceipt,
};
use refined_candidate::ChildReplayEvidence;
pub use refined_candidate::Qwen36RefinedCandidateExecutionReceipt;

use core::{convert::Infallible, fmt};
use std::{
    error::Error,
    io::{Read, Seek, Write},
    path::Path,
};

use tritium_format::{
    ModelId, PackageId, RuntimeOutputScope,
    salt_v2_package::{
        SaltV2PackageReader, SaltV2ScaleUpdateChild, SaltV2ScaleUpdateChildError,
        write_salt_v2_scale_update_child,
    },
};
use tritium_nn::{
    NnError, Projection, Qwen35ExecutionOutputBatch, Qwen35ExecutionVisitError,
    Qwen35SaltV2LanguageMtpModel, Qwen35UntrustedRuntimeTranscript,
};
use tritium_quantize::{
    FixedTritScaleUpdateCandidate, FixedTritScaleUpdateCandidateBuilder,
    OutputReconstructionActivationSet, OutputReconstructionActivationSource,
    OutputReconstructionError, OutputReconstructionScaleCandidate, OutputReconstructionScope,
    OutputReconstructionSpec, SaltV2Profile,
};

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

/// Failure while fitting one fixed-trit scale plane from admitted Qwen activations.
#[derive(Debug)]
pub enum Qwen36ScaleRefitWindowError {
    /// The parent package admission changed during the operation.
    Admission(Qwen36PackageAdmissionError),
    /// The parent execution or Qwen projection computation was invalid.
    Runtime(NnError),
    /// The activation scope or scale-fit observation did not match its contract.
    Fit(OutputReconstructionError),
    /// Tensor name did not identify a canonical language-layer projection.
    InvalidTensorName,
    /// Spec, parent, or in-progress candidate identities disagree.
    ProvenanceMismatch,
}

impl fmt::Display for Qwen36ScaleRefitWindowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(error) => write!(formatter, "scale refit admission: {error}"),
            Self::Runtime(error) => write!(formatter, "scale refit Qwen runtime: {error}"),
            Self::Fit(error) => write!(formatter, "scale refit fit: {error}"),
            Self::InvalidTensorName => formatter
                .write_str("scale refit tensor name is not a canonical language projection"),
            Self::ProvenanceMismatch => {
                formatter.write_str("scale refit spec, parent, execution, or candidate differs")
            }
        }
    }
}

impl Error for Qwen36ScaleRefitWindowError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Admission(error) => Some(error),
            Self::Runtime(error) => Some(error),
            Self::Fit(error) => Some(error),
            Self::InvalidTensorName | Self::ProvenanceMismatch => None,
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

/// Exact inputs for a fresh, campaign-admitted scale-refined child replay.
#[derive(Debug)]
pub struct Qwen36RefinedCandidateReplay<'a> {
    /// Current admitted PTQ parent execution receipt.
    pub parent_execution: &'a Qwen36AdmittedExecutionReceipt,
    /// Bundle directory containing the parent manifest and preserved tensors.
    pub bundle_dir: &'a Path,
    /// Immutable child SALT package bytes to execute.
    pub child_package_path: &'a Path,
    /// Verified parent/update/child package lineage.
    pub lineage: SaltV2ScaleUpdateChild,
    /// Frozen output-evaluation specification.
    pub spec: &'a OutputReconstructionSpec,
    /// Canonical output-reconstruction receipt bytes selected for this child.
    pub output_bytes: &'a [u8],
    /// Content-bound scale candidate matching the child lineage.
    pub scale_candidate: OutputReconstructionScaleCandidate<'a>,
    /// Ordered tokens and row masks for all frozen block and final-logit scopes.
    pub scope_batches: &'a [(&'a [u32], &'a [bool])],
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
    /// Materialize one fitted restart as a strict immutable child of this session's package.
    ///
    /// The parent reader must be the strict package instance used by this admitted
    /// execution session. The returned child lineage binds the exact parent and
    /// fitted scale update. Callers must stage `output` and publish it atomically
    /// only after this method succeeds; the writer may contain partial bytes on error.
    /// Each restart may be materialized for candidate scoring, then the selected
    /// restart must be resolved from its output receipt with
    /// [`tritium_quantize::OutputReconstructionReceipt::selected_fitted_scale_update_candidate`].
    ///
    /// # Errors
    /// Rejects changed admission, a different parent execution/package, mismatched
    /// fit specification or parent identity, invalid package updates, and I/O errors.
    pub fn materialize_scale_update_candidate_child<R, W>(
        &self,
        parent_execution: &Qwen36AdmittedExecutionReceipt,
        spec: &OutputReconstructionSpec,
        candidate: &FixedTritScaleUpdateCandidate,
        parent: &mut SaltV2PackageReader<R>,
        output: W,
    ) -> Result<(W, SaltV2ScaleUpdateChild), Qwen36ExecutionVisitError<Infallible>>
    where
        R: Read + Seek,
        W: Read + Write + Seek,
    {
        self.admission
            .verify_current()
            .map_err(Qwen36ExecutionVisitError::Admission)?;
        output_binding::validate_execution(&self.authority, parent_execution)
            .map_err(Qwen36ExecutionVisitError::Runtime)?;
        if spec.source_model_id() != self.authority.source_model_id
            || spec.token_stream_digest() != parent_execution.token_stream_digest()
            || parent_execution.package_id() != self.authority.package_id
            || parent.package_id() != self.authority.package_id
        {
            return Err(Qwen36ExecutionVisitError::Runtime(NnError::Provenance(
                "scale-update child parent differs from admitted execution".to_owned(),
            )));
        }
        let scale_candidate = candidate.as_scale_candidate(spec).map_err(|error| {
            Qwen36ExecutionVisitError::Runtime(NnError::InvalidArtifact(format!(
                "validate scale-update candidate: {error}"
            )))
        })?;
        if scale_candidate.parent_package_digest() != self.authority.package_id.as_bytes() {
            return Err(Qwen36ExecutionVisitError::Runtime(NnError::Provenance(
                "scale-update candidate is bound to a different admitted package".to_owned(),
            )));
        }
        let (output, lineage) =
            write_salt_v2_scale_update_child(parent, output, scale_candidate.updates()).map_err(
                |error: SaltV2ScaleUpdateChildError| {
                    Qwen36ExecutionVisitError::Runtime(NnError::InvalidArtifact(format!(
                        "materialize scale-update child: {error}"
                    )))
                },
            )?;
        parent.verify_unchanged().map_err(|error| {
            Qwen36ExecutionVisitError::Runtime(NnError::InvalidArtifact(format!(
                "verify scale-update parent after child materialization: {error}"
            )))
        })?;
        self.admission
            .verify_current()
            .map_err(Qwen36ExecutionVisitError::Admission)?;
        Ok((output, lineage))
    }

    /// Observe one frozen output-reconstruction scope in an exact fixed-trit fit.
    ///
    /// The activation set is reopened against `spec` for every call, and the
    /// parent execution, model identity, in-progress builder, and live package
    /// admission are checked before the dense teacher/current-package outputs
    /// reach the scale fitter. The builder must already be active on the exact
    /// named projection's packed parent plane. Call once for each frozen scope
    /// that contains that projection's layer.
    ///
    /// # Errors
    /// Rejects changed admission, mismatched model/token/spec/parent identities,
    /// malformed activation sources or scopes, and projection/fit errors.
    #[allow(clippy::too_many_arguments)]
    pub fn observe_scale_refit_scope<S: OutputReconstructionActivationSource + ?Sized>(
        &self,
        parent_execution: &Qwen36AdmittedExecutionReceipt,
        spec: &OutputReconstructionSpec,
        activation_source: &S,
        scope: OutputReconstructionScope,
        token_start: u64,
        token_count: u64,
        max_decoded_bytes: u64,
        tensor_name: &str,
        teacher: &Projection,
        builder: &mut FixedTritScaleUpdateCandidateBuilder<'_>,
    ) -> Result<(), Qwen36ScaleRefitWindowError> {
        self.admission
            .verify_current()
            .map_err(Qwen36ScaleRefitWindowError::Admission)?;
        output_binding::validate_execution(&self.authority, parent_execution)
            .map_err(Qwen36ScaleRefitWindowError::Runtime)?;
        if spec.source_model_id() != self.authority.source_model_id
            || spec.token_stream_digest() != parent_execution.token_stream_digest()
            || parent_execution.package_id() != self.authority.package_id
            || !builder.is_bound_to(spec, self.authority.package_id.as_bytes())
        {
            return Err(Qwen36ScaleRefitWindowError::ProvenanceMismatch);
        }

        let layer_index = qwen_language_projection_layer(tensor_name)
            .ok_or(Qwen36ScaleRefitWindowError::InvalidTensorName)?;
        let OutputReconstructionScope::Block { start, end } = scope else {
            return Err(Qwen36ScaleRefitWindowError::Fit(
                OutputReconstructionError::InvalidActivationWindowScope,
            ));
        };
        if layer_index < start || layer_index >= end {
            return Err(Qwen36ScaleRefitWindowError::Fit(
                OutputReconstructionError::InvalidActivationWindowScope,
            ));
        }

        let activation_set = OutputReconstructionActivationSet::new(spec, activation_source)
            .map_err(Qwen36ScaleRefitWindowError::Fit)?;
        let windows = activation_set
            .read_window(scope, token_start, token_count, max_decoded_bytes)
            .map_err(Qwen36ScaleRefitWindowError::Fit)?;
        let activation_window =
            windows
                .layer(layer_index)
                .ok_or(Qwen36ScaleRefitWindowError::Fit(
                    OutputReconstructionError::InvalidActivationWindowScope,
                ))?;
        let rows = usize::try_from(activation_window.token_count()).map_err(|_| {
            Qwen36ScaleRefitWindowError::Fit(OutputReconstructionError::InvalidGeometry)
        })?;
        let mut fit_error = None;
        self.model
            .runner()
            .visit_named_projection_output_pairs(
                tensor_name,
                teacher,
                activation_window.values(),
                rows,
                |teacher_outputs, current_outputs| {
                    if fit_error.is_none() {
                        fit_error = builder
                            .observe_window_from_current_projection(
                                activation_window,
                                teacher_outputs,
                                current_outputs,
                            )
                            .err();
                    }
                },
            )
            .map_err(Qwen36ScaleRefitWindowError::Runtime)?;
        if let Some(error) = fit_error {
            return Err(Qwen36ScaleRefitWindowError::Fit(error));
        }
        self.admission
            .verify_current()
            .map_err(Qwen36ScaleRefitWindowError::Admission)
    }

    /// Observe every frozen scheduled block window that contains one projection.
    ///
    /// Token windows are supplied in increasing, non-overlapping order and must
    /// match the spec's frozen batch count. Windows are streamed one at a time;
    /// each admitted scope is reopened and checked by
    /// [`Self::observe_scale_refit_scope`]. Final-logit scopes are deliberately
    /// excluded because they are scored by the output-reconstruction receipt,
    /// not used as layer-local scale-fit observations.
    ///
    /// The builder is incremental. If any observation fails, the caller must
    /// discard it rather than finish or publish a partial candidate.
    ///
    /// # Errors
    /// Rejects malformed batch schedules, absent projection scopes, changed
    /// admission, mismatched identities, or invalid activation/output data.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn observe_scale_refit_scheduled_windows<
        S: OutputReconstructionActivationSource + ?Sized,
    >(
        &self,
        parent_execution: &Qwen36AdmittedExecutionReceipt,
        spec: &OutputReconstructionSpec,
        activation_source: &S,
        token_windows: &[(u64, u64)],
        max_decoded_bytes: u64,
        tensor_name: &str,
        teacher: &Projection,
        builder: &mut FixedTritScaleUpdateCandidateBuilder<'_>,
    ) -> Result<(), Qwen36ScaleRefitWindowError> {
        if token_windows.len() != usize::try_from(spec.batches_per_scope()).unwrap_or(usize::MAX) {
            return Err(Qwen36ScaleRefitWindowError::Fit(
                OutputReconstructionError::InvalidCount,
            ));
        }
        let mut previous_end = None;
        for &(token_start, token_count) in token_windows {
            let Some(token_end) = token_start.checked_add(token_count) else {
                return Err(Qwen36ScaleRefitWindowError::Fit(
                    OutputReconstructionError::InvalidGeometry,
                ));
            };
            if token_count == 0 || previous_end.is_some_and(|end| token_start < end) {
                return Err(Qwen36ScaleRefitWindowError::Fit(
                    OutputReconstructionError::InvalidGeometry,
                ));
            }
            previous_end = Some(token_end);
        }

        let layer_index = qwen_language_projection_layer(tensor_name)
            .ok_or(Qwen36ScaleRefitWindowError::InvalidTensorName)?;
        let mut observed_scope = false;
        for &scope in spec.scopes() {
            let OutputReconstructionScope::Block { start, end } = scope else {
                continue;
            };
            if layer_index < start || layer_index >= end {
                continue;
            }
            observed_scope = true;
            for &(token_start, token_count) in token_windows {
                self.observe_scale_refit_scope(
                    parent_execution,
                    spec,
                    activation_source,
                    scope,
                    token_start,
                    token_count,
                    max_decoded_bytes,
                    tensor_name,
                    teacher,
                    builder,
                )?;
            }
        }
        if !observed_scope {
            return Err(Qwen36ScaleRefitWindowError::Fit(
                OutputReconstructionError::InvalidActivationWindowScope,
            ));
        }
        Ok(())
    }

    /// Freshly replay and admit an immutable scale-refined child of this exact
    /// campaign package. The supplied scope batches are executed once for final
    /// logits and again for the frozen output-reconstruction scopes; both passes
    /// use the same ordered tokens.
    ///
    /// A structurally valid output receipt is not enough: this method reloads the
    /// child package, validates its parent lineage and physical ledgers, replays
    /// it on the sealed built-in backend, and binds the resulting transcript to
    /// the selected output receipt and current campaign admission.
    ///
    /// # Errors
    /// Fails closed if the campaign admission changes, the child is not descended
    /// from this session's exact package, package loading or runtime replay fails,
    /// or any output/candidate/transcript identity differs.
    pub fn replay_refined_candidate(
        &self,
        replay: Qwen36RefinedCandidateReplay<'_>,
    ) -> Result<Qwen36RefinedCandidateExecutionReceipt, Qwen36ExecutionVisitError<Infallible>> {
        self.admission
            .verify_current()
            .map_err(Qwen36ExecutionVisitError::Admission)?;
        output_binding::validate_execution(&self.authority, replay.parent_execution)
            .map_err(Qwen36ExecutionVisitError::Runtime)?;
        if replay.lineage.parent_package_id() != self.authority.package_id
            || replay.scale_candidate.parent_package_digest()
                != self.authority.package_id.as_bytes()
            || replay.scale_candidate.spec_id() != replay.spec.spec_id()
            || replay.spec.source_model_id() != self.authority.source_model_id
            || replay.spec.token_stream_digest() != replay.parent_execution.token_stream_digest()
            || replay.scope_batches.is_empty()
        {
            return Err(Qwen36ExecutionVisitError::Runtime(NnError::Provenance(
                "refined candidate does not match this admitted parent and output schedule"
                    .to_owned(),
            )));
        }

        let mut child_authority = self.authority.clone();
        child_authority.package_id = replay.lineage.child_package_id();
        let child_model = match child_authority.backend {
            Qwen36ExecutionBackend::Cpu => {
                Qwen35SaltV2LanguageMtpModel::load_bundle_scale_update_child(
                    replay.bundle_dir,
                    profile_name(child_authority.profile),
                    replay.child_package_path,
                    replay.lineage,
                    Box::new(tritium_cpu::CpuBackend::new()),
                )
            }
            #[cfg(feature = "cuda")]
            Qwen36ExecutionBackend::Cuda { ordinal } => {
                let ordinal = usize::try_from(ordinal).map_err(|_| {
                    Qwen36ExecutionVisitError::Runtime(NnError::Backend(
                        "CUDA ordinal exceeds usize".to_owned(),
                    ))
                })?;
                let backend = tritium_cuda::CudaBackend::new(ordinal)
                    .map_err(|error| Qwen36ExecutionVisitError::Runtime(NnError::from(error)))?;
                Qwen35SaltV2LanguageMtpModel::load_bundle_scale_update_child(
                    replay.bundle_dir,
                    profile_name(child_authority.profile),
                    replay.child_package_path,
                    replay.lineage,
                    Box::new(backend),
                )
            }
            #[cfg(not(feature = "cuda"))]
            Qwen36ExecutionBackend::Cuda { .. } => {
                return Err(Qwen36ExecutionVisitError::Runtime(NnError::Backend(
                    "CUDA refined replay requires the cuda feature".to_owned(),
                )));
            }
        }
        .map_err(Qwen36ExecutionVisitError::Runtime)?;

        self.replay_refined_candidate_with_model(replay, child_model)
    }

    fn replay_refined_candidate_with_model(
        &self,
        replay: Qwen36RefinedCandidateReplay<'_>,
        child_model: Qwen35SaltV2LanguageMtpModel,
    ) -> Result<Qwen36RefinedCandidateExecutionReceipt, Qwen36ExecutionVisitError<Infallible>> {
        self.admission
            .verify_current()
            .map_err(Qwen36ExecutionVisitError::Admission)?;
        output_binding::validate_execution(&self.authority, replay.parent_execution)
            .map_err(Qwen36ExecutionVisitError::Runtime)?;
        if replay.lineage.parent_package_id() != self.authority.package_id
            || replay.scale_candidate.parent_package_digest()
                != self.authority.package_id.as_bytes()
            || replay.scale_candidate.spec_id() != replay.spec.spec_id()
            || replay.spec.source_model_id() != self.authority.source_model_id
            || replay.spec.token_stream_digest() != replay.parent_execution.token_stream_digest()
            || replay.scope_batches.is_empty()
        {
            return Err(Qwen36ExecutionVisitError::Runtime(NnError::Provenance(
                "refined candidate does not match this admitted parent and output schedule"
                    .to_owned(),
            )));
        }
        let mut child_authority = self.authority.clone();
        child_authority.package_id = replay.lineage.child_package_id();
        validate_loaded_model(&child_authority, &child_model)
            .map_err(Qwen36ExecutionVisitError::Runtime)?;

        let scopes = output_runtime_scopes(replay.spec)?;
        let mut final_batches = Vec::new();
        final_batches
            .try_reserve_exact(replay.scope_batches.len())
            .map_err(|_| {
                Qwen36ExecutionVisitError::Runtime(NnError::ResourceExhausted(
                    "allocate refined replay batch references".to_owned(),
                ))
            })?;
        final_batches.extend(replay.scope_batches.iter().map(|(tokens, _)| *tokens));
        let transcript = child_model
            .try_visit_untrusted_final_logits(final_batches.iter().copied(), |_| {
                Ok::<_, Infallible>(())
            })
            .map_err(map_execution_error)?;
        validate_transcript(&child_authority, &transcript)
            .map_err(Qwen36ExecutionVisitError::Runtime)?;
        let scope_transcript = child_model
            .try_visit_untrusted_output_scopes(
                replay.spec.spec_id(),
                replay.scale_candidate.candidate_id(),
                replay.scale_candidate.initialization_seed(),
                &scopes,
                replay.scope_batches.iter().copied(),
            )
            .map_err(map_execution_error)?;

        self.admission
            .verify_current()
            .map_err(Qwen36ExecutionVisitError::Admission)?;
        Qwen36RefinedCandidateExecutionReceipt::from_child_replay(
            replay.parent_execution,
            replay.lineage,
            ChildReplayEvidence {
                spec: replay.spec,
                output_bytes: replay.output_bytes,
                scale_candidate: replay.scale_candidate,
                transcript: &transcript,
                scope_transcript: &scope_transcript,
                backend: child_authority.backend,
            },
        )
        .map_err(Qwen36ExecutionVisitError::Runtime)
    }

    #[cfg(test)]
    pub(crate) fn replay_refined_candidate_test_fixture(
        &self,
        replay: Qwen36RefinedCandidateReplay<'_>,
    ) -> Result<Qwen36RefinedCandidateExecutionReceipt, Qwen36ExecutionVisitError<Infallible>> {
        let child_model =
            Qwen35SaltV2LanguageMtpModel::load_bundle_scale_update_child_test_fixture(
                replay.bundle_dir,
                profile_name(self.authority.profile),
                replay.child_package_path,
                replay.lineage,
                Box::new(tritium_cpu::CpuBackend::new()),
            )
            .map_err(Qwen36ExecutionVisitError::Runtime)?;
        self.replay_refined_candidate_with_model(replay, child_model)
    }

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

fn output_runtime_scopes(
    spec: &OutputReconstructionSpec,
) -> Result<Vec<RuntimeOutputScope>, Qwen36ExecutionVisitError<Infallible>> {
    let mut scopes = Vec::new();
    scopes.try_reserve_exact(spec.scopes().len()).map_err(|_| {
        Qwen36ExecutionVisitError::Runtime(NnError::ResourceExhausted(
            "allocate refined output-scope schedule".to_owned(),
        ))
    })?;
    scopes.extend(spec.scopes().iter().map(|scope| match scope {
        OutputReconstructionScope::Block { start, end } => RuntimeOutputScope::Block {
            start: *start,
            end: *end,
        },
        OutputReconstructionScope::FinalLogits => RuntimeOutputScope::FinalLogits,
    }));
    Ok(scopes)
}

fn qwen_language_projection_layer(tensor_name: &str) -> Option<u32> {
    let suffix = tensor_name.strip_prefix("model.language_model.layers.")?;
    let (index, projection) = suffix.split_once('.')?;
    if projection.is_empty() {
        return None;
    }
    let parsed = index.parse::<u32>().ok()?;
    (parsed.to_string() == index).then_some(parsed)
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

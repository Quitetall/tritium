//! ADR 0044 D8-D10: one immutable tensor handle for projection and gather.

use std::sync::Arc;

use tritium_format::AdditiveTensor;
use tritium_runtime::{
    TensorPayloadBudget, TensorUploadContext, TensorUploadResources, upload_tensor_logged_admitted,
    upload_tensor_logged_with_resources,
};
use tritium_spec::{
    DeviceBuffer, TensorCaps, TensorMatmul, TensorUploadPlan, TensorUploadPolicy, TensorView,
    TernaryBackend,
};

use crate::NnError;

/// A shared admitted-upload context for model weights on any tensor backend.
///
/// Successful uploads append registered observations, not model-load or release
/// verdicts. Failure drops the attempted buffer/reservation. Earlier observations
/// remain valid observations, even when a later model tensor fails. The budget
/// covers owned payloads only, not source decode, norms, scratch, KV or drivers.
pub struct TensorLoader<'a> {
    backend: &'a dyn TernaryBackend,
    policy: TensorUploadPolicy,
    budget: &'a TensorPayloadBudget,
    context: TensorUploadContext<'a>,
    planned: Option<(
        TensorUploadResources<'a>,
        &'a std::collections::BTreeMap<String, TensorUploadPlan>,
    )>,
    stream_plans: Option<&'a std::collections::BTreeMap<String, tritium_spec::TensorStreamPlan>>,
}

impl core::fmt::Debug for TensorLoader<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorLoader")
            .field("backend", &self.backend.device_id())
            .field("policy", &self.policy)
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

impl<'a> TensorLoader<'a> {
    /// Bind one backend, explicit policy, shared budget and evidence span.
    pub fn new(
        backend: &'a dyn TernaryBackend,
        policy: TensorUploadPolicy,
        budget: &'a TensorPayloadBudget,
        context: TensorUploadContext<'a>,
    ) -> Self {
        Self {
            backend,
            policy,
            budget,
            context,
            planned: None,
            stream_plans: None,
        }
    }

    #[cfg(feature = "tokenizer")]
    pub(crate) fn planned(
        backend: &'a dyn TernaryBackend,
        policy: TensorUploadPolicy,
        resources: TensorUploadResources<'a>,
        plans: &'a std::collections::BTreeMap<String, TensorUploadPlan>,
        streams: Option<&'a std::collections::BTreeMap<String, tritium_spec::TensorStreamPlan>>,
        context: TensorUploadContext<'a>,
    ) -> Self {
        Self {
            backend,
            policy,
            budget: resources.payload,
            context,
            planned: Some((resources, plans)),
            stream_plans: streams,
        }
    }

    pub(crate) fn is_streaming(&self) -> bool {
        self.stream_plans.is_some()
    }

    #[cfg(feature = "tokenizer")]
    pub(crate) fn upload_stream(
        &mut self,
        name: &str,
        source: &dyn tritium_spec::TensorStreamSource,
    ) -> Result<Linear, NnError> {
        let expected = self
            .stream_plans
            .and_then(|p| p.get(name))
            .ok_or_else(|| NnError::Backend(format!("{name}: absent aggregate stream plan")))?;
        let actual = self
            .backend
            .tensor_stream_plan(source.metadata(), source.encoding())
            .map_err(|e| NnError::Backend(e.to_string()))?;
        if actual != Some(*expected) || source.encoded_bytes() != expected.encoded_bytes {
            return Err(NnError::Backend(format!(
                "{name}: stream plan changed after aggregate admission"
            )));
        }
        let (rows, cols) = match source.metadata() {
            tritium_spec::TensorMetadata::Dense { rows, cols } => (rows, cols),
            tritium_spec::TensorMetadata::Additive(a) => (
                usize::try_from(a.layout().rows)
                    .map_err(|_| NnError::Backend("rows exceed usize".into()))?,
                usize::try_from(a.layout().cols)
                    .map_err(|_| NnError::Backend("columns exceed usize".into()))?,
            ),
        };
        if rows == 0 || cols == 0 {
            return Err(NnError::Backend("empty streamed model geometry".into()));
        }
        let (resources, _) = self
            .planned
            .as_ref()
            .ok_or_else(|| NnError::Backend("stream resources absent".into()))?;
        let (dev, caps) = tritium_runtime::upload_tensor_stream_logged_with_resources(
            self.backend,
            source,
            self.policy,
            name,
            TensorUploadContext {
                recorder: self.context.recorder,
                span: self.context.span,
                parent: self.context.parent,
                logical_time: self.context.logical_time,
            },
            TensorUploadResources {
                payload: resources.payload,
                host_workspace: resources.host_workspace,
                device_workspace: resources.device_workspace,
            },
        )
        .map_err(|e| NnError::Backend(format!("{name}: {e}")))?;
        if caps != expected.upload.caps {
            return Err(NnError::Backend(format!("{name}: streamed caps changed")));
        }
        Ok(Linear {
            rows,
            cols,
            dev: Arc::from(dev),
            caps,
        })
    }

    pub(crate) fn check_evidence(&self) -> Result<(), NnError> {
        self.context
            .recorder
            .check_emit::<tritium_evidence::TensorUploaded>(
                self.context.span,
                self.context.parent,
                self.context.logical_time,
            )
            .map_err(|error| NnError::Backend(format!("tensor upload evidence: {error}")))
    }

    /// Validate finite model weights and upload without retaining a host shadow.
    ///
    /// The returned handle owns its reservation independently of this loader.
    /// Clones and embedding/tied-head consumers share one handle and lease.
    ///
    /// # Errors
    /// Rejects empty geometry, invalid scales/weights, unavailable capability,
    /// policy/budget/backend failure or unregistered/failed upload evidence.
    pub fn upload(&mut self, name: &str, view: TensorView<'_>) -> Result<Linear, NnError> {
        if self.is_streaming() {
            return Err(NnError::Backend(
                "mandatory streaming cannot use decoded upload fallback".into(),
            ));
        }
        let (rows, cols) = match view {
            TensorView::Dense { rows, cols, values } => {
                tritium_core::DenseView::new(rows, cols, values)
                    .map_err(|error| NnError::Backend(format!("{name}: {error:?}")))?;
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(NnError::Backend(format!(
                        "{name}: non-finite model weights"
                    )));
                }
                (rows, cols)
            }
            TensorView::Additive(view) => {
                AdditiveTensor::validate_view(view)
                    .map_err(|error| NnError::Backend(format!("{name}: {error:?}")))?;
                let layout = view.layout();
                (
                    usize::try_from(layout.rows)
                        .map_err(|_| NnError::Backend("tensor rows exceed usize".into()))?,
                    usize::try_from(layout.cols)
                        .map_err(|_| NnError::Backend("tensor columns exceed usize".into()))?,
                )
            }
        };
        if rows == 0 || cols == 0 {
            return Err(NnError::Backend(format!(
                "{name}: empty model weight geometry"
            )));
        }
        let context = TensorUploadContext {
            recorder: self.context.recorder,
            span: self.context.span,
            parent: self.context.parent,
            logical_time: self.context.logical_time,
        };
        let result = if let Some((resources, plans)) = &self.planned {
            let expected = plans
                .get(name)
                .ok_or_else(|| NnError::Backend(format!("{name}: absent aggregate upload plan")))?;
            let actual = self
                .backend
                .tensor_upload_plan(
                    tritium_spec::TensorMetadata::from_view(view)
                        .map_err(|e| NnError::Backend(e.to_string()))?,
                )
                .map_err(|e| NnError::Backend(e.to_string()))?;
            if actual != Some(*expected) {
                return Err(NnError::Backend(format!(
                    "{name}: upload plan changed after aggregate admission"
                )));
            }
            upload_tensor_logged_with_resources(
                self.backend,
                view,
                self.policy,
                name,
                context,
                TensorUploadResources {
                    payload: resources.payload,
                    host_workspace: resources.host_workspace,
                    device_workspace: resources.device_workspace,
                },
            )
        } else {
            upload_tensor_logged_admitted(
                self.backend,
                view,
                self.policy,
                name,
                context,
                self.budget,
            )
        };
        let (dev, caps) = result.map_err(|error| NnError::Backend(format!("{name}: {error}")))?;
        if let Some((_, plans)) = &self.planned
            && plans.get(name).is_none_or(|plan| plan.caps != caps)
        {
            return Err(NnError::Backend(format!(
                "{name}: actual caps differ from aggregate plan"
            )));
        }
        Ok(Linear {
            rows,
            cols,
            dev: Arc::from(dev),
            caps,
        })
    }
}

/// Backend-independent linear weight with private validated geometry and storage.
///
/// The scale law and basis remain inside the backend tensor. Cloning never copies
/// weights or acquires a second budget lease. No backend downcast or host fallback
/// occurs here. The supplied execution backend must accept the uploaded handle.
#[derive(Clone)]
pub struct Linear {
    rows: usize,
    cols: usize,
    dev: Arc<dyn DeviceBuffer>,
    caps: TensorCaps,
}

impl core::fmt::Debug for Linear {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Linear")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("caps", &self.caps)
            .finish_non_exhaustive()
    }
}

impl Linear {
    #[cfg(feature = "tokenizer")]
    pub(crate) fn device_arc(&self) -> &Arc<dyn DeviceBuffer> {
        &self.dev
    }
    pub(crate) fn device(&self) -> &dyn DeviceBuffer {
        self.dev.as_ref()
    }
    /// Number of output rows.
    #[must_use]
    pub const fn n_out(&self) -> usize {
        self.rows
    }

    /// Number of input columns.
    #[must_use]
    pub const fn k_in(&self) -> usize {
        self.cols
    }

    /// Checked upload tier and payload declaration; not a physical-memory verdict.
    #[must_use]
    pub const fn tensor_caps(&self) -> TensorCaps {
        self.caps
    }

    /// Owned payload bytes, excluding allocator/object/backend overhead.
    /// Summing clones double-counts a shared handle; the budget counts it once.
    #[must_use]
    pub fn payload_bytes(&self) -> usize {
        self.dev.len_bytes()
    }

    /// Execute with caller-owned basis scratch, without allocating in this adapter.
    ///
    /// # Errors
    /// Shape overflow/mismatch or backend ownership/execution failure.
    pub fn forward_with_scratch(
        &self,
        backend: &dyn TernaryBackend,
        act: &[f32],
        batch: usize,
        scratch: &mut [f32],
        out: &mut [f32],
    ) -> Result<(), NnError> {
        self.check_shape(act, batch, out)?;
        if scratch.len() != act.len() {
            return Err(NnError::Shape {
                expected: act.len(),
                got: scratch.len(),
            });
        }
        backend
            .matmul(TensorMatmul {
                act,
                tensor: self.dev.as_ref(),
                batch,
                transformed_act: scratch,
                out,
            })
            .map_err(Into::into)
    }

    /// Execute using bounded temporary basis scratch (not covered by the weight budget).
    /// Repeated-call executors should reuse scratch with [`Self::forward_with_scratch`].
    ///
    /// # Errors
    /// Shape, temporary allocation, backend ownership or execution failure.
    pub fn forward(
        &self,
        backend: &dyn TernaryBackend,
        act: &[f32],
        batch: usize,
        out: &mut [f32],
    ) -> Result<(), NnError> {
        self.check_shape(act, batch, out)?;
        let mut scratch = Vec::new();
        scratch.try_reserve_exact(act.len()).map_err(|error| {
            NnError::ResourceExhausted(format!(
                "tensor activation scratch {} f32 values: {error}",
                act.len()
            ))
        })?;
        scratch.resize(act.len(), 0.);
        self.forward_with_scratch(backend, act, batch, &mut scratch, out)
    }

    /// Gather original-coordinate rows, preserving order and duplicates.
    ///
    /// # Errors
    /// Rejects every invalid ID/shape before execution, or returns backend failure.
    pub fn gather(
        &self,
        backend: &dyn TernaryBackend,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), NnError> {
        let expected = ids
            .len()
            .checked_mul(self.cols)
            .ok_or_else(|| NnError::Backend("gather size overflows".into()))?;
        if out.len() != expected {
            return Err(NnError::Shape {
                expected,
                got: out.len(),
            });
        }
        if let Some(row) = ids.iter().find(|&&row| row >= self.rows) {
            return Err(NnError::MissingTensor(format!("embedding row {row}")));
        }
        backend
            .embed_rows(self.dev.as_ref(), ids, out)
            .map_err(Into::into)
    }

    fn check_shape(&self, act: &[f32], batch: usize, out: &[f32]) -> Result<(), NnError> {
        let inputs = batch
            .checked_mul(self.cols)
            .ok_or_else(|| NnError::Backend("tensor input size overflows".into()))?;
        let outputs = batch
            .checked_mul(self.rows)
            .ok_or_else(|| NnError::Backend("tensor output size overflows".into()))?;
        if act.len() != inputs {
            return Err(NnError::Shape {
                expected: inputs,
                got: act.len(),
            });
        }
        if out.len() != outputs {
            return Err(NnError::Shape {
                expected: outputs,
                got: out.len(),
            });
        }
        Ok(())
    }
}

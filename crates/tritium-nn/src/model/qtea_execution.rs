//! Explicit loading target; resource domains are logical, not physical admission.
use super::qtea::HostQteaModelLimits;

/// Logical limits for streaming packed weights into host-orchestrated CUDA.
/// Metadata, vector decode byte buffers, runtime activations/caches, allocator/
/// driver/module overhead and other models are excluded. No dense matrix shadow.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub struct CudaQteaModelLimits {
    /// Source/directory limits, per-owner admission and host load/staging caps.
    /// `max_weight_bytes` here bounds retained preserved Vec capacities plus the
    /// one current encoded upload Vec, NOT the aggregate device matrix storage.
    pub host: HostQteaModelLimits,
    /// Aggregate unique canonical encoded device weight bytes, including MTP.
    pub max_device_weight_bytes: u64,
    /// Per-call input/index plus output device staging; not concurrent aggregate.
    pub max_device_staging_bytes: u64,
}

#[derive(Clone, Copy)]
pub(super) enum Execution {
    Host(HostQteaModelLimits),
    #[cfg(feature = "cuda")]
    Cuda(CudaQteaModelLimits),
}
impl Execution {
    pub(super) fn host(self) -> HostQteaModelLimits {
        match self {
            Self::Host(limits) => limits,
            #[cfg(feature = "cuda")]
            Self::Cuda(limits) => limits.host,
        }
    }
    #[cfg(feature = "tokenizer")]
    pub(super) fn is_cuda(self) -> bool {
        match self {
            Self::Host(_) => false,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => true,
        }
    }
    #[cfg(feature = "tokenizer")]
    pub(super) fn label(
        self,
        #[allow(unused_variables)] backend: &dyn tritium_spec::TernaryBackend,
    ) -> &'static str {
        if !self.is_cuda() {
            return "qtea-scalar-host";
        }
        #[cfg(feature = "cuda")]
        if backend
            .as_concrete()
            .and_then(|b| b.downcast_ref::<tritium_cuda::CudaBackend>())
            .is_some_and(|b| b.device_swiglu_enabled())
        {
            if backend
                .as_concrete()
                .and_then(|b| b.downcast_ref::<tritium_cuda::CudaBackend>())
                .is_some_and(|b| b.swiglu_workspace_admission_enabled())
            {
                return "qtea-cuda-ordered-device-swiglu-reuse-host-orchestrated";
            }
            return "qtea-cuda-ordered-device-swiglu-host-orchestrated";
        }
        "qtea-cuda-ordered-host-orchestrated"
    }
}

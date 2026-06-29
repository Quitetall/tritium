//! Device capability description used by the runtime to choose a backend.

/// What a backend device can do. The runtime reads this to pick a backend for a
/// given problem; later milestones grow the fields (fp8, IMMA, tensor cores).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceCaps {
    /// Backend family, e.g. `"cpu"` or `"cuda"`.
    pub backend: String,
    /// Human-readable device/arch name, e.g. `"x86_64 (avx2)"` or `"NVIDIA RTX 4090"`.
    pub device_name: String,
    /// Detected SIMD / ISA feature flags, e.g. `["avx2"]` or `["sm_89"]`.
    pub features: Vec<String>,
    /// Total device memory in bytes (host RAM for CPU), `0` if unknown.
    pub total_memory_bytes: u64,
    /// Whether an int8 tensor-core (IMMA) path is available (false until 0.30).
    pub supports_imma: bool,
    /// Whether an fp8 path is available (false until later milestones).
    pub supports_fp8: bool,
}

impl DeviceCaps {
    /// Construct caps for a backend with the given family and device name.
    /// Optional capability flags default off and are set with the builder methods.
    pub fn new(backend: impl Into<String>, device_name: impl Into<String>) -> Self {
        Self {
            backend: backend.into(),
            device_name: device_name.into(),
            features: Vec::new(),
            total_memory_bytes: 0,
            supports_imma: false,
            supports_fp8: false,
        }
    }

    /// Set the detected feature flags.
    #[must_use]
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.features = features;
        self
    }

    /// Set total device memory in bytes.
    #[must_use]
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.total_memory_bytes = bytes;
        self
    }

    /// Set whether an int8 tensor-core (IMMA) path is available.
    #[must_use]
    pub fn with_imma(mut self, supported: bool) -> Self {
        self.supports_imma = supported;
        self
    }

    /// Set whether an fp8 path is available.
    #[must_use]
    pub fn with_fp8(mut self, supported: bool) -> Self {
        self.supports_fp8 = supported;
        self
    }

    /// True if the named feature flag is present.
    pub fn has_feature(&self, name: &str) -> bool {
        self.features.iter().any(|f| f == name)
    }
}

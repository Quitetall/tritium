//! Fail-closed architecture capability and tensor-role contracts.
//!
//! Additive PTQ can pack many matrix shapes without understanding a model, but a
//! product converter must understand every tensor and every execution-semantic
//! feature. These types keep architecture discovery separate from capability
//! negotiation and from the conservative default quantization policy.

use core::fmt;

/// Execution or layout feature an architecture adapter may require.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ArchitectureFeature {
    /// Conventional full causal attention.
    FullAttention,
    /// Local or sliding-window attention layers.
    SlidingWindowAttention,
    /// Recurrent or gated linear-attention layers.
    LinearAttention,
    /// Low-rank latent attention or a learned sparse-attention indexer.
    LatentAttention,
    /// Per-head query/key normalization.
    QkNorm,
    /// Learned attention-output gating.
    AttentionOutputGate,
    /// Routed mixture-of-experts feed-forward layers.
    MixtureOfExperts,
    /// A dense shared expert alongside routed experts.
    SharedExpert,
    /// Key/value projection weights shared within or across layers.
    SharedKvProjection,
    /// Token embeddings injected separately at decoder layers.
    PerLayerEmbeddings,
    /// Attention head counts or dimensions that vary by layer type.
    PerLayerAttentionGeometry,
    /// Architecture-specific RoPE schedules beyond one global theta.
    CustomRope,
    /// Non-text towers or projectors packaged with the text model.
    Multimodal,
}

/// Canonical feature requirements reported by one architecture adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArchitectureRequirements {
    architecture: String,
    features: Vec<ArchitectureFeature>,
}

impl ArchitectureRequirements {
    /// Build requirements, sorting and deduplicating feature declarations.
    ///
    /// # Errors
    /// Returns [`AdapterError::EmptyArchitecture`] for an empty identifier.
    pub fn new(
        architecture: impl Into<String>,
        features: impl IntoIterator<Item = ArchitectureFeature>,
    ) -> Result<Self, AdapterError> {
        let architecture = architecture.into();
        if architecture.is_empty() {
            return Err(AdapterError::EmptyArchitecture);
        }
        let mut features: Vec<_> = features.into_iter().collect();
        features.sort_unstable();
        features.dedup();
        Ok(Self {
            architecture,
            features,
        })
    }

    /// Architecture identifier supplied by the adapter.
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Sorted, unique required features.
    pub fn features(&self) -> &[ArchitectureFeature] {
        &self.features
    }
}

/// Features supported by a converter, evaluator, or runtime implementation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CapabilitySet {
    features: Vec<ArchitectureFeature>,
}

impl CapabilitySet {
    /// Build a sorted, deduplicated capability set.
    pub fn new(features: impl IntoIterator<Item = ArchitectureFeature>) -> Self {
        let mut features: Vec<_> = features.into_iter().collect();
        features.sort_unstable();
        features.dedup();
        Self { features }
    }

    /// Sorted, unique supported features.
    pub fn features(&self) -> &[ArchitectureFeature] {
        &self.features
    }

    /// Whether this set includes a feature.
    pub fn supports(&self, feature: ArchitectureFeature) -> bool {
        self.features.binary_search(&feature).is_ok()
    }

    /// Require every architecture feature, returning the complete deterministic gap.
    ///
    /// # Errors
    /// Returns [`CapabilityGap`] when one or more requirements are unsupported.
    pub fn validate(&self, requirements: &ArchitectureRequirements) -> Result<(), CapabilityGap> {
        let missing: Vec<_> = requirements
            .features
            .iter()
            .copied()
            .filter(|feature| !self.supports(*feature))
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(CapabilityGap {
                architecture: requirements.architecture.clone(),
                missing,
            })
        }
    }
}

/// Complete set of unsupported requirements for an architecture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityGap {
    architecture: String,
    missing: Vec<ArchitectureFeature>,
}

impl CapabilityGap {
    /// Architecture that could not be handled.
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Sorted, unique unsupported features.
    pub fn missing(&self) -> &[ArchitectureFeature] {
        &self.missing
    }
}

impl fmt::Display for CapabilityGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "architecture `{}` requires unsupported features: {:?}",
            self.architecture, self.missing
        )
    }
}

impl std::error::Error for CapabilityGap {}

/// Borrowed tensor metadata presented to an [`ArchitectureAdapter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorDescriptor<'a> {
    name: &'a str,
    shape: &'a [u64],
}

impl<'a> TensorDescriptor<'a> {
    /// Validate borrowed tensor metadata.
    ///
    /// # Errors
    /// Returns [`AdapterError`] for an empty name, empty shape, or zero dimension.
    pub fn new(name: &'a str, shape: &'a [u64]) -> Result<Self, AdapterError> {
        if name.is_empty() {
            return Err(AdapterError::EmptyTensorName);
        }
        if shape.is_empty() {
            return Err(AdapterError::EmptyTensorShape(name.to_owned()));
        }
        if let Some(dimension) = shape.iter().position(|&size| size == 0) {
            return Err(AdapterError::ZeroTensorDimension {
                tensor: name.to_owned(),
                dimension,
            });
        }
        Ok(Self { name, shape })
    }

    /// Canonical source tensor name.
    pub fn name(&self) -> &'a str {
        self.name
    }

    /// Logical source tensor shape.
    pub fn shape(&self) -> &'a [u64] {
        self.shape
    }
}

/// Semantic role assigned to a tensor by an architecture adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TensorRole {
    /// Dense attention, MLP, or linear-attention matrix projection.
    Projection,
    /// Routed or shared-expert matrix projection.
    ExpertProjection,
    /// MoE routing or gating tensor.
    Router,
    /// Token or per-layer embedding table.
    Embedding,
    /// Normalization scale or bias.
    Normalization,
    /// Additive bias outside normalization.
    Bias,
    /// Learned positional or RoPE parameter.
    PositionalEncoding,
    /// Recurrent, convolutional, or state-space state parameter.
    StateParameter,
    /// Convolution kernel not represented as a matrix projection.
    Convolution,
    /// Untied language-model output head.
    OutputHead,
}

/// Conservative default action for a recognized tensor role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TensorDisposition {
    /// Run sensitivity-aware additive ternary quantization.
    AdditiveTernary,
    /// Copy the source tensor without lossy conversion.
    PreserveSource,
}

impl TensorDisposition {
    /// Choose the fail-safe default for a recognized role.
    ///
    /// Only explicit matrix projection roles are ternarized. Routers, norms,
    /// embeddings, biases, state, convolution, positional tensors, and output
    /// heads remain at source precision until a recipe opts them in separately.
    /// The match is intentionally exhaustive so adding a role requires an
    /// explicit safety decision here.
    pub fn for_role(role: TensorRole) -> Self {
        match role {
            TensorRole::Projection | TensorRole::ExpertProjection => Self::AdditiveTernary,
            TensorRole::Router
            | TensorRole::Embedding
            | TensorRole::Normalization
            | TensorRole::Bias
            | TensorRole::PositionalEncoding
            | TensorRole::StateParameter
            | TensorRole::Convolution
            | TensorRole::OutputHead => Self::PreserveSource,
        }
    }
}

/// Fail-closed model-family adapter used during conversion ingest.
pub trait ArchitectureAdapter {
    /// Describe every execution-semantic feature the source model requires.
    fn requirements(&self) -> ArchitectureRequirements;

    /// Assign a known semantic role to a source tensor.
    ///
    /// # Errors
    /// Must return [`AdapterError::UnrecognizedTensor`] rather than guessing when
    /// the model contains an unknown name or layout.
    fn classify_tensor(&self, tensor: TensorDescriptor<'_>) -> Result<TensorRole, AdapterError>;
}

/// Why architecture discovery or tensor classification failed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdapterError {
    /// Architecture identifier was empty.
    EmptyArchitecture,
    /// Tensor name was empty.
    EmptyTensorName,
    /// Tensor shape was empty.
    EmptyTensorShape(String),
    /// Tensor contained a zero-sized dimension.
    ZeroTensorDimension {
        /// Tensor containing the zero dimension.
        tensor: String,
        /// Zero-based dimension index.
        dimension: usize,
    },
    /// Adapter did not recognize a tensor and refused to guess.
    UnrecognizedTensor(String),
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyArchitecture => f.write_str("architecture identifier is empty"),
            Self::EmptyTensorName => f.write_str("tensor name is empty"),
            Self::EmptyTensorShape(name) => write!(f, "tensor `{name}` has no dimensions"),
            Self::ZeroTensorDimension { tensor, dimension } => {
                write!(f, "tensor `{tensor}` has zero-sized dimension {dimension}")
            }
            Self::UnrecognizedTensor(name) => {
                write!(f, "architecture adapter does not recognize tensor `{name}`")
            }
        }
    }
}

impl std::error::Error for AdapterError {}

#[cfg(test)]
mod tests {
    use super::*;

    struct QwenMoeAdapter;

    impl ArchitectureAdapter for QwenMoeAdapter {
        fn requirements(&self) -> ArchitectureRequirements {
            ArchitectureRequirements::new(
                "qwen3.5-moe",
                [
                    ArchitectureFeature::FullAttention,
                    ArchitectureFeature::LinearAttention,
                    ArchitectureFeature::QkNorm,
                    ArchitectureFeature::AttentionOutputGate,
                    ArchitectureFeature::MixtureOfExperts,
                    ArchitectureFeature::SharedExpert,
                ],
            )
            .expect("valid requirements")
        }

        fn classify_tensor(
            &self,
            tensor: TensorDescriptor<'_>,
        ) -> Result<TensorRole, AdapterError> {
            if tensor.name().contains("experts") {
                Ok(TensorRole::ExpertProjection)
            } else if tensor.name().ends_with("mlp.gate.weight") {
                Ok(TensorRole::Router)
            } else if tensor.name().contains("norm") {
                Ok(TensorRole::Normalization)
            } else if tensor.name().ends_with("proj.weight") {
                Ok(TensorRole::Projection)
            } else {
                Err(AdapterError::UnrecognizedTensor(tensor.name().to_owned()))
            }
        }
    }

    #[test]
    fn missing_capabilities_are_complete_sorted_and_fail_closed() {
        let adapter = QwenMoeAdapter;
        let available = CapabilitySet::new([
            ArchitectureFeature::FullAttention,
            ArchitectureFeature::QkNorm,
        ]);
        let gap = available
            .validate(&adapter.requirements())
            .expect_err("partial support must fail");

        assert_eq!(gap.architecture(), "qwen3.5-moe");
        assert_eq!(
            gap.missing(),
            &[
                ArchitectureFeature::LinearAttention,
                ArchitectureFeature::AttentionOutputGate,
                ArchitectureFeature::MixtureOfExperts,
                ArchitectureFeature::SharedExpert,
            ]
        );
    }

    #[test]
    fn tensor_policy_quantizes_only_explicit_matrix_roles() {
        assert_eq!(
            TensorDisposition::for_role(TensorRole::Projection),
            TensorDisposition::AdditiveTernary
        );
        assert_eq!(
            TensorDisposition::for_role(TensorRole::ExpertProjection),
            TensorDisposition::AdditiveTernary
        );
        assert_eq!(
            TensorDisposition::for_role(TensorRole::Router),
            TensorDisposition::PreserveSource
        );
        assert_eq!(
            TensorDisposition::for_role(TensorRole::Normalization),
            TensorDisposition::PreserveSource
        );
    }

    #[test]
    fn adapter_rejects_unknown_tensor_instead_of_guessing() {
        let adapter = QwenMoeAdapter;
        let unknown = TensorDescriptor::new("model.layers.0.mystery.weight", &[32, 32])
            .expect("valid descriptor");
        assert!(matches!(
            adapter.classify_tensor(unknown),
            Err(AdapterError::UnrecognizedTensor(_))
        ));
    }
}

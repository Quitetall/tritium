//! Fallible backend seam for portable whole-Tape training.

use core::fmt;

use crate::{TrainingOpCategoryV1, TrainingOpManifestV1, TrainingVjpV1};

/// Portable tensor/storage dtype.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainDTypeV1 {
    /// IEEE-754 binary32 tensor data; mandatory on every v1 backend.
    F32,
    /// Unsigned 32-bit index data.
    U32,
    /// Opaque canonical bytes for checkpoint/artifact state.
    Bytes,
}

impl TrainDTypeV1 {
    const fn element_bytes(self) -> u64 {
        match self {
            Self::F32 | Self::U32 => 4,
            Self::Bytes => 1,
        }
    }
}

/// Immutable named request-buffer payload.
#[derive(Clone, Copy, Debug)]
pub enum TrainBufferDataRefV1<'a> {
    /// F32 tensor elements.
    F32(&'a [f32]),
    /// U32 tensor/index elements.
    U32(&'a [u32]),
    /// Opaque canonical bytes.
    Bytes(&'a [u8]),
}

impl TrainBufferDataRefV1<'_> {
    const fn len(&self) -> usize {
        match self {
            Self::F32(data) => data.len(),
            Self::U32(data) => data.len(),
            Self::Bytes(data) => data.len(),
        }
    }

    const fn dtype(&self) -> TrainDTypeV1 {
        match self {
            Self::F32(_) => TrainDTypeV1::F32,
            Self::U32(_) => TrainDTypeV1::U32,
            Self::Bytes(_) => TrainDTypeV1::Bytes,
        }
    }
}

/// Mutable named output-buffer payload.
#[derive(Debug)]
pub enum TrainBufferDataMutV1<'a> {
    /// F32 tensor elements.
    F32(&'a mut [f32]),
    /// U32 tensor/index elements.
    U32(&'a mut [u32]),
    /// Opaque canonical bytes.
    Bytes(&'a mut [u8]),
}

impl TrainBufferDataMutV1<'_> {
    const fn len(&self) -> usize {
        match self {
            Self::F32(data) => data.len(),
            Self::U32(data) => data.len(),
            Self::Bytes(data) => data.len(),
        }
    }

    const fn dtype(&self) -> TrainDTypeV1 {
        match self {
            Self::F32(_) => TrainDTypeV1::F32,
            Self::U32(_) => TrainDTypeV1::U32,
            Self::Bytes(_) => TrainDTypeV1::Bytes,
        }
    }
}

/// Owned payload used by compilers, fixture loaders and FFI adapters.
#[derive(Clone, Debug, PartialEq)]
pub enum TrainOwnedBufferDataV1 {
    /// Owned f32 elements.
    F32(Vec<f32>),
    /// Owned u32 elements.
    U32(Vec<u32>),
    /// Owned canonical bytes.
    Bytes(Vec<u8>),
}

/// Owned counterpart of borrowed named request/output buffers.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainOwnedBufferV1 {
    /// Stable operation-local role name.
    pub name: String,
    /// Row-major language-neutral dimensions.
    pub shape: Vec<u64>,
    /// Owned typed payload.
    pub data: TrainOwnedBufferDataV1,
}

impl TrainOwnedBufferV1 {
    /// Borrow as an immutable request buffer without copying payload data.
    #[must_use]
    pub fn as_ref(&self) -> TrainNamedBufferRefV1<'_> {
        let data = match &self.data {
            TrainOwnedBufferDataV1::F32(data) => TrainBufferDataRefV1::F32(data),
            TrainOwnedBufferDataV1::U32(data) => TrainBufferDataRefV1::U32(data),
            TrainOwnedBufferDataV1::Bytes(data) => TrainBufferDataRefV1::Bytes(data),
        };
        TrainNamedBufferRefV1::new(&self.name, &self.shape, data)
    }

    /// Borrow as a mutable output buffer without copying payload data.
    ///
    /// Rust's exclusive borrow enforces v1 aliasing policy: one owned buffer
    /// cannot simultaneously back immutable input and mutable output views in
    /// safe code. FFI adapters must reproduce this check before construction.
    #[must_use]
    pub fn as_mut(&mut self) -> TrainNamedBufferMutV1<'_> {
        let data = match &mut self.data {
            TrainOwnedBufferDataV1::F32(data) => TrainBufferDataMutV1::F32(data),
            TrainOwnedBufferDataV1::U32(data) => TrainBufferDataMutV1::U32(data),
            TrainOwnedBufferDataV1::Bytes(data) => TrainBufferDataMutV1::Bytes(data),
        };
        TrainNamedBufferMutV1::new(&self.name, &self.shape, data)
    }
}

/// Immutable named input/state buffer with language-neutral u64 shape.
#[derive(Clone, Copy, Debug)]
pub struct TrainNamedBufferRefV1<'a> {
    /// Stable operation-local role name.
    pub name: &'a str,
    /// Row-major dimensions; empty means scalar.
    pub shape: &'a [u64],
    /// Typed payload.
    pub data: TrainBufferDataRefV1<'a>,
}

impl<'a> TrainNamedBufferRefV1<'a> {
    /// Construct one borrowed request buffer.
    #[must_use]
    pub const fn new(name: &'a str, shape: &'a [u64], data: TrainBufferDataRefV1<'a>) -> Self {
        Self { name, shape, data }
    }
}

/// Mutable named output/state buffer with language-neutral u64 shape.
#[derive(Debug)]
pub struct TrainNamedBufferMutV1<'a> {
    /// Stable operation-local role name.
    pub name: &'a str,
    /// Row-major dimensions; empty means scalar.
    pub shape: &'a [u64],
    /// Typed writable payload.
    pub data: TrainBufferDataMutV1<'a>,
}

impl<'a> TrainNamedBufferMutV1<'a> {
    /// Construct one borrowed output buffer.
    #[must_use]
    pub const fn new(name: &'a str, shape: &'a [u64], data: TrainBufferDataMutV1<'a>) -> Self {
        Self { name, shape, data }
    }
}

/// Typed operation attribute.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrainAttributeValueV1<'a> {
    /// Finite f32 scalar.
    F32(f32),
    /// Unsigned integer scalar.
    U64(u64),
    /// Boolean flag.
    Bool(bool),
    /// UTF-8 identifier/text value.
    Text(&'a str),
    /// Unsigned integer list, used for shapes/strides/padding.
    U64List(&'a [u64]),
    /// Unsigned index list.
    U32List(&'a [u32]),
}

/// One named operation attribute.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainAttributeV1<'a> {
    /// Stable operation-local attribute name.
    pub name: &'a str,
    /// Typed value.
    pub value: TrainAttributeValueV1<'a>,
}

impl<'a> TrainAttributeV1<'a> {
    /// Construct one attribute.
    #[must_use]
    pub const fn new(name: &'a str, value: TrainAttributeValueV1<'a>) -> Self {
        Self { name, value }
    }
}

/// Requested semantic phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainExecutionV1 {
    /// Forward tensor evaluation.
    Forward,
    /// First-order vector-Jacobian product.
    Vjp,
    /// In-place optimizer update.
    Step,
    /// Serialize training state.
    Checkpoint,
    /// Restore training state.
    Resume,
    /// Export canonical hard artifact.
    Export,
    /// Reload canonical hard artifact.
    Reload,
}

/// Backend-specific validation ceilings for one request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrainLimitsV1 {
    /// Maximum admitted rank.
    pub max_rank: u32,
    /// Maximum admitted elements in any one buffer.
    pub max_elements: u64,
    /// Maximum admitted payload bytes in any one buffer.
    pub max_bytes: u64,
}

impl TrainLimitsV1 {
    /// Language-level maximum used by generic validation before adapter limits.
    pub const UNBOUNDED: Self = Self {
        max_rank: u32::MAX,
        max_elements: u64::MAX,
        max_bytes: u64::MAX,
    };
}

/// Borrowed, validated-at-execution portable training request.
#[derive(Clone, Copy, Debug)]
pub struct TrainRequestV1<'a> {
    /// Permanent manifest operation ID.
    pub operation: &'a str,
    /// Requested semantic phase.
    pub execution: TrainExecutionV1,
    /// Named input and pre-mutation state buffers.
    pub inputs: &'a [TrainNamedBufferRefV1<'a>],
    /// Named typed attributes.
    pub attributes: &'a [TrainAttributeV1<'a>],
    /// Exact semantic-vector corpus digest when execution is conformance-bound.
    pub vector_digest: Option<[u8; 32]>,
}

impl<'a> TrainRequestV1<'a> {
    /// Construct one borrowed request.
    #[must_use]
    pub const fn new(
        operation: &'a str,
        execution: TrainExecutionV1,
        inputs: &'a [TrainNamedBufferRefV1<'a>],
        attributes: &'a [TrainAttributeV1<'a>],
    ) -> Self {
        Self {
            operation,
            execution,
            inputs,
            attributes,
            vector_digest: None,
        }
    }

    /// Bind this execution to the exact semantic-vector corpus driving it.
    #[must_use]
    pub const fn with_vector_digest(mut self, digest: [u8; 32]) -> Self {
        self.vector_digest = Some(digest);
        self
    }

    /// Validate manifest membership, phase, names, shapes and buffer lengths.
    ///
    /// Adapters call this before reading or mutating caller buffers. Individual
    /// operation adapters then validate their exact roles and attribute schema.
    ///
    /// # Errors
    /// Returns [`TrainRequestError`] for every malformed contract; this method
    /// never indexes tensor data or allocates from untrusted shape products.
    pub fn validate(&self, output: &TrainOutputV1<'_>) -> Result<(), TrainRequestError> {
        self.validate_with_limits(output, TrainLimitsV1::UNBOUNDED)
    }

    /// Validate against explicit backend rank, element and byte ceilings.
    ///
    /// # Errors
    /// Returns [`TrainRequestError`] before adapter dispatch or mutation.
    pub fn validate_with_limits(
        &self,
        output: &TrainOutputV1<'_>,
        limits: TrainLimitsV1,
    ) -> Result<(), TrainRequestError> {
        let descriptor = TrainingOpManifestV1::operations()
            .iter()
            .find(|descriptor| descriptor.id == self.operation)
            .ok_or_else(|| TrainRequestError::UnknownOperation(self.operation.to_owned()))?;
        if !execution_allowed(
            descriptor.id,
            descriptor.category,
            descriptor.forward,
            descriptor.vjp,
            self.execution,
        ) {
            return Err(TrainRequestError::IllegalExecution {
                operation: self.operation.to_owned(),
                execution: self.execution,
            });
        }
        validate_ref_buffers(self.inputs, limits)?;
        validate_mut_buffers(output.buffers, limits)?;
        validate_attributes(self.attributes)?;
        Ok(())
    }
}

/// Writable buffers supplied by caller for one execution.
#[derive(Debug)]
pub struct TrainOutputV1<'a> {
    /// Named forward/VJP/state/artifact outputs.
    pub buffers: &'a mut [TrainNamedBufferMutV1<'a>],
}

impl<'a> TrainOutputV1<'a> {
    /// Construct an output view.
    #[must_use]
    pub const fn new(buffers: &'a mut [TrainNamedBufferMutV1<'a>]) -> Self {
        Self { buffers }
    }
}

/// Why a portable request failed validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrainRequestError {
    /// Operation ID is not present in the frozen manifest.
    UnknownOperation(String),
    /// Operation does not implement the requested phase.
    IllegalExecution {
        /// Permanent operation ID.
        operation: String,
        /// Rejected phase.
        execution: TrainExecutionV1,
    },
    /// Role/attribute name is empty or not portable lowercase ASCII.
    InvalidName {
        /// Input, output or attribute namespace.
        namespace: &'static str,
        /// Rejected name.
        name: String,
    },
    /// Role/attribute name appears twice in one namespace.
    DuplicateName {
        /// Input, output or attribute namespace.
        namespace: &'static str,
        /// Duplicated name.
        name: String,
    },
    /// Shape product does not fit host `usize`.
    ShapeOverflow {
        /// Buffer role.
        name: String,
    },
    /// Buffer rank exceeds backend ceiling.
    RankLimit {
        /// Buffer role.
        name: String,
        /// Observed rank.
        got: usize,
        /// Backend ceiling.
        max: u32,
    },
    /// Shape element count exceeds backend ceiling.
    ElementLimit {
        /// Buffer role.
        name: String,
        /// Shape-derived elements.
        got: u64,
        /// Backend ceiling.
        max: u64,
    },
    /// Shape-derived payload byte count overflows u64.
    ByteCountOverflow {
        /// Buffer role.
        name: String,
    },
    /// Payload byte count exceeds backend ceiling.
    ByteLimit {
        /// Buffer role.
        name: String,
        /// Shape-derived bytes.
        got: u64,
        /// Backend ceiling.
        max: u64,
    },
    /// Shape element count differs from payload length.
    BufferLength {
        /// Buffer role.
        name: String,
        /// Shape-derived element count.
        expected: usize,
        /// Payload element count.
        got: usize,
    },
    /// F32 attribute is NaN or infinite.
    NonFiniteAttribute(String),
}

impl fmt::Display for TrainRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownOperation(operation) => {
                write!(f, "unknown training operation {operation:?}")
            }
            Self::IllegalExecution {
                operation,
                execution,
            } => write!(
                f,
                "training operation {operation:?} does not support {execution:?}"
            ),
            Self::InvalidName { namespace, name } => {
                write!(f, "invalid {namespace} name {name:?}")
            }
            Self::DuplicateName { namespace, name } => {
                write!(f, "duplicate {namespace} name {name:?}")
            }
            Self::ShapeOverflow { name } => write!(f, "shape product overflows for {name:?}"),
            Self::RankLimit { name, got, max } => {
                write!(f, "buffer {name:?} rank {got} exceeds limit {max}")
            }
            Self::ElementLimit { name, got, max } => {
                write!(f, "buffer {name:?} has {got} elements, limit is {max}")
            }
            Self::ByteCountOverflow { name } => {
                write!(f, "payload byte count overflows for {name:?}")
            }
            Self::ByteLimit { name, got, max } => {
                write!(f, "buffer {name:?} has {got} bytes, limit is {max}")
            }
            Self::BufferLength {
                name,
                expected,
                got,
            } => write!(
                f,
                "buffer {name:?} has {got} elements, shape requires {expected}"
            ),
            Self::NonFiniteAttribute(name) => write!(f, "attribute {name:?} must be finite"),
        }
    }
}

impl std::error::Error for TrainRequestError {}

/// Backend capability declaration bound to one manifest digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainCapabilitiesV1 {
    /// Stable physical/backend identity.
    pub backend_id: String,
    /// Exact [`TrainingOpManifestV1`] digest.
    pub manifest_digest: [u8; 32],
    /// Manifest IDs this adapter executes without fallback.
    pub supported_operations: Vec<String>,
    /// Supported storage dtypes; f32 is mandatory for conforming v1 backends.
    pub dtypes: Vec<TrainDTypeV1>,
    /// Rank, element and byte ceilings enforced before dispatch.
    pub limits: TrainLimitsV1,
    /// True only when steady-state execution remains on declared device.
    pub device_resident: bool,
}

/// Content-bound receipt for one backend execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainReceiptV1 {
    /// Stable backend adapter identity.
    pub backend_id: String,
    /// Version/build identity of adapter implementation.
    pub backend_build: String,
    /// Physical device identity when receipted; `None` is never release-admissible.
    pub physical_device: Option<String>,
    /// Exact manifest digest used for dispatch.
    pub manifest_digest: [u8; 32],
    /// Canonical semantic-vector corpus digest, when execution is vector-bound.
    pub vector_digest: Option<[u8; 32]>,
    /// Executed permanent operation ID.
    pub operation: String,
    /// Executed semantic phase.
    pub execution: TrainExecutionV1,
    /// Storage dtype used by primary numeric tensors.
    pub dtype: TrainDTypeV1,
    /// Enforced backend ceilings.
    pub limits: TrainLimitsV1,
    /// Digest of canonical request buffers/attributes.
    pub input_digest: [u8; 32],
    /// Digest of canonical output buffers.
    pub output_digest: [u8; 32],
    /// Peak caller-visible resident tensor bytes.
    pub peak_resident_bytes: u64,
    /// Peak temporary tensor scratch bytes.
    pub scratch_bytes: u64,
    /// Tensor host-transfer count after setup.
    pub host_transfers: u64,
    /// Whether tensor execution remained on declared device.
    pub device_resident: bool,
}

/// Structured operation-schema failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrainOperationErrorV1 {
    /// Named input, attribute or output roles differ from frozen operation schema.
    Roles {
        /// Input, attribute or output namespace.
        namespace: &'static str,
    },
    /// Named buffer has wrong dtype.
    DType {
        /// Buffer role.
        name: String,
        /// Required dtype.
        expected: TrainDTypeV1,
        /// Supplied dtype.
        got: TrainDTypeV1,
    },
    /// Operation-specific shapes disagree.
    Shape,
    /// Numeric buffer contains NaN or infinity where forbidden.
    NonFinite {
        /// Buffer role.
        name: String,
    },
    /// Named attribute has wrong typed variant.
    AttributeType {
        /// Attribute role.
        name: String,
        /// Required type name.
        expected: &'static str,
    },
    /// Named attribute value violates operation constraints.
    AttributeValue {
        /// Attribute role.
        name: String,
        /// Stable constraint code.
        constraint: &'static str,
    },
}

impl fmt::Display for TrainOperationErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Roles { namespace } => {
                write!(f, "{namespace} roles differ from operation schema")
            }
            Self::DType {
                name,
                expected,
                got,
            } => write!(f, "buffer {name:?} has {got:?}, expected {expected:?}"),
            Self::Shape => write!(f, "operation shapes disagree"),
            Self::NonFinite { name } => write!(f, "buffer {name:?} contains non-finite values"),
            Self::AttributeType { name, expected } => {
                write!(f, "attribute {name:?} must be {expected}")
            }
            Self::AttributeValue { name, constraint } => {
                write!(f, "attribute {name:?} violates {constraint}")
            }
        }
    }
}

impl std::error::Error for TrainOperationErrorV1 {}

/// Fallible backend execution failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrainBackendError {
    /// Shared request-contract validation failed.
    InvalidRequest(TrainRequestError),
    /// Manifest operation is valid but adapter does not implement it.
    UnsupportedOperation(String),
    /// Adapter rejected operation-specific roles/attributes.
    InvalidOperation(TrainOperationErrorV1),
    /// Device/runtime failure.
    Backend {
        /// Stable backend-specific error code.
        code: String,
        /// Human-readable diagnostic.
        message: String,
    },
}

impl fmt::Display for TrainBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(error) => write!(f, "invalid training request: {error}"),
            Self::UnsupportedOperation(operation) => {
                write!(f, "unsupported training operation {operation:?}")
            }
            Self::InvalidOperation(error) => write!(f, "invalid operation contract: {error}"),
            Self::Backend { code, message } => {
                write!(f, "training backend failed ({code}): {message}")
            }
        }
    }
}

impl std::error::Error for TrainBackendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidRequest(error) => Some(error),
            Self::InvalidOperation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TrainRequestError> for TrainBackendError {
    fn from(error: TrainRequestError) -> Self {
        Self::InvalidRequest(error)
    }
}

/// Fallible portable whole-Tape execution seam implemented by each backend.
pub trait TrainBackendV1: Send + Sync {
    /// Declare exact manifest coverage, shape ceilings and residency.
    fn capabilities(&self) -> TrainCapabilitiesV1;

    /// Execute one manifest operation into caller-owned buffers.
    ///
    /// Implementations must call [`TrainRequestV1::validate_with_limits`] before
    /// reading or mutating any buffer and must not claim device residency after
    /// fallback.
    ///
    /// # Errors
    /// Returns [`TrainBackendError`] for malformed contracts, unsupported
    /// operations or backend failures. Public adapters must not panic.
    fn execute(
        &self,
        request: TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<TrainReceiptV1, TrainBackendError>;
}

fn execution_allowed(
    id: &str,
    category: TrainingOpCategoryV1,
    forward: bool,
    vjp: TrainingVjpV1,
    execution: TrainExecutionV1,
) -> bool {
    match execution {
        TrainExecutionV1::Forward => forward,
        TrainExecutionV1::Vjp => vjp == TrainingVjpV1::FirstOrder,
        TrainExecutionV1::Step => category == TrainingOpCategoryV1::Optimizer,
        TrainExecutionV1::Checkpoint => id == "lifecycle.checkpoint",
        TrainExecutionV1::Resume => id == "lifecycle.resume",
        TrainExecutionV1::Export => id == "lifecycle.export",
        TrainExecutionV1::Reload => id == "lifecycle.reload",
    }
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'.')
        })
}

fn checked_elements(
    name: &str,
    shape: &[u64],
    dtype: TrainDTypeV1,
    limits: TrainLimitsV1,
) -> Result<usize, TrainRequestError> {
    if shape.len() > limits.max_rank as usize {
        return Err(TrainRequestError::RankLimit {
            name: name.to_owned(),
            got: shape.len(),
            max: limits.max_rank,
        });
    }
    let elements = shape.iter().try_fold(1_u64, |elements, &dimension| {
        elements
            .checked_mul(dimension)
            .ok_or_else(|| TrainRequestError::ShapeOverflow {
                name: name.to_owned(),
            })
    })?;
    if elements > limits.max_elements {
        return Err(TrainRequestError::ElementLimit {
            name: name.to_owned(),
            got: elements,
            max: limits.max_elements,
        });
    }
    let bytes = elements.checked_mul(dtype.element_bytes()).ok_or_else(|| {
        TrainRequestError::ByteCountOverflow {
            name: name.to_owned(),
        }
    })?;
    if bytes > limits.max_bytes {
        return Err(TrainRequestError::ByteLimit {
            name: name.to_owned(),
            got: bytes,
            max: limits.max_bytes,
        });
    }
    usize::try_from(elements).map_err(|_| TrainRequestError::ShapeOverflow {
        name: name.to_owned(),
    })
}

trait BufferMetadata {
    fn name(&self) -> &str;
    fn shape(&self) -> &[u64];
    fn len(&self) -> usize;
    fn dtype(&self) -> TrainDTypeV1;
}

impl BufferMetadata for TrainNamedBufferRefV1<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn shape(&self) -> &[u64] {
        self.shape
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn dtype(&self) -> TrainDTypeV1 {
        self.data.dtype()
    }
}

impl BufferMetadata for TrainNamedBufferMutV1<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn shape(&self) -> &[u64] {
        self.shape
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn dtype(&self) -> TrainDTypeV1 {
        self.data.dtype()
    }
}

fn validate_buffers<T: BufferMetadata>(
    buffers: &[T],
    namespace: &'static str,
    limits: TrainLimitsV1,
) -> Result<(), TrainRequestError> {
    for (index, buffer) in buffers.iter().enumerate() {
        validate_name(namespace, buffer.name())?;
        reject_prior_name(
            namespace,
            buffer.name(),
            &buffers[..index],
            BufferMetadata::name,
        )?;
        let expected = checked_elements(buffer.name(), buffer.shape(), buffer.dtype(), limits)?;
        let got = buffer.len();
        if expected != got {
            return Err(TrainRequestError::BufferLength {
                name: buffer.name().to_owned(),
                expected,
                got,
            });
        }
    }
    Ok(())
}

fn validate_ref_buffers(
    buffers: &[TrainNamedBufferRefV1<'_>],
    limits: TrainLimitsV1,
) -> Result<(), TrainRequestError> {
    validate_buffers(buffers, "input", limits)
}

fn validate_mut_buffers(
    buffers: &[TrainNamedBufferMutV1<'_>],
    limits: TrainLimitsV1,
) -> Result<(), TrainRequestError> {
    validate_buffers(buffers, "output", limits)
}

fn validate_attributes(attributes: &[TrainAttributeV1<'_>]) -> Result<(), TrainRequestError> {
    for (index, attribute) in attributes.iter().enumerate() {
        validate_name("attribute", attribute.name)?;
        reject_prior_name("attribute", attribute.name, &attributes[..index], |item| {
            item.name
        })?;
        if matches!(attribute.value, TrainAttributeValueV1::F32(value) if !value.is_finite()) {
            return Err(TrainRequestError::NonFiniteAttribute(
                attribute.name.to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_name(namespace: &'static str, name: &str) -> Result<(), TrainRequestError> {
    if valid_name(name) {
        Ok(())
    } else {
        Err(TrainRequestError::InvalidName {
            namespace,
            name: name.to_owned(),
        })
    }
}

fn reject_prior_name<T>(
    namespace: &'static str,
    name: &str,
    prior: &[T],
    get_name: impl Fn(&T) -> &str,
) -> Result<(), TrainRequestError> {
    if prior.iter().any(|item| get_name(item) == name) {
        Err(TrainRequestError::DuplicateName {
            namespace,
            name: name.to_owned(),
        })
    } else {
        Ok(())
    }
}

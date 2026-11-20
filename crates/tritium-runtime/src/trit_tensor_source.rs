//! Shared retained typed-model adapter to the encoded-upload interface.
use std::{
    cell::RefCell,
    io::{Read, Seek},
};
use tritium_format::trit_model::TritTensorStream;
use tritium_schema::{PlaneCounts, TensorMeta, TensorStorage};
use tritium_spec::{
    AdditiveMetadata, BackendError, TensorMetadata, TensorStreamEncoding, TensorStreamSource,
};

/// Borrow validated typed matrix geometry without cloning allocation counts.
/// # Errors
/// Unsupported storage/rank/counts or platform-unrepresentable dimensions.
pub fn trit_tensor_metadata(meta: &TensorMeta) -> Result<TensorMetadata<'_>, BackendError> {
    match &meta.storage {
        TensorStorage::Additive { layout, planes } => Ok(TensorMetadata::Additive(match planes {
            PlaneCounts::Uniform(n) => AdditiveMetadata::uniform(*layout, *n)?,
            PlaneCounts::PerTile(counts) => AdditiveMetadata::per_tile(*layout, counts)?,
            _ => return Err(BackendError::UnsupportedTensor),
        })),
        TensorStorage::Dense { shape, .. } if shape.len() == 2 => Ok(TensorMetadata::Dense {
            rows: usize::try_from(shape[0])
                .map_err(|_| BackendError::InvalidInput("rows exceed usize".into()))?,
            cols: usize::try_from(shape[1])
                .map_err(|_| BackendError::InvalidInput("columns exceed usize".into()))?,
        }),
        _ => Err(BackendError::UnsupportedTensor),
    }
}

/// Preserve the original typed payload encoding; no compatibility decoding.
/// # Errors
/// Unknown storage interpretation.
pub fn trit_tensor_encoding(meta: &TensorMeta) -> Result<TensorStreamEncoding, BackendError> {
    Ok(match &meta.storage {
        TensorStorage::Additive { .. } => TensorStreamEncoding::CanonicalRawAdditive,
        TensorStorage::Dense { precision, .. } => {
            TensorStreamEncoding::DenseLittleEndian(*precision)
        }
        _ => return Err(BackendError::UnsupportedTensor),
    })
}

/// Bounded repeatable source over one admitted model stream. Complete terminal
/// source verification remains the enclosing loader's responsibility.
#[derive(Debug)]
pub struct TritTensorSource<'a, R> {
    metadata: TensorMetadata<'a>,
    encoding: TensorStreamEncoding,
    encoded_bytes: u64,
    reader: RefCell<TritTensorStream<'a, R>>,
}
impl<'a, R: Read + Seek> TritTensorSource<'a, R> {
    /// Bind borrowed metadata/counts and the retained reader together.
    /// # Errors
    /// Unknown matrix geometry or encoding.
    pub fn new(reader: TritTensorStream<'a, R>) -> Result<Self, BackendError> {
        Ok(Self {
            metadata: trit_tensor_metadata(reader.metadata())?,
            encoding: trit_tensor_encoding(reader.metadata())?,
            encoded_bytes: reader.encoded_bytes(),
            reader: RefCell::new(reader),
        })
    }
}
impl<R: Read + Seek> TensorStreamSource for TritTensorSource<'_, R> {
    fn metadata(&self) -> TensorMetadata<'_> {
        self.metadata
    }
    fn encoding(&self) -> TensorStreamEncoding {
        self.encoding
    }
    fn encoded_bytes(&self) -> u64 {
        self.encoded_bytes
    }
    fn visit(
        &self,
        visitor: &mut dyn FnMut(&[u8]) -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        self.reader
            .try_borrow_mut()
            .map_err(|_| BackendError::InvalidInput("recursive retained stream visit".into()))?
            .visit(visitor)
            .map_err(|e| BackendError::Backend(format!("retained tensor stream: {e}")))
    }
}

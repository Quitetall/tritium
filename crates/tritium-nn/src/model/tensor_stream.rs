//! Retained typed source adapter at the common encoded-upload seam.
use crate::{Linear, NnError, TensorLoader};
use std::{
    cell::RefCell,
    io::{Read, Seek},
};
use tritium_format::trit_model::{TritModelReader, TritTensorStream};
use tritium_schema::{PlaneCounts, TensorMeta, TensorStorage};
use tritium_spec::{
    AdditiveMetadata, BackendError, TensorMetadata, TensorStreamEncoding, TensorStreamSource,
};

pub(super) fn metadata(meta: &TensorMeta) -> Result<TensorMetadata<'_>, BackendError> {
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
pub(super) fn encoding(meta: &TensorMeta) -> Result<TensorStreamEncoding, BackendError> {
    Ok(match &meta.storage {
        TensorStorage::Additive { .. } => TensorStreamEncoding::CanonicalRawAdditive,
        TensorStorage::Dense { precision, .. } => {
            TensorStreamEncoding::DenseLittleEndian(*precision)
        }
        _ => return Err(BackendError::UnsupportedTensor),
    })
}
struct Source<'a, R> {
    metadata: TensorMetadata<'a>,
    encoding: TensorStreamEncoding,
    encoded_bytes: u64,
    reader: RefCell<TritTensorStream<'a, R>>,
}
impl<R: Read + Seek> TensorStreamSource for Source<'_, R> {
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
pub(super) fn upload<R: Read + Seek>(
    loader: &mut TensorLoader<'_>,
    model: &mut TritModelReader<R>,
    level: &str,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Linear, NnError> {
    let reader = model
        .tensor_stream(level, name)
        .map_err(|e| NnError::InvalidArtifact(e.to_string()))?;
    let meta = reader.metadata();
    if super::source::matrix_shape(meta)? != (rows, cols) {
        return Err(NnError::InvalidArtifact(format!(
            "{name}: stream geometry mismatch"
        )));
    }
    let source = Source {
        metadata: metadata(meta).map_err(|e| NnError::Backend(e.to_string()))?,
        encoding: encoding(meta).map_err(|e| NnError::Backend(e.to_string()))?,
        encoded_bytes: reader.encoded_bytes(),
        reader: RefCell::new(reader),
    };
    loader.upload_stream(name, &source)
}

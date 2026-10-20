//! Retained typed source adapter at the common encoded-upload seam.
use crate::{Linear, NnError, TensorLoader};
use std::io::{Read, Seek};
use tritium_format::trit_model::TritModelReader;
pub(super) use tritium_runtime::{
    trit_tensor_encoding as encoding, trit_tensor_metadata as metadata,
};
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
    let source = tritium_runtime::TritTensorSource::new(reader)
        .map_err(|e| NnError::Backend(e.to_string()))?;
    loader.upload_stream(name, &source)
}

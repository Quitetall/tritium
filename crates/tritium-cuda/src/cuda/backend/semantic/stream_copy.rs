//! Checked windows into unpublished device payloads.
use super::*;
mod pinned;
use pinned::Window;

pub(super) fn dense(
    backend: &CudaBackend,
    source: &dyn TensorStreamSource,
    rows: usize,
    cols: usize,
    precision: tritium_schema::ScalePrecision,
) -> Result<Option<CudaSlice<f32>>, BackendError> {
    let elements = TensorMetadata::dense_bytes(rows, cols)? / 4;
    let expected = tritium_format::trit_model::DensePayloadDecoder::byte_len(precision, elements)
        .map_err(|e| invalid(&e.to_string()))?;
    if elements == 0 {
        visit(source, expected, |_| {
            Err(invalid("nonempty zero-size dense source"))
        })?;
        return Ok(None);
    }
    let width = match precision {
        tritium_schema::ScalePrecision::F16 | tritium_schema::ScalePrecision::Bf16 => 2,
        tritium_schema::ScalePrecision::F32 => 4,
        _ => return Err(BackendError::UnsupportedTensor),
    };
    let mut device = backend
        .stream
        .alloc_zeros::<f32>(elements)
        .map_err(|e| alloc_or_backend("allocate private dense stream payload", &e, elements * 4))?;
    let capacity = elements.min(TENSOR_STREAM_CHUNK_BYTES / width);
    let mut windows = [
        Window::<f32>::new(backend, capacity)?,
        Window::<f32>::new(backend, capacity)?,
    ];
    let mut active = 0usize;
    let mut pending = [0u8; TENSOR_STREAM_CHUNK_BYTES];
    let mut used = 0;
    let mut at = 0usize;
    let mut flush = |bytes: &[u8]| -> Result<(), BackendError> {
        let count = bytes.len() / width;
        let mut decoder = tritium_format::trit_model::DensePayloadDecoder::new(
            precision,
            &mut windows[active].mutable()?[..count],
        )
        .map_err(|e| invalid(&e.to_string()))?;
        decoder.push(bytes).map_err(|e| invalid(&e.to_string()))?;
        decoder.finish().map_err(|e| invalid(&e.to_string()))?;
        let end = at
            .checked_add(count)
            .ok_or_else(|| invalid("dense copy offset overflow"))?;
        if end > elements {
            return Err(invalid("dense copy exceeds private allocation"));
        }
        windows[active].copy(&mut device, at, count)?;
        active ^= 1;
        at = end;
        Ok(())
    };
    visit(source, expected, |mut bytes| {
        while !bytes.is_empty() {
            let n = bytes.len().min(pending.len() - used);
            pending[used..used + n].copy_from_slice(&bytes[..n]);
            used += n;
            bytes = &bytes[n..];
            if used == pending.len() {
                flush(&pending)?;
                used = 0;
            }
        }
        Ok(())
    })?;
    if used != 0 {
        flush(&pending[..used])?;
    }
    if at != elements {
        return Err(invalid("dense stream final extent mismatch"));
    }
    for window in &mut windows {
        window.wait()?;
    }
    drop(windows);
    backend
        .stream
        .synchronize()
        .map_err(|e| driver_err("finish private dense CUDA upload", &e))?;
    backend
        .stream
        .context()
        .check_err()
        .map_err(|e| driver_err("check dense pinned cleanup", &e))?;
    Ok(Some(device))
}

pub(super) fn visit(
    source: &dyn TensorStreamSource,
    expected: usize,
    mut visitor: impl FnMut(&[u8]) -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    let mut seen = 0usize;
    let mut poisoned = false;
    source.visit(&mut |bytes| {
        if poisoned {
            return Err(invalid("CUDA source ignored a prior visitor error"));
        }
        let result = (|| {
            let end = seen
                .checked_add(bytes.len())
                .ok_or_else(|| invalid("CUDA stream size overflow"))?;
            if bytes.len() > TENSOR_STREAM_CHUNK_BYTES || end > expected {
                return Err(invalid("CUDA source exceeds bounded stream geometry"));
            }
            visitor(bytes)?;
            seen = end;
            Ok(())
        })();
        if result.is_err() {
            poisoned = true;
        }
        result
    })?;
    if poisoned || seen != expected {
        return Err(invalid("CUDA source failed exact bounded visit"));
    }
    Ok(())
}

pub(super) struct ByteWriter<'a> {
    device: &'a mut CudaSlice<u8>,
    windows: [Window<u8>; 2],
    active: usize,
    used: usize,
    at: usize,
}
impl<'a> ByteWriter<'a> {
    pub(super) fn new(
        backend: &'a CudaBackend,
        device: &'a mut CudaSlice<u8>,
        at: usize,
    ) -> Result<Self, BackendError> {
        let windows = [
            Window::new(backend, TENSOR_STREAM_CHUNK_BYTES)?,
            Window::new(backend, TENSOR_STREAM_CHUNK_BYTES)?,
        ];
        Ok(Self {
            device,
            windows,
            active: 0,
            used: 0,
            at,
        })
    }
    pub(super) fn push(&mut self, mut bytes: &[u8]) -> Result<(), BackendError> {
        while !bytes.is_empty() {
            let n = bytes.len().min(TENSOR_STREAM_CHUNK_BYTES - self.used);
            self.windows[self.active].mutable()?[self.used..self.used + n]
                .copy_from_slice(&bytes[..n]);
            self.used += n;
            bytes = &bytes[n..];
            if self.used == TENSOR_STREAM_CHUNK_BYTES {
                self.flush()?;
            }
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<(), BackendError> {
        if self.used == 0 {
            return Ok(());
        }
        let end = self
            .at
            .checked_add(self.used)
            .ok_or_else(|| invalid("CUDA copy offset overflow"))?;
        if end > self.device.len() {
            return Err(invalid("CUDA copy exceeds private allocation"));
        }
        self.windows[self.active].copy(self.device, self.at, self.used)?;
        self.at = end;
        self.active ^= 1;
        self.used = 0;
        Ok(())
    }
    pub(super) fn finish(mut self, expected: usize) -> Result<(), BackendError> {
        self.flush()?;
        for window in &mut self.windows {
            window.wait()?;
        }
        if self.at != expected {
            return Err(invalid("CUDA byte writer extent mismatch"));
        }
        Ok(())
    }
}

use std::cell::Cell;
use tritium_spec::{BackendError, TensorMetadata, TensorStreamEncoding, TensorStreamSource};

#[derive(Debug)]
pub(crate) struct Source<'a> {
    pub metadata: TensorMetadata<'a>,
    pub encoding: TensorStreamEncoding,
    pub bytes: &'a [u8],
    pub chunk: usize,
    pub declared: u64,
    pub calls: Cell<usize>,
    pub fail: bool,
}
impl<'a> Source<'a> {
    pub(crate) fn new(
        metadata: TensorMetadata<'a>,
        encoding: TensorStreamEncoding,
        bytes: &'a [u8],
    ) -> Self {
        Self {
            metadata,
            encoding,
            bytes,
            chunk: 7,
            declared: bytes.len() as u64,
            calls: Cell::new(0),
            fail: false,
        }
    }
}
impl TensorStreamSource for Source<'_> {
    fn metadata(&self) -> TensorMetadata<'_> {
        self.metadata
    }
    fn encoding(&self) -> TensorStreamEncoding {
        self.encoding
    }
    fn encoded_bytes(&self) -> u64 {
        self.declared
    }
    fn visit(
        &self,
        visitor: &mut dyn FnMut(&[u8]) -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        self.calls.set(self.calls.get() + 1);
        for bytes in self.bytes.chunks(self.chunk) {
            visitor(bytes)?;
        }
        if self.fail {
            return Err(BackendError::Backend(
                "injected terminal source fault".into(),
            ));
        }
        Ok(())
    }
}

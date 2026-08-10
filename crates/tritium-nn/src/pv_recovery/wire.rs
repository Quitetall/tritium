use super::DevicePvRecoveryError;

pub(super) struct Reader<'bytes> {
    bytes: &'bytes [u8],
    position: usize,
}

impl<'bytes> Reader<'bytes> {
    pub(super) const fn new(bytes: &'bytes [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(super) fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    pub(super) fn take(&mut self, count: usize) -> Result<&'bytes [u8], DevicePvRecoveryError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| checkpoint_error("checkpoint range overflows host range"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| checkpoint_error("checkpoint is truncated"))?;
        self.position = end;
        Ok(value)
    }

    pub(super) fn array<const N: usize>(&mut self) -> Result<[u8; N], DevicePvRecoveryError> {
        self.take(N)?
            .try_into()
            .map_err(|_| checkpoint_error("checkpoint is truncated"))
    }

    pub(super) fn u8(&mut self) -> Result<u8, DevicePvRecoveryError> {
        Ok(self.take(1)?[0])
    }

    pub(super) fn u64(&mut self) -> Result<u64, DevicePvRecoveryError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    pub(super) fn usize(&mut self) -> Result<usize, DevicePvRecoveryError> {
        usize::try_from(self.u64()?)
            .map_err(|_| checkpoint_error("checkpoint value exceeds host range"))
    }

    pub(super) fn blob(&mut self) -> Result<&'bytes [u8], DevicePvRecoveryError> {
        let length = self.usize()?;
        self.take(length)
    }
}

fn checkpoint_error(reason: &str) -> DevicePvRecoveryError {
    DevicePvRecoveryError::Checkpoint(reason.into())
}

use crate::optim::AdamState;

use super::PvTuningError;

pub(super) fn write_adam_state(out: &mut Vec<u8>, state: &AdamState) {
    for values in [&state.m, &state.v] {
        for value in values {
            out.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
}

pub(super) fn read_adam_state(
    reader: &mut Reader<'_>,
    len: usize,
) -> Result<AdamState, PvTuningError> {
    let mut read_values = || -> Result<Vec<f32>, PvTuningError> {
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            let value = f32::from_bits(reader.u32()?);
            if !value.is_finite() {
                return Err(PvTuningError::checkpoint(
                    "optimizer state contains a non-finite value",
                ));
            }
            values.push(value);
        }
        Ok(values)
    };
    let m = read_values()?;
    let v = read_values()?;
    if v.iter().any(|value| *value < 0.0) {
        return Err(PvTuningError::checkpoint(
            "optimizer second moment contains a negative value",
        ));
    }
    Ok(AdamState { m, v })
}

#[derive(Debug)]
pub(super) struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    pub(super) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(super) fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], PvTuningError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| PvTuningError::checkpoint("checkpoint offset overflow"))?;
        if end > self.bytes.len() {
            return Err(PvTuningError::checkpoint("checkpoint is truncated"));
        }
        let bytes = &self.bytes[self.position..end];
        self.position = end;
        Ok(bytes)
    }

    pub(super) fn array<const N: usize>(&mut self) -> Result<[u8; N], PvTuningError> {
        self.take(N)?
            .try_into()
            .map_err(|_| PvTuningError::checkpoint("checkpoint is truncated"))
    }

    pub(super) fn u8(&mut self) -> Result<u8, PvTuningError> {
        Ok(self.array::<1>()?[0])
    }

    pub(super) fn u16(&mut self) -> Result<u16, PvTuningError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    pub(super) fn u32(&mut self) -> Result<u32, PvTuningError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    pub(super) fn u64(&mut self) -> Result<u64, PvTuningError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    pub(super) fn usize(&mut self) -> Result<usize, PvTuningError> {
        usize::try_from(self.u64()?)
            .map_err(|_| PvTuningError::checkpoint("checkpoint geometry exceeds host range"))
    }
}

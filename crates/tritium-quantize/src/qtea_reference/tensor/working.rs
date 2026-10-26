//! Private resident/seek-backed adapters for the same ordered feedback sweep.
use std::io::{Read, Seek, SeekFrom, Write};

use crate::qtea_reference::{ArithmeticError, filled};

pub(super) trait ScratchIo: Read + Write + Seek {}
impl<T: Read + Write + Seek> ScratchIo for T {}

fn io(error: std::io::Error) -> ArithmeticError {
    ArithmeticError::ScratchIo { kind: error.kind() }
}

fn seek(stream: &mut dyn ScratchIo, position: u64) -> Result<(), ArithmeticError> {
    if stream.seek(SeekFrom::Start(position)).map_err(io)? != position {
        return Err(io(std::io::ErrorKind::InvalidData.into()));
    }
    Ok(())
}

pub(super) fn check_empty(stream: &mut dyn ScratchIo) -> Result<(), ArithmeticError> {
    if stream.seek(SeekFrom::End(0)).map_err(io)? != 0 {
        return Err(ArithmeticError::ScratchNotEmpty);
    }
    seek(stream, 0)
}

enum Storage<'a> {
    Resident(Vec<f32>),
    SeekBacked {
        stream: &'a mut dyn ScratchIo,
        bytes: Vec<u8>,
    },
}

pub(super) struct Working<'a> {
    rows: usize,
    columns: usize,
    storage: Storage<'a>,
}

impl<'a> Working<'a> {
    pub(super) fn new(
        original: &[f32],
        rows: usize,
        columns: usize,
        order: &[usize],
        stream: Option<&'a mut dyn ScratchIo>,
    ) -> Result<Self, ArithmeticError> {
        let storage = match stream {
            Some(stream) => Storage::SeekBacked {
                stream,
                bytes: filled(columns * 4, 0)?,
            },
            None => Storage::Resident(filled(original.len(), 0.0)?),
        };
        let mut result = Self {
            rows,
            columns,
            storage,
        };
        let mut row_values = filled(columns, 0.0)?;
        for row in 0..rows {
            for (column, &source) in order.iter().enumerate() {
                row_values[column] = original[row * columns + source];
            }
            result.write(row, 0, &row_values)?;
        }
        Ok(result)
    }

    fn range(&self, row: usize, start: usize, length: usize) -> Result<usize, ArithmeticError> {
        if row >= self.rows
            || start
                .checked_add(length)
                .is_none_or(|end| end > self.columns)
        {
            return Err(ArithmeticError::InvalidShape);
        }
        Ok(row * self.columns + start) // Complete geometry was checked before construction.
    }

    pub(super) fn read(
        &mut self,
        row: usize,
        start: usize,
        output: &mut [f32],
    ) -> Result<(), ArithmeticError> {
        let index = self.range(row, start, output.len())?;
        match &mut self.storage {
            Storage::Resident(values) => {
                output.copy_from_slice(&values[index..index + output.len()])
            }
            Storage::SeekBacked { stream, bytes } => {
                let bytes = &mut bytes[..output.len() * 4];
                let offset = (index as u64) * 4;
                seek(*stream, offset)?;
                stream.read_exact(bytes).map_err(io)?;
                if stream.stream_position().map_err(io)? != offset + bytes.len() as u64 {
                    return Err(io(std::io::ErrorKind::InvalidData.into()));
                }
                for (value, encoded) in output.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    *value = f32::from_le_bytes(*encoded);
                    if !value.is_finite() {
                        return Err(ArithmeticError::NonFiniteArithmetic);
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn write(
        &mut self,
        row: usize,
        start: usize,
        values: &[f32],
    ) -> Result<(), ArithmeticError> {
        let index = self.range(row, start, values.len())?;
        match &mut self.storage {
            Storage::Resident(output) => {
                output[index..index + values.len()].copy_from_slice(values)
            }
            Storage::SeekBacked { stream, bytes } => {
                let bytes = &mut bytes[..values.len() * 4];
                for (encoded, value) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                    if !value.is_finite() {
                        return Err(ArithmeticError::NonFiniteArithmetic);
                    }
                    encoded.copy_from_slice(&value.to_le_bytes());
                }
                let offset = (index as u64) * 4;
                seek(*stream, offset)?;
                stream.write_all(bytes).map_err(io)?;
                if stream.stream_position().map_err(io)? != offset + bytes.len() as u64 {
                    return Err(io(std::io::ErrorKind::InvalidData.into()));
                }
            }
        }
        Ok(())
    }

    pub(super) fn finish(&mut self) -> Result<(), ArithmeticError> {
        if let Storage::SeekBacked { stream, .. } = &mut self.storage {
            stream.flush().map_err(io)?;
            let expected = self.rows as u64 * self.columns as u64 * 4;
            if stream.seek(SeekFrom::End(0)).map_err(io)? != expected {
                return Err(io(std::io::ErrorKind::InvalidData.into()));
            }
        }
        Ok(())
    }
}

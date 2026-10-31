//! Private shared streaming mechanics, not a journal profile or admission API.
use std::io::{Read, Seek, SeekFrom, Write};

/// Private sequential read/seek loan shared by region readers and companion guards.
pub(super) struct Shared<'a, R>(pub &'a std::cell::RefCell<R>);
impl<R: Read> Read for Shared<'_, R> {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().read(b)
    }
}
impl<R: Seek> Seek for Shared<'_, R> {
    fn seek(&mut self, s: SeekFrom) -> std::io::Result<u64> {
        self.0.borrow_mut().seek(s)
    }
}

pub(super) const SCRATCH: usize = 8192;
#[derive(Debug)]
pub(super) enum Error {
    Limit(&'static str),
    Io(std::io::Error),
    Poisoned,
}
impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
pub(super) fn sum(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b)
        .ok_or(Error::Limit("native journal size overflow"))
}

pub(super) fn write<W: Write>(
    output: &mut W,
    hash: &mut blake3::Hasher,
    count: &mut u64,
    expected: u64,
    poisoned: &mut bool,
    bytes: &[u8],
) -> Result<(), Error> {
    if *poisoned {
        return Err(Error::Poisoned);
    }
    let next = sum(*count, bytes.len() as u64)?;
    if next > expected {
        return Err(Error::Limit("native journal emitted bytes exceed limit"));
    }
    *poisoned = true;
    output.write_all(bytes)?;
    hash.update(bytes);
    *count = next;
    *poisoned = false;
    Ok(())
}
pub(super) fn flush<W: Write>(output: &mut W, poisoned: &mut bool) -> Result<(), Error> {
    *poisoned = true;
    output.flush()?;
    *poisoned = false;
    Ok(())
}
pub(super) fn read<R: Read>(
    source: &mut R,
    hash: &mut blake3::Hasher,
    count: &mut u64,
    bytes: &mut [u8],
) -> Result<(), Error> {
    source.read_exact(bytes)?;
    hash.update(bytes);
    *count = sum(*count, bytes.len() as u64)?;
    Ok(())
}
pub(super) fn hash_exact<R: Read>(
    source: &mut R,
    mut count: u64,
    scratch: &mut [u8; SCRATCH],
) -> std::io::Result<[u8; 32]> {
    let mut hash = blake3::Hasher::new();
    while count > 0 {
        let n = count.min(SCRATCH as u64) as usize;
        source.read_exact(&mut scratch[..n])?;
        hash.update(&scratch[..n]);
        count -= n as u64;
    }
    Ok(*hash.finalize().as_bytes())
}
pub(super) fn encode<E>(
    values: &[f32],
    frame: &mut blake3::Hasher,
    mut write: impl FnMut(&[u8]) -> Result<(), E>,
) -> Result<(), E> {
    let mut scratch = [0; SCRATCH];
    for chunk in values.chunks(SCRATCH / 4) {
        for (bytes, value) in scratch.as_chunks_mut::<4>().0.iter_mut().zip(chunk) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        let bytes = &scratch[..chunk.len() * 4];
        write(bytes)?;
        frame.update(bytes);
    }
    Ok(())
}
pub(super) fn decode<E>(
    values: &mut [f32],
    scratch: &mut [u8; SCRATCH],
    frame: &mut blake3::Hasher,
    mut read: impl FnMut(&mut [u8]) -> Result<(), E>,
) -> Result<(), E> {
    for chunk in values.chunks_mut(SCRATCH / 4) {
        let bytes = &mut scratch[..chunk.len() * 4];
        read(bytes)?;
        frame.update(bytes);
        for (value, encoded) in chunk.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *value = f32::from_le_bytes(*encoded);
        }
    }
    Ok(())
}

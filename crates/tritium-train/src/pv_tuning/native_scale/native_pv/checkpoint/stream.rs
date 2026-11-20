//! Shared TNPV owner codec and bounded stream integrity, not a new wire format.
use super::super::*;
use std::io::{self, Read, Seek, SeekFrom, Write};
use tritium_core::Trit;

pub(in super::super) const IO_BYTES: usize = 8192;

#[derive(Debug)]
pub(in super::super) enum WireError {
    Invalid(PvTuningError),
    Io(io::Error),
}
impl From<io::Error> for WireError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
impl From<PvTuningError> for WireError {
    fn from(value: PvTuningError) -> Self {
        Self::Invalid(value)
    }
}
fn bad(text: &str) -> WireError {
    PvTuningError::checkpoint(text).into()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) struct Header {
    pub plan: NativeRefinementPlanId,
    pub parent: [u8; 32],
    pub completed: u64,
    pub owners: u64,
}
impl Header {
    pub(in super::super) fn bytes(self) -> [u8; 85] {
        let mut bytes = [0; 85];
        bytes[..5].copy_from_slice(b"TNPV\x01");
        bytes[5..37].copy_from_slice(self.plan.as_bytes());
        bytes[37..69].copy_from_slice(&self.parent);
        bytes[69..77].copy_from_slice(&self.completed.to_le_bytes());
        bytes[77..85].copy_from_slice(&self.owners.to_le_bytes());
        bytes
    }
}
fn array<const N: usize>(reader: &mut impl Read) -> Result<[u8; N], WireError> {
    let mut bytes = [0; N];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}
fn count(reader: &mut impl Read) -> Result<u64, WireError> {
    Ok(u64::from_le_bytes(array(reader)?))
}
pub(in super::super) fn read_header(reader: &mut impl Read) -> Result<Header, WireError> {
    if array::<5>(reader)? != *b"TNPV\x01" {
        return Err(bad("unsupported native PV checkpoint format"));
    }
    Ok(Header {
        plan: NativeRefinementPlanId::from_bytes(array(reader)?),
        parent: array(reader)?,
        completed: count(reader)?,
        owners: count(reader)?,
    })
}

fn write_values<const BLOCK: usize>(
    writer: &mut impl Write,
    values: &[f32],
    precision: ScalePrecision,
) -> Result<(), WireError> {
    let width = scale_width(precision)? as usize;
    let mut buffer = [0; BLOCK];
    for chunk in values.chunks(BLOCK / width) {
        for (dst, &value) in buffer.chunks_mut(width).zip(chunk) {
            match precision {
                ScalePrecision::F16 => {
                    dst.copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes())
                }
                ScalePrecision::Bf16 => {
                    dst.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes())
                }
                ScalePrecision::F32 => dst.copy_from_slice(&value.to_le_bytes()),
                _ => return Err(bad("unsupported native PV scale precision")),
            }
        }
        writer.write_all(&buffer[..chunk.len() * width])?;
    }
    Ok(())
}
pub(in super::super) fn write_owner<const BLOCK: usize>(
    writer: &mut impl Write,
    owner: &PvOwner,
) -> Result<(), WireError> {
    writer.write_all(&(owner.weight.trits().len() as u64).to_le_bytes())?;
    let mut buffer = [0; BLOCK];
    for chunk in owner.weight.trits().chunks(BLOCK) {
        for (byte, code) in buffer.iter_mut().zip(chunk) {
            *byte = code.get() as u8;
        }
        writer.write_all(&buffer[..chunk.len()])?;
    }
    writer.write_all(&(owner.weight.scales().len() as u64).to_le_bytes())?;
    write_values::<BLOCK>(
        writer,
        owner.weight.scales(),
        owner.weight.layout().law.precision,
    )?;
    for values in [&owner.scale.m, &owner.scale.v] {
        write_values::<BLOCK>(writer, values, ScalePrecision::F32)?;
    }
    writer.write_all(&(owner.code.m.len() as u64).to_le_bytes())?;
    for values in [&owner.code.m, &owner.code.v] {
        write_values::<BLOCK>(writer, values, ScalePrecision::F32)?;
    }
    Ok(())
}
fn read_values<const BLOCK: usize>(
    reader: &mut impl Read,
    n: usize,
    precision: ScalePrecision,
) -> Result<Vec<f32>, WireError> {
    let width = scale_width(precision)? as usize;
    let mut values = reserved(n)?;
    let mut buffer = [0; BLOCK];
    while values.len() < n {
        let amount = (n - values.len()).min(BLOCK / width);
        reader.read_exact(&mut buffer[..amount * width])?;
        for bytes in buffer[..amount * width].chunks_exact(width) {
            values.push(match precision {
                ScalePrecision::F16 => {
                    f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
                }
                ScalePrecision::Bf16 => {
                    bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
                }
                ScalePrecision::F32 => {
                    f32::from_le_bytes(bytes.try_into().map_err(|_| bad("invalid scale width"))?)
                }
                _ => return Err(bad("unsupported native PV scale precision")),
            });
        }
    }
    Ok(values)
}
fn read_state<const BLOCK: usize>(
    reader: &mut impl Read,
    n: usize,
) -> Result<AdamState, WireError> {
    let m = read_values::<BLOCK>(reader, n, ScalePrecision::F32)?;
    let v = read_values::<BLOCK>(reader, n, ScalePrecision::F32)?;
    if m.iter().chain(&v).any(|v| !v.is_finite()) || v.iter().any(|v| v.is_sign_negative()) {
        return Err(bad("native PV checkpoint invalid optimizer moment"));
    }
    Ok(AdamState { m, v })
}
pub(in super::super) fn read_owner<const BLOCK: usize>(
    reader: &mut impl Read,
    mut weight: AdditiveTensor,
    completed: u64,
    updater: PvUpdater,
) -> Result<PvOwner, WireError> {
    let active = active_count(&weight)?;
    if count(reader)? != weight.trits().len() as u64 {
        return Err(bad("native PV checkpoint trit count mismatch"));
    }
    let mut codes = reserved(weight.trits().len())?;
    let mut buffer = [0; BLOCK];
    while codes.len() < weight.trits().len() {
        let n = (weight.trits().len() - codes.len()).min(BLOCK);
        reader.read_exact(&mut buffer[..n])?;
        for &byte in &buffer[..n] {
            codes.push(
                Trit::from_i8(byte as i8).map_err(|_| bad("invalid native PV checkpoint trit"))?,
            );
        }
    }
    let scales = weight.scales().len();
    if count(reader)? != scales as u64 {
        return Err(bad("native PV checkpoint scale count mismatch"));
    }
    let values = read_values::<BLOCK>(reader, scales, weight.layout().law.precision)?;
    let scale = read_state::<BLOCK>(reader, scales)?;
    if count(reader)? != active as u64 {
        return Err(bad("native PV checkpoint code count mismatch"));
    }
    let code = read_state::<BLOCK>(reader, active)?;
    if completed == 0 {
        if codes != weight.trits()
            || values
                .iter()
                .zip(weight.scales())
                .any(|(a, b)| a.to_bits() != b.to_bits())
            || scale
                .m
                .iter()
                .chain(&scale.v)
                .chain(&code.m)
                .chain(&code.v)
                .any(|v| v.to_bits() != 0)
        {
            return Err(bad("native PV zero-step state differs from parent"));
        }
    } else {
        check_corrected_moments(updater.continuous, completed, &scale)
            .map_err(PvTuningError::checkpoint)?;
        check_corrected_moments(updater.code_optimizer, completed, &code)
            .map_err(PvTuningError::checkpoint)?;
    }
    weight
        .replace_scales(values)
        .map_err(|e| bad(&format!("native PV checkpoint scales: {e:?}")))?;
    weight
        .replace_trits(codes)
        .map_err(|e| bad(&format!("native PV checkpoint codes: {e:?}")))?;
    Ok(PvOwner {
        weight,
        scale,
        code,
    })
}

pub(in super::super) struct HashRead<'a, R> {
    pub source: &'a mut R,
    pub hash: blake3::Hasher,
}
impl<R: Read> Read for HashRead<'_, R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let n = self.source.read(bytes)?;
        if n > bytes.len() {
            return Err(io::Error::other("reader returned invalid length"));
        }
        self.hash.update(&bytes[..n]);
        Ok(n)
    }
}
pub(in super::super) struct HashWrite<'a, W> {
    pub output: &'a mut W,
    pub hash: blake3::Hasher,
}
impl<W: Write> Write for HashWrite<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let n = self.output.write(bytes)?;
        if n > bytes.len() {
            return Err(io::Error::other("writer returned invalid length"));
        }
        self.hash.update(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in super::super) struct Inspection {
    pub header: Header,
    pub digest: [u8; 32],
    pub body: [u8; 32],
    pub bytes: u64,
}
pub(in super::super) fn inspect<R: Read + Seek>(
    source: &mut R,
    cap: u64,
) -> Result<Inspection, WireError> {
    let bytes = source.seek(SeekFrom::End(0))?;
    if bytes < super::FIXED_BYTES || bytes > cap {
        return Err(bad("native PV checkpoint length/cap mismatch"));
    }
    source.rewind()?;
    let mut reader = HashRead {
        source,
        hash: blake3::Hasher::new(),
    };
    let header = read_header(&mut reader)?;
    let mut remaining = bytes - super::FIXED_BYTES;
    let mut buffer = [0; IO_BYTES];
    while remaining > 0 {
        let n = remaining.min(IO_BYTES as u64) as usize;
        reader.read_exact(&mut buffer[..n])?;
        remaining -= n as u64;
    }
    let body = *reader.hash.finalize().as_bytes();
    let checksum = array::<32>(&mut reader)?;
    if body != checksum {
        return Err(bad("native PV checkpoint checksum mismatch"));
    }
    Ok(Inspection {
        header,
        body,
        bytes,
        digest: *reader.hash.finalize().as_bytes(),
    })
}
pub(in super::super) fn seal<W: Read + Write + Seek>(
    output: &mut W,
    header: Header,
    bytes: u64,
    suffix: [u8; 32],
) -> Result<Inspection, WireError> {
    output.flush()?;
    if output.seek(SeekFrom::End(0))? != bytes - 32 {
        return Err(bad("native PV checkpoint output length mismatch"));
    }
    output.rewind()?;
    if array::<85>(output)? != header.bytes() {
        return Err(bad("native PV checkpoint output header drift"));
    }
    let mut whole = blake3::Hasher::new();
    whole.update(&header.bytes());
    let mut owner_hash = blake3::Hasher::new();
    let mut remaining = bytes - super::FIXED_BYTES;
    let mut buffer = [0; IO_BYTES];
    while remaining > 0 {
        let n = remaining.min(IO_BYTES as u64) as usize;
        output.read_exact(&mut buffer[..n])?;
        whole.update(&buffer[..n]);
        owner_hash.update(&buffer[..n]);
        remaining -= n as u64;
    }
    if *owner_hash.finalize().as_bytes() != suffix {
        return Err(bad("native PV checkpoint output payload drift"));
    }
    let body = *whole.finalize().as_bytes();
    output.write_all(&body)?;
    whole.update(&body);
    output.flush()?;
    let observed = inspect(output, bytes)?;
    if observed.digest != *whole.finalize().as_bytes() {
        return Err(bad("native PV checkpoint output changed after flush"));
    }
    Ok(observed)
}

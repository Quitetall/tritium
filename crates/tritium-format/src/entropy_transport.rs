//! Seekable entropy-coded transport for already-packed ternary artifacts.
//!
//! This layer is deliberately separate from resident/runtime formats. Each chunk is either
//! stored verbatim or encoded with a deterministic canonical byte Huffman table, and the index
//! keeps every logical range independently addressable. The expanded fixed-codec bytes therefore
//! remain the resident-byte denominator; this module only reduces bytes at rest or in transit.

use core::fmt;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::{Read, Seek, SeekFrom};

use blake3::Hasher;

/// File magic for the seekable entropy transport.
pub const ENTROPY_TRANSPORT_MAGIC: [u8; 4] = *b"TRNS";
/// Current transport version.
pub const ENTROPY_TRANSPORT_VERSION: u8 = 1;
/// Default independently seekable chunk size.
pub const ENTROPY_TRANSPORT_DEFAULT_CHUNK_BYTES: usize = 64 * 1024;
/// Minimum accepted chunk size.
pub const ENTROPY_TRANSPORT_MIN_CHUNK_BYTES: usize = 64;
/// Maximum accepted chunk size.
pub const ENTROPY_TRANSPORT_MAX_CHUNK_BYTES: usize = 1 << 20;
const HEADER_BYTES: usize = 36;
const INDEX_ENTRY_BYTES: usize = 60;
const MODE_RAW: u8 = 0;
const MODE_HUFFMAN: u8 = 1;

/// A single independently addressable transport chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntropyChunkInfo {
    /// Logical byte offset in the uncompressed stream.
    pub logical_offset: u64,
    /// Physical byte offset of encoded payload in the transport.
    pub payload_offset: u64,
    /// Number of logical bytes represented by this chunk.
    pub logical_len: u32,
    /// Number of physical payload bytes.
    pub payload_len: u32,
    /// Whether payload is raw (`false`) or canonical Huffman (`true`).
    pub huffman: bool,
    /// BLAKE3 digest of decoded logical bytes.
    pub digest: [u8; 32],
}

/// Parsed, seekable view over an entropy transport.
#[derive(Debug)]
pub struct EntropyTransport<'a> {
    bytes: &'a [u8],
    logical_len: usize,
    chunk_size: usize,
    chunks: Vec<EntropyChunkInfo>,
}

/// File-backed parsed view over an entropy transport.
///
/// Construction reads only the fixed header and index. Payload bytes are fetched with an
/// absolute seek for each requested chunk, so callers can inspect or decode large artifacts
/// without first materializing the complete transport in memory.
#[derive(Debug)]
pub struct SeekableEntropyTransport<R> {
    source: R,
    logical_len: usize,
    chunk_size: usize,
    chunks: Vec<EntropyChunkInfo>,
}

/// Errors emitted while encoding or parsing entropy transport.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EntropyTransportError {
    /// Transport magic was not recognized.
    BadMagic,
    /// Transport version is not supported.
    UnsupportedVersion(u8),
    /// Header or index reserved bytes were non-zero.
    NonZeroReserved,
    /// Chunk size is not within the supported bounded range.
    InvalidChunkSize(usize),
    /// A declared integer could not be represented or overflowed a calculation.
    LengthOverflow,
    /// Header/index/payload bytes were truncated or inconsistent.
    Truncated,
    /// Index entries were not canonical contiguous chunks.
    NonCanonicalIndex,
    /// Chunk mode is unknown.
    UnknownMode(u8),
    /// Huffman table is malformed or not prefix-free.
    InvalidHuffmanTable,
    /// Huffman bitstream is malformed.
    InvalidHuffmanStream,
    /// Decoded chunk length differs from its index declaration.
    DecodedLengthMismatch,
    /// Decoded bytes failed their content digest.
    DigestMismatch,
    /// Requested chunk does not exist.
    ChunkOutOfRange(usize),
    /// Requested logical range lies outside the stream.
    RangeOutOfRange,
}

impl fmt::Display for EntropyTransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "entropy transport: bad magic"),
            Self::UnsupportedVersion(v) => write!(f, "entropy transport: unsupported version {v}"),
            Self::NonZeroReserved => write!(f, "entropy transport: reserved bytes are non-zero"),
            Self::InvalidChunkSize(n) => write!(f, "entropy transport: invalid chunk size {n}"),
            Self::LengthOverflow => write!(f, "entropy transport: length overflow"),
            Self::Truncated => write!(f, "entropy transport: truncated or inconsistent bytes"),
            Self::NonCanonicalIndex => write!(f, "entropy transport: non-canonical index"),
            Self::UnknownMode(m) => write!(f, "entropy transport: unknown chunk mode {m}"),
            Self::InvalidHuffmanTable => write!(f, "entropy transport: invalid Huffman table"),
            Self::InvalidHuffmanStream => write!(f, "entropy transport: invalid Huffman stream"),
            Self::DecodedLengthMismatch => write!(f, "entropy transport: decoded length mismatch"),
            Self::DigestMismatch => write!(f, "entropy transport: chunk digest mismatch"),
            Self::ChunkOutOfRange(i) => write!(f, "entropy transport: chunk {i} out of range"),
            Self::RangeOutOfRange => write!(f, "entropy transport: logical range out of range"),
        }
    }
}

impl std::error::Error for EntropyTransportError {}

/// Encode bytes using the default 64 KiB independently seekable chunks.
pub fn write_entropy_transport(bytes: &[u8]) -> Result<Vec<u8>, EntropyTransportError> {
    write_entropy_transport_with_chunk_size(bytes, ENTROPY_TRANSPORT_DEFAULT_CHUNK_BYTES)
}

/// Encode bytes using independently seekable chunks of `chunk_size` bytes.
pub fn write_entropy_transport_with_chunk_size(
    bytes: &[u8],
    chunk_size: usize,
) -> Result<Vec<u8>, EntropyTransportError> {
    validate_chunk_size(chunk_size)?;
    let chunk_count = bytes.len().div_ceil(chunk_size);
    let index_bytes = chunk_count
        .checked_mul(INDEX_ENTRY_BYTES)
        .ok_or(EntropyTransportError::LengthOverflow)?;
    let payload_start = HEADER_BYTES
        .checked_add(index_bytes)
        .ok_or(EntropyTransportError::LengthOverflow)?;
    let mut encoded_chunks = Vec::with_capacity(chunk_count);
    let mut infos = Vec::with_capacity(chunk_count);
    let mut payload_len = 0usize;
    for (index, logical) in bytes.chunks(chunk_size).enumerate() {
        let (huffman, payload) = encode_chunk(logical)?;
        let payload_offset = payload_start
            .checked_add(payload_len)
            .ok_or(EntropyTransportError::LengthOverflow)?;
        payload_len = payload_len
            .checked_add(payload.len())
            .ok_or(EntropyTransportError::LengthOverflow)?;
        let logical_offset = index
            .checked_mul(chunk_size)
            .ok_or(EntropyTransportError::LengthOverflow)?;
        let logical_offset =
            u64::try_from(logical_offset).map_err(|_| EntropyTransportError::LengthOverflow)?;
        infos.push(EntropyChunkInfo {
            logical_offset,
            payload_offset: u64::try_from(payload_offset)
                .map_err(|_| EntropyTransportError::LengthOverflow)?,
            logical_len: u32::try_from(logical.len())
                .map_err(|_| EntropyTransportError::LengthOverflow)?,
            payload_len: u32::try_from(payload.len())
                .map_err(|_| EntropyTransportError::LengthOverflow)?,
            huffman,
            digest: digest(logical),
        });
        encoded_chunks.push(payload);
    }
    let mut out = Vec::with_capacity(payload_start + payload_len);
    out.extend_from_slice(&ENTROPY_TRANSPORT_MAGIC);
    out.push(ENTROPY_TRANSPORT_VERSION);
    out.push(0);
    out.push(chunk_size.trailing_zeros() as u8);
    out.push(0);
    out.extend_from_slice(
        &u64::try_from(bytes.len())
            .map_err(|_| EntropyTransportError::LengthOverflow)?
            .to_le_bytes(),
    );
    out.extend_from_slice(
        &u32::try_from(chunk_count)
            .map_err(|_| EntropyTransportError::LengthOverflow)?
            .to_le_bytes(),
    );
    out.extend_from_slice(&u64::try_from(HEADER_BYTES).unwrap().to_le_bytes());
    out.extend_from_slice(
        &u64::try_from(index_bytes)
            .map_err(|_| EntropyTransportError::LengthOverflow)?
            .to_le_bytes(),
    );
    for info in &infos {
        out.extend_from_slice(&info.logical_offset.to_le_bytes());
        out.extend_from_slice(&info.payload_offset.to_le_bytes());
        out.extend_from_slice(&info.logical_len.to_le_bytes());
        out.extend_from_slice(&info.payload_len.to_le_bytes());
        out.push(if info.huffman { MODE_HUFFMAN } else { MODE_RAW });
        out.extend_from_slice(&[0u8; 3]);
        out.extend_from_slice(&info.digest);
    }
    for payload in encoded_chunks {
        out.extend_from_slice(&payload);
    }
    Ok(out)
}

/// Parse a transport without eagerly decoding all chunks.
pub fn read_entropy_transport(bytes: &[u8]) -> Result<EntropyTransport<'_>, EntropyTransportError> {
    let (logical_len, chunk_size, chunks) = parse_transport_metadata(
        bytes,
        u64::try_from(bytes.len()).map_err(|_| EntropyTransportError::LengthOverflow)?,
    )?;
    Ok(EntropyTransport {
        bytes,
        logical_len,
        chunk_size,
        chunks,
    })
}

/// Parse a file-backed transport by reading only its header and fixed index.
pub fn read_entropy_transport_seekable<R: Read + Seek>(
    mut source: R,
) -> Result<SeekableEntropyTransport<R>, EntropyTransportError> {
    let source_len = source
        .seek(SeekFrom::End(0))
        .map_err(|_| EntropyTransportError::Truncated)?;
    source
        .seek(SeekFrom::Start(0))
        .map_err(|_| EntropyTransportError::Truncated)?;
    let mut header = [0u8; HEADER_BYTES];
    source
        .read_exact(&mut header)
        .map_err(|_| EntropyTransportError::Truncated)?;
    let index_bytes = usize::try_from(read_u64(&header, 28)?)
        .map_err(|_| EntropyTransportError::LengthOverflow)?;
    let index_end = HEADER_BYTES
        .checked_add(index_bytes)
        .ok_or(EntropyTransportError::LengthOverflow)?;
    if u64::try_from(index_end).map_err(|_| EntropyTransportError::LengthOverflow)? > source_len {
        return Err(EntropyTransportError::NonCanonicalIndex);
    }
    let mut metadata = vec![0u8; index_end];
    metadata[..HEADER_BYTES].copy_from_slice(&header);
    source
        .read_exact(&mut metadata[HEADER_BYTES..])
        .map_err(|_| EntropyTransportError::Truncated)?;
    let (logical_len, chunk_size, chunks) = parse_transport_metadata(&metadata, source_len)?;
    Ok(SeekableEntropyTransport {
        source,
        logical_len,
        chunk_size,
        chunks,
    })
}

fn parse_transport_metadata(
    bytes: &[u8],
    source_len: u64,
) -> Result<(usize, usize, Vec<EntropyChunkInfo>), EntropyTransportError> {
    if bytes.len() < HEADER_BYTES || bytes[..4] != ENTROPY_TRANSPORT_MAGIC {
        return Err(EntropyTransportError::BadMagic);
    }
    if bytes[4] != ENTROPY_TRANSPORT_VERSION {
        return Err(EntropyTransportError::UnsupportedVersion(bytes[4]));
    }
    if bytes[5] != 0 || bytes[7] != 0 {
        return Err(EntropyTransportError::NonZeroReserved);
    }
    let chunk_log2 = bytes[6];
    if chunk_log2 >= usize::BITS as u8 {
        return Err(EntropyTransportError::InvalidChunkSize(usize::MAX));
    }
    let chunk_size = 1usize << chunk_log2;
    validate_chunk_size(chunk_size)?;
    let logical_len =
        usize::try_from(read_u64(bytes, 8)?).map_err(|_| EntropyTransportError::LengthOverflow)?;
    let chunk_count =
        usize::try_from(read_u32(bytes, 16)?).map_err(|_| EntropyTransportError::LengthOverflow)?;
    let index_offset =
        usize::try_from(read_u64(bytes, 20)?).map_err(|_| EntropyTransportError::LengthOverflow)?;
    let index_bytes =
        usize::try_from(read_u64(bytes, 28)?).map_err(|_| EntropyTransportError::LengthOverflow)?;
    let expected_index_bytes = chunk_count
        .checked_mul(INDEX_ENTRY_BYTES)
        .ok_or(EntropyTransportError::LengthOverflow)?;
    if index_offset != HEADER_BYTES
        || index_bytes != expected_index_bytes
        || index_offset
            .checked_add(index_bytes)
            .ok_or(EntropyTransportError::LengthOverflow)?
            > bytes.len()
    {
        return Err(EntropyTransportError::NonCanonicalIndex);
    }
    let payload_start = index_offset + index_bytes;
    let expected_chunks = logical_len.div_ceil(chunk_size);
    if chunk_count != expected_chunks {
        return Err(EntropyTransportError::NonCanonicalIndex);
    }
    let mut chunks = Vec::with_capacity(chunk_count);
    let mut logical_cursor = 0u64;
    let mut payload_cursor =
        u64::try_from(payload_start).map_err(|_| EntropyTransportError::LengthOverflow)?;
    for index in 0..chunk_count {
        let entry = index_offset + index * INDEX_ENTRY_BYTES;
        let logical_offset = read_u64(bytes, entry)?;
        let payload_offset = read_u64(bytes, entry + 8)?;
        let logical_chunk_len = read_u32(bytes, entry + 16)?;
        let payload_len = read_u32(bytes, entry + 20)?;
        let mode = bytes
            .get(entry + 24)
            .copied()
            .ok_or(EntropyTransportError::Truncated)?;
        if bytes.get(entry + 25..entry + 28) != Some(&[0, 0, 0][..]) {
            return Err(EntropyTransportError::NonZeroReserved);
        }
        let digest_bytes = bytes
            .get(entry + 28..entry + INDEX_ENTRY_BYTES)
            .ok_or(EntropyTransportError::Truncated)?;
        let digest: [u8; 32] = digest_bytes
            .try_into()
            .map_err(|_| EntropyTransportError::Truncated)?;
        let logical_cursor_usize =
            usize::try_from(logical_cursor).map_err(|_| EntropyTransportError::LengthOverflow)?;
        let expected_logical_len = logical_len
            .checked_sub(logical_cursor_usize)
            .ok_or(EntropyTransportError::NonCanonicalIndex)?
            .min(chunk_size);
        if logical_offset != logical_cursor
            || u64::from(logical_chunk_len) != u64::try_from(expected_logical_len).unwrap()
            || payload_offset != payload_cursor
            || mode > MODE_HUFFMAN
        {
            return Err(if mode > MODE_HUFFMAN {
                EntropyTransportError::UnknownMode(mode)
            } else {
                EntropyTransportError::NonCanonicalIndex
            });
        }
        let payload_end = payload_offset
            .checked_add(u64::from(payload_len))
            .ok_or(EntropyTransportError::LengthOverflow)?;
        if payload_end > source_len {
            return Err(EntropyTransportError::Truncated);
        }
        logical_cursor = logical_cursor
            .checked_add(u64::from(logical_chunk_len))
            .ok_or(EntropyTransportError::LengthOverflow)?;
        payload_cursor = payload_end;
        chunks.push(EntropyChunkInfo {
            logical_offset,
            payload_offset,
            logical_len: logical_chunk_len,
            payload_len,
            huffman: mode == MODE_HUFFMAN,
            digest,
        });
    }
    if logical_cursor != u64::try_from(logical_len).unwrap() || payload_cursor != source_len {
        return Err(EntropyTransportError::NonCanonicalIndex);
    }
    Ok((logical_len, chunk_size, chunks))
}

impl<'a> EntropyTransport<'a> {
    /// Number of logical bytes in the transport.
    #[must_use]
    pub const fn logical_len(&self) -> usize {
        self.logical_len
    }

    /// Independently addressable chunk size.
    #[must_use]
    pub const fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Number of chunks in the transport index.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Return indexed metadata without decoding payload bytes.
    pub fn chunk_info(&self, index: usize) -> Result<EntropyChunkInfo, EntropyTransportError> {
        self.chunks
            .get(index)
            .copied()
            .ok_or(EntropyTransportError::ChunkOutOfRange(index))
    }

    /// Decode and integrity-check one independently seekable chunk.
    pub fn read_chunk(&self, index: usize) -> Result<Vec<u8>, EntropyTransportError> {
        let info = self.chunk_info(index)?;
        let start = usize::try_from(info.payload_offset)
            .map_err(|_| EntropyTransportError::LengthOverflow)?;
        let end = start
            .checked_add(info.payload_len as usize)
            .ok_or(EntropyTransportError::LengthOverflow)?;
        let payload = self
            .bytes
            .get(start..end)
            .ok_or(EntropyTransportError::Truncated)?;
        decode_chunk(info, payload)
    }

    /// Decode and integrity-check an arbitrary logical byte range.
    pub fn read_range(&self, offset: usize, len: usize) -> Result<Vec<u8>, EntropyTransportError> {
        let end = offset
            .checked_add(len)
            .ok_or(EntropyTransportError::RangeOutOfRange)?;
        if end > self.logical_len {
            return Err(EntropyTransportError::RangeOutOfRange);
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        let first = offset / self.chunk_size;
        let last = (end - 1) / self.chunk_size;
        let mut out = Vec::with_capacity(len);
        for index in first..=last {
            let chunk = self.read_chunk(index)?;
            let chunk_start = index * self.chunk_size;
            let from = offset.saturating_sub(chunk_start);
            let to = (end - chunk_start).min(chunk.len());
            out.extend_from_slice(&chunk[from..to]);
        }
        Ok(out)
    }

    /// Decode and integrity-check the complete logical stream.
    pub fn read_all(&self) -> Result<Vec<u8>, EntropyTransportError> {
        self.read_range(0, self.logical_len)
    }
}

impl<R: Read + Seek> SeekableEntropyTransport<R> {
    /// Return the underlying source after releasing the parsed transport view.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.source
    }

    /// Number of logical bytes in the transport.
    #[must_use]
    pub const fn logical_len(&self) -> usize {
        self.logical_len
    }

    /// Independently addressable chunk size.
    #[must_use]
    pub const fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Number of chunks in the transport index.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Return indexed metadata without reading any payload bytes.
    pub fn chunk_info(&self, index: usize) -> Result<EntropyChunkInfo, EntropyTransportError> {
        self.chunks
            .get(index)
            .copied()
            .ok_or(EntropyTransportError::ChunkOutOfRange(index))
    }

    /// Seek to and decode one independently addressable chunk.
    pub fn read_chunk(&mut self, index: usize) -> Result<Vec<u8>, EntropyTransportError> {
        let info = self.chunk_info(index)?;
        let payload_len =
            usize::try_from(info.payload_len).map_err(|_| EntropyTransportError::LengthOverflow)?;
        let mut payload = vec![0u8; payload_len];
        self.source
            .seek(SeekFrom::Start(info.payload_offset))
            .map_err(|_| EntropyTransportError::Truncated)?;
        self.source
            .read_exact(&mut payload)
            .map_err(|_| EntropyTransportError::Truncated)?;
        decode_chunk(info, &payload)
    }

    /// Seek to and decode an arbitrary logical byte range.
    pub fn read_range(
        &mut self,
        offset: usize,
        len: usize,
    ) -> Result<Vec<u8>, EntropyTransportError> {
        let end = offset
            .checked_add(len)
            .ok_or(EntropyTransportError::RangeOutOfRange)?;
        if end > self.logical_len {
            return Err(EntropyTransportError::RangeOutOfRange);
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        let first = offset / self.chunk_size;
        let last = (end - 1) / self.chunk_size;
        let mut out = Vec::with_capacity(len);
        for index in first..=last {
            let chunk = self.read_chunk(index)?;
            let chunk_start = index * self.chunk_size;
            let from = offset.saturating_sub(chunk_start);
            let to = (end - chunk_start).min(chunk.len());
            out.extend_from_slice(&chunk[from..to]);
        }
        Ok(out)
    }

    /// Seek to and decode the complete logical stream.
    pub fn read_all(&mut self) -> Result<Vec<u8>, EntropyTransportError> {
        self.read_range(0, self.logical_len)
    }
}

fn decode_chunk(info: EntropyChunkInfo, payload: &[u8]) -> Result<Vec<u8>, EntropyTransportError> {
    let decoded = if info.huffman {
        decode_huffman(payload, info.logical_len as usize)?
    } else {
        if payload.len() != info.logical_len as usize {
            return Err(EntropyTransportError::DecodedLengthMismatch);
        }
        payload.to_vec()
    };
    if digest(&decoded) != info.digest {
        return Err(EntropyTransportError::DigestMismatch);
    }
    Ok(decoded)
}

fn validate_chunk_size(chunk_size: usize) -> Result<(), EntropyTransportError> {
    if !(ENTROPY_TRANSPORT_MIN_CHUNK_BYTES..=ENTROPY_TRANSPORT_MAX_CHUNK_BYTES)
        .contains(&chunk_size)
        || !chunk_size.is_power_of_two()
    {
        return Err(EntropyTransportError::InvalidChunkSize(chunk_size));
    }
    Ok(())
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn encode_chunk(bytes: &[u8]) -> Result<(bool, Vec<u8>), EntropyTransportError> {
    let lengths = huffman_lengths(bytes)?;
    let codes = canonical_codes(&lengths)?;
    let mut stream = BitWriter::default();
    for &byte in bytes {
        stream.push(&codes[byte as usize]);
    }
    let mut encoded = Vec::with_capacity(256 + 8 + stream.bytes.len());
    encoded.extend(lengths);
    encoded.extend_from_slice(&stream.bits.to_le_bytes());
    encoded.extend(stream.bytes);
    if encoded.len() < bytes.len() {
        Ok((true, encoded))
    } else {
        Ok((false, bytes.to_vec()))
    }
}

fn decode_huffman(payload: &[u8], expected_len: usize) -> Result<Vec<u8>, EntropyTransportError> {
    if payload.len() < 264 {
        return Err(EntropyTransportError::InvalidHuffmanStream);
    }
    let lengths: [u8; 256] = payload[..256]
        .try_into()
        .map_err(|_| EntropyTransportError::InvalidHuffmanTable)?;
    let bit_len = u64::from_le_bytes(payload[256..264].try_into().unwrap());
    let stream = &payload[264..];
    if bit_len > u64::try_from(stream.len()).unwrap().saturating_mul(8) {
        return Err(EntropyTransportError::InvalidHuffmanStream);
    }
    let codes = canonical_codes(&lengths)?;
    let trie = build_trie(&codes)?;
    let mut out = Vec::with_capacity(expected_len);
    let mut bit_pos = 0u64;
    for _ in 0..expected_len {
        let mut node = 0usize;
        loop {
            if bit_pos >= bit_len {
                return Err(EntropyTransportError::InvalidHuffmanStream);
            }
            let bit = (stream[(bit_pos / 8) as usize] >> (7 - (bit_pos % 8))) & 1;
            bit_pos += 1;
            let next = trie[node].children[bit as usize];
            if next < 0 {
                return Err(EntropyTransportError::InvalidHuffmanStream);
            }
            node = next as usize;
            if let Some(symbol) = trie[node].symbol {
                out.push(symbol);
                break;
            }
        }
    }
    if bit_pos != bit_len {
        return Err(EntropyTransportError::InvalidHuffmanStream);
    }
    Ok(out)
}

fn huffman_lengths(bytes: &[u8]) -> Result<[u8; 256], EntropyTransportError> {
    let mut frequencies = [0u64; 256];
    for &byte in bytes {
        frequencies[byte as usize] += 1;
    }
    let mut nodes = Vec::new();
    let mut heap = BinaryHeap::new();
    for (symbol, &frequency) in frequencies.iter().enumerate() {
        if frequency != 0 {
            let index = nodes.len();
            nodes.push(Node {
                frequency,
                min_symbol: symbol as u16,
                symbol: Some(symbol as u8),
                children: None,
            });
            heap.push(HeapNode {
                frequency,
                min_symbol: symbol as u16,
                index,
            });
        }
    }
    if heap.is_empty() {
        return Err(EntropyTransportError::InvalidHuffmanTable);
    }
    if heap.len() == 1 {
        let symbol = heap.peek().unwrap().index;
        let mut lengths = [0u8; 256];
        lengths[nodes[symbol].symbol.unwrap() as usize] = 1;
        return Ok(lengths);
    }
    while heap.len() > 1 {
        let left = heap.pop().unwrap();
        let right = heap.pop().unwrap();
        let index = nodes.len();
        nodes.push(Node {
            frequency: left
                .frequency
                .checked_add(right.frequency)
                .ok_or(EntropyTransportError::LengthOverflow)?,
            min_symbol: left.min_symbol.min(right.min_symbol),
            symbol: None,
            children: Some((left.index, right.index)),
        });
        heap.push(HeapNode {
            frequency: nodes[index].frequency,
            min_symbol: nodes[index].min_symbol,
            index,
        });
    }
    let root = heap.pop().unwrap().index;
    let mut lengths = [0u8; 256];
    assign_lengths(&nodes, root, 0, &mut lengths)?;
    Ok(lengths)
}

#[derive(Clone, Copy, Debug)]
struct Node {
    frequency: u64,
    min_symbol: u16,
    symbol: Option<u8>,
    children: Option<(usize, usize)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HeapNode {
    frequency: u64,
    min_symbol: u16,
    index: usize,
}

impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .frequency
            .cmp(&self.frequency)
            .then_with(|| other.min_symbol.cmp(&self.min_symbol))
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn assign_lengths(
    nodes: &[Node],
    index: usize,
    depth: usize,
    lengths: &mut [u8; 256],
) -> Result<(), EntropyTransportError> {
    let node = nodes
        .get(index)
        .ok_or(EntropyTransportError::InvalidHuffmanTable)?;
    if let Some(symbol) = node.symbol {
        lengths[symbol as usize] =
            u8::try_from(depth.max(1)).map_err(|_| EntropyTransportError::InvalidHuffmanTable)?;
    } else if let Some((left, right)) = node.children {
        assign_lengths(nodes, left, depth + 1, lengths)?;
        assign_lengths(nodes, right, depth + 1, lengths)?;
    } else {
        return Err(EntropyTransportError::InvalidHuffmanTable);
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Code {
    bits: Vec<u8>,
}

fn canonical_codes(lengths: &[u8; 256]) -> Result<Vec<Code>, EntropyTransportError> {
    let mut order: Vec<(u8, u8)> = lengths
        .iter()
        .enumerate()
        .filter_map(|(symbol, &len)| (len != 0).then_some((len, symbol as u8)))
        .collect();
    if order.is_empty() {
        return Err(EntropyTransportError::InvalidHuffmanTable);
    }
    order.sort_unstable();
    let mut codes = (0..256)
        .map(|_| Code { bits: Vec::new() })
        .collect::<Vec<_>>();
    let mut current = vec![0u8; order[0].0 as usize];
    let mut previous_len = order[0].0;
    for (position, &(len, symbol)) in order.iter().enumerate() {
        if position != 0 {
            if !increment_bits(&mut current) {
                return Err(EntropyTransportError::InvalidHuffmanTable);
            }
            if len < previous_len {
                return Err(EntropyTransportError::InvalidHuffmanTable);
            }
            current.resize(len as usize, 0);
        }
        codes[symbol as usize] = Code {
            bits: current.clone(),
        };
        previous_len = len;
    }
    Ok(codes)
}

fn increment_bits(bits: &mut [u8]) -> bool {
    for bit in bits.iter_mut().rev() {
        if *bit == 0 {
            *bit = 1;
            return true;
        }
        *bit = 0;
    }
    false
}

#[derive(Clone, Debug)]
struct TrieNode {
    children: [i32; 2],
    symbol: Option<u8>,
}

fn build_trie(codes: &[Code]) -> Result<Vec<TrieNode>, EntropyTransportError> {
    let mut trie = vec![TrieNode {
        children: [-1, -1],
        symbol: None,
    }];
    for (symbol, code) in codes
        .iter()
        .enumerate()
        .filter(|(_, code)| !code.bits.is_empty())
    {
        let mut node = 0usize;
        for (position, &bit) in code.bits.iter().enumerate() {
            if trie[node].symbol.is_some() {
                return Err(EntropyTransportError::InvalidHuffmanTable);
            }
            if bit > 1 {
                return Err(EntropyTransportError::InvalidHuffmanTable);
            }
            let child = trie[node].children[bit as usize];
            if child < 0 {
                let next =
                    i32::try_from(trie.len()).map_err(|_| EntropyTransportError::LengthOverflow)?;
                trie[node].children[bit as usize] = next;
                trie.push(TrieNode {
                    children: [-1, -1],
                    symbol: None,
                });
                node = next as usize;
            } else {
                node = child as usize;
            }
            if position + 1 == code.bits.len() {
                if trie[node].symbol.is_some() || trie[node].children != [-1, -1] {
                    return Err(EntropyTransportError::InvalidHuffmanTable);
                }
                trie[node].symbol = Some(symbol as u8);
            }
        }
    }
    Ok(trie)
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bits: u64,
}

impl BitWriter {
    fn push(&mut self, code: &Code) {
        for &bit in &code.bits {
            if self.bits.is_multiple_of(8) {
                self.bytes.push(0);
            }
            if bit != 0 {
                let index = self.bytes.len() - 1;
                self.bytes[index] |= 1 << (7 - (self.bits % 8));
            }
            self.bits += 1;
        }
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, EntropyTransportError> {
    let end = offset
        .checked_add(4)
        .ok_or(EntropyTransportError::LengthOverflow)?;
    bytes
        .get(offset..end)
        .ok_or(EntropyTransportError::Truncated)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, EntropyTransportError> {
    let end = offset
        .checked_add(8)
        .ok_or(EntropyTransportError::LengthOverflow)?;
    bytes
        .get(offset..end)
        .ok_or(EntropyTransportError::Truncated)
        .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Read, Seek, SeekFrom};

    #[derive(Debug)]
    struct CountingCursor {
        cursor: Cursor<Vec<u8>>,
        reads: usize,
    }

    impl Read for CountingCursor {
        fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
            let count = self.cursor.read(bytes)?;
            self.reads += count;
            Ok(count)
        }
    }

    impl Seek for CountingCursor {
        fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
            self.cursor.seek(position)
        }
    }

    #[test]
    fn deterministic_compressible_roundtrip_and_range() {
        let source = vec![0u8; 200]
            .into_iter()
            .chain((0..200).map(|n| (n % 3) as u8))
            .collect::<Vec<_>>();
        let encoded = write_entropy_transport_with_chunk_size(&source, 64).unwrap();
        assert_eq!(
            encoded,
            write_entropy_transport_with_chunk_size(&source, 64).unwrap()
        );
        let reader = read_entropy_transport(&encoded).unwrap();
        assert_eq!(reader.chunk_count(), 7);
        assert_eq!(reader.read_all().unwrap(), source);
        assert_eq!(reader.read_range(57, 149).unwrap(), source[57..206]);
        assert!(reader.chunk_info(99).is_err());
    }

    #[test]
    fn incompressible_chunk_falls_back_to_raw() {
        let source: Vec<u8> = (0..255).collect();
        let encoded = write_entropy_transport_with_chunk_size(&source, 256).unwrap();
        let reader = read_entropy_transport(&encoded).unwrap();
        assert!(!reader.chunk_info(0).unwrap().huffman);
        assert_eq!(reader.read_all().unwrap(), source);
    }

    #[test]
    fn empty_roundtrip() {
        let encoded = write_entropy_transport(&[]).unwrap();
        let reader = read_entropy_transport(&encoded).unwrap();
        assert_eq!(reader.logical_len(), 0);
        assert_eq!(reader.chunk_count(), 0);
        assert_eq!(reader.read_all().unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn malformed_and_tampered_payloads_fail_closed() {
        let mut encoded = write_entropy_transport_with_chunk_size(&[7; 128], 64).unwrap();
        encoded[0] = b'X';
        assert!(matches!(
            read_entropy_transport(&encoded),
            Err(EntropyTransportError::BadMagic)
        ));
        let mut encoded = write_entropy_transport_with_chunk_size(&[7; 128], 64).unwrap();
        let payload = read_entropy_transport(&encoded)
            .unwrap()
            .chunk_info(0)
            .unwrap()
            .payload_offset as usize;
        encoded[payload] ^= 1;
        let reader = read_entropy_transport(&encoded).unwrap();
        assert_eq!(
            reader.read_chunk(0),
            Err(EntropyTransportError::DigestMismatch)
        );
    }

    #[test]
    fn chunk_size_is_bounded_power_of_two() {
        assert_eq!(
            write_entropy_transport_with_chunk_size(&[1], 63),
            Err(EntropyTransportError::InvalidChunkSize(63))
        );
        assert_eq!(
            write_entropy_transport_with_chunk_size(&[1], 128),
            Ok(write_entropy_transport_with_chunk_size(&[1], 128).unwrap())
        );
    }

    #[test]
    fn malformed_prefixes_fail_without_panicking() {
        let encoded = write_entropy_transport_with_chunk_size(&[3; 128], 64).unwrap();
        for end in 0..=encoded.len() {
            let result = std::panic::catch_unwind(|| read_entropy_transport(&encoded[..end]));
            assert!(result.is_ok(), "parser panicked at prefix length {end}");
        }
        let mut malformed = encoded;
        malformed[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            read_entropy_transport(&malformed),
            Err(EntropyTransportError::NonCanonicalIndex | EntropyTransportError::Truncated)
        ));
    }

    #[test]
    fn file_backed_reader_loads_index_then_only_selected_payloads() {
        let source = vec![0u8; 192]
            .into_iter()
            .chain((0..192).map(|n| n as u8))
            .collect::<Vec<_>>();
        let encoded = write_entropy_transport_with_chunk_size(&source, 64).unwrap();
        let mut reader = read_entropy_transport_seekable(CountingCursor {
            cursor: Cursor::new(encoded.clone()),
            reads: 0,
        })
        .unwrap();
        assert_eq!(reader.chunk_count(), 6);
        let header_and_index = HEADER_BYTES + 6 * INDEX_ENTRY_BYTES;
        assert_eq!(reader.source.reads, header_and_index);
        let info = reader.chunk_info(4).unwrap();
        let before = reader.source.reads;
        assert_eq!(reader.read_chunk(4).unwrap(), source[256..320]);
        assert_eq!(reader.source.reads - before, info.payload_len as usize);
        assert_eq!(reader.read_range(17, 141).unwrap(), source[17..158]);
        assert_eq!(reader.read_all().unwrap(), source);
    }

    #[test]
    fn file_backed_reader_rejects_truncated_source_before_payload_read() {
        let encoded = write_entropy_transport_with_chunk_size(&[5; 128], 64).unwrap();
        let truncated = encoded[..encoded.len() - 1].to_vec();
        assert!(matches!(
            read_entropy_transport_seekable(Cursor::new(truncated)),
            Err(EntropyTransportError::NonCanonicalIndex | EntropyTransportError::Truncated)
        ));
    }
}

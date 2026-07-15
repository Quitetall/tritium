//! TQ2_0 **residual sidecar** — the on-disk container for SALT multi-plane
//! weights (ADR 0001/0006).
//!
//! A SALT-quantized row is a sum of `T` ternary planes,
//! `W ≈ Σ_p scale_p · trit_p` (ADR 0001 §1). Each plane is a *standard* TQ2_0
//! row — per-256-block `f16` scales over `K` trits — so the sidecar invents no
//! new trit packing: it reuses [`pack_tq2_0_row`]/[`unpack_tq2_0_row`] unchanged
//! and a `T = 1` SALT row is byte-identical to a legacy plain-TQ2 row. The group
//! granularity is therefore one 256-block: a plane's per-group AbsMean scale is
//! exactly that block's `f16` scale.
//!
//! Both row versions start with a 10-byte little-endian header: magic `b"TSLT"`,
//! `u8` version, `u8` plane count `T`, and `u32` row length `K`. Legacy v1 then
//! stores `T` dense TQ2_0 planes back to back. Progressive v2 adds one tag/length
//! descriptor per plane and may encode residuals sparsely; plane 0 stays dense.
//!
//! Back-compat is explicit, not sniffed: a caller that knows (from the tensor's
//! declared type) it holds legacy plain-TQ2 calls [`read_legacy_as_salt`] to wrap
//! those bytes as a one-plane row — a pre-SALT model loads as flat AbsMean.

use core::mem::size_of;

use half::f16;
use tritium_core::Trit;

use crate::{
    FormatError, PlaneRepr, QK_K, SparsePlaneRef, TQ2_0_BLOCK_BYTES, choose_plane_repr,
    expand_plane_repr, num_blocks, pack_sparse_plane, sparse::validate_sparse_plane,
    sparse_to_tq2_0, unpack_tq2_0_row,
};

/// Sidecar magic: `b"TSLT"` (Tritium SALT).
pub const SALT_MAGIC: [u8; 4] = *b"TSLT";

/// Legacy dense row version written by [`pack_salt_row`].
pub const SALT_VERSION: u8 = 1;

/// Progressive row version: framed dense base plus dense-or-sparse residual payloads.
pub const SALT_PROGRESSIVE_VERSION: u8 = 2;

/// Default residual-density cutoff used by the progressive bundle writer.
pub const DEFAULT_SPARSE_RESIDUAL_DENSITY: f32 = 0.10;

/// Header size: magic(4) + version(1) + plane-count(1) + K as `u32`(4) = 10.
pub const SALT_HEADER_BYTES: usize = 4 + 1 + 1 + 4;

const PLANE_TAG_DENSE: u8 = 0;
const PLANE_TAG_SPARSE: u8 = 1;
const PLANE_DESCRIPTOR_BYTES: usize = 1 + 4;

/// One SALT-quantized row: `T` ternary planes that sum to the dequantized weight.
///
/// `planes[0]` is the dense base; `planes[1..]` are residual planes. Each plane
/// is `num_blocks(k) · `[`TQ2_0_BLOCK_BYTES`] bytes of TQ2_0. An empty `planes`
/// is a fully pruned row (dequantizes to all-zero).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SaltRow {
    /// Trits per plane (the row length `K`).
    pub k: usize,
    /// The `T` planes, each a TQ2_0 row of `num_blocks(k)` blocks.
    pub planes: Vec<Vec<u8>>,
}

impl SaltRow {
    /// Realized plane count `T`.
    #[inline]
    pub fn plane_count(&self) -> usize {
        self.planes.len()
    }

    /// Expected byte length of each plane for this `k`.
    #[inline]
    fn plane_bytes(&self) -> usize {
        num_blocks(self.k) * TQ2_0_BLOCK_BYTES
    }
}

/// One SALT row retaining each plane in its selected dense or sparse storage form.
///
/// Unlike [`SaltRow`], this representation does not expand progressive sparse residuals
/// into dense TQ2_0 bytes. Fields stay private so every instance has validated plane
/// geometry, a dense base plane when non-empty, and canonical sparse coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedSaltRow {
    k: usize,
    planes: Vec<PlaneRepr>,
}

impl PackedSaltRow {
    /// Validate and retain one row of adaptive dense/sparse planes.
    ///
    /// An empty plane list is a valid all-zero row. A non-empty row must begin with
    /// a dense plane; only residual planes may be sparse.
    ///
    /// # Errors
    /// Returns a typed format error for oversized rows, malformed dense payloads,
    /// a sparse base plane, or invalid sparse geometry/coordinates/signs.
    pub fn new(k: usize, planes: Vec<PlaneRepr>) -> Result<Self, FormatError> {
        if planes.len() > u8::MAX as usize {
            return Err(FormatError::SaltTooManyPlanes(planes.len()));
        }
        if k > u32::MAX as usize {
            return Err(FormatError::SaltRowTooLong(k));
        }
        if matches!(planes.first(), Some(PlaneRepr::Sparse(_))) {
            return Err(FormatError::SaltSparseBasePlane);
        }
        let plane_bytes =
            num_blocks(k)
                .checked_mul(TQ2_0_BLOCK_BYTES)
                .ok_or(FormatError::WrongBlockLen {
                    expected: usize::MAX,
                    got: k,
                })?;
        for plane in &planes {
            match plane {
                PlaneRepr::Dense(bytes) => validate_dense_plane(bytes, plane_bytes)?,
                PlaneRepr::Sparse(sparse) => {
                    if sparse.k != k {
                        return Err(FormatError::WrongBlockLen {
                            expected: k,
                            got: sparse.k,
                        });
                    }
                    validate_sparse_plane(sparse)?;
                }
            }
        }
        Ok(Self { k, planes })
    }

    /// Logical row length `K`.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Realized additive plane count.
    #[must_use]
    pub fn plane_count(&self) -> usize {
        self.planes.len()
    }

    /// Validated plane representations in additive order.
    #[must_use]
    pub fn planes(&self) -> &[PlaneRepr] {
        &self.planes
    }

    /// Number of residual planes retained sparsely.
    #[must_use]
    pub fn sparse_plane_count(&self) -> usize {
        self.planes
            .iter()
            .filter(|plane| matches!(plane, PlaneRepr::Sparse(_)))
            .count()
    }

    /// Payload bytes represented by the retained dense bytes, sparse scales, and
    /// packed signed coordinates. Container metadata and allocator overhead are excluded.
    #[must_use]
    pub fn resident_payload_bytes(&self) -> usize {
        self.planes.iter().fold(0usize, |total, plane| {
            let bytes = match plane {
                PlaneRepr::Dense(bytes) => bytes.len(),
                PlaneRepr::Sparse(sparse) => sparse
                    .scales
                    .len()
                    .saturating_mul(size_of::<f16>())
                    .saturating_add(sparse.idx.len().saturating_mul(size_of::<u32>()))
                    .saturating_add(sparse.sign.len().saturating_mul(size_of::<i8>())),
            };
            total.saturating_add(bytes)
        })
    }

    /// Expand sparse residuals into the legacy all-dense [`SaltRow`] representation.
    ///
    /// # Errors
    /// Returns a format error if a retained sparse plane cannot be reconstructed.
    pub fn to_dense(&self) -> Result<SaltRow, FormatError> {
        let planes = self
            .planes
            .iter()
            .map(expand_plane_repr)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SaltRow { k: self.k, planes })
    }

    /// Consume this row and expand only its sparse residuals into legacy dense planes.
    /// Dense plane allocations are moved directly into the returned [`SaltRow`].
    ///
    /// # Errors
    /// Returns a format error if a retained sparse plane cannot be reconstructed.
    pub fn into_dense(self) -> Result<SaltRow, FormatError> {
        let planes = self
            .planes
            .into_iter()
            .map(|plane| match plane {
                PlaneRepr::Dense(bytes) => Ok(bytes),
                PlaneRepr::Sparse(sparse) => sparse_to_tq2_0(&sparse),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SaltRow { k: self.k, planes })
    }

    fn from_validated(k: usize, planes: Vec<PlaneRepr>) -> Self {
        Self { k, planes }
    }
}

impl TryFrom<SaltRow> for PackedSaltRow {
    type Error = FormatError;

    fn try_from(row: SaltRow) -> Result<Self, Self::Error> {
        Self::new(
            row.k,
            row.planes.into_iter().map(PlaneRepr::Dense).collect(),
        )
    }
}

/// Serialize a [`SaltRow`]: header + each plane's TQ2_0 bytes in plane order.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] if any plane is not `num_blocks(k)·66` bytes;
/// [`FormatError::SaltTooManyPlanes`] if `T` does not fit a `u8`;
/// [`FormatError::SaltRowTooLong`] if `k` does not fit a `u32`.
pub fn pack_salt_row(row: &SaltRow) -> Result<Vec<u8>, FormatError> {
    let plane_bytes = validate_salt_row(row)?;
    let mut out = Vec::with_capacity(SALT_HEADER_BYTES + row.planes.len() * plane_bytes);
    out.extend_from_slice(&SALT_MAGIC);
    out.push(SALT_VERSION);
    out.push(row.planes.len() as u8);
    out.extend_from_slice(&(row.k as u32).to_le_bytes());
    for plane in &row.planes {
        out.extend_from_slice(plane);
    }
    Ok(out)
}

/// Serialize a progressive SALT row with a dense base and compact residual planes.
///
/// Residual planes below `max_sparse_density` use [`crate::SparsePlane`] only when
/// its encoded payload is also smaller than dense TQ2_0. Other residuals remain
/// dense. Plane order stays unchanged, so decoding all planes is byte-identical to
/// the source row and decoding a prefix yields an additive lower-memory tier.
///
/// # Errors
/// Same shape/size errors as [`pack_salt_row`], plus
/// [`FormatError::InvalidSparseDensity`] when threshold is non-finite or outside
/// `[0, 1]`, and sparse-codec errors for malformed plane bytes.
pub fn pack_progressive_salt_row(
    row: &SaltRow,
    max_sparse_density: f32,
) -> Result<Vec<u8>, FormatError> {
    if !max_sparse_density.is_finite() || !(0.0..=1.0).contains(&max_sparse_density) {
        return Err(FormatError::InvalidSparseDensity);
    }
    let plane_bytes = validate_salt_row(row)?;
    let mut encoded = Vec::with_capacity(row.planes.len());
    for (index, plane) in row.planes.iter().enumerate() {
        if index == 0 {
            encoded.push((PLANE_TAG_DENSE, plane.clone()));
            continue;
        }
        match choose_plane_repr(plane, row.k, max_sparse_density)? {
            PlaneRepr::Dense(bytes) => encoded.push((PLANE_TAG_DENSE, bytes)),
            PlaneRepr::Sparse(sparse) => {
                let bytes = pack_sparse_plane(&sparse)?;
                if bytes.len() < plane_bytes {
                    encoded.push((PLANE_TAG_SPARSE, bytes));
                } else {
                    encoded.push((PLANE_TAG_DENSE, plane.clone()));
                }
            }
        }
    }

    let descriptor_bytes = encoded
        .len()
        .checked_mul(PLANE_DESCRIPTOR_BYTES)
        .ok_or(FormatError::SaltTooManyPlanes(encoded.len()))?;
    let payload_bytes = encoded.iter().try_fold(0usize, |total, (_, bytes)| {
        total
            .checked_add(bytes.len())
            .ok_or(FormatError::WrongBlockLen {
                expected: usize::MAX,
                got: total,
            })
    })?;
    let capacity = SALT_HEADER_BYTES
        .checked_add(descriptor_bytes)
        .and_then(|n| n.checked_add(payload_bytes))
        .ok_or(FormatError::WrongBlockLen {
            expected: usize::MAX,
            got: payload_bytes,
        })?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&SALT_MAGIC);
    out.push(SALT_PROGRESSIVE_VERSION);
    out.push(encoded.len() as u8);
    out.extend_from_slice(&(row.k as u32).to_le_bytes());
    for (tag, bytes) in &encoded {
        let len = u32::try_from(bytes.len()).map_err(|_| FormatError::WrongBlockLen {
            expected: u32::MAX as usize,
            got: bytes.len(),
        })?;
        out.push(*tag);
        out.extend_from_slice(&len.to_le_bytes());
    }
    for (_, bytes) in encoded {
        out.extend_from_slice(&bytes);
    }
    Ok(out)
}

fn validate_salt_row(row: &SaltRow) -> Result<usize, FormatError> {
    if row.planes.len() > u8::MAX as usize {
        return Err(FormatError::SaltTooManyPlanes(row.planes.len()));
    }
    if row.k > u32::MAX as usize {
        return Err(FormatError::SaltRowTooLong(row.k));
    }
    let plane_bytes = row.plane_bytes();
    for plane in &row.planes {
        validate_dense_plane(plane, plane_bytes)?;
    }
    Ok(plane_bytes)
}

#[derive(Clone, Copy, Debug)]
struct SaltRowLayout {
    version: u8,
    planes: usize,
    k: usize,
    plane_bytes: usize,
    payload_start: usize,
    encoded_len: usize,
}

fn read_plane_descriptor(bytes: &[u8], index: usize) -> (u8, usize) {
    let offset = SALT_HEADER_BYTES + index * PLANE_DESCRIPTOR_BYTES;
    let len = u32::from_le_bytes(
        bytes[offset + 1..offset + PLANE_DESCRIPTOR_BYTES]
            .try_into()
            .expect("descriptor bounds validated"),
    ) as usize;
    (bytes[offset], len)
}

fn parse_salt_row_layout(bytes: &[u8]) -> Result<SaltRowLayout, FormatError> {
    let (version, planes, k) = read_header(bytes)?;
    let plane_bytes =
        num_blocks(k)
            .checked_mul(TQ2_0_BLOCK_BYTES)
            .ok_or(FormatError::WrongBlockLen {
                expected: usize::MAX,
                got: bytes.len(),
            })?;
    let (payload_start, encoded_len) =
        match version {
            SALT_VERSION => {
                let payload_bytes =
                    planes
                        .checked_mul(plane_bytes)
                        .ok_or(FormatError::WrongBlockLen {
                            expected: usize::MAX,
                            got: bytes.len(),
                        })?;
                let encoded_len = SALT_HEADER_BYTES.checked_add(payload_bytes).ok_or(
                    FormatError::WrongBlockLen {
                        expected: usize::MAX,
                        got: bytes.len(),
                    },
                )?;
                (SALT_HEADER_BYTES, encoded_len)
            }
            SALT_PROGRESSIVE_VERSION => {
                let descriptor_bytes = planes.checked_mul(PLANE_DESCRIPTOR_BYTES).ok_or(
                    FormatError::WrongBlockLen {
                        expected: usize::MAX,
                        got: bytes.len(),
                    },
                )?;
                let payload_start = SALT_HEADER_BYTES.checked_add(descriptor_bytes).ok_or(
                    FormatError::WrongBlockLen {
                        expected: usize::MAX,
                        got: bytes.len(),
                    },
                )?;
                if payload_start > bytes.len() {
                    return Err(FormatError::WrongBlockLen {
                        expected: payload_start,
                        got: bytes.len(),
                    });
                }
                let mut payload_bytes = 0usize;
                for index in 0..planes {
                    let (tag, len) = read_plane_descriptor(bytes, index);
                    validate_plane_tag(tag, index)?;
                    payload_bytes =
                        payload_bytes
                            .checked_add(len)
                            .ok_or(FormatError::WrongBlockLen {
                                expected: usize::MAX,
                                got: bytes.len(),
                            })?;
                }
                let encoded_len =
                    payload_start
                        .checked_add(payload_bytes)
                        .ok_or(FormatError::WrongBlockLen {
                            expected: usize::MAX,
                            got: bytes.len(),
                        })?;
                (payload_start, encoded_len)
            }
            other => return Err(FormatError::UnsupportedSaltVersion(other)),
        };
    if encoded_len > bytes.len() {
        return Err(FormatError::WrongBlockLen {
            expected: encoded_len,
            got: bytes.len(),
        });
    }
    Ok(SaltRowLayout {
        version,
        planes,
        k,
        plane_bytes,
        payload_start,
        encoded_len,
    })
}

/// Allocation-free validated view of one dense or sparse SALT plane.
#[derive(Clone, Copy, Debug)]
pub struct PackedSaltPlaneRef<'a> {
    repr: PackedSaltPlaneRefRepr<'a>,
}

#[derive(Clone, Copy, Debug)]
enum PackedSaltPlaneRefRepr<'a> {
    Dense(&'a [u8]),
    Sparse(SparsePlaneRef<'a>),
}

impl<'a> PackedSaltPlaneRef<'a> {
    /// Dense TQ2_0 payload, or `None` when this plane is sparse.
    #[must_use]
    pub const fn dense_bytes(self) -> Option<&'a [u8]> {
        match self.repr {
            PackedSaltPlaneRefRepr::Dense(bytes) => Some(bytes),
            PackedSaltPlaneRefRepr::Sparse(_) => None,
        }
    }

    /// Sparse payload view, or `None` when this plane is dense.
    #[must_use]
    pub const fn sparse(self) -> Option<SparsePlaneRef<'a>> {
        match self.repr {
            PackedSaltPlaneRefRepr::Dense(_) => None,
            PackedSaltPlaneRefRepr::Sparse(sparse) => Some(sparse),
        }
    }
}

/// Allocation-free validated view of one encoded SALT row.
///
/// Plane payloads borrow the encoded row. Callers can copy them directly into final
/// storage while a seek-backed reader reuses one row-sized scratch allocation.
#[derive(Clone, Copy, Debug)]
pub struct PackedSaltRowRef<'a> {
    bytes: &'a [u8],
    layout: SaltRowLayout,
}

impl<'a> PackedSaltRowRef<'a> {
    /// Parse and fully validate one exact encoded row without allocating.
    ///
    /// # Errors
    /// Returns the same malformed-row errors as [`unpack_packed_salt_row`].
    pub fn parse(bytes: &'a [u8]) -> Result<Self, FormatError> {
        let layout = parse_salt_row_layout(bytes)?;
        if bytes.len() != layout.encoded_len {
            return Err(FormatError::WrongBlockLen {
                expected: layout.encoded_len,
                got: bytes.len(),
            });
        }
        let mut payload_offset = layout.payload_start;
        for index in 0..layout.planes {
            let (tag, len) = if layout.version == SALT_VERSION {
                (PLANE_TAG_DENSE, layout.plane_bytes)
            } else {
                read_plane_descriptor(bytes, index)
            };
            let payload = &bytes[payload_offset..payload_offset + len];
            if tag == PLANE_TAG_DENSE {
                validate_dense_plane(payload, layout.plane_bytes)?;
            } else {
                let sparse = SparsePlaneRef::parse(payload)?;
                if sparse.k() != layout.k {
                    return Err(FormatError::WrongBlockLen {
                        expected: layout.k,
                        got: sparse.k(),
                    });
                }
            }
            payload_offset += len;
        }
        Ok(Self { bytes, layout })
    }

    /// Logical row length.
    #[must_use]
    pub const fn k(self) -> usize {
        self.layout.k
    }

    /// Number of retained additive planes.
    #[must_use]
    pub const fn plane_count(self) -> usize {
        self.layout.planes
    }

    /// Exact encoded bytes backing this validated row view.
    #[must_use]
    pub const fn encoded_bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Validated planes in additive order.
    pub fn planes(self) -> PackedSaltPlaneRefs<'a> {
        PackedSaltPlaneRefs {
            row: self,
            index: 0,
            payload_offset: self.layout.payload_start,
        }
    }
}

/// Iterator over allocation-free plane views in one [`PackedSaltRowRef`].
#[derive(Clone, Debug)]
pub struct PackedSaltPlaneRefs<'a> {
    row: PackedSaltRowRef<'a>,
    index: usize,
    payload_offset: usize,
}

impl<'a> Iterator for PackedSaltPlaneRefs<'a> {
    type Item = PackedSaltPlaneRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.row.layout.planes {
            return None;
        }
        let (tag, len) = if self.row.layout.version == SALT_VERSION {
            (PLANE_TAG_DENSE, self.row.layout.plane_bytes)
        } else {
            read_plane_descriptor(self.row.bytes, self.index)
        };
        let start = self.payload_offset;
        let end = start + len;
        self.index += 1;
        self.payload_offset = end;
        let payload = &self.row.bytes[start..end];
        Some(PackedSaltPlaneRef {
            repr: match tag {
                PLANE_TAG_DENSE => PackedSaltPlaneRefRepr::Dense(payload),
                PLANE_TAG_SPARSE => {
                    PackedSaltPlaneRefRepr::Sparse(SparsePlaneRef::from_validated(payload))
                }
                _ => unreachable!("row validation covered plane tag"),
            },
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.row.layout.planes - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PackedSaltPlaneRefs<'_> {}

/// Return encoded length of first SALT row in `bytes` without consuming later rows.
///
/// Supports legacy dense v1 and progressive framed v2. Returned length is bounded
/// by `bytes.len()`, making this safe for walking concatenated bundle payloads.
///
/// # Errors
/// Returns typed format errors for bad magic/version, invalid representation tags,
/// arithmetic overflow, or truncated descriptors/payloads.
pub fn packed_salt_row_len(bytes: &[u8]) -> Result<usize, FormatError> {
    Ok(parse_salt_row_layout(bytes)?.encoded_len)
}

fn read_header(bytes: &[u8]) -> Result<(u8, usize, usize), FormatError> {
    if bytes.len() < SALT_HEADER_BYTES {
        return Err(FormatError::WrongBlockLen {
            expected: SALT_HEADER_BYTES,
            got: bytes.len(),
        });
    }
    if bytes[0..4] != SALT_MAGIC {
        return Err(FormatError::SaltBadMagic);
    }
    let version = bytes[4];
    let planes = bytes[5] as usize;
    let k = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]) as usize;
    Ok((version, planes, k))
}

fn validate_plane_tag(tag: u8, index: usize) -> Result<(), FormatError> {
    match tag {
        PLANE_TAG_DENSE => Ok(()),
        PLANE_TAG_SPARSE if index == 0 => Err(FormatError::SaltSparseBasePlane),
        PLANE_TAG_SPARSE => Ok(()),
        other => Err(FormatError::SaltInvalidPlaneTag(other)),
    }
}

fn validate_dense_plane(payload: &[u8], expected_len: usize) -> Result<(), FormatError> {
    if payload.len() != expected_len {
        return Err(FormatError::WrongBlockLen {
            expected: expected_len,
            got: payload.len(),
        });
    }
    for block in payload.chunks_exact(TQ2_0_BLOCK_BYTES) {
        for packed in &block[..QK_K / 4] {
            for shift in [0, 2, 4, 6] {
                if (packed >> shift) & 0b11 == 0b11 {
                    return Err(FormatError::DecodedOutOfRange(2));
                }
            }
        }
    }
    Ok(())
}

/// Parse a [`SaltRow`] from sidecar bytes, enforcing magic, version, and length.
///
/// # Errors
/// [`FormatError::SaltBadMagic`] if the magic does not match;
/// [`FormatError::UnsupportedSaltVersion`] on a version this build can't read;
/// [`FormatError::WrongBlockLen`] if the buffer length disagrees with the header.
pub fn unpack_salt_row(bytes: &[u8]) -> Result<SaltRow, FormatError> {
    unpack_salt_row_prefix(bytes, usize::MAX)
}

/// Parse a SALT row while retaining progressive sparse residuals.
///
/// # Errors
/// Same malformed-input errors as [`unpack_salt_row`].
pub fn unpack_packed_salt_row(bytes: &[u8]) -> Result<PackedSaltRow, FormatError> {
    unpack_packed_salt_row_prefix(bytes, usize::MAX)
}

/// Parse a SALT row while retaining at most `max_planes` in their encoded dense/sparse form.
///
/// Every descriptor and payload is still validated, including omitted planes. Legacy v1
/// rows are represented as dense planes; progressive v2 sparse residuals remain sparse.
///
/// # Errors
/// Same errors as [`packed_salt_row_len`], plus malformed dense/sparse payload errors.
pub fn unpack_packed_salt_row_prefix(
    bytes: &[u8],
    max_planes: usize,
) -> Result<PackedSaltRow, FormatError> {
    let row = PackedSaltRowRef::parse(bytes)?;
    let keep = row.plane_count().min(max_planes);
    let mut planes = Vec::with_capacity(keep);
    for plane in row.planes().take(keep) {
        if let Some(dense) = plane.dense_bytes() {
            planes.push(PlaneRepr::Dense(dense.to_vec()));
        } else if let Some(sparse) = plane.sparse() {
            planes.push(PlaneRepr::Sparse(sparse.to_owned()));
        }
    }
    Ok(PackedSaltRow::from_validated(row.k(), planes))
}

/// Parse a SALT row while materializing at most `max_planes` additive planes.
///
/// All descriptors and payloads are still validated. Legacy v1 rows and progressive
/// v2 rows share the same returned dense [`SaltRow`] runtime representation.
///
/// # Errors
/// Same errors as [`packed_salt_row_len`], plus malformed dense/sparse payload errors.
pub fn unpack_salt_row_prefix(bytes: &[u8], max_planes: usize) -> Result<SaltRow, FormatError> {
    unpack_packed_salt_row_prefix(bytes, max_planes)?.into_dense()
}

/// Wrap a legacy plain-TQ2 row (no SALT header) as a one-plane [`SaltRow`].
///
/// Back-compat path: a model quantized before SALT has bare TQ2_0 rows; loading
/// one as `T = 1` makes it flat BitNet AbsMean, the SALT base case. The caller
/// supplies `k` (known from the tensor shape) and the raw `num_blocks(k)·66`
/// bytes.
///
/// # Errors
/// [`FormatError::WrongBlockLen`] if `tq2_row` is not `num_blocks(k)·66` bytes.
pub fn read_legacy_as_salt(tq2_row: &[u8], k: usize) -> Result<SaltRow, FormatError> {
    let plane_bytes = num_blocks(k) * TQ2_0_BLOCK_BYTES;
    if tq2_row.len() != plane_bytes {
        return Err(FormatError::WrongBlockLen {
            expected: plane_bytes,
            got: tq2_row.len(),
        });
    }
    Ok(SaltRow {
        k,
        planes: vec![tq2_row.to_vec()],
    })
}

/// Dequantize a SALT row to fp32: `Σ_p (block_scale_p · trit_p)`, summed in plane
/// order. This is the host reference the multi-plane accumulate kernel must match
/// (the GPU exit gate).
///
/// # Errors
/// Propagates any [`unpack_tq2_0_row`] error from a malformed plane.
pub fn dequant_salt_row(row: &SaltRow) -> Result<Vec<f32>, FormatError> {
    let nb = num_blocks(row.k);
    let mut acc = vec![0.0f32; row.k];
    let mut trits = vec![Trit::ZERO; row.k];
    let mut scales = vec![f16::ZERO; nb];
    for plane in &row.planes {
        unpack_tq2_0_row(plane, &mut trits, &mut scales)?;
        for (i, t) in trits.iter().enumerate() {
            acc[i] += scales[i / QK_K].to_f32() * t.to_f32();
        }
    }
    Ok(acc)
}

/// Dequantize a full SALT tensor — one [`SaltRow`] per output channel — to a **row-major
/// `[rows.len(), k]` dense fp32 matrix**, concatenating [`dequant_salt_row`] over the rows.
/// Reused to build a runnable dense projection from SALT-quantized weights (both a bundle's
/// `SaltTensor` and a live `QuantizedTensor`).
///
/// # Errors
/// Propagates [`dequant_salt_row`] errors from any malformed row plane.
pub fn salt_rows_to_dense(rows: &[SaltRow]) -> Result<Vec<f32>, FormatError> {
    let mut out = Vec::new();
    for row in rows {
        out.extend_from_slice(&dequant_salt_row(row)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack_tq2_0_row;

    /// Deterministic ternary row of length `k` from an LCG (matches rows.rs).
    fn make_trits(k: usize, seed: u64) -> Vec<Trit> {
        let mut s = seed;
        (0..k)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                Trit::from_i8(((s >> 33) % 3) as i8 - 1).unwrap()
            })
            .collect()
    }

    /// Pack `t` planes of pseudo-random trits/scales into a `SaltRow`.
    fn make_salt_row(k: usize, t: usize) -> SaltRow {
        let nb = num_blocks(k);
        let planes = (0..t)
            .map(|p| {
                let trits = make_trits(k, 0x51A17 ^ ((p as u64) << 8) ^ k as u64);
                let scales: Vec<f16> = (0..nb)
                    .map(|b| f16::from_f32((0.5 + b as f32) / (p as f32 + 1.0)))
                    .collect();
                let mut bytes = vec![0u8; nb * TQ2_0_BLOCK_BYTES];
                pack_tq2_0_row(&trits, &scales, &mut bytes).unwrap();
                bytes
            })
            .collect();
        SaltRow { k, planes }
    }

    fn make_sparse_residual(k: usize, stride: usize) -> Vec<u8> {
        let trits: Vec<Trit> = (0..k)
            .map(|i| {
                if i % stride == 0 {
                    Trit::from_i8(if (i / stride).is_multiple_of(2) {
                        1
                    } else {
                        -1
                    })
                    .unwrap()
                } else {
                    Trit::ZERO
                }
            })
            .collect();
        let scales = vec![f16::from_f32(0.125); num_blocks(k)];
        let mut packed = vec![0u8; num_blocks(k) * TQ2_0_BLOCK_BYTES];
        pack_tq2_0_row(&trits, &scales, &mut packed).unwrap();
        packed
    }

    #[test]
    fn progressive_row_is_smaller_exact_and_prefix_loadable() {
        let k = 4096;
        let mut row = make_salt_row(k, 1);
        row.planes.push(make_sparse_residual(k, 64));
        row.planes.push(make_sparse_residual(k, 2));

        let legacy = pack_salt_row(&row).expect("legacy row");
        let progressive = pack_progressive_salt_row(&row, 0.10).expect("progressive row");

        assert!(progressive.len() < legacy.len());
        assert_eq!(progressive[SALT_HEADER_BYTES], PLANE_TAG_DENSE);
        assert_eq!(
            progressive[SALT_HEADER_BYTES + PLANE_DESCRIPTOR_BYTES],
            PLANE_TAG_SPARSE
        );
        assert_eq!(
            progressive[SALT_HEADER_BYTES + 2 * PLANE_DESCRIPTOR_BYTES],
            PLANE_TAG_DENSE
        );
        assert_eq!(
            packed_salt_row_len(&progressive).expect("framed length"),
            progressive.len()
        );
        assert_eq!(unpack_salt_row(&progressive).expect("full row"), row);
        for max_planes in [0, 1, 2, 3, usize::MAX] {
            let expected = SaltRow {
                k,
                planes: row.planes[..row.plane_count().min(max_planes)].to_vec(),
            };
            assert_eq!(
                unpack_salt_row_prefix(&progressive, max_planes).expect("plane prefix"),
                expected
            );
        }
    }

    #[test]
    fn legacy_v1_row_layout_stays_byte_stable() {
        let row = SaltRow {
            k: 1,
            planes: vec![vec![0xA5; TQ2_0_BLOCK_BYTES]],
        };
        let packed = pack_salt_row(&row).expect("legacy row");

        assert_eq!(&packed[..SALT_HEADER_BYTES], b"TSLT\x01\x01\x01\0\0\0");
        assert_eq!(&packed[SALT_HEADER_BYTES..], &[0xA5; TQ2_0_BLOCK_BYTES]);
    }

    #[test]
    fn progressive_row_rejects_malformed_framing() {
        let k = 4096;
        let mut row = make_salt_row(k, 1);
        row.planes.push(make_sparse_residual(k, 64));
        let packed = pack_progressive_salt_row(&row, 0.10).expect("progressive row");

        let mut sparse_base = packed.clone();
        sparse_base[SALT_HEADER_BYTES] = PLANE_TAG_SPARSE;
        assert_eq!(
            unpack_salt_row(&sparse_base),
            Err(FormatError::SaltSparseBasePlane)
        );

        let mut bad_tag = packed.clone();
        bad_tag[SALT_HEADER_BYTES + PLANE_DESCRIPTOR_BYTES] = 9;
        assert_eq!(
            unpack_salt_row(&bad_tag),
            Err(FormatError::SaltInvalidPlaneTag(9))
        );

        let payload_start = SALT_HEADER_BYTES + 2 * PLANE_DESCRIPTOR_BYTES;
        let sparse_start = payload_start + row.planes[0].len();
        let mut wrong_k = packed.clone();
        wrong_k[sparse_start + 6..sparse_start + 10]
            .copy_from_slice(&((k - 1) as u32).to_le_bytes());
        assert!(matches!(
            unpack_salt_row(&wrong_k),
            Err(FormatError::WrongBlockLen { .. })
        ));

        assert!(matches!(
            unpack_salt_row(&packed[..packed.len() - 1]),
            Err(FormatError::WrongBlockLen { .. })
        ));
        let mut trailing = packed.clone();
        trailing.push(0);
        assert!(matches!(
            unpack_salt_row(&trailing),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }

    #[test]
    fn prefix_reader_validates_omitted_dense_payloads() {
        let row = make_salt_row(256, 1);
        let mut progressive = pack_progressive_salt_row(&row, 0.10).expect("progressive row");
        let payload_start = SALT_HEADER_BYTES + PLANE_DESCRIPTOR_BYTES;
        progressive[payload_start] = 0xff; // four reserved TQ2_0 codes
        assert!(matches!(
            unpack_salt_row_prefix(&progressive, 0),
            Err(FormatError::DecodedOutOfRange(2))
        ));

        let mut legacy = pack_salt_row(&row).expect("legacy row");
        legacy[SALT_HEADER_BYTES] = 0xff;
        assert!(matches!(
            unpack_salt_row_prefix(&legacy, 0),
            Err(FormatError::DecodedOutOfRange(2))
        ));
    }

    #[test]
    fn progressive_row_preserves_nonzero_partial_padding_as_dense() {
        let k = 257;
        let mut row = make_salt_row(k, 1);
        let mut residual = make_sparse_residual(k, 64);
        let padded_code = &mut residual[TQ2_0_BLOCK_BYTES + 1];
        *padded_code = (*padded_code & !0b11) | 0b10;
        row.planes.push(residual);

        let progressive = pack_progressive_salt_row(&row, 1.0).expect("progressive row");
        assert_eq!(
            progressive[SALT_HEADER_BYTES + PLANE_DESCRIPTOR_BYTES],
            PLANE_TAG_DENSE
        );
        assert_eq!(unpack_salt_row(&progressive).expect("roundtrip"), row);
    }

    #[test]
    fn progressive_row_never_uses_larger_sparse_payload() {
        let k = 4096;
        let mut row = make_salt_row(k, 1);
        row.planes.push(make_sparse_residual(k, 12)); // ~8.3%: below 10%, above byte break-even

        let progressive = pack_progressive_salt_row(&row, 0.10).expect("progressive row");
        assert_eq!(
            progressive[SALT_HEADER_BYTES + PLANE_DESCRIPTOR_BYTES],
            PLANE_TAG_DENSE
        );
        assert_eq!(
            progressive.len(),
            pack_salt_row(&row).expect("legacy row").len()
                + row.plane_count() * PLANE_DESCRIPTOR_BYTES
        );
        assert_eq!(
            pack_progressive_salt_row(&row, f32::NAN),
            Err(FormatError::InvalidSparseDensity)
        );
        assert_eq!(
            pack_progressive_salt_row(&row, -0.01),
            Err(FormatError::InvalidSparseDensity)
        );
        assert_eq!(
            pack_progressive_salt_row(&row, 1.01),
            Err(FormatError::InvalidSparseDensity)
        );
    }

    #[test]
    fn salt_rows_to_dense_concatenates_per_row_dequant() {
        let k = 300; // 2 blocks (256 + 44)
        let rows = [
            make_salt_row(k, 2),
            make_salt_row(k, 1),
            make_salt_row(k, 3),
        ];
        let dense = salt_rows_to_dense(&rows).unwrap();
        assert_eq!(dense.len(), rows.len() * k);
        for (r, row) in rows.iter().enumerate() {
            let want = dequant_salt_row(row).unwrap();
            assert_eq!(&dense[r * k..r * k + k], &want[..], "row {r}");
        }
    }

    // ── Gate (ADR 0006): multi-plane roundtrip. ──────────────────────────────
    #[test]
    fn multiplane_roundtrip() {
        for &k in &[1usize, 255, 256, 257, 2560] {
            for t in 1..=3usize {
                let row = make_salt_row(k, t);
                let packed = pack_salt_row(&row).unwrap();
                let got = unpack_salt_row(&packed).unwrap();
                assert_eq!(got, row, "k={k} t={t} roundtrip");
                assert_eq!(got.plane_count(), t);
            }
        }
    }

    // ── Gate (ADR 0006): reads legacy plain-TQ2 (no residual) as T=1. ─────────
    #[test]
    fn legacy_plain_tq2_loads_as_t1() {
        let k = 300;
        let nb = num_blocks(k);
        let trits = make_trits(k, 0xBEEF);
        let scales: Vec<f16> = (0..nb).map(|b| f16::from_f32(1.0 + b as f32)).collect();
        let mut legacy = vec![0u8; nb * TQ2_0_BLOCK_BYTES];
        pack_tq2_0_row(&trits, &scales, &mut legacy).unwrap();

        let row = read_legacy_as_salt(&legacy, k).unwrap();
        assert_eq!(row.plane_count(), 1, "legacy → single base plane");
        // The single plane is byte-identical to the legacy row.
        assert_eq!(row.planes[0], legacy);
        // And it dequantizes to the plain-TQ2 reference (scale · trit).
        let deq = dequant_salt_row(&row).unwrap();
        for (i, t) in trits.iter().enumerate() {
            let want = scales[i / QK_K].to_f32() * t.to_f32();
            assert_eq!(deq[i].to_bits(), want.to_bits(), "elem {i}");
        }
        // A legacy row re-packs + roundtrips through the sidecar unchanged.
        let rt = unpack_salt_row(&pack_salt_row(&row).unwrap()).unwrap();
        assert_eq!(rt, row);
    }

    // ── Gate (ADR 0006): version + magic enforced. ───────────────────────────
    #[test]
    fn bad_magic_and_version_rejected() {
        let row = make_salt_row(256, 2);
        let mut packed = pack_salt_row(&row).unwrap();

        let mut bad_magic = packed.clone();
        bad_magic[0] = b'X';
        assert_eq!(unpack_salt_row(&bad_magic), Err(FormatError::SaltBadMagic));

        packed[4] = 99; // bump the version byte
        assert_eq!(
            unpack_salt_row(&packed),
            Err(FormatError::UnsupportedSaltVersion(99))
        );

        // Truncated buffer (shorter than the header) is a length error.
        assert!(matches!(
            unpack_salt_row(&[0u8; 4]),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }

    // A header claiming more planes than the buffer holds is rejected.
    #[test]
    fn truncated_plane_data_rejected() {
        let row = make_salt_row(256, 3);
        let mut packed = pack_salt_row(&row).unwrap();
        packed.truncate(packed.len() - 10); // chop into the last plane
        assert!(matches!(
            unpack_salt_row(&packed),
            Err(FormatError::WrongBlockLen { .. })
        ));
    }

    // ── Gate (ADR 0006): determinism — same row ⇒ byte-identical output. ──────
    #[test]
    fn pack_is_deterministic() {
        let row = make_salt_row(513, 3);
        assert_eq!(pack_salt_row(&row).unwrap(), pack_salt_row(&row).unwrap());
    }

    // ── Gate (ADR 0006): edge — pruned (T=0) and zero-variance planes. ───────
    #[test]
    fn edge_pruned_and_zero_variance() {
        // T=0: fully pruned row → empty stack, dequantizes to all-zero.
        let pruned = SaltRow {
            k: 256,
            planes: vec![],
        };
        let packed = pack_salt_row(&pruned).unwrap();
        assert_eq!(packed.len(), SALT_HEADER_BYTES);
        let back = unpack_salt_row(&packed).unwrap();
        assert_eq!(back, pruned);
        assert!(dequant_salt_row(&back).unwrap().iter().all(|&x| x == 0.0));

        // Zero-variance group: a base plane of all-zero trits at scale 0 → 0.
        let nb = num_blocks(256);
        let mut zero_plane = vec![0u8; nb * TQ2_0_BLOCK_BYTES];
        pack_tq2_0_row(&vec![Trit::ZERO; 256], &[f16::ZERO; 1], &mut zero_plane).unwrap();
        let zrow = SaltRow {
            k: 256,
            planes: vec![zero_plane],
        };
        assert!(dequant_salt_row(&zrow).unwrap().iter().all(|&x| x == 0.0));
    }

    // Dequant sums planes additively: a 2-plane row equals plane0 + plane1.
    #[test]
    fn dequant_is_additive_over_planes() {
        let k = 256;
        let row = make_salt_row(k, 2);
        let full = dequant_salt_row(&row).unwrap();
        let p0 = dequant_salt_row(&SaltRow {
            k,
            planes: vec![row.planes[0].clone()],
        })
        .unwrap();
        let p1 = dequant_salt_row(&SaltRow {
            k,
            planes: vec![row.planes[1].clone()],
        })
        .unwrap();
        for i in 0..k {
            assert_eq!(full[i].to_bits(), (p0[i] + p1[i]).to_bits(), "elem {i}");
        }
    }
}

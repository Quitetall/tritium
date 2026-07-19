//! **Codec bundle** (`TCDC`) — a single-file, codec-complete ternary artifact for a biosignal codec
//! (LamQuant, ADR 0030 Tier 2). Where a SALT bundle carries only 2-D transformer weights, a codec bundle
//! also carries the non-weight quantizer state a conv/FSQ codec needs: per-tensor **Conv1d geometry**,
//! the **FSQ level schedule**, learned **LSQ α**, empirical **CDF LUTs**, and the learned **rotation
//! matrix** — plus an embedded [`SaltBundle`](crate::write_salt_bundle) holding the conv weights (packed
//! 2-D as `[C_out, (C_in/groups)·K]`, the exact ternary reshape). The whole thing is blake3-checksummed.
//!
//! `SaltTensor` and the existing 2-D SALT path are left untouched; the conv weights ride inside the
//! embedded bundle, and `Conv1dTensorMeta` records the rank-3→2-D reshape so a loader reinterprets each
//! `[C_out, K_g]` SALT tensor as `[C_out, C_in/groups, K]`.
//!
//! Layout (little-endian), all sections length-prefixed, checksum over everything before it:
//! ```text
//! magic b"TCDC" (4) | version u32
//! conv:     count u32 | { name | c_in c_out k stride dilation pad_left pad_right groups (u32×8) }
//! fsq:      count u32 | { name | levels(u32 vec) | bound u8 | ste u8 | alpha f32 | seed u64 }
//! lsq:      count u32 | { name | per_channel(f32 vec) }
//! cdf:      count u32 | { name | bins u32 | values(u16 vec) }
//! rotation: dim u32 | data(dim·dim f32)          (dim == 0 ⇒ no rotation)
//! weights:  len u64 | bytes (an embedded SALT bundle)
//! checksum: blake3(all bytes above) (32)
//! ```

/// Codec-bundle magic: `b"TCDC"` (Tritium CoDeC).
pub const CODEC_BUNDLE_MAGIC: [u8; 4] = *b"TCDC";
/// Current codec-bundle format version.
pub const CODEC_BUNDLE_VERSION: u32 = 1;
/// Upper bound on a codec bundle (guards a crafted length against unbounded allocation): 1 GiB.
pub const MAX_CODEC_BUNDLE_BYTES: usize = 1 << 30;

/// Errors reading or writing a codec bundle (self-contained, mirroring `SaltV2MasterError`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecBundleError {
    /// The buffer did not start with [`CODEC_BUNDLE_MAGIC`].
    BadMagic,
    /// A format version this build cannot read.
    UnsupportedVersion(u32),
    /// The buffer was truncated (a length ran past the end).
    Truncated,
    /// The trailing blake3 checksum did not match the body.
    ChecksumMismatch,
    /// A declared length would exceed [`MAX_CODEC_BUNDLE_BYTES`].
    TooLarge,
    /// A name field was not valid UTF-8.
    InvalidUtf8,
    /// The rotation matrix `data` length did not equal `dim·dim`.
    RotationShape,
}

impl core::fmt::Display for CodecBundleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a codec bundle (bad magic)"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported codec-bundle version {v}"),
            Self::Truncated => write!(f, "codec bundle is truncated"),
            Self::ChecksumMismatch => write!(f, "codec-bundle checksum mismatch"),
            Self::TooLarge => write!(f, "codec-bundle section exceeds the size cap"),
            Self::InvalidUtf8 => write!(f, "codec-bundle name is not valid UTF-8"),
            Self::RotationShape => write!(f, "rotation data length != dim*dim"),
        }
    }
}

impl std::error::Error for CodecBundleError {}

/// Geometry of one ternary Conv1d tensor (the weights live in the embedded SALT bundle, keyed by `name`,
/// packed 2-D as `[C_out, (C_in/groups)·K]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conv1dTensorMeta {
    /// Tensor name (matches the embedded SALT bundle entry).
    pub name: String,
    /// Input channels.
    pub c_in: u32,
    /// Output channels.
    pub c_out: u32,
    /// Kernel size.
    pub k: u32,
    /// Convolution stride.
    pub stride: u32,
    /// Kernel dilation.
    pub dilation: u32,
    /// Zero-padding on the left.
    pub pad_left: u32,
    /// Zero-padding on the right.
    pub pad_right: u32,
    /// Convolution groups.
    pub groups: u32,
}

/// FSQ configuration for one quantized latent (`bound`: 0=tanh, 1=clamp; `ste`: 0=hard, 1=soft,
/// 2=stochastic; `alpha` is the soft-round anneal, `seed` the stochastic seed).
#[derive(Clone, Debug, PartialEq)]
pub struct FsqLayerMeta {
    /// Latent/tensor name.
    pub name: String,
    /// Per-channel level count `L`.
    pub levels: Vec<u32>,
    /// Bounding nonlinearity: 0 = tanh, 1 = clamp.
    pub bound: u8,
    /// STE variant: 0 = hard, 1 = soft-round, 2 = stochastic.
    pub ste: u8,
    /// Soft-round anneal coefficient (used when `ste == 1`).
    pub alpha: f32,
    /// Stochastic-rounding seed (used when `ste == 2`).
    pub seed: u64,
}

/// Learned LSQ per-output-channel step size for one tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct LsqAlpha {
    /// Tensor name (matches the embedded SALT bundle entry).
    pub name: String,
    /// One learned step size `α` per output channel.
    pub per_channel: Vec<f32>,
}

/// An empirical-CDF lookup table (`bins` entries, u16-quantized).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CdfLut {
    /// LUT name.
    pub name: String,
    /// Number of CDF bins.
    pub bins: u32,
    /// The `bins` u16-quantized CDF values.
    pub values: Vec<u16>,
}

/// The learned orthogonal pre-rotation (`dim × dim` row-major; `dim == 0` ⇒ none).
#[derive(Clone, Debug, PartialEq)]
pub struct RotationMatrix {
    /// Matrix dimension (`0` = no rotation).
    pub dim: u32,
    /// Row-major `dim × dim` matrix data.
    pub data: Vec<f32>,
}

/// A codec-complete ternary artifact: quantizer metadata + an embedded SALT weights bundle.
#[derive(Clone, Debug, PartialEq)]
pub struct CodecBundle {
    /// Per-tensor ternary Conv1d geometry.
    pub conv: Vec<Conv1dTensorMeta>,
    /// Per-latent FSQ configuration.
    pub fsq: Vec<FsqLayerMeta>,
    /// Per-tensor learned LSQ step sizes.
    pub lsq: Vec<LsqAlpha>,
    /// Empirical-CDF LUTs.
    pub cdf: Vec<CdfLut>,
    /// The learned orthogonal pre-rotation.
    pub rotation: RotationMatrix,
    /// An embedded SALT bundle (the conv weights, 2-D `[C_out, K_g]`), from [`crate::write_salt_bundle`].
    pub weights: Vec<u8>,
}

// ── write ────────────────────────────────────────────────────────────────────────────────────────

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}
fn put_u32s(out: &mut Vec<u8>, v: &[u32]) {
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}
fn put_f32s(out: &mut Vec<u8>, v: &[f32]) {
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}
fn put_u16s(out: &mut Vec<u8>, v: &[u16]) {
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// Serialize a [`CodecBundle`] to bytes (magic + sections + trailing blake3 checksum).
///
/// # Errors
/// [`CodecBundleError::RotationShape`] if `rotation.data.len() != dim·dim`.
pub fn write_codec_bundle(b: &CodecBundle) -> Result<Vec<u8>, CodecBundleError> {
    if b.rotation.data.len() != (b.rotation.dim as usize) * (b.rotation.dim as usize) {
        return Err(CodecBundleError::RotationShape);
    }
    let mut out = Vec::new();
    out.extend_from_slice(&CODEC_BUNDLE_MAGIC);
    out.extend_from_slice(&CODEC_BUNDLE_VERSION.to_le_bytes());

    out.extend_from_slice(&(b.conv.len() as u32).to_le_bytes());
    for c in &b.conv {
        put_str(&mut out, &c.name);
        for x in [
            c.c_in,
            c.c_out,
            c.k,
            c.stride,
            c.dilation,
            c.pad_left,
            c.pad_right,
            c.groups,
        ] {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }

    out.extend_from_slice(&(b.fsq.len() as u32).to_le_bytes());
    for q in &b.fsq {
        put_str(&mut out, &q.name);
        put_u32s(&mut out, &q.levels);
        out.push(q.bound);
        out.push(q.ste);
        out.extend_from_slice(&q.alpha.to_le_bytes());
        out.extend_from_slice(&q.seed.to_le_bytes());
    }

    out.extend_from_slice(&(b.lsq.len() as u32).to_le_bytes());
    for a in &b.lsq {
        put_str(&mut out, &a.name);
        put_f32s(&mut out, &a.per_channel);
    }

    out.extend_from_slice(&(b.cdf.len() as u32).to_le_bytes());
    for l in &b.cdf {
        put_str(&mut out, &l.name);
        out.extend_from_slice(&l.bins.to_le_bytes());
        put_u16s(&mut out, &l.values);
    }

    out.extend_from_slice(&b.rotation.dim.to_le_bytes());
    for &x in &b.rotation.data {
        out.extend_from_slice(&x.to_le_bytes());
    }

    out.extend_from_slice(&(b.weights.len() as u64).to_le_bytes());
    out.extend_from_slice(&b.weights);

    let checksum = blake3::hash(&out);
    out.extend_from_slice(checksum.as_bytes());
    Ok(out)
}

// ── read ─────────────────────────────────────────────────────────────────────────────────────────

/// A bounds-checked little-endian cursor over the codec-bundle body (errors, never panics).
struct Cur<'a> {
    b: &'a [u8],
    o: usize,
}
impl<'a> Cur<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, o: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], CodecBundleError> {
        let end = self.o.checked_add(n).ok_or(CodecBundleError::Truncated)?;
        if end > self.b.len() {
            return Err(CodecBundleError::Truncated);
        }
        let s = &self.b[self.o..end];
        self.o = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, CodecBundleError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, CodecBundleError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, CodecBundleError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u16(&mut self) -> Result<u16, CodecBundleError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, CodecBundleError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    /// A length-prefixed count, rejected if it exceeds the size cap (guards huge allocations).
    fn count(&mut self) -> Result<usize, CodecBundleError> {
        let n = self.u32()? as usize;
        if n > MAX_CODEC_BUNDLE_BYTES {
            return Err(CodecBundleError::TooLarge);
        }
        Ok(n)
    }
    fn string(&mut self) -> Result<String, CodecBundleError> {
        let n = self.count()?;
        let bytes = self.take(n)?;
        core::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| CodecBundleError::InvalidUtf8)
    }
    fn u32s(&mut self) -> Result<Vec<u32>, CodecBundleError> {
        let n = self.count()?;
        (0..n).map(|_| self.u32()).collect()
    }
    fn f32s(&mut self) -> Result<Vec<f32>, CodecBundleError> {
        let n = self.count()?;
        (0..n).map(|_| self.f32()).collect()
    }
    fn u16s(&mut self) -> Result<Vec<u16>, CodecBundleError> {
        let n = self.count()?;
        (0..n).map(|_| self.u16()).collect()
    }
}

/// Parse a [`CodecBundle`] from bytes, verifying the trailing blake3 checksum.
///
/// # Errors
/// [`CodecBundleError`] on bad magic, unsupported version, truncation, checksum mismatch, an
/// oversized length, invalid UTF-8, or a malformed rotation.
pub fn read_codec_bundle(bytes: &[u8]) -> Result<CodecBundle, CodecBundleError> {
    if bytes.len() < 32 {
        return Err(CodecBundleError::Truncated);
    }
    let (body, checksum) = bytes.split_at(bytes.len() - 32);
    if blake3::hash(body).as_bytes() != checksum {
        return Err(CodecBundleError::ChecksumMismatch);
    }
    let mut c = Cur::new(body);
    if c.take(4)? != CODEC_BUNDLE_MAGIC {
        return Err(CodecBundleError::BadMagic);
    }
    let version = c.u32()?;
    if version != CODEC_BUNDLE_VERSION {
        return Err(CodecBundleError::UnsupportedVersion(version));
    }

    let n_conv = c.count()?;
    let mut conv = Vec::with_capacity(n_conv);
    for _ in 0..n_conv {
        let name = c.string()?;
        conv.push(Conv1dTensorMeta {
            name,
            c_in: c.u32()?,
            c_out: c.u32()?,
            k: c.u32()?,
            stride: c.u32()?,
            dilation: c.u32()?,
            pad_left: c.u32()?,
            pad_right: c.u32()?,
            groups: c.u32()?,
        });
    }

    let n_fsq = c.count()?;
    let mut fsq = Vec::with_capacity(n_fsq);
    for _ in 0..n_fsq {
        let name = c.string()?;
        let levels = c.u32s()?;
        fsq.push(FsqLayerMeta {
            name,
            levels,
            bound: c.u8()?,
            ste: c.u8()?,
            alpha: c.f32()?,
            seed: c.u64()?,
        });
    }

    let n_lsq = c.count()?;
    let mut lsq = Vec::with_capacity(n_lsq);
    for _ in 0..n_lsq {
        let name = c.string()?;
        lsq.push(LsqAlpha {
            name,
            per_channel: c.f32s()?,
        });
    }

    let n_cdf = c.count()?;
    let mut cdf = Vec::with_capacity(n_cdf);
    for _ in 0..n_cdf {
        let name = c.string()?;
        let bins = c.u32()?;
        cdf.push(CdfLut {
            name,
            bins,
            values: c.u16s()?,
        });
    }

    let dim = c.u32()?;
    let n_rot = (dim as usize)
        .checked_mul(dim as usize)
        .ok_or(CodecBundleError::TooLarge)?;
    if n_rot > MAX_CODEC_BUNDLE_BYTES {
        return Err(CodecBundleError::TooLarge);
    }
    let data = (0..n_rot).map(|_| c.f32()).collect::<Result<Vec<_>, _>>()?;
    let rotation = RotationMatrix { dim, data };

    let w_len = c.u64()? as usize;
    if w_len > MAX_CODEC_BUNDLE_BYTES {
        return Err(CodecBundleError::TooLarge);
    }
    let weights = c.take(w_len)?.to_vec();

    Ok(CodecBundle {
        conv,
        fsq,
        lsq,
        cdf,
        rotation,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CodecBundle {
        CodecBundle {
            conv: vec![
                Conv1dTensorMeta {
                    name: "enc.conv0".into(),
                    c_in: 168,
                    c_out: 64,
                    k: 3,
                    stride: 1,
                    dilation: 1,
                    pad_left: 1,
                    pad_right: 1,
                    groups: 1,
                },
                Conv1dTensorMeta {
                    name: "enc.dw1".into(),
                    c_in: 64,
                    c_out: 64,
                    k: 7,
                    stride: 2,
                    dilation: 2,
                    pad_left: 6,
                    pad_right: 0,
                    groups: 64,
                },
            ],
            fsq: vec![FsqLayerMeta {
                name: "latent".into(),
                levels: vec![2, 3, 5, 8, 16, 32],
                bound: 1,
                ste: 2,
                alpha: 0.5,
                seed: 0xDEAD_BEEF,
            }],
            lsq: vec![LsqAlpha {
                name: "enc.conv0".into(),
                per_channel: vec![0.11, 0.22, 0.33, 0.44],
            }],
            cdf: vec![CdfLut {
                name: "cdf0".into(),
                bins: 4,
                values: vec![0, 16384, 32768, 65535],
            }],
            rotation: RotationMatrix {
                dim: 2,
                data: vec![0.0, 1.0, -1.0, 0.0],
            },
            weights: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], // opaque embedded SALT bundle bytes
        }
    }

    #[test]
    fn round_trips() {
        let b = sample();
        let bytes = write_codec_bundle(&b).unwrap();
        let back = read_codec_bundle(&bytes).unwrap();
        assert_eq!(b, back);
    }

    #[test]
    fn empty_sections_round_trip() {
        let b = CodecBundle {
            conv: vec![],
            fsq: vec![],
            lsq: vec![],
            cdf: vec![],
            rotation: RotationMatrix {
                dim: 0,
                data: vec![],
            },
            weights: vec![],
        };
        let bytes = write_codec_bundle(&b).unwrap();
        assert_eq!(read_codec_bundle(&bytes).unwrap(), b);
    }

    #[test]
    fn tamper_fails_checksum() {
        let bytes = write_codec_bundle(&sample()).unwrap();
        // Flip a byte in the body (not the checksum tail).
        let mut bad = bytes.clone();
        bad[20] ^= 0xFF;
        assert_eq!(
            read_codec_bundle(&bad),
            Err(CodecBundleError::ChecksumMismatch)
        );
    }

    #[test]
    fn bad_magic_and_version_rejected() {
        let mut bytes = write_codec_bundle(&sample()).unwrap();
        // Corrupt the magic, then re-checksum so it fails on magic (not checksum).
        bytes[0] = b'X';
        let body_len = bytes.len() - 32;
        let cs = blake3::hash(&bytes[..body_len]);
        bytes[body_len..].copy_from_slice(cs.as_bytes());
        assert_eq!(read_codec_bundle(&bytes), Err(CodecBundleError::BadMagic));
    }

    #[test]
    fn truncated_rejected() {
        let bytes = write_codec_bundle(&sample()).unwrap();
        assert_eq!(
            read_codec_bundle(&bytes[..10]),
            Err(CodecBundleError::Truncated)
        );
    }

    #[test]
    fn rotation_shape_validated() {
        let mut b = sample();
        b.rotation = RotationMatrix {
            dim: 2,
            data: vec![1.0, 2.0, 3.0],
        }; // 3 != 2*2
        assert_eq!(write_codec_bundle(&b), Err(CodecBundleError::RotationShape));
    }
}

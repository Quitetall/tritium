//! Common package preparation for buffered/writer export and owned reconstruction.
//! Scalar staging and decoded caches remain owned; not spooled/mmap or full RSS admission.
use super::*;
use std::collections::BTreeMap;
use tritium_format::trit_calibration::{
    decode_calibration_cache_manifest, encode_calibration_cache_manifest,
};
use tritium_format::{
    PackageId, TritBlob, TritBlobKind, read_trit_package, write_trit_package_bounded,
    write_trit_package_to,
};
use tritium_schema::{CalibrationCacheLimits, CalibrationCacheManifest};

impl CalibrationCache {
    /// Encode immutable content in the existing `.trit` envelope and typed cache profile.
    /// Retains staging buffers plus the complete output; not a full RSS/streaming claim.
    /// # Errors
    /// Logical package bound, allocation or common manifest/package codec failure.
    pub fn encode_package(
        &self,
        max_package_bytes: usize,
    ) -> Result<Vec<u8>, CalibrationCacheError> {
        self.with_package(max_package_bytes, |manifest, blobs| {
            write_trit_package_bounded(manifest, blobs, max_package_bytes).map_err(transport)
        })
    }

    /// Emit the same immutable package through the common bounded writer.
    ///
    /// Does not retain a complete output package, but retains this cache and scalar
    /// staging buffers/metadata. This is not spooled input, mmap or full RSS admission.
    /// Writes at the current destination position; never seeks, flushes, syncs or
    /// publishes. Use a caller-owned pending destination before no-clobber publication.
    /// # Errors
    /// Admission completes before any destination write. I/O failure may leave a
    /// prefix and returns `CalibrationCacheError::Write` without a success identity.
    pub fn write_package_to(
        &self,
        max_package_bytes: usize,
        destination: &mut impl std::io::Write,
    ) -> Result<PackageId, CalibrationCacheError> {
        self.with_package(max_package_bytes, |manifest, blobs| {
            write_trit_package_to(manifest, blobs, max_package_bytes, destination)
                .map_err(CalibrationCacheError::Write)
        })
    }

    fn with_package<R>(
        &self,
        max_package_bytes: usize,
        emit: impl FnOnce(&[u8], &[TritBlob<'_>]) -> Result<R, CalibrationCacheError>,
    ) -> Result<R, CalibrationCacheError> {
        if self.binding.payload_bytes > max_package_bytes as u64 {
            return Err(CalibrationCacheError::Limit);
        }
        let manifest = CalibrationCacheManifest {
            binding: self.binding.clone(),
            window_samples: self.windows.iter().map(|w| w.tokens.len() as u64).collect(),
        };
        let manifest = encode_calibration_cache_manifest(&manifest).map_err(transport)?;
        let mut owned = Vec::new();
        let count = self
            .windows
            .len()
            .checked_mul(5)
            .ok_or(CalibrationCacheError::Allocation)?;
        owned
            .try_reserve_exact(count)
            .map_err(|_| CalibrationCacheError::Allocation)?;
        for (index, window) in self.windows.iter().enumerate() {
            owned.push((
                name(index, "tokens"),
                encode(&window.tokens, |v| v.to_le_bytes())?,
            ));
            if let Some(v) = &window.input {
                owned.push((
                    name(index, "input"),
                    encode(v, |v| v.to_bits().to_le_bytes())?,
                ));
            }
            match &window.output {
                Output::None => {}
                Output::Dense(v) => owned.push((
                    name(index, "output"),
                    encode(v, |v| v.to_bits().to_le_bytes())?,
                )),
                Output::Indexed { indices, factors } => {
                    let len = indices
                        .len()
                        .checked_mul(12)
                        .ok_or(CalibrationCacheError::Allocation)?;
                    let mut bytes = reserved(len)?;
                    for (&row, factor) in indices.iter().zip(factors) {
                        bytes.extend_from_slice(&(row as u64).to_le_bytes());
                        bytes.extend_from_slice(&factor.to_bits().to_le_bytes());
                    }
                    owned.push((name(index, "output"), bytes));
                }
            }
            owned.push((
                name(index, "weights"),
                encode(&window.weights, |v| v.to_bits().to_le_bytes())?,
            ));
            owned.push((
                name(index, "mask"),
                encode(&window.mask, |v| [u8::from(*v)])?,
            ));
        }
        let mut blobs = Vec::new();
        blobs
            .try_reserve_exact(
                owned
                    .len()
                    .checked_add(2)
                    .ok_or(CalibrationCacheError::Allocation)?,
            )
            .map_err(|_| CalibrationCacheError::Allocation)?;
        blobs.push(TritBlob {
            kind: TritBlobKind::Asset("tokenizer".into()),
            bytes: &self.tokenizer,
        });
        blobs.push(TritBlob {
            kind: TritBlobKind::Asset("objective".into()),
            bytes: &self.objective,
        });
        blobs.extend(owned.iter().map(|(name, bytes)| TritBlob {
            kind: TritBlobKind::Asset(name.clone()),
            bytes,
        }));
        emit(&manifest, &blobs)
    }

    /// Verify a complete borrowed `.trit` package, then reconstruct actual immutable cache.
    /// Actual content is revalidated/rehashed through the same builder before exposure.
    /// Caller limits supplement the stored producer limits and precede payload copies.
    /// The caller owns the input bytes; this is not mmap/spooled or aggregate RSS admission.
    /// # Errors
    /// Package/profile/inventory/shape/scalar/context identity or receiving limits mismatch.
    pub fn decode_package(
        bytes: &[u8],
        expected: CalibrationCacheId,
        limits: CalibrationCacheLimits,
        max_package_bytes: usize,
    ) -> Result<Self, CalibrationCacheError> {
        if bytes.len() > max_package_bytes {
            return Err(CalibrationCacheError::Limit);
        }
        let package = read_trit_package(bytes).map_err(transport)?;
        let manifest = decode_calibration_cache_manifest(package.manifest()).map_err(transport)?;
        manifest
            .validate(limits)
            .map_err(CalibrationCacheError::Plan)?;
        let b = &manifest.binding;
        if b.source.cache.0 != *expected.as_bytes() {
            return Err(CalibrationCacheError::Binding);
        }
        let request = &b.request;
        let per_window = 3usize
            + usize::from(request.consumer == CalibrationConsumer::Matmul)
            + usize::from(request.output != CalibrationOutput::None);
        let expected_blobs = manifest
            .window_samples
            .len()
            .checked_mul(per_window)
            .and_then(|n| n.checked_add(2))
            .ok_or(CalibrationCacheError::Allocation)?;
        if package.blobs().len() != expected_blobs {
            return Err(CalibrationCacheError::Binding);
        }
        let mut assets = BTreeMap::new();
        let mut actual_payload = 0u64;
        for info in package.blobs() {
            let TritBlobKind::Asset(name) = &info.kind else {
                return Err(CalibrationCacheError::Binding);
            };
            actual_payload = actual_payload
                .checked_add(info.length)
                .ok_or(CalibrationCacheError::Allocation)?;
            let blob = package.blob(info).map_err(transport)?;
            if assets.insert(name.as_str(), blob).is_some() {
                return Err(CalibrationCacheError::Binding);
            }
        }
        if actual_payload != b.payload_bytes || actual_payload > limits.max_payload_bytes {
            return Err(CalibrationCacheError::Binding);
        }
        let tokenizer = take(&mut assets, "tokenizer")?;
        let objective = take(&mut assets, "objective")?;
        let mut builder = CalibrationCacheBuilder::new(request.clone(), tokenizer, objective)?;
        let rows = usize::try_from(request.tensor.shape[0])
            .map_err(|_| CalibrationCacheError::Allocation)?;
        let cols = usize::try_from(request.tensor.shape[1])
            .map_err(|_| CalibrationCacheError::Allocation)?;
        for (index, &count) in manifest.window_samples.iter().enumerate() {
            let samples = usize::try_from(count).map_err(|_| CalibrationCacheError::Allocation)?;
            let tokens = decode(
                take(&mut assets, &name(index, "tokens"))?,
                samples,
                u32::from_le_bytes,
            )?;
            let input = if request.consumer == CalibrationConsumer::Matmul {
                Some(decode(
                    take(&mut assets, &name(index, "input"))?,
                    samples
                        .checked_mul(cols)
                        .ok_or(CalibrationCacheError::Allocation)?,
                    f32::from_le_bytes,
                )?)
            } else {
                None
            };
            let output = match request.output {
                CalibrationOutput::None => Output::None,
                CalibrationOutput::DenseFisher => Output::Dense(decode(
                    take(&mut assets, &name(index, "output"))?,
                    samples
                        .checked_mul(rows)
                        .ok_or(CalibrationCacheError::Allocation)?,
                    f32::from_le_bytes,
                )?),
                CalibrationOutput::IndexedFisher => {
                    let bytes = take(&mut assets, &name(index, "output"))?;
                    check_len(bytes, samples, 12)?;
                    let mut indices = Vec::new();
                    let mut factors = Vec::new();
                    indices
                        .try_reserve_exact(samples)
                        .map_err(|_| CalibrationCacheError::Allocation)?;
                    factors
                        .try_reserve_exact(samples)
                        .map_err(|_| CalibrationCacheError::Allocation)?;
                    for chunk in bytes.as_chunks::<12>().0 {
                        let row = u64::from_le_bytes(chunk[..8].try_into().expect("eight bytes"));
                        indices
                            .push(usize::try_from(row).map_err(|_| CalibrationCacheError::Batch)?);
                        factors.push(f32::from_le_bytes(
                            chunk[8..].try_into().expect("four bytes"),
                        ));
                    }
                    Output::Indexed { indices, factors }
                }
                _ => return Err(CalibrationCacheError::Batch),
            };
            let weights = decode(
                take(&mut assets, &name(index, "weights"))?,
                samples,
                f64::from_le_bytes,
            )?;
            let mask_bytes = take(&mut assets, &name(index, "mask"))?;
            check_len(mask_bytes, samples, 1)?;
            if mask_bytes.iter().any(|v| *v > 1) {
                return Err(CalibrationCacheError::Batch);
            }
            let mut mask = Vec::new();
            mask.try_reserve_exact(samples)
                .map_err(|_| CalibrationCacheError::Allocation)?;
            mask.extend(mask_bytes.iter().map(|v| *v == 1));
            let window = Window {
                tokens,
                input,
                output,
                weights,
                mask,
            };
            builder.capture(&window.tokens, window.batch(0, samples, rows, cols))?;
        }
        if !assets.is_empty() {
            return Err(CalibrationCacheError::Binding);
        }
        let cache = builder.finish()?;
        if cache.id != expected || cache.binding != *b {
            return Err(CalibrationCacheError::Binding);
        }
        Ok(cache)
    }
}

fn transport(e: impl core::fmt::Display) -> CalibrationCacheError {
    CalibrationCacheError::Transport(e.to_string())
}
fn name(index: usize, role: &str) -> String {
    format!("window/{index}/{role}")
}
fn take<'a>(
    assets: &mut BTreeMap<&str, &'a [u8]>,
    name: &str,
) -> Result<&'a [u8], CalibrationCacheError> {
    assets.remove(name).ok_or(CalibrationCacheError::Binding)
}
fn reserved(n: usize) -> Result<Vec<u8>, CalibrationCacheError> {
    super::super::cache_scalar::reserved(n).map_err(scalar_error)
}
fn encode<T, const N: usize>(
    values: &[T],
    convert: impl Fn(&T) -> [u8; N],
) -> Result<Vec<u8>, CalibrationCacheError> {
    super::super::cache_scalar::encode(values, convert).map_err(scalar_error)
}
fn check_len(bytes: &[u8], count: usize, width: usize) -> Result<(), CalibrationCacheError> {
    super::super::cache_scalar::check_len(bytes, count, width).map_err(scalar_error)
}
fn decode<T, const N: usize>(
    bytes: &[u8],
    count: usize,
    convert: impl Fn([u8; N]) -> T,
) -> Result<Vec<T>, CalibrationCacheError> {
    super::super::cache_scalar::decode(bytes, count, convert).map_err(scalar_error)
}
fn scalar_error(error: super::super::cache_scalar::ScalarError) -> CalibrationCacheError {
    match error {
        super::super::cache_scalar::ScalarError::Allocation => CalibrationCacheError::Allocation,
        super::super::cache_scalar::ScalarError::Length => CalibrationCacheError::Batch,
    }
}

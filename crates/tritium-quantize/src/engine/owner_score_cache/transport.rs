//! Common container/profile/scalar codecs; no second cache identity or reducer.
use super::*;
use std::collections::BTreeMap;
use tritium_format::trit_owner_score::{decode_owner_score_manifest, encode_owner_score_manifest};
use tritium_format::{
    PackageId, TritBlob, TritBlobKind, read_trit_package, write_trit_package_bounded,
    write_trit_package_to,
};

impl OwnerScoreCache {
    /// Encode actual immutable observations with the common bounded .trit writer.
    /// Retains scalar staging and output bytes; not a full RSS/spooling claim.
    pub fn encode_package(
        &self,
        max_package_bytes: usize,
    ) -> Result<Vec<u8>, OwnerScoreCacheError> {
        self.with_package(max_package_bytes, |m, b| {
            write_trit_package_bounded(m, b, max_package_bytes).map_err(transport_error)
        })
    }
    /// Same actual package through the common Write seam. Admission precedes writes;
    /// failed I/O may leave a prefix. Does not seek, flush, sync or publish.
    pub fn write_package_to(
        &self,
        max_package_bytes: usize,
        destination: &mut impl std::io::Write,
    ) -> Result<PackageId, OwnerScoreCacheError> {
        self.with_package(max_package_bytes, |m, b| {
            write_trit_package_to(m, b, max_package_bytes, destination)
                .map_err(OwnerScoreCacheError::Write)
        })
    }
    fn with_package<T>(
        &self,
        max: usize,
        emit: impl FnOnce(&[u8], &[TritBlob<'_>]) -> Result<T, OwnerScoreCacheError>,
    ) -> Result<T, OwnerScoreCacheError> {
        if self.binding.payload_bytes > max as u64 {
            return Err(OwnerScoreCacheError::Limit);
        }
        let manifest = encode_owner_score_manifest(&OwnerScoreCacheManifest {
            binding: self.binding.clone(),
            limits: self.limits,
        })
        .map_err(transport_error)?;
        let mut numerical = cache_scalar::reserved(
            self.observations
                .use_count()
                .checked_mul(2)
                .ok_or(OwnerScoreCacheError::Allocation)?,
        )?;
        for (index, usage) in self.observations.uses().enumerate() {
            let (input, adjoint) = match usage {
                OwnerScoreUse::Matmul {
                    inputs, adjoints, ..
                } => (
                    cache_scalar::encode(inputs, |v| v.to_bits().to_le_bytes())?,
                    cache_scalar::encode(adjoints, |v| v.to_bits().to_le_bytes())?,
                ),
                OwnerScoreUse::Gather { indices, adjoints } => (
                    cache_scalar::encode(indices, |v| v.to_le_bytes())?,
                    cache_scalar::encode(adjoints, |v| v.to_bits().to_le_bytes())?,
                ),
            };
            numerical.push((name(index, "input"), input));
            numerical.push((name(index, "adjoint"), adjoint));
        }
        let count = 2usize
            .checked_add(self.context.len())
            .and_then(|n| n.checked_add(numerical.len()))
            .ok_or(OwnerScoreCacheError::Allocation)?;
        let mut blobs = cache_scalar::reserved(count)?;
        blobs.push(TritBlob {
            kind: TritBlobKind::Asset("tokenizer".into()),
            bytes: &self.tokenizer,
        });
        blobs.push(TritBlob {
            kind: TritBlobKind::Asset("objective".into()),
            bytes: &self.objective,
        });
        for (index, bytes) in self.context.iter().enumerate() {
            blobs.push(TritBlob {
                kind: TritBlobKind::Asset(format!("context/{index}")),
                bytes,
            });
        }
        blobs.extend(numerical.iter().map(|(n, b)| TritBlob {
            kind: TritBlobKind::Asset(n.clone()),
            bytes: b,
        }));
        emit(&manifest, &blobs)
    }
    /// Verify canonical container/profile/inventory and actual immutable content.
    /// Exact expected binding and receiving limits are checked before payload copies.
    /// Caller owns complete input bytes; decoder overlap/metadata/full RSS is separate.
    pub fn decode_package(
        bytes: &[u8],
        expected: &OwnerScoreCacheBinding,
        receiving: OwnerScoreCacheLimits,
        max_package_bytes: usize,
    ) -> Result<Self, OwnerScoreCacheError> {
        use OwnerScoreCacheError as E;
        if bytes.len() > max_package_bytes {
            return Err(E::Limit);
        }
        let package = read_trit_package(bytes).map_err(transport_error)?;
        let manifest = decode_owner_score_manifest(package.manifest()).map_err(transport_error)?;
        if !manifest.valid(receiving) {
            return Err(E::Limit);
        }
        if &manifest.binding != expected {
            return Err(E::Binding);
        }
        let content = &manifest.binding.content;
        let count = content
            .invocations
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(content.context.len()))
            .and_then(|n| n.checked_add(2))
            .ok_or(E::Allocation)?;
        if package.blobs().len() != count {
            return Err(E::Binding);
        }
        let mut assets = BTreeMap::new();
        let mut payload = 0u64;
        for info in package.blobs() {
            let TritBlobKind::Asset(name) = &info.kind else {
                return Err(E::Binding);
            };
            payload = payload.checked_add(info.length).ok_or(E::Allocation)?;
            if assets
                .insert(name.as_str(), package.blob(info).map_err(transport_error)?)
                .is_some()
            {
                return Err(E::Binding);
            }
        }
        if payload != manifest.binding.payload_bytes || payload > receiving.max_payload_bytes {
            return Err(E::Binding);
        }
        let tokenizer = take(&mut assets, "tokenizer")?;
        let objective = take(&mut assets, "objective")?;
        if blob(tokenizer) != content.tokenizer || blob(objective) != content.objective {
            return Err(E::Binding);
        }
        let mut context = cache_scalar::reserved(content.context.len())?;
        for (index, binding) in content.context.iter().enumerate() {
            let bytes = take(&mut assets, &format!("context/{index}"))?;
            validate_context(&binding.meta, bytes)?;
            if blob(bytes) != binding.content {
                return Err(E::Binding);
            }
            context.push(bytes);
        }
        let rows = usize::try_from(content.tensor.shape[0]).map_err(|_| E::Allocation)?;
        let columns = usize::try_from(content.tensor.shape[1]).map_err(|_| E::Allocation)?;
        // Admit every numerical buffer's geometry/hash before any numerical
        // decode copies, including malformed late invocations.
        for (index, binding) in content.invocations.iter().enumerate() {
            let input = *assets
                .get(name(index, "input").as_str())
                .ok_or(E::Binding)?;
            let adjoint = *assets
                .get(name(index, "adjoint").as_str())
                .ok_or(E::Binding)?;
            let n = usize::try_from(binding.meta.samples).map_err(|_| E::Allocation)?;
            match binding.meta.role {
                OwnerScoreRole::Matmul => {
                    cache_scalar::check_len(
                        input,
                        n.checked_mul(columns).ok_or(E::Allocation)?,
                        4,
                    )?;
                    cache_scalar::check_len(adjoint, n.checked_mul(rows).ok_or(E::Allocation)?, 4)?;
                }
                OwnerScoreRole::Gather => {
                    cache_scalar::check_len(input, n, 8)?;
                    cache_scalar::check_len(
                        adjoint,
                        n.checked_mul(columns).ok_or(E::Allocation)?,
                        4,
                    )?;
                }
                _ => return Err(E::Structure),
            }
            if blob(input) != binding.input || blob(adjoint) != binding.adjoint {
                return Err(E::Binding);
            }
        }
        let mut decoded = cache_scalar::reserved(content.invocations.len())?;
        for (index, binding) in content.invocations.iter().enumerate() {
            let input = take(&mut assets, &name(index, "input"))?;
            let adjoint = take(&mut assets, &name(index, "adjoint"))?;
            if blob(input) != binding.input || blob(adjoint) != binding.adjoint {
                return Err(E::Binding);
            }
            let n = usize::try_from(binding.meta.samples).map_err(|_| E::Allocation)?;
            decoded.push(match binding.meta.role {
                OwnerScoreRole::Matmul => DecodedUse::Matmul {
                    samples: n,
                    inputs: cache_scalar::decode(
                        input,
                        n.checked_mul(columns).ok_or(E::Allocation)?,
                        f32::from_le_bytes,
                    )?,
                    adjoints: cache_scalar::decode(
                        adjoint,
                        n.checked_mul(rows).ok_or(E::Allocation)?,
                        f32::from_le_bytes,
                    )?,
                },
                OwnerScoreRole::Gather => DecodedUse::Gather {
                    indices: cache_scalar::decode(input, n, i64::from_le_bytes)?,
                    adjoints: cache_scalar::decode(
                        adjoint,
                        n.checked_mul(columns).ok_or(E::Allocation)?,
                        f32::from_le_bytes,
                    )?,
                },
                _ => return Err(E::Structure),
            });
        }
        if !assets.is_empty() {
            return Err(E::Binding);
        }
        let uses: Vec<_> = decoded.iter().map(DecodedUse::borrowed).collect();
        let observations = super::super::OwnerScoreWindow::new(
            rows,
            columns,
            &uses,
            manifest.binding.payload_bytes,
        )?
        .snapshot()?;
        let meta = OwnerScoreWindowMeta {
            context: content.context.iter().map(|b| b.meta.clone()).collect(),
            invocations: content.invocations.iter().map(|b| b.meta.clone()).collect(),
        };
        let cache = Self::new(
            OwnerScoreCacheInput {
                source: content.source,
                tensor: content.tensor.clone(),
                tokenizer,
                objective,
                meta,
                context: &context,
                observations: Arc::new(observations),
            },
            manifest.limits,
        )?;
        if cache.binding() != expected {
            return Err(E::Binding);
        }
        Ok(cache)
    }
}
enum DecodedUse {
    Matmul {
        samples: usize,
        inputs: Vec<f32>,
        adjoints: Vec<f32>,
    },
    Gather {
        indices: Vec<i64>,
        adjoints: Vec<f32>,
    },
}
impl DecodedUse {
    fn borrowed(&self) -> OwnerScoreUse<'_> {
        match self {
            Self::Matmul {
                samples,
                inputs,
                adjoints,
            } => OwnerScoreUse::Matmul {
                samples: *samples,
                inputs,
                adjoints,
            },
            Self::Gather { indices, adjoints } => OwnerScoreUse::Gather { indices, adjoints },
        }
    }
}
fn name(index: usize, role: &str) -> String {
    format!("invocation/{index}/{role}")
}
fn take<'a>(
    assets: &mut BTreeMap<&str, &'a [u8]>,
    name: &str,
) -> Result<&'a [u8], OwnerScoreCacheError> {
    assets.remove(name).ok_or(OwnerScoreCacheError::Binding)
}

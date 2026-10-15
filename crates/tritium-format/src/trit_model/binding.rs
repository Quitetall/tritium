//! One typed binding/digest/prefix validator for borrowed and retained sources.
use super::*;
use crate::TritBlobInfo;
use crate::additive_tensor::{PayloadPlan, PayloadSource, RANGE_BYTES};
use std::collections::BTreeMap;

pub(super) trait ModelSource: PayloadSource {
    fn blobs(&self) -> &[TritBlobInfo];
}

pub(super) struct BorrowedSource<'a, 'b>(pub &'b TritPackage<'a>);
impl PayloadSource for BorrowedSource<'_, '_> {
    type Error = TritModelError;
    fn read_range(
        &mut self,
        blob: usize,
        offset: usize,
        out: &mut [u8],
    ) -> Result<(), Self::Error> {
        let end = offset
            .checked_add(out.len())
            .ok_or(TritPackageError::LengthOverflow)?;
        let bytes = self
            .0
            .indexed_blob(blob)?
            .get(offset..end)
            .ok_or(TritPackageError::InvalidDirectory)?;
        out.copy_from_slice(bytes);
        Ok(())
    }
}
impl ModelSource for BorrowedSource<'_, '_> {
    fn blobs(&self) -> &[TritBlobInfo] {
        self.0.blobs()
    }
}

#[derive(Debug)]
pub(super) struct ModelBindings {
    pub(super) manifest: ModelManifest,
    index: BTreeMap<[u8; 32], Vec<usize>>,
    tensor_digests: Vec<Vec<SemanticTensorDigest>>,
    model_ids: Vec<ModelId>,
}
impl ModelBindings {
    fn level_index(&self, name: &str) -> Result<usize, TritModelError> {
        self.manifest
            .levels
            .binary_search_by(|l| l.name.as_str().cmp(name))
            .map_err(|_| TritModelError::MissingBinding(name.into()))
    }
    fn level(&self, name: &str) -> Result<&Level, TritModelError> {
        Ok(&self.manifest.levels[self.level_index(name)?])
    }
    pub(super) fn model_id(&self, level: &str) -> Result<ModelId, TritModelError> {
        Ok(self.model_ids[self.level_index(level)?])
    }
    pub(super) fn tensor_digest(
        &self,
        level: &str,
        name: &str,
    ) -> Result<SemanticTensorDigest, TritModelError> {
        let level = self.level_index(level)?;
        let tensor = self.manifest.levels[level]
            .tensors
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .map_err(|_| TritModelError::MissingBinding(name.into()))?;
        Ok(self.tensor_digests[level][tensor])
    }
    pub(super) fn tensor_meta(
        &self,
        level: &str,
        name: &str,
    ) -> Result<&TensorMeta, TritModelError> {
        let level = self.level(level)?;
        let index = level
            .tensors
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .map_err(|_| TritModelError::MissingBinding(name.into()))?;
        Ok(&level.tensors[index])
    }
    fn payload_index(
        &self,
        id: BlobId,
        kind: &TritBlobKind,
        name: &str,
        blobs: &[TritBlobInfo],
    ) -> Result<usize, TritModelError> {
        self.index
            .get(id.as_bytes())
            .and_then(|indices| indices.iter().copied().find(|&i| &blobs[i].kind == kind))
            .ok_or_else(|| TritModelError::MissingOrWrongBlob(name.into()))
    }
    pub(super) fn asset_index(
        &self,
        name: &str,
        blobs: &[TritBlobInfo],
    ) -> Result<usize, TritModelError> {
        let index = self
            .manifest
            .assets
            .binary_search_by(|a| a.name.as_str().cmp(name))
            .map_err(|_| TritModelError::MissingBinding(name.into()))?;
        let asset = &self.manifest.assets[index];
        self.payload_index(
            asset.blob.id,
            &TritBlobKind::Asset(asset.name.clone()),
            name,
            blobs,
        )
    }
    pub(super) fn tensor_index(
        &self,
        meta: &TensorMeta,
        blobs: &[TritBlobInfo],
    ) -> Result<usize, TritModelError> {
        let kind = match &meta.storage {
            TensorStorage::Dense { precision, .. } => TritBlobKind::DenseTensor(match precision {
                ScalePrecision::F16 => DType::F16,
                ScalePrecision::Bf16 => DType::BF16,
                ScalePrecision::F32 => DType::F32,
                _ => return Err(TritModelError::UnsupportedFeature("dense precision")),
            }),
            TensorStorage::Additive { .. } => TritBlobKind::AdditiveTensor,
            _ => return Err(TritModelError::UnsupportedFeature("tensor storage")),
        };
        self.payload_index(meta.blob.id, &kind, &meta.name, blobs)
    }
    pub(super) fn validate_complete_payload_inventory(
        &self,
        blobs: &[TritBlobInfo],
    ) -> Result<(), TritModelError> {
        let mut referenced = vec![false; blobs.len()];
        for asset in &self.manifest.assets {
            referenced[self.asset_index(&asset.name, blobs)?] = true;
        }
        for level in &self.manifest.levels {
            for tensor in &level.tensors {
                referenced[self.tensor_index(tensor, blobs)?] = true;
            }
        }
        if referenced.contains(&false) {
            return Err(TritModelError::Schema("unreferenced model payload".into()));
        }
        Ok(())
    }
}

pub(super) fn validate_bindings<S: ModelSource>(
    manifest: ModelManifest,
    source: &mut S,
) -> Result<ModelBindings, S::Error>
where
    S::Error: From<TritModelError>,
{
    manifest.validate().map_err(TritModelError::from)?;
    if !manifest.shards.is_empty() {
        return Err(TritModelError::UnsupportedFeature("sharded model payloads").into());
    }
    let mut index = BTreeMap::<_, Vec<_>>::new();
    for (i, blob) in source.blobs().iter().enumerate() {
        index.entry(*blob.id.as_bytes()).or_default().push(i);
    }
    let mut binding = ModelBindings {
        manifest,
        index,
        tensor_digests: Vec::new(),
        model_ids: Vec::new(),
    };
    for asset in &binding.manifest.assets {
        binding.asset_index(&asset.name, source.blobs())?;
    }
    for level in &binding.manifest.levels {
        let mut digests = Vec::new();
        for tensor in &level.tensors {
            let index = binding.tensor_index(tensor, source.blobs())?;
            let length = blob_length(source, index)?;
            digests.push(match &tensor.storage {
                TensorStorage::Dense { shape, precision } => {
                    dense_semantics(&tensor.name, shape, *precision, index, length, source)?
                }
                TensorStorage::Additive { layout, planes } => {
                    if !matches!(layout.transport, Transport::Raw) {
                        return Err(
                            TritModelError::UnsupportedFeature("entropy model payloads").into()
                        );
                    }
                    let plan = PayloadPlan::new(*layout, planes, length)?;
                    // The semantic visitors validate canonical trits and every
                    // scale window before producing the private digest. Avoid
                    // decoding the same payload once solely for validation.
                    additive_semantics(*layout, planes, &plan, index, source)?
                }
                _ => return Err(TritModelError::UnsupportedFeature("tensor storage").into()),
            });
        }
        binding.tensor_digests.push(digests);
        if let LevelFit::ImportedPrefixOf { level: parent } = &level.fit {
            let parent = binding.level(parent)?;
            for (child, parent) in level.tensors.iter().zip(&parent.tensors) {
                let ci = binding.tensor_index(child, source.blobs())?;
                let pi = binding.tensor_index(parent, source.blobs())?;
                if !is_prefix(child, ci, parent, pi, source)? {
                    return Err(TritModelError::InvalidPrefix(level.name.clone()).into());
                }
            }
        }
    }
    binding.model_ids = binding
        .manifest
        .levels
        .iter()
        .zip(&binding.tensor_digests)
        .map(|(level, digests)| model_digest(&binding.manifest, level, digests))
        .collect::<Result<_, _>>()?;
    Ok(binding)
}

fn blob_length<S: ModelSource>(source: &S, index: usize) -> Result<usize, S::Error>
where
    S::Error: From<TritModelError>,
{
    usize::try_from(source.blobs()[index].length)
        .map_err(|_| TritModelError::Package(TritPackageError::LengthOverflow).into())
}
fn additive_semantics<S: PayloadSource>(
    layout: tritium_schema::AdditiveLayout,
    planes: &PlaneCounts,
    plan: &PayloadPlan<'_>,
    blob: usize,
    source: &mut S,
) -> Result<SemanticTensorDigest, S::Error>
where
    S::Error: From<TritModelError>,
{
    let metadata = TensorSemanticMeta::Additive {
        rows: layout.rows,
        cols: layout.cols,
        tile: layout.tile,
        group: layout.group,
        planes: planes.clone(),
        law: layout.law,
        basis: layout.basis,
    };
    let mut hash = metadata_hasher("tritium.trit.additive.v1", &metadata)?;
    hash.update(&frame_length(plan.trit_count())?);
    // Codec visitors yield small windows. Coalesce their canonical bytes so
    // BLAKE3 can process bulk chunks without retaining a tensor-sized master.
    // Buffering changes only update boundaries, never the framed byte stream.
    let mut staging = [0u8; 16 * 1024];
    let mut used = 0;
    plan.visit_trits(source, blob, |mut chunk| {
        while !chunk.is_empty() {
            let count = (staging.len() - used).min(chunk.len());
            for (byte, trit) in staging[used..used + count].iter_mut().zip(&chunk[..count]) {
                *byte = (trit.get() + 1) as u8;
            }
            used += count;
            chunk = &chunk[count..];
            if used == staging.len() {
                hash.update(&staging);
                used = 0;
            }
        }
        Ok(())
    })?;
    // The partial trit window belongs before the scale frame, including when
    // it is empty (all-pruned tensors or an exact buffer boundary).
    hash.update(&staging[..used]);
    hash.update(&frame_length(plan.scale_len())?);
    plan.visit_scale_bytes(source, blob, |chunk| {
        hash.update(chunk);
        Ok(())
    })?;
    Ok(SemanticTensorDigest::from_bytes(
        *hash.finalize().as_bytes(),
    ))
}
fn dense_semantics<S: PayloadSource>(
    name: &str,
    shape: &[u64],
    precision: ScalePrecision,
    blob: usize,
    length: usize,
    source: &mut S,
) -> Result<SemanticTensorDigest, S::Error>
where
    S::Error: From<TritModelError>,
{
    validate_dense_length(name, shape, precision, length)?;
    let metadata = TensorSemanticMeta::Dense {
        shape: shape.to_vec(),
        precision,
    };
    let mut hash = metadata_hasher("tritium.trit.dense.v1", &metadata)?;
    hash.update(&frame_length(length)?);
    let mut scratch = [0u8; RANGE_BYTES];
    let mut at = 0;
    while at < length {
        let count = (length - at).min(scratch.len());
        source.read_range(blob, at, &mut scratch[..count])?;
        if !dense_finite(precision, &scratch[..count]) {
            return Err(TritModelError::DensePayload(name.into()).into());
        }
        hash.update(&scratch[..count]);
        at += count;
    }
    Ok(SemanticTensorDigest::from_bytes(
        *hash.finalize().as_bytes(),
    ))
}
fn is_prefix<S: ModelSource>(
    child: &TensorMeta,
    ci: usize,
    parent: &TensorMeta,
    pi: usize,
    source: &mut S,
) -> Result<bool, S::Error>
where
    S::Error: From<TritModelError>,
{
    let cl = blob_length(source, ci)?;
    let pl = blob_length(source, pi)?;
    match (&child.storage, &parent.storage) {
        (TensorStorage::Dense { .. }, TensorStorage::Dense { .. }) => {
            if child.storage != parent.storage || cl != pl {
                return Ok(false);
            }
            let mut c = [0u8; RANGE_BYTES];
            let mut p = [0u8; RANGE_BYTES];
            let mut at = 0;
            while at < cl {
                let count = (cl - at).min(c.len());
                source.read_range(ci, at, &mut c[..count])?;
                source.read_range(pi, at, &mut p[..count])?;
                if c[..count] != p[..count] {
                    return Ok(false);
                }
                at += count;
            }
            Ok(true)
        }
        (
            TensorStorage::Additive {
                layout: c,
                planes: cp,
            },
            TensorStorage::Additive {
                layout: p,
                planes: pp,
            },
        ) => {
            let mut normalized = *p;
            normalized.max_planes = c.max_planes;
            normalized.codec = c.codec;
            normalized.allocation = c.allocation;
            if normalized != *c || c.max_planes > p.max_planes {
                return Ok(false);
            }
            let child = PayloadPlan::new(*c, cp, cl)?;
            child.validate(source, ci)?;
            let parent = PayloadPlan::new(*p, pp, pl)?;
            parent.validate(source, pi)?;
            child.is_logical_prefix_of(ci, &parent, pi, source)
        }
        _ => Ok(false),
    }
}

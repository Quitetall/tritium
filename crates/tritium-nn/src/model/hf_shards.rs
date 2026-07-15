//! Seek-backed access to HuggingFace safetensors shards.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::de::{self, IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use tritium_format::{SafeTensorsError, SafeTensorsReader, VisitTensorBytesError};

use crate::error::NnError;

const MAX_HF_INDEX_BYTES: u64 = 100_000_000;
const MAX_HF_TENSORS: usize = 2_000_000;
const MAX_HF_SHARDS: usize = 4_096;
const MAX_HF_TENSOR_NAME_BYTES: usize = 100_000_000;
const MAX_HF_SHAPE_DIMS: usize = 16_000_000;
const MAX_HF_TENSOR_RANK: usize = 32;

#[derive(Debug)]
struct ResolvedShards {
    paths: Vec<PathBuf>,
    weight_map: Option<BTreeMap<String, PathBuf>>,
}

#[derive(Debug)]
struct HfIndex {
    weight_map: BTreeMap<String, String>,
}

struct UniqueWeightMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for UniqueWeightMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueWeightMapVisitor;

        impl<'de> Visitor<'de> for UniqueWeightMapVisitor {
            type Value = UniqueWeightMap;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a weight_map object with unique string mappings")
            }

            fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut entries = BTreeMap::new();
                while let Some(name) = access.next_key::<String>()? {
                    if entries.contains_key(&name) {
                        return Err(de::Error::custom(format!(
                            "duplicate weight_map tensor `{name}`"
                        )));
                    }
                    if entries.len() == MAX_HF_TENSORS {
                        return Err(de::Error::custom(format!(
                            "weight_map exceeds {MAX_HF_TENSORS} tensors"
                        )));
                    }
                    let shard = access.next_value::<String>()?;
                    entries.insert(name, shard);
                }
                Ok(UniqueWeightMap(entries))
            }
        }

        deserializer.deserialize_map(UniqueWeightMapVisitor)
    }
}

impl<'de> Deserialize<'de> for HfIndex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct HfIndexVisitor;

        impl<'de> Visitor<'de> for HfIndexVisitor {
            type Value = HfIndex;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a Hugging Face shard index object")
            }

            fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut keys = BTreeSet::new();
                let mut weight_map = None;
                while let Some(key) = access.next_key::<String>()? {
                    if !keys.insert(key.clone()) {
                        return Err(de::Error::custom(format!("duplicate index key `{key}`")));
                    }
                    if key == "weight_map" {
                        weight_map = Some(access.next_value::<UniqueWeightMap>()?.0);
                    } else {
                        access.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(HfIndex {
                    weight_map: weight_map.ok_or_else(|| de::Error::missing_field("weight_map"))?,
                })
            }
        }

        deserializer.deserialize_map(HfIndexVisitor)
    }
}

#[derive(Debug)]
struct HfShard {
    path: PathBuf,
    reader: RefCell<SafeTensorsReader<File>>,
}

#[derive(Debug)]
struct HfTensorRecord {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    shard_id: usize,
}

#[derive(Debug)]
struct HfTensorLocation {
    dtype: String,
    shape: Vec<usize>,
    shard_id: usize,
}

/// Immutable metadata for one indexed Hugging Face tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct HfTensorMetadata<'a> {
    pub(super) name: &'a str,
    pub(super) dtype: &'a str,
    pub(super) shape: &'a [usize],
    /// Stable ordinal assigned by lexical shard-path order within this set.
    pub(super) shard_id: usize,
    pub(super) shard_path: &'a Path,
}

/// Error from streaming a tensor out of an indexed Hugging Face shard.
#[allow(dead_code, reason = "consumed by the source-bound Qwen adapter seam")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum HfTensorBytesError<E> {
    /// The set does not index the requested tensor.
    MissingTensor(String),
    /// A callback attempted to read the same shard recursively.
    ReentrantShard(PathBuf),
    /// A tensor record references no opened shard.
    InvalidShardId { name: String, shard_id: usize },
    /// The canonical metadata index and its originating reader disagree.
    MetadataMismatch { name: String, shard_path: PathBuf },
    /// The already-open safetensors source failed.
    Source {
        name: String,
        shard_path: PathBuf,
        error: SafeTensorsError,
    },
    /// The caller-provided byte sink stopped the stream.
    Sink(E),
}

impl<E: std::fmt::Display> std::fmt::Display for HfTensorBytesError<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTensor(name) => write!(formatter, "missing HF tensor `{name}`"),
            Self::ReentrantShard(path) => {
                write!(
                    formatter,
                    "reentrant read of safetensors shard {}",
                    path.display()
                )
            }
            Self::InvalidShardId { name, shard_id } => {
                write!(
                    formatter,
                    "HF tensor `{name}` references absent shard {shard_id}"
                )
            }
            Self::MetadataMismatch { name, shard_path } => write!(
                formatter,
                "HF tensor `{name}` metadata disagrees with shard {}",
                shard_path.display()
            ),
            Self::Source {
                name,
                shard_path,
                error,
            } => write!(
                formatter,
                "read HF tensor `{name}` from {}: {error}",
                shard_path.display()
            ),
            Self::Sink(error) => write!(formatter, "HF tensor byte sink: {error}"),
        }
    }
}

impl<E> std::error::Error for HfTensorBytesError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Source { error, .. } => Some(error),
            Self::Sink(error) => Some(error),
            Self::MissingTensor(_)
            | Self::ReentrantShard(_)
            | Self::InvalidShardId { .. }
            | Self::MetadataMismatch { .. } => None,
        }
    }
}

/// Indexed HF shards. Only their JSON headers remain resident; tensor payloads
/// are seek-read on demand.
#[derive(Debug)]
pub(super) struct HfShardSet {
    shards: Vec<HfShard>,
    tensors: Vec<HfTensorRecord>,
}

impl HfShardSet {
    pub(super) fn open(dir: &Path) -> Result<Self, NnError> {
        let ResolvedShards { paths, weight_map } = resolve_shards(dir)?;
        let mut shards = Vec::with_capacity(paths.len());
        let mut by_name = HashMap::new();
        let mut shard_by_path = HashMap::with_capacity(paths.len());
        let mut total_tensors = 0usize;
        let mut total_name_bytes = 0usize;
        let mut total_shape_dims = 0usize;
        for path in paths {
            let file = File::open(&path).map_err(|error| {
                NnError::MissingTensor(format!(
                    "open safetensors shard {}: {error}",
                    path.display()
                ))
            })?;
            let reader = SafeTensorsReader::new(file).map_err(|error| {
                NnError::MissingTensor(format!(
                    "index safetensors shard {}: {error}",
                    path.display()
                ))
            })?;
            let shard_index = shards.len();
            total_tensors = checked_budget_add(
                total_tensors,
                reader.len(),
                MAX_HF_TENSORS,
                "safetensors tensors",
            )?;
            by_name.try_reserve(reader.len()).map_err(|_| {
                NnError::MissingTensor(format!(
                    "could not reserve metadata for {} safetensors tensors",
                    reader.len()
                ))
            })?;
            for name in reader.names() {
                total_name_bytes = checked_budget_add(
                    total_name_bytes,
                    name.len(),
                    MAX_HF_TENSOR_NAME_BYTES,
                    "safetensors tensor-name bytes",
                )?;
                let shape = reader.shape(name).ok_or_else(|| {
                    NnError::MissingTensor(format!(
                        "safetensors tensor `{name}` has no indexed shape"
                    ))
                })?;
                let rank = shape.len();
                if rank > MAX_HF_TENSOR_RANK {
                    return Err(NnError::MissingTensor(format!(
                        "safetensors tensor `{name}` has rank {rank}, limit is {MAX_HF_TENSOR_RANK}"
                    )));
                }
                total_shape_dims = checked_budget_add(
                    total_shape_dims,
                    rank,
                    MAX_HF_SHAPE_DIMS,
                    "safetensors shape dimensions",
                )?;
                let dtype = reader.dtype(name).ok_or_else(|| {
                    NnError::MissingTensor(format!(
                        "safetensors tensor `{name}` has no indexed dtype"
                    ))
                })?;
                if by_name.contains_key(name) {
                    return Err(NnError::MissingTensor(format!(
                        "duplicate safetensors tensor `{name}`"
                    )));
                }
                by_name.insert(
                    try_owned(name, "safetensors tensor name")?,
                    HfTensorLocation {
                        dtype: try_owned(dtype, "safetensors tensor dtype")?,
                        shape: try_owned_shape(shape)?,
                        shard_id: shard_index,
                    },
                );
            }
            shard_by_path.insert(path.clone(), shard_index);
            shards.push(HfShard {
                path,
                reader: RefCell::new(reader),
            });
        }
        let mut tensors = Vec::new();
        tensors.try_reserve_exact(by_name.len()).map_err(|_| {
            NnError::MissingTensor(format!(
                "could not reserve canonical metadata for {} safetensors tensors",
                by_name.len()
            ))
        })?;
        for (name, location) in by_name {
            tensors.push(HfTensorRecord {
                name,
                dtype: location.dtype,
                shape: location.shape,
                shard_id: location.shard_id,
            });
        }
        tensors.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        if let Some(weight_map) = weight_map {
            for (name, expected_path) in &weight_map {
                let expected_shard =
                    shard_by_path.get(expected_path).copied().ok_or_else(|| {
                        NnError::MissingTensor(format!(
                            "index.json references unresolved shard {}",
                            expected_path.display()
                        ))
                    })?;
                let actual_shard = find_tensor(&tensors, name).map(|tensor| tensor.shard_id).ok_or_else(|| {
                    NnError::MissingTensor(format!(
                        "index.json maps tensor `{name}` to {}, but that shard does not contain it",
                        expected_path.display()
                    ))
                })?;
                if actual_shard != expected_shard {
                    return Err(NnError::MissingTensor(format!(
                        "index.json maps tensor `{name}` to {}, but it is stored in {}",
                        expected_path.display(),
                        shards[actual_shard].path.display()
                    )));
                }
            }
            if let Some(unmapped) = tensors
                .iter()
                .find(|tensor| !weight_map.contains_key(&tensor.name))
            {
                return Err(NnError::MissingTensor(format!(
                    "index.json omits tensor `{}` stored in {}",
                    unmapped.name,
                    shards[unmapped.shard_id].path.display()
                )));
            }
        }
        Ok(Self { shards, tensors })
    }

    pub(super) fn names(&self) -> impl Iterator<Item = &str> {
        self.metadata().map(|tensor| tensor.name)
    }

    /// Tensor metadata in canonical global tensor-name order.
    pub(super) fn metadata(&self) -> impl ExactSizeIterator<Item = HfTensorMetadata<'_>> {
        self.tensors.iter().map(|tensor| {
            let shard = &self.shards[tensor.shard_id];
            HfTensorMetadata {
                name: &tensor.name,
                dtype: &tensor.dtype,
                shape: &tensor.shape,
                shard_id: tensor.shard_id,
                shard_path: &shard.path,
            }
        })
    }

    /// Stream one tensor's exact stored payload from its already-open shard.
    ///
    /// Chunks retain [`SafeTensorsReader::try_visit_tensor_bytes`] ordering and
    /// bounds. Sink errors remain typed and stop the source read immediately.
    #[allow(dead_code, reason = "consumed by the source-bound Qwen adapter seam")]
    pub(super) fn try_visit_tensor_bytes<E>(
        &self,
        name: &str,
        max_chunk_bytes: usize,
        visit: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<(), HfTensorBytesError<E>> {
        let tensor = self
            .tensor_record(name)
            .ok_or_else(|| HfTensorBytesError::MissingTensor(name.to_owned()))?;
        let shard =
            self.shards
                .get(tensor.shard_id)
                .ok_or_else(|| HfTensorBytesError::InvalidShardId {
                    name: name.to_owned(),
                    shard_id: tensor.shard_id,
                })?;
        let mut reader = shard
            .reader
            .try_borrow_mut()
            .map_err(|_| HfTensorBytesError::ReentrantShard(shard.path.clone()))?;
        if reader.dtype(name) != Some(tensor.dtype.as_str())
            || reader.shape(name) != Some(tensor.shape.as_slice())
        {
            return Err(HfTensorBytesError::MetadataMismatch {
                name: name.to_owned(),
                shard_path: shard.path.clone(),
            });
        }
        reader
            .try_visit_tensor_bytes(name, max_chunk_bytes, visit)
            .map_err(|error| match error {
                VisitTensorBytesError::Source(error) => HfTensorBytesError::Source {
                    name: name.to_owned(),
                    shard_path: shard.path.clone(),
                    error,
                },
                VisitTensorBytesError::Sink(error) => HfTensorBytesError::Sink(error),
            })
    }

    /// Read a tensor only after its metadata shape exactly matches `expected`.
    pub(super) fn tensor_f32_exact(
        &self,
        name: &str,
        expected: &[usize],
    ) -> Result<Vec<f32>, NnError> {
        self.read_checked(
            name,
            |actual| actual == expected,
            || format!("shape {expected:?}"),
        )
    }

    /// Read a matrix with a known width and an optional exact row count.
    pub(super) fn tensor_f32_matrix(
        &self,
        name: &str,
        rows: Option<usize>,
        columns: usize,
    ) -> Result<Vec<f32>, NnError> {
        self.read_checked(
            name,
            |actual| {
                actual.len() == 2
                    && actual[1] == columns
                    && rows.is_none_or(|expected_rows| actual[0] == expected_rows)
            },
            || match rows {
                Some(rows) => format!("shape [{rows}, {columns}]"),
                None => format!("a rank-2 shape with {columns} columns"),
            },
        )
    }

    fn read_checked(
        &self,
        name: &str,
        shape_matches: impl FnOnce(&[usize]) -> bool,
        expected: impl FnOnce() -> String,
    ) -> Result<Vec<f32>, NnError> {
        let shard_index = self
            .tensor_record(name)
            .map(|tensor| tensor.shard_id)
            .ok_or_else(|| NnError::MissingTensor(name.to_owned()))?;
        let shard = &self.shards[shard_index];
        let mut reader = shard.reader.try_borrow_mut().map_err(|_| {
            NnError::Backend(format!(
                "reentrant read of safetensors shard {}",
                shard.path.display()
            ))
        })?;
        let actual = reader
            .shape(name)
            .ok_or_else(|| NnError::MissingTensor(name.to_owned()))?;
        if !shape_matches(actual) {
            return Err(NnError::MissingTensor(format!(
                "tensor `{name}` in {} has shape {actual:?}, expected {}",
                shard.path.display(),
                expected()
            )));
        }
        reader.tensor_f32(name).map_err(|error| {
            NnError::MissingTensor(format!(
                "read tensor `{name}` from {}: {error}",
                shard.path.display()
            ))
        })
    }

    fn tensor_record(&self, name: &str) -> Option<&HfTensorRecord> {
        find_tensor(&self.tensors, name)
    }
}

fn find_tensor<'a>(tensors: &'a [HfTensorRecord], name: &str) -> Option<&'a HfTensorRecord> {
    tensors
        .binary_search_by(|tensor| tensor.name.as_str().cmp(name))
        .ok()
        .map(|index| &tensors[index])
}

/// Resolve a model directory to its safetensors shard files, in deterministic order:
/// `model.safetensors.index.json` (`weight_map`) → a lone `model.safetensors` → else every
/// `*.safetensors` in the directory, sorted.
fn resolve_shards(dir: &Path) -> Result<ResolvedShards, NnError> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let text = read_bounded_index(&index_path)?;
        let HfIndex {
            weight_map: entries,
        } = serde_json::from_str(&text)
            .map_err(|error| NnError::MissingTensor(format!("parse index: {error}")))?;
        let mut paths = BTreeSet::new();
        let mut weight_map = BTreeMap::new();
        for (name, relative) in entries {
            let relative_path = Path::new(&relative);
            if relative.is_empty()
                || relative_path.is_absolute()
                || relative_path.components().any(|component| {
                    matches!(
                        component,
                        Component::ParentDir | Component::RootDir | Component::Prefix(_)
                    )
                })
            {
                return Err(NnError::MissingTensor(format!(
                    "index.json contains unsafe shard path `{relative}`"
                )));
            }
            let path = dir.join(relative_path);
            paths.insert(path.clone());
            weight_map.insert(name, path);
        }
        if paths.is_empty() {
            return Err(NnError::MissingTensor(
                "index.json lists no shards".to_owned(),
            ));
        }
        if paths.len() > MAX_HF_SHARDS {
            return Err(NnError::MissingTensor(format!(
                "index.json lists {} shards, limit is {MAX_HF_SHARDS}",
                paths.len()
            )));
        }
        return Ok(ResolvedShards {
            paths: paths.into_iter().collect(),
            weight_map: Some(weight_map),
        });
    }

    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(ResolvedShards {
            paths: vec![single],
            weight_map: None,
        });
    }
    let entries = std::fs::read_dir(dir)
        .map_err(|error| NnError::MissingTensor(format!("read dir {}: {error}", dir.display())))?;
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|error| NnError::MissingTensor(format!("read dir entry: {error}")))?
            .path();
        if path
            .extension()
            .is_none_or(|extension| extension != "safetensors")
        {
            continue;
        }
        if paths.len() == MAX_HF_SHARDS {
            return Err(NnError::MissingTensor(format!(
                "model directory lists more than {MAX_HF_SHARDS} safetensors shards"
            )));
        }
        paths.try_reserve(1).map_err(|_| {
            NnError::MissingTensor("could not reserve safetensors shard path".to_owned())
        })?;
        paths.push(path);
    }
    paths.sort();
    if paths.is_empty() {
        return Err(NnError::MissingTensor(format!(
            "no `.safetensors` in {}",
            dir.display()
        )));
    }
    Ok(ResolvedShards {
        paths,
        weight_map: None,
    })
}

fn checked_budget_add(
    current: usize,
    added: usize,
    limit: usize,
    label: &str,
) -> Result<usize, NnError> {
    let total = current.checked_add(added).ok_or_else(|| {
        NnError::MissingTensor(format!("{label} overflow the supported metadata budget"))
    })?;
    if total > limit {
        return Err(NnError::MissingTensor(format!(
            "{label} total {total} exceeds limit {limit}"
        )));
    }
    Ok(total)
}

fn try_owned(value: &str, label: &str) -> Result<String, NnError> {
    let mut owned = String::new();
    owned.try_reserve_exact(value.len()).map_err(|_| {
        NnError::MissingTensor(format!(
            "could not reserve {} bytes for {label}",
            value.len()
        ))
    })?;
    owned.push_str(value);
    Ok(owned)
}

fn try_owned_shape(shape: &[usize]) -> Result<Vec<usize>, NnError> {
    let mut owned = Vec::new();
    owned.try_reserve_exact(shape.len()).map_err(|_| {
        NnError::MissingTensor(format!(
            "could not reserve {} dimensions for safetensors tensor shape",
            shape.len()
        ))
    })?;
    owned.extend_from_slice(shape);
    Ok(owned)
}

fn read_bounded_index(path: &Path) -> Result<String, NnError> {
    // HF cache snapshots intentionally use symlinks into their sibling blob store. The
    // lexical index path is confined, but following those operator-provided symlinks is required
    // for standard cache compatibility; `model_dir` is therefore a trusted filesystem boundary.
    let file =
        File::open(path).map_err(|error| NnError::MissingTensor(format!("open index: {error}")))?;
    let declared_len = file
        .metadata()
        .map_err(|error| NnError::MissingTensor(format!("stat index: {error}")))?
        .len();
    if declared_len > MAX_HF_INDEX_BYTES {
        return Err(NnError::MissingTensor(format!(
            "index.json is {declared_len} bytes, limit is {MAX_HF_INDEX_BYTES}"
        )));
    }
    let mut text = String::new();
    text.try_reserve_exact(usize::try_from(declared_len).unwrap_or(usize::MAX))
        .map_err(|_| {
            NnError::MissingTensor(format!(
                "could not reserve {declared_len} bytes for index.json"
            ))
        })?;
    file.take(MAX_HF_INDEX_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|error| NnError::MissingTensor(format!("read index: {error}")))?;
    if text.len() as u64 > MAX_HF_INDEX_BYTES {
        return Err(NnError::MissingTensor(format!(
            "index.json exceeds the {MAX_HF_INDEX_BYTES}-byte limit"
        )));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::{HfShardSet, HfTensorBytesError};
    use tritium_format::SafeTensorsError;

    fn safetensors(name: &str) -> Vec<u8> {
        safetensors_values(name, &[1.0])
    }

    fn safetensors_values(name: &str, values: &[f32]) -> Vec<u8> {
        let payload = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        safetensors_raw(name, "F32", &[values.len()], &payload)
    }

    fn safetensors_raw(name: &str, dtype: &str, shape: &[usize], payload: &[u8]) -> Vec<u8> {
        let byte_len = payload.len();
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":{shape:?},"data_offsets":[0,{byte_len}]}}}}"#
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    fn temp_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("tritium-hf-shards-{label}-{}", std::process::id()))
    }

    #[test]
    fn metadata_is_global_name_order_with_stable_shard_identity() {
        let dir = temp_dir("metadata-order");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.safetensors"), safetensors("z.weight")).unwrap();
        std::fs::write(
            dir.join("b.safetensors"),
            safetensors_raw("a.weight", "BF16", &[2], &[0, 0, 128, 63]),
        )
        .unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"z.weight":"a.safetensors","a.weight":"b.safetensors"}}"#,
        )
        .unwrap();

        let shards = HfShardSet::open(&dir).unwrap();
        let actual = shards
            .metadata()
            .map(|tensor| {
                (
                    tensor.name,
                    tensor.dtype,
                    tensor.shape,
                    tensor.shard_id,
                    tensor
                        .shard_path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                )
            })
            .collect::<Vec<_>>();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(
            actual,
            [
                ("a.weight", "BF16", &[2][..], 1, "b.safetensors".to_owned()),
                ("z.weight", "F32", &[1][..], 0, "a.safetensors".to_owned()),
            ]
        );
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SinkStop {
        Enough,
    }

    #[test]
    fn raw_visitor_uses_open_handle_and_preserves_sink_error() {
        let dir = temp_dir("raw-visitor");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let values = [1.0f32, -2.0, 3.5];
        std::fs::write(&path, safetensors_values("x", &values)).unwrap();
        let shards = HfShardSet::open(&dir).unwrap();
        std::fs::remove_file(&path).unwrap();

        let mut actual = Vec::new();
        let mut reentrant_checked = false;
        shards
            .try_visit_tensor_bytes("x", 5, |chunk| {
                assert!(!chunk.is_empty());
                assert!(chunk.len() <= 5);
                if !reentrant_checked {
                    let nested = shards.try_visit_tensor_bytes("x", 2, |_| Ok::<(), SinkStop>(()));
                    assert!(matches!(nested, Err(HfTensorBytesError::ReentrantShard(_))));
                    reentrant_checked = true;
                }
                actual.extend_from_slice(chunk);
                Ok::<(), SinkStop>(())
            })
            .unwrap();
        let expected = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert!(reentrant_checked);

        let error = shards
            .try_visit_tensor_bytes("x", 2, |_| Err(SinkStop::Enough))
            .unwrap_err();
        let source_error = shards
            .try_visit_tensor_bytes("x", 0, |_| Ok::<(), SinkStop>(()))
            .unwrap_err();
        let missing_error = shards
            .try_visit_tensor_bytes("missing", 2, |_| Ok::<(), SinkStop>(()))
            .unwrap_err();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(error, HfTensorBytesError::Sink(SinkStop::Enough));
        assert!(matches!(
            source_error,
            HfTensorBytesError::Source {
                error: SafeTensorsError::InvalidChunkSize { requested: 0 },
                ..
            }
        ));
        assert_eq!(
            missing_error,
            HfTensorBytesError::MissingTensor("missing".to_owned())
        );
    }

    #[test]
    fn duplicate_tensor_names_across_shards_fail_closed() {
        let dir = temp_dir("duplicate");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.safetensors"), safetensors("x")).unwrap();
        std::fs::write(dir.join("b.safetensors"), safetensors("x")).unwrap();
        std::fs::write(dir.join("z.safetensors"), b"unreadable later shard").unwrap();

        let error = HfShardSet::open(&dir).unwrap_err();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            error
                .to_string()
                .contains("duplicate safetensors tensor `x`")
        );
    }

    #[test]
    fn shard_index_rejects_duplicate_non_string_and_escaping_mappings() {
        for (label, index) in [
            ("non-string", r#"{"weight_map":{"x":7}}"#),
            (
                "duplicate-root",
                r#"{"weight_map":{"x":"a.safetensors"},"weight_map":{"x":"b.safetensors"}}"#,
            ),
            (
                "duplicate-tensor",
                r#"{"weight_map":{"x":"a.safetensors","x":"b.safetensors"}}"#,
            ),
            ("parent", r#"{"weight_map":{"x":"../outside.safetensors"}}"#),
            (
                "absolute",
                r#"{"weight_map":{"x":"/tmp/outside.safetensors"}}"#,
            ),
        ] {
            let dir = temp_dir(label);
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
            assert!(HfShardSet::open(&dir).is_err(), "index case {label}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn shard_index_must_match_the_declared_tensor_location() {
        for (label, index) in [
            ("missing", r#"{"weight_map":{"missing":"a.safetensors"}}"#),
            (
                "misplaced",
                r#"{"weight_map":{"x":"b.safetensors","y":"a.safetensors"}}"#,
            ),
        ] {
            let dir = temp_dir(label);
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("a.safetensors"), safetensors("x")).unwrap();
            if label == "misplaced" {
                std::fs::write(dir.join("b.safetensors"), safetensors("y")).unwrap();
            }
            std::fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
            assert!(HfShardSet::open(&dir).is_err(), "index case {label}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn shard_index_must_cover_every_tensor_in_referenced_shards() {
        let dir = temp_dir("incomplete-index");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let header = concat!(
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"#,
            r#""y":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        std::fs::write(dir.join("a.safetensors"), bytes).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":"a.safetensors"}}"#,
        )
        .unwrap();

        let error = HfShardSet::open(&dir).unwrap_err();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(error.to_string().contains("omits tensor `y`"));
    }

    #[test]
    fn oversized_sparse_index_fails_before_reading() {
        let dir = temp_dir("oversized");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let index = std::fs::File::create(dir.join("model.safetensors.index.json")).unwrap();
        index.set_len(super::MAX_HF_INDEX_BYTES + 1).unwrap();
        let error = HfShardSet::open(&dir).unwrap_err();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(error.to_string().contains("index.json is"));
    }

    #[test]
    fn excessive_tensor_rank_fails_the_aggregate_metadata_budget() {
        let dir = temp_dir("rank");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let shape = std::iter::repeat_n("1", super::MAX_HF_TENSOR_RANK + 1)
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(r#"{{"x":{{"dtype":"F32","shape":[{shape}],"data_offsets":[0,4]}}}}"#);
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();

        let error = HfShardSet::open(&dir).unwrap_err();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(error.to_string().contains("rank 33"));
    }
}

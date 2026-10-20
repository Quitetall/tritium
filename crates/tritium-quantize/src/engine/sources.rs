//! Raw-byte adapters; source identity is computed by the engine, not trusted here.
use super::model::bound;
use super::{QuantEngineError, WeightSource};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    io::{Read, Seek},
};
use tritium_format::SafeTensorsReader;
use tritium_schema::{QuantSourceTensor, ScalePrecision};

/// Original in-memory tensor, preserving dtype bits until the engine widens it.
#[derive(Clone, Debug)]
pub struct RawTensor {
    /// Exact original logical shape.
    pub shape: Vec<u64>,
    /// Exact original float precision.
    pub precision: ScalePrecision,
    /// Original row-major little-endian payload, not predecoded weights.
    pub bytes: Vec<u8>,
}
/// Borrowing original in-memory tensor/asset/Gram adapter.
#[derive(Clone, Debug, Default)]
pub struct InMemoryWeightSource {
    /// Complete named originals. No hashes are supplied.
    pub tensors: BTreeMap<String, RawTensor>,
    /// Complete original assets, retained byte-for-byte.
    pub assets: BTreeMap<String, Vec<u8>>,
    /// Explicit original input Grams for requested non-identity metrics.
    pub grams: BTreeMap<String, Vec<f64>>,
}
impl WeightSource for InMemoryWeightSource {
    fn inventory(&self) -> Result<Vec<QuantSourceTensor>, QuantEngineError> {
        Ok(self
            .tensors
            .iter()
            .map(|(name, t)| QuantSourceTensor {
                name: name.clone(),
                shape: t.shape.clone(),
                precision: t.precision,
            })
            .collect())
    }
    fn asset_names(&self) -> Vec<String> {
        self.assets.keys().cloned().collect()
    }
    fn tensor(&mut self, name: &str, max_bytes: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        borrowed(
            &self
                .tensors
                .get(name)
                .ok_or_else(|| QuantEngineError::Source("missing tensor".into()))?
                .bytes,
            max_bytes,
        )
    }
    fn asset(&mut self, name: &str, max_bytes: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        borrowed(
            self.assets
                .get(name)
                .ok_or_else(|| QuantEngineError::Source("missing asset".into()))?,
            max_bytes,
        )
    }
    fn input_gram(
        &mut self,
        name: &str,
        max_entries: usize,
    ) -> Result<Cow<'_, [f64]>, QuantEngineError> {
        let values = self
            .grams
            .get(name)
            .ok_or_else(|| QuantEngineError::Source("missing requested Gram".into()))?;
        bound(values.len() as u64, max_entries as u64, "input Gram")?;
        Ok(Cow::Borrowed(values))
    }
}
fn borrowed(bytes: &[u8], maximum: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
    bound(bytes.len() as u64, maximum as u64, "original bytes")?;
    Ok(Cow::Borrowed(bytes))
}

/// Already-open multi-shard safetensors adapter with bounded raw-byte reads.
/// Files are not frozen: this is observed tensor content, not HF/container identity.
pub struct SafeTensorsWeightSource<R: Read + Seek> {
    readers: Vec<SafeTensorsReader<R>>,
    inventory: Vec<QuantSourceTensor>,
    shards: BTreeMap<String, usize>,
    assets: BTreeMap<String, Vec<u8>>,
    grams: BTreeMap<String, Vec<f64>>,
}
impl<R: Read + Seek> core::fmt::Debug for SafeTensorsWeightSource<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SafeTensorsWeightSource")
            .field("shard_count", &self.readers.len())
            .field("tensor_count", &self.inventory.len())
            .finish_non_exhaustive()
    }
}
impl<R: Read + Seek> SafeTensorsWeightSource<R> {
    /// Validate all source shard metadata and reject duplicate cross-shard names.
    /// # Errors
    /// Malformed sources, unsupported original precision or duplicate tensor names.
    pub fn new(
        readers: impl IntoIterator<Item = R>,
        assets: BTreeMap<String, Vec<u8>>,
        grams: BTreeMap<String, Vec<f64>>,
    ) -> Result<Self, QuantEngineError> {
        let readers = readers
            .into_iter()
            .map(|reader| {
                SafeTensorsReader::new(reader).map_err(|e| QuantEngineError::Source(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut shards = BTreeMap::new();
        let mut inventory = Vec::new();
        for (index, reader) in readers.iter().enumerate() {
            for name in reader.names() {
                if shards.insert(name.to_owned(), index).is_some() {
                    return Err(QuantEngineError::Binding("duplicate cross-shard tensor"));
                }
                let precision = match reader.dtype(name) {
                    Some("F16") => ScalePrecision::F16,
                    Some("BF16") => ScalePrecision::Bf16,
                    Some("F32") => ScalePrecision::F32,
                    _ => return Err(QuantEngineError::Binding("unsupported source dtype")),
                };
                let shape = reader
                    .shape(name)
                    .ok_or(QuantEngineError::Binding("source shape absent"))?
                    .iter()
                    .map(|&d| u64::try_from(d).map_err(|_| QuantEngineError::Limit("source shape")))
                    .collect::<Result<Vec<_>, _>>()?;
                let tensor = QuantSourceTensor {
                    name: name.into(),
                    shape,
                    precision,
                };
                tensor.byte_len().map_err(QuantEngineError::Plan)?;
                inventory.push(tensor);
            }
        }
        inventory.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self {
            readers,
            inventory,
            shards,
            assets,
            grams,
        })
    }
}
impl<R: Read + Seek> WeightSource for SafeTensorsWeightSource<R> {
    fn inventory(&self) -> Result<Vec<QuantSourceTensor>, QuantEngineError> {
        Ok(self.inventory.clone())
    }
    fn asset_names(&self) -> Vec<String> {
        self.assets.keys().cloned().collect()
    }
    fn tensor(&mut self, name: &str, max_bytes: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        let index = *self
            .shards
            .get(name)
            .ok_or_else(|| QuantEngineError::Source("missing tensor".into()))?;
        let meta = self
            .inventory
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .map_err(|_| QuantEngineError::Binding("source metadata absent"))?;
        let len = self.inventory[meta]
            .byte_len()
            .map_err(QuantEngineError::Plan)?;
        bound(len, max_bytes as u64, "source bytes")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(max_bytes.min(super::model::platform(len, "source bytes")?))
            .map_err(|_| QuantEngineError::Allocation)?;
        self.readers[index]
            .try_visit_tensor_bytes(name, 64 * 1024, |chunk| {
                let next = bytes
                    .len()
                    .checked_add(chunk.len())
                    .ok_or(QuantEngineError::Limit("source bytes"))?;
                bound(next as u64, len, "source bytes")?;
                bytes.extend_from_slice(chunk);
                Ok::<_, QuantEngineError>(())
            })
            .map_err(|e| match e {
                tritium_format::VisitTensorBytesError::Source(e) => {
                    QuantEngineError::Source(e.to_string())
                }
                tritium_format::VisitTensorBytesError::Sink(e) => e,
            })?;
        Ok(Cow::Owned(bytes))
    }
    fn asset(&mut self, name: &str, max_bytes: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        borrowed(
            self.assets
                .get(name)
                .ok_or_else(|| QuantEngineError::Source("missing asset".into()))?,
            max_bytes,
        )
    }
    fn input_gram(
        &mut self,
        name: &str,
        max_entries: usize,
    ) -> Result<Cow<'_, [f64]>, QuantEngineError> {
        let values = self
            .grams
            .get(name)
            .ok_or_else(|| QuantEngineError::Source("missing requested Gram".into()))?;
        bound(values.len() as u64, max_entries as u64, "input Gram")?;
        Ok(Cow::Borrowed(values))
    }
}

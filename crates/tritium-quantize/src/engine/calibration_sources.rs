//! Actual immutable caches and bounded .trit files at one injected cache seam.
use super::model::{bound, platform};
use super::{CalibrationCache, QuantEngineError};
use std::{collections::BTreeMap, fs::File, io::Read, path::PathBuf};
use tritium_schema::{CalibrationCacheBinding, CalibrationCacheId};

/// Actual cache provider; expected hashes never substitute for reading actual data.
pub trait CalibrationSource {
    /// Complete sorted configured cache names, checked against the model request.
    fn names(&self) -> Vec<String>;
    /// Borrow the actual immutable cache after physical/logical admission.
    /// File-backed adapters verify the existing .trit transport and actual identity.
    fn cache(
        &mut self,
        expected: &CalibrationCacheBinding,
        max_package_bytes: u64,
    ) -> Result<&CalibrationCache, QuantEngineError>;
}
/// Own actual already-finalized caches; caller residency is not full-RSS admission.
#[derive(Default)]
pub struct InMemoryCalibrationSource {
    caches: BTreeMap<String, CalibrationCache>,
}
impl core::fmt::Debug for InMemoryCalibrationSource {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("InMemoryCalibrationSource")
            .field("cache_count", &self.caches.len())
            .finish()
    }
}
impl InMemoryCalibrationSource {
    /// Own actual caches without copying their payload arrays; reject duplicate names.
    /// # Errors
    /// Duplicate tensor cache inventory.
    pub fn new(
        caches: impl IntoIterator<Item = CalibrationCache>,
    ) -> Result<Self, QuantEngineError> {
        let mut source = Self::default();
        for cache in caches {
            if source
                .caches
                .insert(cache.binding().request.tensor.name.clone(), cache)
                .is_some()
            {
                return Err(QuantEngineError::Binding("duplicate calibration cache"));
            }
        }
        Ok(source)
    }
}
impl CalibrationSource for InMemoryCalibrationSource {
    fn names(&self) -> Vec<String> {
        self.caches.keys().cloned().collect()
    }
    fn cache(
        &mut self,
        expected: &CalibrationCacheBinding,
        _: u64,
    ) -> Result<&CalibrationCache, QuantEngineError> {
        let cache = self
            .caches
            .get(&expected.request.tensor.name)
            .ok_or(QuantEngineError::Binding("missing calibration cache"))?;
        if cache.binding() != expected {
            return Err(QuantEngineError::Binding(
                "actual calibration cache differs",
            ));
        }
        Ok(cache)
    }
}
/// Configured .trit files; only the current verified decoded cache is retained.
/// Whole encoded input and decoded cache overlap during loading; not mmap/spool/RSS.
#[derive(Default)]
pub struct TritCalibrationSource {
    paths: BTreeMap<String, PathBuf>,
    current: Option<CalibrationCache>,
}
impl core::fmt::Debug for TritCalibrationSource {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TritCalibrationSource")
            .field("cache_count", &self.paths.len())
            .finish_non_exhaustive()
    }
}
impl TritCalibrationSource {
    /// Configure explicit named cache files; do not derive paths from tensor names.
    /// # Errors
    /// Duplicate configured tensor name.
    pub fn new(
        paths: impl IntoIterator<Item = (String, PathBuf)>,
    ) -> Result<Self, QuantEngineError> {
        let mut source = Self::default();
        for (name, path) in paths {
            if source.paths.insert(name, path).is_some() {
                return Err(QuantEngineError::Binding("duplicate calibration cache"));
            }
        }
        Ok(source)
    }
}
impl CalibrationSource for TritCalibrationSource {
    fn names(&self) -> Vec<String> {
        self.paths.keys().cloned().collect()
    }
    fn cache(
        &mut self,
        expected: &CalibrationCacheBinding,
        maximum: u64,
    ) -> Result<&CalibrationCache, QuantEngineError> {
        // The previous borrow has ended before a new mutable request. Release
        // its decoded payload before loading another encoded/decoded pair.
        self.current = None;
        let path = self
            .paths
            .get(&expected.request.tensor.name)
            .ok_or(QuantEngineError::Binding("missing calibration cache"))?;
        let mut file = File::open(path)?;
        let size = file.metadata()?.len();
        bound(size, maximum, "calibration package bytes")?;
        let size = platform(size, "calibration package bytes")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| QuantEngineError::Allocation)?;
        bytes.resize(size, 0);
        file.read_exact(&mut bytes)?;
        if file.read(&mut [0u8; 1])? != 0 {
            return Err(QuantEngineError::Binding(
                "calibration file grew during read",
            ));
        }
        let cache = CalibrationCache::decode_package(
            &bytes,
            CalibrationCacheId::from_bytes(expected.source.cache.0),
            expected.request.limits,
            platform(maximum, "calibration package bytes")?,
        )
        .map_err(QuantEngineError::Cache)?;
        if cache.binding() != expected {
            return Err(QuantEngineError::Binding(
                "actual calibration cache differs",
            ));
        }
        self.current = Some(cache);
        self.current
            .as_ref()
            .ok_or(QuantEngineError::Binding("calibration cache absent"))
    }
}

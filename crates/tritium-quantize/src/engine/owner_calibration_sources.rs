//! Actual owner streams; numerical/cache interpretation stays in the common receiver.
use super::{OwnerScoreCache, QuantEngineError, read_original_file_bounded};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};
use tritium_schema::{OwnerScoreCacheLimits, OwnerScoreFitPlan};

/// Actual immutable windows for one requested owner; no missing-stream fallback.
pub trait OwnerCalibrationSource {
    /// Complete sorted canonical tensor-owner names.
    fn names(&self) -> Vec<String>;
    /// Actual verified references in requested window order, including repeats.
    fn stream(
        &mut self,
        plan: &OwnerScoreFitPlan,
        max_package_bytes: u64,
    ) -> Result<Vec<&OwnerScoreCache>, QuantEngineError>;
}
/// Own distinct actual caches without cloning their numerical payloads.
#[derive(Debug, Default)]
pub struct InMemoryOwnerCalibrationSource {
    caches: BTreeMap<String, BTreeMap<[u8; 32], OwnerScoreCache>>,
}
impl InMemoryOwnerCalibrationSource {
    /// Construct from actual caches; reject duplicate owner/identity entries.
    pub fn new(
        streams: impl IntoIterator<Item = (String, Vec<OwnerScoreCache>)>,
    ) -> Result<Self, QuantEngineError> {
        let mut result = Self::default();
        for (name, caches) in streams {
            let mut unique = BTreeMap::new();
            for cache in caches {
                if cache.binding().content.tensor.name != name
                    || unique.insert(*cache.id().as_bytes(), cache).is_some()
                {
                    return Err(QuantEngineError::Binding(
                        "duplicate or foreign owner cache",
                    ));
                }
            }
            if unique.is_empty() || result.caches.insert(name, unique).is_some() {
                return Err(QuantEngineError::Binding("duplicate or empty owner stream"));
            }
        }
        Ok(result)
    }
}
impl OwnerCalibrationSource for InMemoryOwnerCalibrationSource {
    fn names(&self) -> Vec<String> {
        self.caches.keys().cloned().collect()
    }
    fn stream(
        &mut self,
        plan: &OwnerScoreFitPlan,
        _: u64,
    ) -> Result<Vec<&OwnerScoreCache>, QuantEngineError> {
        plan.validate()
            .map_err(|_| QuantEngineError::Binding("invalid owner fit request"))?;
        let name = &plan.windows[0].binding.content.tensor.name;
        let caches = self
            .caches
            .get(name)
            .ok_or(QuantEngineError::Binding("owner stream absent"))?;
        check_ids(caches.keys(), plan)?;
        ordered(caches, plan)
    }
}
/// Configured .trit inputs; only the current owner's distinct windows are decoded.
/// Whole input/decoded overlaps remain; this is not mmap/spool/process-RSS admission.
#[derive(Debug, Default)]
pub struct TritOwnerCalibrationSource {
    paths: BTreeMap<String, BTreeMap<[u8; 32], PathBuf>>,
    current: BTreeMap<[u8; 32], OwnerScoreCache>,
}
impl TritOwnerCalibrationSource {
    /// Configure identity-addressed files; actual bindings are verified on load.
    pub fn new(
        paths: BTreeMap<String, BTreeMap<[u8; 32], PathBuf>>,
    ) -> Result<Self, QuantEngineError> {
        if paths
            .iter()
            .any(|(name, files)| name.is_empty() || files.is_empty())
        {
            return Err(QuantEngineError::Binding("empty owner path inventory"));
        }
        Ok(Self {
            paths,
            current: BTreeMap::new(),
        })
    }
}
impl OwnerCalibrationSource for TritOwnerCalibrationSource {
    fn names(&self) -> Vec<String> {
        self.paths.keys().cloned().collect()
    }
    fn stream(
        &mut self,
        plan: &OwnerScoreFitPlan,
        max_package_bytes: u64,
    ) -> Result<Vec<&OwnerScoreCache>, QuantEngineError> {
        plan.validate()
            .map_err(|_| QuantEngineError::Binding("invalid owner fit request"))?;
        let paths = self
            .paths
            .get(&plan.windows[0].binding.content.tensor.name)
            .ok_or(QuantEngineError::Binding("owner paths absent"))?;
        check_ids(paths.keys(), plan)?;
        self.current.clear();
        let mut pending = BTreeMap::new();
        for window in &plan.windows {
            let b = &window.binding;
            if pending.contains_key(b.cache.as_bytes()) {
                continue;
            }
            let bytes = read_original_file_bounded(&paths[b.cache.as_bytes()], max_package_bytes)?;
            let receiving = OwnerScoreCacheLimits {
                max_payload_bytes: b.payload_bytes,
                max_context_bytes: b.context_bytes,
                max_context_tensors: b.content.context.len() as u64,
                max_invocations: b.content.invocations.len() as u64,
                max_samples: b.samples,
            };
            let maximum = usize::try_from(max_package_bytes)
                .map_err(|_| QuantEngineError::Limit("owner package bytes"))?;
            let cache = OwnerScoreCache::decode_package(&bytes, b, receiving, maximum)
                .map_err(QuantEngineError::OwnerCache)?;
            pending.insert(*b.cache.as_bytes(), cache);
        }
        self.current = pending;
        ordered(&self.current, plan)
    }
}
fn check_ids<'a>(
    actual: impl Iterator<Item = &'a [u8; 32]>,
    plan: &OwnerScoreFitPlan,
) -> Result<(), QuantEngineError> {
    let expected = plan
        .windows
        .iter()
        .map(|w| *w.binding.cache.as_bytes())
        .collect::<BTreeSet<_>>();
    if !actual.copied().eq(expected) {
        return Err(QuantEngineError::Binding(
            "complete owner window inventory changed",
        ));
    }
    Ok(())
}
fn ordered<'a>(
    caches: &'a BTreeMap<[u8; 32], OwnerScoreCache>,
    plan: &OwnerScoreFitPlan,
) -> Result<Vec<&'a OwnerScoreCache>, QuantEngineError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(plan.windows.len())
        .map_err(|_| QuantEngineError::Limit("owner references"))?;
    for window in &plan.windows {
        let cache = caches
            .get(window.binding.cache.as_bytes())
            .ok_or(QuantEngineError::Binding("owner window absent"))?;
        if cache.binding() != &window.binding {
            return Err(QuantEngineError::Binding("actual owner binding changed"));
        }
        result.push(cache);
    }
    Ok(result)
}

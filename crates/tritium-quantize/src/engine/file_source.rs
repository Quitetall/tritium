//! Shared CLI/Python filesystem adapter; the engine owns numerics and identities.
use super::{QuantEngineError, SafeTensorsWeightSource, TritCalibrationSource, WeightSource};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};
use tritium_schema::{DirectFitMetric, ModelQuantPlan, QuantActionRef, QuantSourceTensor};

/// File-backed original sources and lazily loaded requested raw f64 Grams.
/// Observed content is not an immutable writer snapshot or official source authentication.
#[derive(Debug)]
pub struct FileWeightSource {
    raw: SafeTensorsWeightSource<File>,
    grams: BTreeMap<String, PathBuf>,
    gram_sizes: BTreeMap<String, u64>,
}
impl FileWeightSource {
    /// Admit plan/binding inventories before opening original payload files.
    /// Source/header/solver costs retain their existing bounds, not full RSS admission.
    /// # Errors
    /// Invalid plan, missing/extra paths, unsupported action, file or source failure.
    pub fn open(
        plan: ModelQuantPlan<'_>,
        shards: &[PathBuf],
        assets: BTreeMap<String, PathBuf>,
        grams: BTreeMap<String, PathBuf>,
    ) -> Result<Self, QuantEngineError> {
        plan.validate().map_err(QuantEngineError::Plan)?;
        let mut gram_sizes = BTreeMap::new();
        for tensor in plan.tensors() {
            match tensor.action {
                QuantActionRef::Fit {
                    plan,
                    metric: DirectFitMetric::InputGramF64 { .. },
                } => {
                    let entries = plan
                        .layout
                        .cols
                        .checked_mul(plan.layout.cols)
                        .ok_or(QuantEngineError::Limit("Gram shape overflow"))?;
                    if entries > plan.limits.max_metric_entries {
                        return Err(QuantEngineError::Limit(
                            "Gram shape exceeds requested metric limit",
                        ));
                    }
                    gram_sizes.insert(
                        tensor.source.name.clone(),
                        entries
                            .checked_mul(8)
                            .ok_or(QuantEngineError::Limit("Gram byte overflow"))?,
                    );
                }
                QuantActionRef::KeepDense
                | QuantActionRef::CalibratedFit(_)
                | QuantActionRef::Fit {
                    metric: DirectFitMetric::Identity,
                    ..
                } => {}
                _ => {
                    return Err(QuantEngineError::Binding(
                        "unsupported typed plan action or metric",
                    ));
                }
            }
        }
        check_inventory(&grams, gram_sizes.keys().map(String::as_str), "raw Gram")?;
        check_inventory(&assets, plan.assets().iter().map(String::as_str), "asset")?;
        if shards.is_empty() {
            return Err(QuantEngineError::Source(
                "at least one --source-shard is required".into(),
            ));
        }
        let readers = shards
            .iter()
            .map(|path| open_original_file(path))
            .collect::<Result<Vec<_>, _>>()?;
        let mut remaining = plan.limits().max_asset_bytes;
        let assets = assets
            .into_iter()
            .map(|(name, path)| {
                let bytes = read_original_file_bounded(&path, remaining)?;
                remaining -= bytes.len() as u64;
                Ok((name, bytes))
            })
            .collect::<Result<_, QuantEngineError>>()?;
        Ok(Self {
            raw: SafeTensorsWeightSource::new(readers, assets, BTreeMap::new())?,
            grams,
            gram_sizes,
        })
    }
}

/// Configure the exact named calibrated-cache inventory without payload reads.
/// # Errors
/// Invalid plan, missing/extra cache bindings or duplicate configuration.
pub fn file_calibration_source(
    plan: ModelQuantPlan<'_>,
    paths: BTreeMap<String, PathBuf>,
) -> Result<TritCalibrationSource, QuantEngineError> {
    plan.validate().map_err(QuantEngineError::Plan)?;
    let expected = plan.tensors().filter_map(|tensor| match tensor.action {
        QuantActionRef::CalibratedFit(_) => Some(tensor.source.name.as_str()),
        _ => None,
    });
    check_inventory(&paths, expected, "calibration cache")?;
    TritCalibrationSource::new(paths)
}

fn check_inventory<'a>(
    paths: &BTreeMap<String, PathBuf>,
    expected: impl Iterator<Item = &'a str>,
    kind: &str,
) -> Result<(), QuantEngineError> {
    if !paths.keys().map(String::as_str).eq(expected) {
        return Err(QuantEngineError::Source(format!(
            "{kind} input inventory does not match plan"
        )));
    }
    Ok(())
}

/// Open a caller-selected ordinary file, retaining its handle for consumption.
/// Symlinked caller paths retain the existing CLI policy; this is not path custody.
/// # Errors
/// Metadata/open failure or a non-regular source.
pub fn open_original_file(path: &Path) -> Result<File, QuantEngineError> {
    if !std::fs::metadata(path)?.is_file() {
        return Err(QuantEngineError::Source(format!(
            "input must be a regular file: {}",
            path.display()
        )));
    }
    let file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(QuantEngineError::Source(
            "opened input is not a regular file".into(),
        ));
    }
    Ok(file)
}

/// Read an exact ordinary file after checking its initial physical byte ceiling.
/// Detects short/growing reads, not all same-size concurrent mutations.
/// # Errors
/// File failure, excessive/platform-unrepresentable size or changed length.
pub fn read_original_file_bounded(path: &Path, max: u64) -> Result<Vec<u8>, QuantEngineError> {
    let mut file = open_original_file(path)?;
    let len = file.metadata()?.len();
    if len > max {
        return Err(QuantEngineError::Limit("input exceeds byte limit"));
    }
    let len = usize::try_from(len)
        .map_err(|_| QuantEngineError::Limit("file size exceeds platform limit"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| QuantEngineError::Allocation)?;
    bytes.resize(len, 0);
    file.read_exact(&mut bytes)?;
    if file.read(&mut [0])? != 0 {
        return Err(QuantEngineError::Source("input grew during read".into()));
    }
    Ok(bytes)
}

impl WeightSource for FileWeightSource {
    fn inventory(&self) -> Result<Vec<QuantSourceTensor>, QuantEngineError> {
        self.raw.inventory()
    }
    fn asset_names(&self) -> Vec<String> {
        self.raw.asset_names()
    }
    fn tensor(&mut self, name: &str, max: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        self.raw.tensor(name, max)
    }
    fn asset(&mut self, name: &str, max: usize) -> Result<Cow<'_, [u8]>, QuantEngineError> {
        self.raw.asset(name, max)
    }
    fn input_gram(&mut self, name: &str, max: usize) -> Result<Cow<'_, [f64]>, QuantEngineError> {
        let path = self
            .grams
            .get(name)
            .ok_or_else(|| QuantEngineError::Source("missing requested raw Gram file".into()))?;
        let expected = *self
            .gram_sizes
            .get(name)
            .ok_or(QuantEngineError::Binding("missing requested Gram shape"))?;
        if expected / 8 > max as u64 {
            return Err(QuantEngineError::Limit(
                "requested Gram exceeds engine limit",
            ));
        }
        let bytes = read_original_file_bounded(path, expected)?;
        if bytes.len() as u64 != expected {
            return Err(QuantEngineError::Source(format!(
                "Gram file has wrong square length: {name}"
            )));
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(bytes.len() / 8)
            .map_err(|_| QuantEngineError::Allocation)?;
        values.extend(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|bytes| f64::from_le_bytes(*bytes)),
        );
        Ok(Cow::Owned(values))
    }
}

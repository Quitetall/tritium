//! Bounded paired payload capture, final identity, then existing reducer replay.
//! Byte verification is not independent proof of forward/backward/recipe execution.

mod transport;

use super::{
    CalibratedTensor, CalibrationBatch, CalibrationCollector, CalibrationError, CalibrationInput,
    CalibrationOutputBatch,
};
use tritium_schema::{
    BlobId, CalibrationCacheBinding, CalibrationCacheId, CalibrationCacheRequest,
    CalibrationConsumer, CalibrationOutput, CalibrationPartition, CalibrationPlan,
    CalibrationPlanError, CalibrationSource, FitInputDigest, ScalePrecision,
};

/// Cache admission/capture/replay failure; no fallback is performed.
#[derive(Debug)]
#[non_exhaustive]
pub enum CalibrationCacheError {
    /// Existing canonical package or typed manifest verification failed.
    Transport(String),
    /// Common writer admission or I/O failed; I/O may leave partial caller output.
    Write(tritium_format::TritPackageWriteError),
    /// Actual reconstructed content differs from the expected or stored binding.
    Binding,
    /// Invalid declared context or budget.
    Plan(CalibrationPlanError),
    /// Empty tokenizer/objective bytes or no captured window.
    Empty,
    /// Incompatible input/output/selection/token encoding, or nonfinite masked data.
    Batch,
    /// Overflow or allocation failure.
    Allocation,
    /// A declared retained byte/sample/window bound was exceeded.
    Limit,
    /// The existing numerical collector rejected replay.
    Calibration(CalibrationError),
}
impl core::fmt::Display for CalibrationCacheError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "calibration cache rejected: {self:?}")
    }
}
impl std::error::Error for CalibrationCacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Write(error) => Some(error),
            _ => None,
        }
    }
}
impl From<CalibrationError> for CalibrationCacheError {
    fn from(error: CalibrationError) -> Self {
        Self::Calibration(error)
    }
}

#[derive(Debug)]
enum Output {
    None,
    Dense(Vec<f32>),
    Indexed {
        indices: Vec<usize>,
        factors: Vec<f32>,
    },
}
#[derive(Debug)]
struct Window {
    tokens: Vec<u32>,
    input: Option<Vec<f32>>,
    output: Output,
    weights: Vec<f64>,
    mask: Vec<bool>,
}
impl Window {
    fn batch(
        &self,
        start: usize,
        count: usize,
        rows: usize,
        columns: usize,
    ) -> CalibrationBatch<'_> {
        let end = start + count;
        CalibrationBatch {
            samples: count,
            input: self
                .input
                .as_deref()
                .map_or(CalibrationInput::GatherIdentity, |values| {
                    CalibrationInput::Activations(&values[start * columns..end * columns])
                }),
            output: match &self.output {
                Output::None => CalibrationOutputBatch::None,
                Output::Dense(values) => {
                    CalibrationOutputBatch::Dense(&values[start * rows..end * rows])
                }
                Output::Indexed { indices, factors } => CalibrationOutputBatch::Indexed {
                    indices: &indices[start..end],
                    factors: &factors[start..end],
                },
            },
            weights: Some(&self.weights[start..end]),
            mask: Some(&self.mask[start..end]),
        }
    }
}

/// Captures actual supplied paired buffers under explicit logical limits.
/// No numerical reduction happens until a real cache identity has been finalized.
#[derive(Debug)]
pub struct CalibrationCacheBuilder {
    request: CalibrationCacheRequest,
    tokenizer: Vec<u8>,
    objective: Vec<u8>,
    windows: Vec<Window>,
    samples: u64,
    payload_bytes: u64,
}
impl CalibrationCacheBuilder {
    /// Admit and retain exact tokenizer/recipe bytes before allocating payload copies.
    /// These are byte-bound context, not independent model/tokenization/objective proofs.
    /// # Errors
    /// Invalid context, empty recipe/tokenizer, logical limit or allocation failure.
    pub fn new(
        request: CalibrationCacheRequest,
        tokenizer: &[u8],
        objective: &[u8],
    ) -> Result<Self, CalibrationCacheError> {
        request.validate().map_err(CalibrationCacheError::Plan)?;
        if tokenizer.is_empty() || objective.is_empty() {
            return Err(CalibrationCacheError::Empty);
        }
        let payload_bytes = (tokenizer.len() as u64)
            .checked_add(objective.len() as u64)
            .ok_or(CalibrationCacheError::Allocation)?;
        if payload_bytes > request.limits.max_payload_bytes {
            return Err(CalibrationCacheError::Limit);
        }
        Ok(Self {
            request,
            tokenizer: copy(tokenizer)?,
            objective: copy(objective)?,
            windows: Vec::new(),
            samples: 0,
            payload_bytes,
        })
    }

    /// Capture one complete declared context window atomically.
    /// One actual token ID is required per observed sample. Changing model context
    /// window boundaries changes cache/token identity; replay chunking does not.
    /// # Errors
    /// Shape/encoding/nonfinite/index/selection mismatch or pre-copy logical limits.
    pub fn capture(
        &mut self,
        tokens: &[u32],
        batch: CalibrationBatch<'_>,
    ) -> Result<(), CalibrationCacheError> {
        use CalibrationCacheError as E;
        if batch.samples == 0 || tokens.len() != batch.samples {
            return Err(E::Batch);
        }
        let rows = usize::try_from(self.request.tensor.shape[0]).map_err(|_| E::Allocation)?;
        let cols = usize::try_from(self.request.tensor.shape[1]).map_err(|_| E::Allocation)?;
        let input = match (self.request.consumer, batch.input) {
            (CalibrationConsumer::Matmul, CalibrationInput::Activations(values))
                if values.len() == batch.samples.checked_mul(cols).ok_or(E::Allocation)? =>
            {
                Some(values)
            }
            (CalibrationConsumer::GatherIdentity, CalibrationInput::GatherIdentity) => None,
            _ => return Err(E::Batch),
        };
        let output_entries = match (self.request.output, batch.output) {
            (CalibrationOutput::None, CalibrationOutputBatch::None) => 0,
            (CalibrationOutput::DenseFisher, CalibrationOutputBatch::Dense(values))
                if values.len() == batch.samples.checked_mul(rows).ok_or(E::Allocation)? =>
            {
                (values.len() as u64).checked_mul(4).ok_or(E::Allocation)?
            }
            (
                CalibrationOutput::IndexedFisher,
                CalibrationOutputBatch::Indexed { indices, factors },
            ) if indices.len() == batch.samples
                && factors.len() == batch.samples
                && indices.iter().all(|&row| row < rows) =>
            {
                (batch.samples as u64)
                    .checked_mul(12)
                    .ok_or(E::Allocation)?
            }
            _ => return Err(E::Batch),
        };
        if batch.weights.is_some_and(|v| {
            v.len() != batch.samples || v.iter().any(|v| !v.is_finite() || *v < 0.0)
        }) || batch.mask.is_some_and(|v| v.len() != batch.samples)
            || input.is_some_and(|v| v.iter().any(|v| !v.is_finite()))
        {
            return Err(E::Batch);
        }
        let gradients = match batch.output {
            CalibrationOutputBatch::None => &[][..],
            CalibrationOutputBatch::Dense(v) => v,
            CalibrationOutputBatch::Indexed { factors, .. } => factors,
        };
        if gradients.iter().any(|v| !v.is_finite()) {
            return Err(E::Batch);
        }
        let bytes = (tokens.len() as u64)
            .checked_mul(13)
            .and_then(|n| {
                (input.map_or(0, |v| v.len()) as u64)
                    .checked_mul(4)
                    .and_then(|i| n.checked_add(i))
            })
            .and_then(|n| n.checked_add(output_entries))
            .ok_or(E::Allocation)?;
        let samples = self
            .samples
            .checked_add(batch.samples as u64)
            .ok_or(E::Allocation)?;
        self.request
            .sample_start
            .checked_add(samples)
            .ok_or(E::Allocation)?;
        let payload_bytes = self.payload_bytes.checked_add(bytes).ok_or(E::Allocation)?;
        if samples > self.request.limits.max_samples
            || self.windows.len() as u64 >= self.request.limits.max_windows
            || payload_bytes > self.request.limits.max_payload_bytes
        {
            return Err(E::Limit);
        }
        let output = match batch.output {
            CalibrationOutputBatch::None => Output::None,
            CalibrationOutputBatch::Dense(v) => Output::Dense(copy(v)?),
            CalibrationOutputBatch::Indexed { indices, factors } => Output::Indexed {
                indices: copy(indices)?,
                factors: copy(factors)?,
            },
        };
        let weights = match batch.weights {
            Some(v) => copy(v)?,
            None => filled(batch.samples, 1.0)?,
        };
        let mask = match batch.mask {
            Some(v) => copy(v)?,
            None => filled(batch.samples, true)?,
        };
        let window = Window {
            tokens: copy(tokens)?,
            input: input.map(copy).transpose()?,
            output,
            weights,
            mask,
        };
        self.windows.try_reserve(1).map_err(|_| E::Allocation)?;
        self.windows.push(window);
        self.samples = samples;
        self.payload_bytes = payload_bytes;
        Ok(())
    }

    /// Seal owned immutable payloads and derive actual cache/token context identities.
    /// This consumes the builder, preventing subsequent hash/content mutation.
    /// # Errors
    /// No captured context or unsupported precision/consumer/output interpretation.
    pub fn finish(self) -> Result<CalibrationCache, CalibrationCacheError> {
        if self.windows.is_empty() {
            return Err(CalibrationCacheError::Empty);
        }
        let tokenizer = BlobId::from_bytes(*blake3::hash(&self.tokenizer).as_bytes());
        let objective = BlobId::from_bytes(*blake3::hash(&self.objective).as_bytes());
        let mut cache = blake3::Hasher::new_derive_key("tritium.calibration.cache.v1");
        let mut tokens = blake3::Hasher::new_derive_key("tritium.calibration.tokens.v1");
        metadata(&mut cache, &self.request)?;
        cache.update(tokenizer.as_bytes());
        cache.update(objective.as_bytes());
        tokens.update(tokenizer.as_bytes());
        number(&mut tokens, self.request.sample_start);
        number(&mut cache, self.windows.len() as u64);
        number(&mut tokens, self.windows.len() as u64);
        for window in &self.windows {
            number(&mut cache, window.tokens.len() as u64);
            number(&mut tokens, window.tokens.len() as u64);
            for token in &window.tokens {
                cache.update(&token.to_le_bytes());
                tokens.update(&token.to_le_bytes());
            }
            if let Some(input) = &window.input {
                for value in input {
                    cache.update(&value.to_bits().to_le_bytes());
                }
            }
            match &window.output {
                Output::None => {}
                Output::Dense(v) => {
                    for value in v {
                        cache.update(&value.to_bits().to_le_bytes());
                    }
                }
                Output::Indexed { indices, factors } => {
                    for (row, value) in indices.iter().zip(factors) {
                        number(&mut cache, *row as u64);
                        cache.update(&value.to_bits().to_le_bytes());
                    }
                }
            }
            for value in &window.weights {
                cache.update(&value.to_bits().to_le_bytes());
            }
            for value in &window.mask {
                cache.update(&[u8::from(*value)]);
            }
        }
        let id = CalibrationCacheId::from_bytes(*cache.finalize().as_bytes());
        let source = CalibrationSource {
            model: FitInputDigest(*self.request.model.as_bytes()),
            cache: FitInputDigest(*id.as_bytes()),
            tokens: FitInputDigest(*tokens.finalize().as_bytes()),
        };
        let binding = CalibrationCacheBinding {
            request: self.request,
            source,
            tokenizer,
            objective,
            samples: self.samples,
            windows: self.windows.len() as u64,
            payload_bytes: self.payload_bytes,
        };
        Ok(CalibrationCache {
            id,
            binding,
            tokenizer: self.tokenizer,
            objective: self.objective,
            windows: self.windows,
        })
    }
}

/// Owned immutable actual-payload cache with finalized byte/context identity.
/// Not an independent forward/backward receipt, persisted format or full RSS gate.
#[derive(Debug)]
pub struct CalibrationCache {
    id: CalibrationCacheId,
    binding: CalibrationCacheBinding,
    tokenizer: Vec<u8>,
    objective: Vec<u8>,
    windows: Vec<Window>,
}
impl CalibrationCache {
    /// Actual finalized cache identity, not an adapter-supplied placeholder.
    #[must_use]
    pub fn id(&self) -> CalibrationCacheId {
        self.id
    }
    /// Borrow immutable actual binding metadata.
    #[must_use]
    pub fn binding(&self) -> &CalibrationCacheBinding {
        &self.binding
    }
    /// Borrow the exact retained tokenizer bytes, never a reserialized copy.
    #[must_use]
    pub fn tokenizer(&self) -> &[u8] {
        &self.tokenizer
    }
    /// Borrow the exact declared objective/teacher/recovery recipe bytes.
    #[must_use]
    pub fn objective(&self) -> &[u8] {
        &self.objective
    }
    /// Replay actual immutable paired buffers through the common numerical collector.
    /// No new forward pass, Gram/Fisher loop or source rebinding is introduced.
    /// # Errors
    /// Unsupported partition/damping/budget or existing numerical reducer failure.
    pub fn collect(
        &self,
        partition: CalibrationPartition,
        damping: f64,
        max_numeric_bytes: u64,
    ) -> Result<CalibratedTensor, CalibrationCacheError> {
        self.collect_chunked(partition, damping, max_numeric_bytes, usize::MAX)
    }

    /// Bound replay scratch by chunking within immutable declared context windows.
    /// Chunk size does not change cache identity, model context or sample selection.
    /// # Errors
    /// Zero chunk size or ordinary numerical collection admission/failure.
    pub fn collect_chunked(
        &self,
        partition: CalibrationPartition,
        damping: f64,
        max_numeric_bytes: u64,
        max_batch_samples: usize,
    ) -> Result<CalibratedTensor, CalibrationCacheError> {
        if max_batch_samples == 0 {
            return Err(CalibrationCacheError::Batch);
        }
        let request = &self.binding.request;
        let rows = usize::try_from(request.tensor.shape[0])
            .map_err(|_| CalibrationCacheError::Allocation)?;
        let columns = usize::try_from(request.tensor.shape[1])
            .map_err(|_| CalibrationCacheError::Allocation)?;
        let mut collector = CalibrationCollector::new(CalibrationPlan {
            tensor: request.tensor.clone(),
            source: self.binding.source,
            partition,
            output: request.output,
            consumer: request.consumer,
            sample_start: request.sample_start,
            damping,
            max_numeric_bytes,
        })?;
        for window in &self.windows {
            for start in (0..window.tokens.len()).step_by(max_batch_samples) {
                let count = max_batch_samples.min(window.tokens.len() - start);
                collector.accumulate(window.batch(start, count, rows, columns))?;
            }
        }
        Ok(collector.finish()?)
    }
}

fn copy<T: Copy>(values: &[T]) -> Result<Vec<T>, CalibrationCacheError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(values.len())
        .map_err(|_| CalibrationCacheError::Allocation)?;
    result.extend_from_slice(values);
    Ok(result)
}
fn filled<T: Clone>(count: usize, value: T) -> Result<Vec<T>, CalibrationCacheError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| CalibrationCacheError::Allocation)?;
    result.resize(count, value);
    Ok(result)
}
fn number(hash: &mut blake3::Hasher, value: u64) {
    hash.update(&value.to_le_bytes());
}
fn text(hash: &mut blake3::Hasher, value: &str) {
    number(hash, value.len() as u64);
    hash.update(value.as_bytes());
}
fn metadata(
    hash: &mut blake3::Hasher,
    request: &CalibrationCacheRequest,
) -> Result<(), CalibrationCacheError> {
    use CalibrationCacheError::Batch;
    hash.update(request.model.as_bytes());
    text(hash, &request.tensor.name);
    number(hash, request.tensor.shape.len() as u64);
    for &dimension in &request.tensor.shape {
        number(hash, dimension);
    }
    let precision = match request.tensor.precision {
        ScalePrecision::F16 => 0,
        ScalePrecision::Bf16 => 1,
        ScalePrecision::F32 => 2,
        _ => return Err(Batch),
    };
    let consumer = match request.consumer {
        CalibrationConsumer::Matmul => 0,
        CalibrationConsumer::GatherIdentity => 1,
        _ => return Err(Batch),
    };
    let output = match request.output {
        CalibrationOutput::None => 0,
        CalibrationOutput::DenseFisher => 1,
        CalibrationOutput::IndexedFisher => 2,
        _ => return Err(Batch),
    };
    hash.update(&[precision, consumer, output]);
    number(hash, request.sample_start);
    Ok(())
}

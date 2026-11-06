//! Bounded device-owned KV with transactional visibility and online causal GQA.
use super::*;
use tritium_spec::{AttentionGeometry, AttentionLimits, AttentionSegment, ResidentAttention};

#[derive(Debug)]
pub(super) struct Cache {
    stream: Arc<CudaStream>,
    function: CudaFunction,
    geometry: AttentionGeometry,
    limits: AttentionLimits,
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    cursor: usize,
    pending: Option<usize>,
}

impl Cache {
    pub(super) fn new(
        backend: &CudaBackend,
        geometry: AttentionGeometry,
        limits: AttentionLimits,
    ) -> Result<Self, BackendError> {
        geometry.payload_bytes(limits)?;
        let elements = limits.max_context * geometry.widths()?.1;
        let alloc = || {
            backend
                .stream
                .alloc_zeros::<f32>(elements)
                .map_err(|e| alloc_or_backend("resident decoder KV", &e, elements * 4))
        };
        Ok(Self {
            stream: Arc::clone(&backend.stream),
            function: backend.func_semantic_attention.clone(),
            geometry,
            limits,
            key: alloc()?,
            value: alloc()?,
            cursor: 0,
            pending: None,
        })
    }
    pub(super) fn len(&self) -> usize {
        self.cursor
    }
    pub(super) fn publish(&mut self, end: usize) {
        self.cursor = end;
        self.pending = None;
    }
    fn extent(&self, tokens: usize, scale: f32) -> Result<usize, BackendError> {
        let end = self
            .cursor
            .checked_add(tokens)
            .ok_or_else(|| invalid("resident attention cursor overflow"))?;
        if tokens == 0
            || tokens > self.limits.max_segment_tokens
            || end > self.limits.max_context
            || !scale.is_finite()
        {
            return Err(invalid(
                "resident attention segment exceeds limits or has invalid scale",
            ));
        }
        Ok(end)
    }

    /// Device operands share the exact owning stream. This path never reads KV
    /// or activations back; callers establish completion before publishing end.
    pub(super) fn stage_device(
        &mut self,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        tokens: usize,
        scale: f32,
    ) -> Result<usize, BackendError> {
        self.pending = None;
        let end = self.extent(tokens, scale)?;
        let (q_width, kv_width) = self.geometry.widths()?;
        for (buffer, expected) in [
            (query, tokens * q_width),
            (key, tokens * kv_width),
            (value, tokens * kv_width),
            (&*output, tokens * q_width),
        ] {
            if !Arc::ptr_eq(buffer.stream(), &self.stream) {
                return Err(invalid(
                    "resident attention operand belongs to a different stream",
                ));
            }
            if buffer.len() < expected {
                return Err(BackendError::ShapeMismatch {
                    expected,
                    got: buffer.len(),
                });
            }
        }
        let start = self.cursor * kv_width;
        let stop = end * kv_width;
        // Only inaccessible tail rows are overwritten. Abort/reset need neither
        // a prefix copy nor a host snapshot; the next stage overwrites this tail.
        self.stream
            .memcpy_dtod(
                &key.slice(..tokens * kv_width),
                &mut self.key.slice_mut(start..stop),
            )
            .map_err(|e| driver_err("stage resident attention keys", &e))?;
        self.stream
            .memcpy_dtod(
                &value.slice(..tokens * kv_width),
                &mut self.value.slice_mut(start..stop),
            )
            .map_err(|e| driver_err("stage resident attention values", &e))?;
        let heads = self.geometry.query_heads as u32;
        let kv_heads = self.geometry.kv_heads as u32;
        let dim = self.geometry.head_dim as u32;
        let offset = self.cursor as u32;
        let threads = dim.next_power_of_two().max(32);
        let cfg = LaunchConfig {
            grid_dim: ((tokens * self.geometry.query_heads) as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: (threads + 3) * 4,
        };
        let mut launch = self.stream.launch_builder(&self.function);
        launch
            .arg(query)
            .arg(&self.key)
            .arg(&self.value)
            .arg(output)
            .arg(&heads)
            .arg(&kv_heads)
            .arg(&dim)
            .arg(&offset)
            .arg(&scale);
        // SAFETY: construction validates all ABI bounds and maximum payloads;
        // extent/length/owner checks above prove each indexed operand range.
        // One block owns one query/head; all block lanes participate in barriers.
        #[allow(unsafe_code)]
        unsafe { launch.launch(cfg) }.map_err(|e| driver_err("resident causal attention", &e))?;
        Ok(end)
    }
}

#[derive(Debug)]
struct CudaAttention {
    cache: Cache,
    payload_bytes: usize,
    query: CudaSlice<f32>,
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    output: CudaSlice<f32>,
    host_output: Vec<f32>,
}

fn invalid(message: &str) -> BackendError {
    BackendError::InvalidInput(message.into())
}

pub(super) fn create(
    backend: &CudaBackend,
    geometry: AttentionGeometry,
    limits: AttentionLimits,
) -> Result<Option<Box<dyn ResidentAttention>>, BackendError> {
    let bytes = geometry.payload_bytes(limits)?;
    let (q_width, kv_width) = geometry.widths()?;
    if geometry.head_dim > 1024
        || [
            geometry.query_heads,
            geometry.kv_heads,
            limits.max_context,
            limits.max_segment_tokens * geometry.query_heads,
        ]
        .into_iter()
        .any(|n| u32::try_from(n).is_err())
    {
        return Ok(None);
    }
    let q_len = limits.max_segment_tokens * q_width;
    let kv_len = limits.max_segment_tokens * kv_width;
    let cache_len = limits.max_context * kv_width;
    let alloc = |len| {
        backend
            .stream
            .alloc_zeros::<f32>(len)
            .map_err(|e| alloc_or_backend("resident attention payload", &e, len * 4))
    };
    let mut host_output = Vec::new();
    host_output
        .try_reserve_exact(q_len)
        .map_err(|_| BackendError::OutOfMemory {
            requested: q_len * 4,
        })?;
    host_output.resize(q_len, 0.);
    Ok(Some(Box::new(CudaAttention {
        cache: Cache {
            stream: Arc::clone(&backend.stream),
            function: backend.func_semantic_attention.clone(),
            geometry,
            limits,
            key: alloc(cache_len)?,
            value: alloc(cache_len)?,
            cursor: 0,
            pending: None,
        },
        payload_bytes: bytes,
        query: alloc(q_len)?,
        key: alloc(kv_len)?,
        value: alloc(kv_len)?,
        output: alloc(q_len)?,
        host_output,
    })))
}

impl ResidentAttention for CudaAttention {
    fn geometry(&self) -> AttentionGeometry {
        self.cache.geometry
    }
    fn limits(&self) -> AttentionLimits {
        self.cache.limits
    }
    fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
    fn len(&self) -> usize {
        self.cache.cursor
    }
    fn staged_len(&self) -> Option<usize> {
        self.cache.pending
    }

    fn stage(&mut self, segment: AttentionSegment<'_>) -> Result<(), BackendError> {
        // Even preflight failure invalidates an earlier pending transaction.
        self.cache.pending = None;
        self.cache.extent(segment.tokens, segment.scale)?;
        let (q_width, kv_width) = self.cache.geometry.widths()?;
        let q_len = segment.tokens * q_width;
        let kv_len = segment.tokens * kv_width;
        for (buffer, expected) in [
            (segment.query, q_len),
            (segment.key, kv_len),
            (segment.value, kv_len),
            (&*segment.output, q_len),
        ] {
            if buffer.len() != expected {
                return Err(BackendError::ShapeMismatch {
                    expected,
                    got: buffer.len(),
                });
            }
        }
        if segment
            .query
            .iter()
            .chain(segment.key)
            .chain(segment.value)
            .any(|v| !v.is_finite())
        {
            return Err(invalid("resident attention operands must be finite"));
        }
        let stream = &self.cache.stream;
        stream
            .memcpy_htod(segment.query, &mut self.query.slice_mut(..q_len))
            .map_err(|e| driver_err("upload resident attention query", &e))?;
        stream
            .memcpy_htod(segment.key, &mut self.key.slice_mut(..kv_len))
            .map_err(|e| driver_err("upload resident attention key", &e))?;
        stream
            .memcpy_htod(segment.value, &mut self.value.slice_mut(..kv_len))
            .map_err(|e| driver_err("upload resident attention value", &e))?;
        let end = self.cache.stage_device(
            &self.query,
            &self.key,
            &self.value,
            &mut self.output,
            segment.tokens,
            segment.scale,
        )?;
        self.cache
            .stream
            .memcpy_dtoh(&self.output.slice(..q_len), &mut self.host_output[..q_len])
            .map_err(|e| driver_err("read resident attention output", &e))?;
        if self.host_output[..q_len].iter().any(|v| !v.is_finite()) {
            return Err(invalid("resident attention produced non-finite output"));
        }
        segment.output.copy_from_slice(&self.host_output[..q_len]);
        self.cache.pending = Some(end);
        Ok(())
    }
    fn commit(&mut self) {
        if let Some(end) = self.cache.pending.take() {
            self.cache.cursor = end;
        }
    }
    fn abort(&mut self) {
        self.cache.pending = None;
    }
    fn reset(&mut self) {
        self.cache.cursor = 0;
        self.cache.pending = None;
    }
}

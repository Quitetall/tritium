//! M=N continuous-batching decode: KV adoption, the batched step,
//! mdecode split kernels and the batched capture/replay graphs
//! (P2a split: move-only from `cuda/mod.rs`; same `impl CudaDecodeModel`,
//! continued in a sibling module — field/parent-method access unchanged).

use super::*;

impl CudaDecodeModel {
    /// Debug/test access: dtoh one K row of a batch slot (f32 bytes).
    #[doc(hidden)]
    pub fn debug_batch_kv_row(
        &self,
        batch: &BatchKv,
        li: usize,
        row_slot: usize,
        row: usize,
        v: bool,
    ) -> Result<Vec<u8>, BackendError> {
        let kw = self.kv_width;
        let off = (row_slot * batch.max_ctx + row) * kw;
        let arena = if v { &batch.kv_v[li] } else { &batch.kv_k[li] };
        let view = arena.slice(off..off + kw);
        let mut out = vec![0f32; kw];
        self.stream
            .memcpy_dtoh(&view, &mut out)
            .map_err(|e| driver_err("debug batch kv row dtoh", &e))?;
        Ok(out.iter().flat_map(|v| v.to_le_bytes()).collect())
    }

    /// Debug/test access: dtoh one K row of a batch slot (f32 bytes).
    #[doc(hidden)]
    pub fn debug_batch_kv_k_row(
        &self,
        batch: &BatchKv,
        li: usize,
        row_slot: usize,
        row: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let kw = self.kv_width;
        let off = (row_slot * batch.max_ctx + row) * kw;
        let view = batch.kv_k[li].slice(off..off + kw);
        let mut out = vec![0f32; kw];
        self.stream
            .memcpy_dtoh(&view, &mut out)
            .map_err(|e| driver_err("debug batch kv row dtoh", &e))?;
        Ok(out.iter().flat_map(|v| v.to_le_bytes()).collect())
    }

    /// Continuous-batching admission: copy this model's single-sequence KV
    /// rows `[0, len)` (every layer, K and V) into batch slot `row`'s arena.
    /// The caller prefills the prompt through the SINGLE-sequence path (the
    /// optimized prefill), then adopts the cache into the slot and
    /// [`BatchKv::set_position`]s it — zero new kernels.
    ///
    /// Phase-1 constraint: batch arenas are f32, so this requires the f32 KV
    /// rung (`kv_elem == 4`); other rungs are rejected loudly.
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] on a bad row/len or a non-f32 KV rung.
    pub fn copy_kv_into_batch_row(
        &self,
        batch: &mut BatchKv,
        row: usize,
        len: usize,
    ) -> Result<(), BackendError> {
        if self.kv_elem != 4 {
            return Err(BackendError::InvalidInput(
                "continuous batching requires the f32 KV rung (batch arenas are f32); \
                 unset TRITIUM_KV"
                    .into(),
            ));
        }
        if row >= batch.n {
            return Err(BackendError::InvalidInput(format!(
                "copy_kv_into_batch_row: row {row} >= batch n {}",
                batch.n
            )));
        }
        if len > batch.max_ctx || len > self.max_ctx {
            return Err(BackendError::InvalidInput(format!(
                "copy_kv_into_batch_row: len {len} exceeds max_ctx {}",
                batch.max_ctx.min(self.max_ctx)
            )));
        }
        if len > self.cache_len {
            return Err(BackendError::InvalidInput(format!(
                "copy_kv_into_batch_row: len {len} > cache_len {} — prefill the \
                 prompt through the single-sequence path first",
                self.cache_len
            )));
        }
        let s = &self.stream;
        let bytes = len * self.kv_width * 4;
        for li in 0..self.layers.len() {
            for (src, dst) in [
                (&self.kv_k[li], &mut batch.kv_k[li]),
                (&self.kv_v[li], &mut batch.kv_v[li]),
            ] {
                let (src_ptr, sg) = src.device_ptr(s);
                let dst_off = row * batch.max_ctx * self.kv_width;
                // f32 elements → byte pointer offset ×4.
                let (dst_base, dg) = dst.device_ptr(s);
                let dst_ptr = dst_base + (dst_off * 4) as sys::CUdeviceptr;
                // SAFETY: raw byte copy between live device allocations on this
                // model's stream: src holds `len·kv_width` f32 rows from
                // position 0 (single-seq arena, byte-typed) and dst is the
                // slot's leading `len·kv_width` f32 span; sizes checked above.
                #[allow(unsafe_code)]
                unsafe { result::memcpy_dtod_async(dst_ptr, src_ptr, bytes, s.cu_stream()) }
                    .map_err(|e| driver_err("batch row adopt dtod", &e))?;
                drop(sg);
                drop(dg);
            }
        }
        // Belt-and-braces ordering: decode_batch_graph's replay already
        // syncs the default stream first, so this is redundant for the
        // current caller — kept so future capture-stream consumers can't
        // read a half-landed adoption.
        s.synchronize()
            .map_err(|e| driver_err("batch row adopt sync", &e))?;
        Ok(())
    }


    /// Allocate batched-decode state for `n` concurrent sequences: a per-sequence KV arena
    /// (`[n, max_ctx, kv_width]` per layer) + the M=N scratch, all starting empty.
    ///
    /// # Errors
    /// [`BackendError`] on a device allocation failure.
    pub fn new_batch(&self, n: usize) -> Result<BatchKv, BackendError> {
        let s = &self.stream;
        let alloc =
            |k: usize, what: &str| s.alloc_zeros::<f32>(k).map_err(|e| driver_err(what, &e));
        let mut kv_k = Vec::with_capacity(self.layers.len());
        let mut kv_v = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            kv_k.push(alloc(n * self.max_ctx * self.kv_width, "batch kv_k")?);
            kv_v.push(alloc(n * self.max_ctx * self.kv_width, "batch kv_v")?);
        }
        Ok(BatchKv {
            n,
            max_ctx: self.max_ctx,
            kv_k,
            kv_v,
            positions: vec![0; n],
            d_tokens: s
                .alloc_zeros::<i32>(n)
                .map_err(|e| driver_err("batch d_tokens", &e))?,
            d_positions: s
                .alloc_zeros::<i32>(n)
                .map_err(|e| driver_err("batch d_positions", &e))?,
            d_x: alloc(n * self.n_embd, "batch d_x")?,
            d_normed: alloc(n * self.n_embd, "batch d_normed")?,
            d_q: alloc(n * self.q_width, "batch d_q")?,
            d_k: alloc(n * self.kv_width, "batch d_k")?,
            d_v: alloc(n * self.kv_width, "batch d_v")?,
            d_attn: alloc(n * self.q_width, "batch d_attn")?,
            d_attn_sn: alloc(n * self.q_width, "batch d_attn_sn")?,
            d_proj: alloc(n * self.n_embd, "batch d_proj")?,
            d_gate: alloc(n * self.n_ff, "batch d_gate")?,
            d_up: alloc(n * self.n_ff, "batch d_up")?,
            d_gate_sn: alloc(n * self.n_ff, "batch d_gate_sn")?,
            d_qact: s
                .alloc_zeros::<i8>(n * self.n_ff)
                .map_err(|e| driver_err("batch d_qact", &e))?,
            d_act_scale: alloc(n, "batch d_act_scale")?,
            d_scores: alloc(n * self.n_head * self.max_ctx, "batch d_scores")?,
            d_attn_partials: alloc(
                n * self.n_head * self.max_ctx.div_ceil(ATTN_SPLIT_CHUNK) * (self.head_dim + 2),
                "batch d_attn_partials",
            )?,
            d_h: alloc(self.n_embd, "batch d_h")?,
            d_logits: alloc(self.vocab, "batch d_logits")?,
            d_logits_batch: alloc(n * self.vocab, "batch d_logits_batch")?,
            d_argmax: s
                .alloc_zeros::<i32>(n)
                .map_err(|e| driver_err("batch d_argmax", &e))?,
            graph: None,
            graph_argmax: None,
            raw_keepalive: None,
        })
    }

    /// One batched decode step: `tokens[r]` is sequence `r`'s next token (at its own
    /// position `batch.positions[r]`). Runs the M=N forward — each row attends its OWN KV
    /// slice — appends `n` k/v rows, advances every position by 1, and returns each
    /// sequence's next-token logits `[vocab]`. Bit-identical per row to a single-sequence
    /// `step_graph` (the batch kernels share the M=1 reduction order).
    ///
    /// # Errors
    /// [`BackendError`] on a length/token guard, capacity overflow, or device failure.
    pub fn decode_batch(
        &mut self,
        batch: &mut BatchKv,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        let n = batch.n;
        if tokens.len() != n {
            return Err(BackendError::InvalidInput(format!(
                "decode_batch expects {n} tokens, got {}",
                tokens.len()
            )));
        }
        for (&t, &p) in tokens.iter().zip(&batch.positions) {
            if t as usize >= self.vocab {
                return Err(BackendError::InvalidInput(format!(
                    "decode_batch token {t} out of range"
                )));
            }
            if p >= self.max_ctx {
                return Err(BackendError::InvalidInput(
                    "decode_batch context overflow".into(),
                ));
            }
        }
        let s = &self.stream;
        let (n_embd, q_width, kv_width, n_ff) =
            (self.n_embd, self.q_width, self.kv_width, self.n_ff);
        let (n_head, n_head_kv, head_dim, max_ctx) =
            (self.n_head, self.n_head_kv, self.head_dim, self.max_ctx);

        let tok_i: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let pos_i: Vec<i32> = batch.positions.iter().map(|&p| p as i32).collect();
        s.memcpy_htod(&tok_i, &mut batch.d_tokens)
            .map_err(|e| driver_err("batch tokens htod", &e))?;
        s.memcpy_htod(&pos_i, &mut batch.d_positions)
            .map_err(|e| driver_err("batch pos htod", &e))?;

        Self::bl_embed(
            s,
            &self.f_embed_batch,
            &self.d_token_embd,
            &batch.d_tokens,
            n_embd,
            n,
            &mut batch.d_x,
        )?;

        for li in 0..self.layers.len() {
            Self::bl_rmsnorm(
                s,
                &self.f_rmsnorm_batch,
                &batch.d_x,
                &self.layers[li].attn_norm,
                self.rms_eps,
                n_embd,
                n,
                &mut batch.d_normed,
            )?;
            Self::bl_quant(
                s,
                &self.f_quant_batch,
                &batch.d_normed,
                n_embd,
                n,
                &mut batch.d_qact,
                &mut batch.d_act_scale,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].q,
                &batch.d_act_scale,
                n,
                &mut batch.d_q,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].k,
                &batch.d_act_scale,
                n,
                &mut batch.d_k,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].v,
                &batch.d_act_scale,
                n,
                &mut batch.d_v,
            )?;
            Self::bl_rope(
                s,
                &self.f_rope_batch,
                &mut batch.d_q,
                &self.d_cos,
                &self.d_sin,
                &batch.d_positions,
                n_head,
                head_dim,
                n,
            )?;
            Self::bl_rope(
                s,
                &self.f_rope_batch,
                &mut batch.d_k,
                &self.d_cos,
                &self.d_sin,
                &batch.d_positions,
                n_head_kv,
                head_dim,
                n,
            )?;
            Self::md_kv_append(
                s,
                &self.f_kv_append_mdecode,
                &batch.d_k,
                &mut batch.kv_k[li],
                &batch.d_positions,
                max_ctx,
                kv_width,
                n,
            )?;
            Self::md_kv_append(
                s,
                &self.f_kv_append_mdecode,
                &batch.d_v,
                &mut batch.kv_v[li],
                &batch.d_positions,
                max_ctx,
                kv_width,
                n,
            )?;
            Self::md_attn(
                s,
                &self.f_attn_split_partial,
                &self.f_attn_combine,
                &batch.d_q,
                &batch.kv_k[li],
                &batch.kv_v[li],
                &mut batch.d_attn,
                &mut batch.d_attn_partials,
                &batch.d_positions,
                max_ctx,
                n_head,
                n_head_kv,
                head_dim,
                self.attn_scale,
                n,
            )?;
            let attn_in: &CudaSlice<f32> = if let Some(sn) = self.layers[li].attn_sub_norm.as_ref()
            {
                Self::bl_rmsnorm(
                    s,
                    &self.f_rmsnorm_batch,
                    &batch.d_attn,
                    sn,
                    self.rms_eps,
                    q_width,
                    n,
                    &mut batch.d_attn_sn,
                )?;
                &batch.d_attn_sn
            } else {
                &batch.d_attn
            };
            Self::bl_quant(
                s,
                &self.f_quant_batch,
                attn_in,
                q_width,
                n,
                &mut batch.d_qact,
                &mut batch.d_act_scale,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].o,
                &batch.d_act_scale,
                n,
                &mut batch.d_proj,
            )?;
            Self::bl_residual(
                s,
                &self.f_residual,
                &mut batch.d_x,
                &batch.d_proj,
                n * n_embd,
            )?;

            Self::bl_rmsnorm(
                s,
                &self.f_rmsnorm_batch,
                &batch.d_x,
                &self.layers[li].ffn_norm,
                self.rms_eps,
                n_embd,
                n,
                &mut batch.d_normed,
            )?;
            Self::bl_quant(
                s,
                &self.f_quant_batch,
                &batch.d_normed,
                n_embd,
                n,
                &mut batch.d_qact,
                &mut batch.d_act_scale,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].gate,
                &batch.d_act_scale,
                n,
                &mut batch.d_gate,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].up,
                &batch.d_act_scale,
                n,
                &mut batch.d_up,
            )?;
            Self::bl_relu2(s, &self.f_relu2, &mut batch.d_gate, &batch.d_up, n * n_ff)?;
            let down_in: &CudaSlice<f32> = if let Some(sn) = self.layers[li].ffn_sub_norm.as_ref() {
                Self::bl_rmsnorm(
                    s,
                    &self.f_rmsnorm_batch,
                    &batch.d_gate,
                    sn,
                    self.rms_eps,
                    n_ff,
                    n,
                    &mut batch.d_gate_sn,
                )?;
                &batch.d_gate_sn
            } else {
                &batch.d_gate
            };
            Self::bl_quant(
                s,
                &self.f_quant_batch,
                down_in,
                n_ff,
                n,
                &mut batch.d_qact,
                &mut batch.d_act_scale,
            )?;
            Self::bl_matmul(
                s,
                &self.f_tiled_scaled,
                &batch.d_qact,
                &self.layers[li].down,
                &batch.d_act_scale,
                n,
                &mut batch.d_proj,
            )?;
            Self::bl_residual(
                s,
                &self.f_residual,
                &mut batch.d_x,
                &batch.d_proj,
                n * n_embd,
            )?;
        }

        // Final norm (all n rows) then per-row LM head.
        Self::bl_rmsnorm(
            s,
            &self.f_rmsnorm_batch,
            &batch.d_x,
            &self.d_output_norm,
            self.rms_eps,
            n_embd,
            n,
            &mut batch.d_normed,
        )?;
        let mut out = Vec::with_capacity(n);
        for r in 0..n {
            {
                let row = batch.d_normed.slice(r * n_embd..(r + 1) * n_embd);
                s.memcpy_dtod(&row, &mut batch.d_h)
                    .map_err(|e| driver_err("batch row copy", &e))?;
            }
            Self::bl_lm_head_f16(
                s,
                &self.f_lm_head_f16,
                &batch.d_h,
                &self.d_token_embd_f16,
                n_embd,
                self.vocab,
                &mut batch.d_logits,
            )?;
            let mut logits = vec![0.0f32; self.vocab];
            s.memcpy_dtoh(&batch.d_logits, &mut logits)
                .map_err(|e| driver_err("batch logits dtoh", &e))?;
            out.push(logits);
        }
        for p in &mut batch.positions {
            *p += 1;
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn md_kv_append(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        src: &CudaSlice<f32>,
        kv_base: &mut CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        max_ctx: usize,
        kv_width: usize,
        n: usize,
    ) -> Result<(), BackendError> {
        let (mc_i, kw_i, n_i) = (max_ctx as i32, kv_width as i32, n as i32);
        let cfg = LaunchConfig {
            grid_dim: (((n * kv_width) as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(src)
            .arg(kv_base)
            .arg(positions)
            .arg(&mc_i)
            .arg(&kw_i)
            .arg(&n_i);
        // SAFETY: `kv_append_mdecode_f32(const float* src, float* kv_base, const int* pos, int max_ctx, int kv_width, int n)`.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch mdecode kv_append", &e))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn md_attn(
        s: &Arc<CudaStream>,
        f_partial: &CudaFunction,
        f_combine: &CudaFunction,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        partials: &mut CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        max_ctx: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
        n: usize,
    ) -> Result<(), BackendError> {
        let n_split = max_ctx.div_ceil(ATTN_SPLIT_CHUNK);
        let (mc_i, nh_i, nhkv_i, hd_i, n_i, ns_i, ck_i) = (
            max_ctx as i32,
            n_head as i32,
            n_head_kv as i32,
            head_dim as i32,
            n as i32,
            n_split as i32,
            ATTN_SPLIT_CHUNK as i32,
        );
        // Partial: one warp (32 threads) per (row, head, split).
        {
            let cfg = LaunchConfig {
                grid_dim: ((n * n_head * n_split) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = s.launch_builder(f_partial);
            l.arg(q)
                .arg(k)
                .arg(v)
                .arg(&mut *partials)
                .arg(positions)
                .arg(&mc_i)
                .arg(&nh_i)
                .arg(&nhkv_i)
                .arg(&hd_i)
                .arg(&scale)
                .arg(&n_i)
                .arg(&ns_i)
                .arg(&ck_i);
            // SAFETY: matches `gqa_attention_split_partial_f32(q, k, v, partials, positions,
            // max_ctx, n_head, n_head_kv, head_dim, scale, n, n_split, chunk)`; only `partials`
            // mutable; partials is `n·n_head·n_split·(head_dim+2)`; grid covers n·n_head·n_split warps.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch md split partial", &e))?;
            }
        }
        // Combine: one warp per (row, head).
        {
            let cfg = LaunchConfig {
                grid_dim: ((n * n_head) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = s.launch_builder(f_combine);
            l.arg(&*partials)
                .arg(out)
                .arg(&nh_i)
                .arg(&hd_i)
                .arg(&n_i)
                .arg(&ns_i);
            // SAFETY: matches `gqa_attention_combine_f32(partials, out, n_head, head_dim, n, n_split)`;
            // only `out` mutable; out is `n·n_head·head_dim`; grid covers n·n_head warps.
            #[allow(unsafe_code)]
            unsafe {
                l.launch(cfg)
                    .map_err(|e| driver_err("launch md split combine", &e))?;
            }
        }
        Ok(())
    }

    /// **Graph-captured batched (M=N) decode** — the Track-2 perf sibling of
    /// [`decode_batch`](Self::decode_batch). The device-resident M=N body is recorded once
    /// into a CUDA graph (per batch, since the capture bakes in *these* buffers' pointers)
    /// and replayed per step, eliminating the per-kernel launch overhead that left the
    /// eager M=N path slower than the M=1 [`step_graph`](Self::step_graph). Bit-identical
    /// to `decode_batch` per row — the graph replays the exact same kernels in the same
    /// order over the same buffers; only the launch mechanism differs — gated by
    /// `cuda_batch_decode_graph_matches_eager`. The LM head stays eager (per-row), like
    /// `decode_batch`.
    ///
    /// # Errors
    /// [`BackendError`] on a length/token guard, capacity overflow, or device failure.
    pub fn decode_batch_graph(
        &mut self,
        batch: &mut BatchKv,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        let n = batch.n;
        if tokens.len() != n {
            return Err(BackendError::InvalidInput(format!(
                "decode_batch_graph expects {n} tokens, got {}",
                tokens.len()
            )));
        }
        for (&t, &p) in tokens.iter().zip(&batch.positions) {
            if t as usize >= self.vocab {
                return Err(BackendError::InvalidInput(format!(
                    "decode_batch_graph token {t} out of range"
                )));
            }
            if p >= self.max_ctx {
                return Err(BackendError::InvalidInput(
                    "decode_batch_graph context overflow".into(),
                ));
            }
        }

        // Lazily load the raw batch kernels, then capture this batch's graph (per-N).
        if self.batch_raw.is_none() {
            let ctx = self.cap_stream.context().clone();
            self.batch_raw = Some(Arc::new(BatchRawKernels::load(&ctx)?));
        }
        if batch.graph.is_none() {
            let g = self.record_graph_batch(batch, false)?;
            batch.graph = Some(SendGraph(g));
            // Keep the modules the captured graph references alive for as long as this
            // batch lives (see `BatchKv::raw_keepalive`).
            batch.raw_keepalive = self.batch_raw.clone();
        }

        // Drain any pending default-stream work before the graph (on `cap_stream`) touches
        // the shared batch buffers, exactly as `step_graph` does for the M=1 path.
        self.stream
            .synchronize()
            .map_err(|e| driver_err("batch graph pre default sync", &e))?;

        // Upload this step's tokens + positions on the capture stream, ordered before the
        // replay (the captured embed/rope/kv/attn read them as stable pointers — the M=N
        // analogue of the M=1 `d_ctrl`).
        let tok_i: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let pos_i: Vec<i32> = batch.positions.iter().map(|&p| p as i32).collect();
        self.cap_stream
            .memcpy_htod(&tok_i, &mut batch.d_tokens)
            .map_err(|e| driver_err("batch graph tokens htod", &e))?;
        self.cap_stream
            .memcpy_htod(&pos_i, &mut batch.d_positions)
            .map_err(|e| driver_err("batch graph pos htod", &e))?;
        batch
            .graph
            .as_ref()
            .expect("graph captured above")
            .launch()
            .map_err(|e| driver_err("batch graph launch", &e))?;
        self.cap_stream
            .synchronize()
            .map_err(|e| driver_err("batch graph sync", &e))?;

        // Final norm landed in `d_normed`; run the per-row LM head eagerly (one warp head
        // per row over the f16 token table), mirroring `decode_batch`'s tail bit-for-bit.
        // The tail stays on `cap_stream` — the same stream the graph ran on — so the read
        // of `d_normed` is plain stream-ordered after the graph's write, with no
        // cross-stream handoff (the M=1 `step_graph` keeps its post-graph dtoh on
        // `cap_stream` for the same reason).
        let s = &self.cap_stream;
        let n_embd = self.n_embd;
        let mut out = Vec::with_capacity(n);
        for r in 0..n {
            {
                let row = batch.d_normed.slice(r * n_embd..(r + 1) * n_embd);
                s.memcpy_dtod(&row, &mut batch.d_h)
                    .map_err(|e| driver_err("batch graph row copy", &e))?;
            }
            Self::bl_lm_head_f16(
                s,
                &self.f_lm_head_f16,
                &batch.d_h,
                &self.d_token_embd_f16,
                n_embd,
                self.vocab,
                &mut batch.d_logits,
            )?;
            let mut logits = vec![0.0f32; self.vocab];
            s.memcpy_dtoh(&batch.d_logits, &mut logits)
                .map_err(|e| driver_err("batch graph logits dtoh", &e))?;
            out.push(logits);
        }
        for p in &mut batch.positions {
            *p += 1;
        }
        Ok(out)
    }

    /// **On-device-sampling batched decode** — the serving fast path. Same M=N forward as
    /// [`decode_batch_graph`](Self::decode_batch_graph), but the captured graph also runs a
    /// batched LM head + greedy argmax, so only `n` token ids (`n·4` bytes) come back instead
    /// of `n·vocab·4` bytes of logits (the readback that caps the eager-tail path). Each
    /// returned token equals the host `sample_greedy` of the logits the logits-path would
    /// produce (the batched LM head is bit-identical per row to the single-row kernel; the
    /// argmax tie rule matches `max_by`), gated by `cuda_batch_decode_graph_argmax_matches_greedy`.
    ///
    /// # Errors
    /// [`BackendError`] on a length/token guard, capacity overflow, or device failure.
    pub fn decode_batch_graph_argmax(
        &mut self,
        batch: &mut BatchKv,
        tokens: &[u32],
    ) -> Result<Vec<u32>, BackendError> {
        let n = batch.n;
        if tokens.len() != n {
            return Err(BackendError::InvalidInput(format!(
                "decode_batch_graph_argmax expects {n} tokens, got {}",
                tokens.len()
            )));
        }
        for (&t, &p) in tokens.iter().zip(&batch.positions) {
            if t as usize >= self.vocab {
                return Err(BackendError::InvalidInput(format!(
                    "decode_batch_graph_argmax token {t} out of range"
                )));
            }
            if p >= self.max_ctx {
                return Err(BackendError::InvalidInput(
                    "decode_batch_graph_argmax context overflow".into(),
                ));
            }
        }

        if self.batch_raw.is_none() {
            let ctx = self.cap_stream.context().clone();
            self.batch_raw = Some(Arc::new(BatchRawKernels::load(&ctx)?));
        }
        if batch.graph_argmax.is_none() {
            let g = self.record_graph_batch(batch, true)?;
            batch.graph_argmax = Some(g);
            batch.raw_keepalive = self.batch_raw.clone();
        }

        self.stream
            .synchronize()
            .map_err(|e| driver_err("batch argmax pre default sync", &e))?;

        let tok_i: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let pos_i: Vec<i32> = batch.positions.iter().map(|&p| p as i32).collect();
        self.cap_stream
            .memcpy_htod(&tok_i, &mut batch.d_tokens)
            .map_err(|e| driver_err("batch argmax tokens htod", &e))?;
        self.cap_stream
            .memcpy_htod(&pos_i, &mut batch.d_positions)
            .map_err(|e| driver_err("batch argmax pos htod", &e))?;
        batch
            .graph_argmax
            .as_ref()
            .expect("argmax graph captured above")
            .launch()
            .map_err(|e| driver_err("batch argmax graph launch", &e))?;
        self.cap_stream
            .synchronize()
            .map_err(|e| driver_err("batch argmax graph sync", &e))?;

        // The graph wrote the n greedy token ids into d_argmax; copy back just those.
        let mut ids = vec![0i32; n];
        self.cap_stream
            .memcpy_dtoh(&batch.d_argmax, &mut ids)
            .map_err(|e| driver_err("batch argmax dtoh", &e))?;
        for p in &mut batch.positions {
            *p += 1;
        }
        Ok(ids.into_iter().map(|t| t as u32).collect())
    }


    /// Extract every batch + weight buffer's stable device pointer (guards dropped here,
    /// outside capture), then capture the full M=N forward via raw launches on
    /// `cap_stream`. Mirrors [`record_graph`](Self::record_graph) for the batched path,
    /// reading the **per-batch** KV arenas and the **unfused** q/k/v/gate/up projections
    /// (the eager `decode_batch` is unfused — fusing is the follow-on).
    ///
    /// `with_head`: when true, the capture also runs the batched LM head + greedy argmax
    /// after the final RMSNorm (the on-device-sampling graph, ending at `d_argmax`); when
    /// false it ends at `d_normed` and the LM head is the eager per-row tail.
    fn record_graph_batch(
        &self,
        batch: &BatchKv,
        with_head: bool,
    ) -> Result<CudaGraph, BackendError> {
        let s = &self.cap_stream;
        let n = batch.n;
        let lin = |l: &ResidentLinear| LinPtrs {
            w: dptr(l.device.as_ref(), s),
            sc: dptr(&l.scales, s),
            n: l.n,
            k: l.k,
            rb: l.row_bytes,
        };
        let layers: Vec<BatchLayerPtrs> = self
            .layers
            .iter()
            .enumerate()
            .map(|(li, l)| BatchLayerPtrs {
                attn_norm: dptr(&l.attn_norm, s),
                attn_sub_norm: l.attn_sub_norm.as_ref().map(|b| dptr(b, s)),
                ffn_norm: dptr(&l.ffn_norm, s),
                ffn_sub_norm: l.ffn_sub_norm.as_ref().map(|b| dptr(b, s)),
                q: lin(&l.q),
                k: lin(&l.k),
                v: lin(&l.v),
                o: lin(&l.o),
                gate: lin(&l.gate),
                up: lin(&l.up),
                down: lin(&l.down),
                kv_k: dptr(&batch.kv_k[li], s),
                kv_v: dptr(&batch.kv_v[li], s),
            })
            .collect();
        let p = BatchPtrs {
            d_tokens: dptr(&batch.d_tokens, s),
            d_positions: dptr(&batch.d_positions, s),
            d_x: dptr(&batch.d_x, s),
            d_normed: dptr(&batch.d_normed, s),
            d_q: dptr(&batch.d_q, s),
            d_k: dptr(&batch.d_k, s),
            d_v: dptr(&batch.d_v, s),
            d_attn: dptr(&batch.d_attn, s),
            d_attn_sn: dptr(&batch.d_attn_sn, s),
            d_proj: dptr(&batch.d_proj, s),
            d_gate: dptr(&batch.d_gate, s),
            d_up: dptr(&batch.d_up, s),
            d_gate_sn: dptr(&batch.d_gate_sn, s),
            d_qact: dptr(&batch.d_qact, s),
            d_act_scale: dptr(&batch.d_act_scale, s),
            d_scores: dptr(&batch.d_scores, s),
            d_attn_partials: dptr(&batch.d_attn_partials, s),
            d_cos: dptr(&self.d_cos, s),
            d_sin: dptr(&self.d_sin, s),
            d_token_embd: dptr(&self.d_token_embd, s),
            d_output_norm: dptr(&self.d_output_norm, s),
            d_token_embd_f16: dptr(&self.d_token_embd_f16, s),
            d_logits_batch: dptr(&batch.d_logits_batch, s),
            d_argmax: dptr(&batch.d_argmax, s),
        };
        // Drain the events the device_ptr extraction recorded, so the capture (raw
        // launches only) carries no pre-capture dependency.
        s.synchronize()
            .map_err(|e| driver_err("batch pre-capture cap sync", &e))?;
        self.stream
            .synchronize()
            .map_err(|e| driver_err("batch pre-capture default sync", &e))?;

        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| driver_err("batch begin_capture", &e))?;

        capture_body(s, || {
            // The exact op order of `decode_batch`, all raw-launched on `cap_stream`.
            self.gb_embed(p.d_token_embd, p.d_tokens, p.d_x, n)?;
            for lp in &layers {
                self.gb_layer(&p, lp, n)?;
            }
            self.gb_rmsnorm(p.d_x, p.d_output_norm, self.n_embd, p.d_normed, n)?;
            if with_head {
                // Batched LM head over all n rows → d_logits_batch, then per-row greedy argmax →
                // d_argmax. Both raw-launched into the capture; only d_argmax is read back.
                self.gb_lm_head_batch(p.d_normed, p.d_token_embd_f16, p.d_logits_batch, n)?;
                self.gb_argmax(p.d_logits_batch, p.d_argmax, n)?;
            }
            Ok(())
        })?;

        let graph = s
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| driver_err("batch end_capture", &e))?
            .ok_or_else(|| BackendError::Backend("batch graph capture produced no graph".into()))?;
        Ok(graph)
    }

    /// One transformer block of the M=N forward, raw-launched into the capture. Mirrors
    /// the per-layer body of [`decode_batch`](Self::decode_batch) op-for-op.
    pub(super) fn gb_layer(&self, p: &BatchPtrs, l: &BatchLayerPtrs, n: usize) -> Result<(), BackendError> {
        let (n_embd, q_width, kv_width, n_ff) =
            (self.n_embd, self.q_width, self.kv_width, self.n_ff);
        let (n_head, n_head_kv, head_dim) = (self.n_head, self.n_head_kv, self.head_dim);

        // pre-norm attention. q/k/v share ONE quant of d_normed, then three unfused GEMMs.
        self.gb_rmsnorm(p.d_x, l.attn_norm, n_embd, p.d_normed, n)?;
        self.gb_quant(p.d_normed, n_embd, p.d_qact, p.d_act_scale, n)?;
        self.gb_matmul(&l.q, p.d_qact, p.d_act_scale, p.d_q, n)?;
        self.gb_matmul(&l.k, p.d_qact, p.d_act_scale, p.d_k, n)?;
        self.gb_matmul(&l.v, p.d_qact, p.d_act_scale, p.d_v, n)?;
        self.gb_rope(p.d_q, p.d_cos, p.d_sin, p.d_positions, n_head, head_dim, n)?;
        self.gb_rope(
            p.d_k,
            p.d_cos,
            p.d_sin,
            p.d_positions,
            n_head_kv,
            head_dim,
            n,
        )?;
        self.gb_kv_append(p.d_k, l.kv_k, p.d_positions, kv_width, n)?;
        self.gb_kv_append(p.d_v, l.kv_v, p.d_positions, kv_width, n)?;
        self.gb_attn(
            p.d_q,
            l.kv_k,
            l.kv_v,
            p.d_attn,
            p.d_attn_partials,
            p.d_positions,
            n,
        )?;
        let attn_in = if let Some(sn) = l.attn_sub_norm {
            self.gb_rmsnorm(p.d_attn, sn, q_width, p.d_attn_sn, n)?;
            p.d_attn_sn
        } else {
            p.d_attn
        };
        self.gb_quant(attn_in, q_width, p.d_qact, p.d_act_scale, n)?;
        self.gb_matmul(&l.o, p.d_qact, p.d_act_scale, p.d_proj, n)?;
        self.gb_residual(p.d_x, p.d_proj, n * n_embd)?;

        // pre-norm ReLU² MLP. gate/up unfused; relu2 writes gate = relu(gate)² ⊙ up.
        self.gb_rmsnorm(p.d_x, l.ffn_norm, n_embd, p.d_normed, n)?;
        self.gb_quant(p.d_normed, n_embd, p.d_qact, p.d_act_scale, n)?;
        self.gb_matmul(&l.gate, p.d_qact, p.d_act_scale, p.d_gate, n)?;
        self.gb_matmul(&l.up, p.d_qact, p.d_act_scale, p.d_up, n)?;
        self.gb_relu2(p.d_gate, p.d_up, n * n_ff)?;
        let down_in = if let Some(sn) = l.ffn_sub_norm {
            self.gb_rmsnorm(p.d_gate, sn, n_ff, p.d_gate_sn, n)?;
            p.d_gate_sn
        } else {
            p.d_gate
        };
        self.gb_quant(down_in, n_ff, p.d_qact, p.d_act_scale, n)?;
        self.gb_matmul(&l.down, p.d_qact, p.d_act_scale, p.d_proj, n)?;
        self.gb_residual(p.d_x, p.d_proj, n * n_embd)?;
        Ok(())
    }

    // Raw-launch helpers for the batched capture (`gb_*`): each mirrors the matching safe
    // `bl_*`/`md_*` helper 1:1 — same grid/block/smem and the same kernel-param order —
    // but builds the params from pre-extracted device pointers and raw-launches on
    // `cap_stream`. `n` is the batch (row) count.

    pub(super) fn gb_embed(
        &self,
        table: sys::CUdeviceptr,
        tokens: sys::CUdeviceptr,
        out: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let (ne_i, n_i) = (self.n_embd as i32, n as i32);
        let grid = (((n * self.n_embd) as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&table), pp(&tokens), pp(&ne_i), pp(&n_i), pp(&out)];
        raw_launch(
            self.batch_raw().embed,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    pub(super) fn gb_rmsnorm(
        &self,
        x: sys::CUdeviceptr,
        w: sys::CUdeviceptr,
        dim: usize,
        out: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let eps = self.rms_eps;
        let (dim_i, n_i) = (dim as i32, n as i32);
        let smem = (dim * 4) as u32;
        let mut params = [pp(&x), pp(&w), pp(&eps), pp(&dim_i), pp(&n_i), pp(&out)];
        raw_launch(
            self.batch_raw().rmsnorm,
            (n as u32, 1, 1),
            (256, 1, 1),
            smem,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    pub(super) fn gb_quant(
        &self,
        d_in: sys::CUdeviceptr,
        k: usize,
        d_qact: sys::CUdeviceptr,
        d_act_scale: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let (k_i, n_i) = (k as i32, n as i32);
        let mut params = [pp(&d_in), pp(&k_i), pp(&n_i), pp(&d_qact), pp(&d_act_scale)];
        raw_launch(
            self.batch_raw().quant,
            (n as u32, 1, 1),
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    pub(super) fn gb_matmul(
        &self,
        lin: &LinPtrs,
        d_qact: sys::CUdeviceptr,
        d_act_scale: sys::CUdeviceptr,
        d_out: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let cs = self.cap_stream.cu_stream();
        let (m_i, n_out_i, k_i, rb_i) = (n as i32, lin.n as i32, lin.k as i32, lin.rb as i32);
        // v1.x: one fused i8 dp4a launch — the `_scaled` epilogue folds the per-row
        // act_scale, replacing the former tiled + scale_mul_batch pair. The multiply
        // order is unchanged ((acc·weight_scale)·act_scale) and the int32 contraction
        // is exact, so the batch-graph argmax lockstep gate is unaffected.
        let grid = ((lin.n as u32).div_ceil(WARPS_PER_BLOCK), n as u32, 1);
        let mut params = [
            pp(&d_qact),
            pp(&lin.w),
            pp(&lin.sc),
            pp(&d_act_scale),
            pp(&d_out),
            pp(&m_i),
            pp(&n_out_i),
            pp(&k_i),
            pp(&rb_i),
        ];
        raw_launch(
            self.batch_raw().tiled_scaled,
            grid,
            (WARPS_PER_BLOCK * 32, 1, 1),
            0,
            cs,
            &mut params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gb_rope(
        &self,
        x: sys::CUdeviceptr,
        cos_t: sys::CUdeviceptr,
        sin_t: sys::CUdeviceptr,
        positions: sys::CUdeviceptr,
        n_head: usize,
        head_dim: usize,
        n: usize,
    ) -> Result<(), BackendError> {
        let (nh_i, hd_i, n_i) = (n_head as i32, head_dim as i32, n as i32);
        let total = (n * n_head * (head_dim / 2)) as u32;
        let grid = (total.div_ceil(256), 1, 1);
        let mut params = [
            pp(&x),
            pp(&cos_t),
            pp(&sin_t),
            pp(&positions),
            pp(&nh_i),
            pp(&hd_i),
            pp(&n_i),
        ];
        raw_launch(
            self.batch_raw().rope,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    pub(super) fn gb_kv_append(
        &self,
        src: sys::CUdeviceptr,
        kv_base: sys::CUdeviceptr,
        positions: sys::CUdeviceptr,
        kv_width: usize,
        n: usize,
    ) -> Result<(), BackendError> {
        let (mc_i, kw_i, n_i) = (self.max_ctx as i32, kv_width as i32, n as i32);
        let grid = (((n * kv_width) as u32).div_ceil(256), 1, 1);
        let mut params = [
            pp(&src),
            pp(&kv_base),
            pp(&positions),
            pp(&mc_i),
            pp(&kw_i),
            pp(&n_i),
        ];
        raw_launch(
            self.batch_raw().kv_append,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gb_attn(
        &self,
        q: sys::CUdeviceptr,
        k: sys::CUdeviceptr,
        v: sys::CUdeviceptr,
        out: sys::CUdeviceptr,
        partials: sys::CUdeviceptr,
        positions: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let n_split = self.max_ctx.div_ceil(ATTN_SPLIT_CHUNK);
        let (mc_i, nh_i, nhkv_i, hd_i, n_i, ns_i, ck_i) = (
            self.max_ctx as i32,
            self.n_head as i32,
            self.n_head_kv as i32,
            self.head_dim as i32,
            n as i32,
            n_split as i32,
            ATTN_SPLIT_CHUNK as i32,
        );
        let scale = self.attn_scale;
        let cs = self.cap_stream.cu_stream();
        {
            let grid = ((n * self.n_head * n_split) as u32, 1, 1);
            let mut params = [
                pp(&q),
                pp(&k),
                pp(&v),
                pp(&partials),
                pp(&positions),
                pp(&mc_i),
                pp(&nh_i),
                pp(&nhkv_i),
                pp(&hd_i),
                pp(&scale),
                pp(&n_i),
                pp(&ns_i),
                pp(&ck_i),
            ];
            raw_launch(
                self.batch_raw().attn_split_partial,
                grid,
                (32, 1, 1),
                0,
                cs,
                &mut params,
            )?;
        }
        {
            let grid = ((n * self.n_head) as u32, 1, 1);
            let mut params = [
                pp(&partials),
                pp(&out),
                pp(&nh_i),
                pp(&hd_i),
                pp(&n_i),
                pp(&ns_i),
            ];
            raw_launch(
                self.batch_raw().attn_combine,
                grid,
                (32, 1, 1),
                0,
                cs,
                &mut params,
            )?;
        }
        Ok(())
    }

    pub(super) fn gb_residual(
        &self,
        x: sys::CUdeviceptr,
        y: sys::CUdeviceptr,
        total: usize,
    ) -> Result<(), BackendError> {
        let total_i = total as i32;
        let grid = ((total as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&x), pp(&y), pp(&total_i)];
        raw_launch(
            self.batch_raw().residual,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    pub(super) fn gb_relu2(
        &self,
        gate: sys::CUdeviceptr,
        up: sys::CUdeviceptr,
        total: usize,
    ) -> Result<(), BackendError> {
        let total_i = total as i32;
        let grid = ((total as u32).div_ceil(256), 1, 1);
        let mut params = [pp(&gate), pp(&up), pp(&total_i)];
        raw_launch(
            self.batch_raw().relu2,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    /// Batched LM head over all `n` rows: `d_normed[n, n_embd] · token_embd_f16 → d_logits[n, vocab]`.
    /// One warp per vocab row, computing `LMHEAD_ROW_TILE` output rows per launch so the embd
    /// table is read once per row-tile (not once per row); `grid.y = ceil(n / TILE)`.
    /// Bit-identical per row to the single-row head.
    pub(super) fn gb_lm_head_batch(
        &self,
        h: sys::CUdeviceptr,
        embd: sys::CUdeviceptr,
        d_logits: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let (ne_i, v_i, n_i) = (self.n_embd as i32, self.vocab as i32, n as i32);
        let grid = (
            (self.vocab as u32).div_ceil(8),
            (n as u32).div_ceil(LMHEAD_ROW_TILE),
            1,
        );
        let mut params = [
            pp(&h),
            pp(&embd),
            pp(&ne_i),
            pp(&v_i),
            pp(&n_i),
            pp(&d_logits),
        ];
        raw_launch(
            self.batch_raw().lm_head_tiled,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    /// Per-row greedy argmax `d_logits[n, vocab] → d_out[n]` (i32). One block per row.
    pub(super) fn gb_argmax(
        &self,
        d_logits: sys::CUdeviceptr,
        d_out: sys::CUdeviceptr,
        n: usize,
    ) -> Result<(), BackendError> {
        let (v_i, n_i) = (self.vocab as i32, n as i32);
        let grid = (n as u32, 1, 1);
        let mut params = [pp(&d_logits), pp(&v_i), pp(&n_i), pp(&d_out)];
        raw_launch(
            self.batch_raw().argmax,
            grid,
            (256, 1, 1),
            0,
            self.cap_stream.cu_stream(),
            &mut params,
        )
    }

    pub(super) fn batch_raw(&self) -> &BatchRawKernels {
        self.batch_raw
            .as_ref()
            .expect("batch raw kernels loaded before record")
    }
}

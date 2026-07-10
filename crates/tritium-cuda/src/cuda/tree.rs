//! Tree-verify decode (BASTION spec-decode verifier): tree forward,
//! greedy/logits verify, commit/promote and the fixed-shape tree graphs
//! (P2a split: move-only from `cuda/mod.rs`; same `impl CudaDecodeModel`,
//! continued in a sibling module — field/parent-method access unchanged).

use super::*;

impl CudaDecodeModel {
    /// **BASTION greedy tree-verify** (ADR 0014) — verify a draft token tree in ONE
    /// batched forward and commit the longest greedy-accepted path.
    ///
    /// `tokens[i]` / `parents[i]` describe the tree: node 0 is the single root
    /// (`parents[0] == -1`) and MUST be the token the caller was about to `step`
    /// (greedy target semantics: the root is already committed by the caller's
    /// previous argmax); every other node has `parents[i] < i`. Node `i` is a
    /// draft candidate for the position after its parent. Duplicate sibling
    /// tokens are allowed; the first matching child wins.
    ///
    /// The whole tree runs as an M=N batched forward (rows = nodes, RoPE at
    /// `cache_len + depth(i)`, K/V written provisionally at arena rows
    /// `cache_len + i`) with the tree-masked attention
    /// (`gqa_attention_tree_f32`): each node attends the committed prefix plus
    /// its own ancestor chain. Greedy acceptance walks from the root taking the
    /// child whose token equals the target argmax at the current node; the
    /// accepted path's K/V rows are then promoted (compacted) into
    /// `cache_len..cache_len+L` and the watermark advances by `L` — rejected
    /// rows sit past the watermark and are dead (O(1) rollback).
    ///
    /// Returns the `L` newly determined tokens: `out[k]` = target argmax at the
    /// k-th accepted node. `out[k] == tokens[path[k+1]]` for the accepted
    /// drafts and `out[L-1]` is the bonus token (feed it back as the next
    /// root). `L >= 1` always: a full draft reject degenerates to exactly one
    /// plain greedy step — losslessness is by construction, since every
    /// returned token IS the target's greedy argmax at its position.
    ///
    /// Intended tree sizes are small (BASTION budgets N at the roofline knee,
    /// typically ≲ 64 nodes): the ancestor table is O(N²) and the scores
    /// scratch O(N · n_head · (cache_len + N)) — both per-call allocations.
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] on a malformed tree (root not at 0,
    /// non-topological parents, out-of-range token) or capacity overflow;
    /// device errors otherwise.
    ///
    /// This is the shared FORWARD half: it validates the tree, appends
    /// provisional K/V at arena rows [cache_len, cache_len + m) and leaves
    /// every node's logits in `tree_scratch.d_logits_all` WITHOUT committing.
    /// [`Self::tree_verify_greedy`] adds the device argmax + greedy walk;
    /// [`Self::tree_verify_logits`] hands the logits to the host for the
    /// speculative-sampling accept rule, with [`Self::tree_commit`] promoting
    /// the host-chosen path.
    fn tree_forward(&mut self, tokens: &[u32], parents: &[i32]) -> Result<usize, BackendError> {
        // A new forward invalidates any uncommitted previous tree.
        self.pending_tree = None;
        let m = tokens.len();
        if m == 0 || parents.len() != m {
            return Err(BackendError::InvalidInput(
                "tree_verify: empty or mismatched parents".into(),
            ));
        }
        if parents[0] != -1 {
            return Err(BackendError::InvalidInput(
                "tree_verify: node 0 must be the root (parent -1)".into(),
            ));
        }
        for (i, &p) in parents.iter().enumerate().skip(1) {
            if p < 0 || p as usize >= i {
                return Err(BackendError::InvalidInput(format!(
                    "tree_verify: parents[{i}]={p} is not topological (0 <= parent < i)"
                )));
            }
        }
        for &t in tokens {
            if t as usize >= self.vocab {
                return Err(BackendError::InvalidInput(format!(
                    "tree_verify token {t} out of range"
                )));
            }
        }
        if self.cache_len + m > self.max_ctx {
            return Err(BackendError::InvalidInput(format!(
                "tree_verify overflow: cache_len={} + {m} nodes > max_ctx={}",
                self.cache_len, self.max_ctx
            )));
        }

        // Graph route: pad to the smallest captured bucket and replay ONE
        // graph instead of ~420 eager launches. Requirements: the split
        // attention geometry (the ctrl twins are float4 kernels), a bucketable
        // size, room for the PADDED tree in the arena, and a context whose
        // score staging fits the default shared-memory limit (the raw-handle
        // capture path doesn't carry the opt-in attribute the safe handles
        // get at load).
        let bucket = if self.head_dim.is_multiple_of(4)
            && m <= TREE_BUCKET_MAX
            && self.max_ctx * 4 <= 48 * 1024
            && self.kv_elem == 4 // non-f32 KV: eager tree (no ctrl twins; graph measured ≈ no win)
            && std::env::var_os("TRITIUM_TREE_EAGER").is_none()
        {
            TREE_BUCKETS
                .iter()
                .copied()
                .find(|&b| b >= m && self.cache_len + b <= self.max_ctx)
        } else {
            None
        };
        // mb = padded node count (bucket) or the real m on the eager path;
        // every host array below is built at stride/length mb. Pad rows are
        // root-token duplicates at depth 1 — valid math whose results only
        // the pads themselves ever see.
        let mb = bucket.unwrap_or(m);

        // Depths, RoPE positions, and per-node ancestor slot lists (root-first,
        // including self). Ancestors are arena slots (cache_len + node index).
        let mut depth = vec![0usize; m];
        let mut anc: Vec<i32> = vec![0; mb * mb]; // [mb, max_anc=mb], row-major
        let mut n_anc = vec![0i32; mb];
        for i in 0..m {
            if parents[i] >= 0 {
                let p = parents[i] as usize;
                depth[i] = depth[p] + 1;
                let (dst_off, src_off) = (i * mb, p * mb);
                let np = n_anc[p] as usize;
                // anc[i] = anc[parent] ++ [slot(i)] (rows are disjoint: p < i).
                anc.copy_within(src_off..src_off + np, dst_off);
                anc[dst_off + np] = (self.cache_len + i) as i32;
                n_anc[i] = n_anc[p] + 1;
            } else {
                anc[i * mb] = (self.cache_len + i) as i32;
                n_anc[i] = 1;
            }
        }
        for i in m..mb {
            // Pad: a root child (its ancestor list is the root's slot + its own).
            anc[i * mb] = self.cache_len as i32;
            anc[i * mb + 1] = (self.cache_len + i) as i32;
            n_anc[i] = 2;
        }
        let mut positions: Vec<i32> = depth.iter().map(|&d| (self.cache_len + d) as i32).collect();
        positions.resize(mb, (self.cache_len + 1) as i32);

        let s = &self.stream;
        let (n_embd, q_width, kv_width, n_ff) =
            (self.n_embd, self.q_width, self.kv_width, self.n_ff);
        let (n_head, n_head_kv, head_dim) = (self.n_head, self.n_head_kv, self.head_dim);
        let prefix_len = self.cache_len;
        let ctx_max = self.cache_len + m;

        // Reusable M=N scratch: allocated on first use (or re-grown for a larger
        // tree), then cached on the model — per-call alloc/free measurably ate
        // the speculative gains. Scores are sized by `max_ctx` (the largest any
        // verify can need) so growth is driven by `m` alone.
        // Capacity covers the graph buckets from the start so the captured
        // graphs' baked pointers stay valid; an oversized eager tree (> the
        // bucket max) re-grows the scratch and must drop every graph.
        let m_cap_want = mb.max(TREE_BUCKET_MAX);
        if self
            .tree_scratch
            .as_ref()
            .is_none_or(|t| t.m_cap < m_cap_want)
        {
            // Unconditional: an error-path `?` between take and put-back drops
            // the scratch while captured graphs keep its baked pointers — on
            // the next call the scratch is None but stale graphs would replay
            // into freed memory if this drop were guarded on `is_some()`.
            self.tree_graphs = None;
            let m = m_cap_want;
            let alloc =
                |n: usize, what: &str| s.alloc_zeros::<f32>(n).map_err(|e| driver_err(what, &e));
            self.tree_scratch = Some(TreeScratch {
                m_cap: m,
                d_x: alloc(m * n_embd, "tree d_x")?,
                d_normed: alloc(m * n_embd, "tree d_normed")?,
                d_q: alloc(m * q_width, "tree d_q")?,
                d_k: alloc(m * kv_width, "tree d_k")?,
                d_v: alloc(m * kv_width, "tree d_v")?,
                d_attn: alloc(m * q_width, "tree d_attn")?,
                d_attn_sn: alloc(m * q_width, "tree d_attn_sn")?,
                d_proj: alloc(m * n_embd, "tree d_proj")?,
                d_gate: alloc(m * n_ff, "tree d_gate")?,
                d_up: alloc(m * n_ff, "tree d_up")?,
                d_gate_sn: alloc(m * n_ff, "tree d_gate_sn")?,
                d_qact: s
                    .alloc_zeros::<i8>(m * n_ff)
                    .map_err(|e| driver_err("tree d_qact", &e))?,
                d_act_scale: alloc(m, "tree d_act_scale")?,
                d_scores: alloc(m * n_head * self.max_ctx, "tree d_scores")?,
                d_logits_all: alloc(m * self.vocab, "tree d_logits")?,
                d_norm_all: alloc(m * n_embd, "tree d_norm_all")?,
                d_ids: s
                    .alloc_zeros::<i32>(m)
                    .map_err(|e| driver_err("tree d_ids", &e))?,
                d_tok: s
                    .alloc_zeros::<i32>(m)
                    .map_err(|e| driver_err("tree d_tok", &e))?,
                d_pos: s
                    .alloc_zeros::<i32>(m)
                    .map_err(|e| driver_err("tree d_pos", &e))?,
                d_anc: s
                    .alloc_zeros::<i32>(m * m)
                    .map_err(|e| driver_err("tree d_anc", &e))?,
                d_nanc: s
                    .alloc_zeros::<i32>(m)
                    .map_err(|e| driver_err("tree d_nanc", &e))?,
                d_amax_val: alloc(m * ARGMAX_CHUNKS, "tree d_amax_val")?,
                d_amax_idx: s
                    .alloc_zeros::<i32>(m * ARGMAX_CHUNKS)
                    .map_err(|e| driver_err("tree d_amax_idx", &e))?,
            });
        }
        // Move the scratch out for disjoint borrows vs `self.kv_*` below; put
        // it back once the device work completes.
        let mut ts = self.tree_scratch.take().expect("tree scratch just ensured");

        // Uploads go into the cached buffers too (oversized is fine — kernels
        // read exactly the first m / m·m entries; `max_anc == m` is a stride
        // into linear memory, not a buffer shape).
        let mut tok_i: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        tok_i.resize(mb, tok_i[0]); // pads embed the root token (valid, unread)
        s.memcpy_htod(&tok_i, &mut ts.d_tok)
            .map_err(|e| driver_err("tree tokens htod", &e))?;
        s.memcpy_htod(&positions, &mut ts.d_pos)
            .map_err(|e| driver_err("tree positions htod", &e))?;
        s.memcpy_htod(&anc, &mut ts.d_anc)
            .map_err(|e| driver_err("tree anc htod", &e))?;
        s.memcpy_htod(&n_anc, &mut ts.d_nanc)
            .map_err(|e| driver_err("tree n_anc htod", &e))?;

        if let Some(bucket) = bucket {
            // ── Graph route: replay the captured trunk (1 launch), then the
            // eager tail at the REAL node count (a padded LM head would read
            // the 656 MB f16 table once per extra 8-row tile).
            if self.batch_raw.is_none() {
                let ctx = self.cap_stream.context().clone();
                self.batch_raw = Some(Arc::new(BatchRawKernels::load(&ctx)?));
            }
            if self.tree_graphs.is_none() {
                let d_ctrl = self
                    .cap_stream
                    .alloc_zeros::<i32>(2)
                    .map_err(|e| driver_err("tree ctrl alloc", &e))?;
                self.tree_graphs = Some(TreeGraphs {
                    d_ctrl,
                    graphs: HashMap::new(),
                    raw_keepalive: self.batch_raw.clone(),
                });
            }
            let have = self
                .tree_graphs
                .as_ref()
                .expect("tree graphs just ensured")
                .graphs
                .contains_key(&bucket);
            if !have {
                if std::env::var_os("TRITIUM_TREE_DEBUG").is_some() {
                    eprintln!("tree-graph: capturing bucket {bucket}");
                }
                let g = self.record_graph_tree(&ts, bucket)?;
                self.tree_graphs
                    .as_mut()
                    .expect("tree graphs just ensured")
                    .graphs
                    .insert(bucket, SendGraph(g));
            }
            // The uploads above ran on the default stream; the graph replays
            // on the capture stream — order them before the ctrl write.
            s.synchronize()
                .map_err(|e| driver_err("tree pre-replay sync", &e))?;
            let ctrl = [prefix_len as i32, m as i32];
            let tg = self.tree_graphs.as_mut().expect("tree graphs just ensured");
            self.cap_stream
                .memcpy_htod(&ctrl, &mut tg.d_ctrl)
                .map_err(|e| driver_err("tree ctrl htod", &e))?;
            self.tree_graphs
                .as_ref()
                .expect("tree graphs just ensured")
                .graphs
                .get(&bucket)
                .expect("tree graph just inserted")
                .launch()
                .map_err(|e| driver_err("tree graph launch", &e))?;
            let cs = &self.cap_stream;
            Self::bl_rmsnorm(
                cs,
                &self.f_rmsnorm_batch,
                &ts.d_x,
                &self.d_output_norm,
                self.rms_eps,
                n_embd,
                m,
                &mut ts.d_norm_all,
            )?;
            Self::bl_lm_head_tiled(
                cs,
                &self.f_lm_head_tiled,
                &ts.d_norm_all,
                &self.d_token_embd_f16,
                n_embd,
                self.vocab,
                m,
                &mut ts.d_logits_all,
            )?;
            // The verify's writes (KV appends included) must be visible to the
            // default-stream consumers that follow (argmax/logits dtoh/promote).
            cs.synchronize()
                .map_err(|e| driver_err("tree post-replay sync", &e))?;
        } else {
            Self::bl_embed(
                s,
                &self.f_embed_batch,
                &self.d_token_embd,
                &ts.d_tok,
                n_embd,
                m,
                &mut ts.d_x,
            )?;

            for li in 0..self.layers.len() {
                Self::bl_rmsnorm(
                    s,
                    &self.f_rmsnorm_batch,
                    &ts.d_x,
                    &self.layers[li].attn_norm,
                    self.rms_eps,
                    n_embd,
                    m,
                    &mut ts.d_normed,
                )?;
                Self::bl_quant(
                    s,
                    &self.f_quant_batch,
                    &ts.d_normed,
                    n_embd,
                    m,
                    &mut ts.d_qact,
                    &mut ts.d_act_scale,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].q,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_q,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].k,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_k,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].v,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_v,
                )?;
                Self::bl_rope(
                    s,
                    &self.f_rope_batch,
                    &mut ts.d_q,
                    &self.d_cos,
                    &self.d_sin,
                    &ts.d_pos,
                    n_head,
                    head_dim,
                    m,
                )?;
                Self::bl_rope(
                    s,
                    &self.f_rope_batch,
                    &mut ts.d_k,
                    &self.d_cos,
                    &self.d_sin,
                    &ts.d_pos,
                    n_head_kv,
                    head_dim,
                    m,
                )?;
                // Provisional K/V at arena rows [cache_len, cache_len + m) — node i's
                // row is cache_len + i regardless of its depth (attention resolves
                // rows through the ancestor table, not contiguity).
                Self::bl_kv_append(
                    s,
                    &self.f_kv_append_batch,
                    &ts.d_k,
                    &mut self.kv_k[li],
                    prefix_len,
                    kv_width,
                    m,
                    if self.kv_dtype.has_scales() {
                        Some(&mut self.kv_k_scales[li])
                    } else {
                        None
                    },
                )?;
                Self::bl_kv_append(
                    s,
                    &self.f_kv_append_batch,
                    &ts.d_v,
                    &mut self.kv_v[li],
                    prefix_len,
                    kv_width,
                    m,
                    if self.kv_dtype.has_scales() {
                        Some(&mut self.kv_v_scales[li])
                    } else {
                        None
                    },
                )?;
                if head_dim.is_multiple_of(4) {
                    Self::bl_attn_tree_split(
                        s,
                        &self.f_attn_tree_scores,
                        &self.f_attn_tree_reduce,
                        &ts.d_q,
                        &self.kv_k[li],
                        &self.kv_v[li],
                        if self.kv_dtype.has_scales() {
                            Some(&self.kv_k_scales[li])
                        } else {
                            None
                        },
                        if self.kv_dtype.has_scales() {
                            Some(&self.kv_v_scales[li])
                        } else {
                            None
                        },
                        &mut ts.d_attn,
                        &mut ts.d_scores,
                        &ts.d_anc,
                        &ts.d_nanc,
                        ctx_max,
                        n_head,
                        n_head_kv,
                        head_dim,
                        self.attn_scale,
                        prefix_len,
                        m,
                    )?;
                } else {
                    Self::bl_attn_tree(
                        s,
                        &self.f_attn_tree,
                        &ts.d_q,
                        &self.kv_k[li],
                        &self.kv_v[li],
                        &mut ts.d_attn,
                        &mut ts.d_scores,
                        &ts.d_anc,
                        &ts.d_nanc,
                        ctx_max,
                        n_head,
                        n_head_kv,
                        head_dim,
                        self.attn_scale,
                        prefix_len,
                        m,
                    )?;
                }
                let attn_in: &CudaSlice<f32> =
                    if let Some(sn) = self.layers[li].attn_sub_norm.as_ref() {
                        Self::bl_rmsnorm(
                            s,
                            &self.f_rmsnorm_batch,
                            &ts.d_attn,
                            sn,
                            self.rms_eps,
                            q_width,
                            m,
                            &mut ts.d_attn_sn,
                        )?;
                        &ts.d_attn_sn
                    } else {
                        &ts.d_attn
                    };
                Self::bl_quant(
                    s,
                    &self.f_quant_batch,
                    attn_in,
                    q_width,
                    m,
                    &mut ts.d_qact,
                    &mut ts.d_act_scale,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].o,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_proj,
                )?;
                Self::bl_residual(s, &self.f_residual, &mut ts.d_x, &ts.d_proj, m * n_embd)?;

                Self::bl_rmsnorm(
                    s,
                    &self.f_rmsnorm_batch,
                    &ts.d_x,
                    &self.layers[li].ffn_norm,
                    self.rms_eps,
                    n_embd,
                    m,
                    &mut ts.d_normed,
                )?;
                Self::bl_quant(
                    s,
                    &self.f_quant_batch,
                    &ts.d_normed,
                    n_embd,
                    m,
                    &mut ts.d_qact,
                    &mut ts.d_act_scale,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].gate,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_gate,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].up,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_up,
                )?;
                Self::bl_relu2(s, &self.f_relu2, &mut ts.d_gate, &ts.d_up, m * n_ff)?;
                let down_in: &CudaSlice<f32> =
                    if let Some(sn) = self.layers[li].ffn_sub_norm.as_ref() {
                        Self::bl_rmsnorm(
                            s,
                            &self.f_rmsnorm_batch,
                            &ts.d_gate,
                            sn,
                            self.rms_eps,
                            n_ff,
                            m,
                            &mut ts.d_gate_sn,
                        )?;
                        &ts.d_gate_sn
                    } else {
                        &ts.d_gate
                    };
                Self::bl_quant(
                    s,
                    &self.f_quant_batch,
                    down_in,
                    n_ff,
                    m,
                    &mut ts.d_qact,
                    &mut ts.d_act_scale,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].down,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_proj,
                )?;
                Self::bl_residual(s, &self.f_residual, &mut ts.d_x, &ts.d_proj, m * n_embd)?;
            }

            // Final norm over ALL rows, batched LM head, per-row greedy argmax.
            Self::bl_rmsnorm(
                s,
                &self.f_rmsnorm_batch,
                &ts.d_x,
                &self.d_output_norm,
                self.rms_eps,
                n_embd,
                m,
                &mut ts.d_norm_all,
            )?;
            Self::bl_lm_head_tiled(
                s,
                &self.f_lm_head_tiled,
                &ts.d_norm_all,
                &self.d_token_embd_f16,
                n_embd,
                self.vocab,
                m,
                &mut ts.d_logits_all,
            )?;
        }
        // Forward complete: logits for every tree node sit in
        // `tree_scratch.d_logits_all[0..m*vocab]`; provisional K/V occupy arena
        // rows [cache_len, cache_len + m). Nothing is committed yet.
        self.tree_scratch = Some(ts);
        Ok(m)
    }

    /// Greedy tree verify (ADR 0014): forward the draft tree, device-argmax
    /// every node, walk the accepted path and commit it. Returns the target's
    /// greedy tokens along the accepted path (+ the bonus token).
    pub fn tree_verify_greedy(
        &mut self,
        tokens: &[u32],
        parents: &[i32],
    ) -> Result<Vec<u32>, BackendError> {
        let m = self.tree_forward(tokens, parents)?;
        let s = &self.stream;
        let mut ts = self
            .tree_scratch
            .take()
            .expect("tree scratch after forward");
        Self::bl_argmax_rows_chunked(
            s,
            &self.f_argmax_partial,
            &self.f_argmax_combine,
            &ts.d_logits_all,
            self.vocab,
            m,
            &mut ts.d_amax_val,
            &mut ts.d_amax_idx,
            &mut ts.d_ids,
        )?;
        let tok_i: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let mut ids = vec![0i32; m];
        // The cached buffer may exceed this call's `m` — copy exactly m ids.
        let ids_view = ts.d_ids.slice(0..m);
        s.memcpy_dtoh(&ids_view, &mut ids)
            .map_err(|e| driver_err("tree ids dtoh", &e))?;

        // Device work is done — return the scratch to the cache. (An early `?`
        // above drops it instead; the next call simply re-allocates.)
        self.tree_scratch = Some(ts);

        // Greedy accept walk: from the root, descend into the (first) child whose
        // draft token equals the target argmax at the current node.
        let mut path = vec![0usize];
        loop {
            let cur = *path.last().expect("path non-empty");
            let want = ids[cur];
            let next = (cur + 1..m).find(|&c| parents[c] as usize == cur && tok_i[c] == want);
            match next {
                Some(c) => path.push(c),
                None => break,
            }
        }
        self.tree_promote(&path)?;

        Ok(path.iter().map(|&n| ids[n] as u32).collect())
    }

    /// Forward a draft tree and return every node's logits `[m, vocab]`
    /// row-major on the host, for a HOST-side accept rule (speculative
    /// sampling). Provisional K/V occupy arena rows [cache_len, cache_len+m);
    /// nothing is committed until [`Self::tree_commit`]. Any other decode
    /// operation (or another tree forward) in between invalidates the
    /// provisional rows — commit refuses once the pending tree is gone.
    pub fn tree_verify_logits(
        &mut self,
        tokens: &[u32],
        parents: &[i32],
    ) -> Result<Vec<f32>, BackendError> {
        let m = self.tree_forward(tokens, parents)?;
        let s = &self.stream;
        let ts = self
            .tree_scratch
            .take()
            .expect("tree scratch after forward");
        let mut logits = vec![0.0f32; m * self.vocab];
        let view = ts.d_logits_all.slice(0..m * self.vocab);
        s.memcpy_dtoh(&view, &mut logits)
            .map_err(|e| driver_err("tree logits dtoh", &e))?;
        self.tree_scratch = Some(ts);
        self.pending_tree = Some((m, self.cache_len, parents.to_vec()));
        Ok(logits)
    }

    /// Commit the host-chosen accepted path of the pending tree (from
    /// [`Self::tree_verify_logits`]): promote its K/V rows and advance the
    /// cache. `path` holds tree-node indices, starting at the root (0), each
    /// subsequent node a child of the previous.
    pub fn tree_commit(&mut self, path: &[usize]) -> Result<(), BackendError> {
        let Some((m, fwd_cache_len, parents)) = self.pending_tree.take() else {
            return Err(BackendError::InvalidInput(
                "tree_commit: no pending tree (call tree_verify_logits first; any \
                 intervening decode operation invalidates the provisional rows)"
                    .into(),
            ));
        };
        if fwd_cache_len != self.cache_len {
            return Err(BackendError::InvalidInput(format!(
                "tree_commit: the cache moved since the tree forward ({} -> {}) — the \
                 provisional rows were overwritten by an intervening decode operation; \
                 re-run tree_verify_logits",
                fwd_cache_len, self.cache_len
            )));
        }
        if path.is_empty() || path[0] != 0 {
            return Err(BackendError::InvalidInput(
                "tree_commit: path must start at the root (node 0)".into(),
            ));
        }
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            if b >= m || parents[b] as usize != a {
                return Err(BackendError::InvalidInput(format!(
                    "tree_commit: node {b} is not a child of {a} (m={m})"
                )));
            }
        }
        self.tree_promote(path)
    }

    /// Promote the accepted path: node path[k] (arena slot cache_len + path[k])
    /// moves to arena row cache_len + k, then the cache advances by the path
    /// length. `path` is strictly increasing (children follow parents), so
    /// src >= dst and no promoted row is overwritten before it is read.
    fn tree_promote(&mut self, path: &[usize]) -> Result<(), BackendError> {
        let s = &self.stream;
        // Arena rows are addressed in BYTES (kv_elem = 4/2/1); a row copy is
        // dtype-agnostic. Under the i8 rung each token also owns a scale row.
        let row_bytes = self.kv_width * self.kv_elem;
        let sc_row = if self.kv_dtype.has_scales() {
            self.n_head_kv * (self.head_dim / KV_QGROUP)
        } else {
            0
        };
        for (k, &node) in path.iter().enumerate() {
            if node == k {
                continue; // already in place (chain prefix)
            }
            let src = (self.cache_len + node) * row_bytes;
            let dst = (self.cache_len + k) * row_bytes;
            for li in 0..self.layers.len() {
                for arena in [&mut self.kv_k[li], &mut self.kv_v[li]] {
                    // Copy via a device temporary: src/dst rows never overlap (path is
                    // strictly increasing, src >= dst), but the tmp keeps the copy
                    // trivially safe for any future path shape.
                    let row = {
                        let src_slice = arena.slice(src..src + row_bytes);
                        let mut tmp = s
                            .alloc_zeros::<u8>(row_bytes)
                            .map_err(|e| driver_err("tree promote tmp", &e))?;
                        s.memcpy_dtod(&src_slice, &mut tmp)
                            .map_err(|e| driver_err("tree promote read", &e))?;
                        tmp
                    };
                    let mut dst_slice = arena.slice_mut(dst..dst + row_bytes);
                    s.memcpy_dtod(&row, &mut dst_slice)
                        .map_err(|e| driver_err("tree promote write", &e))?;
                }
                if sc_row > 0 {
                    let s_src = (self.cache_len + node) * sc_row;
                    let s_dst = (self.cache_len + k) * sc_row;
                    for arena in [&mut self.kv_k_scales[li], &mut self.kv_v_scales[li]] {
                        let row = {
                            let src_slice = arena.slice(s_src..s_src + sc_row);
                            let mut tmp = s
                                .alloc_zeros::<f32>(sc_row)
                                .map_err(|e| driver_err("tree promote sc tmp", &e))?;
                            s.memcpy_dtod(&src_slice, &mut tmp)
                                .map_err(|e| driver_err("tree promote sc read", &e))?;
                            tmp
                        };
                        let mut dst_slice = arena.slice_mut(s_dst..s_dst + sc_row);
                        s.memcpy_dtod(&row, &mut dst_slice)
                            .map_err(|e| driver_err("tree promote sc write", &e))?;
                    }
                }
            }
        }
        self.cache_len += path.len();
        Ok(())
    }

    // --- batched (M>1) prefill launch helpers (safe launches; eager one-shot path) ---

    /// Capture the tree-verify trunk (embed → 30 layers, NO final norm / LM
    /// head — those run eagerly at the real node count) for a padded tree of
    /// `bucket` nodes, raw-launched on `cap_stream`. Mirrors the eager
    /// `tree_verify_greedy` trunk OP-FOR-OP: every kernel is the same function
    /// with the same geometry, except the three ctrl-driven twins
    /// (`kv_append_tree_g`, `gqa_attention_tree_{scores,reduce}_ctrl_g`) that
    /// read [prefix_len, real_m] from `TreeGraphs::d_ctrl` at replay. Real
    /// rows' math is row-independent, so their results are bit-identical to
    /// the eager path (gated by `cuda_tree_verify_greedy_lossless`).
    fn record_graph_tree(
        &self,
        ts: &TreeScratch,
        bucket: usize,
    ) -> Result<CudaGraph, BackendError> {
        let s = &self.cap_stream;
        let mb = bucket;
        let (n_embd, q_width, kv_width, n_ff) =
            (self.n_embd, self.q_width, self.kv_width, self.n_ff);
        let (n_head, n_head_kv, head_dim) = (self.n_head, self.n_head_kv, self.head_dim);
        let raw = self.batch_raw();
        let (f_kv, f_sc, f_rd) = (
            raw.kv_append_tree,
            raw.attn_tree_scores_ctrl,
            raw.attn_tree_reduce_ctrl,
        );

        let lin = |l: &ResidentLinear| LinPtrs {
            w: dptr(l.device.as_ref(), s),
            sc: dptr(&l.scales, s),
            n: l.n,
            k: l.k,
            rb: l.row_bytes,
        };
        struct TreeLayerPtrs {
            attn_norm: sys::CUdeviceptr,
            attn_sub_norm: Option<sys::CUdeviceptr>,
            ffn_norm: sys::CUdeviceptr,
            ffn_sub_norm: Option<sys::CUdeviceptr>,
            q: LinPtrs,
            k: LinPtrs,
            v: LinPtrs,
            o: LinPtrs,
            gate: LinPtrs,
            up: LinPtrs,
            down: LinPtrs,
            kv_k: sys::CUdeviceptr,
            kv_v: sys::CUdeviceptr,
        }
        let layers: Vec<TreeLayerPtrs> = self
            .layers
            .iter()
            .enumerate()
            .map(|(li, l)| TreeLayerPtrs {
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
                kv_k: dptr(&self.kv_k[li], s),
                kv_v: dptr(&self.kv_v[li], s),
            })
            .collect();
        let tg = self
            .tree_graphs
            .as_ref()
            .expect("TreeGraphs created before record");
        let d_ctrl = dptr(&tg.d_ctrl, s);
        let (d_tok, d_pos, d_anc, d_nanc) = (
            dptr(&ts.d_tok, s),
            dptr(&ts.d_pos, s),
            dptr(&ts.d_anc, s),
            dptr(&ts.d_nanc, s),
        );
        let (d_x, d_normed, d_q, d_k, d_v) = (
            dptr(&ts.d_x, s),
            dptr(&ts.d_normed, s),
            dptr(&ts.d_q, s),
            dptr(&ts.d_k, s),
            dptr(&ts.d_v, s),
        );
        let (d_attn, d_attn_sn, d_proj, d_gate, d_up, d_gate_sn) = (
            dptr(&ts.d_attn, s),
            dptr(&ts.d_attn_sn, s),
            dptr(&ts.d_proj, s),
            dptr(&ts.d_gate, s),
            dptr(&ts.d_up, s),
            dptr(&ts.d_gate_sn, s),
        );
        let (d_qact, d_act_scale, d_scores) = (
            dptr(&ts.d_qact, s),
            dptr(&ts.d_act_scale, s),
            dptr(&ts.d_scores, s),
        );
        let (d_cos, d_sin, d_token_embd) = (
            dptr(&self.d_cos, s),
            dptr(&self.d_sin, s),
            dptr(&self.d_token_embd, s),
        );

        // The ctrl-driven launches, closed over the baked dims.
        let cs = self.cap_stream.cu_stream();
        let (kw_i, mb_i) = (kv_width as i32, mb as i32);
        let (stride_i, nh_i, nhkv_i, hd_i, ma_i) = (
            self.max_ctx as i32,
            n_head as i32,
            n_head_kv as i32,
            head_dim as i32,
            mb as i32,
        );
        let scale = self.attn_scale;
        let kv_append = |src: sys::CUdeviceptr, base: sys::CUdeviceptr| {
            let grid = (((mb * kv_width) as u32).div_ceil(256), 1, 1);
            let mut params = [pp(&src), pp(&base), pp(&d_ctrl), pp(&kw_i), pp(&mb_i)];
            raw_launch(f_kv, grid, (256, 1, 1), 0, cs, &mut params)
        };
        let attn = |kv_k: sys::CUdeviceptr, kv_v: sys::CUdeviceptr| {
            const TREE_SCORE_CHUNK: usize = 128; // keep in sync with decode.cu
            let grid = (
                (mb * n_head) as u32,
                (self.max_ctx.div_ceil(TREE_SCORE_CHUNK)) as u32,
                1,
            );
            let mut params = [
                pp(&d_q),
                pp(&kv_k),
                pp(&d_scores),
                pp(&d_anc),
                pp(&d_nanc),
                pp(&d_ctrl),
                pp(&stride_i),
                pp(&nh_i),
                pp(&nhkv_i),
                pp(&hd_i),
                pp(&scale),
                pp(&ma_i),
                pp(&mb_i),
            ];
            raw_launch(f_sc, grid, (32, 1, 1), 0, cs, &mut params)?;
            let grid = ((mb * n_head) as u32, 1, 1);
            let smem = (self.max_ctx * 4) as u32;
            let mut params = [
                pp(&kv_v),
                pp(&d_scores),
                pp(&d_attn),
                pp(&d_anc),
                pp(&d_nanc),
                pp(&d_ctrl),
                pp(&stride_i),
                pp(&nh_i),
                pp(&nhkv_i),
                pp(&hd_i),
                pp(&ma_i),
                pp(&mb_i),
            ];
            raw_launch(f_rd, grid, (128, 1, 1), smem, cs, &mut params)
        };

        // Drain the device_ptr events so the capture carries no pre-capture deps.
        s.synchronize()
            .map_err(|e| driver_err("tree pre-capture cap sync", &e))?;
        self.stream
            .synchronize()
            .map_err(|e| driver_err("tree pre-capture default sync", &e))?;

        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| driver_err("tree begin_capture", &e))?;

        capture_body(s, || {
            self.gb_embed(d_token_embd, d_tok, d_x, mb)?;
            for lp in &layers {
                self.gb_rmsnorm(d_x, lp.attn_norm, n_embd, d_normed, mb)?;
                self.gb_quant(d_normed, n_embd, d_qact, d_act_scale, mb)?;
                self.gb_matmul(&lp.q, d_qact, d_act_scale, d_q, mb)?;
                self.gb_matmul(&lp.k, d_qact, d_act_scale, d_k, mb)?;
                self.gb_matmul(&lp.v, d_qact, d_act_scale, d_v, mb)?;
                self.gb_rope(d_q, d_cos, d_sin, d_pos, n_head, head_dim, mb)?;
                self.gb_rope(d_k, d_cos, d_sin, d_pos, n_head_kv, head_dim, mb)?;
                kv_append(d_k, lp.kv_k)?;
                kv_append(d_v, lp.kv_v)?;
                attn(lp.kv_k, lp.kv_v)?;
                let attn_in = if let Some(sn) = lp.attn_sub_norm {
                    self.gb_rmsnorm(d_attn, sn, q_width, d_attn_sn, mb)?;
                    d_attn_sn
                } else {
                    d_attn
                };
                self.gb_quant(attn_in, q_width, d_qact, d_act_scale, mb)?;
                self.gb_matmul(&lp.o, d_qact, d_act_scale, d_proj, mb)?;
                self.gb_residual(d_x, d_proj, mb * n_embd)?;

                self.gb_rmsnorm(d_x, lp.ffn_norm, n_embd, d_normed, mb)?;
                self.gb_quant(d_normed, n_embd, d_qact, d_act_scale, mb)?;
                self.gb_matmul(&lp.gate, d_qact, d_act_scale, d_gate, mb)?;
                self.gb_matmul(&lp.up, d_qact, d_act_scale, d_up, mb)?;
                self.gb_relu2(d_gate, d_up, mb * n_ff)?;
                let down_in = if let Some(sn) = lp.ffn_sub_norm {
                    self.gb_rmsnorm(d_gate, sn, n_ff, d_gate_sn, mb)?;
                    d_gate_sn
                } else {
                    d_gate
                };
                self.gb_quant(down_in, n_ff, d_qact, d_act_scale, mb)?;
                self.gb_matmul(&lp.down, d_qact, d_act_scale, d_proj, mb)?;
                self.gb_residual(d_x, d_proj, mb * n_embd)?;
            }
            Ok(())
        })?;

        let graph = s
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| driver_err("tree end_capture", &e))?
            .ok_or_else(|| BackendError::Backend("tree graph capture produced no graph".into()))?;
        Ok(graph)
    }
}

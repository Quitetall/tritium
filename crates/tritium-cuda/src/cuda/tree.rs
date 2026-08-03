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
    /// Returns `(m, on_cap)`: `on_cap` = the forward ran on the graph route
    /// with its tail (norm/head) IN FLIGHT on `cap_stream` — consumers must
    /// order their reads on that stream (L2, ADR 0032: the two per-verify
    /// full syncs this fn used to carry are gone; the single ordering point
    /// is the consumer's own readback).
    ///
    /// I2 (L3 batch-slot spec decode): `slot = Some((batch, r))` runs the SAME
    /// forward against dense batch slot `r` of a `BatchKv` instead of the
    /// single-seq cache — the prefix is the slot's rows `[0, positions[r])`,
    /// provisional K/V land at region rows `[positions[r], positions[r]+m)`,
    /// and every KV row index is offset by `r · max_ctx` (graph route: via
    /// `ctrl[2]`; eager route: via a region-offset arena view / a shifted
    /// `cache_len` scalar). `slot = None` is the pre-I2 single-seq path,
    /// bit-identical (base 0). The scratch + captured graphs live on the
    /// TARGET (`self.tree_scratch`/`tree_graphs` vs the batch's own fields),
    /// so the two never invalidate each other.
    ///
    /// I3: a PAGED `BatchKv` slot runs the same forward through the `_paged`
    /// ctrl twins — `ctrl[2]` carries the slot's page-table offset
    /// (`r · tstride`) instead of a row base, every KV row is translated
    /// logical → physical through the shared `d_table` (pointer baked at
    /// capture, content uploaded per verify), and an up-front reservation
    /// guard pins `[0, prefix + padded m)` to mapped pages. Callers
    /// guarantee for `slot`: f32 KV rung, valid live row, and (paged only)
    /// `head_dim % 4 == 0`.
    fn tree_forward(
        &mut self,
        tokens: &[u32],
        parents: &[i32],
        mut slot: Option<(&mut BatchKv, usize)>,
    ) -> Result<(usize, bool), BackendError> {
        // A new single-seq forward invalidates any uncommitted previous tree.
        // A batch-slot forward touches neither the single-seq arenas nor the
        // single-seq scratch, so a pending single-seq tree stays committable.
        if slot.is_none() {
            self.pending_tree = None;
        }
        // Region geometry: watermark, capacity and KV row base of the target.
        // A PAGED slot (I3) has no row base — its rows are page-scattered, so
        // `table_off` (the slot's page-table row offset, `r · tstride`) rides
        // in ctrl word 2 instead and the paged twins translate every KV row
        // through the shared `d_table`. Exactly one of `row_base`/`table_off`
        // is meaningful; the dense/single-seq paths keep base semantics.
        let (prefix_len, region_ctx) = match &slot {
            Some((b, r)) => (b.positions[*r], b.max_ctx),
            None => (self.cache_len, self.max_ctx),
        };
        let table_off: Option<usize> = match &slot {
            Some((b, r)) => b.pages.as_ref().map(|pg| *r * pg.tstride),
            None => None,
        };
        let row_base = match (&slot, table_off) {
            (Some((b, r)), None) => *r * b.max_ctx,
            _ => 0,
        };
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
        if prefix_len + m > region_ctx {
            return Err(BackendError::InvalidInput(format!(
                "tree_verify overflow: prefix_len={prefix_len} + {m} nodes > region max_ctx={region_ctx}"
            )));
        }

        // Graph route: pad to the smallest captured bucket and replay ONE
        // graph instead of ~420 eager launches. Requirements: the split
        // attention geometry (the ctrl twins are float4 kernels), a bucketable
        // size, room for the PADDED tree in the arena, and a context whose
        // score staging fits the default shared-memory limit (the raw-handle
        // capture path doesn't carry the opt-in attribute the safe handles
        // get at load).
        // (Batch slots share every condition: the slot-route wrapper requires
        // the f32 KV rung, batch arenas are f32, and `batch.max_ctx` equals
        // `self.max_ctx` by construction — the score stride and reduce smem
        // stay `self.max_ctx` for both targets.)
        let bucket = if self.head_dim.is_multiple_of(4)
            && m <= TREE_BUCKET_MAX
            && self.max_ctx * 4 <= 48 * 1024
            && self.kv_elem == 4 // non-f32 KV: eager tree (no ctrl twins; graph measured ≈ no win)
            && std::env::var_os("TRITIUM_TREE_EAGER").is_none()
        {
            TREE_BUCKETS
                .iter()
                .copied()
                .find(|&b| b >= m && prefix_len + b <= region_ctx)
        } else {
            None
        };
        if bucket.is_none() {
            // ~600 eager launches per verify instead of 1 replay — worth one
            // loud line per process (ctx > 12288, non-f32 KV, or m > 48).
            static EAGER_WARN: std::sync::Once = std::sync::Once::new();
            EAGER_WARN.call_once(|| {
                eprintln!(
                    "tritium-cuda: tree verify falling back to the EAGER route \
                     (m={m}, kv_elem={}, max_ctx={}) — ~600 launches/verify",
                    self.kv_elem, self.max_ctx
                );
            });
        }
        // mb = padded node count (bucket) or the real m on the eager path;
        // every host array below is built at stride/length mb. Pad rows are
        // root-token duplicates at depth 1 — valid math whose results only
        // the pads themselves ever see.
        let mb = bucket.unwrap_or(m);

        // I3 paged guard: every row the trunk touches — the committed prefix
        // plus the PADDED provisional rows [prefix_len, prefix_len + mb) (the
        // graph route's pads write real bytes) — must sit on a reserved page
        // BEFORE any device work. An unmapped (-1) table entry inside a
        // kernel is UB, so this is a loud host error with zero state change
        // (draft_batch's up-front guard shape; reservation is prefix-
        // contiguous, so the last row's page implies every earlier one).
        if let Some((b, r)) = &slot
            && b.pages.is_some()
            && !b.page_mapped(*r, prefix_len + mb - 1)
        {
            return Err(BackendError::InvalidInput(format!(
                "tree_verify: slot {r} rows {prefix_len}..{} not page-reserved \
                 (reserve_pages for the prefix + the padded tree before verifying)",
                prefix_len + mb
            )));
        }

        // Depths, RoPE positions, and per-node ancestor slot lists (root-first,
        // including self). Ancestors are REGION-LOCAL arena slots
        // (prefix_len + node index): the kernels add the KV row base, so the
        // same table shape serves the single-seq cache and any batch slot.
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
                anc[dst_off + np] = (prefix_len + i) as i32;
                n_anc[i] = n_anc[p] + 1;
            } else {
                anc[i * mb] = (prefix_len + i) as i32;
                n_anc[i] = 1;
            }
        }
        for i in m..mb {
            // Pad: a root child (its ancestor list is the root's slot + its own).
            anc[i * mb] = prefix_len as i32;
            anc[i * mb + 1] = (prefix_len + i) as i32;
            n_anc[i] = 2;
        }
        let mut positions: Vec<i32> = depth.iter().map(|&d| (prefix_len + d) as i32).collect();
        positions.resize(mb, (prefix_len + 1) as i32);

        let s = &self.stream;
        let (n_embd, q_width, kv_width, n_ff) =
            (self.n_embd, self.q_width, self.kv_width, self.n_ff);
        let (n_head, n_head_kv, head_dim) = (self.n_head, self.n_head_kv, self.head_dim);
        let ctx_max = prefix_len + m;

        // Reusable M=N scratch: allocated on first use (or re-grown for a larger
        // tree), then cached on the model — per-call alloc/free measurably ate
        // the speculative gains. Scores are sized by `max_ctx` (the largest any
        // verify can need) so growth is driven by `m` alone.
        // Capacity covers the graph buckets from the start so the captured
        // graphs' baked pointers stay valid; an oversized eager tree (> the
        // bucket max) re-grows the scratch and must drop every graph.
        let m_cap_want = mb.max(TREE_BUCKET_MAX);
        let scratch_stale = match &slot {
            Some((b, _)) => b.tree_scratch.as_ref().is_none_or(|t| t.m_cap < m_cap_want),
            None => self
                .tree_scratch
                .as_ref()
                .is_none_or(|t| t.m_cap < m_cap_want),
        };
        if scratch_stale {
            // Drop the owner's graphs FIRST, unconditionally: an error-path
            // `?` on a previous call may have dropped the scratch while
            // captured graphs kept its baked pointers — stale graphs must be
            // gone before anything could ever replay them (and before the
            // alloc below, so an alloc failure can't resurrect the hazard).
            // A batch owner's SLOTS graphs (I4) bake the same scratch, so
            // they go too.
            match slot.as_mut() {
                Some((b, _)) => {
                    b.tree_graphs = None;
                    b.tree_slots_graphs = None;
                }
                None => self.tree_graphs = None,
            }
            let ts_new = self.alloc_tree_scratch(m_cap_want)?;
            match slot.as_mut() {
                Some((b, _)) => b.tree_scratch = Some(ts_new),
                None => self.tree_scratch = Some(ts_new),
            }
        }
        // Move the scratch out for disjoint borrows vs the KV arenas below;
        // put it back on the same owner once the device work completes. (An
        // early `?` drops it instead; the `scratch_stale` check above then
        // re-allocates AND drops the owner's stale graphs on the next call.)
        let mut ts = match slot.as_mut() {
            Some((b, _)) => b.tree_scratch.take(),
            None => self.tree_scratch.take(),
        }
        .expect("tree scratch just ensured");

        // Uploads go into the cached buffers too (oversized is fine — kernels
        // read exactly the first m / m·m entries; `max_anc == m` is a stride
        // into linear memory, not a buffer shape).
        let mut tok_i: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        tok_i.resize(mb, tok_i[0]); // pads embed the root token (valid, unread)
        // Graph route: uploads go straight onto `cap_stream` (the only reader
        // is the captured graph, on that stream) — stream order replaces the
        // old full pre-replay sync. Eager route: default stream, as before.
        let up = if bucket.is_some() {
            &self.cap_stream
        } else {
            s
        };
        up.memcpy_htod(&tok_i, &mut ts.d_tok)
            .map_err(|e| driver_err("tree tokens htod", &e))?;
        up.memcpy_htod(&positions, &mut ts.d_pos)
            .map_err(|e| driver_err("tree positions htod", &e))?;
        up.memcpy_htod(&anc, &mut ts.d_anc)
            .map_err(|e| driver_err("tree anc htod", &e))?;
        up.memcpy_htod(&n_anc, &mut ts.d_nanc)
            .map_err(|e| driver_err("tree n_anc htod", &e))?;

        if let Some(bucket) = bucket {
            // ── Graph route: replay the captured trunk (1 launch), then the
            // eager tail at the REAL node count (a padded LM head would read
            // the 656 MB f16 table once per extra 8-row tile).
            //
            // Drain any pending default-stream work before the replay (on
            // `cap_stream`) touches the KV arena — the same edge `step_graph`
            // and `decode_batch_graph` order. Without it, back-to-back
            // verifies race: the PREVIOUS verify's `tree_promote` dtods run
            // on the default stream, and this replay's kv_append overwrites
            // their SOURCE rows (found as a wrong promoted row in the I2
            // bitwise KV gate; the eager route shares the default stream and
            // never races).
            self.stream
                .synchronize()
                .map_err(|e| driver_err("tree pre-replay default sync", &e))?;
            if self.batch_raw.is_none() {
                let ctx = self.cap_stream.context().clone();
                self.batch_raw = Some(Arc::new(BatchRawKernels::load(&ctx)?));
            }
            let owner_missing = match &slot {
                Some((b, _)) => b.tree_graphs.is_none(),
                None => self.tree_graphs.is_none(),
            };
            if owner_missing {
                let d_ctrl = self
                    .cap_stream
                    .alloc_zeros::<i32>(3)
                    .map_err(|e| driver_err("tree ctrl alloc", &e))?;
                let tg = TreeGraphs {
                    d_ctrl,
                    graphs: HashMap::new(),
                    raw_keepalive: self.batch_raw.clone(),
                };
                match slot.as_mut() {
                    Some((b, _)) => b.tree_graphs = Some(tg),
                    None => self.tree_graphs = Some(tg),
                }
            }
            let have = match &slot {
                Some((b, _)) => b
                    .tree_graphs
                    .as_ref()
                    .expect("tree graphs just ensured")
                    .graphs
                    .contains_key(&bucket),
                None => self
                    .tree_graphs
                    .as_ref()
                    .expect("tree graphs just ensured")
                    .graphs
                    .contains_key(&bucket),
            };
            if !have {
                if std::env::var_os("TRITIUM_TREE_DEBUG").is_some() {
                    eprintln!("tree-graph: capturing bucket {bucket}");
                }
                // Bake the TARGET's ctrl + KV arena pointers into the capture
                // (the row base / table offset rides in ctrl at replay, so
                // one batch capture serves every slot). Paged batch (I3): the
                // arenas are the page POOLS and the page-table POINTER is
                // baked too — its CONTENT is per-replay data, uploaded below
                // exactly like `decode_batch_graph`'s.
                let cs = &self.cap_stream;
                type KvPtrs = Vec<(sys::CUdeviceptr, sys::CUdeviceptr)>;
                let (d_ctrl, kv, d_table): (sys::CUdeviceptr, KvPtrs, Option<sys::CUdeviceptr>) =
                    match &slot {
                        Some((b, _)) => (
                            dptr(&b.tree_graphs.as_ref().expect("just ensured").d_ctrl, cs),
                            (0..self.layers.len())
                                .map(|li| (dptr(&b.kv_k[li], cs), dptr(&b.kv_v[li], cs)))
                                .collect(),
                            b.pages.as_ref().map(|pg| dptr(&pg.d_table, cs)),
                        ),
                        None => (
                            dptr(&self.tree_graphs.as_ref().expect("just ensured").d_ctrl, cs),
                            (0..self.layers.len())
                                .map(|li| (dptr(&self.kv_k[li], cs), dptr(&self.kv_v[li], cs)))
                                .collect(),
                            None,
                        ),
                    };
                let g = self.record_graph_tree(&ts, bucket, d_ctrl, &kv, d_table, false)?;
                let tg = match slot.as_mut() {
                    Some((b, _)) => b.tree_graphs.as_mut(),
                    None => self.tree_graphs.as_mut(),
                }
                .expect("tree graphs just ensured");
                tg.graphs.insert(bucket, SendGraph(g));
                // The eager paged route may have created this TreeGraphs
                // before `batch_raw` existed — (re)pin the modules the
                // freshly captured graph references.
                tg.raw_keepalive = self.batch_raw.clone();
            }
            // Uploads, ctrl write, replay and tail all sit on `cap_stream` —
            // stream order is the only ordering needed (no host sync).
            // Word 2: dense = the slot's TOKEN-ROW base; paged = the slot's
            // page-TABLE offset (`r · tstride`). Either is a token/table
            // index; i32 truncation would need a >2^31-entry arena/table
            // (multi-TB allocs fail long before).
            let ctrl2 = table_off.unwrap_or(row_base);
            debug_assert!(ctrl2 <= i32::MAX as usize, "tree ctrl word 2 overflows i32");
            let ctrl = [prefix_len as i32, m as i32, ctrl2 as i32];
            {
                // Paged: refresh the baked table's CONTENT for this replay
                // (`decode_batch_graph`'s shape — pointer at capture, data
                // per step).
                if let Some((b, _)) = slot.as_mut()
                    && let Some(pg) = b.pages.as_mut()
                {
                    self.cap_stream
                        .memcpy_htod(&pg.table, &mut pg.d_table)
                        .map_err(|e| driver_err("tree table htod", &e))?;
                }
                let tg = match slot.as_mut() {
                    Some((b, _)) => b.tree_graphs.as_mut(),
                    None => self.tree_graphs.as_mut(),
                }
                .expect("tree graphs just ensured");
                self.cap_stream
                    .memcpy_htod(&ctrl, &mut tg.d_ctrl)
                    .map_err(|e| driver_err("tree ctrl htod", &e))?;
                tg.graphs
                    .get(&bucket)
                    .expect("tree graph just inserted")
                    .launch()
                    .map_err(|e| driver_err("tree graph launch", &e))?;
            }
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
            // No sync here (L2): consumers run their argmax/readback on
            // `cap_stream` (see `on_cap`), and the pageable dtoh they end with
            // drains the stream before any default-stream work (promote)
            // touches the arena.
        } else {
            // I3 eager PAGED slot: region-offset views can't address page-
            // scattered rows, so the eager route launches the SAME ctrl-driven
            // paged twins the graph route captures (the decode split's shape:
            // per-batch paged handles), reading ctrl + table from device.
            // Upload both once per verify on the default stream (all eager
            // work is stream-ordered there, promote dtods included). The
            // batch's `TreeGraphs` is ensured just for its `d_ctrl` — its
            // graph map stays empty on this route.
            if let Some(toff) = table_off {
                let (b, _) = slot.as_mut().expect("table_off implies a slot");
                if b.tree_graphs.is_none() {
                    let d_ctrl = s
                        .alloc_zeros::<i32>(3)
                        .map_err(|e| driver_err("tree ctrl alloc (eager paged)", &e))?;
                    b.tree_graphs = Some(TreeGraphs {
                        d_ctrl,
                        graphs: HashMap::new(),
                        raw_keepalive: self.batch_raw.clone(),
                    });
                }
                debug_assert!(toff <= i32::MAX as usize, "tree ctrl word 2 overflows i32");
                let ctrl = [prefix_len as i32, m as i32, toff as i32];
                s.memcpy_htod(
                    &ctrl,
                    &mut b.tree_graphs.as_mut().expect("just ensured").d_ctrl,
                )
                .map_err(|e| driver_err("tree ctrl htod (eager paged)", &e))?;
                let pg = b.pages.as_mut().expect("table_off implies pages");
                s.memcpy_htod(&pg.table, &mut pg.d_table)
                    .map_err(|e| driver_err("tree table htod (eager paged)", &e))?;
            }
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
                    &self.f_tq1_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].q,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_q,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &self.f_tq1_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].k,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_k,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &self.f_tq1_tiled_scaled,
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
                // Provisional K/V at region rows [prefix_len, prefix_len + m) —
                // node i's row is prefix_len + i regardless of its depth
                // (attention resolves rows through the ancestor table, not
                // contiguity). Single-seq: the model arenas at base 0. Batch
                // slot: the batch's f32 arena, with the row base folded into
                // the kv_append `cache_len` scalar and the attention views'
                // pointer offset (region-relative indexing either way).
                match slot.as_mut() {
                    None => {
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
                        let (kview, vview) = (self.kv_k[li].as_view(), self.kv_v[li].as_view());
                        if head_dim.is_multiple_of(4) {
                            Self::bl_attn_tree_split(
                                s,
                                &self.f_attn_tree_scores,
                                &self.f_attn_tree_reduce,
                                &ts.d_q,
                                &kview,
                                &vview,
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
                                &kview,
                                &vview,
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
                    }
                    Some((b, _)) if b.pages.is_some() => {
                        // I3 paged slot: absolute pool addressing through the
                        // page table — the ctrl-driven paged twins, fed by the
                        // ctrl/table uploaded before the loop. score stride is
                        // self.max_ctx (the scratch's row pitch); grid/smem
                        // are bounded by the REAL context like the dense
                        // eager launches.
                        let tg = b.tree_graphs.as_ref().expect("ensured before the loop");
                        let pg = b.pages.as_ref().expect("paged arm");
                        Self::bl_kv_append_tree_paged(
                            s,
                            &self.f_kv_append_tree_paged,
                            &ts.d_k,
                            &mut b.kv_k[li],
                            &tg.d_ctrl,
                            &pg.d_table,
                            kv_width,
                            m,
                        )?;
                        Self::bl_kv_append_tree_paged(
                            s,
                            &self.f_kv_append_tree_paged,
                            &ts.d_v,
                            &mut b.kv_v[li],
                            &tg.d_ctrl,
                            &pg.d_table,
                            kv_width,
                            m,
                        )?;
                        // head_dim % 4 == 0 is guaranteed (the slot wrapper
                        // refuses paged verifies otherwise — the paged twins
                        // are float4 kernels).
                        Self::bl_attn_tree_split_paged(
                            s,
                            &self.f_attn_tree_scores_ctrl_paged,
                            &self.f_attn_tree_reduce_ctrl_paged,
                            &ts.d_q,
                            &b.kv_k[li],
                            &b.kv_v[li],
                            &tg.d_ctrl,
                            &pg.d_table,
                            &mut ts.d_attn,
                            &mut ts.d_scores,
                            &ts.d_anc,
                            &ts.d_nanc,
                            self.max_ctx,
                            ctx_max,
                            n_head,
                            n_head_kv,
                            head_dim,
                            self.attn_scale,
                            m,
                        )?;
                    }
                    Some((b, _)) => {
                        // Batch arenas are f32 (no scale planes); the slot
                        // wrapper guarantees kv_elem == 4, so the dtype-
                        // selected handles below are the f32 kernels.
                        Self::bl_kv_append(
                            s,
                            &self.f_kv_append_batch,
                            &ts.d_k,
                            &mut b.kv_k[li],
                            row_base + prefix_len,
                            kv_width,
                            m,
                            None,
                        )?;
                        Self::bl_kv_append(
                            s,
                            &self.f_kv_append_batch,
                            &ts.d_v,
                            &mut b.kv_v[li],
                            row_base + prefix_len,
                            kv_width,
                            m,
                            None,
                        )?;
                        let (lo, hi) = (row_base * kv_width, (row_base + region_ctx) * kv_width);
                        let (kview, vview) = (b.kv_k[li].slice(lo..hi), b.kv_v[li].slice(lo..hi));
                        if head_dim.is_multiple_of(4) {
                            Self::bl_attn_tree_split(
                                s,
                                &self.f_attn_tree_scores,
                                &self.f_attn_tree_reduce,
                                &ts.d_q,
                                &kview,
                                &vview,
                                None,
                                None,
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
                                &kview,
                                &vview,
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
                    }
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
                    &self.f_tq1_tiled_scaled,
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
                    &self.f_tq1_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].gate,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_gate,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &self.f_tq1_tiled_scaled,
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
                    &self.f_tq1_tiled_scaled,
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
        // Forward complete: logits for every tree node sit in the target's
        // `tree_scratch.d_logits_all[0..m*vocab]`; provisional K/V occupy
        // region rows [prefix_len, prefix_len + m). Nothing is committed yet.
        match slot.as_mut() {
            Some((b, _)) => b.tree_scratch = Some(ts),
            None => self.tree_scratch = Some(ts),
        }
        Ok((m, bucket.is_some()))
    }

    /// Allocate a [`TreeScratch`] sized for `m_cap` tree nodes (shared by the
    /// single-seq target and each `BatchKv`'s own scratch — the buffers are
    /// tree-shaped, not target-shaped).
    fn alloc_tree_scratch(&self, m_cap: usize) -> Result<TreeScratch, BackendError> {
        let s = &self.stream;
        let (n_embd, q_width, kv_width, n_ff) =
            (self.n_embd, self.q_width, self.kv_width, self.n_ff);
        let m = m_cap;
        let alloc =
            |n: usize, what: &str| s.alloc_zeros::<f32>(n).map_err(|e| driver_err(what, &e));
        Ok(TreeScratch {
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
            d_scores: alloc(m * self.n_head * self.max_ctx, "tree d_scores")?,
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
        })
    }

    /// Greedy tree verify (ADR 0014): forward the draft tree, device-argmax
    /// every node, walk the accepted path and commit it. Returns the target's
    /// greedy tokens along the accepted path (+ the bonus token).
    pub fn tree_verify_greedy(
        &mut self,
        tokens: &[u32],
        parents: &[i32],
    ) -> Result<Vec<u32>, BackendError> {
        self.tree_verify_greedy_in(tokens, parents, None)
    }

    /// **I2/I3 batch-slot greedy tree verify** (L3 batch-slot spec decode):
    /// the SAME verify as [`Self::tree_verify_greedy`], run against batch
    /// slot `row` of a [`BatchKv`] instead of the single-sequence cache — the
    /// tree attends the slot's committed rows `[0, positions[row])`, the
    /// accepted path is promoted into the slot's region, and
    /// `batch.positions[row]` advances by the accepted length. Other slots'
    /// KV is untouched, so a per-slot spec loop can verify each slot in turn
    /// (I4 builds on this).
    ///
    /// Dense batches (I2) address the slot's rows via a KV row base
    /// (`row · max_ctx`). PAGED batches (I3, ADR 0025) translate every KV row
    /// through the slot's page table instead; the caller must have
    /// [`BatchKv::reserve_pages`] covering `positions[row]` PLUS the padded
    /// tree (`prefix + bucket` tokens — the graph route's pad rows write real
    /// bytes; [`Self::tree_reservation_rows`] computes the exact demand)
    /// before verifying, or the verify refuses loudly with zero state
    /// change. Paged verifies additionally require `head_dim % 4 == 0` (the
    /// paged ctrl twins are float4 kernels; the non-split tree kernel has no
    /// paged twin).
    ///
    /// Common requirements: the f32 KV rung (batch arenas are f32 — the same
    /// constraint as [`Self::copy_kv_into_batch_row`]) and a live row. The
    /// tree-shape contract (root at node 0 = the slot's last committed token,
    /// topological parents) is identical to the single-seq verify.
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] on a bad/dead target, a malformed
    /// tree, region overflow, or a short page reservation; device errors
    /// otherwise.
    pub fn tree_verify_greedy_slot(
        &mut self,
        batch: &mut BatchKv,
        row: usize,
        tokens: &[u32],
        parents: &[i32],
    ) -> Result<Vec<u32>, BackendError> {
        if row >= batch.n {
            return Err(BackendError::InvalidInput(format!(
                "tree_verify_greedy_slot: row {row} >= batch n {}",
                batch.n
            )));
        }
        if batch.pages.is_some() && !self.head_dim.is_multiple_of(4) {
            return Err(BackendError::InvalidInput(format!(
                "tree_verify_greedy_slot: paged tree verify requires \
                 head_dim % 4 == 0 (float4 paged ctrl twins), got head_dim {}",
                self.head_dim
            )));
        }
        if self.kv_elem != 4 {
            return Err(BackendError::InvalidInput(
                "tree_verify_greedy_slot requires the f32 KV rung (batch arenas \
                 are f32); unset TRITIUM_KV"
                    .into(),
            ));
        }
        if !batch.live[row] {
            return Err(BackendError::InvalidInput(format!(
                "tree_verify_greedy_slot: row {row} is dead"
            )));
        }
        // Geometry check (review 69649e8): the slot route sizes smem/scores
        // off self.max_ctx and indexes b.kv_k[li] per layer — a batch built
        // by a different-geometry model would corrupt device memory. (The
        // same exposure predates I2 in decode_batch/draft_batch; this entry
        // is the newly load-bearing one for scratch sizing.)
        if batch.max_ctx != self.max_ctx || batch.kv_k.len() != self.layers.len() {
            return Err(BackendError::InvalidInput(format!(
                "tree_verify_greedy_slot: batch geometry (max_ctx {}, {} layers) \
                 does not match this model (max_ctx {}, {} layers) — the BatchKv \
                 must come from this model's new_batch",
                batch.max_ctx,
                batch.kv_k.len(),
                self.max_ctx,
                self.layers.len()
            )));
        }
        self.tree_verify_greedy_in(tokens, parents, Some((batch, row)))
    }

    /// Shared greedy-verify body — `slot` selects the target region (see
    /// [`Self::tree_forward`]).
    fn tree_verify_greedy_in(
        &mut self,
        tokens: &[u32],
        parents: &[i32],
        mut slot: Option<(&mut BatchKv, usize)>,
    ) -> Result<Vec<u32>, BackendError> {
        let (m, on_cap) =
            self.tree_forward(tokens, parents, slot.as_mut().map(|(b, r)| (&mut **b, *r)))?;
        // Consume on the stream that produced the logits (graph route:
        // cap_stream — the trailing pageable dtoh below is the ONE ordering
        // point and drains the whole verify before tree_promote's
        // default-stream work).
        let s = if on_cap {
            &self.cap_stream
        } else {
            &self.stream
        };
        let mut ts = match slot.as_mut() {
            Some((b, _)) => b.tree_scratch.take(),
            None => self.tree_scratch.take(),
        }
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
        let mut ids = vec![0i32; m];
        // The cached buffer may exceed this call's `m` — copy exactly m ids.
        let ids_view = ts.d_ids.slice(0..m);
        s.memcpy_dtoh(&ids_view, &mut ids)
            .map_err(|e| driver_err("tree ids dtoh", &e))?;

        // Device work is done — return the scratch to its owner. (An early `?`
        // above drops it instead; the next call simply re-allocates.)
        match slot.as_mut() {
            Some((b, _)) => b.tree_scratch = Some(ts),
            None => self.tree_scratch = Some(ts),
        }

        // Greedy accept walk: from the root, descend into the (first) child whose
        // draft token equals the target argmax at the current node.
        let mut path = vec![0usize];
        loop {
            let cur = *path.last().expect("path non-empty");
            let want = ids[cur];
            let next =
                (cur + 1..m).find(|&c| parents[c] as usize == cur && tokens[c] as i32 == want);
            match next {
                Some(c) => path.push(c),
                None => break,
            }
        }
        self.tree_promote_in(&path, slot)?;

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
        let (m, on_cap) = self.tree_forward(tokens, parents, None)?;
        let s = if on_cap {
            &self.cap_stream
        } else {
            &self.stream
        };
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

    /// Region dispatcher for the accepted-path promotion (I2/I3): `None` =
    /// the single-seq cache ([`Self::tree_promote`]); `Some((batch, r))`
    /// compacts the SAME strictly-increasing path inside batch slot `r`
    /// (slot-logical row `positions[r] + path[k]` → `positions[r] + k`) and
    /// advances `positions[r]` by the path length. Dense slots offset both
    /// rows by `r · max_ctx`; PAGED slots translate each row through the HOST
    /// page-table copy (the promote is host-issued dtods — no kernel, so no
    /// device table read; every row lives inside ONE page, so a per-row
    /// lookup suffices). Batch arenas are f32 with no scale planes, so the
    /// slot arm is the plain row-copy loop.
    fn tree_promote_in(
        &mut self,
        path: &[usize],
        slot: Option<(&mut BatchKv, usize)>,
    ) -> Result<(), BackendError> {
        let Some((batch, row)) = slot else {
            return self.tree_promote(path);
        };
        let s = &self.stream;
        let prefix = batch.positions[row];
        let row_base = row * batch.max_ctx;
        let row_bytes = self.kv_width * 4; // batch arenas are f32
        // Physical token row of a slot-logical row. Paged: the forward's
        // reservation guard already pinned every row in
        // [0, prefix + padded m) to a mapped page, and nothing between the
        // forward and this promote can unmap one (same &mut borrow).
        let phys_row = |logical: usize| -> usize {
            match batch.pages.as_ref() {
                None => row_base + logical,
                Some(pg) => {
                    let e = pg.table[row * pg.tstride + logical / KV_PAGE_TOKENS];
                    debug_assert!(e >= 0, "tree promote row unmapped (guarded in tree_forward)");
                    e as usize * KV_PAGE_TOKENS + logical % KV_PAGE_TOKENS
                }
            }
        };
        for (k, &node) in path.iter().enumerate() {
            if node == k {
                continue; // already in place (chain prefix)
            }
            debug_assert!(node > k, "tree_promote: path must be strictly increasing");
            let src = phys_row(prefix + node) * row_bytes;
            let dst = phys_row(prefix + k) * row_bytes;
            for li in 0..self.layers.len() {
                for arena in [&mut batch.kv_k[li], &mut batch.kv_v[li]] {
                    let (base, guard) = arena.device_ptr(s);
                    // SAFETY: one dtod within the live batch arena; src/dst
                    // are row-aligned, equal-length, DISJOINT byte ranges —
                    // node > k gives distinct logical rows, and dense base-add
                    // / the paged table (distinct pages from the free list ⇒
                    // injective) both map distinct logical rows to distinct
                    // physical rows inside the allocation (the forward's
                    // overflow + reservation guards bound them); ordered on
                    // this stream.
                    #[allow(unsafe_code)]
                    unsafe {
                        result::memcpy_dtod_async(
                            base + dst as sys::CUdeviceptr,
                            base + src as sys::CUdeviceptr,
                            row_bytes,
                            s.cu_stream(),
                        )
                    }
                    .map_err(|e| driver_err("tree promote slot row", &e))?;
                    drop(guard);
                }
            }
        }
        batch.positions[row] += path.len();
        Ok(())
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
            // node > k always (a node's index is >= its depth in the accepted
            // path), so src and dst are DISJOINT row-aligned ranges — a direct
            // device-to-device copy is safe. The previous shape allocated a
            // fresh device temp per (row, layer, arena) and copied twice —
            // an alloc storm on long accepts (review D1-corrected item, plan
            // SOTA P3); the raw-pointer copy is the copy_kv_into_batch_row
            // pattern.
            debug_assert!(node > k, "tree_promote: path must be strictly increasing");
            let src = (self.cache_len + node) * row_bytes;
            let dst = (self.cache_len + k) * row_bytes;
            for li in 0..self.layers.len() {
                for arena in [&mut self.kv_k[li], &mut self.kv_v[li]] {
                    let (base, guard) = arena.device_ptr(s);
                    // SAFETY: one dtod within a live arena; src/dst are
                    // row-aligned, equal-length, disjoint (node > k) byte
                    // ranges inside the allocation; ordered on this stream.
                    #[allow(unsafe_code)]
                    unsafe {
                        result::memcpy_dtod_async(
                            base + dst as sys::CUdeviceptr,
                            base + src as sys::CUdeviceptr,
                            row_bytes,
                            s.cu_stream(),
                        )
                    }
                    .map_err(|e| driver_err("tree promote row", &e))?;
                    drop(guard);
                }
                if sc_row > 0 {
                    let s_src = (self.cache_len + node) * sc_row * 4;
                    let s_dst = (self.cache_len + k) * sc_row * 4;
                    for arena in [&mut self.kv_k_scales[li], &mut self.kv_v_scales[li]] {
                        let (base, guard) = arena.device_ptr(s);
                        // SAFETY: as above — f32 arena addressed in bytes
                        // (offsets ×4), disjoint row-aligned ranges.
                        #[allow(unsafe_code)]
                        unsafe {
                            result::memcpy_dtod_async(
                                base + s_dst as sys::CUdeviceptr,
                                base + s_src as sys::CUdeviceptr,
                                sc_row * 4,
                                s.cu_stream(),
                            )
                        }
                        .map_err(|e| driver_err("tree promote sc row", &e))?;
                        drop(guard);
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
    /// read `[prefix_len, real_m, kv_row_base]` from the owning
    /// `TreeGraphs::d_ctrl` at replay. Real rows' math is row-independent, so
    /// their results are bit-identical to the eager path (gated by
    /// `cuda_tree_verify_greedy_lossless`).
    ///
    /// `d_ctrl` and the per-layer `kv` (K, V) arena base pointers are the
    /// TARGET's (I2): the model's single-seq cache, or a `BatchKv`'s dense
    /// arenas + own ctrl — whichever owns the capture. The slot row base is
    /// per-replay ctrl data, so one batch capture serves every slot.
    ///
    /// `paged_table` (I3): `Some(d_table)` marks a PAGED BatchKv owner — the
    /// three ctrl twins become their `_paged` siblings, `kv` holds the page
    /// POOL base pointers, the table POINTER is baked here (its content is
    /// per-replay data) and ctrl word 2 is read as the slot's table offset
    /// (`r · tstride`) instead of a KV row base.
    ///
    /// `slots` (I4): capture the BATCHED-slots twins instead — `d_ctrl` is
    /// then the owner's PER-ROW ctrl plane (`[3 · TREE_BUCKET_MAX]` i32,
    /// row g = `[prefix_len, local_node_or_-1, word2]`). The slots twins'
    /// signatures match the single-slot ctrl twins word for word (only the
    /// ctrl INTERPRETATION differs), so the capture body below is shared
    /// verbatim across all four routes.
    fn record_graph_tree(
        &self,
        ts: &TreeScratch,
        bucket: usize,
        d_ctrl: sys::CUdeviceptr,
        kv: &[(sys::CUdeviceptr, sys::CUdeviceptr)],
        paged_table: Option<sys::CUdeviceptr>,
        slots: bool,
    ) -> Result<CudaGraph, BackendError> {
        let s = &self.cap_stream;
        let mb = bucket;
        let (n_embd, q_width, kv_width, n_ff) =
            (self.n_embd, self.q_width, self.kv_width, self.n_ff);
        let (n_head, n_head_kv, head_dim) = (self.n_head, self.n_head_kv, self.head_dim);
        let raw = self.batch_raw();
        let paged = paged_table.is_some();
        let d_table: sys::CUdeviceptr = paged_table.unwrap_or(0);
        let (f_kv, f_sc, f_rd) = match (slots, paged) {
            (false, true) => (
                raw.kv_append_tree_paged,
                raw.attn_tree_scores_ctrl_paged,
                raw.attn_tree_reduce_ctrl_paged,
            ),
            (false, false) => (
                raw.kv_append_tree,
                raw.attn_tree_scores_ctrl,
                raw.attn_tree_reduce_ctrl,
            ),
            (true, true) => (
                raw.kv_append_tree_slots_paged,
                raw.attn_tree_scores_slots_paged,
                raw.attn_tree_reduce_slots_paged,
            ),
            (true, false) => (
                raw.kv_append_tree_slots,
                raw.attn_tree_scores_slots,
                raw.attn_tree_reduce_slots,
            ),
        };

        let lin = |l: &ResidentLinear| LinPtrs {
            w: dptr(l.device.as_ref(), s),
            sc: dptr(&l.scales, s),
            // Batch/tree launches use the dense kernel (no _sparse residual /
            // batch twins yet — Track B consolidation first); fields unused
            // there but kept uniform.
            bm: l.bitmap.as_ref().map_or(0, |b| dptr(b, s)),
            wpr: l.k.div_ceil(256).div_ceil(32) as i32,
            tq1: l.tq1,
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
                kv_k: kv[li].0,
                kv_v: kv[li].1,
            })
            .collect();
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
            if paged {
                let mut params = [
                    pp(&src),
                    pp(&base),
                    pp(&d_ctrl),
                    pp(&d_table),
                    pp(&kw_i),
                    pp(&mb_i),
                ];
                raw_launch(f_kv, grid, (256, 1, 1), 0, cs, &mut params)
            } else {
                let mut params = [pp(&src), pp(&base), pp(&d_ctrl), pp(&kw_i), pp(&mb_i)];
                raw_launch(f_kv, grid, (256, 1, 1), 0, cs, &mut params)
            }
        };
        let attn = |kv_k: sys::CUdeviceptr, kv_v: sys::CUdeviceptr| {
            const TREE_SCORE_CHUNK: usize = 128; // keep in sync with decode.cu
            let grid = (
                (mb * n_head) as u32,
                (self.max_ctx.div_ceil(TREE_SCORE_CHUNK)) as u32,
                1,
            );
            if paged {
                let mut params = [
                    pp(&d_q),
                    pp(&kv_k),
                    pp(&d_scores),
                    pp(&d_anc),
                    pp(&d_nanc),
                    pp(&d_ctrl),
                    pp(&d_table),
                    pp(&stride_i),
                    pp(&nh_i),
                    pp(&nhkv_i),
                    pp(&hd_i),
                    pp(&scale),
                    pp(&ma_i),
                    pp(&mb_i),
                ];
                raw_launch(f_sc, grid, (32, 1, 1), 0, cs, &mut params)?;
            } else {
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
            }
            let grid = ((mb * n_head) as u32, 1, 1);
            let smem = (self.max_ctx * 4) as u32;
            if paged {
                let mut params = [
                    pp(&kv_v),
                    pp(&d_scores),
                    pp(&d_attn),
                    pp(&d_anc),
                    pp(&d_nanc),
                    pp(&d_ctrl),
                    pp(&d_table),
                    pp(&stride_i),
                    pp(&nh_i),
                    pp(&nhkv_i),
                    pp(&hd_i),
                    pp(&ma_i),
                    pp(&mb_i),
                ];
                raw_launch(f_rd, grid, (128, 1, 1), smem, cs, &mut params)
            } else {
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
            }
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

    /// I3 eager-paged launch: `kv_append_tree_paged_g` — append `m`
    /// provisional K/V rows at slot-logical rows `[ctrl[0], ctrl[0] + m)`,
    /// each translated through the page table (`ctrl[2]` = the slot's table
    /// offset, `row · tstride`). `kv_base` is the page POOL.
    #[allow(clippy::too_many_arguments)]
    fn bl_kv_append_tree_paged(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        src: &CudaSlice<f32>,
        kv_base: &mut CudaSlice<f32>,
        ctrl: &CudaSlice<i32>,
        table: &CudaSlice<i32>,
        kv_width: usize,
        m: usize,
    ) -> Result<(), BackendError> {
        let (kw_i, m_i) = (kv_width as i32, m as i32);
        let cfg = LaunchConfig {
            grid_dim: (((m * kv_width) as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(src)
            .arg(kv_base)
            .arg(ctrl)
            .arg(table)
            .arg(&kw_i)
            .arg(&m_i);
        // SAFETY: matches `kv_append_tree_paged_g(src, pool, tree_ctrl, table,
        // kv_width, m)`; only `kv_base` mutable; the host reservation guard in
        // `tree_forward` pins every written row to a mapped page.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch tree kv_append paged", &e))?;
        }
        Ok(())
    }

    /// I3 eager-paged split tree attention: the ctrl-driven PAGED twins
    /// (scores fan-out + per-(node, head) 128-thread softmax/reduce), every
    /// f32 fold in the dense pair's order — paging changes addresses, never
    /// arithmetic. `k`/`v` are the page POOLS (absolute addressing through
    /// the table; no region views). `score_stride` is the scores scratch row
    /// pitch (`self.max_ctx`, what the graph route bakes); `ctx_bound` =
    /// `prefix_len + m` bounds the grid and shared staging (the kernels
    /// guard per row, mirroring the dense eager launches).
    #[allow(clippy::too_many_arguments)]
    fn bl_attn_tree_split_paged(
        s: &Arc<CudaStream>,
        f_scores: &CudaFunction,
        f_reduce: &CudaFunction,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        ctrl: &CudaSlice<i32>,
        table: &CudaSlice<i32>,
        out: &mut CudaSlice<f32>,
        scores: &mut CudaSlice<f32>,
        anc: &CudaSlice<i32>,
        n_anc: &CudaSlice<i32>,
        score_stride: usize,
        ctx_bound: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
        m: usize,
    ) -> Result<(), BackendError> {
        // Keep in sync with `#define TREE_SCORE_CHUNK` in decode.cu.
        const TREE_SCORE_CHUNK: usize = 128;
        let (ss_i, nh_i, nhkv_i, hd_i) = (
            score_stride as i32,
            n_head as i32,
            n_head_kv as i32,
            head_dim as i32,
        );
        let (ma_i, m_i) = (m as i32, m as i32);
        let cfg = LaunchConfig {
            grid_dim: (
                (m * n_head) as u32,
                (ctx_bound.div_ceil(TREE_SCORE_CHUNK)) as u32,
                1,
            ),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f_scores);
        l.arg(q)
            .arg(k)
            .arg(&mut *scores)
            .arg(anc)
            .arg(n_anc)
            .arg(ctrl)
            .arg(table)
            .arg(&ss_i)
            .arg(&nh_i)
            .arg(&nhkv_i)
            .arg(&hd_i)
            .arg(&scale)
            .arg(&ma_i)
            .arg(&m_i);
        // SAFETY: matches `gqa_attention_tree_scores_ctrl_paged_g(q, k, scores,
        // anc, n_anc, tree_ctrl, table, score_stride, n_head, n_head_kv,
        // head_dim, scale, max_anc, m)`; max_anc == m (the eager ancestor
        // table is [m, m]); only `scores` mutable.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch tree attention scores paged", &e))?;
        }
        let cfg = LaunchConfig {
            grid_dim: ((m * n_head) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (ctx_bound * 4) as u32,
        };
        let mut l = s.launch_builder(f_reduce);
        l.arg(v)
            .arg(&*scores)
            .arg(out)
            .arg(anc)
            .arg(n_anc)
            .arg(ctrl)
            .arg(table)
            .arg(&ss_i)
            .arg(&nh_i)
            .arg(&nhkv_i)
            .arg(&hd_i)
            .arg(&ma_i)
            .arg(&m_i);
        // SAFETY: matches `gqa_attention_tree_reduce_ctrl_paged_g(v, scores,
        // out, anc, n_anc, tree_ctrl, table, score_stride, n_head, n_head_kv,
        // head_dim, max_anc, m)` with ctx_bound·4 B of dynamic shared (the
        // handle's opt-in at load covers up to max_ctx·4).
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch tree attention reduce paged", &e))?;
        }
        Ok(())
    }

    // ─── I4: batched-slots tree verify (L3 batch-slot spec decode) ─────────

    /// Padded KV-row demand of ONE slot's [`Self::tree_verify_greedy_slot`]
    /// for a verify of `m` tree nodes at committed prefix `prefix_len`:
    /// `prefix_len + b` where `b` is the graph bucket `m` pads to (the pad
    /// rows write real bytes past the watermark), or `prefix_len + m` when
    /// the verify would take the eager route (non-bucketable geometry or
    /// `TRITIUM_TREE_EAGER`). Paged callers must
    /// [`BatchKv::reserve_pages`] AT LEAST this many tokens before the
    /// verify or it refuses loudly. The value is also a safe upper bound for
    /// [`Self::tree_verify_greedy_slots`], whose per-slot demand is only
    /// `prefix_len + m` (batched pads belong to no slot and write nothing) —
    /// serve-side callers can use this one helper for both entry points.
    #[must_use]
    pub fn tree_reservation_rows(&self, prefix_len: usize, m: usize) -> usize {
        // Mirrors `tree_forward`'s bucket predicate exactly (batch slots
        // share every condition; region_ctx == self.max_ctx by the geometry
        // guard).
        let bucket = if self.head_dim.is_multiple_of(4)
            && m <= TREE_BUCKET_MAX
            && self.max_ctx * 4 <= 48 * 1024
            && self.kv_elem == 4
            && std::env::var_os("TRITIUM_TREE_EAGER").is_none()
        {
            TREE_BUCKETS
                .iter()
                .copied()
                .find(|&b| b >= m && prefix_len + b <= self.max_ctx)
        } else {
            None
        };
        prefix_len + bucket.unwrap_or(m)
    }

    /// **I4 batched-slots greedy tree verify** (L3 batch-slot spec decode,
    /// the payoff rung): verify SEVERAL slots' draft trees in ONE tree
    /// forward. The forward's row set is the concatenation of every listed
    /// slot's tree rows (`m_total = Σ mᵢ`, padded to one graph bucket); each
    /// row attends ITS slot's committed prefix plus ITS ancestors within its
    /// own tree, provisional K/V land in ITS slot's region, and the LM head
    /// and argmax run ONCE over all `m_total` rows — amortizing the ~657 MB
    /// f16 lm_head table read (44.5% of spec-loop GPU, ADR 0032) N-wide
    /// instead of paying it per slot. Per-slot greedy accept walks, promotes
    /// and position advances then run host-side exactly as in
    /// [`Self::tree_verify_greedy_slot`]; `out[k]` is slot `rows[k]`'s
    /// accepted tokens (identical, bit-for-bit, to verifying the slots
    /// sequentially — gated by `cuda_tree_verify_slots_matches_sequential`).
    ///
    /// `trees[k] = (tokens, parents)` is slot `rows[k]`'s tree in the
    /// [`Self::tree_verify_greedy`] shape (root at node 0 = the slot's last
    /// committed token, topological parents). Chains of different lengths
    /// and branchy trees mix freely; an early accept stop in one slot (e.g.
    /// its drafts reject at the root) never affects the others.
    ///
    /// Requirements: the f32 KV rung, `head_dim % 4 == 0` (the slots twins
    /// are float4 kernels — dense AND paged), a batch from this model's
    /// `new_batch`/`new_batch_paged`, live non-duplicate rows, and
    /// `m_total <= 48` (`TREE_BUCKET_MAX` — the batched forward pads to ONE
    /// bucket; split larger verify sets across calls). PAGED slots must be
    /// reserved through `positions[r] + tokensᵣ.len()` tokens (batched pads
    /// write nothing, so — unlike the single-slot verify — no padded
    /// reservation is needed; [`Self::tree_reservation_rows`] remains a safe
    /// upper bound covering both entry points).
    ///
    /// # Errors
    /// [`BackendError::InvalidInput`] on any bad target or tree — checked
    /// UP FRONT, before any device work, so a refusal leaves every listed
    /// slot's position/KV/pages untouched; device errors otherwise.
    pub fn tree_verify_greedy_slots(
        &mut self,
        batch: &mut BatchKv,
        rows: &[usize],
        trees: &[(&[u32], &[i32])],
    ) -> Result<Vec<Vec<u32>>, BackendError> {
        let (m_total, on_cap) = self.tree_forward_slots(batch, rows, trees)?;
        // Consume on the stream that produced the logits; the pageable ids
        // dtoh below is the ONE ordering point (drains the whole verify
        // before the promotes' default-stream dtods touch the arenas) — the
        // single-slot readback shape.
        let s = if on_cap {
            &self.cap_stream
        } else {
            &self.stream
        };
        let mut ts = batch
            .tree_scratch
            .take()
            .expect("tree scratch after forward");
        Self::bl_argmax_rows_chunked(
            s,
            &self.f_argmax_partial,
            &self.f_argmax_combine,
            &ts.d_logits_all,
            self.vocab,
            m_total,
            &mut ts.d_amax_val,
            &mut ts.d_amax_idx,
            &mut ts.d_ids,
        )?;
        let mut ids = vec![0i32; m_total];
        let ids_view = ts.d_ids.slice(0..m_total);
        s.memcpy_dtoh(&ids_view, &mut ids)
            .map_err(|e| driver_err("tree slots ids dtoh", &e))?;
        batch.tree_scratch = Some(ts);

        // Per-slot accept walks over the CONCATENATED ids buffer: slot k's
        // rows are ids[row_start .. row_start + mᵢ] (local node i ↦ global
        // row row_start + i — the one strided mapping in the whole path,
        // kept explicit here). Walk + promote are per-slot independent;
        // regions are disjoint, so the sequential promotes can't interact.
        let mut outs: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
        let mut row_start = 0usize;
        for (&row, &(tokens, parents)) in rows.iter().zip(trees) {
            let m_i = tokens.len();
            let ids_r = &ids[row_start..row_start + m_i];
            let mut path = vec![0usize];
            loop {
                let cur = *path.last().expect("path non-empty");
                let want = ids_r[cur];
                let next = (cur + 1..m_i)
                    .find(|&c| parents[c] as usize == cur && tokens[c] as i32 == want);
                match next {
                    Some(c) => path.push(c),
                    None => break,
                }
            }
            self.tree_promote_in(&path, Some((batch, row)))?;
            outs.push(path.iter().map(|&n| ids_r[n] as u32).collect());
            row_start += m_i;
        }
        Ok(outs)
    }

    /// The batched-slots FORWARD half: validate every target + tree UP
    /// FRONT (zero state change on refusal), build the concatenated per-row
    /// host tables, run ONE trunk over `mb` rows (graph replay of the slots
    /// twins, or the eager slots launches) and leave all `m_total` real
    /// rows' logits in `batch.tree_scratch.d_logits_all`. Returns
    /// `(m_total, on_cap)` with `tree_forward`'s stream-ordering contract.
    ///
    /// Per-ROW ctrl layout (the I4 ctrl plane, `[mb, 3]` i32 row-major,
    /// uploaded into the batch's `tree_slots_graphs.d_ctrl`):
    /// word 0 = the row's slot prefix_len; word 1 = the row's LOCAL node
    /// index in its slot's tree (−1 = pad row: skips attention AND writes no
    /// KV — batched pads belong to no slot); word 2 = the slot's dense KV
    /// row base (`r · max_ctx`) or paged table offset (`r · tstride`).
    /// Kernels read their own 3 words — no slot loop anywhere on device.
    fn tree_forward_slots(
        &mut self,
        batch: &mut BatchKv,
        rows: &[usize],
        trees: &[(&[u32], &[i32])],
    ) -> Result<(usize, bool), BackendError> {
        // ── Guard phase: EVERYTHING host-checked before any device work, so
        // a refusal is atomic across all listed slots. ──
        if rows.is_empty() || rows.len() != trees.len() {
            return Err(BackendError::InvalidInput(format!(
                "tree_verify_slots: rows ({}) and trees ({}) must be non-empty and equal-length",
                rows.len(),
                trees.len()
            )));
        }
        if self.kv_elem != 4 {
            return Err(BackendError::InvalidInput(
                "tree_verify_slots requires the f32 KV rung (batch arenas are \
                 f32); unset TRITIUM_KV"
                    .into(),
            ));
        }
        if !self.head_dim.is_multiple_of(4) {
            return Err(BackendError::InvalidInput(format!(
                "tree_verify_slots requires head_dim % 4 == 0 (the batched-slots \
                 twins are float4 kernels), got head_dim {}",
                self.head_dim
            )));
        }
        if batch.max_ctx != self.max_ctx || batch.kv_k.len() != self.layers.len() {
            return Err(BackendError::InvalidInput(format!(
                "tree_verify_slots: batch geometry (max_ctx {}, {} layers) does \
                 not match this model (max_ctx {}, {} layers) — the BatchKv must \
                 come from this model's new_batch",
                batch.max_ctx,
                batch.kv_k.len(),
                self.max_ctx,
                self.layers.len()
            )));
        }
        let mut seen = vec![false; batch.n];
        for (&r, &(tokens, parents)) in rows.iter().zip(trees) {
            if r >= batch.n {
                return Err(BackendError::InvalidInput(format!(
                    "tree_verify_slots: row {r} >= batch n {}",
                    batch.n
                )));
            }
            if !batch.live[r] {
                return Err(BackendError::InvalidInput(format!(
                    "tree_verify_slots: row {r} is dead"
                )));
            }
            if seen[r] {
                return Err(BackendError::InvalidInput(format!(
                    "tree_verify_slots: row {r} listed twice"
                )));
            }
            seen[r] = true;
            let m_i = tokens.len();
            if m_i == 0 || parents.len() != m_i {
                return Err(BackendError::InvalidInput(format!(
                    "tree_verify_slots: row {r}: empty or mismatched parents"
                )));
            }
            if parents[0] != -1 {
                return Err(BackendError::InvalidInput(format!(
                    "tree_verify_slots: row {r}: node 0 must be the root (parent -1)"
                )));
            }
            for (i, &p) in parents.iter().enumerate().skip(1) {
                if p < 0 || p as usize >= i {
                    return Err(BackendError::InvalidInput(format!(
                        "tree_verify_slots: row {r}: parents[{i}]={p} is not \
                         topological (0 <= parent < i)"
                    )));
                }
            }
            for &t in tokens {
                if t as usize >= self.vocab {
                    return Err(BackendError::InvalidInput(format!(
                        "tree_verify_slots: row {r}: token {t} out of range"
                    )));
                }
            }
            let prefix = batch.positions[r];
            if prefix + m_i > batch.max_ctx {
                return Err(BackendError::InvalidInput(format!(
                    "tree_verify_slots: row {r} overflow: prefix {prefix} + {m_i} \
                     nodes > max_ctx {}",
                    batch.max_ctx
                )));
            }
            // Paged: every REAL row this verify writes — [prefix, prefix +
            // mᵢ); batched pads write nothing — must sit on a reserved page
            // (reservation is prefix-contiguous, so the last row's page
            // implies the rest).
            if batch.pages.is_some() && !batch.page_mapped(r, prefix + m_i - 1) {
                return Err(BackendError::InvalidInput(format!(
                    "tree_verify_slots: row {r} rows {prefix}..{} not \
                     page-reserved (reserve_pages for the prefix + the tree \
                     before verifying)",
                    prefix + m_i
                )));
            }
        }
        let m_total: usize = trees.iter().map(|&(t, _)| t.len()).sum();
        if m_total > TREE_BUCKET_MAX {
            return Err(BackendError::InvalidInput(format!(
                "tree_verify_slots: {m_total} total tree nodes > the one-bucket \
                 cap {TREE_BUCKET_MAX} — split the verify across calls (v1 caps \
                 the batch rather than growing TREE_BUCKETS, which would change \
                 capture VRAM)"
            )));
        }

        // Route: pad m_total to the smallest bucket and replay ONE captured
        // slots graph. head_dim % 4 and kv_elem == 4 are guaranteed above;
        // the remaining conditions mirror `tree_forward` (a bucket always
        // exists — m_total <= TREE_BUCKET_MAX — and pads write nothing, so
        // there is no region-fit condition on the padded size).
        let bucket = if self.max_ctx * 4 <= 48 * 1024
            && std::env::var_os("TRITIUM_TREE_EAGER").is_none()
        {
            TREE_BUCKETS.iter().copied().find(|&b| b >= m_total)
        } else {
            None
        };
        if bucket.is_none() {
            static EAGER_WARN: std::sync::Once = std::sync::Once::new();
            EAGER_WARN.call_once(|| {
                eprintln!(
                    "tritium-cuda: batched-slots tree verify falling back to the \
                     EAGER route (m_total={m_total}, max_ctx={}) — ~600 \
                     launches/verify",
                    self.max_ctx
                );
            });
        }
        let mb = bucket.unwrap_or(m_total);

        // ── Concatenated per-row host tables. Ancestor entries stay
        // region-LOGICAL within each row's OWN slot (prefixᵣ + local node
        // index), exactly as in the single-slot path; the per-row ctrl
        // carries the slot attribution. ──
        let mut anc: Vec<i32> = vec![0; mb * mb]; // [mb, max_anc=mb], row-major
        let mut n_anc = vec![0i32; mb];
        let mut tok_i = vec![0i32; mb];
        let mut positions = vec![0i32; mb];
        let mut row_ctrl = vec![0i32; mb * 3];
        let mut ctx_bound = 0usize; // max over rows of (prefix + n_anc) — eager grid/smem
        let mut row_start = 0usize;
        for (&r, &(tokens, parents)) in rows.iter().zip(trees) {
            let prefix = batch.positions[r];
            let word2 = match batch.pages.as_ref() {
                Some(pg) => r * pg.tstride,
                None => r * batch.max_ctx,
            };
            debug_assert!(word2 <= i32::MAX as usize, "slots ctrl word 2 overflows i32");
            let m_i = tokens.len();
            let mut depth = vec![0usize; m_i];
            for i in 0..m_i {
                let g = row_start + i;
                if parents[i] >= 0 {
                    let p = parents[i] as usize;
                    depth[i] = depth[p] + 1;
                    let (dst_off, src_off) = (g * mb, (row_start + p) * mb);
                    let np = n_anc[row_start + p] as usize;
                    // anc[g] = anc[parent] ++ [slot(i)] (rows disjoint: p < i).
                    anc.copy_within(src_off..src_off + np, dst_off);
                    anc[dst_off + np] = (prefix + i) as i32;
                    n_anc[g] = n_anc[row_start + p] + 1;
                } else {
                    anc[g * mb] = (prefix + i) as i32;
                    n_anc[g] = 1;
                }
                tok_i[g] = tokens[i] as i32;
                positions[g] = (prefix + depth[i]) as i32;
                row_ctrl[g * 3] = prefix as i32;
                row_ctrl[g * 3 + 1] = i as i32;
                row_ctrl[g * 3 + 2] = word2 as i32;
            }
            ctx_bound = ctx_bound.max(prefix + m_i);
            row_start += m_i;
        }
        for g in m_total..mb {
            // Pad rows: valid token/position for the row-independent trunk
            // math (embed/rope read them), but ctrl word 1 = -1 — attention
            // early-exits and kv_append skips, so a pad touches NO slot's
            // region (unlike single-slot pads, which park junk past their
            // one slot's watermark).
            tok_i[g] = tok_i[0];
            positions[g] = positions[0];
            n_anc[g] = 0;
            row_ctrl[g * 3] = 0;
            row_ctrl[g * 3 + 1] = -1;
            row_ctrl[g * 3 + 2] = 0;
        }

        let s = &self.stream;
        let (n_embd, q_width, kv_width, n_ff) =
            (self.n_embd, self.q_width, self.kv_width, self.n_ff);
        let (n_head, n_head_kv, head_dim) = (self.n_head, self.n_head_kv, self.head_dim);

        // Scratch: the batch's own TreeScratch (shared with the single-slot
        // path — buffers are tree-shaped). A re-grow invalidates BOTH graph
        // sets (they bake these pointers).
        let m_cap_want = mb.max(TREE_BUCKET_MAX);
        let scratch_stale = batch
            .tree_scratch
            .as_ref()
            .is_none_or(|t| t.m_cap < m_cap_want);
        if scratch_stale {
            batch.tree_graphs = None;
            batch.tree_slots_graphs = None;
            let ts_new = self.alloc_tree_scratch(m_cap_want)?;
            batch.tree_scratch = Some(ts_new);
        }
        let mut ts = batch
            .tree_scratch
            .take()
            .expect("tree scratch just ensured");

        let up = if bucket.is_some() {
            &self.cap_stream
        } else {
            s
        };
        up.memcpy_htod(&tok_i, &mut ts.d_tok)
            .map_err(|e| driver_err("tree slots tokens htod", &e))?;
        up.memcpy_htod(&positions, &mut ts.d_pos)
            .map_err(|e| driver_err("tree slots positions htod", &e))?;
        up.memcpy_htod(&anc, &mut ts.d_anc)
            .map_err(|e| driver_err("tree slots anc htod", &e))?;
        up.memcpy_htod(&n_anc, &mut ts.d_nanc)
            .map_err(|e| driver_err("tree slots n_anc htod", &e))?;

        if let Some(bucket) = bucket {
            // ── Graph route: replay the captured SLOTS trunk, then the eager
            // tail (norm + LM head) at the REAL row count m_total. Same
            // pre-replay default-stream drain as `tree_forward` (the previous
            // verify's promote dtods race this replay's kv_append otherwise).
            self.stream
                .synchronize()
                .map_err(|e| driver_err("tree slots pre-replay default sync", &e))?;
            if self.batch_raw.is_none() {
                let ctx = self.cap_stream.context().clone();
                self.batch_raw = Some(Arc::new(BatchRawKernels::load(&ctx)?));
            }
            if batch.tree_slots_graphs.is_none() {
                let d_ctrl = self
                    .cap_stream
                    .alloc_zeros::<i32>(3 * TREE_BUCKET_MAX)
                    .map_err(|e| driver_err("tree slots ctrl alloc", &e))?;
                batch.tree_slots_graphs = Some(TreeGraphs {
                    d_ctrl,
                    graphs: HashMap::new(),
                    raw_keepalive: self.batch_raw.clone(),
                });
            }
            let have = batch
                .tree_slots_graphs
                .as_ref()
                .expect("slots graphs just ensured")
                .graphs
                .contains_key(&bucket);
            if !have {
                if std::env::var_os("TRITIUM_TREE_DEBUG").is_some() {
                    eprintln!("tree-slots-graph: capturing bucket {bucket}");
                }
                let cs = &self.cap_stream;
                let d_ctrl = dptr(
                    &batch
                        .tree_slots_graphs
                        .as_ref()
                        .expect("just ensured")
                        .d_ctrl,
                    cs,
                );
                let kv: Vec<(sys::CUdeviceptr, sys::CUdeviceptr)> = (0..self.layers.len())
                    .map(|li| (dptr(&batch.kv_k[li], cs), dptr(&batch.kv_v[li], cs)))
                    .collect();
                let d_table = batch.pages.as_ref().map(|pg| dptr(&pg.d_table, cs));
                let g = self.record_graph_tree(&ts, bucket, d_ctrl, &kv, d_table, true)?;
                let tg = batch
                    .tree_slots_graphs
                    .as_mut()
                    .expect("slots graphs just ensured");
                tg.graphs.insert(bucket, SendGraph(g));
                tg.raw_keepalive = self.batch_raw.clone();
            }
            // Per-replay data: page-table content (paged), then the per-row
            // ctrl plane; replay + tail all on cap_stream (stream order is
            // the only ordering needed).
            if let Some(pg) = batch.pages.as_mut() {
                self.cap_stream
                    .memcpy_htod(&pg.table, &mut pg.d_table)
                    .map_err(|e| driver_err("tree slots table htod", &e))?;
            }
            {
                let tg = batch
                    .tree_slots_graphs
                    .as_mut()
                    .expect("slots graphs just ensured");
                self.cap_stream
                    .memcpy_htod(&row_ctrl, &mut tg.d_ctrl)
                    .map_err(|e| driver_err("tree slots ctrl htod", &e))?;
                tg.graphs
                    .get(&bucket)
                    .expect("slots graph just inserted")
                    .launch()
                    .map_err(|e| driver_err("tree slots graph launch", &e))?;
            }
            let cs = &self.cap_stream;
            Self::bl_rmsnorm(
                cs,
                &self.f_rmsnorm_batch,
                &ts.d_x,
                &self.d_output_norm,
                self.rms_eps,
                n_embd,
                m_total,
                &mut ts.d_norm_all,
            )?;
            Self::bl_lm_head_tiled(
                cs,
                &self.f_lm_head_tiled,
                &ts.d_norm_all,
                &self.d_token_embd_f16,
                n_embd,
                self.vocab,
                m_total,
                &mut ts.d_logits_all,
            )?;
        } else {
            // ── Eager route: the SAME slots kernels launched per layer on
            // the default stream (the I3 eager-paged shape — no region views
            // can express per-row slot attribution, so ctrl-driven kernels
            // serve both dense and paged). mb == m_total here (no pads).
            if batch.tree_slots_graphs.is_none() {
                let d_ctrl = s
                    .alloc_zeros::<i32>(3 * TREE_BUCKET_MAX)
                    .map_err(|e| driver_err("tree slots ctrl alloc (eager)", &e))?;
                batch.tree_slots_graphs = Some(TreeGraphs {
                    d_ctrl,
                    graphs: HashMap::new(),
                    raw_keepalive: self.batch_raw.clone(),
                });
            }
            s.memcpy_htod(
                &row_ctrl,
                &mut batch
                    .tree_slots_graphs
                    .as_mut()
                    .expect("just ensured")
                    .d_ctrl,
            )
            .map_err(|e| driver_err("tree slots ctrl htod (eager)", &e))?;
            if let Some(pg) = batch.pages.as_mut() {
                s.memcpy_htod(&pg.table, &mut pg.d_table)
                    .map_err(|e| driver_err("tree slots table htod (eager)", &e))?;
            }
            let paged = batch.pages.is_some();
            let (f_kv, f_sc, f_rd) = if paged {
                (
                    &self.f_kv_append_tree_slots_paged,
                    &self.f_attn_tree_scores_slots_paged,
                    &self.f_attn_tree_reduce_slots_paged,
                )
            } else {
                (
                    &self.f_kv_append_tree_slots,
                    &self.f_attn_tree_scores_slots,
                    &self.f_attn_tree_reduce_slots,
                )
            };
            let m = m_total;
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
                    &self.f_tq1_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].q,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_q,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &self.f_tq1_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].k,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_k,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &self.f_tq1_tiled_scaled,
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
                {
                    let tg = batch.tree_slots_graphs.as_ref().expect("ensured above");
                    let table = batch.pages.as_ref().map(|pg| &pg.d_table);
                    Self::bl_kv_append_tree_slots(
                        s,
                        f_kv,
                        &ts.d_k,
                        &mut batch.kv_k[li],
                        &tg.d_ctrl,
                        table,
                        kv_width,
                        m,
                    )?;
                    let tg = batch.tree_slots_graphs.as_ref().expect("ensured above");
                    let table = batch.pages.as_ref().map(|pg| &pg.d_table);
                    Self::bl_kv_append_tree_slots(
                        s,
                        f_kv,
                        &ts.d_v,
                        &mut batch.kv_v[li],
                        &tg.d_ctrl,
                        table,
                        kv_width,
                        m,
                    )?;
                    let tg = batch.tree_slots_graphs.as_ref().expect("ensured above");
                    let table = batch.pages.as_ref().map(|pg| &pg.d_table);
                    Self::bl_attn_tree_split_slots(
                        s,
                        f_sc,
                        f_rd,
                        &ts.d_q,
                        &batch.kv_k[li],
                        &batch.kv_v[li],
                        &tg.d_ctrl,
                        table,
                        &mut ts.d_attn,
                        &mut ts.d_scores,
                        &ts.d_anc,
                        &ts.d_nanc,
                        self.max_ctx,
                        ctx_bound,
                        n_head,
                        n_head_kv,
                        head_dim,
                        self.attn_scale,
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
                    &self.f_tq1_tiled_scaled,
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
                    &self.f_tq1_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].gate,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_gate,
                )?;
                Self::bl_matmul(
                    s,
                    &self.f_tiled_scaled,
                    &self.f_tq1_tiled_scaled,
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
                    &self.f_tq1_tiled_scaled,
                    &ts.d_qact,
                    &self.layers[li].down,
                    &ts.d_act_scale,
                    m,
                    &mut ts.d_proj,
                )?;
                Self::bl_residual(s, &self.f_residual, &mut ts.d_x, &ts.d_proj, m * n_embd)?;
            }
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
        batch.tree_scratch = Some(ts);
        Ok((m_total, bucket.is_some()))
    }

    /// I4 eager launch: `kv_append_tree_slots[_paged]_g` — each REAL row's
    /// provisional K/V lands at ITS slot's region row (per-row ctrl); pad
    /// rows write nothing. `table = Some` selects the paged twin (`kv_base`
    /// is then the page POOL).
    #[allow(clippy::too_many_arguments)]
    fn bl_kv_append_tree_slots(
        s: &Arc<CudaStream>,
        f: &CudaFunction,
        src: &CudaSlice<f32>,
        kv_base: &mut CudaSlice<f32>,
        row_ctrl: &CudaSlice<i32>,
        table: Option<&CudaSlice<i32>>,
        kv_width: usize,
        m: usize,
    ) -> Result<(), BackendError> {
        let (kw_i, m_i) = (kv_width as i32, m as i32);
        let cfg = LaunchConfig {
            grid_dim: (((m * kv_width) as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f);
        l.arg(src).arg(kv_base).arg(row_ctrl);
        if let Some(t) = table {
            l.arg(t);
        }
        l.arg(&kw_i).arg(&m_i);
        // SAFETY: matches `kv_append_tree_slots_g(src, kv, row_ctrl,
        // kv_width, m)` / the `_paged` twin's extra `table` arg; only
        // `kv_base` mutable; the guard phase pinned every real row inside
        // its slot's region (dense) or onto a reserved page (paged).
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch tree slots kv_append", &e))?;
        }
        Ok(())
    }

    /// I4 eager split attention: the batched-slots twins (per-row ctrl in
    /// place of launch-wide scalars), dense or paged by `table`. Fold orders
    /// are the single-slot ctrl pair's, so values are bit-identical per row.
    /// `ctx_bound` = max over listed rows of (prefix + m_row): bounds the
    /// scores grid and the reduce's shared staging; each row guards by its
    /// OWN ctx = its prefix + its n_anc.
    #[allow(clippy::too_many_arguments)]
    fn bl_attn_tree_split_slots(
        s: &Arc<CudaStream>,
        f_scores: &CudaFunction,
        f_reduce: &CudaFunction,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        row_ctrl: &CudaSlice<i32>,
        table: Option<&CudaSlice<i32>>,
        out: &mut CudaSlice<f32>,
        scores: &mut CudaSlice<f32>,
        anc: &CudaSlice<i32>,
        n_anc: &CudaSlice<i32>,
        score_stride: usize,
        ctx_bound: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        scale: f32,
        m: usize,
    ) -> Result<(), BackendError> {
        // Keep in sync with `#define TREE_SCORE_CHUNK` in decode.cu.
        const TREE_SCORE_CHUNK: usize = 128;
        let (ss_i, nh_i, nhkv_i, hd_i) = (
            score_stride as i32,
            n_head as i32,
            n_head_kv as i32,
            head_dim as i32,
        );
        let (ma_i, m_i) = (m as i32, m as i32);
        let cfg = LaunchConfig {
            grid_dim: (
                (m * n_head) as u32,
                (ctx_bound.div_ceil(TREE_SCORE_CHUNK)) as u32,
                1,
            ),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(f_scores);
        l.arg(q).arg(k).arg(&mut *scores).arg(anc).arg(n_anc).arg(row_ctrl);
        if let Some(t) = table {
            l.arg(t);
        }
        l.arg(&ss_i)
            .arg(&nh_i)
            .arg(&nhkv_i)
            .arg(&hd_i)
            .arg(&scale)
            .arg(&ma_i)
            .arg(&m_i);
        // SAFETY: matches `gqa_attention_tree_scores_slots_g(q, k, scores,
        // anc, n_anc, row_ctrl, score_stride, n_head, n_head_kv, head_dim,
        // scale, max_anc, m)` / the `_paged` twin's extra `table`; max_anc ==
        // m (the eager ancestor table is [m, m]); only `scores` mutable.
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch tree slots attention scores", &e))?;
        }
        let cfg = LaunchConfig {
            grid_dim: ((m * n_head) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (ctx_bound * 4) as u32,
        };
        let mut l = s.launch_builder(f_reduce);
        l.arg(v).arg(&*scores).arg(out).arg(anc).arg(n_anc).arg(row_ctrl);
        if let Some(t) = table {
            l.arg(t);
        }
        l.arg(&ss_i)
            .arg(&nh_i)
            .arg(&nhkv_i)
            .arg(&hd_i)
            .arg(&ma_i)
            .arg(&m_i);
        // SAFETY: matches `gqa_attention_tree_reduce_slots_g(v, scores, out,
        // anc, n_anc, row_ctrl, score_stride, n_head, n_head_kv, head_dim,
        // max_anc, m)` / the `_paged` twin's extra `table` with ctx_bound·4 B
        // of dynamic shared (the handle's opt-in at load covers max_ctx·4).
        #[allow(unsafe_code)]
        unsafe {
            l.launch(cfg)
                .map_err(|e| driver_err("launch tree slots attention reduce", &e))?;
        }
        Ok(())
    }
}

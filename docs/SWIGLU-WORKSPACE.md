# Budgeted SwiGLU workspace reuse (experimental)

Proposed ADR0176 adds one capacity-only CUDA cache shared by typed-tensor and
packed-QTEA SwiGLU. It does not enable numerical execution, admit weights, measure
physical fit or establish speed. Existing profiles remain unchanged.

```rust,ignore
let device = TensorPayloadBudget::new(device_workspace_cap);
let host = TensorPayloadBudget::new(host_workspace_cap);
let resources = SwiGluWorkspaceResources::new(&device, &host)?;
let backend = CudaBackend::new(0)?.with_device_swiglu()
    .with_swiglu_workspace_resources(resources)?;
```

Configure before loading. Resources clone positive ledgers without reserving yet.
Clones coordinate; independent ledgers do not. Rust callers can explicitly share
a ledger with admitted weights/other models or use partitions. Raw uploads and
unrelated operations are not automatically charged. This is logical payload
admission, excluding allocator pools/overhead, driver/module/events, model caches
and process-wide RAM/VRAM. It is not a physical fit or qualification receipt.

For nonempty batch `m`, input/output width `k`, hidden width `f`, device footprints
are `4*(3*m*k + 2*m*f + max(k,f))+4` for shared tensors (rotation, scales, status)
and `4*(2*m*k + 2*m*f)+4` for QTEA. Host retention is actual output Vec capacity
bytes plus four status bytes. Shared construction additionally reserves temporary
unit scales. Expected requests are reserved before allocation; actual capacities
are checked/charged before GPU allocation. Temporary storage drops before its
reservation shrinks. Physical allocator over-allocation is not controlled.

The slot is keyed by kind and exact `(m,k,f)`, not weight values. Matching calls
reuse capacity only: all inputs/results are overwritten and status resets. A miss
evicts obsolete private scratch before reserving/allocating replacement, avoiding
an unaccounted old-plus-new logical peak. Later failure can leave an empty cache;
weights, graph state and caller output stay unchanged. Selected errors are not
fallback. Numeric refusals allow valid retries; driver/context errors do not imply
recovery/reset. Per-call QTEA caps still apply. Empty calls allocate/resize nothing;
older admitted retention may remain separately visible.

`SwiGluWorkspacePlan::new` is pure checked buffer arithmetic, not weight admission.
`swiglu_workspace_snapshot()` reports geometry, held reservations and saturating
allocation/selection diagnostics, not activations or performance/physical evidence.
Selections can include calls which later refuse. `release_swiglu_workspace()` and
final backend drop release owned capacity/guards, not models, services or device
contexts. CUDA/host allocator pools may retain physical storage.

Without resources, shared execution keeps its prior unadmitted shape cache and
transactional replacement policy; QTEA stays uncached/per-call. The new JSON
profiles are `tritium.qtea-cuda-swiglu-reuse-host-load.v1` and
`tritium.qtea-hybrid-cuda-swiglu-reuse-host-load.v1`, using the existing complete
CUDA request and `--backend cuda`. Workspace ledgers use
`cuda.max_device_staging_bytes` / `max_staging_bytes`, excluding matrix weights
and other backends. Hybrid still requires its preserved-tensor cap. The distinct
observation is `qtea-cuda-ordered-device-swiglu-reuse-host-orchestrated`.
Other operations/orchestration remain host-side and MTP remains Unverified.

`TensorStorageReservation::resize` grows atomically before caller storage growth
and shrinks only after storage is gone. Parent lifetime remains held even at zero;
failed growth leaves guard/counter unchanged. It does not allocate/free storage or
authorize deleting model state. Full residency and all independent empirical,
security and public-activation gates remain open.

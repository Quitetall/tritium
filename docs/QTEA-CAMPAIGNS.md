# Explicit QTEA file campaigns (experimental)

The Rust file runner, `tritium qtea` and `tritium.qtea` use one implementation.
They build **unqualified, residual-bearing QTEA candidates**, not pure SALT,
authenticated source artifacts or qualified SOTA models (proposed ADR0170).
Existing SALT conversion/refinement commands are unchanged.

## Inputs and scope

Supply a strict `tritium.qtea.file-campaign.v1` request with:

- `campaign` and `run`: explicit nonzero content identities;
- `source`: the complete existing `QuantPlan`, all actions `KeepDense` for
  observation only. These actions do not request dense output or claim fitting;
- `model_profile`: `standard` or `qwen35_hybrid`, with complete original inventory;
- `limits`: positive, separate package/metadata/model/widened/native/logical/spool/
  observation/curvature-file caps from `QteaCampaignLimits`;
- `owners`: every matrix in canonical name order. Each supplies a complete
  `TensorFitConfig`, explicit curvature, a fixed single-QTEA Experimental frontier
  profile without automatic selection/fallback, and a caller resource estimate.

Paths are separate: every original safetensors shard, exact named original assets
and one exact named curvature file per matrix. No path is discovered or downloaded.
Native curvature uses existing canonical SIG2/version2 bytes plus an expected
digest and decoded-value capacity. Cache curvature uses an existing `.trit`
calibration cache, expected cache ID, cache/collector/chunk limits. It is replayed
as Full/Matmul/input-only data, with zero collector damping; the explicit numerical
recipe controls fitting damping. Gather/grouped/Fisher inputs are refused, never
relabelled. Embedding native metric selection is explicit caller work.

Inspection reads all original bytes and each curvature input; it is not a free
metadata query. It computes exact work IDs/logical bounds and reconstructs registry
admission from the caller estimate, without fitting, writing a spool or publishing.
Estimates remain assertions, not measured RAM, authenticated machine inventory or
physical admission. Imported Gram/cache identity is not capture authentication,
dataset/teacher attestation, PSD/SPD/fit success or model-quality evidence.

## CLI

Repeat the asset and curvature flags for the **complete** request inventories:

```sh
timeout 120 tritium --evidence off qtea inspect \
  --request request.json --source-shard original.safetensors \
  --asset config.json=config.json --asset tokenizer.json=tokenizer.json \
  --curvature OWNER=curvature-file > selected.json

timeout 3600 tritium --evidence off qtea run \
  --request request.json --source-shard original.safetensors \
  --asset config.json=config.json --asset tokenizer.json=tokenizer.json \
  --curvature OWNER=curvature-file --inspection selected.json \
  --spool /mnt/4tb/tmp/tritium-example/private-spool --out new-candidate
```

These are command shapes, not a complete model recipe. Create the private spool's
parent first. The spool file and output directory must not already exist. JSON
input/output caps default to16MiB and are configurable; they are not RSS bounds.

## Python

```python
from tritium import qtea

files = dict(source_shards=shards, asset_paths=assets, curvature_paths=curvature)
selected = qtea.inspect(request, **files)
result = qtea.run(request, selected, **files,
                  spool_path=new_private_spool, output_dir=new_candidate_directory)
assert result["qualification"] is False
```

Python accepts request/selection dictionaries and path-like inputs. Native work
releases the GIL. Paths are made absolute before that release; there is no Python
numerical planner or copy of the fitting/publication policy.

## Live Torch/Hugging Face capture (opt-in CPU profile)

`tritium.torch.prepare_qtea_source` binds a caller's live CPU model to the exact
existing source files before calibration. It accepts a separate
`tritium.qtea.source-preparation.v1` request with `campaign`, `source`,
`model_profile` and `limits`; no placeholder run/cache identities are required.
Supply `source_shards`, `asset_paths` and a positive `max_live_tensor_bytes`.
The adapter does not load/download a model, save another full checkpoint, invent
SALT fitting actions or mutate parameters. It compares complete persistent state,
canonical aliases, shapes/dtypes and actual bytes through the shared Rust observer.
Unsupported devices/state/views/configurations refuse. Preparation and verification
read whole-source content; this is bounded state comparison, not cheap metadata
inspection or a lock against concurrent writers.

```python
from tritium.torch import prepare_qtea_source

source = prepare_qtea_source(model, source_request,
    source_shards=shards, asset_paths=assets, max_live_tensor_bytes=tensor_cap)
calibration = source.calibrate(windows, objective,
    native_curvature=explicit_native_metrics, objective_bytes=objective_recipe,
    cache_limits=cache_limits, max_total_payload_bytes=total_cache_cap)
selected = calibration.inspect(run_id, owner_options, work_dir=work,
    max_numeric_bytes=numeric_cap, max_batch_samples=chunk)
result = calibration.run(run_id, owner_options, selected,
    output_dir=new_candidate, work_dir=work,
    max_numeric_bytes=numeric_cap, max_batch_samples=chunk)
```

Every matrix must be covered either by exact, unmodified `nn.Linear` consumers or
`native_curvature[owner] = {"spec": native_spec, "path": canonical_gram_file}`.
Embedding/shared gather-Matmul owners require the latter explicitly. Native file
digest/shape/source are checked **before objective execution**. There is no default
embedding identity or scientifically qualified metric choice supplied here.

The existing cache producer records actual Linear inputs/token selection, guarded
by the existing owner-dispatch checks against functional/view/regularizer reads.
Actual finished cache IDs fill the subsequent file request. `owner_options` must
provide every matrix's explicit `recipe`, `profile` and caller `estimate`. The run
ID is supplied by the caller. The complete content request is selected, so changes
to operational request fields (including replay chunk caps) require reinspection,
even where the numerical work/encoded candidate remains identical.

Private cache packages/spools live only under the supplied existing work directory
and are removed after each inspection/run. Source files/live ownership and graph
options are revalidated; source/cleanup errors after commit are not rollback.
This is not GPU/out-of-core capture, arbitrary custom-module graph proof, source/
objective/data/teacher attestation, MTP coverage qualification, physical admission
or a large-model quality/performance result (proposed ADR0171). The existing generic
native calibration API continues to use the same capture loop and keeps its profile.

## Execution and publication

Execution reobserves source bytes, reloads curvature and reconstructs opaque
admissions through the registry. It checks the selected request/source/work/plan
identities and logical bounds. Received JSON is data, never deserialized admission
authority or independent approval. Changed source/assets/recipes/metrics/estimates
cannot reuse a stale selection. No widened original matrices are retained between
phases, and final source/package verification remains mandatory.

The shared no-clobber transaction produces `model.trit` and `candidate.json` with
`qualification=false`, not a sealed `evidence.jsonl`. Only `model.trit` is the commit
marker. Private spool cleanup runs after execution, including errors. Post-commit
cleanup/sync/report/stdout failures are not rollback: inspect the committed files.
Caller-owned immutable/exclusive source, spool-path and destination-parent custody
is required; adversarial concurrent writers and power-loss cleanup are not proved.

Cold CLI/serving requires the separate explicit host-QTEA loading feature/config.
Hybrid MTP remains Unverified. This software route does not qualify pretrained
quality, official source, physical memory/backends, latency, reproduction, security,
Stage7 or public release, and does not authorize an expensive model campaign.

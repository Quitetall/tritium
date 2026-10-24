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

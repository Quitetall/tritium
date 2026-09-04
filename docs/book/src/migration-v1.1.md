# Migrating from v1.0 to v1.1

Tritium v1.1 is a platform expansion over the frozen v1.0 infrastructure
milestone. The repository currently carries candidate version `1.1.0-rc.2`; it
is not yet an authorized or receipt-qualified public release. Use locally built
candidate artifacts until the [compatibility matrix](../../compatibility.md)
marks your exact target `qualified`.

## Compatibility at a glance

| Surface | v1.0 → v1.1 contract | Required action |
|---|---|---|
| Frozen Rust core | Source compatible across 1.x; checked against `v1.0.0` | Upgrade normally and run your tests |
| C ABI v1 | Header and ABI version unchanged | Relink and run the ABI smoke; no source migration |
| Evolving Rust crates | May break in a 1.x minor | Pin the minor and review its current rustdoc/API diff |
| Python distribution | Distribution renamed; import namespace retained | Replace the installed distribution, not imports |
| Python Torch API | New typed PTQ/QAT/refinement/ONNX/diagnostics facade | Adopt explicitly; do not mix latent and hard phases |
| SALT/ONNX artifacts | Current writers and compatibility gates are versioned | Keep originals; migrate only through strict loaders/exporters |
| Browser package | New in v1.1 | No v1.0 migration; qualify the npm archive independently |

## Rust: know which tier you use

The frozen tier remains:

- `tritium-core`
- `tritium-spec`
- `tritium-format`
- `tritium-runtime`
- `tritium-cpu`
- `tritium-quantize`
- `tritium-testkit`

These crates follow SemVer across the 1.x line. The release gate now selects
the latest reachable stable tag automatically; for this candidate it selects
`v1.0.0`:

```sh
./scripts/check-semver.sh --print-baseline
./scripts/check-semver.sh
```

The current candidate passes all seven comparisons with no required SemVer
update. CI must rerun that gate on the eventual release revision; this local
result is not a publication receipt.

The generated [v1.0 → v1.1 API diff](../../generated/api-diff-v1.0-v1.1.md)
also verifies that the Python root namespace retains `Model` and
`ternary_matmul` while listing the new v1.1 names. Regenerate or drift-check it
with:

```sh
python scripts/generate-api-diff.py
python scripts/generate-api-diff.py --check
```

That report is structural source evidence. The clean-wheel functional receipt
remains the authority for installed imports and runtime behavior.

The evolving tier includes `tritium-nn`, `tritium-train`, `tritium-cuda`, the
framework/ONNX interop crates, and `tritium-serve`. v1.1 adds substantial model,
portable-training, SALT V2, Qwen, and serving surfaces there. Those crates were
explicitly outside the v1.0 stable-core promise. Pin `1.1` (or the exact RC
while evaluating candidates), compile against the new API, and treat rustdoc
plus the source diff as authoritative. Do not infer evolving-tier compatibility
from the stable-core gate.

### C ABI

`TRITIUM_ABI_VERSION` remains `1`, and the v1.0 header/export surface is
unchanged. Existing C/C++ source should not need edits. Still rebuild or relink
against the candidate library and run the size-then-fill generation, last-error,
load-bytes, and null/error-path contract tests. ABI identity does not qualify a
backend or model artifact by itself.

```sh
cargo test -p tritium-ffi
```

Compile consumers against the public
[`tritium.h`](../../../crates/tritium-ffi/include/tritium.h), not declarations
copied into an application.

## Python: distribution rename, stable import

v1.0 used the distribution name `tritium`. v1.1 uses `pytritium` so the
published project name describes the integration and avoids claiming the
generic package name. The import namespace remains `tritium`.

Remove the old distribution before installing a candidate wheel; two
distributions owning the same import package are not a supported environment:

```sh
python -m pip show tritium
python -m pip uninstall -y tritium
TRITIUM_WHEEL=./dist/exact-digest-bound-pytritium-wheel.whl
python -m pip install "$TRITIUM_WHEEL"
python -c "import tritium; print(tritium.compiled_backends())"
```

The placeholder path above is not a release command. Set it to one digest-bound
wheel for the exact OS, architecture, Python, Torch, CUDA, and GPU cell. Check
`pip show` before uninstalling so an unrelated package that happens to own the
generic distribution name is not removed accidentally.

Normal v1.0 imports remain source-compatible:

```python
from tritium import Model, ternary_matmul

model = Model.load("model.gguf")
```

`ternary_matmul` gains an optional `device=` selector and still defaults to
CPU. The compiled extension now lives at `tritium._tritium` behind a
pure-Python package. Importing `_tritium` directly is internal and unsupported;
use the root namespace or the public subpackages.

## Adopt the differentiable facade by phase

v1.1 adds `tritium.nn`, `tritium.autograd`, and `tritium.torch`. Their types make
the lifecycle explicit:

```python
import torch
from tritium.torch import TernaryConfig, convert, prepare

prepared = prepare(
    torch.nn.Sequential(torch.nn.Linear(128, 64)),
    TernaryConfig.qat(estimator="salt-ste", planes=1),
    inplace=True,
)
model = prepared.model

# Use ordinary PyTorch optimizers while the graph owns latent floating masters.
loss = model(torch.randn(4, 128)).square().mean()
loss.backward()
torch.optim.AdamW(model.parameters(), lr=1e-4).step()

# Conversion consumes the trainable phase and returns inference-only hard state.
hard = convert(prepared)
```

Do not save a latent QAT checkpoint and label it compact: it contains floating
masters and estimator state. Do not pass an already-hard result back into QAT.
PTQ, QAT-hard, scale-only refinement, hard-PV refinement, and their typed ONNX
lineages have distinct manifests and ancestry.

For diagnostics, use bounded path selection and deliberate cadence. The default
collector refuses more than one million selected latent elements; external
estimators require an explicit purity opt-in. See
[Ternary observability](./observability.md).

## Artifact migration admission is copy-on-write

Never rewrite the only copy of a v1.0 model or sidecar. Keep the dense source,
tokenizer/config assets, original artifact, and their digests. The following is
the admission procedure for a future backward-read receipt, not a claim that
arbitrary v1.0 development artifacts are currently supported:

1. use the v1.1 strict loader for the artifact kind;
2. verify source, schema, tensor coverage, aliases, and preserved state;
3. export to a new directory through the typed v1.1 API;
4. strict-reload that directory in a fresh process;
5. compare generation/numerics and physical bytes before promotion.

For artifacts produced by the current v1.1 candidate, use only these public
entry points. Every loader validates the manifest and payload before returning;
exporters publish to a new directory and refuse an existing destination.

| Admitted v1.1 artifact | Strict load | Copy/export |
|---|---|---|
| Full Qwen PTQ bundle | `load(path)` | `export(result, new_path)` |
| Generic module PTQ | `load_module_conversion(path)`; bind with `load_quantized_module(model, artifact)` | Preserve the sealed conversion directory; no general republisher is promised |
| QAT-hard module | `load_qat_hard(path)`; optionally pass the source model shell | `export_qat_hard(result, new_path)` from the in-memory conversion result |
| Refinement result | `load_refinement(path)` | `export_refinement(result, new_path)` |
| Typed hard ONNX bundle | `load_onnx(path)` | `export_onnx(typed_result, new_path, ...)` |

QAT-hard artifacts use `tritium.module-qat-hard-v2`. Development-only v1
bundles did not bind complete per-consumer module semantics and are rejected;
re-run hard conversion from the retained latent checkpoint and export a new v2
directory. V2 also binds every canonical state tensor and all exact aliases,
including persistent buffers. Relabeling or rehashing a v1 manifest is not
migration.

For example, republish a strictly admitted current Qwen PTQ bundle without
mutating its source:

```python
from tritium.torch import export, load

admitted = load("./source-bundle")
receipt = export(admitted, "./migrated-bundle")  # destination must not exist
reopened = load(receipt.artifact_dir)
assert reopened.completion_id == admitted.completion_id
```

Current SALT schema-v3 writes, SALT V2 packages, and Tritium ONNX manifest-v2
writes are not a blanket promise that every development artifact is readable.
The generated compatibility matrix remains `pending` until backward-read and
clean-install receipts are admitted. An unsupported or unknown schema must fail
closed; manual JSON edits or filename changes are not migration.

Trainable whole-model ONNX import is explicitly deferred beyond v1.1. v1.1 ONNX
support is for typed hard PTQ/QAT/refinement inference artifacts. Exporting one
of those lineages does not recreate a trainable floating master.

## Deprecation and support windows

Stable APIs receive at least one stable minor release of deprecation notice
before removal unless an urgent security break is required. The evolving tier
does not inherit that promise; pin its minor version. Artifact readers document
their backward-read window, and new writes use only the current schema.

The latest stable 1.x minor receives correctness and security fixes. The
immediately previous stable 1.x minor receives critical/high security fixes for
90 days after the next stable minor. Release candidates receive development
fixes but may change before `1.1.0`. See [Support and version policy](../../../SUPPORT.md).

## Migration checklist

- Remove the old Python distribution before installing `pytritium`.
- Verify one exact candidate artifact digest and candidate source revision.
- Run the stable Rust SemVer gate against `v1.0.0` if consuming frozen crates.
- Recompile and review diffs for every evolving-tier Rust crate you consume.
- Keep C ABI v1 smoke tests even though the header is unchanged.
- Preserve original model/artifact bytes; migrate into a new directory.
- Separate latent, PTQ, QAT-hard, refined, and ONNX artifact types.
- Re-measure complete artifact/resident/peak bytes; do not carry forward a
  logical `1.58-bit` compression claim.
- Consult the generated compatibility matrix; `pending` is not support.
- Do not publish packages, models, tags, or claims from a local candidate run.

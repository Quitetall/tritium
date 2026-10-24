# tritium-nn

Model runner: transformer inference over packed ternary weights (BitNet, Qwen, SmolLM2 families).

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

## Packed host QTEA projections (experimental)

Enable `host-qtea` explicitly; the default normal dependency graph does not link
the host CPU backend for this path. This feature is not a browser backend.
`HostQteaLinear::read_from` reads one strict bounded QTEA tensor with its affine
centers, column multipliers and sparse FP8 corrections intact. Wrap it in
`Projection::HostQtea(Arc::new(matrix))` when explicitly constructing model
weights. Existing attention/MLP/head execution then consumes the packed bytes
without retaining a decoded dense weight matrix. `from_tensor` instead moves
already checked storage without a copy. These are proposed ADR 0162 interfaces.

This is explicitly scalar **host** execution with F32 activations, not a native
backend upload, GPU decoder, pure SALT representation or pretrained model loader.
Inspectable-master training and requested-resident graph compilation refuse this
projection. Embedded digests do not authenticate its original source or owner.

Each matrix requires an explicit per-call staging-byte ceiling. `forward` checks
batch geometry and allocates one fallible output-sized F32 staging buffer;
`forward_into` uses exact caller-owned staging without a heap request. Failures
leave output unchanged (staging may be dirty). Encoded limits, staging limits,
caller/cache/model memory and physical RSS are separate domains. Limits do not
aggregate across calls or concurrent matrices. Packed token gather retains the
same shared QTEA bytes for tied-head execution; `is_packed_salt()` remains false.

The separate proposed ADR 0163 `tritium.qtea-model.v1` carrier uses the existing
seek-backed `.trit` envelope with an explicit QTEA directory kind. It binds a
declared source, config/assets and complete named matrix/vector inventory.
`ModelWeights::load_qtea` returns config, architecture variation, owned packed
weights and the exact PackageId. With `tokenizer`, `Runner::load_qtea` loads this
standard-transformer carrier into the common host executor under explicit
`HostQteaModelLimits` and `StreamLimits`; `qtea_package_id()` retains its exact
byte identity. This method does not select or interpret tokenizer/chat assets.

With both features, `LoadedModel::load_qtea(source, expected_source, limits,
stream, context)` returns the existing text-model interface instead. It reads
both tokenizer assets from the same retained package, checks their encoded size
and complete vocabulary IDs before materializing weights, and uses the existing
source-template renderer and common runner. Missing, invalid or ambiguous assets
refuse; no family-based tokenizer/template or dense/GPU fallback is introduced.
Only `MODEL_LOADED_REGISTRATION` is required by this explicit host path. Its
observation records `LoadedSourceIdentity::Qtea` with exact PackageId and a clearly
declared source digest, never an invented semantic ModelId/SALT level or tensor
upload. Older strict observation readers reject this additive proposed ADR 0164
source tag. Existing source variants retain their V1 serialization.

All matrices, including embedding and untied head, stay packed. Only preserved
F16/BF16/F32 vectors widen to F32. Complete inventory/roles/shapes, source agreement
and logical payload/cache caps are admitted before weight materialization;
actual payload checks and terminal source verification precede successful return.
Identical immutable packed blobs share one allocation without inventing trainable
aliases. Config tying requires one embedding/head owner, not a second head tensor.
The payload ceiling excludes metadata, transient decode, allocator/Arc overhead,
staging, caches and physical RSS. Ordinary typed SALT loading rejects this profile.
Official-model Qwen/hybrid/MTP capture, source authentication, pretrained coherence,
production/UI integration, GPU/browser and performance remain unqualified;
toy text-model tests do not pass those gates. Initial load recording/replay is
not independent source, memory, quality or release qualification.

### Explicit QTEA command request (experimental)

With `host-qtea,tokenizer`, `QteaLoadConfig::read` and `load` share one request
contract between CLI generation and loopback serving (proposed ADR0165). JSON
must be a regular file of at most16KiB. Unknown/duplicate fields, invalid profile,
relative paths, zero digest/caps or invalid stream limits refuse before loading.
The actual model is opened once as a regular file; no directory discovery,
SALT level selection, GPU upload or dense fallback is inferred.

Every field is mandatory. This is an illustrative tiny-fixture configuration,
not capacity guidance or an official-model identity. Replace the absolute path,
declared source digest and caps for the actual artifact:

```json
{
  "profile": "tritium.qtea-host-load.v1",
  "path": "/absolute/path/to/model.trit",
  "expected_source_model": [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
  "max_source_bytes": 1048576,
  "max_manifest_bytes": 65536,
  "max_directory_bytes": 65536,
  "max_tensors": 64,
  "max_assets": 8,
  "max_asset_bytes": 65536,
  "max_config_bytes": 4096,
  "max_vector_elements": 32,
  "max_tensor_bytes": 1048576,
  "max_validation_bytes": 65536,
  "max_weight_bytes": 1048576,
  "max_staging_bytes": 65536,
  "stream_limits": {
    "max_context": 8,
    "max_segment_tokens": 3,
    "max_host_cache_payload_bytes": 4096
  }
}
```

Logical weights, per-call staging, source/metadata and host cache caps are separate
from tokenizer/allocator overhead and physical RSS. Source agreement is not
authentication. `HfJsonTokenizer::has_source_chat_template()` reports parsed
template presence only, not whether every message/control combination renders.

### Explicit hybrid language/MTP profile (experimental)

The separate proposed ADR0166 `tritium.qtea-hybrid-model.v1` profile carries the
complete canonical Qwen-family language/MTP inventory. Its `Preserved` role
retains exact rank/dimensions for non-matrix state such as convolution; it is not
a flattened norm vector or permission for dense matrix fallback. All schema
matrices, including embedding/head/MTP projections, must be QTEA. Higher-rank
preserved state has its own mandatory element cap. F16/BF16/F32 state alone widens
to finite F32; matrix bytes remain packed and share immutable storage.

`QteaModelReader::open_hybrid` and `encode_qtea_hybrid_model_manifest` select this
profile explicitly. Standard readers/encoders still reject it and the new role;
old standard-profile bytes and vector limits are unchanged. Exhaustive matches
on the pre-release Rust `QteaModelTensor` enum must now handle or reject
`Preserved`. No standard-artifact migration/reinterpretation is implied.

With `host-qtea,tokenizer`, `LoadedModel::load_qtea_hybrid` takes
`HostQteaHybridModelLimits { base, max_preserved_tensor_elements }`. It checks
actual checkpoint configuration and exact language/MTP names, roles and original
shapes, then reuses the existing Qwen assembler and common host runner. Real
tokenizer/cache admission precedes weights; final source verification precedes
the load observation. Unsupported MTP geometry and vision tensors do not become
an inferred partial model. MTP is retained as `Unverified`, not executable draft
authority or a quality claim.

CLI and serving use the same `--qtea` selector and base JSON above, but explicitly
set `profile` to `tritium.qtea-hybrid-host-load.v1` and add a positive
`max_preserved_tensor_elements`. The standard load profile rejects this field;
hybrid omission, null and zero refuse. Existing CPU/loopback/template/no-fallback
restrictions remain. Tiny native-fit DeltaNet/attention graph tests do not prove
official pretrained Qwen quality, physical memory, performance or release status.

## Source-bound chat prompts (experimental)

With `tokenizer`, `HfJsonTokenizer` retains the embedded `chat_template` from the
loaded tokenizer configuration. `Tokenizer::render_chat_prompt` renders text-role
messages plus a generation prompt without adding BOS/EOS again. It returns
`None` only when there is no source template. Declared invalid, ambiguous or
unsupported templates return an error; serving must not silently replace them
with an architecture-name approximation. Named map/list configurations require
an unambiguous `default`; standalone unbound template files are not loaded.

The fixed MiniJinja 3.0 guest is interpreted by Wasmi 2.0 without imports. Each
compile/render gets fresh state, 32 MiB maximum linear memory, 50 million Wasm
fuel, bounded stack/tables and a four-request nonblocking concurrency limit.
Templates are at most 32 KiB, configurations/input/output at most 1 MiB, and
conversations at most 256 messages. These are experimental policy limits, not
whole-process RSS limits, guaranteed wall deadlines or security qualification.
Only a fixed packaged guest is accepted; model-provided Wasm is never loaded.

This text-generation interface does not yet expose tools/documents, multimodal
content, thinking overrides, assistant continuation or training generation masks.
Unknown functions/extensions and unsupported `tojson` options fail closed. It is
not a claim of universal Hugging Face compatibility or model quality.

Reproduce the checked-in payload with `bash scripts/build-chat-renderer.sh --check`
from the repository root (pinned Rust toolchain and installed Wasm target).
`--write` deliberately regenerates it after source changes. Normal crate/wheel
builds consume the packaged bytes and do not invoke a guest build or Python.
Regeneration normalizes installed and virtual standard-library source paths to
the compiler's commit-bound `/rustc` paths, so installing `rust-src` cannot change
the payload. The build-record check rejects embedded host home/toolchain paths.
The `source_chat_prompt` example and `scripts/verify-chat-template.py` compare
exact text and token IDs offline using tokenizer assets only, not model weights.

## Experimental admitted sources

`ModelSource` selects an HF directory, ordinary typed `.trit` bytes with an
explicit complete level, or an already admitted joined salience model:

```rust,ignore
use tritium_nn::ModelSource;

// `joined` was opened by JoinedSalienceModel::open against the actual B stage
// and complete transform/fit logs, with explicit package/manifest limits.
let source = ModelSource::JoinedSalience {
    model: &joined,
    level: "standard",
};
```

Pass the selector to `ModelRunner::from_source_admitted` with a backend, tensor
upload policy, payload budget and evidence context. With the `tokenizer` feature,
`tritium_nn::load` additionally loads the artifact's actual tokenizer and records
the loaded C package/selected semantic model identity. Both use the same config,
complete tensor-inventory and backend-upload implementation. Ordinary `.trit`
loading rejects the mandatory joined carrier; it cannot silently ignore lineage.

Raw additive matrices may use either Uniform counts or the proposed ADR 0066
PerTile compact allocation, including pruned tiles and row-aligned ragged tails.
Both use the typed format decoder and the same checked/logged tensor uploader;
the selected backend must explicitly support the declared law, basis and codec.
Packed CPU and the production reference own their uploaded data, so the runner
can outlive the borrowed source bytes. `native_only` and lifetime-bound shared
payload admission remain enforced. Host norm/bias vectors, source bytes, transient
decode/index capacity, KV and scratch are outside that payload ledger. This is
not full RSS admission, profile freeze, physical speed or pretrained-model quality
qualification. Sharded/entropy payloads and resident hybrid/MTP require additional
reader/executor support rather than a silent fallback.

The admitted joined view borrows C, not the dense B stage. B and its logs may be
released after joined admission; the resulting runner owns uploaded handles and
may outlive C. Structural lineage admission is not actual A-to-B numerical replay,
activation-capture, quality or release qualification. These proposed ADR 0054/0064
interfaces currently cover the standard host-orchestrated transformer path, not
resident hybrid/MTP execution. Payload limits are not whole-process memory limits.

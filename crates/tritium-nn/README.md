# tritium-nn

Model runner: transformer inference over packed ternary weights (BitNet, Qwen, SmolLM2 families).

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

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

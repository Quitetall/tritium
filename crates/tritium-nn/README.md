# tritium-nn

Model runner: transformer inference over packed ternary weights (BitNet, Qwen, SmolLM2 families).

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

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

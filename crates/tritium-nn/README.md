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

The admitted joined view borrows C, not the dense B stage. B and its logs may be
released after joined admission; the resulting runner owns uploaded handles and
may outlive C. Structural lineage admission is not actual A-to-B numerical replay,
activation-capture, quality or release qualification. These proposed ADR 0054/0064
interfaces currently cover the standard host-orchestrated transformer path, not
resident hybrid/MTP execution. Payload limits are not whole-process memory limits.

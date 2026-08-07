# tritium-py

PyO3 bindings for the tritium-torch Python package.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

Stage-7 campaign executors consume frozen token evidence through
`tritium.torch.Stage7CausalData.open`. It strictly binds the campaign pack and
model-tokenizer identities, reads only the selected sequence window, performs a
terminal same-handle payload check, and yields replayable PyTorch causal-LM
batches. A data receipt is input provenance only, not an execution or quality
result.

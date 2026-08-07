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

`tritium.torch.run_stage7_smoke_model(model, data, output_dir)` consumes that
validated data and transactionally executes or resumes the five concrete smoke
stages: capture, fit, allocate, package, and evaluate. Its result binds source,
tokens, conversion, allocation, package bytes, and measured causal loss. It is
an engineering receipt, not a release-qualified claim.

`tritium.torch.run_stage7_smollm2_smoke(campaign_path, model_dir, output_dir)`
is the frozen campaign wrapper. It admits only the pinned SmolLM2-135M revision,
exact tokenizer/files, source-derived rank-2 inventory, and first 128 C4 members
from the governed evidence pack, then emits the schemas consumed by the Stage-7
qualifier. The wrapper uses local files only and fails closed on unknown resume
artifacts. G64-shaped matrices are stored using explicit SALT V2 package-version
2 scale geometry; G128-only packages retain byte-identical version 1 encoding.

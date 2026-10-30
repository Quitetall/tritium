# tritium-format

Ternary weight formats: TQ1_0/TQ2_0/Q2_0 pack-unpack, GGUF and safetensors readers, SALT packages.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

`import_salt_rows` preserves legacy TQ2_0 logical trits and independent FP16
plane/block scale bits in the shared `AdditiveTensor`. It uses an explicit
verified-container basis, admitted free-F16 G32/G128 layout and logical limits.
It does not refit or reinterpret a stored ladder as the tied-anchor law. This is
a tensor migration prerequisite (proposed ADR0065), not a model migrator,
legacy identity verifier, Direct-fit proof or empirical qualification. Complete
model import/provenance, variable/pruned rows, higher plane/group admission,
legacy rotation tails and both historical campaign migrations remain required.

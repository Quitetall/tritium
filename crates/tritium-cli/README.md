# tritium-cli

Tritium command-line interface: quantize, report, repack, and seekable
transport (installs the `tritium` binary).

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

CLI evidence output defaults to `summary` on stderr. Use `--evidence off` to
disable, `--evidence-out <PATH>` to write a new immutable JSONL file, and
`--evidence det --run-id <ID>` for deterministic logical-time events. Run
`tritium evidence verify <PATH>` to check canonical encoding, event digests,
span chains, and the run root. This verifies log integrity only, not empirical
or release qualification. Current CLI events record command name and outcome
only—not arguments or a semantic plan fingerprint. Use `--evidence-out` for a
clean JSONL file; stderr may include diagnostics. `det` replay covers command
lifecycle events only, not model execution determinism.

For storage or transfer, wrap an existing fixed-codec artifact without changing
runtime accounting:

```text
tritium transport pack model.salt model.salt.trns
tritium transport inspect model.salt.trns
tritium transport unpack model.salt.trns model.salt
```

`inspect` reports logical bytes separately from transport bytes. Logical bytes
remain resident-byte denominator; `TRNS` is never a serving format.

Serving is the separate `tritium-serve` binary (build with `--features tritium-serve/serve`, or `--features tritium-serve/cuda` for GPU serving).

# tritium-cli

Tritium command-line interface: quantize, report, serve, repack, and seekable
transport (installs the `tritium` binary).

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

For storage or transfer, wrap an existing fixed-codec artifact without changing
runtime accounting:

```text
tritium transport pack model.salt model.salt.trns
tritium transport inspect model.salt.trns
tritium transport unpack model.salt.trns model.salt
```

`inspect` reports logical bytes separately from transport bytes. Logical bytes
remain resident-byte denominator; `TRNS` is never a serving format.

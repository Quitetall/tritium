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

Explicit candidate typed-plan conversion uses the shared Rust engine:

```text
tritium convert --plan request.json --source-shard model.safetensors \
  --asset config.json=source/config.json \
  --asset tokenizer.json=source/tokenizer.json \
  --conversion-run-id conversion-01 --out new-output
tritium evidence verify new-output/evidence.jsonl
```

`request.json` is the generated Rust `QuantPlan` JSON representation, not a
CLI-owned recipe. For the distinct cache-bound `CalibratedQuantPlan`, use
`--calibrated-plan` instead and repeat `--calibration-cache TENSOR=cache.trit`
for its exact calibrated inventory. Repeat `--source-shard` and
`--asset NAME=PATH` for all original shards/assets. Explicit raw-Gram actions
require `--input-gram TENSOR=gram.bin` (exact row-major little-endian f64 bits).
Duplicate/missing/extra bindings, invalid plans or metrics and existing output
directories reject; there is no metric fallback. `--max-plan-bytes` defaults
to 16MiB. Resource checks are logical/file-byte bounds, not complete process RSS.

This mode writes `model.trit` and its mandatory sealed `evidence.jsonl`; stdout
reports actual published identities and post-commit warnings. That conversion
log uses `--conversion-run-id`, independently of the outer CLI summary's
`--run-id` and `--evidence` setting. Neither log proves recipe execution,
model quality, supported NN architecture, performance or release qualification.
Legacy `convert --model` keeps its current behavior; typed plans reject legacy
recipe flags rather than ignoring them. This additive mode is not completion
of paired salience-fold/Qwen/Python migration. Its package targets the shared
`.trit` loader; legacy `generate` is not newly migrated by this command.

An explicit original-captured stage can instead feed the calibrated plan:

```text
tritium convert --calibrated-plan request.json \
  --original-captured-stage stage.trit \
  --transform-evidence transform.jsonl \
  --original-salience-evidence original.jsonl \
  --capture-evidence capture.jsonl \
  --stage-limits stage-limits.json --conversion-limits conversion-limits.json \
  --calibration-cache TENSOR=cache.trit \
  --conversion-run-id conversion-02 --out new-output
```

Repeat `--calibration-cache` for exactly the plan's calibrated tensors. All six
stage/log/limits inputs are required together. This mode rejects raw shards,
assets, Grams, plain plans, legacy recipe flags and old stage profiles. Limits
files use the strict Rust `SalienceStageLimits` and `SalienceConversionLimits`
types; `--max-plan-bytes` bounds both the plan and each limits file.
`--max-evidence-bytes` defaults to a positive aggregate 16MiB ceiling across
the three logs. These are physical/logical file-byte checks, not a full RSS
bound or a streaming/mmap implementation.

The common engine writes C with mandatory original-A and B-capture ancestry.
It checks the supplied stage/log/cache agreement and runs the actual fitter;
it does not rerun A salience or B capture, authenticate logs, or qualify model
quality. Cold C admission still requires B and the mandatory logs; the output
does not copy B or private capture text into C. Preserve those custody inputs.
This command does not migrate legacy generation, retire the complete model-aware
conversion route, or establish a standalone public model's byte-reduction claim.

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

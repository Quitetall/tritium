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

`quantize` receives the same typed plans and filesystem bindings through that
same engine adapter. Use `tritium quantize --plan ... --output new-output`
(`--out` is an alias), or `--calibrated-plan` with its exact cache inventory.
The original-captured stage mode below is also available through `quantize`.
Typed output is a new `model.trit`/`evidence.jsonl` directory, not a `.tslb` file.
`--conversion-run-id` keeps its meaning; outer lifecycle summaries identify the
chosen command and cannot disable the sealed engine log.

Read-only original-bundle migration inspection is available before a large write:

```text
tritium --evidence off migrate salt-v2-bundle --inspect-only \
  --bundle /absolute/original-bundle --profile compact-v1 --codec b3 --group 128
```

It uses the same strict descriptor/HF/preserved/profile reader and shared
count/codec/index planner. It reports matrix/companion inventory and logical
trit, scale, working and raw-spool payload requirements. Qwen language/MTP names,
roles and shapes are compared to the public schema. The original completeness
and official-source flags remain unchanged; a matching inventory is metadata
coverage, not official source authentication or model quality.

Inspection forbids `--output`, `--spool` and `--import-run-id`, creates neither
spool nor output, and does not encode bytes or perform target scale-bit proof.
Its output explicitly excludes container overhead and full physical RSS/VRAM.
Logical caps apply; it never expands them automatically. The selected profile
is fully read/validated, so inspection is not a cheap header-only probe.

Without `--inspect-only`, those three write/run arguments remain required.
The same logical preflight runs before spool creation; actual import still
requires exact scale/trit encoding checks, terminal custody and cold log replay.
This imports existing coefficients without fitting and does not qualify release.

Typed `quantize` rejects legacy `--input`/`--model`, `--bpw`, `--scale-group`,
`--sensitivity`, `--fisher`, `--format`, `--ladder`, `--planes`, `--group`,
`--grid` and `--rotate`, even when explicitly set to their legacy defaults.
Legacy SALT geometric/ITF defaults and execution remain unchanged. This does
not infer plans from old recipes or retire that route: legacy numerical/receipt
parity, model-aware Qwen/Python migration and both campaign migrations remain
separate prerequisites to full consolidation.

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

For cold generation from that original-captured C, supply its exact admission
inputs explicitly:

```text
tritium generate --model new-output/model.trit --tokens prompt-ids.json \
  --original-captured-stage stage.trit \
  --transform-evidence transform.jsonl \
  --original-salience-evidence original.jsonl \
  --capture-evidence capture.jsonl \
  --stage-limits stage-limits.json --conversion-limits conversion-limits.json \
  --conversion-evidence new-output/evidence.jsonl \
  --level LEVEL_NAME --max-payload-bytes BYTE_CEILING --eos EOS_ID \
  --max-new 16
```

`LEVEL_NAME` must be a complete named level actually present in C. The payload
ceiling is a positive backend tensor-byte budget, not a full process, KV or
tokenizer RSS bound. `--max-evidence-bytes` now bounds all four logs together;
`--max-limits-bytes` bounds each limits JSON separately (default 16MiB).
All stage/log/limits inputs, the conversion log, level, payload budget and EOS
policy are required in this mode. `--eos` is the caller's stop-token policy;
a non-vocabulary sentinel may deliberately disable EOS stopping. No tokenizer
or BitNet stop token is inferred. `--native-only` optionally rejects emulated
tensor execution; the default allows the shared loader's explicit emulated tier.

`--tokens` remains exact IDs in/out and does not consult tokenizer assets.
Use `--prompt "text"` instead only when C carries a valid `tokenizer.json` and
`tokenizer_config.json`; the shared text loader validates vocabulary before
upload. Missing text capability rejects, with no fallback tokenizer. Both
paths use the common CPU runner and greedy decoder. Sampling is unsupported.
Legacy GGUF/SALT generation keeps its existing defaults and behavior; it does
not automatically unwrap `.trit` or infer ancestry from adjacent files.

This adapter supports only architectures admitted by the common standard
graph. It does not establish official hybrid-Qwen/MTP coverage, GPU speed,
streaming/mmap, standalone distribution, serving integration or model quality.
Keep B and the required logs with C for cold reproduction. CLI lifecycle logs
remain summaries, not independently qualified model-execution receipts.

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

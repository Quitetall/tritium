# tritium-serve engine contract

What an engine manager (LAMU, systemd, a container runtime) may rely on when it
launches `tritium-serve` as an OpenAI-compatible inference engine, the way it
would launch `llama-server` or `vllm serve`. Additions keep the existing
keys, flags and codes; removing or changing one is a contract change (ADR 0045).

## Binaries

`tritium-serve` links the backends it was built with. A plain build links `cpu`
only; `--features cuda` adds `cuda`. Do not assume a backend: ask the binary
(`--probe`, below). Two builds may coexist on one machine under different names
(for example `tritium-serve` CPU-only and `tritium-serve-cuda`).

## Probe

    tritium-serve --probe [--bundle <dir>]

Prints one JSON object (schema `tritium-serve-probe/1`) on stdout and exits 0,
without loading a model or touching a device. Keys are only ever added.

| Key | Meaning |
|---|---|
| `schema` | `"tritium-serve-probe/1"` |
| `version`, `version_line` | crate version; the `--version` line with source identity |
| `backends` | backends linked into this binary, e.g. `["cpu", "cuda"]` |
| `numerics` | tiers this binary can serve: `"exact"`, plus `"fast"` on CUDA builds |
| `fast_max_context` | the fast tier's largest context in tokens, or `null` |
| `endpoints` | HTTP routes served |
| `bundle` | present with `--bundle`: see below |

`bundle` (read from the bundle's `tritium.json` and `config.json`):

| Key | Meaning |
|---|---|
| `path`, `artifact_kind`, `schema_version`, `admission_id`, `source_revision` | bundle identity |
| `complete_model` | `false` means a measured research bundle |
| `needs_allow_incomplete_bundle` | the launch must pass `--allow-incomplete-bundle` (loopback only) |
| `default_profile` | the profile served when `--profile` is omitted |
| `profiles[]` | `name`, `file`, `present` (file exists), `package_id`, `resident_bytes`, `serialized_bytes` |
| `max_position_embeddings` | the model's trained context window |
| `fast_kv_bytes_per_token` | fast tier KV cost per context token (f32 K+V, full-attention layers) |
| `fast_state_bytes` | fast tier fixed recurrent + convolution state |

Device memory for a fast-tier launch is about `resident_bytes + fast_state_bytes
+ ctx * fast_kv_bytes_per_token`, plus a few hundred MB of CUDA context and scratch.

## Launch flags an engine manager needs

| Flag | Meaning |
|---|---|
| `--bundle <dir>` / `--model <gguf>` / `--converted <dir>` | model source, exactly one |
| `--profile <name>` | bundle profile (default `compact-v1`) |
| `--backend cpu\|cuda` | must be one of the probe's `backends` |
| `--numerics exact\|fast` | bundle numerics tier. `exact` (default): bit-identical host forward. `fast`: the device-resident CUDA executor, gated on relative error and greedy agreement against `exact` (`qwen36_resident_parity`); falls back to `exact` where it cannot run (CPU backend, a bundle it does not serve, a request longer than its context). Must be one of the probe's `numerics`. |
| `--ctx <N>` | context size: the most prompt + completion tokens one request may use, the same quantity as `llama-server -c`. Also sizes the fast tier's KV (capped at `fast_max_context`). |
| `--host <ip>` `--port <N>` | bind address; non-loopback requires `TRITIUM_AUTH_TOKEN(S)` |
| `--model-id <s>` | the id `/v1/models` reports and responses carry |
| `--allow-incomplete-bundle` | admit a measured research bundle; loopback only |

Launch flags also have `--config <json>` forms (kebab-case keys), and the model,
profile, backend, numerics and ctx flags have `TRITIUM_BUNDLE`, `TRITIUM_PROFILE`,
`TRITIUM_BACKEND`, `TRITIUM_NUMERICS` and `TRITIUM_CTX` forms. Precedence is
defaults < config file < environment < CLI.

## Lifecycle and readiness

1. The process validates arguments (for `--bundle`, including that the binary
   was built from one clean Git revision; a dirty build exits 1 at once), then
   loads the model and runs a short
   deterministic generation as a self-test and warm-up, so one-time device
   setup (kernel compiles, graph capture) is done before readiness and the
   first request is not slow. **Nothing listens during this phase:**
   connections to the port are refused. Loading a 12 GB bundle takes on the
   order of 1-2 minutes.
2. It then binds `--host:--port`. From that moment:
   - `GET /healthz` (liveness) returns 200 while the decode worker runs, 503
     if it has stopped.
   - `GET /readyz` (readiness) returns 200 only while the worker runs, the
     server is not draining, the backend has not faulted and the admitted
     artifact is serving; otherwise 503. This is the endpoint to poll.
   - `GET /v1/models` lists exactly one model, whose `id` is `--model-id`.
3. `SIGTERM` or `SIGINT` drains in-flight requests, then exits 0.

A manager should treat "connection refused" as "still loading" until its own
timeout, and read the exit status and stderr if the process exits first.

## Exit codes

| Code | Class (`fatal[<class>]`) | Meaning |
|---|---|---|
| 0 | | clean exit (`--help`, `--version`, `--probe`, or drained shutdown) |
| 1 | | any other error, including invalid arguments or configuration |
| 3 | `backend_unavailable` | the requested `--backend` is not linked into this binary |
| 4 | `backend_init` | the backend is linked but failed to initialize (no device, driver) |
| 5 | `model_invalid` | the model, bundle or profile could not be loaded or admitted |
| 6 | `out_of_memory` | an allocation failed while loading or preparing the model |
| 7 | `bind` | the listener could not bind (for example, the port is in use) |
| 8 | `self_test` | the startup self-test decode failed |

A classified failure writes exactly one line to stderr:

    tritium-serve: fatal[<class>]: <message>

Other errors write `tritium-serve error: <message>`.

## Environment and secrets

`tritium-serve` reads only documented variables: the `TRITIUM_*` launch
overlays, `TRITIUM_AUTH_TOKEN` / `TRITIUM_AUTH_TOKENS` (bearer tokens for
non-loopback binds), and diagnostic `TRITIUM_*` tuning switches. It reads no other credentials and
makes no network requests beyond its own listener.

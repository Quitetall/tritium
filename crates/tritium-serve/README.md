# tritium-serve

OpenAI-compatible serving for ternary models: continuous batching, paged KV, speculative decoding.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

## Local converted models

`tritium convert` directories can be served locally with the same OpenAI wire
contract. This path loads `model.tslb`, `config.json`, and the copied HF
tokenizer; it is a compatibility path and does not claim schema-v3 production
admission or release qualification.

```bash
target/release/tritium-serve \
  --converted /models/my-converted-model \
  --backend cpu \
  --model-id local-ternary \
  --port 8099
```

```bash
curl http://127.0.0.1:8099/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local-ternary","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

`TRITIUM_CONVERTED` and the `converted` JSON configuration key provide the
same model-path setting. `--converted` cannot be combined with speculative
decoding, batching, raw token mode, or a draft model.

### Manual native-model serving checks

These ignored tests use real converted SALT weights and their HF tokenizer:

```bash
TRITIUM_SERVE_E2E=1 TRITIUM_CONVERTED_PATH=/models/my-converted-model \
  timeout 900 cargo test --locked -p tritium-serve --features e2e --test e2e \
  serve_e2e_converted -- --ignored --nocapture --test-threads=1
```

The smoke check requires a completion. The lifecycle check compares greedy
SSE text, finish reason and usage with a non-streaming reference, drops real
responses during observed worker prefill and decode phases, then requires worker/queue
recovery without backend faults and identical subsequent completions. Missing
opt-in or model inputs fail an explicitly selected test; default CI keeps these
tests ignored. The model must produce a content token for the declared prompt
to exercise decode disconnect. Execution timeouts are test bounds, not release
latency thresholds. The phase gauge does not pinpoint an individual native
checkpoint. This is in-process compatibility-router coverage, not
socket transport, strict schema-v3 readiness, paged-KV/memory qualification,
model-quality or performance evidence.

## Deployment configuration

Launch configuration is fail-closed and has one precedence order:

```text
built-in defaults < --config strict.json (or TRITIUM_CONFIG) < TRITIUM_* < CLI
```

The JSON file uses kebab-case keys and rejects unknown keys before model or
backend initialization. Example:

```json
{
  "bundle": "/models/qwen36-salt-v3",
  "profile": "compact-v1",
  "backend": "cpu",
  "queue-cap": 32,
  "max-completion-tokens": 4096
}
```

Every non-secret CLI setting has a matching uppercase `TRITIUM_*` variable
(for example `TRITIUM_BUNDLE`, `TRITIUM_BACKEND`, `TRITIUM_QUEUE_CAP` and
`TRITIUM_MAX_COMPLETION_TOKENS`). Bearer credentials remain environment-only:
use `TRITIUM_AUTH_TOKEN` or bounded rotation via `TRITIUM_AUTH_TOKENS`.

For orchestrated shutdown, opt into a separate loopback-only listener with
`--admin-host 127.0.0.1 --admin-port 9090` (or `admin-host`/`admin-port` in the
strict config). It exposes only `GET|POST /drain`, which sets the same one-way
drain flag as SIGTERM; it never serves model, health, readiness, or metrics
routes. Kubernetes `preStop` hooks should target `127.0.0.1` explicitly.

## Cancellation diagnostics

`GET /healthz` includes `cancellation` metadata in healthy and unhealthy
responses, under the same authentication policy as other probes. Schema
`tritium.cancellation-capabilities.v1` reports configured generation, tree
session/verify, model-draft, batch-prompt, batch-speculation and batch-decode
checkpoints separately. The snapshot is taken before worker handoff; probing
health does not build a resident model or run inference.

Legacy defaults report entry/token-delivery checks, not interruption of an
entered prefill. `unknown` is undeclared behavior; `not_enabled` is an
unconfigured route. `cooperative_if_available` is conditional on native route
availability and request eligibility, not a guarantee that the operation can
run. Ordinary lockstep batch decode reports `cooperative_boundaries`: queries
run outside capture/replay and between eager head rows, before a single
whole-batch commit. Successful cancellation publishes no row result or history
progress and preserves committed prefixes for peer retry. Running kernels and
graphs are not preempted: `kernel_preemption` is always false.

These declarations are descriptive, not evidence of rollback, cancellation
latency, resource reclamation or release readiness. `qualification` is always
`not_assessed`; `/readyz` and independent candidate-bound gates remain unchanged.

## Metrics

`GET /metrics` uses same authentication boundary as generation. Prometheus
exposition has fixed-cardinality labels and schema marker
`tritium_metrics_schema_info{version="1"}`. Paged-KV deployments expose:

- `tritium_kv_pool_capacity_tokens` — shared logical token capacity;
- `tritium_kv_pool_free_tokens` — current free logical tokens;
- `tritium_kv_pool_reservations_total` — successful page-reservation operations;
- `tritium_kv_pool_releases_total` — successful page-release operations.

Dense per-slot KV reports zero for all four pool metrics. Token gauges describe
allocator capacity, not serialized/resident model bytes or compression ratio;
use artifact byte gauges and startup receipt for physical-byte claims.

## OpenTelemetry traces

OTLP trace export is opt-in. Set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` or
`OTEL_EXPORTER_OTLP_ENDPOINT` to enable the standard OTLP/HTTP protobuf exporter;
without either variable the server makes no telemetry export connection. The
server uses parent-based always-on sampling: root requests are sampled, while a
valid incoming W3C `traceparent` sampling decision is preserved. Export shutdown
is bounded to five seconds. Prompts, completions, auth tokens, and model names
are not emitted as trace attributes.

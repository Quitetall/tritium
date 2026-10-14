# tritium-serve

OpenAI-compatible serving for ternary models: continuous batching, paged KV, speculative decoding.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

## Typed portable candidates

`--typed /absolute/path/load.json` loads a complete selected `.trit` level through
the common `tritium_nn::load` interface and streams its standard or hybrid
language graph through `RunnerGenerator`. It uses the real embedded tokenizer,
not a raw-ID substitute. `TRITIUM_TYPED` and the strict launch config's `typed`
key select the same source. Exactly one source is allowed.

The separate load JSON uses snake_case and requires every field below. These
are illustrative ceilings, not a capacity recommendation for any model:

```json
{
  "path": "/models/example/model.trit",
  "level": "imported",
  "max_source_bytes": 1073741824,
  "max_manifest_bytes": 1048576,
  "max_directory_bytes": 1048576,
  "max_config_bytes": 1048576,
  "max_tensor_bytes": 33554432,
  "max_decoded_tensor_bytes": 134217728,
  "max_payload_bytes": 4294967296,
  "native_only": false
}
```

```bash
tritium-serve --typed /models/example/load.json --backend cpu --port 8099
```

The load JSON is capped at 16KiB and rejects missing/unknown fields, nonpositive
limits, relative model paths and absent explicit levels before backend init.
The backend payload cap is not a total RSS/KV/tokenizer/concurrency ledger;
`native_only` rejects backend-declared emulation, not all dense tensors. Choose
ceilings from the actual package and separately assess machine capacity.

Optional `stream_limits` bounds the selected context, each native forward segment
and the existing logical host-cache plan. Serving splits longer prompts into
segments; hybrid DeltaNet continuation stays one token. The selected capacity,
not the source model's maximum, bounds requests/completions.

To require resource-admitted encoded matrix streaming, add `streamed_resources`
and explicit `stream_limits` to the complete load JSON. This selects the common
`load_streamed_with_resources` path, without decoded-upload fallback:

```json
{
  "stream_limits": {
    "max_context": 4096,
    "max_segment_tokens": 128,
    "max_host_cache_payload_bytes": 1073741824
  },
  "streamed_resources": {
    "max_source_workspace_bytes": 67108864,
    "max_host_workspace_bytes": 536870912,
    "max_device_workspace_bytes": 0,
    "max_retained_host_bytes": 2147483648
  }
}
```

These are illustrative ceilings, not model-fit recommendations. Existing
`max_payload_bytes` bounds aggregate native matrix payload; the four additional
ceilings cover concurrent source work, backend-declared host/device upload work,
and preserved widened values plus host cache. A zero device-workspace ceiling
is allowed only when the backend explicitly plans zero. Other ceilings must be
positive. Unknown streaming/resource support refuses before matrix upload.
Reservations follow the loaded runner into the serving worker; failure drops
private partial handles. Preserved host vectors retain their bounded decode path.
Metadata/tokenizer heaps, allocator/driver/pinned costs and activation scratch
remain excluded: this is logical admission, not complete physical fit.
CUDA streams into unpublished device storage using bounded host windows. Its
logical plan charges128KiB additive/192KiB dense host upload workspace plus a
4-byte additive device validation flag. Additive CUDA requests must therefore set
`max_device_workspace_bytes` to at least4; zero deliberately refuses that cost.
Driver/pinned/compiler/allocator costs and physical peaks remain unknown.
`streamed_resources` currently requires host graph execution (CUDA
matrix operations are allowed), not `resident_device_f32`. Omitting it preserves
the explicit compatibility loader and does not invent resource admission.

Optional `execution` defaults to `{"kind":"host"}`. To request the common CUDA
device-f32 graph, build with `--features cuda`, use `--backend cuda`, and add both:

```json
{
  "stream_limits": {
    "max_context": 4096,
    "max_segment_tokens": 128,
    "max_host_cache_payload_bytes": 1073741824
  },
  "execution": {
    "kind": "resident_device_f32",
    "max_device_payload_bytes": 17179869184
  }
}
```

These are illustrative fields to merge into the complete load JSON, not a
standalone load config or model capacity recommendation. Resident mode requires
explicit positive stream/device limits. Unsupported backends/graphs or exceeded
bounds fail without host fallback. This arithmetic profile is not host bit-exact
or independently quality/performance qualified. Weight-upload and resident-graph
payload caps cover different domains; neither is a complete physical-memory ledger.

`GET /readyz` adds nullable `runtime_info`: actual `execution` (`host` or
`resident_device_f32`), admitted `max_context`, `max_segment_tokens` and logical
`resident_payload_bytes` (null for host). Unknown/legacy generators return null.
This snapshot is descriptive, not a receipt or inference/self-test probe.
Its nullable `resource_admission` reports the five covered logical byte domains
when resource admission was selected; compatibility loading returns null.
Neither this field nor successful streaming changes production readiness.

This route is loopback-only and unqualified. It rejects raw-token mode, spec,
draft, batching and KV-pool options. Hybrid MTP remains privately Unverified,
with no draft/tree authority. Qwen3.5 hybrid uses the existing QwenIm template;
standard graphs use Concat. Strict schema-v3 bundle admission is unchanged.
Compatibility readiness means worker liveness: `production_artifact` is false,
`startup_receipt` is null, and it is not public-release clearance. Resident
real-model qualification, typed production readiness, complete memory admission and remaining
thin-surface migrations are still owed.

## Local converted models (legacy)

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

# tritium-serve

OpenAI-compatible serving for ternary models: continuous batching, paged KV, speculative decoding.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

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

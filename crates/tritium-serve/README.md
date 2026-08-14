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

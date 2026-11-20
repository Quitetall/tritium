# Tritium serving dashboard

`tritium-serving-dashboard.json` is generated from `metric-registry.json`.
The registry describes the bounded, model-independent Prometheus series used
by the dashboard. `scripts/generate-serve-dashboard.py` also checks that every
registered series exists in the server's `/metrics` exposition.

Regenerate after changing the registry:

```sh
python scripts/generate-serve-dashboard.py
python scripts/generate-serve-dashboard.py --check
```

Import the generated JSON into Grafana and select the Prometheus datasource in
the `Prometheus` variable. The dashboard intentionally omits prompt, completion,
principal, filesystem, and model-identity labels; it is safe for the fixed
cardinality serving telemetry contract.

# Tritium Helm chart

This chart is digest-only and stages one artifact from a read-only PVC into a
bounded `emptyDir`; the serving container never downloads after startup. A
pinned BusyBox init container checks the exact SHA-256 before publication. An
authenticated loopback probe sidecar reads the bearer token from a Secret so
the Secret is never rendered into probe headers.
The helper is also a same-UID watchdog: after bounded authenticated health
failures it signals `tritium-serve`, allowing Kubernetes to restart the main
container when its worker is dead but its TCP listener remains open.

Default image and artifact digests are zero placeholders for lint/render only.
They are not deployable release identities. Override both with admitted values.
The current binary path is legacy GGUF compatibility; schema-v3 production
readiness and receipt parity remain binding gates. URI-to-PVC staging is not yet
implemented; the chart consumes a pre-provisioned read-only source PVC.

KEDA defaults to `minReplicaCount: 1`; scale-to-zero is not admitted for CPU or
GPU. SIGTERM triggers Tritium's graceful drain. An explicit preStop drain hook
remains unavailable until the admin listener lands.
CUDA uses a `Recreate` Deployment strategy so a one-GPU cluster cannot deadlock
waiting for surge capacity. This permits rollout downtime; use independently
scheduled releases for zero-downtime GPU upgrades.

Helm rollback restores the Deployment revision's image digest and expected
artifact hash together. Operators must retain the prior source bytes; if they
are missing or changed, rollback fails closed in the staging init container.

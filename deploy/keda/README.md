# KEDA queue-pressure example

This low-cardinality trigger scales on one aggregate queue-depth series. It has
bounded replica growth, slow scale-down and `minReplicaCount: 1`. Do not set the
minimum to zero for a stateful GPU deployment without admitted cold-start,
artifact-staging and client-retry evidence. The Helm chart carries the same
contract behind `keda.enabled`.

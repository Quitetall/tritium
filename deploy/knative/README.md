# Knative CPU tutorial profile

This is a separately labeled CPU compatibility example, not production or GPU
evidence. It pins concurrency to one, keeps at least one replica warm, bounds
scale and timeout, and requires a pre-staged read-only PVC. Knative volume
support must be enabled by the cluster operator. Replace the zero image digest
with an admitted image. The example does not prove cold start, schema-v3
readiness, authenticated probes, scale-to-zero, or rollback.

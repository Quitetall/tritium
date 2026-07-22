# 0052 — Hardened serving, deployment and observability

Status: **READY** (2026-07-20; work order frozen, implementation open)

- **Decision:** [ADR 0033](../adr/0033-v11-full-public-release.md)
- **Parent:** [plan 0044](./0044-v11-full-public-release.md)
- **Entry dependency:** stable schema-v3 native artifact/load API from plan 0047;
  local-RC packaging identity from plan 0051 before image qualification
- **External actions:** local images and clusters only until publication or a
  hosted deployment is explicitly authorized

## Goal

Turn the existing `tritium-serve` OpenAI-compatible server into a production
deployment surface for verified ternary artifacts. A release candidate must
admit an authenticated artifact before becoming ready, enforce explicit
resource and request budgets, preserve streaming cancellation/backpressure,
emit low-cardinality metrics and traces, and survive the declared container and
Kubernetes failure matrix without silently changing models or backends.

This plan strengthens the existing server. It does not replace the proven
OpenAI DTO/SSE contract, bounded worker queue, bearer gate, graceful drain,
continuous batching or paged-KV implementation.

## Frozen production contract

- **Artifact first:** startup accepts a strict schema-v3 Tritium directory or a
  declared legacy GGUF compatibility input. Production readiness requires the
  schema-v3 path, whose source, package, recipe, tokenizer, coverage and
  physical-byte identities are verified before any listener reports ready.
- **Immutable generation:** one process serves one admitted artifact identity.
  A model change uses a new process or an explicit transactional reload; an
  in-place partial mutation is impossible.
- **Health is not readiness:** `/healthz` proves the process and worker are
  alive. `/readyz` becomes successful only after artifact admission, backend
  allocation and a bounded inference self-test; it becomes unsuccessful before
  drain or device-loss recovery.
- **Fail-closed backend policy:** a requested CUDA/CPU backend never falls back
  to another backend. The effective backend, reduction tier, artifact/package
  IDs and physical device identity are reported by readiness and receipts.
- **Bounded requests:** body bytes, message count, prompt tokens, requested
  decode tokens, total token budget, queue depth, in-flight streams and per-key
  admission rate have explicit limits. Rejection occurs before expensive
  prefill and uses stable OpenAI error codes plus `Retry-After` where relevant.
- **Authentication:** non-loopback service requires bearer authentication.
  Kubernetes probes use a separately configurable loopback/admin listener or a
  probe credential; the public listener does not exempt health endpoints.
- **TLS boundary:** Tritium supports plaintext behind a trusted proxy and
  documents the trust boundary. TLS certificate issuance and edge WAF are not
  implemented inside the inference process.
- **Telemetry safety:** prompts, generated text, bearer tokens, filesystem paths
  and unbounded model IDs never enter logs, metric labels or trace attributes.

## Slice 1 — strict startup and readiness

Add a production loader/config layer around `tritium-serve`:

- accept config from CLI plus environment with one precedence table and reject
  unknown config-file keys;
- load schema-v3 directories through the strict native reader, bind the exact
  artifact/package/source/recipe identities, and reject corrupt or incomplete
  preserved assets before binding the public socket;
- add `StartupReceiptV1` containing source revision, build, artifact identity,
  backend policy/effective backend, physical device, reduction tier, measured
  resident bytes and self-test digest;
- add `/readyz` and a bounded deterministic one-token self-test; readiness is
  false during startup, drain, worker death, failed self-test or device loss;
- make startup and reload transactional: failed admission leaves no public
  listener and no partially published receipt;
- retain legacy GGUF as an explicitly labeled compatibility mode that cannot
  satisfy the v1.1 production-artifact gate.

Gate: valid schema-v3 CPU and CUDA fixtures become ready with matching receipt
identities; corrupt package bytes, asset digests, coverage, tokenizer or backend
policy fail before readiness. A readiness test must use the actual native
loader, not `MockGenerator`.

Progress (2026-07-21): `tritium-serve` now has an unforgeable admission bridge
from `Qwen35SaltV2LoadReceipt`, a versioned startup receipt, a synchronous
deterministic one-token self-test before worker spawn, and a one-way production
readiness revocation handle. The production builder accepts only an opaque
admitted-generator capability, preventing artifact-A/generator-B receipt
substitution. Production `/readyz` binds package/source/backend/
device/byte-ledger identities and the self-test digest; legacy routers are
explicitly labeled `legacy_compatibility` and cannot satisfy the production
artifact gate. A dedicated `QwenGenerator` now owns that exact strict model and
implements bounded greedy/top-k/top-p decode plus logprobs; the only public
admission function derives both generator and receipt from the same model
value, closing artifact/generator substitution. Strict `tokenizer.json` plus
`tokenizer_config.json` loading and Qwen `<|im_start|>` chat rendering now exist
without implicit BOS injection. Main CLI now accepts exactly one strict
`--bundle DIR --profile ...` or legacy `--model GGUF`, rejects unsupported
strict-mode combinations and unknown arguments, parses tokenizer bytes retained
by the authenticated load transaction, and invokes the production router with
clean compile-time source plus physical backend identity. CPU/CUDA Compose now
mount canonical schema-v3 directories read-only, preflight mandatory ordinary
assets/profile, and invoke `--bundle`; the native loader remains the byte-level
trust boundary. Helm now stages a bounded directory without archive extraction,
rejects links/special nodes, pins copied `tritium.json`, then lets the native
loader verify every referenced byte before readiness. Real CPU/CUDA fixtures
remain open. Until those gates pass,
the production builder is a hardened seam, not release-gate evidence.

## Slice 2 — request security and resource governance

Close the actionable server residuals in `docs/security/threat-model.md`:

- add maximum messages, UTF-8 prompt bytes, tokenized prompt length, requested
  completion tokens and combined token budget to `ServeConfig`;
- reject an excessive `max_tokens` value rather than silently clamping it;
- add per-principal token-bucket admission with bounded cardinality and
  deterministic eviction; anonymous loopback is one principal;
- retain constant-time token comparison and support a bounded set of rotating
  bearer-token digests without logging secrets;
- apply an explicit stream lifetime/decode deadline after SSE headers, propagate
  disconnect/deadline cancellation to queued and active jobs, and reclaim KV
  reservations exactly once;
- distinguish overload (`429`), drain/not-ready (`503`), invalid request
  (`400`) and deadline (`408`) using the OpenAI error envelope;
- make every arithmetic byte/token/slot product checked and return a typed
  error under absurd dimensions instead of panicking in `panic=abort` builds.

Gate: contract and adversarial tests cover queue saturation, principal rate
limits, slow consumers, disconnect at every generation phase, deadline races,
oversized prompts, malformed JSON/artifacts, auth rotation and drain. After
each case, queue, KV pool, active jobs and resident bytes return to baseline.

Progress (2026-07-20): message/byte/prompt/completion/combined admission
ceilings and reject-not-clamp semantics are implemented. The existing request
lifetime now also runs inside lazy SSE bodies from queue admission: expiry
emits a typed OpenAI `request_timeout` event, records
`tritium_stream_timeouts_total`, terminates framing and drops the receiver so
the worker cancels. Non-streaming expiry returns HTTP 408 with the same typed
OpenAI envelope. A startup-validated rotation set of at most 32 bearer keys is
stored as fixed-size BLAKE3 digests; expensive routes use independent,
fixed-point token buckets by key (or one anonymous loopback bucket), so
attacker-controlled principal growth and eviction churn are impossible.
`Retry-After` and `tritium_rate_rejections_total` are contract-tested. Explicit
KV reclamation receipts and the full adversarial matrix remain open.

## Slice 3 — metrics, logs and traces

Replace the three-counter diagnostic surface with a versioned telemetry seam:

- Prometheus counters/histograms/gauges for admission outcome, queue wait,
  prefill/decode latency, time to first token, tokens in/out, active/queued
  requests, KV usage, artifact resident bytes, backend errors, cancellations,
  device loss and speculative acceptance when enabled;
- OpenTelemetry spans for admission, tokenize, queue, prefill, decode and
  stream, with W3C trace-context propagation and explicit sampling policy;
- structured JSON logs with request/trace IDs, stable error code and bounded
  identities; no prompt/completion content by default;
- `/metrics` schema tests, a cardinality budget test, and golden Grafana panels
  generated from the metric registry so dashboard queries cannot drift;
- startup and shutdown flush behavior bounded by a configured deadline.

Gate: an in-process collector and Prometheus parser prove names, units,
histogram buckets and trace parentage. Load tests prove label cardinality is
independent of request/model text. Cancellation, overload, self-test failure
and device loss each emit one coherent metric/log/trace outcome.

## Slice 4 — reproducible OCI images

Replace the developer-oriented root `Dockerfile` path with qualified CPU and
CUDA images:

- build only from the exact plan-0051 local-RC source/archive set with locked
  dependencies and pinned base-image digests;
- run as non-root with a read-only root filesystem, dropped Linux capabilities,
  no shell/package manager in the final image, declared temp/cache mounts and a
  seccomp-compatible syscall surface;
- include license notices, SBOM, provenance, source revision, binary and config
  schema versions; bind them to the image digest;
- expose separate public and optional admin/probe ports with no baked secret or
  model weight;
- supply Docker Compose smoke for CPU and NVIDIA Container Toolkit smoke for
  CUDA, both mounting the admitted artifact read-only;
- scan locally for known vulnerabilities and leaked secrets. A waiver names an
  exact digest, advisory, rationale and expiry.

Gate: the exact image runs as an arbitrary UID with read-only rootfs, reaches
readiness, serves streaming and buffered requests, drains on SIGTERM within the
budget, rejects write attempts and preserves the same startup/artifact receipt
as the unpackaged binary.

Progress (2026-07-21): digest-pinned Linux/amd64 CPU and CUDA Dockerfiles now
use a shell-free distroless runtime, non-root ownership, immutable OCI metadata,
and no implicit writable volume. A clean-tree builder admits the plan-0051
candidate manifest, retains its exact source-revision archive, vendors the
locked Cargo graph, builds offline with frozen resolution and requests an OCI
archive with BuildKit SBOM/provenance attestations. CPU/CUDA Compose profiles
enforce arbitrary UID, read-only root, dropped capabilities,
`no-new-privileges`, bounded tmpfs and read-only model mounts. Structural
contract tests are local-green. OCI archive admission now streams every blob
digest, validates single-platform image/config identity, and requires in-toto
SBOM plus SLSA predicates whose subjects bind exact image-manifest digest.
Exact-image runtime, empirical vulnerability/secret scan, strict schema-v3
receipt parity, and NVIDIA evidence remain open gates.

Runtime qualification harness now requires an exact Docker repository digest,
the verified OCI archive/build receipt/package-candidate lineage, strict bundle
manifest identity, production `/readyz` receipt parity, singular model listing,
one buffered and one SSE generation, read-only root/bundle, dropped capabilities,
`no-new-privileges`, and bounded successful SIGTERM drain. CUDA runs additionally
require physical GPU UUID/name/driver evidence. It emits a content-addressed,
candidate-archive-bound receipt only after all checks pass. The release registry
admits CPU and CUDA runtime evidence separately and neither substitutes for the
remaining deployment gate. No qualifying image or 27B bundle is present locally
yet, so both empirical runtime receipts remain open.

CUDA backend identity now separates logical selection (`cuda:<ordinal>`) from
driver-reported physical identity (`cuda:<ordinal>:GPU-<uuid>`). Strict serving
uses physical identity in startup receipts, and runtime qualification rejects a
receipt unless that UUID matches independent `nvidia-smi` evidence. This closes
ordinal-only substitution structurally; actual CUDA image evidence remains open.

Security qualification now runs separate offline Trivy passes over exact OCI
archive bytes: HIGH/CRITICAL vulnerability detection and unfiltered secret
detection. Admission requires Trivy 0.69.0 or newer, a non-expired vulnerability
database no more than 24 hours old, hashed scanner/DB inputs, zero findings and
a content-addressed candidate-bound receipt. CPU and CUDA scan receipts are
independent release gates. Trivy and its DB are absent locally, so no empirical
security receipt exists; waiver schema remains intentionally unimplemented.

## Slice 5 — Helm, autoscaling and serverless examples

Add `deploy/helm/tritium`, `deploy/keda`, and `deploy/knative`:

- Helm values/schema cover image digest, artifact volume/URI staging, backend,
  GPU resources, node selectors/affinity, topology spread, tolerations,
  runtime class, security context, probes, termination grace, PDB, NetworkPolicy,
  ServiceMonitor, secret references and resource ceilings;
- an init container stages and verifies the immutable artifact; the serving
  container receives a read-only admitted directory and never downloads after
  readiness;
- KEDA scales from an external low-cardinality queue-pressure metric with
  stabilization and max-replica bounds; scale-to-zero is disabled for the
  stateful GPU profile unless cold-start evidence explicitly admits it;
- Knative is a separately labeled CPU/tutorial or pre-warmed GPU example with
  concurrency, timeout, cold-start and storage semantics stated honestly;
- rolling update uses readiness plus `preStop` drain; rollback pins the previous
  image and artifact digests together.

Gate: schema/lint/render tests plus a local kind/k3d CPU deployment prove
install, request, scale signal, restart, rolling update, failed update and
rollback. The CUDA lane runs on an actual NVIDIA Kubernetes node and records
driver/runtime/GPU identity; CPU structural rendering cannot green it.

Progress (2026-07-21): a strict-schema Helm chart now covers digest-only images,
PVC-to-bounded-emptyDir artifact staging with SHA-256 verification, CPU/CUDA
resources, runtime class and scheduling, pod/container security, Secret-backed
auth, authenticated readiness plus bounded worker-death restart through a
same-UID loopback watchdog sidecar, GPU-safe `Recreate` updates, PDB,
NetworkPolicy, ServiceMonitor and bounded KEDA scaling. Standalone KEDA and
Knative CPU tutorial examples are explicitly non-zero-scale compatibility
profiles. Pinned offline Helm lint/render and Python schema contracts are
local-green. Admin/preStop drain, schema-v3 loader wiring, kind install/rollout/
rollback, URI-to-PVC staging, live Prometheus/KEDA, Knative cold start and NVIDIA
Kubernetes evidence remain open gates.

Deployment qualification is now automated by
`scripts/qualify-kubernetes-deployment.py`. It binds exact chart and OCI
archives, package lineage, bundle identity, namespace/PVC/Secret preconditions,
the immutable image digest, Kubernetes node identities and tool binaries. The
single-replica CPU or CUDA release must serve authenticated readiness, model
listing, one-token generation and generation-bearing Prometheus metrics;
survive a pod replacement and a successful CPU `RollingUpdate` or CUDA
`Recreate` update without changing its startup receipt; then record a
real failed atomic Helm upgrade that changes both image and artifact identities,
the exact deployed rollback revision, restoration of the admitted pair and complete
uninstall. CUDA additionally runs a release-policy-allowlisted,
digest-addressed NVIDIA evidence image with a fixed `nvidia-smi` command on the
node that served Tritium to capture the driver, CUDA runtime, GPU name and
physical GPU UUID; that UUID must match the serving startup receipt. The harness
also makes CPU qualification contingent on KEDA and external-metrics API
preflight, a bounded Prometheus-backed generation load, an Active/Ready
ScaledObject, HPA-backed scale-out from one to at least two ready replicas and
settling back to one. Harness and receipt validator are local-green; empirical
Prometheus/KEDA and NVIDIA-cluster receipts remain open release gates. The v2
receipt also binds the exact bundle-manifest and OCI-build-receipt support
bytes. Release-evidence admission is fail-closed and flavor-specific: separate
CPU and CUDA deployment kinds each require the matching OCI runtime and
security receipts as their only parents, all three bind the same candidate
image, and the runtime/deployment startup receipts must be identical.

## Slice 6 — failure injection and release evidence

Run the exact local-RC image/chart through a sealed scenario matrix:

OCI runtime qualification v3 now executes the first production failure subset
against exact CPU/CUDA candidate images: unauthenticated and wrong-token access,
stable malformed-JSON envelopes, deterministic per-principal `429` plus
`Retry-After`, second-principal isolation and an exact rejection-counter delta.
The content-addressed receipt retains statuses, retry duration and counter
baseline/final values. Oversized-body `413` and all
three POST surfaces use the same bounded OpenAI JSON error envelope in router
contract tests. It also uses a one-entry queue under synchronized slow SSE load,
requires exactly one active and one queued admission plus exact capacity-`429`
and queue-counter deltas, holds accepted streams unread after observed token
progress, then closes them and requires disconnect-counter movement, an empty
settled queue, a live worker and a bounded successful recovery generation. The
receipt binds the queue capacity, hold duration, token progress and recovery
latency. Remaining scenarios below still block this slice.

SIGTERM phase qualification now has a causal runtime primitive:
`tritium_worker_phase{phase="idle|prefill|decode"}` is a fixed-cardinality
one-hot gauge across chat and tree work in both single and batched workers.
Once drain begins, queued chat and tree jobs are rejected before model prefill;
tree requests retain the same `503 draining` classification whether rejected
at router admission or dequeued by the worker. Exact candidate SIGTERM receipts
for all three phases remain required below.

1. malformed/truncated/wrong-identity artifact at startup;
2. missing secret, invalid config and unavailable requested backend;
3. SIGTERM during queue, prefill and decode;
4. worker panic/process restart, device loss/OOM and artifact volume loss;
5. telemetry collector unavailable/slow and metrics scrape flood;
6. failed rollout followed by digest-pinned rollback.

Every scenario records configuration, source/image/chart/artifact digests,
hardware, requests, expected/observed state transitions, telemetry assertions,
resource high-water marks and cleanup state. No skipped or zero-case lane is a
pass.

## Verification cadence

```bash
cargo test -p tritium-serve --features serve
cargo clippy -p tritium-serve --features serve --all-targets -- -D warnings
cargo test -p tritium-serve --features cuda
docker buildx build --load --provenance=false -f deploy/oci/Dockerfile.cpu .
helm lint deploy/helm/tritium
helm template tritium deploy/helm/tritium --validate
./scripts/serve-fault-matrix.sh --local-rc <receipt-directory>
git diff --check
```

CUDA, Kubernetes and observability evidence commands are refined in their
slice commits and write machine-readable receipts under an ignored output
directory. Documentation tables are generated only from admitted receipts.

## Stop conditions

- Stop before weakening strict artifact admission, explicit backend selection,
  resource ceilings, secret handling, readiness, cancellation or rollback.
- Stop if a production request can trigger an unbounded allocation, metric
  label, queue wait, stream lifetime or retry loop.
- Stop before pushing an image/chart, provisioning a hosted cluster, reserving
  a public endpoint or paying for infrastructure without explicit approval.
- Stop before calling a CPU-rendered chart test proof of the NVIDIA deployment
  lane.

## Done criterion

The exact local-RC binary and CPU/CUDA OCI images admit only verified artifacts,
pass protocol/security/load/cancellation/shutdown/telemetry tests, deploy and
roll back through the pinned Helm/KEDA/Knative configurations, and survive the
failure matrix with bounded resources and coherent receipts. Published images
and charts, if later authorized, must be byte-for-byte the qualified artifacts
and pass a separate registry pull smoke.

## Commit sequence

```text
feat(serve): admit verified artifacts and readiness receipts
feat(serve): enforce request and principal budgets
feat(serve): export bounded production telemetry
build(oci): qualify hardened CPU and CUDA images
feat(deploy): add Helm KEDA and Knative profiles
test(serve): admit production failure matrix
```

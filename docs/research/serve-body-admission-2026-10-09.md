# Serving response-body admission — 2026-10-09

Source baseline: `d1b7974f5af8698baf36112acf3fa6f3fc5eeb12`.
Contract: ADR 0033 / plan 0052 and ADR 0050 (private planning repository).
Status: local software validation passed; empirical release gates remain open.

## Red-capable feedback loop

```sh
cargo test --locked -p tritium-serve --features serve --test contract \
  inflight_cap_retains_unpolled_sse_until_body_drop -- --exact --nocapture
```

At a configured cap of one, the first response body is retained without
polling. A second ordinary request must be rejected immediately, then dropping
the first body must permit recovery. Repro unit
`tritium-serve-stream-admission-repro-20261009.service`, invocation
`e5c94ba64c7f408280c3fa8c623d94ea`, exited 101: expected 429, received 200,
test time 0.00 seconds. A direct cached repeat reproduced it in 0.01 seconds.
The analogous model-listing-to-chat test also failed, expected 429/actual 200,
in 0.01 seconds without any prior generation work.

Ranked hypotheses were response-future permit lifetime, separate clone budgets,
and worker-completion ownership. Locked Tower 0.5.3 source confirms the permit
lives in `ResponseFuture`, not the body, while service clones share their
semaphore. Axum route-layer application creates separate service limiters.
The non-generation model-listing probe excludes worker completion as the
explanation for the retained-body escape.

## Implementation and limits

One shared, nonblocking budget now spans ordinary routes and router clones.
The handler holds its permit until response creation, then transfers it to a
private `HttpBody` wrapper. EOF, body error or drop releases it exactly once;
pending-handler cancellation drops its owned permit. Frames, size hints and
trailers pass through without a relay, token buffer or body copy.

Authenticated GET health/readiness/metrics use normal shared capacity first,
with one reserved overflow permit when it is exhausted. This preserves idle
collector concurrency while bounding the total admitted population to N+1,
without an authentication exemption. Zero preserves explicit limiter-disable
compatibility. Capacity errors use typed HTTP 429 and `Retry-After: 1`;
per-principal rate counters retain their separate meaning.

The first validation run, invocation `20d18ac8af22450c9ff2c48e346bff7d`, caught
two missing `error.code` fields in the new capacity responses (HTTP status was
correct). It passed 54 library, four binary, two CLI and 40 contract cases,
failed two contract cases, and exited 101 before Clippy/fmt. This is not a
passing gate. The code fields were subsequently added. An earlier focused
120-second build exited 124 without a test verdict; that is a build timeout,
not a model or software correctness result.

Drain precedence is separately regression-tested: saturation must not mask an
authenticated draining/not-ready generation state as overload. The focused
drain repro, invocation `71f2fbf8dc40455fb613812e2fd7d49f`, exited 101 in 0.00
seconds (expected 503, actual 429). Its existing readiness/drain checks were
inside the capacity middleware. The middleware now reuses the same readiness
predicate before capacity admission on all three model-work POST routes,
after authentication/principal admission.

## Completed local validation

Managed unit `tritium-serve-body-admission-final-validation-20261009.service`,
invocation `44b9a1eb6b9f4fa6ae2a07d7fb0b1a78`, finished successfully at
21:16:38 local time (exit 0). Commands, using the existing SSD target,
`RUSTC_WRAPPER=`, two build jobs and the existing scratch TMPDIR:

```sh
timeout 900 cargo test --locked -p tritium-serve --features serve
timeout 900 cargo clippy --locked -p tritium-serve --features serve --all-targets -- -D warnings
timeout 120 cargo fmt --all --check
```

104 executed tests passed: 54 library, four binary, two CLI, 43 contract and
one OpenTelemetry test. Disabled CUDA/batch/spec/e2e lanes and zero-case doc
tests are not qualification evidence. Scoped warnings-denied Clippy and
workspace formatting passed. All eight new in-flight contract cases passed
20 consecutive cached runs each; all five budget/body lifecycle unit cases
also passed 20 consecutive cached runs each. These are developer-worktree
checks, not candidate-bound empirical receipts.

The regression set covers retained SSE and model-listing bodies, cross-route
and clone sharing, EOF without destruction, never-polled drop, body errors and
trailers, cancelled request extraction, exact probe classification and bounded
reserve, idle eight-collector capacity, authentication under saturation,
disabled-cap compatibility and drain precedence on all three work routes.

## Evidence boundary and cleanup

These are software lifecycle checks with controlled generators, not Qwen
quality, GPU performance, model/KV reclamation, OCI/Kubernetes qualification
or independent public-release approval. Existing `tritium_requests_inflight`
counts middleware execution, not held bodies. Router admission does not bound
accepted sockets, incomplete headers, rejected-response transport or native
kernel cancellation latency.

No per-run scratch directory was created. Existing SSD build caches were
reused; durable source, ADR and this note are intentionally retained. Historic
August campaign data was not deleted. Foreign staged docs and optimizer WIP
remain outside this change.

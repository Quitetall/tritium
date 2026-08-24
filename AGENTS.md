# Working in Tritium

Tritium is governed by receipts, frozen contracts, and explicit release gates.
These rules apply to contributors and coding agents.

## Non-negotiable rules

1. **Do not self-approve empirical claims.** The performer may build and report
   evidence, but independent review or a separately executed verifier must clear
   release, quality, physical-byte, performance, and security obligations.
2. **Unknown is neither failure nor pass.** Missing hardware, credentials,
   models, or hosted evidence stays `UNKNOWN`/blocked; never downgrade it to a
   green result.
3. **Do not edit generated projections.** Change source atoms or the owning
   generator, regenerate, then run the drift checker. Release manifests and
   receipts are immutable evidence, not hand-written status files.
4. **Do not weaken a frozen gate to make it pass.** Change the contract through
   an ADR or accepted amendment, with migration and rollback recorded.
5. **Claims are bounded by evidence.** Compile-only, emulated, synthetic, and
   CPU results cannot be reported as GPU, model-quality, performance, or SOTA
   results. Record model, data, artifact, source revision, machine, device, and
   run identity for measured claims.
6. **Public contract changes need an ADR.** This includes APIs, schemas, wire
   formats, backend semantics, release gates, governance, and compatibility.
   Preserve unrelated work and never commit secrets, private model weights, or
   unauthorized redistributable payloads.

## Required loop

```text
read the applicable ADR/plan
run the narrow test while iterating
run the relevant local gate on the commit tree
run independent review before calling work complete
report exact commands, evidence identities, skips, and blockers
```

The canonical local entrypoint is `scripts/verify-gates.sh`. Hooks inspect the
staged tree or pushed commit; `--no-verify` is an explicit local bypass only,
never a CI or branch-protection bypass. CI remains authoritative.

## Release discipline

Release status must remain fail-closed. A passing unit test or package build is
not a qualification receipt. Public activation also requires explicit human
authorization after all independent gates are green.

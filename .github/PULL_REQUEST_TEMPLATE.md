## Outcome

Describe the user-visible result and the contract, ADR, plan, or issue it closes.

## Verification

List exact commands and results. Identify skipped optional dependencies or
hardware lanes; a skip is not a pass.

## Evidence and compatibility

For measured or artifact-producing changes, include immutable source, model,
data, artifact, run, machine, and device identities. State API/schema,
performance, memory, security, and backward-read effects.

## Checklist

- [ ] Tests fail without the change and pass with it, or the documentation-only rationale is stated.
- [ ] Generated files were changed through their owning generator.
- [ ] No secret, private model/data content, or unauthorized redistributable payload is included.
- [ ] New claims are receipt-backed and accurately labeled structural, emulated, or physical.
- [ ] Public API/schema/conformance/release-gate changes have an ADR or accepted amendment.
- [ ] Known limitations and follow-up work are recorded.

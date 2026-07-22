# Tritium governance

Tritium is currently maintainer-led. Brian Lam is the release maintainer and
holds final responsibility for repository access, security response, release
authorization, and accepted architecture decisions. This is an operational
role, not authority to bypass published compatibility or evidence gates.

## How decisions are made

- Bugs, documentation, and internal implementation details use normal pull
  request review.
- Public APIs, artifact schemas, backend semantics, conformance vectors,
  benchmark admission, support policy, and release gates require an ADR or an
  explicit amendment to an accepted ADR.
- ADRs record context, alternatives, consequences, and objective exit gates.
  Accepted decisions remain binding until superseded by another accepted ADR.
- Release claims are generated from admitted receipts. Maintainer judgement
  cannot turn missing or structural-only evidence into a passing claim.

Consensus is preferred. When consensus is unavailable, the release maintainer
records the decision and rationale in the relevant ADR or pull request. Major
disagreements may be escalated to a time-bounded request for comment before a
decision.

## RFC process

Use an RFC when a proposal changes more than one public subsystem, introduces a
new ecosystem contract, or remains materially disputed after ordinary ADR
review. Open a draft under `docs/rfcs/` from a repository template containing
motivation, contract, alternatives, compatibility, evidence plan, security and
unresolved questions. The pull request remains open for at least 14 calendar
days after it is announced in the issue tracker; urgent security work may use a
private review and publish the record after coordinated disclosure.

The release maintainer accepts, rejects, or returns the RFC for revision after
the comment window. Acceptance authorizes an ADR amendment or new ADR; it does
not itself change a frozen contract. The final RFC, decision, dissent, and links
to resulting ADRs remain in repository history. An accepted contract changes
only when its ADR and tests land.

## Maintainers and reviewers

Maintainers merge changes, manage releases and private security reports, and
protect branch, signature, and registry credentials. Reviewers may be granted
area ownership after sustained, accurate contributions. Access follows least
privilege and may be removed for inactivity, compromised credentials, or Code
of Conduct violations.

Reviewers must:

- disclose financial, employment, research, or authorship conflicts relevant to
  a decision;
- avoid approving their own material change without independent review;
- verify automated findings and benchmark evidence against primary artifacts;
- preserve private disclosure and embargo boundaries.

A materially conflicted decision-maker must recuse from approval and release
authorization for that decision. An unconflicted maintainer appoints the
alternate reviewer in the public proposal. If no unconflicted reviewer with the
needed competence is available, the decision remains pending; sole-maintainer
status is not permission to waive recusal.

## Compatibility and deprecation

The stable-core and evolving-tier boundaries are defined by the release ADR and
checked by the semver gate. Stable APIs receive a deprecation notice for at
least one stable minor release before removal unless an urgent security issue
requires a break. Artifact readers document their backward-read window; new
writes use only the current schema. A conformance-vector change requires a new
version and migration note, never silent regeneration.

## Research and model evidence

A benchmark or model-zoo promotion requires immutable source/model/data
identities, physical-byte accounting, environment and hardware identity,
commands, uncertainty where applicable, and a machine-readable receipt.
Demotion occurs when evidence is invalidated, a license changes, a security
issue makes distribution unsafe, or a supported release can no longer reproduce
the claim. Negative results remain part of the record.

## Amendments

Governance changes use an ADR and public pull request. They cannot retroactively
weaken a frozen release gate or erase an evidence failure. Repository history is
the durable archive; future forums or chat services are discussion surfaces,
not substitutes for recorded decisions.

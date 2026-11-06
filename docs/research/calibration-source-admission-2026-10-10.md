# Source-preparation runtime admission candidate

Proposed ADR0060 continuation on original-source identity commit
`32853e49b8374e4dc57e09fa3300f5b6c82c1986`.

`Engine::run_with_source_guard` shares the actual engine implementation with
ordinary `run`. It compares the preparation OriginalSourceId against actual
original byte bindings consumed by conversion, before final package allocation
or any sink call. Changed fitted-weight bytes, preserved dense bytes, assets or
interpretation reject; there is no second original source read, fallback or
publication of a mismatched artifact. A late mismatch may occur after fitting.

This is a runtime control, not a new v1 event or audited calibration claim.
Successful guarded and ordinary runs retain identical existing package/evidence
bytes. Source files are not locked, and cache/objective/token/selection/factor
truth is not established. Typed cache verification, source equality bound in
recorded metric observations, row-specific KFAC and package replay/tamper checks,
immutable-source/RSS admission and all physical/quality/release gates remain open.

The unchanged-source candidate check completed exit 0: all 15 model-engine tests
passed, including consumed-byte mutation, exact read counts, rejection before
publication and unchanged successful v1 package/evidence bytes. Scoped strict
Clippy/docs passed. Full committed gates/publication remain separate obligations.

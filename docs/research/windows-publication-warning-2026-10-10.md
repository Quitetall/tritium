# Directory-sink warning contract on Windows

This is software diagnosis and a test correction, not native Windows or release
qualification. ADR0058 explicitly permits a post-commit directory-sync warning;
the production sink and its commit/cleanup semantics are unchanged.

CI 38100096606 at `9b9d92ab3c72cdbf9cd83e49877a44cd0a187a10` failed
Windows job 114354068217 in
`real_file_source_and_directory_sink_never_overwrite` at the assertion that
publication warnings must always be empty. The assertion did not print the
warning variant. Linux success does not establish that assumption on Windows.

A minimized GNU Windows probe imports the actual unchanged production sink
from `078fc3710a3c55e5113b60332e7ed1354d03a146`. Its caller supplies small
synthetic bytes and JSONL; package/evidence validation is outside this I/O seam
and is not qualified by the probe. Wine 11.19 reports `[DirectorySync]`; opening
the output directory itself fails with Windows error 5 / PermissionDenied /
"Access denied." The old empty-warning assertion exits 101. A capability-aware
assertion exits 0 and checks exact publication bytes/log, no pending file and
no overwrite. The native CI symptom is consistent with this cause, but fresh
native Windows CI must verify the actual complete test.

The complete model-engine regression now independently attempts the same
directory open/sync operation and requires exactly the corresponding warning
list: empty on success, or only `DirectorySync` on failure. `PendingCleanup`
still fails the test. Real source fitting, package/log verification, absence of
pending files and refusal to overwrite existing output remain mandatory.
In-memory sink tests still require no warnings. No platform is silently marked
durable, no warning is suppressed, and no release gate is relaxed.

The corrected complete test passes on Linux using:

```sh
CARGO_TARGET_DIR=/mnt/4tb/build/cargo/70/7e33efa4656acd \
CARGO_BUILD_JOBS=2 RUSTC_WRAPPER= RUSTFLAGS='-D warnings' \
cargo test --locked -p tritium-quantize --test model_engine \
  real_file_source_and_directory_sink_never_overwrite -- --exact
```

Wine startup deadlines, one stale PID-path collision, a server-connect failure
and a later timeout/kill exit 137 are retained separately and are not additional
sink-test verdicts. Only the passing probe has a directly measured sub-second
wall time; emulation startup was intermittent. Owned Wine processes were
prefix-checked before termination and the disposable prefix/binaries removed.

The prior exact `078fc371` verifier passed 740 four-crate tests, 14 actual
NN/tokenizer/loader tests and 108 old-source F64 differential cases. It then
terminated with exit 124 at its CLI compilation deadline, before strict docs,
bare-metal schema or canonical prepush. This is PARTIAL, not a completed gate.
Fresh committed-tree verification must cover this correction and the joint
feedback candidate before publication. Native CI and all empirical model,
hardware, security, reproduction and human release gates remain open.

Diagnostic/terminal custody is under
`~/Projects/Tritium/archive/verification/joint-feedback-078fc371-20261010/`.
Checksum custody is not release approval; the addendum records the corrected
timing language, iteration compiler error and repaired checksum-index mistake.

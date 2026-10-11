# Conversion parity requires matching executed thread context

Hosted CI38120324242 at b4d4ef28 failed all three native OS conversion
parity suites after their strict workspace Clippy checks passed. Three tests
failed at complete package-byte equality; this failure is retained, not retried
into a passing receipt.

On unchanged69067b79 source, the actual command suite passes all four cases
with a two-thread parent pool. Changing only that parent to three threads
reproduces all three failures. The CLI child was hardcoded to two threads,
while the in-memory reference used its actual pool. Direct and calibrated
fitting correctly record actual rayon_threads in their evidence, which feeds
package identity. Different execution contexts must not compare as identical.

The test adapter now passes rayon::current_num_threads() to the child. This
also works inside installed local pools. A new regression exercises the real
plain F16/BF16/F32, calibrated F16/BF16/F32 and raw-Gram command fixtures in
one- and three-thread pools. Complete package/JSONL byte equality, repeated
conversion, independent log/package replay, no-clobber and negative source/
cache/Gram checks remain intact. Production fitting, identity derivation,
wire formats and CLI semantics are unchanged.

Scoped invocation dfe96d5fdbf64908b3b0a107bfa76872 completed exit0 with a
three-thread parent: all five command tests, strict all-target CLI Clippy,
formatting and final diff/HEAD guards pass. A first short post-change command
had no terminal test result while contending for the cache; it is not a pass.
Original hosted failures, unchanged local pass, three-thread failure and
corrected controls/log/exit are retained in
`~/Projects/Tritium/archive/verification/cli-parity-20261011`.

Exact commit-tree/canonical verification and new-source hosted CI follow
separately. This is a software fixture correction, not full P3 conversion
migration, model quality, physical-backend/performance/byte qualification,
SOTA status or release clearance. Full v1.1 and explicit human activation
requirements remain open.

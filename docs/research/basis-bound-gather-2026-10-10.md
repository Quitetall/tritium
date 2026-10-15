# Basis-bound reference gather — 2026-10-10

Code: `8dd43018cac644a0b0bcd7180b53f82625521690` (includes `3e8157fe`).
Base: `0d83bc63cc1b6df35afa7ee29cfe111790903914`.
Implements part of accepted ADR 0044 D3/D8 and plan 0055 P1.

## Implemented interface

`AdditiveView::bind_matmul()` and `bind_gather()` construct private-field
`BasisBound<AdditiveView, Matmul>` and `BasisBound<AdditiveView, Gather>`.
The reference matmul accepts only the former; new `reference_embed` accepts
only the latter. Two compile-fail doctests reject crossed consumers, while
positive public-interface tests exercise both. Binding copies only a borrowed
view and allocates no storage. The codebase-design skill guided this small
interface: transform order and validation stay inside the core module.

Gather reconstructs the additive stored row, then undoes the input-axis basis.
Hadamard is normalized/self-inverse; SignedRht is `H D` forward and `D H`
inverse, so inverse Hadamard must precede the deterministic signs. Forward
sign arithmetic is shared without changing its seed/domain/index semantics.
`dequant_row_into` is explicitly documented as returning stored-basis rows,
not logical embeddings. There is no separate basis argument at consumption.

The core path requires caller-owned output, with no allocation or scratch.
It validates the complete ID list and output length before writing, preserves
duplicates/order, and accepts empty gathers. `TernaryBackend::embed_rows`
adds the accepted tensor-level contract; compatibility defaults fail closed.
The testkit reference backend executes additive gather through the typed core
binding and dense gather by direct row copy, including zero-width dense rows.

No law admission, serialized format, numerical precision default, CUDA kernel,
production loader or model artifact changed. This is a reference implementation,
not physical backend qualification or faster inference.

## Tests and failures retained

The initial public-interface test command fails with absent inverse/gather/
binding methods (exit 101). After implementation, six of seven new gather tests
pass; the remaining fixture exceeds the existing single-plane Tensor/Free/F32
admission. The fixture is corrected to a single plane without changing the cap.
Final narrow run: 7 gather tests plus 6 existing additive vectors pass.

On committed final code:

```sh
cargo test --locked -p tritium-core -p tritium-spec -p tritium-testkit --all-targets
cargo test --locked -p tritium-core --doc
scripts/verify-gates.sh prepush
```

Foundation tests: 67 passed, 0 failed, 0 ignored. The gather vectors cover
Identity, normalized Hadamard and SignedRht, repeated/reordered IDs, logical
matmul agreement, every currently admitted law at supported plane counts,
multiple transform blocks, odd normalization exponents and rejected geometry.
The fixed vectors use group 32 and four columns; they do not establish G64/G256
admission, per-tile allocation, packed-codec or large-model correctness.

Committed doctests: 2 passed, 0 failed. Canonical prepush finishes with
`MainPID=0`, `SubState=exited`, `Result=success`, `ExecMainStatus=0`:
formatting, projection drift, actual bare-metal compilation, default and
all-features Clippy, Windows GNU cross-check and actionlint pass. ShellCheck
is absent locally and explicitly warned; run-block lint is not claimed.
Final invocation: `cf3cabe347e2412ca451054964243d2c`.
The first canonical check fails Clippy's `chunks_exact_to_as_chunks` lint in
the new vector test. A separate correction commit uses fixed-size array chunks;
the original failure log is retained, not relabeled success.

## Previous-source hosted evidence

CI `38065369024` at the base `0d83bc63` succeeds; its required bare-metal steps
pass on Ubuntu, macOS and Windows. Wheel run `38065369027` also succeeds.
Its actual public conversion probe passes at 2.51x (1.769s / 0.704s), with
the unchanged fit digest and new native/other timing observations. These runs
confirm the preceding toolchain repair and diagnostic output, not this gather
source. The older `4e42b029` 1.39x failure remains causally unexplained.

Archive: `/home/brianklam/Projects/Tritium/archive/verification/basis-bound-gather-8dd43018-20261010`.
Byte/source custody checking is not empirical or independent phase clearance.

## Remaining scope

Production load-time consumer binding and migration of all embedding/linear
callers remain for P4–P6; the current reference backend binds its owned view
when executing an operation. Reference registration on wasm/MCU, per-combination
caps, all-backend native/emulated gather and memory admission are still open.
Packed/per-tile views, streaming/shards/fuzzing, schema projections/identities,
bounded tracing evidence, unified engine/loader/runner/surfaces and deletion
remain. No P1/P2 gate or release is declared complete.

Full Stage-7, Qwen language/MTP quality/runtime/physical bytes/reproduction,
physical hardware and distributed training, serving/security, final RC packages,
audited model zoo, docs/community, independent clearance and human activation
remain required for the full v1.1 goal.

# ADR 0044 foundation integration — 2026-10-10

## Scope and source

Code commit: `a863f85df05b08cb529ee6a4f51de30a5c1357ee`.
Merge parents: current release front
`4e42b029a3c93b75d6f953b8ea48ae2a8bf1c826` and existing foundation candidate
`49bad5b7f1657da8f6bac89ad59347a790d14eb2`.

This integrates existing ADR 0044 foundation work into the current release
front, preserving the shipping feedback-policy changes. It is implementation
progress, **not completion of P0, P1, P2 or the v1.1 release**. No Qwen fitting,
large model capture, cloud job, publication or release activation ran.

Integrated modules include:

- `tritium-schema`: additive vocabulary, structural/law validation, typed
  identities, verdicts and generated JSON Schema projections.
- `tritium-core`: decoded additive views, normalized Hadamard/SignedRHT and
  reference additive matmul.
- `tritium-testkit`: frozen additive vectors and a reference backend's
  tensor-level additive/dense upload and matmul.
- `tritium-format`: decoded owned additive tensors, physical payload codec
  reference and an in-memory `.trit` envelope/CBOR/blob-dedup reference codec.
- `tritium-evidence`: registered typed events, canonical JCS hash chains,
  integrity verification and verdict aggregation that retains Unknown and
  rejects self-clearance of independent obligations.
- CLI evidence lifecycle flags, viewing and verification; schema projection
  drift checks in CI and canonical verification tiers.

Tensor-level trait compatibility defaults reject unsupported operations. They
do not yet provide ADR 0044's all-backend dense emulated tier. Decoded views are
not the final packed, zero-copy mmap layout.

## Integration repairs

The CLI merge conflict was resolved by retaining both the evidence dispatcher
and the current `--gptq-scale-refit` selection, including `InLoop`. A parser
regression combines global deterministic evidence flags with in-loop scale
refitting and decay 0.5. No quantization default changed.

The schema's B3 description now correctly says base-3 packing of five
coefficients per byte. Projection changes were produced by the owning schema
generator, not edited by hand.

An actual portability defect was reproduced: `tritium-core
--no-default-features` still enabled `tritium-schema/std`. A Linux host check
passed because that target supplies std. A bare-metal check failed with:

```text
error[E0463]: can't find crate for `std`
```

Workspace schema inheritance now disables default features; core forwards its
`std` and `alloc` features explicitly, and evidence explicitly requests schema
serde/std. The same bare-metal check passes after that repair. CI installs
`thumbv7em-none-eabihf` and runs the check; the canonical prepush/ci/release
tiers also require it rather than silently falling back to a host check.
This is **compilation evidence, not physical MCU execution or qualification**.
Other optional no_std feature combinations, including core's legacy serde
feature, are not covered by this no-default-features check.

## Checks on the clean code commit

Existing SSD build cache:
`/mnt/4tb/build/cargo/70/7e33efa4656acd`.
Host: local Intel Core i9-14900K, Linux x86_64. Default CPU checks are not GPU
execution or model-quality measurements.

```sh
RUSTC_WRAPPER= CARGO_BUILD_JOBS=2 \
CARGO_TARGET_DIR=/mnt/4tb/build/cargo/70/7e33efa4656acd \
timeout 300 cargo test --locked \
  -p tritium-schema -p tritium-core -p tritium-evidence \
  -p tritium-testkit -p tritium-format -p tritium-cli --lib --tests
```

Result: **495 passed, 0 failed, 15 ignored, 38 test binaries**. Ignored
real-model obligations are:

- CLI `repacked_tq1_model_loads_bit_identical`.
- `activation_aware_convert_beats_nearest_point`,
  `converted_model_scores_like_the_fitter_that_made_it`,
  `decayed_gptq_beats_undecayed_at_the_default_calibration`,
  `per_group_activation_scales_are_a_substitute_for_rotation`,
  `rotation_reaches_the_artifact_and_does_not_cost_quality`,
  `ternary_activations_in_the_runtime`,
  `unpadded_container_changes_the_file_and_nothing_else`.
- `convert_then_generate_from_the_directory` and
  `joint_symbols_against_the_byte_transport_on_a_real_bundle`.
- `qwen36_allocation_census`, `qwen36_artifact_error_against_the_fp_master`,
  `single_plane_floor_from_the_fp_master`,
  `stored_scale_versus_refit_scale_on_the_same_trits`, and
  `survey_scale_groups_and_plane_density`.

```sh
cargo run --locked -p tritium-schema --features schema-gen \
  --bin tritium-schema-projections -- --check
cargo check --locked -p tritium-schema -p tritium-core \
  --no-default-features --lib --target thumbv7em-none-eabihf
```

Both pass. The before/after feature trees and bare-metal diagnostics are
retained separately. An earlier host check timed out while waiting for the
build lock; that timeout is not a numerical or compilation failure.

The actual compiled CLI was invoked twice with:

```sh
tritium list-backends --evidence det --run-id foundations-integration \
  --evidence-out cli-a.jsonl
tritium list-backends --evidence det --run-id foundations-integration \
  --evidence-out cli-b.jsonl
cmp cli-a.jsonl cli-b.jsonl
tritium evidence verify cli-a.jsonl --evidence off
```

The JSONL bytes match. Verification reports one event, one span, root
`0716a05a59e12b39248c3262abced2366eb2c8391ec57c52beb2dfe78d2f01d0`.
Both invocations list the real CPU AVX2/FMA backend. The first invocation used
`cargo run`; the command sequence hit its 300-second wall timeout while the
second invocation waited behind concurrent Clippy compilation. Once terminal,
only the missing second invocation and verification were run directly with
the already-built binary. No live job was restarted on an observation timeout.
This tests command lifecycle evidence only: it is not deterministic model
execution or a semantic quantization-plan fingerprint.

The existing shipping consumers were also checked on the code commit:

```sh
cargo test --locked -p tritium-quantize --lib --tests
cargo test --locked -p tritium-nn --lib salt_fit::
cargo test --locked -p tritium-nn --test feedback_decay_legacy
```

Results: quantizer **311 passed, 0 failed, 10 ignored** across 13 binaries;
NN fitter **6 passed**, and original shipping scale/trit payload regressions
**2 passed**, both with zero failures. Quantizer ignores are the four manual
`profile_*` solver tests, `ranking_upgrades_on_pruned_losses_misallocates_planes`,
`salt_recon_error_decreases_with_bpw`, `a_pruned_joint_fit_is_not_a_single_plane_fit`,
`the_fitter_reaches_its_own_objective_at_one_plane`,
`what_each_fix_for_prefix_pruning_costs`, and
`what_frobenius_excess_does_metric_spread_buy`. They require manual profiling
or external model fixtures and were not executed here.

`scripts/verify-gates.sh prepush` finished with systemd `SubState=exited`,
`MainPID=0`, `Result=success`, `ExecMainStatus=0`. It passed formatting,
projection drift, bare-metal compilation, default/all-features workspace
Clippy and the Windows GNU cross-check. Its restricted dispatcher PATH omitted
actionlint, which the script explicitly reported as skipped. Separate
`actionlint -shellcheck <portable-binary>` over all workflows, and
`bash -n scripts/verify-gates.sh` both pass. The hosted wheel result below is
not made green by these local checks.

The host also lacks ShellCheck. This check used the official v0.11.0 portable
Linux x86_64 archive, verified against the release API's SHA-256
`8c3be12b05d5c177a04c29e3c78ce89ac86f1595681cab149b65b97c4e227198`.
The portable tool is temporary and removed with owned scratch; no global
package installation or claim that ShellCheck was preinstalled is made.

Durable source/log custody:
`/home/brianklam/Projects/Tritium/archive/verification/unified-foundations-a863f85d-20261010`.
The standalone verifier compares every changed source file to externally
selected original Git objects and checks exact checksum-inventory coverage.
Custody is not independent numerical or release approval.

## Hosted wheel obligation remains failed

Previous source `4e42b029` passed hosted CI `38061136716`, CodeQL
`38061136784`, CPU capstone `38061136782` and docs `38061136807`.
Wheel workflow `38061136994` failed its installed-wheel public PTQ parallelism
gate: **1.388913829x versus 1.5x required**. Both fitted payloads match.
The failure is retained in the archive. Profile the actual public conversion
path and runner capacity next; no cause, gate waiver or current-source wheel
qualification follows from a single timing or another source's passing run.

## Historical benchmark boundary

The imported `benches/baselines/decode_gate_4090.json` remains bound to source
`61ddf72f8f048edc90179900c874dff2c4fdd1e3`. It records GPU activity of 39%
before the run. It is historical evidence, not a new measurement of this
merge. The accepted quiet-box 360 tok/s gate and per-kernel nsys baseline
obligations remain open. The integration changes no `tritium-cuda` source;
that fact does not substitute for future runner/performance gates.

## Remaining consolidation and release obligations

1. P0: quiet decode matrix and per-kernel baseline/gate qualification.
2. P1: normative CDDL/byte-layout, Python/TS projections, tensor metadata,
   manifest/level/QuantPlan types and identity migration; full conformance
   coverage, inverse gather and basis-bound consumers; tracing evidence,
   semantic run fingerprints, receipt views and resource-bounded verification.
3. P2: streaming/mmap/sharded `.trit`, row-aligned and per-tile packed payloads,
   fuzzing, all-backend caps/emulation/native-only/memory admission and native
   overrides. The reference package still uses the legacy package-id type;
   the final new-domain identity contract is not claimed implemented.
4. P3: unified F32Accum64 engine and fitters, shared calibration, direct-fit
   allocator and shipping-entrypoint migration; original shipping parity and
   matched model-quality A/B. Shared feedback decay is only a prerequisite.
5. P4–P7: unified basis-bound loader, resident executor/runner, Qwen/MTP
   transcripts, all surface adapters, verified legacy migration and deletion.
6. Full Stage-7 producers/measurements and G64/G256 admission; Qwen language/MTP
   quality, physical-byte, runtime/residency, reproduction and NearLossless gates.
7. Physical backend/distributed/browser training, serving/fault/security/OCI
   qualification, final same-source wheels/ONNX/Colab/compatibility, audited
   model zoo, documentation/community infrastructure, second-operator checks,
   signing and explicit human activation/publication.

No missing obligation is converted into a pass by this integration.

# Browser training: cheap local checks

Date: 2026-10-09
Evidence class: developer checks only; not release qualification
Source baseline: `ab23164146bc85a8a76b6b6383788573907334cd`
Contract: ADR 0033 and private research plan 0050.

The worktree is dirty with separate Qwen regression and unrelated training/docs
work. Browser package sources were unchanged by this run. These results cannot
be admitted as clean-source, candidate-archive or physical-browser evidence.

## Executed checks

From the repository root, with a 120-second command limit:

```sh
npm --prefix packages/tritium-web run check:generated
```

Passed: browser-vector metadata, operation bindings and WebGPU kernel catalog
have no generated drift. The checker did not rewrite generated files.

From `packages/tritium-web`, each with a 120-second command limit:

```sh
./node_modules/.bin/tsc -p tsconfig.json --noEmit
env TMPDIR=/mnt/4tb/tmp node --test --test-reporter=spec \
  tests/build-wasm.test.mjs tests/npm-sbom.test.mjs \
  tests/browser-lane-producer.test.mjs
```

TypeScript passed. The Node run passed 12 tests, with zero failures, skips or
cancelled tests. Coverage includes effective Cargo target selection, clean-source
identity checks, package/SBOM identity, canonical browser-vector inventory,
fault-observation admission and the WebDriver protocol boundary. Browser
identities and traces in these tests are fixtures, not physical measurements.
The temporary source fixture was removed by the test's cleanup guard.

## Verified gaps and unexecuted gates

Plan 0050's verification cadence names two commands absent from package.json.
Both were executed with 30-second limits and failed with exit 1:

```text
npm --prefix packages/tritium-web run format:check
npm error Missing script: "format:check"

npm --prefix packages/tritium-web run lint
npm error Missing script: "lint"
```

Implement the intended checks or record an explicit canonical-plan correction;
neither command is a passing gate. Type checking is not reported as a substitute
for these missing commands.

## Subsequent implementation

The planned commands are now implemented with exact-pinned Biome 2.5.15,
following its [official installation guidance](https://biomejs.dev/installation/quick-start/).
`format:check` checks authored formatting; `format` applies it; `lint` runs the
recommended rules with warnings denied. Both checks precede the unchanged
WASM/type/session/installed-archive stages of `npm run check`.

The first actual formatter run reported 49 unformatted files. Formatting was
applied mechanically to authored files only. Lint initially reported eight
errors and 227 warnings: 223 concerned the existing bounded-index assertion
idiom. The documented policy exempts that single style rule in the existing
numeric/compiled-schedule modules explicitly named in `biome.json`, retaining
strict TypeScript and every other recommended rule. Other diagnostics were
fixed: unused imports, explicit no-return iteration callbacks, an optional
abort-signal lookup, an `unknown` result at the adapter-validation boundary and
a real pending Promise in the test device-loss stub. No numerical tolerance,
vector, generated binding/kernel source or hardware gate was changed.

Formatting, warning-denying lint, generated-file checks and strict TypeScript
passed after these fixes. The same 12 tooling/producer tests passed again.
Four new CLI regression tests passed: they demonstrate actual formatter repair,
unreachable-code rejection, warning-denying pipeline retention and rejection of
non-null assertions outside the explicitly exempt modules. Their disposable
fixtures are cleaned after execution.

An attempted TypeScript syntax-tree comparison could not run against the
installed compiler module interface. A subsequent esbuild normalized-output
comparison found identical JavaScript for 36 changed implementation/test files;
seven emitted-code differences match the explicit lint fixes above. All six
changed declaration files normalize to identical Biome-formatted content.
Generated bindings, kernels and vector inventory have no Git diff. These are
bounded mechanical-change checks, not numerical or physical qualification.

Full package verification remains pending until its source/WASM/session/archive
checks execute; the original missing-script results above describe the
pre-implementation state, not current command absence.

The first normal commit-tree gate passed formatting, lint and generated checks,
then stopped with `Error: spawn wasm-bindgen ENOENT` after Cargo's WASM build.
The pinned CLI was listed in Cargo's historical install inventory but its
executable was absent. A disposable copy of the exact upstream 0.2.126 Linux
release was downloaded for the retry; its archive SHA-256 is
`064948d58e2d6c0a745216477a639ba696216d6309aaa902939d1b865b1d869d`,
matching GitHub's release-asset digest and the upstream checksum file. This is
local build tooling, not a published Tritium candidate or qualification receipt.

The second normal commit-tree attempt built the actual WASM guest, passed strict
types and ran 149 Node tests: 148 passed, one failed, none skipped. The failing
browser-producer Git fixture inherited the outer `git commit --only` temporary
index and reported `error: invalid object ... for '.cargo/config.toml'`.
The installed-archive stage did not execute.

A small real-Git hook regression reproduced `error: invalid object ... for
'packages/tritium-web/package.json'` before the fix. The hook now clears Git's
repository-local environment only in the independent snapshot's subshell,
after capturing the exact staged diff. Nested tests cannot reuse the caller's
temporary index. The regression then passed and verifies that unrelated staged
content remains staged, unchanged in the committed tree, and all fixture scratch
is removed. Cargo/npm stand-ins in this isolated regression test hook custody,
not package functionality. Full browser verification must still be rerun.

All three physical
Chrome/Firefox/Safari lanes, no-readback tracing, fault injection and tutorial
timing remain separately required. No registry publication occurred.

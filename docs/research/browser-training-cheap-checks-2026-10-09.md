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

The package's `dist` directory is absent. Full `npm run check`, the actual WASM
guest build/corpus, installed npm archive, complete session tests and native
artifact reload were not executed in this run. They need the normal build path,
which shares the cache with an existing managed push. All three physical
Chrome/Firefox/Safari lanes, no-readback tracing, fault injection and tutorial
timing remain separately required. No registry publication occurred.

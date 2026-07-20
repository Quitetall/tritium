import assert from "node:assert/strict";
import test from "node:test";

import {
  canonicalTrainingManifestJson,
  parseTrainingManifest,
  webGpuCandidateModulesForOperationV1,
  webGpuKernelCandidateBundleV1,
  WebTrainingError,
} from "../dist/index.js";

test("WGSL candidate dependency index keys every frozen tensor operation", () => {
  const manifest = parseTrainingManifest(canonicalTrainingManifestJson());
  const expected = manifest.operations
    .map((operation) => operation.id)
    .filter((operation) => !operation.startsWith("lifecycle."));
  const bundle = webGpuKernelCandidateBundleV1();
  assert.equal(bundle.schemaId, "tritium.webgpu_kernel_candidate_bundle");
  assert.equal(bundle.schemaVersion, 1);
  assert.match(bundle.bundleSha256, /^[0-9a-f]{64}$/);
  assert.deepEqual(Object.keys(bundle.candidateOperationModuleDependencies), expected);
  assert.equal(expected.length, 31);
  assert.ok(Object.isFrozen(bundle));
  assert.ok(Object.isFrozen(bundle.modules));
  assert.ok(Object.isFrozen(bundle.candidateOperationModuleDependencies));

  for (const operation of expected) {
    const dependencies = bundle.candidateOperationModuleDependencies[operation];
    assert.ok(dependencies.length > 0, operation);
    assert.ok(Object.isFrozen(dependencies), operation);
    const modules = webGpuCandidateModulesForOperationV1(operation);
    assert.ok(modules.length > 0, operation);
    assert.ok(Object.isFrozen(modules), operation);
    for (const module of modules) {
      assert.equal(bundle.modules[module.id], module);
      assert.match(module.sha256, /^[0-9a-f]{64}$/);
      assert.ok(Object.isFrozen(module));
    }
  }
  assert.deepEqual(
    bundle.candidateOperationModuleDependencies["optimizer.adamw"],
    ["adamw", "adamw_terms", "adamw_variance", "adamw_finish"],
  );
  assert.deepEqual(
    bundle.candidateOperationModuleDependencies["optimizer.int8_adamw"],
    ["int8_adamw"],
  );
  assert.deepEqual(
    bundle.candidateOperationModuleDependencies["graph.salt_ste"],
    ["salt", "pointwise"],
  );
  assert.deepEqual(
    bundle.candidateOperationModuleDependencies["graph.concat_cols"],
    ["concat", "pointwise"],
  );
});

test("WGSL candidate bundle identity is stable and unknown operations fail closed", () => {
  assert.equal(webGpuKernelCandidateBundleV1(), webGpuKernelCandidateBundleV1());
  assert.throws(
    () => webGpuCandidateModulesForOperationV1("graph.not_real"),
    (error) => {
      assert.ok(error instanceof WebTrainingError);
      assert.equal(error.code, "capability_mismatch");
      return true;
    },
  );
});

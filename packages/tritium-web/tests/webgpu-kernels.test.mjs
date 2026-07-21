import assert from "node:assert/strict";
import test from "node:test";

import {
  canonicalTrainingManifestJson,
  parseTrainingManifest,
  webGpuDispatchCatalogV1,
  webGpuDispatchFormV1,
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
      assert.ok(Object.isFrozen(module.bindings));
      assert.ok(Object.isFrozen(module.entryPoints));
      assert.ok(Object.isFrozen(module.entryPointBindings));
      assert.ok(module.bindings.length > 0);
      assert.ok(module.bindings.every((binding) => binding.group === 0));
      assert.equal(
        new Set(module.bindings.map((binding) => binding.binding)).size,
        module.bindings.length,
      );
    }
  }
  assert.deepEqual(
    bundle.candidateOperationModuleDependencies["optimizer.adamw"],
    ["adamw", "adamw_terms", "adamw_variance", "adamw_finish"],
  );
  assert.deepEqual(
    bundle.candidateOperationModuleDependencies["optimizer.int8_adamw"],
    ["byte_codec", "int8_adamw"],
  );
  assert.deepEqual(
    bundle.modules.int8_adamw.entryPointBindings.dequantize.map((binding) => binding.binding),
    [0, 3, 4, 5, 6],
  );
  assert.deepEqual(
    bundle.modules.int8_adamw.entryPointBindings.square_variance.map(
      (binding) => binding.binding,
    ),
    [0, 4],
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

test("WebGPU dispatch catalog covers all 57 frozen execution forms", () => {
  const manifest = parseTrainingManifest(canonicalTrainingManifestJson());
  const expected = [];
  for (const operation of manifest.operations) {
    if (operation.category === "lifecycle") continue;
    const executions = operation.category === "optimizer"
      ? ["step"]
      : operation.vjp === "first_order"
        ? ["forward", "vjp"]
        : ["forward"];
    for (const execution of executions) expected.push(`${operation.id}|${execution}`);
  }
  const catalog = webGpuDispatchCatalogV1();
  assert.equal(catalog.schemaId, "tritium.webgpu_dispatch_catalog");
  assert.equal(catalog.schemaVersion, 1);
  assert.match(catalog.sha256, /^[0-9a-f]{64}$/);
  assert.deepEqual(Object.keys(catalog.forms), expected);
  assert.equal(expected.length, 57);
  assert.ok(Object.isFrozen(catalog));
  assert.ok(Object.isFrozen(catalog.forms));
  for (const key of expected) {
    const form = catalog.forms[key];
    assert.ok(Object.isFrozen(form));
    assert.ok(Object.isFrozen(form.stages));
    assert.ok(form.stages.length > 0, key);
    for (const stage of form.stages) {
      assert.ok(Object.isFrozen(stage));
      const module = webGpuKernelCandidateBundleV1().modules[stage.moduleId];
      assert.ok(module);
      assert.ok(module.entryPoints[stage.entryPoint]);
    }
  }

  assert.deepEqual(
    webGpuDispatchFormV1("graph.salt_ste", "vjp").stages,
    [{ moduleId: "pointwise", entryPoint: "main", selector: 0,
      dispatch: "linear_output_64", repeat: "once" }],
  );
  assert.equal(webGpuDispatchFormV1("graph.add", "vjp").stages.length, 2);
  assert.equal(
    webGpuDispatchFormV1("graph.concat_cols", "vjp").stages[0].repeat,
    "per_output",
  );
  assert.deepEqual(
    webGpuDispatchFormV1("optimizer.int8_adamw", "step").stages.map(
      (stage) => stage.entryPoint,
    ),
    [
      "unpack",
      "unpack",
      "dequantize",
      "square_variance",
      "products",
      "finish_products",
      "finish_variance",
      "update_parameter",
      "reduce_scales",
      "quantize",
      "pack",
      "pack",
    ],
  );
  assert.throws(
    () => webGpuDispatchFormV1("graph.add", "step"),
    (error) => error instanceof WebTrainingError && error.code === "capability_mismatch",
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

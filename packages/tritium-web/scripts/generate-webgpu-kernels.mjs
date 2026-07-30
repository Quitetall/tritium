import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(packageRoot, "../..");
const shaderRoot = resolve(repoRoot, "crates/tritium-wgpu/src");
const manifestPath = resolve(repoRoot, "spec/training/v2/manifest.json");
const outputPath = resolve(packageRoot, "src/generated-webgpu-kernels.ts");
const dispatchPath = resolve(repoRoot, "spec/training/v2/webgpu-dispatch-v2.json");

const OPERATION_MODULES = Object.freeze({
  "graph.ste_surrogate": ["pointwise"],
  "graph.salt_ste": ["salt", "pointwise"],
  "graph.lsq_ste": ["pointwise"],
  "graph.fsq": ["fsq"],
  "graph.dense_matmul": ["pointwise"],
  "graph.ternary_matmul": ["pointwise"],
  "graph.embedding_gather": ["embedding"],
  "graph.transpose": ["pointwise"],
  "graph.slice_cols": ["pointwise"],
  "graph.concat_cols": ["concat", "pointwise"],
  "graph.detach": ["pointwise"],
  "graph.scale_const": ["pointwise"],
  "graph.bias": ["pointwise"],
  "graph.add": ["pointwise"],
  "graph.mul": ["pointwise"],
  "graph.conv1d": ["conv"],
  "graph.conv2d": ["conv"],
  "graph.attention": ["attention"],
  "graph.relu2": ["pointwise"],
  "graph.silu": ["pointwise"],
  "graph.causal_mask": ["pointwise"],
  "graph.rope": ["rope"],
  "graph.rmsnorm": ["pointwise"],
  "graph.softmax": ["pointwise"],
  "loss.mse": ["pointwise"],
  "loss.softmax_cross_entropy": ["softmax_xent"],
  "loss.topk_knowledge_distillation": ["topk_kd"],
  "optimizer.sgd": ["pointwise"],
  "optimizer.adamw": ["adamw", "adamw_terms", "adamw_variance", "adamw_finish"],
  "optimizer.cautious_adamw": [
    "adamw",
    "adamw_terms",
    "adamw_variance",
    "cautious_adamw_mask",
    "cautious_adamw_lr",
    "cautious_adamw_rescale",
    "cautious_adamw_finish",
  ],
  "optimizer.int8_adamw": ["byte_codec", "int8_adamw"],
  "optimizer.muon": ["muon"],
});

// WebGPU auto layouts contain only resources statically used by one entry point.
// Multi-entry modules therefore need an explicit per-entry subset; single-entry
// modules safely default to every source-declared binding.
const ENTRY_POINT_BINDINGS = Object.freeze({
  int8_adamw: Object.freeze({
    dequantize: [0, 3, 4, 5, 6],
    square_variance: [0, 4],
    products: [0, 1, 2, 3, 4, 7, 8],
    finish_products: [0, 2, 3, 7, 8],
    finish_variance: [0, 4, 8],
    update_parameter: [0, 1, 3, 4],
    reduce_scales: [0, 3, 4, 5, 6],
    quantize: [0, 3, 4, 5, 6],
  }),
});

const stage = (
  moduleId,
  dispatch,
  selector = null,
  entryPoint = "main",
  repeat = "once",
) => Object.freeze({ moduleId, entryPoint, selector, dispatch, repeat });
const pw = (selector, dispatch = "linear_output_64", repeat = "once") =>
  stage("pointwise", dispatch, selector, "main", repeat);
const one = (moduleId, dispatch) => stage(moduleId, dispatch);

const DISPATCH_FORMS = Object.freeze({
  "graph.ste_surrogate|forward": [pw(33)],
  "graph.ste_surrogate|vjp": [pw(34), pw(1)],
  "graph.salt_ste|forward": [one("salt", "single")],
  "graph.salt_ste|vjp": [pw(0)],
  "graph.lsq_ste|forward": [pw(35)],
  "graph.lsq_ste|vjp": [pw(36), pw(37)],
  "graph.fsq|forward": [one("fsq", "linear_input_64")],
  "graph.fsq|vjp": [one("fsq", "linear_input_64")],
  "graph.dense_matmul|forward": [pw(26)],
  "graph.dense_matmul|vjp": [pw(27), pw(28)],
  "graph.ternary_matmul|forward": [pw(29)],
  "graph.ternary_matmul|vjp": [pw(30), pw(31), pw(32)],
  "graph.transpose|forward": [pw(22)],
  "graph.transpose|vjp": [pw(23)],
  "graph.embedding_gather|forward": [one("embedding", "linear_output_64")],
  "graph.embedding_gather|vjp": [one("embedding", "linear_output_64")],
  "graph.slice_cols|forward": [pw(24)],
  "graph.slice_cols|vjp": [pw(25)],
  "graph.concat_cols|forward": [one("concat", "linear_output_64")],
  "graph.concat_cols|vjp": [pw(24, "linear_output_64", "per_output")],
  "graph.detach|forward": [pw(0)],
  "graph.detach|vjp": [pw(1)],
  "graph.scale_const|forward": [pw(2)],
  "graph.scale_const|vjp": [pw(2)],
  "graph.bias|forward": [pw(18)],
  "graph.bias|vjp": [pw(19), pw(20)],
  "graph.add|forward": [pw(3)],
  "graph.add|vjp": [pw(0), pw(0)],
  "graph.mul|forward": [pw(4)],
  "graph.mul|vjp": [pw(4), pw(4)],
  "graph.conv1d|forward": [one("conv", "single")],
  "graph.conv1d|vjp": [one("conv", "single")],
  "graph.conv2d|forward": [one("conv", "single")],
  "graph.conv2d|vjp": [one("conv", "single")],
  "graph.relu2|forward": [pw(5)],
  "graph.relu2|vjp": [pw(6)],
  "graph.silu|forward": [pw(7)],
  "graph.silu|vjp": [pw(8)],
  "graph.rmsnorm|forward": [pw(13)],
  "graph.rmsnorm|vjp": [pw(14), pw(15, "linear_primary_input_64")],
  "graph.softmax|forward": [pw(11)],
  "graph.softmax|vjp": [pw(12)],
  "graph.causal_mask|forward": [pw(9)],
  "graph.causal_mask|vjp": [pw(10)],
  "graph.rope|forward": [one("rope", "rope_pairs_64")],
  "graph.rope|vjp": [one("rope", "rope_pairs_64")],
  "graph.attention|forward": [one("attention", "single")],
  "graph.attention|vjp": [one("attention", "single")],
  "loss.mse|forward": [pw(16, "linear_primary_input_64")],
  "loss.mse|vjp": [pw(17)],
  "loss.softmax_cross_entropy|forward": [one("softmax_xent", "single")],
  "loss.softmax_cross_entropy|vjp": [one("softmax_xent", "single")],
  "loss.topk_knowledge_distillation|forward": [one("topk_kd", "single")],
  "loss.topk_knowledge_distillation|vjp": [one("topk_kd", "single")],
  "optimizer.sgd|step": [pw(21, "linear_parameter_64")],
  "optimizer.adamw|step": [
    one("adamw", "linear_parameter_64"),
    one("adamw_terms", "linear_parameter_64"),
    one("adamw_variance", "linear_parameter_64"),
    one("adamw_finish", "linear_parameter_64"),
  ],
  "optimizer.cautious_adamw|step": [
    one("adamw", "linear_parameter_64"),
    one("adamw_terms", "linear_parameter_64"),
    one("adamw_variance", "linear_parameter_64"),
    one("cautious_adamw_mask", "linear_parameter_64"),
    one("cautious_adamw_lr", "linear_parameter_64"),
    one("cautious_adamw_rescale", "linear_parameter_64"),
    one("cautious_adamw_finish", "linear_parameter_64"),
  ],
  "optimizer.int8_adamw|step": [
    stage("byte_codec", "linear_parameter_64", null, "unpack"),
    stage("byte_codec", "linear_parameter_64", null, "unpack"),
    stage("int8_adamw", "linear_parameter_64", null, "dequantize"),
    stage("int8_adamw", "linear_parameter_64", null, "square_variance"),
    stage("int8_adamw", "linear_parameter_64", null, "products"),
    stage("int8_adamw", "linear_parameter_64", null, "finish_products"),
    stage("int8_adamw", "linear_parameter_64", null, "finish_variance"),
    stage("int8_adamw", "linear_parameter_64", null, "update_parameter"),
    stage("int8_adamw", "optimizer_blocks_256", null, "reduce_scales"),
    stage("int8_adamw", "linear_parameter_64", null, "quantize"),
    stage("byte_codec", "packed_words_64", null, "pack"),
    stage("byte_codec", "packed_words_64", null, "pack"),
  ],
  "optimizer.muon|step": [one("muon", "single")],
});

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function shaderMetadata(source, id) {
  const code = source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/.*$/gm, "");
  const bindings = [];
  const bindingKeys = new Set();
  for (const match of code.matchAll(
    /((?:@\w+(?:\s*\([^)]*\))?\s*)+)var<\s*([^>]+?)\s*>/g,
  )) {
    const group = /@group\s*\(\s*(\d+)\s*\)/.exec(match[1]);
    const binding = /@binding\s*\(\s*(\d+)\s*\)/.exec(match[1]);
    if (group === null && binding === null) continue;
    if (group === null || binding === null) {
      throw new Error(`${id} has an incomplete resource binding declaration`);
    }
    const [addressSpace, access = null] = match[2].split(",").map((part) => part.trim());
    if (!(["uniform", "storage"].includes(addressSpace)) ||
        !(access === null || ["read", "read_write"].includes(access))) {
      throw new Error(`${id} has an unsupported resource binding address/access mode`);
    }
    const key = `${group[1]}|${binding[1]}`;
    if (bindingKeys.has(key)) throw new Error(`${id} duplicates binding ${key}`);
    bindingKeys.add(key);
    bindings.push(Object.freeze({
      group: Number(group[1]),
      binding: Number(binding[1]),
      addressSpace,
      access,
    }));
  }
  const entryPoints = {};
  for (const match of code.matchAll(
    /((?:@\w+(?:\s*\([^)]*\))?\s*)+)fn\s+(\w+)/g,
  )) {
    if (!/@compute(?:\s|$)/.test(match[1])) continue;
    const workgroup = /@workgroup_size\s*\(([^)]+)\)/.exec(match[1]);
    if (workgroup === null) throw new Error(`${id}.${match[2]} lacks @workgroup_size`);
    const dimensions = workgroup[1].split(",").map((part) => Number(part.trim()));
    if (dimensions.length > 3 ||
        dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)) {
      throw new Error(`${id} has a non-literal WebGPU workgroup size`);
    }
    if (Object.hasOwn(entryPoints, match[2])) {
      throw new Error(`${id} duplicates compute entry point ${match[2]}`);
    }
    while (dimensions.length < 3) dimensions.push(1);
    entryPoints[match[2]] = Object.freeze(dimensions);
  }
  const groupAttributeCount = [...code.matchAll(/@group\s*\(/g)].length;
  const bindingAttributeCount = [...code.matchAll(/@binding\s*\(/g)].length;
  const computeAttributeCount = [...code.matchAll(/@compute(?:\s|$)/g)].length;
  if (bindings.length === 0 || Object.keys(entryPoints).length === 0 ||
      groupAttributeCount !== bindings.length || bindingAttributeCount !== bindings.length ||
      computeAttributeCount !== Object.keys(entryPoints).length) {
    throw new Error(`${id} lacks source-derived binding or entry-point metadata`);
  }
  return { bindings: Object.freeze(bindings), entryPoints: Object.freeze(entryPoints) };
}

function generatedSource(modules, operations, forms, bundleDigest, catalogDigest) {
  const moduleEntries = modules.map(({
    id, source, digest, bindings, entryPoints, entryPointBindings,
  }) =>
    `  ${JSON.stringify(id)}: Object.freeze({\n` +
      `    id: ${JSON.stringify(id)},\n` +
      `    sha256: ${JSON.stringify(digest)},\n` +
      `    source: ${JSON.stringify(source)},\n` +
      `    bindings: Object.freeze([${bindings.map((binding) =>
        `Object.freeze(${JSON.stringify(binding)})`,
      ).join(",")}]),\n` +
      `    entryPointBindings: Object.freeze({${Object.entries(entryPointBindings).map(
        ([entryPoint, values]) => `${JSON.stringify(entryPoint)}: Object.freeze([${values.map(
          (binding) => `Object.freeze(${JSON.stringify(binding)})`,
        ).join(",")}])`,
      ).join(",")}}),\n` +
      `    entryPoints: Object.freeze({${Object.entries(entryPoints).map(
        ([entryPoint, workgroupSize]) =>
          `${JSON.stringify(entryPoint)}: Object.freeze(${JSON.stringify(workgroupSize)}) as ` +
          "readonly [number, number, number]",
      ).join(",")}}),\n` +
      "  }),",
  );
  const operationEntries = operations.map(({ operation, moduleIds }) =>
    `  ${JSON.stringify(operation)}: Object.freeze(${JSON.stringify(moduleIds)}),`,
  );
  const formEntries = forms.map((form) =>
    `  ${JSON.stringify(`${form.operation}|${form.execution}`)}: Object.freeze({` +
      ` operation: ${JSON.stringify(form.operation)}, execution: ${JSON.stringify(form.execution)},` +
      ` stages: Object.freeze([${form.stages.map((value) =>
        `Object.freeze(${JSON.stringify(value)})`,
      ).join(",")}]) }),`,
  );
  return `// @generated by scripts/generate-webgpu-kernels.mjs; do not edit.\n` +
    `export const WEBGPU_KERNEL_BUNDLE_SHA256_V1 = ${JSON.stringify(bundleDigest)} as const;\n\n` +
    `export const WEBGPU_DISPATCH_CATALOG_SHA256_V1 = ${JSON.stringify(catalogDigest)} as const;\n\n` +
    "export const WEBGPU_KERNEL_MODULES_V1 = Object.freeze({\n" +
    `${moduleEntries.join("\n")}\n` +
    "});\n\n" +
    "export const WEBGPU_OPERATION_MODULE_DEPENDENCIES_V1 = Object.freeze({\n" +
    `${operationEntries.join("\n")}\n` +
    "});\n\n" +
    "export const WEBGPU_DISPATCH_FORMS_V1 = Object.freeze({\n" +
    `${formEntries.join("\n")}\n` +
    "});\n";
}

async function generate() {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  const tensorOperations = manifest.operations
    .map((operation) => operation.id)
    .filter((operation) => !operation.startsWith("lifecycle."));
  const mapped = Object.keys(OPERATION_MODULES);
  if (
    tensorOperations.length !== 32 ||
    mapped.length !== tensorOperations.length ||
    tensorOperations.some((operation) => !mapped.includes(operation)) ||
    mapped.some((operation) => !tensorOperations.includes(operation))
  ) {
    throw new Error(
      "WebGPU candidate dependency index must key the 32 frozen tensor operations",
    );
  }

  const moduleIds = [...new Set(mapped.flatMap((operation) => OPERATION_MODULES[operation]))].sort();
  const modules = [];
  for (const id of moduleIds) {
    const path = resolve(shaderRoot, `${id}.wgsl`);
    const source = await readFile(path, "utf8");
    const metadata = shaderMetadata(source, id);
    const configured = ENTRY_POINT_BINDINGS[id];
    if (configured !== undefined &&
        (Object.keys(configured).length !== Object.keys(metadata.entryPoints).length ||
          Object.keys(metadata.entryPoints).some((entryPoint) => configured[entryPoint] === undefined))) {
      throw new Error(`${id} entry-point binding map drifted from source`);
    }
    const byId = new Map(metadata.bindings.map((binding) => [binding.binding, binding]));
    const entryPointBindings = Object.fromEntries(
      Object.keys(metadata.entryPoints).map((entryPoint) => {
        const ids = configured?.[entryPoint] ?? metadata.bindings.map((binding) => binding.binding);
        if (new Set(ids).size !== ids.length || ids.some((binding) => !byId.has(binding))) {
          throw new Error(`${id}.${entryPoint} references an unknown or duplicate binding`);
        }
        return [entryPoint, ids.map((binding) => byId.get(binding))];
      }),
    );
    modules.push({
      id, source, digest: sha256(source), ...metadata, entryPointBindings,
    });
  }
  const operations = tensorOperations.map((operation) => ({
    operation,
    moduleIds: [...new Set(OPERATION_MODULES[operation])],
  }));
  const forms = [];
  for (const descriptor of manifest.operations) {
    if (descriptor.category === "lifecycle") continue;
    const executions = descriptor.category === "optimizer"
      ? ["step"]
      : descriptor.vjp === "first_order"
        ? ["forward", "vjp"]
        : ["forward"];
    for (const execution of executions) {
      const key = `${descriptor.id}|${execution}`;
      const stages = DISPATCH_FORMS[key];
      if (stages === undefined) throw new Error(`missing WebGPU dispatch form ${key}`);
      forms.push({ operation: descriptor.id, execution, stages });
    }
  }
  if (forms.length !== 59 || Object.keys(DISPATCH_FORMS).length !== forms.length) {
    throw new Error("WebGPU dispatch catalog must contain exactly 59 execution forms");
  }
  const moduleById = new Map(modules.map((module) => [module.id, module]));
  for (const form of forms) {
    for (const value of form.stages) {
      const module = moduleById.get(value.moduleId);
      if (module === undefined) {
        throw new Error(`${form.operation}|${form.execution} references ${value.moduleId}`);
      }
      if (module.entryPoints[value.entryPoint] === undefined) {
        throw new Error(
          `${form.operation}|${form.execution} references missing entry point ` +
          `${value.moduleId}.${value.entryPoint}`,
        );
      }
    }
  }
  const bundleDigest = sha256(
    modules.map(({ id, digest }) => `module\0${id}\0${digest}\n`).join("") +
      operations.map(({ operation, moduleIds }) =>
        `operation\0${operation}\0${moduleIds.join("\0")}\n`,
      ).join(""),
  );
  const catalog = {
    schema_id: "tritium.webgpu_dispatch_catalog",
    schema_version: 2,
    forms,
  };
  const catalogJson = `${JSON.stringify(catalog, null, 2)}\n`;
  const catalogDigest = sha256(catalogJson);
  return {
    source: generatedSource(modules, operations, forms, bundleDigest, catalogDigest),
    catalogJson,
  };
}

const expected = await generate();
if (process.argv.includes("--check")) {
  const actual = await readFile(outputPath, "utf8").catch(() => "");
  const actualCatalog = await readFile(dispatchPath, "utf8").catch(() => "");
  if (actual !== expected.source || actualCatalog !== expected.catalogJson) {
    throw new Error(
      "generated WebGPU candidate bundle is stale; run npm run generate:webgpu-kernels",
    );
  }
} else {
  await writeFile(outputPath, expected.source);
  await writeFile(dispatchPath, expected.catalogJson);
}

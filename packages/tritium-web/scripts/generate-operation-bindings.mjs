import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repository = resolve(root, "../..");
const manifestPath = resolve(repository, "spec/training/v1/manifest.json");
const vectorsPath = resolve(repository, "spec/training/v1/vectors/v1.json");
const outputPath = resolve(root, "src/operation-bindings.ts");

const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
const vectors = JSON.parse(await readFile(vectorsPath, "utf8"));

const successful = new Map();
for (const fixture of vectors.cases) {
  if (fixture.expected.kind !== "success") continue;
  const key = `${fixture.operation}|${fixture.execution}`;
  const binding = {
    inputs: fixture.inputs.map((input) => input.name),
    attributes: fixture.attributes.map((attribute) => ({
      name: attribute.name,
      kind: attribute.type,
    })),
    outputs: fixture.expected.outputs.map((output) => output.name),
  };
  const previous = successful.get(key);
  if (
    previous !== undefined &&
    JSON.stringify(previous) !== JSON.stringify(binding)
  ) {
    throw new Error(`canonical roles drift within ${key}`);
  }
  successful.set(key, binding);
}

const registry = {};
for (const descriptor of manifest.operations) {
  if (descriptor.category === "lifecycle") continue;
  const executions =
    descriptor.category === "optimizer"
      ? ["step"]
      : descriptor.vjp === "first_order"
        ? ["forward", "vjp"]
        : ["forward"];
  const entry = {};
  for (const execution of executions) {
    const key = `${descriptor.id}|${execution}`;
    const binding = successful.get(key);
    if (binding === undefined) {
      throw new Error(`missing successful canonical binding ${key}`);
    }
    entry[execution] = binding;
    successful.delete(key);
  }
  registry[descriptor.id] = entry;
}

for (const key of successful.keys()) {
  if (!key.startsWith("lifecycle.")) {
    throw new Error(`unregistered canonical binding ${key}`);
  }
}

const lines = [
  "// Generated from spec/training/v1/manifest.json and vectors/v1.json.",
  "// Run `npm run generate:bindings`; manual edits fail `npm run check`.",
  "",
  "export const PORTABLE_OPERATION_BINDINGS_V1 = " +
    JSON.stringify(registry, null, 2) +
    " as const;",
  "",
];
const generated = lines.join("\n");

if (process.argv.includes("--check")) {
  const current = await readFile(outputPath, "utf8").catch(() => "");
  if (current !== generated) {
    throw new Error("src/operation-bindings.ts is stale; run npm run generate:bindings");
  }
} else {
  await writeFile(outputPath, generated);
}

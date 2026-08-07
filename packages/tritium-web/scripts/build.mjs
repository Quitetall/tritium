import { cp, mkdir, readFile, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { blake3 } from "@noble/hashes/blake3.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import { build } from "esbuild";

import { buildPortableWasm } from "./build-wasm.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const output = resolve(root, "dist");
const canonicalVectors = await readFile(resolve(
  root,
  "../../crates/tritium-spec/data/training/v2/vectors/v2.json",
));
const canonicalVectorDigest =
  "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7";
if (bytesToHex(blake3(canonicalVectors)) !== canonicalVectorDigest) {
  throw new Error("canonical training vector bytes differ from the frozen V2 digest");
}

await rm(output, { force: true, recursive: true });
await mkdir(output, { recursive: true });
await buildPortableWasm(output);
await build({
  bundle: true,
  entryPoints: {
    index: resolve(root, "src/index.ts"),
    qualification: resolve(root, "src/webgpu-conformance.ts"),
  },
  format: "esm",
  legalComments: "none",
  minify: false,
  outdir: output,
  platform: "browser",
  sourcemap: true,
  // Keep production stack mapping without embedding the TypeScript source tree.
  sourcesContent: false,
  target: ["es2022"],
  define: {
    __TRITIUM_TRAINING_VECTORS_V2_JSON__: JSON.stringify(
      canonicalVectors.toString("utf8"),
    ),
  },
});
await cp(resolve(root, "src/index.d.ts"), resolve(output, "index.d.ts"));
await cp(
  resolve(root, "src/qualification.d.ts"),
  resolve(output, "qualification.d.ts"),
);
await cp(
  resolve(root, "src/lifecycle-types.d.ts"),
  resolve(output, "lifecycle-types.d.ts"),
);
await cp(
  resolve(root, "src/portable-state-types.d.ts"),
  resolve(output, "portable-state-types.d.ts"),
);
await cp(
  resolve(root, "src/portable-schedule-types.d.ts"),
  resolve(output, "portable-schedule-types.d.ts"),
);
await cp(resolve(root, "src/payload-types.d.ts"), resolve(output, "payload-types.d.ts"));
await cp(resolve(root, "src/portable.d.ts"), resolve(output, "portable.d.ts"));
await cp(resolve(root, "../../LICENSE"), resolve(output, "LICENSE"));
await cp(resolve(root, "../../NOTICE"), resolve(output, "NOTICE"));

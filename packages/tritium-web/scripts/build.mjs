import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

import { buildPortableWasm } from "./build-wasm.mjs";
import { canonicalBrowserVectorMetadataV1 } from "./canonical-browser-vector-metadata.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const output = resolve(root, "dist");
const { sourceBytes: canonicalVectors } = await canonicalBrowserVectorMetadataV1();

await rm(output, { force: true, recursive: true });
await mkdir(output, { recursive: true });
await buildPortableWasm(output);
await build({
  bundle: true,
  entryPoints: {
    index: resolve(root, "src/index.ts"),
    qualification: resolve(root, "src/qualification.ts"),
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

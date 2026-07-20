import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

import { buildPortableWasm } from "./build-wasm.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const output = resolve(root, "dist");

await rm(output, { force: true, recursive: true });
await mkdir(output, { recursive: true });
await buildPortableWasm(output);
await build({
  bundle: true,
  entryPoints: [resolve(root, "src/index.ts")],
  format: "esm",
  legalComments: "none",
  minify: false,
  outfile: resolve(output, "index.js"),
  platform: "browser",
  sourcemap: true,
  target: ["es2022"],
});
await cp(resolve(root, "src/index.d.ts"), resolve(output, "index.d.ts"));
await cp(
  resolve(root, "src/lifecycle-types.d.ts"),
  resolve(output, "lifecycle-types.d.ts"),
);
await cp(
  resolve(root, "src/portable-state-types.d.ts"),
  resolve(output, "portable-state-types.d.ts"),
);
await cp(resolve(root, "src/portable.d.ts"), resolve(output, "portable.d.ts"));
await cp(resolve(root, "../../LICENSE"), resolve(output, "LICENSE"));
await cp(resolve(root, "../../NOTICE"), resolve(output, "NOTICE"));

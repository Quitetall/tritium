import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { canonicalBrowserVectorMetadataV1 } from "./canonical-browser-vector-metadata.mjs";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputPath = resolve(
  packageRoot,
  "../../scripts/data/browser-training-vector-inventory-v1.json",
);
const { metadata } = await canonicalBrowserVectorMetadataV1();
const generated = `${JSON.stringify(metadata, null, 2)}\n`;

if (process.argv.includes("--check")) {
  const current = await readFile(outputPath, "utf8").catch(() => "");
  if (current !== generated) {
    throw new Error(
      "browser vector metadata is stale; run npm run generate:browser-vectors",
    );
  }
} else {
  await writeFile(outputPath, generated);
}

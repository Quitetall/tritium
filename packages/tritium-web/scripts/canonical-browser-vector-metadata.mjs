import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { blake3 } from "@noble/hashes/blake3.js";
import { bytesToHex } from "@noble/hashes/utils.js";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repository = resolve(packageRoot, "../..");
const manifestDigest = "9093a1a7f9a3422c399943782aadf4df6b11833cf2253db0db56ff2d9dedb098";
const vectorDigest = "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7";

function fail(message) {
  throw new Error(`canonical browser vector metadata failed: ${message}`);
}

function deriveCases(corpus) {
  if (corpus?.schema_id !== "tritium.training_vectors" ||
      corpus.schema_version !== 2 || corpus.manifest_digest !== manifestDigest ||
      !Array.isArray(corpus.cases) || corpus.cases.length !== 117) {
    fail("vector corpus identity differs");
  }
  const cases = corpus.cases.map((item, index) => {
    if (typeof item?.case_id !== "string" || typeof item.operation !== "string" ||
        typeof item.expected !== "object" || item.expected === null ||
        !["success", "error"].includes(item.expected.kind)) {
      fail(`vector case ${index} is malformed`);
    }
    const invalid = item.expected.kind === "error";
    const implementation = invalid
      ? "wasm-validation"
      : item.operation.startsWith("lifecycle.") ? "wasm-codec" : "webgpu";
    const scratchBytesMax = invalid ? null : item.expected.scratch_bytes_max;
    if (scratchBytesMax !== null &&
        (!Number.isSafeInteger(scratchBytesMax) || scratchBytesMax < 0)) {
      fail(`vector case ${item.case_id} scratch bound is malformed`);
    }
    return { caseId: item.case_id, implementation, scratchBytesMax };
  });
  if (new Set(cases.map(({ caseId }) => caseId)).size !== 117 ||
      cases.filter(({ implementation }) => implementation === "webgpu").length !== 68 ||
      cases.filter(({ implementation }) => implementation === "wasm-codec").length !== 4 ||
      cases.filter(({ implementation }) => implementation === "wasm-validation").length !== 45) {
    fail("vector corpus inventory differs");
  }
  return cases;
}

export async function canonicalBrowserVectorMetadataV1() {
  const sourceBytes = await readFile(resolve(
    repository,
    "crates/tritium-spec/data/training/v2/vectors/v2.json",
  ));
  const mirroredBytes = await readFile(resolve(
    repository,
    "spec/training/v2/vectors/v2.json",
  ));
  if (!sourceBytes.equals(mirroredBytes)) fail("mirrored vector corpus bytes differ");
  if (bytesToHex(blake3(sourceBytes)) !== vectorDigest) {
    fail("vector source bytes differ from frozen BLAKE3 digest");
  }
  let corpus;
  try {
    corpus = JSON.parse(sourceBytes.toString("utf8"));
  } catch {
    fail("vector source is not UTF-8 JSON");
  }
  const metadata = {
    schema: "tritium.browser-vector-inventory.v1",
    manifestDigest,
    vectorDigest,
    sourceSha256: createHash("sha256").update(sourceBytes).digest("hex"),
    cases: deriveCases(corpus),
  };
  return Object.freeze({ metadata: Object.freeze(metadata), sourceBytes });
}

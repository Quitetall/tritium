import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { blake3 } from "@noble/hashes/blake3.js";
import { bytesToHex } from "@noble/hashes/utils.js";

import {
  canonicalTrainingManifestJson,
  parseTrainingManifest,
  TRAINING_MANIFEST_DIGEST_V2,
  TRAINING_VECTOR_DIGEST_V2,
  TrainingManifestError,
} from "../dist/index.js";

const manifestPath = resolve("../../spec/training/v2/manifest.json");
const vectorsPath = resolve("../../spec/training/v2/vectors/v2.json");

test("package schema mirrors the frozen language-neutral manifest", async () => {
  const manifest = await readFile(manifestPath);
  assert.equal(
    Buffer.compare(Buffer.from(canonicalTrainingManifestJson()), manifest),
    0,
  );
  assert.equal(parseTrainingManifest(manifest).operations.length, 36);
  assert.equal(bytesToHex(blake3(manifest)), TRAINING_MANIFEST_DIGEST_V2);
});

test("package vector identity is frozen and bound to the manifest", async () => {
  const vectors = await readFile(vectorsPath);
  const parsed = JSON.parse(vectors.toString("utf8"));
  assert.equal(parsed.schema_id, "tritium.training_vectors");
  assert.equal(parsed.schema_version, 2);
  assert.equal(parsed.manifest_digest, TRAINING_MANIFEST_DIGEST_V2);
  assert.equal(bytesToHex(blake3(vectors)), TRAINING_VECTOR_DIGEST_V2);
});

test("packed parser fails closed on schema drift", async () => {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  manifest.schema_version = 1;
  assert.throws(
    () => parseTrainingManifest(JSON.stringify(manifest)),
    TrainingManifestError,
  );
  assert.throws(
    () => parseTrainingManifest('{"schema_id":"x","schema_id":"y"}'),
    TrainingManifestError,
  );
});

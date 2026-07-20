import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  runPortableWasmConformance,
  TRAINING_MANIFEST_DIGEST_V1,
  TRAINING_VECTOR_DIGEST_V1,
} from "../dist/index.js";

test("bundled wasm32-unknown guest passes the complete corpus twice", async () => {
  const guest = await readFile(
    new URL("../dist/tritium_wasm_bg.wasm", import.meta.url),
  );
  const corrupted = Uint8Array.from(guest);
  corrupted[corrupted.length - 1] ^= 1;
  await assert.rejects(
    runPortableWasmConformance(corrupted),
    /guest digest mismatch/,
  );
  const receipt = await runPortableWasmConformance(guest);
  assert.deepEqual(receipt, {
    schemaId: "tritium.portable_wasm_conformance_receipt",
    schemaVersion: 1,
    implementation: "wasm-fallback",
    engine: "wasm32-unknown-unknown",
    buildId: receipt.buildId,
    guestDigest: receipt.guestDigest,
    executionDigest: receipt.executionDigest,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V1,
    vectorDigest: TRAINING_VECTOR_DIGEST_V1,
    operationCount: 35,
    caseCount: 114,
    maxCallerBytes: 64 * 1024 * 1024,
    maxLinearMemoryBytes: 192 * 1024 * 1024,
    repeatedExecutions: 2,
  });
  assert.match(receipt.buildId, /^tritium-wasm@1\.0\.0\+source-git:/);
  assert.match(receipt.guestDigest, /^[0-9a-f]{64}$/);
  assert.match(receipt.executionDigest, /^[0-9a-f]{64}$/);
  assert.ok(Object.isFrozen(receipt));

});

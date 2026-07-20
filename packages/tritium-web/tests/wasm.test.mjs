import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  executePortableWasmRequest,
  runPortableWasmConformance,
  TRAINING_MANIFEST_DIGEST_V1,
  TRAINING_VECTOR_DIGEST_V1,
} from "../dist/index.js";

const sgdRequest = {
  schemaId: "tritium.portable_training_request",
  schemaVersion: 1,
  physicalDevice: "node:test-wasm",
  operation: "optimizer.sgd",
  execution: "step",
  vectorDigest: null,
  inputs: [
    {
      name: "parameter",
      shape: [2],
      data: { dtype: "f32", bits: [1065353216, 3221225472] },
    },
    {
      name: "gradient",
      shape: [2],
      data: { dtype: "f32", bits: [1056964608, 3196059648] },
    },
  ],
  attributes: [
    { kind: "u64", name: "step", value: 1 },
    { kind: "f32", name: "lr", bits: 1036831949 },
  ],
  outputs: [
    {
      name: "parameter",
      shape: [2],
      data: { dtype: "f32", bits: [0, 0] },
    },
  ],
};

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

test("bundled guest executes strict portable requests", async () => {
  const guest = await readFile(
    new URL("../dist/tritium_wasm_bg.wasm", import.meta.url),
  );
  const response = await executePortableWasmRequest(sgdRequest, guest);
  assert.equal(response.status, "ok");
  assert.deepEqual(response.outputs[0].data.bits, [1064514355, 3221015757]);
  assert.equal(response.receipt.backendId, "wasm.portable.v1");
  assert.equal(response.receipt.physicalDevice, "node:test-wasm");
  assert.equal(response.receipt.hostTransfers, 0);
  assert.equal(response.receipt.deviceResident, true);
  assert.ok(Object.isFrozen(response));
  assert.ok(Object.isFrozen(response.outputs));
  assert.ok(Object.isFrozen(response.outputs[0].data.bits));

  const u32ListRequest = {
    schemaId: "tritium.portable_training_request",
    schemaVersion: 1,
    physicalDevice: "node:test-wasm",
    operation: "graph.fsq",
    execution: "forward",
    vectorDigest: TRAINING_VECTOR_DIGEST_V1,
    inputs: [
      {
        name: "x",
        shape: [2, 3],
        data: {
          dtype: "f32",
          bits: [3214514586, 3196059648, 1058642330, 1063675494, 1036831949, 3211159142],
        },
      },
    ],
    attributes: [
      { kind: "u64", name: "channels", value: 2 },
      { kind: "u64", name: "len", value: 3 },
      { kind: "u32-list", name: "levels", values: [3, 5] },
      { kind: "text", name: "bound", value: "clamp" },
      { kind: "text", name: "ste", value: "soft_round" },
      { kind: "f32", name: "alpha", bits: 1056964608 },
      { kind: "u64", name: "seed", value: 0 },
    ],
    outputs: [
      {
        name: "result",
        shape: [2, 3],
        data: { dtype: "f32", bits: [0, 0, 0, 0, 0, 0] },
      },
    ],
  };
  const u32ListResponse = await executePortableWasmRequest(u32ListRequest, guest);
  assert.equal(u32ListResponse.status, "ok");

  const mutableRequest = structuredClone(sgdRequest);
  const pending = executePortableWasmRequest(mutableRequest, guest);
  mutableRequest.operation = "graph.not-real";
  mutableRequest.outputs[0].data.bits[0] = 99;
  const snapshotted = await pending;
  assert.equal(snapshotted.status, "ok");
  assert.deepEqual(snapshotted.outputs[0].data.bits, [1064514355, 3221015757]);

  const invalid = structuredClone(sgdRequest);
  invalid.inputs[0].shape = [3];
  invalid.outputs[0].data.bits = [0x7fc00001, 0x7fc00002];
  const failure = await executePortableWasmRequest(invalid, guest);
  assert.equal(failure.status, "error");
  assert.equal(failure.error.code, "buffer_length.parameter.3.2");
  assert.deepEqual(failure.outputs[0].data.bits, [0x7fc00001, 0x7fc00002]);

  const unsafeInteger = structuredClone(sgdRequest);
  unsafeInteger.attributes[0].value = Number.MAX_SAFE_INTEGER + 1;
  const rejected = await executePortableWasmRequest(unsafeInteger, guest);
  assert.equal(rejected.status, "error");
  assert.equal(rejected.error.code, "unsafe_integer");
  assert.deepEqual(rejected.outputs, []);

  const missingDigest = structuredClone(sgdRequest);
  delete missingDigest.vectorDigest;
  const missing = await executePortableWasmRequest(missingDigest, guest);
  assert.equal(missing.status, "error");
  assert.equal(missing.error.code, "missing_field");

  const wrongDigest = structuredClone(sgdRequest);
  wrongDigest.vectorDigest = "0".repeat(64);
  const digestMismatch = await executePortableWasmRequest(wrongDigest, guest);
  assert.equal(digestMismatch.status, "error");
  assert.equal(digestMismatch.error.code, "vector_digest_mismatch");

  const excessiveRank = structuredClone(sgdRequest);
  excessiveRank.inputs[0].shape = [1, 1, 1, 1, 1];
  const rankFailure = await executePortableWasmRequest(excessiveRank, guest);
  assert.equal(rankFailure.status, "error");
  assert.equal(rankFailure.error.category, "capacity");
  assert.equal(rankFailure.error.code, "rank");

  const bigint = structuredClone(sgdRequest);
  bigint.attributes[0].value = 1n;
  const nonJson = await executePortableWasmRequest(bigint, guest);
  assert.equal(nonJson.status, "error");
  assert.equal(nonJson.error.code, "invalid_json");
});

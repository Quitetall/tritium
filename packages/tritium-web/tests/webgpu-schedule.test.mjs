import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  compileWebGpuResidentScheduleV1,
  TRAINING_MANIFEST_DIGEST_V1,
  webGpuDispatchFormV1,
  WebTrainingError,
} from "../dist/index.js";

const corpus = JSON.parse(readFileSync(
  new URL("../../../spec/training/v1/vectors/v1.json", import.meta.url),
  "utf8",
));
const BUDGET = Object.freeze({ maxPeakBytes: 64 * 1024 * 1024 });

function plan(buffers, operations, backwardOperations = []) {
  let residentBytes = 0;
  const placedBuffers = buffers.map((buffer) => {
    const placed = { ...buffer, byteOffset: residentBytes };
    residentBytes += Math.ceil(buffer.byteLength / 16) * 16;
    return placed;
  });
  return {
    schemaId: "tritium.compiled_training_plan",
    schemaVersion: 1,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V1,
    buffers: placedBuffers,
    operations,
    backwardOperations,
    residentBytes,
    batchStagingBytes: 0,
    preparePeakBytes: residentBytes,
    forwardPeakBytes: residentBytes,
    exportPackageBytes: 0,
    exportPeakBytes: residentBytes,
    peakBytes: residentBytes,
  };
}

function f32FromBits(bits) {
  const bytes = new ArrayBuffer(4);
  const view = new DataView(bytes);
  view.setUint32(0, bits, true);
  return view.getFloat32(0, true);
}

function representativePlan(item) {
  const entries = [...item.inputs, ...item.expected.outputs];
  const buffers = entries.map((entry) => {
    const elements = entry.shape.reduce((product, value) => product * value, 1);
    const bytesPerElement = entry.data.dtype === "bytes" ? 1 : 4;
    return {
      id: entry.name,
      role: "activation",
      dtype: entry.data.dtype,
      shape: [...entry.shape],
      aliasOf: null,
      ownerId: entry.name,
      byteOffset: 0,
      byteLength: elements * bytesPerElement,
      backwardInitialization: "none",
    };
  });
  const attributes = item.attributes.map((attribute) => ({
    name: attribute.name,
    kind: attribute.type === "u32_list"
      ? "u32-list"
      : attribute.type === "u64_list"
        ? "u64-list"
        : attribute.type,
    value: attribute.type === "f32"
      ? f32FromBits(attribute.bits)
      : "values" in attribute
        ? [...attribute.values]
        : attribute.value,
  }));
  const binding = (entry) => ({ role: entry.name, bufferId: entry.name });
  const operationId = `canonical.${item.operation}.${item.execution}`;
  const operations = item.execution === "forward" ? [{
    id: operationId,
    operation: item.operation,
    inputs: item.inputs.map((entry) => entry.name),
    outputs: item.expected.outputs.map((entry) => entry.name),
    attributes,
  }] : [];
  const backwards = item.execution === "vjp" ? [{
    id: operationId,
    sourceOperationId: `source.${item.operation}`,
    operation: item.operation,
    execution: "vjp",
    inputs: item.inputs.map(binding),
    outputs: item.expected.outputs.map(binding),
    attributes,
  }] : [];
  return {
    phase: item.execution === "forward" ? "forward" : "backward",
    operationId,
    plan: plan(buffers, operations, backwards),
  };
}

function view(command) {
  return new DataView(
    command.uniformBytes.buffer,
    command.uniformBytes.byteOffset,
    command.uniformBytes.byteLength,
  );
}

test("resident schedule covers all 44 first-tranche canonical execution forms", () => {
  const supported = new Set([
    "graph.detach", "graph.scale_const", "graph.add", "graph.mul", "graph.relu2",
    "graph.silu", "graph.causal_mask", "graph.softmax", "graph.rmsnorm", "loss.mse",
    "graph.bias", "graph.transpose", "graph.slice_cols", "graph.dense_matmul",
    "graph.ternary_matmul", "graph.ste_surrogate", "graph.lsq_ste",
    "graph.salt_ste", "graph.fsq", "graph.embedding_gather", "graph.rope",
    "loss.softmax_cross_entropy",
  ]);
  const representatives = new Map();
  for (const item of corpus.cases) {
    const key = `${item.operation}|${item.execution}`;
    if (supported.has(item.operation) && item.expected.kind === "success" &&
        !representatives.has(key)) representatives.set(key, item);
  }
  assert.equal(representatives.size, 44);
  for (const [key, item] of representatives) {
    const representative = representativePlan(item);
    const schedule = compileWebGpuResidentScheduleV1(representative.plan, BUDGET);
    const transaction = schedule.transaction(
      representative.phase, representative.operationId, 3,
    );
    assert.equal(
      transaction.commands.length,
      webGpuDispatchFormV1(item.operation, item.execution).stages.length,
      key,
    );
    assert.deepEqual(
      transaction.commands.map((command, index) => command.uniformSlot - index),
      transaction.commands.map(() => 3),
      key,
    );
  }
});

test("specialized constants and scratch are immutable auxiliary snapshots", () => {
  const item = corpus.cases.find((candidate) =>
    candidate.operation === "graph.fsq" && candidate.execution === "forward" &&
    candidate.expected.kind === "success");
  const representative = representativePlan(item);
  const schedule = compileWebGpuResidentScheduleV1(representative.plan, BUDGET);
  const first = schedule.auxiliaryResources();
  assert.equal(first.maxBytes, 8);
  assert.equal(first.resources.length, 1);
  assert.deepEqual([...first.resources[0].initialBytes], [3, 0, 0, 0, 5, 0, 0, 0]);
  first.resources[0].initialBytes[0] = 99;
  assert.equal(schedule.auxiliaryResources().resources[0].initialBytes[0], 3);

  const initial = schedule.transaction(representative.phase, representative.operationId, 0);
  assert.equal(initial.commands[0].storageBindings[2], first.resources[0].id);
  initial.commands[0].uniformBytes[0] = 99;
  const fresh = schedule.transaction(representative.phase, representative.operationId, 1);
  assert.equal(view(fresh.commands[0]).getUint32(0, true), 6);
  assert.equal(fresh.commands[0].uniformSlot, 1);
});

test("SALT allocates bounded scratch and softmax VJP keeps cotangent resident", () => {
  const saltItem = corpus.cases.find((candidate) =>
    candidate.operation === "graph.salt_ste" && candidate.execution === "forward" &&
    candidate.expected.kind === "success");
  const salt = representativePlan(saltItem);
  const saltSchedule = compileWebGpuResidentScheduleV1(salt.plan, BUDGET);
  const saltResources = saltSchedule.auxiliaryResources();
  assert.equal(saltResources.maxBytes, 12);
  assert.equal(saltResources.resources[0].initialBytes, null);
  assert.equal(
    saltSchedule.transaction(salt.phase, salt.operationId, 0).commands[0].storageBindings[2],
    saltResources.resources[0].id,
  );

  const xentItem = corpus.cases.find((candidate) =>
    candidate.operation === "loss.softmax_cross_entropy" && candidate.execution === "vjp" &&
    candidate.expected.kind === "success");
  const xent = representativePlan(xentItem);
  const command = compileWebGpuResidentScheduleV1(xent.plan, BUDGET)
    .transaction(xent.phase, xent.operationId, 0).commands[0];
  assert.equal(command.storageBindings[3], "grad_output");
  assert.equal(command.storageBindings[4], "grad_logits");
  assert.equal(view(command).getUint32(8, true), 1);
  assert.equal(view(command).getFloat32(12, true), 0);
});

test("unsupported forms and malformed specialized geometry fail closed", () => {
  const unsupported = corpus.cases.find((candidate) =>
    candidate.operation === "graph.concat_cols" && candidate.execution === "forward" &&
    candidate.expected.kind === "success");
  const concat = representativePlan(unsupported);
  const schedule = compileWebGpuResidentScheduleV1(concat.plan, BUDGET);
  assert.throws(
    () => schedule.transaction(concat.phase, concat.operationId, 0),
    (error) => error instanceof WebTrainingError && error.code === "capability_mismatch",
  );

  const saltItem = structuredClone(corpus.cases.find((candidate) =>
    candidate.operation === "graph.salt_ste" && candidate.execution === "forward" &&
    candidate.expected.kind === "success"));
  saltItem.attributes.find((attribute) => attribute.name === "planes").value = 65;
  const salt = representativePlan(saltItem);
  assert.throws(
    () => compileWebGpuResidentScheduleV1(salt.plan, BUDGET),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
});

test("compiler snapshots hostile plans once and admits aggregate peak before resources", () => {
  const item = corpus.cases.find((candidate) =>
    candidate.operation === "graph.rope" && candidate.execution === "forward" &&
    candidate.expected.kind === "success");
  const representative = representativePlan(item);
  let operationReads = 0;
  const hostile = { ...representative.plan };
  Object.defineProperty(hostile, "operations", {
    enumerable: true,
    get() {
      operationReads += 1;
      if (operationReads > 1) throw new Error("operations reread");
      return representative.plan.operations;
    },
  });
  const schedule = compileWebGpuResidentScheduleV1(hostile, BUDGET);
  assert.equal(operationReads, 1);
  assert.equal(
    schedule.transaction(representative.phase, representative.operationId, 0).commands.length,
    1,
  );

  const throwing = { ...representative.plan };
  Object.defineProperty(throwing, "buffers", {
    enumerable: true,
    get() { throw new Error("attacker exception"); },
  });
  assert.throws(
    () => compileWebGpuResidentScheduleV1(throwing, BUDGET),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  assert.throws(
    () => compileWebGpuResidentScheduleV1(
      representative.plan,
      { maxPeakBytes: representative.plan.peakBytes },
    ),
    (error) => error instanceof WebTrainingError && error.code === "memory_limit",
  );
});

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
  const entries = [...new Map(
    [...item.inputs, ...item.expected.outputs].map((entry) => [entry.name, entry]),
  ).values()];
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
    value: item.execution === "step" && attribute.name === "step"
      ? 0
      : attribute.type === "f32"
      ? f32FromBits(attribute.bits)
      : "values" in attribute
        ? [...attribute.values]
        : attribute.value,
  }));
  const binding = (entry) => ({ role: entry.name, bufferId: entry.name });
  const operationId = `canonical.${item.operation}.${item.execution}`;
  const operations = item.execution !== "vjp" ? [{
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
    phase: item.execution === "vjp" ? "backward" : "forward",
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

test("resident schedule covers all 52 graph/loss forms and five transactional optimizers", () => {
  const supported = new Set([
    "graph.detach", "graph.scale_const", "graph.add", "graph.mul", "graph.relu2",
    "graph.silu", "graph.causal_mask", "graph.softmax", "graph.rmsnorm", "loss.mse",
    "graph.bias", "graph.transpose", "graph.slice_cols", "graph.dense_matmul",
    "graph.ternary_matmul", "graph.ste_surrogate", "graph.lsq_ste",
    "graph.salt_ste", "graph.fsq", "graph.embedding_gather", "graph.rope",
    "graph.concat_cols",
    "graph.conv1d", "graph.conv2d", "graph.attention",
    "loss.softmax_cross_entropy",
    "optimizer.sgd", "optimizer.adamw", "optimizer.cautious_adamw",
    "optimizer.int8_adamw", "optimizer.muon",
  ]);
  const representatives = new Map();
  for (const item of corpus.cases) {
    const key = `${item.operation}|${item.execution}`;
    if (supported.has(item.operation) && item.expected.kind === "success" &&
        !representatives.has(key)) representatives.set(key, item);
  }
  assert.equal(representatives.size, 57);
  for (const [key, item] of representatives) {
    const representative = representativePlan(item);
    const schedule = compileWebGpuResidentScheduleV1(representative.plan, BUDGET);
    const transaction = schedule.transaction(
      representative.phase, representative.operationId, 3,
      item.execution === "step" ? 1 : undefined,
    );
    const form = webGpuDispatchFormV1(item.operation, item.execution);
    const expectedCommands = form.stages.reduce((count, stage) =>
      count + (stage.repeat === "per_output" ? item.expected.outputs.length : 1), 0);
    assert.equal(transaction.commands.length, expectedCommands, key);
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

test("concat packs forward parts with GPU copies and emits ordered VJP slices", () => {
  const forwardItem = corpus.cases.find((candidate) =>
    candidate.operation === "graph.concat_cols" && candidate.execution === "forward" &&
    candidate.expected.kind === "success");
  const forward = representativePlan(forwardItem);
  const forwardSchedule = compileWebGpuResidentScheduleV1(forward.plan, BUDGET);
  const transaction = forwardSchedule.transaction(forward.phase, forward.operationId, 0);
  assert.equal(transaction.copies.length, 2);
  assert.deepEqual(transaction.copies.map((copy) => [
    copy.source, copy.destinationOffset, copy.byteLength,
  ]), [["part.0", 0, 16], ["part.1", 16, 8]]);
  assert.equal(transaction.commands[0].storageBindings[1], transaction.copies[0].destination);
  assert.equal(forwardSchedule.auxiliaryResources().maxBytes, 40);

  const vjpItem = corpus.cases.find((candidate) =>
    candidate.operation === "graph.concat_cols" && candidate.execution === "vjp" &&
    candidate.expected.kind === "success");
  const vjp = representativePlan(vjpItem);
  const commands = compileWebGpuResidentScheduleV1(vjp.plan, BUDGET)
    .transaction(vjp.phase, vjp.operationId, 4).commands;
  assert.deepEqual(commands.map((command) => command.uniformSlot), [4, 5]);
  assert.deepEqual(commands.map((command) => [
    view(command).getUint32(4, true),
    view(command).getUint32(16, true),
    view(command).getUint32(20, true),
  ]), [[24, 0, 2], [24, 2, 1]]);
  assert.deepEqual(commands.map((command) => command.storageBindings[4]), [
    "grad_part.0", "grad_part.1",
  ]);
});

test("malformed specialized geometry fails closed", () => {
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

test("concat VJP rejects duplicate and aliased output owners", () => {
  const item = structuredClone(corpus.cases.find((candidate) =>
    candidate.operation === "graph.concat_cols" && candidate.execution === "vjp" &&
    candidate.expected.kind === "success"));
  item.attributes.find((attribute) => attribute.name === "lens").values = [2, 2];
  item.inputs[0].shape = [2, 4];
  item.expected.outputs[1].shape = [2, 2];

  const duplicate = representativePlan(item);
  duplicate.plan.backwardOperations[0].outputs[1].bufferId = "grad_part.0";
  assert.throws(
    () => compileWebGpuResidentScheduleV1(duplicate.plan, BUDGET),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );

  const aliased = representativePlan(item);
  const owner = aliased.plan.buffers.find((buffer) => buffer.id === "grad_part.0");
  const alias = aliased.plan.buffers.find((buffer) => buffer.id === "grad_part.1");
  owner.role = "parameter";
  alias.role = "parameter";
  alias.aliasOf = owner.id;
  alias.ownerId = owner.id;
  alias.byteOffset = owner.byteOffset;
  assert.throws(
    () => compileWebGpuResidentScheduleV1(aliased.plan, BUDGET),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
});

test("convolution VJP clears resident accumulators and packs exact 80-byte ABI", () => {
  for (const operation of ["graph.conv1d", "graph.conv2d"]) {
    const item = corpus.cases.find((candidate) =>
      candidate.operation === operation && candidate.execution === "vjp" &&
      candidate.expected.kind === "success");
    const representative = representativePlan(item);
    const schedule = compileWebGpuResidentScheduleV1(representative.plan, BUDGET);
    const transaction = schedule.transaction(
      representative.phase, representative.operationId, 0,
    );
    assert.equal(transaction.commands.length, 1);
    assert.equal(transaction.commands[0].uniformBytes.byteLength, 80);
    assert.equal(view(transaction.commands[0]).getUint32(64, true), 1);
    assert.deepEqual(transaction.copies.map((copy) => copy.destination), [
      "grad_x", "grad_weight", "grad_scale",
    ]);
    assert.equal(new Set(transaction.copies.map((copy) => copy.source)).size, 1);
    assert.deepEqual(transaction.commands[0].storageBindings, {
      1: "x", 2: "weight", 3: "scale", 4: "grad_output",
      5: "grad_x", 6: "grad_weight", 7: "grad_scale",
    });
  }
});

test("attention owns probability scratch and zeroes three VJP outputs", () => {
  const item = corpus.cases.find((candidate) =>
    candidate.operation === "graph.attention" && candidate.execution === "vjp" &&
    candidate.expected.kind === "success");
  const representative = representativePlan(item);
  const schedule = compileWebGpuResidentScheduleV1(representative.plan, BUDGET);
  const resources = schedule.auxiliaryResources();
  assert.deepEqual(resources.resources.map((resource) => resource.byteLength), [36, 48, 36]);
  const transaction = schedule.transaction(representative.phase, representative.operationId, 0);
  assert.deepEqual(transaction.copies.map((copy) => copy.destination), [
    "grad_q", "grad_k", "grad_v",
  ]);
  assert.deepEqual(transaction.commands[0].storageBindings, {
    1: "q", 2: "k", 3: "v", 4: "grad_output", 5: "grad_q",
    6: "grad_k", 7: "grad_v",
    8: resources.resources[0].id,
    9: resources.resources[2].id,
  });
  assert.deepEqual([
    view(transaction.commands[0]).getUint32(0, true),
    view(transaction.commands[0]).getUint32(4, true),
    view(transaction.commands[0]).getUint32(8, true),
    view(transaction.commands[0]).getUint32(12, true),
    view(transaction.commands[0]).getUint32(16, true),
    view(transaction.commands[0]).getUint32(20, true),
  ], [3, 2, 1, 2, 1, 1]);
});

test("SGD computes into a candidate and commits only after dispatch", () => {
  const item = corpus.cases.find((candidate) =>
    candidate.operation === "optimizer.sgd" && candidate.execution === "step" &&
    candidate.expected.kind === "success");
  const representative = representativePlan(item);
  const schedule = compileWebGpuResidentScheduleV1(representative.plan, BUDGET);
  const resources = schedule.auxiliaryResources();
  assert.deepEqual(resources.resources.map((resource) => resource.byteLength), [8]);
  assert.throws(
    () => schedule.transaction(representative.phase, representative.operationId, 0),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  const transaction = schedule.transaction(
    representative.phase, representative.operationId, 0, 7,
  );
  assert.deepEqual(transaction.commands[0].storageBindings, {
    1: "parameter", 2: "gradient", 3: "gradient", 4: resources.resources[0].id,
  });
  assert.equal(view(transaction.commands[0]).getUint32(4, true), 21);
  assert.deepEqual(transaction.commitCopies, [{
    source: resources.resources[0].id,
    sourceOffset: 0,
    destination: "parameter",
    destinationOffset: 0,
    byteLength: 8,
  }]);
  assert.throws(
    () => schedule.transaction(representative.phase, representative.operationId, 0, 0),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  const forged = structuredClone(representative.plan);
  forged.operations[0].attributes.find((attribute) => attribute.name === "step").value = 1;
  assert.throws(
    () => compileWebGpuResidentScheduleV1(forged, BUDGET),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
});

test("AdamW stages candidates and commits all three state planes transactionally", () => {
  const item = corpus.cases.find((candidate) =>
    candidate.operation === "optimizer.adamw" && candidate.execution === "step" &&
    candidate.expected.kind === "success");
  const representative = representativePlan(item);
  const schedule = compileWebGpuResidentScheduleV1(representative.plan, BUDGET);
  const resources = schedule.auxiliaryResources().resources;
  const bytes = item.inputs[0].shape.reduce((size, dimension) => size * dimension, 4);
  assert.deepEqual(resources.map((resource) => resource.byteLength), Array(5).fill(bytes));
  const transaction = schedule.transaction(
    representative.phase, representative.operationId, 4, 7,
  );
  assert.deepEqual(transaction.commands.map((command) => command.stageIndex), [0, 1, 2, 3]);
  assert.deepEqual(transaction.commands.map((command) => command.uniformSlot), [4, 5, 6, 7]);
  assert.deepEqual(transaction.commands[0].storageBindings, {
    1: "parameter", 2: "gradient", 3: "moment1", 4: "moment2",
    5: resources[0].id, 6: resources[1].id, 7: resources[2].id,
    8: resources[3].id, 9: resources[4].id,
  });
  assert.deepEqual(transaction.commitCopies.map((copy) => [copy.source, copy.destination]), [
    [resources[0].id, "parameter"],
    [resources[1].id, "moment1"],
    [resources[2].id, "moment2"],
  ]);
  assert.equal(view(transaction.commands[0]).getUint32(0, true), bytes / 4);
  assert.equal(view(transaction.commands[0]).getUint32(32, true), 0x3f4a501a);
  assert.equal(view(transaction.commands[0]).getUint32(36, true), 0x3e9a738c);
});

test("cautious AdamW resets its atomic mask before every staged transaction", () => {
  const item = corpus.cases.find((candidate) =>
    candidate.operation === "optimizer.cautious_adamw" && candidate.execution === "step" &&
    candidate.expected.kind === "success");
  const representative = representativePlan(item);
  const schedule = compileWebGpuResidentScheduleV1(representative.plan, BUDGET);
  const resources = schedule.auxiliaryResources().resources;
  const transaction = schedule.transaction(
    representative.phase, representative.operationId, 0, 3,
  );
  assert.deepEqual(transaction.commands.map((command) => command.stageIndex),
    [0, 1, 2, 3, 4, 5, 6]);
  assert.deepEqual(transaction.copies, [{
    source: resources[6].id,
    sourceOffset: 0,
    destination: resources[5].id,
    destinationOffset: 0,
    byteLength: 4,
  }]);
  assert.deepEqual([...resources[6].initialBytes], [0, 0, 0, 0]);
  assert.equal(transaction.commands[3].storageBindings[10], resources[5].id);
  assert.equal(transaction.commands[5].storageBindings[10], resources[5].id);
  assert.equal(view(transaction.commands[0]).getUint32(32, true), 0x3ef9db22);
  assert.equal(view(transaction.commands[0]).getUint32(36, true), 0x3e120c4c);
});

test("int8 AdamW unpacks compact state, stages widened math, and repacks commits", () => {
  const item = corpus.cases.find((candidate) =>
    candidate.operation === "optimizer.int8_adamw" && candidate.execution === "step" &&
    candidate.expected.kind === "success");
  const representative = representativePlan(item);
  const schedule = compileWebGpuResidentScheduleV1(representative.plan, BUDGET);
  const resources = schedule.auxiliaryResources().resources;
  const transaction = schedule.transaction(
    representative.phase, representative.operationId, 1, 7,
  );
  assert.equal(transaction.commands.length, 12);
  assert.deepEqual(transaction.commands.map((command) => command.stageIndex),
    Array.from({ length: 12 }, (_, index) => index));
  assert.deepEqual(transaction.commands.map((command) => command.uniformSlot),
    Array.from({ length: 12 }, (_, index) => index + 1));
  assert.deepEqual(transaction.commands[0].storageBindings, {
    1: "moment1_q8", 2: resources[1].id,
  });
  assert.deepEqual(transaction.commands[1].storageBindings, {
    1: "moment2_q8", 2: resources[2].id,
  });
  assert.deepEqual(transaction.commands[2].storageBindings, {
    3: resources[1].id, 4: resources[2].id, 5: resources[3].id, 6: resources[4].id,
  });
  assert.deepEqual(transaction.commands[4].storageBindings, {
    1: resources[0].id, 2: "gradient", 3: resources[1].id, 4: resources[2].id,
    7: resources[5].id, 8: resources[6].id,
  });
  assert.deepEqual(transaction.commands[10].storageBindings, {
    1: resources[1].id, 2: resources[7].id,
  });
  assert.deepEqual(transaction.commands[11].storageBindings, {
    1: resources[2].id, 2: resources[8].id,
  });
  assert.deepEqual(transaction.commitCopies.map((copy) => [copy.source, copy.destination]), [
    [resources[0].id, "parameter"],
    [resources[7].id, "moment1_q8"],
    [resources[8].id, "moment2_q8"],
    [resources[3].id, "moment1_scale"],
    [resources[4].id, "moment2_scale"],
  ]);
  assert.equal(view(transaction.commands[2]).getUint32(32, true), 0x3f4a501a);
  assert.equal(view(transaction.commands[2]).getUint32(36, true), 0x3e9a738c);
});

test("Muon snapshots mutable inputs, owns exact workspace, then commits state", () => {
  const item = corpus.cases.find((candidate) =>
    candidate.operation === "optimizer.muon" && candidate.execution === "step" &&
    candidate.expected.kind === "success");
  const representative = representativePlan(item);
  const schedule = compileWebGpuResidentScheduleV1(representative.plan, BUDGET);
  const resources = schedule.auxiliaryResources().resources;
  const attribute = (name) => item.attributes.find((candidate) => candidate.name === name).value;
  const rows = attribute("rows");
  const cols = attribute("cols");
  const len = rows * cols;
  const square = Math.min(rows, cols) ** 2;
  assert.deepEqual(resources.map((resource) => resource.byteLength), [
    (3 * len + 3 * square + 2) * 4, len * 4, len * 4,
  ]);
  const transaction = schedule.transaction(
    representative.phase, representative.operationId, 2, 1,
  );
  assert.deepEqual(transaction.copies.map((copy) => [copy.source, copy.destination]), [
    ["parameter", resources[1].id], ["momentum", resources[2].id],
  ]);
  assert.deepEqual(transaction.commands[0].storageBindings, {
    1: resources[1].id, 2: "gradient", 3: resources[2].id, 4: resources[0].id,
  });
  assert.deepEqual(transaction.commitCopies.map((copy) => [copy.source, copy.destination]), [
    [resources[1].id, "parameter"], [resources[2].id, "momentum"],
  ]);
});

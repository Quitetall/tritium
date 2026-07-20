import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  lowerPointwiseWebGpuOperationV1,
  TRAINING_MANIFEST_DIGEST_V1,
  webGpuDispatchFormV1,
  WebTrainingError,
} from "../dist/index.js";

const corpus = JSON.parse(readFileSync(
  new URL("../../../spec/training/v1/vectors/v1.json", import.meta.url),
  "utf8",
));

const tensor = (id, elements, index, role = "activation") => ({
  id,
  role,
  dtype: "f32",
  shape: [elements],
  aliasOf: null,
  ownerId: id,
  byteOffset: index * 64,
  byteLength: elements * 4,
  backwardInitialization: "none",
});

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

function uniform(command) {
  return new DataView(
    command.uniformBytes.buffer,
    command.uniformBytes.byteOffset,
    command.uniformBytes.byteLength,
  );
}

test("forward add lowers catalog selector and resident roles without tensor values", () => {
  const buffers = ["left", "right", "result"].map((id, index) => tensor(id, 2, index));
  const compiled = plan(buffers, [{
    id: "add",
    operation: "graph.add",
    inputs: ["left", "right"],
    outputs: ["result"],
    attributes: [],
  }]);
  const commands = lowerPointwiseWebGpuOperationV1(compiled, "forward", "add", 5);
  assert.equal(commands.length, 1);
  assert.equal(commands[0].uniformSlot, 5);
  assert.equal(uniform(commands[0]).getUint32(0, true), 2);
  assert.equal(uniform(commands[0]).getUint32(4, true), 3);
  assert.deepEqual(commands[0].storageBindings, {
    1: "left", 2: "right", 3: "left", 4: "result",
  });
  assert.deepEqual(commands[0].workgroups, [1, 1, 1]);
});

test("mul VJP lowers two ordered stages with exact gradient operands", () => {
  const ids = ["left", "right", "grad_output", "grad_left", "grad_right"];
  const buffers = ids.map((id, index) => tensor(id, 65, index));
  const backward = {
    id: "mul.vjp",
    sourceOperationId: "mul",
    operation: "graph.mul",
    execution: "vjp",
    inputs: [
      { role: "left", bufferId: "left" },
      { role: "right", bufferId: "right" },
      { role: "grad_output", bufferId: "grad_output" },
    ],
    outputs: [
      { role: "grad_left", bufferId: "grad_left" },
      { role: "grad_right", bufferId: "grad_right" },
    ],
    attributes: [],
  };
  const commands = lowerPointwiseWebGpuOperationV1(
    plan(buffers, [], [backward]), "backward", "mul.vjp", 9,
  );
  assert.equal(commands.length, 2);
  assert.deepEqual(commands.map((command) => uniform(command).getUint32(4, true)), [4, 4]);
  assert.deepEqual(commands.map((command) => command.uniformSlot), [9, 10]);
  assert.deepEqual(commands.map((command) => command.storageBindings[2]), ["right", "left"]);
  assert.deepEqual(commands.map((command) => command.storageBindings[4]), [
    "grad_left", "grad_right",
  ]);
  assert.deepEqual(commands.map((command) => command.workgroups), [[2, 1, 1], [2, 1, 1]]);
});

test("softmax lowering dispatches full input while carrying column geometry", () => {
  const buffers = [tensor("x", 4, 0), tensor("result", 4, 1)];
  buffers[0].shape = [2, 2];
  buffers[1].shape = [2, 2];
  const compiled = plan(buffers, [{
    id: "softmax",
    operation: "graph.softmax",
    inputs: ["x"],
    outputs: ["result"],
    attributes: [
      { name: "rows", kind: "u64", value: 2 },
      { name: "cols", kind: "u64", value: 2 },
    ],
  }]);
  const [command] = lowerPointwiseWebGpuOperationV1(compiled, "forward", "softmax", 0);
  assert.equal(uniform(command).getUint32(0, true), 4);
  assert.equal(uniform(command).getUint32(4, true), 11);
  assert.equal(uniform(command).getUint32(12, true), 2);
});

test("unlowered or malformed operations fail closed", () => {
  const buffers = [tensor("x", 2, 0), tensor("result", 2, 1)];
  const compiled = plan(buffers, [{
    id: "fsq",
    operation: "graph.fsq",
    inputs: ["x"],
    outputs: ["result"],
    attributes: [
      { name: "channels", kind: "u64", value: 1 },
      { name: "len", kind: "u64", value: 2 },
      { name: "levels", kind: "u32-list", value: [3] },
      { name: "bound", kind: "text", value: "tanh" },
      { name: "ste", kind: "text", value: "identity" },
      { name: "alpha", kind: "f32", value: 1 },
      { name: "seed", kind: "u64", value: 0 },
    ],
  }]);
  assert.throws(
    () => lowerPointwiseWebGpuOperationV1(compiled, "forward", "fsq", 0),
    (error) => error instanceof WebTrainingError && error.code === "capability_mismatch",
  );
  assert.throws(
    () => lowerPointwiseWebGpuOperationV1(compiled, "forward", "missing", 0),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  assert.throws(
    () => lowerPointwiseWebGpuOperationV1(
      plan(buffers, [null]), "forward", "fsq", 0,
    ),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
});

test("phase selection disambiguates colliding compiled operation IDs", () => {
  const buffers = ["left", "right", "result", "grad_output", "grad_x"]
    .map((id, index) => tensor(id, 2, index));
  const operation = {
    id: "backward.0",
    operation: "graph.add",
    inputs: ["left", "right"],
    outputs: ["result"],
    attributes: [],
  };
  const backward = {
    id: "backward.0",
    sourceOperationId: "source",
    operation: "graph.detach",
    execution: "vjp",
    inputs: [{ role: "grad_output", bufferId: "grad_output" }],
    outputs: [{ role: "grad_x", bufferId: "grad_x" }],
    attributes: [],
  };
  const compiled = plan(buffers, [operation], [backward]);
  assert.equal(
    lowerPointwiseWebGpuOperationV1(compiled, "forward", "backward.0", 0)[0]
      .operation,
    "graph.add",
  );
  assert.equal(
    lowerPointwiseWebGpuOperationV1(compiled, "backward", "backward.0", 1)[0]
      .operation,
    "graph.detach",
  );
});

test("multi-stage lowering rejects uniform slots outside the compiled arena", () => {
  const ids = ["left", "right", "grad_output", "grad_left", "grad_right"];
  const buffers = ids.map((id, index) => tensor(id, 2, index));
  const backward = {
    id: "mul.vjp",
    sourceOperationId: "mul",
    operation: "graph.mul",
    execution: "vjp",
    inputs: [
      { role: "left", bufferId: "left" },
      { role: "right", bufferId: "right" },
      { role: "grad_output", bufferId: "grad_output" },
    ],
    outputs: [
      { role: "grad_left", bufferId: "grad_left" },
      { role: "grad_right", bufferId: "grad_right" },
    ],
    attributes: [],
  };
  const compiled = plan(buffers, [], [backward]);
  assert.throws(
    () => lowerPointwiseWebGpuOperationV1(compiled, "backward", "mul.vjp", 15),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  assert.throws(
    () => lowerPointwiseWebGpuOperationV1(
      compiled, "backward", "mul.vjp", Number.MAX_SAFE_INTEGER,
    ),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
});

test("MSE VJP keeps scalar cotangent resident instead of reading it into a uniform", () => {
  const ids = ["prediction", "target", "grad_output", "grad_prediction"];
  const buffers = ids.map((id, index) => tensor(id, id === "grad_output" ? 1 : 8, index));
  buffers[2].shape = [];
  const backward = {
    id: "mse.vjp",
    sourceOperationId: "mse",
    operation: "loss.mse",
    execution: "vjp",
    inputs: [
      { role: "prediction", bufferId: "prediction" },
      { role: "target", bufferId: "target" },
      { role: "grad_output", bufferId: "grad_output" },
    ],
    outputs: [{ role: "grad_prediction", bufferId: "grad_prediction" }],
    attributes: [],
  };
  const [command] = lowerPointwiseWebGpuOperationV1(
    plan(buffers, [], [backward]),
    "backward",
    "mse.vjp",
    0,
  );
  assert.equal(command.storageBindings[3], "grad_output");
  assert.equal(uniform(command).getFloat32(8, true), 0);
  assert.equal(uniform(command).getUint32(0, true), 8);
  assert.equal(uniform(command).getUint32(4, true), 17);
});

test("dense VJP lowers shape scalars and stage-specific output lengths", () => {
  const definitions = [
    ["x", 6], ["weight", 12], ["grad_output", 8], ["grad_x", 6], ["grad_weight", 12],
  ];
  const buffers = definitions.map(([id, elements], index) => tensor(id, elements, index));
  Object.assign(buffers[0], { shape: [2, 3] });
  Object.assign(buffers[1], { shape: [4, 3] });
  Object.assign(buffers[2], { shape: [2, 4] });
  Object.assign(buffers[3], { shape: [2, 3] });
  Object.assign(buffers[4], { shape: [4, 3] });
  const backward = {
    id: "dense.vjp",
    sourceOperationId: "dense",
    operation: "graph.dense_matmul",
    execution: "vjp",
    inputs: [
      { role: "x", bufferId: "x" },
      { role: "weight", bufferId: "weight" },
      { role: "grad_output", bufferId: "grad_output" },
    ],
    outputs: [
      { role: "grad_x", bufferId: "grad_x" },
      { role: "grad_weight", bufferId: "grad_weight" },
    ],
    attributes: [
      { name: "m", kind: "u64", value: 2 },
      { name: "n", kind: "u64", value: 4 },
      { name: "k", kind: "u64", value: 3 },
    ],
  };
  const commands = lowerPointwiseWebGpuOperationV1(
    plan(buffers, [], [backward]),
    "backward",
    "dense.vjp",
    3,
  );
  assert.deepEqual(commands.map((command) => uniform(command).getUint32(0, true)), [6, 12]);
  for (const command of commands) {
    assert.deepEqual([
      uniform(command).getUint32(12, true),
      uniform(command).getUint32(16, true),
      uniform(command).getUint32(20, true),
    ], [2, 4, 3]);
  }
});

test("RMSNorm weight VJP dispatches matrix length into vector output", () => {
  const definitions = [
    ["x", 6], ["weight", 3], ["grad_output", 6], ["grad_x", 6], ["grad_weight", 3],
  ];
  const buffers = definitions.map(([id, elements], index) => tensor(id, elements, index));
  Object.assign(buffers[0], { shape: [2, 3] });
  Object.assign(buffers[1], { shape: [3] });
  Object.assign(buffers[2], { shape: [2, 3] });
  Object.assign(buffers[3], { shape: [2, 3] });
  Object.assign(buffers[4], { shape: [3] });
  const backward = {
    id: "rms.vjp",
    sourceOperationId: "rms",
    operation: "graph.rmsnorm",
    execution: "vjp",
    inputs: [
      { role: "x", bufferId: "x" },
      { role: "weight", bufferId: "weight" },
      { role: "grad_output", bufferId: "grad_output" },
    ],
    outputs: [
      { role: "grad_x", bufferId: "grad_x" },
      { role: "grad_weight", bufferId: "grad_weight" },
    ],
    attributes: [
      { name: "rows", kind: "u64", value: 2 },
      { name: "cols", kind: "u64", value: 3 },
      { name: "eps", kind: "f32", value: 1e-5 },
    ],
  };
  const commands = lowerPointwiseWebGpuOperationV1(
    plan(buffers, [], [backward]),
    "backward",
    "rms.vjp",
    0,
  );
  assert.equal(uniform(commands[1]).getUint32(0, true), 6);
  assert.equal(commands[1].storageBindings[4], "grad_weight");
});

test("all 34 pointwise-backed canonical forms lower with catalog stage parity", () => {
  const supported = new Set([
    "graph.detach", "graph.scale_const", "graph.add", "graph.mul", "graph.relu2",
    "graph.silu", "graph.causal_mask", "graph.softmax", "graph.rmsnorm", "loss.mse",
    "graph.bias", "graph.transpose", "graph.slice_cols", "graph.dense_matmul",
    "graph.ternary_matmul", "graph.ste_surrogate", "graph.lsq_ste",
  ]);
  // Count is 17 operations x forward/VJP = 34; dense estimator families included above.
  assert.equal(supported.size, 17);
  const representatives = new Map();
  for (const item of corpus.cases) {
    const key = `${item.operation}|${item.execution}`;
    if (supported.has(item.operation) && item.expected.kind === "success" &&
        !representatives.has(key)) representatives.set(key, item);
  }
  assert.equal(representatives.size, 34);
  for (const [key, item] of representatives) {
    const entries = [...item.inputs, ...item.expected.outputs];
    let offset = 0;
    const buffers = entries.map((entry) => {
      const elements = entry.shape.reduce((product, value) => product * value, 1);
      const byteLength = elements * (entry.data.dtype === "bytes" ? 1 : 4);
      const result = {
        id: entry.name,
        role: "activation",
        dtype: entry.data.dtype,
        shape: [...entry.shape],
        aliasOf: null,
        ownerId: entry.name,
        byteOffset: offset,
        byteLength,
        backwardInitialization: "none",
      };
      offset += Math.ceil(byteLength / 16) * 16;
      return result;
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
    const operationId = `canonical.${key}`;
    const operations = item.execution === "forward"
      ? [{
        id: operationId,
        operation: item.operation,
        inputs: item.inputs.map((entry) => entry.name),
        outputs: item.expected.outputs.map((entry) => entry.name),
        attributes,
      }]
      : [];
    const backwards = item.execution === "vjp"
      ? [{
        id: operationId,
        sourceOperationId: `source.${key}`,
        operation: item.operation,
        execution: "vjp",
        inputs: item.inputs.map(binding),
        outputs: item.expected.outputs.map(binding),
        attributes,
      }]
      : [];
    const commands = lowerPointwiseWebGpuOperationV1(
      plan(buffers, operations, backwards),
      item.execution === "forward" ? "forward" : "backward",
      operationId,
      0,
    );
    const form = webGpuDispatchFormV1(item.operation, item.execution);
    assert.equal(commands.length, form.stages.length, key);
    assert.deepEqual(
      commands.map((command) => uniform(command).getUint32(4, true)),
      form.stages.map((stage) => stage.selector),
      key,
    );
  }
});

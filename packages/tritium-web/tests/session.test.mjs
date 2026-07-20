import assert from "node:assert/strict";
import test from "node:test";

import {
  canonicalTrainingManifestJson,
  parseTrainingManifest,
  prepareTraining,
  TRAINING_MANIFEST_DIGEST_V1,
  TRAINING_VECTOR_DIGEST_V1,
  WebTrainingError,
} from "../dist/index.js";

const operations = parseTrainingManifest(
  canonicalTrainingManifestJson(),
).operations.map((operation) => operation.id);

const model = {
  schemaId: "tritium.web_training_model",
  schemaVersion: 1,
  recipe: {
    schemaId: "tritium.training_recipe",
    schemaVersion: 1,
    tensors: [
      { id: "x", dtype: "f32", shape: [1], role: "batch", aliasOf: null },
      { id: "target", dtype: "f32", shape: [1], role: "batch", aliasOf: null },
      { id: "weight", dtype: "f32", shape: [1], role: "parameter", aliasOf: null },
      { id: "tied-weight", dtype: "f32", shape: [1], role: "parameter", aliasOf: "weight" },
      { id: "grad", dtype: "f32", shape: [1], role: "gradient", aliasOf: null },
      { id: "sum", dtype: "f32", shape: [1], role: "activation", aliasOf: null },
      { id: "loss", dtype: "f32", shape: [], role: "result", aliasOf: null },
    ],
    operations: [
      { id: "add", operation: "graph.add", inputs: ["x", "weight"], outputs: ["sum"], attributes: [] },
      { id: "mse", operation: "loss.mse", inputs: ["sum", "target"], outputs: ["loss"], attributes: [] },
      { id: "sgd", operation: "optimizer.sgd", inputs: ["weight", "grad"], outputs: ["weight"], attributes: [] },
    ],
  },
  payload: Buffer.from([1, 2, 3]),
};

const config = {
  backend: "webgpu",
  allowWasmFallback: false,
  maxResidentBytes: 2048,
  seed: 7,
  requiredOperations: ["lifecycle.checkpoint", "lifecycle.export"],
};

function batch() {
  return {
    inputs: {
      x: new Float32Array([1]),
      target: new Float32Array([0]),
    },
  };
}

function capabilities(implementation = "webgpu") {
  return {
    schemaId: "tritium.web_training_capabilities",
    schemaVersion: 1,
    implementation,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V1,
    vectorDigest: TRAINING_VECTOR_DIGEST_V1,
    buildId: "test-adapter-v1",
    physicalDevice: implementation === "webgpu" ? "test-gpu" : "test-wasm",
    supportedOperations: operations,
    maxResidentBytes: 4096,
  };
}

function receipt(operation, implementation = "webgpu", overrides = {}) {
  return {
    schemaId: "tritium.web_training_receipt",
    schemaVersion: 1,
    implementation,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V1,
    vectorDigest: TRAINING_VECTOR_DIGEST_V1,
    buildId: "test-adapter-v1",
    physicalDevice: implementation === "webgpu" ? "test-gpu" : "test-wasm",
    operation,
    completedSteps: operation === "session.step" ? 1 : 0,
    peakResidentBytes: 1024,
    ...overrides,
  };
}

class MockAdapter {
  constructor(implementation = "webgpu") {
    this.capabilities = capabilities(implementation);
    this.implementation = implementation;
    this.calls = [];
    this.forwardBarrier = null;
    this.completedSteps = 0;
  }

  async validate(preparedModel) {
    this.calls.push("validate");
    preparedModel.payload[0] = 77;
  }

  async prepare(preparedModel, _config, plan) {
    this.calls.push("prepare");
    this.plan = plan;
    this.prepareSaw = preparedModel.payload[0];
    preparedModel.payload[0] = 99;
    return receipt("session.prepare", this.implementation, {
      completedSteps: this.completedSteps,
    });
  }

  async forward(preparedBatch) {
    this.calls.push("forward");
    preparedBatch.inputs.x[0] = 99;
    if (this.forwardBarrier !== null) await this.forwardBarrier;
    return {
      loss: 0.25,
      receipt: receipt("session.forward", this.implementation, {
        completedSteps: this.completedSteps,
      }),
    };
  }

  async backward() {
    this.calls.push("backward");
    return receipt("session.backward", this.implementation, {
      completedSteps: this.completedSteps,
    });
  }

  async step() {
    this.calls.push("step");
    this.completedSteps += 1;
    return receipt("session.step", this.implementation, {
      completedSteps: this.completedSteps,
    });
  }

  async checkpoint() {
    this.calls.push("checkpoint");
    this.checkpointBytes = Buffer.from([4, 5]);
    return {
      bytes: this.checkpointBytes,
      receipt: receipt("session.checkpoint", this.implementation, {
        completedSteps: this.completedSteps,
      }),
    };
  }

  async resume(bytes) {
    this.calls.push(`resume:${bytes[0]}`);
    bytes[0] = 88;
    return receipt("session.resume", this.implementation, {
      completedSteps: this.completedSteps,
    });
  }

  async export() {
    this.calls.push("export");
    return {
      bytes: new Uint8Array([6, 7]),
      receipt: receipt("session.export", this.implementation, {
        completedSteps: this.completedSteps,
      }),
    };
  }

  async dispose() {
    this.calls.push("dispose");
  }
}

async function rejectsCode(promise, code) {
  await assert.rejects(promise, (error) => {
    assert.ok(error instanceof WebTrainingError);
    assert.equal(error.code, code);
    return true;
  });
}

test("checked session executes the complete lifecycle in order", async () => {
  const adapter = new MockAdapter();
  const session = await prepareTraining(model, config, adapter);
  assert.equal(model.payload[0], 1, "adapter receives an isolated model payload");
  assert.equal(adapter.prepareSaw, 1, "validation cannot mutate prepare payload");
  assert.equal(session.state, "prepared");
  assert.equal(session.plan.residentBytes, 132);
  assert.equal(session.plan.batchStagingBytes, 8);
  assert.equal(session.plan.preparePeakBytes, 138);
  assert.equal(session.plan.forwardPeakBytes, 140);
  assert.equal(session.plan.peakBytes, 140);
  const weight = session.plan.buffers.find((buffer) => buffer.id === "weight");
  const tied = session.plan.buffers.find((buffer) => buffer.id === "tied-weight");
  assert.equal(tied.ownerId, "weight");
  assert.equal(tied.byteOffset, weight.byteOffset);
  assert.equal(tied.byteLength, weight.byteLength);
  assert.ok(Object.isFrozen(session.plan));
  assert.ok(Object.isFrozen(session.plan.buffers));
  assert.deepEqual(
    session.plan.operations.find((operation) => operation.id === "sgd").inputs,
    ["weight", "grad"],
  );
  assert.deepEqual(
    session.plan.backwardOperations.map((operation) => [
      operation.sourceOperationId,
      operation.execution,
    ]),
    [
      ["mse", "vjp"],
      ["add", "vjp"],
    ],
  );
  assert.equal(
    session.plan.buffers.find((buffer) => buffer.backwardInitialization === "one").shape.length,
    0,
  );
  assert.equal(
    session.plan.buffers.find((buffer) => buffer.id === "grad").backwardInitialization,
    "zero",
  );
  assert.equal(
    session.plan.backwardOperations[1].outputs.find(
      (binding) => binding.role === "grad_right",
    ).bufferId,
    "grad",
  );

  await rejectsCode(session.step(), "invalid_state");
  const callerBatch = batch();
  const result = await session.forward(callerBatch);
  assert.equal(callerBatch.inputs.x[0], 1, "adapter receives isolated batch staging");
  assert.ok(Object.isFrozen(result));
  assert.ok(Object.isFrozen(result.receipt));
  assert.equal(session.state, "forward-complete");
  await rejectsCode(session.backward({ ...result }), "invalid_state");
  await session.backward(result);
  assert.equal(session.state, "backward-complete");
  await rejectsCode(session.checkpoint(), "invalid_state");
  await session.step();
  assert.equal(session.state, "prepared");

  const checkpoint = await session.checkpoint();
  checkpoint.bytes[0] = 42;
  assert.equal(adapter.checkpointBytes[0], 4, "checkpoint output does not alias Buffer storage");
  await session.resume(checkpoint.bytes);
  assert.equal(checkpoint.bytes[0], 42, "resume receives an isolated byte copy");
  const artifact = await session.export();
  assert.deepEqual([...artifact.bytes], [6, 7]);
  await session.dispose();
  await session.dispose();
  assert.equal(session.state, "disposed");
  await rejectsCode(
    session.forward(batch()),
    "disposed",
  );
  assert.equal(adapter.calls.filter((call) => call === "dispose").length, 1);
});

test("backend policy and manifest coverage fail before adapter preparation", async () => {
  const wasm = new MockAdapter("wasm-fallback");
  await rejectsCode(prepareTraining(model, config, wasm), "backend_policy");
  assert.deepEqual(wasm.calls, []);

  const adapter = new MockAdapter();
  const badModel = {
    ...model,
    recipe: {
      ...model.recipe,
      operations: [
        { ...model.recipe.operations[0], operation: "graph.not_real" },
      ],
    },
  };
  await rejectsCode(
    prepareTraining(badModel, config, adapter),
    "invalid_schema",
  );
  assert.deepEqual(adapter.calls, []);
  await rejectsCode(prepareTraining(model, config), "adapter_unavailable");

  const validator = new MockAdapter();
  validator.validate = async () => {
    validator.calls.push("validate");
    throw new Error("invalid geometry");
  };
  await assert.rejects(prepareTraining(model, config, validator), /invalid geometry/);
  assert.deepEqual(validator.calls, ["validate"]);
});

test("planner rejects invalid ownership and memory before adapter allocation", async () => {
  const badOwner = {
    ...model,
    recipe: {
      ...model.recipe,
      tensors: model.recipe.tensors.map((tensor) =>
        tensor.id === "tied-weight" ? { ...tensor, aliasOf: "missing" } : tensor,
      ),
    },
  };
  const ownerAdapter = new MockAdapter();
  await rejectsCode(prepareTraining(badOwner, config, ownerAdapter), "invalid_schema");
  assert.deepEqual(ownerAdapter.calls, []);

  const nonScalarLoss = {
    ...model,
    recipe: {
      ...model.recipe,
      tensors: model.recipe.tensors.map((tensor) =>
        tensor.id === "loss" ? { ...tensor, role: "activation" } : tensor,
      ),
    },
  };
  const lossAdapter = new MockAdapter();
  await rejectsCode(prepareTraining(nonScalarLoss, config, lossAdapter), "invalid_schema");
  assert.deepEqual(lossAdapter.calls, []);

  const illegalWrite = {
    ...model,
    recipe: {
      ...model.recipe,
      operations: [
        { ...model.recipe.operations[0], outputs: ["weight"] },
        ...model.recipe.operations.slice(1),
      ],
    },
  };
  const writeAdapter = new MockAdapter();
  await rejectsCode(prepareTraining(illegalWrite, config, writeAdapter), "invalid_schema");
  assert.deepEqual(writeAdapter.calls, []);

  const memoryAdapter = new MockAdapter();
  await rejectsCode(
    prepareTraining(model, { ...config, maxResidentBytes: 90 }, memoryAdapter),
    "memory_limit",
  );
  assert.deepEqual(memoryAdapter.calls, []);
});

test("planner rejects non-representable and sparse typed attributes", async () => {
  for (const attribute of [
    { name: "rate", kind: "f32", value: Number.MAX_VALUE },
    { name: "shape", kind: "u32-list", value: new Array(2) },
  ]) {
    const badModel = {
      ...model,
      recipe: {
        ...model.recipe,
        operations: model.recipe.operations.map((operation, index) =>
          index === 0 ? { ...operation, attributes: [attribute] } : operation,
        ),
      },
    };
    const adapter = new MockAdapter();
    await rejectsCode(prepareTraining(badModel, config, adapter), "invalid_schema");
    assert.deepEqual(adapter.calls, []);
  }
});

test("planner enforces one optimizer and gradient per tied parameter owner", async () => {
  const aliasUpdate = {
    ...model,
    recipe: {
      ...model.recipe,
      operations: model.recipe.operations.map((operation) =>
        operation.id === "sgd"
          ? {
              ...operation,
              inputs: ["tied-weight", "grad"],
              outputs: ["tied-weight"],
            }
          : operation,
      ),
    },
  };
  const aliasAdapter = new MockAdapter();
  await rejectsCode(prepareTraining(aliasUpdate, config, aliasAdapter), "invalid_schema");
  assert.deepEqual(aliasAdapter.calls, []);

  const duplicateUpdate = {
    ...model,
    recipe: {
      ...model.recipe,
      operations: [
        ...model.recipe.operations,
        {
          ...model.recipe.operations[2],
          id: "sgd-again",
        },
      ],
    },
  };
  const duplicateAdapter = new MockAdapter();
  await rejectsCode(
    prepareTraining(duplicateUpdate, config, duplicateAdapter),
    "invalid_schema",
  );
  assert.deepEqual(duplicateAdapter.calls, []);

  const wrongGradientShape = {
    ...model,
    recipe: {
      ...model.recipe,
      tensors: model.recipe.tensors.map((tensor) =>
        tensor.id === "grad" ? { ...tensor, shape: [2] } : tensor,
      ),
    },
  };
  const shapeAdapter = new MockAdapter();
  await rejectsCode(
    prepareTraining(wrongGradientShape, config, shapeAdapter),
    "invalid_schema",
  );
  assert.deepEqual(shapeAdapter.calls, []);

  const orphanGradient = {
    ...model,
    recipe: {
      ...model.recipe,
      tensors: [
        ...model.recipe.tensors,
        { id: "orphan-grad", dtype: "f32", shape: [1], role: "gradient", aliasOf: null },
      ],
    },
  };
  const orphanAdapter = new MockAdapter();
  await rejectsCode(
    prepareTraining(orphanGradient, config, orphanAdapter),
    "invalid_schema",
  );
  assert.deepEqual(orphanAdapter.calls, []);
});

test("planner emits deterministic fan-in accumulation for tied parameters", async () => {
  const tiedModel = {
    ...model,
    recipe: {
      ...model.recipe,
      tensors: [
        ...model.recipe.tensors,
        { id: "sum2", dtype: "f32", shape: [1], role: "activation", aliasOf: null },
      ],
      operations: [
        model.recipe.operations[0],
        {
          id: "add-tied",
          operation: "graph.add",
          inputs: ["sum", "tied-weight"],
          outputs: ["sum2"],
          attributes: [],
        },
        { ...model.recipe.operations[1], inputs: ["sum2", "target"] },
        model.recipe.operations[2],
      ],
    },
  };
  const session = await prepareTraining(tiedModel, config, new MockAdapter());
  const reductions = session.plan.backwardOperations.filter(
    (operation) => operation.operation === "graph.add" && operation.execution === "forward",
  );
  assert.equal(reductions.length, 1);
  assert.equal(reductions[0].outputs[0].bufferId, "grad");
  assert.ok(
    reductions[0].inputs.every((binding) =>
      binding.bufferId.startsWith("__tritium.contribution."),
    ),
  );
  assert.equal(
    session.plan.backwardOperations.filter(
      (operation) =>
        operation.execution === "vjp" &&
        operation.outputs.some((binding) => binding.role === "grad_right"),
    ).length,
    2,
  );
  await session.dispose();
});

test("planner treats detach as a reverse-reachability barrier", async () => {
  const detachedModel = {
    ...model,
    recipe: {
      ...model.recipe,
      operations: [
        {
          id: "detach",
          operation: "graph.detach",
          inputs: ["weight"],
          outputs: ["sum"],
          attributes: [],
        },
        model.recipe.operations[1],
        model.recipe.operations[2],
      ],
    },
  };
  const session = await prepareTraining(detachedModel, config, new MockAdapter());
  assert.deepEqual(
    session.plan.backwardOperations.map((operation) => operation.sourceOperationId),
    ["mse"],
  );
  assert.equal(
    session.plan.buffers.find((buffer) => buffer.id === "grad").backwardInitialization,
    "zero",
  );
  await session.dispose();
});

test("planner binds stateful optimizer slots exclusively and positionally", async () => {
  const adamModel = {
    ...model,
    recipe: {
      ...model.recipe,
      tensors: [
        ...model.recipe.tensors,
        { id: "moment1", dtype: "f32", shape: [1], role: "optimizer-state", aliasOf: null },
        { id: "moment2", dtype: "f32", shape: [1], role: "optimizer-state", aliasOf: null },
      ],
      operations: model.recipe.operations.map((operation) =>
        operation.id === "sgd"
          ? {
              ...operation,
              operation: "optimizer.adamw",
              inputs: ["weight", "grad", "moment1", "moment2"],
              outputs: ["weight", "moment1", "moment2"],
            }
          : operation,
      ),
    },
  };
  const adapter = new MockAdapter();
  const session = await prepareTraining(adamModel, config, adapter);
  assert.deepEqual(session.plan.operations.find((item) => item.id === "sgd").inputs.slice(2), [
    "moment1",
    "moment2",
  ]);
  await session.dispose();

  const wrongStateRole = {
    ...adamModel,
    recipe: {
      ...adamModel.recipe,
      operations: adamModel.recipe.operations.map((operation) =>
        operation.id === "sgd"
          ? {
              ...operation,
              inputs: ["weight", "grad", "tied-weight", "moment2"],
              outputs: ["weight", "tied-weight", "moment2"],
            }
          : operation,
      ),
    },
  };
  await rejectsCode(
    prepareTraining(wrongStateRole, config, new MockAdapter()),
    "invalid_schema",
  );

  const wrongStateShape = {
    ...adamModel,
    recipe: {
      ...adamModel.recipe,
      tensors: adamModel.recipe.tensors.map((tensor) =>
        tensor.id === "moment2" ? { ...tensor, shape: [2] } : tensor,
      ),
    },
  };
  await rejectsCode(
    prepareTraining(wrongStateShape, config, new MockAdapter()),
    "invalid_schema",
  );
});

test("planner orders multiple groups and rejects shared optimizer state", async () => {
  const secondParameter = [
    { id: "weight2", dtype: "f32", shape: [1], role: "parameter", aliasOf: null },
    { id: "grad2", dtype: "f32", shape: [1], role: "gradient", aliasOf: null },
  ];
  const twoGroupModel = {
    ...model,
    recipe: {
      ...model.recipe,
      tensors: [...model.recipe.tensors, ...secondParameter],
      operations: [
        ...model.recipe.operations,
        {
          id: "sgd-second",
          operation: "optimizer.sgd",
          inputs: ["weight2", "grad2"],
          outputs: ["weight2"],
          attributes: [],
        },
      ],
    },
  };
  const session = await prepareTraining(twoGroupModel, config, new MockAdapter());
  assert.deepEqual(
    session.plan.operations
      .filter((operation) => operation.operation.startsWith("optimizer."))
      .map((operation) => operation.inputs[0]),
    ["weight", "weight2"],
  );
  await session.dispose();

  const sharedStateModel = {
    ...twoGroupModel,
    recipe: {
      ...twoGroupModel.recipe,
      tensors: [
        ...twoGroupModel.recipe.tensors,
        { id: "shared-momentum", dtype: "f32", shape: [1], role: "optimizer-state", aliasOf: null },
      ],
      operations: twoGroupModel.recipe.operations.map((operation) =>
        operation.operation === "optimizer.sgd"
          ? {
              ...operation,
              operation: "optimizer.muon",
              inputs: [operation.inputs[0], operation.inputs[1], "shared-momentum"],
              outputs: [operation.outputs[0], "shared-momentum"],
            }
          : operation,
      ),
    },
  };
  await rejectsCode(
    prepareTraining(sharedStateModel, config, new MockAdapter()),
    "invalid_schema",
  );
});

test("planner validates int8 AdamW block-state geometry", async () => {
  const tensors = [
    { id: "wide-weight", dtype: "f32", shape: [260], role: "parameter", aliasOf: null },
    { id: "wide-grad", dtype: "f32", shape: [260], role: "gradient", aliasOf: null },
    { id: "moment1-q8", dtype: "bytes", shape: [260], role: "optimizer-state", aliasOf: null },
    { id: "moment2-q8", dtype: "bytes", shape: [260], role: "optimizer-state", aliasOf: null },
    { id: "moment1-scale", dtype: "f32", shape: [2], role: "optimizer-state", aliasOf: null },
    { id: "moment2-scale", dtype: "f32", shape: [2], role: "optimizer-state", aliasOf: null },
  ];
  const operation = {
    id: "int8-adamw",
    operation: "optimizer.int8_adamw",
    inputs: [
      "wide-weight",
      "wide-grad",
      "moment1-q8",
      "moment2-q8",
      "moment1-scale",
      "moment2-scale",
    ],
    outputs: [
      "wide-weight",
      "moment1-q8",
      "moment2-q8",
      "moment1-scale",
      "moment2-scale",
    ],
    attributes: [],
  };
  const int8Model = {
    ...model,
    recipe: {
      ...model.recipe,
      tensors: [...model.recipe.tensors, ...tensors],
      operations: [...model.recipe.operations, operation],
    },
  };
  const int8Config = { ...config, maxResidentBytes: 4096 };
  const int8Adapter = new MockAdapter();
  int8Adapter.prepare = async (_preparedModel, _preparedConfig, plan) =>
    receipt("session.prepare", "webgpu", {
      peakResidentBytes: plan.residentBytes,
    });
  const session = await prepareTraining(int8Model, int8Config, int8Adapter);
  assert.deepEqual(session.plan.operations.find((item) => item.id === "int8-adamw").inputs.slice(2), [
    "moment1-q8",
    "moment2-q8",
    "moment1-scale",
    "moment2-scale",
  ]);
  await session.dispose();

  const badScaleModel = {
    ...int8Model,
    recipe: {
      ...int8Model.recipe,
      tensors: int8Model.recipe.tensors.map((tensor) =>
        tensor.id === "moment2-scale" ? { ...tensor, shape: [1] } : tensor,
      ),
    },
  };
  await rejectsCode(
    prepareTraining(badScaleModel, int8Config, new MockAdapter()),
    "invalid_schema",
  );
});

test("prepare uses one pre-await caller snapshot", async () => {
  const mutableModel = {
    ...model,
    payload: Uint8Array.from(model.payload),
  };
  const mutableConfig = { ...config };
  const adapter = new MockAdapter();
  let releaseValidation;
  adapter.validate = async (preparedModel) => {
    adapter.calls.push("validate");
    assert.equal(preparedModel.payload[0], 1);
    await new Promise((resolve) => {
      releaseValidation = resolve;
    });
  };
  const preparing = prepareTraining(mutableModel, mutableConfig, adapter);
  mutableModel.payload = Uint8Array.from([9, 9, 9, 9]);
  mutableConfig.maxResidentBytes = 1;
  releaseValidation();
  const session = await preparing;
  assert.equal(adapter.prepareSaw, 1);
  assert.equal(session.plan.preparePeakBytes, 138);
  await session.dispose();
});

test("capability getters are snapshotted exactly once", async () => {
  const adapter = new MockAdapter();
  const source = capabilities();
  let buildReads = 0;
  adapter.capabilities = {
    ...source,
    get buildId() {
      buildReads += 1;
      return buildReads === 1 ? source.buildId : "unvalidated-build";
    },
  };
  const session = await prepareTraining(model, config, adapter);
  assert.equal(buildReads, 1);
  assert.equal(session.capabilities.buildId, source.buildId);
  await session.dispose();
});

test("capability snapshot rejects unknown and malformed fields", async () => {
  const unknownAdapter = new MockAdapter();
  unknownAdapter.capabilities = { ...capabilities(), futureField: true };
  await rejectsCode(
    prepareTraining(model, config, unknownAdapter),
    "capability_mismatch",
  );
  assert.deepEqual(unknownAdapter.calls, []);

  const malformedAdapter = new MockAdapter();
  malformedAdapter.capabilities = {
    ...capabilities(),
    supportedOperations: null,
  };
  await rejectsCode(
    prepareTraining(model, config, malformedAdapter),
    "capability_mismatch",
  );
  assert.deepEqual(malformedAdapter.calls, []);
});

test("adapter outputs are snapshotted before validation", async () => {
  const adapter = new MockAdapter();
  const session = await prepareTraining(model, config, adapter);
  let lossReads = 0;
  let receiptReads = 0;
  let buildReads = 0;
  const forwardReceipt = receipt("session.forward");
  Object.defineProperty(forwardReceipt, "buildId", {
    enumerable: true,
    get() {
      buildReads += 1;
      return buildReads === 1 ? "test-adapter-v1" : "drifted-build";
    },
  });
  adapter.forward = async () => ({
    get loss() {
      lossReads += 1;
      return lossReads === 1 ? 0.5 : Number.NaN;
    },
    get receipt() {
      receiptReads += 1;
      return receiptReads === 1 ? forwardReceipt : receipt("wrong-operation");
    },
  });
  const result = await session.forward(batch());
  assert.equal(result.loss, 0.5);
  assert.equal(result.receipt.buildId, "test-adapter-v1");
  assert.equal(lossReads, 1);
  assert.equal(receiptReads, 1);
  assert.equal(buildReads, 1);

  await session.backward(result);
  await session.step();
  let byteReads = 0;
  adapter.checkpoint = async () => ({
    get bytes() {
      byteReads += 1;
      return byteReads === 1 ? Uint8Array.from([4, 5]) : new Uint8Array();
    },
    receipt: receipt("session.checkpoint", "webgpu", { completedSteps: 1 }),
  });
  const checkpoint = await session.checkpoint();
  assert.deepEqual([...checkpoint.bytes], [4, 5]);
  assert.equal(byteReads, 1);
  await session.dispose();
});

test("receipt resident and step counters gate state commits", async () => {
  const residentAdapter = new MockAdapter();
  const residentSession = await prepareTraining(model, config, residentAdapter);
  residentAdapter.forward = async () => ({
    loss: 1,
    receipt: receipt("session.forward", "webgpu", { peakResidentBytes: 0 }),
  });
  await rejectsCode(residentSession.forward(batch()), "memory_limit");
  assert.equal(residentSession.state, "prepared");

  const stepAdapter = new MockAdapter();
  const stepSession = await prepareTraining(model, config, stepAdapter);
  const result = await stepSession.forward(batch());
  await stepSession.backward(result);
  stepAdapter.step = async () =>
    receipt("session.step", "webgpu", { completedSteps: 7 });
  await rejectsCode(stepSession.step(), "invalid_receipt");
  assert.equal(stepSession.state, "backward-complete");
});

test("forward rejects batches that differ from the compiled plan", async () => {
  const adapter = new MockAdapter();
  const session = await prepareTraining(model, config, adapter);
  await rejectsCode(
    session.forward({ inputs: { x: new Float32Array([1]) } }),
    "invalid_schema",
  );
  await rejectsCode(
    session.forward({
      inputs: {
        x: new Uint32Array([1]),
        target: new Float32Array([0]),
      },
    }),
    "invalid_schema",
  );
  assert.deepEqual(adapter.calls, ["validate", "prepare"]);
  assert.equal(session.state, "prepared");
});

test("invalid receipts and adapter failures leave state uncommitted", async () => {
  const adapter = new MockAdapter();
  const session = await prepareTraining(model, config, adapter);
  adapter.forward = async () => {
    throw new Error("device lost");
  };
  await assert.rejects(
    session.forward(batch()),
    /device lost/,
  );
  assert.equal(session.state, "prepared");

  adapter.forward = async () => ({
    loss: 1,
    receipt: receipt("session.forward", "webgpu", {
      manifestDigest: "0".repeat(64),
    }),
  });
  await rejectsCode(
    session.forward(batch()),
    "invalid_receipt",
  );
  assert.equal(session.state, "prepared");
});

test("concurrent session mutations fail closed", async () => {
  const adapter = new MockAdapter();
  let release;
  adapter.forwardBarrier = new Promise((resolve) => {
    release = resolve;
  });
  const session = await prepareTraining(model, config, adapter);
  const forward = session.forward(batch());
  await new Promise((resolve) => setImmediate(resolve));
  await rejectsCode(session.checkpoint(), "busy");
  release();
  await forward;
  assert.equal(session.state, "forward-complete");
});

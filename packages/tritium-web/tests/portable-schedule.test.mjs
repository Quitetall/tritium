import assert from "node:assert/strict";
import test from "node:test";

import {
  PortableSchedulePlanError,
  compilePortableBackwardOperationRequest,
  compilePortablePlanOperationRequest,
  compileTrainingPlan,
} from "../dist/index.js";

const MAX_PORTABLE_TEST_BYTES = 8 * 1024 * 1024;

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
      {
        id: "add",
        operation: "graph.add",
        inputs: ["x", "tied-weight"],
        outputs: ["sum"],
        attributes: [],
      },
      {
        id: "mse",
        operation: "loss.mse",
        inputs: ["sum", "target"],
        outputs: ["loss"],
        attributes: [],
      },
      {
        id: "sgd",
        operation: "optimizer.sgd",
        inputs: ["weight", "grad"],
        outputs: ["weight"],
        attributes: [
          { name: "step", kind: "u64", value: 0 },
          { name: "lr", kind: "f32", value: 0.1 },
        ],
      },
    ],
  },
  payload: new Uint8Array([1]),
};

const config = {
  backend: "wasm",
  allowWasmFallback: true,
  maxResidentBytes: 4096,
  seed: 7,
  requiredOperations: ["graph.add", "loss.mse", "optimizer.sgd"],
};

function fixture() {
  const plan = compileTrainingPlan(model, config);
  const store = {};
  for (const buffer of plan.buffers) {
    if (store[buffer.ownerId] !== undefined) continue;
    const elements = buffer.dtype === "bytes" ? buffer.byteLength : buffer.byteLength / 4;
    store[buffer.ownerId] =
      buffer.dtype === "f32"
        ? new Float32Array(elements)
        : buffer.dtype === "u32"
          ? new Uint32Array(elements)
          : new Uint8Array(elements);
    if (buffer.backwardInitialization === "one") store[buffer.ownerId].fill(1);
  }
  store.x[0] = 1;
  store.target[0] = 2;
  store.weight[0] = 3;
  store.grad[0] = 4;
  return { plan, store };
}

function throwsCode(fn, code) {
  assert.throws(fn, (error) => {
    assert.ok(error instanceof PortableSchedulePlanError);
    assert.equal(error.code, code);
    return true;
  });
}

test("schedule compiler binds forward, VJP, and step roles exactly", () => {
  const { plan, store } = fixture();
  const add = compilePortablePlanOperationRequest(plan, "add", store, "node:schedule");
  assert.equal(add.request.execution, "forward");
  assert.deepEqual(add.request.inputs.map((buffer) => buffer.name), ["left", "right"]);
  assert.deepEqual(add.request.outputs.map((buffer) => buffer.name), ["result"]);
  assert.deepEqual(add.request.inputs[0].data.bits, [0x3f800000]);
  assert.deepEqual(
    add.request.inputs[1].data.bits,
    [0x40400000],
    "alias input resolves through its canonical owner",
  );
  assert.deepEqual(add.outputBufferIds, ["sum"]);

  const mse = compilePortablePlanOperationRequest(plan, "mse", store);
  assert.deepEqual(mse.request.inputs.map((buffer) => buffer.name), ["prediction", "target"]);
  assert.deepEqual(mse.request.outputs.map((buffer) => buffer.name), ["result"]);

  for (const operation of plan.backwardOperations) {
    const dispatch = compilePortableBackwardOperationRequest(plan, operation.id, store);
    assert.deepEqual(
      dispatch.request.inputs.map((buffer) => buffer.name),
      operation.inputs.map((binding) => binding.role),
    );
    assert.deepEqual(
      dispatch.request.outputs.map((buffer) => buffer.name),
      operation.outputs.map((binding) => binding.role),
    );
  }

  const step = compilePortablePlanOperationRequest(plan, "sgd", store);
  assert.equal(step.request.execution, "step");
  assert.deepEqual(step.request.inputs.map((buffer) => buffer.name), ["parameter", "gradient"]);
  assert.deepEqual(step.request.outputs.map((buffer) => buffer.name), ["parameter"]);
  assert.deepEqual(step.outputBufferIds, ["weight"]);
});

test("compiled dispatch owns immutable snapshots of caller tensors", () => {
  const { plan, store } = fixture();
  const dispatch = compilePortablePlanOperationRequest(plan, "add", store);
  store.x[0] = 99;
  store.weight[0] = 88;
  assert.deepEqual(dispatch.request.inputs[0].data.bits, [0x3f800000]);
  assert.deepEqual(dispatch.request.inputs[1].data.bits, [0x40400000]);
  assert.ok(Object.isFrozen(dispatch));
  assert.ok(Object.isFrozen(dispatch.request.inputs));
  assert.ok(Object.isFrozen(dispatch.request.inputs[0].data.bits));
});

test("f32 snapshots preserve raw IEEE lanes", () => {
  const { plan, store } = fixture();
  const lanes = new Uint32Array(store.x.buffer, store.x.byteOffset, store.x.length);
  for (const bits of [0x7fa12345, 0x80000000, 0x7f800000, 0xff800000]) {
    lanes[0] = bits;
    const dispatch = compilePortablePlanOperationRequest(plan, "add", store);
    assert.equal(dispatch.request.inputs[0].data.bits[0], bits);
  }
});

test("schedule compiler rejects malformed plans, stores, and role drift", () => {
  const { plan, store } = fixture();
  throwsCode(() => compilePortablePlanOperationRequest(null, "add", store), "invalid_schema");
  throwsCode(() => compilePortablePlanOperationRequest(plan, "missing", store), "invalid_schema");
  throwsCode(
    () => compilePortablePlanOperationRequest(plan, "add", { ...store, weight: undefined }),
    "missing_buffer",
  );
  throwsCode(
    () => compilePortablePlanOperationRequest(plan, "add", { ...store, weight: new Uint32Array(1) }),
    "buffer_mismatch",
  );
  throwsCode(
    () => compilePortablePlanOperationRequest(plan, "add", { ...store, weight: new Float32Array(2) }),
    "buffer_mismatch",
  );
  const aliasLocalStore = { ...store, "tied-weight": new Float32Array([5]) };
  delete aliasLocalStore.weight;
  throwsCode(
    () => compilePortablePlanOperationRequest(plan, "add", aliasLocalStore),
    "missing_buffer",
  );
  const redirectedBuffers = plan.buffers.map((buffer) =>
    buffer.id === "tied-weight"
      ? {
          ...buffer,
          ownerId: "x",
          byteOffset: plan.buffers.find((candidate) => candidate.id === "x").byteOffset,
        }
      : buffer,
  );
  throwsCode(
    () => compilePortablePlanOperationRequest({ ...plan, buffers: redirectedBuffers }, "add", store),
    "invalid_schema",
  );
  const coherentRedirect = plan.buffers.map((buffer) =>
    buffer.id === "tied-weight"
      ? {
          ...buffer,
          aliasOf: "x",
          ownerId: "x",
          byteOffset: plan.buffers.find((candidate) => candidate.id === "x").byteOffset,
        }
      : buffer,
  );
  throwsCode(
    () => compilePortablePlanOperationRequest({ ...plan, buffers: coherentRedirect }, "add", store),
    "invalid_schema",
  );
  const roleDriftBuffers = plan.buffers.map((buffer) =>
    buffer.id === "tied-weight" ? { ...buffer, role: "activation" } : buffer,
  );
  throwsCode(
    () => compilePortablePlanOperationRequest({ ...plan, buffers: roleDriftBuffers }, "add", store),
    "invalid_schema",
  );
  const forgedSgd = {
    ...plan,
    operations: plan.operations.map((operation) =>
      operation.id === "sgd" ? { ...operation, attributes: [] } : operation,
    ),
  };
  throwsCode(
    () => compilePortablePlanOperationRequest(forgedSgd, "sgd", store),
    "invalid_schema",
  );
  const oversizedBuffers = plan.buffers.map((buffer) =>
    buffer.id === "x"
      ? {
          ...buffer,
          shape: [MAX_PORTABLE_TEST_BYTES / 4 + 1],
          byteLength: MAX_PORTABLE_TEST_BYTES + 4,
        }
      : buffer,
  );
  throwsCode(
    () => compilePortablePlanOperationRequest({ ...plan, buffers: oversizedBuffers }, "add", store),
    "capacity",
  );
  throwsCode(
    () => compilePortablePlanOperationRequest({ ...plan, buffers: [null, ...plan.buffers] }, "add", store),
    "invalid_schema",
  );
  const sparseInputs = new Array(2);
  sparseInputs[0] = "x";
  throwsCode(
    () =>
      compilePortablePlanOperationRequest(
        {
          ...plan,
          operations: [
            { ...plan.operations[0], inputs: sparseInputs },
            ...plan.operations.slice(1),
          ],
        },
        "add",
        store,
      ),
    "invalid_schema",
  );

  const original = plan.backwardOperations[0];
  const forged = {
    ...plan,
    backwardOperations: [
      {
        ...original,
        inputs: [{ ...original.inputs[0], role: "forged_role" }, ...original.inputs.slice(1)],
      },
      ...plan.backwardOperations.slice(1),
    ],
  };
  throwsCode(
    () => compilePortableBackwardOperationRequest(forged, original.id, store),
    "invalid_schema",
  );
});

test("schedule compiler rejects aggregate JSON transport overflow", () => {
  const elements = 393_216;
  const largeModel = {
    ...model,
    recipe: {
      ...model.recipe,
      tensors: model.recipe.tensors.map((tensor) =>
        tensor.id === "loss" ? tensor : { ...tensor, shape: [elements] },
      ),
    },
  };
  const plan = compileTrainingPlan(largeModel, {
    ...config,
    maxResidentBytes: 64 * 1024 * 1024,
  });
  const store = {};
  for (const buffer of plan.buffers) {
    if (store[buffer.ownerId] !== undefined) continue;
    store[buffer.ownerId] = new Float32Array(buffer.byteLength / 4);
  }
  new Uint32Array(store.x.buffer).fill(0xffff_ffff);
  new Uint32Array(store.weight.buffer).fill(0xffff_ffff);
  throwsCode(
    () => compilePortablePlanOperationRequest(plan, "add", store),
    "capacity",
  );
});

import assert from "node:assert/strict";
import test from "node:test";

import {
  WebTrainingError,
  WebTrainingPayloadError,
  compileTrainingPlan,
  decodeWebTrainingPayload,
  encodeWebTrainingPayload,
  prepareTraining,
} from "../dist/index.js";

const model = {
  schemaId: "tritium.web_training_model",
  schemaVersion: 1,
  recipe: {
    schemaId: "tritium.training_recipe",
    schemaVersion: 1,
    tensors: [
      { id: "x", dtype: "f32", shape: [2], role: "batch", aliasOf: null },
      { id: "target", dtype: "f32", shape: [2], role: "batch", aliasOf: null },
      { id: "weight", dtype: "f32", shape: [2], role: "parameter", aliasOf: null },
      { id: "tied-weight", dtype: "f32", shape: [2], role: "parameter", aliasOf: "weight" },
      { id: "grad", dtype: "f32", shape: [2], role: "gradient", aliasOf: null },
      { id: "sum", dtype: "f32", shape: [2], role: "activation", aliasOf: null },
      { id: "loss", dtype: "f32", shape: [], role: "result", aliasOf: null },
    ],
    operations: [
      { id: "add", operation: "graph.add", inputs: ["x", "tied-weight"], outputs: ["sum"], attributes: [] },
      { id: "mse", operation: "loss.mse", inputs: ["sum", "target"], outputs: ["loss"], attributes: [] },
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
  payload: Uint8Array.of(1),
};

const config = {
  backend: "wasm",
  allowWasmFallback: true,
  maxResidentBytes: 4096,
  seed: 7,
  requiredOperations: ["graph.add", "loss.mse", "optimizer.sgd"],
};

test("canonical payload materializes exact owned training buffers", () => {
  const plan = compileTrainingPlan(model, config);
  const weight = new Float32Array(2);
  const bits = new Uint32Array(weight.buffer);
  bits[0] = 0x7fa12345;
  bits[1] = 0x80000000;

  const payload = encodeWebTrainingPayload({ weight });
  const store = decodeWebTrainingPayload(plan, payload);

  assert.deepEqual([...new Uint32Array(store.weight.buffer)], [...bits]);
  assert.equal(store["tied-weight"], undefined, "aliases never receive separate storage");
  assert.deepEqual([...store.grad], [0, 0]);
  const lossSeed = plan.buffers.find(
    (buffer) => buffer.backwardInitialization === "one",
  );
  assert.deepEqual([...store[lossSeed.ownerId]], [1]);
  assert.deepEqual(
    Object.keys(store).sort(),
    [...new Set(plan.buffers.map((buffer) => buffer.ownerId))].sort(),
  );
});

function throwsPayloadCode(run, code) {
  assert.throws(run, (error) => {
    assert.ok(error instanceof WebTrainingPayloadError);
    assert.equal(error.code, code, error.message);
    return true;
  });
}

test("payload wire is deterministic and fails closed on corruption or drift", () => {
  const plan = compileTrainingPlan(model, config);
  const weight = new Float32Array([1, -0]);
  const first = encodeWebTrainingPayload({ weight });
  const second = encodeWebTrainingPayload({ weight: Float32Array.from(weight) });
  assert.deepEqual(first, second);
  assert.deepEqual([...first.subarray(0, 12)], [
    0x54, 0x52, 0x57, 0x45, 0x42, 0x50, 0x31, 0,
    1, 0, 0, 0,
  ]);

  weight[0] = 99;
  assert.deepEqual([...decodeWebTrainingPayload(plan, first).weight], [1, -0]);

  const corrupt = Uint8Array.from(first);
  corrupt[corrupt.length - 1] ^= 1;
  throwsPayloadCode(() => decodeWebTrainingPayload(plan, corrupt), "integrity");
  throwsPayloadCode(
    () => decodeWebTrainingPayload(plan, encodeWebTrainingPayload({ weight: new Uint32Array(2) })),
    "buffer_mismatch",
  );
  throwsPayloadCode(
    () => decodeWebTrainingPayload(plan, encodeWebTrainingPayload({ weight: new Float32Array(1) })),
    "buffer_mismatch",
  );
  throwsPayloadCode(
    () => decodeWebTrainingPayload(plan, encodeWebTrainingPayload({ weight: new Float32Array(2), extra: Uint8Array.of(1) })),
    "buffer_mismatch",
  );
});

test("bundled WASM adapter executes compiled forward, backward, and step", async () => {
  const payload = encodeWebTrainingPayload({
    weight: new Float32Array([2, 3]),
  });
  const session = await prepareTraining({ ...model, payload }, config);
  assert.equal(session.capabilities.implementation, "wasm-fallback");

  const first = await session.forward({
    inputs: {
      x: new Float32Array([1, 1]),
      target: new Float32Array([0, 0]),
    },
  });
  assert.ok(Math.abs(first.loss - 12.5) < 1e-6);
  await session.backward(first);
  const step = await session.step();
  assert.equal(step.completedSteps, 1);

  const second = await session.forward({
    inputs: {
      x: new Float32Array([1, 1]),
      target: new Float32Array([0, 0]),
    },
  });
  assert.ok(Math.abs(second.loss - 10.125) < 1e-5);
  await session.dispose();
});

test("bundled WASM adapter checkpoints and resumes exact optimizer state", async () => {
  const train = await prepareTraining(
    { ...model, payload: encodeWebTrainingPayload({ weight: new Float32Array([2, 3]) }) },
    config,
  );
  const result = await train.forward({
    inputs: { x: new Float32Array([1, 1]), target: new Float32Array([0, 0]) },
  });
  await train.backward(result);
  await train.step();
  const checkpoint = await train.checkpoint();

  const resumed = await prepareTraining(
    { ...model, payload: encodeWebTrainingPayload({ weight: new Float32Array([9, 9]) }) },
    config,
  );
  const receipt = await resumed.resume(checkpoint.bytes);
  assert.equal(receipt.completedSteps, 1);
  const replay = await resumed.forward({
    inputs: { x: new Float32Array([1, 1]), target: new Float32Array([0, 0]) },
  });
  assert.ok(Math.abs(replay.loss - 10.125) < 1e-5);
  await train.dispose();
  await resumed.dispose();
});

test("bundled adapter exposes stable session errors", async () => {
  const validPayload = encodeWebTrainingPayload({
    weight: new Float32Array([2, 3]),
  });
  const corruptPayload = Uint8Array.from(validPayload);
  corruptPayload[corruptPayload.length - 1] ^= 1;
  await assert.rejects(
    prepareTraining({ ...model, payload: corruptPayload }, config),
    (error) => {
      assert.ok(error instanceof WebTrainingError);
      assert.equal(error.code, "invalid_schema");
      return true;
    },
  );

  const session = await prepareTraining({ ...model, payload: validPayload }, config);
  await assert.rejects(session.export(), (error) => {
    assert.ok(error instanceof WebTrainingError);
    assert.equal(error.code, "invalid_state");
    return true;
  });
  await session.dispose();
});

function saltTrainingModel(width, planes = 3, weights = undefined) {
  return {
    schemaId: "tritium.web_training_model",
    schemaVersion: 1,
    recipe: {
      schemaId: "tritium.training_recipe",
      schemaVersion: 1,
      tensors: [
        { id: "target", dtype: "f32", shape: [1, width], role: "batch", aliasOf: null },
        { id: "weight", dtype: "f32", shape: [1, width], role: "parameter", aliasOf: null },
        { id: "grad", dtype: "f32", shape: [1, width], role: "gradient", aliasOf: null },
        { id: "quant", dtype: "f32", shape: [1, width], role: "activation", aliasOf: null },
        { id: "loss", dtype: "f32", shape: [], role: "result", aliasOf: null },
      ],
      operations: [
        {
          id: "salt",
          operation: "graph.salt_ste",
          inputs: ["weight"],
          outputs: ["quant"],
          attributes: [
            { name: "rows", kind: "u64", value: 1 },
            { name: "cols", kind: "u64", value: width },
            { name: "planes", kind: "u64", value: planes },
          ],
        },
        {
          id: "mse",
          operation: "loss.mse",
          inputs: ["quant", "target"],
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
    payload: encodeWebTrainingPayload({
      weight: weights ?? Float32Array.from(
        { length: width },
        (_, index) => (index % 7 - 3) / 4,
      ),
    }),
  };
}

const saltConfig = {
  backend: "wasm",
  allowWasmFallback: true,
  maxResidentBytes: 1024 * 1024,
  seed: 7,
  requiredOperations: [
    "graph.salt_ste",
    "loss.mse",
    "optimizer.sgd",
    "lifecycle.export",
    "lifecycle.reload",
  ],
};

test("bundled adapter exports live additive parameters as strict canonical B3 SALT", async () => {
  const width = 1152; // four full allocation tiles plus one ragged tile
  const saltModel = saltTrainingModel(width);
  const compiled = compileTrainingPlan(saltModel, saltConfig);
  assert.ok(compiled.exportPackageBytes > 0);
  assert.ok(compiled.exportPeakBytes > compiled.forwardPeakBytes);
  assert.equal(
    compiled.peakBytes,
    Math.max(
      compiled.preparePeakBytes,
      compiled.forwardPeakBytes,
      compiled.exportPeakBytes,
    ),
  );
  const session = await prepareTraining(saltModel, saltConfig);
  const first = await session.export();
  const replay = await session.export();
  assert.deepEqual(first.bytes, replay.bytes, "unchanged live state exports deterministically");
  assert.equal(new TextDecoder().decode(first.bytes.subarray(0, 8)), "TSLT2PKG");
  assert.equal(first.bytes[10], 2, "browser export uses compact B3 coding");
  assert.equal(first.bytes.length, compiled.exportPackageBytes);
  assert.equal(first.receipt.operation, "session.export");
  assert.equal(first.receipt.peakResidentBytes, compiled.exportPeakBytes);

  const result = await session.forward({ inputs: { target: new Float32Array(width) } });
  await session.backward(result);
  await session.step();
  const changed = await session.export();
  assert.notDeepEqual(changed.bytes, first.bytes, "export is derived from updated parameters");
  assert.equal(changed.receipt.completedSteps, 1);
  await session.dispose();
});

function singleSaltTensorLayout(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const nameBytes = view.getUint32(24, true);
  const rank = view.getUint32(28, true);
  const payloadBytes = Number(view.getBigUint64(48, true));
  const scalesBytes = Number(view.getBigUint64(56, true));
  const payloadOffset = 24 + 64 + nameBytes + rank * 8;
  return {
    packedTensorCount: view.getUint32(12, true),
    packedTileCount: view.getBigUint64(40, true),
    payload: bytes.subarray(payloadOffset, payloadOffset + payloadBytes),
    scales: bytes.subarray(
      payloadOffset + payloadBytes,
      payloadOffset + payloadBytes + scalesBytes,
    ),
  };
}

test("B3 export carries terminal 2/4/6-bit and ragged allocation maps canonically", async () => {
  for (const [width, planes, embedded] of [
    [256, 2, 0b01],
    [512, 2, 0b0101],
    [768, 3, 0b11_1111],
  ]) {
    const session = await prepareTraining(saltTrainingModel(width, planes), saltConfig);
    const layout = singleSaltTensorLayout((await session.export()).bytes);
    assert.equal(layout.packedTensorCount & 0x03ff_ffff, 1);
    assert.equal(layout.packedTensorCount >>> 26, embedded, `${width} terminal map`);
    assert.equal(layout.packedTileCount >> 62n, 0n, `${width} has no ragged tile`);
    await session.dispose();
  }

  const ragged = await prepareTraining(saltTrainingModel(128, 3), saltConfig);
  const layout = singleSaltTensorLayout((await ragged.export()).bytes);
  assert.equal(layout.packedTensorCount >>> 26, 0);
  assert.equal(layout.packedTileCount >> 62n, 3n);
  await ragged.dispose();
});

test("B3 export matches additive trits and f16 round-to-nearest-even scale oracles", async () => {
  const cases = [
    {
      name: "mixed",
      weights: new Float32Array([-1, -0.5, 0, 0.5, 1]),
      payload: 225,
      scale: [0xcd, 0x38],
    },
    {
      name: "even tie",
      weights: new Float32Array(5).fill(1.00048828125),
      payload: 242,
      scale: [0x00, 0x3c],
    },
    {
      name: "minimum subnormal",
      weights: new Float32Array(5).fill(2 ** -24),
      payload: 242,
      scale: [0x01, 0x00],
    },
  ];
  for (const oracle of cases) {
    const session = await prepareTraining(
      saltTrainingModel(5, 1, oracle.weights),
      saltConfig,
    );
    const layout = singleSaltTensorLayout((await session.export()).bytes);
    assert.deepEqual([...layout.payload], [oracle.payload], `${oracle.name} trits`);
    assert.deepEqual([...layout.scales], oracle.scale, `${oracle.name} scale`);
    await session.dispose();
  }

  for (const [name, value] of [
    ["underflow", 2 ** -25],
    ["overflow", 65520],
  ]) {
    const session = await prepareTraining(
      saltTrainingModel(5, 1, new Float32Array(5).fill(value)),
      saltConfig,
    );
    await assert.rejects(session.export(), (error) => {
      assert.ok(error instanceof WebTrainingError);
      assert.equal(error.code, "invalid_state", name);
      return true;
    });
    await session.dispose();
  }
});

async function rejectsWebCode(promise, code) {
  await assert.rejects(promise, (error) => {
    assert.ok(error instanceof WebTrainingError);
    assert.equal(error.code, code, error.message);
    return true;
  });
}

test("built-in preflight rejects portable buffer and aggregate JSON capacity", async () => {
  const oversized = {
    ...model,
    recipe: {
      ...model.recipe,
      tensors: model.recipe.tensors.map((tensor) =>
        tensor.id !== "loss"
          ? { ...tensor, shape: [2_097_153] }
          : tensor,
      ),
    },
  };
  const largeConfig = { ...config, maxResidentBytes: 512 * 1024 * 1024 };
  await rejectsWebCode(prepareTraining(oversized, largeConfig), "memory_limit");

  const aggregate = {
    ...oversized,
    recipe: {
      ...oversized.recipe,
      tensors: oversized.recipe.tensors.map((tensor) =>
        tensor.id !== "loss"
          ? { ...tensor, shape: [1_000_000] }
          : tensor,
      ),
    },
  };
  await rejectsWebCode(prepareTraining(aggregate, largeConfig), "memory_limit");
});

test("built-in preflight rejects lifecycle capacity before guest creation", async () => {
  const boundaryModel = {
    ...model,
    recipe: {
      ...model.recipe,
      tensors: [
        ...model.recipe.tensors.map((tensor) =>
          tensor.id === "loss" ? tensor : { ...tensor, shape: [155_000] },
        ),
        { id: "moment1", dtype: "f32", shape: [155_000], role: "optimizer-state", aliasOf: null },
        { id: "moment2", dtype: "f32", shape: [155_000], role: "optimizer-state", aliasOf: null },
      ],
      operations: model.recipe.operations.map((operation) =>
        operation.id === "sgd"
          ? {
              ...operation,
              operation: "optimizer.adamw",
              inputs: ["weight", "grad", "moment1", "moment2"],
              outputs: ["weight", "moment1", "moment2"],
              attributes: [
                { name: "step", kind: "u64", value: 0 },
                { name: "lr", kind: "f32", value: 0.001 },
                { name: "beta1", kind: "f32", value: 0.9 },
                { name: "beta2", kind: "f32", value: 0.999 },
                { name: "eps", kind: "f32", value: 1e-8 },
                { name: "weight_decay", kind: "f32", value: 0 },
              ],
            }
          : operation,
      ),
    },
  };
  await rejectsWebCode(
    prepareTraining(boundaryModel, {
      ...config,
      maxResidentBytes: 512 * 1024 * 1024,
      requiredOperations: ["graph.add", "loss.mse", "optimizer.adamw"],
    }),
    "memory_limit",
  );

  const tensors = [
    { id: "x", dtype: "f32", shape: [1], role: "batch", aliasOf: null },
    { id: "target", dtype: "f32", shape: [1], role: "batch", aliasOf: null },
    { id: "loss", dtype: "f32", shape: [], role: "result", aliasOf: null },
  ];
  const operations = [];
  let input = "x";
  for (let index = 0; index < 22; index += 1) {
    const parameter = `weight-${index}`;
    const gradient = `grad-${index}`;
    const moment1 = `moment1-${index}`;
    const moment2 = `moment2-${index}`;
    const output = `sum-${index}`;
    tensors.push(
      { id: parameter, dtype: "f32", shape: [1], role: "parameter", aliasOf: null },
      { id: gradient, dtype: "f32", shape: [1], role: "gradient", aliasOf: null },
      { id: moment1, dtype: "f32", shape: [1], role: "optimizer-state", aliasOf: null },
      { id: moment2, dtype: "f32", shape: [1], role: "optimizer-state", aliasOf: null },
      { id: output, dtype: "f32", shape: [1], role: "activation", aliasOf: null },
    );
    operations.push({
      id: `add-${index}`,
      operation: "graph.add",
      inputs: [input, parameter],
      outputs: [output],
      attributes: [],
    });
    input = output;
  }
  operations.push({
    id: "mse",
    operation: "loss.mse",
    inputs: [input, "target"],
    outputs: ["loss"],
    attributes: [],
  });
  for (let index = 0; index < 22; index += 1) {
    operations.push({
      id: `adam-${index}`,
      operation: "optimizer.adamw",
      inputs: [`weight-${index}`, `grad-${index}`, `moment1-${index}`, `moment2-${index}`],
      outputs: [`weight-${index}`, `moment1-${index}`, `moment2-${index}`],
      attributes: [
        { name: "step", kind: "u64", value: 0 },
        { name: "lr", kind: "f32", value: 0.001 },
        { name: "beta1", kind: "f32", value: 0.9 },
        { name: "beta2", kind: "f32", value: 0.999 },
        { name: "eps", kind: "f32", value: 1e-8 },
        { name: "weight_decay", kind: "f32", value: 0 },
      ],
    });
  }
  const lifecycleModel = {
    schemaId: "tritium.web_training_model",
    schemaVersion: 1,
    recipe: {
      schemaId: "tritium.training_recipe",
      schemaVersion: 1,
      tensors,
      operations,
    },
    payload: Uint8Array.of(1),
  };
  await rejectsWebCode(
    prepareTraining(lifecycleModel, {
      ...config,
      maxResidentBytes: 64 * 1024 * 1024,
      requiredOperations: ["graph.add", "loss.mse", "optimizer.adamw"],
    }),
    "memory_limit",
  );
});

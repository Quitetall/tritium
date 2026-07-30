import assert from "node:assert/strict";
import test from "node:test";

import {
  compileWebGpuResidentScheduleV1,
  createWebGpuTrainingAdapter,
  encodeWebTrainingPayload,
  prepareTraining,
  TRAINING_MANIFEST_DIGEST_V2,
  WebGpuResidentRuntimeV1,
  WebTrainingError,
} from "../dist/index.js";

class FakeBuffer {
  constructor(size, device, label) {
    this.size = size;
    this.label = label;
    this.bytes = new Uint8Array(size);
    this.device = device;
  }
  async mapAsync() {
    this.device.maps += 1;
    if (this.device.mapGate !== null) await this.device.mapGate;
  }
  getMappedRange(offset = 0, size = this.size) {
    return this.bytes.slice(offset, offset + size).buffer;
  }
  unmap() {}
  destroy() { this.destroyed = true; }
}

class FakeDevice {
  constructor(overrides = {}) {
    this.limits = {
      maxBufferSize: 1 << 24,
      maxStorageBufferBindingSize: 1 << 24,
      maxComputeWorkgroupsPerDimension: 65535,
      maxBindingsPerBindGroup: 16,
      maxStorageBuffersPerShaderStage: 16,
      maxUniformBuffersPerShaderStage: 12,
      maxUniformBufferBindingSize: 65536,
      minUniformBufferOffsetAlignment: 256,
      ...overrides,
    };
    this.maps = 0;
    this.mapGate = null;
    this.submits = 0;
    this.bindGroups = 0;
    this.pipelines = 0;
    this.destroyed = false;
    this.events = [];
    this.buffers = new Map();
    this.lost = new Promise((resolve) => { this.lose = resolve; });
    this.queue = {
      writeBuffer: (buffer, offset, data) => buffer.bytes.set(data, offset),
      submit: (commands) => {
        this.submits += 1;
        for (const command of commands) command();
      },
      onSubmittedWorkDone: async () => {},
    };
  }
  createShaderModule(descriptor) { return descriptor; }
  async createComputePipelineAsync() {
    this.pipelines += 1;
    return { getBindGroupLayout: () => ({}) };
  }
  createBuffer({ label, size }) {
    const buffer = new FakeBuffer(size, this, label);
    this.buffers.set(label, buffer);
    return buffer;
  }
  createBindGroup(descriptor) {
    this.bindGroups += 1;
    return descriptor;
  }
  createCommandEncoder() {
    const copies = [];
    return {
      beginComputePass: () => ({
        setPipeline() {},
        setBindGroup() {},
        dispatchWorkgroups: () => this.events.push("dispatch"),
        end() {},
      }),
      copyBufferToBuffer: (source, sourceOffset, destination, destinationOffset, size) => {
        this.events.push(`copy:${source.label}>${destination.label}`);
        copies.push(() => destination.bytes.set(
          source.bytes.slice(sourceOffset, sourceOffset + size),
          destinationOffset,
        ));
      },
      clearBuffer: (buffer, offset = 0, size = buffer.size - offset) => {
        this.events.push(`clear:${buffer.label}`);
        copies.push(() => buffer.bytes.fill(0, offset, offset + size));
      },
      finish: () => () => copies.forEach((copy) => copy()),
    };
  }
  destroy() { this.destroyed = true; }
}

const buffer = (id, byteOffset) => ({
  id,
  role: id,
  dtype: "f32",
  shape: [1],
  aliasOf: null,
  ownerId: id,
  byteOffset,
  byteLength: 4,
  backwardInitialization: "none",
});

function plan() {
  return {
    schemaId: "tritium.compiled_training_plan",
    schemaVersion: 1,
    manifestDigest: "unused-by-low-level-runtime",
    buffers: [buffer("left", 0), buffer("right", 16), buffer("result", 32)],
    operations: [{
      id: "add",
      operation: "graph.add",
      inputs: ["left", "right"],
      outputs: ["result"],
      attributes: [],
    }],
    backwardOperations: [],
    residentBytes: 48,
    batchStagingBytes: 0,
    preparePeakBytes: 48,
    forwardPeakBytes: 48,
    exportPackageBytes: 0,
    exportPeakBytes: 48,
    peakBytes: 48,
  };
}

const command = Object.freeze({
  operation: "graph.add",
  execution: "forward",
  stageIndex: 0,
  uniformSlot: 0,
  uniformBytes: new Uint8Array(32),
  storageBindings: Object.freeze({ 1: "left", 2: "right", 3: "left", 4: "result" }),
  workgroups: Object.freeze([1, 1, 1]),
});

test("resident WebGPU transactions cache bindings and never map or read back", async () => {
  const device = new FakeDevice();
  const runtime = await WebGpuResidentRuntimeV1.prepare(device, plan(), [
    { bufferId: "left", bytes: Uint8Array.of(1, 2, 3, 4) },
    { bufferId: "right", bytes: Uint8Array.of(5, 6, 7, 8) },
  ]);
  assert.equal(device.pipelines, 1, "only plan-reachable pipelines compile");
  runtime.dispatch([command]);
  const bindGroups = device.bindGroups;
  runtime.dispatch([command]);
  assert.equal(device.bindGroups, bindGroups, "resident binding layout must be cached");
  assert.equal(device.maps, 0, "dispatch cannot map a GPU buffer");
  assert.equal(device.submits, 2);
  let commandReads = 0;
  await runtime.dispatchTransactions([{
    get commands() { commandReads += 1; return [command]; },
    copies: [],
    commitCopies: [],
  }]);
  assert.equal(commandReads, 1);
  const fieldReads = new Map();
  const accessorCommand = {};
  for (const key of Reflect.ownKeys(command)) {
    Object.defineProperty(accessorCommand, key, {
      enumerable: true,
      get() {
        fieldReads.set(key, (fieldReads.get(key) ?? 0) + 1);
        return command[key];
      },
    });
  }
  await runtime.dispatchTransactions([{
    commands: [accessorCommand], copies: [], commitCopies: [],
  }]);
  assert.deepEqual([...fieldReads.values()], Array(fieldReads.size).fill(1));
  assert.throws(
    () => runtime.dispatchTransactions([{
      get commands() { throw new Error("hostile"); },
      copies: [],
      commitCopies: [],
    }]),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  assert.throws(
    () => runtime.dispatchTransactions([{
      commands: [{
        ...command,
        get operation() { throw new Error("hostile"); },
      }],
      copies: [],
      commitCopies: [],
    }]),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );

  const result = await runtime.read("left");
  assert.deepEqual(result, Uint8Array.of(1, 2, 3, 4));
  assert.equal(device.maps, 1, "only explicit readback maps a staging buffer");
  runtime.dispose();
  assert.equal(device.destroyed, true);
});

test("resident replacement preserves every owner when one candidate upload fails", async () => {
  const device = new FakeDevice();
  const runtime = await WebGpuResidentRuntimeV1.prepare(device, plan(), [
    { bufferId: "left", bytes: Uint8Array.of(1, 2, 3, 4) },
    { bufferId: "right", bytes: Uint8Array.of(5, 6, 7, 8) },
    { bufferId: "result", bytes: Uint8Array.of(9, 10, 11, 12) },
  ]);
  const write = device.queue.writeBuffer;
  let replacementWrites = 0;
  device.queue.writeBuffer = (target, offset, data) => {
    if (target.label.startsWith("tritium:replacement:")) {
      replacementWrites += 1;
      if (replacementWrites === 2) throw new Error("injected upload failure");
    }
    write(target, offset, data);
  };
  await assert.rejects(
    runtime.replace([
      { bufferId: "left", bytes: Uint8Array.of(13, 14, 15, 16) },
      { bufferId: "right", bytes: Uint8Array.of(17, 18, 19, 20) },
    ], { residentPeakBytes: 12, maxPeakBytes: 20 }),
    (error) => error instanceof WebTrainingError && error.code === "adapter_failure",
  );
  assert.deepEqual(await runtime.read("left"), Uint8Array.of(1, 2, 3, 4));
  assert.deepEqual(await runtime.read("right"), Uint8Array.of(5, 6, 7, 8));
  runtime.dispose();
});

test("resident replacement rejects over-budget candidates before allocation", async () => {
  const device = new FakeDevice();
  const runtime = await WebGpuResidentRuntimeV1.prepare(device, plan(), [
    { bufferId: "left", bytes: Uint8Array.of(1, 2, 3, 4) },
    { bufferId: "right", bytes: Uint8Array.of(5, 6, 7, 8) },
    { bufferId: "result", bytes: Uint8Array.of(9, 10, 11, 12) },
  ]);
  const allocatedBefore = device.buffers.size;
  await assert.rejects(
    runtime.replace(
      [{ bufferId: "left", bytes: Uint8Array.of(13, 14, 15, 16) }],
      { residentPeakBytes: 12, maxPeakBytes: 15 },
    ),
    (error) => error instanceof WebTrainingError && error.code === "memory_limit",
  );
  assert.equal(device.buffers.size, allocatedBefore);
  assert.deepEqual(await runtime.read("left"), Uint8Array.of(1, 2, 3, 4));
  runtime.dispose();
});

test("resident replacement cancellation preserves the committed owner", async () => {
  const device = new FakeDevice();
  const runtime = await WebGpuResidentRuntimeV1.prepare(device, plan(), [
    { bufferId: "left", bytes: Uint8Array.of(1, 2, 3, 4) },
    { bufferId: "right", bytes: Uint8Array.of(5, 6, 7, 8) },
    { bufferId: "result", bytes: Uint8Array.of(9, 10, 11, 12) },
  ]);
  device.queue.onSubmittedWorkDone = () => new Promise(() => {});
  const cancellation = new AbortController();
  const replacing = runtime.replace(
    [{ bufferId: "left", bytes: Uint8Array.of(13, 14, 15, 16) }],
    { residentPeakBytes: 12, maxPeakBytes: 16 },
    cancellation.signal,
  );
  cancellation.abort();
  await assert.rejects(
    replacing,
    (error) => error instanceof WebTrainingError && error.code === "cancelled",
  );
  assert.deepEqual(await runtime.read("left"), Uint8Array.of(1, 2, 3, 4));
  assert.equal(device.buffers.get("tritium:replacement:left").destroyed, true);
  runtime.dispose();
});

test("auxiliary resources stay resident and receive same-submission GPU copies", async () => {
  const device = new FakeDevice();
  const auxiliaryBytes = Uint8Array.of(9, 10, 11, 12, 13, 14, 15, 16);
  const preparing = WebGpuResidentRuntimeV1.prepare(
    device,
    plan(),
    [{ bufferId: "left", bytes: Uint8Array.of(1, 2, 3, 4) }],
    {
      maxBytes: 8,
      resources: [{ id: "scratch", byteLength: 8, initialBytes: auxiliaryBytes }],
    },
  );
  auxiliaryBytes.fill(99);
  const runtime = await preparing;
  runtime.dispatch(
    [{ ...command, storageBindings: { ...command.storageBindings, 3: "scratch" } }],
    [{
      source: "left",
      sourceOffset: 0,
      destination: "scratch",
      destinationOffset: 4,
      byteLength: 4,
    }],
  );
  assert.deepEqual(
    await runtime.read("scratch"),
    Uint8Array.of(9, 10, 11, 12, 1, 2, 3, 4),
  );
  assert.equal(device.submits, 2, "copy and dispatch share one submission; read is explicit");
  runtime.dispose();
});

test("candidate commits encode after compute in one submission", async () => {
  const device = new FakeDevice();
  const runtime = await WebGpuResidentRuntimeV1.prepare(
    device,
    plan(),
    [{ bufferId: "left", bytes: Uint8Array.of(1, 2, 3, 4) }],
    {
      maxBytes: 4,
      resources: [{ id: "candidate", byteLength: 4, initialBytes: null }],
    },
  );
  device.events.length = 0;
  runtime.dispatch(
    [command],
    [{
      source: "left", sourceOffset: 0,
      destination: "candidate", destinationOffset: 0, byteLength: 4,
    }],
    [{
      source: "candidate", sourceOffset: 0,
      destination: "right", destinationOffset: 0, byteLength: 4,
    }],
  );
  assert.deepEqual(device.events, [
    "copy:tritium:resident:left>tritium:auxiliary:candidate",
    "dispatch",
    "copy:tritium:auxiliary:candidate>tritium:resident:right",
  ]);
  assert.equal(device.submits, 1);
  assert.deepEqual(await runtime.read("right"), Uint8Array.of(1, 2, 3, 4));
  runtime.dispose();
});

test("candidate commits reject duplicate and chained physical destinations", async () => {
  const alias = {
    ...buffer("right_alias", 16),
    role: "parameter",
    aliasOf: "right",
    ownerId: "right",
  };
  const aliasedPlan = { ...plan(), buffers: [...plan().buffers, alias] };
  const device = new FakeDevice();
  const runtime = await WebGpuResidentRuntimeV1.prepare(
    device,
    aliasedPlan,
    [],
    {
      maxBytes: 8,
      resources: [
        { id: "candidate-a", byteLength: 4, initialBytes: null },
        { id: "candidate-b", byteLength: 4, initialBytes: null },
      ],
    },
  );
  const copy = (source, destination) => ({
    source, sourceOffset: 0, destination, destinationOffset: 0, byteLength: 4,
  });
  for (const commits of [
    [copy("candidate-a", "right"), copy("candidate-b", "right_alias")],
    [copy("candidate-a", "right"), copy("right_alias", "result")],
  ]) {
    assert.throws(
      () => runtime.dispatch([command], [], commits),
      (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
    );
  }
  assert.equal(device.submits, 0);
  runtime.dispose();
});

test("auxiliary getters are captured exactly once before preparation awaits", async () => {
  const reads = new Map();
  const once = (name, value) => ({
    enumerable: true,
    get() {
      const count = (reads.get(name) ?? 0) + 1;
      reads.set(name, count);
      if (count !== 1) throw new Error(`${name} read twice`);
      return value;
    },
  });
  const resource = {};
  Object.defineProperties(resource, {
    id: once("resource.id", "scratch"),
    byteLength: once("resource.byteLength", 4),
    initialBytes: once("resource.initialBytes", Uint8Array.of(1, 2, 3, 4)),
  });
  const auxiliary = {};
  Object.defineProperties(auxiliary, {
    maxBytes: once("set.maxBytes", 4),
    resources: once("set.resources", [resource]),
  });
  const runtime = await WebGpuResidentRuntimeV1.prepare(
    new FakeDevice(), plan(), [], auxiliary,
  );
  assert.deepEqual([...reads.values()], [1, 1, 1, 1, 1]);
  assert.deepEqual(await runtime.read("scratch"), Uint8Array.of(1, 2, 3, 4));
  runtime.dispose();
});

test("auxiliary and copy admission reject collisions, drift, and unsafe ranges", async () => {
  for (const auxiliary of [
    [{ id: "left", byteLength: 4, initialBytes: null }],
    [{ id: "scratch", byteLength: 3, initialBytes: null }],
    [{ id: "scratch", byteLength: 4, initialBytes: Uint8Array.of(1) }],
  ]) {
    await assert.rejects(
      WebGpuResidentRuntimeV1.prepare(new FakeDevice(), plan(), [], {
        maxBytes: 8,
        resources: auxiliary,
      }),
      (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
    );
  }
  await assert.rejects(
    WebGpuResidentRuntimeV1.prepare(new FakeDevice(), plan(), [], {
      maxBytes: 4,
      resources: [
        { id: "one", byteLength: 4, initialBytes: null },
        { id: "two", byteLength: 4, initialBytes: null },
      ],
    }),
    (error) => error instanceof WebTrainingError && error.code === "memory_limit",
  );
  const forgedAliasPlan = {
    ...plan(),
    buffers: [
      ...plan().buffers,
      { ...buffer("forged", 48), ownerId: "scratch" },
    ],
  };
  await assert.rejects(
    WebGpuResidentRuntimeV1.prepare(new FakeDevice(), forgedAliasPlan, [], {
      maxBytes: 4,
      resources: [{ id: "scratch", byteLength: 4, initialBytes: null }],
    }),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  const device = new FakeDevice();
  const runtime = await WebGpuResidentRuntimeV1.prepare(
    device,
    plan(),
    [],
    {
      maxBytes: 8,
      resources: [{ id: "scratch", byteLength: 8, initialBytes: null }],
    },
  );
  for (const copy of [
    { source: "missing", sourceOffset: 0, destination: "scratch", destinationOffset: 0, byteLength: 4 },
    { source: "left", sourceOffset: 0, destination: "scratch", destinationOffset: 6, byteLength: 4 },
    { source: "scratch", sourceOffset: 0, destination: "scratch", destinationOffset: 4, byteLength: 4 },
  ]) {
    assert.throws(
      () => runtime.dispatch([command], [copy]),
      (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
    );
  }
  assert.equal(device.submits, 0);
  runtime.dispose();
});

test("resident WebGPU admission and loss fail closed", async () => {
  await assert.rejects(
    WebGpuResidentRuntimeV1.prepare(
      new FakeDevice({ maxBindingsPerBindGroup: 1 }),
      plan(),
      [],
    ),
    (error) => error instanceof WebTrainingError && error.code === "capability_mismatch",
  );
  const device = new FakeDevice();
  const runtime = await WebGpuResidentRuntimeV1.prepare(device, plan(), []);
  device.lose({ reason: "destroyed" });
  await Promise.resolve();
  assert.throws(
    () => runtime.dispatch([command]),
    (error) => error instanceof WebTrainingError && error.code === "device_lost",
  );
  const rejectingDevice = new FakeDevice();
  const rejectingRuntime = await WebGpuResidentRuntimeV1.prepare(
    rejectingDevice, plan(), [],
  );
  rejectingDevice.queue.onSubmittedWorkDone = async () => {
    rejectingDevice.lose({ reason: "queue-rejected" });
    throw new Error("raw device loss");
  };
  await assert.rejects(
    rejectingRuntime.dispatchTransactions([{ commands: [command], copies: [], commitCopies: [] }]),
    (error) => error instanceof WebTrainingError && error.code === "device_lost",
  );
});

test("one transaction cannot alias mutable uniform slots", async () => {
  const runtime = await WebGpuResidentRuntimeV1.prepare(new FakeDevice(), plan(), []);
  assert.throws(
    () => runtime.dispatch([command, command]),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  runtime.dispose();
});

test("preparation snapshots caller bytes before first pipeline await", async () => {
  const bytes = Uint8Array.of(1, 2, 3, 4);
  const preparing = WebGpuResidentRuntimeV1.prepare(new FakeDevice(), plan(), [
    { bufferId: "left", bytes },
  ]);
  bytes.fill(9);
  const runtime = await preparing;
  assert.deepEqual(await runtime.read("left"), Uint8Array.of(1, 2, 3, 4));
  runtime.dispose();
});

test("dispatch rejects binding drift before submission", async () => {
  const device = new FakeDevice();
  const runtime = await WebGpuResidentRuntimeV1.prepare(device, plan(), []);
  assert.throws(
    () => runtime.dispatch([{ ...command, storageBindings: { ...command.storageBindings, 9: "left" } }]),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  assert.equal(device.submits, 0);
  runtime.dispose();
});

test("GPU transfers pad byte tensors and clear reused uniform tails", async () => {
  const device = new FakeDevice();
  const tiny = {
    ...buffer("tiny", 48),
    dtype: "bytes",
    shape: [3],
    byteLength: 3,
  };
  const tinyPlan = { ...plan(), buffers: [...plan().buffers, tiny], residentBytes: 64 };
  const runtime = await WebGpuResidentRuntimeV1.prepare(device, tinyPlan, [
    { bufferId: "tiny", bytes: Uint8Array.of(4, 5, 6) },
  ], {
    maxBytes: 4,
    resources: [{
      id: "packed-candidate", byteLength: 4, initialBytes: Uint8Array.of(7, 8, 9, 0),
    }],
  });
  assert.deepEqual(await runtime.read("tiny"), Uint8Array.of(4, 5, 6));

  runtime.dispatch([], [], [{
    source: "packed-candidate", sourceOffset: 0,
    destination: "tiny", destinationOffset: 0, byteLength: 4,
  }]);
  assert.deepEqual(await runtime.read("tiny"), Uint8Array.of(7, 8, 9));
  assert.throws(
    () => runtime.dispatch([], [], [{
      source: "packed-candidate", sourceOffset: 0,
      destination: "tiny", destinationOffset: 0, byteLength: 8,
    }]),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );

  runtime.dispatch([{ ...command, uniformBytes: new Uint8Array(256).fill(7) }]);
  runtime.dispatch([{ ...command, uniformBytes: Uint8Array.of(1, 2, 3, 4) }]);
  const uniform = device.buffers.get("tritium:uniform-arena").bytes;
  assert.deepEqual(uniform.slice(0, 6), Uint8Array.of(1, 2, 3, 4, 0, 0));
  runtime.dispose();
});

test("resident writes and same-submission clears stay root-owned and ordered", async () => {
  const device = new FakeDevice();
  const runtime = await WebGpuResidentRuntimeV1.prepare(device, plan(), []);
  runtime.write("left", Uint8Array.of(1, 2, 3, 4));
  assert.deepEqual(await runtime.read("left"), Uint8Array.of(1, 2, 3, 4));
  device.events.length = 0;
  runtime.dispatch([command], [], [], ["left"]);
  assert.deepEqual(device.events, ["clear:tritium:resident:left", "dispatch"]);
  assert.deepEqual(await runtime.read("left"), Uint8Array.of(0, 0, 0, 0));
  assert.throws(
    () => runtime.dispatch([], [], [], ["left", "left"]),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  assert.throws(
    () => runtime.write("left", Uint8Array.of(1)),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  runtime.dispose();
});

test("ordered phase submission copies a produced value only after its producer dispatch", async () => {
  const specs = [
    ["left", [1, 1]], ["right", [1, 1]], ["produced", [1, 1]],
    ["other", [1, 1]], ["joined", [1, 2]],
  ];
  const buffers = specs.map(([id, shape], index) => ({
    id, role: "activation", dtype: "f32", shape, aliasOf: null, ownerId: id,
    byteOffset: index * 16, byteLength: shape[1] * 4, backwardInitialization: "none",
  }));
  const dependentPlan = {
    schemaId: "tritium.compiled_training_plan",
    schemaVersion: 1,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V2,
    buffers,
    operations: [
      { id: "produce", operation: "graph.add", inputs: ["left", "right"], outputs: ["produced"], attributes: [] },
      {
        id: "join", operation: "graph.concat_cols", inputs: ["produced", "other"], outputs: ["joined"],
        attributes: [
          { name: "rows", kind: "u64", value: 1 },
          { name: "lens", kind: "u64-list", value: [1, 1] },
        ],
      },
    ],
    backwardOperations: [],
    residentBytes: 72,
    batchStagingBytes: 0,
    preparePeakBytes: 72,
    forwardPeakBytes: 72,
    exportPackageBytes: 0,
    exportPeakBytes: 72,
    peakBytes: 72,
  };
  const schedule = compileWebGpuResidentScheduleV1(
    dependentPlan, { maxPeakBytes: 1 << 20 },
  );
  const device = new FakeDevice();
  const runtime = await WebGpuResidentRuntimeV1.prepare(
    device, dependentPlan, [], schedule.auxiliaryResources(),
  );
  await runtime.dispatchTransactions([
    schedule.transaction("forward", "produce", 0),
    schedule.transaction("forward", "join", 1),
  ]);
  const producerDispatch = device.events.indexOf("dispatch");
  const producedCopy = device.events.findIndex((event) =>
    event.startsWith("copy:tritium:resident:produced>tritium:auxiliary:"),
  );
  assert.ok(producerDispatch >= 0 && producedCopy > producerDispatch, device.events);
  assert.equal(device.submits, 1);
  runtime.dispose();
});

function adapterTrainingFixture() {
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
        { id: "gradient", dtype: "f32", shape: [1], role: "gradient", aliasOf: null },
        { id: "sum", dtype: "f32", shape: [1], role: "activation", aliasOf: null },
        { id: "loss", dtype: "f32", shape: [], role: "result", aliasOf: null },
      ],
      operations: [
        { id: "add", operation: "graph.add", inputs: ["x", "weight"], outputs: ["sum"], attributes: [] },
        { id: "mse", operation: "loss.mse", inputs: ["sum", "target"], outputs: ["loss"], attributes: [] },
        {
          id: "sgd", operation: "optimizer.sgd", inputs: ["weight", "gradient"], outputs: ["weight"],
          attributes: [
            { name: "step", kind: "u64", value: 0 },
            { name: "lr", kind: "f32", value: 0.1 },
          ],
        },
      ],
    },
    payload: encodeWebTrainingPayload({ weight: new Float32Array([2]) }),
  };
  const config = {
    backend: "webgpu",
    allowWasmFallback: false,
    maxResidentBytes: 1 << 20,
    seed: 7,
    requiredOperations: ["graph.add", "loss.mse", "optimizer.sgd"],
  };
  return { model, config };
}

function adapterOptimizerFixture(optimizer) {
  const fixture = adapterTrainingFixture();
  if (optimizer === "sgd") return fixture;
  const tensors = [...fixture.model.recipe.tensors];
  const operation = { ...fixture.model.recipe.operations[2] };
  const payload = { weight: new Float32Array([2]) };
  if (optimizer === "adamw" || optimizer === "cautious_adamw") {
    tensors.push(
      { id: "moment1", dtype: "f32", shape: [1], role: "optimizer-state", aliasOf: null },
      { id: "moment2", dtype: "f32", shape: [1], role: "optimizer-state", aliasOf: null },
    );
    payload.moment1 = new Float32Array([0]);
    payload.moment2 = new Float32Array([0]);
    Object.assign(operation, {
      id: optimizer,
      operation: `optimizer.${optimizer}`,
      inputs: ["weight", "gradient", "moment1", "moment2"],
      outputs: ["weight", "moment1", "moment2"],
      attributes: [
        { name: "step", kind: "u64", value: 0 },
        { name: "lr", kind: "f32", value: 0.001 },
        { name: "beta1", kind: "f32", value: 0.9 },
        { name: "beta2", kind: "f32", value: 0.999 },
        { name: "eps", kind: "f32", value: 1e-8 },
        { name: "weight_decay", kind: "f32", value: 0 },
      ],
    });
  } else if (optimizer === "int8_adamw") {
    tensors.push(
      { id: "moment1", dtype: "bytes", shape: [1], role: "optimizer-state", aliasOf: null },
      { id: "moment2", dtype: "bytes", shape: [1], role: "optimizer-state", aliasOf: null },
      { id: "moment1-scale", dtype: "f32", shape: [1], role: "optimizer-state", aliasOf: null },
      { id: "moment2-scale", dtype: "f32", shape: [1], role: "optimizer-state", aliasOf: null },
    );
    payload.moment1 = new Uint8Array([0]);
    payload.moment2 = new Uint8Array([0]);
    payload["moment1-scale"] = new Float32Array([0]);
    payload["moment2-scale"] = new Float32Array([0]);
    Object.assign(operation, {
      id: optimizer,
      operation: `optimizer.${optimizer}`,
      inputs: ["weight", "gradient", "moment1", "moment2", "moment1-scale", "moment2-scale"],
      outputs: ["weight", "moment1", "moment2", "moment1-scale", "moment2-scale"],
      attributes: [
        { name: "step", kind: "u64", value: 0 },
        { name: "lr", kind: "f32", value: 0.001 },
        { name: "beta1", kind: "f32", value: 0.9 },
        { name: "beta2", kind: "f32", value: 0.999 },
        { name: "eps", kind: "f32", value: 1e-8 },
        { name: "weight_decay", kind: "f32", value: 0 },
      ],
    });
  } else {
    for (const id of ["x", "target", "weight", "gradient", "sum"]) {
      const index = tensors.findIndex((tensor) => tensor.id === id);
      tensors[index] = { ...tensors[index], shape: [1, 1] };
    }
    tensors.push(
      { id: "momentum", dtype: "f32", shape: [1, 1], role: "optimizer-state", aliasOf: null },
    );
    payload.momentum = new Float32Array([0]);
    Object.assign(operation, {
      id: optimizer,
      operation: `optimizer.${optimizer}`,
      inputs: ["weight", "gradient", "momentum"],
      outputs: ["weight", "momentum"],
      attributes: [
        { name: "step", kind: "u64", value: 0 },
        { name: "lr", kind: "f32", value: 0.01 },
        { name: "momentum", kind: "f32", value: 0.95 },
        { name: "weight_decay", kind: "f32", value: 0 },
        { name: "rows", kind: "u64", value: 1 },
        { name: "cols", kind: "u64", value: 1 },
        { name: "ns_steps", kind: "u64", value: 5 },
      ],
    });
  }
  return {
    model: {
      ...fixture.model,
      recipe: {
        ...fixture.model.recipe,
        tensors,
        operations: [...fixture.model.recipe.operations.slice(0, 2), operation],
      },
      payload: encodeWebTrainingPayload(payload),
    },
    config: {
      ...fixture.config,
      requiredOperations: ["graph.add", "loss.mse", `optimizer.${optimizer}`],
    },
  };
}

function adapterSaltFixture() {
  const width = 256;
  const model = {
    schemaId: "tritium.web_training_model",
    schemaVersion: 1,
    recipe: {
      schemaId: "tritium.training_recipe",
      schemaVersion: 1,
      tensors: [
        { id: "target", dtype: "f32", shape: [1, width], role: "batch", aliasOf: null },
        { id: "weight", dtype: "f32", shape: [1, width], role: "parameter", aliasOf: null },
        { id: "gradient", dtype: "f32", shape: [1, width], role: "gradient", aliasOf: null },
        { id: "quant", dtype: "f32", shape: [1, width], role: "activation", aliasOf: null },
        { id: "loss", dtype: "f32", shape: [], role: "result", aliasOf: null },
      ],
      operations: [
        {
          id: "salt", operation: "graph.salt_ste", inputs: ["weight"], outputs: ["quant"],
          attributes: [
            { name: "rows", kind: "u64", value: 1 },
            { name: "cols", kind: "u64", value: width },
            { name: "planes", kind: "u64", value: 2 },
          ],
        },
        { id: "mse", operation: "loss.mse", inputs: ["quant", "target"], outputs: ["loss"], attributes: [] },
        {
          id: "sgd", operation: "optimizer.sgd", inputs: ["weight", "gradient"], outputs: ["weight"],
          attributes: [
            { name: "step", kind: "u64", value: 0 },
            { name: "lr", kind: "f32", value: 0.1 },
          ],
        },
      ],
    },
    payload: encodeWebTrainingPayload({
      weight: Float32Array.from({ length: width }, (_, index) => (index % 9 - 4) / 8),
    }),
  };
  const config = {
    backend: "webgpu",
    allowWasmFallback: false,
    maxResidentBytes: 1 << 20,
    seed: 7,
    requiredOperations: [
      "graph.salt_ste", "loss.mse", "optimizer.sgd", "lifecycle.export", "lifecycle.reload",
    ],
  };
  return { model, config };
}

test("resident WebGPU adapter executes one command buffer per training phase", async () => {
  const device = new FakeDevice();
  const adapter = createWebGpuTrainingAdapter(device, {
    buildId: "test-webgpu-adapter",
    physicalDevice: "fake-gpu",
    maxResidentBytes: 1 << 20,
  });
  const { model, config } = adapterTrainingFixture();
  const session = await prepareTraining(model, config, adapter);
  device.events.length = 0;
  const result = await session.forward({
    inputs: { x: new Float32Array([3]), target: new Float32Array([0]) },
  });
  assert.equal(result.loss, 0, "the fake device intentionally does not execute WGSL math");
  assert.ok(result.receipt.peakResidentBytes > session.plan.peakBytes);
  assert.equal(device.events.filter((event) => event === "dispatch").length, 2);
  assert.equal(device.submits, 2, "forward dispatch and scalar readback are separate submissions");

  device.events.length = 0;
  const backward = await session.backward(result);
  assert.equal(backward.completedSteps, 0);
  assert.ok(device.events[0].startsWith("clear:"));
  assert.equal(device.submits, 3, "the complete backward graph uses one submission");

  device.events.length = 0;
  const step = await session.step();
  assert.equal(step.completedSteps, 1);
  assert.equal(device.submits, 4, "the optimizer transaction and commit use one submission");
  assert.ok(device.events.some((event) => event.startsWith("copy:")));

  const second = await session.forward({
    inputs: { x: new Float32Array([4]), target: new Float32Array([0]) },
  });
  await session.backward(second);
  device.queue.onSubmittedWorkDone = () => new Promise(() => {});
  const lostStep = session.step();
  device.lose({ reason: "test" });
  await assert.rejects(
    lostStep,
    (error) => error instanceof WebTrainingError && error.code === "device_lost",
  );
  assert.equal(session.state, "terminal");
  assert.equal(device.destroyed, true);
});

test("resident WebGPU checkpoint resumes exact optimizer state through public session", async () => {
  const firstFixture = adapterTrainingFixture();
  const first = await prepareTraining(
    firstFixture.model,
    firstFixture.config,
    createWebGpuTrainingAdapter(new FakeDevice(), {
      buildId: "test-webgpu-lifecycle",
      physicalDevice: "fake-gpu-a",
      maxResidentBytes: 1 << 20,
    }),
  );
  const saved = await first.checkpoint();
  assert.equal(saved.receipt.operation, "session.checkpoint");
  assert.equal(saved.receipt.completedSteps, 0);

  const secondFixture = adapterTrainingFixture();
  const changedModel = {
    ...secondFixture.model,
    payload: encodeWebTrainingPayload({ weight: new Float32Array([9]) }),
  };
  const second = await prepareTraining(
    changedModel,
    secondFixture.config,
    createWebGpuTrainingAdapter(new FakeDevice(), {
      buildId: "test-webgpu-lifecycle",
      physicalDevice: "fake-gpu-b",
      maxResidentBytes: 1 << 20,
    }),
  );
  const changed = await second.checkpoint();
  assert.notDeepEqual(changed.bytes, saved.bytes);
  const resumed = await second.resume(saved.bytes);
  assert.equal(resumed.operation, "session.resume");
  assert.equal(resumed.completedSteps, 0);
  const restored = await second.checkpoint();
  assert.deepEqual(restored.bytes, saved.bytes);

  await first.dispose();
  await second.dispose();
});

test("resident WebGPU resume honors the session ceiling before candidate allocation", async () => {
  const fixture = adapterTrainingFixture();
  const probe = await prepareTraining(
    fixture.model,
    fixture.config,
    createWebGpuTrainingAdapter(new FakeDevice(), { maxResidentBytes: 1 << 20 }),
  );
  const saved = await probe.checkpoint();
  const schedulePeak = compileWebGpuResidentScheduleV1(
    probe.plan, { maxPeakBytes: 1 << 20 },
  ).peakBytes();
  await probe.dispose();

  const device = new FakeDevice();
  const limited = await prepareTraining(
    fixture.model,
    { ...fixture.config, maxResidentBytes: schedulePeak },
    createWebGpuTrainingAdapter(device, { maxResidentBytes: 1 << 20 }),
  );
  await assert.rejects(
    limited.resume(saved.bytes),
    (error) => error instanceof WebTrainingError && error.code === "memory_limit",
  );
  assert.equal(limited.state, "prepared");
  assert.equal(
    [...device.buffers.keys()].some((label) => label.startsWith("tritium:replacement:")),
    false,
  );
  await limited.dispose();
});

test("resident WebGPU lifecycle matches portable WASM for every optimizer layout", async () => {
  for (const optimizer of ["sgd", "adamw", "cautious_adamw", "int8_adamw", "muon"]) {
    const { model, config } = adapterOptimizerFixture(optimizer);
    const portable = await prepareTraining(model, {
      ...config,
      backend: "wasm",
      allowWasmFallback: true,
    });
    const expected = await portable.checkpoint();
    const resident = await prepareTraining(
      model,
      config,
      createWebGpuTrainingAdapter(new FakeDevice(), {
        buildId: `test-webgpu-${optimizer}`,
        physicalDevice: "fake-gpu",
        maxResidentBytes: 1 << 20,
      }),
    );
    const actual = await resident.checkpoint();
    assert.deepEqual(actual.bytes, expected.bytes, optimizer);
    const receipt = await resident.resume(expected.bytes);
    assert.equal(receipt.completedSteps, 0, optimizer);
    assert.ok(receipt.peakResidentBytes > resident.plan.peakBytes, optimizer);
    assert.deepEqual((await resident.checkpoint()).bytes, expected.bytes, optimizer);
    await resident.dispose();
    await portable.dispose();
  }
});

test("resident WebGPU checkpoint cancellation leaves the public session reusable", async () => {
  const device = new FakeDevice();
  let releaseMap;
  device.mapGate = new Promise((resolve) => { releaseMap = resolve; });
  const { model, config } = adapterTrainingFixture();
  const session = await prepareTraining(
    model,
    config,
    createWebGpuTrainingAdapter(device, {
      buildId: "test-webgpu-cancellation",
      physicalDevice: "fake-gpu",
      maxResidentBytes: 1 << 20,
    }),
  );
  const cancellation = new AbortController();
  const checkpoint = session.checkpoint({ signal: cancellation.signal });
  await new Promise((resolve) => setImmediate(resolve));
  cancellation.abort();
  await assert.rejects(
    checkpoint,
    (error) => error instanceof WebTrainingError && error.code === "cancelled",
  );
  assert.equal(session.state, "prepared");
  releaseMap();
  device.mapGate = null;
  await new Promise((resolve) => setImmediate(resolve));
  const saved = await session.checkpoint();
  assert.equal(saved.receipt.operation, "session.checkpoint");
  await session.dispose();
});

test("resident WebGPU export derives deterministic strict SALT from live parameters", async () => {
  const { model, config } = adapterSaltFixture();
  const session = await prepareTraining(
    model,
    config,
    createWebGpuTrainingAdapter(new FakeDevice(), {
      buildId: "test-webgpu-export",
      physicalDevice: "fake-gpu",
      maxResidentBytes: 1 << 20,
    }),
  );
  const first = await session.export();
  const second = await session.export();
  assert.equal(new TextDecoder().decode(first.bytes.subarray(0, 8)), "TSLT2PKG");
  assert.deepEqual(second.bytes, first.bytes);
  assert.equal(first.receipt.operation, "session.export");
  assert.equal(first.receipt.completedSteps, 0);
  assert.equal(first.receipt.peakResidentBytes, session.plan.exportPeakBytes);
  await session.dispose();
});

test("prepareTraining acquires WebGPU and preserves strict fallback policy", async () => {
  const { model, config } = adapterTrainingFixture();
  const automaticDevice = new FakeDevice();
  let requested = 0;
  const priorNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      gpu: {
        async requestAdapter(options) {
          assert.deepEqual(options, { powerPreference: "high-performance" });
          return {
            async requestDevice() {
              requested += 1;
              return automaticDevice;
            },
          };
        },
      },
    },
  });
  try {
    const automatic = await prepareTraining(model, config);
    assert.equal(automatic.capabilities.implementation, "webgpu");
    assert.equal(requested, 1);
    await automatic.dispose();
    assert.equal(automaticDevice.destroyed, true);

    globalThis.navigator.gpu.requestAdapter = async () => null;
    await assert.rejects(
      prepareTraining(model, config),
      (error) => error instanceof WebTrainingError && error.code === "adapter_unavailable",
    );
    const fallback = await prepareTraining(model, {
      ...config,
      backend: "auto",
      allowWasmFallback: true,
    });
    assert.equal(fallback.capabilities.implementation, "wasm-fallback");
    await fallback.dispose();

    let destroyedInvalidDevice = false;
    globalThis.navigator.gpu.requestAdapter = async () => ({
      async requestDevice() {
        return {
          destroy() { destroyedInvalidDevice = true; },
        };
      },
    });
    await assert.rejects(
      prepareTraining(model, config),
      (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
    );
    assert.equal(destroyedInvalidDevice, true);

    const undersizedDevice = new FakeDevice({ maxBufferSize: 1024 });
    globalThis.navigator.gpu.requestAdapter = async () => ({
      async requestDevice() { return undersizedDevice; },
    });
    await assert.rejects(
      prepareTraining(model, config),
      (error) => error instanceof WebTrainingError && error.code === "memory_limit",
    );
    assert.equal(undersizedDevice.destroyed, true);
  } finally {
    if (priorNavigator === undefined) delete globalThis.navigator;
    else Object.defineProperty(globalThis, "navigator", priorNavigator);
  }
});

test("resident WebGPU adapter rejects malformed factory inputs with stable errors", async () => {
  assert.throws(
    () => createWebGpuTrainingAdapter(null),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  assert.throws(
    () => createWebGpuTrainingAdapter(new FakeDevice(), { maxResidentBytes: 0 }),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  assert.throws(
    () => createWebGpuTrainingAdapter(new FakeDevice(), new Proxy({}, {
      ownKeys() { throw new Error("hostile"); },
    })),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
  let reads = 0;
  const adapter = createWebGpuTrainingAdapter(new FakeDevice(), {
    get maxResidentBytes() { reads += 1; return 4096; },
  });
  assert.equal(reads, 1);
  assert.equal(adapter.capabilities.maxResidentBytes, 4096);
  const unpreparedDevice = new FakeDevice();
  const unprepared = createWebGpuTrainingAdapter(unpreparedDevice);
  await unprepared.dispose();
  assert.equal(unpreparedDevice.destroyed, true);
});

test("resident int8 AdamW binds each auto-layout entry point exactly", async () => {
  const specs = [
    ["parameter", "f32", [4], 16],
    ["gradient", "f32", [4], 16],
    ["moment1_q8", "bytes", [4], 4],
    ["moment2_q8", "bytes", [4], 4],
    ["moment1_scale", "f32", [1], 4],
    ["moment2_scale", "f32", [1], 4],
  ];
  let byteOffset = 0;
  const buffers = specs.map(([id, dtype, shape, byteLength]) => {
    const result = {
      id, role: "activation", dtype, shape, aliasOf: null, ownerId: id,
      byteOffset, byteLength, backwardInitialization: "none",
    };
    byteOffset += 16;
    return result;
  });
  const int8Plan = {
    schemaId: "tritium.compiled_training_plan",
    schemaVersion: 1,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V2,
    buffers,
    operations: [{
      id: "int8-step",
      operation: "optimizer.int8_adamw",
      inputs: specs.map(([id]) => id),
      outputs: ["parameter", "moment1_q8", "moment2_q8", "moment1_scale", "moment2_scale"],
      attributes: [
        { name: "step", kind: "u64", value: 0 },
        { name: "lr", kind: "f32", value: Math.fround(0.01) },
        { name: "beta1", kind: "f32", value: Math.fround(0.9) },
        { name: "beta2", kind: "f32", value: Math.fround(0.95) },
        { name: "eps", kind: "f32", value: Math.fround(1e-8) },
        { name: "weight_decay", kind: "f32", value: Math.fround(0.01) },
      ],
    }],
    backwardOperations: [],
    residentBytes: byteOffset,
    batchStagingBytes: 0,
    preparePeakBytes: byteOffset,
    forwardPeakBytes: byteOffset,
    exportPackageBytes: 0,
    exportPeakBytes: byteOffset,
    peakBytes: byteOffset,
  };
  const schedule = compileWebGpuResidentScheduleV1(
    int8Plan, { maxPeakBytes: 1 << 20 },
  );
  const device = new FakeDevice();
  const runtime = await WebGpuResidentRuntimeV1.prepare(
    device,
    int8Plan,
    buffers.map((entry) => ({ bufferId: entry.id, bytes: new Uint8Array(entry.byteLength) })),
    schedule.auxiliaryResources(),
  );
  const transaction = schedule.transaction("forward", "int8-step", 0, 1);
  runtime.dispatch(transaction.commands, transaction.copies, transaction.commitCopies);
  assert.equal(device.pipelines, 12);
  assert.equal(device.bindGroups, 12);
  assert.equal(device.events.filter((event) => event === "dispatch").length, 12);
  runtime.dispose();
});

test("bind-group cache keys cannot collide through tensor IDs", async () => {
  const device = new FakeDevice();
  const ids = ["a|2:s:b", "c", "a", "b|2:s:c", "extra", "result"];
  const collisionPlan = {
    ...plan(),
    buffers: ids.map((id, index) => buffer(id, index * 16)),
    residentBytes: ids.length * 16,
  };
  const runtime = await WebGpuResidentRuntimeV1.prepare(device, collisionPlan, []);
  runtime.dispatch([{
    ...command,
    storageBindings: { 1: ids[0], 2: ids[1], 3: "extra", 4: "result" },
  }]);
  const first = device.bindGroups;
  runtime.dispatch([{
    ...command,
    storageBindings: { 1: ids[2], 2: ids[3], 3: "extra", 4: "result" },
  }]);
  assert.equal(device.bindGroups, first + 1);
  runtime.dispose();
});

test("device loss during pipeline preparation stays typed", async () => {
  class LosingDevice extends FakeDevice {
    async createComputePipelineAsync() {
      this.lose({ reason: "destroyed" });
      return new Promise(() => {});
    }
  }
  await assert.rejects(
    WebGpuResidentRuntimeV1.prepare(new LosingDevice(), plan(), []),
    (error) => error instanceof WebTrainingError && error.code === "device_lost",
  );
});

test("malformed and partially reachable plans fail before GPU work", async () => {
  for (const malformed of [
    null,
    { ...plan(), buffers: [{ ...buffer("bad", 0), shape: null }] },
    { ...plan(), operations: [null] },
    { ...plan(), operations: [...plan().operations, {
      id: "unknown",
      operation: "graph.not_real",
      inputs: [],
      outputs: ["result"],
      attributes: [],
    }] },
  ]) {
    const device = new FakeDevice();
    await assert.rejects(
      WebGpuResidentRuntimeV1.prepare(device, malformed, []),
      (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
    );
    assert.equal(device.pipelines, 0);
    assert.equal(device.destroyed, false);
  }
  const sparse = new Array(1);
  await assert.rejects(
    WebGpuResidentRuntimeV1.prepare(new FakeDevice(), plan(), sparse),
    (error) => error instanceof WebTrainingError && error.code === "invalid_schema",
  );
});

import assert from "node:assert/strict";
import test from "node:test";

import { WebGpuResidentRuntimeV1, WebTrainingError } from "../dist/index.js";

class FakeBuffer {
  constructor(size, device) {
    this.size = size;
    this.bytes = new Uint8Array(size);
    this.device = device;
  }
  async mapAsync() { this.device.maps += 1; }
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
    this.submits = 0;
    this.bindGroups = 0;
    this.pipelines = 0;
    this.destroyed = false;
    this.buffers = new Map();
    this.lost = new Promise((resolve) => { this.lose = resolve; });
    this.queue = {
      writeBuffer: (buffer, offset, data) => buffer.bytes.set(data, offset),
      submit: (commands) => {
        this.submits += 1;
        for (const command of commands) command();
      },
    };
  }
  createShaderModule(descriptor) { return descriptor; }
  async createComputePipelineAsync() {
    this.pipelines += 1;
    return { getBindGroupLayout: () => ({}) };
  }
  createBuffer({ label, size }) {
    const buffer = new FakeBuffer(size, this);
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
        dispatchWorkgroups() {},
        end() {},
      }),
      copyBufferToBuffer: (source, sourceOffset, destination, destinationOffset, size) => {
        copies.push(() => destination.bytes.set(
          source.bytes.slice(sourceOffset, sourceOffset + size),
          destinationOffset,
        ));
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

  const result = await runtime.read("left");
  assert.deepEqual(result, Uint8Array.of(1, 2, 3, 4));
  assert.equal(device.maps, 1, "only explicit readback maps a staging buffer");
  runtime.dispose();
  assert.equal(device.destroyed, true);
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
  ]);
  assert.deepEqual(await runtime.read("tiny"), Uint8Array.of(4, 5, 6));

  runtime.dispatch([{ ...command, uniformBytes: new Uint8Array(256).fill(7) }]);
  runtime.dispatch([{ ...command, uniformBytes: Uint8Array.of(1, 2, 3, 4) }]);
  const uniform = device.buffers.get("tritium:uniform-arena").bytes;
  assert.deepEqual(uniform.slice(0, 6), Uint8Array.of(1, 2, 3, 4, 0, 0));
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

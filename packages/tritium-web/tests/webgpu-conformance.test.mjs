import assert from "node:assert/strict";
import test from "node:test";

import {
  runWebGpuVectorConformanceV1,
  webGpuVectorConformanceInventoryV1,
} from "../dist/qualification.js";
import { FakeDevice } from "./support/fake-webgpu.mjs";

test("qualification inventory binds every canonical vector", () => {
  const inventory = webGpuVectorConformanceInventoryV1();
  assert.deepEqual(inventory, {
    schemaId: "tritium.webgpu_vector_conformance_inventory",
    schemaVersion: 1,
    manifestDigest:
      "9093a1a7f9a3422c399943782aadf4df6b11833cf2253db0db56ff2d9dedb098",
    vectorDigest:
      "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7",
    caseCounts: {
      valid: 72,
      invalid: 45,
      compute: 68,
      lifecycle: 4,
      total: 117,
    },
  });
  assert.ok(Object.isFrozen(inventory));
  assert.ok(Object.isFrozen(inventory.caseCounts));
});

test("qualification runner fails closed before claiming a non-device", async () => {
  await assert.rejects(
    runWebGpuVectorConformanceV1({}),
    (error) => {
      assert.match(String(error), /WebGPU|device|limits/);
      return true;
    },
  );
});

test("qualification runner destroys an admitted device when options fail", async () => {
  let destroyed = 0;
  const device = {
    limits: {
      maxBufferSize: 1 << 30,
      maxStorageBufferBindingSize: 1 << 29,
      maxComputeWorkgroupsPerDimension: 65535,
      maxBindingsPerBindGroup: 16,
      maxStorageBuffersPerShaderStage: 10,
      maxUniformBuffersPerShaderStage: 12,
      maxUniformBufferBindingSize: 65536,
      minUniformBufferOffsetAlignment: 256,
    },
    queue: {
      writeBuffer() {},
      submit() {},
      async onSubmittedWorkDone() {},
    },
    lost: { then() {} },
    createShaderModule() {},
    async createComputePipelineAsync() {},
    createBuffer() {},
    createBindGroup() {},
    createCommandEncoder() {},
    destroy() { destroyed += 1; },
  };
  await assert.rejects(
    runWebGpuVectorConformanceV1(device, { maxPeakBytes: 0 }),
    /maxPeakBytes/,
  );
  assert.equal(destroyed, 1);
});

test("qualification runner cannot pass a no-op structural WebGPU adapter", async () => {
  const device = new FakeDevice();
  await assert.rejects(
    runWebGpuVectorConformanceV1(device),
    /graph\.add\.forward\.basic output differs/,
  );
  assert.ok(device.pipelines > 0);
  assert.ok(device.events.includes("dispatch"));
  assert.ok(device.maps > 0, "grading must cross an explicit readback boundary");
  assert.equal(device.destroyed, true);
});

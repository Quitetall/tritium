import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import { prepareTraining } from "../dist/index.js";

import {
  PhysicalBrowserQualificationError,
  physicalBrowserTrainingScenarioV1,
  runPhysicalBrowserTrainingLaneV1,
} from "../dist/qualification.js";
import { FakeDevice } from "./support/fake-webgpu.mjs";

function rejectsCode(promise, code) {
  return assert.rejects(promise, (error) => {
    assert.ok(error instanceof PhysicalBrowserQualificationError);
    assert.equal(error.code, code);
    return true;
  });
}

test("physical browser scenario identity and inputs are immutable", () => {
  const first = physicalBrowserTrainingScenarioV1();
  const second = physicalBrowserTrainingScenarioV1();
  assert.equal(first.schemaId, "tritium.physical_browser_training_scenario");
  assert.equal(first.schemaVersion, 1);
  assert.equal(first.scenarioId, "salt-ste-sgd-256-v1");
  assert.equal(first.completedSteps, 1);
  assert.ok(Object.isFrozen(first));
  assert.ok(Object.isFrozen(first.model));
  assert.ok(Object.isFrozen(first.model.recipe));
  assert.notEqual(first.model.payload, second.model.payload);
  first.model.payload[0] ^= 1;
  assert.notDeepEqual(first.model.payload, second.model.payload);
  assert.deepEqual(first.batch.inputs.target, second.batch.inputs.target);
});

test("physical browser lane rejects malformed options before GPU acquisition", async () => {
  let requests = 0;
  const priorNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { gpu: { requestAdapter() { requests += 1; } } },
  });
  try {
    await rejectsCode(
      runPhysicalBrowserTrainingLaneV1({
        nativeArtifact: new Uint8Array(),
        nativeReferenceDigest: "0".repeat(64),
      }),
      "invalid_options",
    );
    assert.equal(requests, 0);
  } finally {
    if (priorNavigator === undefined) delete globalThis.navigator;
    else Object.defineProperty(globalThis, "navigator", priorNavigator);
  }
});

test("physical browser lane fails closed without WebGPU", async () => {
  const priorNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {},
  });
  try {
    await rejectsCode(
      runPhysicalBrowserTrainingLaneV1({
        nativeArtifact: Uint8Array.of(1),
        nativeReferenceDigest: "0".repeat(64),
      }),
      "adapter_unavailable",
    );
  } finally {
    if (priorNavigator === undefined) delete globalThis.navigator;
    else Object.defineProperty(globalThis, "navigator", priorNavigator);
  }
});

test("frozen browser scenario emits exact native-reference artifact through WASM", async () => {
  const scenario = physicalBrowserTrainingScenarioV1();
  const session = await prepareTraining(scenario.model, {
    ...scenario.config,
    backend: "wasm",
    allowWasmFallback: false,
  });
  try {
    const result = await session.forward(scenario.batch);
    await session.backward(result);
    await session.step();
    const artifact = await session.export();
    assert.equal(artifact.bytes.byteLength, 224);
    assert.equal(
      createHash("sha256").update(artifact.bytes).digest("hex"),
      "6e889858c06a7eb91133f69a948ab8356a444c677eecd9e800ec689380a6e17e",
    );
  } finally {
    await session.dispose();
  }
});

test("no-op structural WebGPU cannot produce a physical lane", async () => {
  const devices = [];
  const priorNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      gpu: {
        async requestAdapter() {
          return {
            info: {
              vendor: "Test Vendor",
              architecture: "Test Architecture",
              device: "Test Device",
              description: "structural test adapter",
              isFallbackAdapter: false,
            },
            async requestDevice() {
              const device = new FakeDevice();
              devices.push(device);
              return device;
            },
          };
        },
      },
    },
  });
  try {
    await rejectsCode(
      runPhysicalBrowserTrainingLaneV1({
        nativeArtifact: Uint8Array.of(1),
        nativeReferenceDigest: "0".repeat(64),
      }),
      "vector_conformance",
    );
    assert.equal(devices.length, 1);
    assert.equal(devices[0].destroyed, true);
  } finally {
    if (priorNavigator === undefined) delete globalThis.navigator;
    else Object.defineProperty(globalThis, "navigator", priorNavigator);
  }
});

test("Firefox non-fallback adapter uses WebGL hardware identity before vector qualification", async () => {
  const devices = [];
  const priorNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const priorDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
  const webgl = {
    getExtension(name) {
      assert.equal(name, "WEBGL_debug_renderer_info");
      return { UNMASKED_VENDOR_WEBGL: 0x9245, UNMASKED_RENDERER_WEBGL: 0x9246 };
    },
    getParameter(name) {
      return name === 0x9245 ? "NVIDIA Corporation" : "NVIDIA GeForce RTX 4090";
    },
  };
  const document = {
    createElement(name) {
      assert.equal(name, "canvas");
      return { getContext: (kind) => kind === "webgl2" ? webgl : null };
    },
  };
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      gpu: {
        async requestAdapter() {
          return {
            info: {
              vendor: "",
              architecture: "",
              device: "",
              description: "",
              isFallbackAdapter: false,
            },
            async requestDevice() {
              const device = new FakeDevice();
              devices.push(device);
              return device;
            },
          };
        },
      },
    },
  });
  Object.defineProperty(globalThis, "document", { configurable: true, value: document });
  try {
    await rejectsCode(
      runPhysicalBrowserTrainingLaneV1({
        nativeArtifact: Uint8Array.of(1),
        nativeReferenceDigest: "0".repeat(64),
      }),
      "vector_conformance",
    );
    assert.equal(devices.length, 1);
    assert.equal(devices[0].destroyed, true);
  } finally {
    if (priorNavigator === undefined) delete globalThis.navigator;
    else Object.defineProperty(globalThis, "navigator", priorNavigator);
    if (priorDocument === undefined) delete globalThis.document;
    else Object.defineProperty(globalThis, "document", priorDocument);
  }
});

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
    operations: ["graph.add", "loss.mse", "optimizer.sgd"],
  },
  payload: new Uint8Array([1, 2, 3]),
};

const config = {
  backend: "webgpu",
  allowWasmFallback: false,
  maxResidentBytes: 2048,
  seed: 7,
  requiredOperations: ["lifecycle.checkpoint", "lifecycle.export"],
};

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
  }

  async prepare(preparedModel) {
    this.calls.push("prepare");
    preparedModel.payload[0] = 99;
    return receipt("session.prepare", this.implementation);
  }

  async forward() {
    this.calls.push("forward");
    if (this.forwardBarrier !== null) await this.forwardBarrier;
    return {
      loss: 0.25,
      receipt: receipt("session.forward", this.implementation),
    };
  }

  async backward() {
    this.calls.push("backward");
    return receipt("session.backward", this.implementation);
  }

  async step() {
    this.calls.push("step");
    return receipt("session.step", this.implementation);
  }

  async checkpoint() {
    this.calls.push("checkpoint");
    return {
      bytes: new Uint8Array([4, 5]),
      receipt: receipt("session.checkpoint", this.implementation),
    };
  }

  async resume(bytes) {
    this.calls.push(`resume:${bytes[0]}`);
    bytes[0] = 88;
    return receipt("session.resume", this.implementation);
  }

  async export() {
    this.calls.push("export");
    return {
      bytes: new Uint8Array([6, 7]),
      receipt: receipt("session.export", this.implementation),
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
  assert.equal(session.state, "prepared");

  await rejectsCode(session.step(), "invalid_state");
  const result = await session.forward({ inputs: { x: new Float32Array([1]) } });
  assert.equal(session.state, "forward-complete");
  await rejectsCode(session.backward({ ...result }), "invalid_state");
  await session.backward(result);
  assert.equal(session.state, "backward-complete");
  await rejectsCode(session.checkpoint(), "invalid_state");
  await session.step();
  assert.equal(session.state, "prepared");

  const checkpoint = await session.checkpoint();
  checkpoint.bytes[0] = 42;
  await session.resume(checkpoint.bytes);
  assert.equal(checkpoint.bytes[0], 42, "resume receives an isolated byte copy");
  const artifact = await session.export();
  assert.deepEqual([...artifact.bytes], [6, 7]);
  await session.dispose();
  await session.dispose();
  assert.equal(session.state, "disposed");
  await rejectsCode(
    session.forward({ inputs: { x: new Float32Array([1]) } }),
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
    recipe: { ...model.recipe, operations: ["graph.not_real"] },
  };
  await rejectsCode(
    prepareTraining(badModel, config, adapter),
    "invalid_schema",
  );
  assert.deepEqual(adapter.calls, []);
  await rejectsCode(prepareTraining(model, config), "adapter_unavailable");
});

test("invalid receipts and adapter failures leave state uncommitted", async () => {
  const adapter = new MockAdapter();
  const session = await prepareTraining(model, config, adapter);
  adapter.forward = async () => {
    throw new Error("device lost");
  };
  await assert.rejects(
    session.forward({ inputs: { x: new Float32Array([1]) } }),
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
    session.forward({ inputs: { x: new Float32Array([1]) } }),
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
  const forward = session.forward({ inputs: { x: new Float32Array([1]) } });
  await new Promise((resolve) => setImmediate(resolve));
  await rejectsCode(session.checkpoint(), "busy");
  release();
  await forward;
  assert.equal(session.state, "forward-complete");
});

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  PortableLifecyclePlanError,
  compilePortableCheckpointRequest,
  compilePortableExportRequest,
  compilePortableReloadRequest,
  compilePortableResumeRequest,
  executePortableWasmRequest,
} from "../dist/index.js";

const state = {
  optimizer: "adamw",
  step: 7,
  leaves: [
    {
      parameter: [1065353216, 3221225472],
      moment1: [1036831949, 3192704205],
      moment2: [1008981770, 1025758986],
    },
  ],
};

test("typed lifecycle compiler checkpoint/resume round-trips exact state", async () => {
  const guest = await readFile(
    new URL("../dist/tritium_wasm_bg.wasm", import.meta.url),
  );
  const checkpointRequest = compilePortableCheckpointRequest(state, "node:lifecycle");
  assert.equal(checkpointRequest.outputs[0].data.values.length, 49);
  const checkpoint = await executePortableWasmRequest(checkpointRequest, guest);
  assert.equal(checkpoint.status, "ok");
  assert.equal(checkpoint.receipt.operation, "lifecycle.checkpoint");

  const bytes = Uint8Array.from(checkpoint.outputs[0].data.values);
  const resumeRequest = compilePortableResumeRequest(
    "adamw",
    [2],
    bytes,
    "node:lifecycle",
  );
  const resumed = await executePortableWasmRequest(resumeRequest, guest);
  assert.equal(resumed.status, "ok");
  assert.deepEqual(resumed.outputs[0].data.values, [7, 0, 0, 0, 0, 0, 0, 0]);
  assert.deepEqual(resumed.outputs[1].data.bits, state.leaves[0].parameter);
  assert.deepEqual(resumed.outputs[2].data.bits, state.leaves[0].moment1);
  assert.deepEqual(resumed.outputs[3].data.bits, state.leaves[0].moment2);
});

test("every optimizer checkpoint round-trips exact planes", async () => {
  const guest = await readFile(
    new URL("../dist/tritium_wasm_bg.wasm", import.meta.url),
  );
  const states = [
    { optimizer: "sgd", step: 1, leaves: [{ parameter: [0x3f800000] }] },
    {
      optimizer: "cautious_adamw",
      step: 2,
      leaves: [{ parameter: [1], moment1: [2], moment2: [3] }],
    },
    {
      optimizer: "int8_adamw",
      step: 3,
      leaves: [
        {
          parameter: [4],
          moment1Q8: [5],
          moment2Q8: [6],
          moment1Scale: [7],
          moment2Scale: [8],
        },
      ],
    },
    {
      optimizer: "muon",
      step: 4,
      leaves: [{ parameter: [9], momentum: [10] }],
    },
  ];
  for (const optimizerState of states) {
    const checkpointRequest = compilePortableCheckpointRequest(
      optimizerState,
      "node:lifecycle-all",
    );
    const checkpoint = await executePortableWasmRequest(checkpointRequest, guest);
    assert.equal(checkpoint.status, "ok", optimizerState.optimizer);
    const resumeRequest = compilePortableResumeRequest(
      optimizerState.optimizer,
      optimizerState.leaves.map((leaf) => leaf.parameter.length),
      Uint8Array.from(checkpoint.outputs[0].data.values),
      "node:lifecycle-all",
    );
    const resumed = await executePortableWasmRequest(resumeRequest, guest);
    assert.equal(resumed.status, "ok", optimizerState.optimizer);
    const stepBytes = new Uint8Array(8);
    new DataView(stepBytes.buffer).setBigUint64(
      0,
      BigInt(optimizerState.step),
      true,
    );
    assert.deepEqual(resumed.outputs[0].data.values, [...stepBytes]);
    assert.deepEqual(
      resumed.outputs.slice(1),
      checkpointRequest.inputs,
      optimizerState.optimizer,
    );
  }
});

test("typed lifecycle compiler validates SALT export and reload in Rust", async () => {
  const guest = await readFile(
    new URL("../dist/tritium_wasm_bg.wasm", import.meta.url),
  );
  const vectors = JSON.parse(
    await readFile(
      new URL("../../../spec/training/v1/vectors/v1.json", import.meta.url),
      "utf8",
    ),
  );
  const exportCase = vectors.cases.find(
    (candidate) => candidate.case_id === "lifecycle.export.salt_v2_package",
  );
  const packageBytes = Uint8Array.from(exportCase.inputs[0].data.values);
  const exported = await executePortableWasmRequest(
    compilePortableExportRequest(packageBytes, "node:lifecycle"),
    guest,
  );
  assert.equal(exported.status, "ok");
  assert.deepEqual(exported.outputs[0].data.values, [...packageBytes]);

  const reloaded = await executePortableWasmRequest(
    compilePortableReloadRequest(
      Uint8Array.from(exported.outputs[0].data.values),
      "node:lifecycle",
    ),
    guest,
  );
  assert.equal(reloaded.status, "ok");
  assert.deepEqual(reloaded.outputs[0].data.values, [...packageBytes]);
});

test("typed lifecycle compiler fails before guest allocation", () => {
  assert.throws(
    () =>
      compilePortableCheckpointRequest({
        ...state,
        leaves: [{ ...state.leaves[0], unexpected: [] }],
      }),
    (error) => {
      assert.ok(error instanceof PortableLifecyclePlanError);
      assert.equal(error.code, "invalid_schema");
      return true;
    },
  );
  assert.throws(
    () => compilePortableResumeRequest("sgd", [], Uint8Array.of(1)),
    (error) => {
      assert.ok(error instanceof PortableLifecyclePlanError);
      assert.equal(error.code, "invalid_schema");
      return true;
    },
  );
  assert.throws(
    () =>
      compilePortableCheckpointRequest({
        optimizer: "adamw",
        step: 0,
        leaves: [{ parameter: [0], moment1: null, moment2: [0] }],
      }),
    (error) => {
      assert.ok(error instanceof PortableLifecyclePlanError);
      assert.equal(error.code, "invalid_schema");
      return true;
    },
  );
  assert.throws(
    () =>
      compilePortableResumeRequest(
        "sgd",
        [2 * 1024 * 1024 + 1],
        Uint8Array.of(1),
      ),
    (error) => {
      assert.ok(error instanceof PortableLifecyclePlanError);
      assert.equal(error.code, "capacity");
      return true;
    },
  );
  assert.throws(
    () => compilePortableExportRequest(new Uint8Array(8 * 1024 * 1024 + 1)),
    (error) => {
      assert.ok(error instanceof PortableLifecyclePlanError);
      assert.equal(error.code, "capacity");
      return true;
    },
  );
  assert.throws(
    () => compilePortableExportRequest(new Uint8Array(3 * 1024 * 1024).fill(255)),
    (error) => {
      assert.ok(error instanceof PortableLifecyclePlanError);
      assert.equal(error.code, "capacity");
      assert.match(error.message, /request JSON/);
      return true;
    },
  );
  assert.throws(
    () =>
      compilePortableCheckpointRequest({
        optimizer: "int8_adamw",
        step: 0,
        leaves: Array.from({ length: 13 }, () => null),
      }),
    (error) => {
      assert.ok(error instanceof PortableLifecyclePlanError);
      assert.equal(error.code, "capacity");
      return true;
    },
  );
});

test("typed lifecycle compiler derives every optimizer layout", () => {
  const fixtures = [
    [{ optimizer: "sgd", step: 0, leaves: [{ parameter: [0] }] }, 29, ["parameter.0"]],
    [
      {
        optimizer: "cautious_adamw",
        step: 0,
        leaves: [{ parameter: [0], moment1: [0], moment2: [0] }],
      },
      37,
      ["parameter.0", "moment1.0", "moment2.0"],
    ],
    [
      {
        optimizer: "int8_adamw",
        step: 0,
        leaves: [{
          parameter: [0],
          moment1Q8: [0],
          moment2Q8: [0],
          moment1Scale: [0],
          moment2Scale: [0],
        }],
      },
      39,
      [
        "parameter.0",
        "moment1_q8.0",
        "moment2_q8.0",
        "moment1_scale.0",
        "moment2_scale.0",
      ],
    ],
    [
      {
        optimizer: "muon",
        step: 0,
        leaves: [{ parameter: [0], momentum: [0] }],
      },
      33,
      ["parameter.0", "momentum.0"],
    ],
  ];
  for (const [fixture, bytes, names] of fixtures) {
    const request = compilePortableCheckpointRequest(fixture);
    assert.equal(request.outputs[0].data.values.length, bytes);
    assert.deepEqual(request.inputs.map((buffer) => buffer.name), names);
  }

  const mutable = structuredClone(state);
  const request = compilePortableCheckpointRequest(mutable);
  mutable.leaves[0].parameter[0] = 0;
  assert.equal(request.inputs[0].data.bits[0], state.leaves[0].parameter[0]);
});

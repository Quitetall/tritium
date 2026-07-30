import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  PortableWasmLifecycleError,
  PortableWasmLifecycleState,
} from "../dist/index.js";

const initial = {
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

async function guestResponse() {
  const bytes = await readFile(
    new URL("../dist/tritium_wasm_bg.wasm", import.meta.url),
  );
  return new Response(bytes);
}

test("portable lifecycle state owns and atomically resumes optimizer planes", async () => {
  const source = structuredClone(initial);
  const lifecycle = await PortableWasmLifecycleState.create({
    source: await guestResponse(),
    state: source,
    physicalDevice: "node:owned-lifecycle",
  });
  source.leaves[0].parameter[0] = 0;
  assert.equal(lifecycle.state.leaves[0].parameter[0], initial.leaves[0].parameter[0]);

  const checkpoint = await lifecycle.checkpoint();
  assert.equal(checkpoint.receipt.operation, "lifecycle.checkpoint");

  const next = structuredClone(initial);
  next.step = 9;
  next.leaves[0].parameter[0] = 42;
  const committing = lifecycle.commit(next);
  next.leaves[0].parameter[0] = 99;
  await committing;
  assert.equal(lifecycle.state.step, 9);
  assert.equal(lifecycle.state.leaves[0].parameter[0], 42);

  const resumed = await lifecycle.resume(checkpoint.bytes);
  assert.equal(resumed.operation, "lifecycle.resume");
  assert.deepEqual(lifecycle.state, initial);

  const beforeFailure = lifecycle.state;
  const corrupt = Uint8Array.from(checkpoint.bytes);
  corrupt[0] ^= 0xff;
  await assert.rejects(lifecycle.resume(corrupt), (error) => {
    assert.ok(error instanceof PortableWasmLifecycleError);
    assert.equal(error.code, "backend");
    return true;
  });
  assert.deepEqual(lifecycle.state, beforeFailure);
  await assert.rejects(lifecycle.resume([...checkpoint.bytes]), (error) => {
    assert.ok(error instanceof PortableWasmLifecycleError);
    assert.equal(error.code, "invalid_state");
    return true;
  });

  lifecycle.dispose();
  assert.throws(() => lifecycle.state, (error) => {
    assert.ok(error instanceof PortableWasmLifecycleError);
    assert.equal(error.code, "disposed");
    return true;
  });
});

test("portable lifecycle create snapshots before awaiting the guest", async () => {
  const bytes = await readFile(
    new URL("../dist/tritium_wasm_bg.wasm", import.meta.url),
  );
  let release;
  const delayed = new Response(bytes);
  delayed.arrayBuffer = () =>
    new Promise((resolve) => {
      release = () => resolve(Uint8Array.from(bytes).buffer);
    });
  const caller = structuredClone(initial);
  const creating = PortableWasmLifecycleState.create({
    source: delayed,
    state: caller,
    physicalDevice: "node:snapshot",
  });
  await Promise.resolve();
  caller.step = 99;
  caller.leaves[0].parameter[0] = 42;
  release();
  const lifecycle = await creating;
  assert.deepEqual(lifecycle.state, initial);
  lifecycle.dispose();
});

test("portable lifecycle normalizes plan errors", async () => {
  await assert.rejects(
    PortableWasmLifecycleState.create({
      source: await guestResponse(),
      state: { ...initial, unexpected: true },
    }),
    (error) => {
      assert.ok(error instanceof PortableWasmLifecycleError);
      assert.equal(error.code, "invalid_state");
      return true;
    },
  );
  await assert.rejects(
    PortableWasmLifecycleState.create({
      source: await guestResponse(),
      state: initial,
      unexpected: true,
    }),
    (error) => {
      assert.ok(error instanceof PortableWasmLifecycleError);
      assert.equal(error.code, "invalid_state");
      return true;
    },
  );
});

test("portable lifecycle export is admitted by strict reload before release", async () => {
  const lifecycle = await PortableWasmLifecycleState.create({
    source: await guestResponse(),
    state: initial,
    physicalDevice: "node:owned-export",
  });
  const vectors = JSON.parse(
    await readFile(
      new URL("../../../spec/training/v2/vectors/v2.json", import.meta.url),
      "utf8",
    ),
  );
  const exportCase = vectors.cases.find(
    (candidate) => candidate.case_id === "lifecycle.export.salt_v2_package",
  );
  const packageBytes = Uint8Array.from(exportCase.inputs[0].data.values);
  const artifact = await lifecycle.admitExport(packageBytes);
  assert.equal(artifact.receipt.operation, "lifecycle.export");
  assert.deepEqual(artifact.bytes, packageBytes);
  lifecycle.dispose();
});

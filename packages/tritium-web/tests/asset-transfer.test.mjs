import assert from "node:assert/strict";
import test from "node:test";
import { blake3 } from "@noble/hashes/blake3.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import { AssetTransferError, transferAsset } from "../src/asset-transfer.ts";

const description = (bytes) => ({ byteLength: bytes.length, blake3: bytesToHex(blake3(bytes)) });
const limits = { maxAssetBytes: 1024, maxWindowBytes: 4 };

function receiver() {
  const writes = [];
  const events = [];
  let published = null;
  return {
    events,
    writes,
    get published() {
      return published;
    },
    async begin() {
      events.push("begin");
      return {
        async write(offset, bytes) {
          events.push("write");
          writes.push({ offset, bytes: bytes.slice() });
        },
        async complete() {
          events.push("complete");
          published = Buffer.concat(writes.map((value) => value.bytes));
        },
        async abort() {
          events.push("abort");
          writes.length = 0;
        },
      };
    },
  };
}

test("asset transfer streams exact raw bytes from memory and Blob adapters", async () => {
  const bytes = Uint8Array.from({ length: 11 }, (_, index) => index);
  const blob = new Blob([bytes]);
  for (const source of [
    {
      async read(offset, length) {
        return bytes.subarray(offset, offset + length);
      },
    },
    {
      async read(offset, length) {
        return new Uint8Array(await blob.slice(offset, offset + length).arrayBuffer());
      },
    },
  ]) {
    const target = receiver();
    const result = await transferAsset(description(bytes), source, target, limits);
    assert.deepEqual(result, {
      byteLength: 11,
      windows: 3,
      maxSnapshotBytes: 4,
      blake3: description(bytes).blake3,
    });
    assert.deepEqual(target.published, Buffer.from(bytes));
    assert.deepEqual(
      target.writes.map((value) => value.offset),
      [0, 4, 8],
    );
    assert.equal(target.events.at(-1), "complete");
  }
});

test("asset metadata and pre-cancellation fail before allocation or reading", async () => {
  let reads = 0;
  const source = {
    async read() {
      reads += 1;
      return Uint8Array.of(1);
    },
  };
  for (const [value, capacity, code] of [
    [{ byteLength: 0, blake3: "0".repeat(64) }, limits, "invalid_metadata"],
    [{ byteLength: 1025, blake3: "0".repeat(64) }, limits, "capacity"],
    [{ byteLength: 1, blake3: "wrong" }, limits, "invalid_metadata"],
  ]) {
    const target = receiver();
    await assert.rejects(transferAsset(value, source, target, capacity), { code });
    assert.deepEqual(target.events, []);
  }
  const target = receiver();
  await assert.rejects(
    transferAsset(description(Uint8Array.of(1)), source, target, limits, AbortSignal.abort()),
    { code: "cancelled" },
  );
  assert.deepEqual(target.events, []);
  assert.equal(reads, 0);
});

test("asset read-size failures and final digest mismatch never publish", async () => {
  for (const returned of [Uint8Array.of(), new Uint8Array(5), Uint8Array.of(9)]) {
    const target = receiver();
    const expected = Uint8Array.of(1);
    const code = returned.length === 1 ? "integrity" : "read_size";
    await assert.rejects(
      transferAsset(
        description(expected),
        {
          async read() {
            return returned;
          },
        },
        target,
        limits,
      ),
      { code },
    );
    assert.equal(target.published, null);
    assert.equal(target.events.at(-1), "abort");
    assert.deepEqual(target.writes, []);
  }
});

test("asset cancellation after write aborts unpublished state", async () => {
  const control = new AbortController();
  const target = receiver();
  const begin = target.begin.bind(target);
  target.begin = async (value) => {
    const stage = await begin(value);
    return {
      ...stage,
      async write(offset, bytes) {
        await stage.write(offset, bytes);
        control.abort();
      },
    };
  };
  await assert.rejects(
    transferAsset(
      description(Uint8Array.of(1)),
      {
        async read() {
          return Uint8Array.of(1);
        },
      },
      target,
      limits,
      control.signal,
    ),
    { code: "cancelled" },
  );
  assert.equal(target.published, null);
  assert.deepEqual(target.events, ["begin", "write", "abort"]);
});

test("asset read errors preserve their cause and abort failures stay visible", async () => {
  const sourceError = new Error("source stopped");
  const target = receiver();
  await assert.rejects(
    transferAsset(
      description(Uint8Array.of(1)),
      {
        async read() {
          throw sourceError;
        },
      },
      target,
      limits,
    ),
    (error) => error === sourceError,
  );
  assert.equal(target.events.at(-1), "abort");
  const abortError = new Error("abort failed");
  await assert.rejects(
    transferAsset(
      description(Uint8Array.of(1)),
      {
        async read() {
          throw sourceError;
        },
      },
      {
        async begin() {
          return {
            async write() {},
            async complete() {},
            async abort() {
              throw abortError;
            },
          };
        },
      },
      limits,
    ),
    (error) =>
      error instanceof AggregateError &&
      error.errors[0] === sourceError &&
      error.errors[1] === abortError,
  );
});

test("asset snapshots ignore caller iterators and forged byte lengths", async () => {
  const value = Uint8Array.of(1);
  value[Symbol.iterator] = () => {
    throw new Error("iterator must not run");
  };
  Object.defineProperty(value, "byteLength", { value: 9000 });
  const target = receiver();
  await transferAsset(
    description(Uint8Array.of(1)),
    {
      async read() {
        return value;
      },
    },
    target,
    limits,
  );
  assert.deepEqual(target.published, Buffer.from([1]));
  assert.ok(AssetTransferError.prototype instanceof Error);
});

test("asset transfer captures metadata once and isolates delivered caller bytes", async () => {
  const bytes = Uint8Array.of(1, 2);
  const value = description(bytes);
  const target = receiver();
  const begin = target.begin.bind(target);
  target.begin = async (metadata) => {
    const stage = await begin(metadata);
    return {
      ...stage,
      async write(offset, snapshot) {
        bytes[0] = 99;
        await stage.write(offset, snapshot);
      },
    };
  };
  const result = await transferAsset(
    value,
    {
      async read() {
        value.byteLength = 100;
        value.blake3 = "0".repeat(64);
        return bytes;
      },
    },
    target,
    limits,
  );
  assert.equal(result.byteLength, 2);
  assert.deepEqual(target.published, Buffer.from([1, 2]));
});

test("asset stage and seal failure abort private state", async () => {
  for (const method of ["write", "complete"]) {
    const expected = new Error(`${method} refused`);
    const target = receiver();
    const begin = target.begin.bind(target);
    target.begin = async (value) => ({
      ...(await begin(value)),
      async [method]() {
        throw expected;
      },
    });
    await assert.rejects(
      transferAsset(
        description(Uint8Array.of(1)),
        {
          async read() {
            return Uint8Array.of(1);
          },
        },
        target,
        limits,
      ),
      (error) => error === expected,
    );
    assert.equal(target.published, null);
    assert.equal(target.events.at(-1), "abort");
  }
});

test("asset cancellation after the seal commit point does not revoke success", async () => {
  const control = new AbortController();
  const target = receiver();
  const begin = target.begin.bind(target);
  target.begin = async (value) => {
    const stage = await begin(value);
    return {
      ...stage,
      async complete() {
        await stage.complete();
        control.abort();
      },
    };
  };
  const result = await transferAsset(
    description(Uint8Array.of(1)),
    {
      async read() {
        return Uint8Array.of(1);
      },
    },
    target,
    limits,
    control.signal,
  );
  assert.equal(result.byteLength, 1);
  assert.deepEqual(target.published, Buffer.from([1]));
  assert.equal(target.events.at(-1), "complete");
});

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, mkdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import {
  admitArchiveNativeReference,
  readBoundedOrdinaryFile,
} from "../scripts/archive-native-reference.mjs";

async function fixture(context) {
  const directory = await mkdtemp(join(tmpdir(), "tritium-archive-native-"));
  context.after(() => rm(directory, { force: true, recursive: true }));
  return directory;
}

test("native parity input preserves exact bytes at the declared ceiling", async (context) => {
  const directory = await fixture(context);
  const path = join(directory, "bytes");
  await writeFile(path, new Uint8Array([0, 1, 128, 255]));
  assert.deepEqual(await readBoundedOrdinaryFile(path, 4), Buffer.from([0, 1, 128, 255]));
});

test("native parity input rejects empty and oversized files", async (context) => {
  const directory = await fixture(context);
  const path = join(directory, "bytes");
  for (const bytes of [new Uint8Array(), new Uint8Array(5)]) {
    await writeFile(path, bytes);
    await assert.rejects(readBoundedOrdinaryFile(path, 4), /bounded nonempty ordinary file/);
  }
});

test("native parity input rejects symlinks and directories", async (context) => {
  const directory = await fixture(context);
  await writeFile(join(directory, "real"), "bytes");
  await symlink(join(directory, "real"), join(directory, "link"));
  await assert.rejects(readBoundedOrdinaryFile(join(directory, "link"), 8), { code: "ELOOP" });
  await mkdir(join(directory, "folder"));
  await assert.rejects(
    readBoundedOrdinaryFile(join(directory, "folder"), 8),
    /bounded nonempty ordinary file/,
  );
});

test("native parity input rejects a FIFO without waiting for a writer", async (context) => {
  const directory = await fixture(context);
  const path = join(directory, "pipe");
  await promisify(execFile)("mkfifo", [path], { timeout: 5000 });
  await assert.rejects(readBoundedOrdinaryFile(path, 8), /bounded nonempty ordinary file/);
});

test("native parity input fails closed for malformed receipts and directories", async (context) => {
  const directory = await fixture(context);
  await assert.rejects(admitArchiveNativeReference("", "a".repeat(40)), /directory is invalid/);
  await writeFile(join(directory, "native.salt"), "TSLT2PKG bounded artifact");
  await writeFile(join(directory, "receipt.json"), "not json");
  await assert.rejects(admitArchiveNativeReference(directory, "a".repeat(40)), SyntaxError);
  await writeFile(join(directory, "receipt.json"), "{}");
  await assert.rejects(
    admitArchiveNativeReference(directory, "a".repeat(40)),
    /fields do not match the frozen schema/,
  );
});

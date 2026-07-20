import { execFile } from "node:child_process";
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

import { blake3 } from "@noble/hashes/blake3.js";
import { bytesToHex } from "@noble/hashes/utils.js";

const run = promisify(execFile);
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repository = resolve(root, "../..");
const generated = resolve(root, ".generated");
const guest = resolve(
  repository,
  "target/wasm32-unknown-unknown/release/tritium_wasm.wasm",
);
const wasmBindgenVersion = "wasm-bindgen 0.2.126";
const maximumLinearMemoryPages = (192 * 1024 * 1024) / 65536;

function readU32Leb(bytes, cursor) {
  let result = 0;
  let shift = 0;
  for (let count = 0; count < 5; count += 1) {
    const byte = bytes[cursor.offset];
    if (byte === undefined) throw new Error("truncated WASM LEB128 integer");
    cursor.offset += 1;
    result |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) return result >>> 0;
    shift += 7;
  }
  throw new Error("oversized WASM u32 LEB128 integer");
}

function assertLinearMemoryMaximum(bytes) {
  if (
    bytes.length < 8 ||
    !bytes.subarray(0, 8).equals(Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]))
  ) {
    throw new Error("generated guest is not a WebAssembly v1 module");
  }
  const cursor = { offset: 8 };
  while (cursor.offset < bytes.length) {
    const sectionId = bytes[cursor.offset];
    cursor.offset += 1;
    const sectionLength = readU32Leb(bytes, cursor);
    const sectionEnd = cursor.offset + sectionLength;
    if (sectionEnd > bytes.length) throw new Error("truncated WASM section");
    if (sectionId === 5) {
      const memoryCount = readU32Leb(bytes, cursor);
      if (memoryCount !== 1) throw new Error("guest must define exactly one memory");
      const flags = readU32Leb(bytes, cursor);
      readU32Leb(bytes, cursor);
      if ((flags & 1) === 0) throw new Error("guest memory has no declared maximum");
      const maximum = readU32Leb(bytes, cursor);
      if (maximum !== maximumLinearMemoryPages) {
        throw new Error(
          `guest memory maximum is ${maximum} pages, expected ${maximumLinearMemoryPages}`,
        );
      }
      return;
    }
    cursor.offset = sectionEnd;
  }
  throw new Error("guest has no defined linear memory section");
}

export async function buildPortableWasm(output) {
  await mkdir(generated, { recursive: true });
  await run(
    "cargo",
    [
      "build",
      "--locked",
      "--release",
      "--package",
      "tritium-wasm",
      "--target",
      "wasm32-unknown-unknown",
    ],
    { cwd: repository },
  );
  const { stdout: actualWasmBindgenVersion } = await run("wasm-bindgen", [
    "--version",
  ]);
  if (actualWasmBindgenVersion.trim() !== wasmBindgenVersion) {
    throw new Error(
      `expected ${wasmBindgenVersion}, got ${actualWasmBindgenVersion.trim()}`,
    );
  }
  await run(
    "wasm-bindgen",
    [
      "--target",
      "web",
      "--out-dir",
      generated,
      "--out-name",
      "tritium_wasm",
      guest,
    ],
    { cwd: repository },
  );
  const guestBytes = await readFile(resolve(generated, "tritium_wasm_bg.wasm"));
  assertLinearMemoryMaximum(guestBytes);
  const guestDigest = bytesToHex(blake3(guestBytes));
  await writeFile(
    resolve(generated, "wasm_identity.ts"),
    `export const WASM_GUEST_DIGEST_V1 = "${guestDigest}" as const;\n`,
  );
  if (output !== undefined) {
    await cp(
      resolve(generated, "tritium_wasm_bg.wasm"),
      resolve(output, "tritium_wasm_bg.wasm"),
    );
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await buildPortableWasm(undefined);
}

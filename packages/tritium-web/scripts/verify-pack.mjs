import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  readFile,
  realpath,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";

const run = promisify(execFile);
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const packageJson = JSON.parse(await readFile(resolve(root, "package.json"), "utf8"));
const MAX_COMMAND_OUTPUT_BYTES = 8 * 1024 * 1024;
const expectedFiles = Object.freeze([
  "README.md",
  "dist/LICENSE",
  "dist/NOTICE",
  "dist/index.d.ts",
  "dist/index.js",
  "dist/index.js.map",
  "dist/lifecycle-types.d.ts",
  "dist/payload-types.d.ts",
  "dist/portable-schedule-types.d.ts",
  "dist/portable-state-types.d.ts",
  "dist/portable.d.ts",
  "dist/tritium_wasm_bg.wasm",
  "package.json",
]);

function fail(message) {
  throw new Error(`npm archive verification failed: ${message}`);
}

function packedMetadata(stdout) {
  let parsed;
  try {
    parsed = JSON.parse(stdout);
  } catch {
    fail("npm pack did not return JSON");
  }
  const candidates = Array.isArray(parsed)
    ? parsed
    : parsed !== null && typeof parsed === "object"
      ? Object.values(parsed)
      : [];
  if (candidates.length !== 1 || candidates[0] === null ||
      typeof candidates[0] !== "object") {
    fail("npm pack returned an unexpected candidate set");
  }
  return candidates[0];
}

function digest(algorithm, bytes, encoding = "hex") {
  return createHash(algorithm).update(bytes).digest(encoding);
}

function assertRelativeFile(path) {
  if (typeof path !== "string" || path.length === 0 || path.includes("\\") ||
      path.includes("\0") || path.startsWith("/") || path.split("/").some((part) =>
        part.length === 0 || part === "." || part === ".."
      )) {
    fail(`unsafe archive member ${String(path)}`);
  }
}

async function main() {
  if (packageJson.private !== true) fail("publication guard is not enabled");
  const temporary = await mkdtemp(join(tmpdir(), "tritium-npm-pack-"));
  const archiveDirectory = resolve(temporary, "archive");
  await mkdir(archiveDirectory);
  const { stdout } = await run("npm", [
    "pack", "--json", "--pack-destination", archiveDirectory,
  ], { cwd: root, maxBuffer: MAX_COMMAND_OUTPUT_BYTES });
  const metadata = packedMetadata(stdout);
  if (metadata.name !== packageJson.name || metadata.version !== packageJson.version ||
      typeof metadata.filename !== "string" || !Array.isArray(metadata.files)) {
    fail("npm metadata differs from package identity");
  }
  const files = metadata.files.map((entry) => {
    if (entry === null || typeof entry !== "object" ||
        !Number.isSafeInteger(entry.size) || entry.size < 0 || entry.mode !== 0o644) {
      fail("npm returned a malformed archive member");
    }
    assertRelativeFile(entry.path);
    return entry.path;
  }).sort();
  const expected = [...expectedFiles].sort();
  if (files.length !== expected.length ||
      files.some((path, index) => path !== expected[index])) {
    fail(`archive file set drifted: ${JSON.stringify(files)}`);
  }

  const archive = resolve(archiveDirectory, metadata.filename);
  const archiveRoot = `${await realpath(archiveDirectory)}${sep}`;
  const archiveRealPath = await realpath(archive);
  if (!archiveRealPath.startsWith(archiveRoot)) fail("archive escaped its output directory");
  const archiveBytes = await readFile(archiveRealPath);
  const sha512 = digest("sha512", archiveBytes, "base64");
  if (metadata.integrity !== `sha512-${sha512}` ||
      metadata.shasum !== digest("sha1", archiveBytes)) {
    fail("npm archive digest metadata is invalid");
  }

  const sourceMap = JSON.parse(await readFile(resolve(root, "dist/index.js.map"), "utf8"));
  if (!Array.isArray(sourceMap.sources) ||
      sourceMap.sources.some((source) => typeof source !== "string" ||
        source.startsWith("/") || /^[A-Za-z]:[\\/]/.test(source)) ||
      (Array.isArray(sourceMap.sourcesContent) &&
        sourceMap.sourcesContent.some((content) => content !== null))) {
    fail("source map leaks source contents or absolute paths");
  }

  const consumer = resolve(temporary, "consumer");
  await mkdir(consumer);
  await writeFile(resolve(consumer, "package.json"), JSON.stringify({
    name: "tritium-archive-smoke",
    private: true,
    type: "module",
  }));
  await run("npm", [
    "install", "--offline", "--ignore-scripts", "--no-audit", "--no-fund",
    "--package-lock=false", archiveRealPath,
  ], { cwd: consumer, maxBuffer: MAX_COMMAND_OUTPUT_BYTES });
  const installedRoot = await realpath(resolve(consumer, "node_modules"));
  const installedRootUrl = pathToFileURL(`${installedRoot}${sep}`).href;
  await writeFile(resolve(consumer, "smoke.mjs"), `
import assert from "node:assert/strict";
import { runPortableWasmConformance } from "@tritium-ai/web";
const resolved = import.meta.resolve("@tritium-ai/web");
assert.ok(resolved.startsWith(${JSON.stringify(installedRootUrl)}));
const receipt = await runPortableWasmConformance();
assert.equal(receipt.operationCount, 35);
assert.equal(receipt.caseCount, 114);
process.stdout.write(JSON.stringify(receipt));
`);
  const smoke = await run(process.execPath, [resolve(consumer, "smoke.mjs")], {
    cwd: consumer,
    maxBuffer: MAX_COMMAND_OUTPUT_BYTES,
  });
  const wasmReceipt = JSON.parse(smoke.stdout);

  await writeFile(resolve(consumer, "index.ts"), `
import {
  TRAINING_MANIFEST_DIGEST_V1,
  createWebGpuTrainingAdapter,
  type WebGpuDevicePortV1,
  type WebTrainingAdapterV1,
} from "@tritium-ai/web";
const digest: string = TRAINING_MANIFEST_DIGEST_V1;
declare const device: WebGpuDevicePortV1;
const adapter: WebTrainingAdapterV1 = createWebGpuTrainingAdapter(device);
void digest;
void adapter;
`);
  await writeFile(resolve(consumer, "tsconfig.json"), JSON.stringify({
    compilerOptions: {
      strict: true,
      noEmit: true,
      target: "ES2022",
      module: "NodeNext",
      moduleResolution: "NodeNext",
      skipLibCheck: false,
    },
    files: ["index.ts"],
  }));
  await run(process.execPath, [
    resolve(root, "node_modules/typescript/bin/tsc"), "-p", resolve(consumer, "tsconfig.json"),
  ], { cwd: consumer, maxBuffer: MAX_COMMAND_OUTPUT_BYTES });

  const receipt = Object.freeze({
    schemaId: "tritium.npm_archive_receipt",
    schemaVersion: 1,
    package: `${packageJson.name}@${packageJson.version}`,
    archiveSha256: digest("sha256", archiveBytes),
    archiveBytes: archiveBytes.byteLength,
    entryCount: files.length,
    sourceFree: true,
    installedOffline: true,
    strictTypeScript: true,
    wasmBuildId: wasmReceipt.buildId,
    wasmGuestDigest: wasmReceipt.guestDigest,
  });
  process.stdout.write(`${JSON.stringify(receipt, null, 2)}\n`);
}

await main();

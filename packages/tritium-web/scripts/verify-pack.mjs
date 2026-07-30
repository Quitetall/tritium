import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  realpath,
  rm,
  writeFile,
} from "node:fs/promises";
import { arch, hostname, platform, tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { generateNpmSbom } from "./generate-npm-sbom.mjs";

const run = promisify(execFile);
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const packageJson = JSON.parse(await readFile(resolve(root, "package.json"), "utf8"));
const packageLock = JSON.parse(await readFile(resolve(root, "package-lock.json"), "utf8"));
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

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${canonicalJson(value[key])}`
    ).join(",")}}`;
  }
  return JSON.stringify(value);
}

function assertRelativeFile(path) {
  if (typeof path !== "string" || path.length === 0 || path.includes("\\") ||
      path.includes("\0") || path.startsWith("/") || path.split("/").some((part) =>
        part.length === 0 || part === "." || part === ".."
      )) {
    fail(`unsafe archive member ${String(path)}`);
  }
}

async function verifyArchive(temporary, startedAtUtc, started) {
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
  if (metadata.filename !== basename(metadata.filename) ||
      metadata.filename.includes("\\") || metadata.filename.includes("\0")) {
    fail("npm returned an unsafe archive filename");
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
  const archiveRoot = await realpath(archiveDirectory);
  const archiveRealPath = await realpath(archive);
  const archiveRelativePath = relative(archiveRoot, archiveRealPath);
  if (archiveRelativePath.length === 0 || archiveRelativePath === ".." ||
      archiveRelativePath.startsWith(`..${sep}`) || isAbsolute(archiveRelativePath)) {
    fail("archive escaped its output directory");
  }
  const archiveBytes = await readFile(archiveRealPath);
  const sha512 = digest("sha512", archiveBytes, "base64");
  if (metadata.integrity !== `sha512-${sha512}` ||
      metadata.shasum !== digest("sha1", archiveBytes)) {
    fail("npm archive digest metadata is invalid");
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
  const installedPackage = resolve(installedRoot, "@tritium-ai/web");
  const sourceMap = JSON.parse(
    await readFile(resolve(installedPackage, "dist/index.js.map"), "utf8"),
  );
  if (!Array.isArray(sourceMap.sources) ||
      sourceMap.sources.some((source) => typeof source !== "string" ||
        source.startsWith("/") || /^[A-Za-z]:[\\/]/.test(source)) ||
      (Array.isArray(sourceMap.sourcesContent) &&
        sourceMap.sourcesContent.some((content) => content !== null))) {
    fail("installed source map leaks source contents or absolute paths");
  }
  await writeFile(resolve(consumer, "smoke.mjs"), `
import assert from "node:assert/strict";
import { runPortableWasmConformance } from "@tritium-ai/web";
const resolved = import.meta.resolve("@tritium-ai/web");
assert.ok(resolved.startsWith(${JSON.stringify(installedRootUrl)}));
const receipt = await runPortableWasmConformance();
assert.equal(receipt.operationCount, 36);
assert.equal(receipt.caseCount, 117);
process.stdout.write(JSON.stringify(receipt));
`);
  const smoke = await run(process.execPath, [resolve(consumer, "smoke.mjs")], {
    cwd: consumer,
    maxBuffer: MAX_COMMAND_OUTPUT_BYTES,
  });
  const wasmReceipt = JSON.parse(smoke.stdout);
  const sourceIdentity = typeof wasmReceipt.buildId === "string"
    ? /\+source-git:([0-9a-f]{40})(?:\+dirty-blake3:([0-9a-f]{64}))?$/.exec(
      wasmReceipt.buildId,
    )
    : null;
  if (sourceIdentity === null ||
      !wasmReceipt.buildId.startsWith(`tritium-wasm@${packageJson.version}+`) ||
      typeof wasmReceipt.guestDigest !== "string" ||
      !/^[0-9a-f]{64}$/.test(wasmReceipt.guestDigest)) {
    fail("installed WASM returned an invalid identity receipt");
  }

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

  const npmVersion = (await run("npm", ["--version"], {
    maxBuffer: MAX_COMMAND_OUTPUT_BYTES,
  })).stdout.trim();
  const machineMaterial = {
    node: hostname(),
    system: platform(),
    architecture: arch(),
  };
  const runId = process.env.TRITIUM_NPM_RUN_ID ?? (
    process.env.GITHUB_RUN_ID !== undefined && process.env.GITHUB_RUN_ATTEMPT !== undefined
      ? `github-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}-npm-archive`
      : `local-${process.pid}`
  );
  const unsignedReceipt = {
    schema: "tritium.npm-archive-qualification.v1",
    release: packageJson.version,
    source_revision: sourceIdentity[1],
    run_id: runId,
    started_at_utc: startedAtUtc,
    duration_ms: Number(process.hrtime.bigint() - started) / 1e6,
    machine: {
      machine_id: `sha256:${digest("sha256", Buffer.from(canonicalJson(machineMaterial)))}`,
      system: machineMaterial.system,
      architecture: machineMaterial.architecture,
    },
    toolchain: { node: process.version, npm: npmVersion },
    artifact: {
      kind: "npm-archive",
      name: metadata.filename,
      package: `${packageJson.name}@${packageJson.version}`,
      bytes: archiveBytes.byteLength,
      sha256: digest("sha256", archiveBytes),
      integrity: metadata.integrity,
    },
    evidence: {
      source_dirty: sourceIdentity[2] !== undefined,
      entry_count: files.length,
      source_free: true,
      installed_offline: true,
      strict_typescript: true,
      wasm_build_id: wasmReceipt.buildId,
      wasm_guest_digest: wasmReceipt.guestDigest,
    },
    result: "pass",
  };
  const receipt = Object.freeze({
    ...unsignedReceipt,
    receipt_id: `sha256:${digest("sha256", Buffer.from(canonicalJson(unsignedReceipt)))}`,
  });
  const receiptJson = `${JSON.stringify(receipt, null, 2)}\n`;
  const evidenceDirectory = process.env.TRITIUM_NPM_EVIDENCE_DIR;
  if (evidenceDirectory !== undefined) {
    if (evidenceDirectory.length === 0 || evidenceDirectory.includes("\0")) {
      fail("TRITIUM_NPM_EVIDENCE_DIR is invalid");
    }
    const output = resolve(evidenceDirectory);
    await mkdir(output, { recursive: true });
    await copyFile(
      archiveRealPath,
      resolve(output, metadata.filename),
      constants.COPYFILE_EXCL,
    );
    await writeFile(resolve(output, "npm-archive-receipt.json"), receiptJson, {
      flag: "wx",
    });
    const sbom = generateNpmSbom(
      packageJson,
      packageLock,
      receipt,
      metadata.filename,
    );
    await writeFile(
      resolve(output, "tritium-web-node22.cdx.json"),
      `${JSON.stringify(sbom, null, 2)}\n`,
      { flag: "wx" },
    );
  }
  process.stdout.write(receiptJson);
}

async function main() {
  if (packageJson.private !== true) fail("publication guard is not enabled");
  const startedAtUtc = new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
  const started = process.hrtime.bigint();
  const temporary = await mkdtemp(join(tmpdir(), "tritium-npm-pack-"));
  try {
    await verifyArchive(temporary, startedAtUtc, started);
  } finally {
    await rm(temporary, { force: true, recursive: true });
  }
}

await main();

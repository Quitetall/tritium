#!/usr/bin/env node
/** Produce one candidate-bound physical browser lane through W3C WebDriver. */

import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream, constants as fsConstants } from "node:fs";
import {
  access,
  mkdir,
  mkdtemp,
  lstat,
  open,
  readFile,
  realpath,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { createServer } from "node:http";
import { arch as hostArchitecture, platform as hostPlatform, release as hostRelease } from "node:os";
import { basename, dirname, extname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import {
  canonicalBrowserVectorMetadataV1,
} from "../packages/tritium-web/scripts/canonical-browser-vector-metadata.mjs";

const run = promisify(execFile);
const SCRIPT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const MANIFEST_DIGEST = "9093a1a7f9a3422c399943782aadf4df6b11833cf2253db0db56ff2d9dedb098";
const VECTOR_DIGEST = "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7";
const SCENARIO_ID = "salt-ste-sgd-256-v1";
const ENGINES = new Set(["chrome", "firefox", "safari"]);
const SOFTWARE_MARKERS = ["swiftshader", "llvmpipe", "software", "emulator", "lavapipe", "warp"];
const MAX_ARCHIVE_BYTES = 512 * 1024 * 1024;
const MAX_RECEIPT_BYTES = 1024 * 1024;
const MAX_ARTIFACT_BYTES = 8 * 1024 * 1024;
const MAX_WEBDRIVER_RESULT_BYTES = 128 * 1024 * 1024;
const MAX_COMMAND_OUTPUT_BYTES = 8 * 1024 * 1024;
const EXPECTED_NPM_ENTRY_COUNT = 16;
const PACKAGE_RELEASE = JSON.parse(await readFile(
  join(SCRIPT_ROOT, "packages/tritium-web/package.json"),
  "utf8",
)).version;

export class BrowserLaneProducerError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "BrowserLaneProducerError";
    this.code = code;
  }
}

function fail(code, message) {
  throw new BrowserLaneProducerError(code, message);
}

function object(value, label) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail("schema", `${label} must be an object`);
  }
  return value;
}

function exactFields(value, fields, label) {
  const candidate = object(value, label);
  const actual = Object.keys(candidate).sort();
  const expected = [...fields].sort();
  if (actual.length !== expected.length || actual.some((field, index) => field !== expected[index])) {
    fail("schema", `${label} fields do not match the frozen schema`);
  }
  return candidate;
}

function nonEmpty(value, label) {
  if (typeof value !== "string" || value.trim().length === 0) {
    fail("schema", `${label} must be a non-empty string`);
  }
  return value.trim();
}

function hex(value, label, length = 64) {
  if (typeof value !== "string" || value.length !== length || !/^[0-9a-f]+$/.test(value)) {
    fail("schema", `${label} must be ${length} lowercase hexadecimal characters`);
  }
  return value;
}

function positive(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) fail("schema", `${label} must be positive`);
  return value;
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

function canonicalBytes(value) {
  return Buffer.from(canonicalJson(value));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

const { metadata: canonicalVectorMetadata } =
  await canonicalBrowserVectorMetadataV1();
const CANONICAL_VECTOR_CASES = Object.freeze(canonicalVectorMetadata.cases);

function validateCanonicalVectorTraceCases(value) {
  if (!Array.isArray(value) || value.length !== CANONICAL_VECTOR_CASES.length) {
    fail("browser_trace", "browser vector trace does not retain the canonical inventory");
  }
  for (let index = 0; index < CANONICAL_VECTOR_CASES.length; index += 1) {
    const expected = CANONICAL_VECTOR_CASES[index];
    const actual = value[index];
    if (typeof actual !== "object" || actual === null || Array.isArray(actual) ||
        Object.keys(actual).sort().join("\0") !==
          ["caseId", "implementation", "outputDigest", "scratchBytes", "scratchBytesMax"]
            .sort().join("\0") ||
        actual.caseId !== expected.caseId || actual.implementation !== expected.implementation ||
        typeof actual.outputDigest !== "string" || !/^[0-9a-f]{64}$/.test(actual.outputDigest) ||
        actual.scratchBytesMax !== expected.scratchBytesMax) {
      fail("browser_trace", `browser vector case ${index} differs from canonical metadata`);
    }
    if (expected.implementation === "wasm-validation") {
      if (actual.scratchBytes !== null) {
        fail("browser_trace", `browser vector case ${expected.caseId} has invalid scratch evidence`);
      }
    } else if (!Number.isSafeInteger(actual.scratchBytes) || actual.scratchBytes < 0 ||
               actual.scratchBytes > expected.scratchBytesMax) {
      fail("browser_trace", `browser vector case ${expected.caseId} exceeds its scratch bound`);
    }
  }
}

async function fileDescriptor(path, label, maximum) {
  const metadata = await lstat(path).catch(() => null);
  if (metadata === null || !metadata.isFile() || metadata.isSymbolicLink() ||
      metadata.size <= 0 || metadata.size > maximum) {
    fail("file", `${label} must be a bounded ordinary file`);
  }
  return { name: basename(path), bytes: metadata.size, sha256: sha256(await readFile(path)) };
}

function safeRelativeFile(value, label) {
  const path = nonEmpty(value, label);
  if (path.includes("\\") || path.includes("\0") || path.startsWith("/") ||
      path.split("/").some((part) => part === "" || part === "." || part === "..")) {
    fail("schema", `${label} is unsafe`);
  }
  return path;
}

export function validateNativeReferenceV1(receiptValue, artifactBytes, artifactName, revision) {
  const receipt = exactFields(receiptValue, [
    "artifact", "backend", "backend_build", "backend_id", "export", "manifest_digest",
    "physical_device", "receipt_id", "reload", "result", "scenario_id", "schema",
    "source_revision", "vector_digest",
  ], "native reference");
  if (!(artifactBytes instanceof Uint8Array) || artifactBytes.byteLength === 0 ||
      artifactBytes.byteLength > MAX_ARTIFACT_BYTES) {
    fail("native_reference", "native reference artifact is invalid");
  }
  if (!/^[0-9a-f]{40}$/.test(revision)) {
    fail("native_reference", "native reference source revision is invalid");
  }
  const artifact = exactFields(receipt.artifact, ["bytes", "name", "sha256"], "native artifact");
  const artifactDigest = sha256(artifactBytes);
  const admitLifecycle = (value, operation, reload = false) => {
    const fields = [
      "artifact_sha256", "device_resident", "host_transfers", "input_digest", "operation",
      "output_digest", "peak_resident_bytes", "result", "scratch_bytes",
    ];
    if (reload) fields.push("reloaded_sha256");
    const lifecycle = exactFields(value, fields, `native ${operation}`);
    if (lifecycle.result !== "pass" || lifecycle.operation !== operation ||
        lifecycle.artifact_sha256 !== artifactDigest || lifecycle.device_resident !== true ||
        lifecycle.host_transfers !== 0 ||
        !Number.isSafeInteger(lifecycle.peak_resident_bytes) || lifecycle.peak_resident_bytes <= 0 ||
        !Number.isSafeInteger(lifecycle.scratch_bytes) || lifecycle.scratch_bytes < 0 ||
        typeof lifecycle.input_digest !== "string" ||
        !/^[0-9a-f]{64}$/.test(lifecycle.input_digest) ||
        typeof lifecycle.output_digest !== "string" ||
        !/^[0-9a-f]{64}$/.test(lifecycle.output_digest) ||
        (reload && lifecycle.reloaded_sha256 !== artifactDigest)) {
      fail("native_reference", `native ${operation} receipt differs`);
    }
    return Object.freeze({
      result: lifecycle.result,
      operation: lifecycle.operation,
      artifactSha256: lifecycle.artifact_sha256,
      inputDigest: lifecycle.input_digest,
      outputDigest: lifecycle.output_digest,
      peakResidentBytes: lifecycle.peak_resident_bytes,
      scratchBytes: lifecycle.scratch_bytes,
      hostTransfers: lifecycle.host_transfers,
      deviceResident: lifecycle.device_resident,
      ...(reload ? { reloadedSha256: lifecycle.reloaded_sha256 } : {}),
    });
  };
  const exportReceipt = admitLifecycle(receipt.export, "lifecycle.export");
  const reloadReceipt = admitLifecycle(receipt.reload, "lifecycle.reload", true);
  if (receipt.schema !== "tritium.browser-native-reference.v1" ||
      receipt.result !== "pass" || receipt.scenario_id !== SCENARIO_ID ||
      receipt.source_revision !== revision || receipt.backend !== "cpu" ||
      receipt.backend_id !== "cpu.reference.v1" ||
      receipt.backend_build !==
        `tritium-train@${PACKAGE_RELEASE}+source-git:${revision}` ||
      receipt.manifest_digest !== MANIFEST_DIGEST || receipt.vector_digest !== VECTOR_DIGEST ||
      !nonEmpty(receipt.physical_device, "native reference physical_device").startsWith("cpu:") ||
      artifact.name !== artifactName || basename(artifact.name) !== artifact.name ||
      artifact.bytes !== artifactBytes.byteLength || artifact.sha256 !== artifactDigest) {
    fail("native_reference", "native CPU reference identity or artifact differs");
  }
  const unsigned = { ...receipt };
  delete unsigned.receipt_id;
  const expectedId = `sha256:${sha256(canonicalBytes(unsigned))}`;
  if (receipt.receipt_id !== expectedId) fail("native_reference", "native reference receipt identity differs");
  return Object.freeze({
    schema: receipt.schema,
    scenarioId: receipt.scenario_id,
    sourceRevision: receipt.source_revision,
    backend: receipt.backend,
    backendId: receipt.backend_id,
    backendBuild: receipt.backend_build,
    physicalDevice: receipt.physical_device,
    artifactName: artifact.name,
    artifactBytes: artifact.bytes,
    artifactSha256: artifactDigest,
    receiptId: receipt.receipt_id,
    receiptDigest: sha256(canonicalBytes(receipt)),
    receipt: JSON.parse(canonicalBytes(receipt).toString("utf8")),
    export: exportReceipt,
    reload: reloadReceipt,
  });
}

export function validateNpmReceiptV1(receiptValue, archive, archiveBytes, revision) {
  const receipt = exactFields(receiptValue, [
    "artifact", "duration_ms", "evidence", "machine", "receipt_id", "release",
    "result", "run_id", "schema", "source_revision", "started_at_utc", "toolchain",
  ], "npm archive receipt");
  if (!(archiveBytes instanceof Uint8Array) || archiveBytes.byteLength !== archive.bytes ||
      sha256(archiveBytes) !== archive.sha256) {
    fail("npm_receipt", "npm archive bytes differ from admitted descriptor");
  }
  const machine = exactFields(
    receipt.machine,
    ["architecture", "machine_id", "system"],
    "npm receipt machine",
  );
  const toolchain = exactFields(receipt.toolchain, ["node", "npm"], "npm receipt toolchain");
  const artifact = exactFields(receipt.artifact, [
    "bytes", "integrity", "kind", "name", "package", "sha256",
  ], "npm receipt artifact");
  const evidence = exactFields(receipt.evidence, [
    "entry_count", "installed_offline", "source_dirty", "source_free",
    "strict_typescript", "wasm_build_id", "wasm_guest_digest",
  ], "npm receipt evidence");
  const expectedIntegrity = `sha512-${createHash("sha512").update(archiveBytes).digest("base64")}`;
  if (receipt.schema !== "tritium.npm-archive-qualification.v1" ||
      receipt.result !== "pass" || receipt.release !== PACKAGE_RELEASE ||
      receipt.source_revision !== revision ||
      typeof receipt.run_id !== "string" || receipt.run_id.length === 0 ||
      !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/.test(receipt.started_at_utc) ||
      typeof receipt.duration_ms !== "number" || !Number.isFinite(receipt.duration_ms) ||
      receipt.duration_ms <= 0 ||
      !/^sha256:[0-9a-f]{64}$/.test(machine.machine_id) ||
      typeof machine.system !== "string" || machine.system.length === 0 ||
      typeof machine.architecture !== "string" || machine.architecture.length === 0 ||
      !/^v[0-9]+(?:\.[0-9]+){2}$/.test(toolchain.node) ||
      !/^[0-9]+(?:\.[0-9]+){1,2}$/.test(toolchain.npm) ||
      artifact.kind !== "npm-archive" || artifact.name !== archive.name ||
      basename(artifact.name) !== artifact.name || artifact.name.includes("\\") ||
      artifact.name.includes("\0") || !artifact.name.endsWith(".tgz") ||
      artifact.package !== `@tritium-ai/web@${PACKAGE_RELEASE}` ||
      artifact.bytes !== archive.bytes || artifact.sha256 !== archive.sha256 ||
      artifact.integrity !== expectedIntegrity || evidence.source_dirty !== false ||
      evidence.source_free !== true || evidence.installed_offline !== true ||
      evidence.strict_typescript !== true || evidence.entry_count !== EXPECTED_NPM_ENTRY_COUNT ||
      evidence.wasm_build_id !==
        `tritium-wasm@${PACKAGE_RELEASE}+source-git:${revision}` ||
      typeof evidence.wasm_guest_digest !== "string" ||
      !/^[0-9a-f]{64}$/.test(evidence.wasm_guest_digest)) {
    fail("npm_receipt", "npm archive receipt is malformed, stale, dirty, or artifact-unbound");
  }
  const unsigned = { ...receipt };
  delete unsigned.receipt_id;
  const expectedId = `sha256:${sha256(canonicalBytes(unsigned))}`;
  if (receipt.receipt_id !== expectedId) {
    fail("npm_receipt", "npm archive receipt identity differs");
  }
  return Object.freeze({
    schema: receipt.schema,
    receiptId: receipt.receipt_id,
    sourceRevision: revision,
    archive,
    receipt: JSON.parse(canonicalBytes(receipt).toString("utf8")),
  });
}

function expectedBrowserName(engine, browserName) {
  const normalized = String(browserName ?? "").toLowerCase();
  if (engine === "chrome") return normalized === "chrome" || normalized === "chromium";
  if (engine === "firefox") return normalized === "firefox";
  return normalized === "safari";
}

function expectedPlatformName(osName, platformName) {
  const os = String(osName).toLowerCase();
  const platform = String(platformName ?? "").toLowerCase();
  if (os === "linux") return platform === "linux";
  if (os === "windows" || os === "win32") return platform === "windows";
  if (os === "macos" || os === "darwin") {
    return platform === "mac" || platform === "macos" || platform === "darwin";
  }
  return os === platform;
}

function validateBrowserTrace(traceValue, nativeReference) {
  const trace = object(traceValue, "browser trace");
  if (trace.schemaId !== "tritium.physical_browser_training_lane_trace" ||
      trace.schemaVersion !== 1 || trace.scenarioId !== SCENARIO_ID ||
      trace.implementation !== "webgpu" || trace.manifestDigest !== MANIFEST_DIGEST ||
      trace.vectorDigest !== VECTOR_DIGEST || !/^[0-9a-f]{64}$/.test(trace.executionDigest)) {
    fail("browser_trace", "browser trace identity differs");
  }
  if (typeof trace.buildId !== "string" ||
      !/^wgsl:[0-9a-f]{64}:browser-qualification:salt-ste-sgd-256-v1$/.test(trace.buildId)) {
    fail("browser_trace", "browser trace WGSL build identity differs");
  }
  const unsignedTrace = { ...trace };
  delete unsignedTrace.executionDigest;
  if (trace.executionDigest !== sha256(canonicalBytes(unsignedTrace))) {
    fail("browser_trace", "browser trace execution digest differs");
  }
  const adapter = object(trace.adapter, "browser adapter");
  const description = [adapter.vendor, adapter.architecture, adapter.device, adapter.description]
    .map((value, index) => nonEmpty(value, `browser adapter field ${index}`)).join(" ");
  if (adapter.software !== false || SOFTWARE_MARKERS.some((marker) =>
    description.toLowerCase().includes(marker))) {
    fail("browser_trace", "browser trace does not prove a physical adapter");
  }
  const limits = object(trace.limits, "browser limits");
  for (const field of [
    "maxBufferSize", "maxStorageBufferBindingSize",
    "maxComputeWorkgroupsPerDimension", "maxStorageBuffersPerShaderStage",
  ]) positive(limits[field], `browser limits.${field}`);
  const vector = object(trace.vector, "browser vector trace");
  const cases = object(vector.caseCounts, "browser vector case counts");
  const vectorCases = Array.isArray(vector.cases) ? vector.cases : [];
  validateCanonicalVectorTraceCases(vectorCases);
  const vectorDigest = sha256(canonicalBytes(vectorCases));
  const implementations = vectorCases.reduce((counts, item) => {
    const implementation = item?.implementation;
    counts[implementation] = (counts[implementation] ?? 0) + 1;
    return counts;
  }, {});
  if (vector.schemaId !== "tritium.webgpu_vector_conformance_trace" ||
      vector.schemaVersion !== 1 || vector.implementation !== "webgpu" ||
      vector.manifestDigest !== MANIFEST_DIGEST || vector.vectorDigest !== VECTOR_DIGEST ||
      vector.executionDigest !== vectorDigest || implementations.webgpu !== 68 ||
      implementations["wasm-codec"] !== 4 || implementations["wasm-validation"] !== 45 ||
      cases.valid !== 72 || cases.invalid !== 45 || cases.skipped !== 0 ||
      vector.webgpuCaseTransactions !== 68 || !Number.isSafeInteger(vector.webgpuDispatches) ||
      vector.webgpuDispatches <= 0 || vector.wasmCodecCalls !== 4 ||
      vector.wasmValidationCalls !== 45 || vector.wasmDispatches !== 0 ||
      trace.wasmDispatches !== 0 ||
      trace.steadyStateReadbacks !== 0 || !Number.isSafeInteger(trace.explicitReadbacks) ||
      trace.explicitReadbacks <= 0 || !Number.isSafeInteger(trace.peakBufferBytes) ||
      trace.peakBufferBytes <= 0) {
    fail("browser_trace", "browser execution counts or readback ledger differs");
  }
  const lifecycle = object(trace.lifecycle, "browser lifecycle");
  for (const field of [
    "prepare", "forward", "backward", "optimizerStep", "checkpointResume",
    "exportReload", "nativeArtifactParity",
  ]) {
    if (lifecycle[field] !== true) fail("browser_trace", `browser lifecycle ${field} did not pass`);
  }
  if (lifecycle.completedSteps !== 1 ||
      lifecycle.nativeArtifactSha256 !== nativeReference.artifactSha256 ||
      lifecycle.artifactSha256 !== nativeReference.artifactSha256 ||
      lifecycle.nativeReferenceDigest !== nativeReference.receiptDigest) {
    fail("browser_trace", "browser native artifact parity is unbound");
  }
  const lifecycleReceipts = Array.isArray(lifecycle.receipts) ? lifecycle.receipts : [];
  const requiredReceiptOperations = new Set([
    "session.forward", "session.backward", "session.step", "session.checkpoint",
    "session.resume", "session.export",
  ]);
  if (lifecycleReceipts.length !== requiredReceiptOperations.size ||
      lifecycleReceipts.some((receipt) =>
        !requiredReceiptOperations.delete(receipt?.operation) ||
        receipt?.physicalDevice !== trace.physicalDevice ||
        receipt?.buildId !== trace.buildId ||
        !Number.isSafeInteger(receipt?.peakResidentBytes) || receipt.peakResidentBytes <= 0
      ) || requiredReceiptOperations.size !== 0) {
    fail("browser_trace", "browser lifecycle receipts are incomplete or identity-drifted");
  }
  const faults = object(trace.faults, "browser faults");
  for (const field of [
    "deviceLoss", "allocationFailure", "malformedCheckpoint", "malformedSalt",
    "cancellation", "outOfOrder",
  ]) {
    if (faults[field]?.passed !== true || typeof faults[field]?.errorCode !== "string") {
      fail("browser_trace", `browser fault ${field} did not pass`);
    }
  }
  if (faults.cancellation?.errorCode !== "cancelled" ||
      !Number.isSafeInteger(faults.cancellation?.observedEvents) ||
      faults.cancellation.observedEvents < 1 ||
      faults.allocationFailure?.errorCode !== "injected_allocation_failure" ||
      faults.allocationFailure?.observedEvents !== 1) {
    fail("browser_trace", "browser physical fault observations are incomplete");
  }
  return trace;
}

export function assembleBrowserLaneV1(options) {
  const engine = nonEmpty(options.engine, "engine");
  if (!ENGINES.has(engine)) fail("lane", "engine must be chrome, firefox, or safari");
  const browserVersion = nonEmpty(options.browserVersion, "browser version");
  if (!/^[0-9]+(?:\.[0-9]+){1,3}$/.test(browserVersion)) {
    fail("lane", "browser version is not an exact numeric stable version");
  }
  if (!/^[0-9a-f]{40}$/.test(options.sourceRevision)) fail("lane", "source revision is invalid");
  const runId = nonEmpty(options.runId, "run id");
  const os = exactFields(options.os, ["architecture", "name", "version"], "host OS");
  for (const field of ["architecture", "name", "version"]) nonEmpty(os[field], `host OS.${field}`);
  if (engine === "safari" && !["macos", "darwin"].includes(os.name.toLowerCase())) {
    fail("lane", "Safari physical lane must run on macOS");
  }
  const capabilities = object(options.webdriverCapabilities, "WebDriver capabilities");
  if (!expectedBrowserName(engine, capabilities.browserName) ||
      capabilities.browserVersion !== browserVersion ||
      !expectedPlatformName(os.name, capabilities.platformName)) {
    fail("lane", "WebDriver browser identity differs from requested lane");
  }
  const trace = validateBrowserTrace(options.browserTrace, options.nativeReference);
  const { receipt: nativeReceipt, ...nativeReference } = options.nativeReference;
  if (typeof nativeReceipt !== "object" || nativeReceipt === null ||
      nativeReceipt.receipt_id !== nativeReference.receiptId) {
    fail("lane", "native receipt differs from validated native reference");
  }
  const npmQualification = exactFields(options.npmQualification, [
    "archive", "receipt", "receiptId", "schema", "sourceRevision",
  ], "npm qualification");
  if (npmQualification.schema !== "tritium.npm-archive-qualification.v1" ||
      npmQualification.sourceRevision !== options.sourceRevision ||
      npmQualification.archive.name !== options.archive.name ||
      npmQualification.archive.bytes !== options.archive.bytes ||
      npmQualification.archive.sha256 !== options.archive.sha256 ||
      npmQualification.receipt?.receipt_id !== npmQualification.receiptId) {
    fail("lane", "npm qualification differs from candidate lane identity");
  }
  const traceFile = safeRelativeFile(options.traceFile, "trace file");
  const traceEvidence = {
    schema: "tritium.browser-training-lane-evidence.v1",
    run_id: runId,
    engine,
    source_revision: options.sourceRevision,
    archive: options.archive,
    npm_receipt: npmQualification.receipt,
    native_receipt: nativeReceipt,
    native_reference: nativeReference,
    webdriver_capabilities: capabilities,
    browser_trace: trace,
  };
  const traceBytes = Buffer.concat([canonicalBytes(traceEvidence), Buffer.from("\n")]);
  const lane = {
    engine,
    browser_version: browserVersion,
    os: { name: os.name, version: os.version, architecture: os.architecture },
    adapter: {
      vendor: trace.adapter.vendor,
      architecture: trace.adapter.architecture,
      device: trace.adapter.device,
      description: trace.adapter.description,
      software: false,
    },
    limits: {
      max_buffer_size: trace.limits.maxBufferSize,
      max_storage_buffer_binding_size: trace.limits.maxStorageBufferBindingSize,
      max_compute_workgroups_per_dimension: trace.limits.maxComputeWorkgroupsPerDimension,
      max_storage_buffers_per_shader_stage: trace.limits.maxStorageBuffersPerShaderStage,
    },
    case_counts: { valid: 72, invalid: 45, skipped: 0 },
    lifecycle: {
      prepare: true,
      forward: true,
      backward: true,
      optimizer_step: true,
      checkpoint_resume: true,
      export_reload: true,
      native_artifact_parity: true,
    },
    faults: {
      device_loss: true,
      allocation_failure: true,
      malformed_checkpoint: true,
      malformed_salt: true,
      cancellation: true,
      out_of_order: true,
    },
    trace: {
      file: traceFile,
      bytes: traceBytes.byteLength,
      sha256: sha256(traceBytes),
      steady_state_readbacks: 0,
      wasm_dispatches: 0,
      explicit_readbacks: trace.explicitReadbacks,
      peak_buffer_bytes: trace.peakBufferBytes,
    },
  };
  return Object.freeze({ lane: Object.freeze(lane), traceBytes });
}

export class WebDriverClassicClient {
  constructor(baseUrl) {
    let parsed;
    try { parsed = new URL(baseUrl); } catch { fail("webdriver", "WebDriver URL is invalid"); }
    if (!["http:", "https:"].includes(parsed.protocol) || parsed.username || parsed.password || parsed.search || parsed.hash) {
      fail("webdriver", "WebDriver URL must be an unauthenticated HTTP endpoint");
    }
    if (!["127.0.0.1", "localhost", "[::1]"].includes(parsed.hostname)) {
      fail("webdriver", "WebDriver endpoint must be loopback-local to bind host OS identity");
    }
    this.baseUrl = parsed.href.replace(/\/$/, "");
  }

  async request(method, path, body = undefined, timeoutMs = 60_000) {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers: body === undefined ? undefined : { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    }).catch((error) => fail("webdriver", `WebDriver transport failed: ${error.message}`));
    const declaredLength = Number(response.headers.get("content-length"));
    if (Number.isFinite(declaredLength) && declaredLength > MAX_WEBDRIVER_RESULT_BYTES) {
      fail("webdriver", "WebDriver response exceeded limit");
    }
    const bytes = Buffer.from(await response.arrayBuffer());
    if (bytes.byteLength > MAX_WEBDRIVER_RESULT_BYTES) fail("webdriver", "WebDriver response exceeded limit");
    let envelope;
    try { envelope = JSON.parse(bytes.toString("utf8")); } catch { fail("webdriver", "WebDriver returned non-JSON"); }
    const value = envelope?.value;
    if (!response.ok || (value && typeof value === "object" && typeof value.error === "string")) {
      fail("webdriver", `WebDriver command failed: ${value?.error ?? response.status}`);
    }
    return value;
  }

  async createSession(engine) {
    if (!ENGINES.has(engine)) fail("webdriver", "unsupported WebDriver engine");
    const value = await this.request("POST", "/session", {
      capabilities: { alwaysMatch: { browserName: engine === "chrome" ? "chrome" : engine } },
    });
    if (!value || typeof value.sessionId !== "string" || value.sessionId.length === 0 ||
        typeof value.capabilities !== "object" || value.capabilities === null) {
      fail("webdriver", "WebDriver new-session response is malformed");
    }
    return Object.freeze({ id: value.sessionId, capabilities: value.capabilities });
  }

  setTimeouts(sessionId, timeouts) {
    return this.request("POST", `/session/${encodeURIComponent(sessionId)}/timeouts`, timeouts);
  }

  navigate(sessionId, url) {
    return this.request("POST", `/session/${encodeURIComponent(sessionId)}/url`, { url });
  }

  executeAsync(sessionId, script, args) {
    return this.request("POST", `/session/${encodeURIComponent(sessionId)}/execute/async`, {
      script,
      args,
    }, 31 * 60_000);
  }

  deleteSession(sessionId) {
    return this.request("DELETE", `/session/${encodeURIComponent(sessionId)}`);
  }
}

const RUNNER_HTML = `<!doctype html>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'">
<title>Tritium physical browser qualification</title>
<script type="module" src="/runner.mjs"></script>`;

const RUNNER_MODULE = `
import { runPhysicalBrowserTrainingLaneV1 } from "/node_modules/@tritium-ai/web/dist/qualification.js";
globalThis.__tritiumPhysicalLaneReady = Promise.resolve(async (options) =>
  runPhysicalBrowserTrainingLaneV1({
    nativeArtifact: Uint8Array.from(options.nativeArtifact),
    nativeReferenceDigest: options.nativeReferenceDigest,
  })
);
`;

const ASYNC_SCRIPT = `
const nativeArtifact = arguments[0];
const nativeReferenceDigest = arguments[1];
const done = arguments[arguments.length - 1];
Promise.resolve(globalThis.__tritiumPhysicalLaneReady)
  .then((run) => run({ nativeArtifact, nativeReferenceDigest }))
  .then(
    (value) => done({ ok: true, value }),
    (error) => done({ ok: false, error: {
      name: String(error?.name ?? "Error"),
      code: String(error?.code ?? "unknown"),
      message: String(error?.message ?? error),
      stack: String(error?.stack ?? ""),
    } }),
  );
`;

async function serveDirectory(root) {
  const admittedRoot = await realpath(root);
  const server = createServer(async (request, response) => {
    try {
      if (request.method !== "GET" && request.method !== "HEAD") {
        response.writeHead(405).end();
        return;
      }
      const parsed = new URL(request.url, "http://127.0.0.1");
      let logical;
      try { logical = decodeURIComponent(parsed.pathname); } catch { response.writeHead(400).end(); return; }
      const parts = logical.split("/").filter(Boolean);
      if (parts.some((part) => part === "." || part === ".." || part.includes("\0"))) {
        response.writeHead(400).end();
        return;
      }
      const candidate = await realpath(join(admittedRoot, ...parts)).catch(() => null);
      if (candidate === null || (candidate !== admittedRoot && !candidate.startsWith(`${admittedRoot}${sep}`))) {
        response.writeHead(404).end();
        return;
      }
      const metadata = await stat(candidate);
      if (!metadata.isFile()) { response.writeHead(404).end(); return; }
      const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".wasm": "application/wasm" };
      response.writeHead(200, {
        "content-type": types[extname(candidate)] ?? "application/octet-stream",
        "content-length": metadata.size,
        "cache-control": "no-store",
        "x-content-type-options": "nosniff",
      });
      if (request.method === "HEAD") response.end();
      else createReadStream(candidate).pipe(response);
    } catch {
      if (!response.headersSent) response.writeHead(500);
      response.end();
    }
  });
  await new Promise((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolveListen);
  });
  const address = server.address();
  return {
    origin: `http://127.0.0.1:${address.port}`,
    close: () => new Promise((resolveClose, reject) =>
      server.close((error) => error ? reject(error) : resolveClose())),
  };
}

async function installCandidate(archive, temporary) {
  await writeFile(join(temporary, "package.json"), JSON.stringify({
    name: "tritium-browser-qualification",
    private: true,
    type: "module",
  }));
  await run("npm", [
    "install", "--offline", "--ignore-scripts", "--no-audit", "--no-fund",
    "--package-lock=false", archive,
  ], { cwd: temporary, maxBuffer: MAX_COMMAND_OUTPUT_BYTES });
  await writeFile(join(temporary, "runner.html"), RUNNER_HTML);
  await writeFile(join(temporary, "runner.mjs"), RUNNER_MODULE);
  const packageRoot = await realpath(join(temporary, "node_modules", "@tritium-ai", "web"));
  if (!packageRoot.startsWith(`${await realpath(temporary)}${sep}`)) {
    fail("archive", "installed candidate escaped temporary consumer");
  }
}

async function fsyncFile(path) {
  const descriptor = await open(path, fsConstants.O_RDONLY);
  try { await descriptor.sync(); } finally { await descriptor.close(); }
}

async function fsyncDirectory(path) {
  if (process.platform === "win32") return;
  const descriptor = await open(path, fsConstants.O_RDONLY);
  try { await descriptor.sync(); } finally { await descriptor.close(); }
}

async function requireCleanRevision(repo, revision) {
  const head = (await run("git", ["rev-parse", "HEAD"], { cwd: repo })).stdout.trim();
  const status = (await run("git", ["status", "--short", "--untracked-files=all"], { cwd: repo })).stdout.trim();
  if (head !== revision || status !== "") fail("source", "browser lane requires clean exact source revision");
}

export async function qualifyBrowserLaneV1(options) {
  const engine = nonEmpty(options.engine, "engine");
  if (!ENGINES.has(engine)) fail("options", "unsupported browser engine");
  const revision = hex(options.sourceRevision, "source revision", 40);
  const outputDir = resolve(options.outputDir);
  await access(outputDir).then(
    () => fail("output", "browser lane output directory already exists"),
    () => undefined,
  );
  await requireCleanRevision(resolve(options.repo ?? SCRIPT_ROOT), revision);
  const archivePath = resolve(options.artifact);
  const nativeArtifactPath = resolve(options.nativeArtifact);
  const archive = await fileDescriptor(archivePath, "npm archive", MAX_ARCHIVE_BYTES);
  const archiveBytes = await readFile(archivePath);
  if (!archive.name.endsWith(".tgz")) fail("archive", "candidate archive must end in .tgz");
  const nativeArtifactDescriptor = await fileDescriptor(nativeArtifactPath, "native artifact", MAX_ARTIFACT_BYTES);
  const nativeArtifactBytes = await readFile(nativeArtifactPath);
  const npmReceiptPath = resolve(options.npmReceipt);
  const nativeReceiptPath = resolve(options.nativeReferenceReceipt);
  await fileDescriptor(npmReceiptPath, "npm receipt", MAX_RECEIPT_BYTES);
  await fileDescriptor(nativeReceiptPath, "native reference receipt", MAX_RECEIPT_BYTES);
  const npmReceiptBytes = await readFile(npmReceiptPath);
  const nativeReceiptBytes = await readFile(nativeReceiptPath);
  let npmReceiptValue;
  let nativeReceiptValue;
  try {
    npmReceiptValue = JSON.parse(npmReceiptBytes);
    nativeReceiptValue = JSON.parse(nativeReceiptBytes);
  } catch {
    fail("receipt", "qualification receipt is not UTF-8 JSON");
  }
  const npmQualification = validateNpmReceiptV1(
    npmReceiptValue,
    archive,
    archiveBytes,
    revision,
  );
  const nativeReference = validateNativeReferenceV1(
    nativeReceiptValue,
    nativeArtifactBytes,
    nativeArtifactDescriptor.name,
    revision,
  );

  await mkdir(dirname(outputDir), { recursive: true });
  const temporary = await mkdtemp(join(dirname(outputDir), `.${basename(outputDir)}.consumer-`));
  const stage = await mkdtemp(join(dirname(outputDir), `.${basename(outputDir)}.stage-`));
  let server = null;
  let session = null;
  const client = new WebDriverClassicClient(options.webdriverUrl);
  try {
    await installCandidate(archivePath, temporary);
    server = await serveDirectory(temporary);
    session = await client.createSession(engine);
    await client.setTimeouts(session.id, { pageLoad: 60_000, script: 30 * 60_000 });
    await client.navigate(session.id, `${server.origin}/runner.html`);
    const result = await client.executeAsync(session.id, ASYNC_SCRIPT, [
      [...nativeArtifactBytes],
      nativeReference.receiptDigest,
    ]);
    if (!result || result.ok !== true || typeof result.value !== "object" || result.value === null) {
      const code = result?.error?.code ?? "malformed result";
      const message = typeof result?.error?.message === "string" ? `: ${result.error.message}` : "";
      fail("browser", `physical browser qualification failed: ${code}${message}`);
    }
    const browserVersion = nonEmpty(options.expectedBrowserVersion, "expected browser version");
    if (session.capabilities.browserVersion !== browserVersion) {
      fail("browser", "WebDriver browser version differs from predeclared stable version");
    }
    const assembled = assembleBrowserLaneV1({
      engine,
      browserVersion,
      os: options.os,
      runId: options.runId,
      sourceRevision: revision,
      archive,
      npmQualification,
      nativeReference,
      webdriverCapabilities: session.capabilities,
      browserTrace: result.value,
      traceFile: "trace.json",
    });
    const tracePath = join(stage, "trace.json");
    const lanePath = join(stage, "lane.json");
    await writeFile(tracePath, assembled.traceBytes, { flag: "wx" });
    await writeFile(lanePath, Buffer.concat([canonicalBytes(assembled.lane), Buffer.from("\n")]), { flag: "wx" });
    await fsyncFile(tracePath);
    await fsyncFile(lanePath);
    await fsyncDirectory(stage);
    await rename(stage, outputDir);
    await fsyncDirectory(dirname(outputDir));
    return assembled.lane;
  } finally {
    if (session !== null) await client.deleteSession(session.id).catch(() => undefined);
    if (server !== null) await server.close().catch(() => undefined);
    await rm(temporary, { recursive: true, force: true });
    await rm(stage, { recursive: true, force: true });
  }
}

function parseArgs(argv) {
  const values = {};
  const allowed = new Set([
    "artifact", "engine", "expected-browser-version", "native-artifact", "native-reference-receipt", "npm-receipt",
    "output-dir", "repo", "run-id", "source-revision", "webdriver-url",
  ]);
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    if (!name?.startsWith("--") || value === undefined) fail("options", "arguments must be --name value pairs");
    const key = name.slice(2);
    if (!allowed.has(key)) fail("options", `unknown argument --${key}`);
    if (key in values) fail("options", `duplicate argument --${key}`);
    values[key] = value;
  }
  const required = [
    "artifact", "engine", "expected-browser-version", "native-artifact", "native-reference-receipt", "npm-receipt",
    "output-dir", "run-id", "source-revision", "webdriver-url",
  ];
  for (const key of required) if (!(key in values)) fail("options", `missing --${key}`);
  return {
    artifact: values.artifact,
    engine: values.engine,
    expectedBrowserVersion: values["expected-browser-version"],
    nativeArtifact: values["native-artifact"],
    nativeReferenceReceipt: values["native-reference-receipt"],
    npmReceipt: values["npm-receipt"],
    os: {
      architecture: hostArchitecture(),
      name: hostPlatform() === "darwin" ? "macOS"
        : hostPlatform() === "win32" ? "Windows"
          : hostPlatform() === "linux" ? "Linux" : hostPlatform(),
      version: hostRelease(),
    },
    outputDir: values["output-dir"],
    repo: values.repo ?? SCRIPT_ROOT,
    runId: values["run-id"],
    sourceRevision: values["source-revision"],
    webdriverUrl: values["webdriver-url"],
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  qualifyBrowserLaneV1(parseArgs(process.argv.slice(2))).then(
    (lane) => process.stdout.write(`${canonicalJson(lane)}\n`),
    (error) => {
      process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
      process.exitCode = 1;
    },
  );
}

import init, {
  tritium_execute_portable_request_json,
  tritium_portable_build_id,
  tritium_portable_conformance_case_count,
  tritium_portable_manifest_digest,
  tritium_portable_max_caller_bytes,
  tritium_portable_max_linear_memory_bytes,
  tritium_portable_operation_count,
  tritium_portable_report_digest,
  tritium_portable_vector_digest,
} from "../.generated/tritium_wasm.js";
import { WASM_GUEST_DIGEST_V1 } from "../.generated/wasm_identity.ts";
import { blake3 } from "@noble/hashes/blake3.js";
import { bytesToHex } from "@noble/hashes/utils.js";

import {
  TRAINING_MANIFEST_DIGEST_V1,
  TRAINING_VECTOR_DIGEST_V1,
} from "./identity.ts";
import type {
  PortableBufferV1,
  PortableExecutionV1,
  PortableTrainingRequestV1,
  PortableTrainingResponseV1,
  PortableWasmConformanceReceiptV1,
  PortableWasmSourceV1,
} from "./portable.js";
export type {
  PortableAttributeV1,
  PortableBufferDataV1,
  PortableBufferV1,
  PortableExecutionV1,
  PortableTrainingErrorV1,
  PortableTrainingReceiptV1,
  PortableTrainingRequestV1,
  PortableTrainingResponseV1,
  PortableWasmConformanceReceiptV1,
  PortableWasmSourceV1,
} from "./portable.js";

const MAX_REQUEST_JSON_BYTES = 8 * 1024 * 1024;
const MAX_CALLER_BYTES = 64 * 1024 * 1024;
const MAX_LINEAR_MEMORY_BYTES = 192 * 1024 * 1024;
const UTF8 = new TextEncoder();

let initialized: Promise<void> | null = null;

class GuestTrapError extends Error {}

export async function snapshotPortableWasmSource(
  source: PortableWasmSourceV1,
): Promise<Uint8Array> {
  if (source instanceof Response) {
    return new Uint8Array(await source.arrayBuffer());
  }
  if (ArrayBuffer.isView(source)) {
    return new Uint8Array(
      source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength),
    );
  }
  if (source instanceof ArrayBuffer) return new Uint8Array(source.slice(0));
  if (source instanceof URL && source.protocol === "file:") {
    const moduleName = "node:fs/promises";
    const fileSystem = await import(moduleName) as { readFile(url: URL): Promise<Uint8Array> };
    return Uint8Array.from(await fileSystem.readFile(source));
  }
  const response = await fetch(source);
  if (!response.ok) {
    throw new Error(`portable WASM fetch failed with HTTP ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function initializeGuest(source: PortableWasmSourceV1): Promise<string> {
  const guestBytes = await snapshotPortableWasmSource(source);
  const guestDigest = bytesToHex(blake3(guestBytes));
  if (guestDigest !== WASM_GUEST_DIGEST_V1) {
    throw new Error(
      `portable WASM guest digest mismatch: expected ${WASM_GUEST_DIGEST_V1}, got ${guestDigest}`,
    );
  }
  if (initialized === null) {
    const attempt = init({ module_or_path: guestBytes }).then(() => undefined);
    initialized = attempt;
    try {
      await attempt;
    } catch (error) {
      if (initialized === attempt) initialized = null;
      throw new GuestTrapError("portable WASM guest failed to initialize", {
        cause: error,
      });
    }
  } else {
    try {
      await initialized;
    } catch (error) {
      throw new GuestTrapError("portable WASM guest failed to initialize", {
        cause: error,
      });
    }
  }

  let buildId: string;
  let manifestDigest: string;
  let vectorDigest: string;
  let maxCallerBytes: number;
  let maxLinearMemoryBytes: number;
  try {
    buildId = tritium_portable_build_id();
    manifestDigest = tritium_portable_manifest_digest();
    vectorDigest = tritium_portable_vector_digest();
    maxCallerBytes = tritium_portable_max_caller_bytes();
    maxLinearMemoryBytes = tritium_portable_max_linear_memory_bytes();
  } catch (error) {
    throw new GuestTrapError("portable WASM identity export trapped", {
      cause: error,
    });
  }
  if (
    manifestDigest !== TRAINING_MANIFEST_DIGEST_V1 ||
    vectorDigest !== TRAINING_VECTOR_DIGEST_V1 ||
    maxCallerBytes !== MAX_CALLER_BYTES ||
    maxLinearMemoryBytes !== MAX_LINEAR_MEMORY_BYTES
  ) {
    throw new Error("portable WASM guest identity or limits mismatch");
  }
  return buildId;
}

function deepFreeze<T>(value: T): T {
  if (typeof value !== "object" || value === null || Object.isFrozen(value)) {
    return value;
  }
  for (const nested of Object.values(value)) deepFreeze(nested);
  return Object.freeze(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
): boolean {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return (
    actual.length === wanted.length &&
    actual.every((key, index) => key === wanted[index])
  );
}

function isDenseNumericArray(
  value: unknown,
  maximum: number,
): value is number[] {
  if (!Array.isArray(value)) return false;
  for (let index = 0; index < value.length; index += 1) {
    const item = value[index];
    if (
      !(index in value) ||
      typeof item !== "number" ||
      !Number.isSafeInteger(item) ||
      item < 0 ||
      item > maximum
    ) {
      return false;
    }
  }
  return true;
}

function validOutputBuffer(value: unknown, expected: PortableBufferV1): boolean {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["data", "name", "shape"]) ||
    value.name !== expected.name ||
    !Array.isArray(value.shape) ||
    value.shape.length !== expected.shape.length ||
    value.shape.some((dimension, index) => dimension !== expected.shape[index]) ||
    !isRecord(value.data) ||
    value.data.dtype !== expected.data.dtype
  ) {
    return false;
  }
  if (value.data.dtype === "f32") {
    return (
      hasExactKeys(value.data, ["bits", "dtype"]) &&
      isDenseNumericArray(value.data.bits, 0xffff_ffff) &&
      expected.data.dtype === "f32" &&
      value.data.bits.length === expected.data.bits.length
    );
  }
  if (value.data.dtype === "u32") {
    return (
      hasExactKeys(value.data, ["dtype", "values"]) &&
      isDenseNumericArray(value.data.values, 0xffff_ffff) &&
      expected.data.dtype === "u32" &&
      value.data.values.length === expected.data.values.length
    );
  }
  return (
    value.data.dtype === "bytes" &&
    hasExactKeys(value.data, ["dtype", "values"]) &&
    isDenseNumericArray(value.data.values, 0xff) &&
    expected.data.dtype === "bytes" &&
    value.data.values.length === expected.data.values.length
  );
}

function sameNumericArray(
  value: readonly number[],
  expected: readonly number[],
): boolean {
  return (
    value.length === expected.length &&
    value.every((item, index) => item === expected[index])
  );
}

function outputValuesMatch(value: unknown, expected: PortableBufferV1): boolean {
  if (!isRecord(value) || !isRecord(value.data)) return false;
  if (value.data.dtype === "f32" && expected.data.dtype === "f32") {
    return (
      Array.isArray(value.data.bits) &&
      sameNumericArray(value.data.bits, expected.data.bits)
    );
  }
  if (value.data.dtype === "u32" && expected.data.dtype === "u32") {
    return (
      Array.isArray(value.data.values) &&
      sameNumericArray(value.data.values, expected.data.values)
    );
  }
  if (value.data.dtype === "bytes" && expected.data.dtype === "bytes") {
    return (
      Array.isArray(value.data.values) &&
      sameNumericArray(value.data.values, expected.data.values)
    );
  }
  return false;
}

function outputsMatch(
  outputs: unknown[],
  expected: readonly PortableBufferV1[],
  allowEmpty: boolean,
  requireExactValues: boolean,
): boolean {
  if (allowEmpty && outputs.length === 0) return true;
  return (
    outputs.length === expected.length &&
    outputs.every(
      (output, index) =>
        validOutputBuffer(output, expected[index]!) &&
        (!requireExactValues || outputValuesMatch(output, expected[index]!)),
    )
  );
}

interface DigestWriter {
  update(data: Uint8Array): unknown;
}

function hashU64(writer: DigestWriter, value: number): void {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(value), true);
  writer.update(bytes);
}

function hashString(writer: DigestWriter, value: string): void {
  const bytes = UTF8.encode(value);
  hashU64(writer, bytes.length);
  writer.update(bytes);
}

function hashU32Array(writer: DigestWriter, values: readonly number[]): void {
  const bytes = new Uint8Array(4);
  const view = new DataView(bytes.buffer);
  for (const value of values) {
    view.setUint32(0, value, true);
    writer.update(bytes);
  }
}

function hashBuffer(writer: DigestWriter, buffer: PortableBufferV1): void {
  hashString(writer, buffer.name);
  hashU64(writer, buffer.shape.length);
  for (const dimension of buffer.shape) hashU64(writer, dimension);
  if (buffer.data.dtype === "f32") {
    writer.update(Uint8Array.of(0));
    hashU64(writer, buffer.data.bits.length);
    hashU32Array(writer, buffer.data.bits);
  } else if (buffer.data.dtype === "u32") {
    writer.update(Uint8Array.of(1));
    hashU64(writer, buffer.data.values.length);
    hashU32Array(writer, buffer.data.values);
  } else {
    writer.update(Uint8Array.of(2));
    hashU64(writer, buffer.data.values.length);
    writer.update(Uint8Array.from(buffer.data.values));
  }
}

function requestDigest(request: PortableTrainingRequestV1): string {
  const writer = blake3.create();
  hashString(writer, request.operation);
  writer.update(
    Uint8Array.of(
      ["forward", "vjp", "step", "checkpoint", "resume", "export", "reload"].indexOf(
        request.execution,
      ),
    ),
  );
  hashU64(writer, request.inputs.length);
  for (const buffer of request.inputs) hashBuffer(writer, buffer);
  hashU64(writer, request.attributes.length);
  for (const attribute of request.attributes) {
    hashString(writer, attribute.name);
    if (attribute.kind === "f32") {
      writer.update(Uint8Array.of(0));
      hashU32Array(writer, [attribute.bits]);
    } else if (attribute.kind === "u64") {
      writer.update(Uint8Array.of(1));
      hashU64(writer, attribute.value);
    } else if (attribute.kind === "bool") {
      writer.update(Uint8Array.of(2, Number(attribute.value)));
    } else if (attribute.kind === "text") {
      writer.update(Uint8Array.of(3));
      hashString(writer, attribute.value);
    } else if (attribute.kind === "u64-list") {
      writer.update(Uint8Array.of(4));
      hashU64(writer, attribute.values.length);
      for (const value of attribute.values) hashU64(writer, value);
    } else {
      writer.update(Uint8Array.of(5));
      writer.update(Uint8Array.of(1));
      hashU64(writer, attribute.values.length);
      hashU32Array(writer, attribute.values);
    }
  }
  return bytesToHex(writer.digest());
}

function outputDigest(outputs: readonly PortableBufferV1[]): string {
  const writer = blake3.create();
  hashU64(writer, outputs.length);
  for (const buffer of outputs) hashBuffer(writer, buffer);
  return bytesToHex(writer.digest());
}

function expectedReceiptDtype(execution: PortableExecutionV1): "f32" | "bytes" {
  return ["checkpoint", "resume", "export", "reload"].includes(execution)
    ? "bytes"
    : "f32";
}

function validateResponse(
  value: unknown,
  request: PortableTrainingRequestV1,
  backendBuild: string,
): PortableTrainingResponseV1 {
  if (
    !isRecord(value) ||
    !hasExactKeys(
      value,
      value.status === "ok"
        ? ["outputs", "receipt", "schemaId", "schemaVersion", "status"]
        : ["error", "outputs", "schemaId", "schemaVersion", "status"],
    ) ||
    (value.status !== "ok" && value.status !== "error") ||
    value.schemaId !== "tritium.portable_training_response" ||
    value.schemaVersion !== 1 ||
    !Array.isArray(value.outputs) ||
    !outputsMatch(
      value.outputs,
      request.outputs,
      value.status === "error",
      value.status === "error",
    )
  ) {
    throw new Error("portable WASM returned an invalid response envelope");
  }
  if (value.status === "ok") {
    const expectedInputDigest = requestDigest(request);
    const expectedOutputDigest = outputDigest(
      value.outputs as PortableBufferV1[],
    );
    if (
      !isRecord(value.receipt) ||
      !hasExactKeys(value.receipt, [
        "backendBuild",
        "backendId",
        "deviceResident",
        "dtype",
        "execution",
        "hostTransfers",
        "inputDigest",
        "manifestDigest",
        "maxBytes",
        "maxElements",
        "maxRank",
        "operation",
        "outputDigest",
        "peakResidentBytes",
        "physicalDevice",
        "scratchBytes",
        "vectorDigest",
      ]) ||
      value.receipt.backendId !== "wasm.portable.v1" ||
      value.receipt.backendBuild !== backendBuild ||
      value.receipt.manifestDigest !== TRAINING_MANIFEST_DIGEST_V1 ||
      value.receipt.vectorDigest !== request.vectorDigest ||
      value.receipt.operation !== request.operation ||
      value.receipt.execution !== request.execution ||
      value.receipt.physicalDevice !== request.physicalDevice ||
      value.receipt.hostTransfers !== 0 ||
      value.receipt.deviceResident !== true ||
      value.receipt.dtype !== expectedReceiptDtype(request.execution) ||
      value.receipt.maxRank !== 4 ||
      value.receipt.maxElements !== 8 * 1024 * 1024 ||
      value.receipt.maxBytes !== 8 * 1024 * 1024 ||
      typeof value.receipt.peakResidentBytes !== "number" ||
      !Number.isSafeInteger(value.receipt.peakResidentBytes) ||
      value.receipt.peakResidentBytes < 0 ||
      value.receipt.peakResidentBytes > MAX_CALLER_BYTES ||
      typeof value.receipt.scratchBytes !== "number" ||
      !Number.isSafeInteger(value.receipt.scratchBytes) ||
      value.receipt.scratchBytes < 0 ||
      value.receipt.scratchBytes > 128 * 1024 * 1024 ||
      value.receipt.peakResidentBytes + value.receipt.scratchBytes >
        MAX_LINEAR_MEMORY_BYTES ||
      value.receipt.inputDigest !== expectedInputDigest ||
      value.receipt.outputDigest !== expectedOutputDigest
    ) {
      throw new Error("portable WASM returned an invalid success receipt");
    }
  } else if (
    !isRecord(value.error) ||
    !hasExactKeys(value.error, ["category", "code", "message"]) ||
    typeof value.error.category !== "string" ||
    typeof value.error.code !== "string" ||
    typeof value.error.message !== "string"
  ) {
    throw new Error("portable WASM returned an invalid error response");
  }
  if (
    value.status === "error" &&
    value.outputs.length === 0 &&
    request.outputs.length !== 0 &&
    isRecord(value.error) &&
    typeof value.error.code === "string" &&
    ![
      "caller_bytes",
      "collection_items",
      "invalid_digest",
      "invalid_json",
      "invalid_name",
      "list_items",
      "missing_field",
      "physical_device",
      "rank",
      "request_bytes",
      "serialization",
      "text_bytes",
      "unsafe_integer",
      "unsupported_schema",
      "vector_digest_mismatch",
    ].includes(value.error.code)
  ) {
    throw new Error("portable WASM backend error omitted output sentinels");
  }
  return deepFreeze(value as PortableTrainingResponseV1);
}

function localError(
  code: string,
  message: string,
  category = "invalid_request",
): PortableTrainingResponseV1 {
  return deepFreeze({
    status: "error",
    schemaId: "tritium.portable_training_response",
    schemaVersion: 1,
    outputs: [],
    error: { category, code, message },
  });
}

export type PreparedPortableWasmExecutor = Readonly<{
  buildId: string;
  execute(request: PortableTrainingRequestV1): Promise<PortableTrainingResponseV1>;
}>;

type AdmittedPortableRequest = Readonly<{
  requestJson: string;
  requestSnapshot: PortableTrainingRequestV1;
}>;

function admitPortableRequest(
  request: PortableTrainingRequestV1,
): AdmittedPortableRequest | PortableTrainingResponseV1 {
  let requestJson: string;
  let requestSnapshot: PortableTrainingRequestV1;
  try {
    requestJson = JSON.stringify(request);
    if (requestJson === undefined) throw new Error("undefined JSON result");
    requestSnapshot = JSON.parse(requestJson) as PortableTrainingRequestV1;
  } catch {
    return localError(
      "invalid_json",
      "portable WASM request is not JSON serializable",
    );
  }
  if (UTF8.encode(requestJson).byteLength > MAX_REQUEST_JSON_BYTES) {
    return localError(
      "request_bytes",
      "portable WASM request JSON exceeds 8 MiB",
      "capacity",
    );
  }
  return Object.freeze({ requestJson, requestSnapshot });
}

async function executeAdmittedRequest(
  admitted: AdmittedPortableRequest,
  backendBuild: string,
): Promise<PortableTrainingResponseV1> {
  let responseJson: string;
  try {
    responseJson = tritium_execute_portable_request_json(admitted.requestJson);
  } catch {
    return localError("guest_trap", "portable WASM guest trapped", "internal");
  }
  return validateResponse(
    JSON.parse(responseJson) as unknown,
    admitted.requestSnapshot,
    backendBuild,
  );
}

async function executeInitializedRequest(
  request: PortableTrainingRequestV1,
  backendBuild: string,
): Promise<PortableTrainingResponseV1> {
  const admitted = admitPortableRequest(request);
  return "status" in admitted
    ? admitted
    : executeAdmittedRequest(admitted, backendBuild);
}

/** Snapshot, admit, and initialize one guest for repeated request execution. */
export async function preparePortableWasmExecutor(
  source: PortableWasmSourceV1 = new URL(
    "./tritium_wasm_bg.wasm",
    import.meta.url,
  ),
): Promise<PreparedPortableWasmExecutor> {
  const buildId = await initializeGuest(source);
  return Object.freeze({
    buildId,
    execute: (request: PortableTrainingRequestV1) => executeInitializedRequest(request, buildId),
  });
}

/** Execute one strict request through Rust-owned WASM semantics. */
export async function executePortableWasmRequest(
  request: PortableTrainingRequestV1,
  source: PortableWasmSourceV1 = new URL(
    "./tritium_wasm_bg.wasm",
    import.meta.url,
  ),
): Promise<PortableTrainingResponseV1> {
  const admitted = admitPortableRequest(request);
  if ("status" in admitted) return admitted;
  let executor: PreparedPortableWasmExecutor;
  try {
    executor = await preparePortableWasmExecutor(source);
  } catch (error) {
    if (error instanceof GuestTrapError) {
      return localError("guest_trap", "portable WASM guest trapped", "internal");
    }
    throw error;
  }
  return executeAdmittedRequest(admitted, executor.buildId);
}

/** Execute the complete canonical vector corpus twice inside the bundled guest. */
export async function runPortableWasmConformance(
  source: PortableWasmSourceV1 = new URL(
    "./tritium_wasm_bg.wasm",
    import.meta.url,
  ),
): Promise<PortableWasmConformanceReceiptV1> {
  const buildId = await initializeGuest(source);
  let firstExecutionDigest: string;
  let secondExecutionDigest: string;
  let operationCount: number;
  let caseCount: number;
  let maxCallerBytes: number;
  let maxLinearMemoryBytes: number;
  let guestManifestDigest: string;
  let guestVectorDigest: string;
  try {
    firstExecutionDigest = tritium_portable_report_digest();
    secondExecutionDigest = tritium_portable_report_digest();
    operationCount = tritium_portable_operation_count();
    caseCount = tritium_portable_conformance_case_count();
    maxCallerBytes = tritium_portable_max_caller_bytes();
    maxLinearMemoryBytes = tritium_portable_max_linear_memory_bytes();
    guestManifestDigest = tritium_portable_manifest_digest();
    guestVectorDigest = tritium_portable_vector_digest();
  } catch (error) {
    throw new GuestTrapError("portable WASM conformance export trapped", {
      cause: error,
    });
  }
  if (
    operationCount !== 35 ||
    caseCount !== 114 ||
    maxCallerBytes !== MAX_CALLER_BYTES ||
    maxLinearMemoryBytes !== MAX_LINEAR_MEMORY_BYTES ||
    guestManifestDigest !== TRAINING_MANIFEST_DIGEST_V1 ||
    guestVectorDigest !== TRAINING_VECTOR_DIGEST_V1 ||
    !/^[0-9a-f]{64}$/.test(firstExecutionDigest) ||
    secondExecutionDigest !== firstExecutionDigest
  ) {
    throw new Error(
      `portable WASM conformance failed: operations=${operationCount}, cases=${caseCount}, maxCallerBytes=${maxCallerBytes}, maxLinearMemoryBytes=${maxLinearMemoryBytes}, manifest=${guestManifestDigest}, vectors=${guestVectorDigest}, execution=${firstExecutionDigest}/${secondExecutionDigest}`,
    );
  }
  return Object.freeze({
    schemaId: "tritium.portable_wasm_conformance_receipt",
    schemaVersion: 1,
    implementation: "wasm-fallback",
    engine: "wasm32-unknown-unknown",
    buildId,
    guestDigest: WASM_GUEST_DIGEST_V1,
    executionDigest: firstExecutionDigest,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V1,
    vectorDigest: TRAINING_VECTOR_DIGEST_V1,
    operationCount,
    caseCount,
    maxCallerBytes,
    maxLinearMemoryBytes,
    repeatedExecutions: 2,
  });
}

import init, {
  tritium_execute_portable_request_json,
  tritium_portable_build_id,
  tritium_portable_conformance_case_count,
  tritium_portable_max_caller_bytes,
  tritium_portable_manifest_digest,
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

const MAX_REQUEST_JSON_BYTES = 8 * 1024 * 1024;
const MAX_CALLER_BYTES = 64 * 1024 * 1024;
const MAX_LINEAR_MEMORY_BYTES = 192 * 1024 * 1024;
const MAX_COLLECTION_ITEMS = 64;
const MAX_NAME_BYTES = 128;
const MAX_TEXT_BYTES = 4096;
const MAX_LIST_ITEMS = 1024;

export interface PortableWasmConformanceReceiptV1 {
  readonly schemaId: "tritium.portable_wasm_conformance_receipt";
  readonly schemaVersion: 1;
  readonly implementation: "wasm-fallback";
  readonly engine: "wasm32-unknown-unknown";
  readonly buildId: string;
  readonly guestDigest: typeof WASM_GUEST_DIGEST_V1;
  readonly executionDigest: string;
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V1;
  readonly vectorDigest: typeof TRAINING_VECTOR_DIGEST_V1;
  readonly operationCount: number;
  readonly caseCount: number;
  readonly maxCallerBytes: number;
  readonly maxLinearMemoryBytes: number;
  readonly repeatedExecutions: 2;
}

export type PortableExecutionV1 =
  | "forward"
  | "vjp"
  | "step"
  | "checkpoint"
  | "resume"
  | "export"
  | "reload";

export type PortableBufferDataV1 =
  | { readonly dtype: "f32"; readonly bits: readonly number[] }
  | { readonly dtype: "u32"; readonly values: readonly number[] }
  | { readonly dtype: "bytes"; readonly values: readonly number[] };

export interface PortableBufferV1 {
  readonly name: string;
  readonly shape: readonly number[];
  readonly data: PortableBufferDataV1;
}

export type PortableAttributeV1 =
  | { readonly kind: "f32"; readonly name: string; readonly bits: number }
  /** V1 JSON transports u64 values only through Number safe integers. */
  | { readonly kind: "u64"; readonly name: string; readonly value: number }
  | { readonly kind: "bool"; readonly name: string; readonly value: boolean }
  | { readonly kind: "text"; readonly name: string; readonly value: string }
  /** Every V1 u64-list value must be a non-negative Number safe integer. */
  | { readonly kind: "u64-list"; readonly name: string; readonly values: readonly number[] }
  | { readonly kind: "u32-list"; readonly name: string; readonly values: readonly number[] };

export interface PortableTrainingRequestV1 {
  readonly schemaId: "tritium.portable_training_request";
  readonly schemaVersion: 1;
  readonly physicalDevice: string;
  readonly operation: string;
  readonly execution: PortableExecutionV1;
  readonly vectorDigest: typeof TRAINING_VECTOR_DIGEST_V1 | null;
  readonly inputs: readonly PortableBufferV1[];
  readonly attributes: readonly PortableAttributeV1[];
  readonly outputs: readonly PortableBufferV1[];
}

export interface PortableTrainingReceiptV1 {
  readonly backendId: "wasm.portable.v1";
  readonly backendBuild: string;
  readonly physicalDevice: string;
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V1;
  readonly vectorDigest: typeof TRAINING_VECTOR_DIGEST_V1 | null;
  readonly operation: string;
  readonly execution: PortableExecutionV1;
  readonly dtype: "f32" | "u32" | "bytes";
  readonly maxRank: number;
  readonly maxElements: number;
  readonly maxBytes: number;
  readonly inputDigest: string;
  readonly outputDigest: string;
  readonly peakResidentBytes: number;
  readonly scratchBytes: number;
  readonly hostTransfers: 0;
  readonly deviceResident: true;
}

export interface PortableTrainingErrorV1 {
  readonly category: string;
  readonly code: string;
  readonly message: string;
}

export type PortableTrainingResponseV1 =
  | {
      readonly status: "ok";
      readonly schemaId: "tritium.portable_training_response";
      readonly schemaVersion: 1;
      readonly outputs: readonly PortableBufferV1[];
      readonly receipt: PortableTrainingReceiptV1;
    }
  | {
      readonly status: "error";
      readonly schemaId: "tritium.portable_training_response";
      readonly schemaVersion: 1;
      readonly outputs: readonly PortableBufferV1[];
      readonly error: PortableTrainingErrorV1;
    };

let initialized: Promise<void> | null = null;

export type PortableWasmSourceV1 =
  | RequestInfo
  | URL
  | Response
  | BufferSource;

async function readGuestBytes(source: PortableWasmSourceV1): Promise<Uint8Array> {
  if (source instanceof Response) {
    return new Uint8Array(await source.arrayBuffer());
  }
  if (ArrayBuffer.isView(source)) {
    return new Uint8Array(
      source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength),
    );
  }
  if (source instanceof ArrayBuffer) return new Uint8Array(source.slice(0));
  const response = await fetch(source);
  if (!response.ok) {
    throw new Error(`portable WASM fetch failed with HTTP ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function initializeGuest(source: PortableWasmSourceV1): Promise<void> {
  const guestBytes = await readGuestBytes(source);
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
      throw error;
    }
  } else {
    await initialized;
  }
  if (
    tritium_portable_manifest_digest() !== TRAINING_MANIFEST_DIGEST_V1 ||
    tritium_portable_vector_digest() !== TRAINING_VECTOR_DIGEST_V1 ||
    tritium_portable_max_caller_bytes() !== 64 * 1024 * 1024 ||
    tritium_portable_max_linear_memory_bytes() !== 192 * 1024 * 1024
  ) {
    throw new Error("portable WASM guest identity or limits mismatch");
  }
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
  const data = value.data;
  if (data.dtype === "f32") {
    return (
      hasExactKeys(data, ["bits", "dtype"]) &&
      isDenseNumericArray(data.bits, 0xffff_ffff) &&
      data.bits.length === ("bits" in expected.data ? expected.data.bits.length : -1)
    );
  }
  const maximum = data.dtype === "u32" ? 0xffff_ffff : 0xff;
  return (
    (data.dtype === "u32" || data.dtype === "bytes") &&
    hasExactKeys(data, ["dtype", "values"]) &&
    isDenseNumericArray(data.values, maximum) &&
    data.values.length === ("values" in expected.data ? expected.data.values.length : -1)
  );
}

function sameNumericArray(value: readonly number[], expected: readonly number[]): boolean {
  return (
    value.length === expected.length &&
    value.every((item, index) => item === expected[index])
  );
}

function outputValuesMatch(value: unknown, expected: PortableBufferV1): boolean {
  if (!isRecord(value) || !isRecord(value.data)) return false;
  if (value.data.dtype === "f32" && expected.data.dtype === "f32") {
    return Array.isArray(value.data.bits) && sameNumericArray(value.data.bits, expected.data.bits);
  }
  if (value.data.dtype === "u32" && expected.data.dtype === "u32") {
    return Array.isArray(value.data.values) && sameNumericArray(value.data.values, expected.data.values);
  }
  if (value.data.dtype === "bytes" && expected.data.dtype === "bytes") {
    return Array.isArray(value.data.values) && sameNumericArray(value.data.values, expected.data.values);
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

function validateResponse(
  value: unknown,
  request: PortableTrainingRequestV1,
): PortableTrainingResponseV1 {
  if (
    !isRecord(value) ||
    !hasExactKeys(
      value,
      value.status === "ok"
        ? ["outputs", "receipt", "schemaId", "schemaVersion", "status"]
        : ["error", "outputs", "schemaId", "schemaVersion", "status"],
    ) ||
    !("status" in value) ||
    !("schemaId" in value) ||
    !("schemaVersion" in value) ||
    !("outputs" in value) ||
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
    if (
      !("receipt" in value) ||
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
      !("backendId" in value.receipt) ||
      !("manifestDigest" in value.receipt) ||
      !("vectorDigest" in value.receipt) ||
      !("operation" in value.receipt) ||
      !("execution" in value.receipt) ||
      !("physicalDevice" in value.receipt) ||
      !("inputDigest" in value.receipt) ||
      !("outputDigest" in value.receipt) ||
      !("hostTransfers" in value.receipt) ||
      !("deviceResident" in value.receipt) ||
      value.receipt.backendId !== "wasm.portable.v1" ||
      value.receipt.backendBuild !== tritium_portable_build_id() ||
      value.receipt.manifestDigest !== TRAINING_MANIFEST_DIGEST_V1 ||
      value.receipt.vectorDigest !== request.vectorDigest ||
      value.receipt.operation !== request.operation ||
      value.receipt.execution !== request.execution ||
      value.receipt.physicalDevice !== request.physicalDevice ||
      value.receipt.hostTransfers !== 0 ||
      value.receipt.deviceResident !== true ||
      value.receipt.maxRank !== 4 ||
      value.receipt.maxElements !== 8 * 1024 * 1024 ||
      value.receipt.maxBytes !== 8 * 1024 * 1024 ||
      typeof value.receipt.peakResidentBytes !== "number" ||
      value.receipt.peakResidentBytes > MAX_CALLER_BYTES ||
      typeof value.receipt.scratchBytes !== "number" ||
      value.receipt.scratchBytes > 128 * 1024 * 1024 ||
      value.receipt.peakResidentBytes + value.receipt.scratchBytes >
        MAX_LINEAR_MEMORY_BYTES ||
      !["f32", "u32", "bytes"].includes(String(value.receipt.dtype)) ||
      ![
        value.receipt.maxRank,
        value.receipt.maxElements,
        value.receipt.maxBytes,
        value.receipt.peakResidentBytes,
        value.receipt.scratchBytes,
      ].every(
        (field) =>
          typeof field === "number" &&
          Number.isSafeInteger(field) &&
          field >= 0,
      ) ||
      typeof value.receipt.inputDigest !== "string" ||
      typeof value.receipt.outputDigest !== "string" ||
      !/^[0-9a-f]{64}$/.test(value.receipt.inputDigest) ||
      !/^[0-9a-f]{64}$/.test(value.receipt.outputDigest)
    ) {
      throw new Error("portable WASM returned an invalid success receipt");
    }
  } else if (
    !("error" in value) ||
    !isRecord(value.error) ||
    !hasExactKeys(value.error, ["category", "code", "message"]) ||
    !("category" in value.error) ||
    !("code" in value.error) ||
    !("message" in value.error) ||
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
      "invalid_json",
      "unsupported_schema",
      "invalid_digest",
      "vector_digest_mismatch",
      "physical_device",
      "request_bytes",
      "missing_field",
      "unsafe_integer",
      "collection_items",
      "caller_bytes",
      "rank",
      "text_bytes",
      "list_items",
      "serialization",
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

function utf8Length(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}

function containsUnsafeNumber(value: unknown): boolean {
  if (typeof value === "number") {
    return !Number.isSafeInteger(value) || value < 0;
  }
  if (Array.isArray(value)) return value.some(containsUnsafeNumber);
  if (isRecord(value)) return Object.values(value).some(containsUnsafeNumber);
  return false;
}

function validRequestBuffer(value: unknown): value is PortableBufferV1 {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["data", "name", "shape"]) ||
    typeof value.name !== "string" ||
    utf8Length(value.name) === 0 ||
    utf8Length(value.name) > MAX_NAME_BYTES ||
    !isDenseNumericArray(value.shape, Number.MAX_SAFE_INTEGER) ||
    value.shape.length > 4 ||
    !isRecord(value.data)
  ) {
    return false;
  }
  if (value.data.dtype === "f32") {
    return (
      hasExactKeys(value.data, ["bits", "dtype"]) &&
      isDenseNumericArray(value.data.bits, 0xffff_ffff)
    );
  }
  if (value.data.dtype === "u32") {
    return (
      hasExactKeys(value.data, ["dtype", "values"]) &&
      isDenseNumericArray(value.data.values, 0xffff_ffff)
    );
  }
  return (
    value.data.dtype === "bytes" &&
    hasExactKeys(value.data, ["dtype", "values"]) &&
    isDenseNumericArray(value.data.values, 0xff)
  );
}

function validRequestAttribute(value: unknown): value is PortableAttributeV1 {
  if (
    !isRecord(value) ||
    typeof value.name !== "string" ||
    utf8Length(value.name) === 0 ||
    utf8Length(value.name) > MAX_NAME_BYTES
  ) {
    return false;
  }
  if (value.kind === "f32") {
    return (
      hasExactKeys(value, ["bits", "kind", "name"]) &&
      typeof value.bits === "number" &&
      Number.isSafeInteger(value.bits) &&
      value.bits >= 0 &&
      value.bits <= 0xffff_ffff
    );
  }
  if (value.kind === "u64") {
    return (
      hasExactKeys(value, ["kind", "name", "value"]) &&
      typeof value.value === "number" &&
      Number.isSafeInteger(value.value) &&
      value.value >= 0
    );
  }
  if (value.kind === "bool") {
    return (
      hasExactKeys(value, ["kind", "name", "value"]) &&
      typeof value.value === "boolean"
    );
  }
  if (value.kind === "text") {
    return (
      hasExactKeys(value, ["kind", "name", "value"]) &&
      typeof value.value === "string" &&
      utf8Length(value.value) <= MAX_TEXT_BYTES
    );
  }
  if (value.kind === "u64-list") {
    return (
      hasExactKeys(value, ["kind", "name", "values"]) &&
      isDenseNumericArray(value.values, Number.MAX_SAFE_INTEGER) &&
      value.values.length <= MAX_LIST_ITEMS
    );
  }
  return (
    value.kind === "u32-list" &&
    hasExactKeys(value, ["kind", "name", "values"]) &&
    isDenseNumericArray(value.values, 0xffff_ffff) &&
    value.values.length <= MAX_LIST_ITEMS
  );
}

interface RequestPreflightError {
  readonly code: string;
  readonly message: string;
}

function preflightError(code: string, message: string): RequestPreflightError {
  return { code, message };
}

function requestPreflight(value: unknown): RequestPreflightError | null {
  if (isRecord(value) && !("vectorDigest" in value)) {
    return preflightError("missing_field", "vectorDigest is required");
  }
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "attributes",
      "execution",
      "inputs",
      "operation",
      "outputs",
      "physicalDevice",
      "schemaId",
      "schemaVersion",
      "vectorDigest",
    ]) ||
    value.schemaId !== "tritium.portable_training_request" ||
    value.schemaVersion !== 1 ||
    !["forward", "vjp", "step", "checkpoint", "resume", "export", "reload"].includes(
      String(value.execution),
    ) ||
    (value.vectorDigest !== null && value.vectorDigest !== TRAINING_VECTOR_DIGEST_V1) ||
    !Array.isArray(value.inputs) ||
    !Array.isArray(value.attributes) ||
    !Array.isArray(value.outputs)
  ) {
    return preflightError(
      "invalid_schema",
      "request violates portable training schema",
    );
  }
  if (
    typeof value.physicalDevice !== "string" ||
    utf8Length(value.physicalDevice) === 0 ||
    utf8Length(value.physicalDevice) > MAX_TEXT_BYTES
  ) {
    return preflightError(
      "physical_device",
      "physicalDevice must contain 1..=4096 UTF-8 bytes",
    );
  }
  if (
    typeof value.operation !== "string" ||
    utf8Length(value.operation) === 0 ||
    utf8Length(value.operation) > MAX_NAME_BYTES
  ) {
    return preflightError(
      "invalid_name",
      "operation must contain 1..=128 UTF-8 bytes",
    );
  }
  if (
    value.inputs.length > MAX_COLLECTION_ITEMS ||
    value.attributes.length > MAX_COLLECTION_ITEMS ||
    value.outputs.length > MAX_COLLECTION_ITEMS
  ) {
    return preflightError(
      "collection_items",
      "inputs, attributes, or outputs exceed 64 items",
    );
  }
  for (const buffer of [...value.inputs, ...value.outputs]) {
    if (
      isRecord(buffer) &&
      typeof buffer.name === "string" &&
      (utf8Length(buffer.name) === 0 || utf8Length(buffer.name) > MAX_NAME_BYTES)
    ) {
      return preflightError(
        "invalid_name",
        "buffer name must contain 1..=128 UTF-8 bytes",
      );
    }
    if (!validRequestBuffer(buffer)) {
      return preflightError(
        "unsafe_integer",
        "buffer fields must use bounded JavaScript safe integers",
      );
    }
  }
  for (const attribute of value.attributes) {
    if (isRecord(attribute) && typeof attribute.name === "string") {
      if (
        utf8Length(attribute.name) === 0 ||
        utf8Length(attribute.name) > MAX_NAME_BYTES
      ) {
        return preflightError(
          "invalid_name",
          "attribute name must contain 1..=128 UTF-8 bytes",
        );
      }
      if (
        attribute.kind === "text" &&
        typeof attribute.value === "string" &&
        utf8Length(attribute.value) > MAX_TEXT_BYTES
      ) {
        return preflightError(
          "text_bytes",
          "text attribute exceeds 4096 UTF-8 bytes",
        );
      }
      if (
        (attribute.kind === "u64-list" || attribute.kind === "u32-list") &&
        Array.isArray(attribute.values) &&
        attribute.values.length > MAX_LIST_ITEMS
      ) {
        return preflightError(
          "list_items",
          "attribute list exceeds 1024 items",
        );
      }
    }
    if (!validRequestAttribute(attribute)) {
      return preflightError(
        "unsafe_integer",
        "attribute fields must use bounded JavaScript safe integers",
      );
    }
  }
  let callerBytes = 0;
  for (const buffer of [...value.inputs, ...value.outputs]) {
    const elements =
      "bits" in buffer.data ? buffer.data.bits.length : buffer.data.values.length;
    callerBytes += buffer.data.dtype === "bytes" ? elements : elements * 4;
  }
  return callerBytes <= MAX_CALLER_BYTES
    ? null
    : preflightError(
        "caller_bytes",
        `decoded caller buffers exceed ${MAX_CALLER_BYTES} bytes`,
      );
}

/** Execute one strict request through Rust-owned WASM semantics. */
export async function executePortableWasmRequest(
  request: PortableTrainingRequestV1,
  source: PortableWasmSourceV1 = new URL(
    "./tritium_wasm_bg.wasm",
    import.meta.url,
  ),
): Promise<PortableTrainingResponseV1> {
  let requestJson: string;
  let requestSnapshot: PortableTrainingRequestV1;
  try {
    requestJson = JSON.stringify(request);
    if (requestJson === undefined) throw new Error("undefined JSON result");
    requestSnapshot = deepFreeze(
      JSON.parse(requestJson) as PortableTrainingRequestV1,
    );
  } catch {
    return localError(
      "invalid_json",
      "portable WASM request is not JSON serializable",
    );
  }
  const requestBytes = new TextEncoder().encode(requestJson).byteLength;
  if (requestBytes > MAX_REQUEST_JSON_BYTES) {
    return localError(
      "request_bytes",
      "portable WASM request JSON exceeds 8 MiB",
      "capacity",
    );
  }
  if (containsUnsafeNumber(requestSnapshot)) {
    return localError(
      "unsafe_integer",
      "portable WASM numeric fields must use non-negative JavaScript safe integers",
    );
  }
  const preflightError = requestPreflight(requestSnapshot);
  if (preflightError !== null) {
    const category = [
      "caller_bytes",
      "collection_items",
      "list_items",
      "rank",
      "text_bytes",
    ].includes(preflightError.code)
      ? "capacity"
      : "invalid_request";
    return localError(preflightError.code, preflightError.message, category);
  }
  await initializeGuest(source);
  let responseJson: string;
  try {
    responseJson = tritium_execute_portable_request_json(requestJson);
  } catch {
    return localError(
      "guest_trap",
      "portable WASM guest trapped",
      "internal",
    );
  }
  return validateResponse(
    JSON.parse(responseJson) as unknown,
    requestSnapshot,
  );
}

/** Execute the complete canonical vector corpus twice inside the bundled guest. */
export async function runPortableWasmConformance(
  source: PortableWasmSourceV1 = new URL(
    "./tritium_wasm_bg.wasm",
    import.meta.url,
  ),
): Promise<PortableWasmConformanceReceiptV1> {
  await initializeGuest(source);
  const firstExecutionDigest = tritium_portable_report_digest();
  const secondExecutionDigest = tritium_portable_report_digest();
  const operationCount = tritium_portable_operation_count();
  const caseCount = tritium_portable_conformance_case_count();
  const maxCallerBytes = tritium_portable_max_caller_bytes();
  const maxLinearMemoryBytes = tritium_portable_max_linear_memory_bytes();
  const guestManifestDigest = tritium_portable_manifest_digest();
  const guestVectorDigest = tritium_portable_vector_digest();
  if (
    operationCount !== 35 ||
    caseCount !== 114 ||
    maxCallerBytes !== 64 * 1024 * 1024 ||
    maxLinearMemoryBytes !== 192 * 1024 * 1024 ||
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
    buildId: tritium_portable_build_id(),
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

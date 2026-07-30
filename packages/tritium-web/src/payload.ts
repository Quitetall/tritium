import { blake3 } from "@noble/hashes/blake3.js";

import { TRAINING_MANIFEST_DIGEST_V2 } from "./identity.ts";
import type {
  WebTrainingInitialTensorsV1,
  WebTrainingPayloadErrorCode,
} from "./payload-types.js";
import type {
  PortableScheduleTensorStoreV1,
  PortableScheduleTensorV1,
} from "./portable-schedule-types.js";
import {
  PortableSchedulePlanError,
  admittedCompiledBufferMap,
} from "./portable-schedule.ts";
import type {
  CompiledTrainingBufferV1,
  CompiledTrainingPlanV1,
} from "./session.ts";

export type {
  WebTrainingInitialTensorsV1,
  WebTrainingPayloadErrorCode,
} from "./payload-types.js";

const MAGIC = Uint8Array.of(0x54, 0x52, 0x57, 0x45, 0x42, 0x50, 0x31, 0x00);
const HEADER_BYTES = 56;
const ENTRY_HEADER_BYTES = 8;
const MAX_PAYLOAD_BYTES = 64 * 1024 * 1024;
const UTF8 = new TextEncoder();
const UTF8_FATAL = new TextDecoder("utf-8", { fatal: true });

type DType = CompiledTrainingBufferV1["dtype"];
type EncodedEntry = {
  readonly name: string;
  readonly nameBytes: Uint8Array;
  readonly dtype: DType;
  readonly data: Uint8Array;
};

export class WebTrainingPayloadError extends Error {
  readonly code: WebTrainingPayloadErrorCode;

  constructor(code: WebTrainingPayloadErrorCode, message: string) {
    super(message);
    this.name = "WebTrainingPayloadError";
    this.code = code;
  }
}

function fail(code: WebTrainingPayloadErrorCode, message: string): never {
  throw new WebTrainingPayloadError(code, message);
}

function byteOrder(left: Uint8Array, right: Uint8Array): number {
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    const difference = left[index]! - right[index]!;
    if (difference !== 0) return difference;
  }
  return left.length - right.length;
}

function exactNameBytes(name: string): Uint8Array {
  if (name.length === 0) fail("invalid_schema", "tensor name must be nonempty");
  const encoded = UTF8.encode(name);
  if (encoded.length === 0 || encoded.length > 0xffff) {
    fail("capacity", `tensor name ${name} exceeds payload V1 capacity`);
  }
  if (UTF8_FATAL.decode(encoded) !== name) {
    fail("invalid_schema", `tensor name ${name} is not canonical UTF-8`);
  }
  return encoded;
}

function encodeTensor(tensor: PortableScheduleTensorV1): Readonly<{ dtype: DType; data: Uint8Array }> {
  if (tensor instanceof Float32Array) {
    const data = new Uint8Array(tensor.byteLength);
    const output = new DataView(data.buffer);
    const lanes = new Uint32Array(tensor.buffer, tensor.byteOffset, tensor.length);
    lanes.forEach((value, index) => output.setUint32(index * 4, value, true));
    return { dtype: "f32", data };
  }
  if (tensor instanceof Uint32Array) {
    const data = new Uint8Array(tensor.byteLength);
    const output = new DataView(data.buffer);
    tensor.forEach((value, index) => output.setUint32(index * 4, value, true));
    return { dtype: "u32", data };
  }
  if (tensor instanceof Uint8Array) {
    return { dtype: "bytes", data: Uint8Array.from(tensor) };
  }
  fail("invalid_schema", "initial tensor must be Float32Array, Uint32Array, or Uint8Array");
}

function dtypeCode(dtype: DType): number {
  return dtype === "f32" ? 0 : dtype === "u32" ? 1 : 2;
}

function codeDtype(code: number): DType {
  if (code === 0) return "f32";
  if (code === 1) return "u32";
  if (code === 2) return "bytes";
  fail("invalid_schema", `unknown payload dtype ${code}`);
}

function checkedSize(current: number, addition: number): number {
  const result = current + addition;
  if (!Number.isSafeInteger(result) || result > MAX_PAYLOAD_BYTES) {
    fail("capacity", "web training payload exceeds 64 MiB");
  }
  return result;
}

/** Encode canonical, checksummed root parameter and optimizer-state bytes. */
export function encodeWebTrainingPayload(
  tensors: WebTrainingInitialTensorsV1,
): Uint8Array {
  if (typeof tensors !== "object" || tensors === null || Array.isArray(tensors)) {
    fail("invalid_schema", "initial tensors must be an object");
  }
  if (Reflect.ownKeys(tensors).some((key) => typeof key !== "string")) {
    fail("invalid_schema", "initial tensors cannot contain symbol keys");
  }
  const entries: EncodedEntry[] = Object.keys(tensors).map((name) => {
    const encoded = encodeTensor(tensors[name]!);
    return { name, nameBytes: exactNameBytes(name), ...encoded };
  });
  if (entries.length === 0) fail("invalid_schema", "initial tensors cannot be empty");
  entries.sort((left, right) => byteOrder(left.nameBytes, right.nameBytes));

  let bodyBytes = 0;
  for (const entry of entries) {
    bodyBytes = checkedSize(bodyBytes, ENTRY_HEADER_BYTES);
    bodyBytes = checkedSize(bodyBytes, entry.nameBytes.length);
    bodyBytes = checkedSize(bodyBytes, entry.data.length);
  }
  const body = new Uint8Array(bodyBytes);
  const view = new DataView(body.buffer);
  let offset = 0;
  for (const entry of entries) {
    view.setUint16(offset, entry.nameBytes.length, true);
    view.setUint8(offset + 2, dtypeCode(entry.dtype));
    view.setUint8(offset + 3, 0);
    view.setUint32(offset + 4, entry.data.length, true);
    offset += ENTRY_HEADER_BYTES;
    body.set(entry.nameBytes, offset);
    offset += entry.nameBytes.length;
    body.set(entry.data, offset);
    offset += entry.data.length;
  }

  const outputBytes = checkedSize(HEADER_BYTES, body.length);
  const output = new Uint8Array(outputBytes);
  const header = new DataView(output.buffer);
  output.set(MAGIC, 0);
  header.setUint32(8, 1, true);
  header.setUint32(12, entries.length, true);
  header.setUint32(16, body.length, true);
  header.setUint32(20, 0, true);
  output.set(blake3(body), 24);
  output.set(body, HEADER_BYTES);
  return output;
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.length !== right.length) return false;
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) {
    difference |= left[index]! ^ right[index]!;
  }
  return difference === 0;
}

function decodeTensor(dtype: DType, bytes: Uint8Array): PortableScheduleTensorV1 {
  if (dtype === "bytes") return Uint8Array.from(bytes);
  if (bytes.length % 4 !== 0) fail("invalid_schema", `${dtype} payload is not lane-aligned`);
  const lanes = new Uint32Array(bytes.length / 4);
  const source = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < lanes.length; index += 1) {
    lanes[index] = source.getUint32(index * 4, true);
  }
  return dtype === "u32" ? lanes : new Float32Array(lanes.buffer);
}

function parsePayload(payload: Uint8Array): ReadonlyMap<string, Readonly<{ dtype: DType; tensor: PortableScheduleTensorV1 }>> {
  if (!(payload instanceof Uint8Array)) fail("invalid_schema", "payload must be Uint8Array");
  if (payload.length < HEADER_BYTES || payload.length > MAX_PAYLOAD_BYTES) {
    fail("capacity", "payload length is outside V1 bounds");
  }
  const source = Uint8Array.from(payload);
  if (!equalBytes(source.subarray(0, MAGIC.length), MAGIC)) {
    fail("invalid_schema", "payload magic is invalid");
  }
  const header = new DataView(source.buffer);
  const count = header.getUint32(12, true);
  const bodyLength = header.getUint32(16, true);
  if (
    header.getUint32(8, true) !== 1 ||
    header.getUint32(20, true) !== 0 ||
    count === 0 ||
    bodyLength !== source.length - HEADER_BYTES
  ) {
    fail("invalid_schema", "payload header is invalid");
  }
  const body = source.subarray(HEADER_BYTES);
  if (!equalBytes(source.subarray(24, 56), blake3(body))) {
    fail("integrity", "payload body digest mismatch");
  }

  const entries = new Map<string, Readonly<{ dtype: DType; tensor: PortableScheduleTensorV1 }>>();
  const view = new DataView(body.buffer, body.byteOffset, body.byteLength);
  let offset = 0;
  let previousName: Uint8Array | null = null;
  for (let index = 0; index < count; index += 1) {
    if (offset + ENTRY_HEADER_BYTES > body.length) fail("invalid_schema", "payload entry header is truncated");
    const nameLength = view.getUint16(offset, true);
    const dtype = codeDtype(view.getUint8(offset + 2));
    const flags = view.getUint8(offset + 3);
    const dataLength = view.getUint32(offset + 4, true);
    offset += ENTRY_HEADER_BYTES;
    if (nameLength === 0 || flags !== 0 || offset + nameLength + dataLength > body.length) {
      fail("invalid_schema", "payload entry bounds are invalid");
    }
    const nameBytes = body.slice(offset, offset + nameLength);
    offset += nameLength;
    let name: string;
    try {
      name = UTF8_FATAL.decode(nameBytes);
    } catch {
      fail("invalid_schema", "payload tensor name is invalid UTF-8");
    }
    if (!equalBytes(UTF8.encode(name), nameBytes)) fail("invalid_schema", "payload tensor name is not canonical UTF-8");
    if (previousName !== null && byteOrder(previousName, nameBytes) >= 0) {
      fail("invalid_schema", "payload tensor names are not strictly ordered");
    }
    previousName = nameBytes;
    const tensorBytes = body.subarray(offset, offset + dataLength);
    offset += dataLength;
    entries.set(name, Object.freeze({ dtype, tensor: decodeTensor(dtype, tensorBytes) }));
  }
  if (offset !== body.length || entries.size !== count) {
    fail("invalid_schema", "payload contains trailing or duplicate entries");
  }
  return entries;
}

function emptyTensor(buffer: CompiledTrainingBufferV1): PortableScheduleTensorV1 {
  const elements = buffer.dtype === "bytes" ? buffer.byteLength : buffer.byteLength / 4;
  if (buffer.dtype === "f32") return new Float32Array(elements);
  if (buffer.dtype === "u32") return new Uint32Array(elements);
  return new Uint8Array(elements);
}

/** Verify payload against plan, then materialize one mutable tensor per owner. */
export function decodeWebTrainingPayload(
  plan: CompiledTrainingPlanV1,
  payload: Uint8Array,
): PortableScheduleTensorStoreV1 {
  let admitted: ReadonlyMap<string, CompiledTrainingBufferV1>;
  try {
    admitted = admittedCompiledBufferMap(plan);
  } catch (error) {
    if (error instanceof PortableSchedulePlanError) {
      fail(error.code === "capacity" ? "capacity" : "invalid_schema", error.message);
    }
    throw error;
  }
  const entries = parsePayload(payload);
  const owners = new Map<string, CompiledTrainingBufferV1>();
  for (const buffer of admitted.values()) {
    if (buffer.ownerId === buffer.id) owners.set(buffer.id, buffer);
  }
  const expected = new Set(
    [...owners.values()]
      .filter((buffer) => buffer.role === "parameter" || buffer.role === "optimizer-state")
      .map((buffer) => buffer.id),
  );
  for (const name of entries.keys()) {
    if (!expected.has(name)) fail("buffer_mismatch", `payload contains unexpected tensor ${name}`);
  }
  for (const name of expected) {
    if (!entries.has(name)) fail("missing_buffer", `payload omits persistent tensor ${name}`);
  }

  const store = Object.create(null) as Record<string, PortableScheduleTensorV1>;
  for (const [id, buffer] of owners) {
    const entry = entries.get(id);
    const tensor = entry?.tensor ?? emptyTensor(buffer);
    if (entry !== undefined && (entry.dtype !== buffer.dtype || tensor.byteLength !== buffer.byteLength)) {
      fail("buffer_mismatch", `payload tensor ${id} differs from compiled dtype/shape`);
    }
    if (entry === undefined && buffer.backwardInitialization === "one") tensor.fill(1);
    store[id] = tensor;
  }
  return Object.freeze(store);
}

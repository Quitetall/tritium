import { blake3 } from "@noble/hashes/blake3.js";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";

import {
  TRAINING_MANIFEST_DIGEST_V2,
  TRAINING_VECTOR_DIGEST_V2,
} from "./identity.ts";
import type {
  PortableAttributeV1,
  PortableBufferV1,
  PortableWasmSourceV1,
} from "./portable.js";
import type { CompiledTrainingPlanV1 } from "./session.ts";
import { WebTrainingError } from "./session.ts";
import { preparePortableWasmExecutor } from "./wasm.ts";
import type {
  WebGpuDevicePortV1,
  WebGpuResidentTensorV1,
} from "./webgpu-runtime.ts";
import { WebGpuResidentRuntimeV1 } from "./webgpu-runtime.ts";
import { compileWebGpuResidentScheduleV1 } from "./webgpu-schedule.ts";

type VectorData =
  | Readonly<{ dtype: "f32"; bits: readonly number[] }>
  | Readonly<{ dtype: "u32"; values: readonly number[] }>
  | Readonly<{ dtype: "bytes"; values: readonly number[] }>;

type VectorBuffer = Readonly<{
  name: string;
  shape: readonly number[];
  data: VectorData;
}>;

type VectorAttribute = Readonly<{
  type: "f32" | "u64" | "bool" | "text" | "u64_list" | "u32_list";
  name: string;
  bits?: number;
  value?: number | boolean | string;
  values?: readonly number[];
}>;

type VectorTolerance =
  | Readonly<{ kind: "bit_exact" }>
  | Readonly<{
    kind: "absolute_relative";
    absolute_bits: number;
    relative_bits: number;
  }>;

type VectorExpected =
  | Readonly<{
    kind: "success";
    outputs: readonly VectorBuffer[];
    scratch_bytes_max: number;
  }>
  | Readonly<{
    kind: "error";
    category: string;
    code: string;
    outputs: readonly VectorBuffer[];
  }>;

type VectorCase = Readonly<{
  case_id: string;
  operation: string;
  execution: "forward" | "vjp" | "step" | "checkpoint" | "resume" | "export" | "reload";
  tolerance: VectorTolerance;
  inputs: readonly VectorBuffer[];
  attributes: readonly VectorAttribute[];
  expected: VectorExpected;
}>;

type SuccessfulVectorCase = VectorCase & Readonly<{
  expected: Extract<VectorExpected, Readonly<{ kind: "success" }>>;
}>;

type VectorCorpus = Readonly<{
  schema_id: string;
  schema_version: number;
  manifest_digest: string;
  cases: readonly VectorCase[];
}>;

export type WebGpuVectorConformanceInventoryV1 = Readonly<{
  schemaId: "tritium.webgpu_vector_conformance_inventory";
  schemaVersion: 1;
  manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V2;
  vectorDigest: typeof TRAINING_VECTOR_DIGEST_V2;
  caseCounts: Readonly<{
    valid: 72;
    invalid: 45;
    compute: 68;
    lifecycle: 4;
    total: 117;
  }>;
}>;

export type WebGpuVectorCaseTraceV1 = Readonly<{
  caseId: string;
  implementation: "webgpu" | "wasm-codec" | "wasm-validation";
  outputDigest: string;
  scratchBytes: number | null;
  scratchBytesMax: number | null;
}>;

export type WebGpuVectorConformanceTraceV1 = Readonly<{
  schemaId: "tritium.webgpu_vector_conformance_trace";
  schemaVersion: 1;
  implementation: "webgpu";
  manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V2;
  vectorDigest: typeof TRAINING_VECTOR_DIGEST_V2;
  caseCounts: Readonly<{ valid: 72; invalid: 45; skipped: 0 }>;
  webgpuCaseTransactions: 68;
  webgpuDispatches: number;
  wasmDispatches: 0;
  wasmCodecCalls: 4;
  wasmValidationCalls: 45;
  explicitReadbacks: number;
  peakBufferBytes: number;
  executionDigest: string;
  cases: readonly WebGpuVectorCaseTraceV1[];
}>;

export type WebGpuVectorConformanceOptionsV1 = Readonly<{
  wasmSource?: PortableWasmSourceV1;
  maxPeakBytes?: number;
  physicalDevice?: string;
}>;

type ComputeEntry = Readonly<{
  item: SuccessfulVectorCase;
  phase: "forward" | "backward";
  operationId: string;
  outputIds: readonly string[];
  optimizerStep: number | undefined;
  scratchBytes: number;
  scratchBytesMax: number;
}>;

const UTF8 = new TextEncoder();
declare const __TRITIUM_TRAINING_VECTORS_V2_JSON__: string;
const RAW_CORPUS_JSON = __TRITIUM_TRAINING_VECTORS_V2_JSON__;
const corpus = JSON.parse(RAW_CORPUS_JSON) as unknown as VectorCorpus;
const CANONICAL_COUNTS = Object.freeze({
  valid: 72 as const,
  invalid: 45 as const,
  compute: 68 as const,
  lifecycle: 4 as const,
  total: 117 as const,
});

function fail(message: string): never {
  throw new WebTrainingError("invalid_schema", `WebGPU conformance ${message}`);
}

function denseArray(value: unknown): value is readonly unknown[] {
  if (!Array.isArray(value)) return false;
  for (let index = 0; index < value.length; index += 1) {
    if (!(index in value)) return false;
  }
  return true;
}

function record(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateCorpus(): WebGpuVectorConformanceInventoryV1 {
  if (
    bytesToHex(blake3(UTF8.encode(RAW_CORPUS_JSON))) !== TRAINING_VECTOR_DIGEST_V2 ||
    !record(corpus) ||
    corpus.schema_id !== "tritium.training_vectors" ||
    corpus.schema_version !== 2 ||
    corpus.manifest_digest !== TRAINING_MANIFEST_DIGEST_V2 ||
    !denseArray(corpus.cases) ||
    corpus.cases.length !== CANONICAL_COUNTS.total
  ) {
    fail("embedded vector corpus identity is invalid");
  }
  const ids = new Set<string>();
  let valid = 0;
  let invalid = 0;
  let compute = 0;
  let lifecycle = 0;
  for (const item of corpus.cases) {
    if (
      !record(item) ||
      typeof item.case_id !== "string" ||
      item.case_id.length === 0 ||
      ids.has(item.case_id) ||
      typeof item.operation !== "string" ||
      !denseArray(item.inputs) ||
      !denseArray(item.attributes) ||
      !record(item.expected) ||
      (item.expected.kind !== "success" && item.expected.kind !== "error")
    ) {
      fail("embedded vector corpus structure is invalid");
    }
    ids.add(item.case_id);
    if (item.expected.kind === "success") {
      valid += 1;
      if (item.operation.startsWith("lifecycle.")) lifecycle += 1;
      else compute += 1;
    } else {
      invalid += 1;
    }
  }
  if (
    valid !== CANONICAL_COUNTS.valid ||
    invalid !== CANONICAL_COUNTS.invalid ||
    compute !== CANONICAL_COUNTS.compute ||
    lifecycle !== CANONICAL_COUNTS.lifecycle
  ) {
    fail("embedded vector corpus case counts drifted");
  }
  return Object.freeze({
    schemaId: "tritium.webgpu_vector_conformance_inventory",
    schemaVersion: 1,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V2,
    vectorDigest: TRAINING_VECTOR_DIGEST_V2,
    caseCounts: CANONICAL_COUNTS,
  });
}

const INVENTORY = validateCorpus();

/** Return exact identities and counts for source-free lane preflight. */
export function webGpuVectorConformanceInventoryV1(): WebGpuVectorConformanceInventoryV1 {
  return INVENTORY;
}

function snapshotOptions(
  value: WebGpuVectorConformanceOptionsV1 | undefined,
): Required<Pick<WebGpuVectorConformanceOptionsV1, "maxPeakBytes" | "physicalDevice">> &
  Pick<WebGpuVectorConformanceOptionsV1, "wasmSource"> {
  if (value === undefined) {
    return Object.freeze({
      maxPeakBytes: 64 * 1024 * 1024,
      physicalDevice: "browser:webgpu-conformance",
    });
  }
  if (!record(value)) fail("options must be an object");
  const keys = Reflect.ownKeys(value);
  if (
    keys.some((key) => typeof key !== "string") ||
    (keys as readonly string[]).some((key) =>
      !["maxPeakBytes", "physicalDevice", "wasmSource"].includes(key)
    )
  ) {
    fail("options contain an unknown field");
  }
  const maxPeakBytes = value.maxPeakBytes ?? 64 * 1024 * 1024;
  const physicalDevice = value.physicalDevice ?? "browser:webgpu-conformance";
  if (!Number.isSafeInteger(maxPeakBytes) || maxPeakBytes <= 0) {
    fail("maxPeakBytes must be a positive safe integer");
  }
  if (typeof physicalDevice !== "string" || physicalDevice.length === 0) {
    fail("physicalDevice must be a nonempty string");
  }
  return value.wasmSource === undefined
    ? Object.freeze({ maxPeakBytes, physicalDevice })
    : Object.freeze({ maxPeakBytes, physicalDevice, wasmSource: value.wasmSource });
}

function validateDevice(device: unknown): asserts device is WebGpuDevicePortV1 {
  if (!record(device) || !record(device.limits) || !record(device.queue)) {
    fail("device and limits are required");
  }
  for (const method of [
    "createShaderModule",
    "createComputePipelineAsync",
    "createBuffer",
    "createBindGroup",
    "createCommandEncoder",
    "destroy",
  ]) {
    if (typeof device[method] !== "function") fail(`device.${method} is required`);
  }
  for (const method of ["writeBuffer", "submit", "onSubmittedWorkDone"]) {
    if (typeof device.queue[method] !== "function") fail(`device.queue.${method} is required`);
  }
  // Firefox exposes GPUDevice.lost as a cross-realm thenable rather than an
  // instanceof-our-global-Promise. Awaitability is the WebGPU contract; an
  // instanceof check rejects valid browser implementations.
  const lost = device.lost as unknown;
  if (!record(lost) || typeof lost.then !== "function") {
    fail("device.lost must be a Promise-like thenable");
  }
}

function align16(value: number): number {
  return Math.ceil(value / 16) * 16;
}

function product(shape: readonly number[], label: string): number {
  let value = 1;
  for (const dimension of shape) {
    if (!Number.isSafeInteger(dimension) || dimension < 0 || value > Number.MAX_SAFE_INTEGER / Math.max(1, dimension)) {
      fail(`${label} shape is invalid`);
    }
    value *= dimension;
  }
  return value;
}

function checkedProduct(values: readonly number[], label: string): number {
  return product(values, label);
}

function checkedSum(values: readonly number[], label: string): number {
  let total = 0;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0 || total > Number.MAX_SAFE_INTEGER - value) {
      fail(`${label} sum is invalid`);
    }
    total += value;
  }
  return total;
}

function u64Attribute(item: VectorCase, name: string): number {
  const attribute = item.attributes.find((candidate) => candidate.name === name);
  if (
    attribute?.type !== "u64" ||
    typeof attribute.value !== "number" ||
    !Number.isSafeInteger(attribute.value) ||
    attribute.value < 0
  ) {
    fail(`${item.case_id} ${name} attribute is invalid`);
  }
  return attribute.value;
}

function inputElements(item: VectorCase, name: string): number {
  const buffer = item.inputs.find((candidate) => candidate.name === name);
  if (buffer === undefined) fail(`${item.case_id} ${name} input is missing`);
  return checkedProduct(buffer.shape, `${item.case_id} ${name}`);
}

function convolutionOutput(
  input: number,
  kernel: number,
  stride: number,
  dilation: number,
  before: number,
  after: number,
  label: string,
): number {
  const effective = checkedSum([
    checkedProduct([dilation, kernel - 1], label),
    1,
  ], label);
  const padded = checkedSum([input, before, after], label);
  if (stride <= 0 || kernel <= 0 || padded < effective) fail(`${label} geometry is invalid`);
  return Math.floor((padded - effective) / stride) + 1;
}

/** Mirror frozen backend scratch semantics, excluding immutable constants and commit candidates. */
function semanticScratchBytes(item: VectorCase): number {
  if (item.expected.kind !== "success") return 0;
  if (item.operation === "graph.salt_ste") {
    return item.execution === "forward"
      ? checkedProduct([u64Attribute(item, "cols"), 4], `${item.case_id} scratch`)
      : 0;
  }
  if (item.operation === "graph.attention") {
    const seq = u64Attribute(item, "seq");
    const query = checkedProduct([
      seq,
      u64Attribute(item, "n_head"),
      u64Attribute(item, "head_dim"),
    ], `${item.case_id} query`);
    const keyValue = checkedProduct([
      seq,
      u64Attribute(item, "n_kv_head"),
      u64Attribute(item, "head_dim"),
    ], `${item.case_id} key/value`);
    const scores = checkedProduct([seq, seq], `${item.case_id} scores`);
    const elements = item.execution === "vjp"
      ? checkedSum(
        [query, keyValue, keyValue, scores, scores],
        `${item.case_id} scratch`,
      )
      : checkedSum([query, scores], `${item.case_id} scratch`);
    return checkedProduct([elements, 4], `${item.case_id} scratch bytes`);
  }
  if (item.operation === "graph.conv1d") {
    const batch = u64Attribute(item, "batch");
    const cIn = u64Attribute(item, "c_in");
    const cOut = u64Attribute(item, "c_out");
    const inputLength = u64Attribute(item, "l_in");
    const kernel = u64Attribute(item, "k");
    const groups = u64Attribute(item, "groups");
    const outputLength = convolutionOutput(
      inputLength,
      kernel,
      u64Attribute(item, "stride"),
      u64Attribute(item, "dilation"),
      u64Attribute(item, "pad_left"),
      u64Attribute(item, "pad_right"),
      `${item.case_id} output`,
    );
    const input = checkedProduct([batch, cIn, inputLength], `${item.case_id} input`);
    const patchColumns = checkedProduct([cIn / groups, kernel], `${item.case_id} patch`);
    const weight = checkedProduct([cOut, patchColumns], `${item.case_id} weight`);
    const columns = checkedProduct([outputLength, patchColumns], `${item.case_id} columns`);
    const groupOutput = checkedProduct(
      [outputLength, cOut / groups],
      `${item.case_id} group output`,
    );
    const elements = item.execution === "forward"
      ? checkedSum([
        checkedProduct([batch, cOut, outputLength], `${item.case_id} result`),
        columns,
        groupOutput,
      ], `${item.case_id} scratch`)
      : checkedSum([
        input,
        weight,
        cOut,
        columns,
        groupOutput,
        columns,
        weight / groups,
        cOut / groups,
      ], `${item.case_id} scratch`);
    return checkedProduct([elements, 4], `${item.case_id} scratch bytes`);
  }
  if (item.operation === "graph.conv2d") {
    const batch = u64Attribute(item, "batch");
    const cIn = u64Attribute(item, "c_in");
    const cOut = u64Attribute(item, "c_out");
    const inputHeight = u64Attribute(item, "input_h");
    const inputWidth = u64Attribute(item, "input_w");
    const kernelHeight = u64Attribute(item, "kernel_h");
    const kernelWidth = u64Attribute(item, "kernel_w");
    const groups = u64Attribute(item, "groups");
    const outputHeight = convolutionOutput(
      inputHeight,
      kernelHeight,
      u64Attribute(item, "stride_h"),
      u64Attribute(item, "dilation_h"),
      u64Attribute(item, "pad_top"),
      u64Attribute(item, "pad_bottom"),
      `${item.case_id} output height`,
    );
    const outputWidth = convolutionOutput(
      inputWidth,
      kernelWidth,
      u64Attribute(item, "stride_w"),
      u64Attribute(item, "dilation_w"),
      u64Attribute(item, "pad_left"),
      u64Attribute(item, "pad_right"),
      `${item.case_id} output width`,
    );
    const tileRows = Math.min(32, checkedProduct(
      [outputHeight, outputWidth],
      `${item.case_id} tile rows`,
    ));
    const patchColumns = checkedProduct(
      [cIn / groups, kernelHeight, kernelWidth],
      `${item.case_id} patch`,
    );
    const groupChannels = cOut / groups;
    const columns = checkedProduct([tileRows, patchColumns], `${item.case_id} columns`);
    const groupOutput = checkedProduct(
      [tileRows, groupChannels],
      `${item.case_id} group output`,
    );
    const output = checkedProduct(
      [batch, cOut, outputHeight, outputWidth],
      `${item.case_id} output`,
    );
    const input = checkedProduct(
      [batch, cIn, inputHeight, inputWidth],
      `${item.case_id} input`,
    );
    const weight = checkedProduct([cOut, patchColumns], `${item.case_id} weight`);
    const elements = item.execution === "forward"
      ? checkedSum([output, columns, groupOutput], `${item.case_id} scratch`)
      : checkedSum([
        input,
        weight,
        cOut,
        columns,
        groupOutput,
        columns,
        checkedProduct([groupChannels, patchColumns], `${item.case_id} group weight`),
        groupChannels,
      ], `${item.case_id} scratch`);
    return checkedProduct([elements, 4], `${item.case_id} scratch bytes`);
  }
  if (item.operation === "optimizer.adamw") {
    return checkedProduct(
      [inputElements(item, "parameter"), 2, 4],
      `${item.case_id} scratch bytes`,
    );
  }
  if (item.operation === "optimizer.cautious_adamw") {
    return checkedProduct(
      [inputElements(item, "parameter"), 3, 4],
      `${item.case_id} scratch bytes`,
    );
  }
  if (item.operation === "optimizer.int8_adamw") {
    const length = inputElements(item, "parameter");
    const blocks = Math.ceil(length / 256);
    const stateBytes = checkedSum([
      checkedProduct([length, 2], `${item.case_id} compact state`),
      checkedProduct([blocks, 8], `${item.case_id} scales`),
    ], `${item.case_id} state`);
    const blockBytes = checkedProduct(
      [Math.min(length, 256), 2, 4],
      `${item.case_id} block workspace`,
    );
    return checkedSum([stateBytes, blockBytes], `${item.case_id} scratch bytes`);
  }
  if (item.operation === "optimizer.muon") {
    const rows = u64Attribute(item, "rows");
    const cols = u64Attribute(item, "cols");
    const matrix = checkedProduct([rows, cols], `${item.case_id} matrix`);
    const gramAxis = Math.min(rows, cols);
    const gram = checkedProduct([gramAxis, gramAxis], `${item.case_id} gram`);
    const elements = checkedSum([
      checkedProduct([matrix, 4], `${item.case_id} matrix workspace`),
      checkedProduct([gram, 3], `${item.case_id} gram workspace`),
    ], `${item.case_id} workspace`);
    return checkedProduct([elements, 4], `${item.case_id} scratch bytes`);
  }
  return 0;
}

function admitScratch(item: VectorCase, scratchBytes: number): number {
  if (
    item.expected.kind !== "success" ||
    !Number.isSafeInteger(item.expected.scratch_bytes_max) ||
    item.expected.scratch_bytes_max < 0 ||
    !Number.isSafeInteger(scratchBytes) ||
    scratchBytes < 0 ||
    scratchBytes > item.expected.scratch_bytes_max
  ) {
    fail(`${item.case_id} scratch exceeds the canonical ceiling`);
  }
  return scratchBytes;
}

function f32FromBits(bits: number): number {
  const bytes = new ArrayBuffer(4);
  const view = new DataView(bytes);
  view.setUint32(0, bits, true);
  return view.getFloat32(0, true);
}

function portableAttribute(attribute: VectorAttribute): PortableAttributeV1 {
  switch (attribute.type) {
    case "f32":
      if (!Number.isSafeInteger(attribute.bits)) fail("f32 attribute bits are invalid");
      return Object.freeze({
        kind: "f32",
        name: attribute.name,
        bits: attribute.bits as number,
      });
    case "u64":
      if (!Number.isSafeInteger(attribute.value) || (attribute.value as number) < 0) {
        fail("u64 attribute is invalid");
      }
      return Object.freeze({ kind: "u64", name: attribute.name, value: attribute.value as number });
    case "bool":
      if (typeof attribute.value !== "boolean") fail("bool attribute is invalid");
      return Object.freeze({ kind: "bool", name: attribute.name, value: attribute.value });
    case "text":
      if (typeof attribute.value !== "string") fail("text attribute is invalid");
      return Object.freeze({ kind: "text", name: attribute.name, value: attribute.value });
    case "u64_list":
      if (!denseArray(attribute.values)) fail("u64-list attribute is invalid");
      return Object.freeze({ kind: "u64-list", name: attribute.name, values: [...attribute.values] });
    case "u32_list":
      if (!denseArray(attribute.values)) fail("u32-list attribute is invalid");
      return Object.freeze({ kind: "u32-list", name: attribute.name, values: [...attribute.values] });
  }
}

function planAttribute(attribute: VectorAttribute, execution: VectorCase["execution"]): Readonly<{
  name: string;
  kind: "f32" | "u64" | "bool" | "text" | "u64-list" | "u32-list";
  value: number | boolean | string | readonly number[];
}> {
  const portable = portableAttribute(attribute);
  if (portable.kind === "f32") {
    return Object.freeze({
      name: portable.name,
      kind: portable.kind,
      value: f32FromBits(portable.bits),
    });
  }
  if (portable.kind === "u64" && portable.name === "step" && execution === "step") {
    return Object.freeze({ name: portable.name, kind: portable.kind, value: 0 });
  }
  if (portable.kind === "u64-list" || portable.kind === "u32-list") {
    return Object.freeze({ name: portable.name, kind: portable.kind, value: [...portable.values] });
  }
  return Object.freeze({ name: portable.name, kind: portable.kind, value: portable.value });
}

function bufferBytes(buffer: VectorBuffer): Uint8Array {
  const elements = product(buffer.shape, buffer.name);
  if (buffer.data.dtype === "bytes") {
    if (buffer.data.values.length !== elements) fail(`${buffer.name} byte count differs`);
    return Uint8Array.from(buffer.data.values);
  }
  const values = buffer.data.dtype === "f32" ? buffer.data.bits : buffer.data.values;
  if (values.length !== elements) fail(`${buffer.name} lane count differs`);
  const bytes = new Uint8Array(elements * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setUint32(index * 4, value, true));
  return bytes;
}

function poisonBytes(buffer: VectorBuffer): Uint8Array {
  const bytes = new Uint8Array(
    product(buffer.shape, buffer.name) * (buffer.data.dtype === "bytes" ? 1 : 4),
  );
  bytes.fill(0xa5);
  return bytes;
}

function compileComputePlan(): Readonly<{
  plan: CompiledTrainingPlanV1;
  initial: readonly WebGpuResidentTensorV1[];
  entries: readonly ComputeEntry[];
}> {
  const buffers: CompiledTrainingPlanV1["buffers"][number][] = [];
  const operations: CompiledTrainingPlanV1["operations"][number][] = [];
  const backwardOperations: CompiledTrainingPlanV1["backwardOperations"][number][] = [];
  const initial: WebGpuResidentTensorV1[] = [];
  const entries: ComputeEntry[] = [];
  let residentBytes = 0;

  for (const [caseIndex, item] of corpus.cases.entries()) {
    if (item.expected.kind !== "success" || item.operation.startsWith("lifecycle.")) continue;
    const prefix = `case.${caseIndex}`;
    const byName = new Map<string, VectorBuffer>();
    for (const buffer of [...item.inputs, ...item.expected.outputs]) {
      const previous = byName.get(buffer.name);
      if (
        previous !== undefined &&
        (previous.data.dtype !== buffer.data.dtype ||
          JSON.stringify(previous.shape) !== JSON.stringify(buffer.shape))
      ) {
        fail(`${item.case_id} reuses a buffer name with different geometry`);
      }
      if (previous === undefined) byName.set(buffer.name, buffer);
    }
    const inputByName = new Map(item.inputs.map((buffer) => [buffer.name, buffer]));
    for (const buffer of byName.values()) {
      const id = `${prefix}.${buffer.name}`;
      const byteLength = product(buffer.shape, id) * (buffer.data.dtype === "bytes" ? 1 : 4);
      buffers.push(Object.freeze({
        id,
        role: "activation",
        dtype: buffer.data.dtype,
        shape: Object.freeze([...buffer.shape]),
        aliasOf: null,
        ownerId: id,
        byteOffset: residentBytes,
        byteLength,
        backwardInitialization: "none",
      }));
      residentBytes += align16(byteLength);
      const input = inputByName.get(buffer.name);
      initial.push(Object.freeze({
        bufferId: id,
        bytes: input === undefined ? poisonBytes(buffer) : bufferBytes(input),
      }));
    }
    const operationId = `${prefix}.${item.operation}.${item.execution}`;
    const id = (name: string) => `${prefix}.${name}`;
    const attributes = Object.freeze(item.attributes.map((attribute) =>
      planAttribute(attribute, item.execution)));
    if (item.execution === "vjp") {
      backwardOperations.push(Object.freeze({
        id: operationId,
        sourceOperationId: `${prefix}.source`,
        operation: item.operation,
        execution: "vjp",
        inputs: Object.freeze(item.inputs.map((buffer) =>
          Object.freeze({ role: buffer.name, bufferId: id(buffer.name) }))),
        outputs: Object.freeze(item.expected.outputs.map((buffer) =>
          Object.freeze({ role: buffer.name, bufferId: id(buffer.name) }))),
        attributes,
      }));
    } else {
      operations.push(Object.freeze({
        id: operationId,
        operation: item.operation,
        inputs: Object.freeze(item.inputs.map((buffer) => id(buffer.name))),
        outputs: Object.freeze(item.expected.outputs.map((buffer) => id(buffer.name))),
        attributes,
      }));
    }
    const step = item.attributes.find((attribute) => attribute.name === "step")?.value;
    const scratchBytes = admitScratch(item, semanticScratchBytes(item));
    entries.push(Object.freeze({
      item: item as SuccessfulVectorCase,
      phase: item.execution === "vjp" ? "backward" : "forward",
      operationId,
      outputIds: Object.freeze(item.expected.outputs.map((buffer) => id(buffer.name))),
      optimizerStep: item.execution === "step"
        ? typeof step === "number" && Number.isSafeInteger(step) && step > 0 ? step : 1
        : undefined,
      scratchBytes,
      scratchBytesMax: item.expected.scratch_bytes_max,
    }));
  }

  const plan = Object.freeze({
    schemaId: "tritium.compiled_training_plan",
    schemaVersion: 1,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V2,
    buffers: Object.freeze(buffers),
    operations: Object.freeze(operations),
    backwardOperations: Object.freeze(backwardOperations),
    residentBytes,
    batchStagingBytes: 0,
    preparePeakBytes: residentBytes,
    forwardPeakBytes: residentBytes,
    exportPackageBytes: 0,
    exportPeakBytes: residentBytes,
    peakBytes: residentBytes,
  }) satisfies CompiledTrainingPlanV1;
  if (entries.length !== CANONICAL_COUNTS.compute) {
    fail("compiled compute case count drifted");
  }
  return Object.freeze({ plan, initial: Object.freeze(initial), entries: Object.freeze(entries) });
}

function bufferFromVector(buffer: VectorBuffer): PortableBufferV1 {
  if (buffer.data.dtype === "f32") {
    return Object.freeze({
      name: buffer.name,
      shape: Object.freeze([...buffer.shape]),
      data: Object.freeze({ dtype: "f32", bits: Object.freeze([...buffer.data.bits]) }),
    });
  }
  return Object.freeze({
    name: buffer.name,
    shape: Object.freeze([...buffer.shape]),
    data: Object.freeze({
      dtype: buffer.data.dtype,
      values: Object.freeze([...buffer.data.values]),
    }),
  });
}

function portableRequest(item: VectorCase, physicalDevice: string) {
  return Object.freeze({
    schemaId: "tritium.portable_training_request" as const,
    schemaVersion: 1 as const,
    physicalDevice,
    operation: item.operation,
    execution: item.execution,
    vectorDigest: TRAINING_VECTOR_DIGEST_V2,
    inputs: Object.freeze(item.inputs.map(bufferFromVector)),
    attributes: Object.freeze(item.attributes.map(portableAttribute)),
    outputs: Object.freeze(item.expected.outputs.map(bufferFromVector)),
  });
}

function u32Bits(bytes: Uint8Array): readonly number[] {
  if (bytes.byteLength % 4 !== 0) fail("f32/u32 output is not lane aligned");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Object.freeze(Array.from(
    { length: bytes.byteLength / 4 },
    (_, index) => view.getUint32(index * 4, true),
  ));
}

function actualBuffer(expected: VectorBuffer, bytes: Uint8Array): PortableBufferV1 {
  if (expected.data.dtype === "f32") {
    return Object.freeze({
      name: expected.name,
      shape: Object.freeze([...expected.shape]),
      data: Object.freeze({ dtype: "f32", bits: u32Bits(bytes) }),
    });
  }
  if (expected.data.dtype === "u32") {
    return Object.freeze({
      name: expected.name,
      shape: Object.freeze([...expected.shape]),
      data: Object.freeze({ dtype: "u32", values: u32Bits(bytes) }),
    });
  }
  return Object.freeze({
    name: expected.name,
    shape: Object.freeze([...expected.shape]),
    data: Object.freeze({ dtype: "bytes", values: Object.freeze([...bytes]) }),
  });
}

function numericValues(buffer: PortableBufferV1): readonly number[] {
  return buffer.data.dtype === "f32" ? buffer.data.bits : buffer.data.values;
}

function compareOutput(
  item: VectorCase,
  actual: PortableBufferV1,
  expected: VectorBuffer,
): void {
  const wanted = bufferFromVector(expected);
  if (
    actual.name !== wanted.name ||
    actual.data.dtype !== wanted.data.dtype ||
    JSON.stringify(actual.shape) !== JSON.stringify(wanted.shape)
  ) {
    fail(`${item.case_id} output envelope differs`);
  }
  const left = numericValues(actual);
  const right = numericValues(wanted);
  if (left.length !== right.length) fail(`${item.case_id} output length differs`);
  if (actual.data.dtype !== "f32" || wanted.data.dtype !== "f32" || item.tolerance.kind === "bit_exact") {
    if (left.some((value, index) => value !== right[index])) {
      fail(`${item.case_id} output differs under bit-exact grading`);
    }
    return;
  }
  const absolute = f32FromBits(item.tolerance.absolute_bits);
  const relative = f32FromBits(item.tolerance.relative_bits);
  for (let index = 0; index < left.length; index += 1) {
    const actualValue = f32FromBits(left[index]!);
    const expectedValue = f32FromBits(right[index]!);
    if (
      !Number.isFinite(actualValue) ||
      Math.abs(actualValue - expectedValue) > absolute + relative * Math.abs(expectedValue)
    ) {
      fail(`${item.case_id} output differs at lane ${index}`);
    }
  }
}

function outputDigest(outputs: readonly PortableBufferV1[]): string {
  return bytesToHex(sha256(UTF8.encode(JSON.stringify(outputs))));
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const recordValue = value as Readonly<Record<string, unknown>>;
    return `{${Object.keys(recordValue).sort().map((key) =>
      `${JSON.stringify(key)}:${canonicalJson(recordValue[key])}`
    ).join(",")}}`;
  }
  return JSON.stringify(value);
}

function samePortableOutputs(
  actual: readonly PortableBufferV1[],
  expected: readonly VectorBuffer[],
): boolean {
  return JSON.stringify(actual) === JSON.stringify(expected.map(bufferFromVector));
}

/**
 * Execute all 117 canonical cases from installed bytes. Compute successes run on
 * resident WebGPU. WASM remains limited to canonical lifecycle codecs and
 * expected-invalid admission; no successful tensor request enters WASM.
 * Takes exclusive device ownership and always destroys it.
 */
export async function runWebGpuVectorConformanceV1(
  candidateDevice: WebGpuDevicePortV1,
  options?: WebGpuVectorConformanceOptionsV1,
): Promise<WebGpuVectorConformanceTraceV1> {
  validateDevice(candidateDevice);
  const device = candidateDevice;
  let runtime: WebGpuResidentRuntimeV1 | null = null;
  try {
    const captured = snapshotOptions(options);
    const compiled = compileComputePlan();
    const uniformStride = Math.max(256, device.limits.minUniformBufferOffsetAlignment);
    if (!Number.isSafeInteger(uniformStride) || uniformStride % 256 !== 0) {
      fail("device uniform alignment is unsupported");
    }
    const schedule = compileWebGpuResidentScheduleV1(compiled.plan, {
      maxPeakBytes: captured.maxPeakBytes,
      uniformStride,
    });
    const traces = new Map<string, WebGpuVectorCaseTraceV1>();
    let explicitReadbacks = 0;
    let webgpuCaseTransactions = 0;
    let webgpuDispatches = 0;
    let wasmTensorDispatches = 0;
    let wasmCodecCalls = 0;
    let wasmValidationCalls = 0;
    const executor = captured.wasmSource === undefined
      ? await preparePortableWasmExecutor()
      : await preparePortableWasmExecutor(captured.wasmSource);
    const executeControlPlane = (item: VectorCase) => {
      if (item.expected.kind === "success" && !item.operation.startsWith("lifecycle.")) {
        wasmTensorDispatches += 1;
      }
      return executor.execute(portableRequest(item, captured.physicalDevice));
    };
    for (const item of corpus.cases) {
      if (item.expected.kind === "success" && !item.operation.startsWith("lifecycle.")) {
        continue;
      }
      const response = await executeControlPlane(item);
      if (item.expected.kind === "success") {
        if (response.status !== "ok" || response.outputs.length !== item.expected.outputs.length) {
          fail(`${item.case_id} lifecycle codec failed`);
        }
        response.outputs.forEach((output, index) => {
          const expected = item.expected.outputs[index];
          if (expected === undefined) fail(`${item.case_id} codec output index is missing`);
          compareOutput(item, output, expected);
        });
        const scratchBytes = admitScratch(item, response.receipt.scratchBytes);
        wasmCodecCalls += 1;
        traces.set(item.case_id, Object.freeze({
          caseId: item.case_id,
          implementation: "wasm-codec",
          outputDigest: outputDigest(response.outputs),
          scratchBytes,
          scratchBytesMax: item.expected.scratch_bytes_max,
        }));
      } else {
        if (
          response.status !== "error" ||
          response.error.category !== item.expected.category ||
          response.error.code !== item.expected.code ||
          !samePortableOutputs(response.outputs, item.expected.outputs)
        ) {
          fail(`${item.case_id} expected-invalid admission differs`);
        }
        wasmValidationCalls += 1;
        traces.set(item.case_id, Object.freeze({
          caseId: item.case_id,
          implementation: "wasm-validation",
          outputDigest: outputDigest(response.outputs),
          scratchBytes: null,
          scratchBytesMax: null,
        }));
      }
    }

    runtime = await WebGpuResidentRuntimeV1.prepare(
      device,
      compiled.plan,
      compiled.initial,
      schedule.auxiliaryResources(),
      uniformStride,
    );
    for (const entry of compiled.entries) {
      const transaction = schedule.transaction(
        entry.phase,
        entry.operationId,
        0,
        entry.optimizerStep,
      );
      await runtime.dispatchTransactions([transaction]);
      webgpuCaseTransactions += 1;
      webgpuDispatches += transaction.commands.length;
      const outputs: PortableBufferV1[] = [];
      for (let index = 0; index < entry.outputIds.length; index += 1) {
        const expected = entry.item.expected.outputs[index];
        if (expected === undefined) fail(`${entry.item.case_id} output index is missing`);
        const bytes = await runtime.read(entry.outputIds[index]!);
        explicitReadbacks += 1;
        const actual = actualBuffer(expected, bytes);
        compareOutput(entry.item, actual, expected);
        outputs.push(actual);
      }
      traces.set(entry.item.case_id, Object.freeze({
        caseId: entry.item.case_id,
        implementation: "webgpu",
        outputDigest: outputDigest(outputs),
        scratchBytes: entry.scratchBytes,
        scratchBytesMax: entry.scratchBytesMax,
      }));
    }

    if (
      webgpuCaseTransactions !== CANONICAL_COUNTS.compute ||
      webgpuDispatches <= 0 ||
      wasmTensorDispatches !== 0 ||
      wasmCodecCalls !== CANONICAL_COUNTS.lifecycle ||
      wasmValidationCalls !== CANONICAL_COUNTS.invalid
    ) {
      fail("observed execution path counts differ from the canonical inventory");
    }
    const observedWebGpuCaseTransactions = webgpuCaseTransactions as 68;
    const observedWasmDispatches = wasmTensorDispatches as 0;
    const observedWasmCodecCalls = wasmCodecCalls as 4;
    const observedWasmValidationCalls = wasmValidationCalls as 45;

    const ordered = Object.freeze(corpus.cases.map((item) => {
      const trace = traces.get(item.case_id);
      if (trace === undefined) fail(`${item.case_id} has no execution trace`);
      return trace;
    }));
    const executionDigest = bytesToHex(sha256(UTF8.encode(canonicalJson(ordered))));
    return Object.freeze({
      schemaId: "tritium.webgpu_vector_conformance_trace",
      schemaVersion: 1,
      implementation: "webgpu",
      manifestDigest: TRAINING_MANIFEST_DIGEST_V2,
      vectorDigest: TRAINING_VECTOR_DIGEST_V2,
      caseCounts: Object.freeze({
        valid: CANONICAL_COUNTS.valid,
        invalid: CANONICAL_COUNTS.invalid,
        skipped: 0,
      }),
      webgpuCaseTransactions: observedWebGpuCaseTransactions,
      webgpuDispatches,
      wasmDispatches: observedWasmDispatches,
      wasmCodecCalls: observedWasmCodecCalls,
      wasmValidationCalls: observedWasmValidationCalls,
      explicitReadbacks,
      peakBufferBytes: schedule.peakBytes(),
      executionDigest,
      cases: ordered,
    });
  } finally {
    if (runtime === null) device.destroy();
    else runtime.dispose();
  }
}

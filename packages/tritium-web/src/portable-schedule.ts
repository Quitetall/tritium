import { TRAINING_MANIFEST_DIGEST_V1, TRAINING_VECTOR_DIGEST_V1 } from "./identity.ts";
import { PORTABLE_OPERATION_BINDINGS_V1 } from "./operation-bindings.ts";
import type {
  PortableAttributeV1,
  PortableBufferDataV1,
  PortableBufferV1,
  PortableExecutionV1,
  PortableTrainingRequestV1,
} from "./portable.js";
import type {
  PortableCompiledDispatchV1,
  PortableSchedulePlanErrorCode,
  PortableScheduleTensorStoreV1,
  PortableScheduleTensorV1,
} from "./portable-schedule-types.js";
import type {
  CompiledBackwardOperationV1,
  CompiledTrainingBufferV1,
  CompiledTrainingOperationV1,
  CompiledTrainingPlanV1,
  TrainingAttributeSpecV1,
} from "./session.ts";

export type {
  PortableCompiledDispatchV1,
  PortableSchedulePlanErrorCode,
  PortableScheduleTensorStoreV1,
  PortableScheduleTensorV1,
} from "./portable-schedule-types.js";

type Binding = {
  readonly inputs: readonly string[];
  readonly attributes: readonly Readonly<{ name: string; kind: TrainingAttributeSpecV1["kind"] }>[];
  readonly outputs: readonly string[];
};
type BindingRegistry = Readonly<
  Record<string, Readonly<Partial<Record<"forward" | "vjp" | "step", Binding>>>>
>;
const BINDINGS = PORTABLE_OPERATION_BINDINGS_V1 as unknown as BindingRegistry;
const MAX_PORTABLE_BUFFER_BYTES = 8 * 1024 * 1024;
const MAX_PORTABLE_REQUEST_JSON_BYTES = 8 * 1024 * 1024;

export class PortableSchedulePlanError extends Error {
  readonly code: PortableSchedulePlanErrorCode;

  constructor(code: PortableSchedulePlanErrorCode, message: string) {
    super(message);
    this.name = "PortableSchedulePlanError";
    this.code = code;
  }
}

function fail(code: PortableSchedulePlanErrorCode, message: string): never {
  throw new PortableSchedulePlanError(code, message);
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isDenseArray(value: readonly unknown[]): boolean {
  return Object.keys(value).length === value.length;
}

function checkPlan(plan: CompiledTrainingPlanV1): void {
  if (
    typeof plan !== "object" ||
    plan === null ||
    plan.schemaId !== "tritium.compiled_training_plan" ||
    plan.schemaVersion !== 1 ||
    plan.manifestDigest !== TRAINING_MANIFEST_DIGEST_V1 ||
    !Array.isArray(plan.buffers) ||
    !isDenseArray(plan.buffers) ||
    !Array.isArray(plan.operations) ||
    !isDenseArray(plan.operations) ||
    !Array.isArray(plan.backwardOperations) ||
    !isDenseArray(plan.backwardOperations)
  ) {
    fail("invalid_schema", "compiled plan identity is invalid");
  }
}

function checkDevice(physicalDevice: string): void {
  if (typeof physicalDevice !== "string" || physicalDevice.length === 0) {
    fail("invalid_schema", "physicalDevice must be nonempty");
  }
}

function findOne<T extends { readonly id: string }>(
  items: readonly T[],
  id: string,
  name: string,
): T {
  if (typeof id !== "string" || id.length === 0) {
    fail("invalid_schema", `${name} id must be nonempty`);
  }
  if (items.some((item) => !isRecord(item) || typeof item.id !== "string" || item.id.length === 0)) {
    fail("invalid_schema", `${name} entries are invalid`);
  }
  const matches = items.filter((item) => item.id === id);
  if (matches.length !== 1) {
    fail("invalid_schema", `${name} ${id} is missing or duplicated`);
  }
  return matches[0]!;
}

function bufferMap(
  plan: CompiledTrainingPlanV1,
): ReadonlyMap<string, CompiledTrainingBufferV1> {
  const buffers = new Map<string, CompiledTrainingBufferV1>();
  for (const buffer of plan.buffers) {
    if (
      !isRecord(buffer) ||
      typeof buffer.id !== "string" ||
      buffer.id.length === 0 ||
      typeof buffer.ownerId !== "string" ||
      buffer.ownerId.length === 0 ||
      !(buffer.aliasOf === null || (typeof buffer.aliasOf === "string" && buffer.aliasOf.length > 0)) ||
      !(["batch", "parameter", "gradient", "optimizer-state", "activation", "result"] as const).includes(buffer.role) ||
      !(["f32", "u32", "bytes"] as const).includes(buffer.dtype) ||
      !Array.isArray(buffer.shape) ||
      !isDenseArray(buffer.shape) ||
      buffer.shape.some(
        (dimension) => !Number.isSafeInteger(dimension) || dimension <= 0,
      ) ||
      !Number.isSafeInteger(buffer.byteOffset) ||
      buffer.byteOffset < 0 ||
      !Number.isSafeInteger(buffer.byteLength) ||
      buffer.byteLength <= 0
    ) {
      fail("invalid_schema", "compiled buffer entry is invalid");
    }
    let elements = 1;
    for (const dimension of buffer.shape) {
      elements *= dimension;
      if (!Number.isSafeInteger(elements)) {
        fail("invalid_schema", `compiled buffer ${buffer.id} shape exceeds safe range`);
      }
    }
    const expectedBytes = elements * (buffer.dtype === "bytes" ? 1 : 4);
    if (!Number.isSafeInteger(expectedBytes) || expectedBytes !== buffer.byteLength) {
      fail("invalid_schema", `compiled buffer ${buffer.id} byte length is inconsistent`);
    }
    if (buffer.byteLength > MAX_PORTABLE_BUFFER_BYTES) {
      fail("capacity", `compiled buffer ${buffer.id} exceeds portable 8 MiB limit`);
    }
    if (buffers.has(buffer.id)) {
      fail("invalid_schema", `compiled buffer ${buffer.id} is duplicated`);
    }
    buffers.set(buffer.id, buffer);
  }
  for (const buffer of buffers.values()) {
    const owner = buffers.get(buffer.ownerId);
    const aliasTarget = buffer.aliasOf === null ? null : buffers.get(buffer.aliasOf);
    if (
      owner === undefined ||
      owner.aliasOf !== null ||
      owner.ownerId !== owner.id ||
      (buffer.aliasOf === null && buffer.ownerId !== buffer.id) ||
      (buffer.aliasOf !== null &&
        (buffer.aliasOf === buffer.id ||
          buffer.aliasOf !== buffer.ownerId ||
          buffer.role !== "parameter" ||
          owner.role !== "parameter" ||
          aliasTarget === undefined ||
          aliasTarget === null ||
          aliasTarget.aliasOf !== null ||
          aliasTarget.ownerId !== owner.id ||
          buffer.ownerId !== owner.id)) ||
      buffer.dtype !== owner.dtype ||
      buffer.byteOffset !== owner.byteOffset ||
      buffer.byteLength !== owner.byteLength ||
      buffer.shape.length !== owner.shape.length ||
      buffer.shape.some((dimension, index) => dimension !== owner.shape[index])
    ) {
      fail("invalid_schema", `compiled buffer ${buffer.id} owner is inconsistent`);
    }
  }
  return buffers;
}

function tensorFor(
  store: PortableScheduleTensorStoreV1,
  buffer: CompiledTrainingBufferV1,
): PortableScheduleTensorV1 {
  if (typeof store !== "object" || store === null || Array.isArray(store)) {
    fail("invalid_schema", "tensor store must be an object");
  }
  const value = store[buffer.ownerId];
  if (value === undefined) {
    fail("missing_buffer", `tensor store omits ${buffer.ownerId}`);
  }
  const matches =
    (buffer.dtype === "f32" && value instanceof Float32Array) ||
    (buffer.dtype === "u32" && value instanceof Uint32Array) ||
    (buffer.dtype === "bytes" && value instanceof Uint8Array);
  if (!matches || value.byteLength !== buffer.byteLength) {
    fail("buffer_mismatch", `tensor ${buffer.ownerId} does not match compiled dtype/shape`);
  }
  return value;
}

function f32Bits(values: Float32Array): readonly number[] {
  return Object.freeze(
    Array.from(new Uint32Array(values.buffer, values.byteOffset, values.length)),
  );
}

function inputData(
  dtype: CompiledTrainingBufferV1["dtype"],
  tensor: PortableScheduleTensorV1,
): PortableBufferDataV1 {
  if (dtype === "f32" && tensor instanceof Float32Array) {
    return Object.freeze({ dtype, bits: f32Bits(tensor) });
  }
  if (dtype === "u32" && tensor instanceof Uint32Array) {
    return Object.freeze({ dtype, values: Object.freeze(Array.from(tensor)) });
  }
  if (dtype === "bytes" && tensor instanceof Uint8Array) {
    return Object.freeze({ dtype, values: Object.freeze(Array.from(tensor)) });
  }
  fail("buffer_mismatch", "tensor dtype changed after validation");
}

function outputData(buffer: CompiledTrainingBufferV1): PortableBufferDataV1 {
  const elements = buffer.dtype === "bytes" ? buffer.byteLength : buffer.byteLength / 4;
  if (buffer.dtype === "f32") {
    return Object.freeze({ dtype: "f32", bits: Object.freeze(new Array(elements).fill(0)) });
  }
  if (buffer.dtype === "u32") {
    return Object.freeze({ dtype: "u32", values: Object.freeze(new Array(elements).fill(0)) });
  }
  return Object.freeze({ dtype: "bytes", values: Object.freeze(new Array(elements).fill(0)) });
}

function portableInput(
  role: string,
  bufferId: string,
  buffers: ReadonlyMap<string, CompiledTrainingBufferV1>,
  store: PortableScheduleTensorStoreV1,
): PortableBufferV1 {
  const buffer = buffers.get(bufferId);
  if (buffer === undefined) fail("invalid_schema", `unknown compiled buffer ${bufferId}`);
  const tensor = tensorFor(store, buffer);
  return Object.freeze({
    name: role,
    shape: Object.freeze([...buffer.shape]),
    data: inputData(buffer.dtype, tensor),
  });
}

function portableOutput(
  role: string,
  bufferId: string,
  buffers: ReadonlyMap<string, CompiledTrainingBufferV1>,
): PortableBufferV1 {
  const buffer = buffers.get(bufferId);
  if (buffer === undefined) fail("invalid_schema", `unknown compiled buffer ${bufferId}`);
  return Object.freeze({
    name: role,
    shape: Object.freeze([...buffer.shape]),
    data: outputData(buffer),
  });
}

function f32AttributeBits(value: number): number {
  const bytes = new ArrayBuffer(4);
  const view = new DataView(bytes);
  view.setFloat32(0, value, true);
  return view.getUint32(0, true);
}

function portableAttribute(attribute: TrainingAttributeSpecV1): PortableAttributeV1 {
  if (!isRecord(attribute) || typeof attribute.name !== "string" || attribute.name.length === 0) {
    fail("invalid_schema", "compiled attribute is invalid");
  }
  switch (attribute.kind) {
    case "f32":
      if (typeof attribute.value !== "number" || !Number.isFinite(Math.fround(attribute.value))) {
        fail("invalid_schema", `${attribute.name} must be finite f32`);
      }
      return Object.freeze({
        kind: "f32",
        name: attribute.name,
        bits: f32AttributeBits(attribute.value as number),
      });
    case "u64":
      if (!Number.isSafeInteger(attribute.value) || (attribute.value as number) < 0) {
        fail("invalid_schema", `${attribute.name} must be safe u64`);
      }
      return Object.freeze({ kind: "u64", name: attribute.name, value: attribute.value as number });
    case "bool":
      if (typeof attribute.value !== "boolean") {
        fail("invalid_schema", `${attribute.name} must be boolean`);
      }
      return Object.freeze({ kind: "bool", name: attribute.name, value: attribute.value as boolean });
    case "text":
      if (typeof attribute.value !== "string" || attribute.value.length === 0) {
        fail("invalid_schema", `${attribute.name} must be nonempty text`);
      }
      return Object.freeze({ kind: "text", name: attribute.name, value: attribute.value as string });
    case "u64-list":
    case "u32-list":
      if (
        !Array.isArray(attribute.value) ||
        !isDenseArray(attribute.value) ||
        attribute.value.some(
          (value) =>
            !Number.isSafeInteger(value) ||
            value < 0 ||
            (attribute.kind === "u32-list" && value > 0xffff_ffff),
        )
      ) {
        fail("invalid_schema", `${attribute.name} integer list is invalid`);
      }
      return Object.freeze({
        kind: attribute.kind,
        name: attribute.name,
        values: Object.freeze([...(attribute.value as readonly number[])]),
      });
    default:
      fail("invalid_schema", "compiled attribute kind is invalid");
  }
}

function checkIds(ids: readonly string[], name: string): void {
  if (
    !Array.isArray(ids) ||
    !isDenseArray(ids) ||
    ids.some((id) => typeof id !== "string" || id.length === 0)
  ) {
    fail("invalid_schema", `${name} buffer ids are invalid`);
  }
}

function checkAttributes(attributes: readonly TrainingAttributeSpecV1[], name: string): void {
  if (!Array.isArray(attributes) || !isDenseArray(attributes)) {
    fail("invalid_schema", `${name} attributes are invalid`);
  }
  for (const attribute of attributes) portableAttribute(attribute);
}

function binding(operation: string, execution: "forward" | "vjp" | "step"): Binding {
  const selected = BINDINGS[operation]?.[execution];
  if (selected === undefined) {
    fail("invalid_schema", `no canonical ${execution} binding for ${operation}`);
  }
  return selected;
}

function compile(
  operation: string,
  execution: "forward" | "vjp" | "step",
  inputIds: readonly string[],
  outputIds: readonly string[],
  attributes: readonly TrainingAttributeSpecV1[],
  plan: CompiledTrainingPlanV1,
  store: PortableScheduleTensorStoreV1,
  physicalDevice: string,
): PortableCompiledDispatchV1 {
  checkPlan(plan);
  checkDevice(physicalDevice);
  checkIds(inputIds, `${operation}.${execution} input`);
  checkIds(outputIds, `${operation}.${execution} output`);
  checkAttributes(attributes, `${operation}.${execution}`);
  const roles = binding(operation, execution);
  if (roles.inputs.length !== inputIds.length || roles.outputs.length !== outputIds.length) {
    fail("invalid_schema", `${operation}.${execution} arity differs from canonical ABI`);
  }
  if (
    roles.attributes.length !== attributes.length ||
    attributes.some(
      (attribute, index) =>
        attribute.name !== roles.attributes[index]?.name ||
        attribute.kind !== roles.attributes[index]?.kind,
    )
  ) {
    fail("invalid_schema", `${operation}.${execution} attributes differ from canonical ABI`);
  }
  const buffers = bufferMap(plan);
  const request: PortableTrainingRequestV1 = Object.freeze({
    schemaId: "tritium.portable_training_request",
    schemaVersion: 1,
    physicalDevice,
    operation,
    execution: execution as PortableExecutionV1,
    vectorDigest: TRAINING_VECTOR_DIGEST_V1,
    inputs: Object.freeze(
      inputIds.map((id, index) => portableInput(roles.inputs[index]!, id, buffers, store)),
    ),
    attributes: Object.freeze(attributes.map(portableAttribute)),
    outputs: Object.freeze(
      outputIds.map((id, index) => portableOutput(roles.outputs[index]!, id, buffers)),
    ),
  });
  if (new TextEncoder().encode(JSON.stringify(request)).byteLength > MAX_PORTABLE_REQUEST_JSON_BYTES) {
    fail("capacity", "compiled portable request JSON exceeds 8 MiB");
  }
  return Object.freeze({ request, outputBufferIds: Object.freeze([...outputIds]) });
}

/** Compile one frozen forward or optimizer operation into portable ABI V1. */
export function compilePortablePlanOperationRequest(
  plan: CompiledTrainingPlanV1,
  operationId: string,
  store: PortableScheduleTensorStoreV1,
  physicalDevice = "wasm32:browser",
): PortableCompiledDispatchV1 {
  checkPlan(plan);
  const operation: CompiledTrainingOperationV1 = findOne(
    plan.operations,
    operationId,
    "compiled operation",
  );
  if (
    typeof operation.operation !== "string" ||
    operation.operation.length === 0 ||
    !Array.isArray(operation.inputs) ||
    !Array.isArray(operation.outputs) ||
    !Array.isArray(operation.attributes)
  ) {
    fail("invalid_schema", `${operationId} compiled operation is invalid`);
  }
  const execution = operation.operation.startsWith("optimizer.") ? "step" : "forward";
  return compile(
    operation.operation,
    execution,
    operation.inputs,
    operation.outputs,
    operation.attributes,
    plan,
    store,
    physicalDevice,
  );
}

/** Compile one frozen reverse dispatch and verify its generated role bindings. */
export function compilePortableBackwardOperationRequest(
  plan: CompiledTrainingPlanV1,
  operationId: string,
  store: PortableScheduleTensorStoreV1,
  physicalDevice = "wasm32:browser",
): PortableCompiledDispatchV1 {
  checkPlan(plan);
  const operation: CompiledBackwardOperationV1 = findOne(
    plan.backwardOperations,
    operationId,
    "compiled backward operation",
  );
  if (
    typeof operation.operation !== "string" ||
    operation.operation.length === 0 ||
    !(operation.execution === "forward" || operation.execution === "vjp") ||
    !Array.isArray(operation.inputs) ||
    !isDenseArray(operation.inputs) ||
    !Array.isArray(operation.outputs) ||
    !isDenseArray(operation.outputs) ||
    !Array.isArray(operation.attributes) ||
    operation.inputs.some(
      (item) => !isRecord(item) || typeof item.role !== "string" || typeof item.bufferId !== "string",
    ) ||
    operation.outputs.some(
      (item) => !isRecord(item) || typeof item.role !== "string" || typeof item.bufferId !== "string",
    )
  ) {
    fail("invalid_schema", `${operationId} compiled backward operation is invalid`);
  }
  const execution = operation.execution;
  const roles = binding(operation.operation, execution);
  const actualInputs = operation.inputs.map((item) => item.role);
  const actualOutputs = operation.outputs.map((item) => item.role);
  if (
    actualInputs.length !== roles.inputs.length ||
    actualInputs.some((role, index) => role !== roles.inputs[index]) ||
    actualOutputs.length !== roles.outputs.length ||
    actualOutputs.some((role, index) => role !== roles.outputs[index])
  ) {
    fail("invalid_schema", `${operation.id} role bindings drift from canonical ABI`);
  }
  return compile(
    operation.operation,
    execution,
    operation.inputs.map((item) => item.bufferId),
    operation.outputs.map((item) => item.bufferId),
    operation.attributes,
    plan,
    store,
    physicalDevice,
  );
}

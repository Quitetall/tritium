import {
  canonicalTrainingManifestJson,
  parseTrainingManifest,
} from "../../../bindings/typescript/src/training_manifest.ts";

import {
  TRAINING_MANIFEST_DIGEST_V1,
  TRAINING_VECTOR_DIGEST_V1,
} from "./identity.ts";

export type WebTrainingBackendPolicyV1 = "auto" | "webgpu" | "wasm";
export type WebTrainingImplementationV1 = "webgpu" | "wasm-fallback";
export type WebTrainingState =
  | "prepared"
  | "forward-complete"
  | "backward-complete"
  | "disposed";

export type WebTrainingErrorCode =
  | "adapter_unavailable"
  | "backend_policy"
  | "busy"
  | "capability_mismatch"
  | "disposed"
  | "invalid_config"
  | "invalid_receipt"
  | "invalid_schema"
  | "invalid_state"
  | "memory_limit";

export class WebTrainingError extends Error {
  readonly code: WebTrainingErrorCode;
  readonly state: WebTrainingState | null;

  constructor(
    code: WebTrainingErrorCode,
    message: string,
    state: WebTrainingState | null = null,
  ) {
    super(message);
    this.name = "WebTrainingError";
    this.code = code;
    this.state = state;
  }
}

export interface TrainingRecipeV1 {
  readonly schemaId: "tritium.training_recipe";
  readonly schemaVersion: 1;
  readonly tensors: readonly TrainingTensorSpecV1[];
  readonly operations: readonly TrainingOperationSpecV1[];
}

export type TrainingDTypeV1 = "f32" | "u32" | "bytes";
export type TrainingTensorRoleV1 =
  | "batch"
  | "parameter"
  | "gradient"
  | "optimizer-state"
  | "activation"
  | "result";

export interface TrainingTensorSpecV1 {
  readonly id: string;
  readonly dtype: TrainingDTypeV1;
  readonly shape: readonly number[];
  readonly role: TrainingTensorRoleV1;
  readonly aliasOf: string | null;
}

export interface TrainingOperationSpecV1 {
  readonly id: string;
  readonly operation: string;
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
  readonly attributes: readonly TrainingAttributeSpecV1[];
}

export type TrainingAttributeKindV1 =
  | "f32"
  | "u64"
  | "bool"
  | "text"
  | "u64-list"
  | "u32-list";

export interface TrainingAttributeSpecV1 {
  readonly name: string;
  readonly kind: TrainingAttributeKindV1;
  readonly value: number | boolean | string | readonly number[];
}

export interface CompiledTrainingBufferV1 extends TrainingTensorSpecV1 {
  readonly ownerId: string;
  readonly byteOffset: number;
  readonly byteLength: number;
}

export interface CompiledTrainingOperationV1 extends TrainingOperationSpecV1 {}

export interface CompiledTrainingPlanV1 {
  readonly schemaId: "tritium.compiled_training_plan";
  readonly schemaVersion: 1;
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V1;
  readonly buffers: readonly CompiledTrainingBufferV1[];
  readonly operations: readonly CompiledTrainingOperationV1[];
  readonly residentBytes: number;
  readonly batchStagingBytes: number;
  readonly preparePeakBytes: number;
  readonly forwardPeakBytes: number;
  readonly peakBytes: number;
}

export interface WebTrainingModelV1 {
  readonly schemaId: "tritium.web_training_model";
  readonly schemaVersion: 1;
  readonly recipe: TrainingRecipeV1;
  readonly payload: Uint8Array;
}

export interface TrainingBatchV1 {
  readonly inputs: Readonly<
    Record<string, Float32Array | Uint32Array | Uint8Array>
  >;
}

export interface WebTrainingConfigV1 {
  readonly backend: WebTrainingBackendPolicyV1;
  readonly allowWasmFallback: boolean;
  readonly maxResidentBytes: number;
  readonly seed: number;
  readonly requiredOperations: readonly string[];
}

export interface WebTrainingCapabilitiesV1 {
  readonly schemaId: "tritium.web_training_capabilities";
  readonly schemaVersion: 1;
  readonly implementation: WebTrainingImplementationV1;
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V1;
  readonly vectorDigest: typeof TRAINING_VECTOR_DIGEST_V1;
  readonly buildId: string;
  readonly physicalDevice: string | null;
  readonly supportedOperations: readonly string[];
  readonly maxResidentBytes: number;
}

export interface WebTrainingReceiptV1 {
  readonly schemaId: "tritium.web_training_receipt";
  readonly schemaVersion: 1;
  readonly implementation: WebTrainingImplementationV1;
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V1;
  readonly vectorDigest: typeof TRAINING_VECTOR_DIGEST_V1;
  readonly buildId: string;
  readonly physicalDevice: string | null;
  readonly operation: string;
  readonly completedSteps: number;
  readonly peakResidentBytes: number;
}

export interface TrainingResultV1 {
  readonly loss: number;
  readonly receipt: WebTrainingReceiptV1;
}

export interface WebBinaryResultV1 {
  readonly bytes: Uint8Array;
  readonly receipt: WebTrainingReceiptV1;
}

/** Low-level adapter implemented by generated WASM/WebGPU packages.
 * `validate` must be allocation-free, and neither `validate` nor `prepare` may
 * mutate or retain their arguments. Validation completes operation-specific
 * geometry and attribute checks before `prepare` allocates persistent state.
 */
export interface WebTrainingAdapterV1 {
  readonly capabilities: WebTrainingCapabilitiesV1;
  validate(
    model: WebTrainingModelV1,
    config: WebTrainingConfigV1,
    plan: CompiledTrainingPlanV1,
  ): Promise<void>;
  prepare(
    model: WebTrainingModelV1,
    config: WebTrainingConfigV1,
    plan: CompiledTrainingPlanV1,
  ): Promise<WebTrainingReceiptV1>;
  forward(batch: TrainingBatchV1): Promise<TrainingResultV1>;
  backward(result: TrainingResultV1): Promise<WebTrainingReceiptV1>;
  step(): Promise<WebTrainingReceiptV1>;
  checkpoint(): Promise<WebBinaryResultV1>;
  resume(checkpoint: Uint8Array): Promise<WebTrainingReceiptV1>;
  export(): Promise<WebBinaryResultV1>;
  dispose(): Promise<void>;
}

const CONFIG_KEYS = [
  "allowWasmFallback",
  "backend",
  "maxResidentBytes",
  "requiredOperations",
  "seed",
] as const;
const MODEL_KEYS = ["payload", "recipe", "schemaId", "schemaVersion"] as const;
const RECIPE_KEYS = ["operations", "schemaId", "schemaVersion", "tensors"] as const;
const TENSOR_KEYS = ["aliasOf", "dtype", "id", "role", "shape"] as const;
const OPERATION_KEYS = ["attributes", "id", "inputs", "operation", "outputs"] as const;
const ATTRIBUTE_KEYS = ["kind", "name", "value"] as const;
const CAPABILITY_KEYS = [
  "buildId",
  "implementation",
  "manifestDigest",
  "maxResidentBytes",
  "physicalDevice",
  "schemaId",
  "schemaVersion",
  "supportedOperations",
  "vectorDigest",
] as const;
const RECEIPT_KEYS = [
  "buildId",
  "completedSteps",
  "implementation",
  "manifestDigest",
  "operation",
  "peakResidentBytes",
  "physicalDevice",
  "schemaId",
  "schemaVersion",
  "vectorDigest",
] as const;

function fail(
  code: WebTrainingErrorCode,
  message: string,
  state: WebTrainingState | null = null,
): never {
  throw new WebTrainingError(code, message, state);
}

function exactKeys(
  value: unknown,
  expected: readonly string[],
  name: string,
  code: WebTrainingErrorCode = "invalid_schema",
): void {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail(code, `${name} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (
    actual.length !== wanted.length ||
    actual.some((key, index) => key !== wanted[index])
  ) {
    fail(code, `${name} fields do not match schema v1`);
  }
}

function copyTypedArray(
  value: Float32Array | Uint32Array | Uint8Array,
): Float32Array | Uint32Array | Uint8Array {
  if (value instanceof Float32Array) return Float32Array.from(value);
  if (value instanceof Uint32Array) return Uint32Array.from(value);
  return Uint8Array.from(value);
}

function validateAndCopyBatch(
  batch: TrainingBatchV1,
  plan: CompiledTrainingPlanV1,
  state: WebTrainingState,
): TrainingBatchV1 {
  exactKeys(batch, ["inputs"], "batch");
  if (
    typeof batch.inputs !== "object" ||
    batch.inputs === null ||
    Array.isArray(batch.inputs)
  ) {
    fail("invalid_schema", "batch.inputs must be an object", state);
  }
  const expected = plan.buffers.filter((buffer) => buffer.role === "batch");
  const actualNames = Object.keys(batch.inputs).sort();
  const expectedNames = expected.map((buffer) => buffer.id).sort();
  if (
    actualNames.length !== expectedNames.length ||
    actualNames.some((name, index) => name !== expectedNames[index])
  ) {
    fail("invalid_schema", "batch inputs do not match the compiled plan", state);
  }
  const copied: Record<string, Float32Array | Uint32Array | Uint8Array> = {};
  for (const buffer of expected) {
    const value = batch.inputs[buffer.id];
    const dtypeMatches =
      (buffer.dtype === "f32" && value instanceof Float32Array) ||
      (buffer.dtype === "u32" && value instanceof Uint32Array) ||
      (buffer.dtype === "bytes" && value instanceof Uint8Array);
    if (!dtypeMatches || value === undefined || value.byteLength !== buffer.byteLength) {
      fail(
        "invalid_schema",
        `batch input ${buffer.id} does not match compiled dtype/shape`,
        state,
      );
    }
    copied[buffer.id] = copyTypedArray(value);
  }
  return Object.freeze({ inputs: Object.freeze(copied) });
}

function safeNonnegativeInteger(
  value: number,
  name: string,
  code: WebTrainingErrorCode = "invalid_config",
): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    fail(code, `${name} must be a nonnegative safe integer`);
  }
}

function isDenseArray(values: readonly unknown[]): boolean {
  for (let index = 0; index < values.length; index += 1) {
    if (!(index in values)) return false;
  }
  return true;
}

function nonemptyUniqueStrings(values: readonly string[], name: string): void {
  if (
    !Array.isArray(values) ||
    !isDenseArray(values) ||
    values.length === 0 ||
    values.some((value) => typeof value !== "string" || value.length === 0) ||
    new Set(values).size !== values.length
  ) {
    fail("invalid_schema", `${name} must contain unique nonempty strings`);
  }
}

function uniqueStrings(
  values: readonly string[],
  name: string,
  allowEmpty: boolean,
): void {
  if (
    !Array.isArray(values) ||
    !isDenseArray(values) ||
    (!allowEmpty && values.length === 0) ||
    values.some((value) => typeof value !== "string" || value.length === 0) ||
    new Set(values).size !== values.length
  ) {
    fail("invalid_schema", `${name} must contain unique nonempty strings`);
  }
}

function validateAttribute(attribute: TrainingAttributeSpecV1, operationId: string): void {
  exactKeys(attribute, ATTRIBUTE_KEYS, `${operationId} attribute`);
  if (typeof attribute.name !== "string" || attribute.name.length === 0) {
    fail("invalid_schema", `${operationId} attribute name is invalid`);
  }
  switch (attribute.kind) {
    case "f32":
      if (
        typeof attribute.value !== "number" ||
        !Number.isFinite(attribute.value) ||
        !Number.isFinite(Math.fround(attribute.value))
      ) {
        fail("invalid_schema", `${operationId}.${attribute.name} must be finite f32`);
      }
      return;
    case "u64":
      if (
        typeof attribute.value !== "number" ||
        !Number.isSafeInteger(attribute.value) ||
        attribute.value < 0
      ) {
        fail("invalid_schema", `${operationId}.${attribute.name} must be safe u64`);
      }
      return;
    case "bool":
      if (typeof attribute.value !== "boolean") {
        fail("invalid_schema", `${operationId}.${attribute.name} must be boolean`);
      }
      return;
    case "text":
      if (typeof attribute.value !== "string" || attribute.value.length === 0) {
        fail("invalid_schema", `${operationId}.${attribute.name} must be nonempty text`);
      }
      return;
    case "u64-list":
    case "u32-list": {
      const maximum = attribute.kind === "u32-list" ? 0xffff_ffff : Number.MAX_SAFE_INTEGER;
      if (
        !Array.isArray(attribute.value) ||
        !isDenseArray(attribute.value) ||
        attribute.value.some(
          (value: unknown) =>
            typeof value !== "number" ||
            !Number.isSafeInteger(value) ||
            value < 0 ||
            value > maximum,
        )
      ) {
        fail("invalid_schema", `${operationId}.${attribute.name} integer list is invalid`);
      }
      return;
    }
    default:
      fail("invalid_schema", `${operationId}.${attribute.name} kind is invalid`);
  }
}

function validateModel(model: WebTrainingModelV1): void {
  exactKeys(model, MODEL_KEYS, "model");
  if (
    model.schemaId !== "tritium.web_training_model" ||
    model.schemaVersion !== 1 ||
    !(model.payload instanceof Uint8Array) ||
    model.payload.byteLength === 0
  ) {
    fail("invalid_schema", "model is not a nonempty WebTrainingModelV1");
  }
  exactKeys(model.recipe, RECIPE_KEYS, "recipe");
  if (
    model.recipe.schemaId !== "tritium.training_recipe" ||
    model.recipe.schemaVersion !== 1 ||
    !Array.isArray(model.recipe.tensors) ||
    !isDenseArray(model.recipe.tensors) ||
    model.recipe.tensors.length === 0 ||
    !Array.isArray(model.recipe.operations) ||
    !isDenseArray(model.recipe.operations) ||
    model.recipe.operations.length === 0
  ) {
    fail("invalid_schema", "recipe schema identity is not v1");
  }
  const tensorIds: string[] = [];
  for (const tensor of model.recipe.tensors) {
    exactKeys(tensor, TENSOR_KEYS, "recipe tensor");
    if (
      typeof tensor.id !== "string" ||
      tensor.id.length === 0 ||
      !(["f32", "u32", "bytes"] as const).includes(tensor.dtype) ||
      !([
        "batch",
        "parameter",
        "gradient",
        "optimizer-state",
        "activation",
        "result",
      ] as const).includes(tensor.role) ||
      !Array.isArray(tensor.shape) ||
      !isDenseArray(tensor.shape) ||
      tensor.shape.some(
        (dimension: unknown) =>
          typeof dimension !== "number" ||
          !Number.isSafeInteger(dimension) ||
          dimension <= 0,
      ) ||
      !(tensor.aliasOf === null ||
        (typeof tensor.aliasOf === "string" && tensor.aliasOf.length > 0))
    ) {
      fail("invalid_schema", `invalid tensor ${String(tensor.id)}`);
    }
    tensorIds.push(tensor.id);
  }
  uniqueStrings(tensorIds, "recipe tensor ids", false);

  const operationIds: string[] = [];
  for (const operation of model.recipe.operations) {
    exactKeys(operation, OPERATION_KEYS, "recipe operation");
    if (
      typeof operation.id !== "string" ||
      operation.id.length === 0 ||
      typeof operation.operation !== "string" ||
      operation.operation.length === 0
    ) {
      fail("invalid_schema", "recipe operation identity is invalid");
    }
    uniqueStrings(operation.inputs, `${operation.id}.inputs`, true);
    uniqueStrings(operation.outputs, `${operation.id}.outputs`, true);
    if (
      !Array.isArray(operation.attributes) ||
      !isDenseArray(operation.attributes)
    ) {
      fail("invalid_schema", `${operation.id}.attributes must be an array`);
    }
    for (const attribute of operation.attributes) {
      validateAttribute(attribute, operation.id);
    }
    uniqueStrings(
      operation.attributes.map((attribute: TrainingAttributeSpecV1) => attribute.name),
      `${operation.id} attribute names`,
      true,
    );
    operationIds.push(operation.id);
  }
  uniqueStrings(operationIds, "recipe operation ids", false);
}

function validateConfig(config: WebTrainingConfigV1): void {
  exactKeys(config, CONFIG_KEYS, "config");
  if (!(["auto", "webgpu", "wasm"] as const).includes(config.backend)) {
    fail("invalid_config", `unknown backend policy ${String(config.backend)}`);
  }
  if (typeof config.allowWasmFallback !== "boolean") {
    fail("invalid_config", "allowWasmFallback must be boolean");
  }
  safeNonnegativeInteger(config.maxResidentBytes, "maxResidentBytes");
  if (config.maxResidentBytes === 0) {
    fail("invalid_config", "maxResidentBytes must be positive");
  }
  safeNonnegativeInteger(config.seed, "seed");
  nonemptyUniqueStrings(config.requiredOperations, "requiredOperations");
}

function checkedMultiply(left: number, right: number, name: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    fail("invalid_schema", `${name} byte size exceeds safe integer range`);
  }
  return product;
}

function checkedAdd(left: number, right: number, name: string): number {
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    fail("memory_limit", `${name} exceeds safe integer range`);
  }
  return sum;
}

function tensorByteLength(tensor: TrainingTensorSpecV1): number {
  let elements = 1;
  for (const dimension of tensor.shape) {
    elements = checkedMultiply(elements, dimension, tensor.id);
  }
  const width = tensor.dtype === "bytes" ? 1 : 4;
  return checkedMultiply(elements, width, tensor.id);
}

function align16(value: number): number {
  return checkedAdd(value, (16 - (value % 16)) % 16, "buffer alignment");
}

function operationDTypes(
  operation: string,
): { readonly inputs: readonly TrainingDTypeV1[]; readonly outputs: readonly TrainingDTypeV1[] } {
  switch (operation) {
    case "graph.causal_mask":
    case "graph.detach":
    case "graph.fsq":
    case "graph.relu2":
    case "graph.rope":
    case "graph.salt_ste":
    case "graph.scale_const":
    case "graph.silu":
    case "graph.slice_cols":
    case "graph.softmax":
    case "graph.transpose":
      return { inputs: ["f32"], outputs: ["f32"] };
    case "graph.add":
    case "graph.bias":
    case "graph.concat_cols":
    case "graph.dense_matmul":
    case "graph.lsq_ste":
    case "graph.mul":
    case "graph.rmsnorm":
    case "graph.ste_surrogate":
    case "loss.mse":
    case "loss.softmax_cross_entropy":
      return { inputs: ["f32", "f32"], outputs: ["f32"] };
    case "graph.attention":
    case "graph.conv1d":
    case "graph.conv2d":
    case "graph.ternary_matmul":
      return { inputs: ["f32", "f32", "f32"], outputs: ["f32"] };
    case "graph.embedding_gather":
      return { inputs: ["f32", "u32"], outputs: ["f32"] };
    case "optimizer.sgd":
      return { inputs: ["f32", "f32"], outputs: ["f32"] };
    case "optimizer.adamw":
    case "optimizer.cautious_adamw":
      return {
        inputs: ["f32", "f32", "f32", "f32"],
        outputs: ["f32", "f32", "f32"],
      };
    case "optimizer.int8_adamw":
      return {
        inputs: ["f32", "f32", "bytes", "bytes", "f32", "f32"],
        outputs: ["f32", "bytes", "bytes", "f32", "f32"],
      };
    case "optimizer.muon":
      return {
        inputs: ["f32", "f32", "f32"],
        outputs: ["f32", "f32"],
      };
    default:
      fail("invalid_schema", `operation ${operation} is not schedulable`);
  }
}

function sameShape(left: TrainingTensorSpecV1, right: TrainingTensorSpecV1): boolean {
  return (
    left.shape.length === right.shape.length &&
    left.shape.every((dimension, index) => dimension === right.shape[index])
  );
}

/** Compile and freeze the allocation/schedule plan before adapter allocation. */
export function compileTrainingPlan(
  model: WebTrainingModelV1,
  config: WebTrainingConfigV1,
): CompiledTrainingPlanV1 {
  validateModel(model);
  validateConfig(config);
  const descriptors = new Map(
    parseTrainingManifest(canonicalTrainingManifestJson()).operations.map(
      (operation) => [operation.id, operation] as const,
    ),
  );
  const tensors = new Map(
    model.recipe.tensors.map((tensor) => [tensor.id, tensor] as const),
  );
  const allocations = new Map<
    string,
    { readonly byteOffset: number; readonly byteLength: number }
  >();
  let residentBytes = 0;
  for (const tensor of model.recipe.tensors) {
    if (tensor.aliasOf !== null) continue;
    const byteOffset = align16(residentBytes);
    const byteLength = tensorByteLength(tensor);
    residentBytes = checkedAdd(byteOffset, byteLength, "resident buffer plan");
    allocations.set(tensor.id, { byteOffset, byteLength });
  }

  const buffers = model.recipe.tensors.map((tensor) => {
    const ownerId = tensor.aliasOf ?? tensor.id;
    const owner = tensors.get(ownerId);
    if (
      owner === undefined ||
      owner.aliasOf !== null ||
      (tensor.aliasOf !== null &&
        (tensor.role !== "parameter" ||
          owner.role !== "parameter" ||
          tensor.dtype !== owner.dtype ||
          tensor.shape.length !== owner.shape.length ||
          tensor.shape.some((dimension, index) => dimension !== owner.shape[index])))
    ) {
      fail("invalid_schema", `invalid allocation owner ${ownerId} for ${tensor.id}`);
    }
    const allocation = allocations.get(ownerId);
    if (allocation === undefined) {
      fail("invalid_schema", `missing allocation owner ${ownerId}`);
    }
    return Object.freeze({
      id: tensor.id,
      dtype: tensor.dtype,
      shape: Object.freeze([...tensor.shape]),
      role: tensor.role,
      aliasOf: tensor.aliasOf,
      ownerId,
      byteOffset: allocation.byteOffset,
      byteLength: allocation.byteLength,
    });
  });

  const defined = new Set(
    model.recipe.tensors
      .filter((tensor) =>
        tensor.role === "batch" ||
        tensor.role === "parameter" ||
        tensor.role === "optimizer-state",
      )
      .map((tensor) => tensor.id),
  );
  let phase = 0;
  const operations = model.recipe.operations.map((operation) => {
    const descriptor = descriptors.get(operation.operation);
    if (descriptor === undefined) {
      fail("invalid_schema", `unknown training operation ${operation.operation}`);
    }
    if (descriptor.category === "lifecycle") {
      fail("invalid_schema", "lifecycle operations are session methods, not recipe steps");
    }
    const operationPhase = descriptor.category === "graph" ? 0 : descriptor.category === "loss" ? 1 : 2;
    if (operationPhase < phase) {
      fail("invalid_schema", `${operation.id} violates graph/loss/optimizer phase order`);
    }
    phase = operationPhase;
    if (descriptor.category === "optimizer") {
      for (const tensor of model.recipe.tensors) {
        if (tensor.role === "gradient") defined.add(tensor.id);
      }
    }
    const signature = operationDTypes(operation.operation);
    if (
      operation.inputs.length !== signature.inputs.length ||
      operation.outputs.length !== signature.outputs.length
    ) {
      fail("invalid_schema", `${operation.id} arity does not match ${operation.operation}`);
    }
    const inputTensors = operation.inputs.map((tensorId, index) => {
      const tensor = tensors.get(tensorId);
      if (tensor === undefined || !defined.has(tensorId)) {
        fail("invalid_schema", `${operation.id} reads undefined tensor ${tensorId}`);
      }
      if (tensor.dtype !== signature.inputs[index]) {
        fail("invalid_schema", `${operation.id} input ${tensorId} has wrong dtype`);
      }
      return tensor;
    });
    const outputTensors = operation.outputs.map((tensorId, index) => {
      const tensor = tensors.get(tensorId);
      if (tensor === undefined) {
        fail("invalid_schema", `${operation.id} references unknown tensor ${tensorId}`);
      }
      if (tensor.dtype !== signature.outputs[index]) {
        fail("invalid_schema", `${operation.id} output ${tensorId} has wrong dtype`);
      }
      if (
        descriptor.category !== "optimizer" &&
        (tensor.role === "batch" ||
          tensor.role === "parameter" ||
          tensor.role === "gradient" ||
          tensor.role === "optimizer-state")
      ) {
        fail("invalid_schema", `${operation.id} illegally writes persistent tensor ${tensorId}`);
      }
      if (descriptor.category === "optimizer") {
        const expectedRole = index === 0 ? "parameter" : "optimizer-state";
        if (tensor.role !== expectedRole) {
          fail(
            "invalid_schema",
            `${operation.id} output ${tensorId} must be ${expectedRole}`,
          );
        }
      }
      if (
        descriptor.category === "optimizer" &&
        !operation.inputs.includes(tensorId)
      ) {
        fail("invalid_schema", `${operation.id} output ${tensorId} is not in-place state`);
      }
      if (!descriptor.mutates && defined.has(tensorId)) {
        fail("invalid_schema", `${operation.id} overwrites defined tensor ${tensorId}`);
      }
      return tensor;
    });
    if (
      ["graph.add", "graph.mul"].includes(operation.operation) &&
      (!sameShape(inputTensors[0]!, inputTensors[1]!) ||
        !sameShape(inputTensors[0]!, outputTensors[0]!))
    ) {
      fail("invalid_schema", `${operation.id} elementwise shapes differ`);
    }
    if (
      operation.operation === "loss.mse" &&
      (!sameShape(inputTensors[0]!, inputTensors[1]!) || outputTensors[0]!.shape.length !== 0)
    ) {
      fail("invalid_schema", `${operation.id} MSE shapes are invalid`);
    }
    if (
      operation.operation === "optimizer.sgd" &&
      (!sameShape(inputTensors[0]!, inputTensors[1]!) ||
        !sameShape(inputTensors[0]!, outputTensors[0]!))
    ) {
      fail("invalid_schema", `${operation.id} SGD shapes differ`);
    }
    for (const tensor of outputTensors) {
      defined.add(tensor.id);
    }
    return Object.freeze({
      id: operation.id,
      operation: operation.operation,
      inputs: Object.freeze([...operation.inputs]),
      outputs: Object.freeze([...operation.outputs]),
      attributes: Object.freeze(
        operation.attributes.map((attribute) =>
          Object.freeze({
            ...attribute,
            value: Array.isArray(attribute.value)
              ? Object.freeze([...attribute.value])
              : attribute.value,
          }),
        ),
      ),
    });
  });
  const parameterGroupsByOwner = new Map<string, string>();
  const claimedGradients = new Set<string>();
  const claimedOptimizerStates = new Set<string>();
  for (const operation of operations) {
    if (!operation.operation.startsWith("optimizer.")) continue;
    const parameter = tensors.get(operation.inputs[0]!);
    const gradient = tensors.get(operation.inputs[1]!);
    if (
      parameter === undefined ||
      parameter.role !== "parameter" ||
      parameter.aliasOf !== null ||
      operation.outputs[0] !== parameter.id
    ) {
      fail(
        "invalid_schema",
        `${operation.id} must update a canonical parameter owner in place`,
      );
    }
    if (
      gradient === undefined ||
      gradient.role !== "gradient" ||
      gradient.aliasOf !== null ||
      gradient.dtype !== parameter.dtype ||
      !sameShape(gradient, parameter)
    ) {
      fail(
        "invalid_schema",
        `${operation.id} gradient does not match parameter owner ${parameter.id}`,
      );
    }
    if (parameterGroupsByOwner.has(parameter.id)) {
      fail("invalid_schema", `parameter owner ${parameter.id} has multiple optimizers`);
    }
    if (claimedGradients.has(gradient.id)) {
      fail("invalid_schema", `gradient ${gradient.id} has multiple parameter owners`);
    }
    claimedGradients.add(gradient.id);
    const stateInputIds = operation.inputs.slice(2);
    const stateOutputIds = operation.outputs.slice(1);
    if (
      stateInputIds.length !== stateOutputIds.length ||
      stateInputIds.some((stateId, index) => stateOutputIds[index] !== stateId)
    ) {
      fail("invalid_schema", `${operation.id} optimizer state is not positionally in place`);
    }
    const parameterElements = tensorByteLength(parameter) / 4;
    for (const [index, stateId] of stateInputIds.entries()) {
      const state = tensors.get(stateId);
      if (
        state === undefined ||
        state.role !== "optimizer-state" ||
        state.aliasOf !== null
      ) {
        fail("invalid_schema", `${operation.id} has invalid optimizer state ${stateId}`);
      }
      const expectedScale =
        operation.operation === "optimizer.int8_adamw" && index >= 2;
      const shapeMatches = expectedScale
        ? state.shape.length === 1 &&
          state.shape[0] === Math.ceil(parameterElements / 256)
        : sameShape(state, parameter);
      if (!shapeMatches) {
        fail("invalid_schema", `${operation.id} optimizer state ${stateId} has wrong shape`);
      }
      if (claimedOptimizerStates.has(stateId)) {
        fail("invalid_schema", `optimizer state ${stateId} has multiple owners`);
      }
      claimedOptimizerStates.add(stateId);
    }
    parameterGroupsByOwner.set(parameter.id, operation.id);
  }
  const parameterOwners = model.recipe.tensors.filter(
    (tensor) => tensor.role === "parameter" && tensor.aliasOf === null,
  );
  for (const parameter of parameterOwners) {
    if (!parameterGroupsByOwner.has(parameter.id)) {
      fail("invalid_schema", `parameter owner ${parameter.id} has no optimizer`);
    }
  }
  for (const tensor of model.recipe.tensors) {
    if (tensor.role === "gradient" && !claimedGradients.has(tensor.id)) {
      fail("invalid_schema", `gradient ${tensor.id} has no parameter owner`);
    }
    if (
      tensor.role === "optimizer-state" &&
      !claimedOptimizerStates.has(tensor.id)
    ) {
      fail("invalid_schema", `optimizer state ${tensor.id} has no parameter owner`);
    }
  }
  const isolatedPayloadBytes = checkedMultiply(
    model.payload.byteLength,
    2,
    "validation and preparation payloads",
  );
  const preparePeakBytes = checkedAdd(
    residentBytes,
    isolatedPayloadBytes,
    "prepared model memory",
  );
  const batchStagingBytes = buffers
    .filter((buffer) => buffer.role === "batch")
    .reduce(
      (total, buffer) => checkedAdd(total, buffer.byteLength, "batch staging"),
      0,
    );
  const forwardPeakBytes = checkedAdd(
    residentBytes,
    batchStagingBytes,
    "forward memory",
  );
  const peakBytes = Math.max(preparePeakBytes, forwardPeakBytes);
  if (peakBytes > config.maxResidentBytes) {
    fail(
      "memory_limit",
      `compiled plan requires ${peakBytes} bytes; ceiling is ${config.maxResidentBytes}`,
    );
  }
  return Object.freeze({
    schemaId: "tritium.compiled_training_plan",
    schemaVersion: 1,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V1,
    buffers: Object.freeze(buffers),
    operations: Object.freeze(operations),
    residentBytes,
    batchStagingBytes,
    preparePeakBytes,
    forwardPeakBytes,
    peakBytes,
  });
}

function validateCapabilities(
  capabilities: WebTrainingCapabilitiesV1,
  config: WebTrainingConfigV1,
  recipe: TrainingRecipeV1,
): void {
  exactKeys(
    capabilities,
    CAPABILITY_KEYS,
    "capabilities",
    "capability_mismatch",
  );
  if (
    capabilities.schemaId !== "tritium.web_training_capabilities" ||
    capabilities.schemaVersion !== 1 ||
    capabilities.manifestDigest !== TRAINING_MANIFEST_DIGEST_V1 ||
    capabilities.vectorDigest !== TRAINING_VECTOR_DIGEST_V1 ||
    !(["webgpu", "wasm-fallback"] as const).includes(
      capabilities.implementation,
    ) ||
    typeof capabilities.buildId !== "string" ||
    capabilities.buildId.length === 0 ||
    !(capabilities.physicalDevice === null ||
      (typeof capabilities.physicalDevice === "string" &&
        capabilities.physicalDevice.length > 0))
  ) {
    fail("capability_mismatch", "adapter capability identity is invalid");
  }
  safeNonnegativeInteger(
    capabilities.maxResidentBytes,
    "adapter maxResidentBytes",
    "capability_mismatch",
  );
  if (capabilities.maxResidentBytes === 0) {
    fail("capability_mismatch", "adapter maxResidentBytes must be positive");
  }
  nonemptyUniqueStrings(
    capabilities.supportedOperations,
    "capabilities.supportedOperations",
  );
  if (
    config.backend === "webgpu" &&
    capabilities.implementation !== "webgpu"
  ) {
    fail("backend_policy", "backend webgpu cannot use a WASM adapter");
  }
  if (
    config.backend === "wasm" &&
    capabilities.implementation !== "wasm-fallback"
  ) {
    fail("backend_policy", "backend wasm cannot use a WebGPU adapter");
  }
  if (
    capabilities.implementation === "wasm-fallback" &&
    config.backend === "auto" &&
    !config.allowWasmFallback
  ) {
    fail("backend_policy", "automatic WASM fallback is disabled");
  }
  if (config.maxResidentBytes > capabilities.maxResidentBytes) {
    fail("memory_limit", "configured memory ceiling exceeds adapter capacity");
  }

  const supported = new Set(capabilities.supportedOperations);
  const canonical = new Set(
    parseTrainingManifest(canonicalTrainingManifestJson()).operations.map(
      (operation) => operation.id,
    ),
  );
  for (const operation of [
    ...config.requiredOperations,
    ...recipe.operations.map((operation) => operation.operation),
  ]) {
    if (!canonical.has(operation)) {
      fail("invalid_schema", `unknown training operation ${operation}`);
    }
    if (!supported.has(operation)) {
      fail("capability_mismatch", `adapter does not support ${operation}`);
    }
  }
}

function validateReceipt(
  receipt: WebTrainingReceiptV1,
  capabilities: WebTrainingCapabilitiesV1,
  expectedOperation: string,
  minimumResidentBytes: number,
  maxResidentBytes: number,
  expectedCompletedSteps: number | null,
): void {
  exactKeys(receipt, RECEIPT_KEYS, "receipt", "invalid_receipt");
  if (
    receipt.schemaId !== "tritium.web_training_receipt" ||
    receipt.schemaVersion !== 1 ||
    receipt.manifestDigest !== TRAINING_MANIFEST_DIGEST_V1 ||
    receipt.vectorDigest !== TRAINING_VECTOR_DIGEST_V1 ||
    receipt.implementation !== capabilities.implementation ||
    receipt.buildId !== capabilities.buildId ||
    receipt.physicalDevice !== capabilities.physicalDevice ||
    receipt.operation !== expectedOperation
  ) {
    fail("invalid_receipt", `invalid ${expectedOperation} receipt identity`);
  }
  safeNonnegativeInteger(
    receipt.completedSteps,
    "receipt.completedSteps",
    "invalid_receipt",
  );
  safeNonnegativeInteger(
    receipt.peakResidentBytes,
    "receipt.peakResidentBytes",
    "invalid_receipt",
  );
  if (
    receipt.peakResidentBytes < minimumResidentBytes ||
    receipt.peakResidentBytes > maxResidentBytes
  ) {
    fail("memory_limit", `${expectedOperation} exceeded the memory ceiling`);
  }
  if (
    expectedCompletedSteps !== null &&
    receipt.completedSteps !== expectedCompletedSteps
  ) {
    fail("invalid_receipt", `${expectedOperation} reported the wrong step count`);
  }
}

function snapshotReceipt(receipt: WebTrainingReceiptV1): WebTrainingReceiptV1 {
  exactKeys(receipt, RECEIPT_KEYS, "receipt", "invalid_receipt");
  return Object.freeze({
    schemaId: receipt.schemaId,
    schemaVersion: receipt.schemaVersion,
    implementation: receipt.implementation,
    manifestDigest: receipt.manifestDigest,
    vectorDigest: receipt.vectorDigest,
    buildId: receipt.buildId,
    physicalDevice: receipt.physicalDevice,
    operation: receipt.operation,
    completedSteps: receipt.completedSteps,
    peakResidentBytes: receipt.peakResidentBytes,
  });
}

function snapshotBinaryResult(
  result: WebBinaryResultV1,
  operation: string,
): WebBinaryResultV1 {
  exactKeys(result, ["bytes", "receipt"], `${operation} result`, "invalid_receipt");
  const bytes = result.bytes;
  const receipt = snapshotReceipt(result.receipt);
  if (!(bytes instanceof Uint8Array) || bytes.byteLength === 0) {
    fail("invalid_receipt", `${operation} returned an empty binary artifact`);
  }
  return Object.freeze({ bytes: Uint8Array.from(bytes), receipt });
}

function snapshotCapabilities(
  capabilities: WebTrainingCapabilitiesV1,
): WebTrainingCapabilitiesV1 {
  exactKeys(
    capabilities,
    CAPABILITY_KEYS,
    "capabilities",
    "capability_mismatch",
  );
  const supportedOperations = capabilities.supportedOperations;
  if (!Array.isArray(supportedOperations)) {
    fail("capability_mismatch", "supportedOperations must be an array");
  }
  const snapshot: WebTrainingCapabilitiesV1 = {
    schemaId: capabilities.schemaId,
    schemaVersion: capabilities.schemaVersion,
    implementation: capabilities.implementation,
    manifestDigest: capabilities.manifestDigest,
    vectorDigest: capabilities.vectorDigest,
    buildId: capabilities.buildId,
    physicalDevice: capabilities.physicalDevice,
    supportedOperations: Array.from(supportedOperations),
    maxResidentBytes: capabilities.maxResidentBytes,
  };
  return Object.freeze({
    ...snapshot,
    supportedOperations: Object.freeze(snapshot.supportedOperations),
  });
}

export class WebTrainingSession {
  readonly capabilities: WebTrainingCapabilitiesV1;
  readonly plan: CompiledTrainingPlanV1;
  readonly #adapter: WebTrainingAdapterV1;
  readonly #maxResidentBytes: number;
  #state: WebTrainingState = "prepared";
  #busy = false;
  #lastResult: TrainingResultV1 | null = null;
  #completedSteps = 0;

  private constructor(
    adapter: WebTrainingAdapterV1,
    maxResidentBytes: number,
    capabilities: WebTrainingCapabilitiesV1,
    plan: CompiledTrainingPlanV1,
  ) {
    this.#adapter = adapter;
    this.#maxResidentBytes = maxResidentBytes;
    this.capabilities = capabilities;
    this.plan = plan;
  }

  static async prepare(
    model: WebTrainingModelV1,
    config: WebTrainingConfigV1,
    adapter: WebTrainingAdapterV1,
  ): Promise<WebTrainingSession> {
    validateModel(model);
    validateConfig(config);
    const safeConfig: WebTrainingConfigV1 = Object.freeze({
      backend: config.backend,
      allowWasmFallback: config.allowWasmFallback,
      maxResidentBytes: config.maxResidentBytes,
      seed: config.seed,
      requiredOperations: Object.freeze([...config.requiredOperations]),
    });
    const plan = compileTrainingPlan(model, safeConfig);
    const capabilities = snapshotCapabilities(adapter.capabilities);
    validateCapabilities(capabilities, safeConfig, model.recipe);
    const capturedPayload = Uint8Array.from(model.payload);
    const safeModel: WebTrainingModelV1 = Object.freeze({
      schemaId: model.schemaId,
      schemaVersion: model.schemaVersion,
      recipe: Object.freeze({
        schemaId: model.recipe.schemaId,
        schemaVersion: model.recipe.schemaVersion,
        tensors: Object.freeze(
          model.recipe.tensors.map((tensor) =>
            Object.freeze({
              ...tensor,
              shape: Object.freeze([...tensor.shape]),
            }),
          ),
        ),
        operations: Object.freeze(
          model.recipe.operations.map((operation) =>
            Object.freeze({
              ...operation,
              inputs: Object.freeze([...operation.inputs]),
              outputs: Object.freeze([...operation.outputs]),
              attributes: Object.freeze(
                operation.attributes.map((attribute) =>
                  Object.freeze({
                    ...attribute,
                    value: Array.isArray(attribute.value)
                      ? Object.freeze([...attribute.value])
                      : attribute.value,
                  }),
                ),
              ),
            }),
          ),
        ),
      }),
      payload: Uint8Array.from(capturedPayload),
    });
    const validation = await adapter.validate(safeModel, safeConfig, plan);
    if (validation !== undefined) {
      fail("capability_mismatch", "adapter.validate must return undefined");
    }
    const prepareModel = Object.freeze({
      ...safeModel,
      payload: capturedPayload,
    });
    const receipt = snapshotReceipt(
      await adapter.prepare(prepareModel, safeConfig, plan),
    );
    validateReceipt(
      receipt,
      capabilities,
      "session.prepare",
      plan.residentBytes,
      safeConfig.maxResidentBytes - safeModel.payload.byteLength * 2,
      0,
    );
    return new WebTrainingSession(
      adapter,
      safeConfig.maxResidentBytes,
      capabilities,
      plan,
    );
  }

  get state(): WebTrainingState {
    return this.#state;
  }

  async #exclusive<T>(run: () => Promise<T>): Promise<T> {
    if (this.#state === "disposed") {
      fail("disposed", "training session is disposed", this.#state);
    }
    if (this.#busy) {
      fail("busy", "another session operation is in flight", this.#state);
    }
    this.#busy = true;
    try {
      return await run();
    } finally {
      this.#busy = false;
    }
  }

  #require(expected: WebTrainingState, operation: string): void {
    if (this.#state !== expected) {
      fail(
        "invalid_state",
        `${operation} requires ${expected}; current state is ${this.#state}`,
        this.#state,
      );
    }
  }

  async forward(batch: TrainingBatchV1): Promise<TrainingResultV1> {
    return this.#exclusive(async () => {
      this.#require("prepared", "forward");
      const safeBatch = validateAndCopyBatch(batch, this.plan, this.#state);
      const result = await this.#adapter.forward(safeBatch);
      exactKeys(result, ["loss", "receipt"], "forward result", "invalid_receipt");
      const safeResult = Object.freeze({
        loss: result.loss,
        receipt: snapshotReceipt(result.receipt),
      });
      if (!Number.isFinite(safeResult.loss)) {
        fail("invalid_receipt", "forward loss must be finite", this.#state);
      }
      validateReceipt(
        safeResult.receipt,
        this.capabilities,
        "session.forward",
        this.plan.residentBytes,
        this.#maxResidentBytes - this.plan.batchStagingBytes,
        this.#completedSteps,
      );
      this.#lastResult = safeResult;
      this.#state = "forward-complete";
      return safeResult;
    });
  }

  async backward(result: TrainingResultV1): Promise<WebTrainingReceiptV1> {
    return this.#exclusive(async () => {
      this.#require("forward-complete", "backward");
      if (result !== this.#lastResult) {
        fail("invalid_state", "backward result is not the active forward", this.#state);
      }
      const receipt = snapshotReceipt(await this.#adapter.backward(result));
      validateReceipt(
        receipt,
        this.capabilities,
        "session.backward",
        this.plan.residentBytes,
        this.#maxResidentBytes,
        this.#completedSteps,
      );
      this.#state = "backward-complete";
      return receipt;
    });
  }

  async step(): Promise<WebTrainingReceiptV1> {
    return this.#exclusive(async () => {
      this.#require("backward-complete", "step");
      const receipt = snapshotReceipt(await this.#adapter.step());
      validateReceipt(
        receipt,
        this.capabilities,
        "session.step",
        this.plan.residentBytes,
        this.#maxResidentBytes,
        this.#completedSteps + 1,
      );
      this.#completedSteps = receipt.completedSteps;
      this.#lastResult = null;
      this.#state = "prepared";
      return receipt;
    });
  }

  async checkpoint(): Promise<WebBinaryResultV1> {
    return this.#exclusive(async () => {
      this.#require("prepared", "checkpoint");
      const result = snapshotBinaryResult(
        await this.#adapter.checkpoint(),
        "checkpoint",
      );
      validateReceipt(
        result.receipt,
        this.capabilities,
        "session.checkpoint",
        this.plan.residentBytes,
        this.#maxResidentBytes,
        this.#completedSteps,
      );
      return result;
    });
  }

  async resume(checkpoint: Uint8Array): Promise<WebTrainingReceiptV1> {
    return this.#exclusive(async () => {
      this.#require("prepared", "resume");
      if (!(checkpoint instanceof Uint8Array) || checkpoint.byteLength === 0) {
        fail("invalid_schema", "checkpoint must not be empty", this.#state);
      }
      const receipt = snapshotReceipt(
        await this.#adapter.resume(Uint8Array.from(checkpoint)),
      );
      validateReceipt(
        receipt,
        this.capabilities,
        "session.resume",
        this.plan.residentBytes,
        this.#maxResidentBytes,
        null,
      );
      this.#completedSteps = receipt.completedSteps;
      return receipt;
    });
  }

  async export(): Promise<WebBinaryResultV1> {
    return this.#exclusive(async () => {
      this.#require("prepared", "export");
      const result = snapshotBinaryResult(
        await this.#adapter.export(),
        "export",
      );
      validateReceipt(
        result.receipt,
        this.capabilities,
        "session.export",
        this.plan.residentBytes,
        this.#maxResidentBytes,
        this.#completedSteps,
      );
      return result;
    });
  }

  async dispose(): Promise<void> {
    if (this.#state === "disposed") return;
    await this.#exclusive(async () => {
      await this.#adapter.dispose();
      this.#lastResult = null;
      this.#state = "disposed";
    });
  }
}

export async function prepareTraining(
  model: WebTrainingModelV1,
  config: WebTrainingConfigV1,
  adapter?: WebTrainingAdapterV1,
): Promise<WebTrainingSession> {
  if (adapter === undefined) {
    fail(
      "adapter_unavailable",
      "no generated WebGPU or WASM adapter was supplied",
    );
  }
  return WebTrainingSession.prepare(model, config, adapter);
}

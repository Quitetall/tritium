import {
  canonicalTrainingManifestJson,
  parseTrainingManifest,
} from "../../../bindings/typescript/src/training_manifest.ts";

import {
  TRAINING_MANIFEST_DIGEST_V2,
  TRAINING_VECTOR_DIGEST_V2,
} from "./identity.ts";
import {
  TrainingGeometryError,
  validateTrainingOperationGeometry,
} from "./geometry.ts";
import {
  compileSaltExportTargets,
  saltExportLayout,
  SaltExportError,
} from "./salt-export.ts";

export type WebTrainingBackendPolicyV1 = "auto" | "webgpu" | "wasm";
export type WebTrainingImplementationV1 = "webgpu" | "wasm-fallback";
export type WebTrainingState =
  | "preparing"
  | "prepared"
  | "forward-complete"
  | "backward-complete"
  | "terminal"
  | "disposed";

export type WebTrainingErrorCode =
  | "adapter_unavailable"
  | "backend_policy"
  | "busy"
  | "capability_mismatch"
  | "adapter_failure"
  | "cancelled"
  | "device_lost"
  | "disposed"
  | "invalid_config"
  | "attribute_value.indices.in_range"
  | "attribute_value.probabilities.finite_nonnegative"
  | "invalid_receipt"
  | "invalid_schema"
  | "invalid_state"
  | "memory_limit";

export class WebTrainingError extends Error {
  readonly code: WebTrainingErrorCode;
  readonly state: WebTrainingState | null;
  readonly failureReceipt: WebTrainingFailureReceiptV1 | null;

  constructor(
    code: WebTrainingErrorCode,
    message: string,
    state: WebTrainingState | null = null,
    failureReceipt: WebTrainingFailureReceiptV1 | null = null,
  ) {
    super(message);
    this.name = "WebTrainingError";
    this.code = code;
    this.state = state;
    this.failureReceipt = failureReceipt;
  }
}

export interface WebTrainingFailureReceiptV1 {
  readonly schemaId: "tritium.web_training_failure_receipt";
  readonly schemaVersion: 1;
  readonly implementation: WebTrainingImplementationV1;
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V2;
  readonly vectorDigest: typeof TRAINING_VECTOR_DIGEST_V2;
  readonly buildId: string;
  readonly physicalDevice: string | null;
  readonly operation: string;
  readonly completedSteps: number;
  readonly cause: "adapter_failure" | "cancelled" | "device_lost";
  readonly stateBefore: WebTrainingState;
  readonly stateAfter: WebTrainingState;
  readonly recoverable: boolean;
}

export interface WebTrainingOperationOptionsV1 {
  readonly signal?: AbortSignal;
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
  readonly backwardInitialization: "none" | "zero" | "one";
}

export interface CompiledTrainingOperationV1 extends TrainingOperationSpecV1 {}

export interface CompiledTrainingBindingV1 {
  readonly role: string;
  readonly bufferId: string;
}

export interface CompiledBackwardOperationV1 {
  readonly id: string;
  readonly sourceOperationId: string;
  readonly operation: string;
  readonly execution: "forward" | "vjp";
  readonly inputs: readonly CompiledTrainingBindingV1[];
  readonly outputs: readonly CompiledTrainingBindingV1[];
  readonly attributes: readonly TrainingAttributeSpecV1[];
}

export interface CompiledTrainingPlanV1 {
  readonly schemaId: "tritium.compiled_training_plan";
  readonly schemaVersion: 1;
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V2;
  readonly buffers: readonly CompiledTrainingBufferV1[];
  readonly operations: readonly CompiledTrainingOperationV1[];
  readonly backwardOperations: readonly CompiledBackwardOperationV1[];
  readonly residentBytes: number;
  readonly batchStagingBytes: number;
  readonly preparePeakBytes: number;
  readonly forwardPeakBytes: number;
  readonly exportPackageBytes: number;
  readonly exportPeakBytes: number;
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
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V2;
  readonly vectorDigest: typeof TRAINING_VECTOR_DIGEST_V2;
  readonly buildId: string;
  readonly physicalDevice: string | null;
  readonly supportedOperations: readonly string[];
  readonly maxResidentBytes: number;
}

export interface WebTrainingReceiptV1 {
  readonly schemaId: "tritium.web_training_receipt";
  readonly schemaVersion: 1;
  readonly implementation: WebTrainingImplementationV1;
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V2;
  readonly vectorDigest: typeof TRAINING_VECTOR_DIGEST_V2;
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
 * A recoverable typed rejection must happen before committed-state mutation.
 * Cancellation after phase submission must preserve parameter and optimizer
 * owners and permit exact retry from the last public state. Transient batch,
 * activation, result, and gradient owners may be discarded only when retry
 * deterministically reconstructs them before observation. Device loss must
 * reject with `WebTrainingError("device_lost", ...)`; the session makes that
 * state terminal.
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
  forward(batch: TrainingBatchV1, signal?: AbortSignal | null): Promise<TrainingResultV1>;
  backward(result: TrainingResultV1, signal?: AbortSignal | null): Promise<WebTrainingReceiptV1>;
  step(signal?: AbortSignal | null): Promise<WebTrainingReceiptV1>;
  checkpoint(signal?: AbortSignal | null): Promise<WebBinaryResultV1>;
  resume(checkpoint: Uint8Array, signal?: AbortSignal | null): Promise<WebTrainingReceiptV1>;
  export(signal?: AbortSignal | null): Promise<WebBinaryResultV1>;
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
  const copied = Object.create(null) as Record<
    string,
    Float32Array | Uint32Array | Uint8Array
  >;
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
  for (const operation of plan.operations) {
    if (operation.operation !== "loss.topk_knowledge_distillation") continue;
    const cols = operation.attributes.find((attribute) => attribute.name === "cols")?.value;
    const indices = copied[operation.inputs[1]!];
    const probabilities = copied[operation.inputs[2]!];
    if (
      typeof cols !== "number" ||
      !(indices instanceof Uint32Array) ||
      !(probabilities instanceof Float32Array)
    ) {
      fail(
        "invalid_schema",
        `${operation.id} sparse targets must be batch-owned u32/f32 tensors`,
        state,
      );
    }
    if (indices.some((index) => index >= cols)) {
      fail(
        "attribute_value.indices.in_range",
        `${operation.id} top-k index must be smaller than cols`,
        state,
      );
    }
    if (probabilities.some((probability) => !Number.isFinite(probability) || probability < 0)) {
      fail(
        "attribute_value.probabilities.finite_nonnegative",
        `${operation.id} top-k probabilities must be finite and nonnegative`,
        state,
      );
    }
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
      tensor.id.startsWith("__tritium.") ||
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
    case "loss.topk_knowledge_distillation":
      return { inputs: ["f32", "u32", "f32"], outputs: ["f32"] };
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

interface DifferentiationRuleV1 {
  readonly savedInputRoles: readonly string[];
  readonly gradientInputIndexes: readonly number[];
  readonly gradientOutputRoles: readonly string[];
}

function differentiationRule(operation: string): DifferentiationRuleV1 {
  switch (operation) {
    case "graph.ste_surrogate":
      return {
        savedInputRoles: ["weight", "scale"],
        gradientInputIndexes: [0, 1],
        gradientOutputRoles: ["grad_weight", "grad_scale"],
      };
    case "graph.salt_ste":
      return {
        savedInputRoles: ["weight"],
        gradientInputIndexes: [0],
        gradientOutputRoles: ["grad_weight"],
      };
    case "graph.lsq_ste":
      return {
        savedInputRoles: ["weight", "alpha"],
        gradientInputIndexes: [0, 1],
        gradientOutputRoles: ["grad_weight", "grad_alpha"],
      };
    case "graph.fsq":
      return {
        savedInputRoles: ["x"],
        gradientInputIndexes: [0],
        gradientOutputRoles: ["grad_x"],
      };
    case "graph.dense_matmul":
      return {
        savedInputRoles: ["x", "weight"],
        gradientInputIndexes: [0, 1],
        gradientOutputRoles: ["grad_x", "grad_weight"],
      };
    case "graph.ternary_matmul":
      return {
        savedInputRoles: ["activation", "weight", "scale"],
        gradientInputIndexes: [0, 1, 2],
        gradientOutputRoles: ["grad_activation", "grad_weight", "grad_scale"],
      };
    case "graph.transpose":
    case "graph.slice_cols":
    case "graph.scale_const":
    case "graph.causal_mask":
    case "graph.rope":
      return {
        savedInputRoles: [],
        gradientInputIndexes: [0],
        gradientOutputRoles: ["grad_x"],
      };
    case "graph.embedding_gather":
      return {
        savedInputRoles: ["weight", "tokens"],
        gradientInputIndexes: [0],
        gradientOutputRoles: ["grad_weight"],
      };
    case "graph.concat_cols":
      return {
        savedInputRoles: [],
        gradientInputIndexes: [0, 1],
        gradientOutputRoles: ["grad_part.0", "grad_part.1"],
      };
    case "graph.add":
      return {
        savedInputRoles: [],
        gradientInputIndexes: [0, 1],
        gradientOutputRoles: ["grad_left", "grad_right"],
      };
    case "graph.mul":
      return {
        savedInputRoles: ["left", "right"],
        gradientInputIndexes: [0, 1],
        gradientOutputRoles: ["grad_left", "grad_right"],
      };
    case "graph.bias":
      return {
        savedInputRoles: ["x", "bias"],
        gradientInputIndexes: [0, 1],
        gradientOutputRoles: ["grad_x", "grad_bias"],
      };
    case "graph.conv1d":
    case "graph.conv2d":
      return {
        savedInputRoles: ["x", "weight", "scale"],
        gradientInputIndexes: [0, 1, 2],
        gradientOutputRoles: ["grad_x", "grad_weight", "grad_scale"],
      };
    case "graph.relu2":
    case "graph.silu":
      return {
        savedInputRoles: ["x"],
        gradientInputIndexes: [0],
        gradientOutputRoles: ["grad_x"],
      };
    case "graph.rmsnorm":
      return {
        savedInputRoles: ["x", "weight"],
        gradientInputIndexes: [0, 1],
        gradientOutputRoles: ["grad_x", "grad_weight"],
      };
    case "graph.softmax":
      return {
        savedInputRoles: ["x"],
        gradientInputIndexes: [0],
        gradientOutputRoles: ["grad_x"],
      };
    case "graph.attention":
      return {
        savedInputRoles: ["q", "k", "v"],
        gradientInputIndexes: [0, 1, 2],
        gradientOutputRoles: ["grad_q", "grad_k", "grad_v"],
      };
    case "loss.mse":
      return {
        savedInputRoles: ["prediction", "target"],
        gradientInputIndexes: [0],
        gradientOutputRoles: ["grad_prediction"],
      };
    case "loss.softmax_cross_entropy":
      return {
        savedInputRoles: ["logits", "target"],
        gradientInputIndexes: [0],
        gradientOutputRoles: ["grad_logits"],
      };
    case "loss.topk_knowledge_distillation":
      return {
        savedInputRoles: ["logits", "indices", "probabilities"],
        gradientInputIndexes: [0],
        gradientOutputRoles: ["grad_logits"],
      };
    default:
      fail("invalid_schema", `operation ${operation} has no first-order VJP rule`);
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

  const buffers: CompiledTrainingBufferV1[] = model.recipe.tensors.map((tensor) => {
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
      backwardInitialization:
        tensor.role === "gradient" ? ("zero" as const) : ("none" as const),
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
    if (
      operation.operation === "loss.topk_knowledge_distillation" &&
      (inputTensors[1]!.role !== "batch" || inputTensors[2]!.role !== "batch")
    ) {
      fail(
        "invalid_schema",
        `${operation.id} sparse indices and probabilities must be batch tensors`,
      );
    }
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
    try {
      validateTrainingOperationGeometry(operation, inputTensors, outputTensors, config.seed);
    } catch (error) {
      if (error instanceof TrainingGeometryError) fail("invalid_schema", error.message);
      throw error;
    }
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

  const lossOperations = operations.filter((operation) =>
    operation.operation.startsWith("loss."),
  );
  if (lossOperations.length !== 1) {
    fail("invalid_schema", "a training recipe must contain exactly one loss operation");
  }
  const lossOutput = tensors.get(lossOperations[0]!.outputs[0]!);
  if (
    lossOutput === undefined ||
    lossOutput.role !== "result" ||
    lossOutput.dtype !== "f32" ||
    lossOutput.shape.length !== 0
  ) {
    fail("invalid_schema", "the training loss must be a scalar f32 result");
  }

  let internalBufferIndex = 0;
  const allocateInternal = (
    source: TrainingTensorSpecV1,
    purpose: string,
    initialization: "none" | "one" = "none",
  ): string => {
    const id = `__tritium.${purpose}.${internalBufferIndex}`;
    internalBufferIndex += 1;
    const byteOffset = align16(residentBytes);
    const byteLength = tensorByteLength(source);
    residentBytes = checkedAdd(byteOffset, byteLength, "resident buffer plan");
    buffers.push(
      Object.freeze({
        id,
        dtype: source.dtype,
        shape: Object.freeze([...source.shape]),
        role: "gradient",
        aliasOf: null,
        ownerId: id,
        byteOffset,
        byteLength,
        backwardInitialization: initialization,
      }),
    );
    return id;
  };

  const parameterGradientByOwner = new Map<string, string>();
  for (const operation of operations) {
    if (!operation.operation.startsWith("optimizer.")) continue;
    parameterGradientByOwner.set(operation.inputs[0]!, operation.inputs[1]!);
  }
  const gradientTargetKey = (tensorId: string): string => {
    const tensor = tensors.get(tensorId)!;
    return tensor.role === "parameter" ? tensor.aliasOf ?? tensor.id : tensor.id;
  };

  interface ActiveBackwardNode {
    readonly operation: CompiledTrainingOperationV1;
    readonly rule: DifferentiationRuleV1;
    readonly outputGradientKey: string;
    readonly contributionKeys: readonly string[];
  }
  const neededGradients = new Set<string>([lossOperations[0]!.outputs[0]!]);
  const activeBackwardNodes: ActiveBackwardNode[] = [];
  for (const operation of [...operations].reverse()) {
    if (operation.operation.startsWith("optimizer.")) continue;
    const outputId = operation.outputs[0]!;
    if (!neededGradients.has(outputId)) continue;
    if (operation.operation === "graph.detach") continue;
    const rule = differentiationRule(operation.operation);
    const contributionKeys = rule.gradientInputIndexes.map((inputIndex) => {
      const inputId = operation.inputs[inputIndex]!;
      const key = gradientTargetKey(inputId);
      neededGradients.add(key);
      return key;
    });
    activeBackwardNodes.push({
      operation,
      rule,
      outputGradientKey: gradientTargetKey(outputId),
      contributionKeys,
    });
  }
  for (const ownerId of parameterGradientByOwner.keys()) {
    if (!neededGradients.has(ownerId)) {
      fail(
        "invalid_schema",
        `optimized parameter owner ${ownerId} is disconnected from the loss`,
      );
    }
  }

  const contributionCounts = new Map<string, number>();
  for (const node of activeBackwardNodes) {
    for (const key of node.contributionKeys) {
      contributionCounts.set(key, (contributionCounts.get(key) ?? 0) + 1);
    }
  }
  const gradientBuffers = new Map<string, string>();
  const ensureGradientBuffer = (key: string): string => {
    const existing = gradientBuffers.get(key);
    if (existing !== undefined) return existing;
    const parameterGradient = parameterGradientByOwner.get(key);
    if (parameterGradient !== undefined) {
      gradientBuffers.set(key, parameterGradient);
      return parameterGradient;
    }
    const source = tensors.get(key);
    if (source === undefined || source.dtype !== "f32") {
      fail("invalid_schema", `cannot allocate gradient for ${key}`);
    }
    const allocated = allocateInternal(source, "gradient");
    gradientBuffers.set(key, allocated);
    return allocated;
  };

  const lossSeed = allocateInternal(lossOutput, "loss_seed", "one");
  gradientBuffers.set(lossOutput.id, lossSeed);
  const contributions = new Map<string, string[]>();
  const backwardOperations: CompiledBackwardOperationV1[] = [];
  let dispatchIndex = 0;
  for (const node of activeBackwardNodes) {
    const { operation, rule } = node;
    const outputBindings: CompiledTrainingBindingV1[] = [];
    for (const [index, key] of node.contributionKeys.entries()) {
      const total = contributionCounts.get(key)!;
      const source = tensors.get(key)!;
      const bufferId =
        total === 1
          ? ensureGradientBuffer(key)
          : allocateInternal(source, "contribution");
      const targetContributions = contributions.get(key) ?? [];
      targetContributions.push(bufferId);
      contributions.set(key, targetContributions);
      outputBindings.push(
        Object.freeze({
          role: rule.gradientOutputRoles[index]!,
          bufferId,
        }),
      );
    }
    const savedInputs = rule.savedInputRoles.map((role, index) =>
      Object.freeze({ role, bufferId: operation.inputs[index]! }),
    );
    backwardOperations.push(
      Object.freeze({
        id: `backward.${dispatchIndex}`,
        sourceOperationId: operation.id,
        operation: operation.operation,
        execution: "vjp",
        inputs: Object.freeze([
          ...savedInputs,
          Object.freeze({
            role: "grad_output",
            bufferId: ensureGradientBuffer(node.outputGradientKey),
          }),
        ]),
        outputs: Object.freeze(outputBindings),
        attributes: operation.attributes,
      }),
    );
    dispatchIndex += 1;

    for (const key of new Set(node.contributionKeys)) {
      const parts = contributions.get(key)!;
      if (parts.length !== contributionCounts.get(key) || parts.length < 2) continue;
      let accumulated = parts[0]!;
      for (let index = 1; index < parts.length; index += 1) {
        const last = index === parts.length - 1;
        const result = last
          ? ensureGradientBuffer(key)
          : allocateInternal(tensors.get(key)!, "accumulation");
        backwardOperations.push(
          Object.freeze({
            id: `backward.${dispatchIndex}`,
            sourceOperationId: operation.id,
            operation: "graph.add",
            execution: "forward",
            inputs: Object.freeze([
              Object.freeze({ role: "left", bufferId: accumulated }),
              Object.freeze({ role: "right", bufferId: parts[index]! }),
            ]),
            outputs: Object.freeze([
              Object.freeze({ role: "result", bufferId: result }),
            ]),
            attributes: Object.freeze([]),
          }),
        );
        dispatchIndex += 1;
        accumulated = result;
      }
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
  const provisionalPeakBytes = Math.max(preparePeakBytes, forwardPeakBytes);
  const provisionalPlan: CompiledTrainingPlanV1 = Object.freeze({
    schemaId: "tritium.compiled_training_plan",
    schemaVersion: 1,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V2,
    buffers: Object.freeze(buffers),
    operations: Object.freeze(operations),
    backwardOperations: Object.freeze(backwardOperations),
    residentBytes,
    batchStagingBytes,
    preparePeakBytes,
    forwardPeakBytes,
    exportPackageBytes: 0,
    exportPeakBytes: 0,
    peakBytes: provisionalPeakBytes,
  });
  let exportPackageBytes = 0;
  let exportPeakBytes = 0;
  try {
    const targets = compileSaltExportTargets(
      provisionalPlan,
      config.requiredOperations.includes("lifecycle.export"),
    );
    if (targets.length !== 0) {
      const layout = saltExportLayout(targets);
      exportPackageBytes = layout.packageBytes;
      const fitBytes = checkedAdd(
        checkedMultiply(layout.packageBytes, 2, "SALT export fit"),
        layout.maxFitScratchBytes,
        "SALT export fit",
      );
      const admissionBytes = checkedAdd(
        checkedAdd(
          checkedMultiply(layout.packageBytes, 6, "SALT export admission"),
          checkedMultiply(layout.semanticBytes, 2, "SALT export admission"),
          "SALT export admission",
        ),
        64 * 1024,
        "SALT export admission",
      );
      exportPeakBytes = checkedAdd(
        residentBytes,
        Math.max(fitBytes, admissionBytes),
        "SALT export peak",
      );
    }
  } catch (error) {
    if (error instanceof SaltExportError) {
      fail(error.code === "capacity" ? "memory_limit" : "invalid_schema", error.message);
    }
    throw error;
  }
  const peakBytes = Math.max(preparePeakBytes, forwardPeakBytes, exportPeakBytes);
  if (peakBytes > config.maxResidentBytes) {
    fail(
      "memory_limit",
      `compiled plan requires ${peakBytes} bytes; ceiling is ${config.maxResidentBytes}`,
    );
  }
  const plan: CompiledTrainingPlanV1 = Object.freeze({
    ...provisionalPlan,
    exportPackageBytes,
    exportPeakBytes,
    peakBytes,
  });
  return plan;
}

function validateCapabilities(
  capabilities: WebTrainingCapabilitiesV1,
  config: WebTrainingConfigV1,
  recipe: TrainingRecipeV1,
  plan: CompiledTrainingPlanV1,
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
    capabilities.manifestDigest !== TRAINING_MANIFEST_DIGEST_V2 ||
    capabilities.vectorDigest !== TRAINING_VECTOR_DIGEST_V2 ||
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
    ...plan.backwardOperations.map((operation) => operation.operation),
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
    receipt.manifestDigest !== TRAINING_MANIFEST_DIGEST_V2 ||
    receipt.vectorDigest !== TRAINING_VECTOR_DIGEST_V2 ||
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

function failureReceipt(
  capabilities: WebTrainingCapabilitiesV1,
  operation: string,
  completedSteps: number,
  cause: "adapter_failure" | "cancelled" | "device_lost",
  stateBefore: WebTrainingState,
  stateAfter: WebTrainingState,
): WebTrainingFailureReceiptV1 {
  return Object.freeze({
    schemaId: "tritium.web_training_failure_receipt",
    schemaVersion: 1,
    implementation: capabilities.implementation,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V2,
    vectorDigest: TRAINING_VECTOR_DIGEST_V2,
    buildId: capabilities.buildId,
    physicalDevice: capabilities.physicalDevice,
    operation,
    completedSteps,
    cause,
    stateBefore,
    stateAfter,
    recoverable: cause === "cancelled",
  });
}

function operationSignal(
  options: WebTrainingOperationOptionsV1 | undefined,
): AbortSignal | null {
  if (options === undefined) return null;
  if (typeof options !== "object" || options === null || Array.isArray(options)) {
    fail("invalid_schema", "operation options must be an object");
  }
  const keys = Object.keys(options);
  if (keys.length > 1 || (keys.length === 1 && keys[0] !== "signal")) {
    fail("invalid_schema", "operation options contain unknown fields");
  }
  const signal = options.signal;
  if (signal === undefined) return null;
  if (
    typeof signal !== "object" ||
    signal === null ||
    typeof signal.aborted !== "boolean" ||
    typeof signal.addEventListener !== "function" ||
    typeof signal.removeEventListener !== "function"
  ) {
    fail("invalid_schema", "operation signal must be an AbortSignal");
  }
  return signal;
}

function signalAborted(signal: AbortSignal | null): boolean {
  return signal !== null && signal.aborted;
}

function adapterFailureCause(
  error: unknown,
): "cancelled" | "device_lost" | null {
  if (typeof error !== "object" || error === null) return null;
  try {
    const code = Reflect.get(error, "code");
    return code === "cancelled" || code === "device_lost" ? code : null;
  } catch {
    return null;
  }
}

function errorMessage(error: unknown, fallback: string): string {
  if (typeof error !== "object" || error === null) return fallback;
  try {
    const message = Reflect.get(error, "message");
    return typeof message === "string" ? message : fallback;
  } catch {
    return fallback;
  }
}

function isAbortError(error: unknown): boolean {
  if (typeof error !== "object" || error === null) return false;
  try {
    return Reflect.get(error, "name") === "AbortError";
  } catch {
    return false;
  }
}

class PostDispatchAdmissionError extends Error {
  readonly original: unknown;

  constructor(original: unknown) {
    super(errorMessage(original, "adapter result failed admission"));
    this.name = "PostDispatchAdmissionError";
    this.original = original;
  }
}

function admitPostDispatch<T>(admit: () => T): T {
  try {
    return admit();
  } catch (error) {
    throw new PostDispatchAdmissionError(error);
  }
}

function recoverableAdapterErrorCode(error: unknown): WebTrainingErrorCode | null {
  if (typeof error !== "object" || error === null) return null;
  try {
    const code = Reflect.get(error, "code");
    return code === "capability_mismatch" ||
      code === "invalid_config" ||
      code === "invalid_schema" ||
      code === "invalid_state" ||
      code === "memory_limit"
      ? code
      : null;
  } catch {
    return null;
  }
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
  #adapterDisposed = false;

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
    validateCapabilities(capabilities, safeConfig, model.recipe, plan);
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
    const prepareModel = Object.freeze({
      ...safeModel,
      payload: capturedPayload,
    });
    let validation: void;
    try {
      validation = await adapter.validate(safeModel, safeConfig, plan);
    } catch (error) {
      if (adapterFailureCause(error) !== "device_lost") throw error;
      try {
        await adapter.dispose();
      } catch {
        // Device loss is authoritative; cleanup failure cannot replace it.
      }
      const failed = failureReceipt(
        capabilities,
        "session.validate",
        0,
        "device_lost",
        "preparing",
        "terminal",
      );
      throw new WebTrainingError(
        "device_lost",
        errorMessage(error, "device lost during validation"),
        "terminal",
        failed,
      );
    }
    if (validation !== undefined) {
      fail("capability_mismatch", "adapter.validate must return undefined");
    }

    let receipt: WebTrainingReceiptV1;
    try {
      receipt = snapshotReceipt(await adapter.prepare(prepareModel, safeConfig, plan));
      validateReceipt(
        receipt,
        capabilities,
        "session.prepare",
        plan.preparePeakBytes,
        safeConfig.maxResidentBytes,
        0,
      );
    } catch (error) {
      try {
        await adapter.dispose();
      } catch {
        // Preparation's primary failure remains authoritative.
      }
      if (adapterFailureCause(error) !== "device_lost") throw error;
      const failed = failureReceipt(
        capabilities,
        "session.prepare",
        0,
        "device_lost",
        "preparing",
        "terminal",
      );
      throw new WebTrainingError(
        "device_lost",
        errorMessage(error, "device lost during preparation"),
        "terminal",
        failed,
      );
    }
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

  #failureReceipt(
    operation: string,
    cause: "adapter_failure" | "cancelled" | "device_lost",
    stateBefore: WebTrainingState,
    stateAfter: WebTrainingState,
  ): WebTrainingFailureReceiptV1 {
    return failureReceipt(
      this.capabilities,
      operation,
      this.#completedSteps,
      cause,
      stateBefore,
      stateAfter,
    );
  }

  async #disposeAdapter(permanent: boolean): Promise<void> {
    if (this.#adapterDisposed) return;
    if (permanent) this.#adapterDisposed = true;
    try {
      await this.#adapter.dispose();
      this.#adapterDisposed = true;
    } catch (error) {
      if (!permanent) this.#adapterDisposed = false;
      throw error;
    }
  }

  async #adapterTransaction<T>(
    operation: string,
    signal: AbortSignal | null,
    run: (signal: AbortSignal | null) => Promise<T>,
  ): Promise<T> {
    const stateBefore = this.#state;
    if (signalAborted(signal)) {
      const receipt = this.#failureReceipt(
        operation,
        "cancelled",
        stateBefore,
        stateBefore,
      );
      throw new WebTrainingError(
        "cancelled",
        `${operation} was cancelled before dispatch`,
        stateBefore,
        receipt,
      );
    }
    try {
      return await run(signal);
    } catch (error) {
      const recoverableCode =
        error instanceof PostDispatchAdmissionError
          ? null
          : recoverableAdapterErrorCode(error);
      if (recoverableCode !== null) {
        throw new WebTrainingError(
          recoverableCode,
          errorMessage(error, `${operation} was rejected before mutation`),
          stateBefore,
        );
      }
      const classified = adapterFailureCause(error);
      const cause =
        classified ??
        (signalAborted(signal) && isAbortError(error) ? "cancelled" : "adapter_failure");
      const message =
        error instanceof PostDispatchAdmissionError
          ? errorMessage(error.original, `${operation} result failed admission`)
          : errorMessage(error, `${operation} failed`);
      if (cause !== "cancelled") {
        this.#lastResult = null;
        this.#state = "terminal";
        try {
          await this.#disposeAdapter(true);
        } catch {
          // Terminal failure is authoritative; cleanup failure cannot replace it.
        }
      }
      const stateAfter = this.#state;
      const receipt = this.#failureReceipt(operation, cause, stateBefore, stateAfter);
      throw new WebTrainingError(cause, message, stateAfter, receipt);
    }
  }

  #rejectPreDispatchCancellation(
    operation: string,
    signal: AbortSignal | null,
  ): void {
    if (!signalAborted(signal)) return;
    const receipt = this.#failureReceipt(
      operation,
      "cancelled",
      this.#state,
      this.#state,
    );
    throw new WebTrainingError(
      "cancelled",
      `${operation} was cancelled before dispatch`,
      this.#state,
      receipt,
    );
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

  async forward(
    batch: TrainingBatchV1,
    options?: WebTrainingOperationOptionsV1,
  ): Promise<TrainingResultV1> {
    return this.#exclusive(async () => {
      this.#require("prepared", "forward");
      const signal = operationSignal(options);
      this.#rejectPreDispatchCancellation("session.forward", signal);
      const safeBatch = validateAndCopyBatch(batch, this.plan, this.#state);
      return this.#adapterTransaction(
        "session.forward",
        signal,
        async (admittedSignal) => {
          const result = await this.#adapter.forward(safeBatch, admittedSignal);
          return admitPostDispatch(() => {
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
              this.plan.forwardPeakBytes,
              this.#maxResidentBytes,
              this.#completedSteps,
            );
            this.#lastResult = safeResult;
            this.#state = "forward-complete";
            return safeResult;
          });
        },
      );
    });
  }

  async backward(
    result: TrainingResultV1,
    options?: WebTrainingOperationOptionsV1,
  ): Promise<WebTrainingReceiptV1> {
    return this.#exclusive(async () => {
      this.#require("forward-complete", "backward");
      const signal = operationSignal(options);
      this.#rejectPreDispatchCancellation("session.backward", signal);
      if (result !== this.#lastResult) {
        fail("invalid_state", "backward result is not the active forward", this.#state);
      }
      return this.#adapterTransaction(
        "session.backward",
        signal,
        async (admittedSignal) => {
          const rawReceipt = await this.#adapter.backward(result, admittedSignal);
          return admitPostDispatch(() => {
            const receipt = snapshotReceipt(rawReceipt);
            validateReceipt(
              receipt,
              this.capabilities,
              "session.backward",
              this.plan.peakBytes,
              this.#maxResidentBytes,
              this.#completedSteps,
            );
            this.#state = "backward-complete";
            return receipt;
          });
        },
      );
    });
  }

  async step(options?: WebTrainingOperationOptionsV1): Promise<WebTrainingReceiptV1> {
    return this.#exclusive(async () => {
      this.#require("backward-complete", "step");
      const signal = operationSignal(options);
      return this.#adapterTransaction(
        "session.step",
        signal,
        async (admittedSignal) => {
          const rawReceipt = await this.#adapter.step(admittedSignal);
          return admitPostDispatch(() => {
            const receipt = snapshotReceipt(rawReceipt);
            validateReceipt(
              receipt,
              this.capabilities,
              "session.step",
              this.plan.peakBytes,
              this.#maxResidentBytes,
              this.#completedSteps + 1,
            );
            this.#completedSteps = receipt.completedSteps;
            this.#lastResult = null;
            this.#state = "prepared";
            return receipt;
          });
        },
      );
    });
  }

  async checkpoint(
    options?: WebTrainingOperationOptionsV1,
  ): Promise<WebBinaryResultV1> {
    return this.#exclusive(async () => {
      this.#require("prepared", "checkpoint");
      const signal = operationSignal(options);
      return this.#adapterTransaction(
        "session.checkpoint",
        signal,
        async (admittedSignal) => {
          const rawResult = await this.#adapter.checkpoint(admittedSignal);
          return admitPostDispatch(() => {
            const result = snapshotBinaryResult(rawResult, "checkpoint");
            validateReceipt(
              result.receipt,
              this.capabilities,
              "session.checkpoint",
              this.plan.peakBytes,
              this.#maxResidentBytes,
              this.#completedSteps,
            );
            return result;
          });
        },
      );
    });
  }

  async resume(
    checkpoint: Uint8Array,
    options?: WebTrainingOperationOptionsV1,
  ): Promise<WebTrainingReceiptV1> {
    return this.#exclusive(async () => {
      this.#require("prepared", "resume");
      const signal = operationSignal(options);
      this.#rejectPreDispatchCancellation("session.resume", signal);
      if (!(checkpoint instanceof Uint8Array) || checkpoint.byteLength === 0) {
        fail("invalid_schema", "checkpoint must not be empty", this.#state);
      }
      return this.#adapterTransaction(
        "session.resume",
        signal,
        async (admittedSignal) => {
          const rawReceipt = await this.#adapter.resume(
            Uint8Array.from(checkpoint),
            admittedSignal,
          );
          return admitPostDispatch(() => {
            const receipt = snapshotReceipt(rawReceipt);
            validateReceipt(
              receipt,
              this.capabilities,
              "session.resume",
              this.plan.peakBytes,
              this.#maxResidentBytes,
              null,
            );
            this.#completedSteps = receipt.completedSteps;
            return receipt;
          });
        },
      );
    });
  }

  async export(options?: WebTrainingOperationOptionsV1): Promise<WebBinaryResultV1> {
    return this.#exclusive(async () => {
      this.#require("prepared", "export");
      const signal = operationSignal(options);
      return this.#adapterTransaction(
        "session.export",
        signal,
        async (admittedSignal) => {
          const rawResult = await this.#adapter.export(admittedSignal);
          return admitPostDispatch(() => {
            const result = snapshotBinaryResult(rawResult, "export");
            validateReceipt(
              result.receipt,
              this.capabilities,
              "session.export",
              this.plan.exportPeakBytes,
              this.#maxResidentBytes,
              this.#completedSteps,
            );
            return result;
          });
        },
      );
    });
  }

  async dispose(): Promise<void> {
    if (this.#state === "disposed") return;
    await this.#exclusive(async () => {
      this.#lastResult = null;
      this.#state = "terminal";
      await this.#disposeAdapter(false);
      this.#state = "disposed";
    });
  }
}

/** Read one host property without trusting browser/proxy getters. */
function capturedMember(value: object, name: PropertyKey): unknown {
  try {
    return Reflect.get(value, name);
  } catch {
    return undefined;
  }
}

async function requestDefaultWebGpuAdapter(): Promise<WebTrainingAdapterV1 | null> {
  const navigatorValue = capturedMember(globalThis, "navigator");
  if (typeof navigatorValue !== "object" || navigatorValue === null) return null;
  const gpu = capturedMember(navigatorValue, "gpu");
  if (typeof gpu !== "object" || gpu === null) return null;
  const requestAdapter = capturedMember(gpu, "requestAdapter");
  if (typeof requestAdapter !== "function") return null;
  const physicalAdapter = await Reflect.apply(requestAdapter, gpu, [{
    powerPreference: "high-performance",
  }]);
  if (typeof physicalAdapter !== "object" || physicalAdapter === null) return null;
  const requestDevice = capturedMember(physicalAdapter, "requestDevice");
  if (typeof requestDevice !== "function") return null;
  const device = await Reflect.apply(requestDevice, physicalAdapter, []);
  if (typeof device !== "object" || device === null) {
    fail("adapter_unavailable", "WebGPU requestDevice returned no device");
  }
  try {
    const { createWebGpuTrainingAdapter } = await import("./webgpu-adapter.ts");
    return createWebGpuTrainingAdapter(
      device as Parameters<typeof createWebGpuTrainingAdapter>[0],
    );
  } catch (error) {
    const destroy = capturedMember(device, "destroy");
    if (typeof destroy === "function") {
      try {
        Reflect.apply(destroy, device, []);
      } catch {
        // Factory admission failed; preserve that primary error.
      }
    }
    throw error;
  }
}

export async function prepareTraining(
  model: WebTrainingModelV1,
  config: WebTrainingConfigV1,
  adapter?: WebTrainingAdapterV1,
): Promise<WebTrainingSession> {
  validateModel(model);
  validateConfig(config);
  let ownsAutomaticAdapter = false;
  if (adapter === undefined) {
    if (config.backend !== "wasm") {
      try {
        adapter = await requestDefaultWebGpuAdapter() ?? undefined;
        ownsAutomaticAdapter = adapter !== undefined;
      } catch (error) {
        if (config.backend === "webgpu" || !config.allowWasmFallback) {
          if (error instanceof WebTrainingError) throw error;
          fail(
            "adapter_unavailable",
            `WebGPU device could not be created: ${
              error instanceof Error ? error.message : "unknown failure"
            }`,
          );
        }
      }
    }
    if (adapter === undefined &&
        (config.backend === "wasm" ||
          (config.backend === "auto" && config.allowWasmFallback))) {
      const plan = compileTrainingPlan(model, config);
      const {
        createPortableWasmTrainingAdapter,
        validatePortableWasmPlan,
      } = await import("./wasm-adapter.ts");
      validatePortableWasmPlan(plan);
      try {
        adapter = await createPortableWasmTrainingAdapter();
      } catch (error) {
        if (error instanceof WebTrainingError) throw error;
        fail(
          "adapter_unavailable",
          `portable WASM guest could not be created: ${
            error instanceof Error ? error.message : "unknown failure"
          }`,
        );
      }
    } else if (adapter === undefined) {
      fail(
        "adapter_unavailable",
        "no WebGPU adapter or device is available",
      );
    }
  }
  try {
    return await WebTrainingSession.prepare(model, config, adapter);
  } catch (error) {
    if (ownsAutomaticAdapter) {
      try {
        await adapter.dispose();
      } catch {
        // Admission failure is authoritative; cleanup cannot replace it.
      }
    }
    throw error;
  }
}

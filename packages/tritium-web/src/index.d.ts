import type {
  PortableTrainingRequestV1,
  PortableTrainingResponseV1,
  PortableWasmConformanceReceiptV1,
  PortableWasmSourceV1,
} from "./portable.js";
import type {
  PortableCheckpointOptimizerV1,
  PortableCheckpointStateV1,
} from "./lifecycle-types.js";

export type {
  PortableAdamLeafV1,
  PortableCheckpointOptimizerV1,
  PortableCheckpointStateV1,
  PortableInt8AdamLeafV1,
  PortableMuonLeafV1,
  PortableSgdLeafV1,
} from "./lifecycle-types.js";

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

export type TrainingOpCategoryV1 =
  | "graph"
  | "loss"
  | "optimizer"
  | "lifecycle";
export type TrainingVjpV1 = "none" | "first_order";

export interface TrainingOpDescriptorV1 {
  readonly id: string;
  readonly category: TrainingOpCategoryV1;
  readonly forward: boolean;
  readonly vjp: TrainingVjpV1;
  readonly mutates: boolean;
  readonly checkpoint_planes: readonly string[];
}

export interface TrainingOpManifestV1 {
  readonly schema_id: "tritium.training_op_manifest";
  readonly schema_version: 1;
  readonly dtype: "f32";
  readonly operations: readonly TrainingOpDescriptorV1[];
}

export declare class TrainingManifestError extends Error {
  constructor(message: string);
}

export declare function canonicalTrainingManifestJson(): Uint8Array;
export declare function parseTrainingManifest(
  data: string | Uint8Array,
): TrainingOpManifestV1;

export declare const TRAINING_MANIFEST_DIGEST_V1:
  "aefb352d04db145e48394b392a106ab0ad831e09e62d8c76ceddedb36a564083";
export declare const TRAINING_VECTOR_DIGEST_V1:
  "fcb250733b991aac165871f8c54b0b063337a3ed01bd1da02de220916887fbd6";

export declare function runPortableWasmConformance(
  source?: PortableWasmSourceV1,
): Promise<PortableWasmConformanceReceiptV1>;

export declare function executePortableWasmRequest(
  request: PortableTrainingRequestV1,
  source?: PortableWasmSourceV1,
): Promise<PortableTrainingResponseV1>;

export declare class PortableLifecyclePlanError extends Error {
  readonly code: "invalid_schema" | "capacity";
  constructor(code: "invalid_schema" | "capacity", message: string);
}

export declare function compilePortableCheckpointRequest(
  state: PortableCheckpointStateV1,
  physicalDevice?: string,
): PortableTrainingRequestV1;

export declare function compilePortableResumeRequest(
  optimizer: PortableCheckpointOptimizerV1,
  leafLengths: readonly number[],
  checkpoint: Uint8Array,
  physicalDevice?: string,
): PortableTrainingRequestV1;

export declare function compilePortableExportRequest(
  packageBytes: Uint8Array,
  physicalDevice?: string,
): PortableTrainingRequestV1;

export declare function compilePortableReloadRequest(
  artifact: Uint8Array,
  physicalDevice?: string,
): PortableTrainingRequestV1;

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

export declare class WebTrainingError extends Error {
  readonly code: WebTrainingErrorCode;
  readonly state: WebTrainingState | null;
  constructor(
    code: WebTrainingErrorCode,
    message: string,
    state?: WebTrainingState | null,
  );
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

/** Low-level generated adapter. `validate` is allocation-free; neither
 * `validate` nor `prepare` may mutate or retain their arguments.
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

export declare class WebTrainingSession {
  readonly capabilities: WebTrainingCapabilitiesV1;
  readonly plan: CompiledTrainingPlanV1;
  static prepare(
    model: WebTrainingModelV1,
    config: WebTrainingConfigV1,
    adapter: WebTrainingAdapterV1,
  ): Promise<WebTrainingSession>;
  get state(): WebTrainingState;
  forward(batch: TrainingBatchV1): Promise<TrainingResultV1>;
  backward(result: TrainingResultV1): Promise<WebTrainingReceiptV1>;
  step(): Promise<WebTrainingReceiptV1>;
  checkpoint(): Promise<WebBinaryResultV1>;
  resume(checkpoint: Uint8Array): Promise<WebTrainingReceiptV1>;
  export(): Promise<WebBinaryResultV1>;
  dispose(): Promise<void>;
}

export declare function compileTrainingPlan(
  model: WebTrainingModelV1,
  config: WebTrainingConfigV1,
): CompiledTrainingPlanV1;

export declare function prepareTraining(
  model: WebTrainingModelV1,
  config: WebTrainingConfigV1,
  adapter?: WebTrainingAdapterV1,
): Promise<WebTrainingSession>;

import type {
  PortableTrainingRequestV1,
  PortableTrainingReceiptV1,
  PortableTrainingResponseV1,
  PortableWasmConformanceReceiptV1,
  PortableWasmSourceV1,
} from "./portable.js";
import type {
  PortableCheckpointOptimizerV1,
  PortableCheckpointStateV1,
} from "./lifecycle-types.js";
import type {
  PortableWasmLifecycleBinaryV1,
  PortableWasmLifecycleErrorCode,
  PortableWasmLifecycleOptionsV1,
} from "./portable-state-types.js";
import type {
  PortableCompiledDispatchV1,
  PortableSchedulePlanErrorCode,
  PortableScheduleTensorStoreV1,
} from "./portable-schedule-types.js";
import type {
  WebTrainingInitialTensorsV1,
  WebTrainingPayloadErrorCode,
} from "./payload-types.js";

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

export type {
  PortableWasmLifecycleBinaryV1,
  PortableWasmLifecycleErrorCode,
  PortableWasmLifecycleOptionsV1,
  PortableWasmLifecycleStateV1,
} from "./portable-state-types.js";

export type {
  PortableCompiledDispatchV1,
  PortableSchedulePlanErrorCode,
  PortableScheduleTensorStoreV1,
  PortableScheduleTensorV1,
} from "./portable-schedule-types.js";

export type {
  WebTrainingInitialTensorsV1,
  WebTrainingPayloadErrorCode,
} from "./payload-types.js";

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

export declare function createPortableWasmTrainingAdapter(
  source?: PortableWasmSourceV1,
): Promise<WebTrainingAdapterV1>;

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

export declare class PortableWasmLifecycleError extends Error {
  readonly code: PortableWasmLifecycleErrorCode;
  constructor(code: PortableWasmLifecycleErrorCode, message: string);
}

export declare class PortableWasmLifecycleState {
  static create(
    options: PortableWasmLifecycleOptionsV1,
  ): Promise<PortableWasmLifecycleState>;
  get state(): PortableCheckpointStateV1;
  checkpoint(): Promise<PortableWasmLifecycleBinaryV1>;
  commit(state: PortableCheckpointStateV1): Promise<PortableTrainingReceiptV1>;
  resume(checkpoint: Uint8Array): Promise<PortableTrainingReceiptV1>;
  admitExport(packageBytes: Uint8Array): Promise<PortableWasmLifecycleBinaryV1>;
  dispose(): void;
}

export declare class PortableSchedulePlanError extends Error {
  readonly code: PortableSchedulePlanErrorCode;
  constructor(code: PortableSchedulePlanErrorCode, message: string);
}

export declare function compilePortablePlanOperationRequest(
  plan: CompiledTrainingPlanV1,
  operationId: string,
  store: PortableScheduleTensorStoreV1,
  physicalDevice?: string,
): PortableCompiledDispatchV1;

export declare function compilePortableBackwardOperationRequest(
  plan: CompiledTrainingPlanV1,
  operationId: string,
  store: PortableScheduleTensorStoreV1,
  physicalDevice?: string,
): PortableCompiledDispatchV1;

export declare class WebTrainingPayloadError extends Error {
  readonly code: WebTrainingPayloadErrorCode;
  constructor(code: WebTrainingPayloadErrorCode, message: string);
}

export declare function encodeWebTrainingPayload(
  tensors: WebTrainingInitialTensorsV1,
): Uint8Array;

export declare function decodeWebTrainingPayload(
  plan: CompiledTrainingPlanV1,
  payload: Uint8Array,
): PortableScheduleTensorStoreV1;

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
  | "invalid_receipt"
  | "invalid_schema"
  | "invalid_state"
  | "memory_limit";

export declare class WebTrainingError extends Error {
  readonly code: WebTrainingErrorCode;
  readonly state: WebTrainingState | null;
  readonly failureReceipt: WebTrainingFailureReceiptV1 | null;
  constructor(
    code: WebTrainingErrorCode,
    message: string,
    state?: WebTrainingState | null,
    failureReceipt?: WebTrainingFailureReceiptV1 | null,
  );
}

export interface WebTrainingFailureReceiptV1 {
  readonly schemaId: "tritium.web_training_failure_receipt";
  readonly schemaVersion: 1;
  readonly implementation: WebTrainingImplementationV1;
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V1;
  readonly vectorDigest: typeof TRAINING_VECTOR_DIGEST_V1;
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
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V1;
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
 * `validate` nor `prepare` may mutate or retain their arguments. Recoverable
 * typed rejections happen before mutation. Cancellation
 * must roll back partial writes before rejecting with `cancelled`; device loss
 * rejects with `device_lost` and is terminal.
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

export declare class WebTrainingSession {
  readonly capabilities: WebTrainingCapabilitiesV1;
  readonly plan: CompiledTrainingPlanV1;
  static prepare(
    model: WebTrainingModelV1,
    config: WebTrainingConfigV1,
    adapter: WebTrainingAdapterV1,
  ): Promise<WebTrainingSession>;
  get state(): WebTrainingState;
  forward(batch: TrainingBatchV1, options?: WebTrainingOperationOptionsV1): Promise<TrainingResultV1>;
  backward(result: TrainingResultV1, options?: WebTrainingOperationOptionsV1): Promise<WebTrainingReceiptV1>;
  step(options?: WebTrainingOperationOptionsV1): Promise<WebTrainingReceiptV1>;
  checkpoint(options?: WebTrainingOperationOptionsV1): Promise<WebBinaryResultV1>;
  resume(checkpoint: Uint8Array, options?: WebTrainingOperationOptionsV1): Promise<WebTrainingReceiptV1>;
  export(options?: WebTrainingOperationOptionsV1): Promise<WebBinaryResultV1>;
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

export interface WebGpuKernelModuleV1 {
  readonly id: string;
  readonly sha256: string;
  readonly source: string;
  readonly bindings: readonly WebGpuKernelBindingV1[];
  readonly entryPoints: Readonly<Record<string, readonly [number, number, number]>>;
}

export interface WebGpuKernelBindingV1 {
  readonly group: number;
  readonly binding: number;
  readonly addressSpace: "uniform" | "storage";
  readonly access: "read" | "read_write" | null;
}

export interface WebGpuKernelCandidateBundleV1 {
  readonly schemaId: "tritium.webgpu_kernel_candidate_bundle";
  readonly schemaVersion: 1;
  readonly bundleSha256: string;
  readonly modules: Readonly<Record<string, WebGpuKernelModuleV1>>;
  readonly candidateOperationModuleDependencies: Readonly<
    Record<string, readonly string[]>
  >;
}

export declare function webGpuKernelCandidateBundleV1(): WebGpuKernelCandidateBundleV1;
export declare function webGpuCandidateModulesForOperationV1(
  operation: string,
): readonly WebGpuKernelModuleV1[];

export type WebGpuDispatchExecutionV1 = "forward" | "vjp" | "step";
export type WebGpuDispatchRepeatV1 = "once" | "per_output";
export type WebGpuDispatchGeometryV1 =
  | "linear_input_64"
  | "linear_output_64"
  | "linear_parameter_64"
  | "linear_primary_input_64"
  | "linear_rows_64"
  | "optimizer_blocks_256"
  | "rope_pairs_64"
  | "single";

export interface WebGpuDispatchStageV1 {
  readonly moduleId: string;
  readonly entryPoint: string;
  readonly selector: number | null;
  readonly dispatch: WebGpuDispatchGeometryV1;
  readonly repeat: WebGpuDispatchRepeatV1;
}

export interface WebGpuDispatchFormV1 {
  readonly operation: string;
  readonly execution: WebGpuDispatchExecutionV1;
  readonly stages: readonly WebGpuDispatchStageV1[];
}

export interface WebGpuDispatchCatalogV1 {
  readonly schemaId: "tritium.webgpu_dispatch_catalog";
  readonly schemaVersion: 1;
  readonly sha256: string;
  readonly forms: Readonly<Record<string, WebGpuDispatchFormV1>>;
}

export declare function webGpuDispatchCatalogV1(): WebGpuDispatchCatalogV1;
export declare function webGpuDispatchFormV1(
  operation: string,
  execution: WebGpuDispatchExecutionV1,
): WebGpuDispatchFormV1;

export interface WebGpuBufferPortV1 {
  readonly size: number;
  mapAsync(mode: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

export interface WebGpuPipelinePortV1 {
  getBindGroupLayout(index: number): unknown;
}

export interface WebGpuComputePassPortV1 {
  setPipeline(pipeline: WebGpuPipelinePortV1): void;
  setBindGroup(index: number, bindGroup: unknown): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

export interface WebGpuCommandEncoderPortV1 {
  beginComputePass(descriptor?: Readonly<Record<string, unknown>>): WebGpuComputePassPortV1;
  copyBufferToBuffer(
    source: WebGpuBufferPortV1,
    sourceOffset: number,
    destination: WebGpuBufferPortV1,
    destinationOffset: number,
    size: number,
  ): void;
  finish(): unknown;
}

export interface WebGpuDevicePortV1 {
  readonly limits: Readonly<{
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
    maxComputeWorkgroupsPerDimension: number;
    maxBindingsPerBindGroup: number;
    maxStorageBuffersPerShaderStage: number;
    maxUniformBuffersPerShaderStage: number;
    maxUniformBufferBindingSize: number;
    minUniformBufferOffsetAlignment: number;
  }>;
  readonly queue: Readonly<{
    writeBuffer(buffer: WebGpuBufferPortV1, bufferOffset: number, data: Uint8Array): void;
    submit(commands: readonly unknown[]): void;
  }>;
  readonly lost: Promise<unknown>;
  createShaderModule(descriptor: Readonly<{ label: string; code: string }>): unknown;
  createComputePipelineAsync(descriptor: Readonly<{
    label: string;
    layout: "auto";
    compute: Readonly<{ module: unknown; entryPoint: string }>;
  }>): Promise<WebGpuPipelinePortV1>;
  createBuffer(descriptor: Readonly<{
    label: string;
    size: number;
    usage: number;
  }>): WebGpuBufferPortV1;
  createBindGroup(descriptor: Readonly<{
    label: string;
    layout: unknown;
    entries: readonly Readonly<{
      binding: number;
      resource: Readonly<{
        buffer: WebGpuBufferPortV1;
        offset?: number;
        size?: number;
      }>;
    }>[];
  }>): unknown;
  createCommandEncoder(descriptor: Readonly<{ label: string }>): WebGpuCommandEncoderPortV1;
  destroy(): void;
}

export interface WebGpuResidentTensorV1 {
  readonly bufferId: string;
  readonly bytes: Uint8Array;
}

export interface WebGpuResidentAuxiliaryV1 {
  readonly id: string;
  readonly byteLength: number;
  readonly initialBytes: Uint8Array | null;
}

export interface WebGpuResidentAuxiliarySetV1 {
  readonly maxBytes: number;
  readonly resources: readonly WebGpuResidentAuxiliaryV1[];
}

export interface WebGpuResidentCopyV1 {
  readonly source: string;
  readonly sourceOffset: number;
  readonly destination: string;
  readonly destinationOffset: number;
  readonly byteLength: number;
}

export interface WebGpuResidentDispatchV1 {
  readonly operation: string;
  readonly execution: WebGpuDispatchExecutionV1;
  readonly stageIndex: number;
  readonly uniformSlot: number;
  readonly uniformBytes: Uint8Array | null;
  readonly storageBindings: Readonly<Record<number, string>>;
  readonly workgroups: readonly [number, number, number];
}

export declare class WebGpuResidentRuntimeV1 {
  private constructor();
  static prepare(
    device: WebGpuDevicePortV1,
    plan: CompiledTrainingPlanV1,
    initial: readonly WebGpuResidentTensorV1[],
    auxiliary?: WebGpuResidentAuxiliarySetV1,
  ): Promise<WebGpuResidentRuntimeV1>;
  dispatch(
    commands: readonly WebGpuResidentDispatchV1[],
    copies?: readonly WebGpuResidentCopyV1[],
  ): void;
  read(bufferId: string): Promise<Uint8Array>;
  dispose(): void;
}

export declare function lowerPointwiseWebGpuOperationV1(
  plan: CompiledTrainingPlanV1,
  phase: "forward" | "backward",
  operationId: string,
  firstUniformSlot: number,
): readonly WebGpuResidentDispatchV1[];

export interface WebGpuResidentTransactionV1 {
  readonly commands: readonly WebGpuResidentDispatchV1[];
  readonly copies: readonly WebGpuResidentCopyV1[];
}

export interface WebGpuResidentScheduleBudgetV1 {
  readonly maxPeakBytes: number;
}

export interface WebGpuResidentScheduleV1 {
  auxiliaryResources(): WebGpuResidentAuxiliarySetV1;
  transaction(
    phase: "forward" | "backward",
    operationId: string,
    firstUniformSlot: number,
  ): WebGpuResidentTransactionV1;
}

export declare function compileWebGpuResidentScheduleV1(
  plan: CompiledTrainingPlanV1,
  budget: WebGpuResidentScheduleBudgetV1,
): WebGpuResidentScheduleV1;

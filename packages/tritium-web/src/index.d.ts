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
  readonly operations: readonly string[];
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

export interface WebTrainingAdapterV1 {
  readonly capabilities: WebTrainingCapabilitiesV1;
  prepare(
    model: WebTrainingModelV1,
    config: WebTrainingConfigV1,
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

export declare function prepareTraining(
  model: WebTrainingModelV1,
  config: WebTrainingConfigV1,
  adapter?: WebTrainingAdapterV1,
): Promise<WebTrainingSession>;

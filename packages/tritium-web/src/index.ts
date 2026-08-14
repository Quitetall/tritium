export {
  canonicalTrainingManifestV1Json,
  canonicalTrainingManifestJson,
  parseTrainingManifest,
  TrainingManifestError,
} from "../../../bindings/typescript/src/training_manifest.ts";
export type {
  TrainingOpCategoryV1,
  TrainingOpDescriptorV1,
  TrainingOpManifestV1,
  TrainingOpManifestV2,
  TrainingVjpV1,
} from "../../../bindings/typescript/src/training_manifest.ts";
export {
  TRAINING_MANIFEST_DIGEST_V1,
  TRAINING_MANIFEST_DIGEST_V2,
  TRAINING_VECTOR_DIGEST_V1,
  TRAINING_VECTOR_DIGEST_V2,
} from "./identity.ts";
export {
  WebTrainingError,
  WebTrainingSession,
  compileTrainingPlan,
  prepareTraining,
} from "./session.ts";
export { executePortableWasmRequest, runPortableWasmConformance } from "./wasm.ts";
export { createPortableWasmTrainingAdapter } from "./wasm-adapter.ts";
export {
  webGpuDispatchCatalogV2,
  webGpuDispatchFormV1,
  webGpuCandidateModulesForOperationV1,
  webGpuKernelCandidateBundleV1,
} from "./webgpu-kernels.ts";
export { WebGpuResidentRuntimeV1 } from "./webgpu-runtime.ts";
export { createWebGpuTrainingAdapter } from "./webgpu-adapter.ts";
export { lowerPointwiseWebGpuOperationV1 } from "./webgpu-lowering.ts";
export { compileWebGpuResidentScheduleV1 } from "./webgpu-schedule.ts";
export {
  PortableLifecyclePlanError,
  compilePortableCheckpointRequest,
  compilePortableExportRequest,
  compilePortableReloadRequest,
  compilePortableResumeRequest,
} from "./lifecycle.ts";
export {
  PortableWasmLifecycleError,
  PortableWasmLifecycleState,
} from "./portable-state.ts";
export {
  PortableSchedulePlanError,
  compilePortableBackwardOperationRequest,
  compilePortablePlanOperationRequest,
} from "./portable-schedule.ts";
export {
  WebTrainingPayloadError,
  decodeWebTrainingPayload,
  encodeWebTrainingPayload,
} from "./payload.ts";
export type {
  PortableAdamLeafV1,
  PortableCheckpointOptimizerV1,
  PortableCheckpointStateV1,
  PortableInt8AdamLeafV1,
  PortableMuonLeafV1,
  PortableSgdLeafV1,
} from "./lifecycle.ts";
export type {
  PortableWasmLifecycleBinaryV1,
  PortableWasmLifecycleErrorCode,
  PortableWasmLifecycleOptionsV1,
  PortableWasmLifecycleStateV1,
} from "./portable-state.ts";
export type {
  PortableCompiledDispatchV1,
  PortableSchedulePlanErrorCode,
  PortableScheduleTensorStoreV1,
  PortableScheduleTensorV1,
} from "./portable-schedule.ts";
export type {
  WebTrainingInitialTensorsV1,
  WebTrainingPayloadErrorCode,
} from "./payload.ts";
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
  CompiledTrainingBufferV1,
  CompiledTrainingBindingV1,
  CompiledBackwardOperationV1,
  CompiledTrainingOperationV1,
  CompiledTrainingPlanV1,
  TrainingBatchV1,
  TrainingAttributeKindV1,
  TrainingAttributeSpecV1,
  TrainingDTypeV1,
  TrainingOperationSpecV1,
  TrainingRecipeV1,
  TrainingResultV1,
  TrainingTensorRoleV1,
  TrainingTensorSpecV1,
  WebBinaryResultV1,
  WebTrainingAdapterV1,
  WebTrainingBackendPolicyV1,
  WebTrainingCapabilitiesV1,
  WebTrainingConfigV1,
  WebTrainingErrorCode,
  WebTrainingFailureReceiptV1,
  WebTrainingImplementationV1,
  WebTrainingModelV1,
  WebTrainingOperationOptionsV1,
  WebTrainingReceiptV1,
  WebTrainingState,
} from "./session.ts";
export type {
  WebGpuDispatchCatalogV2,
  WebGpuDispatchExecutionV1,
  WebGpuDispatchFormV1,
  WebGpuDispatchGeometryV1,
  WebGpuDispatchRepeatV1,
  WebGpuDispatchStageV1,
  WebGpuKernelBindingV1,
  WebGpuKernelCandidateBundleV1,
  WebGpuKernelModuleV1,
} from "./webgpu-kernels.ts";
export type {
  WebGpuBufferPortV1,
  WebGpuCommandEncoderPortV1,
  WebGpuComputePassPortV1,
  WebGpuDevicePortV1,
  WebGpuPipelinePortV1,
  WebGpuResidentDispatchV1,
  WebGpuResidentSubmissionV1,
  WebGpuResidentTensorV1,
} from "./webgpu-runtime.ts";
export type { WebGpuTrainingAdapterOptionsV1 } from "./webgpu-adapter.ts";
// Pre-commit staged-tree probe: no runtime behavior.

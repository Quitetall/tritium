import type {
  PortableWasmSourceV1,
  TrainingBatchV1,
  WebTrainingConfigV1,
  WebTrainingModelV1,
  WebGpuDevicePortV1,
} from "./index.js";

export interface WebGpuVectorConformanceInventoryV1 {
  readonly schemaId: "tritium.webgpu_vector_conformance_inventory";
  readonly schemaVersion: 1;
  readonly manifestDigest:
    "9093a1a7f9a3422c399943782aadf4df6b11833cf2253db0db56ff2d9dedb098";
  readonly vectorDigest:
    "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7";
  readonly caseCounts: Readonly<{
    valid: 72;
    invalid: 45;
    compute: 68;
    lifecycle: 4;
    total: 117;
  }>;
}

export interface WebGpuVectorCaseTraceV1 {
  readonly caseId: string;
  readonly implementation: "webgpu" | "wasm-codec" | "wasm-validation";
  readonly outputDigest: string;
  readonly scratchBytes: number | null;
  readonly scratchBytesMax: number | null;
}

export interface WebGpuVectorConformanceTraceV1 {
  readonly schemaId: "tritium.webgpu_vector_conformance_trace";
  readonly schemaVersion: 1;
  readonly implementation: "webgpu";
  readonly manifestDigest: WebGpuVectorConformanceInventoryV1["manifestDigest"];
  readonly vectorDigest: WebGpuVectorConformanceInventoryV1["vectorDigest"];
  readonly caseCounts: Readonly<{ valid: 72; invalid: 45; skipped: 0 }>;
  readonly webgpuCaseTransactions: 68;
  readonly webgpuDispatches: number;
  readonly wasmDispatches: 0;
  readonly wasmCodecCalls: 4;
  readonly wasmValidationCalls: 45;
  readonly explicitReadbacks: number;
  readonly peakBufferBytes: number;
  readonly executionDigest: string;
  readonly cases: readonly WebGpuVectorCaseTraceV1[];
}

export interface WebGpuVectorConformanceOptionsV1 {
  readonly wasmSource?: PortableWasmSourceV1;
  readonly maxPeakBytes?: number;
  readonly physicalDevice?: string;
}

export declare function webGpuVectorConformanceInventoryV1():
  WebGpuVectorConformanceInventoryV1;

/** Takes exclusive ownership of device and destroys it on success or failure. */
export declare function runWebGpuVectorConformanceV1(
  device: WebGpuDevicePortV1,
  options?: WebGpuVectorConformanceOptionsV1,
): Promise<WebGpuVectorConformanceTraceV1>;

export type PhysicalBrowserQualificationErrorCode =
  | "adapter_unavailable"
  | "device_identity"
  | "fault_injection"
  | "instrumentation"
  | "invalid_options"
  | "lifecycle"
  | "native_artifact_parity"
  | "vector_conformance";

export declare class PhysicalBrowserQualificationError extends Error {
  readonly code: PhysicalBrowserQualificationErrorCode;
  constructor(code: PhysicalBrowserQualificationErrorCode, message: string);
}

export interface PhysicalBrowserAdapterIdentityV1 {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
  readonly software: false;
}

export interface PhysicalBrowserLimitsV1 {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxComputeWorkgroupsPerDimension: number;
  readonly maxStorageBuffersPerShaderStage: number;
}

export interface PhysicalBrowserFaultTraceV1 {
  readonly passed: true;
  readonly errorCode: string;
  readonly stateAfter: string | null;
  /** Count of concrete injected events when physical observation is required. */
  readonly observedEvents?: number;
}

export interface PhysicalBrowserLifecycleReceiptV1 {
  readonly operation: string;
  readonly completedSteps: number;
  readonly peakResidentBytes: number;
  readonly buildId: string;
  readonly physicalDevice: string;
}

export interface PhysicalBrowserLifecycleTraceV1 {
  readonly prepare: true;
  readonly forward: true;
  readonly backward: true;
  readonly optimizerStep: true;
  readonly checkpointResume: true;
  readonly exportReload: true;
  readonly nativeArtifactParity: true;
  readonly completedSteps: 1;
  readonly checkpointSha256: string;
  readonly artifactSha256: string;
  readonly nativeArtifactSha256: string;
  readonly nativeReferenceDigest: string;
  readonly receipts: readonly PhysicalBrowserLifecycleReceiptV1[];
}

export interface PhysicalBrowserTrainingScenarioV1 {
  readonly schemaId: "tritium.physical_browser_training_scenario";
  readonly schemaVersion: 1;
  readonly scenarioId: "salt-ste-sgd-256-v1";
  readonly completedSteps: 1;
  readonly model: WebTrainingModelV1;
  readonly config: WebTrainingConfigV1;
  readonly batch: TrainingBatchV1;
}

export interface PhysicalBrowserTrainingLaneTraceV1 {
  readonly schemaId: "tritium.physical_browser_training_lane_trace";
  readonly schemaVersion: 1;
  readonly scenarioId: "salt-ste-sgd-256-v1";
  readonly implementation: "webgpu";
  readonly manifestDigest: WebGpuVectorConformanceInventoryV1["manifestDigest"];
  readonly vectorDigest: WebGpuVectorConformanceInventoryV1["vectorDigest"];
  readonly physicalDevice: string;
  readonly buildId: string;
  readonly adapter: PhysicalBrowserAdapterIdentityV1;
  readonly limits: PhysicalBrowserLimitsV1;
  readonly vector: WebGpuVectorConformanceTraceV1;
  readonly lifecycle: PhysicalBrowserLifecycleTraceV1;
  readonly faults: Readonly<{
    deviceLoss: PhysicalBrowserFaultTraceV1;
    allocationFailure: PhysicalBrowserFaultTraceV1;
    malformedCheckpoint: PhysicalBrowserFaultTraceV1;
    malformedSalt: PhysicalBrowserFaultTraceV1;
    cancellation: PhysicalBrowserFaultTraceV1;
    outOfOrder: PhysicalBrowserFaultTraceV1;
  }>;
  readonly explicitReadbacks: number;
  readonly steadyStateReadbacks: 0;
  readonly wasmDispatches: 0;
  readonly peakBufferBytes: number;
  readonly executionDigest: string;
}

export interface PhysicalBrowserTrainingLaneOptionsV1 {
  readonly nativeArtifact: Uint8Array;
  readonly nativeReferenceDigest: string;
  readonly maxPeakBytes?: number;
}

export declare function physicalBrowserTrainingScenarioV1():
  PhysicalBrowserTrainingScenarioV1;

/** Acquires and destroys every physical WebGPU device used by the lane. */
export declare function runPhysicalBrowserTrainingLaneV1(
  options: PhysicalBrowserTrainingLaneOptionsV1,
): Promise<PhysicalBrowserTrainingLaneTraceV1>;

import type {
  PortableWasmSourceV1,
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

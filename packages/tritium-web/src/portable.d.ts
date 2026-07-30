export type PortableExecutionV1 =
  | "forward"
  | "vjp"
  | "step"
  | "checkpoint"
  | "resume"
  | "export"
  | "reload";

export type PortableBufferDataV1 =
  | { readonly dtype: "f32"; readonly bits: readonly number[] }
  | { readonly dtype: "u32"; readonly values: readonly number[] }
  | { readonly dtype: "bytes"; readonly values: readonly number[] };

export interface PortableBufferV1 {
  readonly name: string;
  readonly shape: readonly number[];
  readonly data: PortableBufferDataV1;
}

export type PortableAttributeV1 =
  | { readonly kind: "f32"; readonly name: string; readonly bits: number }
  /** V1 JSON transports u64 values only through Number safe integers. */
  | { readonly kind: "u64"; readonly name: string; readonly value: number }
  | { readonly kind: "bool"; readonly name: string; readonly value: boolean }
  | { readonly kind: "text"; readonly name: string; readonly value: string }
  /** Every V1 u64-list value must be a non-negative Number safe integer. */
  | {
      readonly kind: "u64-list";
      readonly name: string;
      readonly values: readonly number[];
    }
  | {
      readonly kind: "u32-list";
      readonly name: string;
      readonly values: readonly number[];
    };

export interface PortableTrainingRequestV1 {
  readonly schemaId: "tritium.portable_training_request";
  readonly schemaVersion: 1;
  readonly physicalDevice: string;
  readonly operation: string;
  readonly execution: PortableExecutionV1;
  readonly vectorDigest:
    | "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7"
    | null;
  readonly inputs: readonly PortableBufferV1[];
  readonly attributes: readonly PortableAttributeV1[];
  readonly outputs: readonly PortableBufferV1[];
}

export interface PortableTrainingReceiptV1 {
  readonly backendId: "wasm.portable.v1";
  readonly backendBuild: string;
  readonly physicalDevice: string;
  readonly manifestDigest:
    "9093a1a7f9a3422c399943782aadf4df6b11833cf2253db0db56ff2d9dedb098";
  readonly vectorDigest:
    | "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7"
    | null;
  readonly operation: string;
  readonly execution: PortableExecutionV1;
  readonly dtype: "f32" | "u32" | "bytes";
  readonly maxRank: number;
  readonly maxElements: number;
  readonly maxBytes: number;
  readonly inputDigest: string;
  readonly outputDigest: string;
  readonly peakResidentBytes: number;
  readonly scratchBytes: number;
  readonly hostTransfers: 0;
  readonly deviceResident: true;
}

export interface PortableTrainingErrorV1 {
  readonly category: string;
  readonly code: string;
  readonly message: string;
}

export type PortableTrainingResponseV1 =
  | {
      readonly status: "ok";
      readonly schemaId: "tritium.portable_training_response";
      readonly schemaVersion: 1;
      readonly outputs: readonly PortableBufferV1[];
      readonly receipt: PortableTrainingReceiptV1;
    }
  | {
      readonly status: "error";
      readonly schemaId: "tritium.portable_training_response";
      readonly schemaVersion: 1;
      readonly outputs: readonly PortableBufferV1[];
      readonly error: PortableTrainingErrorV1;
    };

export interface PortableWasmConformanceReceiptV1 {
  readonly schemaId: "tritium.portable_wasm_conformance_receipt";
  readonly schemaVersion: 1;
  readonly implementation: "wasm-fallback";
  readonly engine: "wasm32-unknown-unknown";
  readonly buildId: string;
  readonly guestDigest: string;
  readonly executionDigest: string;
  readonly manifestDigest:
    "9093a1a7f9a3422c399943782aadf4df6b11833cf2253db0db56ff2d9dedb098";
  readonly vectorDigest:
    "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7";
  readonly operationCount: number;
  readonly caseCount: number;
  readonly maxCallerBytes: number;
  readonly maxLinearMemoryBytes: number;
  readonly repeatedExecutions: 2;
}

export type PortableWasmSourceV1 =
  | RequestInfo
  | URL
  | Response
  | BufferSource;

import type { PortableTrainingRequestV1 } from "./portable.js";

export type PortableScheduleTensorV1 =
  | Float32Array
  | Uint32Array
  | Uint8Array;

export type PortableScheduleTensorStoreV1 = Readonly<
  Record<string, PortableScheduleTensorV1>
>;

export type PortableSchedulePlanErrorCode =
  | "buffer_mismatch"
  | "capacity"
  | "invalid_schema"
  | "missing_buffer";

export interface PortableCompiledDispatchV1 {
  readonly request: PortableTrainingRequestV1;
  readonly outputBufferIds: readonly string[];
}

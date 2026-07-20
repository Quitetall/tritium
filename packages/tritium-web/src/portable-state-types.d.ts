import type { PortableCheckpointStateV1 } from "./lifecycle-types.js";
import type {
  PortableTrainingReceiptV1,
  PortableWasmSourceV1,
} from "./portable.js";

export type PortableWasmLifecycleErrorCode =
  | "backend"
  | "busy"
  | "disposed"
  | "invalid_state";

export interface PortableWasmLifecycleBinaryV1 {
  readonly bytes: Uint8Array;
  readonly receipt: PortableTrainingReceiptV1;
}

export type PortableWasmLifecycleStateV1 = PortableCheckpointStateV1;

export interface PortableWasmLifecycleOptionsV1 {
  readonly source: PortableWasmSourceV1;
  readonly state: PortableCheckpointStateV1;
  readonly physicalDevice?: string;
}

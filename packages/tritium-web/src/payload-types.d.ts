import type { PortableScheduleTensorV1 } from "./portable-schedule-types.js";

export type WebTrainingPayloadErrorCode =
  | "buffer_mismatch"
  | "capacity"
  | "integrity"
  | "invalid_schema"
  | "missing_buffer";

export type WebTrainingInitialTensorsV1 = Readonly<
  Record<string, PortableScheduleTensorV1>
>;

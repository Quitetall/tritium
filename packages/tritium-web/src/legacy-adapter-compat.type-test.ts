import type { TrainingBatchV1, TrainingResultV1, WebTrainingAdapterV1 } from "./session.ts";

declare const adapter: WebTrainingAdapterV1;
declare const batch: TrainingBatchV1;
declare const result: TrainingResultV1;

// Direct v1 adapter callers compiled before operation cancellation existed.
void adapter.forward(batch);
void adapter.backward(result);
void adapter.step();
void adapter.checkpoint();
void adapter.resume(new Uint8Array([1]));
void adapter.export();

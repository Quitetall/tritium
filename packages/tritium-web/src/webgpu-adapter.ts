import {
  canonicalTrainingManifestJson,
  parseTrainingManifest,
} from "../../../bindings/typescript/src/training_manifest.ts";

import {
  TRAINING_MANIFEST_DIGEST_V1,
  TRAINING_VECTOR_DIGEST_V1,
} from "./identity.ts";
import {
  decodeWebTrainingPayload,
  WebTrainingPayloadError,
} from "./payload.ts";
import type { PortableScheduleTensorV1 } from "./portable-schedule-types.js";
import type {
  CompiledTrainingPlanV1,
  TrainingBatchV1,
  TrainingResultV1,
  WebBinaryResultV1,
  WebTrainingAdapterV1,
  WebTrainingCapabilitiesV1,
  WebTrainingConfigV1,
  WebTrainingModelV1,
  WebTrainingReceiptV1,
} from "./session.ts";
import { WebTrainingError } from "./session.ts";
import { webGpuKernelCandidateBundleV1 } from "./webgpu-kernels.ts";
import {
  WebGpuResidentRuntimeV1,
  type WebGpuDevicePortV1,
  type WebGpuResidentSubmissionV1,
  type WebGpuResidentTensorV1,
} from "./webgpu-runtime.ts";
import {
  compileWebGpuResidentScheduleV1,
  type WebGpuResidentScheduleV1,
} from "./webgpu-schedule.ts";

export interface WebGpuTrainingAdapterOptionsV1 {
  readonly buildId?: string;
  readonly physicalDevice?: string | null;
  readonly maxResidentBytes?: number;
}

function fail(code: "cancelled" | "capability_mismatch" | "invalid_schema" | "invalid_state" | "memory_limit", message: string): never {
  throw new WebTrainingError(code, message);
}

function rejectPreDispatchCancellation(signal?: AbortSignal | null): void {
  if (signal?.aborted === true) fail("cancelled", "WebGPU operation was cancelled before dispatch");
}

function capturedProperty(
  value: Readonly<Record<PropertyKey, unknown>>,
  name: PropertyKey,
  context: string,
): unknown {
  try {
    return Reflect.get(value, name);
  } catch {
    fail("invalid_schema", `${context}.${String(name)} could not be read`);
  }
}

function capturedKeys(value: object, context: string): readonly PropertyKey[] {
  try {
    return Reflect.ownKeys(value);
  } catch {
    fail("invalid_schema", `${context} keys could not be read`);
  }
}

function bytes(tensor: PortableScheduleTensorV1): Uint8Array {
  return Uint8Array.from(
    new Uint8Array(tensor.buffer, tensor.byteOffset, tensor.byteLength),
  );
}

function receipt(
  capabilities: WebTrainingCapabilitiesV1,
  operation: string,
  completedSteps: number,
  peakResidentBytes: number,
): WebTrainingReceiptV1 {
  return Object.freeze({
    schemaId: "tritium.web_training_receipt",
    schemaVersion: 1,
    implementation: capabilities.implementation,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V1,
    vectorDigest: TRAINING_VECTOR_DIGEST_V1,
    buildId: capabilities.buildId,
    physicalDevice: capabilities.physicalDevice,
    operation,
    completedSteps,
    peakResidentBytes,
  });
}

class ResidentWebGpuTrainingAdapter implements WebTrainingAdapterV1 {
  readonly capabilities: WebTrainingCapabilitiesV1;
  readonly #device: WebGpuDevicePortV1;
  readonly #uniformStride: number;
  #plan: CompiledTrainingPlanV1 | null = null;
  #runtime: WebGpuResidentRuntimeV1 | null = null;
  #schedule: WebGpuResidentScheduleV1 | null = null;
  #completedSteps = 0;
  #disposed = false;

  constructor(
    device: WebGpuDevicePortV1,
    capabilities: WebTrainingCapabilitiesV1,
    uniformStride: number,
  ) {
    this.#device = device;
    this.capabilities = capabilities;
    this.#uniformStride = uniformStride;
  }

  async validate(
    _model: WebTrainingModelV1,
    config: WebTrainingConfigV1,
    plan: CompiledTrainingPlanV1,
  ): Promise<void> {
    compileWebGpuResidentScheduleV1(plan, {
      maxPeakBytes: config.maxResidentBytes,
      uniformStride: this.#uniformStride,
    });
  }

  async prepare(
    model: WebTrainingModelV1,
    config: WebTrainingConfigV1,
    plan: CompiledTrainingPlanV1,
  ): Promise<WebTrainingReceiptV1> {
    if (this.#disposed || this.#runtime !== null) {
      fail("invalid_state", "WebGPU adapter is not fresh");
    }
    const schedule = compileWebGpuResidentScheduleV1(
      plan, {
        maxPeakBytes: config.maxResidentBytes,
        uniformStride: this.#uniformStride,
      },
    );
    let store: Readonly<Record<string, PortableScheduleTensorV1>>;
    try {
      store = decodeWebTrainingPayload(plan, model.payload);
    } catch (error) {
      if (error instanceof WebTrainingPayloadError) {
        fail(error.code === "capacity" ? "memory_limit" : "invalid_schema", error.message);
      }
      throw error;
    }
    const initial: WebGpuResidentTensorV1[] = [];
    for (const buffer of plan.buffers) {
      if (buffer.ownerId !== buffer.id) continue;
      const tensor = store[buffer.id];
      if (tensor === undefined) fail("invalid_schema", `payload omitted owner ${buffer.id}`);
      initial.push(Object.freeze({ bufferId: buffer.id, bytes: bytes(tensor) }));
    }
    this.#runtime = await WebGpuResidentRuntimeV1.prepare(
      this.#device, plan, initial, schedule.auxiliaryResources(), this.#uniformStride,
    );
    this.#schedule = schedule;
    this.#plan = plan;
    return receipt(this.capabilities, "session.prepare", 0, schedule.peakBytes());
  }

  #ready(): Readonly<{
    plan: CompiledTrainingPlanV1;
    runtime: WebGpuResidentRuntimeV1;
    schedule: WebGpuResidentScheduleV1;
  }> {
    if (this.#disposed) fail("invalid_state", "WebGPU adapter is disposed");
    if (this.#plan === null || this.#runtime === null || this.#schedule === null) {
      fail("invalid_state", "WebGPU adapter is not prepared");
    }
    return { plan: this.#plan, runtime: this.#runtime, schedule: this.#schedule };
  }

  async #dispatch(
    phase: "forward" | "backward",
    operationIds: readonly string[],
    optimizerStep?: number,
    clears: readonly string[] = [],
  ): Promise<void> {
    const { runtime, schedule } = this.#ready();
    const transactions: WebGpuResidentSubmissionV1[] = [];
    let firstUniformSlot = 0;
    for (const operationId of operationIds) {
      const transaction = schedule.transaction(
        phase, operationId, firstUniformSlot, optimizerStep,
      );
      transactions.push(transaction);
      firstUniformSlot += transaction.commands.length;
    }
    await runtime.dispatchTransactions(transactions, clears);
  }

  async forward(
    batch: TrainingBatchV1,
    signal?: AbortSignal | null,
  ): Promise<TrainingResultV1> {
    const { plan, runtime, schedule } = this.#ready();
    rejectPreDispatchCancellation(signal);
    for (const [bufferId, tensor] of Object.entries(batch.inputs)) {
      runtime.write(bufferId, bytes(tensor));
    }
    await this.#dispatch(
      "forward",
      plan.operations
        .filter((operation) => !operation.operation.startsWith("optimizer."))
        .map((operation) => operation.id),
    );
    const lossOperation = [...plan.operations].reverse().find((operation) =>
      operation.operation.startsWith("loss."),
    );
    const lossId = lossOperation?.outputs[0];
    if (lossId === undefined) fail("invalid_schema", "compiled plan has no loss output");
    const lossBytes = await runtime.read(lossId);
    if (lossBytes.byteLength !== 4) fail("invalid_schema", "loss output is not scalar f32");
    const loss = new DataView(
      lossBytes.buffer, lossBytes.byteOffset, lossBytes.byteLength,
    ).getFloat32(0, true);
    return Object.freeze({
      loss,
      receipt: receipt(
        this.capabilities, "session.forward", this.#completedSteps, schedule.peakBytes(),
      ),
    });
  }

  async backward(
    _result: TrainingResultV1,
    signal?: AbortSignal | null,
  ): Promise<WebTrainingReceiptV1> {
    const { plan, schedule } = this.#ready();
    rejectPreDispatchCancellation(signal);
    const clears = plan.buffers
      .filter((buffer) =>
        buffer.ownerId === buffer.id && buffer.backwardInitialization === "zero"
      )
      .map((buffer) => buffer.id);
    await this.#dispatch(
      "backward", plan.backwardOperations.map((operation) => operation.id), undefined, clears,
    );
    return receipt(
      this.capabilities, "session.backward", this.#completedSteps, schedule.peakBytes(),
    );
  }

  async step(signal?: AbortSignal | null): Promise<WebTrainingReceiptV1> {
    const { plan, schedule } = this.#ready();
    rejectPreDispatchCancellation(signal);
    const nextStep = this.#completedSteps + 1;
    await this.#dispatch(
      "forward",
      plan.operations
        .filter((operation) => operation.operation.startsWith("optimizer."))
        .map((operation) => operation.id),
      nextStep,
    );
    this.#completedSteps = nextStep;
    return receipt(this.capabilities, "session.step", nextStep, schedule.peakBytes());
  }

  async checkpoint(): Promise<WebBinaryResultV1> {
    fail("capability_mismatch", "resident WebGPU checkpoint integration is not available");
  }

  async resume(_checkpoint: Uint8Array): Promise<WebTrainingReceiptV1> {
    fail("capability_mismatch", "resident WebGPU resume integration is not available");
  }

  async export(): Promise<WebBinaryResultV1> {
    fail("capability_mismatch", "resident WebGPU SALT export integration is not available");
  }

  async dispose(): Promise<void> {
    if (this.#disposed) return;
    this.#disposed = true;
    if (this.#runtime === null) this.#device.destroy();
    else this.#runtime.dispose();
    this.#runtime = null;
    this.#schedule = null;
    this.#plan = null;
  }
}

/** Create a session adapter over one already-authorized WebGPU device.
 * The adapter takes exclusive ownership and destroys the device on disposal.
 */
export function createWebGpuTrainingAdapter(
  device: WebGpuDevicePortV1,
  options: WebGpuTrainingAdapterOptionsV1 = {},
): WebTrainingAdapterV1 {
  if (typeof device !== "object" || device === null) {
    fail("invalid_schema", "WebGPU adapter device is invalid");
  }
  const limits = capturedProperty(
    device as unknown as Readonly<Record<PropertyKey, unknown>>, "limits", "device",
  );
  if (typeof limits !== "object" || limits === null) {
    fail("invalid_schema", "WebGPU adapter device is invalid");
  }
  const deviceMaxBufferSize = capturedProperty(
    limits as Readonly<Record<PropertyKey, unknown>>, "maxBufferSize", "device.limits",
  );
  if (!Number.isSafeInteger(deviceMaxBufferSize) || (deviceMaxBufferSize as number) <= 0) {
    fail("invalid_schema", "WebGPU adapter device is invalid");
  }
  const deviceUniformAlignment = capturedProperty(
    limits as Readonly<Record<PropertyKey, unknown>>,
    "minUniformBufferOffsetAlignment",
    "device.limits",
  );
  if (!Number.isSafeInteger(deviceUniformAlignment) ||
      (deviceUniformAlignment as number) <= 0) {
    fail("invalid_schema", "WebGPU adapter uniform alignment is invalid");
  }
  const uniformStride = Math.max(256, deviceUniformAlignment as number);
  if (uniformStride % 256 !== 0) {
    fail("invalid_schema", "WebGPU adapter uniform alignment is not a 256-byte multiple");
  }
  if (typeof options !== "object" || options === null || Array.isArray(options) ||
      capturedKeys(options, "WebGPU adapter options").some((key) =>
        typeof key !== "string" || !["buildId", "maxResidentBytes", "physicalDevice"].includes(key)
      )) {
    fail("invalid_schema", "WebGPU adapter options are invalid");
  }
  const optionRecord = options as Readonly<Record<PropertyKey, unknown>>;
  const configuredMax = capturedProperty(optionRecord, "maxResidentBytes", "options");
  const configuredBuildId = capturedProperty(optionRecord, "buildId", "options");
  const configuredPhysicalDevice = capturedProperty(optionRecord, "physicalDevice", "options");
  const maxResidentBytes = configuredMax ?? deviceMaxBufferSize;
  if (!Number.isSafeInteger(maxResidentBytes) || (maxResidentBytes as number) <= 0) {
    fail("invalid_schema", "WebGPU adapter maxResidentBytes must be positive");
  }
  const admittedMaxResidentBytes = maxResidentBytes as number;
  const buildId = configuredBuildId ?? `wgsl:${webGpuKernelCandidateBundleV1().bundleSha256}`;
  const physicalDevice = configuredPhysicalDevice ?? null;
  if (typeof buildId !== "string" || buildId.length === 0 ||
      !(physicalDevice === null ||
        (typeof physicalDevice === "string" && physicalDevice.length > 0))) {
    fail("invalid_schema", "WebGPU adapter identity is invalid");
  }
  const supportedOperations = parseTrainingManifest(canonicalTrainingManifestJson())
    .operations
    .filter((operation) => operation.category !== "lifecycle")
    .map((operation) => operation.id);
  const capabilities: WebTrainingCapabilitiesV1 = Object.freeze({
    schemaId: "tritium.web_training_capabilities",
    schemaVersion: 1,
    implementation: "webgpu",
    manifestDigest: TRAINING_MANIFEST_DIGEST_V1,
    vectorDigest: TRAINING_VECTOR_DIGEST_V1,
    buildId,
    physicalDevice,
    supportedOperations: Object.freeze(supportedOperations),
    maxResidentBytes: admittedMaxResidentBytes,
  });
  return new ResidentWebGpuTrainingAdapter(device, capabilities, uniformStride);
}

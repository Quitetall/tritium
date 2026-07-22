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
import {
  PortableWasmLifecycleError,
  PortableWasmLifecycleState,
} from "./portable-state.ts";
import type {
  PortableAdamLeafV1,
  PortableCheckpointOptimizerV1,
  PortableCheckpointStateV1,
  PortableInt8AdamLeafV1,
  PortableMuonLeafV1,
  PortableSgdLeafV1,
} from "./lifecycle-types.js";
import {
  compileSaltExportTargets,
  encodeStateDerivedSaltV2,
  SaltExportError,
} from "./salt-export.ts";
import type { PortableScheduleTensorV1 } from "./portable-schedule-types.js";
import type {
  CompiledTrainingBufferV1,
  CompiledTrainingOperationV1,
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
import type { WebTrainingErrorCode } from "./session.ts";
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

function fail(code: WebTrainingErrorCode, message: string): never {
  throw new WebTrainingError(code, message);
}

function rejectPreDispatchCancellation(signal?: AbortSignal | null): void {
  if (signal?.aborted === true) fail("cancelled", "WebGPU operation was cancelled before dispatch");
}

function rejectBeforeCommitCancellation(signal?: AbortSignal | null): void {
  if (signal?.aborted === true) fail("cancelled", "WebGPU lifecycle was cancelled before commit");
}

async function cancellable<T>(
  operation: () => Promise<T>, signal: AbortSignal | null | undefined, action: string,
): Promise<T> {
  rejectPreDispatchCancellation(signal);
  if (signal === null || signal === undefined) return operation();
  let abort: (() => void) | null = null;
  try {
    const cancelled = new Promise<never>((_resolve, reject) => {
      abort = () => reject(new WebTrainingError(
        "cancelled", `WebGPU ${action} was cancelled before commit`,
      ));
      signal.addEventListener("abort", abort, { once: true });
    });
    return await Promise.race([operation(), cancelled]);
  } finally {
    if (abort !== null) signal.removeEventListener("abort", abort);
  }
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

function optimizerKind(operation: string): PortableCheckpointOptimizerV1 {
  const value = operation.slice("optimizer.".length);
  if (["sgd", "adamw", "cautious_adamw", "int8_adamw", "muon"].includes(value)) {
    return value as PortableCheckpointOptimizerV1;
  }
  fail("invalid_schema", `unsupported WebGPU checkpoint optimizer ${operation}`);
}

function rawF32Bits(value: Uint8Array, name: string): readonly number[] {
  if (value.byteLength % 4 !== 0) {
    fail("invalid_schema", `WebGPU checkpoint plane ${name} is not f32-aligned`);
  }
  const view = new DataView(value.buffer, value.byteOffset, value.byteLength);
  return Object.freeze(
    Array.from({ length: value.byteLength / 4 }, (_, index) => view.getUint32(index * 4, true)),
  );
}

function f32Bytes(bits: readonly number[], name: string): Uint8Array {
  const result = new Uint8Array(bits.length * 4);
  const view = new DataView(result.buffer);
  for (const [index, value] of bits.entries()) {
    if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
      fail("invalid_receipt", `WebGPU resume returned invalid ${name} bits`);
    }
    view.setUint32(index * 4, value, true);
  }
  return result;
}

function gpuBufferBytes(byteLength: number): number {
  if (!Number.isSafeInteger(byteLength) || byteLength < 0 ||
      byteLength > Number.MAX_SAFE_INTEGER - 3) {
    fail("memory_limit", "WebGPU lifecycle buffer size exceeds the safe integer range");
  }
  return Math.max(4, Math.ceil(byteLength / 4) * 4);
}

function normalizeLifecycleError(error: unknown, action: string): never {
  if (error instanceof WebTrainingError) throw error;
  if (error instanceof SaltExportError) {
    fail(error.code === "capacity" ? "memory_limit" : error.code, error.message);
  }
  if (error instanceof PortableWasmLifecycleError) {
    fail(
      error.code === "busy" ? "busy"
        : error.code === "disposed" ? "disposed" : "invalid_receipt",
      `WebGPU ${action} failed strict WASM admission: ${error.message}`,
    );
  }
  fail("adapter_failure", `WebGPU ${action} failed: ${String(error)}`);
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
  #maxPeakBytes: number | null = null;
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
    this.#maxPeakBytes = config.maxResidentBytes;
    return receipt(this.capabilities, "session.prepare", 0, schedule.peakBytes());
  }

  #ready(): Readonly<{
    plan: CompiledTrainingPlanV1;
    runtime: WebGpuResidentRuntimeV1;
    schedule: WebGpuResidentScheduleV1;
  }> {
    if (this.#disposed) fail("invalid_state", "WebGPU adapter is disposed");
    if (this.#plan === null || this.#runtime === null || this.#schedule === null ||
        this.#maxPeakBytes === null) {
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

  async #lifecycleState(signal?: AbortSignal | null): Promise<Readonly<{
    operations: readonly CompiledTrainingOperationV1[];
    state: PortableCheckpointStateV1;
  }>> {
    const { plan, runtime } = this.#ready();
    const operations = plan.operations.filter((operation) =>
      operation.operation.startsWith("optimizer."),
    );
    if (operations.length === 0) fail("invalid_schema", "compiled plan has no optimizer operations");
    const optimizer = optimizerKind(operations[0]!.operation);
    if (operations.some((operation) => optimizerKind(operation.operation) !== optimizer)) {
      fail("capability_mismatch", "WebGPU checkpoints require one optimizer kind");
    }
    const read = (id: string): Promise<Uint8Array> =>
      cancellable(() => runtime.read(id), signal, `lifecycle read ${id}`);
    let state: PortableCheckpointStateV1;
    if (optimizer === "sgd") {
      const leaves: PortableSgdLeafV1[] = [];
      for (const operation of operations) {
        const id = operation.inputs[0]!;
        leaves.push(Object.freeze({ parameter: rawF32Bits(await read(id), id) }));
      }
      state = Object.freeze({ optimizer, step: this.#completedSteps, leaves: Object.freeze(leaves) });
    } else if (optimizer === "adamw" || optimizer === "cautious_adamw") {
      const leaves: PortableAdamLeafV1[] = [];
      for (const operation of operations) {
        const [parameter, , moment1, moment2] = operation.inputs;
        leaves.push(Object.freeze({
          parameter: rawF32Bits(await read(parameter!), parameter!),
          moment1: rawF32Bits(await read(moment1!), moment1!),
          moment2: rawF32Bits(await read(moment2!), moment2!),
        }));
      }
      state = Object.freeze({ optimizer, step: this.#completedSteps, leaves: Object.freeze(leaves) });
    } else if (optimizer === "int8_adamw") {
      const leaves: PortableInt8AdamLeafV1[] = [];
      for (const operation of operations) {
        const [parameter, , moment1, moment2, moment1Scale, moment2Scale] = operation.inputs;
        leaves.push(Object.freeze({
          parameter: rawF32Bits(await read(parameter!), parameter!),
          moment1Q8: Object.freeze(Array.from(await read(moment1!))),
          moment2Q8: Object.freeze(Array.from(await read(moment2!))),
          moment1Scale: rawF32Bits(await read(moment1Scale!), moment1Scale!),
          moment2Scale: rawF32Bits(await read(moment2Scale!), moment2Scale!),
        }));
      }
      state = Object.freeze({ optimizer, step: this.#completedSteps, leaves: Object.freeze(leaves) });
    } else {
      const leaves: PortableMuonLeafV1[] = [];
      for (const operation of operations) {
        const [parameter, , momentum] = operation.inputs;
        leaves.push(Object.freeze({
          parameter: rawF32Bits(await read(parameter!), parameter!),
          momentum: rawF32Bits(await read(momentum!), momentum!),
        }));
      }
      state = Object.freeze({ optimizer, step: this.#completedSteps, leaves: Object.freeze(leaves) });
    }
    return Object.freeze({ operations: Object.freeze(operations), state });
  }

  #rootBuffer(
    buffers: ReadonlyMap<string, CompiledTrainingBufferV1>, id: string,
  ): CompiledTrainingBufferV1 {
    const buffer = buffers.get(id);
    const root = buffer === undefined ? undefined : buffers.get(buffer.ownerId);
    if (buffer === undefined || root === undefined || root.ownerId !== root.id) {
      fail("invalid_schema", `WebGPU lifecycle buffer ${id} has no root owner`);
    }
    return root;
  }

  async #applyLifecycleState(
    state: PortableCheckpointStateV1,
    operations: readonly CompiledTrainingOperationV1[],
    signal?: AbortSignal | null,
  ): Promise<number> {
    const { plan, runtime, schedule } = this.#ready();
    if (state.leaves.length !== operations.length ||
        operations.some((operation) => optimizerKind(operation.operation) !== state.optimizer)) {
      fail("invalid_receipt", "WebGPU resume changed optimizer topology");
    }
    const buffers = new Map(plan.buffers.map((buffer) => [buffer.id, buffer] as const));
    const candidates = new Map<string, Uint8Array>();
    const stageRootReplacement = (id: string, value: Uint8Array): void => {
      const root = this.#rootBuffer(buffers, id);
      if (candidates.has(root.id) || value.byteLength !== root.byteLength) {
        fail("invalid_receipt", `WebGPU resume changed ${id} layout or ownership`);
      }
      candidates.set(root.id, value);
    };
    for (const [index, operation] of operations.entries()) {
      const leaf = state.leaves[index]!;
      stageRootReplacement(operation.inputs[0]!, f32Bytes(leaf.parameter, operation.inputs[0]!));
      if (state.optimizer === "adamw" || state.optimizer === "cautious_adamw") {
        const typed = leaf as PortableAdamLeafV1;
        stageRootReplacement(operation.inputs[2]!, f32Bytes(typed.moment1, operation.inputs[2]!));
        stageRootReplacement(operation.inputs[3]!, f32Bytes(typed.moment2, operation.inputs[3]!));
      } else if (state.optimizer === "int8_adamw") {
        const typed = leaf as PortableInt8AdamLeafV1;
        stageRootReplacement(operation.inputs[2]!, Uint8Array.from(typed.moment1Q8));
        stageRootReplacement(operation.inputs[3]!, Uint8Array.from(typed.moment2Q8));
        stageRootReplacement(operation.inputs[4]!, f32Bytes(typed.moment1Scale, operation.inputs[4]!));
        stageRootReplacement(operation.inputs[5]!, f32Bytes(typed.moment2Scale, operation.inputs[5]!));
      } else if (state.optimizer === "muon") {
        const typed = leaf as PortableMuonLeafV1;
        stageRootReplacement(operation.inputs[2]!, f32Bytes(typed.momentum, operation.inputs[2]!));
      }
    }
    const candidateBytes = [...candidates.values()].reduce((total, value) => {
      const size = gpuBufferBytes(value.byteLength);
      if (total > Number.MAX_SAFE_INTEGER - size) {
        fail("memory_limit", "WebGPU resume candidate size exceeds the safe integer range");
      }
      return total + size;
    }, 0);
    const maxPeakBytes = this.#maxPeakBytes!;
    await runtime.replace(
      [...candidates].map(([bufferId, value]) => Object.freeze({ bufferId, bytes: value })),
      { residentPeakBytes: schedule.peakBytes(), maxPeakBytes },
      signal,
    );
    return schedule.peakBytes() + candidateBytes;
  }

  async #lifecycleController(state: PortableCheckpointStateV1): Promise<PortableWasmLifecycleState> {
    return PortableWasmLifecycleState.create({
      source: new URL("./tritium_wasm_bg.wasm", import.meta.url),
      state,
      physicalDevice: `webgpu:${this.capabilities.physicalDevice ?? "unknown"}`,
    });
  }

  async checkpoint(signal?: AbortSignal | null): Promise<WebBinaryResultV1> {
    const { schedule } = this.#ready();
    rejectPreDispatchCancellation(signal);
    let controller: PortableWasmLifecycleState | null = null;
    try {
      const { state } = await this.#lifecycleState(signal);
      controller = await this.#lifecycleController(state);
      rejectBeforeCommitCancellation(signal);
      const result = await controller.checkpoint();
      rejectBeforeCommitCancellation(signal);
      return Object.freeze({
        bytes: Uint8Array.from(result.bytes),
        receipt: receipt(
          this.capabilities, "session.checkpoint", this.#completedSteps, schedule.peakBytes(),
        ),
      });
    } catch (error) {
      return normalizeLifecycleError(error, "checkpoint");
    } finally {
      controller?.dispose();
    }
  }

  async resume(
    checkpoint: Uint8Array, signal?: AbortSignal | null,
  ): Promise<WebTrainingReceiptV1> {
    this.#ready();
    rejectPreDispatchCancellation(signal);
    if (!(checkpoint instanceof Uint8Array)) fail("invalid_schema", "checkpoint must be Uint8Array");
    let controller: PortableWasmLifecycleState | null = null;
    try {
      const current = await this.#lifecycleState(signal);
      controller = await this.#lifecycleController(current.state);
      rejectBeforeCommitCancellation(signal);
      await controller.resume(Uint8Array.from(checkpoint));
      rejectBeforeCommitCancellation(signal);
      const candidate = controller.state;
      const peakBytes = await this.#applyLifecycleState(candidate, current.operations, signal);
      this.#completedSteps = candidate.step;
      return receipt(
        this.capabilities, "session.resume", candidate.step, peakBytes,
      );
    } catch (error) {
      return normalizeLifecycleError(error, "resume");
    } finally {
      controller?.dispose();
    }
  }

  async export(signal?: AbortSignal | null): Promise<WebBinaryResultV1> {
    const { plan, runtime } = this.#ready();
    rejectPreDispatchCancellation(signal);
    let controller: PortableWasmLifecycleState | null = null;
    try {
      const lifecycle = await this.#lifecycleState(signal);
      const targets = compileSaltExportTargets(plan, false);
      const store: Record<string, PortableScheduleTensorV1> = {};
      for (const target of targets) {
        const value = await cancellable(
          () => runtime.read(target.ownerId), signal, `export read ${target.ownerId}`,
        );
        if (value.byteLength % 4 !== 0) {
          fail("invalid_state", `WebGPU export parameter ${target.ownerId} is not f32-aligned`);
        }
        store[target.ownerId] = new Float32Array(
          value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength),
        );
      }
      const artifact = encodeStateDerivedSaltV2(targets, store);
      controller = await this.#lifecycleController(lifecycle.state);
      rejectBeforeCommitCancellation(signal);
      const admitted = await controller.admitExport(artifact);
      rejectBeforeCommitCancellation(signal);
      return Object.freeze({
        bytes: Uint8Array.from(admitted.bytes),
        receipt: receipt(
          this.capabilities, "session.export", this.#completedSteps, plan.exportPeakBytes,
        ),
      });
    } catch (error) {
      return normalizeLifecycleError(error, "export");
    } finally {
      controller?.dispose();
    }
  }

  async dispose(): Promise<void> {
    if (this.#disposed) return;
    this.#disposed = true;
    if (this.#runtime === null) this.#device.destroy();
    else this.#runtime.dispose();
    this.#runtime = null;
    this.#schedule = null;
    this.#plan = null;
    this.#maxPeakBytes = null;
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

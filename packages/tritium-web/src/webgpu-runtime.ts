import type { CompiledTrainingPlanV1 } from "./session.ts";
import { WebTrainingError } from "./session.ts";
import {
  webGpuDispatchCatalogV2,
  webGpuDispatchFormV1,
  webGpuKernelCandidateBundleV1,
} from "./webgpu-kernels.ts";
import type {
  WebGpuDispatchExecutionV1,
  WebGpuKernelBindingV1,
} from "./webgpu-kernels.ts";

const COPY_SRC = 4;
const COPY_DST = 8;
const MAP_READ = 1;
const UNIFORM = 64;
const STORAGE = 128;
const UNIFORM_BYTES = 256;

export interface WebGpuBufferPortV1 {
  readonly size: number;
  mapAsync(mode: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

export interface WebGpuPipelinePortV1 {
  getBindGroupLayout(index: number): unknown;
}

export interface WebGpuComputePassPortV1 {
  setPipeline(pipeline: WebGpuPipelinePortV1): void;
  setBindGroup(index: number, bindGroup: unknown): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

export interface WebGpuCommandEncoderPortV1 {
  beginComputePass(descriptor?: Readonly<Record<string, unknown>>): WebGpuComputePassPortV1;
  copyBufferToBuffer(
    source: WebGpuBufferPortV1,
    sourceOffset: number,
    destination: WebGpuBufferPortV1,
    destinationOffset: number,
    size: number,
  ): void;
  clearBuffer(buffer: WebGpuBufferPortV1, offset?: number, size?: number): void;
  finish(): unknown;
}

export interface WebGpuDevicePortV1 {
  readonly limits: Readonly<{
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
    maxComputeWorkgroupsPerDimension: number;
    maxBindingsPerBindGroup: number;
    maxStorageBuffersPerShaderStage: number;
    maxUniformBuffersPerShaderStage: number;
    maxUniformBufferBindingSize: number;
    minUniformBufferOffsetAlignment: number;
  }>;
  readonly queue: Readonly<{
    writeBuffer(
      buffer: WebGpuBufferPortV1,
      bufferOffset: number,
      data: Uint8Array,
    ): void;
    submit(commands: readonly unknown[]): void;
    onSubmittedWorkDone(): Promise<void>;
  }>;
  readonly lost: Promise<unknown>;
  createShaderModule(descriptor: Readonly<{ label: string; code: string }>): unknown;
  createComputePipelineAsync(descriptor: Readonly<{
    label: string;
    layout: "auto";
    compute: Readonly<{ module: unknown; entryPoint: string }>;
  }>): Promise<WebGpuPipelinePortV1>;
  createBuffer(descriptor: Readonly<{
    label: string;
    size: number;
    usage: number;
  }>): WebGpuBufferPortV1;
  createBindGroup(descriptor: Readonly<{
    label: string;
    layout: unknown;
    entries: readonly Readonly<{
      binding: number;
      resource: Readonly<{
        buffer: WebGpuBufferPortV1;
        offset?: number;
        size?: number;
      }>;
    }>[];
  }>): unknown;
  createCommandEncoder(descriptor: Readonly<{ label: string }>): WebGpuCommandEncoderPortV1;
  destroy(): void;
}

export interface WebGpuResidentTensorV1 {
  readonly bufferId: string;
  readonly bytes: Uint8Array;
}

export interface WebGpuResidentAuxiliaryV1 {
  readonly id: string;
  readonly byteLength: number;
  readonly initialBytes: Uint8Array | null;
}

export interface WebGpuResidentAuxiliarySetV1 {
  readonly maxBytes: number;
  readonly resources: readonly WebGpuResidentAuxiliaryV1[];
}

export interface WebGpuResidentCopyV1 {
  readonly source: string;
  readonly sourceOffset: number;
  readonly destination: string;
  readonly destinationOffset: number;
  readonly byteLength: number;
}

export interface WebGpuResidentDispatchV1 {
  readonly operation: string;
  readonly execution: WebGpuDispatchExecutionV1;
  readonly stageIndex: number;
  readonly uniformSlot: number;
  readonly uniformBytes: Uint8Array | null;
  readonly storageBindings: Readonly<Record<number, string>>;
  readonly workgroups: readonly [number, number, number];
}

export interface WebGpuResidentSubmissionV1 {
  readonly commands: readonly WebGpuResidentDispatchV1[];
  readonly copies: readonly WebGpuResidentCopyV1[];
  readonly commitCopies: readonly WebGpuResidentCopyV1[];
}

type ResidentBuffer = Readonly<{
  id: string;
  ownerId: string;
  byteLength: number;
}>;
type PreparedStage = Readonly<{
  pipeline: WebGpuPipelinePortV1;
  bindings: readonly WebGpuKernelBindingV1[];
  hasUniform: boolean;
}>;

function fail(code: "adapter_unavailable" | "adapter_failure" | "cancelled" | "capability_mismatch" | "device_lost" | "invalid_schema" | "memory_limit", message: string): never {
  throw new WebTrainingError(code, message);
}

function safeLimit(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    fail("capability_mismatch", `WebGPU ${name} must be a positive safe integer`);
  }
  return value;
}

function key(operation: string, execution: string, stageIndex: number): string {
  return `${operation}|${execution}|${stageIndex}`;
}

function paddedBytes(bytes: number): number {
  return Math.max(4, Math.ceil(bytes / 4) * 4);
}

function denseArray(value: unknown): value is readonly unknown[] {
  if (!Array.isArray(value)) return false;
  for (let index = 0; index < value.length; index += 1) {
    if (!(index in value)) return false;
  }
  return true;
}

function record(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function property(
  value: Readonly<Record<string, unknown>>,
  key: string,
  name: string,
): unknown {
  try {
    return Reflect.get(value, key);
  } catch {
    fail("invalid_schema", `${name}.${key} could not be read`);
  }
}

function ownKeys(value: object, name: string): readonly PropertyKey[] {
  try {
    return Reflect.ownKeys(value);
  } catch {
    fail("invalid_schema", `${name} keys could not be read`);
  }
}

/** Derive the uniform arena capacity from a fully compiled plan. */
export function webGpuUniformSlotCapacityV1(plan: CompiledTrainingPlanV1): number {
  if (!record(plan) || !denseArray(plan.operations) || !denseArray(plan.backwardOperations)) {
    fail("invalid_schema", "WebGPU compiled operations must be dense arrays");
  }
  let total = 0;
  for (const [phase, operations] of [
    ["forward", plan.operations],
    ["backward", plan.backwardOperations],
  ] as const) {
    for (const operation of operations) {
      if (!record(operation) || !denseArray(operation.outputs) ||
          (phase === "forward"
            ? operation.outputs.some((output) => typeof output !== "string")
            : operation.outputs.some((output) =>
              !record(output) || typeof output.role !== "string" ||
              typeof output.bufferId !== "string"
            ))) {
        fail("invalid_schema", `WebGPU compiled ${phase} outputs must be arrays`);
      }
      const increment = Math.max(1, operation.outputs.length) * 8;
      if (!Number.isSafeInteger(increment) || total > Number.MAX_SAFE_INTEGER - increment) {
        fail("memory_limit", "WebGPU uniform slot capacity exceeds safe integer range");
      }
      total += increment;
    }
  }
  return Math.max(1, total);
}

export class WebGpuResidentRuntimeV1 {
  readonly #device: WebGpuDevicePortV1;
  readonly #resident: Map<string, WebGpuBufferPortV1>;
  readonly #zero: WebGpuBufferPortV1;
  readonly #uniformArena: WebGpuBufferPortV1;
  readonly #uniformSlots: number;
  readonly #uniformStride: number;
  readonly #buffers: ReadonlyMap<string, ResidentBuffer>;
  readonly #stages: ReadonlyMap<string, PreparedStage>;
  readonly #bindGroups = new Map<string, unknown>();
  #lost = false;
  #disposed = false;

  private constructor(
    device: WebGpuDevicePortV1,
    resident: Map<string, WebGpuBufferPortV1>,
    zero: WebGpuBufferPortV1,
    uniformArena: WebGpuBufferPortV1,
    uniformSlots: number,
    uniformStride: number,
    buffers: ReadonlyMap<string, ResidentBuffer>,
    stages: ReadonlyMap<string, PreparedStage>,
  ) {
    this.#device = device;
    this.#resident = resident;
    this.#zero = zero;
    this.#uniformArena = uniformArena;
    this.#uniformSlots = uniformSlots;
    this.#uniformStride = uniformStride;
    this.#buffers = buffers;
    this.#stages = stages;
    void device.lost.then(() => {
      this.#lost = true;
      this.#bindGroups.clear();
    });
  }

  static async prepare(
    device: WebGpuDevicePortV1,
    plan: CompiledTrainingPlanV1,
    initial: readonly WebGpuResidentTensorV1[],
    auxiliary: WebGpuResidentAuxiliarySetV1 = Object.freeze({
      maxBytes: 0,
      resources: Object.freeze([]),
    }),
    expectedUniformStride?: number,
  ): Promise<WebGpuResidentRuntimeV1> {
    const auxiliaryMaxBytes = record(auxiliary)
      ? property(auxiliary, "maxBytes", "WebGPU auxiliary set")
      : undefined;
    const auxiliaryResources = record(auxiliary)
      ? property(auxiliary, "resources", "WebGPU auxiliary set")
      : undefined;
    if (!record(plan) || !denseArray(plan.buffers) || !denseArray(plan.operations) ||
        !denseArray(plan.backwardOperations) || !denseArray(initial) ||
        !record(auxiliary) || !Number.isSafeInteger(auxiliaryMaxBytes) ||
        (auxiliaryMaxBytes as number) < 0 || !denseArray(auxiliaryResources)) {
      fail("invalid_schema", "WebGPU runtime inputs must be compiled arrays");
    }
    const capturedBuffers = plan.buffers.map((buffer) => {
      if (!record(buffer) || !denseArray(buffer.shape) ||
          buffer.shape.some((dimension) =>
            !Number.isSafeInteger(dimension) || (dimension as number) < 0
          )) {
        fail("invalid_schema", "WebGPU compiled buffer shape is invalid");
      }
      return Object.freeze({
        ...buffer,
        shape: Object.freeze([...buffer.shape]),
      }) as ResidentBuffer;
    });
    const capturedInitial = initial.map((tensor) => {
      if (!record(tensor) || typeof tensor.bufferId !== "string" ||
          !(tensor.bytes instanceof Uint8Array)) {
        fail("invalid_schema", "WebGPU initial tensors must contain owned bytes");
      }
      return Object.freeze({
        bufferId: tensor.bufferId,
        bytes: Uint8Array.from(tensor.bytes),
      });
    });
    const admittedAuxiliary = auxiliaryResources.map((resource) => {
      const id = record(resource)
        ? property(resource, "id", "WebGPU auxiliary resource")
        : undefined;
      const byteLength = record(resource)
        ? property(resource, "byteLength", "WebGPU auxiliary resource")
        : undefined;
      const initialBytes = record(resource)
        ? property(resource, "initialBytes", "WebGPU auxiliary resource")
        : undefined;
      if (!record(resource) || typeof id !== "string" || id.length === 0 ||
          !Number.isSafeInteger(byteLength) || (byteLength as number) <= 0 ||
          (byteLength as number) % 4 !== 0 ||
          !(initialBytes === null || initialBytes instanceof Uint8Array) ||
          (initialBytes !== null && initialBytes.byteLength !== byteLength)) {
        fail("invalid_schema", "WebGPU auxiliary resource is invalid");
      }
      return Object.freeze({
        id,
        ownerId: id,
        byteLength: byteLength as number,
        initialBytes,
      });
    });
    const uniformSlots = webGpuUniformSlotCapacityV1(plan);
    const reachableForms = new Set<string>();
    for (const operation of plan.operations) {
      if (!record(operation) || typeof operation.operation !== "string") {
        fail("invalid_schema", "WebGPU compiled operation ID must be a string");
      }
      reachableForms.add(`${operation.operation}|${
        operation.operation.startsWith("optimizer.") ? "step" : "forward"
      }`);
    }
    for (const operation of plan.backwardOperations) {
      if (!record(operation) || typeof operation.operation !== "string" ||
          !(operation.execution === "forward" || operation.execution === "vjp")) {
        fail("invalid_schema", "WebGPU backward operation identity is invalid");
      }
      reachableForms.add(`${operation.operation}|${operation.execution}`);
    }
    const catalogForms = webGpuDispatchCatalogV2().forms;
    for (const form of reachableForms) {
      if (catalogForms[form] === undefined) {
        fail("invalid_schema", `compiled plan references unknown WebGPU form ${form}`);
      }
    }
    const maxBufferSize = safeLimit(device.limits.maxBufferSize, "maxBufferSize");
    const maxStorage = safeLimit(
      device.limits.maxStorageBufferBindingSize,
      "maxStorageBufferBindingSize",
    );
    safeLimit(
      device.limits.maxComputeWorkgroupsPerDimension,
      "maxComputeWorkgroupsPerDimension",
    );
    const maxBindings = safeLimit(
      device.limits.maxBindingsPerBindGroup,
      "maxBindingsPerBindGroup",
    );
    const maxStorageBindings = safeLimit(
      device.limits.maxStorageBuffersPerShaderStage,
      "maxStorageBuffersPerShaderStage",
    );
    const maxUniformBindings = safeLimit(
      device.limits.maxUniformBuffersPerShaderStage,
      "maxUniformBuffersPerShaderStage",
    );
    const maxUniform = safeLimit(
      device.limits.maxUniformBufferBindingSize,
      "maxUniformBufferBindingSize",
    );
    const uniformStride = Math.max(
      UNIFORM_BYTES,
      safeLimit(
        device.limits.minUniformBufferOffsetAlignment,
        "minUniformBufferOffsetAlignment",
      ),
    );
    if (expectedUniformStride !== undefined && uniformStride !== expectedUniformStride) {
      fail("capability_mismatch", "WebGPU uniform alignment changed after admission");
    }
    if (maxUniform < UNIFORM_BYTES) {
      fail("capability_mismatch", "WebGPU uniform binding limit is below 256 bytes");
    }
    let auxiliaryBytes = 0;
    for (const resource of admittedAuxiliary) {
      if (paddedBytes(resource.byteLength) > maxStorage ||
          paddedBytes(resource.byteLength) > maxBufferSize) {
        fail("memory_limit", `${resource.id} exceeds WebGPU storage binding limit`);
      }
      if (auxiliaryBytes > (auxiliaryMaxBytes as number) - resource.byteLength) {
        fail("memory_limit", "WebGPU auxiliary resources exceed declared budget");
      }
      auxiliaryBytes += resource.byteLength;
    }
    const compiledBuffers = new Map<string, ResidentBuffer>();
    for (const buffer of capturedBuffers) {
      if (typeof buffer.id !== "string" || buffer.id.length === 0 ||
          typeof buffer.ownerId !== "string" ||
          !Number.isSafeInteger(buffer.byteLength) || buffer.byteLength < 0 ||
          compiledBuffers.has(buffer.id)) {
        fail("invalid_schema", "WebGPU compiled buffer ownership is invalid");
      }
      compiledBuffers.set(buffer.id, buffer);
      if (paddedBytes(buffer.byteLength) > maxStorage ||
          paddedBytes(buffer.byteLength) > maxBufferSize) {
        fail("memory_limit", `${buffer.id} exceeds WebGPU storage binding limit`);
      }
    }
    for (const buffer of capturedBuffers) {
      const owner = compiledBuffers.get(buffer.ownerId);
      if (owner === undefined || owner.ownerId !== owner.id ||
          owner.byteLength !== buffer.byteLength) {
        fail("invalid_schema", `${buffer.id} has invalid WebGPU root ownership`);
      }
    }
    const auxiliaryIds = new Set<string>();
    for (const resource of admittedAuxiliary) {
      if (compiledBuffers.has(resource.id) || auxiliaryIds.has(resource.id)) {
        fail("invalid_schema", `WebGPU auxiliary resource ${resource.id} collides`);
      }
      auxiliaryIds.add(resource.id);
    }
    const capturedAuxiliary = admittedAuxiliary.map((resource) => Object.freeze({
      ...resource,
      initialBytes: resource.initialBytes === null
        ? null
        : Uint8Array.from(resource.initialBytes),
    }));
    const buffers = new Map(compiledBuffers);
    for (const resource of capturedAuxiliary) {
      buffers.set(resource.id, resource);
    }
    const bundle = webGpuKernelCandidateBundleV1();
    const stages = new Map<string, PreparedStage>();
    let preparing = true;
    const lossDuringPrepare = device.lost.then(() => {
      if (preparing) fail("device_lost", "WebGPU device was lost during preparation");
      return new Promise<never>(() => {});
    });
    void lossDuringPrepare.catch(() => {});
    try {
      for (const form of Object.values(catalogForms)) {
        if (!reachableForms.has(`${form.operation}|${form.execution}`)) continue;
        for (const [stageIndex, stage] of form.stages.entries()) {
          const stageKey = key(form.operation, form.execution, stageIndex);
          const module = bundle.modules[stage.moduleId];
          const stageBindings = module?.entryPointBindings[stage.entryPoint];
          const storageBindings = stageBindings?.filter(
            (binding) => binding.addressSpace === "storage",
          ).length ?? 0;
          const uniformBindings = stageBindings?.filter(
            (binding) => binding.addressSpace === "uniform",
          ).length ?? 0;
          if (module === undefined || stageBindings === undefined ||
              stageBindings.length > maxBindings ||
              storageBindings > maxStorageBindings ||
              uniformBindings > maxUniformBindings) {
            fail("capability_mismatch", `${stage.moduleId} exceeds WebGPU binding limits`);
          }
          const shader = device.createShaderModule({
            label: `tritium:${stage.moduleId}`,
            code: module.source,
          });
          const pipeline = await Promise.race([
            device.createComputePipelineAsync({
              label: `tritium:${stageKey}`,
              layout: "auto",
              compute: { module: shader, entryPoint: stage.entryPoint },
            }),
            lossDuringPrepare,
          ]);
          const hasUniform = stageBindings.some(
            (binding) => binding.addressSpace === "uniform",
          );
          stages.set(stageKey, Object.freeze({
            pipeline,
            bindings: stageBindings,
            hasUniform,
          }));
        }
      }
      if (stages.size === 0) {
        fail("invalid_schema", "compiled plan has no reachable WebGPU dispatch stages");
      }
      const resident = new Map<string, WebGpuBufferPortV1>();
      for (const buffer of capturedBuffers) {
        if (buffer.ownerId !== buffer.id) continue;
        resident.set(buffer.id, device.createBuffer({
          label: `tritium:resident:${buffer.id}`,
          size: paddedBytes(buffer.byteLength),
          usage: STORAGE | COPY_SRC | COPY_DST,
        }));
      }
      for (const resource of capturedAuxiliary) {
        const allocated = device.createBuffer({
          label: `tritium:auxiliary:${resource.id}`,
          size: paddedBytes(resource.byteLength),
          usage: STORAGE | COPY_SRC | COPY_DST,
        });
        resident.set(resource.id, allocated);
        if (resource.initialBytes !== null) {
          device.queue.writeBuffer(allocated, 0, resource.initialBytes);
        }
      }
      const zero = device.createBuffer({
        label: "tritium:zero-binding",
        size: 4,
        usage: STORAGE | COPY_DST,
      });
      const uniformArenaBytes = uniformSlots * uniformStride;
      if (!Number.isSafeInteger(uniformArenaBytes) || uniformArenaBytes > maxBufferSize) {
        fail("memory_limit", "WebGPU uniform arena exceeds maxBufferSize");
      }
      const uniformArena = device.createBuffer({
        label: "tritium:uniform-arena",
        size: uniformArenaBytes,
        usage: UNIFORM | COPY_DST,
      });
      device.queue.writeBuffer(zero, 0, new Uint8Array(4));
      const seen = new Set<string>();
      for (const tensor of capturedInitial) {
        const buffer = buffers.get(tensor.bufferId);
        if (buffer === undefined || buffer.ownerId !== buffer.id ||
            auxiliaryIds.has(tensor.bufferId)) {
          fail("invalid_schema", `initial tensor ${tensor.bufferId} is not a root buffer`);
        }
        if (seen.has(buffer.id) || tensor.bytes.byteLength !== buffer.byteLength) {
          fail("invalid_schema", `initial tensor ${buffer.id} has invalid ownership or length`);
        }
        seen.add(buffer.id);
        if (tensor.bytes.byteLength > 0) {
          const upload = new Uint8Array(paddedBytes(tensor.bytes.byteLength));
          upload.set(tensor.bytes);
          device.queue.writeBuffer(
            resident.get(buffer.id)!,
            0,
            upload,
          );
        }
      }
      preparing = false;
      return new WebGpuResidentRuntimeV1(
        device,
        resident,
        zero,
        uniformArena,
        uniformSlots,
        uniformStride,
        buffers,
        stages,
      );
    } catch (error) {
      device.destroy();
      if (error instanceof WebTrainingError) throw error;
      fail("capability_mismatch", `WebGPU pipeline preparation failed: ${String(error)}`);
    }
  }

  dispatch(
    commands: readonly WebGpuResidentDispatchV1[],
    copies: readonly WebGpuResidentCopyV1[] = [],
    commitCopies: readonly WebGpuResidentCopyV1[] = [],
    clearBufferIds: readonly string[] = [],
  ): void {
    void this.#submitTransactions([
      Object.freeze({ commands, copies, commitCopies }),
    ], clearBufferIds).catch(() => {
      // The synchronous low-level API observes loss through the next operation.
    });
  }

  dispatchTransactions(
    transactions: readonly WebGpuResidentSubmissionV1[],
    clearBufferIds: readonly string[] = [],
  ): Promise<void> {
    return this.#submitTransactions(transactions, clearBufferIds);
  }

  #submitTransactions(
    transactions: readonly WebGpuResidentSubmissionV1[],
    clearBufferIds: readonly string[],
  ): Promise<void> {
    this.#ready();
    if (!denseArray(transactions) || !denseArray(clearBufferIds)) {
      fail("invalid_schema", "WebGPU transaction inputs must be dense arrays");
    }
    const clearOwners = new Set<string>();
    const capturedClears = clearBufferIds.map((bufferId) => {
      if (typeof bufferId !== "string") {
        fail("invalid_schema", "WebGPU clear target must be a buffer ID");
      }
      const buffer = this.#buffers.get(bufferId);
      if (buffer === undefined || clearOwners.has(buffer.ownerId)) {
        fail("invalid_schema", "WebGPU clear targets must have unique physical owners");
      }
      clearOwners.add(buffer.ownerId);
      return Object.freeze({ id: bufferId, byteLength: paddedBytes(buffer.byteLength) });
    });
    const captureCopies = (values: readonly WebGpuResidentCopyV1[]) => values.map((copy) => {
      const sourceId = record(copy) ? property(copy, "source", "WebGPU resident copy") : undefined;
      const destinationId = record(copy)
        ? property(copy, "destination", "WebGPU resident copy")
        : undefined;
      const sourceOffset = record(copy)
        ? property(copy, "sourceOffset", "WebGPU resident copy")
        : undefined;
      const destinationOffset = record(copy)
        ? property(copy, "destinationOffset", "WebGPU resident copy")
        : undefined;
      const byteLength = record(copy)
        ? property(copy, "byteLength", "WebGPU resident copy")
        : undefined;
      if (!record(copy) || typeof sourceId !== "string" ||
          typeof destinationId !== "string" ||
          !Number.isSafeInteger(sourceOffset) || (sourceOffset as number) < 0 ||
          !Number.isSafeInteger(destinationOffset) || (destinationOffset as number) < 0 ||
          !Number.isSafeInteger(byteLength) || (byteLength as number) <= 0 ||
          (sourceOffset as number) % 4 !== 0 ||
          (destinationOffset as number) % 4 !== 0 ||
          (byteLength as number) % 4 !== 0) {
        fail("invalid_schema", "WebGPU resident copy is malformed");
      }
      const safeSourceOffset = sourceOffset as number;
      const safeDestinationOffset = destinationOffset as number;
      const safeByteLength = byteLength as number;
      const source = this.#buffers.get(sourceId);
      const destination = this.#buffers.get(destinationId);
      const sourceEnd = safeSourceOffset + safeByteLength;
      const destinationEnd = safeDestinationOffset + safeByteLength;
      const sourceFits = source !== undefined && (sourceEnd <= source.byteLength ||
        (safeSourceOffset === 0 && safeByteLength === paddedBytes(source.byteLength)));
      const destinationFits = destination !== undefined &&
        (destinationEnd <= destination.byteLength ||
          (safeDestinationOffset === 0 &&
            safeByteLength === paddedBytes(destination.byteLength)));
      if (source === undefined || destination === undefined ||
          !Number.isSafeInteger(sourceEnd) || !sourceFits ||
          !Number.isSafeInteger(destinationEnd) || !destinationFits) {
        fail("invalid_schema", "WebGPU resident copy exceeds a resource view");
      }
      if (source.ownerId === destination.ownerId) {
        fail("invalid_schema", "WebGPU resident copy requires distinct physical buffers");
      }
      return Object.freeze({
        source: sourceId,
        sourceOffset: safeSourceOffset,
        destination: destinationId,
        destinationOffset: safeDestinationOffset,
        byteLength: safeByteLength,
      });
    });
    const captureCommands = (commands: readonly WebGpuResidentDispatchV1[]) => commands.map((command) => {
      if (!record(command)) fail("invalid_schema", "WebGPU dispatch command is malformed");
      const fields = ownKeys(command, "WebGPU dispatch command");
      const expected = [
        "execution",
        "operation",
        "stageIndex",
        "storageBindings",
        "uniformBytes",
        "uniformSlot",
        "workgroups",
      ];
      if (fields.some((field) => typeof field !== "string")) {
        fail("invalid_schema", "WebGPU dispatch command field must be a string");
      }
      const stringFields = [...(fields as readonly string[])].sort();
      if (stringFields.length !== expected.length ||
          stringFields.some((field, index) => field !== expected[index])) {
        fail("invalid_schema", "WebGPU dispatch command is malformed");
      }
      const operation = property(command, "operation", "WebGPU dispatch command");
      const execution = property(command, "execution", "WebGPU dispatch command");
      const stageIndex = property(command, "stageIndex", "WebGPU dispatch command");
      const uniformSlot = property(command, "uniformSlot", "WebGPU dispatch command");
      const uniformBytes = property(command, "uniformBytes", "WebGPU dispatch command");
      const storageBindings = property(command, "storageBindings", "WebGPU dispatch command");
      const workgroups = property(command, "workgroups", "WebGPU dispatch command");
      if (typeof operation !== "string" ||
          !(execution === "forward" || execution === "vjp" || execution === "step") ||
          !Number.isSafeInteger(stageIndex) || (stageIndex as number) < 0 ||
          !record(storageBindings) ||
          !(uniformBytes === null || uniformBytes instanceof Uint8Array) ||
          !denseArray(workgroups) || workgroups.length !== 3) {
        fail("invalid_schema", "WebGPU dispatch command is malformed");
      }
      const capturedBindings: Record<string, string> = {};
      for (const binding of ownKeys(storageBindings, "WebGPU storage bindings")) {
        if (typeof binding !== "string") {
          fail("invalid_schema", "WebGPU storage binding key must be a string");
        }
        const bufferId = property(storageBindings, binding, "WebGPU storage bindings");
        if (typeof bufferId !== "string") {
          fail("invalid_schema", "WebGPU storage binding value must be a string");
        }
        capturedBindings[binding] = bufferId;
      }
      return Object.freeze({
        operation,
        execution: execution as WebGpuDispatchExecutionV1,
        stageIndex: stageIndex as number,
        uniformSlot: uniformSlot as number,
        uniformBytes: uniformBytes === null
          ? null
          : Uint8Array.from(uniformBytes),
        storageBindings: Object.freeze(capturedBindings),
        workgroups: Object.freeze([...workgroups]) as readonly [number, number, number],
      });
    });
    const capturedTransactions = transactions.map((transaction) => {
      if (!record(transaction)) {
        fail("invalid_schema", "WebGPU transaction is malformed");
      }
      const commands = property(transaction, "commands", "WebGPU transaction");
      const copies = property(transaction, "copies", "WebGPU transaction");
      const commitCopies = property(transaction, "commitCopies", "WebGPU transaction");
      if (!denseArray(commands) || !denseArray(copies) || !denseArray(commitCopies)) {
        fail("invalid_schema", "WebGPU transaction is malformed");
      }
      const capturedCopies = captureCopies(copies as readonly WebGpuResidentCopyV1[]);
      const capturedCommitCopies = captureCopies(
        commitCopies as readonly WebGpuResidentCopyV1[],
      );
      const commitSourceOwners = new Set(capturedCommitCopies.map((copy) =>
        this.#buffers.get(copy.source)!.ownerId));
      const commitDestinationOwners = new Set<string>();
      for (const copy of capturedCommitCopies) {
        const owner = this.#buffers.get(copy.destination)!.ownerId;
        if (commitDestinationOwners.has(owner)) {
          fail("invalid_schema", "WebGPU commit copies require unique destination owners");
        }
        commitDestinationOwners.add(owner);
      }
      if ([...commitDestinationOwners].some((owner) => commitSourceOwners.has(owner))) {
        fail("invalid_schema", "WebGPU commit destinations cannot feed another commit copy");
      }
      return Object.freeze({
        commands: Object.freeze(captureCommands(
          commands as readonly WebGpuResidentDispatchV1[],
        )),
        copies: Object.freeze(capturedCopies),
        commitCopies: Object.freeze(capturedCommitCopies),
      });
    });
    const encoder = this.#device.createCommandEncoder({ label: "tritium:transaction" });
    for (const clear of capturedClears) {
      encoder.clearBuffer(this.#physicalBuffer(clear.id), 0, clear.byteLength);
    }
    const usedUniformSlots = new Set<number>();
    for (const transaction of capturedTransactions) {
      for (const copy of transaction.copies) {
        encoder.copyBufferToBuffer(
          this.#physicalBuffer(copy.source),
          copy.sourceOffset,
          this.#physicalBuffer(copy.destination),
          copy.destinationOffset,
          copy.byteLength,
        );
      }
      if (transaction.commands.length > 0) {
        const pass = encoder.beginComputePass({ label: "tritium:resident-dispatch" });
        for (const command of transaction.commands) {
          const form = webGpuDispatchFormV1(command.operation, command.execution);
          const descriptor = form.stages[command.stageIndex];
          const prepared = this.#stages.get(
            key(command.operation, command.execution, command.stageIndex),
          );
          if (descriptor === undefined || prepared === undefined) {
            fail("invalid_schema", "WebGPU dispatch stage index is invalid");
          }
          if (prepared.hasUniform &&
              (!Number.isSafeInteger(command.uniformSlot) || command.uniformSlot < 0 ||
                command.uniformSlot >= this.#uniformSlots ||
                usedUniformSlots.has(command.uniformSlot))) {
            fail("invalid_schema", "WebGPU transaction uniform slots must be unique and in range");
          }
          if (prepared.hasUniform) usedUniformSlots.add(command.uniformSlot);
          if (prepared.hasUniform !== (command.uniformBytes !== null)) {
            fail("invalid_schema", "WebGPU uniform presence differs from shader layout");
          }
          const expectedStorage = prepared.bindings
            .filter((binding) => binding.addressSpace === "storage")
            .map((binding) => String(binding.binding))
            .sort();
          const suppliedStorage = Object.keys(command.storageBindings).sort();
          if (expectedStorage.length !== suppliedStorage.length ||
              expectedStorage.some((binding, index) => binding !== suppliedStorage[index])) {
            fail("invalid_schema", "WebGPU storage bindings differ from shader layout");
          }
          const limit = this.#device.limits.maxComputeWorkgroupsPerDimension;
          if (command.workgroups.some((value) =>
            !Number.isSafeInteger(value) || value < 1 || value > limit
          )) {
            fail("memory_limit", "WebGPU dispatch exceeds workgroup limits");
          }
          const entries = prepared.bindings.map((binding) =>
            this.#bindingEntry(command, prepared, binding),
          );
          const signature = JSON.stringify([
            command.operation,
            command.execution,
            command.stageIndex,
            prepared.bindings.map((binding) => binding.addressSpace === "uniform"
              ? [binding.binding, "uniform", command.uniformSlot]
              : [binding.binding, "storage", command.storageBindings[binding.binding]]
            ),
          ]);
          let bindGroup = this.#bindGroups.get(signature);
          if (bindGroup === undefined) {
            bindGroup = this.#device.createBindGroup({
              label: `tritium:bindings:${signature}`,
              layout: prepared.pipeline.getBindGroupLayout(0),
              entries,
            });
            this.#bindGroups.set(signature, bindGroup);
          }
          pass.setPipeline(prepared.pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(...command.workgroups);
        }
        pass.end();
      }
      for (const copy of transaction.commitCopies) {
        encoder.copyBufferToBuffer(
          this.#physicalBuffer(copy.source),
          copy.sourceOffset,
          this.#physicalBuffer(copy.destination),
          copy.destinationOffset,
          copy.byteLength,
        );
      }
    }
    this.#device.queue.submit([encoder.finish()]);
    return Promise.race([
      this.#device.queue.onSubmittedWorkDone().catch(() =>
        fail("device_lost", "WebGPU queue rejected submitted work")
      ),
      this.#device.lost.then(() =>
        fail("device_lost", "WebGPU device was lost during submitted work")
      ),
    ]).then(() => undefined);
  }

  write(bufferId: string, bytes: Uint8Array): void {
    this.#ready();
    const buffer = this.#buffers.get(bufferId);
    if (buffer === undefined || buffer.ownerId !== buffer.id ||
        !(bytes instanceof Uint8Array) || bytes.byteLength !== buffer.byteLength) {
      fail("invalid_schema", "WebGPU resident write differs from a root buffer");
    }
    const upload = new Uint8Array(paddedBytes(bytes.byteLength));
    upload.set(bytes);
    this.#device.queue.writeBuffer(this.#physicalBuffer(bufferId), 0, upload);
  }

  /** Replace complete root owners only after every candidate upload succeeds. */
  async replace(
    tensors: readonly WebGpuResidentTensorV1[],
    budget: Readonly<{ residentPeakBytes: number; maxPeakBytes: number }>,
    signal?: AbortSignal | null,
  ): Promise<void> {
    this.#ready();
    if (!denseArray(tensors)) {
      fail("invalid_schema", "WebGPU replacement tensors must be a dense array");
    }
    const captured = tensors.map((tensor) => {
      if (!record(tensor) || typeof tensor.bufferId !== "string" ||
          !(tensor.bytes instanceof Uint8Array)) {
        fail("invalid_schema", "WebGPU replacement tensor is invalid");
      }
      const buffer = this.#buffers.get(tensor.bufferId);
      if (buffer === undefined || buffer.ownerId !== buffer.id ||
          tensor.bytes.byteLength !== buffer.byteLength) {
        fail("invalid_schema", "WebGPU replacement differs from a root buffer");
      }
      return Object.freeze({ bufferId: buffer.id, bytes: Uint8Array.from(tensor.bytes) });
    });
    const ids = new Set(captured.map((tensor) => tensor.bufferId));
    if (ids.size !== captured.length) {
      fail("invalid_schema", "WebGPU replacement contains duplicate root buffers");
    }
    const residentPeakBytes = record(budget)
      ? property(budget, "residentPeakBytes", "WebGPU replacement budget")
      : undefined;
    const maxPeakBytes = record(budget)
      ? property(budget, "maxPeakBytes", "WebGPU replacement budget")
      : undefined;
    if (!record(budget) || !Number.isSafeInteger(residentPeakBytes) ||
        !Number.isSafeInteger(maxPeakBytes) || (residentPeakBytes as number) < 0 ||
        (maxPeakBytes as number) < 0) {
      fail("invalid_schema", "WebGPU replacement budget is invalid");
    }
    const candidateBytes = captured.reduce((total, tensor) => {
      const size = paddedBytes(tensor.bytes.byteLength);
      if (total > Number.MAX_SAFE_INTEGER - size) {
        fail("memory_limit", "WebGPU replacement candidate size overflowed");
      }
      return total + size;
    }, 0);
    if ((residentPeakBytes as number) > (maxPeakBytes as number) - candidateBytes) {
      fail("memory_limit", "WebGPU atomic replacement exceeds maxPeakBytes");
    }
    if (signal?.aborted === true) {
      fail("cancelled", "WebGPU replacement was cancelled before allocation");
    }
    const candidates = new Map<string, WebGpuBufferPortV1>();
    let abort: (() => void) | null = null;
    try {
      for (const tensor of captured) {
        const allocated = this.#device.createBuffer({
          label: `tritium:replacement:${tensor.bufferId}`,
          size: paddedBytes(tensor.bytes.byteLength),
          usage: STORAGE | COPY_SRC | COPY_DST,
        });
        candidates.set(tensor.bufferId, allocated);
        if (tensor.bytes.byteLength > 0) {
          const upload = new Uint8Array(paddedBytes(tensor.bytes.byteLength));
          upload.set(tensor.bytes);
          this.#device.queue.writeBuffer(allocated, 0, upload);
        }
      }
      const cancelled = new Promise<never>((_resolve, reject) => {
        if (signal === null || signal === undefined) return;
        abort = () => reject(new WebTrainingError(
          "cancelled", "WebGPU replacement was cancelled before commit",
        ));
        signal.addEventListener("abort", abort, { once: true });
      });
      await Promise.race([
        this.#device.queue.onSubmittedWorkDone().catch(() =>
          fail("device_lost", "WebGPU replacement upload failed")
        ),
        this.#device.lost.then(() =>
          fail("device_lost", "WebGPU device was lost during replacement")
        ),
        cancelled,
      ]);
    } catch (error) {
      for (const candidate of candidates.values()) candidate.destroy();
      if (error instanceof WebTrainingError) throw error;
      fail("adapter_failure", `WebGPU replacement upload failed: ${String(error)}`);
    } finally {
      if (abort !== null) signal?.removeEventListener("abort", abort);
    }
    const replaced: WebGpuBufferPortV1[] = [];
    for (const [bufferId, candidate] of candidates) {
      const previous = this.#resident.get(bufferId);
      if (previous === undefined) {
        candidate.destroy();
        fail("invalid_schema", `WebGPU resident owner ${bufferId} is missing`);
      }
      this.#resident.set(bufferId, candidate);
      replaced.push(previous);
    }
    this.#bindGroups.clear();
    for (const previous of replaced) previous.destroy();
  }

  async read(bufferId: string): Promise<Uint8Array> {
    this.#ready();
    const buffer = this.#buffers.get(bufferId);
    if (buffer === undefined) fail("invalid_schema", `unknown resident buffer ${bufferId}`);
    if (buffer.byteLength === 0) return new Uint8Array();
    const resident = this.#resident.get(buffer.ownerId);
    if (resident === undefined) {
      fail("invalid_schema", `WebGPU resident owner ${buffer.ownerId} is missing`);
    }
    const transferBytes = paddedBytes(buffer.byteLength);
    const staging = this.#device.createBuffer({
      label: `tritium:readback:${bufferId}`,
      size: transferBytes,
      usage: MAP_READ | COPY_DST,
    });
    try {
      const encoder = this.#device.createCommandEncoder({ label: "tritium:explicit-readback" });
      encoder.copyBufferToBuffer(
        resident,
        0,
        staging,
        0,
        transferBytes,
      );
      this.#device.queue.submit([encoder.finish()]);
      await Promise.race([
        staging.mapAsync(MAP_READ).catch(() =>
          fail("device_lost", "WebGPU readback mapping failed after submission")
        ),
        this.#device.lost.then(() =>
          fail("device_lost", "WebGPU device was lost during explicit readback")
        ),
      ]);
      return Uint8Array.from(
        new Uint8Array(staging.getMappedRange(0, transferBytes)).subarray(0, buffer.byteLength),
      );
    } finally {
      staging.unmap();
      staging.destroy();
    }
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    for (const buffer of this.#resident.values()) buffer.destroy();
    this.#zero.destroy();
    this.#uniformArena.destroy();
    this.#bindGroups.clear();
    this.#device.destroy();
  }

  #bindingEntry(
    command: WebGpuResidentDispatchV1,
    prepared: PreparedStage,
    binding: WebGpuKernelBindingV1,
  ): Readonly<{
    binding: number;
    resource: Readonly<{ buffer: WebGpuBufferPortV1; offset?: number; size?: number }>;
  }> {
    if (binding.addressSpace === "uniform") {
      if (!prepared.hasUniform || command.uniformBytes === null ||
          command.uniformBytes.byteLength === 0 || command.uniformBytes.byteLength > UNIFORM_BYTES) {
        fail("invalid_schema", "WebGPU uniform payload is missing or oversized");
      }
      const offset = command.uniformSlot * this.#uniformStride;
      const upload = new Uint8Array(UNIFORM_BYTES);
      upload.set(command.uniformBytes);
      this.#device.queue.writeBuffer(
        this.#uniformArena,
        offset,
        upload,
      );
      return Object.freeze({ binding: binding.binding, resource: Object.freeze({
        buffer: this.#uniformArena,
        offset,
        size: UNIFORM_BYTES,
      }) });
    }
    const bufferId = command.storageBindings[binding.binding];
    const buffer = bufferId === undefined ? undefined : this.#buffers.get(bufferId);
    if (buffer === undefined) {
      fail("invalid_schema", `WebGPU storage binding ${binding.binding} is missing`);
    }
    if (buffer.byteLength === 0) {
      return Object.freeze({ binding: binding.binding, resource: Object.freeze({
        buffer: this.#zero,
        offset: 0,
        size: 4,
      }) });
    }
    const resident = this.#resident.get(buffer.ownerId);
    if (resident === undefined) {
      fail("invalid_schema", `WebGPU resident owner ${buffer.ownerId} is missing`);
    }
    return Object.freeze({ binding: binding.binding, resource: Object.freeze({
      buffer: resident,
      offset: 0,
      size: paddedBytes(buffer.byteLength),
    }) });
  }

  #physicalBuffer(bufferId: string): WebGpuBufferPortV1 {
    const buffer = this.#buffers.get(bufferId);
    const resident = buffer === undefined ? undefined : this.#resident.get(buffer.ownerId);
    if (buffer === undefined || resident === undefined) {
      fail("invalid_schema", `WebGPU resident resource ${bufferId} is missing`);
    }
    return resident;
  }

  #ready(): void {
    if (this.#disposed) fail("invalid_schema", "WebGPU resident runtime is disposed");
    if (this.#lost) fail("device_lost", "WebGPU device was lost");
  }
}

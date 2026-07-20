import type { CompiledTrainingPlanV1 } from "./session.ts";
import { WebTrainingError } from "./session.ts";
import {
  webGpuDispatchCatalogV1,
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

function fail(code: "adapter_unavailable" | "capability_mismatch" | "device_lost" | "invalid_schema" | "memory_limit", message: string): never {
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
  readonly #resident: ReadonlyMap<string, WebGpuBufferPortV1>;
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
    resident: ReadonlyMap<string, WebGpuBufferPortV1>,
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
    const catalogForms = webGpuDispatchCatalogV1().forms;
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
          const storageBindings = module?.bindings.filter(
            (binding) => binding.addressSpace === "storage",
          ).length ?? 0;
          const uniformBindings = module?.bindings.filter(
            (binding) => binding.addressSpace === "uniform",
          ).length ?? 0;
          if (module === undefined || module.bindings.length > maxBindings ||
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
          const hasUniform = module.bindings.some(
            (binding) => binding.addressSpace === "uniform",
          );
          stages.set(stageKey, Object.freeze({
            pipeline,
            bindings: module.bindings,
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
  ): void {
    this.#ready();
    if (!denseArray(commands) || !denseArray(copies)) {
      fail("invalid_schema", "WebGPU transaction inputs must be dense arrays");
    }
    const capturedCopies = copies.map((copy) => {
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
      if (source === undefined || destination === undefined ||
          !Number.isSafeInteger(sourceEnd) || sourceEnd > source.byteLength ||
          !Number.isSafeInteger(destinationEnd) || destinationEnd > destination.byteLength) {
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
    const capturedCommands = commands.map((command) => {
      if (!record(command)) fail("invalid_schema", "WebGPU dispatch command is malformed");
      const fields = Object.keys(command).sort();
      const expected = [
        "execution",
        "operation",
        "stageIndex",
        "storageBindings",
        "uniformBytes",
        "uniformSlot",
        "workgroups",
      ];
      if (fields.length !== expected.length ||
          fields.some((field, index) => field !== expected[index]) ||
          typeof command.operation !== "string" ||
          !(["forward", "vjp", "step"] as const).includes(command.execution) ||
          !Number.isSafeInteger(command.stageIndex) || command.stageIndex < 0 ||
          !record(command.storageBindings) ||
          !(command.uniformBytes === null || command.uniformBytes instanceof Uint8Array) ||
          !denseArray(command.workgroups) || command.workgroups.length !== 3) {
        fail("invalid_schema", "WebGPU dispatch command is malformed");
      }
      return Object.freeze({
        operation: command.operation,
        execution: command.execution,
        stageIndex: command.stageIndex,
        uniformSlot: command.uniformSlot,
        uniformBytes: command.uniformBytes === null
          ? null
          : Uint8Array.from(command.uniformBytes),
        storageBindings: Object.freeze({ ...command.storageBindings }),
        workgroups: Object.freeze([...command.workgroups]) as readonly [number, number, number],
      });
    });
    const encoder = this.#device.createCommandEncoder({ label: "tritium:transaction" });
    for (const copy of capturedCopies) {
      encoder.copyBufferToBuffer(
        this.#physicalBuffer(copy.source),
        copy.sourceOffset,
        this.#physicalBuffer(copy.destination),
        copy.destinationOffset,
        copy.byteLength,
      );
    }
    const pass = encoder.beginComputePass({ label: "tritium:resident-dispatch" });
    const usedUniformSlots = new Set<number>();
    for (const command of capturedCommands) {
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
    this.#device.queue.submit([encoder.finish()]);
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
        staging.mapAsync(MAP_READ),
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

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

export interface WebGpuResidentDispatchV1 {
  readonly operation: string;
  readonly execution: WebGpuDispatchExecutionV1;
  readonly stageIndex: number;
  readonly uniformSlot: number;
  readonly uniformBytes: Uint8Array | null;
  readonly storageBindings: Readonly<Record<number, string>>;
  readonly workgroups: readonly [number, number, number];
}

type ResidentBuffer = CompiledTrainingPlanV1["buffers"][number];
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
  ): Promise<WebGpuResidentRuntimeV1> {
    if (!record(plan) || !denseArray(plan.buffers) || !denseArray(plan.operations) ||
        !denseArray(plan.backwardOperations) || !denseArray(initial)) {
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
    const uniformSlots = Math.max(
      1,
      [...plan.operations, ...plan.backwardOperations].reduce((total, operation) => {
        if (!record(operation) || !denseArray(operation.outputs) ||
            operation.outputs.some((output) => typeof output !== "string")) {
          fail("invalid_schema", "WebGPU compiled operation outputs must be arrays");
        }
        return total + Math.max(1, operation.outputs.length) * 8;
      }, 0),
    );
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
    const buffers = new Map<string, ResidentBuffer>();
    for (const buffer of capturedBuffers) {
      if (typeof buffer.id !== "string" || buffer.id.length === 0 ||
          typeof buffer.ownerId !== "string" ||
          !Number.isSafeInteger(buffer.byteLength) || buffer.byteLength < 0 ||
          buffers.has(buffer.id)) {
        fail("invalid_schema", "WebGPU compiled buffer ownership is invalid");
      }
      buffers.set(buffer.id, buffer);
      if (paddedBytes(buffer.byteLength) > maxStorage ||
          paddedBytes(buffer.byteLength) > maxBufferSize) {
        fail("memory_limit", `${buffer.id} exceeds WebGPU storage binding limit`);
      }
    }
    for (const buffer of capturedBuffers) {
      const owner = buffers.get(buffer.ownerId);
      if (owner === undefined || owner.ownerId !== owner.id ||
          owner.byteLength !== buffer.byteLength) {
        fail("invalid_schema", `${buffer.id} has invalid WebGPU root ownership`);
      }
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
        if (buffer === undefined || buffer.ownerId !== buffer.id) {
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

  dispatch(commands: readonly WebGpuResidentDispatchV1[]): void {
    this.#ready();
    if (!denseArray(commands)) fail("invalid_schema", "WebGPU commands must be a dense array");
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

  #ready(): void {
    if (this.#disposed) fail("invalid_schema", "WebGPU resident runtime is disposed");
    if (this.#lost) fail("device_lost", "WebGPU device was lost");
  }
}

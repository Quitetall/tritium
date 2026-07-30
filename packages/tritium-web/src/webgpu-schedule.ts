import {
  admittedWebGpuBuffersV1,
  compiledWebGpuInvocationV1,
  isPointwiseWebGpuOperationV1,
  lowerPointwiseWebGpuOperationV1,
  requiredWebGpuRoleV1,
  webGpuF32V1,
  webGpuU32V1,
  type WebGpuLoweringInvocationV1,
} from "./webgpu-lowering.ts";
import type { CompiledTrainingPlanV1, TrainingAttributeSpecV1 } from "./session.ts";
import { WebTrainingError } from "./session.ts";
import { webGpuDispatchFormV1 } from "./webgpu-kernels.ts";
import {
  webGpuUniformSlotCapacityV1,
  type WebGpuResidentAuxiliarySetV1,
  type WebGpuResidentAuxiliaryV1,
  type WebGpuResidentCopyV1,
  type WebGpuResidentDispatchV1,
} from "./webgpu-runtime.ts";

export interface WebGpuResidentTransactionV1 {
  readonly commands: readonly WebGpuResidentDispatchV1[];
  readonly copies: readonly WebGpuResidentCopyV1[];
  readonly commitCopies: readonly WebGpuResidentCopyV1[];
}

export interface WebGpuResidentScheduleBudgetV1 {
  readonly maxPeakBytes: number;
  readonly uniformStride?: number;
}

export interface WebGpuResidentScheduleV1 {
  auxiliaryResources(): WebGpuResidentAuxiliarySetV1;
  peakBytes(): number;
  transaction(
    phase: "forward" | "backward",
    operationId: string,
    firstUniformSlot: number,
    optimizerStep?: number,
  ): WebGpuResidentTransactionV1;
}

type BufferMap = ReturnType<typeof admittedWebGpuBuffersV1>;
type Template = Readonly<{
  commands: readonly WebGpuResidentDispatchV1[];
  copies: readonly WebGpuResidentCopyV1[];
  commitCopies?: readonly WebGpuResidentCopyV1[];
  commandFactory?: (optimizerStep: number) => readonly WebGpuResidentDispatchV1[];
}>;
type PendingResource = Readonly<{
  id: string;
  byteLength: number;
  initialValues: readonly number[] | null;
}>;

const SPECIALIZED = new Set([
  "graph.salt_ste",
  "graph.fsq",
  "graph.embedding_gather",
  "graph.rope",
  "graph.concat_cols",
  "graph.conv1d",
  "graph.conv2d",
  "graph.attention",
  "loss.softmax_cross_entropy",
  "loss.topk_knowledge_distillation",
  "optimizer.sgd",
  "optimizer.adamw",
  "optimizer.cautious_adamw",
  "optimizer.int8_adamw",
  "optimizer.muon",
]);

function fail(
  code: "capability_mismatch" | "invalid_schema" | "memory_limit",
  message: string,
): never {
  throw new WebTrainingError(code, message);
}

function key(phase: "forward" | "backward", operationId: string): string {
  return JSON.stringify([phase, operationId]);
}

function dense(values: readonly unknown[]): boolean {
  return Object.keys(values).length === values.length;
}

function property(value: object, name: string, context: string): unknown {
  try {
    return Reflect.get(value, name);
  } catch {
    fail("invalid_schema", `${context}.${name} could not be read`);
  }
}

function arraySnapshot<T>(
  value: unknown,
  context: string,
  capture: (item: unknown, context: string) => T,
): readonly T[] {
  if (!Array.isArray(value)) fail("invalid_schema", `${context} must be an array`);
  let keys: string[];
  let length: unknown;
  try {
    keys = Object.keys(value);
    length = Reflect.get(value, "length");
  } catch {
    fail("invalid_schema", `${context} could not be read`);
  }
  if (!Number.isSafeInteger(length) || (length as number) < 0 ||
      keys.length !== length || keys.some((key, index) => key !== String(index))) {
    fail("invalid_schema", `${context} must be dense without extra fields`);
  }
  const result: T[] = [];
  for (let index = 0; index < (length as number); index += 1) {
    result.push(capture(property(value, String(index), context), `${context}[${index}]`));
  }
  return Object.freeze(result);
}

function recordSnapshot(value: unknown, context: string): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail("invalid_schema", `${context} must be an object`);
  }
  return value as Readonly<Record<string, unknown>>;
}

function snapshotPlan(value: CompiledTrainingPlanV1): CompiledTrainingPlanV1 {
  const source = recordSnapshot(value, "compiled plan");
  const captureShape = (shape: unknown, context: string) =>
    arraySnapshot(shape, context, (dimension) => dimension as number);
  const captureAttribute = (item: unknown, context: string) => {
    const attribute = recordSnapshot(item, context);
    const rawValue = property(attribute, "value", context);
    const capturedValue = Array.isArray(rawValue)
      ? arraySnapshot(rawValue, `${context}.value`, (entry) => entry as number)
      : rawValue;
    return Object.freeze({
      name: property(attribute, "name", context),
      kind: property(attribute, "kind", context),
      value: capturedValue,
    }) as TrainingAttributeSpecV1;
  };
  const captureBinding = (item: unknown, context: string) => {
    const binding = recordSnapshot(item, context);
    return Object.freeze({
      role: property(binding, "role", context),
      bufferId: property(binding, "bufferId", context),
    }) as CompiledTrainingPlanV1["backwardOperations"][number]["inputs"][number];
  };
  const buffers = arraySnapshot(property(source, "buffers", "compiled plan"), "buffers", (item, context) => {
    const buffer = recordSnapshot(item, context);
    return Object.freeze({
      id: property(buffer, "id", context),
      role: property(buffer, "role", context),
      dtype: property(buffer, "dtype", context),
      shape: captureShape(property(buffer, "shape", context), `${context}.shape`),
      aliasOf: property(buffer, "aliasOf", context),
      ownerId: property(buffer, "ownerId", context),
      byteOffset: property(buffer, "byteOffset", context),
      byteLength: property(buffer, "byteLength", context),
      backwardInitialization: property(buffer, "backwardInitialization", context),
    }) as CompiledTrainingPlanV1["buffers"][number];
  });
  const operations = arraySnapshot(
    property(source, "operations", "compiled plan"), "operations", (item, context) => {
      const operation = recordSnapshot(item, context);
      return Object.freeze({
        id: property(operation, "id", context),
        operation: property(operation, "operation", context),
        inputs: arraySnapshot(property(operation, "inputs", context), `${context}.inputs`,
          (entry) => entry as string),
        outputs: arraySnapshot(property(operation, "outputs", context), `${context}.outputs`,
          (entry) => entry as string),
        attributes: arraySnapshot(
          property(operation, "attributes", context), `${context}.attributes`, captureAttribute,
        ),
      }) as CompiledTrainingPlanV1["operations"][number];
    },
  );
  const backwardOperations = arraySnapshot(
    property(source, "backwardOperations", "compiled plan"),
    "backwardOperations",
    (item, context) => {
      const operation = recordSnapshot(item, context);
      return Object.freeze({
        id: property(operation, "id", context),
        sourceOperationId: property(operation, "sourceOperationId", context),
        operation: property(operation, "operation", context),
        execution: property(operation, "execution", context),
        inputs: arraySnapshot(
          property(operation, "inputs", context), `${context}.inputs`, captureBinding,
        ),
        outputs: arraySnapshot(
          property(operation, "outputs", context), `${context}.outputs`, captureBinding,
        ),
        attributes: arraySnapshot(
          property(operation, "attributes", context), `${context}.attributes`, captureAttribute,
        ),
      }) as CompiledTrainingPlanV1["backwardOperations"][number];
    },
  );
  const scalar = (name: keyof CompiledTrainingPlanV1) => property(source, name, "compiled plan");
  return Object.freeze({
    schemaId: scalar("schemaId"),
    schemaVersion: scalar("schemaVersion"),
    manifestDigest: scalar("manifestDigest"),
    buffers,
    operations,
    backwardOperations,
    residentBytes: scalar("residentBytes"),
    batchStagingBytes: scalar("batchStagingBytes"),
    preparePeakBytes: scalar("preparePeakBytes"),
    forwardPeakBytes: scalar("forwardPeakBytes"),
    exportPackageBytes: scalar("exportPackageBytes"),
    exportPeakBytes: scalar("exportPeakBytes"),
    peakBytes: scalar("peakBytes"),
  }) as CompiledTrainingPlanV1;
}

function u32List(value: unknown, name: string): readonly number[] {
  if (!Array.isArray(value) || !dense(value)) {
    fail("invalid_schema", `${name} must be a dense u32 list`);
  }
  return Object.freeze(value.map((item) => webGpuU32V1(item, name)));
}

function safeU64(value: unknown, name: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    fail("invalid_schema", `${name} must be a nonnegative safe integer`);
  }
  return value as number;
}

function positiveU32(value: unknown, name: string): number {
  const result = webGpuU32V1(value, name);
  if (result === 0) fail("invalid_schema", `${name} must be positive`);
  return result;
}

function product(name: string, ...values: number[]): number {
  let result = 1;
  for (const value of values) {
    result *= value;
    if (!Number.isSafeInteger(result) || result > 0xffff_ffff) {
      fail("invalid_schema", `${name} exceeds u32 geometry`);
    }
  }
  return result;
}

function sumU32(name: string, ...values: number[]): number {
  let result = 0;
  for (const value of values) {
    result += value;
    if (!Number.isSafeInteger(result) || result > 0xffff_ffff) {
      fail("invalid_schema", `${name} exceeds u32 geometry`);
    }
  }
  return result;
}

function convAxis(
  input: number,
  kernel: number,
  stride: number,
  dilation: number,
  before: number,
  after: number,
  name: string,
): number {
  const effective = sumU32(name, product(name, kernel - 1, dilation), 1);
  const padded = sumU32(name, input, before, after);
  if (padded < effective) fail("invalid_schema", `${name} kernel exceeds padded input`);
  return Math.floor((padded - effective) / stride) + 1;
}

function uniform(bytes: number, write: (view: DataView) => void): Uint8Array {
  const result = new Uint8Array(bytes);
  write(new DataView(result.buffer));
  return result;
}

function u32Bytes(values: readonly number[]): Uint8Array {
  return uniform(values.length * 4, (view) => {
    values.forEach((value, index) => view.setUint32(index * 4, value, true));
  });
}

// Match Rust f32::powi/LLVM powi: binary32 rounding occurs after every
// multiply, not once after a double-precision Math.pow.
function powiF32(base: number, exponent: number): number {
  let factor = base;
  let power = exponent;
  let result = Math.fround(1);
  while (power !== 0) {
    if (power % 2 === 1) result = Math.fround(result * factor);
    power = Math.floor(power / 2);
    if (power !== 0) factor = Math.fround(factor * factor);
  }
  return result;
}

function expect(
  buffers: BufferMap,
  bufferId: string,
  dtype: "f32" | "u32" | "bytes",
  shape: readonly number[],
  role: string,
): void {
  const buffer = buffers.get(bufferId);
  if (buffer === undefined || buffer.dtype !== dtype ||
      buffer.shape.length !== shape.length ||
      buffer.shape.some((dimension, index) => dimension !== shape[index])) {
    fail("invalid_schema", `${role} differs from specialized WebGPU geometry`);
  }
}

function requireDisjointWrites(
  buffers: BufferMap,
  reads: readonly string[],
  writes: readonly string[],
  operation: string,
): void {
  const readOwners = new Set(reads.map((id) => buffers.get(id)!.ownerId));
  const writeOwners = new Set<string>();
  for (const id of writes) {
    const owner = buffers.get(id)!.ownerId;
    if (readOwners.has(owner) || writeOwners.has(owner)) {
      fail("invalid_schema", `${operation} writes must have distinct non-input owners`);
    }
    writeOwners.add(owner);
  }
}

function indexedStage(
  invocation: WebGpuLoweringInvocationV1,
  stageIndex: number,
  expectedModuleId: string,
  uniformBytes: Uint8Array,
  storageBindings: Readonly<Record<number, string>>,
  workgroups: readonly [number, number, number],
  expectedRepeat: "once" | "per_output" = "once",
  expectedEntryPoint = "main",
): WebGpuResidentDispatchV1 {
  const form = webGpuDispatchFormV1(invocation.operation, invocation.execution);
  if (form.stages[stageIndex]?.repeat !== expectedRepeat ||
      form.stages[stageIndex]?.moduleId !== expectedModuleId ||
      form.stages[stageIndex]?.entryPoint !== expectedEntryPoint) {
    fail("invalid_schema", `${invocation.operation} specialized catalog stage drifted`);
  }
  return Object.freeze({
    operation: invocation.operation,
    execution: invocation.execution,
    stageIndex,
    uniformSlot: 0,
    uniformBytes,
    storageBindings: Object.freeze({ ...storageBindings }),
    workgroups: Object.freeze([...workgroups]) as readonly [number, number, number],
  });
}

function stage(
  invocation: WebGpuLoweringInvocationV1,
  expectedModuleId: string,
  uniformBytes: Uint8Array,
  storageBindings: Readonly<Record<number, string>>,
  workgroups: readonly [number, number, number],
  expectedRepeat: "once" | "per_output" = "once",
): WebGpuResidentDispatchV1 {
  const form = webGpuDispatchFormV1(invocation.operation, invocation.execution);
  if (form.stages.length !== 1) {
    fail("invalid_schema", `${invocation.operation} specialized catalog stage drifted`);
  }
  return indexedStage(
    invocation, 0, expectedModuleId, uniformBytes, storageBindings, workgroups, expectedRepeat,
    "main",
  );
}

/** Compile pointwise and first-tranche specialized forms into resident transactions. */
export function compileWebGpuResidentScheduleV1(
  sourcePlan: CompiledTrainingPlanV1,
  budget: WebGpuResidentScheduleBudgetV1,
): WebGpuResidentScheduleV1 {
  const plan = snapshotPlan(sourcePlan);
  const budgetRecord = recordSnapshot(budget, "WebGPU schedule budget");
  const maxPeakBytes = property(budgetRecord, "maxPeakBytes", "WebGPU schedule budget");
  const uniformStride = property(budgetRecord, "uniformStride", "WebGPU schedule budget") ?? 256;
  if (!Number.isSafeInteger(maxPeakBytes) || (maxPeakBytes as number) < 0) {
    fail("invalid_schema", "WebGPU schedule maxPeakBytes must be a nonnegative safe integer");
  }
  if (!Number.isSafeInteger(uniformStride) || (uniformStride as number) < 256 ||
      (uniformStride as number) % 256 !== 0) {
    fail("invalid_schema", "WebGPU schedule uniformStride must be a positive 256-byte multiple");
  }
  const buffers = admittedWebGpuBuffersV1(plan);
  const metrics = [
    plan.residentBytes,
    plan.batchStagingBytes,
    plan.preparePeakBytes,
    plan.forwardPeakBytes,
    plan.exportPackageBytes,
    plan.exportPeakBytes,
    plan.peakBytes,
  ];
  if (metrics.some((value) => !Number.isSafeInteger(value) || value < 0) ||
      plan.preparePeakBytes < plan.residentBytes ||
      plan.forwardPeakBytes < plan.residentBytes ||
      plan.peakBytes < plan.residentBytes ||
      plan.peakBytes < plan.preparePeakBytes ||
      plan.peakBytes < plan.forwardPeakBytes ||
      plan.peakBytes < plan.exportPeakBytes) {
    fail("invalid_schema", "compiled plan memory metrics are inconsistent");
  }
  let rootBytes = 0;
  let physicalRootBytes = 0;
  for (const buffer of buffers.values()) {
    if (buffer.ownerId !== buffer.id) continue;
    const physicalBytes = Math.max(4, Math.ceil(buffer.byteLength / 4) * 4);
    if (rootBytes > Number.MAX_SAFE_INTEGER - buffer.byteLength ||
        physicalRootBytes > Number.MAX_SAFE_INTEGER - physicalBytes) {
      fail("memory_limit", "compiled root buffers exceed safe integer range");
    }
    rootBytes += buffer.byteLength;
    physicalRootBytes += physicalBytes;
  }
  if (rootBytes > plan.residentBytes) {
    fail("invalid_schema", "compiled residentBytes omits root buffers");
  }
  const rootPaddingBytes = Math.max(0, physicalRootBytes - plan.residentBytes);
  const uniformSlots = webGpuUniformSlotCapacityV1(plan);
  const occupied = new Set(buffers.keys());
  const pendingResources: PendingResource[] = [];
  const templates = new Map<string, Template>();
  let auxiliaryBytes = 0;
  let serial = 0;

  const auxiliary = (
    stem: string,
    byteLength: number,
    initialValues: readonly number[] | null,
  ) => {
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0 || byteLength % 4 !== 0 ||
        (initialValues !== null && initialValues.length * 4 !== byteLength)) {
      fail("invalid_schema", `${stem} auxiliary resource is invalid`);
    }
    let id: string;
    do {
      id = `$tritium.webgpu.${serial}.${stem}`;
      serial += 1;
    } while (occupied.has(id));
    occupied.add(id);
    if (auxiliaryBytes > Number.MAX_SAFE_INTEGER - byteLength) {
      fail("memory_limit", "specialized WebGPU auxiliary budget overflows");
    }
    auxiliaryBytes += byteLength;
    pendingResources.push(Object.freeze({
      id,
      byteLength,
      initialValues,
    }));
    return id;
  };

  const compileSpecialized = (
    invocation: WebGpuLoweringInvocationV1,
  ): Template => {
    const input = invocation.inputs;
    const output = invocation.outputs;
    const attributes = invocation.attributes;
    switch (`${invocation.operation}|${invocation.execution}`) {
      case "graph.salt_ste|forward": {
        const rows = positiveU32(attributes.rows, "rows");
        const cols = positiveU32(attributes.cols, "cols");
        const planes = positiveU32(attributes.planes, "planes");
        if (planes > 64) fail("invalid_schema", "SALT planes exceed the WebGPU kernel limit");
        const weight = requiredWebGpuRoleV1(input, "weight");
        const result = requiredWebGpuRoleV1(output, "result");
        expect(buffers, weight, "f32", [rows, cols], "weight");
        expect(buffers, result, "f32", [rows, cols], "result");
        const residual = auxiliary("salt-residual", product("SALT scratch", cols, 4), null);
        const params = uniform(16, (view) => {
          view.setUint32(0, rows, true);
          view.setUint32(4, cols, true);
          view.setUint32(8, planes, true);
        });
        return Object.freeze({
          commands: Object.freeze([stage(
            invocation, "salt", params, { 1: weight, 2: residual, 3: result }, [1, 1, 1],
          )]),
          copies: Object.freeze([]),
        });
      }
      case "graph.salt_ste|vjp": {
        const rows = positiveU32(attributes.rows, "rows");
        const cols = positiveU32(attributes.cols, "cols");
        const planes = positiveU32(attributes.planes, "planes");
        if (planes > 64) fail("invalid_schema", "SALT planes exceed the WebGPU kernel limit");
        const weight = requiredWebGpuRoleV1(input, "weight");
        const gradient = requiredWebGpuRoleV1(input, "grad_output");
        const result = requiredWebGpuRoleV1(output, "grad_weight");
        expect(buffers, weight, "f32", [rows, cols], "weight");
        expect(buffers, gradient, "f32", [rows, cols], "grad_output");
        expect(buffers, result, "f32", [rows, cols], "grad_weight");
        const len = product("SALT elements", rows, cols);
        const params = uniform(32, (view) => {
          view.setUint32(0, len, true);
          view.setUint32(4, 0, true);
        });
        return Object.freeze({
          commands: Object.freeze([stage(
            invocation,
            "pointwise",
            params,
            { 1: gradient, 2: gradient, 3: gradient, 4: result },
            [Math.ceil(len / 64), 1, 1],
          )]),
          copies: Object.freeze([]),
        });
      }
      case "graph.fsq|forward":
      case "graph.fsq|vjp": {
        const channels = positiveU32(attributes.channels, "channels");
        const len = positiveU32(attributes.len, "len");
        const total = product("FSQ elements", channels, len);
        const levels = u32List(attributes.levels, "levels");
        if (levels.length !== channels || levels.some((level) => level < 2)) {
          fail("invalid_schema", "FSQ levels differ from channel geometry");
        }
        const bound = attributes.bound === "clamp" ? 0 : attributes.bound === "tanh" ? 1 : -1;
        const estimator = attributes.ste === "hard" ? 0
          : attributes.ste === "soft_round" ? 1
            : attributes.ste === "stochastic" ? 2 : -1;
        if (bound < 0 || estimator < 0) fail("invalid_schema", "FSQ mode is unknown");
        const alpha = webGpuF32V1(attributes.alpha, "alpha");
        if (alpha < 0 || alpha > 1) fail("invalid_schema", "FSQ alpha must be in [0,1]");
        const seed = safeU64(attributes.seed, "seed");
        const x = requiredWebGpuRoleV1(input, "x");
        const upstream = invocation.execution === "forward"
          ? x
          : requiredWebGpuRoleV1(input, "grad_output");
        const result = requiredWebGpuRoleV1(
          output,
          invocation.execution === "forward" ? "result" : "grad_x",
        );
        expect(buffers, x, "f32", [channels, len], "x");
        expect(buffers, upstream, "f32", [channels, len], "FSQ upstream");
        expect(buffers, result, "f32", [channels, len], "FSQ result");
        const levelsId = auxiliary("fsq-levels", levels.length * 4, levels);
        const params = uniform(32, (view) => {
          view.setUint32(0, total, true);
          view.setUint32(4, len, true);
          view.setUint32(8, bound, true);
          view.setUint32(12, estimator, true);
          view.setUint32(16, Number(invocation.execution === "vjp"), true);
          view.setFloat32(20, alpha, true);
          view.setUint32(24, seed % 0x1_0000_0000, true);
          view.setUint32(28, Math.floor(seed / 0x1_0000_0000), true);
        });
        return Object.freeze({
          commands: Object.freeze([stage(
            invocation,
            "fsq",
            params,
            { 1: x, 2: levelsId, 3: upstream, 4: result },
            [Math.ceil(total / 64), 1, 1],
          )]),
          copies: Object.freeze([]),
        });
      }
      case "graph.embedding_gather|forward":
      case "graph.embedding_gather|vjp": {
        const vocab = positiveU32(attributes.vocab, "vocab");
        const width = positiveU32(attributes.n_embd, "n_embd");
        const weight = requiredWebGpuRoleV1(input, "weight");
        const tokens = requiredWebGpuRoleV1(input, "tokens");
        const tokenBuffer = buffers.get(tokens);
        if (tokenBuffer === undefined || tokenBuffer.dtype !== "u32" ||
            tokenBuffer.shape.length !== 1) {
          fail("invalid_schema", "embedding tokens differ from WebGPU geometry");
        }
        const sequence = webGpuU32V1(tokenBuffer.shape[0], "sequence");
        const result = requiredWebGpuRoleV1(
          output,
          invocation.execution === "forward" ? "result" : "grad_weight",
        );
        const gradient = invocation.execution === "forward"
          ? weight
          : requiredWebGpuRoleV1(input, "grad_output");
        expect(buffers, weight, "f32", [vocab, width], "embedding weight");
        expect(buffers, gradient, "f32", invocation.execution === "forward"
          ? [vocab, width] : [sequence, width], "embedding gradient");
        expect(buffers, result, "f32", invocation.execution === "forward"
          ? [sequence, width] : [vocab, width], "embedding result");
        const params = uniform(16, (view) => {
          view.setUint32(0, vocab, true);
          view.setUint32(4, width, true);
          view.setUint32(8, sequence, true);
          view.setUint32(12, Number(invocation.execution === "vjp"), true);
        });
        const count = invocation.execution === "forward"
          ? product("embedding output", sequence, width)
          : product("embedding gradient", vocab, width);
        return Object.freeze({
          commands: Object.freeze([stage(
            invocation,
            "embedding",
            params,
            { 1: weight, 2: tokens, 3: gradient, 4: result },
            [Math.max(1, Math.ceil(count / 64)), 1, 1],
          )]),
          copies: Object.freeze([]),
        });
      }
      case "graph.rope|forward":
      case "graph.rope|vjp": {
        const positions = u32List(attributes.positions, "positions");
        if (positions.length === 0) fail("invalid_schema", "RoPE positions are empty");
        const nToken = webGpuU32V1(positions.length, "n_token");
        const nHead = positiveU32(attributes.n_head, "n_head");
        const headDim = positiveU32(attributes.head_dim, "head_dim");
        const theta = webGpuF32V1(attributes.theta, "theta");
        if (headDim === 0 || headDim % 2 !== 0 || theta <= 0) {
          fail("invalid_schema", "RoPE head_dim/theta is invalid");
        }
        const source = requiredWebGpuRoleV1(
          input,
          invocation.execution === "forward" ? "x" : "grad_output",
        );
        const result = requiredWebGpuRoleV1(
          output,
          invocation.execution === "forward" ? "result" : "grad_x",
        );
        const shape = [nToken, nHead, headDim];
        expect(buffers, source, "f32", shape, "RoPE input");
        expect(buffers, result, "f32", shape, "RoPE output");
        const positionsId = auxiliary(
          "rope-positions", positions.length * 4, positions,
        );
        const params = uniform(32, (view) => {
          view.setUint32(0, nToken, true);
          view.setUint32(4, nHead, true);
          view.setUint32(8, headDim, true);
          view.setUint32(12, Number(invocation.execution === "vjp"), true);
          view.setFloat32(16, theta, true);
        });
        const pairs = product("RoPE pairs", nToken, nHead, headDim / 2);
        return Object.freeze({
          commands: Object.freeze([stage(
            invocation,
            "rope",
            params,
            { 1: source, 2: positionsId, 3: result },
            [Math.ceil(pairs / 64), 1, 1],
          )]),
          copies: Object.freeze([]),
        });
      }
      case "graph.concat_cols|forward":
      case "graph.concat_cols|vjp": {
        const rows = positiveU32(attributes.rows, "rows");
        const lens = u32List(attributes.lens, "lens");
        const expectedParts = invocation.execution === "forward"
          ? Object.keys(input).length
          : Object.keys(output).length;
        if (lens.length === 0 || lens.length !== expectedParts ||
            lens.some((width) => width === 0)) {
          fail("invalid_schema", "concat lens must match nonempty positive parts");
        }
        const total = lens.reduce((sum, width) => product("concat columns", 1, sum + width), 0);
        const gradient = invocation.execution === "vjp"
          ? requiredWebGpuRoleV1(input, "grad_output")
          : "";
        if (invocation.execution === "vjp") {
          expect(buffers, gradient, "f32", [rows, total], "concat grad_output");
          let start = 0;
          const writtenOwners = new Set<string>();
          const commands = lens.map((width, index) => {
            const result = requiredWebGpuRoleV1(output, `grad_part.${index}`);
            expect(buffers, result, "f32", [rows, width], `concat grad_part.${index}`);
            const ownerId = buffers.get(result)!.ownerId;
            if (writtenOwners.has(ownerId)) {
              fail("invalid_schema", "concat VJP outputs must have distinct physical owners");
            }
            writtenOwners.add(ownerId);
            const len = product("concat gradient", rows, width);
            const params = uniform(32, (view) => {
              view.setUint32(0, len, true);
              view.setUint32(4, 24, true);
              view.setUint32(12, total, true);
              view.setUint32(16, start, true);
              view.setUint32(20, width, true);
            });
            start += width;
            return stage(
              invocation,
              "pointwise",
              params,
              { 1: gradient, 2: gradient, 3: gradient, 4: result },
              [Math.ceil(len / 64), 1, 1],
              "per_output",
            );
          });
          return Object.freeze({
            commands: Object.freeze(commands),
            copies: Object.freeze([]),
          });
        }
        const values = auxiliary(
          "concat-values", product("concat values", rows, total, 4), null,
        );
        const lengths = auxiliary("concat-lengths", lens.length * 4, lens);
        let elementOffset = 0;
        const offsets = lens.map((width) => {
          const offset = elementOffset;
          elementOffset += product("concat part", rows, width);
          return offset;
        });
        const offsetsId = auxiliary("concat-offsets", offsets.length * 4, offsets);
        const copies: WebGpuResidentCopyV1[] = [];
        let destinationOffset = 0;
        lens.forEach((width, index) => {
          const part = requiredWebGpuRoleV1(input, `part.${index}`);
          expect(buffers, part, "f32", [rows, width], `concat part.${index}`);
          const byteLength = product("concat part bytes", rows, width, 4);
          copies.push(Object.freeze({
            source: part,
            sourceOffset: 0,
            destination: values,
            destinationOffset,
            byteLength,
          }));
          destinationOffset += byteLength;
        });
        const result = requiredWebGpuRoleV1(output, "result");
        expect(buffers, result, "f32", [rows, total], "concat result");
        const params = uniform(16, (view) => {
          view.setUint32(0, rows, true);
          view.setUint32(4, lens.length, true);
          view.setUint32(8, total, true);
        });
        return Object.freeze({
          commands: Object.freeze([stage(
            invocation,
            "concat",
            params,
            { 1: values, 2: lengths, 3: offsetsId, 4: result },
            [Math.ceil(product("concat output", rows, total) / 64), 1, 1],
          )]),
          copies: Object.freeze(copies),
        });
      }
      case "graph.conv1d|forward":
      case "graph.conv1d|vjp":
      case "graph.conv2d|forward":
      case "graph.conv2d|vjp": {
        const batch = positiveU32(attributes.batch, "batch");
        const cIn = positiveU32(attributes.c_in, "c_in");
        const cOut = positiveU32(attributes.c_out, "c_out");
        const groups = positiveU32(attributes.groups, "groups");
        if (cIn % groups !== 0 || cOut % groups !== 0) {
          fail("invalid_schema", "convolution groups must divide channels");
        }
        const is1d = invocation.operation === "graph.conv1d";
        const inputH = is1d ? 1 : positiveU32(attributes.input_h, "input_h");
        const inputW = is1d
          ? positiveU32(attributes.l_in, "l_in")
          : positiveU32(attributes.input_w, "input_w");
        const kernelH = is1d ? 1 : positiveU32(attributes.kernel_h, "kernel_h");
        const kernelW = is1d
          ? positiveU32(attributes.k, "k")
          : positiveU32(attributes.kernel_w, "kernel_w");
        const strideH = is1d ? 1 : positiveU32(attributes.stride_h, "stride_h");
        const strideW = is1d
          ? positiveU32(attributes.stride, "stride")
          : positiveU32(attributes.stride_w, "stride_w");
        const dilationH = is1d ? 1 : positiveU32(attributes.dilation_h, "dilation_h");
        const dilationW = is1d
          ? positiveU32(attributes.dilation, "dilation")
          : positiveU32(attributes.dilation_w, "dilation_w");
        const padTop = is1d ? 0 : webGpuU32V1(attributes.pad_top, "pad_top");
        const padBottom = is1d ? 0 : webGpuU32V1(attributes.pad_bottom, "pad_bottom");
        const padLeft = webGpuU32V1(attributes.pad_left, "pad_left");
        const padRight = webGpuU32V1(attributes.pad_right, "pad_right");
        const outputH = convAxis(
          inputH, kernelH, strideH, dilationH, padTop, padBottom, "conv output_h",
        );
        const outputW = convAxis(
          inputW, kernelW, strideW, dilationW, padLeft, padRight, "conv output_w",
        );
        const maximumH = sumU32(
          "conv h indexing",
          product("conv h indexing", outputH - 1, strideH),
          product("conv h indexing", kernelH - 1, dilationH),
        );
        const maximumW = sumU32(
          "conv w indexing",
          product("conv w indexing", outputW - 1, strideW),
          product("conv w indexing", kernelW - 1, dilationW),
        );
        if (maximumH > 0x7fff_ffff || maximumW > 0x7fff_ffff ||
            padTop > 0x7fff_ffff || padLeft > 0x7fff_ffff) {
          fail("invalid_schema", "convolution indexing exceeds i32");
        }
        const inputShape = is1d
          ? [batch, cIn, inputW]
          : [batch, cIn, inputH, inputW];
        const weightShape = is1d
          ? [cOut, cIn / groups, kernelW]
          : [cOut, cIn / groups, kernelH, kernelW];
        const outputShape = is1d
          ? [batch, cOut, outputW]
          : [batch, cOut, outputH, outputW];
        const x = requiredWebGpuRoleV1(input, "x");
        const weight = requiredWebGpuRoleV1(input, "weight");
        const scale = requiredWebGpuRoleV1(input, "scale");
        expect(buffers, x, "f32", inputShape, "conv x");
        expect(buffers, weight, "f32", weightShape, "conv weight");
        expect(buffers, scale, "f32", [cOut], "conv scale");
        const inputElements = product("conv input", ...inputShape);
        const weightElements = product("conv weight", ...weightShape);
        const outputElements = product("conv output", ...outputShape);
        const patch = product("conv patch", cIn / groups, kernelH, kernelW);
        const tileRows = is1d ? outputW : Math.min(product("conv rows", outputH, outputW), 32);
        const columns = product("conv columns", tileRows, patch);
        const groupOutput = product("conv group output", tileRows, cOut / groups);
        const contractElements = invocation.execution === "forward"
          ? sumU32("conv scratch", outputElements, columns, groupOutput)
          : sumU32(
            "conv scratch", inputElements, weightElements, cOut, columns, groupOutput,
            columns, weightElements / groups, cOut / groups,
          );
        if (contractElements * 4 > 64 * 1024 * 1024) {
          fail("memory_limit", "convolution scratch exceeds 64 MiB");
        }
        const resultRole = invocation.execution === "forward" ? "result" : "grad_x";
        const result = requiredWebGpuRoleV1(output, resultRole);
        expect(buffers, result, "f32",
          invocation.execution === "forward" ? outputShape : inputShape, `conv ${resultRole}`);
        let gradOutput: string;
        let gradWeight: string;
        let gradScale: string;
        const copies: WebGpuResidentCopyV1[] = [];
        if (invocation.execution === "vjp") {
          gradOutput = requiredWebGpuRoleV1(input, "grad_output");
          gradWeight = requiredWebGpuRoleV1(output, "grad_weight");
          gradScale = requiredWebGpuRoleV1(output, "grad_scale");
          expect(buffers, gradOutput, "f32", outputShape, "conv grad_output");
          expect(buffers, gradWeight, "f32", weightShape, "conv grad_weight");
          expect(buffers, gradScale, "f32", [cOut], "conv grad_scale");
          requireDisjointWrites(
            buffers, [x, weight, scale, gradOutput], [result, gradWeight, gradScale],
            invocation.operation,
          );
          const zeroBytes = Math.max(inputElements, weightElements, cOut) * 4;
          const zero = auxiliary("conv-zero", zeroBytes, null);
          for (const [destination, byteLength] of [
            [result, inputElements * 4],
            [gradWeight, weightElements * 4],
            [gradScale, cOut * 4],
          ] as const) {
            copies.push(Object.freeze({
              source: zero,
              sourceOffset: 0,
              destination,
              destinationOffset: 0,
              byteLength,
            }));
          }
        } else {
          requireDisjointWrites(buffers, [x, weight, scale], [result], invocation.operation);
          gradOutput = x;
          gradWeight = auxiliary("conv-unused-grad-weight", 4, null);
          gradScale = auxiliary("conv-unused-grad-scale", 4, null);
        }
        const params = uniform(80, (view) => {
          [
            batch, cIn, cOut, inputH, inputW, kernelH, kernelW, strideH, strideW,
            dilationH, dilationW, padTop, padLeft, groups, outputH, outputW,
            Number(invocation.execution === "vjp"), padBottom, padRight, 0,
          ].forEach((value, index) => view.setUint32(index * 4, value, true));
        });
        return Object.freeze({
          commands: Object.freeze([stage(
            invocation,
            "conv",
            params,
            {
              1: x, 2: weight, 3: scale, 4: gradOutput,
              5: result, 6: gradWeight, 7: gradScale,
            },
            [1, 1, 1],
          )]),
          copies: Object.freeze(copies),
        });
      }
      case "graph.attention|forward":
      case "graph.attention|vjp": {
        const seq = positiveU32(attributes.seq, "seq");
        const nHead = positiveU32(attributes.n_head, "n_head");
        const nKvHead = positiveU32(attributes.n_kv_head, "n_kv_head");
        const headDim = positiveU32(attributes.head_dim, "head_dim");
        if (nHead % nKvHead !== 0 || typeof attributes.causal !== "boolean") {
          fail("invalid_schema", "attention head or causal attributes are invalid");
        }
        const queryShape = [seq, nHead, headDim];
        const kvShape = [seq, nKvHead, headDim];
        const queryElements = product("attention query", ...queryShape);
        const kvElements = product("attention kv", ...kvShape);
        const probabilityElements = product("attention probabilities", seq, seq);
        const contractElements = sumU32(
          "attention scratch", queryElements, kvElements, kvElements,
          probabilityElements, probabilityElements,
        );
        if (contractElements * 4 > 64 * 1024 * 1024) {
          fail("memory_limit", "attention scratch exceeds 64 MiB");
        }
        const q = requiredWebGpuRoleV1(input, "q");
        const k = requiredWebGpuRoleV1(input, "k");
        const v = requiredWebGpuRoleV1(input, "v");
        expect(buffers, q, "f32", queryShape, "attention q");
        expect(buffers, k, "f32", kvShape, "attention k");
        expect(buffers, v, "f32", kvShape, "attention v");
        const probabilities = auxiliary(
          "attention-probabilities", probabilityElements * 4, null,
        );
        let gradOutput: string;
        let output0: string;
        let output1: string;
        let output2: string;
        let gradProbabilities: string;
        const copies: WebGpuResidentCopyV1[] = [];
        if (invocation.execution === "vjp") {
          gradOutput = requiredWebGpuRoleV1(input, "grad_output");
          output0 = requiredWebGpuRoleV1(output, "grad_q");
          output1 = requiredWebGpuRoleV1(output, "grad_k");
          output2 = requiredWebGpuRoleV1(output, "grad_v");
          expect(buffers, gradOutput, "f32", queryShape, "attention grad_output");
          expect(buffers, output0, "f32", queryShape, "attention grad_q");
          expect(buffers, output1, "f32", kvShape, "attention grad_k");
          expect(buffers, output2, "f32", kvShape, "attention grad_v");
          requireDisjointWrites(
            buffers, [q, k, v, gradOutput], [output0, output1, output2],
            invocation.operation,
          );
          const zero = auxiliary(
            "attention-zero", Math.max(queryElements, kvElements) * 4, null,
          );
          for (const [destination, byteLength] of [
            [output0, queryElements * 4],
            [output1, kvElements * 4],
            [output2, kvElements * 4],
          ] as const) {
            copies.push(Object.freeze({
              source: zero,
              sourceOffset: 0,
              destination,
              destinationOffset: 0,
              byteLength,
            }));
          }
          gradProbabilities = auxiliary(
            "attention-grad-probabilities", probabilityElements * 4, null,
          );
        } else {
          gradOutput = q;
          output0 = requiredWebGpuRoleV1(output, "result");
          expect(buffers, output0, "f32", queryShape, "attention result");
          requireDisjointWrites(buffers, [q, k, v], [output0], invocation.operation);
          output1 = auxiliary("attention-unused-output-1", 4, null);
          output2 = auxiliary("attention-unused-output-2", 4, null);
          gradProbabilities = auxiliary("attention-unused-grad-probabilities", 4, null);
        }
        const params = uniform(32, (view) => {
          view.setUint32(0, seq, true);
          view.setUint32(4, nHead, true);
          view.setUint32(8, nKvHead, true);
          view.setUint32(12, headDim, true);
          view.setUint32(16, Number(attributes.causal), true);
          view.setUint32(20, Number(invocation.execution === "vjp"), true);
        });
        return Object.freeze({
          commands: Object.freeze([stage(
            invocation,
            "attention",
            params,
            {
              1: q, 2: k, 3: v, 4: gradOutput, 5: output0,
              6: output1, 7: output2, 8: probabilities, 9: gradProbabilities,
            },
            [1, 1, 1],
          )]),
          copies: Object.freeze(copies),
        });
      }
      case "loss.softmax_cross_entropy|forward":
      case "loss.softmax_cross_entropy|vjp": {
        const rows = positiveU32(attributes.rows, "rows");
        const cols = positiveU32(attributes.cols, "cols");
        product("softmax cross entropy", rows, cols);
        const logits = requiredWebGpuRoleV1(input, "logits");
        const target = requiredWebGpuRoleV1(input, "target");
        const gradOutput = invocation.execution === "forward"
          ? logits
          : requiredWebGpuRoleV1(input, "grad_output");
        const result = requiredWebGpuRoleV1(
          output,
          invocation.execution === "forward" ? "result" : "grad_logits",
        );
        expect(buffers, logits, "f32", [rows, cols], "logits");
        expect(buffers, target, "f32", [rows, cols], "target");
        expect(buffers, gradOutput, "f32", invocation.execution === "forward"
          ? [rows, cols] : [], "cross entropy cotangent");
        expect(buffers, result, "f32", invocation.execution === "forward"
          ? [] : [rows, cols], "cross entropy result");
        const params = uniform(32, (view) => {
          view.setUint32(0, rows, true);
          view.setUint32(4, cols, true);
          view.setUint32(8, Number(invocation.execution === "vjp"), true);
        });
        return Object.freeze({
          commands: Object.freeze([stage(
            invocation,
            "softmax_xent",
            params,
            { 1: logits, 2: target, 3: gradOutput, 4: result },
            [1, 1, 1],
          )]),
          copies: Object.freeze([]),
        });
      }
      case "loss.topk_knowledge_distillation|forward":
      case "loss.topk_knowledge_distillation|vjp": {
        const rows = positiveU32(attributes.rows, "rows");
        const cols = positiveU32(attributes.cols, "cols");
        const k = positiveU32(attributes.k, "k");
        product("top-k knowledge distillation logits", rows, cols);
        product("top-k knowledge distillation sparse target", rows, k);
        if (k > cols) fail("invalid_schema", "top-k knowledge distillation k exceeds cols");
        const logits = requiredWebGpuRoleV1(input, "logits");
        const indices = requiredWebGpuRoleV1(input, "indices");
        const probabilities = requiredWebGpuRoleV1(input, "probabilities");
        const gradOutput = invocation.execution === "forward"
          ? logits
          : requiredWebGpuRoleV1(input, "grad_output");
        const result = requiredWebGpuRoleV1(
          output,
          invocation.execution === "forward" ? "result" : "grad_logits",
        );
        expect(buffers, logits, "f32", [rows, cols], "top-k logits");
        expect(buffers, indices, "u32", [rows, k], "top-k indices");
        expect(buffers, probabilities, "f32", [rows, k], "top-k probabilities");
        expect(
          buffers,
          gradOutput,
          "f32",
          invocation.execution === "forward" ? [rows, cols] : [],
          "top-k cotangent",
        );
        expect(
          buffers,
          result,
          "f32",
          invocation.execution === "forward" ? [] : [rows, cols],
          "top-k result",
        );
        requireDisjointWrites(
          buffers,
          [logits, indices, probabilities, gradOutput],
          [result],
          invocation.operation,
        );
        const params = uniform(32, (view) => {
          view.setUint32(0, rows, true);
          view.setUint32(4, cols, true);
          view.setUint32(8, k, true);
          view.setUint32(12, Number(invocation.execution === "vjp"), true);
        });
        return Object.freeze({
          commands: Object.freeze([stage(
            invocation,
            "topk_kd",
            params,
            {
              1: logits,
              2: indices,
              3: probabilities,
              4: gradOutput,
              5: result,
            },
            [1, 1, 1],
          )]),
          copies: Object.freeze([]),
        });
      }
      case "optimizer.sgd|step": {
        if (safeU64(attributes.step, "step") !== 0) {
          fail("invalid_schema", "SGD compiled recipe step must start at zero");
        }
        const learningRate = webGpuF32V1(attributes.lr, "lr");
        if (learningRate < 0) fail("invalid_schema", "SGD lr must be nonnegative");
        const parameter = requiredWebGpuRoleV1(input, "parameter");
        const gradient = requiredWebGpuRoleV1(input, "gradient");
        const result = requiredWebGpuRoleV1(output, "parameter");
        const parameterBuffer = buffers.get(parameter);
        if (parameterBuffer === undefined || parameterBuffer.dtype !== "f32") {
          fail("invalid_schema", "SGD parameter must be f32");
        }
        expect(buffers, gradient, "f32", parameterBuffer.shape, "SGD gradient");
        expect(buffers, result, "f32", parameterBuffer.shape, "SGD result");
        if (buffers.get(result)!.ownerId !== parameterBuffer.ownerId) {
          fail("invalid_schema", "SGD output must commit to parameter owner");
        }
        const len = product("SGD parameter", ...parameterBuffer.shape);
        const candidate = auxiliary("sgd-parameter-candidate", len * 4, null);
        const commandFactory = (optimizerStep: number) => {
          if (!Number.isSafeInteger(optimizerStep) || optimizerStep <= 0) {
            fail("invalid_schema", "optimizerStep must be a positive safe integer");
          }
          const params = uniform(32, (view) => {
            view.setUint32(0, len, true);
            view.setUint32(4, 21, true);
            view.setFloat32(8, learningRate, true);
          });
          return Object.freeze([stage(
            invocation,
            "pointwise",
            params,
            { 1: parameter, 2: gradient, 3: gradient, 4: candidate },
            [Math.ceil(len / 64), 1, 1],
          )]);
        };
        return Object.freeze({
          commands: Object.freeze([]),
          commandFactory,
          copies: Object.freeze([]),
          commitCopies: Object.freeze([Object.freeze({
            source: candidate,
            sourceOffset: 0,
            destination: result,
            destinationOffset: 0,
            byteLength: len * 4,
          })]),
        });
      }
      case "optimizer.adamw|step":
      case "optimizer.cautious_adamw|step": {
        const cautious = invocation.operation === "optimizer.cautious_adamw";
        const expectedStages = cautious ? 7 : 4;
        if (webGpuDispatchFormV1(invocation.operation, invocation.execution).stages.length !==
            expectedStages) {
          fail("invalid_schema", `${invocation.operation} specialized catalog stage drifted`);
        }
        if (safeU64(attributes.step, "step") !== 0) {
          fail("invalid_schema", `${invocation.operation} compiled recipe step must start at zero`);
        }
        const learningRate = webGpuF32V1(attributes.lr, "lr");
        const beta1 = webGpuF32V1(attributes.beta1, "beta1");
        const beta2 = webGpuF32V1(attributes.beta2, "beta2");
        const epsilon = webGpuF32V1(attributes.eps, "eps");
        const weightDecay = webGpuF32V1(attributes.weight_decay, "weight_decay");
        if (learningRate < 0 || beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 ||
            epsilon <= 0 || weightDecay < 0) {
          fail("invalid_schema", `${invocation.operation} scalar attributes are invalid`);
        }
        const parameter = requiredWebGpuRoleV1(input, "parameter");
        const gradient = requiredWebGpuRoleV1(input, "gradient");
        const moment1 = requiredWebGpuRoleV1(input, "moment1");
        const moment2 = requiredWebGpuRoleV1(input, "moment2");
        const resultParameter = requiredWebGpuRoleV1(output, "parameter");
        const resultMoment1 = requiredWebGpuRoleV1(output, "moment1");
        const resultMoment2 = requiredWebGpuRoleV1(output, "moment2");
        const parameterBuffer = buffers.get(parameter);
        if (parameterBuffer === undefined || parameterBuffer.dtype !== "f32") {
          fail("invalid_schema", `${invocation.operation} parameter must be f32`);
        }
        for (const [id, role] of [
          [gradient, "gradient"], [moment1, "moment1"], [moment2, "moment2"],
          [resultParameter, "result parameter"], [resultMoment1, "result moment1"],
          [resultMoment2, "result moment2"],
        ] as const) {
          expect(buffers, id, "f32", parameterBuffer.shape, `${invocation.operation} ${role}`);
        }
        for (const [source, destination, role] of [
          [parameter, resultParameter, "parameter"],
          [moment1, resultMoment1, "moment1"],
          [moment2, resultMoment2, "moment2"],
        ] as const) {
          if (buffers.get(source)!.ownerId !== buffers.get(destination)!.ownerId) {
            fail("invalid_schema", `${invocation.operation} output must commit to ${role} owner`);
          }
        }
        const len = product(`${invocation.operation} parameter`, ...parameterBuffer.shape);
        const bytes = product(`${invocation.operation} bytes`, len, 4);
        const candidateParameter = auxiliary("adamw-parameter-candidate", bytes, null);
        const candidateMoment1 = auxiliary("adamw-moment1-candidate", bytes, null);
        const candidateMoment2 = auxiliary("adamw-moment2-candidate", bytes, null);
        const scratch1 = auxiliary("adamw-scratch-1", bytes, null);
        const scratch2 = auxiliary("adamw-scratch-2", bytes, null);
        let aligned: string | undefined;
        let zero: string | undefined;
        if (cautious) {
          aligned = auxiliary("cautious-adamw-aligned", 4, null);
          zero = auxiliary("cautious-adamw-zero", 4, [0]);
        }
        const workgroups = [Math.ceil(len / 64), 1, 1] as const;
        const commandFactory = (optimizerStep: number) => {
          if (!Number.isSafeInteger(optimizerStep) || optimizerStep <= 0) {
            fail("invalid_schema", "optimizerStep must be a positive safe integer");
          }
          const exponent = Math.min(optimizerStep, 0x7fff_ffff);
          const correction1 = Math.fround(1 - powiF32(beta1, exponent));
          const correction2 = Math.fround(1 - powiF32(beta2, exponent));
          const shrink = Math.fround(1 - Math.fround(learningRate * weightDecay));
          const params = uniform(48, (view) => {
            view.setUint32(0, len, true);
            view.setFloat32(16, learningRate, true);
            view.setFloat32(20, beta1, true);
            view.setFloat32(24, beta2, true);
            view.setFloat32(28, epsilon, true);
            view.setFloat32(32, correction1, true);
            view.setFloat32(36, correction2, true);
            view.setFloat32(40, shrink, true);
          });
          const commands = [
            indexedStage(invocation, 0, "adamw", params, {
              1: parameter, 2: gradient, 3: moment1, 4: moment2,
              5: candidateParameter, 6: candidateMoment1, 7: candidateMoment2,
              8: scratch1, 9: scratch2,
            }, workgroups),
            indexedStage(invocation, 1, "adamw_terms", params, {
              2: gradient, 6: candidateMoment1, 8: scratch1, 9: scratch2,
            }, workgroups),
            indexedStage(invocation, 2, "adamw_variance", params, {
              7: candidateMoment2, 9: scratch2,
            }, workgroups),
          ];
          if (cautious) {
            commands.push(
              indexedStage(invocation, 3, "cautious_adamw_mask", params, {
                2: gradient, 6: candidateMoment1, 7: candidateMoment2,
                8: scratch1, 10: aligned!,
              }, workgroups),
              indexedStage(invocation, 4, "cautious_adamw_lr", params, {
                8: scratch1,
              }, workgroups),
              indexedStage(invocation, 5, "cautious_adamw_rescale", params, {
                8: scratch1, 10: aligned!,
              }, workgroups),
              indexedStage(invocation, 6, "cautious_adamw_finish", params, {
                5: candidateParameter, 8: scratch1,
              }, workgroups),
            );
          } else {
            commands.push(indexedStage(invocation, 3, "adamw_finish", params, {
              5: candidateParameter, 6: candidateMoment1, 7: candidateMoment2,
            }, workgroups));
          }
          return Object.freeze(commands);
        };
        return Object.freeze({
          commands: Object.freeze([]),
          commandFactory,
          copies: cautious ? Object.freeze([Object.freeze({
            source: zero!, sourceOffset: 0, destination: aligned!, destinationOffset: 0,
            byteLength: 4,
          })]) : Object.freeze([]),
          commitCopies: Object.freeze([
            Object.freeze({
              source: candidateParameter, sourceOffset: 0, destination: resultParameter,
              destinationOffset: 0, byteLength: bytes,
            }),
            Object.freeze({
              source: candidateMoment1, sourceOffset: 0, destination: resultMoment1,
              destinationOffset: 0, byteLength: bytes,
            }),
            Object.freeze({
              source: candidateMoment2, sourceOffset: 0, destination: resultMoment2,
              destinationOffset: 0, byteLength: bytes,
            }),
          ]),
        });
      }
      case "optimizer.int8_adamw|step": {
        if (webGpuDispatchFormV1(invocation.operation, invocation.execution).stages.length !== 12) {
          fail("invalid_schema", "optimizer.int8_adamw specialized catalog stage drifted");
        }
        if (safeU64(attributes.step, "step") !== 0) {
          fail("invalid_schema", "int8 AdamW compiled recipe step must start at zero");
        }
        const learningRate = webGpuF32V1(attributes.lr, "lr");
        const beta1 = webGpuF32V1(attributes.beta1, "beta1");
        const beta2 = webGpuF32V1(attributes.beta2, "beta2");
        const epsilon = webGpuF32V1(attributes.eps, "eps");
        const weightDecay = webGpuF32V1(attributes.weight_decay, "weight_decay");
        if (learningRate < 0 || beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 ||
            epsilon <= 0 || weightDecay < 0) {
          fail("invalid_schema", "int8 AdamW scalar attributes are invalid");
        }
        const parameter = requiredWebGpuRoleV1(input, "parameter");
        const gradient = requiredWebGpuRoleV1(input, "gradient");
        const moment1 = requiredWebGpuRoleV1(input, "moment1_q8");
        const moment2 = requiredWebGpuRoleV1(input, "moment2_q8");
        const scale1 = requiredWebGpuRoleV1(input, "moment1_scale");
        const scale2 = requiredWebGpuRoleV1(input, "moment2_scale");
        const resultParameter = requiredWebGpuRoleV1(output, "parameter");
        const resultMoment1 = requiredWebGpuRoleV1(output, "moment1_q8");
        const resultMoment2 = requiredWebGpuRoleV1(output, "moment2_q8");
        const resultScale1 = requiredWebGpuRoleV1(output, "moment1_scale");
        const resultScale2 = requiredWebGpuRoleV1(output, "moment2_scale");
        const parameterBuffer = buffers.get(parameter);
        if (parameterBuffer === undefined || parameterBuffer.dtype !== "f32") {
          fail("invalid_schema", "int8 AdamW parameter must be f32");
        }
        const len = product("int8 AdamW parameter", ...parameterBuffer.shape);
        const blocks = Math.ceil(len / 256);
        expect(buffers, gradient, "f32", parameterBuffer.shape, "int8 AdamW gradient");
        expect(buffers, resultParameter, "f32", parameterBuffer.shape, "int8 AdamW result");
        for (const [id, role] of [
          [moment1, "moment1_q8"], [moment2, "moment2_q8"],
          [resultMoment1, "result moment1_q8"], [resultMoment2, "result moment2_q8"],
        ] as const) {
          expect(buffers, id, "bytes", [len], `int8 AdamW ${role}`);
        }
        for (const [id, role] of [
          [scale1, "moment1_scale"], [scale2, "moment2_scale"],
          [resultScale1, "result moment1_scale"], [resultScale2, "result moment2_scale"],
        ] as const) {
          expect(buffers, id, "f32", [blocks], `int8 AdamW ${role}`);
        }
        for (const [source, destination, role] of [
          [parameter, resultParameter, "parameter"],
          [moment1, resultMoment1, "moment1_q8"],
          [moment2, resultMoment2, "moment2_q8"],
          [scale1, resultScale1, "moment1_scale"],
          [scale2, resultScale2, "moment2_scale"],
        ] as const) {
          if (buffers.get(source)!.ownerId !== buffers.get(destination)!.ownerId) {
            fail("invalid_schema", `int8 AdamW output must commit to ${role} owner`);
          }
        }
        const tensorBytes = product("int8 AdamW tensor bytes", len, 4);
        const scaleBytes = product("int8 AdamW scale bytes", blocks, 4);
        const packedWords = Math.ceil(len / 4);
        const packedBytes = product("int8 AdamW packed bytes", packedWords, 4);
        const candidateParameter = auxiliary("int8-adamw-parameter-candidate", tensorBytes, null);
        const expandedMoment1 = auxiliary("int8-adamw-expanded-moment1", tensorBytes, null);
        const expandedMoment2 = auxiliary("int8-adamw-expanded-moment2", tensorBytes, null);
        const candidateScale1 = auxiliary("int8-adamw-moment1-scale-candidate", scaleBytes, null);
        const candidateScale2 = auxiliary("int8-adamw-moment2-scale-candidate", scaleBytes, null);
        const scratch1 = auxiliary("int8-adamw-scratch-1", tensorBytes, null);
        const scratch2 = auxiliary("int8-adamw-scratch-2", tensorBytes, null);
        const packedMoment1 = auxiliary("int8-adamw-packed-moment1-candidate", packedBytes, null);
        const packedMoment2 = auxiliary("int8-adamw-packed-moment2-candidate", packedBytes, null);
        const linearWorkgroups = [Math.ceil(len / 64), 1, 1] as const;
        const packedWorkgroups = [Math.ceil(packedWords / 64), 1, 1] as const;
        const codecParams = uniform(16, (view) => view.setUint32(0, len, true));
        const commandFactory = (optimizerStep: number) => {
          if (!Number.isSafeInteger(optimizerStep) || optimizerStep <= 0) {
            fail("invalid_schema", "optimizerStep must be a positive safe integer");
          }
          const exponent = Math.min(optimizerStep, 0x7fff_ffff);
          const params = uniform(48, (view) => {
            view.setUint32(0, len, true);
            view.setFloat32(16, learningRate, true);
            view.setFloat32(20, beta1, true);
            view.setFloat32(24, beta2, true);
            view.setFloat32(28, epsilon, true);
            view.setFloat32(32, Math.fround(1 - powiF32(beta1, exponent)), true);
            view.setFloat32(36, Math.fround(1 - powiF32(beta2, exponent)), true);
            view.setFloat32(
              40, Math.fround(1 - Math.fround(learningRate * weightDecay)), true,
            );
          });
          const commands = [
            indexedStage(invocation, 0, "byte_codec", codecParams, {
              1: moment1, 2: expandedMoment1,
            }, linearWorkgroups, "once", "unpack"),
            indexedStage(invocation, 1, "byte_codec", codecParams, {
              1: moment2, 2: expandedMoment2,
            }, linearWorkgroups, "once", "unpack"),
          ];
          const coreStages: readonly (readonly [
            string, Readonly<Record<number, string>>,
          ])[] = [
            ["dequantize", {
              3: expandedMoment1, 4: expandedMoment2, 5: candidateScale1, 6: candidateScale2,
            }],
            ["square_variance", { 4: expandedMoment2 }],
            ["products", {
              1: candidateParameter, 2: gradient, 3: expandedMoment1, 4: expandedMoment2,
              7: scratch1, 8: scratch2,
            }],
            ["finish_products", {
              2: gradient, 3: expandedMoment1, 7: scratch1, 8: scratch2,
            }],
            ["finish_variance", { 4: expandedMoment2, 8: scratch2 }],
            ["update_parameter", {
              1: candidateParameter, 3: expandedMoment1, 4: expandedMoment2,
            }],
            ["reduce_scales", {
              3: expandedMoment1, 4: expandedMoment2, 5: candidateScale1, 6: candidateScale2,
            }],
            ["quantize", {
              3: expandedMoment1, 4: expandedMoment2, 5: candidateScale1, 6: candidateScale2,
            }],
          ];
          for (const [offset, [entryPoint, stageBindings]] of coreStages.entries()) {
            commands.push(indexedStage(
              invocation,
              offset + 2,
              "int8_adamw",
              params,
              stageBindings,
              entryPoint === "reduce_scales"
                ? [blocks, 1, 1]
                : linearWorkgroups,
              "once",
              entryPoint,
            ));
          }
          commands.push(
            indexedStage(invocation, 10, "byte_codec", codecParams, {
              1: expandedMoment1, 2: packedMoment1,
            }, packedWorkgroups, "once", "pack"),
            indexedStage(invocation, 11, "byte_codec", codecParams, {
              1: expandedMoment2, 2: packedMoment2,
            }, packedWorkgroups, "once", "pack"),
          );
          return Object.freeze(commands);
        };
        return Object.freeze({
          commands: Object.freeze([]),
          commandFactory,
          copies: Object.freeze([
            Object.freeze({
              source: parameter, sourceOffset: 0, destination: candidateParameter,
              destinationOffset: 0, byteLength: tensorBytes,
            }),
            Object.freeze({
              source: scale1, sourceOffset: 0, destination: candidateScale1,
              destinationOffset: 0, byteLength: scaleBytes,
            }),
            Object.freeze({
              source: scale2, sourceOffset: 0, destination: candidateScale2,
              destinationOffset: 0, byteLength: scaleBytes,
            }),
          ]),
          commitCopies: Object.freeze([
            Object.freeze({
              source: candidateParameter, sourceOffset: 0, destination: resultParameter,
              destinationOffset: 0, byteLength: tensorBytes,
            }),
            Object.freeze({
              source: packedMoment1, sourceOffset: 0, destination: resultMoment1,
              destinationOffset: 0, byteLength: packedBytes,
            }),
            Object.freeze({
              source: packedMoment2, sourceOffset: 0, destination: resultMoment2,
              destinationOffset: 0, byteLength: packedBytes,
            }),
            Object.freeze({
              source: candidateScale1, sourceOffset: 0, destination: resultScale1,
              destinationOffset: 0, byteLength: scaleBytes,
            }),
            Object.freeze({
              source: candidateScale2, sourceOffset: 0, destination: resultScale2,
              destinationOffset: 0, byteLength: scaleBytes,
            }),
          ]),
        });
      }
      case "optimizer.muon|step": {
        if (webGpuDispatchFormV1(invocation.operation, invocation.execution).stages.length !== 1) {
          fail("invalid_schema", "optimizer.muon specialized catalog stage drifted");
        }
        if (safeU64(attributes.step, "step") !== 0) {
          fail("invalid_schema", "Muon compiled recipe step must start at zero");
        }
        const learningRate = webGpuF32V1(attributes.lr, "lr");
        const momentumDecay = webGpuF32V1(attributes.momentum, "momentum");
        const weightDecay = webGpuF32V1(attributes.weight_decay, "weight_decay");
        const rows = positiveU32(attributes.rows, "rows");
        const cols = positiveU32(attributes.cols, "cols");
        const steps = positiveU32(attributes.ns_steps, "ns_steps");
        if (learningRate < 0 || momentumDecay < 0 || momentumDecay >= 1 || weightDecay < 0 ||
            steps > 32) {
          fail("invalid_schema", "Muon scalar attributes are invalid");
        }
        const len = product("Muon parameter", rows, cols);
        const parameter = requiredWebGpuRoleV1(input, "parameter");
        const gradient = requiredWebGpuRoleV1(input, "gradient");
        const momentum = requiredWebGpuRoleV1(input, "momentum");
        const resultParameter = requiredWebGpuRoleV1(output, "parameter");
        const resultMomentum = requiredWebGpuRoleV1(output, "momentum");
        for (const [id, role] of [
          [parameter, "parameter"], [gradient, "gradient"], [momentum, "momentum"],
          [resultParameter, "result parameter"], [resultMomentum, "result momentum"],
        ] as const) {
          expect(buffers, id, "f32", [rows, cols], `Muon ${role}`);
        }
        if (buffers.get(parameter)!.ownerId !== buffers.get(resultParameter)!.ownerId ||
            buffers.get(momentum)!.ownerId !== buffers.get(resultMomentum)!.ownerId) {
          fail("invalid_schema", "Muon outputs must commit to their input owners");
        }
        const bytes = product("Muon parameter bytes", len, 4);
        const r = Math.min(rows, cols);
        const square = product("Muon square workspace", r, r);
        const workspaceElements = sumU32(
          "Muon workspace", product("Muon vector workspace", len, 3),
          product("Muon matrix workspace", square, 3), 2,
        );
        const workspace = auxiliary(
          "muon-workspace", product("Muon workspace bytes", workspaceElements, 4), null,
        );
        const candidateParameter = auxiliary("muon-parameter-candidate", bytes, null);
        const candidateMomentum = auxiliary("muon-momentum-candidate", bytes, null);
        const commandFactory = (optimizerStep: number) => {
          if (!Number.isSafeInteger(optimizerStep) || optimizerStep <= 0) {
            fail("invalid_schema", "optimizerStep must be a positive safe integer");
          }
          const scale = Math.fround(
            learningRate * Math.fround(Math.sqrt(Math.fround(Math.max(rows, cols)))),
          );
          const shrink = Math.fround(1 - Math.fround(learningRate * weightDecay));
          const params = uniform(32, (view) => {
            view.setUint32(0, len, true);
            view.setUint32(4, rows, true);
            view.setUint32(8, cols, true);
            view.setUint32(12, steps, true);
            view.setFloat32(16, momentumDecay, true);
            view.setFloat32(20, scale, true);
            view.setFloat32(24, shrink, true);
          });
          return Object.freeze([stage(invocation, "muon", params, {
            1: candidateParameter, 2: gradient, 3: candidateMomentum, 4: workspace,
          }, [1, 1, 1])]);
        };
        return Object.freeze({
          commands: Object.freeze([]),
          commandFactory,
          copies: Object.freeze([
            Object.freeze({
              source: parameter, sourceOffset: 0, destination: candidateParameter,
              destinationOffset: 0, byteLength: bytes,
            }),
            Object.freeze({
              source: momentum, sourceOffset: 0, destination: candidateMomentum,
              destinationOffset: 0, byteLength: bytes,
            }),
          ]),
          commitCopies: Object.freeze([
            Object.freeze({
              source: candidateParameter, sourceOffset: 0, destination: resultParameter,
              destinationOffset: 0, byteLength: bytes,
            }),
            Object.freeze({
              source: candidateMomentum, sourceOffset: 0, destination: resultMomentum,
              destinationOffset: 0, byteLength: bytes,
            }),
          ]),
        });
      }
      default:
        fail("capability_mismatch", `${invocation.operation} has no specialized WebGPU lowering`);
    }
  };

  const entries = [
    ...plan.operations.map((operation) => ({ phase: "forward" as const, id: operation.id })),
    ...plan.backwardOperations.map((operation) => ({ phase: "backward" as const, id: operation.id })),
  ];
  for (const entry of entries) {
    const invocation = compiledWebGpuInvocationV1(plan, entry.phase, entry.id);
    let template: Template;
    if (isPointwiseWebGpuOperationV1(invocation.operation)) {
      template = Object.freeze({
        commands: Object.freeze([...lowerPointwiseWebGpuOperationV1(
          plan, entry.phase, entry.id, 0,
        )]),
        copies: Object.freeze([]),
      });
    } else if (SPECIALIZED.has(invocation.operation)) {
      template = compileSpecialized(invocation);
    } else {
      continue;
    }
    templates.set(key(entry.phase, entry.id), template);
  }

  const uniformBytes = product(
    "WebGPU uniform arena", uniformSlots, uniformStride as number,
  );
  const additionalBytes = auxiliaryBytes + uniformBytes + rootPaddingBytes + 8;
  if (!Number.isSafeInteger(additionalBytes) ||
      !Number.isSafeInteger(plan.peakBytes) || plan.peakBytes < 0 ||
      plan.peakBytes > (maxPeakBytes as number) - additionalBytes) {
    fail("memory_limit", "WebGPU resident schedule exceeds maxPeakBytes");
  }
  const residentPeakBytes = plan.peakBytes + additionalBytes;
  const resources: WebGpuResidentAuxiliaryV1[] = pendingResources.map((resource) =>
    Object.freeze({
      id: resource.id,
      byteLength: resource.byteLength,
      initialBytes: resource.initialValues === null ? null : u32Bytes(resource.initialValues),
    }));

  const snapshotResources = () => Object.freeze(resources.map((resource) => Object.freeze({
    id: resource.id,
    byteLength: resource.byteLength,
    initialBytes: resource.initialBytes === null
      ? null
      : Uint8Array.from(resource.initialBytes),
  })));

  return Object.freeze({
    peakBytes(): number {
      return residentPeakBytes;
    },
    auxiliaryResources(): WebGpuResidentAuxiliarySetV1 {
      return Object.freeze({
        maxBytes: auxiliaryBytes,
        resources: snapshotResources(),
      });
    },
    transaction(
      phase: "forward" | "backward",
      operationId: string,
      firstUniformSlot: number,
      optimizerStep?: number,
    ): WebGpuResidentTransactionV1 {
      if ((phase !== "forward" && phase !== "backward") ||
          typeof operationId !== "string" || operationId.length === 0 ||
          !Number.isSafeInteger(firstUniformSlot) || firstUniformSlot < 0) {
        fail("invalid_schema", "WebGPU transaction selector is invalid");
      }
      const template = templates.get(key(phase, operationId));
      if (template === undefined) {
        fail("capability_mismatch", `compiled operation ${operationId} is not lowered`);
      }
      if ((template.commandFactory === undefined) !== (optimizerStep === undefined)) {
        fail("invalid_schema", "optimizerStep presence differs from compiled operation phase");
      }
      const commands = template.commandFactory === undefined
        ? template.commands
        : template.commandFactory(optimizerStep!);
      const finalSlot = firstUniformSlot + commands.length - 1;
      if (!Number.isSafeInteger(finalSlot) || finalSlot >= uniformSlots) {
        fail("invalid_schema", "WebGPU transaction exceeds uniform arena");
      }
      return Object.freeze({
        commands: Object.freeze(commands.map((command, index) => Object.freeze({
          ...command,
          uniformSlot: firstUniformSlot + index,
          uniformBytes: command.uniformBytes === null
            ? null
            : Uint8Array.from(command.uniformBytes),
          storageBindings: Object.freeze({ ...command.storageBindings }),
          workgroups: Object.freeze([...command.workgroups]) as readonly [number, number, number],
        }))),
        copies: Object.freeze(template.copies.map((copy) => Object.freeze({ ...copy }))),
        commitCopies: Object.freeze(
          (template.commitCopies ?? []).map((copy) => Object.freeze({ ...copy })),
        ),
      });
    },
  });
}

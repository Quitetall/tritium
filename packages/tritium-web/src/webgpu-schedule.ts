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
}

export interface WebGpuResidentScheduleBudgetV1 {
  readonly maxPeakBytes: number;
}

export interface WebGpuResidentScheduleV1 {
  auxiliaryResources(): WebGpuResidentAuxiliarySetV1;
  transaction(
    phase: "forward" | "backward",
    operationId: string,
    firstUniformSlot: number,
  ): WebGpuResidentTransactionV1;
}

type BufferMap = ReturnType<typeof admittedWebGpuBuffersV1>;
type Template = Readonly<{
  commands: readonly WebGpuResidentDispatchV1[];
  copies: readonly WebGpuResidentCopyV1[];
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
  "loss.softmax_cross_entropy",
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

function expect(
  buffers: BufferMap,
  bufferId: string,
  dtype: "f32" | "u32",
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

function stage(
  invocation: WebGpuLoweringInvocationV1,
  expectedModuleId: string,
  uniformBytes: Uint8Array,
  storageBindings: Readonly<Record<number, string>>,
  workgroups: readonly [number, number, number],
): WebGpuResidentDispatchV1 {
  const form = webGpuDispatchFormV1(invocation.operation, invocation.execution);
  if (form.stages.length !== 1 || form.stages[0]?.repeat !== "once" ||
      form.stages[0]?.moduleId !== expectedModuleId) {
    fail("invalid_schema", `${invocation.operation} specialized catalog stage drifted`);
  }
  return Object.freeze({
    operation: invocation.operation,
    execution: invocation.execution,
    stageIndex: 0,
    uniformSlot: 0,
    uniformBytes,
    storageBindings: Object.freeze({ ...storageBindings }),
    workgroups: Object.freeze([...workgroups]) as readonly [number, number, number],
  });
}

/** Compile pointwise and first-tranche specialized forms into resident transactions. */
export function compileWebGpuResidentScheduleV1(
  sourcePlan: CompiledTrainingPlanV1,
  budget: WebGpuResidentScheduleBudgetV1,
): WebGpuResidentScheduleV1 {
  const plan = snapshotPlan(sourcePlan);
  const budgetRecord = recordSnapshot(budget, "WebGPU schedule budget");
  const maxPeakBytes = property(budgetRecord, "maxPeakBytes", "WebGPU schedule budget");
  if (!Number.isSafeInteger(maxPeakBytes) || (maxPeakBytes as number) < 0) {
    fail("invalid_schema", "WebGPU schedule maxPeakBytes must be a nonnegative safe integer");
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
  for (const buffer of buffers.values()) {
    if (buffer.ownerId !== buffer.id) continue;
    if (rootBytes > Number.MAX_SAFE_INTEGER - buffer.byteLength) {
      fail("memory_limit", "compiled root buffers exceed safe integer range");
    }
    rootBytes += buffer.byteLength;
  }
  if (rootBytes > plan.residentBytes) {
    fail("invalid_schema", "compiled residentBytes omits root buffers");
  }
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

  const uniformBytes = product("WebGPU uniform arena", uniformSlots, 256);
  const additionalBytes = auxiliaryBytes + uniformBytes + 4;
  if (!Number.isSafeInteger(additionalBytes) ||
      !Number.isSafeInteger(plan.peakBytes) || plan.peakBytes < 0 ||
      plan.peakBytes > (maxPeakBytes as number) - additionalBytes) {
    fail("memory_limit", "WebGPU resident schedule exceeds maxPeakBytes");
  }
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
      const finalSlot = firstUniformSlot + template.commands.length - 1;
      if (!Number.isSafeInteger(finalSlot) || finalSlot >= uniformSlots) {
        fail("invalid_schema", "WebGPU transaction exceeds uniform arena");
      }
      return Object.freeze({
        commands: Object.freeze(template.commands.map((command, index) => Object.freeze({
          ...command,
          uniformSlot: firstUniformSlot + index,
          uniformBytes: command.uniformBytes === null
            ? null
            : Uint8Array.from(command.uniformBytes),
          storageBindings: Object.freeze({ ...command.storageBindings }),
          workgroups: Object.freeze([...command.workgroups]) as readonly [number, number, number],
        }))),
        copies: Object.freeze(template.copies.map((copy) => Object.freeze({ ...copy }))),
      });
    },
  });
}

import { PORTABLE_OPERATION_BINDINGS_V1 } from "./operation-bindings.ts";
import {
  admittedCompiledBufferMap,
  PortableSchedulePlanError,
} from "./portable-schedule.ts";
import type {
  CompiledBackwardOperationV1,
  CompiledTrainingOperationV1,
  CompiledTrainingPlanV1,
  TrainingAttributeSpecV1,
} from "./session.ts";
import { WebTrainingError } from "./session.ts";
import { webGpuDispatchFormV1 } from "./webgpu-kernels.ts";
import type { WebGpuDispatchExecutionV1 } from "./webgpu-kernels.ts";
import {
  webGpuUniformSlotCapacityV1,
  type WebGpuResidentDispatchV1,
} from "./webgpu-runtime.ts";

type Binding = Readonly<{
  inputs: readonly string[];
  attributes: readonly Readonly<{
    name: string;
    kind: TrainingAttributeSpecV1["kind"];
  }>[];
  outputs: readonly string[];
}>;
type Registry = Readonly<
  Record<string, Readonly<Partial<Record<WebGpuDispatchExecutionV1, Binding>>>>
>;
export type WebGpuLoweringInvocationV1 = Readonly<{
  operation: string;
  execution: WebGpuDispatchExecutionV1;
  inputs: Readonly<Record<string, string>>;
  outputs: Readonly<Record<string, string>>;
  attributes: Readonly<Record<string, TrainingAttributeSpecV1["value"]>>;
}>;

const BINDINGS = PORTABLE_OPERATION_BINDINGS_V1 as unknown as Registry;
const POINTWISE_OPERATIONS = new Set([
  "graph.detach",
  "graph.scale_const",
  "graph.add",
  "graph.mul",
  "graph.relu2",
  "graph.silu",
  "graph.causal_mask",
  "graph.softmax",
  "graph.rmsnorm",
  "loss.mse",
  "graph.bias",
  "graph.transpose",
  "graph.slice_cols",
  "graph.dense_matmul",
  "graph.ternary_matmul",
  "graph.ste_surrogate",
  "graph.lsq_ste",
]);

export function isPointwiseWebGpuOperationV1(operation: string): boolean {
  return POINTWISE_OPERATIONS.has(operation);
}

function fail(
  code: "capability_mismatch" | "invalid_schema" | "memory_limit",
  message: string,
): never {
  throw new WebTrainingError(code, message);
}

function dense(value: readonly unknown[]): boolean {
  return Object.keys(value).length === value.length;
}

function record(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function roleMap(roles: readonly string[], ids: readonly string[], name: string) {
  if (!Array.isArray(ids) || !dense(ids) || ids.length !== roles.length ||
      ids.some((id) => typeof id !== "string" || id.length === 0)) {
    fail("invalid_schema", `compiled ${name} bindings differ from WebGPU registry`);
  }
  return Object.freeze(Object.fromEntries(roles.map((role, index) => [role, ids[index]!])));
}

function attributeMap(
  expected: Binding["attributes"],
  actual: readonly TrainingAttributeSpecV1[],
) {
  if (!Array.isArray(actual) || !dense(actual) || actual.length !== expected.length) {
    fail("invalid_schema", "compiled attributes differ from WebGPU registry");
  }
  const result: Record<string, TrainingAttributeSpecV1["value"]> = Object.create(null);
  expected.forEach((descriptor, index) => {
    const attribute = actual[index];
    if (attribute?.name !== descriptor.name || attribute.kind !== descriptor.kind) {
      fail("invalid_schema", "compiled attribute order differs from WebGPU registry");
    }
    result[descriptor.name] = attribute.value;
  });
  return Object.freeze(result);
}

export function compiledWebGpuInvocationV1(
  plan: CompiledTrainingPlanV1,
  phase: "forward" | "backward",
  operationId: string,
): WebGpuLoweringInvocationV1 {
  const validForward = plan.operations.every((operation) =>
    record(operation) && typeof operation.id === "string" &&
    typeof operation.operation === "string" && Array.isArray(operation.inputs) &&
    Array.isArray(operation.outputs) && Array.isArray(operation.attributes)
  );
  const validBackward = plan.backwardOperations.every((operation) =>
    record(operation) && typeof operation.id === "string" &&
    typeof operation.operation === "string" &&
    (operation.execution === "forward" || operation.execution === "vjp") &&
    Array.isArray(operation.inputs) && Array.isArray(operation.outputs) &&
    Array.isArray(operation.attributes)
  );
  if (!validForward || !validBackward || typeof operationId !== "string" || operationId.length === 0) {
    fail("invalid_schema", "compiled WebGPU operation entries are invalid");
  }
  if (phase === "forward") {
    const matches = plan.operations.filter((operation) => operation.id === operationId);
    if (matches.length !== 1) {
      fail("invalid_schema", `compiled forward operation ${operationId} is missing or duplicated`);
    }
    const operation = matches[0] as CompiledTrainingOperationV1;
    const execution: WebGpuDispatchExecutionV1 = operation.operation.startsWith("optimizer.")
      ? "step"
      : "forward";
    const binding = BINDINGS[operation.operation]?.[execution];
    if (binding === undefined) fail("invalid_schema", "WebGPU operation binding is absent");
    return Object.freeze({
      operation: operation.operation,
      execution,
      inputs: roleMap(binding.inputs, operation.inputs, "input"),
      outputs: roleMap(binding.outputs, operation.outputs, "output"),
      attributes: attributeMap(binding.attributes, operation.attributes),
    });
  }
  const matches = plan.backwardOperations.filter((operation) => operation.id === operationId);
  if (matches.length !== 1) {
    fail("invalid_schema", `compiled backward operation ${operationId} is missing or duplicated`);
  }
  const operation = matches[0] as CompiledBackwardOperationV1;
  const binding = BINDINGS[operation.operation]?.[operation.execution];
  if (binding === undefined) fail("invalid_schema", "WebGPU backward binding is absent");
  const bindingsByRole = (
    actual: CompiledBackwardOperationV1["inputs"],
    expected: readonly string[],
    name: string,
  ) => {
    if (!Array.isArray(actual) || !dense(actual) || actual.length !== expected.length) {
      fail("invalid_schema", `compiled backward ${name} bindings differ from registry`);
    }
    const result: Record<string, string> = Object.create(null);
    actual.forEach((value, index) => {
      if (value?.role !== expected[index] || typeof value.bufferId !== "string") {
        fail("invalid_schema", `compiled backward ${name} role differs from registry`);
      }
      result[value.role] = value.bufferId;
    });
    return Object.freeze(result);
  };
  return Object.freeze({
    operation: operation.operation,
    execution: operation.execution,
    inputs: bindingsByRole(operation.inputs, binding.inputs, "input"),
    outputs: bindingsByRole(operation.outputs, binding.outputs, "output"),
    attributes: attributeMap(binding.attributes, operation.attributes),
  });
}

export function requiredWebGpuRoleV1(
  map: Readonly<Record<string, string>>,
  role: string,
): string {
  const value = map[role];
  if (value === undefined) fail("invalid_schema", `WebGPU lowering omits role ${role}`);
  return value;
}

export function webGpuU32V1(value: unknown, name: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0 || (value as number) > 0xffff_ffff) {
    fail("invalid_schema", `${name} must fit u32`);
  }
  return value as number;
}

export function webGpuF32V1(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(Math.fround(value))) {
    fail("invalid_schema", `${name} must be finite f32`);
  }
  return Math.fround(value);
}

type Invocation = WebGpuLoweringInvocationV1;
const invocationFor = compiledWebGpuInvocationV1;
const required = requiredWebGpuRoleV1;
const u32 = webGpuU32V1;
const f32 = webGpuF32V1;

export function admittedWebGpuBuffersV1(
  plan: CompiledTrainingPlanV1,
): ReturnType<typeof admittedCompiledBufferMap> {
  try {
    return admittedCompiledBufferMap(plan);
  } catch (error) {
    if (error instanceof PortableSchedulePlanError) {
      fail(
        error.code === "capacity" ? "memory_limit" : "invalid_schema",
        error.message,
      );
    }
    throw error;
  }
}

function validatePointwiseGeometry(
  invocation: Invocation,
  buffers: ReadonlyMap<string, CompiledTrainingPlanV1["buffers"][number]>,
): void {
  const expect = (bufferId: string, expected: readonly number[], role: string) => {
    const actual = buffers.get(bufferId);
    if (actual === undefined || actual.dtype !== "f32" ||
        actual.shape.length !== expected.length ||
        actual.shape.some((dimension, index) => dimension !== expected[index])) {
      fail("invalid_schema", `${invocation.operation} ${role} geometry differs from WebGPU ABI`);
    }
  };
  const input = invocation.inputs;
  const output = invocation.outputs;
  const same = (inputRole: string, outputRole: string) => {
    const source = required(input, inputRole);
    expect(required(output, outputRole), buffers.get(source)?.shape ?? [-1], outputRole);
  };
  const rowsCols = () => [
    u32(invocation.attributes.rows, "rows"),
    u32(invocation.attributes.cols, "cols"),
  ] as const;
  switch (`${invocation.operation}|${invocation.execution}`) {
    case "graph.detach|forward": same("x", "result"); return;
    case "graph.detach|vjp": same("grad_output", "grad_x"); return;
    case "graph.scale_const|forward": same("x", "result"); return;
    case "graph.scale_const|vjp": same("grad_output", "grad_x"); return;
    case "graph.add|forward":
      expect(required(input, "right"), buffers.get(required(input, "left"))?.shape ?? [-1], "right");
      same("left", "result");
      return;
    case "graph.add|vjp":
      same("grad_output", "grad_left");
      same("grad_output", "grad_right");
      return;
    case "graph.mul|forward":
      expect(required(input, "right"), buffers.get(required(input, "left"))?.shape ?? [-1], "right");
      same("left", "result");
      return;
    case "graph.mul|vjp": {
      const shape = buffers.get(required(input, "left"))?.shape ?? [-1];
      expect(required(input, "right"), shape, "right");
      expect(required(input, "grad_output"), shape, "grad_output");
      expect(required(output, "grad_left"), shape, "grad_left");
      expect(required(output, "grad_right"), shape, "grad_right");
      return;
    }
    case "graph.relu2|forward":
    case "graph.silu|forward": same("x", "result"); return;
    case "graph.relu2|vjp":
    case "graph.silu|vjp": {
      const shape = buffers.get(required(input, "x"))?.shape ?? [-1];
      expect(required(input, "grad_output"), shape, "grad_output");
      expect(required(output, "grad_x"), shape, "grad_x");
      return;
    }
    case "graph.causal_mask|forward":
    case "graph.softmax|forward": {
      const shape = rowsCols();
      expect(required(input, "x"), shape, "x");
      expect(required(output, "result"), shape, "result");
      return;
    }
    case "graph.causal_mask|vjp": {
      const shape = rowsCols();
      expect(required(input, "grad_output"), shape, "grad_output");
      expect(required(output, "grad_x"), shape, "grad_x");
      return;
    }
    case "graph.softmax|vjp": {
      const shape = rowsCols();
      expect(required(input, "x"), shape, "x");
      expect(required(input, "grad_output"), shape, "grad_output");
      expect(required(output, "grad_x"), shape, "grad_x");
      return;
    }
    case "graph.rmsnorm|forward":
    case "graph.rmsnorm|vjp": {
      const [rows, cols] = rowsCols();
      expect(required(input, "x"), [rows, cols], "x");
      expect(required(input, "weight"), [cols], "weight");
      if (invocation.execution === "forward") {
        expect(required(output, "result"), [rows, cols], "result");
      } else {
        expect(required(input, "grad_output"), [rows, cols], "grad_output");
        expect(required(output, "grad_x"), [rows, cols], "grad_x");
        expect(required(output, "grad_weight"), [cols], "grad_weight");
      }
      return;
    }
    case "loss.mse|forward":
    case "loss.mse|vjp": {
      const shape = buffers.get(required(input, "prediction"))?.shape ?? [-1];
      expect(required(input, "target"), shape, "target");
      if (invocation.execution === "forward") {
        expect(required(output, "result"), [], "result");
      } else {
        expect(required(input, "grad_output"), [], "grad_output");
        expect(required(output, "grad_prediction"), shape, "grad_prediction");
      }
      return;
    }
    case "graph.bias|forward":
    case "graph.bias|vjp": {
      const [rows, cols] = rowsCols();
      expect(required(input, "x"), [rows, cols], "x");
      expect(required(input, "bias"), [cols], "bias");
      if (invocation.execution === "forward") {
        expect(required(output, "result"), [rows, cols], "result");
      } else {
        expect(required(input, "grad_output"), [rows, cols], "grad_output");
        expect(required(output, "grad_x"), [rows, cols], "grad_x");
        expect(required(output, "grad_bias"), [cols], "grad_bias");
      }
      return;
    }
    case "graph.transpose|forward":
    case "graph.transpose|vjp": {
      const [rows, cols] = rowsCols();
      if (invocation.execution === "forward") {
        expect(required(input, "x"), [rows, cols], "x");
        expect(required(output, "result"), [cols, rows], "result");
      } else {
        expect(required(input, "grad_output"), [cols, rows], "grad_output");
        expect(required(output, "grad_x"), [rows, cols], "grad_x");
      }
      return;
    }
    case "graph.slice_cols|forward":
    case "graph.slice_cols|vjp": {
      const [rows, cols] = rowsCols();
      const start = u32(invocation.attributes.start, "start");
      const len = u32(invocation.attributes.len, "len");
      if (start + len > cols) fail("invalid_schema", "slice exceeds WebGPU column bounds");
      if (invocation.execution === "forward") {
        expect(required(input, "x"), [rows, cols], "x");
        expect(required(output, "result"), [rows, len], "result");
      } else {
        expect(required(input, "grad_output"), [rows, len], "grad_output");
        expect(required(output, "grad_x"), [rows, cols], "grad_x");
      }
      return;
    }
    case "graph.dense_matmul|forward":
    case "graph.dense_matmul|vjp": {
      const m = u32(invocation.attributes.m, "m");
      const n = u32(invocation.attributes.n, "n");
      const k = u32(invocation.attributes.k, "k");
      expect(required(input, "x"), [m, k], "x");
      expect(required(input, "weight"), [n, k], "weight");
      if (invocation.execution === "forward") {
        expect(required(output, "result"), [m, n], "result");
      } else {
        expect(required(input, "grad_output"), [m, n], "grad_output");
        expect(required(output, "grad_x"), [m, k], "grad_x");
        expect(required(output, "grad_weight"), [n, k], "grad_weight");
      }
      return;
    }
    case "graph.ternary_matmul|forward":
    case "graph.ternary_matmul|vjp": {
      const m = u32(invocation.attributes.m, "m");
      const n = u32(invocation.attributes.n, "n");
      const k = u32(invocation.attributes.k, "k");
      expect(required(input, "activation"), [m, k], "activation");
      expect(required(input, "weight"), [n, k], "weight");
      expect(required(input, "scale"), [n], "scale");
      if (invocation.execution === "forward") {
        expect(required(output, "result"), [m, n], "result");
      } else {
        expect(required(input, "grad_output"), [m, n], "grad_output");
        expect(required(output, "grad_activation"), [m, k], "grad_activation");
        expect(required(output, "grad_weight"), [n, k], "grad_weight");
        expect(required(output, "grad_scale"), [n], "grad_scale");
      }
      return;
    }
    case "graph.ste_surrogate|forward":
    case "graph.ste_surrogate|vjp":
    case "graph.lsq_ste|forward":
    case "graph.lsq_ste|vjp": {
      const [rows, cols] = rowsCols();
      const scaleRole = invocation.operation === "graph.lsq_ste" ? "alpha" : "scale";
      expect(required(input, "weight"), [rows, cols], "weight");
      expect(required(input, scaleRole), [rows], scaleRole);
      if (invocation.execution === "forward") {
        expect(required(output, "result"), [rows, cols], "result");
      } else {
        expect(required(input, "grad_output"), [rows, cols], "grad_output");
        expect(required(output, "grad_weight"), [rows, cols], "grad_weight");
        expect(required(output, invocation.operation === "graph.lsq_ste" ? "grad_alpha" : "grad_scale"), [rows], "scale gradient");
      }
      return;
    }
    default:
      fail("capability_mismatch", `${invocation.operation} pointwise geometry is not lowered`);
  }
}

function pointwiseUniform(
  len: number,
  selector: number,
  scalar: number,
  auxiliary: number,
  secondary = 0,
  tertiary = 0,
): Uint8Array {
  const bytes = new Uint8Array(32);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, u32(len, "pointwise length"), true);
  view.setUint32(4, u32(selector, "pointwise selector"), true);
  view.setFloat32(8, f32(scalar, "pointwise scalar"), true);
  view.setUint32(12, u32(auxiliary, "pointwise auxiliary"), true);
  view.setUint32(16, u32(secondary, "pointwise secondary"), true);
  view.setUint32(20, u32(tertiary, "pointwise tertiary"), true);
  return bytes;
}

/** Lower one compiled pointwise operation directly to resident WebGPU commands. */
export function lowerPointwiseWebGpuOperationV1(
  plan: CompiledTrainingPlanV1,
  phase: "forward" | "backward",
  operationId: string,
  firstUniformSlot: number,
): readonly WebGpuResidentDispatchV1[] {
  const buffers = admittedWebGpuBuffersV1(plan);
  if (phase !== "forward" && phase !== "backward") {
    fail("invalid_schema", "WebGPU lowering phase must be forward or backward");
  }
  const invocation = invocationFor(plan, phase, operationId);
  if (!POINTWISE_OPERATIONS.has(invocation.operation)) {
    fail("capability_mismatch", `${invocation.operation} has no pointwise WebGPU lowering`);
  }
  validatePointwiseGeometry(invocation, buffers);
  const elements = (bufferId: string): number => {
    const buffer = buffers.get(bufferId);
    if (buffer === undefined || buffer.dtype !== "f32" || buffer.byteLength % 4 !== 0) {
      fail("invalid_schema", `${bufferId} is not a compiled f32 WebGPU buffer`);
    }
    return buffer.byteLength / 4;
  };
  const form = webGpuDispatchFormV1(invocation.operation, invocation.execution);
  if (!Number.isSafeInteger(firstUniformSlot) || firstUniformSlot < 0) {
    fail("invalid_schema", "firstUniformSlot must be a nonnegative safe integer");
  }
  const uniformSlots = webGpuUniformSlotCapacityV1(plan);
  const finalUniformSlot = firstUniformSlot + form.stages.length - 1;
  if (!Number.isSafeInteger(finalUniformSlot) || finalUniformSlot >= uniformSlots) {
    fail("invalid_schema", "pointwise uniform slots exceed the compiled WebGPU arena");
  }
  const stage = (
    stageIndex: number,
    left: string,
    right: string,
    extra: string,
    output: string,
    len = elements(output),
    scalar = 0,
    auxiliary = 0,
    secondary = 0,
    tertiary = 0,
  ): WebGpuResidentDispatchV1 => {
    const descriptor = form.stages[stageIndex];
    if (descriptor?.moduleId !== "pointwise" || descriptor.selector === null) {
      fail("invalid_schema", "shared catalog pointwise stage drifted");
    }
    return Object.freeze({
      operation: invocation.operation,
      execution: invocation.execution,
      stageIndex,
      uniformSlot: firstUniformSlot + stageIndex,
      uniformBytes: pointwiseUniform(
        len,
        descriptor.selector,
        scalar,
        auxiliary,
        secondary,
        tertiary,
      ),
      storageBindings: Object.freeze({ 1: left, 2: right, 3: extra, 4: output }),
      workgroups: Object.freeze([Math.ceil(len / 64), 1, 1]) as readonly [number, number, number],
    });
  };
  const input = invocation.inputs;
  const output = invocation.outputs;
  switch (`${invocation.operation}|${invocation.execution}`) {
    case "graph.detach|forward": {
      const x = required(input, "x");
      return Object.freeze([stage(0, x, x, x, required(output, "result"))]);
    }
    case "graph.detach|vjp": {
      const gradient = required(input, "grad_output");
      return Object.freeze([stage(0, gradient, gradient, gradient, required(output, "grad_x"))]);
    }
    case "graph.scale_const|forward":
    case "graph.scale_const|vjp": {
      const source = required(input, invocation.execution === "forward" ? "x" : "grad_output");
      const target = required(output, invocation.execution === "forward" ? "result" : "grad_x");
      return Object.freeze([
        stage(0, source, source, source, target, elements(target),
          f32(invocation.attributes.scale, "scale")),
      ]);
    }
    case "graph.add|forward": {
      const left = required(input, "left");
      return Object.freeze([stage(
        0,
        left,
        required(input, "right"),
        left,
        required(output, "result"),
      )]);
    }
    case "graph.add|vjp": {
      const gradient = required(input, "grad_output");
      return Object.freeze([
        stage(0, gradient, gradient, gradient, required(output, "grad_left")),
        stage(1, gradient, gradient, gradient, required(output, "grad_right")),
      ]);
    }
    case "graph.mul|forward": {
      const left = required(input, "left");
      return Object.freeze([stage(
        0,
        left,
        required(input, "right"),
        left,
        required(output, "result"),
      )]);
    }
    case "graph.mul|vjp": {
      const gradient = required(input, "grad_output");
      return Object.freeze([
        stage(0, gradient, required(input, "right"), gradient, required(output, "grad_left")),
        stage(1, gradient, required(input, "left"), gradient, required(output, "grad_right")),
      ]);
    }
    case "graph.relu2|forward":
    case "graph.silu|forward": {
      const x = required(input, "x");
      return Object.freeze([stage(0, x, x, x, required(output, "result"))]);
    }
    case "graph.relu2|vjp":
    case "graph.silu|vjp": {
      const x = required(input, "x");
      return Object.freeze([stage(
        0,
        x,
        required(input, "grad_output"),
        x,
        required(output, "grad_x"),
      )]);
    }
    case "graph.causal_mask|forward":
    case "graph.causal_mask|vjp": {
      const source = required(input, invocation.execution === "forward" ? "x" : "grad_output");
      return Object.freeze([stage(
        0,
        source,
        source,
        source,
        required(output, invocation.execution === "forward" ? "result" : "grad_x"),
        elements(source),
        0,
        u32(invocation.attributes.cols, "cols"),
      )]);
    }
    case "graph.softmax|forward": {
      const x = required(input, "x");
      return Object.freeze([stage(
        0,
        x,
        x,
        x,
        required(output, "result"),
        elements(x),
        0,
        u32(invocation.attributes.cols, "cols"),
      )]);
    }
    case "graph.softmax|vjp": {
      const x = required(input, "x");
      return Object.freeze([stage(
        0,
        x,
        required(input, "grad_output"),
        x,
        required(output, "grad_x"),
        elements(x),
        0,
        u32(invocation.attributes.cols, "cols"),
      )]);
    }
    case "graph.rmsnorm|forward": {
      const x = required(input, "x");
      return Object.freeze([stage(
        0,
        x,
        required(input, "weight"),
        x,
        required(output, "result"),
        elements(x),
        f32(invocation.attributes.eps, "eps"),
        u32(invocation.attributes.cols, "cols"),
      )]);
    }
    case "graph.rmsnorm|vjp": {
      const x = required(input, "x");
      const weight = required(input, "weight");
      const gradient = required(input, "grad_output");
      const len = elements(x);
      const eps = f32(invocation.attributes.eps, "eps");
      const cols = u32(invocation.attributes.cols, "cols");
      return Object.freeze([
        stage(0, x, weight, gradient, required(output, "grad_x"), len, eps, cols),
        stage(1, x, weight, gradient, required(output, "grad_weight"), len, eps, cols),
      ]);
    }
    case "loss.mse|forward": {
      const prediction = required(input, "prediction");
      return Object.freeze([stage(
        0,
        prediction,
        required(input, "target"),
        prediction,
        required(output, "result"),
        elements(prediction),
      )]);
    }
    case "loss.mse|vjp": {
      const prediction = required(input, "prediction");
      return Object.freeze([stage(
        0,
        prediction,
        required(input, "target"),
        required(input, "grad_output"),
        required(output, "grad_prediction"),
        elements(prediction),
      )]);
    }
    case "graph.bias|forward": {
      const x = required(input, "x");
      return Object.freeze([stage(
        0,
        x,
        required(input, "bias"),
        x,
        required(output, "result"),
        elements(x),
        0,
        u32(invocation.attributes.cols, "cols"),
      )]);
    }
    case "graph.bias|vjp": {
      const x = required(input, "x");
      const bias = required(input, "bias");
      const gradient = required(input, "grad_output");
      const len = elements(x);
      const cols = u32(invocation.attributes.cols, "cols");
      return Object.freeze([
        stage(0, x, bias, gradient, required(output, "grad_x"), len, 0, cols),
        stage(1, x, bias, gradient, required(output, "grad_bias"), len, 0, cols),
      ]);
    }
    case "graph.transpose|forward":
    case "graph.transpose|vjp": {
      const source = required(input, invocation.execution === "forward" ? "x" : "grad_output");
      return Object.freeze([stage(
        0,
        source,
        source,
        source,
        required(output, invocation.execution === "forward" ? "result" : "grad_x"),
        elements(source),
        0,
        u32(invocation.attributes.cols, "cols"),
      )]);
    }
    case "graph.slice_cols|forward":
    case "graph.slice_cols|vjp": {
      const source = required(
        input,
        invocation.execution === "forward" ? "x" : "grad_output",
      );
      const target = required(
        output,
        invocation.execution === "forward" ? "result" : "grad_x",
      );
      return Object.freeze([stage(
        0,
        source,
        source,
        source,
        target,
        elements(target),
        0,
        u32(invocation.attributes.cols, "cols"),
        u32(invocation.attributes.start, "start"),
        u32(invocation.attributes.len, "len"),
      )]);
    }
    case "graph.dense_matmul|forward": {
      const x = required(input, "x");
      return Object.freeze([stage(
        0,
        x,
        required(input, "weight"),
        x,
        required(output, "result"),
        elements(required(output, "result")),
        0,
        u32(invocation.attributes.m, "m"),
        u32(invocation.attributes.n, "n"),
        u32(invocation.attributes.k, "k"),
      )]);
    }
    case "graph.dense_matmul|vjp": {
      const x = required(input, "x");
      const weight = required(input, "weight");
      const gradient = required(input, "grad_output");
      const m = u32(invocation.attributes.m, "m");
      const n = u32(invocation.attributes.n, "n");
      const k = u32(invocation.attributes.k, "k");
      return Object.freeze([
        stage(0, x, weight, gradient, required(output, "grad_x"), elements(x), 0, m, n, k),
        stage(
          1,
          x,
          weight,
          gradient,
          required(output, "grad_weight"),
          elements(weight),
          0,
          m,
          n,
          k,
        ),
      ]);
    }
    case "graph.ternary_matmul|forward": {
      const activation = required(input, "activation");
      return Object.freeze([stage(
        0,
        activation,
        required(input, "weight"),
        required(input, "scale"),
        required(output, "result"),
        elements(required(output, "result")),
        0,
        u32(invocation.attributes.m, "m"),
        u32(invocation.attributes.n, "n"),
        u32(invocation.attributes.k, "k"),
      )]);
    }
    case "graph.ternary_matmul|vjp": {
      const activation = required(input, "activation");
      const weight = required(input, "weight");
      const scale = required(input, "scale");
      const gradient = required(input, "grad_output");
      const m = u32(invocation.attributes.m, "m");
      const n = u32(invocation.attributes.n, "n");
      const k = u32(invocation.attributes.k, "k");
      return Object.freeze([
        stage(
          0,
          gradient,
          weight,
          scale,
          required(output, "grad_activation"),
          elements(activation),
          0,
          m,
          n,
          k,
        ),
        stage(
          1,
          gradient,
          activation,
          scale,
          required(output, "grad_weight"),
          elements(weight),
          0,
          m,
          n,
          k,
        ),
        stage(
          2,
          gradient,
          activation,
          weight,
          required(output, "grad_scale"),
          elements(scale),
          0,
          m,
          n,
          k,
        ),
      ]);
    }
    case "graph.ste_surrogate|forward": {
      const weight = required(input, "weight");
      return Object.freeze([stage(
        0,
        weight,
        required(input, "scale"),
        weight,
        required(output, "result"),
        elements(weight),
        0,
        u32(invocation.attributes.cols, "cols"),
      )]);
    }
    case "graph.ste_surrogate|vjp": {
      const weight = required(input, "weight");
      const scale = required(input, "scale");
      return Object.freeze([
        stage(
          0,
          weight,
          scale,
          required(input, "grad_output"),
          required(output, "grad_weight"),
          elements(weight),
          0,
          u32(invocation.attributes.cols, "cols"),
        ),
        stage(1, scale, scale, scale, required(output, "grad_scale"), elements(scale)),
      ]);
    }
    case "graph.lsq_ste|forward": {
      const weight = required(input, "weight");
      return Object.freeze([stage(
        0,
        weight,
        required(input, "alpha"),
        weight,
        required(output, "result"),
        elements(weight),
        0,
        u32(invocation.attributes.cols, "cols"),
      )]);
    }
    case "graph.lsq_ste|vjp": {
      const weight = required(input, "weight");
      const alpha = required(input, "alpha");
      const gradient = required(input, "grad_output");
      const cols = u32(invocation.attributes.cols, "cols");
      return Object.freeze([
        stage(0, weight, alpha, gradient, required(output, "grad_weight"), elements(weight), 0, cols),
        stage(1, weight, alpha, gradient, required(output, "grad_alpha"), elements(alpha), 0, cols),
      ]);
    }
    default:
      fail("capability_mismatch", `${invocation.operation} pointwise phase is not lowered`);
  }
}

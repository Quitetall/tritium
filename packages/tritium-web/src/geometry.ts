import { PORTABLE_OPERATION_BINDINGS_V1 } from "./operation-bindings.ts";
import type {
  TrainingAttributeKindV1,
  TrainingAttributeSpecV1,
  TrainingOperationSpecV1,
  TrainingTensorSpecV1,
} from "./session.ts";

const MAX_U32 = 0xffff_ffff;
const MAX_SALT_PLANES = 64;
const MAX_SCRATCH_BYTES = 64 * 1024 * 1024;

type Binding = Readonly<{
  attributes: readonly Readonly<{ name: string; kind: TrainingAttributeKindV1 }>[];
}>;
type BindingRegistry = Readonly<
  Record<string, Readonly<Partial<Record<"forward" | "step", Binding>>>>
>;
const BINDINGS = PORTABLE_OPERATION_BINDINGS_V1 as unknown as BindingRegistry;

export class TrainingGeometryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TrainingGeometryError";
  }
}

function fail(operation: TrainingOperationSpecV1, message: string): never {
  throw new TrainingGeometryError(`${operation.id} ${message}`);
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function requireShape(
  operation: TrainingOperationSpecV1,
  tensor: TrainingTensorSpecV1,
  expected: readonly number[],
  role: string,
): void {
  if (!sameShape(tensor.shape, expected)) {
    fail(operation, `${role} shape must be [${expected.join(",")}]`);
  }
}

function checkedAdd(operation: TrainingOperationSpecV1, ...values: readonly number[]): number {
  let total = 0;
  for (const value of values) {
    total += value;
    if (!Number.isSafeInteger(total)) fail(operation, "geometry addition overflowed");
  }
  return total;
}

function checkedProduct(
  operation: TrainingOperationSpecV1,
  values: readonly number[],
  maximum = Number.MAX_SAFE_INTEGER,
): number {
  let product = 1;
  for (const value of values) {
    product *= value;
    if (!Number.isSafeInteger(product) || product > maximum) {
      fail(operation, "geometry product exceeds its bounded range");
    }
  }
  return product;
}

function attributes(operation: TrainingOperationSpecV1): ReadonlyMap<string, TrainingAttributeSpecV1> {
  const execution = operation.operation.startsWith("optimizer.") ? "step" : "forward";
  const expected = BINDINGS[operation.operation]?.[execution]?.attributes;
  if (
    expected === undefined ||
    expected.length !== operation.attributes.length ||
    operation.attributes.some(
      (attribute, index) =>
        attribute.name !== expected[index]?.name || attribute.kind !== expected[index]?.kind,
    )
  ) {
    fail(operation, "attributes differ from canonical ABI");
  }
  return new Map(operation.attributes.map((attribute) => [attribute.name, attribute]));
}

function numberAttribute(
  operation: TrainingOperationSpecV1,
  values: ReadonlyMap<string, TrainingAttributeSpecV1>,
  name: string,
): number {
  const value = values.get(name)?.value;
  if (typeof value !== "number") fail(operation, `${name} must be numeric`);
  return value;
}

function listAttribute(
  operation: TrainingOperationSpecV1,
  values: ReadonlyMap<string, TrainingAttributeSpecV1>,
  name: string,
): readonly number[] {
  const value = values.get(name)?.value;
  if (!Array.isArray(value)) fail(operation, `${name} must be a list`);
  return value;
}

function textAttribute(
  operation: TrainingOperationSpecV1,
  values: ReadonlyMap<string, TrainingAttributeSpecV1>,
  name: string,
): string {
  const value = values.get(name)?.value;
  if (typeof value !== "string") fail(operation, `${name} must be text`);
  return value;
}

function requirePositiveU32(
  operation: TrainingOperationSpecV1,
  values: ReadonlyMap<string, TrainingAttributeSpecV1>,
  name: string,
): number {
  const value = numberAttribute(operation, values, name);
  if (!Number.isSafeInteger(value) || value <= 0 || value > MAX_U32) {
    fail(operation, `${name} must be positive u32`);
  }
  return value;
}

function requireNonnegativeU32(
  operation: TrainingOperationSpecV1,
  values: ReadonlyMap<string, TrainingAttributeSpecV1>,
  name: string,
): number {
  const value = numberAttribute(operation, values, name);
  if (!Number.isSafeInteger(value) || value < 0 || value > MAX_U32) {
    fail(operation, `${name} must be nonnegative u32`);
  }
  return value;
}

function requireSame(
  operation: TrainingOperationSpecV1,
  tensors: readonly TrainingTensorSpecV1[],
): void {
  const expected = tensors[0]!.shape;
  for (const tensor of tensors.slice(1)) requireShape(operation, tensor, expected, tensor.id);
}

function validateAdamAttributes(
  operation: TrainingOperationSpecV1,
  values: ReadonlyMap<string, TrainingAttributeSpecV1>,
): void {
  if (numberAttribute(operation, values, "step") !== 0) {
    fail(operation, "recipe step must start at zero");
  }
  if (numberAttribute(operation, values, "lr") < 0) fail(operation, "lr must be nonnegative");
  for (const name of ["beta1", "beta2"] as const) {
    const value = numberAttribute(operation, values, name);
    if (value < 0 || value >= 1) fail(operation, `${name} must be in [0,1)`);
  }
  if (numberAttribute(operation, values, "eps") <= 0) fail(operation, "eps must be positive");
  if (numberAttribute(operation, values, "weight_decay") < 0) {
    fail(operation, "weight_decay must be nonnegative");
  }
}

function convOutputAxis(
  operation: TrainingOperationSpecV1,
  input: number,
  kernel: number,
  stride: number,
  dilation: number,
  padBefore: number,
  padAfter: number,
): number {
  const effective = checkedAdd(operation, checkedProduct(operation, [dilation, kernel - 1]), 1);
  const padded = checkedAdd(operation, input, padBefore, padAfter);
  if (effective > MAX_U32 || padded > MAX_U32 || padded < effective) {
    fail(operation, "convolution axis has no bounded output");
  }
  return Math.floor((padded - effective) / stride) + 1;
}

/** Validate canonical forward/step geometry without allocating tensor storage. */
export function validateTrainingOperationGeometry(
  operation: TrainingOperationSpecV1,
  inputs: readonly TrainingTensorSpecV1[],
  outputs: readonly TrainingTensorSpecV1[],
  sessionSeed?: number,
): void {
  const values = attributes(operation);
  switch (operation.operation) {
    case "graph.detach":
    case "graph.relu2":
    case "graph.silu":
      requireSame(operation, [inputs[0]!, outputs[0]!]);
      return;
    case "graph.scale_const":
      numberAttribute(operation, values, "scale");
      requireSame(operation, [inputs[0]!, outputs[0]!]);
      return;
    case "graph.add":
    case "graph.mul":
      requireSame(operation, [inputs[0]!, inputs[1]!, outputs[0]!]);
      return;
    case "graph.causal_mask":
    case "graph.softmax": {
      const rows = requirePositiveU32(operation, values, "rows");
      const cols = requirePositiveU32(operation, values, "cols");
      requireShape(operation, inputs[0]!, [rows, cols], "input");
      requireShape(operation, outputs[0]!, [rows, cols], "output");
      return;
    }
    case "graph.salt_ste": {
      const rows = requirePositiveU32(operation, values, "rows");
      const cols = requirePositiveU32(operation, values, "cols");
      const planes = requirePositiveU32(operation, values, "planes");
      if (planes > MAX_SALT_PLANES) fail(operation, "planes exceeds 64");
      if (cols * 4 > MAX_SCRATCH_BYTES) fail(operation, "SALT scratch exceeds 64 MiB");
      requireShape(operation, inputs[0]!, [rows, cols], "weight");
      requireShape(operation, outputs[0]!, [rows, cols], "result");
      return;
    }
    case "graph.fsq": {
      const channels = requirePositiveU32(operation, values, "channels");
      const len = requirePositiveU32(operation, values, "len");
      checkedProduct(operation, [channels, len], MAX_U32);
      const levels = listAttribute(operation, values, "levels");
      if (levels.length !== channels || levels.some((level) => level < 2)) {
        fail(operation, "levels must contain one value >=2 per channel");
      }
      if (!["clamp", "tanh"].includes(textAttribute(operation, values, "bound"))) {
        fail(operation, "bound is unknown");
      }
      if (!["hard", "soft_round", "stochastic"].includes(textAttribute(operation, values, "ste"))) {
        fail(operation, "ste is unknown");
      }
      const alpha = numberAttribute(operation, values, "alpha");
      if (alpha < 0 || alpha > 1) fail(operation, "alpha must be in [0,1]");
      const seed = numberAttribute(operation, values, "seed");
      if (sessionSeed !== undefined && seed !== sessionSeed) {
        fail(operation, "seed differs from session seed");
      }
      requireShape(operation, inputs[0]!, [channels, len], "x");
      requireShape(operation, outputs[0]!, [channels, len], "result");
      return;
    }
    case "graph.rope": {
      const positions = listAttribute(operation, values, "positions");
      if (positions.length === 0 || positions.some((position) => position > MAX_U32)) {
        fail(operation, "positions must be nonempty u32 values");
      }
      const nHead = requirePositiveU32(operation, values, "n_head");
      const headDim = requirePositiveU32(operation, values, "head_dim");
      if (headDim % 2 !== 0) fail(operation, "head_dim must be even");
      if (numberAttribute(operation, values, "theta") <= 0) fail(operation, "theta must be positive");
      checkedProduct(operation, [positions.length, nHead, headDim], MAX_U32);
      const shape = [positions.length, nHead, headDim];
      requireShape(operation, inputs[0]!, shape, "x");
      requireShape(operation, outputs[0]!, shape, "result");
      return;
    }
    case "graph.rmsnorm": {
      const rows = requirePositiveU32(operation, values, "rows");
      const cols = requirePositiveU32(operation, values, "cols");
      if (numberAttribute(operation, values, "eps") < 0) fail(operation, "eps must be nonnegative");
      requireShape(operation, inputs[0]!, [rows, cols], "x");
      requireShape(operation, inputs[1]!, [cols], "weight");
      requireShape(operation, outputs[0]!, [rows, cols], "result");
      return;
    }
    case "loss.mse":
      requireSame(operation, [inputs[0]!, inputs[1]!]);
      requireShape(operation, outputs[0]!, [], "result");
      return;
    case "loss.softmax_cross_entropy": {
      const rows = requirePositiveU32(operation, values, "rows");
      const cols = requirePositiveU32(operation, values, "cols");
      checkedProduct(operation, [rows, cols], MAX_U32);
      requireShape(operation, inputs[0]!, [rows, cols], "logits");
      requireShape(operation, inputs[1]!, [rows, cols], "target");
      requireShape(operation, outputs[0]!, [], "result");
      return;
    }
    case "loss.topk_knowledge_distillation": {
      const rows = requirePositiveU32(operation, values, "rows");
      const cols = requirePositiveU32(operation, values, "cols");
      const k = requirePositiveU32(operation, values, "k");
      if (k > cols) fail(operation, "k must not exceed cols");
      checkedProduct(operation, [rows, cols], MAX_U32);
      checkedProduct(operation, [rows, k], MAX_U32);
      requireShape(operation, inputs[0]!, [rows, cols], "logits");
      requireShape(operation, inputs[1]!, [rows, k], "indices");
      requireShape(operation, inputs[2]!, [rows, k], "probabilities");
      requireShape(operation, outputs[0]!, [], "result");
      return;
    }
    case "graph.bias": {
      const rows = requirePositiveU32(operation, values, "rows");
      const cols = requirePositiveU32(operation, values, "cols");
      requireShape(operation, inputs[0]!, [rows, cols], "x");
      requireShape(operation, inputs[1]!, [cols], "bias");
      requireShape(operation, outputs[0]!, [rows, cols], "result");
      return;
    }
    case "graph.transpose": {
      const rows = requirePositiveU32(operation, values, "rows");
      const cols = requirePositiveU32(operation, values, "cols");
      requireShape(operation, inputs[0]!, [rows, cols], "x");
      requireShape(operation, outputs[0]!, [cols, rows], "result");
      return;
    }
    case "graph.slice_cols": {
      const rows = requirePositiveU32(operation, values, "rows");
      const cols = requirePositiveU32(operation, values, "cols");
      const start = requireNonnegativeU32(operation, values, "start");
      const len = requirePositiveU32(operation, values, "len");
      if (start + len > cols) fail(operation, "slice exceeds columns");
      requireShape(operation, inputs[0]!, [rows, cols], "x");
      requireShape(operation, outputs[0]!, [rows, len], "result");
      return;
    }
    case "graph.dense_matmul": {
      const m = requirePositiveU32(operation, values, "m");
      const n = requirePositiveU32(operation, values, "n");
      const k = requirePositiveU32(operation, values, "k");
      checkedProduct(operation, [m, n], MAX_U32);
      requireShape(operation, inputs[0]!, [m, k], "x");
      requireShape(operation, inputs[1]!, [n, k], "weight");
      requireShape(operation, outputs[0]!, [m, n], "result");
      return;
    }
    case "graph.ternary_matmul": {
      const m = requirePositiveU32(operation, values, "m");
      const n = requirePositiveU32(operation, values, "n");
      const k = requirePositiveU32(operation, values, "k");
      checkedProduct(operation, [m, n], MAX_U32);
      requireShape(operation, inputs[0]!, [m, k], "activation");
      requireShape(operation, inputs[1]!, [n, k], "weight");
      requireShape(operation, inputs[2]!, [n], "scale");
      requireShape(operation, outputs[0]!, [m, n], "result");
      return;
    }
    case "graph.concat_cols": {
      const rows = requirePositiveU32(operation, values, "rows");
      const lens = listAttribute(operation, values, "lens");
      if (lens.length !== inputs.length || lens.length === 0 || lens.some((length) => length <= 0)) {
        fail(operation, "lens must match nonempty input parts");
      }
      const total = checkedAdd(operation, ...lens);
      if (total > MAX_U32) fail(operation, "concatenated columns exceed u32");
      inputs.forEach((tensor, index) => requireShape(operation, tensor, [rows, lens[index]!], `part.${index}`));
      requireShape(operation, outputs[0]!, [rows, total], "result");
      return;
    }
    case "graph.embedding_gather": {
      const vocab = requirePositiveU32(operation, values, "vocab");
      const width = requirePositiveU32(operation, values, "n_embd");
      requireShape(operation, inputs[0]!, [vocab, width], "weight");
      if (inputs[1]!.shape.length !== 1) fail(operation, "tokens must be rank one");
      const sequence = inputs[1]!.shape[0]!;
      requireShape(operation, outputs[0]!, [sequence, width], "result");
      return;
    }
    case "graph.ste_surrogate":
    case "graph.lsq_ste": {
      const rows = requirePositiveU32(operation, values, "rows");
      const cols = requirePositiveU32(operation, values, "cols");
      requireShape(operation, inputs[0]!, [rows, cols], "weight");
      requireShape(operation, inputs[1]!, [rows], operation.operation === "graph.lsq_ste" ? "alpha" : "scale");
      requireShape(operation, outputs[0]!, [rows, cols], "result");
      return;
    }
    case "graph.attention": {
      const seq = requirePositiveU32(operation, values, "seq");
      const nHead = requirePositiveU32(operation, values, "n_head");
      const nKvHead = requirePositiveU32(operation, values, "n_kv_head");
      const headDim = requirePositiveU32(operation, values, "head_dim");
      if (nHead % nKvHead !== 0) fail(operation, "n_kv_head must divide n_head");
      const query = checkedProduct(operation, [seq, nHead, headDim], MAX_U32);
      const kv = checkedProduct(operation, [seq, nKvHead, headDim], MAX_U32);
      const probability = checkedProduct(operation, [seq, seq], MAX_U32);
      if (checkedAdd(operation, query, kv, kv, probability, probability) * 4 > MAX_SCRATCH_BYTES) {
        fail(operation, "attention scratch exceeds 64 MiB");
      }
      const queryShape = [seq, nHead, headDim];
      const kvShape = [seq, nKvHead, headDim];
      requireShape(operation, inputs[0]!, queryShape, "q");
      requireShape(operation, inputs[1]!, kvShape, "k");
      requireShape(operation, inputs[2]!, kvShape, "v");
      requireShape(operation, outputs[0]!, queryShape, "result");
      return;
    }
    case "graph.conv1d": {
      const batch = requirePositiveU32(operation, values, "batch");
      const cIn = requirePositiveU32(operation, values, "c_in");
      const cOut = requirePositiveU32(operation, values, "c_out");
      const inputLen = requirePositiveU32(operation, values, "l_in");
      const kernel = requirePositiveU32(operation, values, "k");
      const stride = requirePositiveU32(operation, values, "stride");
      const dilation = requirePositiveU32(operation, values, "dilation");
      const padLeft = requireNonnegativeU32(operation, values, "pad_left");
      const padRight = requireNonnegativeU32(operation, values, "pad_right");
      const groups = requirePositiveU32(operation, values, "groups");
      if (cIn % groups !== 0 || cOut % groups !== 0) fail(operation, "groups must divide channels");
      const outputLen = convOutputAxis(operation, inputLen, kernel, stride, dilation, padLeft, padRight);
      const maximumPosition = checkedAdd(
        operation,
        checkedProduct(operation, [outputLen - 1, stride]),
        checkedProduct(operation, [kernel - 1, dilation]),
      );
      if (maximumPosition > 0x7fff_ffff || padLeft > 0x7fff_ffff) {
        fail(operation, "conv1d indexing exceeds i32");
      }
      checkedProduct(operation, [batch, cIn, inputLen], MAX_U32);
      checkedProduct(operation, [cOut, cIn / groups, kernel], MAX_U32);
      checkedProduct(operation, [batch, cOut, outputLen], MAX_U32);
      const inputElements = batch * cIn * inputLen;
      const patch = (cIn / groups) * kernel;
      const weightElements = cOut * patch;
      const columns = outputLen * patch;
      const groupOutput = outputLen * (cOut / groups);
      const forwardScratch = checkedAdd(
        operation,
        checkedProduct(operation, [batch, cOut, outputLen]),
        columns,
        groupOutput,
      );
      const vjpScratch = checkedAdd(
        operation,
        inputElements,
        weightElements,
        cOut,
        columns,
        groupOutput,
        columns,
        weightElements / groups,
        cOut / groups,
      );
      if (Math.max(forwardScratch, vjpScratch) * 4 > MAX_SCRATCH_BYTES) {
        fail(operation, "conv1d scratch exceeds 64 MiB");
      }
      requireShape(operation, inputs[0]!, [batch, cIn, inputLen], "x");
      requireShape(operation, inputs[1]!, [cOut, cIn / groups, kernel], "weight");
      requireShape(operation, inputs[2]!, [cOut], "scale");
      requireShape(operation, outputs[0]!, [batch, cOut, outputLen], "result");
      return;
    }
    case "graph.conv2d": {
      const names = [
        "batch", "c_in", "c_out", "input_h", "input_w", "kernel_h", "kernel_w",
        "stride_h", "stride_w", "dilation_h", "dilation_w", "groups",
      ] as const;
      const dimensions = Object.fromEntries(
        names.map((name) => [name, requirePositiveU32(operation, values, name)]),
      ) as Record<(typeof names)[number], number>;
      const padTop = requireNonnegativeU32(operation, values, "pad_top");
      const padBottom = requireNonnegativeU32(operation, values, "pad_bottom");
      const padLeft = requireNonnegativeU32(operation, values, "pad_left");
      const padRight = requireNonnegativeU32(operation, values, "pad_right");
      if (dimensions.c_in % dimensions.groups !== 0 || dimensions.c_out % dimensions.groups !== 0) {
        fail(operation, "groups must divide channels");
      }
      const outputH = convOutputAxis(operation, dimensions.input_h, dimensions.kernel_h, dimensions.stride_h, dimensions.dilation_h, padTop, padBottom);
      const outputW = convOutputAxis(operation, dimensions.input_w, dimensions.kernel_w, dimensions.stride_w, dimensions.dilation_w, padLeft, padRight);
      checkedProduct(operation, [dimensions.batch, dimensions.c_in, dimensions.input_h, dimensions.input_w], MAX_U32);
      checkedProduct(operation, [dimensions.c_out, dimensions.c_in / dimensions.groups, dimensions.kernel_h, dimensions.kernel_w], MAX_U32);
      checkedProduct(operation, [dimensions.batch, dimensions.c_out, outputH, outputW], MAX_U32);
      const tileRows = Math.min(outputH * outputW, 32);
      const patch = (dimensions.c_in / dimensions.groups) * dimensions.kernel_h * dimensions.kernel_w;
      const groupChannels = dimensions.c_out / dimensions.groups;
      const columns = tileRows * patch;
      const groupOutput = tileRows * groupChannels;
      const outputElements = checkedProduct(
        operation,
        [dimensions.batch, dimensions.c_out, outputH, outputW],
      );
      const forwardScratch = checkedAdd(operation, outputElements, columns, groupOutput);
      const vjpScratch = checkedAdd(
        operation,
        dimensions.batch * dimensions.c_in * dimensions.input_h * dimensions.input_w,
        dimensions.c_out * patch,
        dimensions.c_out,
        columns,
        groupOutput,
        columns,
        groupChannels * patch,
        groupChannels,
      );
      if (Math.max(forwardScratch, vjpScratch) * 4 > MAX_SCRATCH_BYTES) {
        fail(operation, "conv2d scratch exceeds 64 MiB");
      }
      requireShape(operation, inputs[0]!, [dimensions.batch, dimensions.c_in, dimensions.input_h, dimensions.input_w], "x");
      requireShape(operation, inputs[1]!, [dimensions.c_out, dimensions.c_in / dimensions.groups, dimensions.kernel_h, dimensions.kernel_w], "weight");
      requireShape(operation, inputs[2]!, [dimensions.c_out], "scale");
      requireShape(operation, outputs[0]!, [dimensions.batch, dimensions.c_out, outputH, outputW], "result");
      return;
    }
    case "optimizer.sgd":
      if (numberAttribute(operation, values, "step") !== 0) fail(operation, "recipe step must start at zero");
      if (numberAttribute(operation, values, "lr") < 0) fail(operation, "lr must be nonnegative");
      requireSame(operation, [inputs[0]!, inputs[1]!, outputs[0]!]);
      return;
    case "optimizer.adamw":
    case "optimizer.cautious_adamw":
      validateAdamAttributes(operation, values);
      requireSame(operation, [...inputs, ...outputs]);
      return;
    case "optimizer.int8_adamw": {
      validateAdamAttributes(operation, values);
      requireSame(operation, [inputs[0]!, inputs[1]!, outputs[0]!]);
      const elements = checkedProduct(operation, inputs[0]!.shape);
      const blocks = Math.ceil(elements / 256);
      for (const tensor of [inputs[2]!, inputs[3]!, outputs[1]!, outputs[2]!]) {
        requireShape(operation, tensor, [elements], tensor.id);
      }
      for (const tensor of [inputs[4]!, inputs[5]!, outputs[3]!, outputs[4]!]) {
        requireShape(operation, tensor, [blocks], tensor.id);
      }
      return;
    }
    case "optimizer.muon": {
      if (numberAttribute(operation, values, "step") !== 0) fail(operation, "recipe step must start at zero");
      if (numberAttribute(operation, values, "lr") < 0) fail(operation, "lr must be nonnegative");
      const momentum = numberAttribute(operation, values, "momentum");
      if (momentum < 0 || momentum >= 1) fail(operation, "momentum must be in [0,1)");
      if (numberAttribute(operation, values, "weight_decay") < 0) fail(operation, "weight_decay must be nonnegative");
      const rows = requirePositiveU32(operation, values, "rows");
      const cols = requirePositiveU32(operation, values, "cols");
      const steps = requirePositiveU32(operation, values, "ns_steps");
      if (steps > 32) fail(operation, "ns_steps exceeds 32");
      checkedProduct(operation, [rows, cols], MAX_U32);
      requireSame(operation, [...inputs, ...outputs]);
      requireShape(operation, inputs[0]!, [rows, cols], "parameter");
      return;
    }
    default:
      fail(operation, `operation ${operation.operation} has no geometry rule`);
  }
}

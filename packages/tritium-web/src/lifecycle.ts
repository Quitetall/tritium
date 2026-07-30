import { TRAINING_VECTOR_DIGEST_V2 } from "./identity.ts";
import type {
  PortableAdamLeafV1,
  PortableCheckpointOptimizerV1,
  PortableCheckpointStateV1,
  PortableInt8AdamLeafV1,
  PortableMuonLeafV1,
} from "./lifecycle-types.js";
import type {
  PortableAttributeV1,
  PortableBufferV1,
  PortableTrainingRequestV1,
} from "./portable.js";
export type {
  PortableAdamLeafV1,
  PortableCheckpointOptimizerV1,
  PortableCheckpointStateV1,
  PortableInt8AdamLeafV1,
  PortableMuonLeafV1,
  PortableSgdLeafV1,
} from "./lifecycle-types.js";

const MAX_BUFFER_BYTES = 8 * 1024 * 1024;
const MAX_CALLER_BYTES = 64 * 1024 * 1024;
const MAX_REQUEST_JSON_BYTES = 8 * 1024 * 1024;
const INT8_ADAM_BLOCK = 256;
const UTF8 = new TextEncoder();

export class PortableLifecyclePlanError extends Error {
  readonly code: "invalid_schema" | "capacity";

  constructor(code: "invalid_schema" | "capacity", message: string) {
    super(message);
    this.name = "PortableLifecyclePlanError";
    this.code = code;
  }
}

function fail(
  code: "invalid_schema" | "capacity",
  message: string,
): never {
  throw new PortableLifecyclePlanError(code, message);
}

function exactKeys(value: object, expected: readonly string[], name: string): void {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (
    actual.length !== wanted.length ||
    actual.some((key, index) => key !== wanted[index])
  ) {
    fail("invalid_schema", `${name} fields do not match its optimizer schema`);
  }
}

function checkedArray(
  value: readonly number[] | Uint8Array | Uint32Array,
  maximum: number,
  name: string,
  maximumItems: number,
): readonly number[] | Uint8Array | Uint32Array {
  if (!Array.isArray(value) && !(value instanceof Uint8Array) && !(value instanceof Uint32Array)) {
    fail("invalid_schema", `${name} must be an unsigned integer array`);
  }
  if (value.length > maximumItems) {
    fail("capacity", `${name} exceeds the portable 8 MiB buffer limit`);
  }
  for (let index = 0; index < value.length; index += 1) {
    const item = value[index];
    if (
      !(index in value) ||
      typeof item !== "number" ||
      !Number.isSafeInteger(item) ||
      item < 0 ||
      item > maximum
    ) {
      fail("invalid_schema", `${name} must contain bounded unsigned integers`);
    }
  }
  return value;
}

function f32Buffer(
  name: string,
  bits: readonly number[] | Uint8Array | Uint32Array,
): PortableBufferV1 {
  const values = Array.from(
    checkedArray(bits, 0xffff_ffff, name, MAX_BUFFER_BYTES / 4),
  );
  return { name, shape: [values.length], data: { dtype: "f32", bits: values } };
}

function byteBuffer(
  name: string,
  input: readonly number[] | Uint8Array | Uint32Array,
): PortableBufferV1 {
  const values = Array.from(checkedArray(input, 0xff, name, MAX_BUFFER_BYTES));
  return { name, shape: [values.length], data: { dtype: "bytes", values } };
}

function optimizerAttributes(
  optimizer: PortableCheckpointOptimizerV1,
  leafLengths: readonly number[],
  step?: number,
): PortableAttributeV1[] {
  const attributes: PortableAttributeV1[] = [
    { kind: "text", name: "optimizer", value: optimizer },
  ];
  if (step !== undefined) attributes.push({ kind: "u64", name: "step", value: step });
  attributes.push({ kind: "u64-list", name: "leaf_lens", values: [...leafLengths] });
  return attributes;
}

function checkpointSize(
  optimizer: PortableCheckpointOptimizerV1,
  leafLengths: readonly number[],
): number {
  let bytes = 17;
  for (const length of leafLengths) {
    const stateBytes =
      optimizer === "sgd"
        ? 0
        : optimizer === "adamw" || optimizer === "cautious_adamw"
          ? length * 8
          : optimizer === "int8_adamw"
            ? length * 2 + Math.ceil(length / INT8_ADAM_BLOCK) * 8
            : length * 4;
    bytes += 8 + length * 4 + stateBytes;
    if (!Number.isSafeInteger(bytes) || bytes > MAX_BUFFER_BYTES) {
      fail("capacity", "encoded checkpoint exceeds the portable 8 MiB buffer limit");
    }
  }
  return bytes;
}

function boundedLifecyclePlaneJsonBytes(
  elements: number,
  dtype: "f32" | "bytes",
  input: boolean,
): number {
  const digits = input ? (dtype === "bytes" ? 3 : 10) : 1;
  const bytes = 256 + 1 + elements * (digits + 1);
  if (!Number.isSafeInteger(bytes)) fail("capacity", "lifecycle JSON size overflowed");
  return bytes;
}

/** Allocation-free capacity admission for checkpoint and resume requests. */
export function preflightPortableLifecycleLayout(
  optimizer: PortableCheckpointOptimizerV1,
  leafLengths: readonly number[],
): void {
  if (![
    "sgd",
    "adamw",
    "cautious_adamw",
    "int8_adamw",
    "muon",
  ].includes(optimizer)) {
    fail("invalid_schema", "optimizer is not supported by portable checkpoints");
  }
  validateLeafLengths(leafLengths);
  const planes =
    optimizer === "sgd"
      ? 1
      : optimizer === "adamw" || optimizer === "cautious_adamw"
        ? 3
        : optimizer === "int8_adamw"
          ? 5
          : 2;
  if (leafLengths.length * planes > 64) {
    fail("capacity", "checkpoint inputs exceed 64 buffers");
  }
  if (1 + leafLengths.length * planes > 64) {
    fail("capacity", "resume outputs exceed 64 buffers");
  }
  const encodedBytes = checkpointSize(optimizer, leafLengths);
  let callerBytes = encodedBytes + 8;
  let checkpointJsonBytes =
    4096 +
    leafLengths.length * 12 +
    boundedLifecyclePlaneJsonBytes(encodedBytes, "bytes", false);
  let resumeJsonBytes =
    4096 +
    leafLengths.length * 12 +
    boundedLifecyclePlaneJsonBytes(encodedBytes, "bytes", true) +
    boundedLifecyclePlaneJsonBytes(8, "bytes", false);
  for (const length of leafLengths) {
    const blocks = Math.ceil(length / INT8_ADAM_BLOCK);
    const stateBytes =
      optimizer === "sgd"
        ? 0
        : optimizer === "adamw" || optimizer === "cautious_adamw"
          ? length * 8
          : optimizer === "int8_adamw"
            ? length * 2 + blocks * 8
            : length * 4;
    callerBytes = checkedCallerBytes(callerBytes, length * 4 + stateBytes);
    checkpointJsonBytes += boundedLifecyclePlaneJsonBytes(length, "f32", true);
    resumeJsonBytes += boundedLifecyclePlaneJsonBytes(length, "f32", false);
    if (optimizer === "adamw" || optimizer === "cautious_adamw") {
      checkpointJsonBytes += 2 * boundedLifecyclePlaneJsonBytes(length, "f32", true);
      resumeJsonBytes += 2 * boundedLifecyclePlaneJsonBytes(length, "f32", false);
    } else if (optimizer === "int8_adamw") {
      checkpointJsonBytes += 2 * boundedLifecyclePlaneJsonBytes(length, "bytes", true);
      checkpointJsonBytes += 2 * boundedLifecyclePlaneJsonBytes(blocks, "f32", true);
      resumeJsonBytes += 2 * boundedLifecyclePlaneJsonBytes(length, "bytes", false);
      resumeJsonBytes += 2 * boundedLifecyclePlaneJsonBytes(blocks, "f32", false);
    } else if (optimizer === "muon") {
      checkpointJsonBytes += boundedLifecyclePlaneJsonBytes(length, "f32", true);
      resumeJsonBytes += boundedLifecyclePlaneJsonBytes(length, "f32", false);
    }
  }
  if (
    !Number.isSafeInteger(checkpointJsonBytes) ||
    !Number.isSafeInteger(resumeJsonBytes) ||
    checkpointJsonBytes > MAX_REQUEST_JSON_BYTES ||
    resumeJsonBytes > MAX_REQUEST_JSON_BYTES
  ) {
    fail("capacity", "lifecycle request may exceed 8 MiB");
  }
}

function validateLeafLengths(values: readonly number[]): readonly number[] {
  checkedArray(values, 0xffff_ffff, "leafLengths", 1024);
  if (values.length === 0) fail("invalid_schema", "at least one leaf is required");
  if (values.some((value) => value === 0)) {
    fail("invalid_schema", "leaf lengths must be positive");
  }
  if (values.some((value) => value > MAX_BUFFER_BYTES / 4)) {
    fail("capacity", "a resume parameter plane exceeds the portable 8 MiB limit");
  }
  return values;
}

function checkedCallerBytes(current: number, added: number): number {
  const next = current + added;
  if (!Number.isSafeInteger(next) || next > MAX_CALLER_BYTES) {
    fail("capacity", "lifecycle caller buffers exceed 64 MiB");
  }
  return next;
}

function baseRequest(
  physicalDevice: string,
  operation: PortableTrainingRequestV1["operation"],
  execution: PortableTrainingRequestV1["execution"],
  inputs: readonly PortableBufferV1[],
  attributes: readonly PortableAttributeV1[],
  outputs: readonly PortableBufferV1[],
): PortableTrainingRequestV1 {
  const request: PortableTrainingRequestV1 = {
    schemaId: "tritium.portable_training_request",
    schemaVersion: 1,
    physicalDevice,
    operation,
    execution,
    vectorDigest: TRAINING_VECTOR_DIGEST_V2,
    inputs,
    attributes,
    outputs,
  };
  if (UTF8.encode(JSON.stringify(request)).byteLength > MAX_REQUEST_JSON_BYTES) {
    fail("capacity", "lifecycle request JSON exceeds 8 MiB");
  }
  return request;
}

/** Compile typed optimizer state into the canonical lifecycle.checkpoint ABI. */
export function compilePortableCheckpointRequest(
  state: PortableCheckpointStateV1,
  physicalDevice = "wasm32:browser",
): PortableTrainingRequestV1 {
  if (typeof state !== "object" || state === null || Array.isArray(state)) {
    fail("invalid_schema", "checkpoint state must be an object");
  }
  exactKeys(state, ["leaves", "optimizer", "step"], "checkpoint state");
  if (![
    "sgd",
    "adamw",
    "cautious_adamw",
    "int8_adamw",
    "muon",
  ].includes(state.optimizer)) {
    fail("invalid_schema", "optimizer is not supported by portable checkpoints");
  }
  if (!Number.isSafeInteger(state.step) || state.step < 0) {
    fail("invalid_schema", "step must be a non-negative JavaScript safe integer");
  }
  if (!Array.isArray(state.leaves) || state.leaves.length === 0) {
    fail("invalid_schema", "at least one checkpoint leaf is required");
  }
  const buffersPerLeaf =
    state.optimizer === "sgd"
      ? 1
      : state.optimizer === "adamw" ||
          state.optimizer === "cautious_adamw"
        ? 3
        : state.optimizer === "int8_adamw"
          ? 5
          : 2;
  if (state.leaves.length > Math.floor(64 / buffersPerLeaf)) {
    fail("capacity", "checkpoint inputs exceed 64 buffers");
  }
  const inputs: PortableBufferV1[] = [];
  let callerBytes = 0;
  const leafLengths = state.leaves.map((leaf, index) => {
    if (typeof leaf !== "object" || leaf === null || Array.isArray(leaf)) {
      fail("invalid_schema", `leaf ${index} must be an object`);
    }
    const parameter = checkedArray(
      leaf.parameter,
      0xffff_ffff,
      `parameter.${index}`,
      MAX_BUFFER_BYTES / 4,
    );
    if (parameter.length === 0) fail("invalid_schema", "parameter leaves must be nonempty");
    callerBytes = checkedCallerBytes(callerBytes, parameter.length * 4);
    inputs.push(f32Buffer(`parameter.${index}`, parameter));
    if (state.optimizer === "sgd") {
      exactKeys(leaf, ["parameter"], `leaf ${index}`);
    } else if (state.optimizer === "adamw" || state.optimizer === "cautious_adamw") {
      const adam = leaf as PortableAdamLeafV1;
      exactKeys(adam, ["moment1", "moment2", "parameter"], `leaf ${index}`);
      const moment1 = checkedArray(
        adam.moment1,
        0xffff_ffff,
        `moment1.${index}`,
        MAX_BUFFER_BYTES / 4,
      );
      const moment2 = checkedArray(
        adam.moment2,
        0xffff_ffff,
        `moment2.${index}`,
        MAX_BUFFER_BYTES / 4,
      );
      if (moment1.length !== parameter.length || moment2.length !== parameter.length) {
        fail("invalid_schema", `Adam leaf ${index} plane lengths differ`);
      }
      callerBytes = checkedCallerBytes(callerBytes, moment1.length * 4);
      callerBytes = checkedCallerBytes(callerBytes, moment2.length * 4);
      inputs.push(f32Buffer(`moment1.${index}`, moment1));
      inputs.push(f32Buffer(`moment2.${index}`, moment2));
    } else if (state.optimizer === "int8_adamw") {
      const adam = leaf as PortableInt8AdamLeafV1;
      exactKeys(
        adam,
        ["moment1Q8", "moment1Scale", "moment2Q8", "moment2Scale", "parameter"],
        `leaf ${index}`,
      );
      const moment1Q8 = checkedArray(
        adam.moment1Q8,
        0xff,
        `moment1Q8.${index}`,
        MAX_BUFFER_BYTES,
      );
      const moment2Q8 = checkedArray(
        adam.moment2Q8,
        0xff,
        `moment2Q8.${index}`,
        MAX_BUFFER_BYTES,
      );
      const moment1Scale = checkedArray(
        adam.moment1Scale,
        0xffff_ffff,
        `moment1Scale.${index}`,
        MAX_BUFFER_BYTES / 4,
      );
      const moment2Scale = checkedArray(
        adam.moment2Scale,
        0xffff_ffff,
        `moment2Scale.${index}`,
        MAX_BUFFER_BYTES / 4,
      );
      const blocks = Math.ceil(parameter.length / INT8_ADAM_BLOCK);
      if (
        moment1Q8.length !== parameter.length ||
        moment2Q8.length !== parameter.length ||
        moment1Scale.length !== blocks ||
        moment2Scale.length !== blocks
      ) {
        fail("invalid_schema", `int8 Adam leaf ${index} plane lengths differ`);
      }
      callerBytes = checkedCallerBytes(callerBytes, moment1Q8.length);
      callerBytes = checkedCallerBytes(callerBytes, moment2Q8.length);
      callerBytes = checkedCallerBytes(callerBytes, moment1Scale.length * 4);
      callerBytes = checkedCallerBytes(callerBytes, moment2Scale.length * 4);
      inputs.push(byteBuffer(`moment1_q8.${index}`, moment1Q8));
      inputs.push(byteBuffer(`moment2_q8.${index}`, moment2Q8));
      inputs.push(f32Buffer(`moment1_scale.${index}`, moment1Scale));
      inputs.push(f32Buffer(`moment2_scale.${index}`, moment2Scale));
    } else {
      const muon = leaf as PortableMuonLeafV1;
      exactKeys(muon, ["momentum", "parameter"], `leaf ${index}`);
      const momentum = checkedArray(
        muon.momentum,
        0xffff_ffff,
        `momentum.${index}`,
        MAX_BUFFER_BYTES / 4,
      );
      if (momentum.length !== parameter.length) {
        fail("invalid_schema", `Muon leaf ${index} plane lengths differ`);
      }
      callerBytes = checkedCallerBytes(callerBytes, momentum.length * 4);
      inputs.push(f32Buffer(`momentum.${index}`, momentum));
    }
    return parameter.length;
  });
  const bytes = checkpointSize(state.optimizer, leafLengths);
  checkedCallerBytes(callerBytes, bytes);
  return baseRequest(
    physicalDevice,
    "lifecycle.checkpoint",
    "checkpoint",
    inputs,
    optimizerAttributes(state.optimizer, leafLengths, state.step),
    [byteBuffer("checkpoint", new Uint8Array(bytes))],
  );
}

/** Compile canonical checkpoint bytes into an atomic lifecycle.resume ABI. */
export function compilePortableResumeRequest(
  optimizer: PortableCheckpointOptimizerV1,
  leafLengths: readonly number[],
  checkpoint: Uint8Array,
  physicalDevice = "wasm32:browser",
): PortableTrainingRequestV1 {
  if (![
    "sgd",
    "adamw",
    "cautious_adamw",
    "int8_adamw",
    "muon",
  ].includes(optimizer)) {
    fail("invalid_schema", "optimizer is not supported by portable checkpoints");
  }
  validateLeafLengths(leafLengths);
  if (!(checkpoint instanceof Uint8Array) || checkpoint.byteLength === 0) {
    fail("invalid_schema", "checkpoint must be a nonempty Uint8Array");
  }
  if (checkpoint.byteLength > MAX_BUFFER_BYTES) {
    fail("capacity", "checkpoint exceeds the portable 8 MiB buffer limit");
  }
  const planes =
    optimizer === "sgd"
      ? 1
      : optimizer === "adamw" || optimizer === "cautious_adamw"
        ? 3
        : optimizer === "int8_adamw"
          ? 5
          : 2;
  if (1 + leafLengths.length * planes > 64) {
    fail("capacity", "resume outputs exceed 64 buffers");
  }
  let callerBytes = checkpoint.byteLength + 8;
  for (const length of leafLengths) {
    const blocks = Math.ceil(length / INT8_ADAM_BLOCK);
    const stateBytes =
      optimizer === "sgd"
        ? 0
        : optimizer === "adamw" || optimizer === "cautious_adamw"
          ? length * 8
          : optimizer === "int8_adamw"
            ? length * 2 + blocks * 8
            : length * 4;
    callerBytes = checkedCallerBytes(callerBytes, length * 4 + stateBytes);
  }
  const outputs: PortableBufferV1[] = [byteBuffer("step", new Uint8Array(8))];
  for (const [index, length] of leafLengths.entries()) {
    const blocks = Math.ceil(length / INT8_ADAM_BLOCK);
    outputs.push(f32Buffer(`parameter.${index}`, new Uint32Array(length)));
    if (optimizer === "adamw" || optimizer === "cautious_adamw") {
      outputs.push(f32Buffer(`moment1.${index}`, new Uint32Array(length)));
      outputs.push(f32Buffer(`moment2.${index}`, new Uint32Array(length)));
    } else if (optimizer === "int8_adamw") {
      outputs.push(byteBuffer(`moment1_q8.${index}`, new Uint8Array(length)));
      outputs.push(byteBuffer(`moment2_q8.${index}`, new Uint8Array(length)));
      outputs.push(f32Buffer(`moment1_scale.${index}`, new Uint32Array(blocks)));
      outputs.push(f32Buffer(`moment2_scale.${index}`, new Uint32Array(blocks)));
    } else if (optimizer === "muon") {
      outputs.push(f32Buffer(`momentum.${index}`, new Uint32Array(length)));
    }
  }
  return baseRequest(
    physicalDevice,
    "lifecycle.resume",
    "resume",
    [byteBuffer("checkpoint", checkpoint)],
    optimizerAttributes(optimizer, leafLengths),
    outputs,
  );
}

/** Compile strict SALT V2 package validation and byte-identical export. */
export function compilePortableExportRequest(
  packageBytes: Uint8Array,
  physicalDevice = "wasm32:browser",
): PortableTrainingRequestV1 {
  if (!(packageBytes instanceof Uint8Array) || packageBytes.byteLength === 0) {
    fail("invalid_schema", "package must be a nonempty Uint8Array");
  }
  if (packageBytes.byteLength > MAX_BUFFER_BYTES) {
    fail("capacity", "package exceeds the portable 8 MiB buffer limit");
  }
  checkedCallerBytes(packageBytes.byteLength, packageBytes.byteLength);
  return baseRequest(
    physicalDevice,
    "lifecycle.export",
    "export",
    [byteBuffer("package", packageBytes)],
    [{ kind: "text", name: "format", value: "salt_v2_package_v1" }],
    [byteBuffer("artifact", new Uint8Array(packageBytes.byteLength))],
  );
}

/** Compile strict SALT V2 artifact admission and byte-identical reload. */
export function compilePortableReloadRequest(
  artifact: Uint8Array,
  physicalDevice = "wasm32:browser",
): PortableTrainingRequestV1 {
  if (!(artifact instanceof Uint8Array) || artifact.byteLength === 0) {
    fail("invalid_schema", "artifact must be a nonempty Uint8Array");
  }
  if (artifact.byteLength > MAX_BUFFER_BYTES) {
    fail("capacity", "artifact exceeds the portable 8 MiB buffer limit");
  }
  checkedCallerBytes(artifact.byteLength, artifact.byteLength);
  return baseRequest(
    physicalDevice,
    "lifecycle.reload",
    "reload",
    [byteBuffer("artifact", artifact)],
    [{ kind: "text", name: "format", value: "salt_v2_package_v1" }],
    [byteBuffer("package", new Uint8Array(artifact.byteLength))],
  );
}

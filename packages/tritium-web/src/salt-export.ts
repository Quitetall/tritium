import type { PortableScheduleTensorV1 } from "./portable-schedule-types.js";
import type {
  CompiledTrainingBufferV1,
  CompiledTrainingPlanV1,
  TrainingAttributeSpecV1,
} from "./session.ts";

const TILE = 256;
const GROUP = 128;
const MAX_PLANES = 3;
const MAX_PACKAGE_BYTES = 8 * 1024 * 1024;
const UTF8 = new TextEncoder();
const UTF8_FATAL = new TextDecoder("utf-8", { fatal: true });
const F32_BITS = new DataView(new ArrayBuffer(4));

export type SaltExportErrorCode = "invalid_schema" | "invalid_state" | "capacity";

export class SaltExportError extends Error {
  readonly code: SaltExportErrorCode;

  constructor(code: SaltExportErrorCode, message: string) {
    super(message);
    this.name = "SaltExportError";
    this.code = code;
  }
}

export interface CompiledSaltExportTargetV1 {
  readonly name: string;
  readonly ownerId: string;
  readonly dims: readonly number[];
  readonly rows: number;
  readonly cols: number;
  readonly planes: number;
}

export interface SaltExportLayoutV1 {
  readonly packageBytes: number;
  readonly semanticBytes: number;
  readonly maxFitScratchBytes: number;
}

function fail(code: SaltExportErrorCode, message: string): never {
  throw new SaltExportError(code, message);
}

function u64Attribute(
  attributes: readonly TrainingAttributeSpecV1[],
  name: string,
): number {
  const attribute = attributes.find((candidate) => candidate.name === name);
  if (
    attribute?.kind !== "u64" ||
    typeof attribute.value !== "number" ||
    !Number.isSafeInteger(attribute.value)
  ) {
    fail("invalid_schema", `SALT export target has no canonical ${name}`);
  }
  return attribute.value;
}

function checkedAdd(left: number, right: number): number {
  const value = left + right;
  if (!Number.isSafeInteger(value)) fail("capacity", "SALT export size overflowed");
  return value;
}

function checkedMultiply(left: number, right: number): number {
  const value = left * right;
  if (!Number.isSafeInteger(value)) fail("capacity", "SALT export size overflowed");
  return value;
}

function targetPhysicalBytes(target: CompiledSaltExportTargetV1): number {
  const elements = target.rows * target.cols;
  if (!Number.isSafeInteger(elements) || elements <= 0) {
    fail("invalid_schema", `SALT export target ${target.name} has invalid geometry`);
  }
  let bytes = 64 + UTF8.encode(target.name).length + target.dims.length * 8;
  for (let offset = 0; offset < elements; offset += TILE) {
    const length = Math.min(TILE, elements - offset);
    bytes = checkedAdd(bytes, Math.ceil(length / 5) * target.planes);
    bytes = checkedAdd(bytes, Math.ceil(length / GROUP) * 2 * target.planes);
  }
  return bytes;
}

/** Derive immutable SALT outputs from graph.salt_ste parameter sites. */
export function compileSaltExportTargets(
  plan: CompiledTrainingPlanV1,
  requireTarget: boolean,
): readonly CompiledSaltExportTargetV1[] {
  const buffers = new Map(plan.buffers.map((buffer) => [buffer.id, buffer] as const));
  const targets = new Map<string, CompiledSaltExportTargetV1>();
  for (const operation of plan.operations) {
    if (operation.operation !== "graph.salt_ste") continue;
    const input = buffers.get(operation.inputs[0] ?? "");
    if (input?.role !== "parameter" || input.dtype !== "f32") {
      fail("invalid_schema", `${operation.id} SALT export input must be an f32 parameter`);
    }
    const owner = buffers.get(input.ownerId);
    if (owner === undefined) {
      fail("invalid_schema", `${operation.id} SALT export owner is missing`);
    }
    const rows = u64Attribute(operation.attributes, "rows");
    const cols = u64Attribute(operation.attributes, "cols");
    const planes = u64Attribute(operation.attributes, "planes");
    if (planes < 1 || planes > MAX_PLANES) {
      fail("invalid_schema", `${operation.id} SALT export requires 1..=3 planes`);
    }
    if (rows > 1 && cols % GROUP !== 0) {
      fail(
        "invalid_schema",
        `${operation.id} row boundaries do not align to SALT group128 scales`,
      );
    }
    const candidate = Object.freeze({
      name: owner.id,
      ownerId: owner.id,
      dims: Object.freeze([...owner.shape]),
      rows,
      cols,
      planes,
    });
    const encodedName = UTF8.encode(candidate.name);
    try {
      if (UTF8_FATAL.decode(encodedName) !== candidate.name) {
        fail("invalid_schema", `SALT export target ${candidate.name} is not canonical UTF-8`);
      }
    } catch (error) {
      if (error instanceof SaltExportError) throw error;
      fail("invalid_schema", `SALT export target ${candidate.name} is not canonical UTF-8`);
    }
    const previous = targets.get(owner.id);
    if (
      previous !== undefined &&
      (previous.rows !== rows || previous.cols !== cols || previous.planes !== planes)
    ) {
      fail("invalid_schema", `parameter ${owner.id} has conflicting SALT export sites`);
    }
    targets.set(owner.id, candidate);
  }
  if (requireTarget && targets.size === 0) {
    fail("invalid_schema", "lifecycle.export requires at least one graph.salt_ste parameter");
  }
  const compiled = Object.freeze([...targets.values()]);
  if (compiled.length !== 0) {
    if (saltExportLayout(compiled).packageBytes > MAX_PACKAGE_BYTES) {
      fail("capacity", "state-derived SALT package exceeds the portable 8 MiB limit");
    }
  }
  return compiled;
}

/** Exact package and conservative direct-WASM admission geometry. */
export function saltExportLayout(
  targets: readonly CompiledSaltExportTargetV1[],
): SaltExportLayoutV1 {
  const fullTiles = targets.reduce(
    (total, target) => checkedAdd(total, Math.floor((target.rows * target.cols) / TILE)),
    0,
  );
  let packageBytes = 24 + Math.floor((fullTiles * 2) / 8);
  let semanticBytes = 0;
  let maxFitScratchBytes = 0;
  for (const target of targets) {
    const elements = checkedMultiply(target.rows, target.cols);
    packageBytes = checkedAdd(packageBytes, targetPhysicalBytes(target));
    semanticBytes = checkedAdd(
      semanticBytes,
      checkedAdd(
        checkedMultiply(elements, target.planes),
        checkedMultiply(
          checkedMultiply(Math.ceil(elements / GROUP), 2),
          target.planes,
        ),
      ),
    );
    maxFitScratchBytes = Math.max(
      maxFitScratchBytes,
      checkedMultiply(target.cols, 4),
    );
  }
  packageBytes = checkedAdd(packageBytes, (8 - (packageBytes % 8)) % 8);
  return Object.freeze({ packageBytes, semanticBytes, maxFitScratchBytes });
}

function f32Bits(value: number): number {
  F32_BITS.setFloat32(0, value, true);
  return F32_BITS.getUint32(0, true);
}

function f32ToF16(value: number): number {
  const bits = f32Bits(Math.fround(value));
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  let mantissa = bits & 0x7fffff;
  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  let halfExponent = exponent - 127 + 15;
  if (halfExponent >= 31) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    mantissa |= 0x800000;
    const shift = 14 - halfExponent;
    let rounded = mantissa >>> shift;
    const remainder = mantissa & (2 ** shift - 1);
    const halfway = 2 ** (shift - 1);
    if (remainder > halfway || (remainder === halfway && (rounded & 1) !== 0)) rounded += 1;
    return sign | rounded;
  }
  let rounded = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (rounded & 1) !== 0)) {
    rounded += 1;
    if (rounded === 0x400) {
      rounded = 0;
      halfExponent += 1;
      if (halfExponent === 31) return sign | 0x7c00;
    }
  }
  return sign | (halfExponent << 10) | rounded;
}

function f16ToF32(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return Math.fround(sign * mantissa * 2 ** -24);
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : NaN;
  return Math.fround(sign * (1 + mantissa / 1024) * 2 ** (exponent - 15));
}

interface EncodedTensor {
  readonly target: CompiledSaltExportTargetV1;
  readonly payload: Uint8Array;
  readonly scales: Uint8Array;
  readonly tileCount: number;
  readonly elements: number;
}

function encodeTensor(
  target: CompiledSaltExportTargetV1,
  parameter: Float32Array,
): EncodedTensor {
  const elements = target.rows * target.cols;
  if (parameter.length !== elements) {
    fail("invalid_state", `parameter ${target.ownerId} changed length before export`);
  }
  const tiles = Array.from({ length: Math.ceil(elements / TILE) }, (_, tile) => {
    const offset = tile * TILE;
    const length = Math.min(TILE, elements - offset);
    return {
      offset,
      length,
      payloadBytesPerPlane: Math.ceil(length / 5),
      scaleGroupsPerPlane: Math.ceil(length / GROUP),
      payloadOffset: 0,
      scalesOffset: 0,
    };
  });
  let payloadBytes = 0;
  let scaleBytes = 0;
  for (const tile of tiles) {
    tile.payloadOffset = payloadBytes;
    tile.scalesOffset = scaleBytes;
    payloadBytes += tile.payloadBytesPerPlane * target.planes;
    scaleBytes += tile.scaleGroupsPerPlane * 2 * target.planes;
  }
  // 121 is five semantic-zero radix-3 digits. Live trits adjust their place
  // directly, so no whole-tensor trit or boxed-number staging is required.
  const payload = new Uint8Array(payloadBytes);
  payload.fill(121);
  const scales = new Uint8Array(scaleBytes);
  const residual = new Float32Array(target.cols);
  for (let row = 0; row < target.rows; row += 1) {
    const start = row * target.cols;
    for (let column = 0; column < target.cols; column += 1) {
      const value = parameter[start + column]!;
      if (!Number.isFinite(value)) {
        fail("invalid_state", `parameter ${target.ownerId} contains a non-finite value`);
      }
      residual[column] = value;
    }
    for (let plane = 0; plane < target.planes; plane += 1) {
      let sum = 0;
      for (const value of residual) sum = Math.fround(sum + Math.abs(value));
      const scaleBits = f32ToF16(Math.fround(sum / target.cols));
      const scale = f16ToF32(scaleBits);
      if (!Number.isFinite(scale)) {
        fail("invalid_state", `parameter ${target.ownerId} has an unrepresentable f16 scale`);
      }
      if (sum !== 0 && scale === 0) {
        fail("invalid_state", `parameter ${target.ownerId} scale underflows f16`);
      }
      if (scale === 0) continue;
      for (let column = 0; column < target.cols; column += 1) {
        const ratio = Math.fround(residual[column]! / scale);
        const trit = ratio >= 0.5 ? 1 : ratio <= -0.5 ? -1 : 0;
        const index = start + column;
        const tile = tiles[Math.floor(index / TILE)]!;
        const local = index - tile.offset;
        const byte =
          tile.payloadOffset +
          plane * tile.payloadBytesPerPlane +
          Math.floor(local / 5);
        payload[byte] = payload[byte]! + trit * (3 ** (local % 5));
        residual[column] = Math.fround(residual[column]! - Math.fround(scale * trit));
      }
      for (let group = start; group < start + target.cols; group += GROUP) {
        const tile = tiles[Math.floor(group / TILE)]!;
        const localGroup = Math.floor((group - tile.offset) / GROUP);
        const scaleOffset =
          tile.scalesOffset +
          plane * tile.scaleGroupsPerPlane * 2 +
          localGroup * 2;
        scales[scaleOffset] = scaleBits & 0xff;
        scales[scaleOffset + 1] = scaleBits >>> 8;
      }
    }
  }
  return {
    target,
    payload,
    scales,
    tileCount: Math.ceil(elements / TILE),
    elements,
  };
}

function writeU64(view: DataView, offset: number, value: number | bigint): void {
  view.setBigUint64(offset, BigInt(value), true);
}

/** Fit current f32 parameter owners into one canonical B3 SALT V2 package. */
export function encodeStateDerivedSaltV2(
  targets: readonly CompiledSaltExportTargetV1[],
  store: Readonly<Record<string, PortableScheduleTensorV1>>,
): Uint8Array {
  if (targets.length === 0) fail("invalid_state", "training plan has no SALT export targets");
  const tensors = targets.map((target) => {
    const parameter = store[target.ownerId];
    if (!(parameter instanceof Float32Array)) {
      fail("invalid_state", `SALT export parameter ${target.ownerId} is not f32`);
    }
    return encodeTensor(target, parameter);
  });
  const fullTileValues: number[] = [];
  for (const tensor of tensors) {
    const value = tensor.target.planes === 1 ? 0 : tensor.target.planes === 2 ? 1 : 3;
    for (let tile = 0; tile < Math.floor(tensor.elements / TILE); tile += 1) {
      fullTileValues.push(value);
    }
  }
  const mapBytes = new Uint8Array(Math.floor((fullTileValues.length * 2) / 8));
  let embeddedMap = 0;
  const completeBits = mapBytes.length * 8;
  fullTileValues.forEach((value, tile) => {
    for (let bit = 0; bit < 2; bit += 1) {
      const bitIndex = tile * 2 + bit;
      if ((value & (1 << bit)) === 0) continue;
      if (bitIndex < completeBits) {
        const byteIndex = bitIndex >>> 3;
        mapBytes[byteIndex] = mapBytes[byteIndex]! | (1 << (bitIndex & 7));
      } else {
        embeddedMap |= 1 << (bitIndex - completeBits);
      }
    }
  });
  let total = 24 + mapBytes.length;
  for (const tensor of tensors) {
    total +=
      64 +
      UTF8.encode(tensor.target.name).length +
      tensor.target.dims.length * 8 +
      tensor.payload.length +
      tensor.scales.length;
  }
  total += (8 - (total % 8)) % 8;
  if (total > MAX_PACKAGE_BYTES) fail("capacity", "SALT package exceeds 8 MiB");
  const bytes = new Uint8Array(total);
  const view = new DataView(bytes.buffer);
  bytes.set(UTF8.encode("TSLT2PKG"), 0);
  view.setUint16(8, 1, true);
  view.setUint8(10, 2); // B3
  view.setUint8(11, 0);
  view.setUint32(12, tensors.length | (embeddedMap << 26), true);
  writeU64(view, 16, total);
  let cursor = 24;
  for (const tensor of tensors) {
    const name = UTF8.encode(tensor.target.name);
    view.setUint32(cursor, name.length, true);
    view.setUint32(cursor + 4, tensor.target.dims.length, true);
    writeU64(view, cursor + 8, tensor.elements);
    const raggedValue = tensor.target.planes === 1 ? 0n : tensor.target.planes === 2 ? 1n : 3n;
    const packedTileCount = BigInt(tensor.tileCount) |
      (tensor.elements % TILE === 0 ? 0n : raggedValue << 62n);
    view.setBigUint64(cursor + 16, packedTileCount, true);
    writeU64(view, cursor + 24, tensor.payload.length);
    writeU64(view, cursor + 32, tensor.scales.length);
    // transform tag, reserved bytes, seed and domain are already zero.
    cursor += 64;
    bytes.set(name, cursor);
    cursor += name.length;
    for (const dimension of tensor.target.dims) {
      writeU64(view, cursor, dimension);
      cursor += 8;
    }
    bytes.set(tensor.payload, cursor);
    cursor += tensor.payload.length;
    bytes.set(tensor.scales, cursor);
    cursor += tensor.scales.length;
  }
  bytes.set(mapBytes, cursor);
  return bytes;
}

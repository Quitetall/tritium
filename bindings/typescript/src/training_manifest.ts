/** Strict TypeScript representation of TrainingOpManifestV2. */

export type TrainingOpCategoryV1 = "graph" | "loss" | "optimizer" | "lifecycle";
export type TrainingVjpV1 = "none" | "first_order";

export interface TrainingOpDescriptorV1 {
  readonly id: string;
  readonly category: TrainingOpCategoryV1;
  readonly forward: boolean;
  readonly vjp: TrainingVjpV1;
  readonly mutates: boolean;
  readonly checkpoint_planes: readonly string[];
}

export interface TrainingOpManifestV2 {
  readonly schema_id: "tritium.training_op_manifest";
  readonly schema_version: 2;
  readonly dtype: "f32";
  readonly operations: readonly TrainingOpDescriptorV1[];
}

export interface TrainingOpManifestV1 {
  readonly schema_id: "tritium.training_op_manifest";
  readonly schema_version: 1;
  readonly dtype: "f32";
  readonly operations: readonly TrainingOpDescriptorV1[];
}

export class TrainingManifestError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TrainingManifestError";
  }
}

const GRAPH_IDS = [
  "graph.ste_surrogate",
  "graph.salt_ste",
  "graph.lsq_ste",
  "graph.fsq",
  "graph.dense_matmul",
  "graph.ternary_matmul",
  "graph.transpose",
  "graph.embedding_gather",
  "graph.slice_cols",
  "graph.concat_cols",
  "graph.detach",
  "graph.scale_const",
  "graph.bias",
  "graph.add",
  "graph.mul",
  "graph.conv1d",
  "graph.conv2d",
  "graph.relu2",
  "graph.silu",
  "graph.rmsnorm",
  "graph.softmax",
  "graph.causal_mask",
  "graph.rope",
  "graph.attention",
] as const;

const LOSS_IDS = [
  "loss.mse",
  "loss.softmax_cross_entropy",
  "loss.topk_knowledge_distillation",
] as const;
const OPTIMIZERS = [
  ["optimizer.sgd", ["parameter"]],
  ["optimizer.adamw", ["parameter", "moment1", "moment2"]],
  ["optimizer.cautious_adamw", ["parameter", "moment1", "moment2"]],
  [
    "optimizer.int8_adamw",
    ["parameter", "moment1_q8", "moment2_q8", "moment1_scale", "moment2_scale"],
  ],
  ["optimizer.muon", ["parameter", "momentum"]],
] as const;
const LIFECYCLE = [
  ["lifecycle.checkpoint", false],
  ["lifecycle.resume", true],
  ["lifecycle.export", false],
  ["lifecycle.reload", true],
] as const;

const operations: readonly TrainingOpDescriptorV1[] = Object.freeze([
  ...GRAPH_IDS.map((id) =>
    Object.freeze({
      id,
      category: "graph" as const,
      forward: true,
      vjp: "first_order" as const,
      mutates: false,
      checkpoint_planes: Object.freeze([] as string[]),
    })
  ),
  ...LOSS_IDS.map((id) =>
    Object.freeze({
      id,
      category: "loss" as const,
      forward: true,
      vjp: "first_order" as const,
      mutates: false,
      checkpoint_planes: Object.freeze([] as string[]),
    })
  ),
  ...OPTIMIZERS.map(([id, planes]) =>
    Object.freeze({
      id,
      category: "optimizer" as const,
      forward: false,
      vjp: "none" as const,
      mutates: true,
      checkpoint_planes: Object.freeze([...planes]),
    })
  ),
  ...LIFECYCLE.map(([id, mutates]) =>
    Object.freeze({
      id,
      category: "lifecycle" as const,
      forward: false,
      vjp: "none" as const,
      mutates,
      checkpoint_planes: Object.freeze([] as string[]),
    })
  ),
]);

const MANIFEST_V1: TrainingOpManifestV1 = Object.freeze({
  schema_id: "tritium.training_op_manifest",
  schema_version: 1,
  dtype: "f32",
  operations: Object.freeze(
    operations.filter((operation) =>
      operation.id !== "loss.topk_knowledge_distillation"
    ),
  ),
});

const MANIFEST_V2: TrainingOpManifestV2 = Object.freeze({
  schema_id: "tritium.training_op_manifest",
  schema_version: 2,
  dtype: "f32",
  operations,
});

const ROOT_FIELDS = [
  "schema_id",
  "schema_version",
  "dtype",
  "operations",
] as const;
const OP_FIELDS = [
  "id",
  "category",
  "forward",
  "vjp",
  "mutates",
  "checkpoint_planes",
] as const;

function sameKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
): boolean {
  const keys = Object.keys(value);
  return keys.length === expected.length &&
    expected.every((key) => keys.includes(key));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function canonicalManifestJson(
  manifest: TrainingOpManifestV1 | TrainingOpManifestV2,
): Uint8Array {
  const lines = [
    "{",
    `  "schema_id": ${JSON.stringify(manifest.schema_id)},`,
    `  "schema_version": ${manifest.schema_version},`,
    `  "dtype": ${JSON.stringify(manifest.dtype)},`,
    '  "operations": [',
  ];
  manifest.operations.forEach((operation, index) => {
    const suffix = index + 1 < manifest.operations.length ? "," : "";
    lines.push(`    ${JSON.stringify(operation)}${suffix}`);
  });
  lines.push("  ]", "}");
  return new TextEncoder().encode(`${lines.join("\n")}\n`);
}

/** Return byte-identical current V2 manifest JSON with terminal LF. */
export function canonicalTrainingManifestJson(): Uint8Array {
  return canonicalManifestJson(MANIFEST_V2);
}

/** Return byte-identical backward-readable V1 manifest JSON with terminal LF. */
export function canonicalTrainingManifestV1Json(): Uint8Array {
  return canonicalManifestJson(MANIFEST_V1);
}

class DuplicateKeyScanner {
  private index = 0;

  constructor(private readonly text: string) {}

  scan(): void {
    this.value();
    this.whitespace();
    if (this.index !== this.text.length) this.fail("trailing JSON bytes");
  }

  private fail(message: string): never {
    throw new TrainingManifestError(`${message} at byte ${this.index}`);
  }

  private whitespace(): void {
    while (/\s/u.test(this.text[this.index] ?? "")) this.index += 1;
  }

  private value(): void {
    this.whitespace();
    const token = this.text[this.index];
    if (token === "{") return this.object();
    if (token === "[") return this.array();
    if (token === '"') {
      this.string();
      return;
    }
    for (const literal of ["true", "false", "null"] as const) {
      if (this.text.startsWith(literal, this.index)) {
        this.index += literal.length;
        return;
      }
    }
    const number = this.text.slice(this.index).match(
      /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/u,
    );
    if (number !== null) {
      this.index += number[0].length;
      return;
    }
    this.fail("invalid JSON value");
  }

  private object(): void {
    this.index += 1;
    const keys = new Set<string>();
    this.whitespace();
    if (this.text[this.index] === "}") {
      this.index += 1;
      return;
    }
    while (true) {
      this.whitespace();
      if (this.text[this.index] !== '"') this.fail("object key must be string");
      const key = this.string();
      if (keys.has(key)) {
        this.fail(`duplicate manifest field ${JSON.stringify(key)}`);
      }
      keys.add(key);
      this.whitespace();
      if (this.text[this.index] !== ":") this.fail("missing object colon");
      this.index += 1;
      this.value();
      this.whitespace();
      const separator = this.text[this.index];
      if (separator === "}") {
        this.index += 1;
        return;
      }
      if (separator !== ",") this.fail("missing object separator");
      this.index += 1;
    }
  }

  private array(): void {
    this.index += 1;
    this.whitespace();
    if (this.text[this.index] === "]") {
      this.index += 1;
      return;
    }
    while (true) {
      this.value();
      this.whitespace();
      const separator = this.text[this.index];
      if (separator === "]") {
        this.index += 1;
        return;
      }
      if (separator !== ",") this.fail("missing array separator");
      this.index += 1;
    }
  }

  private string(): string {
    const start = this.index;
    this.index += 1;
    while (this.index < this.text.length) {
      const character = this.text[this.index];
      if (character === '"') {
        this.index += 1;
        return JSON.parse(this.text.slice(start, this.index)) as string;
      }
      if (character === "\\") {
        this.index += 1;
        const escape = this.text[this.index];
        if (escape === "u") {
          const digits = this.text.slice(this.index + 1, this.index + 5);
          if (!/^[0-9a-fA-F]{4}$/u.test(digits)) {
            this.fail("invalid Unicode escape");
          }
          this.index += 5;
          continue;
        }
        if (!'"\\/bfnrt'.includes(escape ?? "")) {
          this.fail("invalid string escape");
        }
      } else if ((character?.charCodeAt(0) ?? 0) < 0x20) {
        this.fail("unescaped string control character");
      }
      this.index += 1;
    }
    this.fail("unterminated string");
  }
}

function validateOperation(
  value: unknown,
  index: number,
  manifest: TrainingOpManifestV1 | TrainingOpManifestV2,
): void {
  if (!isRecord(value) || !sameKeys(value, OP_FIELDS)) {
    throw new TrainingManifestError(
      `operation ${index} fields differ from v${manifest.schema_version} contract`,
    );
  }
  const expected = manifest.operations[index];
  if (expected === undefined) {
    throw new TrainingManifestError(
      `operation count differs from v${manifest.schema_version}`,
    );
  }
  if (
    value.id !== expected.id ||
    value.category !== expected.category ||
    value.forward !== expected.forward ||
    value.vjp !== expected.vjp ||
    value.mutates !== expected.mutates ||
    !Array.isArray(value.checkpoint_planes) ||
    value.checkpoint_planes.length !== expected.checkpoint_planes.length ||
    !value.checkpoint_planes.every(
      (plane, planeIndex) =>
        typeof plane === "string" &&
        plane === expected.checkpoint_planes[planeIndex],
    )
  ) {
    throw new TrainingManifestError(
      `operation ${index} differs from frozen v${manifest.schema_version} descriptor`,
    );
  }
}

/** Parse exact V1 or V2 semantics; JSON whitespace and field order may differ. */
export function parseTrainingManifest(
  data: string | Uint8Array,
): TrainingOpManifestV1 | TrainingOpManifestV2 {
  let text: string;
  try {
    text = typeof data === "string"
      ? data
      : new TextDecoder("utf-8", { fatal: true }).decode(data);
  } catch (error) {
    throw new TrainingManifestError(`manifest is not UTF-8: ${String(error)}`);
  }
  let value: unknown;
  try {
    value = JSON.parse(text) as unknown;
  } catch (error) {
    throw new TrainingManifestError(
      `invalid training manifest JSON: ${String(error)}`,
    );
  }
  new DuplicateKeyScanner(text).scan();
  if (!isRecord(value) || !sameKeys(value, ROOT_FIELDS)) {
    throw new TrainingManifestError(
      "manifest root fields differ from supported contract",
    );
  }
  const manifest = value.schema_version === 1
    ? MANIFEST_V1
    : value.schema_version === 2
    ? MANIFEST_V2
    : null;
  if (
    manifest === null ||
    value.schema_id !== manifest.schema_id ||
    value.dtype !== manifest.dtype ||
    !Array.isArray(value.operations) ||
    value.operations.length !== manifest.operations.length
  ) {
    throw new TrainingManifestError(
      "manifest header or operation count differs from supported contract",
    );
  }
  value.operations.forEach((operation, index) =>
    validateOperation(operation, index, manifest)
  );
  return manifest;
}

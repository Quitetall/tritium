import {
  canonicalTrainingManifestJson,
  parseTrainingManifest,
} from "../../../bindings/typescript/src/training_manifest.ts";

import {
  TRAINING_MANIFEST_DIGEST_V1,
  TRAINING_VECTOR_DIGEST_V1,
} from "./identity.ts";

export type WebTrainingBackendPolicyV1 = "auto" | "webgpu" | "wasm";
export type WebTrainingImplementationV1 = "webgpu" | "wasm-fallback";
export type WebTrainingState =
  | "prepared"
  | "forward-complete"
  | "backward-complete"
  | "disposed";

export type WebTrainingErrorCode =
  | "adapter_unavailable"
  | "backend_policy"
  | "busy"
  | "capability_mismatch"
  | "disposed"
  | "invalid_config"
  | "invalid_receipt"
  | "invalid_schema"
  | "invalid_state"
  | "memory_limit";

export class WebTrainingError extends Error {
  readonly code: WebTrainingErrorCode;
  readonly state: WebTrainingState | null;

  constructor(
    code: WebTrainingErrorCode,
    message: string,
    state: WebTrainingState | null = null,
  ) {
    super(message);
    this.name = "WebTrainingError";
    this.code = code;
    this.state = state;
  }
}

export interface TrainingRecipeV1 {
  readonly schemaId: "tritium.training_recipe";
  readonly schemaVersion: 1;
  readonly operations: readonly string[];
}

export interface WebTrainingModelV1 {
  readonly schemaId: "tritium.web_training_model";
  readonly schemaVersion: 1;
  readonly recipe: TrainingRecipeV1;
  readonly payload: Uint8Array;
}

export interface TrainingBatchV1 {
  readonly inputs: Readonly<
    Record<string, Float32Array | Uint32Array | Uint8Array>
  >;
}

export interface WebTrainingConfigV1 {
  readonly backend: WebTrainingBackendPolicyV1;
  readonly allowWasmFallback: boolean;
  readonly maxResidentBytes: number;
  readonly seed: number;
  readonly requiredOperations: readonly string[];
}

export interface WebTrainingCapabilitiesV1 {
  readonly schemaId: "tritium.web_training_capabilities";
  readonly schemaVersion: 1;
  readonly implementation: WebTrainingImplementationV1;
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V1;
  readonly vectorDigest: typeof TRAINING_VECTOR_DIGEST_V1;
  readonly buildId: string;
  readonly physicalDevice: string | null;
  readonly supportedOperations: readonly string[];
  readonly maxResidentBytes: number;
}

export interface WebTrainingReceiptV1 {
  readonly schemaId: "tritium.web_training_receipt";
  readonly schemaVersion: 1;
  readonly implementation: WebTrainingImplementationV1;
  readonly manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V1;
  readonly vectorDigest: typeof TRAINING_VECTOR_DIGEST_V1;
  readonly buildId: string;
  readonly physicalDevice: string | null;
  readonly operation: string;
  readonly completedSteps: number;
  readonly peakResidentBytes: number;
}

export interface TrainingResultV1 {
  readonly loss: number;
  readonly receipt: WebTrainingReceiptV1;
}

export interface WebBinaryResultV1 {
  readonly bytes: Uint8Array;
  readonly receipt: WebTrainingReceiptV1;
}

/** Low-level adapter implemented by the generated WASM and WebGPU packages. */
export interface WebTrainingAdapterV1 {
  readonly capabilities: WebTrainingCapabilitiesV1;
  prepare(
    model: WebTrainingModelV1,
    config: WebTrainingConfigV1,
  ): Promise<WebTrainingReceiptV1>;
  forward(batch: TrainingBatchV1): Promise<TrainingResultV1>;
  backward(result: TrainingResultV1): Promise<WebTrainingReceiptV1>;
  step(): Promise<WebTrainingReceiptV1>;
  checkpoint(): Promise<WebBinaryResultV1>;
  resume(checkpoint: Uint8Array): Promise<WebTrainingReceiptV1>;
  export(): Promise<WebBinaryResultV1>;
  dispose(): Promise<void>;
}

const CONFIG_KEYS = [
  "allowWasmFallback",
  "backend",
  "maxResidentBytes",
  "requiredOperations",
  "seed",
] as const;
const MODEL_KEYS = ["payload", "recipe", "schemaId", "schemaVersion"] as const;
const RECIPE_KEYS = ["operations", "schemaId", "schemaVersion"] as const;
const CAPABILITY_KEYS = [
  "buildId",
  "implementation",
  "manifestDigest",
  "maxResidentBytes",
  "physicalDevice",
  "schemaId",
  "schemaVersion",
  "supportedOperations",
  "vectorDigest",
] as const;
const RECEIPT_KEYS = [
  "buildId",
  "completedSteps",
  "implementation",
  "manifestDigest",
  "operation",
  "peakResidentBytes",
  "physicalDevice",
  "schemaId",
  "schemaVersion",
  "vectorDigest",
] as const;

function fail(
  code: WebTrainingErrorCode,
  message: string,
  state: WebTrainingState | null = null,
): never {
  throw new WebTrainingError(code, message, state);
}

function exactKeys(
  value: object,
  expected: readonly string[],
  name: string,
  code: WebTrainingErrorCode = "invalid_schema",
): void {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (
    actual.length !== wanted.length ||
    actual.some((key, index) => key !== wanted[index])
  ) {
    fail(code, `${name} fields do not match schema v1`);
  }
}

function safeNonnegativeInteger(
  value: number,
  name: string,
  code: WebTrainingErrorCode = "invalid_config",
): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    fail(code, `${name} must be a nonnegative safe integer`);
  }
}

function nonemptyUniqueStrings(values: readonly string[], name: string): void {
  if (
    values.length === 0 ||
    values.some((value) => typeof value !== "string" || value.length === 0) ||
    new Set(values).size !== values.length
  ) {
    fail("invalid_schema", `${name} must contain unique nonempty strings`);
  }
}

function validateModel(model: WebTrainingModelV1): void {
  exactKeys(model, MODEL_KEYS, "model");
  if (
    model.schemaId !== "tritium.web_training_model" ||
    model.schemaVersion !== 1 ||
    !(model.payload instanceof Uint8Array) ||
    model.payload.byteLength === 0
  ) {
    fail("invalid_schema", "model is not a nonempty WebTrainingModelV1");
  }
  exactKeys(model.recipe, RECIPE_KEYS, "recipe");
  if (
    model.recipe.schemaId !== "tritium.training_recipe" ||
    model.recipe.schemaVersion !== 1
  ) {
    fail("invalid_schema", "recipe schema identity is not v1");
  }
  nonemptyUniqueStrings(model.recipe.operations, "recipe.operations");
}

function validateConfig(config: WebTrainingConfigV1): void {
  exactKeys(config, CONFIG_KEYS, "config");
  if (!(["auto", "webgpu", "wasm"] as const).includes(config.backend)) {
    fail("invalid_config", `unknown backend policy ${String(config.backend)}`);
  }
  if (typeof config.allowWasmFallback !== "boolean") {
    fail("invalid_config", "allowWasmFallback must be boolean");
  }
  safeNonnegativeInteger(config.maxResidentBytes, "maxResidentBytes");
  if (config.maxResidentBytes === 0) {
    fail("invalid_config", "maxResidentBytes must be positive");
  }
  safeNonnegativeInteger(config.seed, "seed");
  nonemptyUniqueStrings(config.requiredOperations, "requiredOperations");
}

function validateCapabilities(
  capabilities: WebTrainingCapabilitiesV1,
  config: WebTrainingConfigV1,
  recipe: TrainingRecipeV1,
): void {
  exactKeys(
    capabilities,
    CAPABILITY_KEYS,
    "capabilities",
    "capability_mismatch",
  );
  if (
    capabilities.schemaId !== "tritium.web_training_capabilities" ||
    capabilities.schemaVersion !== 1 ||
    capabilities.manifestDigest !== TRAINING_MANIFEST_DIGEST_V1 ||
    capabilities.vectorDigest !== TRAINING_VECTOR_DIGEST_V1 ||
    !(["webgpu", "wasm-fallback"] as const).includes(
      capabilities.implementation,
    ) ||
    capabilities.buildId.length === 0
  ) {
    fail("capability_mismatch", "adapter capability identity is invalid");
  }
  safeNonnegativeInteger(
    capabilities.maxResidentBytes,
    "adapter maxResidentBytes",
    "capability_mismatch",
  );
  nonemptyUniqueStrings(
    capabilities.supportedOperations,
    "capabilities.supportedOperations",
  );
  if (
    config.backend === "webgpu" &&
    capabilities.implementation !== "webgpu"
  ) {
    fail("backend_policy", "backend webgpu cannot use a WASM adapter");
  }
  if (
    config.backend === "wasm" &&
    capabilities.implementation !== "wasm-fallback"
  ) {
    fail("backend_policy", "backend wasm cannot use a WebGPU adapter");
  }
  if (
    capabilities.implementation === "wasm-fallback" &&
    config.backend === "auto" &&
    !config.allowWasmFallback
  ) {
    fail("backend_policy", "automatic WASM fallback is disabled");
  }
  if (config.maxResidentBytes > capabilities.maxResidentBytes) {
    fail("memory_limit", "configured memory ceiling exceeds adapter capacity");
  }

  const supported = new Set(capabilities.supportedOperations);
  const canonical = new Set(
    parseTrainingManifest(canonicalTrainingManifestJson()).operations.map(
      (operation) => operation.id,
    ),
  );
  for (const operation of [
    ...config.requiredOperations,
    ...recipe.operations,
  ]) {
    if (!canonical.has(operation)) {
      fail("invalid_schema", `unknown training operation ${operation}`);
    }
    if (!supported.has(operation)) {
      fail("capability_mismatch", `adapter does not support ${operation}`);
    }
  }
}

function validateReceipt(
  receipt: WebTrainingReceiptV1,
  capabilities: WebTrainingCapabilitiesV1,
  expectedOperation: string,
  maxResidentBytes: number,
): void {
  exactKeys(receipt, RECEIPT_KEYS, "receipt", "invalid_receipt");
  if (
    receipt.schemaId !== "tritium.web_training_receipt" ||
    receipt.schemaVersion !== 1 ||
    receipt.manifestDigest !== TRAINING_MANIFEST_DIGEST_V1 ||
    receipt.vectorDigest !== TRAINING_VECTOR_DIGEST_V1 ||
    receipt.implementation !== capabilities.implementation ||
    receipt.buildId !== capabilities.buildId ||
    receipt.physicalDevice !== capabilities.physicalDevice ||
    receipt.operation !== expectedOperation
  ) {
    fail("invalid_receipt", `invalid ${expectedOperation} receipt identity`);
  }
  safeNonnegativeInteger(
    receipt.completedSteps,
    "receipt.completedSteps",
    "invalid_receipt",
  );
  safeNonnegativeInteger(
    receipt.peakResidentBytes,
    "receipt.peakResidentBytes",
    "invalid_receipt",
  );
  if (receipt.peakResidentBytes > maxResidentBytes) {
    fail("memory_limit", `${expectedOperation} exceeded the memory ceiling`);
  }
}

function validateBinary(result: WebBinaryResultV1, operation: string): void {
  if (!(result.bytes instanceof Uint8Array) || result.bytes.byteLength === 0) {
    fail("invalid_receipt", `${operation} returned an empty binary artifact`);
  }
}

function freezeCapabilities(
  capabilities: WebTrainingCapabilitiesV1,
): WebTrainingCapabilitiesV1 {
  return Object.freeze({
    ...capabilities,
    supportedOperations: Object.freeze([
      ...capabilities.supportedOperations,
    ]),
  });
}

export class WebTrainingSession {
  readonly capabilities: WebTrainingCapabilitiesV1;
  readonly #adapter: WebTrainingAdapterV1;
  readonly #maxResidentBytes: number;
  #state: WebTrainingState = "prepared";
  #busy = false;
  #lastResult: TrainingResultV1 | null = null;

  private constructor(
    adapter: WebTrainingAdapterV1,
    maxResidentBytes: number,
    capabilities: WebTrainingCapabilitiesV1,
  ) {
    this.#adapter = adapter;
    this.#maxResidentBytes = maxResidentBytes;
    this.capabilities = capabilities;
  }

  static async prepare(
    model: WebTrainingModelV1,
    config: WebTrainingConfigV1,
    adapter: WebTrainingAdapterV1,
  ): Promise<WebTrainingSession> {
    validateModel(model);
    validateConfig(config);
    validateCapabilities(adapter.capabilities, config, model.recipe);
    const capabilities = freezeCapabilities(adapter.capabilities);
    const safeModel: WebTrainingModelV1 = Object.freeze({
      schemaId: model.schemaId,
      schemaVersion: model.schemaVersion,
      recipe: Object.freeze({
        schemaId: model.recipe.schemaId,
        schemaVersion: model.recipe.schemaVersion,
        operations: Object.freeze([...model.recipe.operations]),
      }),
      payload: model.payload.slice(),
    });
    const safeConfig: WebTrainingConfigV1 = Object.freeze({
      backend: config.backend,
      allowWasmFallback: config.allowWasmFallback,
      maxResidentBytes: config.maxResidentBytes,
      seed: config.seed,
      requiredOperations: Object.freeze([...config.requiredOperations]),
    });
    const receipt = await adapter.prepare(safeModel, safeConfig);
    validateReceipt(
      receipt,
      capabilities,
      "session.prepare",
      config.maxResidentBytes,
    );
    return new WebTrainingSession(
      adapter,
      config.maxResidentBytes,
      capabilities,
    );
  }

  get state(): WebTrainingState {
    return this.#state;
  }

  async #exclusive<T>(run: () => Promise<T>): Promise<T> {
    if (this.#state === "disposed") {
      fail("disposed", "training session is disposed", this.#state);
    }
    if (this.#busy) {
      fail("busy", "another session operation is in flight", this.#state);
    }
    this.#busy = true;
    try {
      return await run();
    } finally {
      this.#busy = false;
    }
  }

  #require(expected: WebTrainingState, operation: string): void {
    if (this.#state !== expected) {
      fail(
        "invalid_state",
        `${operation} requires ${expected}; current state is ${this.#state}`,
        this.#state,
      );
    }
  }

  async forward(batch: TrainingBatchV1): Promise<TrainingResultV1> {
    return this.#exclusive(async () => {
      this.#require("prepared", "forward");
      exactKeys(batch, ["inputs"], "batch");
      const inputs = Object.entries(batch.inputs);
      if (inputs.length === 0) {
        fail("invalid_schema", "batch.inputs must not be empty", this.#state);
      }
      if (
        inputs.some(
          ([name, value]) =>
            name.length === 0 ||
            !(
              value instanceof Float32Array ||
              value instanceof Uint32Array ||
              value instanceof Uint8Array
            ) ||
            value.length === 0,
        )
      ) {
        fail(
          "invalid_schema",
          "batch inputs must be named nonempty typed arrays",
          this.#state,
        );
      }
      const result = await this.#adapter.forward(batch);
      exactKeys(result, ["loss", "receipt"], "forward result", "invalid_receipt");
      if (!Number.isFinite(result.loss)) {
        fail("invalid_receipt", "forward loss must be finite", this.#state);
      }
      validateReceipt(
        result.receipt,
        this.capabilities,
        "session.forward",
        this.#maxResidentBytes,
      );
      this.#lastResult = result;
      this.#state = "forward-complete";
      return result;
    });
  }

  async backward(result: TrainingResultV1): Promise<WebTrainingReceiptV1> {
    return this.#exclusive(async () => {
      this.#require("forward-complete", "backward");
      if (result !== this.#lastResult) {
        fail("invalid_state", "backward result is not the active forward", this.#state);
      }
      const receipt = await this.#adapter.backward(result);
      validateReceipt(
        receipt,
        this.capabilities,
        "session.backward",
        this.#maxResidentBytes,
      );
      this.#state = "backward-complete";
      return receipt;
    });
  }

  async step(): Promise<WebTrainingReceiptV1> {
    return this.#exclusive(async () => {
      this.#require("backward-complete", "step");
      const receipt = await this.#adapter.step();
      validateReceipt(
        receipt,
        this.capabilities,
        "session.step",
        this.#maxResidentBytes,
      );
      this.#lastResult = null;
      this.#state = "prepared";
      return receipt;
    });
  }

  async checkpoint(): Promise<WebBinaryResultV1> {
    return this.#exclusive(async () => {
      this.#require("prepared", "checkpoint");
      const result = await this.#adapter.checkpoint();
      exactKeys(result, ["bytes", "receipt"], "checkpoint result", "invalid_receipt");
      validateBinary(result, "checkpoint");
      validateReceipt(
        result.receipt,
        this.capabilities,
        "session.checkpoint",
        this.#maxResidentBytes,
      );
      return { bytes: result.bytes.slice(), receipt: result.receipt };
    });
  }

  async resume(checkpoint: Uint8Array): Promise<WebTrainingReceiptV1> {
    return this.#exclusive(async () => {
      this.#require("prepared", "resume");
      if (!(checkpoint instanceof Uint8Array) || checkpoint.byteLength === 0) {
        fail("invalid_schema", "checkpoint must not be empty", this.#state);
      }
      const receipt = await this.#adapter.resume(checkpoint.slice());
      validateReceipt(
        receipt,
        this.capabilities,
        "session.resume",
        this.#maxResidentBytes,
      );
      return receipt;
    });
  }

  async export(): Promise<WebBinaryResultV1> {
    return this.#exclusive(async () => {
      this.#require("prepared", "export");
      const result = await this.#adapter.export();
      exactKeys(result, ["bytes", "receipt"], "export result", "invalid_receipt");
      validateBinary(result, "export");
      validateReceipt(
        result.receipt,
        this.capabilities,
        "session.export",
        this.#maxResidentBytes,
      );
      return { bytes: result.bytes.slice(), receipt: result.receipt };
    });
  }

  async dispose(): Promise<void> {
    if (this.#state === "disposed") return;
    await this.#exclusive(async () => {
      await this.#adapter.dispose();
      this.#lastResult = null;
      this.#state = "disposed";
    });
  }
}

export async function prepareTraining(
  model: WebTrainingModelV1,
  config: WebTrainingConfigV1,
  adapter?: WebTrainingAdapterV1,
): Promise<WebTrainingSession> {
  if (adapter === undefined) {
    fail(
      "adapter_unavailable",
      "no generated WebGPU or WASM adapter was supplied",
    );
  }
  return WebTrainingSession.prepare(model, config, adapter);
}

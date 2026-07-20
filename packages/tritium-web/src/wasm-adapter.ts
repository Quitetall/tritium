import {
  canonicalTrainingManifestJson,
  parseTrainingManifest,
} from "../../../bindings/typescript/src/training_manifest.ts";
import {
  TRAINING_MANIFEST_DIGEST_V1,
  TRAINING_VECTOR_DIGEST_V1,
} from "./identity.ts";
import {
  compilePortableCheckpointRequest,
  compilePortableResumeRequest,
  PortableLifecyclePlanError,
  preflightPortableLifecycleLayout,
} from "./lifecycle.ts";
import type {
  PortableAdamLeafV1,
  PortableCheckpointOptimizerV1,
  PortableCheckpointStateV1,
  PortableInt8AdamLeafV1,
  PortableMuonLeafV1,
  PortableSgdLeafV1,
} from "./lifecycle-types.js";
import {
  decodeWebTrainingPayload,
  WebTrainingPayloadError,
} from "./payload.ts";
import {
  admittedCompiledBufferMap,
  compilePortableBackwardOperationRequest,
  compilePortablePlanOperationRequest,
  PortableSchedulePlanError,
  preflightPortableSchedulePlan,
} from "./portable-schedule.ts";
import type {
  PortableBufferV1,
  PortableTrainingRequestV1,
  PortableTrainingResponseV1,
  PortableWasmSourceV1,
} from "./portable.js";
import type { PortableScheduleTensorV1 } from "./portable-schedule-types.js";
import type {
  CompiledTrainingBufferV1,
  CompiledTrainingOperationV1,
  CompiledTrainingPlanV1,
  TrainingBatchV1,
  TrainingResultV1,
  WebBinaryResultV1,
  WebTrainingAdapterV1,
  WebTrainingCapabilitiesV1,
  WebTrainingConfigV1,
  WebTrainingModelV1,
  WebTrainingReceiptV1,
} from "./session.ts";
import {
  WebTrainingError,
} from "./session.ts";
import type { WebTrainingErrorCode } from "./session.ts";
import {
  preparePortableWasmExecutor,
} from "./wasm.ts";
import type { PreparedPortableWasmExecutor } from "./wasm.ts";

const PHYSICAL_DEVICE = "wasm32:browser";
const MAX_RESIDENT_BYTES = 64 * 1024 * 1024;

function adapterFail(code: WebTrainingErrorCode, message: string): never {
  throw new WebTrainingError(code, message);
}

function normalizeCompilerError(error: unknown): never {
  if (error instanceof WebTrainingError) throw error;
  if (
    error instanceof PortableSchedulePlanError ||
    error instanceof PortableLifecyclePlanError
  ) {
    adapterFail(error.code === "capacity" ? "memory_limit" : "invalid_schema", error.message);
  }
  throw error;
}

function compileChecked<T>(compile: () => T): T {
  try {
    return compile();
  } catch (error) {
    normalizeCompilerError(error);
  }
}

function probeRequest(): PortableTrainingRequestV1 {
  const scalar = (name: string): PortableBufferV1 => Object.freeze({
    name,
    shape: Object.freeze([]),
    data: Object.freeze({ dtype: "f32", bits: Object.freeze([0]) }),
  });
  return Object.freeze({
    schemaId: "tritium.portable_training_request",
    schemaVersion: 1,
    physicalDevice: PHYSICAL_DEVICE,
    operation: "graph.add",
    execution: "forward",
    vectorDigest: TRAINING_VECTOR_DIGEST_V1,
    inputs: Object.freeze([scalar("left"), scalar("right")]),
    attributes: Object.freeze([]),
    outputs: Object.freeze([scalar("result")]),
  });
}

function requireSuccess(
  response: PortableTrainingResponseV1,
): Extract<PortableTrainingResponseV1, { readonly status: "ok" }> {
  if (response.status === "error") {
    adapterFail(
      "invalid_receipt",
      `portable WASM ${response.error.category}.${response.error.code}: ${response.error.message}`,
    );
  }
  return response;
}

function webReceipt(
  capabilities: WebTrainingCapabilitiesV1,
  operation: string,
  completedSteps: number,
  peakResidentBytes: number,
): WebTrainingReceiptV1 {
  return Object.freeze({
    schemaId: "tritium.web_training_receipt",
    schemaVersion: 1,
    implementation: "wasm-fallback",
    manifestDigest: TRAINING_MANIFEST_DIGEST_V1,
    vectorDigest: TRAINING_VECTOR_DIGEST_V1,
    buildId: capabilities.buildId,
    physicalDevice: capabilities.physicalDevice,
    operation,
    completedSteps,
    peakResidentBytes,
  });
}

function tensorFromOutput(output: PortableBufferV1): PortableScheduleTensorV1 {
  if (output.data.dtype === "f32") {
    const lanes = Uint32Array.from(output.data.bits);
    return new Float32Array(lanes.buffer);
  }
  if (output.data.dtype === "u32") return Uint32Array.from(output.data.values);
  return Uint8Array.from(output.data.values);
}

function rawF32Bits(tensor: PortableScheduleTensorV1, name: string): readonly number[] {
  if (!(tensor instanceof Float32Array)) adapterFail("invalid_schema", `${name} must be f32`);
  return Object.freeze(
    Array.from(new Uint32Array(tensor.buffer, tensor.byteOffset, tensor.length)),
  );
}

function optimizerKind(operation: string): PortableCheckpointOptimizerV1 {
  const kind = operation.startsWith("optimizer.")
    ? operation.slice("optimizer.".length)
    : "";
  if (
    kind === "sgd" ||
    kind === "adamw" ||
    kind === "cautious_adamw" ||
    kind === "int8_adamw" ||
    kind === "muon"
  ) {
    return kind;
  }
  adapterFail("capability_mismatch", `unsupported optimizer operation ${operation}`);
}

function decodeStepOutput(output: PortableBufferV1 | undefined): number {
  if (
    output?.name !== "step" ||
    output.data.dtype !== "bytes" ||
    output.data.values.length !== 8
  ) {
    adapterFail("invalid_receipt", "portable WASM resume returned invalid step output");
  }
  let step = 0n;
  for (let index = 7; index >= 0; index -= 1) {
    step = (step << 8n) | BigInt(output.data.values[index] ?? 0);
  }
  if (step > BigInt(Number.MAX_SAFE_INTEGER)) {
    adapterFail(
      "invalid_receipt",
      "portable WASM resume step exceeds JavaScript safe integer range",
    );
  }
  return Number(step);
}

function sameTensorType(left: PortableScheduleTensorV1, right: PortableScheduleTensorV1): boolean {
  return (
    (left instanceof Float32Array && right instanceof Float32Array) ||
    (left instanceof Uint32Array && right instanceof Uint32Array) ||
    (left instanceof Uint8Array && right instanceof Uint8Array)
  );
}

type PendingTensorWrite = Readonly<{
  target: PortableScheduleTensorV1;
  candidate: PortableScheduleTensorV1;
}>;

/** Validate the built-in WASM subset without creating or probing a guest. */
export function validatePortableWasmPlan(plan: CompiledTrainingPlanV1): void {
  try {
    const buffers = admittedCompiledBufferMap(plan);
    preflightPortableSchedulePlan(plan);
    let kind: PortableCheckpointOptimizerV1 | null = null;
    const leafLengths: number[] = [];
    for (const operation of plan.operations) {
      if (!operation.operation.startsWith("optimizer.")) continue;
      const current = optimizerKind(operation.operation);
      if (kind !== null && current !== kind) {
        adapterFail(
          "capability_mismatch",
          "portable WASM sessions require one optimizer kind",
        );
      }
      kind = current;
      const parameter = buffers.get(operation.inputs[0]!);
      if (parameter === undefined || parameter.dtype !== "f32") {
        adapterFail("invalid_schema", "optimizer parameter must be a compiled f32 buffer");
      }
      leafLengths.push(parameter.byteLength / 4);
    }
    if (kind === null) adapterFail("invalid_schema", "compiled plan has no optimizer operations");
    preflightPortableLifecycleLayout(kind, leafLengths);
  } catch (error) {
    normalizeCompilerError(error);
  }
}

class PortableWasmTrainingAdapter implements WebTrainingAdapterV1 {
  readonly capabilities: WebTrainingCapabilitiesV1;
  readonly #executor: PreparedPortableWasmExecutor;
  #plan: CompiledTrainingPlanV1 | null = null;
  #store: Record<string, PortableScheduleTensorV1> | null = null;
  #buffers = new Map<string, CompiledTrainingBufferV1>();
  #completedSteps = 0;
  #disposed = false;

  constructor(executor: PreparedPortableWasmExecutor) {
    this.#executor = executor;
    const supportedOperations = parseTrainingManifest(
      canonicalTrainingManifestJson(),
    ).operations
      .filter((operation) => operation.category !== "lifecycle")
      .map((operation) => operation.id);
    supportedOperations.push("lifecycle.checkpoint", "lifecycle.resume");
    this.capabilities = Object.freeze({
      schemaId: "tritium.web_training_capabilities",
      schemaVersion: 1,
      implementation: "wasm-fallback",
      manifestDigest: TRAINING_MANIFEST_DIGEST_V1,
      vectorDigest: TRAINING_VECTOR_DIGEST_V1,
      buildId: executor.buildId,
      physicalDevice: PHYSICAL_DEVICE,
      supportedOperations: Object.freeze(supportedOperations),
      maxResidentBytes: MAX_RESIDENT_BYTES,
    });
  }

  async validate(
    _model: WebTrainingModelV1,
    _config: WebTrainingConfigV1,
    plan: CompiledTrainingPlanV1,
  ): Promise<void> {
    validatePortableWasmPlan(plan);
  }

  async prepare(
    model: WebTrainingModelV1,
    _config: WebTrainingConfigV1,
    plan: CompiledTrainingPlanV1,
  ): Promise<WebTrainingReceiptV1> {
    if (this.#disposed || this.#plan !== null) {
      adapterFail("invalid_state", "portable WASM adapter is not fresh");
    }
    let decoded: Readonly<Record<string, PortableScheduleTensorV1>>;
    try {
      decoded = decodeWebTrainingPayload(plan, model.payload);
    } catch (error) {
      if (error instanceof WebTrainingPayloadError) {
        adapterFail(error.code === "capacity" ? "memory_limit" : "invalid_schema", error.message);
      }
      throw error;
    }
    this.#store = Object.assign(
      Object.create(null) as Record<string, PortableScheduleTensorV1>,
      decoded,
    );
    this.#plan = plan;
    this.#buffers = new Map(plan.buffers.map((buffer) => [buffer.id, buffer]));
    return webReceipt(this.capabilities, "session.prepare", 0, plan.preparePeakBytes);
  }

  #ready(): Readonly<{
    plan: CompiledTrainingPlanV1;
    store: Record<string, PortableScheduleTensorV1>;
  }> {
    if (this.#disposed) adapterFail("disposed", "portable WASM adapter is disposed");
    if (this.#plan === null || this.#store === null) {
      adapterFail("invalid_state", "portable WASM adapter is not prepared");
    }
    return { plan: this.#plan, store: this.#store };
  }

  async #execute(
    request: PortableTrainingRequestV1,
    outputBufferIds: readonly string[],
  ): Promise<readonly PendingTensorWrite[]> {
    const { store } = this.#ready();
    const response = requireSuccess(
      await this.#executor.execute(request),
    );
    if (response.outputs.length !== outputBufferIds.length) {
      adapterFail(
        "invalid_receipt",
        "portable WASM output count differs from compiled schedule",
      );
    }
    return response.outputs.map((output, index) => {
      const buffer = this.#buffers.get(outputBufferIds[index]!);
      if (buffer === undefined) {
        adapterFail("invalid_receipt", "portable WASM output targets unknown buffer");
      }
      const target = store[buffer.ownerId];
      const candidate = tensorFromOutput(output);
      if (target === undefined || !sameTensorType(target, candidate) || target.length !== candidate.length) {
        adapterFail(
          "invalid_receipt",
          `portable WASM output ${buffer.id} differs from prepared buffer`,
        );
      }
      return { target, candidate };
    });
  }

  #commit(writes: readonly PendingTensorWrite[]): void {
    for (const { target, candidate } of writes) target.set(candidate as never);
  }

  async #dispatch(
    request: PortableTrainingRequestV1,
    outputBufferIds: readonly string[],
  ): Promise<void> {
    this.#commit(await this.#execute(request, outputBufferIds));
  }

  async forward(batch: TrainingBatchV1): Promise<TrainingResultV1> {
    const { plan, store } = this.#ready();
    for (const [id, value] of Object.entries(batch.inputs)) {
      const buffer = this.#buffers.get(id);
      const target = buffer === undefined ? undefined : store[buffer.ownerId];
      if (target === undefined || !sameTensorType(target, value) || target.length !== value.length) {
        adapterFail("invalid_schema", `batch tensor ${id} differs from prepared buffer`);
      }
      target.set(value as never);
    }
    for (const operation of plan.operations) {
      if (operation.operation.startsWith("optimizer.")) continue;
      const dispatch = compileChecked(() =>
        compilePortablePlanOperationRequest(plan, operation.id, store, PHYSICAL_DEVICE),
      );
      await this.#dispatch(dispatch.request, dispatch.outputBufferIds);
    }
    const lossOperation = [...plan.operations].reverse().find((operation) => operation.operation.startsWith("loss."));
    const lossBuffer = lossOperation === undefined ? undefined : this.#buffers.get(lossOperation.outputs[0]!);
    const lossTensor = lossBuffer === undefined ? undefined : store[lossBuffer.ownerId];
    if (!(lossTensor instanceof Float32Array) || lossTensor.length !== 1) {
      adapterFail(
        "invalid_receipt",
        "compiled schedule did not produce one scalar f32 loss",
      );
    }
    return Object.freeze({
      loss: lossTensor[0]!,
      receipt: webReceipt(
        this.capabilities,
        "session.forward",
        this.#completedSteps,
        plan.forwardPeakBytes,
      ),
    });
  }

  async backward(_result: TrainingResultV1): Promise<WebTrainingReceiptV1> {
    const { plan, store } = this.#ready();
    for (const buffer of plan.buffers) {
      if (buffer.ownerId !== buffer.id) continue;
      if (buffer.backwardInitialization === "zero") store[buffer.id]?.fill(0);
      if (buffer.backwardInitialization === "one") store[buffer.id]?.fill(1);
    }
    for (const operation of plan.backwardOperations) {
      const dispatch = compileChecked(() =>
        compilePortableBackwardOperationRequest(plan, operation.id, store, PHYSICAL_DEVICE),
      );
      await this.#dispatch(dispatch.request, dispatch.outputBufferIds);
    }
    return webReceipt(
      this.capabilities,
      "session.backward",
      this.#completedSteps,
      plan.peakBytes,
    );
  }

  async step(): Promise<WebTrainingReceiptV1> {
    const { plan, store } = this.#ready();
    const writes: PendingTensorWrite[] = [];
    for (const operation of plan.operations) {
      if (!operation.operation.startsWith("optimizer.")) continue;
      const dispatch = compileChecked(() =>
        compilePortablePlanOperationRequest(plan, operation.id, store, PHYSICAL_DEVICE),
      );
      const attributes = dispatch.request.attributes.map((attribute) =>
        attribute.kind === "u64" && attribute.name === "step"
          ? Object.freeze({ ...attribute, value: this.#completedSteps + 1 })
          : attribute,
      );
      writes.push(
        ...(await this.#execute(
          Object.freeze({ ...dispatch.request, attributes: Object.freeze(attributes) }),
          dispatch.outputBufferIds,
        )),
      );
    }
    this.#commit(writes);
    this.#completedSteps += 1;
    return webReceipt(
      this.capabilities,
      "session.step",
      this.#completedSteps,
      plan.peakBytes,
    );
  }

  async checkpoint(): Promise<WebBinaryResultV1> {
    const { plan } = this.#ready();
    const { state } = this.#lifecycleState();
    const response = requireSuccess(
      await this.#executor.execute(
        compileChecked(() => compilePortableCheckpointRequest(state, PHYSICAL_DEVICE)),
      ),
    );
    const output = response.outputs[0];
    if (output?.name !== "checkpoint" || output.data.dtype !== "bytes") {
      adapterFail("invalid_receipt", "portable WASM checkpoint returned invalid bytes");
    }
    return Object.freeze({
      bytes: Uint8Array.from(output.data.values),
      receipt: webReceipt(
        this.capabilities,
        "session.checkpoint",
        this.#completedSteps,
        plan.peakBytes,
      ),
    });
  }

  #lifecycleState(): Readonly<{
    optimizer: PortableCheckpointOptimizerV1;
    operations: readonly CompiledTrainingOperationV1[];
    state: PortableCheckpointStateV1;
  }> {
    const { plan, store } = this.#ready();
    const operations = plan.operations.filter((operation) =>
      operation.operation.startsWith("optimizer."),
    );
    if (operations.length === 0) {
      adapterFail("invalid_schema", "compiled plan has no optimizer operations");
    }
    const optimizer = optimizerKind(operations[0]!.operation);
    if (operations.some((operation) => optimizerKind(operation.operation) !== optimizer)) {
      adapterFail(
        "capability_mismatch",
        "portable WASM sessions require one optimizer kind",
      );
    }
    const parameterBits = (operation: CompiledTrainingOperationV1): readonly number[] => {
      const parameterBuffer = this.#buffers.get(operation.inputs[0]!);
      const parameter = parameterBuffer === undefined
        ? undefined
        : store[parameterBuffer.ownerId];
      return rawF32Bits(parameter!, operation.inputs[0]!);
    };
    let state: PortableCheckpointStateV1;
    if (optimizer === "sgd") {
      const leaves: readonly PortableSgdLeafV1[] = operations.map((operation) =>
        Object.freeze({ parameter: parameterBits(operation) }),
      );
      state = Object.freeze({
        optimizer,
        step: this.#completedSteps,
        leaves: Object.freeze(leaves),
      });
    } else if (optimizer === "adamw" || optimizer === "cautious_adamw") {
      const leaves: readonly PortableAdamLeafV1[] = operations.map((operation) =>
        Object.freeze({
          parameter: parameterBits(operation),
          moment1: rawF32Bits(store[operation.inputs[2]!]!, operation.inputs[2]!),
          moment2: rawF32Bits(store[operation.inputs[3]!]!, operation.inputs[3]!),
        }),
      );
      state = Object.freeze({
        optimizer,
        step: this.#completedSteps,
        leaves: Object.freeze(leaves),
      });
    } else if (optimizer === "int8_adamw") {
      const leaves: readonly PortableInt8AdamLeafV1[] = operations.map((operation) => {
        const moment1 = store[operation.inputs[2]!]!;
        const moment2 = store[operation.inputs[3]!]!;
        if (!(moment1 instanceof Uint8Array) || !(moment2 instanceof Uint8Array)) {
          adapterFail("invalid_schema", "int8 AdamW moment planes must be bytes");
        }
        return Object.freeze({
          parameter: parameterBits(operation),
          moment1Q8: Object.freeze(Array.from(moment1)),
          moment2Q8: Object.freeze(Array.from(moment2)),
          moment1Scale: rawF32Bits(store[operation.inputs[4]!]!, operation.inputs[4]!),
          moment2Scale: rawF32Bits(store[operation.inputs[5]!]!, operation.inputs[5]!),
        });
      });
      state = Object.freeze({
        optimizer,
        step: this.#completedSteps,
        leaves: Object.freeze(leaves),
      });
    } else {
      const leaves: readonly PortableMuonLeafV1[] = operations.map((operation) =>
        Object.freeze({
          parameter: parameterBits(operation),
          momentum: rawF32Bits(store[operation.inputs[2]!]!, operation.inputs[2]!),
        }),
      );
      state = Object.freeze({
        optimizer,
        step: this.#completedSteps,
        leaves: Object.freeze(leaves),
      });
    }
    return Object.freeze({
      optimizer,
      operations: Object.freeze(operations),
      state,
    });
  }

  async resume(checkpoint: Uint8Array): Promise<WebTrainingReceiptV1> {
    const { plan, store } = this.#ready();
    const { optimizer, operations, state } = this.#lifecycleState();
    const leafLengths = state.leaves.map((leaf) => leaf.parameter.length);
    const response = requireSuccess(
      await this.#executor.execute(
        compileChecked(() =>
          compilePortableResumeRequest(
            optimizer,
            leafLengths,
            Uint8Array.from(checkpoint),
            PHYSICAL_DEVICE,
          ),
        ),
      ),
    );
    const completedSteps = decodeStepOutput(response.outputs[0]);
    const writes: PendingTensorWrite[] = [];
    let outputIndex = 1;
    for (const operation of operations) {
      const targetIds = [operation.inputs[0]!, ...operation.inputs.slice(2)];
      for (const targetId of targetIds) {
        const output = response.outputs[outputIndex];
        const buffer = this.#buffers.get(targetId);
        const target = buffer === undefined ? undefined : store[buffer.ownerId];
        if (output === undefined || target === undefined) {
          adapterFail("invalid_receipt", "portable WASM resume omitted optimizer plane");
        }
        const candidate = tensorFromOutput(output);
        if (!sameTensorType(target, candidate) || target.length !== candidate.length) {
          adapterFail(
            "invalid_receipt",
            `portable WASM resume changed ${targetId} layout`,
          );
        }
        writes.push({ target, candidate });
        outputIndex += 1;
      }
    }
    if (outputIndex !== response.outputs.length) {
      adapterFail("invalid_receipt", "portable WASM resume returned extra optimizer planes");
    }
    this.#commit(writes);
    this.#completedSteps = completedSteps;
    return webReceipt(
      this.capabilities,
      "session.resume",
      completedSteps,
      plan.peakBytes,
    );
  }

  async export(): Promise<WebBinaryResultV1> {
    adapterFail(
      "adapter_unavailable",
      "portable WASM state-derived SALT export is not implemented",
    );
  }

  async dispose(): Promise<void> {
    if (this.#disposed) return;
    this.#disposed = true;
    if (this.#store !== null) {
      for (const tensor of Object.values(this.#store)) tensor.fill(0);
    }
    this.#store = null;
    this.#plan = null;
    this.#buffers.clear();
  }
}

/** Create session-owned deterministic WASM fallback adapter. */
export async function createPortableWasmTrainingAdapter(
  source: PortableWasmSourceV1 = new URL("./tritium_wasm_bg.wasm", import.meta.url),
): Promise<WebTrainingAdapterV1> {
  const executor = await preparePortableWasmExecutor(source);
  const probe = requireSuccess(await executor.execute(probeRequest()));
  if (probe.receipt.backendBuild !== executor.buildId) {
    adapterFail(
      "invalid_receipt",
      "portable WASM probe returned different build identity",
    );
  }
  return new PortableWasmTrainingAdapter(executor);
}

import {
  PortableLifecyclePlanError,
  compilePortableCheckpointRequest,
  compilePortableExportRequest,
  compilePortableReloadRequest,
  compilePortableResumeRequest,
} from "./lifecycle.ts";
import type {
  PortableCheckpointOptimizerV1,
  PortableCheckpointStateV1,
} from "./lifecycle-types.js";
import type {
  PortableTrainingReceiptV1,
  PortableTrainingResponseV1,
  PortableWasmSourceV1,
} from "./portable.js";
import type {
  PortableWasmLifecycleBinaryV1,
  PortableWasmLifecycleErrorCode,
  PortableWasmLifecycleOptionsV1,
} from "./portable-state-types.js";
import { executePortableWasmRequest, snapshotPortableWasmSource } from "./wasm.ts";

export type {
  PortableWasmLifecycleBinaryV1,
  PortableWasmLifecycleErrorCode,
  PortableWasmLifecycleOptionsV1,
  PortableWasmLifecycleStateV1,
} from "./portable-state-types.js";

type OwnedState = {
  optimizer: PortableCheckpointOptimizerV1;
  step: number;
  leaves: Record<string, readonly number[]>[];
};

export class PortableWasmLifecycleError extends Error {
  readonly code: PortableWasmLifecycleErrorCode;

  constructor(code: PortableWasmLifecycleErrorCode, message: string) {
    super(message);
    this.name = "PortableWasmLifecycleError";
    this.code = code;
  }
}

function normalizeError(error: unknown): PortableWasmLifecycleError {
  if (error instanceof PortableWasmLifecycleError) return error;
  if (error instanceof PortableLifecyclePlanError) {
    return new PortableWasmLifecycleError("invalid_state", error.message);
  }
  const message = error instanceof Error ? error.message : String(error);
  return new PortableWasmLifecycleError("backend", message);
}

function copyState(state: PortableCheckpointStateV1): OwnedState {
  return {
    optimizer: state.optimizer,
    step: state.step,
    leaves: state.leaves.map((leaf) => {
      const copy: Record<string, readonly number[]> = {};
      for (const [name, values] of Object.entries(
        leaf as unknown as Record<string, readonly number[]>,
      )) {
        copy[name] = Object.freeze(Array.from(values));
      }
      return Object.freeze(copy);
    }),
  };
}

function freezeState(state: OwnedState): PortableCheckpointStateV1 {
  return Object.freeze({
    optimizer: state.optimizer,
    step: state.step,
    leaves: Object.freeze(
      state.leaves.map((leaf) =>
        Object.freeze(
          Object.fromEntries(
            Object.entries(leaf).map(([name, values]) => [
              name,
              Object.freeze(Array.from(values)),
            ]),
          ),
        ),
      ),
    ),
  }) as unknown as PortableCheckpointStateV1;
}

async function snapshotSource(source: PortableWasmSourceV1): Promise<ArrayBuffer> {
  const bytes = await snapshotPortableWasmSource(source);
  return bytes.buffer.slice(
    bytes.byteOffset, bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
}

function requireSuccess(
  response: PortableTrainingResponseV1,
): Extract<PortableTrainingResponseV1, { readonly status: "ok" }> {
  if (response.status === "error") {
    throw new PortableWasmLifecycleError(
      "backend",
      `${response.error.category}.${response.error.code}: ${response.error.message}`,
    );
  }
  return response;
}

function bytesOutput(
  response: Extract<PortableTrainingResponseV1, { readonly status: "ok" }>,
  index: number,
  name: string,
): Uint8Array {
  const output = response.outputs[index];
  if (output?.name !== name || output.data.dtype !== "bytes") {
    throw new PortableWasmLifecycleError(
      "backend",
      `portable WASM returned invalid ${name} output`,
    );
  }
  return Uint8Array.from(output.data.values);
}

function bitsOutput(
  response: Extract<PortableTrainingResponseV1, { readonly status: "ok" }>,
  index: number,
  name: string,
): readonly number[] {
  const output = response.outputs[index];
  if (output?.name !== name || output.data.dtype !== "f32") {
    throw new PortableWasmLifecycleError(
      "backend",
      `portable WASM returned invalid ${name} output`,
    );
  }
  return Object.freeze(Array.from(output.data.bits));
}

function decodeStep(bytes: Uint8Array): number {
  if (bytes.byteLength !== 8) {
    throw new PortableWasmLifecycleError("backend", "resume returned invalid step bytes");
  }
  let step = 0n;
  for (let index = 7; index >= 0; index -= 1) {
    step = (step << 8n) | BigInt(bytes[index] ?? 0);
  }
  if (step > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new PortableWasmLifecycleError(
      "backend",
      "resume step exceeds the JavaScript safe integer range",
    );
  }
  return Number(step);
}

function decodeResume(
  optimizer: PortableCheckpointOptimizerV1,
  leafLengths: readonly number[],
  response: Extract<PortableTrainingResponseV1, { readonly status: "ok" }>,
): OwnedState {
  const step = decodeStep(bytesOutput(response, 0, "step"));
  const leaves: Record<string, readonly number[]>[] = [];
  let output = 1;
  for (const [index, length] of leafLengths.entries()) {
    const leaf: Record<string, readonly number[]> = {
      parameter: bitsOutput(response, output, `parameter.${index}`),
    };
    output += 1;
    if (optimizer === "adamw" || optimizer === "cautious_adamw") {
      leaf.moment1 = bitsOutput(response, output, `moment1.${index}`);
      leaf.moment2 = bitsOutput(response, output + 1, `moment2.${index}`);
      output += 2;
    } else if (optimizer === "int8_adamw") {
      leaf.moment1Q8 = Object.freeze(
        Array.from(bytesOutput(response, output, `moment1_q8.${index}`)),
      );
      leaf.moment2Q8 = Object.freeze(
        Array.from(bytesOutput(response, output + 1, `moment2_q8.${index}`)),
      );
      leaf.moment1Scale = bitsOutput(response, output + 2, `moment1_scale.${index}`);
      leaf.moment2Scale = bitsOutput(response, output + 3, `moment2_scale.${index}`);
      output += 4;
    } else if (optimizer === "muon") {
      leaf.momentum = bitsOutput(response, output, `momentum.${index}`);
      output += 1;
    }
    if (leaf.parameter === undefined || leaf.parameter.length !== length) {
      throw new PortableWasmLifecycleError("backend", "resume changed a leaf length");
    }
    leaves.push(Object.freeze(leaf));
  }
  if (output !== response.outputs.length) {
    throw new PortableWasmLifecycleError("backend", "resume returned extra outputs");
  }
  return { optimizer, step, leaves };
}

/** Owns portable optimizer planes and commits only guest-validated transitions. */
export class PortableWasmLifecycleState {
  readonly #guest: ArrayBuffer;
  readonly #physicalDevice: string;
  #owned: OwnedState;
  #busy = false;
  #disposed = false;

  private constructor(guest: ArrayBuffer, state: OwnedState, physicalDevice: string) {
    this.#guest = guest;
    this.#owned = state;
    this.#physicalDevice = physicalDevice;
  }

  static async create(
    options: PortableWasmLifecycleOptionsV1,
  ): Promise<PortableWasmLifecycleState> {
    try {
      if (typeof options !== "object" || options === null) {
        throw new PortableWasmLifecycleError("invalid_state", "options must be an object");
      }
      const keys = Object.keys(options).sort();
      const expected = options.physicalDevice === undefined
        ? ["source", "state"]
        : ["physicalDevice", "source", "state"];
      if (keys.length !== expected.length || keys.some((key, index) => key !== expected[index])) {
        throw new PortableWasmLifecycleError(
          "invalid_state",
          "lifecycle options fields do not match schema v1",
        );
      }
      const physicalDevice = options.physicalDevice ?? "wasm32:browser";
      if (typeof physicalDevice !== "string" || physicalDevice.length === 0) {
        throw new PortableWasmLifecycleError(
          "invalid_state",
          "physicalDevice must be nonempty",
        );
      }
      compilePortableCheckpointRequest(options.state, physicalDevice);
      const owned = copyState(options.state);
      const guest = await snapshotSource(options.source);
      const controller = new PortableWasmLifecycleState(guest, owned, physicalDevice);
      await controller.checkpoint();
      return controller;
    } catch (error) {
      throw normalizeError(error);
    }
  }

  get state(): PortableCheckpointStateV1 {
    if (this.#disposed) {
      throw new PortableWasmLifecycleError("disposed", "lifecycle state is disposed");
    }
    return freezeState(this.#owned);
  }

  async #exclusive<T>(run: () => Promise<T>): Promise<T> {
    if (this.#disposed) {
      throw new PortableWasmLifecycleError("disposed", "lifecycle state is disposed");
    }
    if (this.#busy) {
      throw new PortableWasmLifecycleError("busy", "a lifecycle transition is in flight");
    }
    this.#busy = true;
    try {
      return await run();
    } catch (error) {
      throw normalizeError(error);
    } finally {
      this.#busy = false;
    }
  }

  async #checkpoint(state: PortableCheckpointStateV1): Promise<PortableWasmLifecycleBinaryV1> {
    const response = requireSuccess(
      await executePortableWasmRequest(
        compilePortableCheckpointRequest(state, this.#physicalDevice),
        this.#guest,
      ),
    );
    return Object.freeze({
      bytes: bytesOutput(response, 0, "checkpoint"),
      receipt: response.receipt,
    });
  }

  async checkpoint(): Promise<PortableWasmLifecycleBinaryV1> {
    return this.#exclusive(() => this.#checkpoint(freezeState(this.#owned)));
  }

  async commit(state: PortableCheckpointStateV1): Promise<PortableTrainingReceiptV1> {
    return this.#exclusive(async () => {
      compilePortableCheckpointRequest(state, this.#physicalDevice);
      const candidate = copyState(state);
      const result = await this.#checkpoint(freezeState(candidate));
      this.#owned = candidate;
      return result.receipt;
    });
  }

  async resume(checkpoint: Uint8Array): Promise<PortableTrainingReceiptV1> {
    return this.#exclusive(async () => {
      if (!(checkpoint instanceof Uint8Array)) {
        throw new PortableWasmLifecycleError(
          "invalid_state",
          "checkpoint must be a Uint8Array",
        );
      }
      const leafLengths = this.#owned.leaves.map((leaf) => leaf.parameter?.length ?? 0);
      const response = requireSuccess(
        await executePortableWasmRequest(
          compilePortableResumeRequest(
            this.#owned.optimizer,
            leafLengths,
            Uint8Array.from(checkpoint),
            this.#physicalDevice,
          ),
          this.#guest,
        ),
      );
      const candidate = decodeResume(this.#owned.optimizer, leafLengths, response);
      this.#owned = candidate;
      return response.receipt;
    });
  }

  async admitExport(packageBytes: Uint8Array): Promise<PortableWasmLifecycleBinaryV1> {
    return this.#exclusive(async () => {
      if (!(packageBytes instanceof Uint8Array)) {
        throw new PortableWasmLifecycleError(
          "invalid_state",
          "SALT package must be a Uint8Array",
        );
      }
      const expected = Uint8Array.from(packageBytes);
      const exported = requireSuccess(
        await executePortableWasmRequest(
          compilePortableExportRequest(expected, this.#physicalDevice),
          this.#guest,
        ),
      );
      const artifact = bytesOutput(exported, 0, "artifact");
      if (
        artifact.length !== expected.length ||
        artifact.some((value, index) => value !== expected[index])
      ) {
        throw new PortableWasmLifecycleError(
          "backend",
          "portable WASM changed the state-derived SALT artifact",
        );
      }
      const reloaded = requireSuccess(
        await executePortableWasmRequest(
          compilePortableReloadRequest(artifact, this.#physicalDevice),
          this.#guest,
        ),
      );
      const admitted = bytesOutput(reloaded, 0, "package");
      if (
        artifact.length !== admitted.length ||
        artifact.some((value, index) => value !== admitted[index])
      ) {
        throw new PortableWasmLifecycleError(
          "backend",
          "strict reload changed the exported SALT package",
        );
      }
      return Object.freeze({ bytes: artifact, receipt: exported.receipt });
    });
  }

  dispose(): void {
    if (this.#busy) {
      throw new PortableWasmLifecycleError("busy", "a lifecycle transition is in flight");
    }
    this.#disposed = true;
    this.#owned = { optimizer: "sgd", step: 0, leaves: [] };
    new Uint8Array(this.#guest).fill(0);
  }
}

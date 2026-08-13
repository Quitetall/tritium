import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";

import {
  TRAINING_MANIFEST_DIGEST_V2,
  TRAINING_VECTOR_DIGEST_V2,
} from "./identity.ts";
import {
  compilePortableReloadRequest,
} from "./lifecycle.ts";
import { encodeWebTrainingPayload } from "./payload.ts";
import {
  prepareTraining,
  WebTrainingError,
  type TrainingBatchV1,
  type WebTrainingConfigV1,
  type WebTrainingModelV1,
  type WebTrainingReceiptV1,
} from "./session.ts";
import { executePortableWasmRequest } from "./wasm.ts";
import { createWebGpuTrainingAdapter } from "./webgpu-adapter.ts";
import { webGpuKernelCandidateBundleV1 } from "./webgpu-kernels.ts";
import type { WebGpuDevicePortV1 } from "./webgpu-runtime.ts";
import {
  runWebGpuVectorConformanceV1,
  type WebGpuVectorConformanceTraceV1,
} from "./webgpu-conformance.ts";

const UTF8 = new TextEncoder();
const SOFTWARE_MARKERS = Object.freeze([
  "swiftshader",
  "llvmpipe",
  "software",
  "emulator",
  "lavapipe",
  "warp",
]);
const MAX_PEAK_BYTES = 64 * 1024 * 1024;

export type PhysicalBrowserQualificationErrorCode =
  | "adapter_unavailable"
  | "device_identity"
  | "fault_injection"
  | "instrumentation"
  | "invalid_options"
  | "lifecycle"
  | "native_artifact_parity"
  | "vector_conformance";

export class PhysicalBrowserQualificationError extends Error {
  readonly code: PhysicalBrowserQualificationErrorCode;

  constructor(code: PhysicalBrowserQualificationErrorCode, message: string) {
    super(message);
    this.name = "PhysicalBrowserQualificationError";
    this.code = code;
  }
}

export type PhysicalBrowserAdapterIdentityV1 = Readonly<{
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  software: false;
}>;

export type PhysicalBrowserLimitsV1 = Readonly<{
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  maxComputeWorkgroupsPerDimension: number;
  maxStorageBuffersPerShaderStage: number;
}>;

export type PhysicalBrowserFaultTraceV1 = Readonly<{
  passed: true;
  errorCode: string;
  stateAfter: string | null;
  /** Count of concrete injected events when physical observation is required. */
  observedEvents?: number;
}>;

export type PhysicalBrowserLifecycleTraceV1 = Readonly<{
  prepare: true;
  forward: true;
  backward: true;
  optimizerStep: true;
  checkpointResume: true;
  exportReload: true;
  nativeArtifactParity: true;
  completedSteps: 1;
  checkpointSha256: string;
  artifactSha256: string;
  nativeArtifactSha256: string;
  nativeReferenceDigest: string;
  receipts: readonly Readonly<{
    operation: string;
    completedSteps: number;
    peakResidentBytes: number;
    buildId: string;
    physicalDevice: string;
  }>[];
}>;

export type PhysicalBrowserTrainingScenarioV1 = Readonly<{
  schemaId: "tritium.physical_browser_training_scenario";
  schemaVersion: 1;
  scenarioId: "salt-ste-sgd-256-v1";
  completedSteps: 1;
  model: WebTrainingModelV1;
  config: WebTrainingConfigV1;
  batch: TrainingBatchV1;
}>;

export type PhysicalBrowserTrainingLaneTraceV1 = Readonly<{
  schemaId: "tritium.physical_browser_training_lane_trace";
  schemaVersion: 1;
  scenarioId: PhysicalBrowserTrainingScenarioV1["scenarioId"];
  implementation: "webgpu";
  manifestDigest: typeof TRAINING_MANIFEST_DIGEST_V2;
  vectorDigest: typeof TRAINING_VECTOR_DIGEST_V2;
  physicalDevice: string;
  buildId: string;
  adapter: PhysicalBrowserAdapterIdentityV1;
  limits: PhysicalBrowserLimitsV1;
  vector: WebGpuVectorConformanceTraceV1;
  lifecycle: PhysicalBrowserLifecycleTraceV1;
  faults: Readonly<{
    deviceLoss: PhysicalBrowserFaultTraceV1;
    allocationFailure: PhysicalBrowserFaultTraceV1;
    malformedCheckpoint: PhysicalBrowserFaultTraceV1;
    malformedSalt: PhysicalBrowserFaultTraceV1;
    cancellation: PhysicalBrowserFaultTraceV1;
    outOfOrder: PhysicalBrowserFaultTraceV1;
  }>;
  explicitReadbacks: number;
  steadyStateReadbacks: 0;
  wasmDispatches: 0;
  peakBufferBytes: number;
  executionDigest: string;
}>;

export type PhysicalBrowserTrainingLaneOptionsV1 = Readonly<{
  nativeArtifact: Uint8Array;
  nativeReferenceDigest: string;
  maxPeakBytes?: number;
}>;

type BrowserGpuAdapter = Readonly<Record<PropertyKey, unknown>> & {
  readonly info?: unknown;
  readonly limits?: unknown;
  requestDevice(): Promise<unknown>;
};

type ReadbackLedger = {
  explicit: number;
  steady: number;
};

type AcquiredDevice = Readonly<{
  identity: PhysicalBrowserAdapterIdentityV1;
  limits: PhysicalBrowserLimitsV1;
  device: WebGpuDevicePortV1;
}>;

function fail(
  code: PhysicalBrowserQualificationErrorCode,
  message: string,
): never {
  throw new PhysicalBrowserQualificationError(code, message);
}

function record(value: unknown): value is Readonly<Record<PropertyKey, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function member(value: object, name: PropertyKey): unknown {
  try {
    return Reflect.get(value, name);
  } catch {
    return undefined;
  }
}

function nonEmpty(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    fail("device_identity", `${label} must be a non-empty string`);
  }
  return value.trim();
}

function positiveLimit(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    fail("device_identity", `${label} must be a positive safe integer`);
  }
  return value as number;
}

function digest(bytes: Uint8Array): string {
  return bytesToHex(sha256(bytes));
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const recordValue = value as Readonly<Record<string, unknown>>;
    return `{${Object.keys(recordValue).sort().map((key) =>
      `${JSON.stringify(key)}:${canonicalJson(recordValue[key])}`
    ).join(",")}}`;
  }
  return JSON.stringify(value);
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= left[index]! ^ right[index]!;
  }
  return difference === 0;
}

function optionsSnapshot(
  options: PhysicalBrowserTrainingLaneOptionsV1,
): Required<PhysicalBrowserTrainingLaneOptionsV1> {
  if (!record(options)) fail("invalid_options", "physical browser options are invalid");
  const keys = Reflect.ownKeys(options);
  if (keys.some((key) => typeof key !== "string" || ![
    "maxPeakBytes",
    "nativeArtifact",
    "nativeReferenceDigest",
  ].includes(key))) {
    fail("invalid_options", "physical browser options contain unknown fields");
  }
  const artifact = member(options, "nativeArtifact");
  const reference = member(options, "nativeReferenceDigest");
  const peak = member(options, "maxPeakBytes") ?? MAX_PEAK_BYTES;
  if (!(artifact instanceof Uint8Array) || artifact.byteLength === 0) {
    fail("invalid_options", "native artifact must be non-empty bytes");
  }
  if (typeof reference !== "string" || !/^[0-9a-f]{64}$/.test(reference)) {
    fail("invalid_options", "native reference digest must be lowercase SHA-256");
  }
  if (!Number.isSafeInteger(peak) || (peak as number) <= 0) {
    fail("invalid_options", "physical browser peak ceiling must be positive");
  }
  if ((peak as number) < MAX_PEAK_BYTES) {
    fail("invalid_options", "physical browser peak ceiling is below the frozen scenario");
  }
  return Object.freeze({
    nativeArtifact: Uint8Array.from(artifact),
    nativeReferenceDigest: reference,
    maxPeakBytes: peak as number,
  });
}

function physicalDeviceIdentity(identity: PhysicalBrowserAdapterIdentityV1): string {
  return [identity.vendor, identity.architecture, identity.device, identity.description]
    .join(":");
}

const TENSORS = Object.freeze([
  Object.freeze({ id: "target", dtype: "f32" as const, shape: Object.freeze([1, 256]), role: "batch" as const, aliasOf: null }),
  Object.freeze({ id: "weight", dtype: "f32" as const, shape: Object.freeze([1, 256]), role: "parameter" as const, aliasOf: null }),
  Object.freeze({ id: "gradient", dtype: "f32" as const, shape: Object.freeze([1, 256]), role: "gradient" as const, aliasOf: null }),
  Object.freeze({ id: "quant", dtype: "f32" as const, shape: Object.freeze([1, 256]), role: "activation" as const, aliasOf: null }),
  Object.freeze({ id: "loss", dtype: "f32" as const, shape: Object.freeze([]), role: "result" as const, aliasOf: null }),
]);

const OPERATIONS = Object.freeze([
  Object.freeze({
    id: "salt",
    operation: "graph.salt_ste",
    inputs: Object.freeze(["weight"]),
    outputs: Object.freeze(["quant"]),
    attributes: Object.freeze([
      Object.freeze({ name: "rows", kind: "u64" as const, value: 1 }),
      Object.freeze({ name: "cols", kind: "u64" as const, value: 256 }),
      Object.freeze({ name: "planes", kind: "u64" as const, value: 2 }),
    ]),
  }),
  Object.freeze({
    id: "mse",
    operation: "loss.mse",
    inputs: Object.freeze(["quant", "target"]),
    outputs: Object.freeze(["loss"]),
    attributes: Object.freeze([]),
  }),
  Object.freeze({
    id: "sgd",
    operation: "optimizer.sgd",
    inputs: Object.freeze(["weight", "gradient"]),
    outputs: Object.freeze(["weight"]),
    attributes: Object.freeze([
      Object.freeze({ name: "step", kind: "u64" as const, value: 0 }),
      Object.freeze({ name: "lr", kind: "f32" as const, value: 0.1 }),
    ]),
  }),
]);

const RECIPE = Object.freeze({
  schemaId: "tritium.training_recipe" as const,
  schemaVersion: 1 as const,
  tensors: TENSORS,
  operations: OPERATIONS,
});

const CONFIG = Object.freeze({
  backend: "webgpu" as const,
  allowWasmFallback: false,
  maxResidentBytes: MAX_PEAK_BYTES,
  seed: 7,
  requiredOperations: Object.freeze([
    "graph.salt_ste",
    "loss.mse",
    "optimizer.sgd",
    "lifecycle.checkpoint",
    "lifecycle.resume",
    "lifecycle.export",
    "lifecycle.reload",
  ]),
});

export function physicalBrowserTrainingScenarioV1(): PhysicalBrowserTrainingScenarioV1 {
  const weights = Float32Array.from(
    { length: 256 },
    (_, index) => (index % 9 - 4) / 8,
  );
  const model = Object.freeze({
    schemaId: "tritium.web_training_model" as const,
    schemaVersion: 1 as const,
    recipe: RECIPE,
    payload: encodeWebTrainingPayload({ weight: weights }),
  });
  const batch = Object.freeze({
    inputs: Object.freeze({ target: new Float32Array(256) }),
  });
  return Object.freeze({
    schemaId: "tritium.physical_browser_training_scenario" as const,
    schemaVersion: 1 as const,
    scenarioId: "salt-ste-sgd-256-v1" as const,
    completedSteps: 1 as const,
    model,
    config: CONFIG,
    batch,
  });
}

async function webglHardwareIdentity(): Promise<Readonly<{
  vendor: string;
  renderer: string;
}> | undefined> {
  const documentValue = member(globalThis, "document");
  const createElement = record(documentValue) ? member(documentValue, "createElement") : undefined;
  if (!record(documentValue) || typeof createElement !== "function") return undefined;
  let canvas: unknown;
  try {
    canvas = Reflect.apply(createElement, documentValue, ["canvas"]);
  } catch {
    return undefined;
  }
  const getContext = record(canvas) ? member(canvas, "getContext") : undefined;
  if (!record(canvas) || typeof getContext !== "function") return undefined;
  let context: unknown;
  try {
    context = Reflect.apply(getContext, canvas, ["webgl2"]);
  } catch {
    return undefined;
  }
  if (!record(context)) return undefined;
  const getExtension = member(context, "getExtension");
  const getParameter = member(context, "getParameter");
  if (typeof getExtension !== "function" || typeof getParameter !== "function") return undefined;
  let extension: unknown;
  try {
    extension = Reflect.apply(getExtension, context, ["WEBGL_debug_renderer_info"]);
  } catch {
    return undefined;
  }
  if (!record(extension)) return undefined;
  const vendorEnum = member(extension, "UNMASKED_VENDOR_WEBGL");
  const rendererEnum = member(extension, "UNMASKED_RENDERER_WEBGL");
  if (!Number.isSafeInteger(vendorEnum) || !Number.isSafeInteger(rendererEnum)) return undefined;
  try {
    const vendor = Reflect.apply(getParameter, context, [vendorEnum]);
    const renderer = Reflect.apply(getParameter, context, [rendererEnum]);
    if (typeof vendor !== "string" || vendor.trim().length === 0 ||
        typeof renderer !== "string" || renderer.trim().length === 0) return undefined;
    return Object.freeze({ vendor: vendor.trim(), renderer: renderer.trim() });
  } catch {
    return undefined;
  }
}

async function adapterInfo(adapter: BrowserGpuAdapter): Promise<PhysicalBrowserAdapterIdentityV1> {
  let info = member(adapter, "info");
  const hasStandardIdentity = (candidate: unknown): candidate is Readonly<Record<PropertyKey, unknown>> =>
    record(candidate) &&
    ["vendor", "architecture", "device", "description"].every((key) => {
      const value = member(candidate, key);
      return typeof value === "string" && value.trim().length > 0;
    });
  if (!hasStandardIdentity(info)) {
    const requestAdapterInfo = member(adapter, "requestAdapterInfo");
    if (typeof requestAdapterInfo === "function") {
      info = await Reflect.apply(requestAdapterInfo, adapter, []);
    }
  }
  if (!record(info)) fail("device_identity", "WebGPU adapter exposes no identity");
  if (member(info, "isFallbackAdapter") !== false) {
    fail("device_identity", "WebGPU adapter is fallback or cannot prove physical execution");
  }

  // Firefox 153 exposes its physical Vulkan identity through wgpu-prefixed
  // fields in privileged diagnostics while leaving standard GPUAdapterInfo
  // strings empty. Named WebGPU fields remain authoritative when exposed.
  const wgpuName = member(info, "wgpuName");
  const wgpuBackend = member(info, "wgpuBackend");
  const wgpuDeviceType = member(info, "wgpuDeviceType");
  const wgpuDriver = member(info, "wgpuDriver");
  const wgpuDriverInfo = member(info, "wgpuDriverInfo");
  const firefoxPhysical = typeof wgpuName === "string" && wgpuName.trim().length > 0 &&
    typeof wgpuBackend === "string" && wgpuBackend.trim().length > 0 &&
    (wgpuDeviceType === "DiscreteGpu" || wgpuDeviceType === "IntegratedGpu");
  const webgl = firefoxPhysical ? undefined : await webglHardwareIdentity();
  const text = (value: unknown): string | undefined =>
    typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
  const standardVendor = text(member(info, "vendor"));
  const standardArchitecture = text(member(info, "architecture"));
  const standardDevice = text(member(info, "device"));
  const standardDescription = text(member(info, "description"));
  const vendor = standardVendor ?? (firefoxPhysical ? text(wgpuDriver) : webgl?.vendor);
  const architecture = standardArchitecture ?? (
    firefoxPhysical ? `${String(wgpuBackend).trim()}/${String(wgpuDeviceType).trim()}` : undefined
  ) ?? (webgl === undefined ? undefined : "WebGL2/WebGPU");
  const device = standardDevice ?? (firefoxPhysical ? text(wgpuName) : webgl?.renderer);
  const description = standardDescription ?? (
    firefoxPhysical
      ? [text(wgpuName), text(wgpuBackend), text(wgpuDriver), text(wgpuDriverInfo)]
        .filter((value): value is string => value !== undefined)
        .join(" ")
      : webgl === undefined ? undefined : `${webgl.renderer} (${webgl.vendor}); WebGPU non-fallback`
  );
  const identity = Object.freeze({
    vendor: nonEmpty(vendor, "adapter.vendor"),
    architecture: nonEmpty(architecture, "adapter.architecture"),
    device: nonEmpty(device, "adapter.device"),
    description: nonEmpty(description, "adapter.description"),
    software: false as const,
  });
  const joined = Object.values(identity).join(" ").toLowerCase();
  if (SOFTWARE_MARKERS.some((marker) => joined.includes(marker))) {
    fail("device_identity", "WebGPU adapter identifies software execution");
  }
  return identity;
}

function admittedLimits(device: Readonly<Record<PropertyKey, unknown>>): PhysicalBrowserLimitsV1 {
  const limits = member(device, "limits");
  if (!record(limits)) fail("device_identity", "WebGPU device limits are unavailable");
  return Object.freeze({
    maxBufferSize: positiveLimit(member(limits, "maxBufferSize"), "limits.maxBufferSize"),
    maxStorageBufferBindingSize: positiveLimit(
      member(limits, "maxStorageBufferBindingSize"),
      "limits.maxStorageBufferBindingSize",
    ),
    maxComputeWorkgroupsPerDimension: positiveLimit(
      member(limits, "maxComputeWorkgroupsPerDimension"),
      "limits.maxComputeWorkgroupsPerDimension",
    ),
    maxStorageBuffersPerShaderStage: positiveLimit(
      member(limits, "maxStorageBuffersPerShaderStage"),
      "limits.maxStorageBuffersPerShaderStage",
    ),
  });
}

function instrumentDevice(
  device: Readonly<Record<PropertyKey, unknown>>,
  ledger: ReadbackLedger,
): WebGpuDevicePortV1 {
  const createBuffer = member(device, "createBuffer");
  if (typeof createBuffer !== "function") {
    fail("instrumentation", "WebGPU createBuffer cannot be instrumented");
  }
  return new Proxy(device, {
    get(target, property) {
      if (property === "createBuffer") {
        return (descriptor: Readonly<{ label?: string }>) => {
          const buffer = Reflect.apply(createBuffer, target, [descriptor]);
          if (!record(buffer)) fail("instrumentation", "WebGPU returned an invalid buffer");
          const mapAsync = member(buffer, "mapAsync");
          if (typeof mapAsync === "function") {
            const label = typeof descriptor.label === "string" ? descriptor.label : "";
            try {
              Object.defineProperty(buffer, "mapAsync", {
                configurable: true,
                value: (...args: readonly unknown[]) => {
                  if (label.startsWith("tritium:readback:")) ledger.explicit += 1;
                  else ledger.steady += 1;
                  return Reflect.apply(mapAsync, buffer, args);
                },
              });
            } catch {
              fail("instrumentation", "WebGPU buffer mapping cannot be instrumented");
            }
          }
          return buffer;
        };
      }
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  }) as unknown as WebGpuDevicePortV1;
}

async function acquireDevice(ledger: ReadbackLedger): Promise<AcquiredDevice> {
  const navigatorValue = member(globalThis, "navigator");
  const gpu = record(navigatorValue) ? member(navigatorValue, "gpu") : undefined;
  const requestAdapter = record(gpu) ? member(gpu, "requestAdapter") : undefined;
  if (!record(gpu) || typeof requestAdapter !== "function") {
    fail("adapter_unavailable", "navigator.gpu.requestAdapter is unavailable");
  }
  const candidate = await Reflect.apply(requestAdapter, gpu, [{
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  }]);
  if (!record(candidate)) fail("adapter_unavailable", "WebGPU returned no adapter");
  const adapter = candidate as BrowserGpuAdapter;
  const identity = await adapterInfo(adapter);
  const requestDevice = member(adapter, "requestDevice");
  if (typeof requestDevice !== "function") {
    fail("adapter_unavailable", "WebGPU adapter cannot request a device");
  }
  const rawDevice = await Reflect.apply(requestDevice, adapter, []);
  if (!record(rawDevice)) fail("adapter_unavailable", "WebGPU returned no device");
  let device: WebGpuDevicePortV1;
  try {
    device = instrumentDevice(rawDevice, ledger);
    const limits = admittedLimits(rawDevice);
    return Object.freeze({ identity, limits, device });
  } catch (error) {
    const destroy = member(rawDevice, "destroy");
    if (typeof destroy === "function") {
      try { Reflect.apply(destroy, rawDevice, []); } catch { /* preserve primary */ }
    }
    throw error;
  }
}

function sameIdentity(expected: AcquiredDevice, actual: AcquiredDevice): void {
  if (JSON.stringify(expected.identity) !== JSON.stringify(actual.identity) ||
      JSON.stringify(expected.limits) !== JSON.stringify(actual.limits)) {
    actual.device.destroy();
    fail("device_identity", "qualification devices changed adapter identity or limits");
  }
}

function receiptTrace(receipt: WebTrainingReceiptV1): PhysicalBrowserLifecycleTraceV1["receipts"][number] {
  if (receipt.physicalDevice === null) {
    fail("lifecycle", `${receipt.operation} omitted physical device identity`);
  }
  return Object.freeze({
    operation: receipt.operation,
    completedSteps: receipt.completedSteps,
    peakResidentBytes: receipt.peakResidentBytes,
    buildId: receipt.buildId,
    physicalDevice: receipt.physicalDevice,
  });
}

async function strictReload(bytes: Uint8Array, label: string): Promise<void> {
  const response = await executePortableWasmRequest(
    compilePortableReloadRequest(bytes, `browser-qualification:${label}`),
  );
  if (response.status !== "ok" || response.outputs.length !== 1 ||
      response.outputs[0]?.data.dtype !== "bytes" ||
      !equalBytes(Uint8Array.from(response.outputs[0].data.values), bytes)) {
    fail("lifecycle", `${label} failed strict SALT reload`);
  }
}

async function expectWebError(
  operation: Promise<unknown>,
  expected: readonly string[],
  label: string,
): Promise<PhysicalBrowserFaultTraceV1> {
  try {
    await operation;
  } catch (error) {
    if (!(error instanceof WebTrainingError) || !expected.includes(error.code)) {
      fail("fault_injection", `${label} returned an unexpected error`);
    }
    return Object.freeze({
      passed: true as const,
      errorCode: error.code,
      stateAfter: error.state,
    });
  }
  fail("fault_injection", `${label} unexpectedly succeeded`);
}

async function malformedSaltFault(): Promise<PhysicalBrowserFaultTraceV1> {
  const response = await executePortableWasmRequest(
    compilePortableReloadRequest(Uint8Array.of(0), "browser-qualification:malformed"),
  );
  if (response.status !== "error") {
    fail("fault_injection", "malformed SALT unexpectedly reloaded");
  }
  return Object.freeze({
    passed: true as const,
    errorCode: response.error.code,
    stateAfter: null,
  });
}

type SubmittedCancellationProbe = Readonly<{
  device: WebGpuDevicePortV1;
  submitted: Promise<void>;
  release(): void;
  submissions(): number;
}>;

function submittedCancellationDevice(device: WebGpuDevicePortV1): SubmittedCancellationProbe {
  let resolveSubmitted!: () => void;
  let resolveGate!: () => void;
  let submissionCount = 0;
  let released = false;
  const submitted = new Promise<void>((resolve) => { resolveSubmitted = resolve; });
  const gate = new Promise<void>((resolve) => { resolveGate = resolve; });
  const queue = new Proxy(device.queue as unknown as object, {
    get(target, property) {
      const value = Reflect.get(target, property, target);
      if (property === "submit") {
        if (typeof value !== "function") fail("instrumentation", "WebGPU queue submit is unavailable");
        return (commands: readonly unknown[]) => {
          Reflect.apply(value, target, [commands]);
          submissionCount += 1;
          resolveSubmitted();
        };
      }
      if (property === "onSubmittedWorkDone") {
        if (typeof value !== "function") {
          fail("instrumentation", "WebGPU queue completion signal is unavailable");
        }
        return async () => {
          const completed = Reflect.apply(value, target, []) as Promise<void>;
          if (submissionCount === 1 && !released) await Promise.all([completed, gate]);
          else await completed;
        };
      }
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
  const observedDevice = new Proxy(device as unknown as object, {
    get(target, property) {
      if (property === "queue") return queue;
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  }) as unknown as WebGpuDevicePortV1;
  return Object.freeze({
    device: observedDevice,
    submitted,
    release() {
      if (!released) {
        released = true;
        resolveGate();
      }
    },
    submissions: () => submissionCount,
  });
}

type AllocationFailureProbe = Readonly<{
  device: WebGpuDevicePortV1;
  sentinel: Error;
  hits(): number;
}>;

function allocationFailingDevice(device: WebGpuDevicePortV1): AllocationFailureProbe {
  const sentinel = new Error("injected WebGPU allocation failure");
  sentinel.name = "TritiumInjectedAllocationFailure";
  let hits = 0;
  const observedDevice = new Proxy(device as unknown as object, {
    get(target, property) {
      if (property === "createBuffer") {
        return () => {
          hits += 1;
          throw sentinel;
        };
      }
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  }) as unknown as WebGpuDevicePortV1;
  return Object.freeze({ device: observedDevice, sentinel, hits: () => hits });
}

async function runLifecycle(
  baseline: AcquiredDevice,
  options: Required<PhysicalBrowserTrainingLaneOptionsV1>,
  ledger: ReadbackLedger,
): Promise<Readonly<{
  lifecycle: PhysicalBrowserLifecycleTraceV1;
  faults: PhysicalBrowserTrainingLaneTraceV1["faults"];
  peakBufferBytes: number;
}>> {
  await strictReload(options.nativeArtifact, "native-reference");
  const scenario = physicalBrowserTrainingScenarioV1();
  const physicalDevice = physicalDeviceIdentity(baseline.identity);
  const buildId = `wgsl:${webGpuKernelCandidateBundleV1().bundleSha256}:browser-qualification:${scenario.scenarioId}`;
  const acquired = await acquireDevice(ledger);
  sameIdentity(baseline, acquired);
  const cancellationProbe = submittedCancellationDevice(acquired.device);
  const adapter = createWebGpuTrainingAdapter(cancellationProbe.device, {
    buildId,
    physicalDevice,
    maxResidentBytes: Math.min(options.maxPeakBytes, acquired.limits.maxBufferSize),
  });
  const receipts: WebTrainingReceiptV1[] = [];
  const session = await prepareTraining(scenario.model, scenario.config, adapter);
  try {
    const outOfOrder = await expectWebError(
      session.step(), ["invalid_state"], "out-of-order lifecycle",
    );
    const cancellationController = new AbortController();
    const cancellationOperation = session.forward(
      scenario.batch, { signal: cancellationController.signal },
    );
    let cancellation: PhysicalBrowserFaultTraceV1;
    try {
      const firstEvent = await Promise.race([
        cancellationProbe.submitted.then(() => "submitted" as const),
        cancellationOperation.then(
          () => "settled" as const,
          () => "rejected" as const,
        ),
      ]);
      if (firstEvent !== "submitted") {
        fail("fault_injection", "cancellation operation settled before GPU submission");
      }
      cancellationController.abort();
      cancellationProbe.release();
      const admitted = await expectWebError(
        cancellationOperation,
        ["cancelled"],
        "submitted cancellation",
      );
      if (admitted.stateAfter !== "prepared" || cancellationProbe.submissions() < 1) {
        fail("fault_injection", "submitted cancellation did not preserve reusable state");
      }
      cancellation = Object.freeze({
        ...admitted,
        observedEvents: cancellationProbe.submissions(),
      });
    } finally {
      cancellationProbe.release();
    }
    const result = await session.forward(scenario.batch);
    receipts.push(result.receipt);
    receipts.push(await session.backward(result));
    receipts.push(await session.step());
    const checkpoint = await session.checkpoint();
    receipts.push(checkpoint.receipt);
    receipts.push(await session.resume(checkpoint.bytes));
    const artifact = await session.export();
    receipts.push(artifact.receipt);
    await strictReload(artifact.bytes, "browser-export");
    if (!equalBytes(artifact.bytes, options.nativeArtifact)) {
      fail("native_artifact_parity", "browser artifact differs from native CPU reference");
    }

    const malformedCheckpointDevice = await acquireDevice(ledger);
    sameIdentity(baseline, malformedCheckpointDevice);
    const malformedCheckpointSession = await prepareTraining(
      scenario.model,
      scenario.config,
      createWebGpuTrainingAdapter(malformedCheckpointDevice.device, {
        buildId,
        physicalDevice,
        maxResidentBytes: Math.min(options.maxPeakBytes, malformedCheckpointDevice.limits.maxBufferSize),
      }),
    );
    let malformedCheckpoint: PhysicalBrowserFaultTraceV1;
    try {
      malformedCheckpoint = await expectWebError(
        malformedCheckpointSession.resume(Uint8Array.of(0)),
        ["adapter_failure", "invalid_receipt", "invalid_schema"],
        "malformed checkpoint",
      );
    } finally {
      await malformedCheckpointSession.dispose();
    }

    const allocationDevice = await acquireDevice(ledger);
    sameIdentity(baseline, allocationDevice);
    const allocationProbe = allocationFailingDevice(allocationDevice.device);
    let allocationError: unknown;
    let unexpectedAllocationSession: Awaited<ReturnType<typeof prepareTraining>> | null = null;
    try {
      unexpectedAllocationSession = await prepareTraining(
        scenario.model,
        scenario.config,
        createWebGpuTrainingAdapter(allocationProbe.device, {
          buildId,
          physicalDevice,
          maxResidentBytes: Math.min(options.maxPeakBytes, allocationDevice.limits.maxBufferSize),
        }),
      );
    } catch (error) {
      allocationError = error;
    } finally {
      if (unexpectedAllocationSession !== null) await unexpectedAllocationSession.dispose();
      allocationDevice.device.destroy();
    }
    const allocationObserved = allocationError === allocationProbe.sentinel ||
      (allocationError instanceof WebTrainingError &&
       allocationError.code === "capability_mismatch" &&
       allocationError.message.includes(allocationProbe.sentinel.message));
    if (!allocationObserved || allocationProbe.hits() !== 1) {
      fail("fault_injection", "allocation injection did not observe its unique sentinel exactly once");
    }
    const allocationFailure = Object.freeze({
      passed: true as const,
      errorCode: "injected_allocation_failure",
      stateAfter: null,
      observedEvents: allocationProbe.hits(),
    });

    const lossDevice = await acquireDevice(ledger);
    sameIdentity(baseline, lossDevice);
    const lossSession = await prepareTraining(
      scenario.model,
      scenario.config,
      createWebGpuTrainingAdapter(lossDevice.device, {
        buildId,
        physicalDevice,
        maxResidentBytes: Math.min(options.maxPeakBytes, lossDevice.limits.maxBufferSize),
      }),
    );
    lossDevice.device.destroy();
    await lossDevice.device.lost;
    const deviceLoss = await expectWebError(
      lossSession.forward(scenario.batch), ["device_lost"], "device loss",
    );
    await lossSession.dispose();

    const admittedReceipts = Object.freeze(receipts.map(receiptTrace));
    if (admittedReceipts.some((receipt) =>
      receipt.physicalDevice !== physicalDevice ||
      receipt.buildId !== buildId
    )) {
      fail("lifecycle", "lifecycle receipt identity drifted");
    }
    const peakBufferBytes = Math.max(
      ...admittedReceipts.map((receipt) => receipt.peakResidentBytes),
    );
    return Object.freeze({
      lifecycle: Object.freeze({
        prepare: true as const,
        forward: true as const,
        backward: true as const,
        optimizerStep: true as const,
        checkpointResume: true as const,
        exportReload: true as const,
        nativeArtifactParity: true as const,
        completedSteps: 1 as const,
        checkpointSha256: digest(checkpoint.bytes),
        artifactSha256: digest(artifact.bytes),
        nativeArtifactSha256: digest(options.nativeArtifact),
        nativeReferenceDigest: options.nativeReferenceDigest,
        receipts: admittedReceipts,
      }),
      faults: Object.freeze({
        deviceLoss,
        allocationFailure,
        malformedCheckpoint,
        malformedSalt: await malformedSaltFault(),
        cancellation,
        outOfOrder,
      }),
      peakBufferBytes,
    });
  } finally {
    await session.dispose();
  }
}

/**
 * Execute complete candidate-bound physical browser qualification. Acquires and
 * destroys every WebGPU device it uses. Structural or software adapters fail.
 */
export async function runPhysicalBrowserTrainingLaneV1(
  rawOptions: PhysicalBrowserTrainingLaneOptionsV1,
): Promise<PhysicalBrowserTrainingLaneTraceV1> {
  const options = optionsSnapshot(rawOptions);
  const ledger: ReadbackLedger = { explicit: 0, steady: 0 };
  const vectorDevice = await acquireDevice(ledger);
  const physicalDevice = physicalDeviceIdentity(vectorDevice.identity);
  let vector: WebGpuVectorConformanceTraceV1;
  try {
    vector = await runWebGpuVectorConformanceV1(vectorDevice.device, {
      maxPeakBytes: options.maxPeakBytes,
      physicalDevice,
    });
  } catch (error) {
    if (error instanceof PhysicalBrowserQualificationError) throw error;
    fail(
      "vector_conformance",
      `physical WebGPU vector conformance failed: ${
        error instanceof Error ? error.message : "unknown failure"
      }`,
    );
  }
  const execution = await runLifecycle(vectorDevice, options, ledger);
  if (ledger.steady !== 0 || vector.wasmDispatches !== 0) {
    fail("instrumentation", "qualification observed hidden readback or WASM tensor dispatch");
  }
  if (ledger.explicit <= vector.explicitReadbacks) {
    fail("instrumentation", "physical lifecycle produced no observed explicit readbacks");
  }
  const unsigned = {
    schemaId: "tritium.physical_browser_training_lane_trace" as const,
    schemaVersion: 1 as const,
    scenarioId: "salt-ste-sgd-256-v1" as const,
    implementation: "webgpu" as const,
    manifestDigest: TRAINING_MANIFEST_DIGEST_V2,
    vectorDigest: TRAINING_VECTOR_DIGEST_V2,
    physicalDevice,
    buildId: `wgsl:${webGpuKernelCandidateBundleV1().bundleSha256}:browser-qualification:salt-ste-sgd-256-v1`,
    adapter: vectorDevice.identity,
    limits: vectorDevice.limits,
    vector,
    lifecycle: execution.lifecycle,
    faults: execution.faults,
    explicitReadbacks: ledger.explicit,
    steadyStateReadbacks: 0 as const,
    wasmDispatches: 0 as const,
    peakBufferBytes: Math.max(vector.peakBufferBytes, execution.peakBufferBytes),
  };
  return Object.freeze({
    ...unsigned,
    executionDigest: digest(UTF8.encode(canonicalJson(unsigned))),
  });
}

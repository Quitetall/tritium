import {
  WEBGPU_KERNEL_MODULES_V1,
  WEBGPU_KERNEL_BUNDLE_SHA256_V1,
  WEBGPU_DISPATCH_CATALOG_SHA256_V1,
  WEBGPU_DISPATCH_FORMS_V1,
  WEBGPU_OPERATION_MODULE_DEPENDENCIES_V1,
} from "./generated-webgpu-kernels.ts";
import { WebTrainingError } from "./session.ts";

export interface WebGpuKernelModuleV1 {
  readonly id: string;
  readonly sha256: string;
  readonly source: string;
  readonly bindings: readonly WebGpuKernelBindingV1[];
  readonly entryPoints: Readonly<Record<string, readonly [number, number, number]>>;
}

export interface WebGpuKernelBindingV1 {
  readonly group: number;
  readonly binding: number;
  readonly addressSpace: "uniform" | "storage";
  readonly access: "read" | "read_write" | null;
}

export interface WebGpuKernelCandidateBundleV1 {
  readonly schemaId: "tritium.webgpu_kernel_candidate_bundle";
  readonly schemaVersion: 1;
  readonly bundleSha256: string;
  readonly modules: Readonly<Record<string, WebGpuKernelModuleV1>>;
  readonly candidateOperationModuleDependencies: Readonly<
    Record<string, readonly string[]>
  >;
}

export type WebGpuDispatchExecutionV1 = "forward" | "vjp" | "step";
export type WebGpuDispatchRepeatV1 = "once" | "per_output";
export type WebGpuDispatchGeometryV1 =
  | "linear_input_64"
  | "linear_output_64"
  | "linear_parameter_64"
  | "linear_primary_input_64"
  | "linear_rows_64"
  | "optimizer_blocks_256"
  | "rope_pairs_64"
  | "single";

export interface WebGpuDispatchStageV1 {
  readonly moduleId: string;
  readonly entryPoint: string;
  readonly selector: number | null;
  readonly dispatch: WebGpuDispatchGeometryV1;
  readonly repeat: WebGpuDispatchRepeatV1;
}

export interface WebGpuDispatchFormV1 {
  readonly operation: string;
  readonly execution: WebGpuDispatchExecutionV1;
  readonly stages: readonly WebGpuDispatchStageV1[];
}

export interface WebGpuDispatchCatalogV1 {
  readonly schemaId: "tritium.webgpu_dispatch_catalog";
  readonly schemaVersion: 1;
  readonly sha256: string;
  readonly forms: Readonly<Record<string, WebGpuDispatchFormV1>>;
}

const BUNDLE: WebGpuKernelCandidateBundleV1 = Object.freeze({
  schemaId: "tritium.webgpu_kernel_candidate_bundle",
  schemaVersion: 1,
  bundleSha256: WEBGPU_KERNEL_BUNDLE_SHA256_V1,
  modules: WEBGPU_KERNEL_MODULES_V1,
  candidateOperationModuleDependencies: WEBGPU_OPERATION_MODULE_DEPENDENCIES_V1,
});

const DISPATCH_CATALOG: WebGpuDispatchCatalogV1 = Object.freeze({
  schemaId: "tritium.webgpu_dispatch_catalog",
  schemaVersion: 1,
  sha256: WEBGPU_DISPATCH_CATALOG_SHA256_V1,
  forms: WEBGPU_DISPATCH_FORMS_V1,
});

/** Return immutable candidate WGSL inputs staged for browser dispatch work. */
export function webGpuKernelCandidateBundleV1(): WebGpuKernelCandidateBundleV1 {
  return BUNDLE;
}

/** Return immutable 57-form WebGPU pipeline-stage metadata. */
export function webGpuDispatchCatalogV1(): WebGpuDispatchCatalogV1 {
  return DISPATCH_CATALOG;
}

/** Resolve one frozen operation/execution dispatch form. */
export function webGpuDispatchFormV1(
  operation: string,
  execution: WebGpuDispatchExecutionV1,
): WebGpuDispatchFormV1 {
  const key = `${operation}|${execution}`;
  const form = DISPATCH_CATALOG.forms[key];
  if (form === undefined) {
    throw new WebTrainingError(
      "capability_mismatch",
      `no WebGPU dispatch form exists for ${key}`,
    );
  }
  return form;
}

/** Resolve the curated candidate modules associated with one tensor operation. */
export function webGpuCandidateModulesForOperationV1(
  operation: string,
): readonly WebGpuKernelModuleV1[] {
  if (!Object.hasOwn(BUNDLE.candidateOperationModuleDependencies, operation)) {
    throw new WebTrainingError(
      "capability_mismatch",
      `no WebGPU candidate dependency entry exists for ${operation}`,
    );
  }
  const moduleIds = BUNDLE.candidateOperationModuleDependencies[operation];
  if (moduleIds === undefined) {
    throw new WebTrainingError(
      "capability_mismatch",
      `no WebGPU candidate dependency entry exists for ${operation}`,
    );
  }
  return Object.freeze(
    moduleIds.map((moduleId) => {
      const module = BUNDLE.modules[moduleId];
      if (module === undefined) {
        throw new WebTrainingError(
          "capability_mismatch",
          `WebGPU kernel module ${moduleId} is absent from the candidate bundle`,
        );
      }
      return module;
    }),
  );
}

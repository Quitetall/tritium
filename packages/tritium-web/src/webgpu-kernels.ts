import {
  WEBGPU_KERNEL_MODULES_V1,
  WEBGPU_KERNEL_BUNDLE_SHA256_V1,
  WEBGPU_OPERATION_MODULE_DEPENDENCIES_V1,
} from "./generated-webgpu-kernels.ts";
import { WebTrainingError } from "./session.ts";

export interface WebGpuKernelModuleV1 {
  readonly id: string;
  readonly sha256: string;
  readonly source: string;
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

const BUNDLE: WebGpuKernelCandidateBundleV1 = Object.freeze({
  schemaId: "tritium.webgpu_kernel_candidate_bundle",
  schemaVersion: 1,
  bundleSha256: WEBGPU_KERNEL_BUNDLE_SHA256_V1,
  modules: WEBGPU_KERNEL_MODULES_V1,
  candidateOperationModuleDependencies: WEBGPU_OPERATION_MODULE_DEPENDENCIES_V1,
});

/** Return immutable candidate WGSL inputs staged for browser dispatch work. */
export function webGpuKernelCandidateBundleV1(): WebGpuKernelCandidateBundleV1 {
  return BUNDLE;
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

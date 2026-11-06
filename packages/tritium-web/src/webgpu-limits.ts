import { WEBGPU_DISPATCH_FORMS_V1, WEBGPU_KERNEL_MODULES_V1 } from "./generated-webgpu-kernels.ts";

/**
 * Return the minimum device limits needed to compile every frozen dispatch
 * form. WebGPU devices otherwise expose only their default limits, even when
 * the physical adapter supports more (for example, attention's nine storage
 * bindings exceed the default of eight).
 */
export function webGpuRequiredDeviceLimitsV1(): Readonly<Record<string, number>> {
  let maxBindingsPerBindGroup = 0;
  let maxStorageBuffersPerShaderStage = 0;
  let maxUniformBuffersPerShaderStage = 0;

  for (const form of Object.values(WEBGPU_DISPATCH_FORMS_V1)) {
    for (const stage of form.stages) {
      const module = Reflect.get(WEBGPU_KERNEL_MODULES_V1, stage.moduleId) as
        | Readonly<{
            entryPointBindings: Readonly<
              Record<
                string,
                readonly {
                  group: number;
                  addressSpace: "uniform" | "storage";
                }[]
              >
            >;
          }>
        | undefined;
      const bindings = module?.entryPointBindings[stage.entryPoint];
      if (bindings === undefined) {
        throw new Error(
          `WebGPU dispatch references missing bindings for ${stage.moduleId}:${stage.entryPoint}`,
        );
      }
      const groupCounts = new Map<number, number>();
      let storage = 0;
      let uniform = 0;
      for (const binding of bindings) {
        groupCounts.set(binding.group, (groupCounts.get(binding.group) ?? 0) + 1);
        if (binding.addressSpace === "storage") storage += 1;
        else uniform += 1;
      }
      maxBindingsPerBindGroup = Math.max(maxBindingsPerBindGroup, ...groupCounts.values());
      maxStorageBuffersPerShaderStage = Math.max(maxStorageBuffersPerShaderStage, storage);
      maxUniformBuffersPerShaderStage = Math.max(maxUniformBuffersPerShaderStage, uniform);
    }
  }

  return Object.freeze({
    maxBindingsPerBindGroup,
    maxStorageBuffersPerShaderStage,
    maxUniformBuffersPerShaderStage,
  });
}

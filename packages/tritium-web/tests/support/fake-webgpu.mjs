class FakeBuffer {
  constructor(size, device, label) {
    this.size = size;
    this.label = label;
    this.bytes = new Uint8Array(size);
    this.device = device;
  }
  async mapAsync() {
    this.device.maps += 1;
    if (this.device.mapGate !== null) await this.device.mapGate;
  }
  getMappedRange(offset = 0, size = this.size) {
    return this.bytes.slice(offset, offset + size).buffer;
  }
  unmap() {}
  destroy() { this.destroyed = true; }
}

export class FakeDevice {
  constructor(overrides = {}) {
    this.limits = {
      maxBufferSize: 1 << 24,
      maxStorageBufferBindingSize: 1 << 24,
      maxComputeWorkgroupsPerDimension: 65535,
      maxBindingsPerBindGroup: 16,
      maxStorageBuffersPerShaderStage: 16,
      maxUniformBuffersPerShaderStage: 12,
      maxUniformBufferBindingSize: 65536,
      minUniformBufferOffsetAlignment: 256,
      ...overrides,
    };
    this.maps = 0;
    this.mapGate = null;
    this.submits = 0;
    this.bindGroups = 0;
    this.pipelines = 0;
    this.destroyed = false;
    this.events = [];
    this.buffers = new Map();
    this.lost = new Promise((resolve) => { this.lose = resolve; });
    this.queue = {
      writeBuffer: (buffer, offset, data) => buffer.bytes.set(data, offset),
      submit: (commands) => {
        this.submits += 1;
        for (const command of commands) command();
      },
      onSubmittedWorkDone: async () => {},
    };
  }
  createShaderModule(descriptor) { return descriptor; }
  async createComputePipelineAsync() {
    this.pipelines += 1;
    return { getBindGroupLayout: () => ({}) };
  }
  createBuffer({ label, size }) {
    const buffer = new FakeBuffer(size, this, label);
    this.buffers.set(label, buffer);
    return buffer;
  }
  createBindGroup(descriptor) {
    this.bindGroups += 1;
    return descriptor;
  }
  createCommandEncoder() {
    const copies = [];
    return {
      beginComputePass: () => ({
        setPipeline() {},
        setBindGroup() {},
        dispatchWorkgroups: () => this.events.push("dispatch"),
        end() {},
      }),
      copyBufferToBuffer: (source, sourceOffset, destination, destinationOffset, size) => {
        this.events.push(`copy:${source.label}>${destination.label}`);
        copies.push(() => destination.bytes.set(
          source.bytes.slice(sourceOffset, sourceOffset + size),
          destinationOffset,
        ));
      },
      clearBuffer: (buffer, offset = 0, size = buffer.size - offset) => {
        this.events.push(`clear:${buffer.label}`);
        copies.push(() => buffer.bytes.fill(0, offset, offset + size));
      },
      finish: () => () => copies.forEach((copy) => copy()),
    };
  }
  destroy() { this.destroyed = true; }
}

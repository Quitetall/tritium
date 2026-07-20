import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  TrainingGeometryError,
  validateTrainingOperationGeometry,
} from "../src/geometry.ts";

const corpus = JSON.parse(
  readFileSync(new URL("../../../spec/training/v1/vectors/v1.json", import.meta.url), "utf8"),
);

function f32(bits) {
  const bytes = new ArrayBuffer(4);
  new DataView(bytes).setUint32(0, bits, true);
  return new DataView(bytes).getFloat32(0, true);
}

function attribute(item) {
  const value = item.type === "f32"
    ? f32(item.bits)
    : "values" in item
      ? [...item.values]
      : item.value;
  const kind = item.type === "u32_list"
    ? "u32-list"
    : item.type === "u64_list"
      ? "u64-list"
      : item.type;
  return { name: item.name, kind, value };
}

function tensor(item, index, namespace) {
  return {
    id: `${namespace}.${index}.${item.name}`,
    dtype: item.data.dtype,
    shape: [...item.shape],
    role: "activation",
    aliasOf: null,
  };
}

function recipeCase(item) {
  const inputs = item.inputs.map((buffer, index) => tensor(buffer, index, "input"));
  const outputs = item.expected.outputs.map((buffer, index) => tensor(buffer, index, "output"));
  const attributes = item.attributes.map(attribute).map((value) =>
    value.name === "step" ? { ...value, value: 0 } : value,
  );
  return {
    operation: {
      id: item.case_id,
      operation: item.operation,
      inputs: inputs.map((value) => value.id),
      outputs: outputs.map((value) => value.id),
      attributes,
    },
    inputs,
    outputs,
  };
}

const successes = corpus.cases.filter(
  (item) =>
    !item.operation.startsWith("lifecycle.") &&
    (item.execution === "forward" || item.execution === "step") &&
    item.expected.kind === "success",
);

test("geometry validator admits every canonical forward and step success", () => {
  const covered = new Set();
  for (const item of successes) {
    const candidate = recipeCase(item);
    validateTrainingOperationGeometry(candidate.operation, candidate.inputs, candidate.outputs);
    covered.add(item.operation);
  }
  assert.equal(covered.size, 31);
});

test("every operation rejects output geometry drift", () => {
  const representatives = new Map();
  for (const item of successes) {
    if (!representatives.has(item.operation)) representatives.set(item.operation, item);
  }
  for (const [operation, item] of representatives) {
    const candidate = recipeCase(item);
    const first = candidate.outputs[0];
    const changedShape = first.shape.length === 0
      ? [1]
      : [first.shape[0] + 1, ...first.shape.slice(1)];
    const outputs = [{ ...first, shape: changedShape }, ...candidate.outputs.slice(1)];
    assert.throws(
      () => validateTrainingOperationGeometry(candidate.operation, candidate.inputs, outputs),
      TrainingGeometryError,
      operation,
    );
  }
});

const INVALID_ATTRIBUTE = {
  "graph.salt_ste": ["planes", 0],
  "graph.fsq": ["alpha", -1],
  "graph.rope": ["head_dim", 3],
  "graph.causal_mask": ["rows", 0],
  "graph.softmax": ["cols", 0],
  "graph.rmsnorm": ["eps", -1],
  "loss.softmax_cross_entropy": ["rows", 0],
  "graph.bias": ["cols", 0],
  "graph.transpose": ["rows", 0],
  "graph.slice_cols": ["len", 0],
  "graph.dense_matmul": ["m", 0],
  "graph.ternary_matmul": ["n", 0],
  "graph.concat_cols": ["lens", []],
  "graph.embedding_gather": ["vocab", 0],
  "graph.ste_surrogate": ["cols", 0],
  "graph.lsq_ste": ["cols", 0],
  "graph.attention": ["n_kv_head", 0],
  "graph.conv1d": ["stride", 0],
  "graph.conv2d": ["stride_h", 0],
  "optimizer.sgd": ["lr", -1],
  "optimizer.adamw": ["beta1", 1],
  "optimizer.cautious_adamw": ["eps", 0],
  "optimizer.int8_adamw": ["weight_decay", -1],
  "optimizer.muon": ["ns_steps", 33],
};

test("bounded attribute domains fail before allocation", () => {
  const representatives = new Map();
  for (const item of successes) {
    if (INVALID_ATTRIBUTE[item.operation] && !representatives.has(item.operation)) {
      representatives.set(item.operation, item);
    }
  }
  assert.equal(representatives.size, Object.keys(INVALID_ATTRIBUTE).length);
  for (const [operation, item] of representatives) {
    const candidate = recipeCase(item);
    const [name, value] = INVALID_ATTRIBUTE[operation];
    const changed = {
      ...candidate.operation,
      attributes: candidate.operation.attributes.map((attribute) =>
        attribute.name === name ? { ...attribute, value } : attribute,
      ),
    };
    assert.throws(
      () => validateTrainingOperationGeometry(changed, candidate.inputs, candidate.outputs),
      TrainingGeometryError,
      operation,
    );
  }
});

test("stochastic estimator seed is session-bound", () => {
  const item = successes.find((candidate) => candidate.operation === "graph.fsq");
  const candidate = recipeCase(item);
  const seed = candidate.operation.attributes.find((attribute) => attribute.name === "seed").value;
  validateTrainingOperationGeometry(candidate.operation, candidate.inputs, candidate.outputs, seed);
  assert.throws(
    () => validateTrainingOperationGeometry(
      candidate.operation,
      candidate.inputs,
      candidate.outputs,
      seed + 1,
    ),
    TrainingGeometryError,
  );
});

test("convolution scratch admits worst forward and reverse phase", () => {
  for (const operationName of ["graph.conv1d", "graph.conv2d"]) {
    const item = successes.find((candidate) => candidate.operation === operationName);
    const candidate = recipeCase(item);
    const replacements = operationName === "graph.conv1d"
      ? {
          batch: 5000,
          c_in: 1,
          c_out: 5000,
          l_in: 1,
          k: 1,
          stride: 1,
          dilation: 1,
          pad_left: 0,
          pad_right: 0,
          groups: 1,
        }
      : {
          batch: 5000,
          c_in: 1,
          c_out: 5000,
          input_h: 1,
          input_w: 1,
          kernel_h: 1,
          kernel_w: 1,
          stride_h: 1,
          stride_w: 1,
          dilation_h: 1,
          dilation_w: 1,
          pad_top: 0,
          pad_bottom: 0,
          pad_left: 0,
          pad_right: 0,
          groups: 1,
        };
    const operation = {
      ...candidate.operation,
      attributes: candidate.operation.attributes.map((attribute) => ({
        ...attribute,
        value: replacements[attribute.name],
      })),
    };
    const inputs = operationName === "graph.conv1d"
      ? candidate.inputs.map((tensor, index) => ({
          ...tensor,
          shape: index === 0 ? [5000, 1, 1] : index === 1 ? [5000, 1, 1] : [5000],
        }))
      : candidate.inputs.map((tensor, index) => ({
          ...tensor,
          shape: index === 0 ? [5000, 1, 1, 1] : index === 1 ? [5000, 1, 1, 1] : [5000],
        }));
    const outputs = [{
      ...candidate.outputs[0],
      shape: operationName === "graph.conv1d"
        ? [5000, 5000, 1]
        : [5000, 5000, 1, 1],
    }];
    assert.throws(
      () => validateTrainingOperationGeometry(operation, inputs, outputs),
      /scratch exceeds 64 MiB/,
      operationName,
    );
  }
});

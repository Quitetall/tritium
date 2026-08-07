import assert from "node:assert/strict";
import { resolve } from "node:path";
import test from "node:test";

import {
  resolveCargoTargetDirectory,
} from "../scripts/build-wasm.mjs";

test("WASM build reads guest from Cargo's effective target directory", () => {
  const repository = resolve("/tmp", "tritium-target-fixture");
  assert.equal(
    resolveCargoTargetDirectory({}, repository),
    resolve(repository, "target"),
  );
  assert.equal(
    resolveCargoTargetDirectory({ CARGO_TARGET_DIR: "build/cargo" }, repository),
    resolve(repository, "build/cargo"),
  );
  assert.equal(
    resolveCargoTargetDirectory({ CARGO_TARGET_DIR: "/var/tmp/tritium-target" }, repository),
    resolve("/var/tmp/tritium-target"),
  );
  assert.throws(
    () => resolveCargoTargetDirectory({ CARGO_TARGET_DIR: "" }, repository),
    /non-empty filesystem path/,
  );
});

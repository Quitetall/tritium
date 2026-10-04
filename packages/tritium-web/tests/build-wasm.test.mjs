import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  canonicalSourceIdentity,
  resolveCargoTargetDirectory,
  resolveEffectiveCargoTargetDirectory,
} from "../scripts/build-wasm.mjs";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");

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

test("effective WASM target directory honors Cargo configuration", async () => {
  const metadata = JSON.parse(
    execFileSync("cargo", ["metadata", "--no-deps", "--format-version", "1"], {
      cwd: repositoryRoot,
      encoding: "utf8",
    }),
  );
  assert.equal(
    await resolveEffectiveCargoTargetDirectory({}, repositoryRoot),
    resolve(metadata.target_directory),
  );
});

test("WASM build binds current clean Git identity", () => {
  assert.equal(
    canonicalSourceIdentity("a".repeat(40), ""),
    `source-git:${"a".repeat(40)}`,
  );
  assert.throws(
    () => canonicalSourceIdentity("a".repeat(40), " M packages/tritium-web/src/index.ts"),
    /clean Git worktree/,
  );
  assert.throws(
    () => canonicalSourceIdentity("not-a-revision", ""),
    /full lowercase object ID/,
  );
});

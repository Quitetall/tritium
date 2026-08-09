import assert from "node:assert/strict";
import { test } from "node:test";

import { generateNpmSbom } from "../scripts/generate-npm-sbom.mjs";

const packageJson = {
  name: "@tritium-ai/web",
  version: "1.1.0-rc.1",
  dependencies: { runtime: "1.2.3" },
};
const packageLock = {
  name: "@tritium-ai/web",
  version: "1.1.0-rc.1",
  lockfileVersion: 3,
  packages: {
    "": { name: "@tritium-ai/web", version: "1.1.0-rc.1" },
    "node_modules/runtime": {
      version: "1.2.3",
      integrity: `sha512-${Buffer.alloc(64, 1).toString("base64")}`,
      license: "MIT",
    },
    "node_modules/builder": {
      version: "4.5.6",
      integrity: `sha256-${Buffer.alloc(32, 2).toString("base64")}`,
      dev: true,
      optional: true,
    },
  },
};
const receipt = {
  source_revision: "b".repeat(40),
  artifact: {
    package: "@tritium-ai/web@1.1.0-rc.1",
    sha256: "a".repeat(64),
    bytes: 123,
  },
  evidence: { source_dirty: false, wasm_guest_digest: "c".repeat(64) },
};

test("npm SBOM binds archive, locked dependencies and runtime edge", () => {
  const first = generateNpmSbom(packageJson, packageLock, receipt, "web.tgz");
  const second = generateNpmSbom(packageJson, packageLock, receipt, "web.tgz");
  assert.deepEqual(first, second);
  assert.equal(first.metadata.component["bom-ref"], "tritium-web-node22");
  assert.equal(first.metadata.component.hashes[0].content, receipt.artifact.sha256);
  assert.deepEqual(first.dependencies, [
    { ref: "tritium-web-node22", dependsOn: ["npm:runtime@1.2.3"] },
  ]);
  assert.equal(first.components.find((item) => item.name === "builder").scope, "excluded");
});

test("npm SBOM rejects weak integrity, drift and unbound receipts", () => {
  assert.throws(
    () => generateNpmSbom(
      packageJson,
      { ...packageLock, version: "wrong" },
      receipt,
      "web.tgz",
    ),
    /package-lock identity/,
  );
  const weak = structuredClone(packageLock);
  weak.packages["node_modules/runtime"].integrity = "sha1-Zm9v";
  assert.throws(
    () => generateNpmSbom(packageJson, weak, receipt, "web.tgz"),
    /SHA-256 or stronger/,
  );
  assert.throws(
    () => generateNpmSbom(
      packageJson,
      packageLock,
      { ...receipt, artifact: { ...receipt.artifact, bytes: 0 } },
      "web.tgz",
    ),
    /archive receipt/,
  );
});

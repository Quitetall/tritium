import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";

import {
  BrowserLaneProducerError,
  WebDriverClassicClient,
  assembleBrowserLaneV1,
  qualifyBrowserLaneV1,
  validateNpmReceiptV1,
  validateNativeReferenceV1,
} from "../../../scripts/run-browser-training-lane.mjs";

const run = promisify(execFile);

// Derived from package.json, not written down. validateNpmReceiptV1 and
// validateNativeReferenceV1 compare every release-bearing field against
// PACKAGE_RELEASE, which run-browser-training-lane.mjs reads from this same
// package.json -- so a literal here is guaranteed to drift on the next bump.
// It did: nine rc.1 fixtures turned this file red at the rc.2 bump.
const RELEASE = JSON.parse(
  readFileSync(new URL("../package.json", import.meta.url), "utf8"),
).version;

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${canonical(value[key])}`
    ).join(",")}}`;
  }
  return JSON.stringify(value);
}

const CANONICAL_VECTOR_CORPUS = JSON.parse(readFileSync(
  new URL("../../../crates/tritium-spec/data/training/v2/vectors/v2.json", import.meta.url),
  "utf8",
));

function canonicalVectorCases() {
  return CANONICAL_VECTOR_CORPUS.cases.map((item) => {
    const invalid = item.expected.kind === "error";
    const codec = !invalid && ["checkpoint", "resume", "export", "reload"].includes(item.execution);
    return {
      caseId: item.case_id,
      implementation: invalid ? "wasm-validation" : codec ? "wasm-codec" : "webgpu",
      outputDigest: "0".repeat(64),
      scratchBytes: invalid ? null : 0,
      scratchBytesMax: invalid ? null : item.expected.scratch_bytes_max,
    };
  });
}

function npmArchiveReceipt(archiveBytes, revision = "a".repeat(40)) {
  const archive = {
    name: `tritium-ai-web-${RELEASE}.tgz`,
    bytes: archiveBytes.byteLength,
    sha256: sha256(archiveBytes),
  };
  const unsigned = {
    schema: "tritium.npm-archive-qualification.v1",
    release: RELEASE,
    source_revision: revision,
    run_id: "npm-physical-1",
    started_at_utc: "2026-08-07T12:00:00Z",
    duration_ms: 1,
    machine: {
      machine_id: `sha256:${"1".repeat(64)}`,
      system: "linux",
      architecture: "x86_64",
    },
    toolchain: { node: "v24.18.1", npm: "11.8.0" },
    artifact: {
      kind: "npm-archive",
      name: archive.name,
      package: `@tritium-ai/web@${RELEASE}`,
      bytes: archive.bytes,
      sha256: archive.sha256,
      integrity: `sha512-${createHash("sha512").update(archiveBytes).digest("base64")}`,
    },
    evidence: {
      source_dirty: false,
      entry_count: 16,
      source_free: true,
      installed_offline: true,
      strict_typescript: true,
      wasm_build_id: `tritium-wasm@${RELEASE}+source-git:${revision}`,
      wasm_guest_digest: "2".repeat(64),
    },
    result: "pass",
  };
  return {
    archive,
    receipt: {
      ...unsigned,
      receipt_id: `sha256:${sha256(Buffer.from(canonical(unsigned)))}`,
    },
  };
}

function admittedNpmQualification(archive, revision = "a".repeat(40)) {
  const receiptId = `sha256:${"4".repeat(64)}`;
  return {
    schema: "tritium.npm-archive-qualification.v1",
    receiptId,
    sourceRevision: revision,
    archive,
    receipt: { receipt_id: receiptId },
  };
}

function nativeReference(artifact, revision = "a".repeat(40)) {
  const artifactSha256 = sha256(artifact);
  const lifecycle = (operation) => ({
    result: "pass",
    operation,
    artifact_sha256: artifactSha256,
    input_digest: "1".repeat(64),
    output_digest: "2".repeat(64),
    peak_resident_bytes: 448,
    scratch_bytes: 131296,
    host_transfers: 0,
    device_resident: true,
  });
  const unsigned = {
    schema: "tritium.browser-native-reference.v1",
    result: "pass",
    scenario_id: "salt-ste-sgd-256-v1",
    source_revision: revision,
    backend: "cpu",
    backend_id: "cpu.reference.v1",
    backend_build: `tritium-train@${RELEASE}+source-git:${revision}`,
    physical_device: "cpu:linux:x86_64:test",
    manifest_digest: "9093a1a7f9a3422c399943782aadf4df6b11833cf2253db0db56ff2d9dedb098",
    vector_digest: "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7",
    artifact: {
      name: "native.salt",
      bytes: artifact.length,
      sha256: artifactSha256,
    },
    export: lifecycle("lifecycle.export"),
    reload: {
      ...lifecycle("lifecycle.reload"),
      reloaded_sha256: artifactSha256,
    },
  };
  return {
    ...unsigned,
    receipt_id: `sha256:${sha256(Buffer.from(canonical(unsigned)))}`,
  };
}

function browserTrace(referenceDigest) {
  const lifecycleOperations = [
    "session.forward", "session.backward", "session.step", "session.checkpoint",
    "session.resume", "session.export",
  ];
  const lifecycle = {
    prepare: true,
    forward: true,
    backward: true,
    optimizerStep: true,
    checkpointResume: true,
    exportReload: true,
    nativeArtifactParity: true,
    completedSteps: 1,
    checkpointSha256: "1".repeat(64),
    artifactSha256: sha256(Buffer.from("native artifact")),
    nativeArtifactSha256: sha256(Buffer.from("native artifact")),
    nativeReferenceDigest: referenceDigest,
    receipts: lifecycleOperations.map((operation) => ({
      operation,
      completedSteps: ["session.forward", "session.backward"].includes(operation) ? 0 : 1,
      peakResidentBytes: 4096,
      buildId: `wgsl:${"9".repeat(64)}:browser-qualification:salt-ste-sgd-256-v1`,
      physicalDevice: "Vendor:Arch:Device:physical",
    })),
  };
  const fault = { passed: true, errorCode: "expected", stateAfter: null };
  const vectorCases = canonicalVectorCases();
  const vector = {
    schemaId: "tritium.webgpu_vector_conformance_trace",
    schemaVersion: 1,
    implementation: "webgpu",
    manifestDigest: "9093a1a7f9a3422c399943782aadf4df6b11833cf2253db0db56ff2d9dedb098",
    vectorDigest: "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7",
    caseCounts: { valid: 72, invalid: 45, skipped: 0 },
    webgpuCaseTransactions: 68,
    webgpuDispatches: 100,
    wasmDispatches: 0,
    wasmCodecCalls: 4,
    wasmValidationCalls: 45,
    explicitReadbacks: 80,
    peakBufferBytes: 2048,
    executionDigest: sha256(Buffer.from(canonical(vectorCases))),
    cases: vectorCases,
  };
  const unsigned = {
    schemaId: "tritium.physical_browser_training_lane_trace",
    schemaVersion: 1,
    scenarioId: "salt-ste-sgd-256-v1",
    implementation: "webgpu",
    manifestDigest: "9093a1a7f9a3422c399943782aadf4df6b11833cf2253db0db56ff2d9dedb098",
    vectorDigest: "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7",
    physicalDevice: "Vendor:Arch:Device:physical",
    buildId: `wgsl:${"9".repeat(64)}:browser-qualification:salt-ste-sgd-256-v1`,
    adapter: {
      vendor: "Vendor",
      architecture: "Arch",
      device: "Device",
      description: "physical",
      software: false,
    },
    limits: {
      maxBufferSize: 1 << 30,
      maxStorageBufferBindingSize: 1 << 29,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBuffersPerShaderStage: 10,
    },
    vector,
    lifecycle,
    faults: {
      deviceLoss: fault,
      allocationFailure: { ...fault, errorCode: "injected_allocation_failure", observedEvents: 1 },
      malformedCheckpoint: fault,
      malformedSalt: fault,
      cancellation: { ...fault, errorCode: "cancelled", observedEvents: 1 },
      outOfOrder: fault,
    },
    explicitReadbacks: 87,
    steadyStateReadbacks: 0,
    wasmDispatches: 0,
    peakBufferBytes: 4096,
  };
  return { ...unsigned, executionDigest: sha256(Buffer.from(canonical(unsigned))) };
}

test("browser producer rejects nonignored untracked source", async () => {
  const repo = await mkdtemp(join(tmpdir(), "tritium-browser-source-"));
  try {
    await run("git", ["init", "-q"], { cwd: repo });
    await run("git", ["config", "user.email", "test@tritium.invalid"], { cwd: repo });
    await run("git", ["config", "user.name", "Tritium Test"], { cwd: repo });
    await writeFile(join(repo, "tracked.txt"), "tracked\n");
    await run("git", ["add", "tracked.txt"], { cwd: repo });
    await run("git", ["commit", "-qm", "fixture"], { cwd: repo });
    const revision = (await run("git", ["rev-parse", "HEAD"], { cwd: repo })).stdout.trim();
    await writeFile(join(repo, "untracked.txt"), "source injection\n");
    await assert.rejects(
      qualifyBrowserLaneV1({
        artifact: join(repo, "missing.tgz"),
        engine: "chrome",
        expectedBrowserVersion: "140.0.1",
        nativeArtifact: join(repo, "missing.salt"),
        nativeReferenceReceipt: join(repo, "missing-native.json"),
        npmReceipt: join(repo, "missing-npm.json"),
        os: { name: "Linux", version: "test", architecture: "x86_64" },
        outputDir: join(repo, "output"),
        repo,
        runId: "source-admission-test",
        sourceRevision: revision,
        webdriverUrl: "http://127.0.0.1:1",
      }),
      (error) => error instanceof BrowserLaneProducerError && error.code === "source",
    );
  } finally {
    await rm(repo, { recursive: true, force: true });
  }
});

test("npm archive receipt validates full schema, integrity, and canonical identity", () => {
  const archiveBytes = Buffer.from("exact npm archive bytes");
  const { archive, receipt } = npmArchiveReceipt(archiveBytes);
  const admitted = validateNpmReceiptV1(
    receipt,
    archive,
    archiveBytes,
    "a".repeat(40),
  );
  assert.equal(admitted.receiptId, receipt.receipt_id);

  const extraField = structuredClone(receipt);
  extraField.evidence.unbound = true;
  assert.throws(
    () => validateNpmReceiptV1(extraField, archive, archiveBytes, "a".repeat(40)),
    (error) => error instanceof BrowserLaneProducerError && error.code === "schema",
  );

  const rehashedPayload = structuredClone(receipt);
  rehashedPayload.receipt_id = `sha256:${"0".repeat(64)}`;
  assert.throws(
    () => validateNpmReceiptV1(rehashedPayload, archive, archiveBytes, "a".repeat(40)),
    (error) => error instanceof BrowserLaneProducerError && error.code === "npm_receipt",
  );
});

test("native CPU reference is revision, artifact, and receipt bound", () => {
  const artifact = Buffer.from("native artifact");
  const receipt = nativeReference(artifact);
  const admitted = validateNativeReferenceV1(
    receipt, artifact, "native.salt", "a".repeat(40),
  );
  assert.equal(admitted.receiptDigest, sha256(Buffer.from(canonical(receipt))));
  assert.equal(admitted.artifactSha256, sha256(artifact));
  assert.equal(admitted.reload.reloadedSha256, sha256(artifact));

  const stale = structuredClone(receipt);
  stale.source_revision = "b".repeat(40);
  assert.throws(
    () => validateNativeReferenceV1(stale, artifact, "native.salt", "a".repeat(40)),
    (error) => error instanceof BrowserLaneProducerError && error.code === "native_reference",
  );
  for (const forgedBuild of [
    `${receipt.backend_build}-suffix`,
    `prefix-${receipt.backend_build}`,
  ]) {
    const forged = structuredClone(receipt);
    forged.backend_build = forgedBuild;
    const unsigned = { ...forged };
    delete unsigned.receipt_id;
    forged.receipt_id = `sha256:${sha256(Buffer.from(canonical(unsigned)))}`;
    assert.throws(
      () => validateNativeReferenceV1(forged, artifact, "native.salt", "a".repeat(40)),
      (error) => error instanceof BrowserLaneProducerError && error.code === "native_reference",
    );
  }
});

test("lane assembly derives every pass claim from browser trace", () => {
  const artifact = Buffer.from("native artifact");
  const archive = {
    name: `tritium-ai-web-${RELEASE}.tgz`,
    bytes: 123,
    sha256: "3".repeat(64),
  };
  const reference = validateNativeReferenceV1(
    nativeReference(artifact), artifact, "native.salt", "a".repeat(40),
  );
  const result = assembleBrowserLaneV1({
    engine: "chrome",
    browserVersion: "140.0.1",
    os: { name: "Linux", version: "6.8", architecture: "x86_64" },
    runId: "chrome-physical-1",
    sourceRevision: "a".repeat(40),
    archive,
    npmQualification: admittedNpmQualification(archive),
    nativeReference: reference,
    webdriverCapabilities: {
      browserName: "chrome",
      browserVersion: "140.0.1",
      platformName: "linux",
    },
    browserTrace: browserTrace(reference.receiptDigest),
    traceFile: "trace.json",
  });
  assert.deepEqual(result.lane.case_counts, { valid: 72, invalid: 45, skipped: 0 });
  assert.deepEqual(result.lane.lifecycle, {
    prepare: true,
    forward: true,
    backward: true,
    optimizer_step: true,
    checkpoint_resume: true,
    export_reload: true,
    native_artifact_parity: true,
  });
  assert.equal(result.lane.trace.steady_state_readbacks, 0);
  assert.equal(result.lane.trace.wasm_dispatches, 0);
  assert.equal(result.lane.trace.sha256, sha256(result.traceBytes));
  const retainedTrace = JSON.parse(result.traceBytes);
  assert.match(retainedTrace.browser_trace.executionDigest, /^[0-9a-f]{64}$/);
  assert.equal(retainedTrace.npm_receipt.receipt_id, admittedNpmQualification(archive).receiptId);
});

test("lane assembly rejects a rehashed non-canonical vector inventory", () => {
  const artifact = Buffer.from("native artifact");
  const reference = validateNativeReferenceV1(
    nativeReference(artifact), artifact, "native.salt", "a".repeat(40),
  );
  const trace = structuredClone(browserTrace(reference.receiptDigest));
  trace.vector.cases[0].caseId = "fabricated.vector.case";
  trace.vector.executionDigest = sha256(Buffer.from(canonical(trace.vector.cases)));
  const unsigned = { ...trace };
  delete unsigned.executionDigest;
  trace.executionDigest = sha256(Buffer.from(canonical(unsigned)));
  assert.throws(
    () => assembleBrowserLaneV1({
      engine: "chrome",
      browserVersion: "140.0.1",
      os: { name: "Linux", version: "6.8", architecture: "x86_64" },
      runId: "chrome-physical-1",
      sourceRevision: "a".repeat(40),
      archive: {
        name: `tritium-ai-web-${RELEASE}.tgz`,
        bytes: 123,
        sha256: "3".repeat(64),
      },
      nativeReference: reference,
      webdriverCapabilities: {
        browserName: "chrome",
        browserVersion: "140.0.1",
        platformName: "linux",
      },
      browserTrace: trace,
      traceFile: "trace.json",
    }),
    (error) => error instanceof BrowserLaneProducerError && error.code === "browser_trace",
  );
});

test("lane assembly rejects unobserved submitted cancellation and allocation injection", () => {
  const artifact = Buffer.from("native artifact");
  const reference = validateNativeReferenceV1(
    nativeReference(artifact), artifact, "native.salt", "a".repeat(40),
  );
  for (const field of ["cancellation", "allocationFailure"]) {
    const trace = structuredClone(browserTrace(reference.receiptDigest));
    trace.faults[field].observedEvents = 0;
    const unsigned = { ...trace };
    delete unsigned.executionDigest;
    trace.executionDigest = sha256(Buffer.from(canonical(unsigned)));
    assert.throws(
      () => assembleBrowserLaneV1({
        engine: "chrome",
        browserVersion: "140.0.1",
        os: { name: "Linux", version: "6.8", architecture: "x86_64" },
        runId: "chrome-physical-1",
        sourceRevision: "a".repeat(40),
        archive: {
          name: `tritium-ai-web-${RELEASE}.tgz`,
          bytes: 123,
          sha256: "3".repeat(64),
        },
        nativeReference: reference,
        webdriverCapabilities: {
          browserName: "chrome",
          browserVersion: "140.0.1",
          platformName: "linux",
        },
        browserTrace: trace,
        traceFile: "trace.json",
      }),
      (error) => error instanceof BrowserLaneProducerError && error.code === "browser_trace",
    );
  }
});

test("classic WebDriver client uses W3C session and async-script routes", async () => {
  assert.throws(
    () => new WebDriverClassicClient("http://example.com:4444"),
    (error) => error instanceof BrowserLaneProducerError && error.code === "webdriver",
  );
  const requests = [];
  const server = createServer(async (request, response) => {
    let body = "";
    for await (const chunk of request) body += chunk;
    requests.push({ method: request.method, url: request.url, body: body && JSON.parse(body) });
    response.setHeader("content-type", "application/json");
    if (request.method === "POST" && request.url === "/session") {
      response.end(JSON.stringify({ value: {
        sessionId: "session-1",
        capabilities: {
          browserName: "chrome",
          browserVersion: "140.0.1",
          platformName: "linux",
        },
      } }));
    } else if (request.url === "/session/session-1/execute/async") {
      response.end(JSON.stringify({ value: { ok: true, value: { passed: true } } }));
    } else {
      response.end(JSON.stringify({ value: null }));
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    const client = new WebDriverClassicClient(`http://127.0.0.1:${address.port}`);
    const session = await client.createSession("chrome");
    await client.setTimeouts(session.id, { pageLoad: 30_000, script: 60_000 });
    await client.navigate(session.id, "http://127.0.0.1:1234/runner.html");
    const result = await client.executeAsync(session.id, "return 1", [[1, 2, 3]]);
    await client.deleteSession(session.id);
    assert.deepEqual(result, { ok: true, value: { passed: true } });
    assert.deepEqual(requests.map(({ method, url }) => [method, url]), [
      ["POST", "/session"],
      ["POST", "/session/session-1/timeouts"],
      ["POST", "/session/session-1/url"],
      ["POST", "/session/session-1/execute/async"],
      ["DELETE", "/session/session-1"],
    ]);
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
});

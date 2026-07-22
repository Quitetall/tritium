import { Buffer } from "node:buffer";

const ARTIFACT_ID = /^[a-z0-9][a-z0-9_.-]*$/;
const DIGEST = /^[0-9a-f]{64}$/;

function fail(message) {
  throw new Error(`npm SBOM generation failed: ${message}`);
}

function packageName(path) {
  const marker = "node_modules/";
  const index = path.lastIndexOf(marker);
  if (index < 0) fail(`invalid package-lock path ${path}`);
  const name = path.slice(index + marker.length);
  if (name.length === 0 || name.includes("node_modules/")) {
    fail(`invalid package-lock path ${path}`);
  }
  return name;
}

function integrityHash(value) {
  if (typeof value !== "string") return [];
  const match = /^(sha256|sha384|sha512)-([A-Za-z0-9+/]+={0,2})$/.exec(value);
  if (match === null) fail("package-lock integrity must use SHA-256 or stronger");
  const labels = { sha256: "SHA-256", sha384: "SHA-384", sha512: "SHA-512" };
  const lengths = { sha256: 32, sha384: 48, sha512: 64 };
  const decoded = Buffer.from(match[2], "base64");
  if (decoded.byteLength !== lengths[match[1]]) fail("package-lock integrity has wrong length");
  return [{ alg: labels[match[1]], content: decoded.toString("hex") }];
}

function component(path, value) {
  if (value === null || typeof value !== "object" || Array.isArray(value) ||
      typeof value.version !== "string" || value.version.length === 0) {
    fail(`invalid package-lock component ${path}`);
  }
  const name = packageName(path);
  const reference = `npm:${name}@${value.version}`;
  const result = {
    type: "library",
    "bom-ref": reference,
    name,
    version: value.version,
    scope: value.dev === true ? "excluded" : "required",
  };
  const hashes = integrityHash(value.integrity);
  if (hashes.length > 0) result.hashes = hashes;
  if (typeof value.license === "string" && /^[A-Za-z0-9.-]+$/.test(value.license)) {
    result.licenses = [{ license: { id: value.license } }];
  }
  const properties = [];
  if (value.optional === true) properties.push({ name: "tritium:npm:optional", value: "true" });
  for (const [name, values] of [["cpu", value.cpu], ["os", value.os]]) {
    if (Array.isArray(values)) {
      properties.push({ name: `tritium:npm:${name}`, value: [...values].sort().join(",") });
    }
  }
  if (properties.length > 0) result.properties = properties;
  return result;
}

export function generateNpmSbom(packageJson, packageLock, receipt, archiveFile,
                                artifactId = "tritium-web-node22") {
  if (!ARTIFACT_ID.test(artifactId)) fail("artifact id is not portable");
  if (packageJson === null || typeof packageJson !== "object" ||
      packageLock === null || typeof packageLock !== "object" ||
      receipt === null || typeof receipt !== "object") {
    fail("package, lock and receipt must be objects");
  }
  if (packageLock.lockfileVersion !== 3 || packageLock.name !== packageJson.name ||
      packageLock.version !== packageJson.version) {
    fail("package-lock identity differs from package.json");
  }
  if (receipt.package !== `${packageJson.name}@${packageJson.version}` ||
      !DIGEST.test(receipt.archiveSha256) || !Number.isSafeInteger(receipt.archiveBytes) ||
      receipt.archiveBytes <= 0 || !/^[0-9a-f]{40}$/.test(receipt.sourceRevision) ||
      typeof receipt.sourceDirty !== "boolean" || !DIGEST.test(receipt.wasmGuestDigest) ||
      typeof archiveFile !== "string" ||
      !/^[A-Za-z0-9][A-Za-z0-9._+-]*\.tgz$/.test(archiveFile)) {
    fail("archive receipt does not bind package identity and bytes");
  }
  const packages = packageLock.packages;
  if (packages === null || typeof packages !== "object" || Array.isArray(packages)) {
    fail("package-lock packages must be an object");
  }
  const components = Object.entries(packages)
    .filter(([path]) => path !== "")
    .map(([path, value]) => component(path, value))
    .sort((left, right) => left["bom-ref"].localeCompare(right["bom-ref"]));
  const refs = new Set();
  for (const item of components) {
    if (refs.has(item["bom-ref"])) fail(`duplicate component ${item["bom-ref"]}`);
    refs.add(item["bom-ref"]);
  }
  const runtimeNames = new Set();
  for (const field of ["dependencies", "optionalDependencies", "peerDependencies"]) {
    const runtime = packageJson[field] ?? {};
    if (runtime === null || typeof runtime !== "object" || Array.isArray(runtime)) {
      fail(`package ${field} must be an object`);
    }
    for (const name of Object.keys(runtime)) runtimeNames.add(name);
  }
  const required = [...runtimeNames].sort().map((name) => {
    const prefix = `npm:${name}@`;
    const matches = components.filter((item) => item["bom-ref"].startsWith(prefix));
    if (matches.length !== 1) fail(`runtime dependency ${name} is not uniquely locked`);
    return matches[0]["bom-ref"];
  });
  return {
    bomFormat: "CycloneDX",
    specVersion: "1.6",
    version: 1,
    metadata: {
      component: {
        type: "library",
        "bom-ref": artifactId,
        name: packageJson.name,
        version: packageJson.version,
        hashes: [{ alg: "SHA-256", content: receipt.archiveSha256 }],
        properties: [
          { name: "tritium:artifact:file", value: archiveFile },
          { name: "tritium:artifact:bytes", value: String(receipt.archiveBytes) },
          { name: "tritium:source:revision", value: receipt.sourceRevision },
          { name: "tritium:source:dirty", value: String(receipt.sourceDirty) },
          { name: "tritium:wasm:guest-digest", value: receipt.wasmGuestDigest },
        ],
      },
      tools: { components: [{ type: "application", name: "tritium-npm-sbom", version: "1" }] },
    },
    components,
    dependencies: [{ ref: artifactId, dependsOn: required }],
  };
}

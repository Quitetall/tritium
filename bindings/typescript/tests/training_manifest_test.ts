import {
  canonicalTrainingManifestV1Json,
  canonicalTrainingManifestJson,
  parseTrainingManifest,
  TrainingManifestError,
} from "../src/training_manifest.ts";

Deno.test("TypeScript reader preserves frozen V1", async () => {
  const fixture = await Deno.readFile("spec/training/v1/manifest.json");
  assert(
    new TextDecoder().decode(canonicalTrainingManifestV1Json()) ===
      new TextDecoder().decode(fixture),
    "V1 canonical bytes differ",
  );
  const parsed = parseTrainingManifest(fixture);
  assert(parsed.schema_version === 1, "V1 schema version differs");
  assert(parsed.operations.length === 35, "V1 operation count differs");
});

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertRejects(data: string | Uint8Array): void {
  try {
    parseTrainingManifest(data);
  } catch (error) {
    assert(
      error instanceof TrainingManifestError,
      `wrong error: ${String(error)}`,
    );
    return;
  }
  throw new Error("manifest unexpectedly accepted");
}

Deno.test("TypeScript canonical bytes equal language-neutral fixture", async () => {
  const fixture = await Deno.readFile("spec/training/v2/manifest.json");
  assert(
    new TextDecoder().decode(canonicalTrainingManifestJson()) ===
      new TextDecoder().decode(fixture),
    "canonical bytes differ",
  );
  assert(
    parseTrainingManifest(fixture).operations.length === 36,
    "operation count differs",
  );
});

Deno.test("TypeScript parser rejects drift, duplicates and bad UTF-8", async () => {
  const fixture = await Deno.readTextFile("spec/training/v2/manifest.json");
  const value = JSON.parse(fixture) as {
    schema_version: number;
    operations: Array<Record<string, unknown>>;
  };
  value.schema_version = 1;
  assertRejects(JSON.stringify(value));
  assertRejects('{"schema_id":"x","schema_id":"y"}');
  assertRejects('{"schema_id":"x","schema\\u005fid":"y"}');
  assertRejects(
    fixture.replace('"forward":true', '"forward":true,"forward":true'),
  );
  assertRejects(new Uint8Array([0xff]));
  assertRejects(fixture.replace('"forward":true', '"forward":false'));
});

Deno.test("TypeScript parser accepts noncanonical field order then re-emits canonical bytes", async () => {
  const fixture = await Deno.readTextFile("spec/training/v2/manifest.json");
  const value = JSON.parse(fixture) as Record<string, unknown>;
  const reordered = JSON.stringify({
    operations: value.operations,
    dtype: value.dtype,
    schema_version: value.schema_version,
    schema_id: value.schema_id,
  });
  assert(
    parseTrainingManifest(reordered).operations.length === 36,
    "reordered fields rejected",
  );
  assert(
    canonicalTrainingManifestJson().at(-1) === 0x0a,
    "canonical LF missing",
  );
});

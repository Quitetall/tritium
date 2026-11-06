import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const run = promisify(execFile);
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const binary = join(root, "node_modules/.bin/biome");

async function fixture(context) {
  const directory = await mkdtemp(join(tmpdir(), "tritium-web-quality-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const config = JSON.parse(await readFile(join(root, "biome.json"), "utf8"));
  // Exercise the package's real rule/format policy without inheriting the
  // host repository's ignore rules for this disposable non-repository fixture.
  config.vcs.enabled = false;
  await writeFile(join(directory, "biome.json"), JSON.stringify(config));
  await mkdir(join(directory, "src"));
  await mkdir(join(directory, "tests"));
  return {
    directory,
    async write(path, source) {
      await writeFile(join(directory, path), source);
    },
    check(command, path, extra = []) {
      return run(binary, [command, ...extra, path], {
        cwd: directory,
        timeout: 15_000,
      });
    },
  };
}

test("package check retains pinned formatter and warning-denying lint stages", async () => {
  const metadata = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
  assert.match(metadata.devDependencies["@biomejs/biome"], /^\d+\.\d+\.\d+$/);
  assert.equal(metadata.scripts["format:check"], "biome format .");
  assert.equal(metadata.scripts.lint, "biome lint --error-on-warnings .");
  assert.match(metadata.scripts.check, /^npm run format:check && npm run lint && /);
});

test("format check rejects drift and passes after actual formatter repair", async (context) => {
  const files = await fixture(context);
  const path = "tests/format-case.mjs";
  await files.write(path, "export const answer={value:42};\n");
  await assert.rejects(
    files.check("format", path),
    (error) => error.code === 1 && /Formatter would have printed/.test(error.stderr),
  );
  await files.check("format", path, ["--write"]);
  await files.check("format", path);
});

test("lint rejects unreachable code rather than only parsing syntax", async (context) => {
  const files = await fixture(context);
  const path = "tests/lint-case.mjs";
  await files.write(path, "export function answer() { return 42; return 0; }\n");
  await assert.rejects(
    files.check("lint", path, ["--error-on-warnings"]),
    (error) => error.code === 1 && /noUnreachable/.test(error.stderr),
  );
  await files.write(path, "export function answer() { return 42; }\n");
  await files.check("lint", path, ["--error-on-warnings"]);
});

test("numeric indexing exception is confined to explicitly named modules", async (context) => {
  const files = await fixture(context);
  const source = "export function read(values: readonly number[]) { return values[0]!; }\n";
  await files.write("src/session.ts", source);
  await files.check("lint", "src/session.ts", ["--error-on-warnings"]);
  await files.write("src/new-boundary.ts", source);
  await assert.rejects(
    files.check("lint", "src/new-boundary.ts", ["--error-on-warnings"]),
    (error) => error.code === 1 && /noNonNullAssertion/.test(error.stderr),
  );
});

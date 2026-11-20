"""Metadata plumbing only: GPU execution and publication are mocked, not qualified."""

import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "run-stage7-native-matrix.py"
SPEC = importlib.util.spec_from_file_location("stage7_native_matrix_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class NativeMatrixReleaseTests(unittest.TestCase):
    def fixture(self, manifest):
        temporary = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "Cargo.toml").write_text(manifest, encoding="utf-8")
        return root

    def test_native_command_and_checked_envelope_use_workspace_release(self):
        for release, stale in itertools.product(("1.1.0-rc.2", "1.1.0", "1.2.3-rc.7"), (False, True)):
            with self.subTest(release=release, stale=stale):
                root = self.fixture(f'[workspace.package]\nversion = "{release}"\n')
                output = root / "evidence" / "native.json"
                target = root / "target"
                revision = "a" * 40
                commands = []

                def execute(command, **kwargs):
                    commands.append(command)
                    if command == ["compute-sanitizer", "--version"]:
                        return subprocess.CompletedProcess(command, 0, "Version unit-test-only", "")
                    if command[:2] == ["cargo", "build"]:
                        executable = target / "debug/examples/stage7_native_matrix"
                        executable.parent.mkdir(parents=True)
                        executable.touch()
                        return subprocess.CompletedProcess(command, 0)
                    self.assertEqual(command[:2], ["compute-sanitizer", "--tool"])
                    arguments = dict(zip(command[command.index("--release"):][::2],
                                         command[command.index("--release") + 1:][::2]))
                    self.assertEqual(arguments["--release"], release)
                    log = Path(command[command.index("--log-file") + 1])
                    log.write_text("ERROR SUMMARY: 0 errors\n", encoding="utf-8")
                    Path(arguments["--output"]).write_text(json.dumps({
                        "schema": "tritium.stage7-native-kernels.v1",
                        "result": "pass",
                        "release": "1.1.0-rc.1" if stale else arguments["--release"],
                        "source_revision": arguments["--source-revision"],
                        "model_revision": arguments["--model-revision"],
                        "sanitizer_version": arguments["--sanitizer-version"],
                        "sanitizer_log": arguments["--sanitizer-log"],
                    }), encoding="utf-8")
                    return subprocess.CompletedProcess(command, 0)

                def validate(path, campaign):
                    self.assertEqual(campaign["release"], release)
                    self.assertEqual(json.loads(path.read_text())["release"], release)
                    return True  # Mocked metadata assertion, never empirical validation.

                with mock.patch.object(runner, "ROOT", root), \
                     mock.patch.object(runner, "_git", side_effect=[revision, ""]), \
                     mock.patch.object(runner.subprocess, "run", side_effect=execute), \
                     mock.patch.object(runner, "_load_qualifier", return_value=SimpleNamespace(_validate_native=validate)) as qualifier, \
                     mock.patch.object(runner, "_publish_immutable") as publish, \
                     mock.patch("builtins.print"):
                    if stale:
                        with self.assertRaisesRegex(RuntimeError, "native measurements do not bind"):
                            runner.run(output, target_dir=target)
                        qualifier.assert_not_called()
                        publish.assert_not_called()
                    else:
                        result = runner.run(output, target_dir=target)
                        self.assertEqual(result["release"], release)
                        publish.assert_called_once()
                        self.assertEqual(json.loads(publish.call_args.args[1])["release"], release)
                self.assertEqual(len(commands), 3)
                self.assertFalse(output.exists())

    def test_invalid_workspace_release_refuses_before_external_execution(self):
        manifests = (
            "[workspace]\n",
            "[workspace.package]\n",
            "[workspace.package]\nversion = 123\n",
            '[workspace.package]\nversion = "v1.1.0"\n',
            '[workspace.package]\nversion = "1.1.0-rc.02"\n',
            '[workspace.package]\nversion = "1.1.0rc2"\n',
            "not = [valid toml\n",
        )
        for manifest in manifests:
            with self.subTest(manifest=manifest):
                root = self.fixture(manifest)
                with mock.patch.object(runner, "ROOT", root), \
                     mock.patch.object(runner, "_git", side_effect=["a" * 40, ""]), \
                     mock.patch.object(runner.subprocess, "run", side_effect=AssertionError("external execution before version refusal")) as execute:
                    with self.assertRaises(ValueError):
                        runner.run(root / "evidence/native.json")
                    execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()

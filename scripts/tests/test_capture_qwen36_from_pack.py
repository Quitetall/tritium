from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "capture-qwen36-from-pack.py"
SPEC = importlib.util.spec_from_file_location("capture_qwen36_from_pack", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CaptureQwenFromPackTests(unittest.TestCase):
    def _base_args(self):
        return [
            "--manifest", "manifest.json",
            "--model-dir", "model",
            "--official-source-identity", "identity.json",
            "--pack-receipt", "pack.json",
            "--replay-contract", "contract.json",
            "--work-dir", "work",
            "--evidence-dir", "evidence",
            "--capture-binding-output", "binding.json",
        ]

    def test_activation_cache_identity_must_be_a_sha256_hex_digest(self):
        MODULE._validate_digest("a" * 64, "activation-cache digest")
        for invalid in ("a" * 63, "a" * 65, "g" * 64, "sha256:" + "a" * 64):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
                    MODULE._validate_digest(invalid, "activation-cache digest")

    def test_preflight_does_not_require_capture_recipe(self):
        args = MODULE._parser().parse_args(self._base_args())
        self.assertIsNone(args.activation_cache_digest)
        self.assertIsNone(args.curvature)
        self.assertIsNone(args.damping)

    def test_execute_requires_stage7_receipt_before_model_loading(self):
        replay = type("Replay", (), {
            "receipt": {"receipt_id": "pack", "revision": MODULE.PINNED_REVISION},
            "contract": {"contract_id": "contract"},
            "token_stream_digest": "batch",
        })()
        replay_module = type("ReplayModule", (), {
            "open": staticmethod(lambda *_args: replay),
        })
        verifier = {"validate_replay_contract": lambda _path: replay.contract}
        stderr = io.StringIO()
        with (
            mock.patch.object(MODULE, "REPLAY", {
                "Qwen36CalibrationReplay": replay_module,
                "_VERIFIER": verifier,
            }),
            mock.patch.object(sys, "argv", [
                str(SCRIPT), *self._base_args(), "--execute",
                "--curvature", "input-hessian", "--damping", "0.01",
                "--activation-cache-digest", "a" * 64,
                "--offload-folder", "offload",
            ]),
            contextlib.redirect_stderr(stderr),
        ):
            with self.assertRaises(SystemExit) as raised:
                MODULE.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--stage7-qualification-receipt", stderr.getvalue())

    def test_execute_validates_freeze_against_candidate_and_checkout_revision(self):
        replay = type("Replay", (), {
            "receipt": {"receipt_id": "pack", "revision": MODULE.PINNED_REVISION},
            "contract": {"contract_id": "contract"},
            "token_stream_digest": "batch",
        })()
        replay_module = type("ReplayModule", (), {
            "open": staticmethod(lambda *_args: replay),
        })
        verifier = {"validate_replay_contract": lambda _path: replay.contract}
        revision = "b" * 40
        checked = []
        runtime_checked = []
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.json"
            candidate.write_text(json.dumps({
                "schema": "tritium.release-candidate.v1",
                "release": "1.1.0-rc.9",
                "source_revision": revision,
            }))
            receipt = root / "stage7.json"
            receipt.write_text("{}\n")

            def validate(receipt_path, actual_revision, release, candidate_path):
                checked.append((receipt_path, actual_revision, release, candidate_path))
                return {"receipt_id": "sha256:" + "c" * 64}

            def validate_runtime(candidate_path, document, actual_revision):
                runtime_checked.append((candidate_path, document, actual_revision))
                return object()

            completed = lambda stdout: type("Completed", (), {"stdout": stdout})()
            torch_fake = ModuleType("torch")
            torch_fake.cuda = SimpleNamespace(is_available=lambda: False)
            transformers_fake = ModuleType("transformers")
            transformers_fake.AutoModelForImageTextToText = object()

            with (
                mock.patch.object(MODULE, "REPLAY", {
                    "Qwen36CalibrationReplay": replay_module,
                    "_VERIFIER": verifier,
                }),
                mock.patch.dict(MODULE.STAGE7, {"validate": validate}),
                mock.patch.object(
                    MODULE.subprocess,
                    "run",
                    side_effect=[completed(revision + "\n"), completed("")],
                ),
                mock.patch.object(
                    MODULE, "_validate_capture_python_environment", validate_runtime
                ),
                mock.patch.object(sys, "argv", [
                    str(SCRIPT), *self._base_args(), "--execute",
                    "--curvature", "input-hessian", "--damping", "0.01",
                    "--activation-cache-digest", "a" * 64,
                    "--offload-folder", str(root / "offload"),
                    "--release-candidate-manifest", str(candidate),
                    "--stage7-qualification-receipt", str(receipt),
                ]),
                mock.patch.dict(sys.modules, {
                    "torch": torch_fake,
                    "transformers": transformers_fake,
                }),
                contextlib.redirect_stderr(stderr),
            ):
                result = MODULE.main()

        self.assertEqual(result, 1)
        self.assertIn("no CUDA device detected", stderr.getvalue())
        self.assertEqual(checked, [(receipt, revision, "1.1.0-rc.9", candidate)])
        self.assertEqual(runtime_checked[0][0], candidate)
        self.assertEqual(runtime_checked[0][1]["source_revision"], revision)
        self.assertEqual(runtime_checked[0][2], revision)

    def test_candidate_wheel_must_match_exact_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "manifest.json"
            wheel = root / "linux" / "pytritium.whl"
            wheel.parent.mkdir()
            wheel.write_bytes(b"candidate wheel bytes")
            digest = MODULE.WHEEL_RUNTIME["_sha256"](wheel)
            candidate = {
                "artifacts": [{
                    "kind": "python-wheel",
                    "path": "linux/pytritium.whl",
                    "identity": {"bytes": wheel.stat().st_size, "sha256": digest},
                }]
            }
            self.assertEqual(
                MODULE._candidate_wheel(candidate_path, candidate, wheel),
                (wheel.resolve(), digest),
            )
            candidate["artifacts"][0]["identity"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "digest differs from candidate"):
                MODULE._candidate_wheel(candidate_path, candidate, wheel)

    def test_capture_environment_binds_candidate_wheel_modules_and_native_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "release" / "manifest.json"
            candidate_path.parent.mkdir()
            wheel = candidate_path.parent / "pytritium.whl"
            wheel.write_bytes(b"exact candidate wheel")
            package = root / "venv" / "site-packages" / "tritium" / "__init__.py"
            native = package.parent / "_tritium.abi3.so"
            qwen36_path = package.parent / "torch" / "qwen36.py"
            for path in (package, native, qwen36_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("candidate module")
            installed_files = frozenset(path.resolve() for path in (package, native, qwen36_path))
            digest = MODULE.WHEEL_RUNTIME["_sha256"](wheel)
            candidate = {
                "artifacts": [{
                    "kind": "python-wheel",
                    "path": wheel.name,
                    "identity": {"bytes": wheel.stat().st_size, "sha256": digest},
                }]
            }
            distribution = SimpleNamespace(
                read_text=lambda name: json.dumps({"url": wheel.as_uri()})
                if name == "direct_url.json" else None
            )
            tritium = SimpleNamespace(
                __file__=str(package),
                _tritium=SimpleNamespace(__file__=str(native)),
            )
            qwen36 = SimpleNamespace(__file__=str(qwen36_path))
            source_checks = []
            wheel_runtime = {
                "installed_distribution_identity": lambda path, actual_digest: (
                    "1.1.0rc.9", installed_files
                ),
                "require_distribution_file": lambda path, files: self.assertIn(
                    path.resolve(), files
                ),
                "require_installed": lambda *_args: None,
                "require_native_source_identity": lambda module, revision: source_checks.append(
                    (module, revision)
                ),
            }
            with (
                mock.patch.object(
                    MODULE.importlib.metadata, "distribution", return_value=distribution
                ),
                mock.patch.object(
                    MODULE.importlib.util,
                    "find_spec",
                    return_value=SimpleNamespace(origin=str(package)),
                ),
                mock.patch.object(
                    MODULE.importlib,
                    "import_module",
                    side_effect=lambda name: tritium if name == "tritium" else qwen36,
                ),
                mock.patch.dict(MODULE.WHEEL_RUNTIME, wheel_runtime),
            ):
                loaded = MODULE._validate_capture_python_environment(
                    candidate_path, candidate, "e" * 40
                )

            self.assertIs(loaded, qwen36)
            self.assertEqual(source_checks, [(tritium._tritium, "e" * 40)])

    def test_execute_rejects_dirty_checkout_before_receipt_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.json"
            revision = "d" * 40
            candidate.write_text(json.dumps({
                "schema": "tritium.release-candidate.v1",
                "release": "1.1.0-rc.9",
                "source_revision": revision,
            }))
            args = MODULE._parser().parse_args(
                self._base_args()
                + [
                    "--release-candidate-manifest", str(candidate),
                    "--stage7-qualification-receipt", str(root / "stage7.json"),
                ]
            )
            completed = lambda stdout: type("Completed", (), {"stdout": stdout})()
            validator = mock.Mock()
            with (
                mock.patch.object(
                    MODULE.subprocess,
                    "run",
                    side_effect=[completed(revision + "\n"), completed(" M local.py\n")],
                ),
                mock.patch.dict(MODULE.STAGE7, {"validate": validator}),
            ):
                with self.assertRaisesRegex(ValueError, "requires a clean checkout"):
                    MODULE._validate_stage7_qualification(args)
                validator.assert_not_called()

    def test_candidate_manifest_rejects_duplicate_identity_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.json"
            candidate.write_text(
                '{"schema":"tritium.release-candidate.v1",'
                '"release":"1.1.0-rc.9",'
                '"source_revision":"' + "a" * 40 + '",'
                '"source_revision":"' + "b" * 40 + '"}'
            )
            args = MODULE._parser().parse_args(
                self._base_args()
                + [
                    "--release-candidate-manifest", str(candidate),
                    "--stage7-qualification-receipt", str(root / "stage7.json"),
                ]
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key 'source_revision'"):
                MODULE._validate_stage7_qualification(args)

    def test_execution_recipe_requires_frozen_cache_digest(self):
        args = MODULE._parser().parse_args(
            self._base_args() + ["--curvature", "input-hessian", "--damping", "0.01"]
        )
        with self.assertRaisesRegex(ValueError, "requires --activation-cache-digest"):
            MODULE._validate_capture_recipe(args)

    def test_execution_recipe_rejects_non_finite_damping(self):
        args = MODULE._parser().parse_args(
            self._base_args()
            + [
                "--curvature", "input-hessian",
                "--damping", "nan",
                "--activation-cache-digest", "a" * 64,
            ]
        )
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            MODULE._validate_capture_recipe(args)

    def test_max_memory_is_parsed_as_device_limits(self):
        self.assertEqual(
            MODULE._parse_max_memory(["0=20GiB", "cpu=48GiB"]),
            {0: "20GiB", "cpu": "48GiB"},
        )

    def test_max_memory_rejects_duplicates_and_malformed_limits(self):
        for values in (["0=20GiB", "0=21GiB"], ["0"], ["cpu="]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    MODULE._parse_max_memory(values)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import io
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts.tests.test_verify_torch_dispatch_overhead_receipt import fixture


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(
    ROOT / "scripts" / "qualify-torch-dispatch-overhead.py"
)
assemble = MODULE["assemble"]
publish_directory_noreplace = MODULE["publish_directory_noreplace"]
RUNNER = runpy.run_path(
    ROOT
    / "crates"
    / "tritium-py"
    / "python"
    / "tritium"
    / "torch"
    / "qualify_dispatch_overhead.py"
)


class _Distribution:
    def __init__(self, root: Path, files: list[str], version: str = "1.1.0rc0"):
        self.root = root
        self.files = files
        self.version = version

    def locate_file(self, logical):
        return self.root / str(logical)


def _wheel_fixture(root: Path, *, name: str = "tritium-torch"):
    dist_info = "tritium_torch-1.1.0rc0.dist-info/"
    payloads = {
        "tritium/__init__.py": b"package",
        "tritium/extra.py": b"complete inventory sentinel",
        "tritium/torch/qualify_dispatch_overhead.py": b"qualifier",
        "tritium/_tritium.abi3.so": b"native",
        dist_info + "METADATA": (
            f"Metadata-Version: 2.4\nName: {name}\nVersion: 1.1.0rc0\n\n"
        ).encode(),
        dist_info + "WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: test\n"
            "Root-Is-Purelib: false\n"
            "Tag: cp39-abi3-linux_x86_64\n\n"
        ).encode(),
    }
    rows = []
    for logical, payload in payloads.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(
            b"="
        )
        rows.append([logical, "sha256=" + digest.decode(), str(len(payload))])
    record_name = dist_info + "RECORD"
    rows.append([record_name, "", ""])
    record = io.StringIO()
    csv.writer(record, lineterminator="\n").writerows(rows)
    payloads[record_name] = record.getvalue().encode()
    wheel = root / "tritium_torch-1.1.0rc0-cp39-abi3-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for logical, payload in payloads.items():
            archive.writestr(logical, payload)
    installed = root / "site-packages"
    for logical, payload in payloads.items():
        path = installed / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    distribution = _Distribution(installed, list(payloads))
    return wheel, payloads, distribution


class QualifyDispatchOverheadTests(unittest.TestCase):
    def test_verifies_complete_named_installed_wheel_inventory(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel, payloads, distribution = _wheel_fixture(Path(raw))
            with (
                wheel.open("rb") as stream,
                mock.patch.object(
                    RUNNER["importlib"].metadata,
                    "distribution",
                    return_value=distribution,
                ),
            ):
                identity = RUNNER["_verify_installed_wheel"](stream, wheel.name)
            self.assertEqual(identity["wheel_file_count"], len(payloads))
            self.assertEqual(
                identity["verified_installed_file_count"], len(payloads)
            )

    def test_rejects_partial_archive_against_full_installed_distribution(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wheel, payloads, distribution = _wheel_fixture(root)
            partial = root / "partial" / wheel.name
            partial.parent.mkdir()
            omitted = "tritium/extra.py"
            with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(partial, "w") as out:
                for info in source.infolist():
                    if info.filename != omitted:
                        out.writestr(info, source.read(info.filename))
            with (
                partial.open("rb") as stream,
                mock.patch.object(
                    RUNNER["importlib"].metadata,
                    "distribution",
                    return_value=distribution,
                ),
                self.assertRaisesRegex(RuntimeError, "RECORD inventory"),
            ):
                RUNNER["_verify_installed_wheel"](stream, partial.name)
            self.assertIn(omitted, payloads)

    def test_rejects_wrong_wheel_distribution_name(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel, _payloads, distribution = _wheel_fixture(
                Path(raw), name="other-project"
            )
            with (
                wheel.open("rb") as stream,
                mock.patch.object(
                    RUNNER["importlib"].metadata,
                    "distribution",
                    return_value=distribution,
                ),
                self.assertRaisesRegex(RuntimeError, "distribution name"),
            ):
                RUNNER["_verify_installed_wheel"](stream, wheel.name)

    def test_rejects_unrelated_installer_inventory_extra(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel, _payloads, distribution = _wheel_fixture(Path(raw))
            unrelated = distribution.root / "tritium" / "__pycache__" / "alien.pyc"
            unrelated.parent.mkdir(parents=True, exist_ok=True)
            unrelated.write_bytes(b"alien")
            distribution.files.append(
                "tritium/__pycache__/alien.pyc"
            )
            with (
                wheel.open("rb") as stream,
                mock.patch.object(
                    RUNNER["importlib"].metadata,
                    "distribution",
                    return_value=distribution,
                ),
                self.assertRaisesRegex(RuntimeError, "inventory differs"),
            ):
                RUNNER["_verify_installed_wheel"](stream, wheel.name)

    def test_snapshot_identity_survives_candidate_path_replacement(self):
        with tempfile.TemporaryDirectory() as raw:
            wheel = Path(raw) / "candidate.whl"
            wheel.write_bytes(b"wheel A")
            snapshot, identity = RUNNER["_snapshot_wheel"](wheel)
            try:
                wheel.write_bytes(b"wheel B")
                self.assertEqual(snapshot.read(), b"wheel A")
                self.assertEqual(
                    identity["sha256"], hashlib.sha256(b"wheel A").hexdigest()
                )
            finally:
                snapshot.close()

    def test_file_and_directory_publication_never_clobber(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stage_file = root / "stage"
            output_file = root / "output"
            stage_file.write_bytes(b"new")
            output_file.write_bytes(b"old")
            descriptor = os.open(root, os.O_TMPFILE | os.O_WRONLY, 0o600)
            try:
                os.write(descriptor, b"new")
                with self.assertRaises(FileExistsError):
                    RUNNER["_publish_fd_noreplace"](descriptor, output_file)
            finally:
                os.close(descriptor)
            self.assertEqual(output_file.read_bytes(), b"old")

            fresh = root / "fresh"
            descriptor = os.open(root, os.O_TMPFILE | os.O_WRONLY, 0o600)
            try:
                os.write(descriptor, b"fresh")
                RUNNER["_publish_fd_noreplace"](descriptor, fresh)
            finally:
                os.close(descriptor)
            self.assertEqual(fresh.read_bytes(), b"fresh")

            stage_dir = root / "stage-dir"
            output_dir = root / "output-dir"
            stage_dir.mkdir()
            (stage_dir / "new").write_bytes(b"new")
            output_dir.mkdir()
            (output_dir / "old").write_bytes(b"old")
            with self.assertRaisesRegex(MODULE["QualificationError"], "already exists"):
                publish_directory_noreplace(stage_dir, output_dir)
            self.assertEqual((output_dir / "old").read_bytes(), b"old")

    def test_installed_runner_policy_matches_release_verifier(self):
        runner = ROOT / (
            "crates/tritium-py/python/tritium/torch/"
            "qualify_dispatch_overhead.py"
        )
        source = runner.read_text(encoding="utf-8")
        tree = ast.parse(source)
        ast.parse(source, feature_version=(3, 9))
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id
            in {
                "TRACE_SCHEMA",
                "POLICY_ID",
                "WARMUP_COUNT",
                "SAMPLE_COUNT",
                "BOOTSTRAP_RESAMPLES",
                "BOOTSTRAP_CONFIDENCE",
                "OVERHEAD_LIMIT_RATIO",
                "POLICY_CASES",
            }
        }
        self.assertEqual(
            set(assignments),
            {
                "TRACE_SCHEMA",
                "POLICY_ID",
                "WARMUP_COUNT",
                "SAMPLE_COUNT",
                "BOOTSTRAP_RESAMPLES",
                "BOOTSTRAP_CONFIDENCE",
                "OVERHEAD_LIMIT_RATIO",
                "POLICY_CASES",
            },
        )
        for name in assignments:
            self.assertEqual(assignments[name], MODULE["VERIFY"][name])

    def test_assembles_and_self_validates_retained_trace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wheel, trace_path, _receipt_path, _trace, _receipt = fixture(root)
            output = root / "qualified"
            receipt = assemble(
                output,
                wheel=wheel,
                trace_path=trace_path,
                source_revision="a" * 40,
                release="1.1.0-rc.0",
                run_id="dispatch-overhead-physical-1",
            )
            self.assertEqual(receipt["result"], "pass")
            self.assertTrue((output / "trace.json").is_file())
            self.assertTrue((output / "receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()

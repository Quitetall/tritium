from __future__ import annotations

import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "generate-release-inputs.py")
ReleaseInputsError = MODULE["ReleaseInputsError"]
build_inputs = MODULE["build_inputs"]
write_inputs = MODULE["write_inputs"]


def sbom(path: Path, *, artifact_id: str, filename: str) -> None:
    path.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "metadata": {
                    "component": {
                        "bom-ref": artifact_id,
                        "properties": [
                            {"name": "tritium:artifact:file", "value": filename}
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


class GenerateReleaseInputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.staged = Path(self.temp.name) / "staged"
        self.staged.mkdir()
        for filename, artifact_id in (
            (
                "pytritium-1.1.0rc2-cp39-abi3-manylinux_2_28_x86_64.whl",
                "wheel-linux",
            ),
            ("tritium-core-1.1.0-rc.2.crate", "crate-core"),
            ("tritium-ai-web-1.1.0-rc.2.tgz", "tritium-web"),
        ):
            (self.staged / filename).write_bytes(b"archive:" + filename.encode())
            sbom(
                self.staged / f"{artifact_id}.cdx.json",
                artifact_id=artifact_id,
                filename=filename,
            )
        (self.staged / "npm-archive-receipt.json").write_text("{}", encoding="utf-8")

    def args(self) -> dict[str, str]:
        return {
            "release": "1.1.0-rc.2",
            "source_revision": "a" * 40,
            "builder_id": "https://example.invalid/workflow",
            "invocation_id": "run-123-attempt-1",
        }

    def test_builds_sorted_exact_sbom_bound_inventory(self):
        document = build_inputs(self.staged, **self.args())
        self.assertEqual(document["schema"], "tritium.release-inputs.v1")
        self.assertEqual(document["source_revision"], "a" * 40)
        self.assertEqual(
            [item["id"] for item in document["artifacts"]],
            ["crate-core", "tritium-web", "wheel-linux"],
        )
        self.assertEqual(
            {item["kind"] for item in document["artifacts"]},
            {"rust-crate", "npm-archive", "python-wheel"},
        )

    def test_rejects_unbound_release_artifact(self):
        (self.staged / "orphan-1.1.0.tgz").write_bytes(b"orphan")
        with self.assertRaisesRegex(ReleaseInputsError, "lack unique SBOM bindings"):
            build_inputs(self.staged, **self.args())

    def test_binds_same_named_nested_wheels_to_colocated_sboms(self):
        filename = "pytritium-1.1.0rc2-cp39-abi3-manylinux_2_28_x86_64.whl"
        for directory, artifact_id in (("cpu", "wheel-cpu"), ("cuda", "wheel-cuda")):
            target = self.staged / directory
            target.mkdir()
            (target / filename).write_bytes(directory.encode())
            sbom(
                target / f"{artifact_id}.cdx.json",
                artifact_id=artifact_id,
                filename=filename,
            )

        document = build_inputs(self.staged, **self.args())
        wheels = {
            item["id"]: item
            for item in document["artifacts"]
            if item["id"].startswith("wheel-")
        }
        self.assertEqual(wheels["wheel-cpu"]["path"], f"cpu/{filename}")
        self.assertEqual(wheels["wheel-cpu"]["sbom"], "cpu/wheel-cpu.cdx.json")
        self.assertEqual(wheels["wheel-cuda"]["path"], f"cuda/{filename}")
        self.assertEqual(wheels["wheel-cuda"]["sbom"], "cuda/wheel-cuda.cdx.json")

    def test_rejects_ambiguous_non_colocated_basename_binding(self):
        filename = "same.whl"
        for directory in ("cpu", "cuda"):
            target = self.staged / directory
            target.mkdir()
            (target / filename).write_bytes(directory.encode())
            sbom(
                target / f"{directory}.cdx.json",
                artifact_id=f"wheel-{directory}",
                filename=filename,
            )
        sbom(
            self.staged / "ambiguous.cdx.json",
            artifact_id="wheel-ambiguous",
            filename=filename,
        )

        with self.assertRaisesRegex(ReleaseInputsError, "ambiguously names artifact"):
            build_inputs(self.staged, **self.args())

    def test_rejects_backslash_in_artifact_directory_path(self):
        directory = self.staged / "nested\\folder"
        directory.mkdir()
        (directory / "portable.whl").write_bytes(b"wheel")
        sbom(
            directory / "wheel.cdx.json",
            artifact_id="nested-wheel",
            filename="portable.whl",
        )

        with self.assertRaisesRegex(ReleaseInputsError, "portable POSIX separators"):
            build_inputs(self.staged, **self.args())

    def test_rejects_backslash_in_sbom_directory_path(self):
        filename = "portable.whl"
        (self.staged / filename).write_bytes(b"wheel")
        directory = self.staged / "metadata\\folder"
        directory.mkdir()
        sbom(
            directory / "wheel.cdx.json",
            artifact_id="wheel",
            filename=filename,
        )

        with self.assertRaisesRegex(ReleaseInputsError, "SBOM path.*portable POSIX"):
            build_inputs(self.staged, **self.args())

    def test_rejects_duplicate_artifact_binding(self):
        filename = "tritium-core-1.1.0-rc.2.crate"
        sbom(
            self.staged / "duplicate.cdx.json",
            artifact_id="another-crate",
            filename=filename,
        )
        with self.assertRaisesRegex(ReleaseInputsError, "duplicate SBOM binding"):
            build_inputs(self.staged, **self.args())

    def test_rejects_sbom_with_unsafe_path_or_missing_artifact(self):
        sbom(
            self.staged / "unsafe.cdx.json",
            artifact_id="unsafe",
            filename="../escape.whl",
        )
        with self.assertRaisesRegex(ReleaseInputsError, "not a basename"):
            build_inputs(self.staged, **self.args())

    def test_rejects_sbom_property_without_value(self):
        path = self.staged / "missing-value.cdx.json"
        path.write_text(
            json.dumps(
                {
                    "bomFormat": "CycloneDX",
                    "metadata": {
                        "component": {
                            "bom-ref": "missing-value",
                            "properties": [{"name": "tritium:artifact:file"}],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ReleaseInputsError, "must be a non-empty string"):
            build_inputs(self.staged, **self.args())

    def test_rejects_noncanonical_revision_and_release(self):
        for args in (
            {**self.args(), "source_revision": "A" * 40},
            {**self.args(), "release": "1.1.0"},
        ):
            with self.subTest(args=args), self.assertRaises(ReleaseInputsError):
                build_inputs(self.staged, **args)

    def test_output_is_write_once(self):
        output = self.staged / "release-inputs.json"
        document = build_inputs(self.staged, **self.args())
        write_inputs(output, document)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), document)
        with self.assertRaisesRegex(ReleaseInputsError, "already exists"):
            write_inputs(output, document)


if __name__ == "__main__":
    unittest.main()

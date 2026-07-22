import json
import runpy
import tempfile
import unittest
from pathlib import Path

from scripts.tests.test_verify_wheel import build_wheel


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "generate-wheel-sbom.py")
SbomError = MODULE["SbomError"]
generate = MODULE["generate"]
write_sbom = MODULE["write_sbom"]


class GenerateWheelSbomTests(unittest.TestCase):
    def test_sbom_binds_wheel_files_and_declared_dependencies(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wheel = root / "tritium_torch-1.1.0rc0-cp39-abi3-linux_x86_64.whl"
            build_wheel(
                wheel,
                extra={
                    "tritium_torch-1.1.0rc0.dist-info/METADATA": (
                        b"Metadata-Version: 2.3\nName: tritium-torch\n"
                        b"Version: 1.1.0rc0\nRequires-Dist: torch>=2.11\n"
                    )
                },
            )
            first = generate(wheel, "tritium-torch-linux-cpu")
            second = generate(wheel, "tritium-torch-linux-cpu")
            self.assertEqual(first, second)
            self.assertEqual(
                first["metadata"]["component"]["bom-ref"],
                "tritium-torch-linux-cpu",
            )
            names = {component["name"] for component in first["components"]}
            self.assertIn("torch", names)
            self.assertIn("tritium/_tritium.abi3.so", names)
            self.assertEqual(
                len(first["dependencies"][0]["dependsOn"]),
                len(first["components"]),
            )

    def test_write_is_canonical_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "wheel.cdx.json"
            document = {
                "bomFormat": "CycloneDX",
                "specVersion": "1.6",
                "metadata": {"component": {"bom-ref": "wheel"}},
            }
            write_sbom(document, output)
            self.assertEqual(json.loads(output.read_text()), document)
            with self.assertRaisesRegex(SbomError, "already exists"):
                write_sbom(document, output)

    def test_invalid_artifact_id_and_corrupt_wheel_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wheel = root / "tritium_torch-1.1.0rc0-cp39-abi3-linux_x86_64.whl"
            build_wheel(wheel, corrupt_record=True)
            with self.assertRaisesRegex(SbomError, "lowercase"):
                generate(wheel, "Uppercase")
            with self.assertRaisesRegex(Exception, "RECORD hash mismatch"):
                generate(wheel, "tritium-wheel")


if __name__ == "__main__":
    unittest.main()

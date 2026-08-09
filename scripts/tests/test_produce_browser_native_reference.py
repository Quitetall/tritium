import hashlib
import runpy
from pathlib import Path
import subprocess
import tempfile
import unittest


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "produce-browser-native-reference.py"
)
build_receipt = MODULE["build_receipt"]
canonical = MODULE["canonical"]
NativeReferenceError = MODULE["NativeReferenceError"]
source_admission = MODULE["source_admission"]


def metadata():
    result = {
        "backend_id": "cpu.reference.v1",
        "backend_build": "tritium-train@1.1.0-rc.1+source-git:" + "a" * 40,
        "physical_device": "cpu:linux:x86_64:test",
        "manifest_digest": MODULE["MANIFEST_DIGEST"],
    }
    for prefix, operation in (
        ("export", "lifecycle.export"),
        ("reload", "lifecycle.reload"),
    ):
        result.update(
            {
                f"{prefix}_operation": operation,
                f"{prefix}_input_digest": "1" * 64,
                f"{prefix}_output_digest": "2" * 64,
                f"{prefix}_peak_resident_bytes": "448",
                f"{prefix}_scratch_bytes": "131296",
                f"{prefix}_host_transfers": "0",
                f"{prefix}_device_resident": "true",
            }
        )
    return result


class NativeBrowserReferenceTests(unittest.TestCase):
    def test_source_admission_rejects_nonignored_untracked_files(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@tritium.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Tritium Test"],
                cwd=root,
                check=True,
            )
            (root / "tracked.txt").write_text("tracked\n")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            source_admission(revision, root)
            (root / "untracked.txt").write_text("source injection\n")
            with self.assertRaisesRegex(NativeReferenceError, "clean source"):
                source_admission(revision, root)

    def test_receipt_binds_exact_native_export_and_reload(self):
        artifact = b"TSLT2PKG exact salt artifact"
        receipt = build_receipt(artifact, artifact, metadata(), "a" * 40)
        digest = hashlib.sha256(artifact).hexdigest()
        self.assertEqual(receipt["artifact"]["sha256"], digest)
        self.assertEqual(receipt["export"]["artifact_sha256"], digest)
        self.assertEqual(receipt["reload"]["artifact_sha256"], digest)
        self.assertEqual(receipt["reload"]["reloaded_sha256"], digest)
        self.assertEqual(
            receipt["receipt_id"],
            "sha256:"
            + hashlib.sha256(
                canonical(
                    {
                        key: value
                        for key, value in receipt.items()
                        if key != "receipt_id"
                    }
                )
            ).hexdigest(),
        )

    def test_rejects_native_reload_or_receipt_identity_drift(self):
        artifact = b"TSLT2PKG exact salt artifact"
        with self.assertRaisesRegex(NativeReferenceError, "reload changed"):
            build_receipt(artifact, artifact + b"x", metadata(), "a" * 40)
        changed = metadata()
        changed["reload_operation"] = "lifecycle.export"
        with self.assertRaisesRegex(NativeReferenceError, "lifecycle"):
            build_receipt(artifact, artifact, changed, "a" * 40)
        changed = metadata()
        changed["reload_host_transfers"] = "1"
        with self.assertRaisesRegex(NativeReferenceError, "resident"):
            build_receipt(artifact, artifact, changed, "a" * 40)
        exact_build = metadata()["backend_build"]
        for forged_build in (exact_build + "-suffix", "prefix-" + exact_build):
            with self.subTest(forged_build=forged_build):
                changed = metadata()
                changed["backend_build"] = forged_build
                with self.assertRaisesRegex(NativeReferenceError, "build"):
                    build_receipt(artifact, artifact, changed, "a" * 40)


if __name__ == "__main__":
    unittest.main()

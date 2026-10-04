from __future__ import annotations

from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "stage-release-candidate.py")
StageError = MODULE["StageError"]
stage = MODULE["stage"]


def write(path: Path, content: bytes = b"fixture") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def downloads(root: Path) -> Path:
    wheels = root / "wheels"
    for target, suffix in (
        ("linux-x86_64-cpu", "manylinux_2_28_x86_64"),
        ("macos-arm64-cpu", "macosx_11_0_arm64"),
        ("windows-x86_64-cpu", "win_amd64"),
    ):
        wheel = f"pytritium-1.1.0rc2-cp39-abi3-{suffix}.whl"
        write(wheels / wheel)
        write(wheels / f"pytritium-{target}.cdx.json")
        write(wheels / f"{target}.json")

    crates = root / "package-evidence" / "crates"
    write(crates / "tritium-core-1.1.0-rc.2.crate")
    write(crates / "tritium-core.cdx.json")
    write(crates / MODULE["CRATE_RECEIPT"])

    npm = root / "package-evidence" / "npm"
    write(npm / "tritium-ai-web-1.1.0-rc.2.tgz")
    write(npm / "tritium-web-node22.cdx.json")
    write(npm / MODULE["NPM_RECEIPT"])

    write(root / "compatibility-evidence" / MODULE["COMPATIBILITY_RECEIPT"])
    return root


class StageReleaseCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.downloads = downloads(self.root / "downloads")
        self.payload = self.root / "candidate"
        self.evidence = self.root / "evidence"

    def test_stages_only_package_bytes_in_candidate_payload(self) -> None:
        result = stage(self.downloads, self.payload, self.evidence)

        self.assertEqual(result["schema"], "tritium.release-candidate-staging.v1")
        self.assertEqual(result["payload_files"], 10)
        self.assertEqual(result["evidence_files"], 6)
        payload_names = {path.name for path in self.payload.iterdir()}
        self.assertEqual(sum(name.endswith(".whl") for name in payload_names), 3)
        self.assertEqual(sum(name.endswith(".crate") for name in payload_names), 1)
        self.assertEqual(sum(name.endswith(".tgz") for name in payload_names), 1)
        self.assertEqual(sum(name.endswith(".cdx.json") for name in payload_names), 5)
        self.assertFalse(any(name.endswith("receipt.json") for name in payload_names))
        self.assertEqual(
            len(list(self.evidence.rglob("*.json"))), 6,
        )

    def test_rejects_unexpected_downloaded_file(self) -> None:
        write(self.downloads / "wheels" / "unexpected.txt")
        with self.assertRaisesRegex(StageError, "wheel artifacts inventory differs"):
            stage(self.downloads, self.payload, self.evidence)
        self.assertFalse(self.payload.exists())
        self.assertFalse(self.evidence.exists())

    def test_rejects_missing_platform_wheel(self) -> None:
        (self.downloads / "wheels" / "pytritium-1.1.0rc2-cp39-abi3-win_amd64.whl").unlink()
        with self.assertRaisesRegex(StageError, "exactly three platform wheels"):
            stage(self.downloads, self.payload, self.evidence)

    def test_rejects_symlink_in_downloads(self) -> None:
        target = self.root / "outside.whl"
        target.write_bytes(b"not staged")
        (self.downloads / "wheels" / "linked.whl").symlink_to(target)
        with self.assertRaisesRegex(StageError, "non-regular file"):
            stage(self.downloads, self.payload, self.evidence)

    def test_rejects_payload_and_evidence_filename_collision(self) -> None:
        name = "tritium-core.cdx.json"
        (self.downloads / "wheels" / "pytritium-linux-x86_64-cpu.cdx.json").unlink()
        write(self.downloads / "wheels" / name)
        with self.assertRaisesRegex(StageError, "colliding artifact or SBOM"):
            stage(self.downloads, self.payload, self.evidence)

    def test_refuses_to_overwrite_output(self) -> None:
        self.payload.mkdir()
        with self.assertRaisesRegex(StageError, "must not already exist"):
            stage(self.downloads, self.payload, self.evidence)

    def test_refuses_output_inside_download_tree(self) -> None:
        with self.assertRaisesRegex(StageError, "outside the downloads tree"):
            stage(
                self.downloads,
                self.downloads / "candidate",
                self.evidence,
            )


if __name__ == "__main__":
    unittest.main()

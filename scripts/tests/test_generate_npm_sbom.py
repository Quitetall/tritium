import io
import json
import runpy
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "generate-npm-sbom.py")
NpmSbomError = MODULE["NpmSbomError"]
inspect_archive = MODULE["inspect_archive"]
write_sbom = MODULE["write_sbom"]


def npm_archive(path: Path, *, name: str = "@tritium-ai/web", version: str = "1.1.0-rc.1", symlink: bool = False) -> None:
    files = {
        "package/package.json": json.dumps({"name": name, "version": version}).encode(),
        "package/README.md": b"Tritium web\n",
        "package/dist/index.js": b"export {};\n",
    }
    with tarfile.open(path, "w:gz") as archive:
        for member_name, payload in files.items():
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if symlink:
            info = tarfile.TarInfo("package/dist/escape")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../escape"
            archive.addfile(info)


class GenerateNpmSbomTests(unittest.TestCase):
    def test_sbom_binds_archive_members_and_source(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "tritium-ai-web-1.1.0-rc.1.tgz"
            npm_archive(archive)
            first = inspect_archive(archive, "tritium-web-node22", "1.1.0-rc.1", "a" * 40)
            second = inspect_archive(archive, "tritium-web-node22", "1.1.0-rc.1", "a" * 40)
            self.assertEqual(first, second)
            component = first["metadata"]["component"]
            self.assertEqual(component["bom-ref"], "tritium-web-node22")
            self.assertEqual(component["properties"][-1]["value"], "false")
            self.assertEqual(len(first["components"]), 3)
            self.assertEqual(len(first["dependencies"][0]["dependsOn"]), 3)

    def test_archive_identity_and_unsafe_members_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "tritium-ai-web-1.1.0-rc.1.tgz"
            npm_archive(archive, symlink=True)
            with self.assertRaisesRegex(NpmSbomError, "not regular"):
                inspect_archive(archive, "tritium-web-node22", "1.1.0-rc.1", "a" * 40)
            wrong = root / "wrong.tgz"
            npm_archive(wrong)
            with self.assertRaisesRegex(NpmSbomError, "filename"):
                inspect_archive(wrong, "tritium-web-node22", "1.1.0-rc.1", "a" * 40)

    def test_write_is_canonical_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "web.cdx.json"
            document = {"bomFormat": "CycloneDX", "specVersion": "1.6"}
            write_sbom(document, output)
            self.assertEqual(json.loads(output.read_text()), document)
            with self.assertRaisesRegex(NpmSbomError, "already exists"):
                write_sbom(document, output)


if __name__ == "__main__":
    unittest.main()

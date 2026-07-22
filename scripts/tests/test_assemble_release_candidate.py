import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "assemble-release-candidate.py")
ReleaseError = MODULE["ReleaseError"]
assemble = MODULE["assemble"]
candidate_validate = MODULE["candidate_validate"]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def fake_tool(path: Path) -> Path:
    tool = path / "tritium"
    tool.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib,json,pathlib,sys\n"
        "b=pathlib.Path(sys.argv[3]).read_bytes()\n"
        "print(json.dumps({'schema':'tritium.file-identity.v1','bytes':len(b),"
        "'sha256':hashlib.sha256(b).hexdigest(),"
        "'blake3':hashlib.sha256(b'B3'+b).hexdigest()},separators=(',',':')))\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    return tool


def candidate_fixture(base: Path, *, corrupt_sbom: bool = False) -> tuple[Path, Path, Path]:
    candidate = base / "candidate"
    candidate.mkdir()
    artifacts = [
        ("web-package", "npm-archive", "tritium-web.tgz", b"npm"),
        ("cpu-wheel", "python-wheel", "tritium.whl", b"wheel"),
    ]
    inputs = []
    for artifact_id, kind, filename, payload in artifacts:
        (candidate / filename).write_bytes(payload)
        sbom_name = f"{artifact_id}.cdx.json"
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "metadata": {"component": {"bom-ref": artifact_id}},
        }
        if corrupt_sbom and artifact_id == "cpu-wheel":
            sbom["metadata"]["component"]["bom-ref"] = "wrong"
        write_json(candidate / sbom_name, sbom)
        inputs.append(
            {"id": artifact_id, "kind": kind, "path": filename, "sbom": sbom_name}
        )
    descriptor = base / "inputs.json"
    write_json(
        descriptor,
        {
            "schema": "tritium.release-inputs.v1",
            "release": "1.1.0-rc.0",
            "source_revision": "a" * 40,
            "builder": {
                "id": "https://github.com/tritium/ci/release",
                "build_type": "https://tritium.ai/build/package/v1",
                "invocation_id": "run-123",
            },
            "artifacts": inputs,
        },
    )
    return candidate, descriptor, fake_tool(base)


class AssembleReleaseCandidateTests(unittest.TestCase):
    def test_assembly_is_sorted_deterministic_and_strictly_reloadable(self):
        manifests = []
        for ordinal in range(2):
            with tempfile.TemporaryDirectory() as raw:
                base = Path(raw)
                candidate, inputs, tool = candidate_fixture(base)
                output = candidate / "manifest.json"
                manifest = assemble(inputs, output, str(tool))
                self.assertEqual(
                    [item["id"] for item in manifest["artifacts"]],
                    ["cpu-wheel", "web-package"],
                )
                self.assertEqual(candidate_validate(output, str(tool)), manifest)
                provenance = json.loads(
                    (candidate / "provenance/cpu-wheel.intoto.json").read_text()
                )
                subject = provenance["subject"][0]
                self.assertEqual(subject["name"], "tritium.whl")
                self.assertEqual(
                    subject["digest"]["sha256"],
                    hashlib.sha256(b"wheel").hexdigest(),
                )
                manifests.append(output.read_bytes())
        self.assertEqual(manifests[0], manifests[1])

    def test_invalid_sbom_rolls_back_publication(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            candidate, inputs, tool = candidate_fixture(base, corrupt_sbom=True)
            with self.assertRaisesRegex(ReleaseError, "does not bind artifact id"):
                assemble(inputs, candidate / "manifest.json", str(tool))
            self.assertFalse((candidate / "manifest.json").exists())
            self.assertFalse((candidate / "provenance").exists())

    def test_failed_strict_reload_rolls_back_published_metadata(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            candidate, inputs, tool = candidate_fixture(base)
            counter = base / "digest-count"
            tool.write_text(
                "#!/usr/bin/env python3\n"
                "import hashlib,json,pathlib,sys\n"
                f"c=pathlib.Path({str(counter)!r}); n=int(c.read_text()) if c.exists() else 0; "
                "c.write_text(str(n+1))\n"
                "b=pathlib.Path(sys.argv[3]).read_bytes(); s=hashlib.sha256(b).hexdigest()\n"
                "print(json.dumps({'schema':'tritium.file-identity.v1','bytes':len(b),"
                "'sha256':('0'*64 if n>=2 else s),"
                "'blake3':hashlib.sha256(b'B3'+b).hexdigest()},separators=(',',':')))\n",
                encoding="utf-8",
            )
            tool.chmod(0o755)
            with self.assertRaisesRegex(ReleaseError, "identity does not match"):
                assemble(inputs, candidate / "manifest.json", str(tool))
            self.assertFalse((candidate / "manifest.json").exists())
            self.assertFalse((candidate / "provenance").exists())


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "release-status")
ReleaseError = MODULE["ReleaseError"]
validate = MODULE["validate"]


def write_json(path: Path, value: object) -> str:
    payload = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def fixture(root: Path) -> tuple[Path, Path, dict]:
    tool = root / "tritium"
    tool.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib,json,pathlib,sys\n"
        "p=pathlib.Path(sys.argv[3]); b=p.read_bytes()\n"
        "print(json.dumps({'schema':'tritium.file-identity.v1','bytes':len(b),"
        "'sha256':hashlib.sha256(b).hexdigest(),'blake3':'0'*64},"
        "separators=(',',':')))\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    artifact = root / "wheel.whl"
    artifact.write_bytes(b"wheel bytes")
    sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    artifact_id = "tritium-torch-linux-cpu"
    sbom_sha = write_json(
        root / "wheel.cdx.json",
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "metadata": {"component": {"bom-ref": artifact_id}},
        },
    )
    revision = "1" * 40
    provenance_sha = write_json(
        root / "wheel.provenance.json",
        {
            "_type": "https://in-toto.io/Statement/v1",
            "subject": [{"name": "wheel.whl", "digest": {"sha256": sha256}}],
            "predicateType": "https://slsa.dev/provenance/v1",
            "predicate": {
                "buildDefinition": {
                    "externalParameters": {"source_revision": revision}
                },
                "runDetails": {"builder": {"id": "test-builder"}},
            },
        },
    )
    document = {
        "schema": "tritium.release-candidate.v1",
        "release": "1.1.0-rc.0",
        "source_revision": revision,
        "artifacts": [
            {
                "id": artifact_id,
                "kind": "python-wheel",
                "path": "wheel.whl",
                "identity": {
                    "schema": "tritium.file-identity.v1",
                    "bytes": len(artifact.read_bytes()),
                    "sha256": sha256,
                    "blake3": "0" * 64,
                },
                "sbom": {"path": "wheel.cdx.json", "sha256": sbom_sha},
                "provenance": {
                    "path": "wheel.provenance.json",
                    "sha256": provenance_sha,
                },
            }
        ],
    }
    candidate = root / "manifest.json"
    write_json(candidate, document)
    return candidate, tool, document


class ReleaseStatusTests(unittest.TestCase):
    def test_candidate_binds_artifact_sbom_and_provenance(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, tool, document = fixture(Path(raw))
            self.assertEqual(validate(candidate, str(tool)), document)

    def test_candidate_rejects_identity_and_lineage_drift(self):
        for mutation in ("artifact", "sbom", "provenance", "revision"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                candidate, tool, document = fixture(root)
                if mutation == "artifact":
                    (root / "wheel.whl").write_bytes(b"changed")
                elif mutation == "sbom":
                    (root / "wheel.cdx.json").write_text("{}", encoding="utf-8")
                elif mutation == "provenance":
                    (root / "wheel.provenance.json").write_text("{}", encoding="utf-8")
                else:
                    document["source_revision"] = "2" * 40
                    write_json(candidate, document)
                with self.assertRaises(ReleaseError):
                    validate(candidate, str(tool))

    def test_candidate_rejects_symlinked_artifact(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, tool, _ = fixture(root)
            artifact = root / "wheel.whl"
            artifact.rename(root / "real.whl")
            artifact.symlink_to("real.whl")
            with self.assertRaisesRegex(ReleaseError, "symlink"):
                validate(candidate, str(tool))


if __name__ == "__main__":
    unittest.main()

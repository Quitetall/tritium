import hashlib
import io
import json
import runpy
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.tests.helm_fixtures import chart_source
from scripts.tests.test_verify_oci_archive import (
    BUILDER_ID,
    fixture as oci_archive_fixture,
)


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "release-status")
ReleaseError = MODULE["ReleaseError"]
validate = MODULE["validate"]
write_report = MODULE["_write_report"]
BUNDLE_MODULE = runpy.run_path(ROOT / "scripts" / "generate-bundle-sbom.py")
generate_bundle_sbom = BUNDLE_MODULE["generate"]
HELM_PACKAGER = runpy.run_path(ROOT / "scripts" / "package-helm-chart.py")
DEPLOYMENT_SBOM = runpy.run_path(ROOT / "scripts" / "generate-deployment-sbom.py")


def fake_blake3(payload: bytes) -> str:
    return hashlib.sha256(b"B3" + payload).hexdigest()


def write_json(path: Path, value: object) -> str:
    payload = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def fixture(root: Path) -> tuple[Path, Path, dict]:
    tool = root / "tritium"
    tool.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib,json,pathlib,sys\n"
        "if sys.argv[1:] == ['release', 'digest-stream']:\n"
        " b=sys.stdin.buffer.read(); schema='tritium.stream-identity.v1'\n"
        " value={'schema':schema,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),"
        "'blake3':hashlib.sha256(b'B3'+b).hexdigest(),"
        "'package_id':'trp1_'+hashlib.sha256(b'PKG'+b).hexdigest()}\n"
        "else:\n"
        " p=pathlib.Path(sys.argv[3]); b=p.read_bytes()\n"
        " value={'schema':'tritium.file-identity.v1','bytes':len(b),"
        "'sha256':hashlib.sha256(b).hexdigest(),"
        "'blake3':hashlib.sha256(b'B3'+b).hexdigest()}\n"
        "print(json.dumps(value,separators=(',',':')))\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    candidate_root = root / "candidate"
    candidate_root.mkdir()
    artifact = candidate_root / "wheel.whl"
    artifact.write_bytes(b"wheel bytes")
    sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    artifact_id = "pytritium-linux-cpu"
    sbom_sha = write_json(
        candidate_root / "wheel.cdx.json",
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "metadata": {
                "component": {
                    "bom-ref": artifact_id,
                    "hashes": [{"alg": "SHA-256", "content": sha256}],
                    "properties": [
                        {"name": "tritium:artifact:file", "value": artifact.name},
                        {
                            "name": "tritium:artifact:bytes",
                            "value": str(artifact.stat().st_size),
                        },
                    ],
                }
            },
        },
    )
    revision = "1" * 40
    provenance_sha = write_json(
        candidate_root / "wheel.provenance.json",
        {
            "_type": "https://in-toto.io/Statement/v1",
            "subject": [{"name": "wheel.whl", "digest": {"sha256": sha256}}],
            "predicateType": "https://slsa.dev/provenance/v1",
            "predicate": {
                "buildDefinition": {
                    "externalParameters": {"source_revision": revision}
                },
                "runDetails": {
                    "builder": {"id": "test-builder"},
                    "metadata": {"invocationID": "test-run"},
                },
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
                    "blake3": fake_blake3(artifact.read_bytes()),
                },
                "sbom": {"path": "wheel.cdx.json", "sha256": sbom_sha},
                "provenance": {
                    "path": "wheel.provenance.json",
                    "sha256": provenance_sha,
                },
            }
        ],
    }
    candidate = candidate_root / "manifest.json"
    write_json(candidate, document)
    return candidate, tool, document


def bundle_fixture(root: Path) -> tuple[Path, Path, dict]:
    _, tool, _ = fixture(root)
    candidate_root = root / "candidate"
    for path in candidate_root.iterdir():
        path.unlink()
    language = b"language"
    mtp = b"mtp"
    weights = b"weights"
    manifest = {
        "schema": "tritium-qwen35-onnx-bundle-v2",
        "sequence_mode": "dynamic-cache-v1",
        "language": {"file": "language.onnx", "blake3": fake_blake3(language)},
        "mtp": {"file": "mtp.onnx", "blake3": fake_blake3(mtp)},
        "weights": {
            "file": "weights.bin",
            "blake3": fake_blake3(weights),
            "bytes": len(weights),
        },
        "identity": {
            "source_model_id": "source",
            "tokenizer_id": "tokenizer",
            "recipe_id": "recipe",
            "package_id": "package",
            "converted_coverage_id": "converted",
            "deferred_coverage_id": "deferred",
        },
        "conversion": {
            "mode": "ptq",
            "completion_id": "completion",
            "campaign_id": "campaign",
            "admission_id": "admission",
            "selection_id": "selection",
        },
    }
    files = {
        "language.onnx": language,
        "mtp.onnx": mtp,
        "weights.bin": weights,
        "tritium-onnx-manifest.json": json.dumps(manifest, sort_keys=True).encode(),
    }
    artifact = candidate_root / "onnx.tar"
    with tarfile.open(artifact, "w") as output:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            output.addfile(info, io.BytesIO(payload))
    artifact_id = "qwen-onnx"
    revision = "1" * 40
    sbom = generate_bundle_sbom(
        artifact, artifact_id, "onnx-bundle", revision, str(tool)
    )
    sbom_sha = write_json(candidate_root / "onnx.cdx.json", sbom)
    payload = artifact.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    provenance_sha = write_json(
        candidate_root / "onnx.provenance.json",
        {
            "_type": "https://in-toto.io/Statement/v1",
            "subject": [{"name": artifact.name, "digest": {"sha256": sha256}}],
            "predicateType": "https://slsa.dev/provenance/v1",
            "predicate": {
                "buildDefinition": {
                    "externalParameters": {"source_revision": revision}
                },
                "runDetails": {
                    "builder": {"id": "test-builder"},
                    "metadata": {"invocationID": "test-run"},
                },
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
                "kind": "onnx-bundle",
                "path": artifact.name,
                "identity": {
                    "schema": "tritium.file-identity.v1",
                    "bytes": len(payload),
                    "sha256": sha256,
                    "blake3": fake_blake3(payload),
                },
                "sbom": {"path": "onnx.cdx.json", "sha256": sbom_sha},
                "provenance": {
                    "path": "onnx.provenance.json",
                    "sha256": provenance_sha,
                },
            }
        ],
    }
    candidate = candidate_root / "manifest.json"
    write_json(candidate, document)
    return candidate, tool, document


def helm_fixture(root: Path) -> tuple[Path, Path, dict]:
    _, tool, _ = fixture(root)
    candidate_root = root / "candidate"
    for path in candidate_root.iterdir():
        path.unlink()
    source = chart_source(root)
    release = "1.1.0-rc.0"
    artifact = candidate_root / f"tritium-{release}.tgz"
    HELM_PACKAGER["package"](source, artifact, release)
    artifact_id = "tritium-helm"
    revision = "1" * 40
    sbom = DEPLOYMENT_SBOM["generate"](
        artifact,
        artifact_id,
        "helm-chart",
        release,
        revision,
        str(tool),
    )
    sbom_sha = write_json(candidate_root / "helm.cdx.json", sbom)
    payload = artifact.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    provenance_sha = write_json(
        candidate_root / "helm.provenance.json",
        {
            "_type": "https://in-toto.io/Statement/v1",
            "subject": [{"name": artifact.name, "digest": {"sha256": sha256}}],
            "predicateType": "https://slsa.dev/provenance/v1",
            "predicate": {
                "buildDefinition": {
                    "externalParameters": {"source_revision": revision}
                },
                "runDetails": {
                    "builder": {"id": "test-builder"},
                    "metadata": {"invocationID": "test-run"},
                },
            },
        },
    )
    document = {
        "schema": "tritium.release-candidate.v1",
        "release": release,
        "source_revision": revision,
        "artifacts": [
            {
                "id": artifact_id,
                "kind": "helm-chart",
                "path": artifact.name,
                "identity": {
                    "schema": "tritium.file-identity.v1",
                    "bytes": len(payload),
                    "sha256": sha256,
                    "blake3": fake_blake3(payload),
                },
                "sbom": {"path": "helm.cdx.json", "sha256": sbom_sha},
                "provenance": {
                    "path": "helm.provenance.json",
                    "sha256": provenance_sha,
                },
            }
        ],
    }
    candidate = candidate_root / "manifest.json"
    write_json(candidate, document)
    return candidate, tool, document


def oci_candidate_fixture(root: Path) -> tuple[Path, Path, dict]:
    _, tool, _ = fixture(root)
    candidate_root = root / "candidate"
    for path in candidate_root.iterdir():
        path.unlink()
    oci_root = root / "oci-source"
    oci_root.mkdir()
    source_archive, _, _ = oci_archive_fixture(oci_root)
    artifact = candidate_root / "image.oci.tar"
    artifact.write_bytes(source_archive.read_bytes())
    artifact_id = "tritium-serve-cpu"
    release = "1.1.0-rc.0"
    revision = "a" * 40
    sbom = DEPLOYMENT_SBOM["generate"](
        artifact,
        artifact_id,
        "oci-image",
        release,
        revision,
        str(tool),
    )
    sbom_sha = write_json(candidate_root / "oci.cdx.json", sbom)
    payload = artifact.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    provenance_sha = write_json(
        candidate_root / "oci.provenance.json",
        {
            "_type": "https://in-toto.io/Statement/v1",
            "subject": [{"name": artifact.name, "digest": {"sha256": sha256}}],
            "predicateType": "https://slsa.dev/provenance/v1",
            "predicate": {
                "buildDefinition": {
                    "externalParameters": {
                        "source_revision": revision,
                        "oci_builder_id": BUILDER_ID,
                        "oci_invocation_id": "buildkit-invocation-123",
                    }
                },
                "runDetails": {
                    "builder": {"id": "test-builder"},
                    "metadata": {"invocationID": "test-run"},
                },
            },
        },
    )
    document = {
        "schema": "tritium.release-candidate.v1",
        "release": release,
        "source_revision": revision,
        "artifacts": [
            {
                "id": artifact_id,
                "kind": "oci-image",
                "path": artifact.name,
                "identity": {
                    "schema": "tritium.file-identity.v1",
                    "bytes": len(payload),
                    "sha256": sha256,
                    "blake3": fake_blake3(payload),
                },
                "sbom": {"path": "oci.cdx.json", "sha256": sbom_sha},
                "provenance": {
                    "path": "oci.provenance.json",
                    "sha256": provenance_sha,
                },
            }
        ],
    }
    candidate = candidate_root / "manifest.json"
    write_json(candidate, document)
    return candidate, tool, document


class ReleaseStatusTests(unittest.TestCase):
    def test_json_report_publish_is_atomic_and_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "status.json"
            write_report(output, {"ready": False})
            self.assertEqual(json.loads(output.read_text()), {"ready": False})
            output.unlink()
            target = root / "target.json"
            target.write_text("unchanged", encoding="utf-8")
            output.symlink_to(target)
            with self.assertRaises(MODULE["ReleaseError"]):
                write_report(output, {"ready": True})
            self.assertEqual(target.read_text(encoding="utf-8"), "unchanged")

    def test_candidate_binds_artifact_sbom_and_provenance(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, tool, document = fixture(Path(raw))
            self.assertEqual(validate(candidate, str(tool)), document)

    def test_oci_candidate_binds_inner_buildkit_identity_in_outer_provenance(self):
        for field, replacement in (
            ("builder", "https://github.com/Quitetall/tritium/actions/runs/999"),
            ("invocation", "other-buildkit-invocation"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                candidate, tool, document = oci_candidate_fixture(Path(raw))
                reference = document["artifacts"][0]["provenance"]
                path = candidate.parent / reference["path"]
                provenance = json.loads(path.read_text())
                key = "oci_builder_id" if field == "builder" else "oci_invocation_id"
                provenance["predicate"]["buildDefinition"]["externalParameters"][
                    key
                ] = replacement
                reference["sha256"] = write_json(path, provenance)
                write_json(candidate, document)
                with self.assertRaisesRegex(ReleaseError, "builder|invocation"):
                    validate(candidate, str(tool))

    def test_candidate_rejects_identity_and_lineage_drift(self):
        for mutation in ("artifact", "sbom", "provenance", "revision"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                candidate, tool, document = fixture(root)
                candidate_root = candidate.parent
                if mutation == "artifact":
                    (candidate_root / "wheel.whl").write_bytes(b"changed")
                elif mutation == "sbom":
                    (candidate_root / "wheel.cdx.json").write_text("{}", encoding="utf-8")
                elif mutation == "provenance":
                    (candidate_root / "wheel.provenance.json").write_text("{}", encoding="utf-8")
                else:
                    document["source_revision"] = "2" * 40
                    write_json(candidate, document)
                with self.assertRaises(ReleaseError):
                    validate(candidate, str(tool))

    def test_candidate_rejects_cosmetic_cyclonedx_root(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, tool, document = fixture(Path(raw))
            sbom_path = candidate.parent / "wheel.cdx.json"
            sbom = json.loads(sbom_path.read_text())
            sbom["metadata"]["component"]["hashes"][0]["content"] = "0" * 64
            document["artifacts"][0]["sbom"]["sha256"] = write_json(sbom_path, sbom)
            write_json(candidate, document)
            with self.assertRaisesRegex(ReleaseError, "exact artifact SHA-256"):
                validate(candidate, str(tool))

    def test_bundle_candidate_requires_canonical_complete_member_inventory(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, tool, document = bundle_fixture(Path(raw))
            self.assertEqual(validate(candidate, str(tool)), document)
            sbom_path = candidate.parent / "onnx.cdx.json"
            sbom = json.loads(sbom_path.read_text())
            sbom["components"] = []
            sbom["dependencies"][0]["dependsOn"] = []
            document["artifacts"][0]["sbom"]["sha256"] = write_json(sbom_path, sbom)
            write_json(candidate, document)
            with self.assertRaisesRegex(ReleaseError, "canonical bundle inventory"):
                validate(candidate, str(tool))

    def test_helm_candidate_requires_canonical_complete_member_inventory(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, tool, document = helm_fixture(Path(raw))
            self.assertEqual(validate(candidate, str(tool)), document)
            sbom_path = candidate.parent / "helm.cdx.json"
            sbom = json.loads(sbom_path.read_text())
            sbom["components"] = []
            sbom["dependencies"][0]["dependsOn"] = []
            document["artifacts"][0]["sbom"]["sha256"] = write_json(sbom_path, sbom)
            write_json(candidate, document)
            with self.assertRaisesRegex(ReleaseError, "canonical deployment inventory"):
                validate(candidate, str(tool))

    def test_oci_candidate_requires_canonical_complete_member_inventory(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, tool, document = oci_candidate_fixture(Path(raw))
            self.assertEqual(validate(candidate, str(tool)), document)
            sbom_path = candidate.parent / "oci.cdx.json"
            sbom = json.loads(sbom_path.read_text())
            sbom["components"] = []
            sbom["dependencies"][0]["dependsOn"] = []
            document["artifacts"][0]["sbom"]["sha256"] = write_json(sbom_path, sbom)
            write_json(candidate, document)
            with self.assertRaisesRegex(ReleaseError, "canonical deployment inventory"):
                validate(candidate, str(tool))

    def test_spdx_candidate_requires_exact_described_package(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, tool, document = fixture(Path(raw))
            artifact = candidate.parent / "wheel.whl"
            artifact_id = document["artifacts"][0]["id"]
            checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
            sbom = {
                "spdxVersion": "SPDX-2.3",
                "SPDXID": "SPDXRef-DOCUMENT",
                "name": artifact_id,
                "documentDescribes": ["SPDXRef-Artifact"],
                "packages": [
                    {
                        "SPDXID": "SPDXRef-Artifact",
                        "name": artifact_id,
                        "packageFileName": artifact.name,
                        "checksums": [
                            {"algorithm": "SHA256", "checksumValue": checksum}
                        ],
                    }
                ],
            }
            sbom_path = candidate.parent / "wheel.cdx.json"
            document["artifacts"][0]["sbom"]["sha256"] = write_json(sbom_path, sbom)
            write_json(candidate, document)
            self.assertEqual(validate(candidate, str(tool)), document)
            sbom["packages"][0]["checksums"][0]["checksumValue"] = "0" * 64
            document["artifacts"][0]["sbom"]["sha256"] = write_json(sbom_path, sbom)
            write_json(candidate, document)
            with self.assertRaisesRegex(ReleaseError, "exact SPDX artifact"):
                validate(candidate, str(tool))

    def test_candidate_rejects_symlinked_artifact(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, tool, _ = fixture(root)
            candidate_root = candidate.parent
            artifact = candidate_root / "wheel.whl"
            artifact.rename(candidate_root / "real.whl")
            artifact.symlink_to("real.whl")
            with self.assertRaisesRegex(ReleaseError, "symlink"):
                validate(candidate, str(tool))

    def test_candidate_rejects_unmanifested_file(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, tool, _ = fixture(root)
            (candidate.parent / "untracked.bin").write_bytes(b"not admitted")
            with self.assertRaisesRegex(ReleaseError, "unknown untracked.bin"):
                validate(candidate, str(tool))

    def test_candidate_rejects_unmanifested_empty_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, tool, _ = fixture(root)
            (candidate.parent / "untracked").mkdir()
            with self.assertRaisesRegex(ReleaseError, "directory topology"):
                validate(candidate, str(tool))


if __name__ == "__main__":
    unittest.main()

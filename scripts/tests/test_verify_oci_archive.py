from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import runpy
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "verify-oci-archive.py")
validate = MODULE["validate"]
OciError = MODULE["OciError"]
MANIFEST = "application/vnd.oci.image.manifest.v1+json"


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def descriptor(payload: bytes, media_type: str) -> dict:
    return {"mediaType": media_type, "digest": "sha256:" + hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def fixture(root: Path, *, wrong_subject: bool = False):
    revision = "a" * 40
    release = "1.1.0-rc.0"
    candidate = root / "manifest.json"
    candidate.write_bytes(encoded({"release": release, "source_revision": revision}))
    config = encoded({"config": {
        "User": "65532:65532", "Entrypoint": ["/usr/local/bin/tritium-serve"],
        "Labels": {
            "org.opencontainers.image.revision": revision,
            "org.opencontainers.image.version": release,
            "io.tritium.artifact.schema": "3",
            "io.tritium.startup-receipt.schema": "1",
        },
    }})
    config_desc = descriptor(config, "application/vnd.oci.image.config.v1+json")
    image = encoded({"schemaVersion": 2, "config": config_desc, "layers": []})
    image_desc = descriptor(image, MANIFEST)
    image_desc["platform"] = {"architecture": "amd64", "os": "linux"}
    image_digest = image_desc["digest"]
    blobs = {config_desc["digest"][7:]: config, image_desc["digest"][7:]: image}
    layers = []
    for predicate in ("https://spdx.dev/Document", "https://slsa.dev/provenance/v1"):
        subject = "f" * 64 if wrong_subject else image_digest[7:]
        payload = encoded({"_type": "https://in-toto.io/Statement/v1", "predicateType": predicate, "subject": [{"name": "image", "digest": {"sha256": subject}}], "predicate": {}})
        item = descriptor(payload, "application/vnd.in-toto+json")
        item["annotations"] = {"in-toto.io/predicate-type": predicate}
        blobs[item["digest"][7:]] = payload
        layers.append(item)
    attestation = encoded({"schemaVersion": 2, "config": config_desc, "layers": layers})
    attestation_desc = descriptor(attestation, MANIFEST)
    attestation_desc["platform"] = {"architecture": "unknown", "os": "unknown"}
    attestation_desc["annotations"] = {
        "vnd.docker.reference.type": "attestation-manifest",
        "vnd.docker.reference.digest": image_digest,
    }
    blobs[attestation_desc["digest"][7:]] = attestation
    index = encoded({"schemaVersion": 2, "manifests": [image_desc, attestation_desc]})
    archive = root / "image.oci.tar"
    with tarfile.open(archive, "w") as tar:
        files = {"oci-layout": encoded({"imageLayoutVersion": "1.0.0"}), "index.json": index}
        files.update({f"blobs/sha256/{name}": payload for name, payload in blobs.items()})
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    receipt = root / "build-receipt.json"
    receipt.write_bytes(encoded({
        "schema": "tritium.oci-build.v1", "release": release, "flavor": "cpu",
        "candidate_manifest_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "source_revision": revision, "source_created": "2026-07-21T00:00:00Z",
        "source_date_epoch": 1, "source_archive": "source.tar", "source_archive_bytes": 1,
        "source_archive_sha256": "b" * 64, "platform": "linux/amd64",
        "archive": archive.name, "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }))
    return archive, receipt, candidate


class VerifyOciArchiveTests(unittest.TestCase):
    def test_accepts_image_bound_sbom_and_provenance(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(Path(raw))
            result = validate(archive, receipt, candidate)
            self.assertEqual(len(result["predicates"]), 2)

    def test_rejects_attestation_for_different_image(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(Path(raw), wrong_subject=True)
            with self.assertRaisesRegex(OciError, "subject"):
                validate(archive, receipt, candidate)

    def test_rejects_archive_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(Path(raw))
            archive.write_bytes(archive.read_bytes() + b"drift")
            with self.assertRaisesRegex(OciError, "identity|SHA-256"):
                validate(archive, receipt, candidate)


if __name__ == "__main__":
    unittest.main()

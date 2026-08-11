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
INDEX = "application/vnd.oci.image.index.v1+json"
BUILDER_ID = "https://github.com/Quitetall/tritium/actions/runs/123"
BUILDKIT_BUILD_TYPE = (
    "https://github.com/moby/buildkit/blob/master/"
    "docs/attestations/slsa-definitions.md"
)


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def descriptor(payload: bytes, media_type: str) -> dict:
    return {
        "mediaType": media_type,
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def fixture(
    root: Path,
    *,
    wrong_subject: bool = False,
    extra_blob: bool = False,
    fake_predicates: bool = False,
    empty_predicates: bool = False,
    provenance_revision: str | None = None,
    builder_id: str = BUILDER_ID,
    receipt_builder_id: str | None = None,
    missing_spdx_download: bool = False,
    malformed_spdx: bool = False,
    impossible_spdx_timestamp: bool = False,
    invalid_llb: bool = False,
    invalid_dependency_digest: bool = False,
    reversed_slsa_timestamps: bool = False,
    pax_metadata: bool = False,
    nested_index: bool = False,
    empty_subject: bool = False,
    invocation_id_key: str = "invocationID",
):
    revision = "a" * 40
    release = "1.1.0-rc.0"
    candidate = root / "manifest.json"
    candidate.write_bytes(encoded({"release": release, "source_revision": revision}))
    config = encoded(
        {
            "config": {
                "User": "65532:65532",
                "Entrypoint": ["/usr/local/bin/tritium-serve"],
                "Labels": {
                    "org.opencontainers.image.revision": revision,
                    "org.opencontainers.image.version": release,
                    "io.tritium.artifact.schema": "3",
                    "io.tritium.startup-receipt.schema": "1",
                },
            }
        }
    )
    config_desc = descriptor(config, "application/vnd.oci.image.config.v1+json")
    image = encoded({"schemaVersion": 2, "config": config_desc, "layers": []})
    image_desc = descriptor(image, MANIFEST)
    image_desc["platform"] = {"architecture": "amd64", "os": "linux"}
    image_digest = image_desc["digest"]
    blobs = {config_desc["digest"][7:]: config, image_desc["digest"][7:]: image}
    spdx = {
        "SPDXID": "SPDXRef-DOCUMENT",
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "name": "tritium-serve",
        "documentNamespace": "https://tritium.ai/spdx/tritium-serve-test",
        "creationInfo": {
            "created": "2026-07-21T00:00:00Z",
            "creators": ["Tool: test-sbom-generator"],
        },
        "packages": [
            {
                "SPDXID": "SPDXRef-Package-tritium-serve",
                "name": "tritium-serve",
                "filesAnalyzed": False,
                "downloadLocation": "NOASSERTION",
            }
        ],
    }
    if missing_spdx_download:
        del spdx["packages"][0]["downloadLocation"]
    if malformed_spdx:
        spdx["creationInfo"]["created"] = "not-a-timestamp"
        spdx["creationInfo"]["creators"] = ["untyped creator"]
        spdx["packages"][0]["SPDXID"] = "SPDXRef-invalid id"
    if impossible_spdx_timestamp:
        spdx["creationInfo"]["created"] = "2026-99-99T99:99:99Z"
    slsa = {
        "buildDefinition": {
            "buildType": BUILDKIT_BUILD_TYPE,
            "externalParameters": {
                "configSource": {"path": "deploy/oci/Dockerfile.cpu"},
                "request": {
                    "frontend": "dockerfile.v0",
                    "args": {
                        "build-arg:SOURCE_REVISION": provenance_revision or revision,
                    },
                },
            },
            "internalParameters": {
                "buildConfig": {
                    "llbDefinition": [{"id": "step0", "op": {"Op": {}}}]
                },
                "builderPlatform": "linux/amd64",
            },
            "resolvedDependencies": [
                {
                    "uri": "pkg:docker/base@sha256:test",
                    "digest": {"sha256": "c" * 64},
                }
            ],
        },
        "runDetails": {
            "builder": {"id": builder_id},
            "metadata": {
                invocation_id_key: "buildkit-invocation-123",
                "startedOn": "2026-07-21T00:00:00Z",
                "finishedOn": "2026-07-21T00:01:00Z",
            },
        },
    }
    if invalid_llb:
        slsa["buildDefinition"]["internalParameters"]["buildConfig"][
            "llbDefinition"
        ] = ["not-an-object"]
    if invalid_dependency_digest:
        slsa["buildDefinition"]["resolvedDependencies"][0]["digest"] = {
            "bad": "not-a-digest"
        }
    if reversed_slsa_timestamps:
        slsa["runDetails"]["metadata"]["startedOn"] = "2026-07-21T00:02:00Z"
        slsa["runDetails"]["metadata"]["finishedOn"] = "2026-07-21T00:01:00Z"
    layers = []
    for predicate in ("https://spdx.dev/Document", "https://slsa.dev/provenance/v1"):
        if fake_predicates:
            predicate = "https://evil.invalid/spdx-and-slsa"
        subject = "f" * 64 if wrong_subject else image_digest[7:]
        body = spdx if predicate == "https://spdx.dev/Document" else slsa
        statement = {
            "_type": "https://in-toto.io/Statement/v1",
            "predicateType": predicate,
            "subject": []
            if empty_subject
            else [{"name": "image", "digest": {"sha256": subject}}],
            "predicate": {} if empty_predicates else body,
        }
        payload = encoded(statement)
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
    if extra_blob:
        orphan = b"unreferenced OCI payload"
        blobs[hashlib.sha256(orphan).hexdigest()] = orphan
    nested = encoded({"schemaVersion": 2, "manifests": [image_desc, attestation_desc]})
    if nested_index:
        nested_desc = descriptor(nested, INDEX)
        blobs[nested_desc["digest"][7:]] = nested
        index = encoded({"schemaVersion": 2, "manifests": [nested_desc]})
    else:
        index = nested
    archive = root / "image.oci.tar"
    with tarfile.open(archive, "w") as tar:
        files = {"oci-layout": encoded({"imageLayoutVersion": "1.0.0"}), "index.json": index}
        files.update({f"blobs/sha256/{name}": payload for name, payload in blobs.items()})
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            if pax_metadata and name == "oci-layout":
                info.pax_headers = {"comment": "hidden transport metadata"}
            tar.addfile(info, io.BytesIO(payload))
    receipt = root / "build-receipt.json"
    receipt.write_bytes(encoded({
        "schema": "tritium.oci-build.v1", "release": release, "flavor": "cpu",
        "candidate_manifest_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "source_revision": revision, "source_created": "2026-07-21T00:00:00Z",
        "source_date_epoch": 1, "source_archive": "source.tar", "source_archive_bytes": 1,
        "source_archive_sha256": "b" * 64, "platform": "linux/amd64",
        "builder_id": receipt_builder_id or BUILDER_ID,
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
            self.assertEqual(result["builder_id"], BUILDER_ID)
            self.assertEqual(result["invocation_id"], "buildkit-invocation-123")

    def test_accepts_buildkit_nested_attested_index(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(Path(raw), nested_index=True)
            result = validate(archive, receipt, candidate)
            self.assertEqual(len(result["predicates"]), 2)

    def test_accepts_buildkit_empty_attestation_subject(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(
                Path(raw), nested_index=True, empty_subject=True,
                invocation_id_key="invocationId",
            )
            result = validate(archive, receipt, candidate)
            self.assertEqual(len(result["predicates"]), 2)

    def test_rejects_empty_predicate_documents(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(Path(raw), empty_predicates=True)
            with self.assertRaisesRegex(OciError, "SPDX|provenance"):
                validate(archive, receipt, candidate)

    def test_rejects_spdx_missing_mandatory_package_download_location(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(
                Path(raw), missing_spdx_download=True
            )
            with self.assertRaisesRegex(OciError, "SPDX"):
                validate(archive, receipt, candidate)

    def test_rejects_malformed_spdx_timestamp_creator_and_identifier(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(Path(raw), malformed_spdx=True)
            with self.assertRaisesRegex(OciError, "SPDX"):
                validate(archive, receipt, candidate)

    def test_rejects_impossible_spdx_timestamp(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(
                Path(raw), impossible_spdx_timestamp=True
            )
            with self.assertRaisesRegex(OciError, "SPDX.*timestamp"):
                validate(archive, receipt, candidate)

    def test_rejects_non_object_llb_definition(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(Path(raw), invalid_llb=True)
            with self.assertRaisesRegex(OciError, "LLB"):
                validate(archive, receipt, candidate)

    def test_rejects_invalid_dependency_digest(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(
                Path(raw), invalid_dependency_digest=True
            )
            with self.assertRaisesRegex(OciError, "digest"):
                validate(archive, receipt, candidate)

    def test_rejects_reversed_slsa_timestamps(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(
                Path(raw), reversed_slsa_timestamps=True
            )
            with self.assertRaisesRegex(OciError, "finish"):
                validate(archive, receipt, candidate)

    def test_rejects_hidden_pax_transport_headers(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(Path(raw), pax_metadata=True)
            with self.assertRaisesRegex(OciError, "extension|PAX"):
                validate(archive, receipt, candidate)

    def test_rejects_provenance_for_different_source_revision(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(
                Path(raw), provenance_revision="d" * 40
            )
            with self.assertRaisesRegex(OciError, "source revision"):
                validate(archive, receipt, candidate)

    def test_rejects_absent_builder_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(Path(raw), builder_id="")
            with self.assertRaisesRegex(OciError, "builder"):
                validate(archive, receipt, candidate)

    def test_rejects_malformed_builder_identity_as_contract_error(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(Path(raw), builder_id="https://[")
            with self.assertRaisesRegex(OciError, "builder"):
                validate(archive, receipt, candidate)

    def test_rejects_builder_identity_different_from_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(
                Path(raw),
                receipt_builder_id="https://github.com/Quitetall/tritium/actions/runs/999",
            )
            with self.assertRaisesRegex(OciError, "builder"):
                validate(archive, receipt, candidate)

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

    def test_rejects_unreferenced_blob(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(Path(raw), extra_blob=True)
            with self.assertRaisesRegex(OciError, "unreferenced"):
                validate(archive, receipt, candidate)

    def test_rejects_predicate_names_that_only_contain_spdx_and_slsa(self):
        with tempfile.TemporaryDirectory() as raw:
            archive, receipt, candidate = fixture(Path(raw), fake_predicates=True)
            with self.assertRaisesRegex(OciError, "SBOM and provenance"):
                validate(archive, receipt, candidate)


if __name__ == "__main__":
    unittest.main()

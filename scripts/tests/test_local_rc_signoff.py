from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "local-rc-signoff.py")
SignoffError = MODULE["SignoffError"]
seal = MODULE["seal"]
verify = MODULE["verify"]


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LocalRcSignoffTests(unittest.TestCase):
    def fixture(self, root: Path, *, ready: bool = True):
        registry = root / "registry.json"
        registry.write_bytes(b'{"schema":"registry"}\n')
        candidate_root = root / "candidate"
        candidate_root.mkdir()
        payload = candidate_root / "payload.bin"
        payload.write_bytes(b"payload")
        payload_sha = digest(payload)
        payload_bytes = payload.stat().st_size
        sbom = candidate_root / "payload.cdx.json"
        sbom.write_bytes(canonical({
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "metadata": {"component": {
                "bom-ref": "payload-artifact",
                "hashes": [{"alg": "SHA-256", "content": payload_sha}],
                "properties": [
                    {"name": "tritium:artifact:file", "value": payload.name},
                    {"name": "tritium:artifact:bytes", "value": str(payload_bytes)},
                ],
            }},
        }))
        provenance = candidate_root / "payload.provenance.json"
        provenance.write_bytes(canonical({
            "_type": "https://in-toto.io/Statement/v1",
            "predicateType": "https://slsa.dev/provenance/v1",
            "subject": [{"name": payload.name, "digest": {"sha256": payload_sha}}],
            "predicate": {
                "buildDefinition": {"externalParameters": {
                    "source_revision": "a" * 40,
                }},
                "runDetails": {
                    "builder": {"id": "https://builder.example"},
                    "metadata": {"invocationID": "test-invocation"},
                },
            },
        }))
        candidate = candidate_root / "manifest.json"
        candidate.write_bytes(canonical({
            "schema": "tritium.release-candidate.v1",
            "release": "1.1.0-rc.0",
            "source_revision": "a" * 40,
            "artifacts": [{
                "id": "payload-artifact",
                "kind": "source-archive",
                "path": payload.name,
                "identity": {
                    "schema": "tritium.file-identity.v1",
                    "bytes": payload_bytes,
                    "sha256": payload_sha,
                    "blake3": "0" * 64,
                },
                "sbom": {"path": sbom.name, "sha256": digest(sbom)},
                "provenance": {"path": provenance.name, "sha256": digest(provenance)},
            }],
        }))
        digest_tool = root / "digest-tool.py"
        digest_tool.write_text(
            "#!/usr/bin/env python3\n"
            "import hashlib, json, pathlib, sys\n"
            "path = pathlib.Path(sys.argv[-1])\n"
            "print(json.dumps({'schema': 'tritium.file-identity.v1',"
            "'bytes': path.stat().st_size,"
            "'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),"
            "'blake3': '0' * 64}))\n",
            encoding="utf-8",
        )
        digest_tool.chmod(digest_tool.stat().st_mode | 0o111)
        report = root / "status.json"
        report.write_bytes(canonical({
            "schema": "tritium.release-gate-report.v1",
            "release": "1.1.0-rc.0",
            "source_revision": "a" * 40,
            "candidate_manifest_sha256": digest(candidate),
            "evidence_registry_sha256": digest(registry),
            "ready": ready,
            "rows": [],
            "external_activation": "EXTERNAL_AUTH_REQUIRED",
        }))
        return report, registry, candidate, digest_tool

    def test_seals_and_verifies_exact_evidence(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report, registry, candidate, digest_tool = self.fixture(root)
            key = root / "release-key"
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
                check=True,
            )
            allowed = root / "allowed_signers"
            allowed.write_text(
                "release-maintainer " + key.with_suffix(".pub").read_text(),
                encoding="utf-8",
            )
            output = root / "signoff.json"
            seal(report, registry, candidate, "release-maintainer", key, output,
                 str(digest_tool))
            value = verify(
                output, Path(str(output) + ".sig"), report, registry, candidate,
                "release-maintainer", allowed, str(digest_tool),
            )
            self.assertEqual(value["release"], "1.1.0-rc.0")

            registry.write_bytes(b'{"schema":"tampered"}\n')
            with self.assertRaisesRegex(SignoffError, "exact registry"):
                verify(
                    output, Path(str(output) + ".sig"), report, registry, candidate,
                    "release-maintainer", allowed, str(digest_tool),
                )

    def test_refuses_partial_report(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report, registry, candidate, digest_tool = self.fixture(root, ready=False)
            key = root / "key"
            key.write_bytes(b"not reached")
            with self.assertRaisesRegex(SignoffError, "complete passing"):
                seal(report, registry, candidate, "maintainer", key,
                     root / "signoff.json", str(digest_tool))

    def test_refuses_candidate_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report, registry, candidate, digest_tool = self.fixture(root)
            candidate_value = json.loads(candidate.read_bytes())
            candidate_value["release"] = "1.1.0-rc.1"
            candidate.write_bytes(canonical(candidate_value))
            report_value = json.loads(report.read_bytes())
            report_value["candidate_manifest_sha256"] = digest(candidate)
            report.write_bytes(canonical(report_value))
            with self.assertRaisesRegex(SignoffError, "candidate release differs"):
                seal(report, registry, candidate, "maintainer", root / "missing-key",
                     root / "signoff.json", str(digest_tool))

    def test_refuses_candidate_source_revision_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report, registry, candidate, digest_tool = self.fixture(root)
            provenance = candidate.parent / "payload.provenance.json"
            provenance_value = json.loads(provenance.read_bytes())
            provenance_value["predicate"]["buildDefinition"]["externalParameters"][
                "source_revision"
            ] = "b" * 40
            provenance.write_bytes(canonical(provenance_value))
            candidate_value = json.loads(candidate.read_bytes())
            candidate_value["source_revision"] = "b" * 40
            candidate_value["artifacts"][0]["provenance"]["sha256"] = digest(provenance)
            candidate.write_bytes(canonical(candidate_value))
            report_value = json.loads(report.read_bytes())
            report_value["candidate_manifest_sha256"] = digest(candidate)
            report.write_bytes(canonical(report_value))
            with self.assertRaisesRegex(
                SignoffError, "candidate source revision differs"
            ):
                seal(report, registry, candidate, "maintainer", root / "missing-key",
                     root / "signoff.json", str(digest_tool))

    def test_refuses_wrong_candidate_schema(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report, registry, candidate, digest_tool = self.fixture(root)
            candidate_value = json.loads(candidate.read_bytes())
            candidate_value["schema"] = "wrong"
            candidate.write_bytes(canonical(candidate_value))
            report_value = json.loads(report.read_bytes())
            report_value["candidate_manifest_sha256"] = digest(candidate)
            report.write_bytes(canonical(report_value))
            with self.assertRaisesRegex(SignoffError, "candidate validation failed"):
                seal(report, registry, candidate, "maintainer", root / "missing-key",
                     root / "signoff.json", str(digest_tool))

    def test_verify_rejects_candidate_schema_mutation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report, registry, candidate, digest_tool = self.fixture(root)
            key = root / "release-key"
            subprocess.run(
                ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
                check=True,
            )
            allowed = root / "allowed_signers"
            allowed.write_text(
                "release-maintainer " + key.with_suffix(".pub").read_text(),
                encoding="utf-8",
            )
            output = root / "signoff.json"
            seal(report, registry, candidate, "release-maintainer", key, output,
                 str(digest_tool))
            candidate_value = json.loads(candidate.read_bytes())
            candidate_value["schema"] = "wrong"
            candidate.write_bytes(canonical(candidate_value))
            report_value = json.loads(report.read_bytes())
            report_value["candidate_manifest_sha256"] = digest(candidate)
            report.write_bytes(canonical(report_value))
            with self.assertRaisesRegex(SignoffError, "candidate validation failed"):
                verify(
                    output, Path(str(output) + ".sig"), report, registry, candidate,
                    "release-maintainer", allowed, str(digest_tool),
                )

if __name__ == "__main__":
    unittest.main()

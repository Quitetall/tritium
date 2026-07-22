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
        candidate = root / "manifest.json"
        candidate.write_bytes(b'{"schema":"candidate"}\n')
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
        return report, registry, candidate

    def test_seals_and_verifies_exact_evidence(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report, registry, candidate = self.fixture(root)
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
            seal(report, registry, candidate, "release-maintainer", key, output)
            value = verify(
                output, Path(str(output) + ".sig"), report, registry, candidate,
                "release-maintainer", allowed,
            )
            self.assertEqual(value["release"], "1.1.0-rc.0")

            registry.write_bytes(b'{"schema":"tampered"}\n')
            with self.assertRaisesRegex(SignoffError, "exact registry"):
                verify(
                    output, Path(str(output) + ".sig"), report, registry, candidate,
                    "release-maintainer", allowed,
                )

    def test_refuses_partial_report(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report, registry, candidate = self.fixture(root, ready=False)
            key = root / "key"
            key.write_bytes(b"not reached")
            with self.assertRaisesRegex(SignoffError, "complete passing"):
                seal(report, registry, candidate, "maintainer", key,
                     root / "signoff.json")


if __name__ == "__main__":
    unittest.main()

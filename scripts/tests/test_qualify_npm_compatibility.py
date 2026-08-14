from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "qualify-npm-compatibility.py"
SPEC = importlib.util.spec_from_file_location("qualify_npm_compatibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def qualified() -> dict:
    revision = "a" * 40
    release = "1.1.0-rc.0"
    return {
        "schema": "tritium.npm-archive-qualification.v1",
        "receipt_id": "sha256:" + "b" * 64,
        "release": release,
        "source_revision": revision,
        "machine": {
            "machine_id": "sha256:" + "c" * 64,
            "system": "linux",
            "architecture": "x64",
        },
        "toolchain": {"node": "v22.18.0", "npm": "11.5.2"},
        "artifact": {
            "kind": "npm-archive",
            "name": "tritium-ai-web-1.1.0-rc.0.tgz",
            "package": "@tritium-ai/web@1.1.0-rc.0",
            "bytes": 123,
            "sha256": "d" * 64,
        },
        "evidence": {
            "source_free": True,
            "installed_offline": True,
            "strict_typescript": True,
            "wasm_build_id": f"tritium-wasm@{release}+source-git:{revision}",
            "wasm_guest_digest": "e" * 64,
        },
    }


class QualifyNpmCompatibilityTests(unittest.TestCase):
    def test_projects_validated_qualification(self):
        value = MODULE.project(qualified())
        self.assertEqual(value["target_id"], "node-22")
        self.assertEqual(value["source_revision"], "a" * 40)
        MODULE.validate_project(value, "a" * 40, "1.1.0-rc.0")

    def test_rejects_release_or_source_drift(self):
        value = MODULE.project(qualified())
        with self.assertRaisesRegex(MODULE.CompatibilityError, "source revision"):
            MODULE.validate_project(value, "f" * 40, "1.1.0-rc.0")
        with self.assertRaisesRegex(MODULE.CompatibilityError, "package/release"):
            MODULE.validate_project(value, "a" * 40, "1.1.0-rc.1")

    def test_rejects_extra_fields(self):
        value = MODULE.project(qualified())
        value["extra"] = True
        with self.assertRaisesRegex(MODULE.CompatibilityError, "fields"):
            MODULE.validate_project(value)

    def test_rejects_unbound_upstream_identity(self):
        value = MODULE.project(qualified())
        value["upstream_receipt_id"] = "not-a-digest"
        with self.assertRaisesRegex(MODULE.CompatibilityError, "upstream_receipt_id"):
            MODULE.validate_project(value)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-release-reproduction.py")
assemble = MODULE["assemble"]
QualificationError = MODULE["QualificationError"]


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def fixture(root: Path, *, failing_command: str | None = None):
    wheel = root / "candidate.whl"
    web = root / "web.tgz"
    wheel.write_bytes(b"wheel")
    web.write_bytes(b"web")
    candidate = root / "manifest.json"
    artifacts = []
    for artifact_id, kind, path in (
        ("wheel", "python-wheel", wheel), ("web", "npm-archive", web)
    ):
        artifacts.append({
            "id": artifact_id, "kind": kind, "path": path.name,
            "identity": {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        })
    write(candidate, {"artifacts": artifacts})

    work = root / "work"
    work.mkdir()
    checks = {field: True for field in MODULE["CHECK_FIELDS"]}
    checks["browser"] = "not-applicable"
    write(work / "checks.json", checks)
    expected = root / "expected"
    observed = work / "generated"
    expected.mkdir()
    observed.mkdir()
    outputs = []
    for name in sorted(MODULE["REQUIRED_OUTPUTS"]):
        (expected / f"{name}.json").write_text(name + "\n", encoding="utf-8")
        (observed / f"{name}.json").write_text(name + "\n", encoding="utf-8")
        outputs.append({
            "name": name, "expected_path": f"expected/{name}.json",
            "observed_path": f"generated/{name}.json",
        })
    commands = []
    for command_id in MODULE["COMMAND_ORDER"]:
        exit_code = "7" if command_id == failing_command else "0"
        commands.append({
            "id": command_id, "argv": ["/bin/sh", "-c", f"exit {exit_code}"],
            "timeout_seconds": 5,
        })
    spec = root / "spec.json"
    write(spec, {
        "schema": MODULE["SPEC_SCHEMA"], "release": "1.1.0-rc.0",
        "source_revision": "a" * 40, "run_id": "second-machine-9",
        "operator": {
            "id": "operator-2", "organization": "Independent Lab",
            "independent": True,
        },
        "machine": {
            "machine_id": "sha256:" + "2" * 64, "system": "Linux",
            "version": "6.17", "architecture": "x86_64", "cpu": "Zen 5",
            "gpus": ["GPU-physical"],
        },
        "primary_machine_id": "sha256:" + "1" * 64,
        "commands": commands, "checks_path": "checks.json", "outputs": outputs,
    })
    return candidate, wheel, spec, work


class QualifyReleaseReproductionTests(unittest.TestCase):
    def setUp(self):
        self.old_compilers = assemble.__globals__["COMPILERS"]
        assemble.__globals__["COMPILERS"] = ()

    def tearDown(self):
        assemble.__globals__["COMPILERS"] = self.old_compilers

    def test_runs_and_seals_content_bound_reproduction(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, wheel, spec, work = fixture(root)
            output = root / "receipt"
            receipt = assemble(
                output, candidate=candidate, anchor=wheel, spec_path=spec,
                work_dir=work, source_revision="a" * 40, release="1.1.0-rc.0",
            )
            self.assertEqual(receipt["result"], "pass")
            self.assertEqual(
                [command["id"] for command in receipt["commands"]],
                list(MODULE["COMMAND_ORDER"]),
            )
            self.assertEqual(len(list((output / "logs").iterdir())), 26)
            self.assertEqual(len(list((output / "outputs").iterdir())), 3)

    def test_fails_closed_on_command_error(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, wheel, spec, work = fixture(
                root, failing_command="qwen-flagship"
            )
            with self.assertRaisesRegex(QualificationError, "qwen-flagship exited 7"):
                assemble(
                    root / "receipt", candidate=candidate, anchor=wheel,
                    spec_path=spec, work_dir=work, source_revision="a" * 40,
                    release="1.1.0-rc.0",
                )

    def test_fails_closed_on_regenerated_output_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, wheel, spec, work = fixture(root)
            (work / "generated/model-card.json").write_text(
                "drift\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(QualificationError, "model-card differs"):
                assemble(
                    root / "receipt", candidate=candidate, anchor=wheel,
                    spec_path=spec, work_dir=work, source_revision="a" * 40,
                    release="1.1.0-rc.0",
                )


if __name__ == "__main__":
    unittest.main()

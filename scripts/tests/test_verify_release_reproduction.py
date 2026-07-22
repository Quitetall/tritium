import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "verify-release-reproduction.py"
)
validate_second = MODULE["validate_second_machine"]
validate_review = MODULE["validate_independent_review"]
canonical = MODULE["canonical"]
ReproductionError = MODULE["ReproductionError"]


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def candidate(root: Path):
    wheel = root / "candidate.whl"
    archive = root / "web.tgz"
    wheel.write_bytes(b"wheel bytes")
    archive.write_bytes(b"web bytes")
    artifacts = []
    for artifact_id, kind, path in (
        ("wheel", "python-wheel", wheel),
        ("web", "npm-archive", archive),
    ):
        artifacts.append(
            {
                "id": artifact_id,
                "kind": kind,
                "path": path.name,
                "identity": {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": artifacts}), encoding="utf-8")
    return manifest, wheel, artifacts


def artifact_records(artifacts):
    return [
        {
            "id": item["id"],
            "kind": item["kind"],
            "name": item["path"],
            "bytes": item["identity"]["bytes"],
            "sha256": item["identity"]["sha256"],
        }
        for item in artifacts
    ]


def anchor_record(wheel: Path):
    return {
        "id": "wheel",
        "kind": "python-wheel",
        "name": wheel.name,
        "bytes": wheel.stat().st_size,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }


def second_receipt(manifest: Path, wheel: Path, artifacts):
    logs = manifest.parent / "logs"
    outputs_dir = manifest.parent / "outputs"
    logs.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    commands = []
    for command_id in sorted(MODULE["REQUIRED_COMMANDS"]):
        stdout = logs / f"{command_id}.stdout"
        stderr = logs / f"{command_id}.stderr"
        stdout.write_bytes((command_id + " stdout").encode())
        stderr.write_bytes((command_id + " stderr").encode())
        commands.append(
            {
                "id": command_id,
                "argv": ["tritium-reproduce", command_id],
                "exit_code": 0,
                "duration_seconds": 1.0,
                "stdout_sha256": digest(stdout.read_bytes()),
                "stderr_sha256": digest(stderr.read_bytes()),
                "stdout_path": stdout.relative_to(manifest.parent).as_posix(),
                "stderr_path": stderr.relative_to(manifest.parent).as_posix(),
            }
        )
    outputs = []
    for name in sorted(MODULE["REQUIRED_OUTPUTS"]):
        path = outputs_dir / f"{name}.json"
        path.write_bytes(name.encode())
        outputs.append({
            "name": name, "expected_sha256": digest(path.read_bytes()),
            "observed_sha256": digest(path.read_bytes()),
            "bytes": path.stat().st_size,
            "path": path.relative_to(manifest.parent).as_posix(),
        })
    value = {
        "schema": MODULE["SECOND_SCHEMA"],
        "result": "pass",
        "release": "1.1.0-rc.0",
        "source_revision": "a" * 40,
        "run_id": "second-machine-1",
        "operator": {
            "id": "operator-2",
            "organization": "independent-lab",
            "independent": True,
        },
        "machine": {
            "machine_id": "sha256:" + "2" * 64,
            "system": "Linux",
            "version": "6.16",
            "architecture": "x86_64",
            "cpu": "Zen 5",
            "gpus": ["NVIDIA RTX 5090"],
        },
        "primary_machine_id": "sha256:" + "1" * 64,
        "candidate_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "anchor_artifact": anchor_record(wheel),
        "artifacts": artifact_records(artifacts),
        "commands": commands,
        "checks": {
            "source_verified": True,
            "artifacts_verified": True,
            "repository_absent": True,
            "compiler_absent": True,
            "tutorial": True,
            "bitnet_native": True,
            "qwen_flagship": True,
            "bounded_validation": True,
            "native_backend": True,
            "onnx": True,
            "serving": True,
            "browser": "not-applicable",
            "generated_tables_exact": True,
        },
        "outputs": outputs,
        "divergences": [],
        "wall_time_seconds": 42.0,
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


def review_receipt(manifest: Path, wheel: Path, reviewed):
    attestation = {
        "schema": MODULE["REVIEW_ATTESTATION_SCHEMA"],
        "release": "1.1.0-rc.0",
        "source_revision": "a" * 40,
        "run_id": "independent-review-1",
        "reviewer": {
            "id": "reviewer-3",
            "organization": "security-lab",
            "independent": True,
            "tool": "review-suite",
            "model": "human+static",
        },
        "candidate_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "review_scope_sha256": "9" * 64,
        "reviewed_receipt_ids": reviewed,
        "scopes": ["code", "security", "evidence"],
        "findings": {
            "total": 3,
            "verified": 2,
            "fixed": 2,
            "false_positive": 1,
            "open": 0,
        },
        "verdict": "pass",
    }
    attestation_path = manifest.parent / "review-attestation.json"
    write(attestation_path, attestation)
    value = {
        **attestation, "schema": MODULE["REVIEW_SCHEMA"], "result": "pass",
        "anchor_artifact": anchor_record(wheel),
        "review_attestation": {
            "path": attestation_path.name, "bytes": attestation_path.stat().st_size,
            "sha256": hashlib.sha256(attestation_path.read_bytes()).hexdigest(),
        },
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


def write(path: Path, value):
    path.write_bytes(canonical(value) + b"\n")


class ReleaseReproductionTests(unittest.TestCase):
    def test_accepts_complete_second_machine_and_review_receipts(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest, wheel, artifacts = candidate(root)
            second = second_receipt(manifest, wheel, artifacts)
            second_path = root / "second.json"
            write(second_path, second)
            self.assertEqual(
                validate_second(second_path, "a" * 40, "1.1.0-rc.0", manifest, wheel),
                second,
            )
            review = review_receipt(manifest, wheel, [second["receipt_id"]])
            review_path = root / "review.json"
            write(review_path, review)
            self.assertEqual(
                validate_review(review_path, "a" * 40, "1.1.0-rc.0", manifest, wheel),
                review,
            )

    def test_rejects_copied_machine_partial_commands_and_divergence(self):
        mutations = (
            (
                lambda value: value.__setitem__(
                    "primary_machine_id", value["machine"]["machine_id"]
                ),
                "differ",
            ),
            (lambda value: value["commands"].pop(), "command inventory"),
            (
                lambda value: value["checks"].__setitem__("qwen_flagship", False),
                "qwen_flagship",
            ),
            (lambda value: value["divergences"].append("quality drift"), "divergences"),
        )
        for mutate, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                manifest, wheel, artifacts = candidate(root)
                value = second_receipt(manifest, wheel, artifacts)
                mutate(value)
                value["receipt_id"] = (
                    "sha256:"
                    + hashlib.sha256(
                        canonical(
                            {
                                key: item
                                for key, item in value.items()
                                if key != "receipt_id"
                            }
                        )
                    ).hexdigest()
                )
                path = root / "second.json"
                write(path, value)
                with self.assertRaisesRegex(ReproductionError, message):
                    validate_second(path, "a" * 40, "1.1.0-rc.0", manifest, wheel)

    def test_rejects_incomplete_review_and_open_findings(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest, wheel, _ = candidate(root)
            value = review_receipt(manifest, wheel, ["sha256:" + "7" * 64])
            value["findings"]["open"] = 1
            attestation_path = root / value["review_attestation"]["path"]
            attestation = json.loads(attestation_path.read_bytes())
            attestation["findings"]["open"] = 1
            write(attestation_path, attestation)
            value["review_attestation"]["bytes"] = attestation_path.stat().st_size
            value["review_attestation"]["sha256"] = hashlib.sha256(
                attestation_path.read_bytes()
            ).hexdigest()
            value["receipt_id"] = (
                "sha256:"
                + hashlib.sha256(
                    canonical(
                        {
                            key: item
                            for key, item in value.items()
                            if key != "receipt_id"
                        }
                    )
                ).hexdigest()
            )
            path = root / "review.json"
            write(path, value)
            with self.assertRaisesRegex(ReproductionError, "unresolved"):
                validate_review(path, "a" * 40, "1.1.0-rc.0", manifest, wheel)


if __name__ == "__main__":
    unittest.main()

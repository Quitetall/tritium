import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "run-baseline-ablation.py")
BASELINES = MODULE["BASELINES"]
CampaignError = MODULE["CampaignError"]
canonical = MODULE["canonical"]
execute = MODULE["execute"]


def campaign(root: Path) -> Path:
    value = {
        "schema": "tritium.baseline-ablation-campaign.v1",
        "release": "1.1.0-rc.0",
        "source_revision": "a" * 40,
        "run_id": "ablation-1",
        "model_artifact_id": "s34",
        "evaluation_id": "sha256:" + "3" * 64,
        "target_bytes": 100,
        "parameter_count": 400,
        "device": "cuda:0:GPU-1",
        "baselines": [
            {
                "method": method,
                "family": family,
                "recipe": {
                    "implementation": f"reference/{method}",
                    "revision": "sha256:" + f"{ordinal + 1:x}" * 64,
                    "arguments": {"target_bytes": 100},
                },
                "build_command": ["baseline-tool", "build", method],
                "evaluation_command": ["baseline-tool", "evaluate", method],
                "artifact": f"{ordinal:02d}-{method}.bin",
            }
            for ordinal, (method, family) in enumerate(BASELINES)
        ],
    }
    path = root / "campaign.json"
    path.write_bytes(canonical(value) + b"\n")
    return path


class FakeCommands:
    def __init__(self, *, device: str = "cuda:0:GPU-1", artifact_bytes: int = 100):
        self.device = device
        self.artifact_bytes = artifact_bytes
        self.calls = []

    def __call__(self, command, *, cwd, env):
        self.calls.append(
            (
                tuple(command),
                env["TRITIUM_ABLATION_PHASE"],
                env["TRITIUM_ABLATION_BASELINE_INDEX"],
                env.get("TRITIUM_ABLATION_SAMPLE_INDEX"),
            )
        )
        artifact = Path(env["TRITIUM_ABLATION_ARTIFACT"])
        if env["TRITIUM_ABLATION_PHASE"] == "build":
            artifact.write_bytes(bytes([int(env["TRITIUM_ABLATION_BASELINE_INDEX"]) + 1])
                                 * self.artifact_bytes)
            return ""
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        return json.dumps(
            {
                "schema": "tritium.baseline-ablation-sample.v1",
                "evaluation_id": env["TRITIUM_ABLATION_EVALUATION_ID"],
                "artifact_sha256": digest,
                "quality_score": 1.0,
                "resident_bytes": 4096,
                "physical_device": self.device,
            }
        )


class DriftingCommands(FakeCommands):
    def __call__(self, command, *, cwd, env):
        raw = super().__call__(command, cwd=cwd, env=env)
        if (
            env["TRITIUM_ABLATION_PHASE"] == "evaluate"
            and env["TRITIUM_ABLATION_BASELINE_INDEX"] == "6"
            and env["TRITIUM_ABLATION_SAMPLE_INDEX"] == "29"
        ):
            with Path(env["TRITIUM_ABLATION_ARTIFACT"]).open("ab") as stream:
                stream.write(b"x")
        return raw


class EarlyDriftingCommands(FakeCommands):
    def __call__(self, command, *, cwd, env):
        raw = super().__call__(command, cwd=cwd, env=env)
        if (
            env["TRITIUM_ABLATION_PHASE"] == "evaluate"
            and env["TRITIUM_ABLATION_BASELINE_INDEX"] == "0"
            and env["TRITIUM_ABLATION_SAMPLE_INDEX"] == "0"
        ):
            with Path(env["TRITIUM_ABLATION_ARTIFACT"]).open("ab") as stream:
                stream.write(b"x")
        return raw


class RunBaselineAblationTests(unittest.TestCase):
    def test_executes_frozen_inventory_and_emits_verifier_trace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            commands = FakeCommands()
            trace = execute(
                campaign(root),
                output=root / "trace.json",
                work_dir=root / "work",
                command_runner=commands,
                environment={
                    "python": "3.13.5",
                    "torch": "2.7.1",
                    "tritium": "1.1.0rc0",
                },
            )
            self.assertEqual(trace["schema"], "tritium.baseline-ablation-execution.v1")
            self.assertEqual(trace["result"], "pass")
            self.assertEqual(len(trace["baselines"]), 7)
            self.assertEqual(len(commands.calls), 7 * 31)
            self.assertTrue(all(len(row["elapsed_samples_ms"]) == 30
                                for row in trace["baselines"]))
            self.assertTrue(all(row["artifact_bytes"] == 100
                                for row in trace["baselines"]))
            self.assertTrue(all(
                row["artifact_sha256"]
                == hashlib.sha256(
                    bytes([ordinal + 1]) * 100
                ).hexdigest()
                for ordinal, row in enumerate(trace["baselines"])
            ))
            self.assertTrue(all(row["recipe"]["implementation"].startswith("reference/")
                                for row in trace["baselines"]))
            self.assertTrue(all(row["build_command"][1] == "build"
                                for row in trace["baselines"]))
            self.assertEqual(
                json.loads((root / "trace.json").read_bytes()),
                trace,
            )
            recipes = [
                {
                    "method": row["method"],
                    "family": row["family"],
                    "recipe_id": row["recipe_id"],
                }
                for row in trace["baselines"]
            ]
            expected_set_id = "sha256:" + hashlib.sha256(
                canonical(
                    {
                        "model_artifact_id": "s34",
                        "evaluation_id": "sha256:" + "3" * 64,
                        "target_bytes": 100,
                        "target_bpw": 2.0,
                        "recipes": recipes,
                    }
                )
            ).hexdigest()
            self.assertEqual(trace["baseline_set_id"], expected_set_id)

    def test_rejects_incomplete_inventory_before_execution(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = campaign(root)
            value = json.loads(path.read_bytes())
            value["baselines"].pop()
            path.write_bytes(canonical(value) + b"\n")
            commands = FakeCommands()
            with self.assertRaisesRegex(CampaignError, "inventory"):
                execute(
                    path,
                    output=root / "trace.json",
                    work_dir=root / "work",
                    command_runner=commands,
                    environment={
                        "python": "3.13.5", "torch": "2.7.1",
                        "tritium": "1.1.0rc0",
                    },
                )
            self.assertEqual(commands.calls, [])

    def test_rejects_late_unsafe_artifact_path_before_execution(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = campaign(root)
            value = json.loads(path.read_bytes())
            value["baselines"][1]["artifact"] = "../escape.bin"
            path.write_bytes(canonical(value) + b"\n")
            commands = FakeCommands()
            with self.assertRaisesRegex(CampaignError, "unsafe"):
                execute(
                    path,
                    output=root / "trace.json",
                    work_dir=root / "work",
                    command_runner=commands,
                    environment={
                        "python": "3.13.5", "torch": "2.7.1",
                        "tritium": "1.1.0rc0",
                    },
                )
            self.assertEqual(commands.calls, [])

    def test_rejects_output_equal_to_work_directory_before_execution(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            commands = FakeCommands()
            with self.assertRaisesRegex(CampaignError, "must differ"):
                execute(
                    campaign(root),
                    output=root / "work",
                    work_dir=root / "work",
                    command_runner=commands,
                    environment={
                        "python": "3.13.5", "torch": "2.7.1",
                        "tritium": "1.1.0rc0",
                    },
                )
            self.assertEqual(commands.calls, [])

    def test_rejects_artifact_that_is_not_matched_to_physical_byte_target(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaisesRegex(CampaignError, "physical-byte"):
                execute(
                    campaign(root),
                    output=root / "trace.json",
                    work_dir=root / "work",
                    command_runner=FakeCommands(artifact_bytes=90),
                    environment={
                        "python": "3.13.5", "torch": "2.7.1",
                        "tritium": "1.1.0rc0",
                    },
                )
            self.assertFalse((root / "trace.json").exists())

    def test_rejects_sample_from_different_device(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaisesRegex(CampaignError, "physical device"):
                execute(
                    campaign(root),
                    output=root / "trace.json",
                    work_dir=root / "work",
                    command_runner=FakeCommands(device="cuda:1:GPU-2"),
                    environment={
                        "python": "3.13.5", "torch": "2.7.1",
                        "tritium": "1.1.0rc0",
                    },
                )
            self.assertFalse((root / "trace.json").exists())

    def test_rejects_artifact_drift_before_publication(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaisesRegex(CampaignError, "artifact drifted"):
                execute(
                    campaign(root),
                    output=root / "trace.json",
                    work_dir=root / "work",
                    command_runner=DriftingCommands(),
                    environment={
                        "python": "3.13.5", "torch": "2.7.1",
                        "tritium": "1.1.0rc0",
                    },
                )
            self.assertFalse((root / "trace.json").exists())

    def test_rejects_artifact_drift_after_each_sample(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaisesRegex(CampaignError, "artifact drifted"):
                execute(
                    campaign(root),
                    output=root / "trace.json",
                    work_dir=root / "work",
                    command_runner=EarlyDriftingCommands(),
                    environment={
                        "python": "3.13.5", "torch": "2.7.1",
                        "tritium": "1.1.0rc0",
                    },
                )
            self.assertFalse((root / "trace.json").exists())


if __name__ == "__main__":
    unittest.main()

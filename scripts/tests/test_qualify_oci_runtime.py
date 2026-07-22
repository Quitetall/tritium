from __future__ import annotations

from pathlib import Path
import runpy
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-oci-runtime.py")
QualificationError = MODULE["QualificationError"]
atomic_create = MODULE["atomic_create"]
canonical = MODULE["canonical"]
validate_ready = MODULE["validate_ready"]
validate_receipt = MODULE["validate_receipt"]
request_json = MODULE["request_json"]
request_error = MODULE["request_error"]


def readiness(flavor: str = "cpu"):
    return {
        "status": "ready",
        "release_gate": "production_artifact_admitted",
        "startup_receipt": {
            "schema_version": 1,
            "artifact_kind": "qwen3.6-language-mtp-salt-v2-hf-bundle",
            "server_source_revision": "a" * 40,
            "server_build_id": "tritium-serve:1.1.0-rc.0:" + "a" * 40,
            "model_source_revision": "b" * 40,
            "manifest_package_id": "c" * 64,
            "salt_package_id": "d" * 64,
            "preserved_package_id": "e" * 64,
            "config_package_id": "f" * 64,
            "profile": "compact-v1",
            "codec": "b3",
            "backend_policy": flavor,
            "effective_backend": flavor,
            "physical_device_id": (
                "cpu" if flavor == "cpu" else "cuda:0:GPU-physical"
            ),
            "loaded_bundle_bytes": 100,
            "resident_bytes": 80,
            "self_test_digest": "1" * 64,
        },
    }


def runtime_receipt(artifact: Path, flavor: str = "cpu") -> dict:
    import hashlib

    startup = readiness(flavor)["startup_receipt"]
    startup_sha256 = hashlib.sha256(canonical(startup)).hexdigest()
    value = {
        "schema": MODULE["SCHEMA"], "release": "1.1.0-rc.0",
        "source_revision": "a" * 40, "run_id": f"{flavor}-1", "flavor": flavor,
        "image": "example@sha256:" + "b" * 64,
        "image_id": "sha256:" + "c" * 64,
        "image_manifest_digest": "sha256:" + "b" * 64,
        "artifact": {"kind": "oci-image", "name": artifact.name,
                     "bytes": artifact.stat().st_size,
                     "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        "manifest": {"schema": "tritium.file-identity.v1", "bytes": 42,
                     "sha256": "2" * 64, "blake3": "c" * 64},
        "profile": "compact-v1", "startup_receipt": startup,
        "faults": {
            "unauthenticated_status": 401, "wrong_token_status": 401,
            "malformed_json_status": 400, "rate_limited_status": 429,
            "retry_after_seconds": 60, "rate_rejections_before": 0,
            "rate_rejections_after": 1, "replacement_principal_status": 400,
            "queue_flood_clients": 3, "slow_reader_tokens": 32,
            "queue_rejections_before": 0, "queue_rejections_after": 1,
            "disconnects_before": 0, "disconnects_after": 1,
            "accepted_streams": 2, "rejected_streams": 1,
            "settled_queue_depth": 0, "worker_alive": 1,
            "queue_capacity": 1, "saturated_queue_depth": 1, "slow_hold_ms": 1000,
            "tokens_out_before_hold": 0, "tokens_out_after_hold": 1,
            "recovery_status": 200, "recovery_ms": 1,
            "recovery_timeout_ms": 60000,
        },
        "shutdown_scenarios": [
            {"phase": "queue", "observed_worker_phase": "decode", "queue_depth": 1,
             "signal": "SIGTERM", "container_id": "1" * 64,
             "image_id": "sha256:" + "c" * 64,
             "startup_receipt_sha256": startup_sha256, "prompt_sha256": "4" * 64,
             "prompt_bytes": 4, "prompt_repetitions": 1, "max_tokens": 32,
             "observation_to_signal_ms": 5, "observation_budget_ms": 2000,
             "exit_code": 0, "shutdown_ms": 10, "budget_ms": 35000},
            {"phase": "prefill", "observed_worker_phase": "prefill", "queue_depth": 0,
             "signal": "SIGTERM", "container_id": "2" * 64,
             "image_id": "sha256:" + "c" * 64,
             "startup_receipt_sha256": startup_sha256, "prompt_sha256": "5" * 64,
             "prompt_bytes": 40, "prompt_repetitions": 256, "max_tokens": 1,
             "observation_to_signal_ms": 5, "observation_budget_ms": 2000,
             "exit_code": 0, "shutdown_ms": 20, "budget_ms": 35000},
            {"phase": "decode", "observed_worker_phase": "decode", "queue_depth": 0,
             "signal": "SIGTERM", "container_id": "3" * 64,
             "image_id": "sha256:" + "c" * 64,
             "startup_receipt_sha256": startup_sha256, "prompt_sha256": "4" * 64,
             "prompt_bytes": 4, "prompt_repetitions": 1, "max_tokens": 32,
             "observation_to_signal_ms": 5, "observation_budget_ms": 2000,
             "exit_code": 0, "shutdown_ms": 10, "budget_ms": 35000},
        ],
        "checks": list(MODULE["CHECKS"]),
        "started_at_utc": "2026-07-21T00:00:00+00:00",
        "timing": {"startup_ms": 10.0, "shutdown_ms": 20},
        "machine": {"id": "sha256:" + "5" * 64, "system": "Linux",
                    "architecture": "x86_64", "docker_server": "28.0.0",
                    "gpu": None if flavor == "cpu" else {
                        "uuid": "GPU-physical", "name": "RTX 4090",
                        "driver_version": "610.43.03",
                    }},
        "result": "pass",
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


class QualifyOciRuntimeTests(unittest.TestCase):
    def test_sigterm_phase_reobserves_immediately_and_closes_response(self):
        events = []

        class Response:
            closed = False

            def close(self):
                self.closed = True

        response = Response()

        def fake_run(command, **_kwargs):
            events.append(("run", tuple(command)))
            return "sha256:" + "c" * 64

        def fake_wait_metric(_base, _token, name, expected, _timeout):
            events.append(("phase", name, expected))
            return expected

        def fake_metric(*_args):
            events.append(("queue-depth",))
            return 0

        def fake_terminate(_container, _timeout, _observed_at):
            events.append(("terminate",))
            return 0, 10, 5

        function = MODULE["qualify_sigterm_phase"]
        with mock.patch.dict(function.__globals__, {
            "run": fake_run,
            "slow_stream_attempt": lambda *_args: ("accepted", response),
            "wait_metric": fake_wait_metric,
            "metric_value": fake_metric,
            "terminate_container": fake_terminate,
        }):
            result = function(
                phase="decode", base_url="http://127.0.0.1", token="a",
                metric_token="b", model_id="m", prompt="prompt", max_tokens=32,
                timeout=10, shutdown_timeout=5, container="1" * 64,
                startup_receipt_sha256="2" * 64,
                expected_image_id="sha256:" + "c" * 64,
                prompt_repetitions=1, observation_budget_ms=2000,
            )
        self.assertEqual(events[-2][0], "phase")
        self.assertEqual(events[-1], ("terminate",))
        self.assertTrue(response.closed)
        self.assertEqual(result["observation_to_signal_ms"], 5)

    def test_sigterm_phase_closes_response_when_signal_fails(self):
        class Response:
            closed = False

            def close(self):
                self.closed = True

        response = Response()
        function = MODULE["qualify_sigterm_phase"]
        with mock.patch.dict(function.__globals__, {
            "run": lambda *_args, **_kwargs: "sha256:" + "c" * 64,
            "slow_stream_attempt": lambda *_args: ("accepted", response),
            "wait_metric": lambda *_args: 1,
            "metric_value": lambda *_args: 0,
            "terminate_container": mock.Mock(side_effect=QualificationError("kill failed")),
        }):
            with self.assertRaisesRegex(QualificationError, "kill failed"):
                function(
                    phase="decode", base_url="http://127.0.0.1", token="a",
                    metric_token="b", model_id="m", prompt="prompt", max_tokens=32,
                    timeout=10, shutdown_timeout=5, container="1" * 64,
                    startup_receipt_sha256="2" * 64,
                    expected_image_id="sha256:" + "c" * 64,
                    prompt_repetitions=1, observation_budget_ms=2000,
                )
        self.assertTrue(response.closed)

    def test_error_client_requires_stable_json_type_and_headers(self):
        class Response:
            status = 429
            headers = {"Retry-After": "60"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return (
                    b'{"error":{"message":"principal request rate exceeded; retry later",'
                    b'"type":"rate_limit_exceeded"}}'
                )

        with mock.patch.object(MODULE["urllib"].request, "urlopen", return_value=Response()):
            _, headers = request_error(
                "http://127.0.0.1/v1/chat/completions", 429,
                "rate_limit_exceeded",
                "principal request rate exceeded; retry later",
                token="token", body=b"{",
            )
        self.assertEqual(headers["retry-after"], "60")

    def test_json_client_rejects_oversized_response(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return b"x" * (MODULE["MAX_JSON_RESPONSE_BYTES"] + 1)

        with mock.patch.object(MODULE["urllib"].request, "urlopen", return_value=Response()):
            with self.assertRaisesRegex(QualificationError, "byte limit"):
                request_json("http://127.0.0.1/readyz", "token")

    def test_accepts_exact_production_readiness(self):
        receipt = validate_ready(
            readiness(), "a" * 40, "cpu", "compact-v1", "c" * 64, "1.1.0-rc.0"
        )
        self.assertEqual(receipt["resident_bytes"], 80)

    def test_rejects_legacy_and_cross_artifact_readiness(self):
        value = readiness()
        value["release_gate"] = "legacy_compatibility"
        with self.assertRaisesRegex(QualificationError, "production artifact"):
            validate_ready(value, "a" * 40, "cpu", "compact-v1", "c" * 64)
        value = readiness()
        with self.assertRaisesRegex(QualificationError, "artifact identity"):
            validate_ready(value, "a" * 40, "cpu", "compact-v1", "0" * 64)

    def test_atomic_receipt_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "receipt.json"
            atomic_create(path, b"first\n")
            with self.assertRaisesRegex(QualificationError, "overwrite"):
                atomic_create(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")

    def test_receipt_validator_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as raw:
            artifact = Path(raw) / "image.oci.tar"
            artifact.write_bytes(b"qualified OCI bytes")
            receipt = runtime_receipt(artifact)
            validate_receipt(
                receipt, revision="a" * 40, release="1.1.0-rc.0",
                artifact_path=artifact,
            )
            receipt["run_id"] = "tampered"
            with self.assertRaisesRegex(QualificationError, "content digest"):
                validate_receipt(receipt)

    def test_receipt_validator_rejects_noncausal_fault_evidence(self):
        with tempfile.TemporaryDirectory() as raw:
            artifact = Path(raw) / "image.oci.tar"
            artifact.write_bytes(b"qualified OCI bytes")
            receipt = runtime_receipt(artifact)
            receipt["faults"]["rate_rejections_after"] = 2
            import hashlib
            del receipt["receipt_id"]
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
            with self.assertRaisesRegex(QualificationError, "fault evidence"):
                validate_receipt(receipt)

            receipt = runtime_receipt(artifact)
            receipt["faults"]["disconnects_after"] = 0
            del receipt["receipt_id"]
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
            with self.assertRaisesRegex(QualificationError, "queue/disconnect evidence"):
                validate_receipt(receipt)

            for field, value in (
                ("queue_capacity", 2),
                ("tokens_out_after_hold", 0),
                ("recovery_ms", 60001),
            ):
                receipt = runtime_receipt(artifact)
                receipt["faults"][field] = value
                del receipt["receipt_id"]
                receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                    canonical(receipt)
                ).hexdigest()
                with self.assertRaisesRegex(
                    QualificationError, "queue/disconnect evidence"
                ):
                    validate_receipt(receipt)

            receipt = runtime_receipt(artifact)
            receipt["shutdown_scenarios"][1]["observed_worker_phase"] = "decode"
            del receipt["receipt_id"]
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical(receipt)
            ).hexdigest()
            with self.assertRaisesRegex(QualificationError, "SIGTERM evidence"):
                validate_receipt(receipt)

            receipt = runtime_receipt(artifact)
            receipt["shutdown_scenarios"][2]["container_id"] = (
                receipt["shutdown_scenarios"][1]["container_id"]
            )
            del receipt["receipt_id"]
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical(receipt)
            ).hexdigest()
            with self.assertRaisesRegex(QualificationError, "not recreated"):
                validate_receipt(receipt)

            receipt = runtime_receipt(artifact)
            receipt["shutdown_scenarios"][1]["prompt_bytes"] = 4
            del receipt["receipt_id"]
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                canonical(receipt)
            ).hexdigest()
            with self.assertRaisesRegex(QualificationError, "workload matrix"):
                validate_receipt(receipt)

            def weak_decode_budget(value):
                value["shutdown_scenarios"][0]["max_tokens"] = 31
                value["shutdown_scenarios"][2]["max_tokens"] = 31

            mutations = (
                lambda value: value["shutdown_scenarios"][1].__setitem__(
                    "prompt_repetitions", 8193
                ),
                weak_decode_budget,
                lambda value: value["shutdown_scenarios"][0].__setitem__(
                    "observation_budget_ms", 99
                ),
                lambda value: value["shutdown_scenarios"][2].__setitem__(
                    "budget_ms", 34000
                ),
            )
            for mutate in mutations:
                receipt = runtime_receipt(artifact)
                mutate(receipt)
                del receipt["receipt_id"]
                receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                    canonical(receipt)
                ).hexdigest()
                with self.assertRaisesRegex(QualificationError, "workload matrix"):
                    validate_receipt(receipt)

    def test_receipt_validator_rejects_cross_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as raw:
            artifact = Path(raw) / "image.oci.tar"
            artifact.write_bytes(b"wrong bytes")
            receipt = runtime_receipt(artifact)
            receipt["artifact"]["bytes"] = 4
            receipt["artifact"]["sha256"] = "0" * 64
            import hashlib
            del receipt["receipt_id"]
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
            with self.assertRaisesRegex(QualificationError, "candidate OCI bytes"):
                validate_receipt(receipt, artifact_path=artifact)

    def test_cuda_receipt_binds_startup_to_physical_gpu_uuid(self):
        with tempfile.TemporaryDirectory() as raw:
            artifact = Path(raw) / "image.oci.tar"
            artifact.write_bytes(b"qualified OCI bytes")
            receipt = runtime_receipt(artifact, "cuda")
            validate_receipt(receipt, artifact_path=artifact)
            receipt["startup_receipt"]["physical_device_id"] = "cuda:0:GPU-other"
            import hashlib
            startup_sha256 = hashlib.sha256(
                canonical(receipt["startup_receipt"])
            ).hexdigest()
            for scenario in receipt["shutdown_scenarios"]:
                scenario["startup_receipt_sha256"] = startup_sha256
            del receipt["receipt_id"]
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
            with self.assertRaisesRegex(QualificationError, "physical NVIDIA UUID"):
                validate_receipt(receipt, artifact_path=artifact)


if __name__ == "__main__":
    unittest.main()

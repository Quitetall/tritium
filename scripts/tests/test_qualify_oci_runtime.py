from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import runpy
import multiprocessing
import signal
import subprocess
import sys
import threading
import time
import tempfile
import traceback
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


class CommandDiagnosticsTests(unittest.TestCase):
    def test_real_failure_does_not_publish_environment_or_arguments(self):
        # Deliberately synthetic markers, not real credentials or private data.
        token = "fixture-only-not-a-credential"
        argument = "fixture-only-private-command-argument"
        environment = os.environ.copy()
        environment["TRITIUM_AUTH_TOKEN"] = token
        command = [
            sys.executable, "-c",
            "import os,sys; "
            "sys.stderr.write(os.environ['TRITIUM_AUTH_TOKEN'] + ' ' + sys.argv[1]); "
            "sys.exit(23)",
            argument,
        ]
        with self.assertRaises(QualificationError) as caught:
            MODULE["run"](command, env=environment, timeout=5)
        diagnostic = "".join(traceback.format_exception(caught.exception))
        self.assertIn("23", str(caught.exception))
        self.assertNotIn(token, diagnostic)
        self.assertNotIn(argument, diagnostic)

    def test_launch_and_timeout_errors_do_not_publish_exception_context(self):
        marker = "fixture-only-sensitive-exception-context"
        command = ["/private/fixture-only-sensitive-executable", marker]
        failures = (
            OSError(marker),
            subprocess.TimeoutExpired(command, 1, stderr=marker.encode()),
            subprocess.SubprocessError(marker),
        )
        for failure in failures:
            with self.subTest(kind=type(failure).__name__):
                with mock.patch.object(MODULE["subprocess"], "run", side_effect=failure):
                    with self.assertRaises(QualificationError) as caught:
                        MODULE["run"](command, timeout=1)
                diagnostic = "".join(traceback.format_exception(caught.exception))
                self.assertNotIn(marker, diagnostic)
                self.assertNotIn(command[0], diagnostic)

    def test_success_retains_the_command_result(self):
        self.assertEqual(
            MODULE["run"]([sys.executable, "-c", "print('verified-output')"], timeout=5),
            "verified-output",
        )

    def test_allowlisted_label_does_not_include_path_or_subprocess_output(self):
        command = ["/private/fixture-sensitive-directory/docker", "fixture-sensitive-argument"]
        result = subprocess.CompletedProcess(
            command, 9, "fixture-sensitive-stdout", "fixture-sensitive-stderr"
        )
        with mock.patch.object(MODULE["subprocess"], "run", return_value=result):
            with self.assertRaises(QualificationError) as caught:
                MODULE["run"](command)
        self.assertEqual(
            str(caught.exception),
            "command failed (9): docker; subprocess diagnostics withheld",
        )

    def test_real_timeout_withholds_command_and_captured_output(self):
        marker = "fixture-only-timeout-marker"
        command = [
            sys.executable, "-c",
            "import sys,time; print(sys.argv[1], flush=True); "
            "sys.stderr.write(sys.argv[1]); sys.stderr.flush(); time.sleep(10)",
            marker,
        ]
        started = time.monotonic()
        with self.assertRaises(QualificationError) as caught:
            MODULE["run"](command, timeout=0.1)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(str(caught.exception), "command failed: external-tool: timeout")
        self.assertNotIn(marker, "".join(traceback.format_exception(caught.exception)))

    def test_http_error_mismatch_does_not_publish_response_or_url(self):
        marker = "fixture-only-sensitive-server-response"
        function = MODULE["request_error"]
        with mock.patch.dict(function.__globals__, {
            "request_response": lambda *_args, **_kwargs: (500, {"error": marker}, {}),
        }):
            with self.assertRaises(QualificationError) as caught:
                function(
                    "http://127.0.0.1/private?fixture=" + marker,
                    401, "invalid_request_error", "missing or invalid bearer token",
                    token="fixture-only-not-a-credential",
                )
        self.assertNotIn(marker, "".join(traceback.format_exception(caught.exception)))

    def test_http_transport_failures_do_not_publish_exception_context(self):
        marker = "fixture-only-sensitive-transport-context"
        url = "http://127.0.0.1/private?fixture=" + marker
        token = "fixture-only-not-a-credential"
        calls = (
            ("request_json", lambda: MODULE["request_json"](url, token, timeout=1)),
            ("request_response", lambda: MODULE["request_response"](url, token=token, timeout=1)),
            ("metric_value", lambda: MODULE["metric_value"](url, token, "tritium_queue_depth", 1)),
            ("slow_stream_attempt", lambda: MODULE["slow_stream_attempt"](url, token, {}, 1, None)),
        )
        for name, call in calls:
            with self.subTest(operation=name):
                with mock.patch.object(MODULE["urllib"].request, "urlopen", side_effect=OSError(marker)):
                    with self.assertRaises(QualificationError) as caught:
                        call()
                diagnostic = "".join(traceback.format_exception(caught.exception))
                self.assertNotIn(marker, diagnostic)
                self.assertNotIn(token, diagnostic)

    def test_failed_http_error_body_read_is_closed_and_sanitized(self):
        marker = "fixture-only-sensitive-error-body"

        class FailingBody(io.BytesIO):
            def read(self, *_args):
                raise OSError(marker)

        for name in ("request_response", "slow_stream_attempt"):
            with self.subTest(operation=name):
                body = FailingBody()
                error = MODULE["urllib"].error.HTTPError(
                    "http://127.0.0.1/private?fixture=" + marker, 429, marker,
                    {"Retry-After": "1"}, body,
                )
                with mock.patch.object(MODULE["urllib"].request, "urlopen", side_effect=error):
                    with self.assertRaises(QualificationError) as caught:
                        if name == "request_response":
                            MODULE[name](error.url, token="fixture-only-not-a-credential", timeout=1)
                        else:
                            MODULE[name](error.url, "fixture-only-not-a-credential", {}, 1, None)
                self.assertTrue(body.closed)
                self.assertNotIn(marker, "".join(traceback.format_exception(caught.exception)))

    def test_sse_rejection_mismatch_does_not_chain_server_reason(self):
        marker = "fixture-only-sensitive-http-reason"
        error = MODULE["urllib"].error.HTTPError(
            "http://127.0.0.1/private?fixture=" + marker, 500, marker,
            {}, io.BytesIO(canonical({"error": marker})),
        )
        with mock.patch.object(MODULE["urllib"].request, "urlopen", side_effect=error):
            with self.assertRaises(QualificationError) as caught:
                MODULE["slow_stream_attempt"](
                    error.url, "fixture-only-not-a-credential", {}, 1, None
                )
        self.assertNotIn(marker, "".join(traceback.format_exception(caught.exception)))


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
            "salt_package_id": "trp1_" + "d" * 64,
            "preserved_package_id": "trp1_" + "e" * 64,
            "config_package_id": "trp1_" + "f" * 64,
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
    def queue_arguments(self):
        return {
            "base_url": "http://127.0.0.1", "token": "private-token",
            "metric_token": "private-metric-token", "model_id": "m",
            "prompt": "private-prompt", "clients": 3, "max_tokens": 32,
            "timeout": 1.0, "hold_seconds": 1.0, "recovery_timeout": 1.0,
            "wall_timeout": 3.0,
        }

    def assert_no_new_children(self, before):
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "fork isolation unavailable")
    def test_queue_workload_success_closes_streams_and_reaps_worker(self):
        closed = multiprocessing.get_context("fork").Value("i", 0)
        attempts = []
        reads = {}

        class Response:
            def close(self):
                with closed.get_lock():
                    closed.value += 1

        def attempt(*_args):
            attempts.append(1)
            return ("accepted", Response()) if len(attempts) <= 2 else ("rejected", None)

        def metric(_base, _token, name, _timeout):
            reads[name] = reads.get(name, 0) + 1
            if name == "tritium_tokens_out_total":
                return reads[name] - 1
            if name == "tritium_queue_depth":
                return int(reads[name] == 1)
            if name == "tritium_worker_alive":
                return 1
            return int(reads[name] > 1)

        function = MODULE["exercise_queue_disconnects"]
        before = {child.pid for child in multiprocessing.active_children()}
        with mock.patch.dict(function.__globals__, {
            "slow_stream_attempt": attempt, "metric_value": metric,
            "request_response": lambda *_args, **_kwargs: (200, {"choices": [{}]}, {}),
        }):
            result = function(**self.queue_arguments())
        self.assertEqual(set(result), MODULE["QUEUE_RESULT_FIELDS"])
        self.assertEqual(closed.value, 2)
        self.assertEqual(result["accepted_streams"], 2)
        self.assertEqual(result["rejected_streams"], 1)
        self.assertEqual(result["queue_rejections_after"], 1)
        self.assertEqual(result["disconnects_after"], 1)
        self.assertEqual(result["settled_queue_depth"], 0)
        self.assertEqual(result["worker_alive"], 1)
        self.assertEqual(result["recovery_status"], 200)
        self.assert_no_new_children(before)

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "fork isolation unavailable")
    def test_queue_workload_kills_sigterm_ignoring_worker(self):
        def ignore_and_stall(**_kwargs):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            threading.Event().wait()

        function = MODULE["exercise_queue_disconnects"]
        before = {child.pid for child in multiprocessing.active_children()}
        started = time.monotonic()
        with mock.patch.dict(function.__globals__, {"_exercise_queue_disconnects": ignore_and_stall}):
            with self.assertRaisesRegex(QualificationError, "wall deadline"):
                function(**(self.queue_arguments() | {"wall_timeout": 0.2}))
        self.assertGreaterEqual(time.monotonic() - started, 1.0)
        self.assertLess(time.monotonic() - started, 2.5)
        self.assert_no_new_children(before)

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "fork isolation unavailable")
    def test_queue_workload_redacts_failure_without_traceback(self):
        def fail(**kwargs):
            raise QualificationError(" ".join((kwargs["token"], kwargs["metric_token"], kwargs["prompt"])))

        function = MODULE["exercise_queue_disconnects"]
        before = {child.pid for child in multiprocessing.active_children()}
        with mock.patch.dict(function.__globals__, {"_exercise_queue_disconnects": fail}):
            with self.assertRaises(QualificationError) as raised:
                function(**self.queue_arguments())
        self.assertIn("queue workload failed", str(raised.exception))
        self.assertEqual(str(raised.exception).count("[redacted]"), 3)
        self.assertNotIn("private", str(raised.exception))
        self.assert_no_new_children(before)

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "fork isolation unavailable")
    def test_queue_workload_rejects_missing_malformed_and_late_evidence(self):
        function = MODULE["exercise_queue_disconnects"]
        fields = {name: 0 for name in MODULE["QUEUE_RESULT_FIELDS"]}
        cases = [
            (None, 3, False, "missing or invalid evidence"),
            (b"not json", 0, False, "missing or invalid evidence"),
            (b"x" * 8192, 0, False, "missing or invalid evidence"),
            (canonical({"ok": 1, "result": fields}), 0, False, "invalid evidence envelope"),
            (canonical({"ok": True, "result": fields | {"worker_alive": True}}), 0, False,
             "invalid evidence fields"),
            (canonical({"ok": True, "result": fields | {"worker_alive": -1}}), 0, False,
             "invalid evidence fields"),
            (canonical({"ok": True, "result": fields | {"extra": 0}}), 0, False,
             "invalid evidence fields"),
            (canonical({"ok": False, "detail": []}), 0, False, "invalid failure envelope"),
            (canonical({"ok": True, "result": fields}), 3, False, "exited unsuccessfully"),
            (canonical({"ok": True, "result": fields}), 0, True, "wall deadline"),
        ]
        before = {child.pid for child in multiprocessing.active_children()}
        for data, exit_code, stall, expected in cases:
            with self.subTest(expected=expected, exit_code=exit_code, stall=stall):
                def worker(connection, _arguments):
                    if data is not None:
                        connection.send_bytes(data)
                    connection.close()
                    if stall:
                        threading.Event().wait()
                    os._exit(exit_code)

                with mock.patch.dict(function.__globals__, {"_queue_disconnect_worker": worker}):
                    with self.assertRaisesRegex(QualificationError, expected):
                        function(**(self.queue_arguments() | {"wall_timeout": 0.2}))
                self.assert_no_new_children(before)

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "fork isolation unavailable")
    def test_queue_worker_oversized_result_is_bounded_failure(self):
        function = MODULE["exercise_queue_disconnects"]
        with mock.patch.dict(function.__globals__, {
            "_exercise_queue_disconnects": lambda **_kwargs: {"oversized": "x" * 8192},
        }):
            with self.assertRaisesRegex(QualificationError, "exceeds IPC limit"):
                function(**self.queue_arguments())

    def test_queue_workload_invalid_durations_do_not_start_worker(self):
        function = MODULE["exercise_queue_disconnects"]
        with mock.patch.object(multiprocessing, "get_context") as context:
            for field in ("timeout", "hold_seconds", "recovery_timeout", "wall_timeout"):
                for duration in (float("nan"), float("inf"), float("-inf"), 0, -1, True):
                    with self.subTest(field=field, duration=duration):
                        with self.assertRaisesRegex(QualificationError, "positive and finite"):
                            function(**(self.queue_arguments() | {field: duration}))
            context.assert_not_called()

    def test_qualify_invalid_durations_fail_before_docker_or_paths(self):
        function = MODULE["qualify"]
        durations = {
            "startup_timeout": 1.0, "request_timeout": 1.0, "shutdown_timeout": 1.0,
            "slow_reader_hold": 1.0, "disconnect_recovery_timeout": 1.0,
            "queue_workload_timeout": 1.0,
        }
        with mock.patch.dict(function.__globals__, {"run": mock.Mock()}) as globals_:
            for field in durations:
                for duration in (float("nan"), float("inf"), float("-inf"), 0, -1):
                    arguments = argparse.Namespace(
                        flavor="cpu", profile="compact-v1", release="1.1.0-rc.0", run_id="test",
                        **(durations | {field: duration}),
                    )
                    with self.subTest(field=field, duration=duration):
                        with self.assertRaisesRegex(QualificationError, "positive and finite"):
                            function(arguments)
            globals_["run"].assert_not_called()

    def test_queue_workload_requires_fork(self):
        function = MODULE["exercise_queue_disconnects"]
        with mock.patch.object(multiprocessing, "get_context", side_effect=ValueError("unsupported")):
            with self.assertRaisesRegex(QualificationError, "requires fork"):
                function(**self.queue_arguments())

    def test_queue_workload_start_failure_closes_ipc(self):
        function = MODULE["exercise_queue_disconnects"]
        context = mock.Mock()
        receive, send, process = mock.Mock(), mock.Mock(), mock.Mock()
        context.Pipe.return_value = receive, send
        context.Process.return_value = process
        process.start.side_effect = OSError("start failure")
        process.is_alive.return_value = False
        with mock.patch.object(multiprocessing, "get_context", return_value=context):
            with self.assertRaisesRegex(QualificationError, "could not start"):
                function(**self.queue_arguments())
        receive.close.assert_called_once()
        send.close.assert_called_once()
        process.close.assert_called_once()

    def test_queue_workload_construction_failure_closes_ipc(self):
        function = MODULE["exercise_queue_disconnects"]
        context = mock.Mock()
        receive, send = mock.Mock(), mock.Mock()
        context.Pipe.return_value = receive, send
        context.Process.side_effect = RuntimeError("construction failure")
        with mock.patch.object(multiprocessing, "get_context", return_value=context):
            with self.assertRaisesRegex(QualificationError, "could not start"):
                function(**self.queue_arguments())
        receive.close.assert_called_once()
        send.close.assert_called_once()

    def test_queue_workload_interrupt_terminates_and_closes_worker(self):
        function = MODULE["exercise_queue_disconnects"]
        context = mock.Mock()
        receive, send, process = mock.Mock(), mock.Mock(), mock.Mock()
        context.Pipe.return_value = receive, send
        context.Process.return_value = process
        receive.poll.side_effect = KeyboardInterrupt
        process.is_alive.side_effect = [True, False, False, False]
        with mock.patch.object(multiprocessing, "get_context", return_value=context):
            with self.assertRaises(KeyboardInterrupt):
                function(**self.queue_arguments())
        process.terminate.assert_called_once()
        process.join.assert_called_once_with(MODULE["QUEUE_WORKER_REAP_GRACE_SECONDS"])
        process.kill.assert_not_called()
        receive.close.assert_called_once()
        self.assertEqual(send.close.call_count, 2)
        process.close.assert_called_once()

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "fork isolation unavailable")
    def test_queue_workload_deadline_reaps_noncooperative_http_thread(self):
        context = multiprocessing.get_context("fork")
        receive, send = context.Pipe(duplex=False)

        def run_case():
            attempts = []
            token_reads = []

            class Response:
                def close(self):
                    pass

            def attempt(*_args):
                attempts.append(1)
                if len(attempts) <= 2:
                    return "accepted", Response()
                threading.Event().wait()  # never returns; executor shutdown cannot help

            def metric(_base, _token, name, _timeout):
                if name == "tritium_tokens_out_total":
                    token_reads.append(1)
                    return int(len(token_reads) > 1)
                return int(name == "tritium_queue_depth")

            function = MODULE["exercise_queue_disconnects"]
            try:
                with mock.patch.dict(function.__globals__, {
                    "slow_stream_attempt": attempt, "metric_value": metric,
                }):
                    function(
                        base_url="http://127.0.0.1", token="private-token",
                        metric_token="private-metric-token", model_id="m", prompt="p",
                        clients=3, max_tokens=32, timeout=1, hold_seconds=1,
                        recovery_timeout=1, wall_timeout=0.15,
                    )
            except QualificationError as error:
                send.send((str(error), len(multiprocessing.active_children())))
            finally:
                send.close()

        process = context.Process(target=run_case)
        started = time.monotonic()
        try:
            process.start()
            send.close()
            process.join(2)
            self.assertFalse(process.is_alive(), "queue qualification stranded after wall deadline")
            self.assertEqual(process.exitcode, 0)
            self.assertTrue(receive.poll())
            error, children = receive.recv()
            self.assertIn("wall deadline", error)
            self.assertEqual(children, 0)
            self.assertLess(time.monotonic() - started, 2)
        finally:
            if process.is_alive():
                process.kill()
                process.join(1)
            receive.close()
            send.close()
            process.close()

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

    def test_rejects_malformed_package_id_in_startup_receipt(self):
        value = readiness()
        value["startup_receipt"]["salt_package_id"] = "d" * 64
        with self.assertRaisesRegex(QualificationError, "trp1 package ID"):
            validate_ready(value, "a" * 40, "cpu", "compact-v1", "c" * 64)

    def test_rejects_package_id_encoding_for_manifest_digest(self):
        value = readiness()
        value["startup_receipt"]["manifest_package_id"] = "trp1_" + "c" * 64
        with self.assertRaisesRegex(QualificationError, "manifest package digest"):
            validate_ready(value, "a" * 40, "cpu", "compact-v1", "c" * 64)

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

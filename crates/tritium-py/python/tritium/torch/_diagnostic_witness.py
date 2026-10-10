"""Draft numerical witnesses; not installed-observability release admission."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct


SCHEMA = "tritium.diagnostic-measurement-witness.draft1"
WEIGHTS = ((-2.0, -0.4, 0.3, 2.1), (0.0, 1.0, -1.0, 0.2))
INPUTS = ((1.0, 1.0, 1.0, 1.0),)
FIELDS = {
    "schema", "binding", "fixture", "teacher_weight", "student_weight",
    "inputs", "teacher_logits", "student_logits", "timing", "memory",
    "metrics", "witness_id",
}
METRICS = {"runtime/forward_ms", "memory/resident_bytes", "teacher_kl"}


def witness_id(value: dict) -> str:
    body = {key: item for key, item in value.items() if key != "witness_id"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _object(value, fields, label):
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"diagnostic witness {label} fields differ")
    return value


def _integer(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"diagnostic witness {label} must be a positive integer")
    return value


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"diagnostic witness {label} must be finite numeric")
    return float(value)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _matrix(value, rows, columns, label):
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"diagnostic witness {label} shape differs")
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError(f"diagnostic witness {label} shape differs")
        result.append([_number(item, label) for item in row])
    try:
        if any(f32(item) != item for row in result for item in row):
            raise ValueError(f"diagnostic witness {label} must retain float32 values")
    except (OverflowError, struct.error) as error:
        raise ValueError(f"diagnostic witness {label} is outside float32") from error
    return result


def fixture_weights():
    return [[f32(item) for item in row] for row in WEIGHTS]


def projected_weights():
    result = []
    for row in fixture_weights():
        scale = f32(math.fsum(abs(item) for item in row) / 4)
        stored = struct.unpack("<e", struct.pack("<e", scale))[0]
        result.append([max(-1, min(1, round(f32(item / scale)))) * stored for item in row])
    return result


def _log_softmax(row):
    maximum = max(row)
    log_total = math.log(math.fsum(math.exp(item - maximum) for item in row))
    return [item - maximum - log_total for item in row]


def teacher_kl(teacher, student):
    values = []
    for left, right in zip(teacher, student):
        p = _log_softmax(left)
        q = _log_softmax(right)
        values.append(math.fsum(math.exp(a) * (a - b) for a, b in zip(p, q)))
    return max(0.0, math.fsum(values) / len(values))


def validate_witness(value: dict, *, expected_binding: dict | None = None) -> dict[str, float]:
    """Recompute draft witness arithmetic; never grant release credit."""
    _object(value, FIELDS, "root")
    if value["schema"] != SCHEMA or value["fixture"] != "tied-linear-4x2-cpu-f32-v1":
        raise ValueError("diagnostic witness schema/fixture differs")
    binding = _object(value["binding"], {"source_revision", "release", "run_id", "wheel_sha256"}, "binding")
    source = binding["source_revision"]
    digest = binding["wheel_sha256"]
    if (
        not isinstance(source, str) or len(source) != 40
        or set(source) - set("0123456789abcdef")
        or not isinstance(digest, str) or len(digest) != 71
        or not digest.startswith("sha256:") or set(digest[7:]) - set("0123456789abcdef")
        or any(not isinstance(binding[key], str) or not binding[key] for key in ("release", "run_id"))
    ):
        raise ValueError("diagnostic witness binding is invalid")
    if expected_binding is not None and binding != expected_binding:
        raise ValueError("diagnostic witness candidate binding differs")
    inputs = _matrix(value["inputs"], 1, 4, "inputs")
    dense = _matrix(value["teacher_weight"], 2, 4, "teacher weight")
    hard = _matrix(value["student_weight"], 2, 4, "student weight")
    if inputs != [list(row) for row in INPUTS] or dense != fixture_weights() or hard != projected_weights():
        raise ValueError("diagnostic witness frozen weight/input projection differs")
    teacher = _matrix(value["teacher_logits"], 1, 2, "teacher logits")
    student = _matrix(value["student_logits"], 1, 2, "student logits")
    for weights, logits in ((dense, teacher), (hard, student)):
        expected = [[2 * math.fsum(x * w for x, w in zip(row, weight)) for weight in weights] for row in inputs]
        if any(not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for left, right in zip(logits, expected) for a, b in zip(left, right)):
            raise ValueError("diagnostic witness forward logits differ from retained weights")
    timing = _object(value["timing"], {"clock", "warmups", "samples"}, "timing")
    if timing["clock"] != "perf_counter_ns" or type(timing["warmups"]) is not int or timing["warmups"] != 5:
        raise ValueError("diagnostic witness clock/warmups differ")
    if not isinstance(timing["samples"], list) or len(timing["samples"]) != 31:
        raise ValueError("diagnostic witness requires 31 timing samples")
    elapsed = []
    previous_end = 0
    for sample in timing["samples"]:
        _object(sample, {"start_ns", "end_ns", "student_logits"}, "timing sample")
        if _matrix(sample["student_logits"], 1, 2, "timed logits") != student:
            raise ValueError("diagnostic witness timed output differs from frozen student")
        start = _integer(sample["start_ns"], "start timestamp")
        end = _integer(sample["end_ns"], "end timestamp")
        if start < previous_end or end <= start:
            raise ValueError("diagnostic witness timing intervals are invalid")
        previous_end = end
        elapsed.append(end - start)
    memory = _object(value["memory"], {"method", "scope", "raw_statm", "page_size"}, "memory")
    if memory["method"] != "linux-proc-self-statm" or memory["scope"] != "whole-process-after-forward-before-telemetry":
        raise ValueError("diagnostic witness memory method/scope differs")
    raw = memory["raw_statm"]
    if not isinstance(raw, str) or len(raw) > 256:
        raise ValueError("diagnostic witness statm record is invalid")
    fields = raw.split()
    if len(fields) != 7 or any(not item.isascii() or not item.isdigit() for item in fields):
        raise ValueError("diagnostic witness statm fields are invalid")
    pages = [int(item) for item in fields]
    page_size = _integer(memory["page_size"], "page size")
    if page_size > 65536 or page_size & (page_size - 1) or not 0 < pages[1] <= pages[0]:
        raise ValueError("diagnostic witness RSS page arithmetic is invalid")
    rss = pages[1] * page_size
    if rss > 2**53:
        raise ValueError("diagnostic witness RSS exceeds exact telemetry integer range")
    expected_metrics = {
        "runtime/forward_ms": statistics.median(elapsed) / 1e6,
        "memory/resident_bytes": float(rss),
        "teacher_kl": teacher_kl(teacher, student),
    }
    metrics = _object(value["metrics"], METRICS, "metrics")
    for name, expected in expected_metrics.items():
        observed = _number(metrics[name], name)
        tolerance = 1e-12 if name == "teacher_kl" else 0.0
        if not math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError(f"diagnostic witness {name} differs from raw evidence")
    if value["witness_id"] != witness_id(value):
        raise ValueError("diagnostic witness identity differs")
    return expected_metrics

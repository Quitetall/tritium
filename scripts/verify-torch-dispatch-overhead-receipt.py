#!/usr/bin/env python3
"""Validate retained PyTorch dispatcher/direct-adapter overhead evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import statistics
from typing import Any


SCHEMA = "tritium.torch-dispatch-overhead-qualification.v1"
TRACE_SCHEMA = "tritium.torch-dispatch-overhead-samples.v1"
POLICY_ID = "torch-dispatch-cpu-linear-v1"
WARMUP_COUNT = 10
SAMPLE_COUNT = 31
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
OVERHEAD_LIMIT_RATIO = 1.05
BOOTSTRAP_SEED = 0x5452_4954_4955_4D
MAX_RECEIPT_BYTES = 32 * 1024 * 1024
RELEASE_PATTERN = re.compile(
    r"^(?P<base>[0-9]+(?:\.[0-9]+)*)(?:-(?P<stage>alpha|beta|rc)\.(?P<number>[0-9]+))?$"
)

POLICY_CASES = (
    {
        "case_id": "decode-forward",
        "phase": "forward",
        "m": 1,
        "n": 1024,
        "k": 1024,
        "bias": True,
        "repetitions": 16,
    },
    {
        "case_id": "decode-backward",
        "phase": "backward",
        "m": 1,
        "n": 1024,
        "k": 1024,
        "bias": True,
        "repetitions": 16,
    },
    {
        "case_id": "microbatch-forward",
        "phase": "forward",
        "m": 8,
        "n": 2048,
        "k": 2048,
        "bias": True,
        "repetitions": 2,
    },
    {
        "case_id": "microbatch-backward",
        "phase": "backward",
        "m": 8,
        "n": 2048,
        "k": 2048,
        "bias": True,
        "repetitions": 2,
    },
    {
        "case_id": "prefill-forward",
        "phase": "forward",
        "m": 32,
        "n": 1024,
        "k": 4096,
        "bias": True,
        "repetitions": 1,
    },
    {
        "case_id": "prefill-backward",
        "phase": "backward",
        "m": 32,
        "n": 1024,
        "k": 4096,
        "bias": True,
        "repetitions": 1,
    },
)

TRACE_FIELDS = {
    "schema",
    "release",
    "source_revision",
    "run_id",
    "result",
    "wheel",
    "runtime",
    "environment",
    "policy",
    "cases",
}
RECEIPT_FIELDS = {
    "schema",
    "receipt_id",
    "result",
    "release",
    "source_revision",
    "run_id",
    "wheel",
    "policy",
    "environment",
    "measurements",
    "trace",
}
WHEEL_FIELDS = {"name", "bytes", "sha256"}
RUNTIME_FIELDS = {
    "python",
    "torch",
    "tritium",
    "source_identity",
    "module_path",
    "native_module_path",
    "wheel_file_count",
    "verified_installed_file_count",
    "installed_tree_sha256",
}
ENVIRONMENT_FIELDS = {
    "system",
    "machine",
    "cpu_model",
    "logical_cpu_count",
    "affinity_before",
    "affinity_used",
    "rayon_threads",
    "torch_threads",
    "torch_interop_threads",
    "omp_threads",
    "mkl_threads",
    "source_tree_absent",
    "clock",
}
POLICY_FIELDS = {
    "policy_id",
    "warmup_count",
    "sample_count",
    "bootstrap_resamples",
    "bootstrap_confidence",
    "overhead_limit_ratio",
}
CASE_FIELDS = {
    "case_id",
    "phase",
    "m",
    "n",
    "k",
    "bias",
    "repetitions",
    "parity_exact",
    "cache_before",
    "cache_after",
    "warmups",
    "samples",
}
CACHE_FIELDS = {"capacity", "entries", "hits", "invalidations", "misses"}
SAMPLE_FIELDS = {
    "ordinal",
    "order",
    "direct_total_ns",
    "wrapper_total_ns",
}
MEASUREMENT_FIELDS = {
    "case_id",
    "phase",
    "m",
    "n",
    "k",
    "bias",
    "repetitions",
    "warmup_count",
    "sample_count",
    "median_direct_ns",
    "median_wrapper_ns",
    "median_ratio",
    "bootstrap_upper_ratio",
    "overhead_limit_ratio",
    "pass",
}
FILE_FIELDS = {"path", "bytes", "sha256"}


class DispatchOverheadError(ValueError):
    """Dispatch-overhead evidence is stale, incomplete, synthetic, or over budget."""


def canonical(value: Any) -> bytes:
    """Return canonical compact JSON bytes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(path: Path) -> str:
    """Hash one file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def distribution_version(release: str) -> str:
    match = RELEASE_PATTERN.fullmatch(release)
    if match is None:
        raise DispatchOverheadError(
            "release must use canonical Tritium version syntax"
        )
    stage = match.group("stage")
    if stage is None:
        return match.group("base")
    marker = {"alpha": "a", "beta": "b", "rc": "rc"}[stage]
    return f"{match.group('base')}{marker}{match.group('number')}"


def object_(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise DispatchOverheadError(f"{label} fields differ from frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DispatchOverheadError(f"{label} must be non-empty")
    return value


def integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DispatchOverheadError(f"{label} must be an integer at least {minimum}")
    return value


def number(value: Any, label: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise DispatchOverheadError(f"{label} must be finite and at least {minimum}")
    return float(value)


def _ordinary(path: Path, label: str, *, max_bytes: int = MAX_RECEIPT_BYTES) -> Path:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
        or path.stat().st_size > max_bytes
    ):
        raise DispatchOverheadError(f"{label} must be a bounded ordinary file")
    return path.resolve(strict=True)


def _canonical_document(path: Path, fields: set[str], label: str) -> dict[str, Any]:
    raw = _ordinary(path, label).read_bytes()
    try:
        value = object_(json.loads(raw), fields, label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DispatchOverheadError(f"{label} must contain UTF-8 JSON") from error
    if raw != canonical(value) + b"\n":
        raise DispatchOverheadError(f"{label} must use canonical JSON plus one newline")
    return value


def _validate_revision(value: Any) -> str:
    revision = string(value, "source revision")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise DispatchOverheadError(
            "source revision must be 40 lowercase hexadecimal characters"
        )
    return revision


def _validate_wheel(record: Any, expected_wheel: Path) -> dict[str, Any]:
    record = object_(record, WHEEL_FIELDS, "wheel")
    expected_wheel = _ordinary(expected_wheel, "expected wheel", max_bytes=4 * 1024**3)
    actual = (
        expected_wheel.name,
        expected_wheel.stat().st_size,
        sha256(expected_wheel),
    )
    declared = (
        string(record["name"], "wheel name"),
        integer(record["bytes"], "wheel bytes", 1),
        string(record["sha256"], "wheel SHA-256"),
    )
    if actual != declared:
        raise DispatchOverheadError("trace does not bind exact candidate wheel bytes")
    return record


def _validate_policy(value: Any) -> dict[str, Any]:
    policy = object_(value, POLICY_FIELDS, "policy")
    expected = {
        "policy_id": POLICY_ID,
        "warmup_count": WARMUP_COUNT,
        "sample_count": SAMPLE_COUNT,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "overhead_limit_ratio": OVERHEAD_LIMIT_RATIO,
    }
    if policy != expected:
        raise DispatchOverheadError("measurement policy differs from frozen policy")
    return policy


def _validate_environment(value: Any) -> dict[str, Any]:
    environment = object_(value, ENVIRONMENT_FIELDS, "environment")
    if string(environment["system"], "environment system") != "Linux":
        raise DispatchOverheadError("dispatch overhead qualification requires Linux")
    string(environment["machine"], "environment machine")
    string(environment["cpu_model"], "CPU model")
    integer(environment["logical_cpu_count"], "logical CPU count", 1)
    affinity = environment["affinity_before"]
    if (
        not isinstance(affinity, list)
        or not affinity
        or any(
            isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
            for cpu in affinity
        )
        or affinity != sorted(set(affinity))
    ):
        raise DispatchOverheadError("original CPU affinity must be sorted and unique")
    used = integer(environment["affinity_used"], "used CPU")
    if used not in affinity:
        raise DispatchOverheadError("used CPU was absent from original affinity")
    for field in (
        "rayon_threads",
        "torch_threads",
        "torch_interop_threads",
        "omp_threads",
        "mkl_threads",
    ):
        if integer(environment[field], field, 1) != 1:
            raise DispatchOverheadError("qualification requires one pinned execution thread")
    if environment["source_tree_absent"] is not True:
        raise DispatchOverheadError("qualification did not run source-tree-free")
    if environment["clock"] != "perf_counter_ns":
        raise DispatchOverheadError("qualification used an unapproved clock")
    return environment


def _validate_cache(value: Any, label: str) -> dict[str, int]:
    cache = object_(value, CACHE_FIELDS, label)
    for field in CACHE_FIELDS:
        integer(cache[field], f"{label}.{field}")
    if (
        cache["capacity"] != 4096
        or cache["entries"] != 1
        or cache["misses"] != 1
        or cache["invalidations"] != 0
    ):
        raise DispatchOverheadError(f"{label} does not prove one stable native cache entry")
    return cache


def _validate_samples(
    values: Any, expected_count: int, label: str
) -> list[dict[str, Any]]:
    if not isinstance(values, list) or len(values) != expected_count:
        raise DispatchOverheadError(f"{label} count differs from frozen policy")
    result = []
    for ordinal, raw in enumerate(values):
        sample = object_(raw, SAMPLE_FIELDS, f"{label}[{ordinal}]")
        if integer(sample["ordinal"], f"{label} ordinal") != ordinal:
            raise DispatchOverheadError(f"{label} ordinals are not contiguous")
        expected_order = "direct-first" if ordinal % 2 == 0 else "wrapper-first"
        if sample["order"] != expected_order:
            raise DispatchOverheadError(f"{label} does not use alternating AB/BA order")
        integer(sample["direct_total_ns"], f"{label} direct time", 1)
        integer(sample["wrapper_total_ns"], f"{label} wrapper time", 1)
        result.append(sample)
    return result


def _bootstrap_upper_ratio(
    direct: list[float], wrapper: list[float], case_ordinal: int
) -> float:
    rng = random.Random(BOOTSTRAP_SEED + case_ordinal)
    count = len(direct)
    ratios = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = [rng.randrange(count) for _ in range(count)]
        baseline = statistics.median(direct[index] for index in indices)
        candidate = statistics.median(wrapper[index] for index in indices)
        ratios.append(candidate / baseline)
    ratios.sort()
    index = math.ceil(BOOTSTRAP_CONFIDENCE * len(ratios)) - 1
    return ratios[index]


def aggregate_case(value: Any, case_ordinal: int) -> dict[str, Any]:
    """Validate and aggregate one frozen paired benchmark case."""

    case = object_(value, CASE_FIELDS, f"cases[{case_ordinal}]")
    if case_ordinal >= len(POLICY_CASES):
        raise DispatchOverheadError("unexpected dispatch benchmark case")
    policy_case = POLICY_CASES[case_ordinal]
    for field, expected in policy_case.items():
        if case[field] != expected:
            raise DispatchOverheadError(
                f"cases[{case_ordinal}].{field} differs from frozen shape"
            )
    if case["parity_exact"] is not True:
        raise DispatchOverheadError("direct and wrapper results were not exactly equal")
    cache_before = _validate_cache(case["cache_before"], "cache_before")
    cache_after = _validate_cache(case["cache_after"], "cache_after")
    if cache_before["hits"] != 0:
        raise DispatchOverheadError("case cache was not reset before parity measurement")
    repetitions = integer(case["repetitions"], "case repetitions", 1)
    expected_hits = 2 + (WARMUP_COUNT + SAMPLE_COUNT) * 2 * repetitions
    if cache_after["hits"] != expected_hits:
        raise DispatchOverheadError(
            "cache-hit delta does not prove every direct and wrapper call used native state"
        )
    _validate_samples(case["warmups"], WARMUP_COUNT, "warmups")
    samples = _validate_samples(case["samples"], SAMPLE_COUNT, "samples")
    direct = [sample["direct_total_ns"] / repetitions for sample in samples]
    wrapper = [sample["wrapper_total_ns"] / repetitions for sample in samples]
    median_direct = statistics.median(direct)
    median_wrapper = statistics.median(wrapper)
    ratio = median_wrapper / median_direct
    upper = _bootstrap_upper_ratio(direct, wrapper, case_ordinal)
    passed = ratio <= OVERHEAD_LIMIT_RATIO and upper <= OVERHEAD_LIMIT_RATIO
    if not passed:
        raise DispatchOverheadError(
            f"{case['case_id']} exceeds five-percent wrapper-overhead gate "
            f"(median={ratio:.6f}, upper={upper:.6f})"
        )
    return {
        **policy_case,
        "warmup_count": WARMUP_COUNT,
        "sample_count": SAMPLE_COUNT,
        "median_direct_ns": median_direct,
        "median_wrapper_ns": median_wrapper,
        "median_ratio": ratio,
        "bootstrap_upper_ratio": upper,
        "overhead_limit_ratio": OVERHEAD_LIMIT_RATIO,
        "pass": True,
    }


def validate_trace(
    path: Path,
    *,
    expected_revision: str,
    expected_release: str,
    expected_wheel: Path,
) -> dict[str, Any]:
    """Validate one source-free raw timing trace."""

    trace = _canonical_document(path, TRACE_FIELDS, "dispatch trace")
    if trace["schema"] != TRACE_SCHEMA or trace["result"] != "complete":
        raise DispatchOverheadError("dispatch trace schema or result differs")
    if (
        _validate_revision(trace["source_revision"]) != expected_revision
        or string(trace["release"], "release") != expected_release
    ):
        raise DispatchOverheadError("dispatch trace source or release is stale")
    string(trace["run_id"], "run id")
    _validate_wheel(trace["wheel"], expected_wheel)
    runtime = object_(trace["runtime"], RUNTIME_FIELDS, "runtime")
    for field in (
        "python",
        "torch",
        "source_identity",
        "module_path",
        "native_module_path",
    ):
        string(runtime[field], f"runtime.{field}")
    if string(runtime["tritium"], "runtime.tritium") != distribution_version(
        expected_release
    ):
        raise DispatchOverheadError(
            "installed Tritium version differs from expected release"
        )
    if runtime["source_identity"] != f"source-git:{expected_revision}":
        raise DispatchOverheadError(
            "installed wheel source identity differs from expected revision"
        )
    wheel_files = integer(runtime["wheel_file_count"], "wheel file count", 1)
    if (
        integer(
            runtime["verified_installed_file_count"],
            "verified installed file count",
            1,
        )
        != wheel_files
    ):
        raise DispatchOverheadError("installed wheel inventory is incomplete")
    tree = string(runtime["installed_tree_sha256"], "installed tree SHA-256")
    if (
        not tree.startswith("sha256:")
        or len(tree) != 71
        or any(character not in "0123456789abcdef" for character in tree[7:])
    ):
        raise DispatchOverheadError("installed tree SHA-256 is malformed")
    _validate_environment(trace["environment"])
    _validate_policy(trace["policy"])
    cases = trace["cases"]
    if not isinstance(cases, list) or len(cases) != len(POLICY_CASES):
        raise DispatchOverheadError("dispatch trace does not contain every frozen case")
    for ordinal, case in enumerate(cases):
        aggregate_case(case, ordinal)
    return trace


def _support_file(root: Path, value: Any) -> Path:
    record = object_(value, FILE_FIELDS, "trace file")
    logical = PurePosixPath(string(record["path"], "trace path"))
    text = str(logical)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise DispatchOverheadError("trace path is unsafe")
    cursor = root.resolve(strict=True)
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise DispatchOverheadError("trace path traverses a symlink")
    path = _ordinary(cursor, "retained trace")
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise DispatchOverheadError("retained trace escapes receipt directory") from error
    if (
        path.stat().st_size != integer(record["bytes"], "trace bytes", 1)
        or sha256(path) != string(record["sha256"], "trace SHA-256")
    ):
        raise DispatchOverheadError("retained trace bytes drifted")
    return path


def validate(
    path: Path,
    *,
    expected_revision: str,
    expected_release: str,
    expected_wheel: Path,
) -> dict[str, Any]:
    """Validate a retained dispatch-overhead qualification receipt."""

    receipt = _canonical_document(path, RECEIPT_FIELDS, "dispatch receipt")
    if receipt["schema"] != SCHEMA or receipt["result"] != "pass":
        raise DispatchOverheadError("dispatch receipt schema or result differs")
    if (
        _validate_revision(receipt["source_revision"]) != expected_revision
        or string(receipt["release"], "release") != expected_release
    ):
        raise DispatchOverheadError("dispatch receipt source or release is stale")
    string(receipt["run_id"], "run id")
    _validate_wheel(receipt["wheel"], expected_wheel)
    _validate_policy(receipt["policy"])
    _validate_environment(receipt["environment"])
    trace_path = _support_file(path.parent, receipt["trace"])
    trace = validate_trace(
        trace_path,
        expected_revision=expected_revision,
        expected_release=expected_release,
        expected_wheel=expected_wheel,
    )
    if (
        receipt["run_id"] != trace["run_id"]
        or receipt["wheel"] != trace["wheel"]
        or receipt["policy"] != trace["policy"]
        or receipt["environment"] != trace["environment"]
    ):
        raise DispatchOverheadError("receipt does not bind retained trace identity")
    expected_measurements = [
        aggregate_case(case, ordinal) for ordinal, case in enumerate(trace["cases"])
    ]
    measurements = receipt["measurements"]
    if (
        not isinstance(measurements, list)
        or len(measurements) != len(POLICY_CASES)
        or any(
            set(measurement) != MEASUREMENT_FIELDS
            for measurement in measurements
            if isinstance(measurement, dict)
        )
        or measurements != expected_measurements
    ):
        raise DispatchOverheadError("receipt aggregates differ from retained trace")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    expected_id = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if receipt["receipt_id"] != expected_id:
        raise DispatchOverheadError("dispatch receipt identity differs")
    return receipt


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    validate(
        args.receipt,
        expected_revision=args.source_revision,
        expected_release=args.release,
        expected_wheel=args.wheel,
    )
    print("torch dispatch overhead receipt: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

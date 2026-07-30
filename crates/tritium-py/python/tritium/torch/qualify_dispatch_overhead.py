"""Source-free paired benchmark for native PyTorch dispatcher overhead."""

from __future__ import annotations

import argparse
import base64
import binascii
import ctypes
import csv
import errno
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import tempfile
import time
from typing import Any, Callable
import zipfile
from email.parser import BytesParser

import torch

from tritium import _tritium as _native
from tritium.torch import ternary_linear


TRACE_SCHEMA = "tritium.torch-dispatch-overhead-samples.v1"
POLICY_ID = "torch-dispatch-cpu-linear-v1"
WARMUP_COUNT = 10
SAMPLE_COUNT = 31
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
OVERHEAD_LIMIT_RATIO = 1.05
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
_THREAD_ENVIRONMENT = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_MAX_WHEEL_BYTES = 4 * 1024**3
_MAX_WHEEL_FILES = 10_000
_MAX_WHEEL_MEMBER_BYTES = 512 * 1024**2
_RELEASE_PATTERN = re.compile(
    r"^(?P<base>[0-9]+(?:\.[0-9]+)*)(?:-(?P<stage>alpha|beta|rc)\.(?P<number>[0-9]+))?$"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _distribution_version(release: str) -> str:
    match = _RELEASE_PATTERN.fullmatch(release)
    if match is None:
        raise ValueError("release must use canonical Tritium version syntax")
    stage = match.group("stage")
    if stage is None:
        return match.group("base")
    marker = {"alpha": "a", "beta": "b", "rc": "rc"}[stage]
    return f"{match.group('base')}{marker}{match.group('number')}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_absent(path: Path) -> bool:
    resolved = path.resolve(strict=True)
    return all(
        not (parent / ".git").exists() and not (parent / "Cargo.toml").exists()
        for parent in (resolved, *resolved.parents)
    )


def _safe_wheel_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or not path.parts
        or str(path) != value
    ):
        raise RuntimeError("wheel RECORD contains an unsafe path")
    return path


def _record_sha256(value: str) -> str:
    algorithm, separator, encoded = value.partition("=")
    if separator != "=" or algorithm != "sha256" or not encoded:
        raise RuntimeError("wheel RECORD must use SHA-256 hashes")
    padding = "=" * (-len(encoded) % 4)
    try:
        digest = base64.urlsafe_b64decode(encoded + padding)
    except (ValueError, binascii.Error) as error:
        raise RuntimeError("wheel RECORD contains malformed SHA-256") from error
    if len(digest) != hashlib.sha256().digest_size:
        raise RuntimeError("wheel RECORD SHA-256 has the wrong length")
    return digest.hex()


def _snapshot_wheel(wheel: Path) -> tuple[Any, dict[str, Any]]:
    snapshot = tempfile.TemporaryFile()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(wheel, flags)
        with os.fdopen(descriptor, "rb") as source:
            source_stat = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(source_stat.st_mode)
                or source_stat.st_size <= 0
                or source_stat.st_size > _MAX_WHEEL_BYTES
            ):
                raise ValueError("wheel must be a bounded ordinary file")
            digest = hashlib.sha256()
            copied = 0
            while chunk := source.read(1024 * 1024):
                snapshot.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            if copied != source_stat.st_size:
                raise RuntimeError("wheel changed while immutable snapshot was captured")
        snapshot.flush()
        snapshot.seek(0)
        return snapshot, {
            "name": wheel.name,
            "bytes": copied,
            "sha256": digest.hexdigest(),
        }
    except BaseException:
        snapshot.close()
        raise


def _publish_fd_noreplace(descriptor: int, output: Path) -> None:
    linkat = getattr(ctypes.CDLL(None, use_errno=True), "linkat", None)
    if linkat is None:
        raise RuntimeError("linkat is required for no-clobber publication")
    linkat.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    linkat.restype = ctypes.c_int
    result = linkat(descriptor, b"", -100, os.fsencode(output), 0x1000)
    if result == 0:
        return
    code = ctypes.get_errno()
    if code == errno.EEXIST:
        raise FileExistsError(f"output already exists: {output}")
    raise RuntimeError(f"no-clobber publication failed: {os.strerror(code)}")


def _allowed_installer_extra(
    logical: str, expected_names: set[str], dist_info: str
) -> bool:
    if logical in {
        dist_info + name
        for name in ("INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json")
    }:
        return True
    path = PurePosixPath(logical)
    if len(path.parts) < 3 or path.parent.name != "__pycache__":
        return False
    cache_tag = sys.implementation.cache_tag
    suffix = f".{cache_tag}.pyc"
    if not cache_tag or not path.name.endswith(suffix):
        return False
    source_name = path.name[: -len(suffix)] + ".py"
    source = (path.parent.parent / source_name).as_posix()
    return source in expected_names


def _verify_installed_wheel(wheel: Any, wheel_name: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution("tritium-torch")
    if distribution.files is None:
        raise RuntimeError("installed tritium-torch distribution has no file inventory")
    distribution_base = Path(distribution.locate_file(""))
    distribution_root = distribution_base.resolve(strict=True)
    wheel.seek(0)
    with zipfile.ZipFile(wheel) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        if (
            not files
            or len(files) > _MAX_WHEEL_FILES
            or any(info.file_size > _MAX_WHEEL_MEMBER_BYTES for info in files)
            or sum(info.file_size for info in files) > _MAX_WHEEL_BYTES
        ):
            raise RuntimeError("wheel uncompressed inventory exceeds safety bounds")
        names = [info.filename for info in files]
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if (
            len(record_names) != 1
            or len(metadata_names) != 1
            or len(wheel_names) != 1
            or len(names) != len(set(names))
        ):
            raise RuntimeError(
                "wheel must contain one METADATA, one WHEEL, one RECORD, and unique file names"
            )
        record_name = record_names[0]
        dist_info = record_name.removesuffix("RECORD")
        if (
            metadata_names[0] != dist_info + "METADATA"
            or wheel_names[0] != dist_info + "WHEEL"
        ):
            raise RuntimeError("wheel metadata roots differ")
        for info in files:
            _safe_wheel_path(info.filename)
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise RuntimeError("wheel contains a symlink member")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        if metadata.get("Name", "").lower().replace("_", "-") != "tritium-torch":
            raise RuntimeError("wheel distribution name differs from tritium-torch")
        wheel_version = metadata.get("Version")
        if not wheel_version or wheel_version != distribution.version:
            raise RuntimeError("installed distribution version differs from wheel")
        filename = wheel_name.removesuffix(".whl").split("-")
        if (
            not wheel_name.endswith(".whl")
            or len(filename) != 5
            or filename[0].lower().replace("-", "_") != "tritium_torch"
            or filename[1] != wheel_version
            or dist_info != f"tritium_torch-{wheel_version}.dist-info/"
        ):
            raise RuntimeError("wheel filename or dist-info identity differs")
        wheel_metadata = BytesParser().parsebytes(archive.read(wheel_names[0]))
        expected_tag = "-".join(filename[2:])
        if (
            wheel_metadata.get("Wheel-Version") != "1.0"
            or wheel_metadata.get("Root-Is-Purelib", "").lower() != "false"
            or expected_tag not in (wheel_metadata.get_all("Tag") or [])
        ):
            raise RuntimeError("wheel compatibility metadata differs from filename")
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
        records: dict[str, tuple[str, int]] = {}
        for row in rows:
            if len(row) != 3:
                raise RuntimeError("wheel RECORD row must contain three fields")
            name = _safe_wheel_path(row[0]).as_posix()
            if name in records:
                raise RuntimeError("wheel RECORD contains a duplicate path")
            if name == record_name:
                if row[1] or row[2]:
                    raise RuntimeError("wheel RECORD self-row must omit hash and size")
                continue
            digest = _record_sha256(row[1])
            try:
                size = int(row[2])
            except ValueError as error:
                raise RuntimeError("wheel RECORD size is malformed") from error
            if size < 0:
                raise RuntimeError("wheel RECORD size is negative")
            records[name] = (digest, size)
        if set(names) != set(records) | {record_name}:
            raise RuntimeError("wheel files and RECORD inventory differ")

        owned_names = [str(item).replace("\\", "/") for item in distribution.files]
        if len(owned_names) != len(set(owned_names)):
            raise RuntimeError("installed distribution inventory contains duplicates")
        expected_names = set(names)
        actual_names = set(owned_names)
        if not expected_names.issubset(actual_names) or any(
            not _allowed_installer_extra(name, expected_names, dist_info)
            for name in actual_names - expected_names
        ):
            raise RuntimeError("installed distribution inventory differs from wheel")

        installed = []
        verified_names = set()
        for name, (expected_digest, expected_size) in sorted(records.items()):
            wheel_bytes = archive.read(name)
            if (
                len(wheel_bytes) != expected_size
                or hashlib.sha256(wheel_bytes).hexdigest() != expected_digest
            ):
                raise RuntimeError("wheel member bytes differ from RECORD")
            unresolved = Path(distribution.locate_file(name))
            cursor = unresolved
            while cursor != distribution_base:
                if cursor.is_symlink():
                    raise RuntimeError("installed wheel path traverses a symlink")
                parent = cursor.parent
                if parent == cursor:
                    raise RuntimeError("installed wheel member escapes distribution root")
                cursor = parent
            path = unresolved.resolve(strict=True)
            try:
                path.relative_to(distribution_root)
            except ValueError as error:
                raise RuntimeError(
                    "installed wheel member escapes distribution root"
                ) from error
            if not path.is_file():
                raise RuntimeError("installed wheel member is missing or not ordinary")
            if path.stat().st_size != expected_size or _sha256(path) != expected_digest:
                raise RuntimeError("installed distribution differs from candidate wheel")
            installed.append(
                {
                    "path": name,
                    "bytes": expected_size,
                    "sha256": expected_digest,
                }
            )
            verified_names.add(name)
        installed_record = Path(distribution.locate_file(record_name))
        if installed_record.is_symlink() or not installed_record.is_file():
            raise RuntimeError("installed distribution RECORD is not an ordinary file")
        verified_names.add(record_name)
    tree_digest = hashlib.sha256(_canonical(installed)).hexdigest()
    return {
        "distribution_version": distribution.version,
        "wheel_file_count": len(names),
        "verified_installed_file_count": len(verified_names),
        "installed_tree_sha256": "sha256:" + tree_digest,
        "distribution_root": distribution_root,
        "verified_names": verified_names,
    }


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == "model name" and value.strip():
                return value.strip()
    value = platform.processor().strip()
    return value or "unknown"


def _cache_info() -> dict[str, int]:
    return {
        key: int(value)
        for key, value in _native._ternary_linear_cache_info().items()
    }


def _consume(value: Any) -> float:
    tensor = value[0] if isinstance(value, tuple) else value
    return float(tensor.detach().reshape(-1)[0])


def _time_batch(function: Callable[[], Any], repetitions: int) -> int:
    result = None
    started = time.perf_counter_ns()
    for _ in range(repetitions):
        result = function()
    elapsed = time.perf_counter_ns() - started
    if result is None:
        raise RuntimeError("timed function produced no result")
    _consume(result)
    return elapsed


def _paired_samples(
    direct: Callable[[], Any],
    wrapper: Callable[[], Any],
    *,
    count: int,
    repetitions: int,
) -> list[dict[str, Any]]:
    samples = []
    for ordinal in range(count):
        direct_first = ordinal % 2 == 0
        order = (
            (("direct", direct), ("wrapper", wrapper))
            if direct_first
            else (("wrapper", wrapper), ("direct", direct))
        )
        timings: dict[str, int] = {}
        for label, function in order:
            timings[label] = _time_batch(function, repetitions)
        samples.append(
            {
                "ordinal": ordinal,
                "order": "direct-first" if direct_first else "wrapper-first",
                "direct_total_ns": timings["direct"],
                "wrapper_total_ns": timings["wrapper"],
            }
        )
    return samples


def _exact_equal(left: Any, right: Any) -> bool:
    left_values = left if isinstance(left, tuple) else (left,)
    right_values = right if isinstance(right, tuple) else (right,)
    return len(left_values) == len(right_values) and all(
        torch.equal(left_tensor, right_tensor)
        for left_tensor, right_tensor in zip(left_values, right_values)
    )


def _run_case(policy: dict[str, Any], seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    m, n, k = policy["m"], policy["n"], policy["k"]
    input_ = torch.randn(m, k, dtype=torch.float32, requires_grad=True)
    master = torch.randn(n, k, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(n, dtype=torch.float32, requires_grad=True)
    grad_output = torch.randn(m, n, dtype=torch.float32)
    detached_input = input_.detach()
    detached_master = master.detach()
    detached_bias = bias.detach()
    version = int(master._version)
    storage_identity = int(master.untyped_storage()._cdata)

    _native._ternary_linear_cache_clear()
    output = ternary_linear(input_, master, bias)
    cache_before = _cache_info()

    def direct_forward():
        capsule = _native._ternary_linear_cpu_dlpack(
            detached_input,
            detached_master,
            detached_bias,
            None,
            master,
            version,
            storage_identity,
        )
        if capsule is None:
            raise RuntimeError("direct CPU forward rejected frozen finite inputs")
        return torch.from_dlpack(capsule)

    def wrapper_forward():
        return ternary_linear(input_, master, bias)

    def direct_backward():
        capsules = _native._ternary_linear_backward_cpu_dlpack(
            grad_output,
            detached_input,
            detached_master,
            None,
            master,
            version,
            storage_identity,
            True,
        )
        if capsules is None:
            raise RuntimeError("direct CPU backward rejected frozen finite inputs")
        return tuple(torch.from_dlpack(capsule) for capsule in capsules)

    def wrapper_backward():
        return torch.autograd.grad(
            output,
            (input_, master, bias),
            grad_output,
            retain_graph=True,
            create_graph=False,
        )

    direct = direct_forward if policy["phase"] == "forward" else direct_backward
    wrapper = wrapper_forward if policy["phase"] == "forward" else wrapper_backward
    parity_exact = _exact_equal(direct(), wrapper())
    if not parity_exact:
        raise RuntimeError(f"{policy['case_id']} direct/wrapper parity failed")
    repetitions = policy["repetitions"]
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        warmups = _paired_samples(
            direct,
            wrapper,
            count=WARMUP_COUNT,
            repetitions=repetitions,
        )
        samples = _paired_samples(
            direct,
            wrapper,
            count=SAMPLE_COUNT,
            repetitions=repetitions,
        )
    finally:
        if was_enabled:
            gc.enable()
    cache_after = _cache_info()
    return {
        **policy,
        "parity_exact": parity_exact,
        "cache_before": cache_before,
        "cache_after": cache_after,
        "warmups": warmups,
        "samples": samples,
    }


def _validate_inputs(
    output: Path,
    wheel: Path,
    source_revision: str,
    release: str,
    run_id: str,
    cpu: int,
) -> tuple[Path, list[int]]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    if wheel.is_symlink() or not wheel.is_file() or not wheel.name.endswith(".whl"):
        raise ValueError("wheel must be an ordinary .whl file")
    if wheel.stat().st_size <= 0 or wheel.stat().st_size > _MAX_WHEEL_BYTES:
        raise ValueError("wheel size exceeds qualification safety bounds")
    if (
        len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise ValueError("source revision must be 40 lowercase hexadecimal characters")
    if not release or not run_id:
        raise ValueError("release and run id must be non-empty")
    _distribution_version(release)
    if platform.system() != "Linux" or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("dispatch overhead qualification requires Linux CPU affinity")
    affinity = sorted(os.sched_getaffinity(0))
    if cpu not in affinity:
        raise ValueError("requested CPU is outside process affinity")
    for name in _THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required before interpreter startup")
    return wheel.resolve(strict=True), affinity


def run(
    output: Path,
    *,
    wheel: Path,
    source_revision: str,
    release: str,
    run_id: str,
    cpu: int,
) -> dict[str, Any]:
    """Run every frozen direct-vs-wrapper case and write one raw trace."""

    wheel, affinity_before = _validate_inputs(
        output, wheel, source_revision, release, run_id, cpu
    )
    wheel_snapshot, wheel_record = _snapshot_wheel(wheel)
    try:
        installed = _verify_installed_wheel(wheel_snapshot, wheel_record["name"])
    finally:
        wheel_snapshot.close()
    if installed["distribution_version"] != _distribution_version(release):
        raise RuntimeError("installed wheel version differs from requested release")
    module_path = Path(__file__).resolve(strict=True)
    native_module_path = Path(_native.__file__).resolve(strict=True)
    for label, path in (
        ("qualifier", module_path),
        ("native extension", native_module_path),
    ):
        try:
            relative = path.relative_to(installed["distribution_root"]).as_posix()
        except ValueError as error:
            raise RuntimeError(f"{label} was not imported from installed wheel") from error
        if relative not in installed["verified_names"]:
            raise RuntimeError(f"{label} bytes are absent from verified wheel inventory")
    source_tree_absent = _source_tree_absent(Path.cwd()) and _source_tree_absent(
        module_path
    )
    if not source_tree_absent:
        raise RuntimeError("dispatch qualification requires source-tree-free execution")
    if "cpu" not in _native.compiled_backends():
        raise RuntimeError("installed wheel does not expose CPU backend")
    source_identity = _native.source_identity()
    expected_source_identity = f"source-git:{source_revision}"
    if source_identity != expected_source_identity:
        raise RuntimeError(
            "installed wheel source identity differs from requested revision: "
            f"expected {expected_source_identity}, got {source_identity}"
        )

    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        cases = [
            _run_case(policy, 0xC0FFEE + ordinal)
            for ordinal, policy in enumerate(POLICY_CASES)
        ]
    finally:
        os.sched_setaffinity(0, set(affinity_before))

    trace = {
        "schema": TRACE_SCHEMA,
        "release": release,
        "source_revision": source_revision,
        "run_id": run_id,
        "result": "complete",
        "wheel": wheel_record,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tritium": installed["distribution_version"],
            "source_identity": source_identity,
            "module_path": str(module_path),
            "native_module_path": str(native_module_path),
            "wheel_file_count": installed["wheel_file_count"],
            "verified_installed_file_count": installed[
                "verified_installed_file_count"
            ],
            "installed_tree_sha256": installed["installed_tree_sha256"],
        },
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count() or 1,
            "affinity_before": affinity_before,
            "affinity_used": cpu,
            "rayon_threads": 1,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "omp_threads": int(os.environ["OMP_NUM_THREADS"]),
            "mkl_threads": int(os.environ["MKL_NUM_THREADS"]),
            "source_tree_absent": source_tree_absent,
            "clock": "perf_counter_ns",
        },
        "policy": {
            "policy_id": POLICY_ID,
            "warmup_count": WARMUP_COUNT,
            "sample_count": SAMPLE_COUNT,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
            "overhead_limit_ratio": OVERHEAD_LIMIT_RATIO,
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(os, "O_TMPFILE"):
        raise RuntimeError("Linux O_TMPFILE is required for atomic trace publication")
    descriptor = os.open(
        output.parent,
        os.O_TMPFILE | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(trace) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            _publish_fd_noreplace(stream.fileno(), output)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    trace = run(
        args.output.absolute(),
        wheel=args.wheel.absolute(),
        source_revision=args.source_revision,
        release=args.release,
        run_id=args.run_id,
        cpu=args.cpu,
    )
    print(
        json.dumps(
            {
                "schema": trace["schema"],
                "result": trace["result"],
                "run_id": trace["run_id"],
                "cases": len(trace["cases"]),
                "output": str(args.output.absolute()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

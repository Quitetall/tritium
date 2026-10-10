"""Public PTQ regression guard for native row-level parallel fitting."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
import unittest


def _native_probe_installation(expected_revision=None):
    """Reject a different distribution, shadowed namespace, or stale native fitter."""
    import base64
    import hashlib
    import importlib.metadata
    from pathlib import Path
    import re

    import tritium
    from tritium import _tritium
    from tritium.torch import ptq

    try:
        distribution = importlib.metadata.distribution("pytritium")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("PTQ parallelism requires an installed pytritium wheel") from error
    if not callable(getattr(_tritium, "_fit_joint_ternary_diagonal_groups_with_objective", None)):
        raise RuntimeError("PTQ parallelism requires the current native PTQ fitter")
    if not callable(getattr(_tritium, "source_identity", None)):
        raise RuntimeError("PTQ parallelism requires native source identity")
    identity = _tritium.source_identity()
    if not isinstance(identity, str) or re.fullmatch(r"source-git:[0-9a-f]{40}", identity) is None:
        raise RuntimeError("PTQ parallelism requires a clean native source identity")
    if expected_revision is not None:
        if (not isinstance(expected_revision, str)
                or re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None):
            raise RuntimeError("PTQ parallelism expected source revision is not canonical")
        if identity != "source-git:" + expected_revision:
            raise RuntimeError("PTQ parallelism native source identity differs from expected revision")

    records = {str(record): record for record in distribution.files or ()}
    native_name = Path(_tritium.__file__).name
    modules = {
        "tritium/__init__.py": tritium,
        "tritium/torch/ptq.py": ptq,
        "tritium/" + native_name: _tritium,
    }
    digests = {}
    for name, module in modules.items():
        record = records.get(name)
        if record is None:
            raise RuntimeError("PTQ parallelism module is missing from pytritium RECORD: " + name)
        path = Path(module.__file__)
        if path.resolve() != Path(distribution.locate_file(record)).resolve():
            raise RuntimeError("PTQ parallelism imported module is not owned by pytritium: " + name)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("PTQ parallelism installed module is not an ordinary file: " + name)
        if (record.hash is None or record.hash.mode not in {"sha256", "sha384", "sha512"}
                or record.size != path.stat().st_size):
            raise RuntimeError("PTQ parallelism module lacks valid RECORD integrity: " + name)
        digest = hashlib.new(record.hash.mode)
        sha256 = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                sha256.update(chunk)
        encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
        if encoded != record.hash.value:
            raise RuntimeError("PTQ parallelism module bytes differ from RECORD: " + name)
        digests[name] = sha256.hexdigest()
    return {
        "distribution": "pytritium", "version": distribution.version,
        "source_identity": identity, "module_sha256": digests,
    }


def _cpu_quota_diagnostics(root="/sys/fs/cgroup", membership="/proc/self/cgroup"):
    """Bounded cgroup-v2 observations, not a capacity guarantee or admission gate."""
    from pathlib import Path

    result = {"complete": False, "observed_quota_cores": None, "scopes_read": 0}

    def read(path):
        try:
            with path.open("rb") as stream:
                value = stream.read(4097)
            return value.decode("ascii") if len(value) <= 4096 else None
        except (OSError, UnicodeError):
            return None

    content = read(Path(membership))
    if content is None:
        return result
    groups = [line[3:] for line in content.splitlines() if line.startswith("0::")]
    if len(groups) != 1 or not groups[0].startswith("/"):
        return result
    components = groups[0].split("/")[1:]
    if groups[0] == "/":
        components = []
    if len(components) > 64 or any(part in ("", ".", "..") for part in components):
        return result
    try:
        root = Path(root).resolve()
        current = root.joinpath(*components)
        if current.resolve() != current:
            return result
    except (OSError, RuntimeError):
        return result

    complete = True
    while True:
        value = read(current / "cpu.max")
        fields = value.split() if value is not None else []
        valid = len(fields) == 2 and fields[1].isascii() and fields[1].isdigit()
        valid = valid and 0 < int(fields[1]) <= 2**64 - 1
        if valid and fields[0] != "max":
            valid = fields[0].isascii() and fields[0].isdigit()
            valid = valid and 0 < int(fields[0]) <= 2**64 - 1
        if not valid:
            complete = False
        else:
            result["scopes_read"] += 1
            if fields[0] != "max":
                quota = int(fields[0]) / int(fields[1])
                previous = result["observed_quota_cores"]
                result["observed_quota_cores"] = quota if previous is None else min(previous, quota)
        if current == root:
            break
        current = current.parent
    result["complete"] = complete
    return result


# Embed the exact guard in the standalone installed-wheel child, without
# importing scripts or adding the source checkout to that child's sys.path.
_PROBE = inspect.getsource(_native_probe_installation) + inspect.getsource(_cpu_quota_diagnostics) + r"""
import hashlib
import json
import os
import tempfile
import time

import torch
from tritium import _tritium
from tritium.torch import TernaryConfig, calibrate, convert, prepare

installation = _native_probe_installation(os.environ.get("TRITIUM_SOURCE_REVISION"))
native_fit = _tritium._fit_joint_ternary_diagonal_groups_with_objective
native_work = {"calls": 0, "rows": 0}
native_timing = {"wall_seconds": 0.0, "cpu_seconds": 0.0}

def record_native_fit(*args, **kwargs):
    native_work["calls"] += 1
    native_work["rows"] += args[1]
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        return native_fit(*args, **kwargs)
    finally:
        native_timing["cpu_seconds"] += time.process_time() - cpu_started
        native_timing["wall_seconds"] += time.perf_counter() - wall_started

_tritium._fit_joint_ternary_diagonal_groups_with_objective = record_native_fit
torch.manual_seed(31)
model = torch.nn.Linear(256, 2048, bias=False)
prepared = prepare(
    model,
    TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
    inplace=False,
)
with tempfile.TemporaryDirectory(prefix="tritium-ptq-parallelism-") as root:
    evidence = calibrate(
        prepared,
        [torch.randn(1, 256)],
        evidence_dir=f"{root}/evidence",
    )
    # Count only work inside the measured public convert call, not preparation.
    native_work.update(calls=0, rows=0)
    native_timing.update(wall_seconds=0.0, cpu_seconds=0.0)
    started = time.perf_counter()
    cpu_started = time.process_time()
    artifact = convert(
        prepared,
        evidence,
        work_dir=f"{root}/conversion",
        max_working_bytes=256 * 1024 * 1024,
    )
    cpu_elapsed = time.process_time() - cpu_started
    elapsed = time.perf_counter() - started
    fitted = artifact.weight("weight")
    digest = hashlib.sha256()
    for plane in fitted.planes:
        digest.update(plane.trits.numpy().tobytes())
        digest.update(plane.scales.numpy().tobytes())
    benchmark_environment = {"cpu_count": os.cpu_count()}
    if hasattr(os, "sched_getaffinity"):
        benchmark_environment["affinity_cpus"] = len(os.sched_getaffinity(0))
    if hasattr(os, "getloadavg"):
        benchmark_environment["load_average_1m"] = os.getloadavg()[0]
    benchmark_environment["cgroup_v2"] = _cpu_quota_diagnostics()
    benchmark_environment["python_version"] = __import__("platform").python_version()
    benchmark_environment["torch_version"] = str(torch.__version__)
    print(json.dumps({
        "algorithm_id": artifact.algorithm_id,
        "installation": installation,
        "native_work": native_work,
        "benchmark_environment": benchmark_environment,
        "elapsed_seconds": elapsed,
        "timing": {
            "native_wall_seconds": native_timing["wall_seconds"],
            "native_cpu_seconds": native_timing["cpu_seconds"],
            "convert_cpu_seconds": cpu_elapsed,
            "other_wall_seconds": max(0.0, elapsed - native_timing["wall_seconds"]),
        },
        "fit_digest": digest.hexdigest(),
        "weighted_mse": fitted.weighted_mse,
    }, sort_keys=True))
"""


@unittest.skipUnless(
    importlib.util.find_spec("torch"),
    "the PTQ parallelism probe runs in the installed-wheel PyTorch lane",
)
class PublicPtqParallelismTests(unittest.TestCase):
    def _convert(self, rayon_threads: int) -> dict[str, object]:
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = str(rayon_threads)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=90,
        )
        self.assertEqual(
            completed.returncode,
            0,
            "public convert probe subprocess failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}",
        )
        return json.loads(completed.stdout)

    def test_public_convert_parallelizes_rows_without_changing_fitted_artifact(self):
        serial = self._convert(1)
        parallel = self._convert(4)

        self.assertEqual(parallel["algorithm_id"], serial["algorithm_id"])
        self.assertEqual(parallel["installation"], serial["installation"])
        self.assertEqual(parallel["native_work"], serial["native_work"])
        self.assertGreater(serial["native_work"]["calls"], 0)
        self.assertEqual(serial["native_work"]["rows"], 2048)
        self.assertEqual(parallel["fit_digest"], serial["fit_digest"])
        self.assertEqual(parallel["weighted_mse"], serial["weighted_mse"])
        speedup = float(serial["elapsed_seconds"]) / float(
            parallel["elapsed_seconds"]
        )
        print(
            "public PTQ convert rows=2048 columns=256 "
            f"serial_seconds={serial['elapsed_seconds']:.3f} "
            f"parallel_seconds={parallel['elapsed_seconds']:.3f} "
            f"speedup={speedup:.2f}x "
            f"source_identity={serial['installation']['source_identity']} "
            f"fit_digest={serial['fit_digest']} native_work={serial['native_work']} "
            f"serial_timing={serial['timing']} parallel_timing={parallel['timing']} "
            f"serial_environment={serial['benchmark_environment']} "
            f"parallel_environment={parallel['benchmark_environment']}"
        )
        self.assertGreaterEqual(
            speedup,
            1.5,
            f"public convert speedup with four Rayon threads was only {speedup:.2f}x",
        )


if __name__ == "__main__":
    unittest.main()

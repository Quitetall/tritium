"""Public PTQ regression guard for native row-level parallel fitting."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import unittest


_PROBE = r"""
import hashlib
import json
import tempfile
import time

import torch
from tritium.torch import TernaryConfig, calibrate, convert, prepare

torch.manual_seed(31)
model = torch.nn.Linear(64, 8192, bias=False)
prepared = prepare(
    model,
    TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
    inplace=False,
)
with tempfile.TemporaryDirectory(prefix="tritium-ptq-parallelism-") as root:
    evidence = calibrate(
        prepared,
        [torch.randn(1, 64)],
        evidence_dir=f"{root}/evidence",
    )
    started = time.perf_counter()
    artifact = convert(
        prepared,
        evidence,
        work_dir=f"{root}/conversion",
        max_working_bytes=256 * 1024 * 1024,
    )
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
    print(json.dumps({
        "algorithm_id": artifact.algorithm_id,
        "benchmark_environment": benchmark_environment,
        "elapsed_seconds": elapsed,
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
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=90,
        )
        return json.loads(completed.stdout)

    def test_public_convert_parallelizes_rows_without_changing_fitted_artifact(self):
        serial = self._convert(1)
        parallel = self._convert(4)

        self.assertEqual(parallel["algorithm_id"], serial["algorithm_id"])
        self.assertEqual(parallel["fit_digest"], serial["fit_digest"])
        self.assertEqual(parallel["weighted_mse"], serial["weighted_mse"])
        speedup = float(serial["elapsed_seconds"]) / float(
            parallel["elapsed_seconds"]
        )
        print(
            "public PTQ convert rows=8192 columns=64 "
            f"serial_seconds={serial['elapsed_seconds']:.3f} "
            f"parallel_seconds={parallel['elapsed_seconds']:.3f} "
            f"speedup={speedup:.2f}x "
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

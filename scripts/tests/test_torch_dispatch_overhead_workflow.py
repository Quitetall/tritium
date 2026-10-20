from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TorchDispatchOverheadWorkflowTests(unittest.TestCase):
    def test_cpu_dispatch_trace_is_measured_sealed_and_uploaded(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(
            encoding="utf-8"
        )
        start = workflow.index("  torch-functional-smoke:")
        match = re.search(r"(?m)^  [a-z0-9_-]+:\s*$", workflow[start + 3 :])
        self.assertIsNotNone(match)
        assert match is not None
        job = workflow[start : start + 3 + match.start()]

        self.assertIn("RAYON_NUM_THREADS: \"1\"", job)
        self.assertIn("OMP_NUM_THREADS: \"1\"", job)
        self.assertIn("MKL_NUM_THREADS: \"1\"", job)
        self.assertIn("python -I -B -m tritium.torch.qualify_dispatch_overhead", job)
        self.assertIn('cd "$RUNNER_TEMP"', job)
        self.assertIn("scripts/qualify-torch-dispatch-overhead.py", job)
        self.assertIn(
            'trace="$GITHUB_WORKSPACE/evidence/torch-dispatch-overhead-trace.json"',
            job,
        )
        self.assertIn('--trace "$trace"', job)
        self.assertIn("--output-dir evidence/torch-dispatch-overhead", job)
        self.assertIn("evidence/torch-dispatch-overhead/**", job)
        self.assertIn("evidence/torch-dispatch-overhead-trace.json", job)

    def test_pull_request_filter_covers_dispatch_measurement_and_sealing(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(
            encoding="utf-8"
        )
        for path in (
            "crates/tritium-py/python/tritium/torch/qualify_dispatch_overhead.py",
            "scripts/qualify-torch-dispatch-overhead.py",
            "scripts/verify-torch-dispatch-overhead-receipt.py",
            "scripts/tests/test_torch_dispatch_overhead_workflow.py",
        ):
            self.assertIn(f'      - "{path}"', workflow)


if __name__ == "__main__":
    unittest.main()

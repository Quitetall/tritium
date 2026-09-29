from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/generate-serve-dashboard.py"
SPEC = importlib.util.spec_from_file_location("generate_serve_dashboard", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ServeDashboardTests(unittest.TestCase):
    def test_registry_metrics_are_in_runtime_exposition(self):
        registry = MODULE.load_registry()
        MODULE.validate_runtime_exposition(
            registry, ROOT / "crates/tritium-serve/src/router.rs"
        )

    def test_rendered_dashboard_is_current_and_every_query_is_registered(self):
        registry = MODULE.load_registry()
        dashboard = json.loads(MODULE.render(registry))
        self.assertEqual(len(dashboard["panels"]), len(registry["metrics"]))
        metric_names = {metric["name"] for metric in registry["metrics"]}
        for panel in dashboard["panels"]:
            expression = panel["targets"][0]["expr"]
            references = MODULE.METRIC_REF.findall(expression)
            canonical = {
                ref.removesuffix("_bucket") if ref.endswith("_bucket") else ref
                for ref in references
            }
            self.assertTrue(canonical)
            self.assertLessEqual(canonical, metric_names)
        self.assertEqual(
            MODULE.DASHBOARD.read_text(encoding="utf-8"), MODULE.render(registry)
        )


if __name__ == "__main__":
    unittest.main()

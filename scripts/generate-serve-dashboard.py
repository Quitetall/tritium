#!/usr/bin/env python3
"""Render and drift-check the Grafana serving dashboard from its metric registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "deploy/grafana/metric-registry.json"
DASHBOARD = ROOT / "deploy/grafana/tritium-serving-dashboard.json"
SCHEMA = "tritium.serve.dashboard-metrics.v1"
METRIC_REF = re.compile(r"\b(tritium_[a-zA-Z0-9_]+)\b")
TOP_LEVEL_FIELDS = {"schema", "title", "refresh", "metrics"}
METRIC_FIELDS = {"name", "type", "unit", "help", "panel"}
PANEL_FIELDS = {"title", "description", "expr", "legend"}
METRIC_TYPES = {"counter", "gauge", "histogram"}
MAX_REGISTRY_BYTES = 64 * 1024


class DashboardError(ValueError):
    """The dashboard metric registry is invalid or has drifted."""


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise DashboardError(f"{label} fields do not match the frozen registry schema")
    return value


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_REGISTRY_BYTES:
        raise DashboardError("metric registry must be a bounded ordinary file")
    try:
        registry = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DashboardError("metric registry must contain UTF-8 JSON") from error
    _object(registry, TOP_LEVEL_FIELDS, "registry")
    if registry["schema"] != SCHEMA:
        raise DashboardError("unsupported metric registry schema")
    if not isinstance(registry["title"], str) or not registry["title"]:
        raise DashboardError("registry.title must be non-empty")
    if not isinstance(registry["refresh"], str) or not registry["refresh"]:
        raise DashboardError("registry.refresh must be non-empty")
    metrics = registry["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise DashboardError("registry.metrics must be a non-empty array")
    known_names: set[str] = set()
    for index, raw in enumerate(metrics):
        metric = _object(raw, METRIC_FIELDS, f"metrics[{index}]")
        name = metric["name"]
        if not isinstance(name, str) or re.fullmatch(r"tritium_[a-z0-9_]+", name) is None:
            raise DashboardError(f"metrics[{index}].name is malformed")
        if name in known_names:
            raise DashboardError(f"duplicate metric {name}")
        known_names.add(name)
    for index, raw in enumerate(metrics):
        metric = _object(raw, METRIC_FIELDS, f"metrics[{index}]")
        name = metric["name"]
        if metric["type"] not in METRIC_TYPES:
            raise DashboardError(f"metrics[{index}].type is invalid")
        for field in ("unit", "help"):
            if not isinstance(metric[field], str) or not metric[field]:
                raise DashboardError(f"metrics[{index}].{field} must be non-empty")
        panel = _object(metric["panel"], PANEL_FIELDS, f"metrics[{index}].panel")
        for field in PANEL_FIELDS:
            if not isinstance(panel[field], str) or not panel[field]:
                raise DashboardError(f"metrics[{index}].panel.{field} must be non-empty")
        references = set(METRIC_REF.findall(panel["expr"]))
        canonical_references = {
            ref.removesuffix("_bucket") if ref.endswith("_bucket") else ref
            for ref in references
        }
        if not canonical_references or not canonical_references <= known_names:
            raise DashboardError(
                f"metrics[{index}] panel references an unregistered metric"
            )
        if metric["type"] == "histogram" and not any(
            ref.endswith("_bucket") for ref in references
        ):
            raise DashboardError(f"histogram panel for {name} must use its bucket series")
    return registry


def validate_runtime_exposition(registry: dict[str, Any], router_source: Path) -> None:
    try:
        source = router_source.read_text(encoding="utf-8")
    except OSError as error:
        raise DashboardError(f"cannot read serving metrics source: {router_source}") from error
    exposed = set(re.findall(r"# HELP (tritium_[a-z0-9_]+)\b", source))
    runtime_types = dict(
        re.findall(r"# TYPE (tritium_[a-z0-9_]+) (counter|gauge|histogram)\b", source)
    )
    exposed.update(
        re.findall(r'render_histogram\(\s*"(tritium_[a-z0-9_]+)"', source)
    )
    runtime_types.update(
        {
            name: "histogram"
            for name in re.findall(
                r'render_histogram\(\s*"(tritium_[a-z0-9_]+)"', source
            )
        }
    )
    absent = sorted(metric["name"] for metric in registry["metrics"] if metric["name"] not in exposed)
    if absent:
        raise DashboardError("registry metrics are absent from /metrics: " + ", ".join(absent))
    mismatches = sorted(
        f"{metric['name']} ({metric['type']} != {runtime_types.get(metric['name'], 'untyped')})"
        for metric in registry["metrics"]
        if metric["type"] != runtime_types.get(metric["name"])
    )
    if mismatches:
        raise DashboardError("registry metric types differ from /metrics: " + ", ".join(mismatches))


def render(registry: dict[str, Any]) -> str:
    panels = []
    for index, metric in enumerate(registry["metrics"], start=1):
        definition = metric["panel"]
        panels.append(
            {
                "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                "description": definition["description"],
                "fieldConfig": {
                    "defaults": {"unit": metric["unit"]},
                    "overrides": [],
                },
                "gridPos": {"h": 8, "w": 12, "x": ((index - 1) % 2) * 12, "y": ((index - 1) // 2) * 8},
                "id": index,
                "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi", "sort": "desc"}},
                "targets": [
                    {
                        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                        "editorMode": "code",
                        "expr": definition["expr"],
                        "legendFormat": definition["legend"],
                        "refId": "A",
                    }
                ],
                "title": definition["title"],
                "type": "timeseries",
            }
        )
    dashboard = {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "liveNow": False,
        "panels": panels,
        "refresh": registry["refresh"],
        "schemaVersion": 39,
        "tags": ["tritium", "serving"],
        "templating": {"list": [{"name": "DS_PROMETHEUS", "type": "datasource", "query": "prometheus", "current": {}, "hide": 0, "includeAll": False, "label": "Prometheus", "multi": False, "options": [], "refresh": 1, "regex": "", "skipUrlSync": False}]},
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": registry["title"],
        "uid": "tritium-serving",
        "version": 1,
        "weekStart": "",
    }
    return json.dumps(dashboard, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if generated dashboard is stale")
    mode.add_argument("--write", action="store_true", help="write the generated dashboard (default)")
    args = parser.parse_args()
    try:
        registry = load_registry()
        validate_runtime_exposition(registry, ROOT / "crates/tritium-serve/src/router.rs")
        rendered = render(registry)
        if args.check:
            if not DASHBOARD.is_file() or DASHBOARD.read_text(encoding="utf-8") != rendered:
                raise DashboardError("Grafana dashboard is stale; run scripts/generate-serve-dashboard.py")
        else:
            DASHBOARD.write_text(rendered, encoding="utf-8")
    except DashboardError as error:
        parser.exit(1, f"generate-serve-dashboard: ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

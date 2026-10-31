"""Explicit experimental inspect-then-run QTEA file campaigns.

Resource estimates and imported metrics are caller assertions, not measurements
or capture authentication. Outputs remain unqualified, residual-bearing QTEA.
No model download, identity fallback, automatic search or trust promotion occurs.
"""

import json
import os

from ._tritium import _inspect_qtea_file_campaign, _run_qtea_file_campaign

__all__ = ["inspect", "run"]


def _json(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _files(source_shards, asset_paths, curvature_paths):
    return _json({
        "source_shards": [os.fspath(p) for p in source_shards],
        "assets": {n: os.fspath(p) for n, p in asset_paths.items()},
        "curvature": {n: os.fspath(p) for n, p in curvature_paths.items()},
    })


def inspect(request, *, source_shards, asset_paths, curvature_paths,
            max_json_bytes=16_777_216, max_report_bytes=16_777_216):
    """Read actual inputs and return unqualified selection data without fitting.

    Supply every original shard/asset and one explicitly typed curvature file per
    matrix. The request includes complete recipes, caps and caller estimates.
    """
    return json.loads(_inspect_qtea_file_campaign(
        _json(request), _files(source_shards, asset_paths, curvature_paths),
        max_json_bytes=max_json_bytes, max_report_bytes=max_report_bytes))


def run(request, inspection, *, source_shards, asset_paths, curvature_paths,
        spool_path, output_dir, max_json_bytes=16_777_216,
        max_report_bytes=16_777_216):
    """Revalidate selection, fit and publish into a new candidate directory.

    The new private spool is removed after execution. Caller owns exclusive,
    immutable path custody. Post-publication report/output errors do not roll back
    a committed model; candidate.json and model.trit bind its unqualified identity.
    A successful run does not qualify quality, hardware, source, MTP or release.
    """
    return json.loads(_run_qtea_file_campaign(
        _json(request), _files(source_shards, asset_paths, curvature_paths),
        _json(inspection), os.fspath(spool_path), os.fspath(output_dir),
        max_json_bytes=max_json_bytes, max_report_bytes=max_report_bytes))

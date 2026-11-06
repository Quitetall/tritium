"""Actual owner-window caches to the shared model engine and native loader.

Scalar-window scores are not relabeled K-FAC/Fisher or qualified recipes. This
adapter retains current model-source snapshot and native cache package overlaps.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile

from .. import _tritium
from .native_calibration import NativeModulePreparation, _encoded
from .native_source import _inventory, _run_native_source
from .owner_capture import NativeOwnerWindowCapture


def _validate_owner_options(
    window_weights, *, damping,
    metric_limits, max_row_bindings, max_total_cache_bytes, max_cache_package_bytes,
    max_plan_bytes=16_777_216,
):
    """Shared frontend admission, not a replacement for Rust plan validation."""
    fields = {"max_windows", "max_payload_bytes", "max_row_bytes", "max_reduction_entries"}
    if (not isinstance(metric_limits, dict) or set(metric_limits) != fields
            or any(type(v) is not int or v <= 0 for v in metric_limits.values())
            or any(type(v) is not int or v <= 0 for v in
                   (max_row_bindings, max_total_cache_bytes, max_cache_package_bytes, max_plan_bytes))):
        raise ValueError("owner fit limits must be explicit positive integers")
    if (not isinstance(window_weights, (list, tuple)) or not window_weights
            or len(window_weights) > metric_limits["max_windows"]):
        raise ValueError("owner window inventory exceeds limit or differs from weights")
    if (any(isinstance(w, bool) or not isinstance(w, (int, float))
            or not math.isfinite(w) or w <= 0 for w in window_weights)
            or isinstance(damping, bool) or not isinstance(damping, (int, float))
            or not math.isfinite(damping) or damping < 0):
        raise ValueError("owner weights and damping must be finite and admitted")
    return tuple(float(weight) for weight in window_weights)


def _owner_caches(source, captures, window_weights, max_total_cache_bytes):
    if not isinstance(source, NativeModulePreparation):
        raise TypeError("actual native preparation required")
    if (not isinstance(captures, (list, tuple)) or not captures
            or len(captures) != len(window_weights)):
        raise ValueError("owner window inventory exceeds limit or differs from weights")
    names = sorted(owner for owner, _, _, _, selected, _ in _inventory(source.prepared) if selected)
    caches, weights = [], []
    total = 0
    for capture, weight in zip(captures, window_weights):
        if not isinstance(capture, NativeOwnerWindowCapture) or not capture.caches:
            raise TypeError("actual durable owner captures required")
        if sorted(cache.tensor_name for cache in capture.caches) != names:
            raise ValueError("owner cache inventory differs from selected source")
        for cache in capture.caches:
            source.observation.validate_owner_cache(cache)
            total += cache.payload_bytes
            if total > max_total_cache_bytes:
                raise ValueError("aggregate owner cache payload exceeds byte limit")
            caches.append(cache)
            weights.append(float(weight))
    return caches, weights


def convert_owner_native_module(
    source, captures, output_dir, run_id, *, window_weights, damping,
    metric_limits, max_row_bindings, max_total_cache_bytes, max_cache_package_bytes,
    max_plan_bytes=16_777_216,
):
    """Fit selected owners with ordered scalar windows through the Rust engine.

    Bounded captures and weights have matching order. Repeating a capture repeats
    its observation, not its numerical storage. This is not a qualified recipe.
    """
    window_weights = _validate_owner_options(
        window_weights, damping=damping, metric_limits=metric_limits,
        max_row_bindings=max_row_bindings, max_total_cache_bytes=max_total_cache_bytes,
        max_cache_package_bytes=max_cache_package_bytes, max_plan_bytes=max_plan_bytes,
    )
    caches, weights = _owner_caches(source, captures, window_weights, max_total_cache_bytes)
    source.verify()
    options = json.loads(source.options_json)

    def consume(plan, shards, output, run, *, asset_paths, input_grams):
        if input_grams:
            raise ValueError("owner calibration cannot reinterpret raw Grams")
        request = source.observation.owner_plan_json(
            plan, caches, weights, float(damping), _encoded(metric_limits),
            max_row_bindings, max_total_cache_bytes, max_cache_package_bytes,
            max_plan_bytes=max_plan_bytes,
        )
        temporary = tempfile.TemporaryDirectory(prefix="tritium-owner-fit-cache-", dir=options["work_dir"])
        result = None
        try:
            paths = {}
            for cache in caches:
                files = paths.setdefault(cache.tensor_name, {})
                if cache.cache_id in files:
                    continue
                path = Path(temporary.name) / f"cache-{len(paths)}-{cache.cache_id}.trit"
                path.write_bytes(cache.package(max_cache_package_bytes))
                cache.verify_file(str(path), max_cache_package_bytes)
                files[cache.cache_id] = str(path)
            result = _tritium.convert_owner_model_plan(
                request, shards, output, run, asset_paths=asset_paths,
                owner_cache_paths=paths, max_plan_bytes=max_plan_bytes,
            )
        finally:
            try:
                temporary.cleanup()
            except OSError as error:
                if result is None:
                    raise
                result = result._with_source_cleanup_warning(f"OwnerCacheCleanup: {error}")
        return result

    return _run_native_source(source.prepared, output_dir, run_id, consume=consume, **options)


__all__ = ["convert_owner_native_module"]

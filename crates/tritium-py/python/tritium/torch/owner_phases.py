"""Explicit owner calibration phases over existing Rust caches and model engine.

These Python carriers freeze orchestration arguments, not a new wire schema,
estimator identity or qualification receipt. Native cache/plan validation remains
authoritative. Arbitrary iterators are not consumed implicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from .native_calibration import NativeModulePreparation, _encoded
from .owner_capture import _cache_limits
from .owner_conversion import _owner_caches, _validate_owner_options, convert_owner_native_module


@dataclass(frozen=True, init=False)
class OwnerCalibrationRecipe:
    """Explicit research recipe; returned option dictionaries are detached copies.

    ``capture_options`` are the existing capture_owner_window limit arguments,
    including cache_limits and max_total_cache_bytes. ``fit_options`` are the
    existing convert_owner_native_module arguments except window_weights.
    Objective code is supplied to calibrate separately; declared objective bytes
    do not prove independent execution or immutable callback custody.
    """
    objective_bytes: bytes
    window_weights: tuple
    _capture_json: str
    _fit_json: str

    def __init__(self, *, objective_bytes, window_weights, capture_options, fit_options):
        capture_fields = {
            "max_observation_bytes", "max_context_bytes", "max_invocations",
            "max_samples", "max_graph_nodes", "cache_limits", "max_total_cache_bytes",
        }
        fit_fields = {
            "damping", "metric_limits", "max_row_bindings", "max_total_cache_bytes",
            "max_cache_package_bytes",
        }
        if not isinstance(capture_options, dict) or set(capture_options) != capture_fields:
            raise ValueError("owner capture options require the exact explicit limit fields")
        if (not isinstance(fit_options, dict)
                or set(fit_options) - {"max_plan_bytes"} != fit_fields):
            raise ValueError("owner fit options require the exact explicit limit fields")
        capture = json.loads(_encoded(capture_options))
        fit = json.loads(_encoded(fit_options))
        fit.setdefault("max_plan_bytes", 16_777_216)
        _cache_limits(capture["cache_limits"], capture["max_total_cache_bytes"])
        if any(type(capture[key]) is not int or capture[key] <= 0
               for key in capture_fields - {"cache_limits"}):
            raise ValueError("owner capture limits must be positive integers")
        if (not isinstance(objective_bytes, bytes) or not objective_bytes
                or len(objective_bytes) > capture["max_context_bytes"]):
            raise ValueError("bounded nonempty objective bytes required")
        weights = _validate_owner_options(window_weights, **fit)
        object.__setattr__(self, "objective_bytes", objective_bytes)
        object.__setattr__(self, "window_weights", weights)
        object.__setattr__(self, "_capture_json", _encoded(capture))
        object.__setattr__(self, "_fit_json", _encoded(fit))

    @property
    def capture_options(self):
        return json.loads(self._capture_json)

    @property
    def fit_options(self):
        return json.loads(self._fit_json)


@dataclass(frozen=True)
class NativeOwnerCalibration:
    """Ordered actual native window caches, not legacy calibration evidence.

    Individual captures can be saved/reopened through their existing native
    cache interface. This runtime carrier introduces no aggregate disk format.
    """
    source: NativeModulePreparation
    captures: tuple
    recipe: OwnerCalibrationRecipe

    def __post_init__(self):
        if not isinstance(self.recipe, OwnerCalibrationRecipe):
            raise TypeError("actual owner calibration recipe required")
        _owner_caches(self.source, self.captures, self.recipe.window_weights,
                      self.recipe.fit_options["max_total_cache_bytes"])
        object.__setattr__(self, "captures", tuple(self.captures))

    def verify(self):
        if not isinstance(self.recipe, OwnerCalibrationRecipe):
            raise TypeError("actual owner calibration recipe required")
        _owner_caches(self.source, self.captures, self.recipe.window_weights,
                      self.recipe.fit_options["max_total_cache_bytes"])
        return self.source.verify()


def calibrate_owner_module(source, windows, objective, recipe):
    if not isinstance(source, NativeModulePreparation):
        raise TypeError("actual native preparation required")
    if not isinstance(recipe, OwnerCalibrationRecipe) or not callable(objective):
        raise TypeError("owner calibration requires an explicit recipe and objective")
    if (not isinstance(windows, (list, tuple))
            or len(windows) != len(recipe.window_weights)):
        raise ValueError("owner calibration requires bounded windows matching recipe weights")
    windows = tuple(windows)
    source.verify()
    captures, total = [], 0
    maximum = recipe.fit_options["max_total_cache_bytes"]
    for window in windows:
        options = recipe.capture_options
        options["max_total_cache_bytes"] = min(options["max_total_cache_bytes"], maximum - total)
        capture = source.capture_owner_window(
            window, objective, objective_bytes=recipe.objective_bytes, **options,
        )
        total += sum(cache.payload_bytes for cache in capture.caches)
        captures.append(capture)
    result = NativeOwnerCalibration(source, tuple(captures), recipe)
    result.verify()
    return result


def convert_owner_calibration(source, calibration, output_dir, run_id):
    if not isinstance(calibration, NativeOwnerCalibration) or calibration.source is not source:
        raise TypeError("owner convert requires calibration from this native preparation")
    if output_dir is None or not isinstance(run_id, str) or not run_id:
        raise ValueError("owner convert requires output_dir and a nonempty run_id")
    calibration.verify()
    return convert_owner_native_module(
        source, calibration.captures, output_dir, run_id,
        window_weights=calibration.recipe.window_weights, **calibration.recipe.fit_options,
    )


__all__ = ["OwnerCalibrationRecipe", "NativeOwnerCalibration"]

"""Whole-Qwen ONNX export and load façade.

The public boundary accepts deployable hard artifacts only. Latent QAT masters,
optimizer state, and training graphs remain a v1.3 feature.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

from .. import _tritium
from .artifacts import QuantizationResult
from .conversion import PreparedModel
from .errors import TritiumError

_MANIFEST_FILE = "tritium-onnx-manifest.json"
_TOP_FIELDS = {"schema", "language", "mtp", "weights", "identity", "conversion"}
_GRAPH_FIELDS = {"file", "blake3"}
_WEIGHT_FIELDS = {"file", "blake3", "bytes"}
_IDENTITY_FIELDS = {
    "source_model_id",
    "tokenizer_id",
    "recipe_id",
    "package_id",
    "converted_coverage_id",
    "deferred_coverage_id",
}
_CONVERSION_FIELDS = {
    "mode",
    "completion_id",
    "campaign_id",
    "admission_id",
    "selection_id",
}


@dataclass(frozen=True)
class OnnxBundleManifest:
    """Strict, content-bound metadata for one language-plus-MTP ONNX bundle."""

    directory: Path
    language_blake3: str
    mtp_blake3: str
    weights_blake3: str
    weights_bytes: int
    source_model_id: str
    tokenizer_id: str
    recipe_id: str
    package_id: str
    converted_coverage_id: str
    deferred_coverage_id: str
    conversion_mode: str
    completion_id: str
    campaign_id: str
    admission_id: str
    selection_id: str


def export_onnx(
    source: Any,
    output_dir: Union[os.PathLike[str], str],
    *,
    profile: str = "compact-v1",
    tokens: int = 1,
    past_tokens: int = 0,
    max_package_bytes: int = 32 * 1024 * 1024 * 1024,
    max_preserved_bytes: int = 8 * 1024 * 1024 * 1024,
    max_salt_resident_bytes: int = 32 * 1024 * 1024 * 1024,
    max_preserved_fp32_bytes: int = 8 * 1024 * 1024 * 1024,
) -> Any:
    """Export one authenticated PTQ result as language, MTP, and shared weights.

    QAT graphs with latent floating masters and refined lineages are not
    silently reinterpreted as PTQ. Their distinct producers must land before
    this function accepts them.
    """
    if isinstance(source, PreparedModel):
        raise TritiumError(
            "trainable ONNX export requires Tritium v1.3",
            code="trainable_onnx_requires_v1_3",
            stage="export_onnx",
        )
    if not isinstance(source, QuantizationResult):
        raise TypeError("export_onnx requires a QuantizationResult")
    if source.schema_version != 3 or source.preserved is None or not source.hf_assets:
        raise TritiumError(
            "ONNX export requires a complete schema-v3 language-plus-MTP bundle",
            code="incomplete_artifact",
            stage="export_onnx",
            details={"schema_version": source.schema_version},
        )
    source.artifact(profile)
    if type(tokens) is not int or tokens <= 0 or type(past_tokens) is not int or past_tokens < 0:
        raise ValueError("tokens must be positive and past_tokens must be nonnegative")
    limits = {
        "max_package_bytes": max_package_bytes,
        "max_preserved_bytes": max_preserved_bytes,
        "max_salt_resident_bytes": max_salt_resident_bytes,
        "max_preserved_fp32_bytes": max_preserved_fp32_bytes,
    }
    if any(type(value) is not int or value <= 0 for value in limits.values()):
        raise ValueError("ONNX export byte ceilings must be positive integers")
    native = getattr(_tritium, "export_qwen35_onnx_bundle", None)
    if native is None:
        raise TritiumError(
            "this wheel does not contain the Qwen ONNX exporter",
            code="onnx_export_unavailable",
            stage="export_onnx",
        )
    return native(
        str(source.artifact_dir),
        str(output_dir),
        profile=profile,
        tokens=tokens,
        past_tokens=past_tokens,
        **limits,
    )


def load_onnx(
    artifact_dir: Union[os.PathLike[str], str], *, device: str = "cpu"
) -> Any:
    """Strictly admit a published bundle and create its native ORT generation runtime."""
    manifest = _read_manifest(Path(artifact_dir))
    native = getattr(_tritium, "QwenOnnxModel", None)
    if native is None:
        raise TritiumError(
            "this wheel does not contain the Qwen ONNX Runtime session",
            code="onnx_runtime_unavailable",
            stage="load_onnx",
            details={"artifact_dir": str(manifest.directory)},
        )
    return native.load(str(manifest.directory), device=device)


def _pairs_without_duplicates(pairs):
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate ONNX manifest field {key!r}")
        value[key] = item
    return value


def _read_manifest(requested: Path) -> OnnxBundleManifest:
    if requested.is_symlink():
        raise ValueError("ONNX artifact must be an ordinary directory")
    directory = requested.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("ONNX artifact must be an ordinary directory")
    path = directory / _MANIFEST_FILE
    value = _read_json_regular(path, max_bytes=1024 * 1024, label="ONNX manifest")
    if not isinstance(value, dict) or set(value) != _TOP_FIELDS:
        raise ValueError("ONNX manifest top-level fields differ from schema v1")
    if value["schema"] != "tritium-qwen35-onnx-bundle-v1":
        raise ValueError("unsupported ONNX bundle schema")
    language = _graph(value["language"], "language.onnx", "language")
    mtp = _graph(value["mtp"], "mtp.onnx", "MTP")
    weights = value["weights"]
    if not isinstance(weights, dict) or set(weights) != _WEIGHT_FIELDS:
        raise ValueError("ONNX weights manifest fields differ from schema v1")
    if weights["file"] != "weights.bin":
        raise ValueError("ONNX weights filename must be canonical")
    weights_digest = _digest(weights["blake3"], "weights")
    if type(weights["bytes"]) is not int or weights["bytes"] <= 0:
        raise ValueError("ONNX weights bytes must be a positive integer")
    identity = _strings(value["identity"], _IDENTITY_FIELDS, "identity")
    conversion = _strings(value["conversion"], _CONVERSION_FIELDS, "conversion")
    if conversion["mode"] not in {"qat-hard", "ptq", "refined"}:
        raise ValueError("unsupported ONNX conversion mode")
    # This descriptor preflight gives deterministic Python errors. The native
    # runtime is still the security authority: it reopens and authenticates all
    # three files immediately before creating the ORT sessions.
    _inspect_regular(
        directory / "language.onnx", "language.onnx", max_bytes=256 * 1024 * 1024
    )
    _inspect_regular(
        directory / "mtp.onnx", "mtp.onnx", max_bytes=256 * 1024 * 1024
    )
    _inspect_regular(
        directory / "weights.bin",
        "weights.bin",
        max_bytes=64 * 1024 * 1024 * 1024,
        expected_bytes=weights["bytes"],
    )
    return OnnxBundleManifest(
        directory=directory,
        language_blake3=language,
        mtp_blake3=mtp,
        weights_blake3=weights_digest,
        weights_bytes=weights["bytes"],
        source_model_id=identity["source_model_id"],
        tokenizer_id=identity["tokenizer_id"],
        recipe_id=identity["recipe_id"],
        package_id=identity["package_id"],
        converted_coverage_id=identity["converted_coverage_id"],
        deferred_coverage_id=identity["deferred_coverage_id"],
        conversion_mode=conversion["mode"],
        completion_id=conversion["completion_id"],
        campaign_id=conversion["campaign_id"],
        admission_id=conversion["admission_id"],
        selection_id=conversion["selection_id"],
    )


def _graph(value: Any, filename: str, label: str) -> str:
    if not isinstance(value, dict) or set(value) != _GRAPH_FIELDS:
        raise ValueError(f"ONNX {label} manifest fields differ from schema v1")
    if value["file"] != filename:
        raise ValueError(f"ONNX {label} filename must be canonical")
    return _digest(value["blake3"], label)


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ONNX {label} BLAKE3 must be 64 lowercase hexadecimal characters")
    return value


def _strings(value: Any, fields: set[str], label: str) -> Dict[str, str]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"ONNX {label} fields differ from schema v1")
    if any(not isinstance(item, str) or not item for item in value.values()):
        raise ValueError(f"ONNX {label} values must be non-empty strings")
    return value


def _read_json_regular(path: Path, *, max_bytes: int, label: str) -> Any:
    descriptor, before, opened = _open_regular(path, label, max_bytes=max_bytes)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            value = json.load(stream, object_pairs_hook=_pairs_without_duplicates)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _require_unchanged(path, before, opened, label)
    return value


def _inspect_regular(
    path: Path,
    label: str,
    *,
    max_bytes: int | None = None,
    expected_bytes: int | None = None,
) -> None:
    descriptor, before, opened = _open_regular(
        path,
        label,
        max_bytes=max_bytes,
        expected_bytes=expected_bytes,
    )
    os.close(descriptor)
    _require_unchanged(path, before, opened, label)


def _open_regular(
    path: Path,
    label: str,
    *,
    max_bytes: int | None = None,
    expected_bytes: int | None = None,
):
    before = path.lstat()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} cannot be opened as an ordinary file") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or not _same_file(before, opened)
            or opened.st_size <= 0
            or (max_bytes is not None and opened.st_size > max_bytes)
            or (expected_bytes is not None and opened.st_size != expected_bytes)
        ):
            raise ValueError(f"{label} type, identity, or length is invalid")
        return descriptor, before, opened
    except Exception:
        os.close(descriptor)
        raise


def _require_unchanged(
    path: Path, before: os.stat_result, opened: os.stat_result, label: str
) -> None:
    after = path.lstat()
    if (
        not _same_file(before, after)
        or before.st_size != opened.st_size
        or after.st_size != opened.st_size
    ):
        raise ValueError(f"{label} changed while being inspected")


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


__all__ = ["OnnxBundleManifest", "export_onnx", "load_onnx"]

"""Cross-language canonical manifest gates for plan 0049."""

import json
from pathlib import Path

import pytest

from tritium.portable import (  # noqa: E402
    TrainingManifestError,
    canonical_training_manifest_json,
    parse_training_manifest,
)


ROOT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "spec" / "training" / "v1" / "manifest.json"
)


def test_python_canonical_bytes_equal_language_neutral_fixture():
    expected = ROOT_MANIFEST.read_bytes()
    assert canonical_training_manifest_json() == expected
    parsed = parse_training_manifest(expected)
    assert parsed.schema_version == 1
    assert len(parsed.operations) == 35


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_version=2),
        lambda value: value.update(dtype="f16"),
        lambda value: value.update(extra=True),
        lambda value: value["operations"].pop(),
        lambda value: value["operations"][0].update(forward=False),
        lambda value: value["operations"].__setitem__(1, value["operations"][0]),
        lambda value: value["operations"].__setitem__(
            slice(0, 2), reversed(value["operations"][0:2])
        ),
    ],
)
def test_python_parser_rejects_schema_and_registry_drift(mutate):
    value = json.loads(ROOT_MANIFEST.read_bytes())
    mutate(value)
    with pytest.raises(TrainingManifestError):
        parse_training_manifest(json.dumps(value))


def test_python_parser_rejects_duplicate_fields_wrong_types_and_bad_utf8():
    with pytest.raises(TrainingManifestError, match="duplicate"):
        parse_training_manifest('{"schema_id":"x","schema_id":"y"}')
    value = json.loads(ROOT_MANIFEST.read_bytes())
    value["schema_version"] = True
    with pytest.raises(TrainingManifestError, match="integer"):
        parse_training_manifest(json.dumps(value))
    with pytest.raises(TrainingManifestError, match="UTF-8"):
        parse_training_manifest(b"\xff")

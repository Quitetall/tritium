"""Ordinary frontend refusals; actual shared native-fit corpus is replayed separately."""
import math

import pytest
from tritium import qtea


@pytest.mark.parametrize("request_data", [{}, {"schema": "unknown"}])
def test_invalid_request_refuses_before_file_work(request_data, tmp_path):
    files = dict(source_shards=[tmp_path / "absent"], asset_paths={}, curvature_paths={})
    with pytest.raises(ValueError):
        qtea.inspect(request_data, **files)
    with pytest.raises(ValueError):
        qtea.run(request_data, {}, **files, spool_path=tmp_path / "spool",
                 output_dir=tmp_path / "candidate")
    assert not (tmp_path / "spool").exists()
    assert not (tmp_path / "candidate").exists()


def test_nonfinite_python_recipe_is_not_serialized_as_null(tmp_path):
    with pytest.raises(ValueError, match="JSON compliant"):
        qtea.inspect({"recipe": math.nan}, source_shards=[], asset_paths={},
                     curvature_paths={})


def test_positive_json_bound_is_required_before_input_work(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        qtea.inspect({}, source_shards=[tmp_path / "absent"], asset_paths={},
                     curvature_paths={}, max_json_bytes=0)

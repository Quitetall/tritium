from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tritium.nn import AdditiveTernaryWeight  # noqa: E402
from tritium.torch import SMOLLM2_MODEL_ID, SMOLLM2_REVISION  # noqa: E402
from tritium.torch.tutorial import _trit_diagnostics, run_smollm2_release_demo  # noqa: E402


def test_tutorial_source_is_immutable_and_public():
    assert SMOLLM2_MODEL_ID == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert SMOLLM2_REVISION == "12fd25f77366fa6b3b4b768ec3050bf629380bac"


def test_tutorial_rejects_bad_arguments_before_creating_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="device"):
        run_smollm2_release_demo(output, device="magic")
    with pytest.raises(ValueError, match="max_seconds"):
        run_smollm2_release_demo(output, max_seconds=0)
    with pytest.raises(ValueError, match="revision"):
        run_smollm2_release_demo(output, revision="main")
    assert not output.exists()


def test_tutorial_trit_diagnostics_count_planes_without_dense_float_shadow():
    trits = torch.tensor([[-1, 0, 1, 0, 1]], dtype=torch.int8)
    plane = SimpleNamespace(
        trits=trits,
        scales=torch.ones((1, 1), dtype=torch.float16),
        group_size=5,
    )
    model = torch.nn.Sequential(AdditiveTernaryWeight((plane,)))

    assert _trit_diagnostics(model) == {
        "negative": 1,
        "zero": 2,
        "positive": 2,
        "zero_rate": 0.4,
        "planes": 1,
    }

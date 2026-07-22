"""Phased public lifecycle gates from ADR 0033 / plan 0047."""

import pytest

torch = pytest.importorskip("torch")

from tritium.nn import TernaryLinear  # noqa: E402
from tritium.torch import (  # noqa: E402
    RefinementConfig,
    TernaryConfig,
    TritiumError,
    inspect,
    prepare,
    prepare_qat,
)


def test_ptq_recipe_and_refinement_are_distinct_versioned_schemas():
    ptq = TernaryConfig.ptq(profile="compact-v1", target_bpw=2.0)
    assert "refinement" not in ptq.to_dict()
    assert TernaryConfig.from_dict(ptq.to_dict()) == ptq

    scale_only = RefinementConfig.scale_only()
    hard_pv = RefinementConfig.hard_pv(structure="s34")
    assert RefinementConfig.from_dict(scale_only.to_dict()) == scale_only
    assert RefinementConfig.from_dict(hard_pv.to_dict()) == hard_pv
    assert scale_only.kind == "scale-only"
    assert hard_pv.kind == "hard-pv"
    assert (
        RefinementConfig.from_dict(
            {"schema_version": 1, "kind": "scale-only", "structure": "dense"}
        )
        == RefinementConfig.scale_only()
    )
    with pytest.raises(ValueError, match="legacy RefinementConfig"):
        RefinementConfig.from_dict(
            {"schema_version": 1, "kind": "unknown", "structure": "dense"}
        )
    with pytest.raises(ValueError, match="legacy RefinementConfig"):
        RefinementConfig.from_dict(
            {"schema_version": 1, "kind": "scale-only", "structure": "s34"}
        )
    with pytest.raises(ValueError, match="iteration schedule"):
        RefinementConfig.from_dict(
            {"schema_version": 1, "kind": "hard-pv", "structure": "dense"}
        )

    with pytest.raises(TypeError):
        TernaryConfig.ptq(profile="compact-v1", refinement="hard-pv")
    legacy = {**ptq.to_dict(), "schema_version": 1, "refinement": "none"}
    assert TernaryConfig.from_dict(legacy) == ptq
    with pytest.raises(ValueError, match="RefinementConfig"):
        TernaryConfig.from_dict({**legacy, "refinement": "hard-pv"})


def test_prepare_requires_explicit_ownership_and_prepare_qat_composes_it():
    source = torch.nn.Sequential(torch.nn.Linear(4, 3))
    with pytest.raises(TypeError):
        prepare(source, TernaryConfig.qat())

    prepared = prepare(source, TernaryConfig.qat(), inplace=True)
    assert prepared.model is source
    assert isinstance(prepared.model[0], TernaryLinear)
    assert prepared.coverage == inspect(prepared)
    assert (
        prepare_qat(torch.nn.Linear(4, 3), TernaryConfig.qat()).__class__
        is TernaryLinear
    )
    with pytest.raises(TypeError, match="bool"):
        prepare(torch.nn.Linear(4, 3), TernaryConfig.qat(), inplace=1)


def test_out_of_place_prepare_preserves_source_and_parameter_ownership():
    source = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
    source_weight = source[0].weight

    prepared = prepare(source, TernaryConfig.qat(), inplace=False)

    assert isinstance(source[0], torch.nn.Linear)
    assert type(source[0]) is torch.nn.Linear
    assert isinstance(prepared.model[0], TernaryLinear)
    assert prepared.model[0].weight is not source_weight
    assert torch.equal(prepared.model[0].weight, source_weight)


def test_ptq_prepare_selects_targets_without_claiming_conversion():
    source = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.LayerNorm(3),
    )
    prepared = prepare(
        source,
        TernaryConfig.ptq(profile="near-lossless-v1", target_modules=("Linear",)),
        inplace=True,
    )

    assert prepared.model is source
    assert type(source[0]) is torch.nn.Linear
    assert prepared.coverage.selected_parameters == 1
    assert prepared.coverage.converted_parameters == 0
    selected = next(
        entry for entry in prepared.coverage.entries if entry.disposition == "selected"
    )
    assert selected.path == "0.weight"
    assert selected.reason == "ptq_target"


def test_ptq_prepare_fails_closed_before_mutation():
    source = torch.nn.Sequential(torch.nn.LayerNorm(3))
    with pytest.raises(TritiumError) as caught:
        prepare(
            source,
            TernaryConfig.ptq(
                profile="compact-v1",
                target_modules=("Linear",),
            ),
            inplace=True,
        )
    assert caught.value.code == "incomplete_coverage"
    assert type(source[0]) is torch.nn.LayerNorm

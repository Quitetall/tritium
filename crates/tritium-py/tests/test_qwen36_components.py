"""Strict Qwen3.6 language/MTP component resolution tests."""

from __future__ import annotations

import pytest
from torch import nn

from tritium.torch import (
    Qwen36ComponentError,
    capture_qwen36_components,
    resolve_qwen36_components,
)


class Graph(nn.Module):
    def __init__(self, *, mtp: bool = True) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        if mtp:
            self.mtp = nn.Linear(4, 4)


def test_resolve_requires_explicit_mtp_component() -> None:
    graph = Graph()
    resolved = resolve_qwen36_components(graph)
    assert resolved.language_path == "model.language_model"
    assert resolved.mtp_path == "mtp"
    assert resolved.mtp_model is graph.mtp
    assert resolved.lm_head is graph.lm_head

    with pytest.raises(Qwen36ComponentError, match="MTP drafter module is missing"):
        resolve_qwen36_components(Graph(mtp=False))


def test_explicit_language_only_diagnostic_mode_does_not_claim_mtp() -> None:
    resolved = resolve_qwen36_components(Graph(mtp=False), require_mtp=False)
    assert resolved.mtp_model is None
    assert resolved.mtp_path is None


def test_aliasing_components_is_rejected() -> None:
    graph = Graph()
    graph.lm_head = graph.model.language_model
    with pytest.raises(Qwen36ComponentError, match="alias unexpectedly"):
        resolve_qwen36_components(graph)

    graph = Graph()
    graph.mtp = graph.lm_head
    with pytest.raises(Qwen36ComponentError, match="alias unexpectedly"):
        resolve_qwen36_components(graph)


def test_capture_resolves_components_before_delegating(monkeypatch) -> None:
    graph = Graph()
    observed = {}
    receipt = object()

    def fake_capture(language_model, data_factory, **kwargs):
        observed["language_model"] = language_model
        observed["data_factory"] = data_factory
        observed.update(kwargs)
        return receipt

    monkeypatch.setattr(
        "tritium.torch.qwen36.capture_qwen36_kronecker_evidence",
        fake_capture,
    )
    factory = lambda task: (task,)
    assert capture_qwen36_components(
        graph,
        factory,
        model_dir="/model",
        declared_revision="a" * 40,
        work_dir="/work",
        evidence_dir="/evidence",
        curvature="input-hessian",
        activation_cache_digest="b" * 64,
        token_stream_digest="c" * 64,
        damping=1e-4,
    ) is receipt
    assert observed["language_model"] is graph.model.language_model
    assert observed["data_factory"] is factory
    assert observed["mtp_model"] is graph.mtp

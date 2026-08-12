"""Strict Qwen3.6 language/MTP component resolution tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from tritium.torch import (
    Qwen36ComponentError,
    Qwen36LanguageMtpOracle,
    Qwen36MtpAdapter,
    attach_qwen36_mtp,
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


def _tiny_text_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["full_attention"],
        max_position_embeddings=32,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 2],
        },
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
    )


class MtpGraph(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        language = nn.Module()
        language.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        language.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        self.model = nn.Module()
        self.model.language_model = language
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = SimpleNamespace(text_config=config)


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


def test_capture_defaults_to_offload_safe_oracle_for_qwen_shape(monkeypatch) -> None:
    graph = Graph()
    graph.model.language_model.norm = nn.Identity()
    sentinel = object()
    observed = {}

    monkeypatch.setattr(
        "tritium.torch.qwen36.Qwen36LanguageMtpOracle",
        lambda model: sentinel,
    )

    def fake_capture(_language_model, _data_factory, **kwargs):
        observed["execution_model"] = kwargs["execution_model"]
        return object()

    monkeypatch.setattr(
        "tritium.torch.qwen36.capture_qwen36_kronecker_evidence",
        fake_capture,
    )
    capture_qwen36_components(
        graph,
        lambda _task: (),
        model_dir="/model",
        declared_revision="a" * 40,
        work_dir="/work",
        evidence_dir="/evidence",
        curvature="input-hessian",
        activation_cache_digest="b" * 64,
        token_stream_digest="c" * 64,
        damping=1e-4,
    )
    assert observed["execution_model"] is sentinel


def test_qwen36_mtp_adapter_matches_checkpoint_namespace(tmp_path) -> None:
    config = _tiny_text_config()
    graph = MtpGraph(config)
    template = Qwen36MtpAdapter(
        config,
        graph.model.language_model.embed_tokens,
        graph.model.language_model.rotary_emb,
    )
    expected = {
        f"mtp.{name}": torch.randn_like(value)
        for name, value in template.state_dict().items()
    }
    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file(expected, str(shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in expected}}),
        encoding="utf-8",
    )

    adapter = attach_qwen36_mtp(graph, tmp_path, device="cpu")
    assert set(adapter.state_dict()) == set(template.state_dict())
    for name, value in template.state_dict().items():
        assert torch.equal(adapter.state_dict()[name], expected[f"mtp.{name}"])
    assert resolve_qwen36_components(graph).mtp_model is adapter


def test_qwen36_mtp_adapter_forward_is_finite() -> None:
    config = _tiny_text_config()
    embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
    adapter = Qwen36MtpAdapter(
        config,
        embeddings,
        Qwen3_5TextRotaryEmbedding(config),
    ).eval()
    output = adapter(
        input_ids=torch.tensor([[1, 2, 3]]),
        hidden_states=torch.randn(1, 3, config.hidden_size),
    )
    assert output.shape == (1, 3, config.hidden_size)
    assert torch.isfinite(output).all()


def test_qwen36_oracle_avoids_hidden_state_capture_path() -> None:
    class Norm(nn.Module):
        def forward(self, values):
            return values

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.norm = Norm()

        def forward(self, input_ids):
            return self.norm(self.embed(input_ids))

    class Mtp(nn.Module):
        def forward(self, *, input_ids, hidden_states, **_kwargs):
            return hidden_states + 1

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = Language()
            self.mtp = Mtp()
            self.lm_head = nn.Linear(4, 8, bias=False)
            self.seen_output_hidden_states = []

        def forward(self, input_ids, output_hidden_states=False, **_kwargs):
            self.seen_output_hidden_states.append(output_hidden_states)
            hidden = self.model.language_model(input_ids)
            return SimpleNamespace(logits=self.lm_head(hidden), loss=None)

    model = Model()
    oracle = Qwen36LanguageMtpOracle(model)
    output = oracle(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones(1, 2, dtype=torch.int64),
        output_hidden_states=True,
    )
    assert model.seen_output_hidden_states == [False]
    assert output.mtp_hidden_states.shape == (1, 2, 4)
    assert torch.equal(output.mtp_logits, model.lm_head(output.mtp_hidden_states))
    assert output.logits.shape == (1, 2, 8)

"""Hugging Face lifecycle gates for ADR 0033 / plan 0047."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tritium.nn import TernaryEmbedding, TernaryLinear  # noqa: E402
from tritium.torch import TernaryConfig, inspect, prepare_qat  # noqa: E402


def _tiny_llama():
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    return transformers.LlamaForCausalLM(config)


def _qat_config():
    return TernaryConfig.qat(
        estimator="salt-ste", target_modules=("Linear", "Embedding"), planes=1
    )


def test_embedding_conversion_preserves_options_and_masked_ste():
    source = torch.nn.Embedding(8, 4, padding_idx=0, max_norm=2.0)
    weight = source.weight
    converted = prepare_qat(source, TernaryConfig.qat(target_modules=("Embedding",)))

    assert isinstance(converted, TernaryEmbedding)
    assert converted.weight is weight
    assert converted.padding_idx == 0
    assert converted.max_norm == 2.0

    output = converted(torch.tensor([[0, 1, 2]])).sum()
    output.backward()
    assert converted.weight.grad is not None
    assert torch.equal(converted.weight.grad[0], torch.zeros(4))


def test_hf_qat_save_reload_is_automatic_tied_and_exact(tmp_path: Path):
    torch.manual_seed(29)
    model = prepare_qat(_tiny_llama(), _qat_config())
    assert isinstance(model.model.embed_tokens, TernaryEmbedding)
    assert isinstance(model.lm_head, TernaryLinear)
    assert model.model.embed_tokens.weight is model.lm_head.weight

    tokens = torch.tensor([[1, 2, 3, 4]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = model(input_ids=tokens, labels=tokens).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    expected = model(input_ids=tokens).logits.detach()

    model.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / "model.safetensors").is_file()
    assert not (tmp_path / "pytorch_model.bin").exists()

    reloaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path)
    reloaded.eval()
    assert isinstance(reloaded.model.embed_tokens, TernaryEmbedding)
    assert isinstance(reloaded.lm_head, TernaryLinear)
    assert reloaded.model.embed_tokens.weight is reloaded.lm_head.weight
    assert inspect(reloaded).converted_parameters > 0
    assert torch.equal(reloaded(input_ids=tokens).logits, expected)


def test_hf_config_records_exact_recipe_not_nominal_bits(tmp_path: Path):
    model = prepare_qat(_tiny_llama(), _qat_config())
    model.save_pretrained(tmp_path, safe_serialization=True)
    config = transformers.AutoConfig.from_pretrained(tmp_path)
    encoded = config.quantization_config

    assert encoded["quant_method"] == "tritium"
    assert encoded["schema_version"] == 1
    assert encoded["target_modules"] == ["Linear", "Embedding"]
    assert "bits" not in encoded

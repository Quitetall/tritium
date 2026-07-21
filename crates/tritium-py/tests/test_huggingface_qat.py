"""Hugging Face lifecycle gates for ADR 0033 / plan 0047."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tritium.nn import (  # noqa: E402
    AdditiveTernaryEmbedding,
    AdditiveTernaryLinear,
    TernaryEmbedding,
    TernaryLinear,
)
from tritium.torch import (  # noqa: E402
    TernaryConfig,
    TritiumError,
    TritiumTrainer,
    calibrate,
    convert,
    inspect,
    load_quantized_module,
    prepare,
    prepare_qat,
)


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


def test_hf_ptq_save_reload_is_compact_automatic_and_exact(tmp_path: Path):
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    source = transformers.LlamaForCausalLM(config)
    prepared = prepare(
        source,
        TernaryConfig.ptq(profile="compact-v1", target_modules=("Linear",)),
        inplace=False,
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    calibration = calibrate(
        prepared,
        [{"input_ids": tokens, "use_cache": False}],
        evidence_dir=tmp_path / "evidence",
    )
    artifact = convert(prepared, calibration, work_dir=tmp_path / "work")
    model = load_quantized_module(prepared.model, artifact)
    expected = model(input_ids=tokens, use_cache=False).logits.detach()
    model.save_pretrained(tmp_path / "hf", safe_serialization=True)

    from safetensors.torch import load_file, save_file

    state = load_file(tmp_path / "hf" / "model.safetensors")
    assert any(name.endswith("packed_trits_0") for name in state)
    assert not any(
        name.endswith(".weight") and "norm" not in name and "embed_tokens" not in name
        for name in state
    )
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "hf")
    assert any(isinstance(module, AdditiveTernaryLinear) for module in reloaded.modules())
    assert not reloaded.hf_quantizer.is_trainable
    assert reloaded.config.tritium_ptq_artifact_id == artifact.artifact_id
    observed = reloaded(input_ids=tokens, use_cache=False).logits
    torch.testing.assert_close(observed, expected)

    packed_name = next(name for name in state if name.endswith("packed_trits_0"))
    original = int(state[packed_name][0])
    state[packed_name][0] = 0 if original != 0 else 1
    save_file(state, tmp_path / "hf" / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(TritiumError) as captured:
        transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "hf")
    assert captured.value.code == "state_identity"

    state[packed_name][0] = 255
    save_file(state, tmp_path / "hf" / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(TritiumError) as captured:
        transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "hf")
    assert captured.value.code == "state_domain"


def test_hf_ptq_save_reload_preserves_tied_embedding_head(tmp_path: Path):
    source = _tiny_llama()
    prepared = prepare(
        source,
        TernaryConfig.ptq(
            profile="compact-v1", target_modules=("Linear", "Embedding")
        ),
        inplace=False,
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    calibration = calibrate(
        prepared,
        [{"input_ids": tokens, "use_cache": False}],
        evidence_dir=tmp_path / "evidence",
    )
    artifact = convert(prepared, calibration, work_dir=tmp_path / "work")
    model = load_quantized_module(prepared.model, artifact)

    assert isinstance(model.model.embed_tokens, AdditiveTernaryEmbedding)
    assert isinstance(model.lm_head, AdditiveTernaryLinear)
    assert model.model.embed_tokens.packed_weight is model.lm_head.packed_weight
    model.tie_weights()
    assert model.model.embed_tokens.packed_weight is model.lm_head.packed_weight
    expected = model(input_ids=tokens, use_cache=False).logits.detach()
    model.save_pretrained(tmp_path / "hf", safe_serialization=True)

    from safetensors.torch import load_file

    state = load_file(tmp_path / "hf" / "model.safetensors")
    assert not any(name.endswith("embed_tokens.weight") for name in state)
    assert not any(name.endswith("lm_head.weight") for name in state)
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "hf")
    assert isinstance(reloaded.model.embed_tokens, AdditiveTernaryEmbedding)
    assert isinstance(reloaded.lm_head, AdditiveTernaryLinear)
    assert reloaded.model.embed_tokens.packed_weight is reloaded.lm_head.packed_weight
    reloaded.tie_weights()
    assert reloaded.model.embed_tokens.packed_weight is reloaded.lm_head.packed_weight
    observed = reloaded(input_ids=tokens, use_cache=False).logits
    torch.testing.assert_close(observed, expected)


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
    assert encoded["schema_version"] == 2
    assert encoded["target_modules"] == ["Linear", "Embedding"]
    assert "bits" not in encoded


class _TokenDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.rows = [
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([2, 3, 4, 5]),
            torch.tensor([3, 4, 5, 6]),
            torch.tensor([4, 5, 6, 7]),
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        tokens = self.rows[index]
        return {"input_ids": tokens, "labels": tokens.clone()}


def test_tritium_trainer_saves_and_resumes_native_hf_checkpoint(tmp_path: Path):
    args = transformers.TrainingArguments(
        output_dir=tmp_path,
        per_device_train_batch_size=1,
        max_steps=1,
        learning_rate=1e-4,
        save_strategy="steps",
        save_steps=1,
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        use_cpu=True,
        seed=41,
        data_seed=41,
        optim="adamw_torch",
    )
    trainer = TritiumTrainer(
        model=_tiny_llama(),
        tritium_config=_qat_config(),
        args=args,
        train_dataset=_TokenDataset(),
    )
    assert trainer.train().global_step == 1
    checkpoint = tmp_path / "checkpoint-1"
    assert checkpoint.is_dir()

    reloaded = transformers.AutoModelForCausalLM.from_pretrained(checkpoint)
    resume_args = transformers.TrainingArguments(
        output_dir=tmp_path,
        per_device_train_batch_size=1,
        max_steps=2,
        learning_rate=1e-4,
        save_strategy="no",
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        use_cpu=True,
        seed=41,
        data_seed=41,
        optim="adamw_torch",
    )
    resumed = TritiumTrainer(
        model=reloaded,
        args=resume_args,
        train_dataset=_TokenDataset(),
    )
    assert resumed.train(resume_from_checkpoint=checkpoint).global_step == 2
    assert isinstance(resumed.model.model.embed_tokens, TernaryEmbedding)
    assert inspect(resumed.model).converted_parameters > 0


def test_accelerate_gradient_accumulation_and_state_resume(tmp_path: Path):
    accelerate = pytest.importorskip("accelerate")
    accelerator = accelerate.Accelerator(
        cpu=True,
        gradient_accumulation_steps=2,
        project_dir=tmp_path,
    )
    model = prepare_qat(_tiny_llama(), _qat_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(_TokenDataset(), batch_size=1)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for index, batch in enumerate(dataloader):
        with accelerator.accumulate(model):
            loss = model(**batch).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if index == 1:
            break

    state_dir = tmp_path / "accelerate-state"
    accelerator.save_state(state_dir)
    expected_random = torch.rand(8)
    unwrapped = accelerator.unwrap_model(model)
    saved = {
        name: tensor.detach().clone()
        for name, tensor in unwrapped.state_dict().items()
        if isinstance(tensor, torch.Tensor)
    }
    with torch.no_grad():
        for parameter in unwrapped.parameters():
            parameter.add_(10)
    accelerator.load_state(state_dir)
    observed_random = torch.rand(8)

    restored = unwrapped.state_dict()
    assert all(torch.equal(restored[name], tensor) for name, tensor in saved.items())
    assert torch.equal(observed_random, expected_random)

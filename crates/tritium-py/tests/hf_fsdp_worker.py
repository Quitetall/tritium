"""Two-rank CPU FSDP worker launched by test_huggingface_distributed.py."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel
import transformers

from tritium.nn import TernaryEmbedding, TernaryLinear
from tritium.torch import TernaryConfig, inspect, prepare_qat


def _model():
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


def _config():
    return TernaryConfig.qat(
        estimator="salt-ste", target_modules=("Linear", "Embedding"), planes=1
    )


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(89)
    model = prepare_qat(_model(), _config())
    assert model.model.embed_tokens.weight is model.lm_head.weight
    wrapped = FullyShardedDataParallel(
        model,
        device_id=torch.device("cpu"),
        use_orig_params=True,
    )
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-4)

    tokens = torch.tensor([[1 + rank, 2 + rank, 3 + rank, 4 + rank]])
    loss = wrapped(input_ids=tokens, labels=tokens).loss
    loss.backward()
    optimizer.step()

    checkpoint = Path(os.environ["TRITIUM_FSDP_CHECKPOINT"])
    model_state, optimizer_state = get_state_dict(wrapped, optimizer)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=checkpoint,
    )

    torch.manual_seed(89)
    restored = prepare_qat(_model(), _config())
    assert isinstance(restored.model.embed_tokens, TernaryEmbedding)
    assert isinstance(restored.lm_head, TernaryLinear)
    assert restored.model.embed_tokens.weight is restored.lm_head.weight
    assert inspect(restored).converted_parameters > 0
    restored_wrapped = FullyShardedDataParallel(
        restored,
        device_id=torch.device("cpu"),
        use_orig_params=True,
    )
    restored_optimizer = torch.optim.AdamW(restored_wrapped.parameters(), lr=1e-4)
    restored_model_state, restored_optimizer_state = get_state_dict(
        restored_wrapped, restored_optimizer
    )
    loaded = {
        "model": restored_model_state,
        "optimizer": restored_optimizer_state,
    }
    dcp.load(loaded, checkpoint_id=checkpoint)
    set_state_dict(
        restored_wrapped,
        restored_optimizer,
        model_state_dict=loaded["model"],
        optim_state_dict=loaded["optimizer"],
    )

    original_loss = wrapped(input_ids=tokens, labels=tokens).loss.detach()
    restored_loss = restored_wrapped(input_ids=tokens, labels=tokens).loss.detach()
    assert torch.equal(restored_loss, original_loss)

    print(f"TRITIUM_FSDP_OK rank={rank}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

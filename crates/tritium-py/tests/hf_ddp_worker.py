"""Two-rank CPU DDP worker launched by test_huggingface_distributed.py."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from torch.nn.parallel import DistributedDataParallel

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
    torch.manual_seed(73)
    model = prepare_qat(_model(), _config())
    assert model.model.embed_tokens.weight is model.lm_head.weight
    wrapped = DistributedDataParallel(model, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-4)

    tokens = torch.tensor([[1 + rank, 2 + rank, 3 + rank, 4 + rank]])
    loss = wrapped(input_ids=tokens, labels=tokens).loss
    loss.backward()
    optimizer.step()

    flat = torch.cat([parameter.detach().reshape(-1) for parameter in wrapped.parameters()])
    gathered = [torch.empty_like(flat) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, flat)
    assert all(torch.equal(gathered[0], candidate) for candidate in gathered[1:])

    checkpoint = Path(os.environ["TRITIUM_DDP_CHECKPOINT"])
    if rank == 0:
        torch.save(wrapped.module.state_dict(), checkpoint)
    dist.barrier()

    restored = prepare_qat(_model(), _config())
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    assert isinstance(restored.model.embed_tokens, TernaryEmbedding)
    assert isinstance(restored.lm_head, TernaryLinear)
    assert restored.model.embed_tokens.weight is restored.lm_head.weight
    assert inspect(restored).converted_parameters > 0
    restored_flat = torch.cat(
        [parameter.detach().reshape(-1) for parameter in restored.parameters()]
    )
    assert torch.equal(restored_flat, flat)

    print(f"TRITIUM_DDP_OK rank={rank}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

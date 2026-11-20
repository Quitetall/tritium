"""Two-rank CPU FSDP worker launched by test_huggingface_distributed.py."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
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

    # PyTorch 2.11 CPU FSDP full-state gathering segfaults on rank 0, including
    # for an ordinary dense module. Materialize the already-committed sharded
    # checkpoint offline instead of calling FSDP.state_dict() on the live model.
    dist.barrier()
    if rank == 0:
        merged_checkpoint = checkpoint.parent / "fsdp-merged-state.pt"
        dcp_to_torch_save(checkpoint, merged_checkpoint)
        merged = torch.load(
            merged_checkpoint, map_location="cpu", weights_only=True
        )
        export_model = prepare_qat(_model(), _config())
        incompatible = export_model.load_state_dict(merged["model"], strict=True)
        assert not incompatible.missing_keys
        assert not incompatible.unexpected_keys
        export_model.eval()
        with torch.no_grad():
            exported_logits = export_model(input_ids=tokens, use_cache=False).logits
            resumed_logits = restored_wrapped(
                input_ids=tokens, use_cache=False
            ).logits
        assert torch.equal(exported_logits, resumed_logits)

        export_dir = checkpoint.parent / "fsdp-export"
        export_model.save_pretrained(export_dir, safe_serialization=True)
        assert (export_dir / "model.safetensors").is_file()
        assert not (export_dir / "pytorch_model.bin").exists()
        print("TRITIUM_FSDP_DCP_EXPORT_OK rank=0", flush=True)
    dist.barrier()

    print(f"TRITIUM_FSDP_OK rank={rank}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Fresh-process CPU bf16 Accelerate gate for plan 0047."""

import torch
import transformers
from accelerate import Accelerator

from tritium.nn import TernaryLinear
from tritium.torch import TernaryConfig, prepare_qat


def main() -> None:
    accelerator = Accelerator(cpu=True, mixed_precision="bf16")
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
    model = prepare_qat(
        transformers.LlamaForCausalLM(config),
        TernaryConfig.qat(target_modules=("Linear", "Embedding")),
    )
    observed_dtypes = []
    first_linear = next(module for module in model.modules() if isinstance(module, TernaryLinear))
    first_linear.register_forward_hook(
        lambda _module, _inputs, output: observed_dtypes.append(output.dtype)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model, optimizer = accelerator.prepare(model, optimizer)
    tokens = torch.tensor([[1, 2, 3, 4]])
    with accelerator.autocast():
        outputs = model(input_ids=tokens, labels=tokens)
    # Accelerate intentionally converts the public model output back to fp32;
    # the inner ternary kernels must still execute under bf16 autocast.
    assert torch.bfloat16 in observed_dtypes
    assert outputs.logits.dtype == torch.float32
    accelerator.backward(outputs.loss)
    optimizer.step()
    print("TRITIUM_ACCELERATE_BF16_OK", flush=True)


if __name__ == "__main__":
    main()

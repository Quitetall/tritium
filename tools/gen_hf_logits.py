"""Generate a transformers greedy/logit reference for a local HF model.

Usage: python3 tools/gen_hf_logits.py <model_dir> <out.json>
Defaults to SmolLM2-135M -> scratch_st/smollm2_ref.json.
"""
import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = os.path.expanduser(
    sys.argv[1] if len(sys.argv) > 1 else "~/.cache/tritium-models/smollm2-135m"
)
out = sys.argv[2] if len(sys.argv) > 2 else "scratch_st/smollm2_ref.json"

tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
text = "The capital of France is Paris. The sky is often a shade of blue."
ids = tok(text, return_tensors="pt").input_ids[0][:16]
with torch.no_grad():
    logits = model(ids.unsqueeze(0)).logits[0]  # [seq, vocab], fp32
json.dump(
    {
        "prompt_ids": ids.tolist(),
        "next_argmax_per_pos": logits.argmax(-1).tolist(),
        "logits_last_row": logits[-1].tolist(),
    },
    open(out, "w"),
)
print("seq", len(ids), "vocab", logits.shape[-1], "->", out)
print("argmax", logits.argmax(-1).tolist())

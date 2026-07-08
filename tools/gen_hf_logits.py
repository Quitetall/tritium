import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = os.path.expanduser("~/.cache/tritium-models/smollm2-135m")
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
text = "The capital of France is Paris. The sky is often a shade of blue."
ids = tok(text, return_tensors="pt").input_ids[0][:16]
with torch.no_grad():
    logits = model(ids.unsqueeze(0)).logits[0]      # [seq, vocab], fp32
argmax = logits.argmax(-1).tolist()
out = {
    "prompt_ids": ids.tolist(),
    "next_argmax_per_pos": argmax,                  # HF greedy next-token at each pos
    "logits_last_row": logits[-1].tolist(),         # full last-position logits
}
json.dump(out, open("scratch_st/smollm2_ref.json", "w"))
print("seq", len(ids), "vocab", logits.shape[-1])
print("argmax", argmax)

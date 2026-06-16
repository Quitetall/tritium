#!/usr/bin/env python3
"""Generate the BitNet 2B4T fidelity-ladder reference (the transformers oracle).

Loads `microsoft/bitnet-b1.58-2B-4T` in **fp32 on CPU** (reorder-free, so the Rust
CPU forward can be compared stage-by-stage), runs a fixed short prompt, and dumps:

  - token ids of the prompt,
  - hidden_states[0..n_layers]  (embedding + each decoder layer output),
  - layer-0 `input_layernorm` output (rung a1),
  - layer-0 attention output, pre-residual, post `attn_sub_norm`+`o_proj` (rung b),
  - final logits at the last position (rung d),
  - a short greedy continuation (token ids) for the end-to-end check.

Output: a JSON file (default tools/reference/bitnet_ladder.json) the Rust
`fidelity_ladder` integration test loads and asserts against.

Run:  python3 tools/gen_reference.py
"""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_DIR = os.path.expanduser("~/.cache/tritium-models/bitnet-2b4t-hf")
GGUF = os.path.expanduser(
    "~/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf"
)
OUT = os.path.join(os.path.dirname(__file__), "reference", "bitnet_ladder.json")

# A fixed short prompt. Kept tiny so one fp32 CPU prefill is tractable.
PROMPT = "The capital of France is"
GREEDY_NEW = 8


def _read_gguf_tensors():
    """Minimal GGUF parse: returns (tensors{name:(dims,ggml_type,abs_off)}, buf)."""
    import io
    import struct

    buf = open(GGUF, "rb").read()
    f = io.BytesIO(buf)

    def rd(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))

    assert f.read(4) == b"GGUF"
    rd("I")
    nt = rd("Q")[0]
    nm = rd("Q")[0]

    def rs():
        return f.read(rd("Q")[0]).decode()

    def rv(vt):
        if vt in (0, 7):
            return rd("B")[0]
        if vt == 1:
            return rd("b")[0]
        if vt == 2:
            return rd("H")[0]
        if vt == 3:
            return rd("h")[0]
        if vt == 4:
            return rd("I")[0]
        if vt == 5:
            return rd("i")[0]
        if vt == 6:
            return rd("f")[0]
        if vt == 8:
            return rs()
        if vt == 10:
            return rd("Q")[0]
        if vt == 11:
            return rd("q")[0]
        if vt == 12:
            return rd("d")[0]
        if vt == 9:
            ct = rd("I")[0]
            cnt = rd("Q")[0]
            return [rv(ct) for _ in range(cnt)]
        raise ValueError(vt)

    for _ in range(nm):
        rs()
        rv(rd("I")[0])
    tensors = {}
    for _ in range(nt):
        name = rs()
        nd = rd("I")[0]
        dims = [rd("Q")[0] for _ in range(nd)]
        tt = rd("I")[0]
        off = rd("Q")[0]
        tensors[name] = (dims, tt)
        tensors[name] = (dims, tt, off)
    data_off = (f.tell() + 31) // 32 * 32
    for name in tensors:
        dims, tt, off = tensors[name]
        tensors[name] = (dims, tt, data_off + off)
    return tensors, buf


def inject_gguf_values(model):
    """Overwrite the model's embedding, norm weights, and I2_S scales with the
    GGUF's exact F16/F32 values, so the oracle and the Tritium runner consume
    byte-identical non-ternary inputs."""
    import struct

    import numpy as np

    tensors, buf = _read_gguf_tensors()

    def f16(name, n):
        _, _, off = tensors[name]
        return torch.from_numpy(
            np.frombuffer(buf[off : off + n * 2], dtype=np.float16)
            .astype(np.float32)
            .copy()
        )

    def f32(name, n):
        _, _, off = tensors[name]
        return torch.from_numpy(
            np.frombuffer(buf[off : off + n * 4], dtype=np.float32).copy()
        )

    def i2s_scale(name):
        dims, _, off = tensors[name]
        n = int(np.prod(dims))
        nqb = n // 4
        return struct.unpack("<f", buf[off + nqb : off + nqb + 4])[0]

    n_embd = model.config.hidden_size
    vocab = model.config.vocab_size
    with torch.no_grad():
        # token embedding (F16 -> f32), [vocab, n_embd]
        model.model.embed_tokens.weight.data = f16(
            "token_embd.weight", vocab * n_embd
        ).reshape(vocab, n_embd)
        # final norm (F32)
        model.model.norm.weight.data = f32("output_norm.weight", n_embd)
        # per-layer norms + sub-norms (F32) and ternary scales (F32 from I2_S).
        n_ff = model.config.intermediate_size
        for li, layer in enumerate(model.model.layers):
            layer.input_layernorm.weight.data = f32(f"blk.{li}.attn_norm.weight", n_embd)
            layer.post_attention_layernorm.weight.data = f32(
                f"blk.{li}.ffn_norm.weight", n_embd
            )
            layer.self_attn.attn_sub_norm.weight.data = f32(
                f"blk.{li}.attn_sub_norm.weight", n_embd
            )
            layer.mlp.ffn_sub_norm.weight.data = f32(
                f"blk.{li}.ffn_sub_norm.weight", n_ff
            )
            scales = {
                "q_proj": f"blk.{li}.attn_q.weight",
                "k_proj": f"blk.{li}.attn_k.weight",
                "v_proj": f"blk.{li}.attn_v.weight",
                "o_proj": f"blk.{li}.attn_output.weight",
            }
            for attr, gname in scales.items():
                getattr(layer.self_attn, attr).weight_scale.data = torch.tensor(
                    [i2s_scale(gname)], dtype=torch.float32
                )
            for attr, gname in {
                "gate_proj": f"blk.{li}.ffn_gate.weight",
                "up_proj": f"blk.{li}.ffn_up.weight",
                "down_proj": f"blk.{li}.ffn_down.weight",
            }.items():
                getattr(layer.mlp, attr).weight_scale.data = torch.tensor(
                    [i2s_scale(gname)], dtype=torch.float32
                )
    print("injected GGUF F16/F32 embedding, norms, and I2_S scales", file=sys.stderr)


def main() -> int:
    if not os.path.exists(os.path.join(HF_DIR, "model.safetensors")):
        print(f"missing {HF_DIR}/model.safetensors; download it first", file=sys.stderr)
        return 1

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(HF_DIR)
    model = AutoModelForCausalLM.from_pretrained(HF_DIR)
    model.eval()

    # transformers 5.5.3 leaves the offline AutoBitLinear `weight` as ternary
    # values stored in **uint8** ({0, 1, 255} == signed int8 {0, +1, -1}) rather
    # than dequantized floats, which makes F.linear raise a dtype error.
    # Materialize the float ternary weight by viewing the uint8 as int8
    # (255 -> -1) — verified to equal `unpack_weights` (the canonical {-1,0,+1})
    # bit-exactly. (NB: `uint8 - 1` is WRONG: it maps 255 -> 254.) Widen the whole
    # model to fp32 so the CPU oracle is reorder-free; weight_scale stays a
    # separate post-multiply, matching AutoBitLinear.forward.
    from transformers.integrations.bitnet import AutoBitLinear

    for mod in model.modules():
        if isinstance(mod, AutoBitLinear) and mod.weight.dtype == torch.uint8:
            w = mod.weight.detach().view(torch.int8).to(torch.float32)
            mod.weight = torch.nn.Parameter(w, requires_grad=False)
            if hasattr(mod, "weight_scale"):
                mod.weight_scale = mod.weight_scale.detach().to(torch.float32)

    # Widen every remaining float param/buffer (embeddings, norms, rotary) to fp32
    # in place; `.float()` is blocked on quantized models, so do it manually.
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype in (torch.bfloat16, torch.float16):
                p.data = p.data.to(torch.float32)
        for name, b in model.named_buffers():
            if b.dtype in (torch.bfloat16, torch.float16):
                b.data = b.data.to(torch.float32)

    # Inject the GGUF's *exact* non-ternary values so the oracle consumes
    # byte-identical inputs to the Tritium runner: the HF checkpoint is bf16 but
    # the GGUF embedding is F16, its norms are F32, and its I2_S scales are F32.
    # Without this the bf16-vs-F16/F32 rounding gap (the same trained weights at
    # different precision) shows up as ~1e-2 relative drift that grows across the
    # 30 layers and would force a loose ladder tolerance, masking real bugs. The
    # ternary weights are already bit-identical (validated), so only embedding,
    # norms, and per-tensor scales need overriding. If the GGUF is absent we skip
    # this and fall back to the bf16 oracle (the ladder then uses a looser bar).
    if os.path.exists(GGUF):
        inject_gguf_values(model)

    # Tokenize WITHOUT a chat template (raw prefill), with the BOS the model expects.
    ids = tok(PROMPT, return_tensors="pt").input_ids  # includes bos by default
    token_ids = ids[0].tolist()
    print("prompt token ids:", token_ids, file=sys.stderr)

    # Hooks for layer-0 stages.
    captured = {}

    layer0 = model.model.layers[0]

    def cap_input_ln(mod, inp, out):
        captured["layer0_input_ln"] = out.detach().float()[0].cpu()

    def cap_attn(mod, inp, out):
        # BitNetAttention.forward returns (attn_output, attn_weights). attn_output
        # is already post attn_sub_norm + o_proj, pre-residual.
        o = out[0] if isinstance(out, tuple) else out
        captured["layer0_attn_out"] = o.detach().float()[0].cpu()

    def cap_final_norm(mod, inp, out):
        captured["final_norm"] = out.detach().float()[0].cpu()

    # Per-layer RAW block outputs via hooks. `output_hidden_states` REPLACES the
    # last entry with the post-final-norm state, so its `hidden_states[n_layers]`
    # is NOT layer (n_layers-1)'s raw output — we hook every decoder layer instead
    # so every rung compares like-for-like against the Rust block output.
    layer_outs = [None] * model.config.num_hidden_layers

    def make_layer_hook(i):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            layer_outs[i] = o.detach().float()[0].cpu()

        return hook

    hooks = [
        layer0.input_layernorm.register_forward_hook(cap_input_ln),
        layer0.self_attn.register_forward_hook(cap_attn),
        model.model.norm.register_forward_hook(cap_final_norm),
    ]
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_layer_hook(i)))

    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)

    for h in hooks:
        h.remove()

    # hidden_states[0] = the embedding; the per-layer raw outputs come from hooks.
    embedding = out.hidden_states[0][0].float().cpu()
    hidden_states = layer_outs  # length n_layers, each the raw block output
    logits = out.logits[0].float().cpu()  # [seq, vocab]
    last_logits = logits[-1]

    # Final norm output (captured via a hook on model.model.norm).
    final_norm = captured["final_norm"]

    # Short greedy continuation (end-to-end token-id check).
    with torch.no_grad():
        gen = model.generate(
            ids,
            max_new_tokens=GREEDY_NEW,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    greedy_ids = gen[0].tolist()[len(token_ids):]
    print("greedy continuation ids:", greedy_ids, file=sys.stderr)

    n_layers = model.config.num_hidden_layers
    data = {
        "prompt": PROMPT,
        "token_ids": token_ids,
        "n_layers": n_layers,
        "n_embd": model.config.hidden_size,
        "vocab": model.config.vocab_size,
        # rung a0: the token embedding (out.hidden_states[0]).
        "embedding": embedding.reshape(-1).tolist(),
        # rung a1
        "layer0_input_ln": captured["layer0_input_ln"].reshape(-1).tolist(),
        # rung b
        "layer0_attn_out": captured["layer0_attn_out"].reshape(-1).tolist(),
        # rung c / c': per-layer RAW block outputs (hooked, length n_layers).
        "hidden_states": [hidden_states[i].reshape(-1).tolist() for i in range(n_layers)],
        # final norm
        "final_norm": final_norm.reshape(-1).tolist(),
        # rung d: last-position logits
        "last_logits": last_logits.tolist(),
        "argmax_last": int(torch.argmax(last_logits).item()),
        "greedy_ids": greedy_ids,
        "eos_token_id": model.config.eos_token_id
        if isinstance(model.config.eos_token_id, int)
        else model.config.eos_token_id[0],
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f)
    print(f"wrote {OUT}  (seq={len(token_ids)}, argmax_last={data['argmax_last']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Ported verbatim from blut-lamu python/lamu/export_i2s_gguf.py by its copyright
# holder (ADR 0037 Stage 4). NOTE: the writer lazily imports 
# (the model definition, which stays cookbook-side by the ADR's ownership rule)
# to rebuild the checkpoint's module — callers must have the cookbook's
# python/lamu dir on PYTHONPATH, which its stages already inject.
"""Export a trained BitNetStudent checkpoint to a Tritium-servable I2_S GGUF.

Strategy: copy EVERY metadata key-value from a reference GGUF (the spec-decode
TARGET, e.g. BitNet 2B4T) — architecture string, rope params, and crucially the
full tokenizer tables (the drafter MUST share the target tokenizer; tritium-serve
rejects a vocab mismatch) — then override only the dimension keys, and write the
student's tensors:

- token_embd.weight            f16   [vocab, n_embd] (tied; no output.weight)
- output_norm.weight           f32
- blk.{i}.attn_norm.weight     f32
- blk.{i}.attn_sub_norm.weight f32
- blk.{i}.ffn_norm.weight      f32
- blk.{i}.ffn_sub_norm.weight  f32
- blk.{i}.attn_q/k/v/output.weight  I2_S (type 36)
- blk.{i}.ffn_gate/up/down.weight   I2_S

I2_S payload contract (verified against tritium-format/src/i2s.rs, 2026-07-12):
128-element blocks of 32 bytes; byte `gp` of a block holds the elements at
positions [gp, 32+gp, 64+gp, 96+gp] in bit-pairs [7:6], [5:4], [3:2], [1:0];
code = trit + 1 (0b00=-1, 0b01=0, 0b10=+1; 0b11 forbidden); ONE trailing
little-endian f32 per-tensor scale after the quant bytes. The exact f32 scale
additionally rides in `tritium.i2s_scale.<tensor>` metadata (the loader
prefers it; f16-unrepresentable scales survive that way).

Usage:
  python export_i2s_gguf.py --checkpoint student.pt --reference target.gguf \
      --out drafter.gguf
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

GGUF_MAGIC = 0x46554747  # 'GGUF'
GGUF_VERSION = 3
ALIGN = 32

# GGUF metadata value types.
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)

GGML_F32, GGML_F16 = 0, 1
GGML_I2_S = 36


# ── minimal GGUF reader (metadata only; enough to clone the target's KVs) ──
class Reader:
    def __init__(self, path: Path) -> None:
        self.b = path.read_bytes()
        self.o = 0

    def u(self, fmt: str) -> int | float:
        v = struct.unpack_from(fmt, self.b, self.o)[0]
        self.o += struct.calcsize(fmt)
        return v

    def string(self) -> str:
        n = self.u("<Q")
        s = self.b[self.o : self.o + n].decode("utf-8")
        self.o += n
        return s

    def value(self, t: int):
        scalars = {
            T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I",
            T_I32: "<i", T_F32: "<f", T_U64: "<Q", T_I64: "<q", T_F64: "<d",
        }
        if t in scalars:
            return self.u(scalars[t])
        if t == T_BOOL:
            return bool(self.u("<B"))
        if t == T_STR:
            return self.string()
        if t == T_ARR:
            et = self.u("<I")
            n = self.u("<Q")
            return (et, [self.value(et) for _ in range(n)])
        raise ValueError(f"unknown gguf value type {t}")


def read_metadata(path: Path) -> list[tuple[str, int, object]]:
    r = Reader(path)
    if r.u("<I") != GGUF_MAGIC:
        raise ValueError(f"{path}: not a GGUF file")
    version = r.u("<I")
    if version != GGUF_VERSION:
        raise ValueError(f"{path}: gguf v{version}, expected v3")
    _n_tensors = r.u("<Q")
    n_kv = r.u("<Q")
    out = []
    for _ in range(n_kv):
        key = r.string()
        t = r.u("<I")
        out.append((key, t, r.value(t)))
    return out


# ── minimal GGUF writer ──
class Writer:
    def __init__(self) -> None:
        self.kv: list[bytes] = []
        self.tensors: list[tuple[str, list[int], int, bytes]] = []

    @staticmethod
    def _string(s: str) -> bytes:
        e = s.encode("utf-8")
        return struct.pack("<Q", len(e)) + e

    def _value(self, t: int, v) -> bytes:
        scalars = {
            T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I",
            T_I32: "<i", T_F32: "<f", T_U64: "<Q", T_I64: "<q", T_F64: "<d",
        }
        if t in scalars:
            return struct.pack(scalars[t], v)
        if t == T_BOOL:
            return struct.pack("<B", 1 if v else 0)
        if t == T_STR:
            return self._string(v)
        if t == T_ARR:
            et, items = v
            out = struct.pack("<IQ", et, len(items))
            return out + b"".join(self._value(et, i) for i in items)
        raise ValueError(f"unknown gguf value type {t}")

    def add_kv(self, key: str, t: int, v) -> None:
        self.kv.append(self._string(key) + struct.pack("<I", t) + self._value(t, v))

    def add_tensor(self, name: str, shape: list[int], ggml_type: int, data: bytes) -> None:
        self.tensors.append((name, shape, ggml_type, data))

    def write(self, path: Path) -> None:
        head = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(self.tensors), len(self.kv))
        kv_blob = b"".join(self.kv)
        infos = b""
        offset = 0
        for name, shape, t, data in self.tensors:
            infos += self._string(name)
            infos += struct.pack("<I", len(shape))
            for d in shape:
                infos += struct.pack("<Q", d)
            infos += struct.pack("<I", t)
            infos += struct.pack("<Q", offset)
            offset += (len(data) + ALIGN - 1) // ALIGN * ALIGN
        base = len(head) + len(kv_blob) + len(infos)
        pad0 = (ALIGN - base % ALIGN) % ALIGN
        with path.open("wb") as f:
            f.write(head)
            f.write(kv_blob)
            f.write(infos)
            f.write(b"\x00" * pad0)
            for _, _, _, data in self.tensors:
                f.write(data)
                pad = (ALIGN - len(data) % ALIGN) % ALIGN
                f.write(b"\x00" * pad)


def pack_i2s(trits: np.ndarray, scale: float) -> bytes:
    """Trits int8 {-1,0,1}, C-order flat → I2_S payload (see module doc)."""
    flat = trits.reshape(-1)
    n = flat.size
    if n % 128 != 0:
        raise ValueError(f"element count {n} not a multiple of 128")
    codes = (flat.astype(np.int16) + 1).astype(np.uint8)  # 0,1,2
    if codes.max(initial=0) > 2:
        raise ValueError("trit out of range")
    blocks = codes.reshape(-1, 4, 32)  # [block, group, gp]
    shifts = np.array([6, 4, 2, 0], dtype=np.uint8).reshape(1, 4, 1)
    packed = (blocks << shifts).sum(axis=1).astype(np.uint8)  # [block, 32]
    return packed.tobytes() + struct.pack("<f", scale)


GGML_TQ2_0 = 35
TQ2_BLOCK = 256
TQ2_BLOCK_BYTES = 66  # 64 packed bytes + LE f16 scale (qs then d)


def pack_tq2_0_row_uniform(row: np.ndarray, scale: float) -> bytes:
    """One row of trits {-1,0,1} -> TQ2_0 blocks, ALL blocks carrying the
    row's scale (row-uniform: the Tritium loader's per-row path, b177594).
    Layout mirrors tritium-format tq2.rs: byte c*32+m holds trits
    [c*128 + n*32 + m] at bit-pair 2n (code = trit+1); f16 scale last.
    """
    k = row.size
    codes = (row.astype(np.int16) + 1).astype(np.uint8)
    if k % TQ2_BLOCK:
        codes = np.pad(codes, (0, TQ2_BLOCK - k % TQ2_BLOCK), constant_values=1)
    out = bytearray()
    sc = np.array([scale], dtype="<f2").tobytes()
    for b in range(codes.size // TQ2_BLOCK):
        blk = codes[b * TQ2_BLOCK : (b + 1) * TQ2_BLOCK]
        data = np.zeros(64, dtype=np.uint8)
        for c in range(2):
            for n in range(4):
                seg = blk[c * 128 + n * 32 : c * 128 + n * 32 + 32]
                data[c * 32 : (c + 1) * 32] |= (seg & 3) << (2 * n)
        out += data.tobytes() + sc
    return bytes(out)


DIM_KEYS = {
    "block_count": lambda c: (T_U32, c.n_layer),
    "embedding_length": lambda c: (T_U32, c.n_embd),
    "feed_forward_length": lambda c: (T_U32, c.n_ff),
    "attention.head_count": lambda c: (T_U32, c.n_head),
    "attention.head_count_kv": lambda c: (T_U32, c.n_kv_head),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path, help="the spec-decode TARGET gguf")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from bitnet_student import BitNetStudent, StudentConfig

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = StudentConfig(**ckpt["config"])
    model = BitNetStudent(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ref_kv = read_metadata(args.reference)
    arch = next(v for k, _, v in ref_kv if k == "general.architecture")

    w = Writer()
    overridden = 0
    for key, t, v in ref_kv:
        hit = None
        for suffix, mk in DIM_KEYS.items():
            if key == f"{arch}.{suffix}":
                hit = mk(cfg)
                break
        if hit is not None:
            w.add_kv(key, hit[0], hit[1])
            overridden += 1
        elif key == "general.name":
            w.add_kv(key, T_STR, f"{v}-drafter-{cfg.n_layer}L{cfg.n_embd}")
        else:
            w.add_kv(key, t, v)
    if overridden != len(DIM_KEYS):
        raise SystemExit(
            f"only {overridden}/{len(DIM_KEYS)} dimension keys found under arch "
            f"'{arch}' in the reference — key naming drift, refusing to export"
        )

    scales: dict[str, float] = {}

    def tern(name: str, lin) -> None:
        trits, scale = lin.export_ternary()
        # GGUF shape convention: [ne0=in, ne1=out] for llama-family matrices.
        w.add_tensor(name, [trits.shape[1], trits.shape[0]], GGML_I2_S,
                     pack_i2s(trits.numpy(), scale))
        scales[name] = scale

    def norm(name: str, m) -> None:
        arr = m.weight.detach().float().numpy()
        w.add_tensor(name, [arr.size], GGML_F32, arr.astype("<f4").tobytes())

    emb = model.embed.weight.detach().to(torch.float16).numpy()
    w.add_tensor("token_embd.weight", [cfg.n_embd, cfg.vocab_size], GGML_F16,
                 emb.astype("<f2").tobytes())
    # Untied ternary head (ADR 0032 L1): emit output.weight as I2_S. The input
    # token_embd stays f16 (quality lookup); only the head is ternary (~8× less
    # table read at serve). Tied models emit no output.weight (loader ties).
    if getattr(model, "lm_head", None) is not None:
        if getattr(model.lm_head, "per_row_scale", False):
            # T2a: per-row scales ride as row-uniform TQ2_0 block scales
            # (f16 precision per row; no i2s_scale metadata — the loader's
            # per-row path is the consumer).
            trits, row_scales = model.lm_head.export_ternary_per_row()
            tn = trits.numpy()
            payload = b"".join(
                pack_tq2_0_row_uniform(tn[r], float(row_scales[r]))
                for r in range(tn.shape[0])
            )
            w.add_tensor("output.weight", [tn.shape[1], tn.shape[0]],
                         GGML_TQ2_0, payload)
        else:
            tern("output.weight", model.lm_head)
    norm("output_norm.weight", model.out_norm)
    for i, blk in enumerate(model.blocks):
        p = f"blk.{i}."
        norm(p + "attn_norm.weight", blk.attn_norm)
        norm(p + "attn_sub_norm.weight", blk.attn_sub_norm)
        norm(p + "ffn_norm.weight", blk.ffn_norm)
        norm(p + "ffn_sub_norm.weight", blk.ffn_sub_norm)
        tern(p + "attn_q.weight", blk.q)
        tern(p + "attn_k.weight", blk.k)
        tern(p + "attn_v.weight", blk.v)
        tern(p + "attn_output.weight", blk.o)
        tern(p + "ffn_gate.weight", blk.gate)
        tern(p + "ffn_up.weight", blk.up)
        tern(p + "ffn_down.weight", blk.down)

    # Authoritative per-tensor scales (the Tritium loader prefers these).
    for name, scale in scales.items():
        w.add_kv(f"tritium.i2s_scale.{name}", T_F32, scale)

    w.write(args.out)
    print(json.dumps({
        "exported": str(args.out),
        "tensors": len(w.tensors),
        "arch": arch,
        "config": ckpt["config"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

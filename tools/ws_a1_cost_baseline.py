#!/usr/bin/env python3
"""Plan 0054 WS-A1/A4: measured capture-cost baseline — per-tensor replay vs shared-forward.

Runs the SAME frozen calibration inputs through:
  (a) per-tensor capture: one full calibration replay per target tensor
  (b) shared-forward capture: one replay feeds every target tensor's writer
and reports wall time, replay counts, artifact bytes, and record digests
(digest equality re-proves WS-A3 byte-identity on the real model).

Receipt JSON is the plan-0043 cost-forecast input (measured, not extrapolated).
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from tritium.torch.ptq import (
    KroneckerCalibrationWriter,
    capture_kronecker_module,
    capture_kronecker_module_group,
)

# Non-zero deterministic digests (the builder rejects all-zero as absent).
# Identical across both paths, so byte-identity still binds.
DIG_MODEL = hashlib.sha256(b"ws-a1 source model").hexdigest()
DIG_CACHE = hashlib.sha256(b"ws-a1 activation cache").hexdigest()
DIG_TOKENS = hashlib.sha256(b"ws-a1 token stream").hexdigest()


def build_batches(tokenizer, device, sequences, seq_len, batch_size, seed):
    """Deterministic synthetic-token calibration batches (frozen by seed)."""
    g = torch.Generator().manual_seed(seed)
    vocab = tokenizer.vocab_size
    batches = []
    done = 0
    while done < sequences:
        n = min(batch_size, sequences - done)
        ids = torch.randint(0, vocab, (n, seq_len), generator=g)
        batches.append({"input_ids": ids.to(device)})
        done += n
    return batches


def target_modules(model, layer_indices):
    """Per layer: the three attention-norm consumers (q/k/v) — one shared input stream."""
    out = []
    for li in layer_indices:
        base = f"model.layers.{li}.self_attn"
        for proj in ("q_proj", "k_proj", "v_proj"):
            path = f"{base}.{proj}"
            mod = model.get_submodule(path)
            assert isinstance(mod, nn.Linear)
            out.append((path, mod.out_features, mod.in_features))
    return out


OBJECTIVE_IDS = {
    "input-hessian": "tritium.input-gram@1",
    "guided-fisher": "tritium.model-loss-guided-fisher.mean-attention-mask@1",
    "forward-kl-kronecker": "tritium.softmax-fisher-rademacher.single-probe@1",
}


def make_writer(evidence_dir, idx, name, rows, cols, curvature, damping):
    return KroneckerCalibrationWriter(
        evidence_dir,
        tensor_index=idx,
        tensor_name=name,
        rows=rows,
        columns=cols,
        curvature=curvature,
        source_model_digest=DIG_MODEL,
        activation_cache_digest=DIG_CACHE,
        token_stream_digest=DIG_TOKENS,
        damping=damping,
        objective_id=OBJECTIVE_IDS[curvature],
    )


def record_digests(evidence_dir: Path):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(evidence_dir.iterdir())
        if p.is_file()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=4, help="number of layers to probe")
    ap.add_argument("--sequences", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--curvature", default="input-hessian",
                    choices=["input-hessian", "forward-kl-kronecker"])
    ap.add_argument("--damping", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    stride = max(1, n_layers // args.layers)
    layer_indices = list(range(0, n_layers, stride))[: args.layers]
    targets = target_modules(model, layer_indices)
    batches = build_batches(tok, device, args.sequences, args.seq_len, args.batch_size, args.seed)

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    per_dir = work / "per_tensor"
    grp_dir = work / "shared_forward"
    per_dir.mkdir(parents=True)
    grp_dir.mkdir(parents=True)

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

    # (a) per-tensor: one full replay per tensor
    t0 = time.perf_counter()
    for idx, (path, rows, cols) in enumerate(targets):
        w = make_writer(per_dir, idx, path, rows, cols, args.curvature, args.damping)
        capture_kronecker_module(model, batches, module=path, writer=w, curvature=args.curvature)
    per_wall = time.perf_counter() - t0
    per_peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

    # (b) shared-forward: one replay per LAYER group (q/k/v share the attn-norm
    # input). An all-24-in-one replay exceeds the bounded-capture snapshot budget
    # (max_capture_bytes, 256 MiB default) — the grouping planner's residency
    # constraint observed in practice; per-layer grouping is the production shape.
    t0 = time.perf_counter()
    group_replays = 0
    for li in layer_indices:
        group = [(i, t) for i, t in enumerate(targets) if t[0].startswith(f"model.layers.{li}.")]
        writers = [
            make_writer(grp_dir, idx, path, rows, cols, args.curvature, args.damping)
            for idx, (path, rows, cols) in group
        ]
        capture_kronecker_module_group(
            model, batches,
            modules=[path for _, (path, _, _) in group],
            writers=writers,
            curvature=args.curvature,
        )
        group_replays += 1
    grp_wall = time.perf_counter() - t0
    grp_peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0

    per_dig = record_digests(per_dir)
    grp_dig = record_digests(grp_dir)
    identical = per_dig == grp_dig

    receipt = {
        "schema": "tritium.ws-a1-cost-baseline.v1",
        "model": args.model,
        "device": device,
        "curvature": args.curvature,
        "calibration": {
            "sequences": args.sequences, "seq_len": args.seq_len,
            "batch_size": args.batch_size, "seed": args.seed,
            "tokens": args.sequences * args.seq_len,
        },
        "targets": {"tensors": len(targets), "layers": layer_indices,
                    "shapes": [(p, r, c) for p, r, c in targets]},
        "per_tensor": {
            "replays": len(targets), "wall_s": round(per_wall, 3),
            "peak_bytes": per_peak,
            "artifact_bytes": sum(p.stat().st_size for p in per_dir.iterdir()),
        },
        "shared_forward": {
            "replays": group_replays, "wall_s": round(grp_wall, 3),
            "peak_bytes": grp_peak,
            "artifact_bytes": sum(p.stat().st_size for p in grp_dir.iterdir()),
        },
        "speedup_wall": round(per_wall / grp_wall, 3) if grp_wall > 0 else None,
        "replay_reduction": round(len(targets) / group_replays, 3),
        "byte_identity": identical,
    }
    Path(args.out).write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("per_tensor", "shared_forward", "speedup_wall",
                       "replay_reduction", "byte_identity")}, indent=1))
    if not identical:
        diff = {k for k in per_dig if per_dig.get(k) != grp_dig.get(k)}
        print("BYTE-IDENTITY FAILED for:", sorted(diff)[:5])
        raise SystemExit(1)


if __name__ == "__main__":
    main()

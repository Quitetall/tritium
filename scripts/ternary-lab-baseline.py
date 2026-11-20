#!/usr/bin/env python3
"""External FP32 STE+Adam baseline for EAT-O. Never imported by the integer experiment.
Same fixed scales, integer-rounded forward activations, bias-free topology, raw
pixel inputs and squared loss. Weight/activation rounding use STE here explicitly.
"""
import argparse
import json
import pathlib
import struct
import time
import hashlib
import platform
import resource
import subprocess
import blake3

import torch

MASK = (1 << 64)-1


def draw(state, n):
    cutoff = MASK - MASK % n
    while True:
        state ^= (state << 13) & MASK
        state ^= state >> 7
        state ^= (state << 17) & MASK
        state &= MASK
        if state < cutoff:
            return state, state % n


def sample(seed, step, n):
    return draw(max(1, (seed + step*0x9e3779b97f4a7c15) & MASK), n)[1]


def load(data):
    images = (data / "train-images-idx3-ubyte").read_bytes()
    labels = (data / "train-labels-idx1-ubyte").read_bytes()
    assert struct.unpack(">4I", images[:16]) == (2051, 60000, 28, 28)
    assert struct.unpack(">2I", labels[:8]) == (2049, 60000)
    x = torch.frombuffer(bytearray(images[16:]), dtype=torch.uint8).reshape(60000, 784).float()
    y = torch.frombuffer(bytearray(labels[8:]), dtype=torch.uint8).long()
    return x, y


def ste(value, quantized):
    return value + (quantized-value).detach()


def run(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    x, y = load(a.data)
    state = a.seed
    values = []
    for _ in range(784*128+128*10):
        state, value = draw(state, 3)
        values.append(value-1)
    w1 = torch.nn.Parameter(torch.tensor(values[:784*128], dtype=torch.float32).reshape(128, 784))
    w2 = torch.nn.Parameter(torch.tensor(values[784*128:], dtype=torch.float32).reshape(10, 128))
    optimizer = torch.optim.Adam([w1, w2], lr=a.lr)

    def forward(inputs):
        q1 = ste(w1, w1.round().clamp(-1, 1))
        q2 = ste(w2, w2.round().clamp(-1, 1))
        h = torch.nn.functional.linear(inputs, q1)/256
        h = ste(h, h.trunc()).relu()
        out = torch.nn.functional.linear(h, q2)/8
        return ste(out, out.trunc())

    start = time.monotonic()
    contraction_terms = 0
    completed = 0
    for step in range(a.steps):
        if contraction_terms >= a.work_limit:
            break
        i = sample(a.seed, step, 55000)
        out = forward(x[i:i+1])
        target = torch.nn.functional.one_hot(y[i:i+1], 10).float()*256
        loss = (out-target).square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        contraction_terms += 2*(784*128+128*10)+128*10
        completed += 1
    elapsed = time.monotonic()-start
    with torch.no_grad():
        out = forward(x[55000+a.eval_offset:55000+a.eval_offset+a.eval_limit])
        labels = y[55000+a.eval_offset:55000+a.eval_offset+a.eval_limit]
        loss = (out-torch.nn.functional.one_hot(labels, 10)*256).square().sum().item()
        correct = (out.argmax(1) == labels).sum().item()
    # Identity hash reads original integer IDX data, independent of float conversion.
    raw_images = (a.data / "train-images-idx3-ubyte").read_bytes()[16:]
    raw_labels = (a.data / "train-labels-idx1-ubyte").read_bytes()[8:]
    def digest_raw(start, end):
        h = blake3.blake3()
        for i in range(start, end):
            h.update(struct.pack("<QQ", raw_labels[i], 784))
            h.update(raw_images[i*784:(i+1)*784])
        return h.hexdigest()
    checkpoint = a.report.with_suffix(".pt")
    with checkpoint.open("xb") as stream:
        torch.save({"w1": w1.detach(), "w2": w2.detach(), "optimizer": optimizer.state_dict(),
                    "seed": a.seed, "completed_steps": completed}, stream)
    result = {"algorithm": "fp32-ste-adam", "seed": a.seed, "steps": completed, "contraction_terms": contraction_terms,
              "lr": a.lr, "evaluation_count": len(labels), "loss_sum": loss,
              "correct": correct, "elapsed_training_seconds": elapsed,
              "master_moments_bytes": sum(p.numel()*12 for p in (w1, w2)),
              "qualification": "UNKNOWN", "split": "validation",
              "evaluation_offset": a.eval_offset,
              "data_digest": digest_raw(0, 55000),
              "evaluation_digest": digest_raw(55000+a.eval_offset, 55000+a.eval_offset+len(labels)),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "implementation_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
              "device": "CPU", "machine": platform.machine(), "torch_version": torch.__version__,
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
              "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    with a.report.open("x") as out_file:
        json.dump(result, out_file, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=pathlib.Path, required=True)
    p.add_argument("--report", type=pathlib.Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval-offset", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=5000)
    p.add_argument("--work-limit", type=int, default=2**63-1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    if args.seed <= 0 or args.steps < 0 or not 0 < args.eval_limit <= 5000 or not 0 <= args.eval_offset < 5000 or args.eval_offset+args.eval_limit > 5000:
        p.error("invalid seed, steps, or evaluation limit")
    run(args)

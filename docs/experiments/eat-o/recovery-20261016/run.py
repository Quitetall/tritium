#!/usr/bin/env python3
"""Recover a ternarized fp model: exact vs fused forward-only carry training (ternary_deep).

Per seed: an fp 784-128-10 model (f64 backprop, 165k examples, lr 1e-4, same init as the ternary
net) is ternarized with a per-layer power-of-two scale fit (folded into the shifts), then
recovered from the trit centres (no fp master on device) on labels or on the fp model's outputs
(distillation). Tune each arm's lr shift on seed 101 at half its budget (validation 0-999);
evaluate on seeds 1-5 (validation 1000-4999).
"""
import concurrent.futures as cf, hashlib, json, os, subprocess, sys
from pathlib import Path

BIN, DATA, OUT, CACHE = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
OUT.mkdir(parents=True, exist_ok=True)
SHIFTS = list(range(2, 12))
BASE = ["--fp-steps", "165000", "--fp-lr", "1e-4", "--fp-cache", CACHE]
N = ["--grad", "noise", "--draws", "1"]
T = ["--target", "teacher"]
ARMS = {}
for b in (2000, 8000, 32000):
    ARMS[f"exact-label-{b // 1000}k"] = ([], b)
    ARMS[f"exact-teacher-{b // 1000}k"] = (T, b)
for b in (2000, 8000, 32000, 128000, 512000):
    ARMS[f"fused-label-{b // 1000}k"] = (N, b)
    ARMS[f"fused-teacher-{b // 1000}k"] = ([*N, *T], b)
for pct in (20, 5):
    ARMS[f"exact-teacher-8k-sparse{pct}"] = ([*T, "--sparse-carry", str(pct)], 8000)
    ARMS[f"fused-teacher-128k-sparse{pct}"] = ([*N, *T, "--sparse-carry", str(pct)], 128000)
ARMS["exact-teacher-8k-fpcarry"] = ([*T, "--carry-init", "fp"], 8000)
ARMS["fused-teacher-128k-fpcarry"] = ([*N, *T, "--carry-init", "fp"], 128000)

def job(tag, args):
    path = OUT / f"{tag}.json"
    if not path.exists():
        out = subprocess.run([BIN, "--data", DATA, *BASE, *args], capture_output=True, text=True, check=True)
        path.write_text(out.stdout)
    return tag, json.loads(path.read_text())

with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", "12"))) as ex:
    order = sorted(ARMS.items(), key=lambda kv: -kv[1][1])
    futs = [ex.submit(job, f"tune-{a}-s{s}", [*k, "--lr", str(s), "--seed", "101", "--steps",
                      str(st // 2), "--eval-offset", "0", "--eval-limit", "1000"])
            for a, (k, st) in order for s in SHIFTS]
    tune = dict(f.result() for f in cf.as_completed(futs))
    selected = {}
    for a in ARMS:
        best = min(SHIFTS, key=lambda s: tune[f"tune-{a}-s{s}"]["mean_loss"])
        if best in (SHIFTS[0], SHIFTS[-1]):
            print(f"WARNING {a}: selected shift {best} is a grid edge", file=sys.stderr)
        selected[a] = best
    futs = [ex.submit(job, f"eval-{a}-{seed}", [*ARMS[a][0], "--lr", str(selected[a]),
                      "--seed", str(seed), "--steps", str(ARMS[a][1])])
            for a, _ in order for seed in range(1, 6)]
    for f in cf.as_completed(futs):
        f.result()

(OUT / "protocol.json").write_text(json.dumps({
    "base": BASE, "arms": ARMS, "lr_shift_grid": SHIFTS, "selected": selected,
    "tune": "seed 101, half the budget, validation 0..999",
    "eval": "seeds 1-5, validation 1000..4999",
    "binary_sha256": hashlib.sha256(Path(BIN).read_bytes()).hexdigest(),
    "quality": "UNKNOWN"}, indent=1))
print(json.dumps(selected))

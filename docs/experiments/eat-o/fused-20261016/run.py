#!/usr/bin/env python3
"""Fusion axis F (and the training half of axis G): forward-only noise credit into the carry.

Protocol as the carry campaign: tune each arm's lr shift on seed 101 at half its evaluation
horizon, validation 0-999; evaluate on seeds 1-5 at the full horizon, validation 1000-4999.
Default noise: ternary antithetic, sigma = 2 quantization steps, integer credit magnitude.
"""
import concurrent.futures as cf, hashlib, json, subprocess, sys
from pathlib import Path

BIN, DATA, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
SHIFTS = [2, 3, 4, 5, 6, 7, 8, 9]
N = ["--grad", "noise"]
ARMS = {  # name: (extra args, evaluation steps)
    "exact": ([], 20000),
    "exact-defer16": (["--defer", "16"], 20000),
    "K1": ([*N, "--draws", "1"], 20000),
    "K4": ([*N, "--draws", "4"], 20000),
    "K16": ([*N, "--draws", "16"], 20000),
    "K1-defer16": ([*N, "--draws", "1", "--defer", "16"], 20000),
    "K1-defer64": ([*N, "--draws", "1", "--defer", "64"], 20000),
    "K1-keep0": ([*N, "--draws", "1", "--keep-pct", "0"], 20000),
    "K1-keep50": ([*N, "--draws", "1", "--keep-pct", "50"], 20000),
    "K1-leak16": ([*N, "--draws", "1", "--leak", "16"], 20000),
    "K1-leak12": ([*N, "--draws", "1", "--leak", "12"], 20000),
    "K1-sigma4": ([*N, "--draws", "1", "--sigma", "4"], 20000),
    "K1-sigma1": ([*N, "--draws", "1", "--sigma", "1"], 20000),
    "K1-gauss": ([*N, "--draws", "1", "--noise", "gauss"], 20000),
    # Equal noisy evaluations to K4 and K16 at 20k: one draw, 4x and 16x the examples.
    "K1-80k": ([*N, "--draws", "1"], 80000),
    "K1-320k": ([*N, "--draws", "1"], 320000),
}

def job(tag, args):
    path = OUT / f"{tag}.json"
    if not path.exists():
        out = subprocess.run([BIN, "--data", DATA, "--optimizer", "carry-sgd", *args],
                             capture_output=True, text=True, check=True)
        path.write_text(out.stdout)
    return tag, json.loads(path.read_text())

with cf.ThreadPoolExecutor(16) as ex:
    futs = [ex.submit(job, f"tune-{a}-s{s}", [*k, "--lr", str(s), "--seed", "101", "--steps",
                      str(st // 2), "--eval-offset", "0", "--eval-limit", "1000"])
            for a, (k, st) in ARMS.items() for s in SHIFTS]
    tune = dict(f.result() for f in cf.as_completed(futs))
    selected = {}
    for a in ARMS:
        best = min(SHIFTS, key=lambda s: tune[f"tune-{a}-s{s}"]["mean_loss"])
        if best in (SHIFTS[0], SHIFTS[-1]):
            print(f"WARNING {a}: selected shift {best} is a grid edge", file=sys.stderr)
        selected[a] = best
    futs = [ex.submit(job, f"eval-{a}-{seed}", [*ARMS[a][0], "--lr", str(selected[a]),
                      "--seed", str(seed), "--steps", str(ARMS[a][1])])
            for a in ARMS for seed in range(1, 6)]
    for f in cf.as_completed(futs):
        f.result()

(OUT / "protocol.json").write_text(json.dumps({
    "arms": ARMS, "lr_shift_grid": SHIFTS, "selected": selected,
    "tune": "seed 101, half the evaluation steps, validation 0..999",
    "eval": "seeds 1-5, validation 1000..4999",
    "binary_sha256": hashlib.sha256(Path(BIN).read_bytes()).hexdigest(),
    "quality": "UNKNOWN"}, indent=1))
print(json.dumps(selected))

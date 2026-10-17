#!/usr/bin/env python3
"""Width/depth scaling of fused forward-only training (ternary_deep).

For each network: exact-gradient carry at 20k examples, and the fused single-draw noise rule at
20k, 80k and 320k examples. Tune each arm's lr shift on seed 101 at half its horizon (validation
0-999), evaluate on seeds 1-5 (validation 1000-4999). The data multiplier is how many more
examples the fused rule needs to match the exact arm's loss.
"""
import concurrent.futures as cf, hashlib, json, os, subprocess, sys
from pathlib import Path

BIN, DATA, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
SHIFTS = list(range(1, 10))
NETS = {"w128": "784,128,10", "w512": "784,512,10", "d256x2": "784,256,256,10"}
ARMS = {}
for n, sizes in NETS.items():
    ARMS[f"{n}-exact"] = (["--sizes", sizes], 20000)
    for steps in (20000, 80000, 320000):
        ARMS[f"{n}-K1-{steps // 1000}k"] = (["--sizes", sizes, "--grad", "noise", "--draws", "1"], steps)

def job(tag, args):
    path = OUT / f"{tag}.json"
    if not path.exists():
        out = subprocess.run([BIN, "--data", DATA, *args], capture_output=True, text=True, check=True)
        path.write_text(out.stdout)
    return tag, json.loads(path.read_text())

workers = int(os.environ.get("WORKERS", "12"))
with cf.ThreadPoolExecutor(workers) as ex:
    # Longest runs first so the pool drains evenly.
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
    "nets": NETS, "arms": ARMS, "lr_shift_grid": SHIFTS, "selected": selected,
    "tune": "seed 101, half the evaluation steps, validation 0..999",
    "eval": "seeds 1-5, validation 1000..4999",
    "binary_sha256": hashlib.sha256(Path(BIN).read_bytes()).hexdigest(),
    "quality": "UNKNOWN"}, indent=1))
print(json.dumps(selected))

#!/usr/bin/env python3
"""Update-rule rungs on carry-sgd (ladder axis U): keep-remainder %, leak, rounding, width.

Protocol as the carry campaign: tune each arm's lr shift (and nothing else) on seed 101, 10,000
steps, validation 0-999; evaluate the selected setting on seeds 1-5, 20,000 examples,
validation 1000-4999. Every arm differs from carry-sgd in exactly one knob (width arms change
storage width + units per trit together, crossed with rounding).
"""
import concurrent.futures as cf, hashlib, json, subprocess, sys
from pathlib import Path

BIN, DATA, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
SHIFTS = [2, 3, 4, 5, 6]
ARMS = {
    "base": [],
    "keep0": ["--keep-pct", "0"],
    "keep50": ["--keep-pct", "50"],
    "leak4": ["--leak", "4"], "leak8": ["--leak", "8"],
    "leak12": ["--leak", "12"], "leak16": ["--leak", "16"],
    "nearest": ["--round", "nearest"], "stochastic": ["--round", "stochastic"],
}
for w, u in [(16, 12), (8, 4), (8, 6)]:
    for r in ["floor", "nearest", "stochastic"]:
        ARMS[f"i{w}u{u}-{r}"] = ["--width", str(w), "--units-log2", str(u), "--round", r]

def run(args):
    out = subprocess.run([BIN, "--data", DATA, "--optimizer", "carry-sgd", *args],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)

def job(tag, args):
    path = OUT / f"{tag}.json"
    if not path.exists():
        path.write_text(json.dumps(run(args)))
    return tag, json.loads(path.read_text())

with cf.ThreadPoolExecutor(16) as ex:
    tune = {}
    futs = [ex.submit(job, f"tune-{a}-s{s}", [*k, "--lr", str(s), "--seed", "101", "--steps",
                      "10000", "--eval-offset", "0", "--eval-limit", "1000"])
            for a, k in ARMS.items() for s in SHIFTS]
    for f in cf.as_completed(futs):
        tag, r = f.result()
        tune[tag] = r
    selected = {}
    for a in ARMS:
        best = min(SHIFTS, key=lambda s: tune[f"tune-{a}-s{s}"]["mean_loss"])
        if best in (SHIFTS[0], SHIFTS[-1]):
            print(f"WARNING {a}: selected shift {best} is a grid edge", file=sys.stderr)
        selected[a] = best
    futs = [ex.submit(job, f"eval-{a}-{seed}", [*ARMS[a], "--lr", str(selected[a]),
                      "--seed", str(seed)])
            for a in ARMS for seed in range(1, 6)]
    for f in cf.as_completed(futs):
        f.result()

(OUT / "protocol.json").write_text(json.dumps({
    "arms": ARMS, "lr_shift_grid": SHIFTS, "selected": selected,
    "tune": "seed 101, 10000 steps, validation 0..999",
    "eval": "seeds 1-5, 20000 steps, validation 1000..4999",
    "binary_sha256": hashlib.sha256(Path(BIN).read_bytes()).hexdigest(),
    "quality": "UNKNOWN"}, indent=1))
print(json.dumps(selected))

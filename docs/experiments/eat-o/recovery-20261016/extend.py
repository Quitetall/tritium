#!/usr/bin/env python3
"""Grid extension for arms that selected the top edge (shift 11): tune shifts 12-14, re-select,
and re-evaluate any arm whose selection moved. Superseded eval runs are kept as eval-*.edge11.json."""
import concurrent.futures as cf, json, os, subprocess, sys
from pathlib import Path

BIN, DATA, OUT, CACHE = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
P = json.loads((OUT / "protocol.json").read_text())
BASE = P["base"][:-1] + [CACHE]
EXTRA = [12, 13, 14]
edge = [a for a, s in P["selected"].items() if s == max(P["lr_shift_grid"])]

def run(path, args):
    if not path.exists():
        out = subprocess.run([BIN, "--data", DATA, *BASE, *args], capture_output=True, text=True, check=True)
        path.write_text(out.stdout)
    return json.loads(path.read_text())

with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", "12"))) as ex:
    futs = {ex.submit(run, OUT / f"tune-{a}-s{s}.json", [*P["arms"][a][0], "--lr", str(s), "--seed", "101",
                      "--steps", str(P["arms"][a][1] // 2), "--eval-offset", "0", "--eval-limit", "1000"]): (a, s)
            for a in edge for s in EXTRA}
    for f in cf.as_completed(futs):
        f.result()
    moved = {}
    for a in edge:
        grid = P["lr_shift_grid"] + EXTRA
        best = min(grid, key=lambda s: json.loads((OUT / f"tune-{a}-s{s}.json").read_text())["mean_loss"])
        if best == grid[-1]:
            print(f"WARNING {a}: still at the extended edge {best}", file=sys.stderr)
        if best != P["selected"][a]:
            moved[a] = best
    futs = []
    for a, s in moved.items():
        for seed in range(1, 6):
            old = OUT / f"eval-{a}-{seed}.json"
            if old.exists():
                old.rename(OUT / f"eval-{a}-{seed}.edge11.json")
            futs.append(ex.submit(run, old, [*P["arms"][a][0], "--lr", str(s), "--seed", str(seed),
                                             "--steps", str(P["arms"][a][1])]))
    for f in cf.as_completed(futs):
        f.result()
P["selected"].update(moved)
P["grid_extension"] = {"arms": edge, "extra_shifts": EXTRA, "moved": moved}
(OUT / "protocol.json").write_text(json.dumps(P, indent=1))
print(json.dumps(moved))

#!/usr/bin/env python3
"""Residual-carry control campaign, same protocol as the EAT-O classifier campaign.

Tune on seed 101 (10,000 steps, validation 0..999, minimum mean squared loss), then five paired
seeds (20,000 steps, validation 1000..4999). Writes new artifacts only; each run's JSON is kept.
"""
import argparse, concurrent.futures as cf, json, pathlib, statistics, subprocess, hashlib

GRIDS = {"carry-adam": [7, 66, 655, 6554], "carry-sgd": [12, 16, 20, 24, 28]}


def run(binary, data, out, name, args):
    path = out / f"{name}.json"
    res = subprocess.run([binary, "--data", data, *map(str, args)], capture_output=True, text=True, check=True)
    path.write_text(res.stdout)
    return json.loads(res.stdout)


def main(a):
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    binary = str(a.binary)
    (out / "protocol.json").write_text(json.dumps({
        "grids": GRIDS, "tune_seed": 101, "tune_steps": 10000, "tune_eval": "validation 0..999",
        "seeds": [1, 2, 3, 4, 5], "steps": 20000, "eval": "validation 1000..4999",
        "binary_sha256": hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest(),
        "quality": "UNKNOWN"}, indent=2))
    jobs, results = {}, {}
    with cf.ThreadPoolExecutor(a.parallel) as pool:
        for opt, grid in GRIDS.items():
            for lr in grid:
                jobs[pool.submit(run, binary, a.data, out, f"tune-{opt}-{lr}",
                                 ["--optimizer", opt, "--lr", lr, "--seed", 101, "--steps", 10000,
                                  "--eval-offset", 0, "--eval-limit", 1000])] = (opt, lr)
        for f in cf.as_completed(jobs):
            opt, lr = jobs[f]
            results.setdefault(opt, []).append((f.result()["mean_loss"], lr, f.result()["accuracy"]))
    selected = {opt: min(rows)[1] for opt, rows in results.items()}
    (out / "selected.json").write_text(json.dumps({"selected": selected, "tuning": results}, indent=2))
    evals = {}
    with cf.ThreadPoolExecutor(a.parallel) as pool:
        futs = {pool.submit(run, binary, a.data, out, f"eval-{opt}-{seed}",
                            ["--optimizer", opt, "--lr", lr, "--seed", seed, "--steps", 20000,
                             "--eval-offset", 1000, "--eval-limit", 4000]): (opt, seed)
                for opt, lr in selected.items() for seed in range(1, 6)}
        for f in cf.as_completed(futs):
            opt, seed = futs[f]
            evals.setdefault(opt, {})[seed] = f.result()
    summary = {}
    for opt, rows in evals.items():
        acc = [rows[s]["accuracy"] * 100 for s in range(1, 6)]
        loss = [rows[s]["mean_loss"] for s in range(1, 6)]
        summary[opt] = {"lr": selected[opt], "accuracy_mean": statistics.mean(acc),
                        "accuracy_sd": statistics.stdev(acc), "accuracies": acc,
                        "mean_loss": statistics.mean(loss),
                        "state_bytes_per_parameter": rows[1]["state_bytes_per_parameter"],
                        "transitions_mean": statistics.mean(rows[s]["transitions"] for s in range(1, 6)),
                        "reversals_mean": statistics.mean(rows[s]["reversals"] for s in range(1, 6))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"selected": selected, "tuning": results, "summary": summary}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=pathlib.Path, required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--parallel", type=int, default=12)
    main(p.parse_args())

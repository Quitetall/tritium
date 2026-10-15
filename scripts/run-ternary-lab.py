#!/usr/bin/env python3
"""Run the EAT-O preregistered small classifier experiment; writes new artifacts only.
Tuning uses seed 101 and first 1000 validation examples. Paired evaluation uses
seeds 1..5 and the disjoint remaining 4000 validation examples. Official test
set is reserved. Equal-work runs match counted linear contraction terms, not
hardware instructions or elapsed time; one-step overshoot is recorded.
"""
import argparse
import json
import math
import pathlib
import statistics
import subprocess
import sys
import shutil
import hashlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
# MNIST's rotating 128-coordinate tile must visit every weight four times before
# the transition controller can act. Reject vacuous non-smoke tuning searches.
MIN_TUNING_STEPS = 4 * ((784*128 + 128*10 + 127)//128)
VARIANTS = {
    "simple": ["--simple", "yes", "--hysteresis", "no", "--precision", "fixed8"],
    "rich": ["--history", "both"],
    "statistics": ["--history", "statistics"],
    "replay": ["--history", "replay"],
    "no-hysteresis": ["--history", "both", "--hysteresis", "no"],
    "fixed8": ["--history", "both", "--precision", "fixed8"],
    "integer32-reference": ["--history", "both", "--precision", "integer32", "--budget", str(24*(784*128+128*10))],
    "probe": ["--history", "both", "--route", "probe"],
}


def execute(command, path):
    with path.with_suffix(".log").open("x") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(path.read_text())


def paired(rows, comparator):
    if len(rows) != 5 or len(comparator) != 5:
        raise ValueError("paired comparison requires five seeds")
    for a, b in zip(rows, comparator):
        for key in ("seed", "evaluation_count", "evaluation_digest", "data_digest"):
            if a.get(key) != b.get(key):
                raise ValueError(f"mismatched paired {key}")
    differences = [a["loss_sum"]/a["evaluation_count"] - b["loss_sum"]/b["evaluation_count"]
                   for a, b in zip(rows, comparator)]
    mean = statistics.mean(differences)
    half = 2.776445105 * statistics.stdev(differences) / math.sqrt(5)
    accuracy_delta = statistics.mean(a["correct"]/a["evaluation_count"] - b["correct"]/b["evaluation_count"]
                                    for a, b in zip(rows, comparator))
    return {"mean_loss_difference": mean, "paired_t_95_ci": [mean-half, mean+half],
            "accuracy_difference": accuracy_delta,
            "advance": mean+half < 0 and accuracy_delta >= -0.01}


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    # Freeze the executable and Python baseline so a concurrent rebuild cannot mix implementations.
    binary = a.output / "ternary_lab"
    shutil.copy2(a.binary, binary)
    baseline_script = a.output / "baseline.py"
    shutil.copy2(ROOT / "scripts/ternary-lab-baseline.py", baseline_script)
    if a.smoke:
        a.steps = a.tune_steps = 4
    plan = {"variants": VARIANTS, "seeds": list(range(1, 6)), "tune_seed": 101,
            "threshold_grid": [0, 4, 16, 64], "lr_grid": [0.0001, 0.001, 0.01, 0.1],
            "steps": a.steps, "tune_steps": a.tune_steps, "smoke": a.smoke,
            "work_matching_tolerance_percent": 1,
            "evaluation": "validation examples 1000..4999", "quality": "UNKNOWN",
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "baseline_sha256": hashlib.sha256(baseline_script.read_bytes()).hexdigest()}
    (a.output / "protocol.json").write_text(json.dumps(plan, indent=2))
    integer = [str(binary), "--data", str(a.data.resolve())]
    baseline = [sys.executable, str(baseline_script), "--data", str(a.data.resolve())]
    tuned = {}
    for name in [*VARIANTS, "ste"]:
        candidates = []
        grid = plan["lr_grid"] if name == "ste" else plan["threshold_grid"]
        for j, setting in enumerate(grid):
            path = a.output / f"tune-{name}-{j}.json"
            command = baseline + ["--lr", str(setting)] if name == "ste" else integer + VARIANTS[name] + ["--threshold", str(setting)]
            row = execute(command + ["--seed", "101", "--steps", str(a.tune_steps),
                                      "--eval-limit", "8" if a.smoke else "1000", "--report", str(path)], path)
            candidates.append((row["loss_sum"]/row["evaluation_count"], j, setting))
        tuned[name] = min(candidates)[2]
    (a.output / "selected.json").write_text(json.dumps(tuned, indent=2))
    results = {}
    for regime in ("data", "work"):
        for name in [*VARIANTS, "ste"]:
            rows = []
            for seed in range(1, 6):
                path = a.output / f"{regime}-{name}-{seed}.json"
                command = baseline + ["--lr", str(tuned[name])] if name == "ste" else integer + VARIANTS[name] + ["--threshold", str(tuned[name])]
                command += ["--seed", str(seed), "--steps", str(a.steps), "--eval-offset", "1000",
                            "--eval-limit", "8" if a.smoke else "4000", "--report", str(path)]
                if regime == "work":
                    # Budget equal to STE's data-matched training contractions.
                    budget = a.steps * (2*(784*128+128*10)+128*10)
                    command += ["--work-limit", str(budget)]
                    # Allow enough steps for cheaper reference methods to reach the same budget.
                    command[command.index("--steps")+1] = str(a.steps*4)
                if name != "ste":
                    checkpoint = path.with_suffix(".checkpoint.json")
                    command += ["--checkpoint", str(checkpoint)]
                row = execute(command, path)
                if name != "ste":
                    subprocess.run([sys.executable, str(ROOT / "scripts/verify-ternary-lab.py"),
                                    str(checkpoint), str(path), "--data", str(a.data.resolve())], check=True,
                                   stdout=subprocess.DEVNULL)
                rows.append(row)
            results[regime, name] = rows
    summary = {regime: {name: paired(results[regime, name], results[regime, "simple"])
                        for name in [*VARIANTS, "ste"] if name != "simple"}
               for regime in ("data", "work")}
    summary["rich_vs_ste"] = {regime: paired(results[regime, "rich"], results[regime, "ste"])
                              for regime in ("data", "work")}
    summary["work_counts"] = {name: [row.get("contraction_terms", row.get("metrics", {}).get("contraction_terms"))
                                    for row in results["work", name]] for name in [*VARIANTS, "ste"]}
    summary["work_target"] = a.steps*(2*(784*128+128*10)+128*10)
    summary["work_overshoot"] = {name: [count-summary["work_target"] for count in counts]
                                 for name, counts in summary["work_counts"].items()}
    for name, comparison in summary["work"].items():
        comparable = all(abs(x) <= summary["work_target"]//100
                         for key in (name, "simple") for x in summary["work_overshoot"][key])
        comparison["work_comparable"] = comparable
        comparison["advance"] &= comparable
    comparable = all(abs(x) <= summary["work_target"]//100
                     for key in ("rich", "ste") for x in summary["work_overshoot"][key])
    summary["rich_vs_ste"]["work"]["work_comparable"] = comparable
    summary["rich_vs_ste"]["work"]["advance"] &= comparable
    summary["advance_to_language_model"] = (not a.smoke and summary["data"]["rich"]["advance"]
                                             and summary["work"]["rich"]["advance"])
    summary["quality"] = "UNKNOWN" if a.smoke else "research comparison only; inspect paired intervals and STE gap"
    if a.smoke:
        for item in summary["rich_vs_ste"].values():
            item["advance"] = False
        for regime in ("data", "work"):
            for item in summary[regime].values():
                item["advance"] = False
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=pathlib.Path, required=True)
    p.add_argument("--data", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--tune-steps", type=int, default=10000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.steps <= 0 or args.tune_steps <= 0:
        p.error("steps must be positive")
    if not args.smoke and args.tune_steps < MIN_TUNING_STEPS:
        p.error(f"non-smoke tuning needs at least {MIN_TUNING_STEPS} steps for four observations per weight")
    args.output = args.output.resolve()
    main(args)

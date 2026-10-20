#!/usr/bin/env python3
"""Freeze and run a bounded EAT-O coverage/replay experiment, without an LM gate.

Six recipes are selected using seed 101 and validation examples 0..999.
Only the selected recipe is evaluated on five seeds and examples 1000..4999.
All runs use exact incremental scoring; prior campaign artifacts stay immutable.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import pathlib
import platform
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def save(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2)


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = out / "ternary_lab"
    shutil.copy2(args.binary, binary)
    snapshot = out / "source"
    files = ["scripts/run-eat-o-followup.py", "scripts/run-ternary-lab.py",
             "scripts/verify-ternary-lab.py", "crates/tritium-train/examples/ternary_lab.rs",
             *[f"crates/tritium-train/examples/ternary_lab/{name}.rs"
               for name in ("engine", "model", "numeric")]]
    hashes = {}
    for name in files:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    recipes = {f"{history}-{coords}": ["--history", history, "--coordinates", str(coords),
                "--replay", "16", "--hysteresis", "no"]
               for history in ("statistics", "both") for coords in (128, 1024, 4096)}
    protocol = {
        "purpose": "exploratory equal-data coverage/replay search; no work-matched or LM qualification",
        "recipes": recipes, "threshold": 0, "tuning_steps": 10000, "evaluation_steps": 20000,
        "tuning_seed": 101, "seeds": list(range(1, 6)), "selection": "minimum tuning loss",
        "tuning_slice": [0, 1000], "evaluation_slice": [1000, 5000],
        "control": "seed 1, original full EAT-O with incremental scoring only",
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "source_sha256": hashes, "machine": platform.platform(),
        "cpu": platform.processor(), "python": sys.version,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(args.data.glob("train-*-idx*-ubyte"))},
        "workers": 2, "advance_to_language_model": False,
    }
    save(out / "protocol.json", protocol)
    verifier = snapshot / "scripts/verify-ternary-lab.py"

    def run(name, flags, seed, tuning=False):
        prefix = f"tune-{name}" if tuning else f"eval-{name}-{seed}"
        report = out / f"{prefix}.json"
        checkpoint = out / f"{prefix}.checkpoint.json"
        command = [str(binary), "--data", str(args.data.resolve()), "--incremental", "yes",
                   "--threshold", "0", "--seed", str(seed), "--steps", "10000" if tuning else "20000",
                   "--eval-limit", "1000" if tuning else "4000", "--eval-offset", "0" if tuning else "1000",
                   "--report", str(report), "--checkpoint", str(checkpoint), *flags]
        save(out / f"{prefix}.command.json", command)
        with (out / f"{prefix}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        checked = subprocess.check_output([sys.executable, str(verifier), str(checkpoint), str(report),
                                          "--data", str(args.data.resolve())], cwd=ROOT, text=True)
        receipt = json.loads(checked)
        assert receipt["arithmetic_and_report"] == "PASS"
        save(out / f"{prefix}.verification.json", receipt)
        row = json.loads(report.read_text())
        print(f"{prefix}: {row['correct']}/{row['evaluation_count']}; loss={row['loss_sum']/row['evaluation_count']:.2f}; work={row['metrics']['contraction_terms']}", flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {name: pool.submit(run, name, flags, 101, True) for name, flags in recipes.items()}
        tuning = {name: job.result() for name, job in jobs.items()}
        selected = min(tuning, key=lambda name: (tuning[name]["loss_sum"], name))
        save(out / "selected.json", {"recipe": selected, "flags": recipes[selected]})
        print("SELECTED " + selected, flush=True)
        jobs = [pool.submit(run, selected, recipes[selected], seed) for seed in range(1, 6)]
        control_job = pool.submit(run, "control", ["--history", "both"], 1)
        rows = [job.result() for job in jobs]
        control = control_job.result()

    old = args.previous.resolve()
    previous = [json.loads((old / f"data-rich-{seed}.json").read_text()) for seed in range(1, 6)]
    ste = [json.loads((old / f"data-ste-{seed}.json").read_text()) for seed in range(1, 6)]
    spec = importlib.util.spec_from_file_location("paired_analysis", snapshot / "scripts/run-ternary-lab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_model = json.loads((old / "data-rich-1.checkpoint.json").read_text())["engine"]["model"]
    control_model = json.loads((out / "eval-control-1.checkpoint.json").read_text())["engine"]["model"]
    assert old_model == control_model, "incremental control changed original model trajectory"
    assert control["correct"] == previous[0]["correct"] and control["loss_sum"] == previous[0]["loss_sum"]
    summary = {
        "selected": selected, "vs_previous_full": module.paired(rows, previous),
        "vs_previous_ste": module.paired(rows, ste),
        "selected_accuracy_percent": sum(r["correct"] for r in rows) / 200,
        "selected_work": [r["metrics"]["contraction_terms"] for r in rows],
        "previous_full_work": [r["metrics"]["contraction_terms"] for r in previous],
        "control_identical_model": True,
        "control_work_ratio": control["metrics"]["contraction_terms"] / previous[0]["metrics"]["contraction_terms"],
        "advance_to_language_model": False,
        "qualification": "exploratory equal-data comparison only; work-matched gate not executed",
    }
    # paired()'s per-contrast advance field is only a loss/accuracy criterion here.
    for key in ("vs_previous_full", "vs_previous_ste"):
        summary[key]["loss_accuracy_criterion"] = summary[key].pop("advance")
    save(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--previous", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    main(parser.parse_args())

#!/usr/bin/env python3
"""Generate/check Tritium's source-controlled SmolLM2 Colab tutorial."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


OUTPUT = Path("examples/colab/tritium-smollm2-v11.ipynb")


def notebook():
    cells = [
        nbformat.v4.new_markdown_cell(
            """# Tritium v1.1: FP model to ternary in under five minutes

## Goal

Convert pinned `HuggingFaceTB/SmolLM2-135M-Instruct` with additive PTQ, run one
full-model QAT step, checkpoint/resume, export compact Hugging Face and ONNX
artifacts, generate tokens, and emit one auditable receipt. First model download
is excluded from measured wall time.""",
            id="goal",
        ),
        nbformat.v4.new_markdown_cell(
            """## Setup

`TRITIUM_WHEEL` may name one local wheel or a directory containing exactly one
wheel. Public release defaults to the pinned PyPI candidate. No compiler or
source checkout is used.""",
            id="setup-heading",
        ),
        nbformat.v4.new_code_cell(
            """import os
import subprocess
import sys
from pathlib import Path

candidate = os.environ.get("TRITIUM_WHEEL", "pytritium==1.1.0rc1")
candidate_path = Path(candidate)
if candidate_path.is_dir():
    wheels = sorted(candidate_path.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"TRITIUM_WHEEL directory contains {len(wheels)} wheels")
    candidate = str(wheels[0].resolve())
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
    "--only-binary=:all:", "--force-reinstall", "--no-deps", candidate,
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
    "--only-binary=:all:", "transformers==5.5.3", "safetensors==0.8.0",
    "onnx==1.22.0", "onnxruntime==1.27.0", "onnxscript==0.7.1",
])
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
""",
            id="install",
        ),
        nbformat.v4.new_markdown_cell(
            """## Steps

One library entry point owns phase ordering and receipt generation. Model ID and
immutable Hub revision remain visible here.""",
            id="steps-heading",
        ),
        nbformat.v4.new_code_cell(
            """import json
import torch
from tritium.torch import (
    SMOLLM2_MODEL_ID,
    SMOLLM2_REVISION,
    run_smollm2_release_demo,
)

output_dir = Path(os.environ.get("TRITIUM_OUTPUT_DIR", "/content/tritium-smollm2-v11"))
receipt = run_smollm2_release_demo(
    output_dir,
    model_id=SMOLLM2_MODEL_ID,
    revision=SMOLLM2_REVISION,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_seconds=300.0,
)
print(json.dumps({
    "elapsed_seconds": receipt["elapsed_seconds_excluding_download"],
    "compression_ratio": receipt["storage"]["selected_dense_to_checkpoint_ratio"],
    "zero_rate": receipt["trits"]["zero_rate"],
    "generated_text": receipt["generated_text"],
    "receipt": str(output_dir / "receipt.json"),
}, indent=2))""",
            id="run",
        ),
        nbformat.v4.new_markdown_cell("## Checks", id="checks-heading"),
        nbformat.v4.new_code_cell(
            """assert receipt["passed"] is True
assert receipt["source_revision"] == SMOLLM2_REVISION
assert receipt["elapsed_seconds_excluding_download"] < 300.0
assert receipt["coverage"]["selected_parameters"] > 0
assert receipt["storage"]["compact_checkpoint_bytes"] < receipt["storage"]["selected_dense_bytes"]
assert 0.0 <= receipt["trits"]["zero_rate"] <= 1.0
assert receipt["qat_optimizer_state_entries"] > 0
assert receipt["onnx_artifact_id"].startswith("sha256:")
print("PASS: pinned SmolLM2 PTQ + QAT + HF + ORT release gate")""",
            id="checks",
        ),
        nbformat.v4.new_markdown_cell(
            """## Next steps

Inspect `receipt.json`, `native-hf/`, `onnx/`, and `qat-checkpoint/`. Production
quality claims require separate held-out evaluation and audited model-zoo
receipts; this notebook proves workflow, packaging, bounded time, and artifact
round trips.""",
            id="next-steps",
        ),
    ]
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )


def rendered() -> str:
    return nbformat.writes(notebook())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = rendered()
    if args.check:
        if args.output.read_text(encoding="utf-8") != value:
            parser.error(f"{args.output} is stale; regenerate it")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(value, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

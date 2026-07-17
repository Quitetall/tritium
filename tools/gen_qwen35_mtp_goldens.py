#!/usr/bin/env python3
"""Run the canonical packaged Qwen3.5 MTP oracle generator."""

from pathlib import Path
import runpy


GENERATOR = (
    Path(__file__).resolve().parents[1]
    / "crates"
    / "tritium-nn"
    / "oracle"
    / "gen_qwen35_mtp_goldens.py"
)

runpy.run_path(str(GENERATOR), run_name="__main__")

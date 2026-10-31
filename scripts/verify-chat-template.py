#!/usr/bin/env python3
"""Offline source-tokenizer differential probe. Never loads model weights."""

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--probe-bin", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="", USE_TORCH="0", USE_TF="0")
    import transformers
    from transformers import AutoTokenizer

    if transformers.__version__ != "5.5.3":
        raise RuntimeError("this reference probe requires pinned transformers 5.5.3")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.source_dir), local_files_only=True, trust_remote_code=False
    )
    fixtures = [
        ("single", [{"role": "user", "content": "Hi"}]),
        ("system-whitespace", [{"role": "system", "content": "  Keep spaces.\n"}, {"role": "user", "content": "  Test  "}]),
        ("unicode-and-literal-jinja", [{"role": "user", "content": "Héllo 🌍 {{ 2 + 2 }} <>&'"}]),
        ("multi-turn", [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "And now?"}]),
        ("reasoning-history", [{"role": "user", "content": "First?"}, {"role": "assistant", "content": "<think>private reasoning</think>\nAnswer."}, {"role": "user", "content": "Next?"}]),
        ("grouped-tool-responses", [{"role": "user", "content": "Results?"}, {"role": "tool", "content": "one"}, {"role": "tool", "content": "two"}]),
        ("invalid-role", [{"role": "invalid", "content": "Hi"}]),
    ]
    results = []
    for name, messages, thinking in [
        (name, messages, thinking)
        for name, messages in fixtures
        for thinking in [None, True, False]
    ]:
        if thinking is not None:
            name = f"{name}-thinking-{str(thinking).lower()}"
        options = {} if thinking is None else {"enable_thinking": thinking}
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **options)
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **options)
            ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            hf_error = None
        except Exception as error:
            hf_error = type(error).__name__
        probe = subprocess.run(
            [str(args.probe_bin.resolve()), str(args.source_dir / "tokenizer.json"),
             str(args.source_dir / "tokenizer_config.json"), json.dumps(messages, ensure_ascii=False)]
            + ([] if thinking is None else [str(thinking).lower()]),
            capture_output=True, text=True, timeout=30, check=False,
        )
        if hf_error:
            results.append({"name": name, "hf_error": hf_error, "rust_refused": probe.returncode != 0,
                            "pass": probe.returncode != 0})
        elif probe.returncode:
            results.append({"name": name, "pass": False, "rust_error": probe.stderr[:4096]})
        else:
            actual = json.loads(probe.stdout)
            results.append({"name": name, "pass": actual == {"prompt": prompt, "input_ids": ids},
                            "hf_prompt": prompt, "hf_ids": ids, "rust": actual})
    report = {
        "schema": "tritium.chat-template-differential.v1",
        "qualified": False,
        "transformers_version": transformers.__version__,
        "source_dir": str(args.source_dir.resolve()),
        "template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((args.source_dir / "tokenizer.json").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((args.source_dir / "tokenizer_config.json").read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(args.probe_bin.read_bytes()).hexdigest(),
        "results": results,
        "pass": all(result["pass"] for result in results),
    }
    # Exclusive creation: never overwrite prior observations or receipts.
    with args.output.open("x") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({"pass": report["pass"], "cases": len(results), "output": str(args.output)}))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

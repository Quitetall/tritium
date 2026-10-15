#!/usr/bin/env python3
"""Record fixed guest build inputs and bundled dependency license texts."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    guest = root / "crates/tritium-nn/chat-renderer"
    destination = root / "crates/tritium-nn/src/model"
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--manifest-path", str(guest / "Cargo.toml"),
         "--format-version", "1"], cwd=root, timeout=60, text=True,
    ))
    licenses = {}
    dependencies = []
    for package in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
        if package["name"] == "tritium-chat-renderer-guest":
            continue
        directory = Path(package["manifest_path"]).parent
        paths = sorted(set(directory.glob("LICENSE*")) | set(directory.glob("COPYING*")))
        if not paths or not package.get("license"):
            raise RuntimeError(f"missing declared license/text for {package['name']}")
        hashes = []
        for path in paths:
            if not path.is_file():
                continue
            text = path.read_text()
            sha = digest(path)
            licenses.setdefault(sha, {"text": text, "packages": []})["packages"].append(
                f"{package['name']}@{package['version']} {path.name}"
            )
            hashes.append(sha)
        dependencies.append({"name": package["name"], "version": package["version"],
                             "license": package["license"], "license_text_sha256": hashes})
    inputs = [guest / "Cargo.toml", guest / "Cargo.lock", guest / "src/lib.rs",
              root / "rust-toolchain.toml", root / "scripts/build-chat-renderer.sh",
              Path(__file__).resolve()]
    record = {
        "schema": "tritium.fixed-chat-renderer-build.v1",
        "qualified": False,
        "target": "wasm32-unknown-unknown",
        "wasm_sha256": digest(destination / "chat_renderer.wasm"),
        "inputs": {str(path.relative_to(root)): digest(path) for path in inputs},
        "dependencies_including_build_time": dependencies,
    }
    notices = "Generated fixed chat renderer dependency notices\n\nIncludes build-time dependencies conservatively. See chat_renderer.build.json.\n"
    for sha, item in sorted(licenses.items()):
        notices += "\n" + "\n".join(sorted(item["packages"])) + f"\nSHA256 {sha}\n\n" + item["text"] + "\n"
    outputs = {"chat_renderer.build.json": json.dumps(record, sort_keys=True, indent=2) + "\n",
               "chat_renderer.NOTICE.txt": notices.rstrip() + "\n"}
    for name, content in outputs.items():
        path = destination / name
        if args.write:
            path.write_text(content)
        elif not path.is_file() or path.read_text() != content:
            raise RuntimeError(f"generated fixed renderer record drift: {name}")
    print(f"fixed chat renderer record: {len(dependencies)} locked dependencies; {'generated' if args.write else 'checked'}")


if __name__ == "__main__":
    main()

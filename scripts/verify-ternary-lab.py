#!/usr/bin/env python3
"""Separately executed integer reference for EAT-O research checkpoints and evaluation reports.
Requires blake3. Does not certify model quality or production qualification.
"""
import argparse
import json
import pathlib
import struct

import blake3


def trunc(value, divisor):
    return (1 if value >= 0 else -1) * (abs(value) // divisor)


def forward(model, pixels):
    n, h, o = (model[k] for k in ("inputs", "hidden", "outputs"))
    weights = model["trits"]
    hidden = [max(0, trunc(sum(weights[r*n+c]*pixels[c] for c in range(n)),
                          1 << model["input_shift"])) for r in range(h)]
    output = [trunc(sum(weights[n*h+r*h+c]*hidden[c] for c in range(h)),
                    1 << model["output_shift"]) for r in range(o)]
    return output


def examples(data, split):
    if data is None:
        return [([255, 0] if i % 2 == 0 else [0, 255], i % 2) for i in range(32)]
    prefix = "train" if split == "validation" else "t10k"
    images = (data / f"{prefix}-images-idx3-ubyte").read_bytes()
    labels = (data / f"{prefix}-labels-idx1-ubyte").read_bytes()
    magic, count, rows, cols = struct.unpack(">4I", images[:16])
    assert (magic, rows, cols) == (2051, 28, 28)
    assert struct.unpack(">2I", labels[:8]) == (2049, count)
    assert len(images) == 16 + count*784 and len(labels) == 8 + count
    start = 55000 if split == "validation" else 0
    return [(images[16+i*784:16+(i+1)*784], labels[8+i]) for i in range(start, count)]


def verify(checkpoint, report, data):
    raw = checkpoint.read_bytes()
    saved = json.loads(raw)
    assert saved["schema"] == 1
    e = saved["engine"]
    m = e["model"]
    assert all(type(q) is int and q in (-1, 0, 1) for q in m["trits"])
    n = m["inputs"]*m["hidden"] + m["hidden"]*m["outputs"]
    assert len(m["trits"]) == n
    for key in ("observations", "age", "direction"):
        assert len(e[key]) == n
    assert m["version"] == e["metrics"]["transitions"]
    for block in e["blocks"]:
        bank = block["bank"]
        digits = bank["digits"]
        assert digits in (0, 8, 16, 24) and 0 <= bank["exponent"] <= 60
        maximum = (3**digits-1)//2 if digits else 2**31-1
        width = (2*maximum).bit_length()
        packed = int.from_bytes(bytes(bank["packed"]), "little")
        assert len(bank["packed"]) == (bank["len"]*width+7)//8
        for i in range(bank["len"]):
            value = ((packed >> (i*width)) & ((1 << width)-1)) - maximum
            assert -maximum <= value <= maximum
            if digits == 0:
                continue
            # Independent balanced-ternary carry/borrow reconstruction.
            remaining, rebuilt, power = value, 0, 1
            for _ in range(digits):
                digit = (remaining+1) % 3 - 1
                rebuilt += digit*power
                remaining = (remaining-digit)//3
                power *= 3
            assert remaining == 0 and rebuilt == value
    r = json.loads(report.read_text())
    assert r["checkpoint_digest"] == blake3.blake3(raw).hexdigest()
    assert r["metrics"] == e["metrics"] and r["config"] == e["config"]
    assert r["optimizer_replay_allocated_bytes"] <= e["config"]["budget"]
    assert r["accumulator_clips"] == 0
    offset = r.get("evaluation_offset", 0)
    xs = examples(data, r["split"])[offset:offset+r["evaluation_count"]]
    assert len(xs) == r["evaluation_count"]
    digest = blake3.blake3()
    loss = correct = 0
    for pixels, label in xs:
        digest.update(struct.pack("<Q", label))
        digest.update(struct.pack("<Q", len(pixels)))
        digest.update(bytes(pixels))
        output = forward(m, pixels)
        loss += sum((v-(256 if i == label else 0))**2 for i, v in enumerate(output))
        correct += max(range(len(output)), key=output.__getitem__) == label
    assert digest.hexdigest() == r["evaluation_digest"]
    assert (loss, correct) == (r["loss_sum"], r["correct"])
    # This is a source audit supplement, not a machine-code proof.
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = root / "crates/tritium-train/examples/ternary_lab"
    import re
    for path in sources.glob("*.rs"):
        code = "\n".join(line.split("//", 1)[0] for line in path.read_text().splitlines())
        assert not re.search(r"\b(?:f32|f64)\b", code), path
    return {"arithmetic_and_report": "PASS", "quality": "UNKNOWN",
            "examples": len(xs), "loss_sum": loss, "correct": correct}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("report", type=pathlib.Path)
    parser.add_argument("--data", type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.checkpoint, args.report, args.data), indent=2))

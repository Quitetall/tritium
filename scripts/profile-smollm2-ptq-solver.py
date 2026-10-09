#!/usr/bin/env python3
"""Capture one pinned SmolLM2 PTQ row group for the Rust solver phase profiler."""

from __future__ import annotations

import argparse
import hashlib
import struct
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tritium.torch import TernaryConfig, calibrate, convert, prepare
from tritium.torch.tutorial import SMOLLM2_MODEL_ID, SMOLLM2_REVISION


_LAYER_PATH = "model.layers.0.mlp.up_proj"
_ROWS = 256
_GROUP_SIZE = 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new binary fixture path (must not exist)")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow fetching the pinned model if it is not already cached",
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing fixture: {output}")

    local_files_only = not args.allow_download
    model = AutoModelForCausalLM.from_pretrained(
        SMOLLM2_MODEL_ID,
        revision=SMOLLM2_REVISION,
        dtype=torch.float32,
        local_files_only=local_files_only,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        SMOLLM2_MODEL_ID,
        revision=SMOLLM2_REVISION,
        local_files_only=local_files_only,
    )
    batch = tokenizer("Ternary models make efficient inference", return_tensors="pt")
    source_layer = model.get_submodule(_LAYER_PATH)
    captured_inputs: list[torch.Tensor] = []
    hook = source_layer.register_forward_pre_hook(
        lambda _module, inputs: captured_inputs.append(inputs[0].detach().cpu())
    )
    with torch.inference_mode():
        model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
    hook.remove()
    if len(captured_inputs) != 1:
        raise RuntimeError(f"expected one {_LAYER_PATH} input capture")

    source_weight = source_layer.weight.detach().to(device="cpu", dtype=torch.float32)
    source_bias = source_layer.bias
    projection = torch.nn.Linear(
        source_weight.shape[1],
        source_weight.shape[0],
        bias=source_bias is not None,
        dtype=torch.float32,
    )
    with torch.no_grad():
        projection.weight.copy_(source_weight)
        if source_bias is not None:
            projection.bias.copy_(source_bias.detach().to(device="cpu", dtype=torch.float32))
    prepared = prepare(
        projection,
        TernaryConfig.ptq(
            profile="compact-v1", target_modules=("Linear",)
        ),
        inplace=True,
    )

    with tempfile.TemporaryDirectory(prefix="tritium-smollm2-profile-") as work_dir:
        calibration_started = time.perf_counter()
        calibration = calibrate(
            prepared,
            captured_inputs,
            evidence_dir=Path(work_dir) / "calibration",
        )
        calibration_seconds = time.perf_counter() - calibration_started
        record = next(
            (
                item
                for item in calibration.records
                if item.module == ""
            ),
            None,
        )
        if record is None:
            raise RuntimeError("calibration did not cover the selected projection")
        parameter = dict(prepared.model.named_parameters(remove_duplicate=False)).get(
            record.weight_aliases[0]
        )
        if parameter is None or parameter.ndim != 2:
            raise RuntimeError("calibration weight alias did not resolve to a matrix")
        if parameter.shape[0] < _ROWS or parameter.shape[1] % _GROUP_SIZE:
            raise RuntimeError(f"unexpected selected matrix shape: {tuple(parameter.shape)}")

        conversion_started = time.perf_counter()
        conversion = convert(
            prepared,
            calibration,
            work_dir=Path(work_dir) / "conversion",
            max_working_bytes=256 * 1024 * 1024,
        )
        conversion_seconds = time.perf_counter() - conversion_started

        curvature_payload = (calibration.evidence_dir / record.file).read_bytes()
        curvature = torch.frombuffer(bytearray(curvature_payload), dtype=torch.float64)
        curvature = curvature / record.samples
        group_diagonal = curvature[:_GROUP_SIZE].contiguous()
        row_group = parameter[:_ROWS, :_GROUP_SIZE].detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        if not bool(torch.isfinite(row_group).all()) or not bool(
            torch.isfinite(group_diagonal).all()
        ):
            raise RuntimeError("profile fixture contains a non-finite model/evidence value")

        data = bytearray(b"TRIPRF01")
        data.extend(struct.pack("<II", _ROWS, _GROUP_SIZE))
        data.extend(row_group.numpy().astype("<f4", copy=False).tobytes(order="C"))
        data.extend(group_diagonal.numpy().astype("<f8", copy=False).tobytes(order="C"))
        artifact_weight = conversion.weight(record.weight_aliases[0])
        fitted_weight_mse = artifact_weight.weighted_mse

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(data)
        stream.flush()

    print(
        " ".join(
            (
                f"fixture_sha256={hashlib.sha256(data).hexdigest()}",
                f"model={SMOLLM2_MODEL_ID}",
                f"revision={SMOLLM2_REVISION}",
                f"layer={_LAYER_PATH}",
                f"shape={_ROWS}x{_GROUP_SIZE}",
                f"projection_shape={tuple(parameter.shape)}",
                f"calibration_seconds={calibration_seconds:.3f}",
                f"public_convert_seconds={conversion_seconds:.3f}",
                f"ptq_artifact_id={conversion.artifact_id}",
                f"weighted_mse={fitted_weight_mse:.9g}",
                f"calibration_id={calibration.evidence_id}",
                f"source_digest={calibration.source_model_digest}",
                f"output={output}",
            )
        )
    )


if __name__ == "__main__":
    main()

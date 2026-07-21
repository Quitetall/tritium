from pathlib import Path

import tritium.onnx as onnx


def _digests():
    return {
        "language_blake3": "11" * 32,
        "mtp_blake3": "22" * 32,
        "weights_blake3": "33" * 32,
    }


def test_verify_forwards_paths_authority_and_limits(monkeypatch, tmp_path: Path):
    observed = {}
    receipt = object()

    def verify(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return receipt

    monkeypatch.setattr(onnx._tritium, "verify_qwen35_onnx_bundle", verify, raising=False)
    result = onnx.verify_qwen35_bundle(
        tmp_path / "language.onnx",
        tmp_path / "mtp.onnx",
        tmp_path / "weights.bin",
        max_graph_bytes=17,
        max_weights_bytes=19,
        **_digests(),
    )

    assert result is receipt
    assert observed["args"] == (
        str(tmp_path / "language.onnx"),
        str(tmp_path / "mtp.onnx"),
        str(tmp_path / "weights.bin"),
    )
    assert observed["kwargs"] == {**_digests(), "max_graph_bytes": 17, "max_weights_bytes": 19}


def test_stage_keeps_output_separate_and_propagates_native_failure(monkeypatch, tmp_path: Path):
    def stage(*args, **kwargs):
        assert args[-1] == str(tmp_path / "published")
        raise RuntimeError("native admission failed")

    monkeypatch.setattr(onnx._tritium, "stage_qwen35_onnx_bundle", stage, raising=False)
    try:
        onnx.stage_qwen35_bundle(
            tmp_path / "language.onnx",
            tmp_path / "mtp.onnx",
            tmp_path / "weights.bin",
            tmp_path / "published",
            **_digests(),
        )
    except RuntimeError as error:
        assert str(error) == "native admission failed"
    else:
        raise AssertionError("native failure was swallowed")


def test_native_export_rejects_invalid_bundle_without_publication(tmp_path: Path):
    output = tmp_path / "published"
    try:
        onnx._tritium.export_qwen35_onnx_bundle(
            str(tmp_path / "missing"), str(output)
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid bundle was exported")
    assert not output.exists()

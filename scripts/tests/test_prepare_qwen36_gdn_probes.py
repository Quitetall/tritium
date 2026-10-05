from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "prepare-qwen36-gdn-probes.py"
SPEC = importlib.util.spec_from_file_location("prepare_qwen36_gdn_probes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(root: Path, *, rank_one: str | None = None) -> list[str]:
    layer_types = ["linear_attention" if layer % 4 != 3 else "full_attention" for layer in range(64)]
    (root / "config.json").write_text(json.dumps({
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 64,
            "layer_types": layer_types,
        }
    }))
    probes = [
        ("deltanet", "qkv", "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
        ("deltanet", "output", "model.language_model.layers.1.linear_attn.out_proj.weight"),
        ("deltanet", "gate_up", "model.language_model.layers.2.mlp.gate_proj.weight"),
        ("deltanet", "down", "model.language_model.layers.4.mlp.down_proj.weight"),
        ("full_attention", "qkv", "model.language_model.layers.3.self_attn.q_proj.weight"),
        ("full_attention", "output", "model.language_model.layers.7.self_attn.o_proj.weight"),
        ("full_attention", "gate_up", "model.language_model.layers.11.mlp.gate_proj.weight"),
        ("full_attention", "down", "model.language_model.layers.15.mlp.down_proj.weight"),
    ]
    shard = "model-00001-of-00001.safetensors"
    header = {
        name: {
            "dtype": "BF16",
            "shape": [32] if name == rank_one else [8, 8],
            "data_offsets": [0, 0],
        }
        for _, _, name in probes
    }
    duplicate_layer_output = "model.language_model.layers.0.linear_attn.out_proj.weight"
    header[duplicate_layer_output] = {
        "dtype": "BF16", "shape": [8, 8], "data_offsets": [0, 0]
    }
    for index in range(497):
        header[f"model.language_model.fixture_matrix_{index:03d}.weight"] = {
            "dtype": "BF16", "shape": [8, 8], "data_offsets": [0, 0]
        }
    for index in range(693):
        header[f"fixture.extra.{index:04d}"] = {
            "dtype": "BF16", "shape": [8], "data_offsets": [0, 0]
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    (root / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    weight_map = {name: shard for _, _, name in probes}
    weight_map[duplicate_layer_output] = shard
    for index in range(497):
        name = f"model.language_model.fixture_matrix_{index:03d}.weight"
        weight_map[name] = shard
    for index in range(693):
        weight_map[f"fixture.extra.{index:04d}"] = shard
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return [f"{family}/{tensor_class}={name}" for family, tensor_class, name in probes]


class QwenGdnProbePreflightTests(unittest.TestCase):
    def test_prepares_all_eight_probes_without_claiming_measurement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selections = fixture(root)
            prepared = MODULE.prepare(root, selections)
        self.assertEqual(prepared["state"], "prepared-not-measured")
        self.assertEqual(prepared["evidence_scope"], "local-config-index-and-safetensors-header-only")
        self.assertEqual(len(prepared["probes"]), 8)
        self.assertTrue(all(probe["shape"] == [8, 8] for probe in prepared["probes"]))
        self.assertEqual(
            {probe["tensor_name"]: probe["tensor_index"] for probe in prepared["probes"]}
            ["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"],
            497,
        )

    def test_rejects_wrong_family_class_and_non_matrix_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selections = fixture(root)
            selections[0] = selections[0].replace("deltanet/qkv=", "deltanet/output=")
            with self.assertRaisesRegex(MODULE.PreflightError, "unique entry"):
                MODULE.prepare(root, selections)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selections = fixture(root, rank_one="model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
            with self.assertRaisesRegex(MODULE.PreflightError, "506 language/MTP matrices"):
                MODULE.prepare(root, selections)

    def test_rejects_duplicate_layer_and_incomplete_selection_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selections = fixture(root)
            selections[1] = selections[1].replace("layers.1.", "layers.0.")
            with self.assertRaisesRegex(MODULE.PreflightError, "four distinct block layers"):
                MODULE.prepare(root, selections)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selections = fixture(root)[:-1]
            with self.assertRaisesRegex(MODULE.PreflightError, "exactly eight"):
                MODULE.prepare(root, selections)


if __name__ == "__main__":
    unittest.main()

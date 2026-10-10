from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "prepare-qwen36-gdn-probes.py"
SPEC = importlib.util.spec_from_file_location("prepare_qwen36_gdn_probes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(
    root: Path,
    *,
    rank_one: str | None = None,
    malformed_matrix: str | None = None,
    index_shard: str | None = None,
    final_full_attention: bool = False,
) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
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
    if final_full_attention:
        probes[7] = (
            "full_attention",
            "down",
            "model.language_model.layers.63.mlp.down_proj.weight",
        )
    shard = "model-00001-of-00001.safetensors"
    header = {
        name: {
            "dtype": "BF16",
            "shape": (
                [32]
                if name == rank_one
                else [0, 8]
                if name == malformed_matrix
                else [8, 8]
            ),
            "data_offsets": [0, 0],
        }
        for _, _, name in probes
    }
    duplicate_layer_output = "model.language_model.layers.0.linear_attn.out_proj.weight"
    header[duplicate_layer_output] = {
        "dtype": "BF16", "shape": [8, 8], "data_offsets": [0, 0]
    }
    for index in range(497):
        name = f"model.language_model.fixture_matrix_{index:03d}.weight"
        header[f"model.language_model.fixture_matrix_{index:03d}.weight"] = {
            "dtype": "BF16",
            "shape": [0, 8] if name == malformed_matrix else [8, 8],
            "data_offsets": [0, 0],
        }
    for index in range(693):
        header[f"fixture.extra.{index:04d}"] = {
            "dtype": "BF16", "shape": [8], "data_offsets": [0, 0]
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    (root / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    weight_map = {name: index_shard or shard for _, _, name in probes}
    weight_map[duplicate_layer_output] = index_shard or shard
    for index in range(497):
        name = f"model.language_model.fixture_matrix_{index:03d}.weight"
        weight_map[name] = index_shard or shard
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
        self.assertEqual(prepared["schema"], "tritium.qwen36-gdn-probe-preflight.v2")
        self.assertEqual(prepared["evidence_scope"], "local-config-index-and-safetensors-header-only")
        self.assertEqual(prepared["state_layer_rule"], MODULE.STATE_LAYER_RULE)
        self.assertEqual(len(prepared["probes"]), 8)
        self.assertTrue(all(probe["shape"] == [8, 8] for probe in prepared["probes"]))
        by_name = {probe["tensor_name"]: probe for probe in prepared["probes"]}
        self.assertEqual(
            by_name["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]["state_layer"],
            0,
        )
        self.assertEqual(
            by_name["model.language_model.layers.15.mlp.down_proj.weight"]["state_layer"],
            16,
        )
        self.assertEqual(
            {probe["tensor_name"]: probe["tensor_index"] for probe in prepared["probes"]}
            ["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"],
            497,
        )

    def test_output_is_canonical_private_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selections = fixture(root / "model")
            prepared = MODULE.prepare(root / "model", selections)
            output = root / "preflight.json"

            MODULE.write_preflight(output, prepared)

            self.assertEqual(output.read_bytes(), MODULE.canonical(prepared) + b"\n")
            self.assertEqual(json.loads(output.read_text()), prepared)
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                MODULE.write_preflight(output, {"state": "replacement"})
            self.assertEqual(output.read_bytes(), MODULE.canonical(prepared) + b"\n")

    def test_cli_persists_the_same_preflight_it_prints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selections = fixture(root / "model")
            expected = MODULE.prepare(root / "model", selections)
            output = root / "cli-preflight.json"
            argv = [str(SCRIPT), str(root / "model"), "--output", str(output)]
            for selection in selections:
                argv.extend(("--probe", selection))
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(MODULE.main(), 0)
            self.assertEqual(json.loads(output.read_text()), expected)

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

    def test_rejects_full_attention_probe_without_following_deltanet_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selections = fixture(root, final_full_attention=True)
            with self.assertRaisesRegex(MODULE.PreflightError, "no following DeltaNet"):
                MODULE.prepare(root, selections)

    def test_rejects_malformed_geometry_outside_selected_probes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selections = fixture(
                root,
                malformed_matrix="model.language_model.fixture_matrix_000.weight",
            )
            with self.assertRaisesRegex(MODULE.PreflightError, "invalid geometry"):
                MODULE.prepare(root, selections)

    def test_rejects_windows_style_shard_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selections = fixture(root, index_shard=r"..\outside.safetensors")
            with self.assertRaisesRegex(MODULE.PreflightError, "unsafe shard path"):
                MODULE.prepare(root, selections)


if __name__ == "__main__":
    unittest.main()

"""Real Rust-published corpus against the installed wheel, never fabricated records."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import pytest
import torch
from torch import nn
from tritium import NativeModelConversion, NativeModelRefinement
from tritium.torch import TernaryConfig, prepare, convert_native_module, load, export

CASES = [f"linear-{precision}-{codec}-{basis}" for precision in ("F16", "F32")
         for codec in ("D2", "B3") for basis in ("Identity", "SignedRht")] + ["hf"]

class TiedState(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(32, 1)
        self.b = nn.Linear(32, 1)
        self.b.weight, self.b.bias = self.a.weight, self.a.bias
        self.norm = nn.LayerNorm(32)
        self.register_buffer("counter", torch.tensor(7))
        self.register_buffer("mask", torch.tensor([True, False]))
        self.register_buffer("mask_alias", self.mask)
        self.register_buffer("scalar", torch.tensor(-0.0))
    def forward(self, x):
        return self.a(self.norm(x)) + self.b(self.norm(x))

def _shell(root, case):
    if case != "hf":
        return TiedState()
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig.from_dict(json.loads((root / "hf-config.json").read_text()))
    return LlamaForCausalLM(config)

def _prepare(root):
    from tritium.torch.native_source import _inventory
    from safetensors.torch import save_file
    root = Path(root)
    for index, case in enumerate(CASES):
        torch.manual_seed(101 + index)
        directory = root / case; directory.mkdir()
        if case == "hf":
            from transformers import LlamaConfig, LlamaForCausalLM
            model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=16, tie_word_embeddings=True))
            (directory / "hf-config.json").write_text(model.config.to_json_string())
            precision, codec, basis = "F32", "B3", "Identity"
        else:
            model = TiedState(); _, precision, codec, basis = case.split("-")
        prepared = prepare(model, TernaryConfig.ptq(profile="compact-v1", planes=2,
            target_modules=("Linear", "Embedding")), inplace=True)
        actions = {}
        for owner, _, tensor, _, targeted, _ in _inventory(prepared):
            if not targeted: continue
            rows, cols = tensor.shape
            rotation = "Identity" if basis == "Identity" else {"SignedRht": dict(block=32, seed=13, domain=29)}
            actions[owner] = {"Fit": dict(metric="Identity", plan=dict(
                layout=dict(rows=rows, cols=cols, tile=256, group=32, max_planes=2,
                    allocation="Uniform", codec=codec, basis=rotation, transport="Raw",
                    law=dict(anchor="Group", relation="Free", precision=precision)),
                profile={"JointIndependentGroups": dict(max_iterations=4, ridge=1e-8,
                    em_restarts=1, ridge_condition_limit=1e6, softened_relay=False, modulated_relay=False)},
                levels=[dict(name="compact", planes=1), dict(name="full", planes=2)],
                limits=dict(max_weights=rows*cols, max_metric_entries=cols*cols, max_output_trits=3*rows*cols)))}
        work = directory / "work"; work.mkdir()
        for name in ("config.json", "tokenizer.json"): (directory / name).write_bytes(b"{}")
        parent = convert_native_module(prepared, directory / "parent", "python-source",
            actions=actions, levels=["compact", "full"], work_dir=work,
            limits=dict(max_source_bytes=4_000_000, max_asset_bytes=1_000_000,
                        max_retained_blob_bytes=4_000_000, max_package_bytes=16_000_000),
            asset_paths={name: directory / name for name in ("config.json", "tokenizer.json")})
        save_file({name: value.detach().contiguous().clone() for name, value in model.state_dict().items()},
                  directory / "source.safetensors")
        optimizer = dict(lr=0.01, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.0)
        request = dict(parent=json.loads(parent.native_refinement_parent("full")), child_level="pv-child",
            owners=sorted(actions), recipe={"AlternatingPv": dict(continuous_optimizer=optimizer,
                code_optimizer={**optimizer, "lr": 0.6}, max_code_change_fraction=1.0,
                max_relative_code_change=None)}, objective=[7]*32, teacher_source=None,
            training=[dict(content=[10+i]*32, weight=1.0) for i in range(4)],
            validation=[dict(content=[30]*32, weight=1.0)],
            limits=dict(max_state_bytes=16_000_000, max_workspace_bytes=16_000_000,
                        max_checkpoint_bytes=16_000_000, max_package_bytes=16_000_000))
        (directory / "plan.json").write_text(json.dumps(request))
        assert not list(work.iterdir())

@pytest.fixture
def corpus():
    value = os.environ.get("TRITIUM_PV_FIXTURES")
    if not value: pytest.skip("requires the explicit Rust P/V producer/installed-wheel probe")
    return Path(value)

def _input(case):
    return torch.tensor([[1, 2, 3]]) if case == "hf" else torch.linspace(-1, 1, 64).reshape(2, 32)

def _forward(model, case):
    return model(_input(case), use_cache=False).logits if case == "hf" else model(_input(case))

def _expected(directory, case):
    from safetensors.torch import load_file
    from tritium.torch.qat import _state_slot
    model = _shell(directory, case); model.load_state_dict(load_file(directory / "source.safetensors"))
    catalog = json.loads(NativeModelConversion.open(str(directory / "parent")).asset("tritium-torch-state.json"))
    aliases = {entry["owner"]: entry["aliases"] for entry in catalog["state"]}
    for entry in json.loads((directory / "expected.json").read_text()):
        weight = nn.Parameter(torch.tensor(entry["values"], dtype=torch.float32).reshape(entry["rows"], entry["cols"]), requires_grad=False)
        for alias in aliases[entry["name"]]:
            module, kind, leaf, _ = _state_slot(model, alias)
            assert kind == "parameter"; module._parameters[leaf] = weight
    return model.eval()

@pytest.mark.parametrize("case", CASES)
def test_actual_rust_pv_child_load_export_replay_and_standalone_parity(corpus, tmp_path, case):
    directory = corpus / case
    parent = NativeModelConversion.open(str(directory / "parent"))
    child = NativeModelRefinement.open(str(directory / "child")); child.verify()
    assert child.recovery_recipe == "alternating-pv"
    assert child.admission == "record-consistent-unqualified" and not child.quality_qualified and not child.objective_authenticated
    assert child.parent_package_id == parent.package_id and child.parent_level == "full"
    assert sorted(p.name for p in (directory / "child").iterdir()) == ["checkpoint.tnpv", "evidence.jsonl", "gradients.tnpj", "model.trit"]
    audit = json.loads(child.verify_with_parent(parent, max_parent_bytes=64_000_000, max_replay_working_bytes=64_000_000))
    assert audit["scope"] == "parent-backed-numerical-replay" and audit["lineage_scope"] == "immediate-parent-only" and not audit["quality_qualified"]
    with pytest.raises(ValueError): NativeModelConversion.open(str(directory / "child"))
    loaded = load(child, model=_shell(directory, case), level="pv-child", load_run_id="pv-load", inplace=False).eval()
    expected = _expected(directory, case)
    with torch.no_grad():
        actual = _forward(loaded, case); torch.testing.assert_close(actual, _forward(expected, case), rtol=2e-4, atol=2e-4)
        tokens = loaded.generate(_input(case), max_new_tokens=3, do_sample=False, pad_token_id=0) if case == "hf" else None
    if case == "hf": assert loaded.model.embed_tokens.packed_weight is loaded.lm_head.packed_weight
    else:
        assert loaded.a.packed_weight is loaded.b.packed_weight and loaded.a.bias is loaded.b.bias
        assert loaded.mask is loaded.mask_alias and loaded.counter.item() == 7
        assert loaded.scalar.view(torch.int32).item() == -2147483648
    copied = export(child, tmp_path / "standalone")
    assert copied.recovery_recipe == "alternating-pv" and copied.package_id == child.package_id and copied.evidence_root == child.evidence_root
    for name in ("model.trit", "evidence.jsonl", "checkpoint.tnpv", "gradients.tnpj"):
        assert (Path(copied.output_dir)/name).read_bytes() == (directory/"child"/name).read_bytes()
    # A copy has no dependency on the original parent or child directories.
    hidden = []
    try:
        for name in ("parent", "child"):
            original = directory / name; absent = directory / (name + ".absent")
            original.rename(absent); hidden.append((absent, original))
        standalone = load(tmp_path / "standalone", model=_shell(directory, case), level="pv-child", load_run_id="pv-copy", inplace=False).eval()
        with torch.no_grad():
            assert torch.equal(_forward(standalone, case), actual)
            if case == "hf": assert torch.equal(standalone.generate(_input(case), max_new_tokens=3, do_sample=False, pad_token_id=0), tokens)
    finally:
        for absent, original in reversed(hidden): absent.rename(original)
    binding = json.loads(copied.native_refinement_parent("pv-child")); assert bytes(binding["package"]).hex() == child.package_id
    before = {p.name:p.read_bytes() for p in Path(copied.output_dir).iterdir()}
    with pytest.raises((ValueError, FileExistsError)): copied.export(str(tmp_path / "standalone"))
    with pytest.raises(ValueError): copied.export(str(tmp_path / "standalone" / "nested"))
    assert {p.name:p.read_bytes() for p in Path(copied.output_dir).iterdir()} == before

@pytest.mark.parametrize("name", ["model.trit", "evidence.jsonl", "checkpoint.tnpv", "gradients.tnpj"])
def test_changed_pv_files_and_tight_caps_are_rejected(corpus, tmp_path, name):
    source = corpus / CASES[0] / "child"; target = tmp_path / "changed"; shutil.copytree(source, target)
    child = NativeModelRefinement.open(str(target))
    with pytest.raises(ValueError): NativeModelRefinement.open(str(target), max_checkpoint_bytes=1)
    with pytest.raises(ValueError): NativeModelRefinement.open(str(target), max_journal_bytes=1)
    path = target / name; value = bytearray(path.read_bytes()); value[len(value)//2] ^= 1; path.write_bytes(value)
    with pytest.raises(ValueError): child.verify()
    with pytest.raises(ValueError): child.asset("tritium-torch-state.json")
    with pytest.raises(ValueError): child.native_refinement_parent("pv-child")
    with pytest.raises(ValueError): child.export(str(tmp_path / "invalid"))
    assert not (tmp_path / "invalid").exists()

def test_profile_companions_are_not_selected_by_existence_and_training_stays_guarded(corpus, tmp_path):
    source = corpus / CASES[0]; target = tmp_path / "mixed"; shutil.copytree(source / "child", target)
    for name in ("checkpoint.tnsr", "gradients.tnsj"): shutil.copyfile(source / "scale" / name, target / name)
    child = NativeModelRefinement.open(str(target)); assert child.recovery_recipe == "alternating-pv"
    (target / "checkpoint.tnpv").unlink()
    with pytest.raises(ValueError): NativeModelRefinement.open(str(target))
    scale = NativeModelRefinement.open(str(source / "scale")); assert scale.recovery_recipe == "scale-only"
    request = json.loads((source / "plan.json").read_text())
    from tritium.torch import native_scale_training
    parent = NativeModelConversion.open(str(source / "parent"))
    with pytest.raises(ValueError, match="ScaleOnly"):
        native_scale_training(TiedState(), parent, level="full", plan=request, load_run_id="still-guarded",
            max_snapshot_bytes=1_000_000, max_parent_bytes=4_000_000, max_saved_input_bytes=1_000_000)

def test_pv_parent_supports_a_fresh_scale_round_and_rejects_wrong_parent(corpus, tmp_path):
    from test_native_scale_training import plan, start
    from test_native_refinement import publish, recording, update
    directory = corpus / CASES[0]; original = NativeModelConversion.open(str(directory / "parent"))
    parent = NativeModelRefinement.open(str(directory / "child"))
    trainer = start(TiedState(), parent, plan(parent, level="pv-child"), level="pv-child", recording=recording(tmp_path))
    assert trainer.revision == 0
    for i in range(4): update(trainer, i, _input(CASES[0]))
    child = publish(trainer, tmp_path / "scale-after-pv"); trainer.close()
    assert child.recovery_recipe == "scale-only" and child.parent_package_id == parent.package_id
    child.verify_with_parent(parent, max_parent_bytes=4_000_000, max_replay_working_bytes=4_000_000)
    with pytest.raises(ValueError): child.verify_with_parent(original, max_parent_bytes=4_000_000, max_replay_working_bytes=4_000_000)

"""Durable, source-bound QAT-hard artifact gates."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from tritium.nn import AdditiveTernaryEmbedding, AdditiveTernaryLinear  # noqa: E402
from tritium.torch import TernaryConfig, convert, export, prepare  # noqa: E402
from tritium.torch.qat_artifacts import (  # noqa: E402
    QatHardArtifact,
    _canonical,
    _digest_bytes,
    _digest_file,
    load_qat_hard,
)


class _TiedLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.norm = torch.nn.LayerNorm(8)
        self.head = torch.nn.Linear(8, 16, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens):
        return self.head(self.norm(self.embed(tokens)))


class _UntiedLanguageModel(_TiedLanguageModel):
    def __init__(self):
        super().__init__()
        self.head.weight = torch.nn.Parameter(self.head.weight.detach().clone())


def _hard_result():
    torch.manual_seed(83)
    prepared = prepare(
        _TiedLanguageModel(),
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Linear", "Embedding"),
            planes=2,
        ),
        inplace=True,
    )
    prepared.model.eval()
    return convert(prepared)


def test_qat_hard_export_strict_reload_and_source_shell_parity(tmp_path):
    result = _hard_result()
    tokens = torch.tensor([[1, 2, 3, 4]])
    expected = result.model(tokens).detach()
    artifact = export(result, tmp_path / "artifact")
    assert isinstance(artifact, QatHardArtifact)
    assert artifact.conversion_artifact_id == result.artifact_id
    assert artifact.source_checkpoint_digest == result.source_checkpoint_digest
    assert artifact.hard_state_digest == result.hard_state_digest
    assert artifact.weights[0].storage_path == "embed._packed_weight"
    assert load_qat_hard(artifact.artifact_dir) == artifact

    shell = _TiedLanguageModel()
    original = shell.embed.weight.detach().clone()
    loaded = load_qat_hard(artifact.artifact_dir, shell, inplace=False)
    assert isinstance(loaded.embed, AdditiveTernaryEmbedding)
    assert isinstance(loaded.head, AdditiveTernaryLinear)
    assert loaded.embed.packed_weight is loaded.head.packed_weight
    assert torch.equal(shell.embed.weight, original)
    torch.testing.assert_close(loaded(tokens), expected, rtol=0, atol=0)


def test_qat_hard_artifact_rejects_corruption_unknown_files_and_wrong_shell(tmp_path):
    artifact = export(_hard_result(), tmp_path / "artifact")
    state = artifact.artifact_dir / "model.safetensors"
    payload = bytearray(state.read_bytes())
    payload[-1] ^= 1
    state.write_bytes(payload)
    with pytest.raises(ValueError, match="state identity mismatch"):
        load_qat_hard(artifact.artifact_dir)

    artifact = export(_hard_result(), tmp_path / "artifact-2")
    (artifact.artifact_dir / "unknown").write_text("x")
    with pytest.raises(ValueError, match="unknown files"):
        load_qat_hard(artifact.artifact_dir)
    (artifact.artifact_dir / "unknown").unlink()
    with pytest.raises(ValueError, match="model shell"):
        load_qat_hard(
            artifact.artifact_dir,
            torch.nn.Sequential(torch.nn.Linear(7, 3)),
            inplace=True,
        )
    with pytest.raises(ValueError, match="tie topology"):
        load_qat_hard(
            artifact.artifact_dir,
            _UntiedLanguageModel(),
            inplace=True,
        )
    wrong_preserved = _TiedLanguageModel()
    wrong_preserved.norm = torch.nn.LayerNorm(7)
    original_embed = wrong_preserved.embed
    with pytest.raises(ValueError, match="preserved model shell state"):
        load_qat_hard(
            artifact.artifact_dir,
            wrong_preserved,
            inplace=True,
        )
    assert wrong_preserved.embed is original_embed


def test_qat_hard_export_rolls_back_failed_state_serialization(monkeypatch, tmp_path):
    import tritium.torch.qat_artifacts as artifacts

    safe_open, load_model, _ = artifacts._dependencies()

    def fail(*args, **kwargs):
        raise RuntimeError("forced serialization failure")

    monkeypatch.setattr(
        artifacts,
        "_dependencies",
        lambda: (safe_open, load_model, fail),
    )
    output = tmp_path / "artifact"
    with pytest.raises(RuntimeError, match="forced serialization failure"):
        export(_hard_result(), output)
    assert not output.exists()


def test_qat_hard_rejects_rehashed_ancestry_and_packed_domain_tampering(tmp_path):
    from safetensors.torch import load_file, save_file

    artifact = export(_hard_result(), tmp_path / "artifact")
    manifest_path = artifact.artifact_dir / "tritium-qat-hard.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["recipe_id"] = "sha256:" + "0" * 64
    identity = dict(manifest)
    identity.pop("artifact_id")
    manifest["artifact_id"] = _digest_bytes(_canonical(identity))
    manifest_path.write_bytes(_canonical(manifest))
    with pytest.raises(ValueError, match="conversion ancestry mismatch"):
        load_qat_hard(artifact.artifact_dir)

    artifact = export(_hard_result(), tmp_path / "artifact-2")
    manifest_path = artifact.artifact_dir / "tritium-qat-hard.json"
    state_path = artifact.artifact_dir / "model.safetensors"
    state = load_file(state_path)
    packed_name = next(name for name in state if name.endswith("packed_trits_0"))
    state[packed_name][0] = 255
    save_file(state, state_path, metadata={"format": "pt", "tritium_mode": "qat-hard"})
    manifest = json.loads(manifest_path.read_text())
    digest, byte_count = _digest_file(state_path, 128 * 1024**3)
    manifest["state"]["sha256"] = digest
    manifest["state"]["bytes"] = byte_count
    identity = dict(manifest)
    identity.pop("artifact_id")
    manifest["artifact_id"] = _digest_bytes(_canonical(identity))
    manifest_path.write_bytes(_canonical(manifest))
    with pytest.raises(ValueError, match="invalid B3 bytes"):
        load_qat_hard(artifact.artifact_dir)


def test_qat_hard_rejects_rehashed_equal_count_tensor_substitution(tmp_path):
    from safetensors.torch import load_file, save_file

    artifact = export(_hard_result(), tmp_path / "artifact")
    manifest_path = artifact.artifact_dir / "tritium-qat-hard.json"
    state_path = artifact.artifact_dir / "model.safetensors"
    state = load_file(state_path)
    state["forged.bias"] = state.pop("norm.bias")
    save_file(state, state_path, metadata={"format": "pt", "tritium_mode": "qat-hard"})

    manifest = json.loads(manifest_path.read_text())
    ledger = manifest["state"]["tensors"]
    next(item for item in ledger if item["name"] == "norm.bias")["name"] = "forged.bias"
    digest, byte_count = _digest_file(state_path, 128 * 1024**3)
    manifest["state"]["sha256"] = digest
    manifest["state"]["bytes"] = byte_count
    identity = dict(manifest)
    identity.pop("artifact_id")
    manifest["artifact_id"] = _digest_bytes(_canonical(identity))
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(ValueError, match="hard-state identity"):
        load_qat_hard(artifact.artifact_dir)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_qat_hard_artifact_reloads_directly_to_cuda(tmp_path):
    prepared = prepare(
        torch.nn.Linear(8, 4, device="cuda"),
        TernaryConfig.qat(estimator="salt-ste", planes=2),
        inplace=True,
    )
    sample = torch.randn(3, 8, device="cuda")
    prepared.model.eval()
    result = convert(prepared)
    expected = result.model(sample).detach()
    artifact = export(result, tmp_path / "artifact")
    loaded = load_qat_hard(
        artifact.artifact_dir,
        torch.nn.Linear(8, 4, device="cuda"),
        inplace=True,
    )
    assert all(buffer.is_cuda for buffer in loaded.buffers())
    torch.testing.assert_close(loaded(sample), expected, rtol=0, atol=0)

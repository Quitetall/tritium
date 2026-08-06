"""Durable, source-bound QAT-hard artifact gates."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from tritium.nn import (  # noqa: E402
    AdditiveTernaryConv1d,
    AdditiveTernaryConv2d,
    AdditiveTernaryEmbedding,
    AdditiveTernaryLinear,
)
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


class _ConvolutionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = torch.nn.Conv1d(
            4,
            6,
            kernel_size=3,
            padding=1,
            groups=2,
            padding_mode="reflect",
        )
        self.spatial = torch.nn.Conv2d(
            4,
            6,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=2,
            bias=False,
        )

    def forward(self, temporal, spatial):
        return self.temporal(temporal), self.spatial(spatial)


class _BufferedConvolutionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 3, kernel_size=3, padding=1)
        self.norm = torch.nn.BatchNorm1d(3)

    def forward(self, inputs):
        return self.norm(self.conv(inputs))


class _SharedConvolutionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shared = torch.nn.Conv1d(2, 3, kernel_size=3, padding=1)
        self.first = shared
        self.second = shared

    def forward(self, inputs):
        return self.first(inputs) + self.second(inputs)


class _TiedConvolutionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv1d(2, 3, kernel_size=3, padding=1)
        self.second = torch.nn.Conv1d(2, 3, kernel_size=3, padding=1)
        self.second.weight = self.first.weight
        self.second.bias = self.first.bias

    def forward(self, inputs):
        return self.first(inputs) + self.second(inputs)


class _CrossBoundaryTiedBiasModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 3, kernel_size=3, padding=1)
        self.norm = torch.nn.LayerNorm(3)
        self.norm.bias = self.conv.bias

    def forward(self, inputs):
        hidden = self.conv(inputs).transpose(1, 2)
        return self.norm(hidden).transpose(1, 2)


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
    assert artifact.schema_version == 2
    assert artifact.source_checkpoint_digest == result.source_checkpoint_digest
    assert artifact.hard_state_digest == result.hard_state_digest
    assert artifact.weights[0].storage_path == "embed._packed_weight"
    assert [consumer.alias for consumer in artifact.weights[0].consumers] == [
        "embed.weight",
        "head.weight",
    ]
    manifest = json.loads(
        (artifact.artifact_dir / "tritium-qat-hard.json").read_text()
    )
    assert manifest["artifact_kind"] == "tritium.module-qat-hard-v2"
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
    wrong_embedding_options = _TiedLanguageModel()
    wrong_embedding_options.embed.padding_idx = 0
    original_head = wrong_embedding_options.head
    with pytest.raises(ValueError, match="consumer contract"):
        load_qat_hard(
            artifact.artifact_dir,
            wrong_embedding_options,
            inplace=True,
        )
    assert wrong_embedding_options.head is original_head
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


def test_qat_hard_convolution_artifact_strict_reload_preserves_geometry(tmp_path):
    torch.manual_seed(97)
    prepared = prepare(
        _ConvolutionModel().eval(),
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Conv1d", "Conv2d"),
            planes=2,
        ),
        inplace=True,
    )
    temporal = torch.randn(2, 4, 11)
    spatial = torch.randn(2, 4, 7, 5)
    hard = convert(prepared)
    expected = hard.model(temporal, spatial)

    artifact = export(hard, tmp_path / "artifact")
    source_shell = _ConvolutionModel().eval()
    loaded = load_qat_hard(artifact.artifact_dir, source_shell, inplace=False)

    assert isinstance(loaded.temporal, AdditiveTernaryConv1d)
    assert isinstance(loaded.spatial, AdditiveTernaryConv2d)
    assert loaded.temporal.groups == 2
    assert loaded.temporal.padding_mode == "reflect"
    assert loaded.spatial.kernel_size == (3, 1)
    assert isinstance(source_shell.temporal, torch.nn.Conv1d)
    actual = loaded(temporal, spatial)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    wrong_geometry = _ConvolutionModel().eval()
    wrong_geometry.spatial = torch.nn.Conv2d(
        4,
        6,
        kernel_size=(1, 3),
        padding=(0, 1),
        groups=2,
        bias=False,
    )
    with pytest.raises(ValueError, match="consumer contract"):
        load_qat_hard(artifact.artifact_dir, wrong_geometry, inplace=True)
    assert isinstance(wrong_geometry.spatial, torch.nn.Conv2d)


def test_qat_hard_artifact_round_trips_preserved_persistent_buffers(tmp_path):
    torch.manual_seed(101)
    source = _BufferedConvolutionModel().eval()
    source.norm.running_mean.copy_(torch.tensor([0.25, -0.5, 0.75]))
    prepared = prepare(
        source,
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Conv1d",),
        ),
        inplace=True,
    )
    sample = torch.randn(2, 2, 9)
    hard = convert(prepared)
    expected = hard.model(sample)

    artifact = export(hard, tmp_path / "artifact")
    loaded = load_qat_hard(
        artifact.artifact_dir,
        _BufferedConvolutionModel().eval(),
        inplace=True,
    )

    assert torch.equal(loaded.norm.running_mean, source.norm.running_mean)
    torch.testing.assert_close(loaded(sample), expected, rtol=0, atol=0)


def test_qat_hard_artifact_preserves_shared_convolution_module_identity(tmp_path):
    torch.manual_seed(103)
    prepared = prepare(
        _SharedConvolutionModel().eval(),
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Conv1d",),
        ),
        inplace=True,
    )
    sample = torch.randn(2, 2, 9)
    hard = convert(prepared)
    expected = hard.model(sample)
    assert hard.model.first is hard.model.second
    assert hard.weights[0].aliases == ("first.weight", "second.weight")

    artifact = export(hard, tmp_path / "artifact")
    loaded = load_qat_hard(
        artifact.artifact_dir,
        _SharedConvolutionModel().eval(),
        inplace=True,
    )

    assert loaded.first is loaded.second
    torch.testing.assert_close(loaded(sample), expected, rtol=0, atol=0)

    wrong_module_topology = _TiedConvolutionModel().eval()
    original_first = wrong_module_topology.first
    original_second = wrong_module_topology.second
    with pytest.raises(ValueError, match="module topology"):
        load_qat_hard(
            artifact.artifact_dir,
            wrong_module_topology,
            inplace=True,
        )
    assert wrong_module_topology.first is original_first
    assert wrong_module_topology.second is original_second


def test_qat_hard_artifact_preserves_distinct_consumers_tied_state(tmp_path):
    torch.manual_seed(109)
    prepared = prepare(
        _TiedConvolutionModel().eval(),
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Conv1d",),
        ),
        inplace=True,
    )
    sample = torch.randn(2, 2, 9)
    hard = convert(prepared)
    expected = hard.model(sample)
    artifact = export(hard, tmp_path / "artifact")

    loaded = load_qat_hard(
        artifact.artifact_dir,
        _TiedConvolutionModel().eval(),
        inplace=True,
    )

    assert loaded.first is not loaded.second
    assert loaded.first.packed_weight is loaded.second.packed_weight
    assert loaded.first.bias is loaded.second.bias
    torch.testing.assert_close(loaded(sample), expected, rtol=0, atol=0)


def test_qat_hard_artifact_preserves_target_to_preserved_bias_tie(tmp_path):
    torch.manual_seed(127)
    prepared = prepare(
        _CrossBoundaryTiedBiasModel().eval(),
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Conv1d",),
        ),
        inplace=True,
    )
    sample = torch.randn(2, 2, 9)
    hard = convert(prepared)
    expected = hard.model(sample)
    artifact = export(hard, tmp_path / "artifact")

    loaded = load_qat_hard(
        artifact.artifact_dir,
        _CrossBoundaryTiedBiasModel().eval(),
        inplace=True,
    )

    assert loaded.conv.bias is loaded.norm.bias
    torch.testing.assert_close(loaded(sample), expected, rtol=0, atol=0)
    loaded.to(dtype=torch.float64)
    assert loaded.conv.bias is loaded.norm.bias

    device_loaded = load_qat_hard(
        artifact.artifact_dir,
        _CrossBoundaryTiedBiasModel().eval(),
        inplace=True,
    )
    device_loaded.to("meta")
    assert device_loaded.conv.bias is device_loaded.norm.bias
    assert device_loaded.conv.bias.device.type == "meta"


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


def test_qat_hard_inplace_load_rolls_back_failed_state_install(monkeypatch, tmp_path):
    import tritium.torch.qat_artifacts as artifacts

    artifact = export(_hard_result(), tmp_path / "artifact")

    def fail(target, *args, **kwargs):
        target.norm.weight = torch.nn.Parameter(
            torch.full_like(target.norm.weight, 17.0)
        )
        raise RuntimeError("forced state install failure")

    monkeypatch.setattr(artifacts, "_load_state_into_shell", fail)
    shell = _TiedLanguageModel()
    original_embed = shell.embed
    original_head = shell.head
    original_norm_weight = shell.norm.weight
    original_norm_value = shell.norm.weight.detach().clone()
    with pytest.raises(RuntimeError, match="forced state install failure"):
        load_qat_hard(artifact.artifact_dir, shell, inplace=True)
    assert shell.embed is original_embed
    assert shell.head is original_head
    assert shell.norm.weight is original_norm_weight
    assert torch.equal(shell.norm.weight, original_norm_value)


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
    forged = next(item for item in ledger if item["name"] == "norm.bias")
    forged["name"] = "forged.bias"
    forged["aliases"] = ["forged.bias"]
    digest, byte_count = _digest_file(state_path, 128 * 1024**3)
    manifest["state"]["sha256"] = digest
    manifest["state"]["bytes"] = byte_count
    identity = dict(manifest)
    identity.pop("artifact_id")
    manifest["artifact_id"] = _digest_bytes(_canonical(identity))
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(ValueError, match="tie topology|hard-state identity"):
        load_qat_hard(artifact.artifact_dir)


@pytest.mark.parametrize("tamper", ("missing", "relabel"))
def test_qat_hard_rejects_rehashed_alias_metadata_tampering(tmp_path, tamper):
    from safetensors.torch import load_file, save_file

    torch.manual_seed(131)
    prepared = prepare(
        _CrossBoundaryTiedBiasModel().eval(),
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Conv1d",),
        ),
        inplace=True,
    )
    artifact = export(convert(prepared), tmp_path / "artifact")
    manifest_path = artifact.artifact_dir / "tritium-qat-hard.json"
    state_path = artifact.artifact_dir / "model.safetensors"
    state = load_file(state_path)
    manifest = json.loads(manifest_path.read_text())
    metadata = {"format": "pt", "tritium_mode": "qat-hard"}
    if tamper == "relabel":
        metadata["evil.bias"] = "conv.bias"
        bias = next(
            item
            for item in manifest["state"]["tensors"]
            if item["name"] == "conv.bias"
        )
        bias["aliases"] = ["conv.bias", "evil.bias"]
    save_file(state, state_path, metadata=metadata)

    digest, byte_count = _digest_file(state_path, 128 * 1024**3)
    manifest["state"]["sha256"] = digest
    manifest["state"]["bytes"] = byte_count
    identity = dict(manifest)
    identity.pop("artifact_id")
    manifest["artifact_id"] = _digest_bytes(_canonical(identity))
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(ValueError, match="state tie topology"):
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

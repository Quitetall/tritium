import hashlib
import io
import os
import sys
import tarfile

import pytest

pytest.importorskip("torch")

from tritium.torch.bundle_binding import BoundBundle, BundleBindingError  # noqa: E402
from tritium.torch import qualify_onnx  # noqa: E402


def candidate(root, kind="onnx-bundle", *, compressed=False, members=None):
    root.mkdir()
    bundle = root / "unpacked"
    bundle.mkdir()
    members = members or [
        ("language.onnx", b"language"), ("mtp.onnx", b"mtp"),
        ("tritium-onnx-manifest.json", b"manifest"), ("weights.bin", b"packed"),
    ]
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in members:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
            if "/" not in name and name not in {".", ".."}:
                (bundle / name).write_bytes(data)
    body = output.getvalue()
    name = "candidate.tar"
    if compressed:
        zstandard = pytest.importorskip("zstandard")
        body = zstandard.ZstdCompressor().compress(body)
        name += ".zst"
    archive = root / name
    archive.write_bytes(body)
    identity = {
        "id": kind, "kind": kind, "name": name, "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    return archive, bundle, identity


@pytest.mark.parametrize("compressed", [False, True])
def test_binds_original_tar_and_zstd_bytes_without_copying(tmp_path, compressed):
    archive, bundle, identity = candidate(tmp_path / "candidate", compressed=compressed)
    with BoundBundle(archive, bundle, identity) as bound:
        assert bound.bundle == bundle
        bound.assert_unchanged()
        # Other tasks/fault injection can write in shared ancestor directories.
        (tmp_path / "unrelated").write_bytes(b"shared workspace")


@pytest.mark.parametrize("fault", ["changed", "missing", "extra", "symlink", "fifo"])
def test_rejects_substituted_directory_before_execution(tmp_path, fault):
    archive, bundle, identity = candidate(tmp_path / "candidate")
    weight = bundle / "weights.bin"
    if fault == "changed":
        weight.write_bytes(b"swappe")  # Same size; file names and metadata can still agree.
    elif fault == "missing":
        weight.unlink()
    elif fault == "extra":
        (bundle / "hidden-shadow").write_bytes(b"dense")
    elif fault == "symlink":
        outside = tmp_path / "outside"
        outside.write_bytes(weight.read_bytes())
        weight.unlink()
        weight.symlink_to(outside)
    else:
        weight.unlink()
        os.mkfifo(weight)
    with pytest.raises(BundleBindingError):
        with BoundBundle(archive, bundle, identity):
            pytest.fail("substituted payload reached execution")


@pytest.mark.parametrize("fault", ["digest", "bytes", "name", "ancestor-symlink"])
def test_rejects_archive_identity_and_path_substitution(tmp_path, fault):
    archive, bundle, identity = candidate(tmp_path / "candidate")
    if fault == "digest":
        identity["sha256"] = "0" * 64
    elif fault == "bytes":
        identity["bytes"] += 1
    elif fault == "name":
        identity["name"] = "other.tar"
    else:
        link = tmp_path / "alias"
        link.symlink_to(archive.parent, target_is_directory=True)
        archive = link / archive.name
    with pytest.raises(BundleBindingError):
        with BoundBundle(archive, bundle, identity):
            pytest.fail("unbound archive reached execution")


@pytest.mark.parametrize("name", ["../weights.bin", "sub/weights.bin", "unknown.bin"])
def test_rejects_unsafe_or_unknown_archive_members(tmp_path, name):
    archive, bundle, identity = candidate(tmp_path / "candidate", members=[(name, b"packed")])
    with pytest.raises(BundleBindingError):
        with BoundBundle(archive, bundle, identity):
            pytest.fail("noncanonical tar reached execution")


def test_rejects_duplicate_archive_member(tmp_path):
    archive, bundle, identity = candidate(
        tmp_path / "candidate", members=[("weights.bin", b"one"), ("weights.bin", b"one")]
    )
    with pytest.raises(BundleBindingError, match="duplicate"):
        with BoundBundle(archive, bundle, identity):
            pass


@pytest.mark.parametrize("fault", ["restore-bytes", "replace-file", "archive", "root-swap"])
def test_retained_handles_detect_changes_during_execution(tmp_path, fault):
    archive, bundle, identity = candidate(tmp_path / "candidate")
    with pytest.raises(BundleBindingError, match="changed"):
        with BoundBundle(archive, bundle, identity):
            weight = bundle / "weights.bin"
            if fault == "restore-bytes":
                original = weight.read_bytes()
                weight.write_bytes(b"swappe")
                weight.write_bytes(original)
            elif fault == "replace-file":
                replacement = tmp_path / "replacement"
                replacement.write_bytes(weight.read_bytes())
                os.replace(replacement, weight)
            elif fault == "archive":
                archive.write_bytes(archive.read_bytes())
            else:
                bundle.rename(bundle.with_name("original"))
                bundle.mkdir()


def test_detects_ancestor_swap_even_if_original_path_is_restored(tmp_path):
    archive, bundle, identity = candidate(tmp_path / "candidate")
    with pytest.raises(BundleBindingError, match="changed"):
        with BoundBundle(archive, bundle, identity):
            parent = archive.parent
            moved = parent.with_name("moved")
            parent.rename(moved)
            parent.mkdir()
            parent.rmdir()
            moved.rename(parent)


@pytest.mark.parametrize("fault", ["link-header", "trailing-payload", "truncated"])
def test_rejects_noncanonical_tar_even_with_correct_archive_digest(tmp_path, fault):
    archive, bundle, identity = candidate(tmp_path / "candidate")
    body = bytearray(archive.read_bytes())
    if fault == "link-header":
        body[156] = ord("2")
    elif fault == "trailing-payload":
        body[-1] = 1
    else:
        body = body[:600]
    archive.write_bytes(body)
    identity.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    with pytest.raises(BundleBindingError):
        with BoundBundle(archive, bundle, identity):
            pytest.fail("malformed archive reached execution")


def test_large_member_is_compared_through_bounded_chunks(tmp_path, monkeypatch):
    from tritium.torch import bundle_binding

    payload = b"p" * (bundle_binding.CHUNK_BYTES * 2 + 17)
    archive, bundle, identity = candidate(
        tmp_path / "candidate", members=[("weights.bin", payload)]
    )
    exact = bundle_binding._exact
    sizes = []

    def observed(stream, size):
        sizes.append(size)
        return exact(stream, size)

    monkeypatch.setattr(bundle_binding, "_exact", observed)
    with BoundBundle(archive, bundle, identity):
        pass
    assert max(sizes) <= bundle_binding.CHUNK_BYTES
    with (bundle / "weights.bin").open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        stream.write(b"q")
    with pytest.raises(BundleBindingError, match="differs from archive"):
        with BoundBundle(archive, bundle, identity):
            pass


def test_zstd_archive_has_no_missing_dependency_fallback(tmp_path, monkeypatch):
    archive, bundle, identity = candidate(tmp_path / "candidate", compressed=True)
    monkeypatch.setitem(sys.modules, "zstandard", None)
    with pytest.raises(BundleBindingError, match="zstandard is required"):
        with BoundBundle(archive, bundle, identity):
            pass


def test_corrupt_zstd_payload_is_rejected_even_with_matching_identity(tmp_path):
    archive, bundle, identity = candidate(tmp_path / "candidate", compressed=True)
    archive.write_bytes(b"invalid zstd")
    identity.update(bytes=12, sha256=hashlib.sha256(b"invalid zstd").hexdigest())
    with pytest.raises(BundleBindingError, match="zstd payload is invalid"):
        with BoundBundle(archive, bundle, identity):
            pass


def worker_inputs(tmp_path):
    onnx_archive, onnx_bundle, onnx_identity = candidate(tmp_path / "onnx")
    native_archive, native_bundle, native_identity = candidate(
        tmp_path / "native", "model-bundle", members=[("tritium.json", b"manifest")]
    )
    wheel = tmp_path / "candidate.whl"
    wheel.write_bytes(b"wheel")
    wheel_identity = {
        "id": "wheel", "kind": "python-wheel", "name": wheel.name, "bytes": 5,
        "sha256": hashlib.sha256(b"wheel").hexdigest(),
    }
    return dict(
        wheel=wheel, wheel_record=wheel_identity,
        artifact_record=onnx_identity, model_record=native_identity,
        model_artifact_id=native_identity["id"],
        onnx_archive=onnx_archive, native_archive=native_archive,
        onnx_bundle=onnx_bundle, native_bundle=native_bundle,
        profile="compact-v1", conversion_mode="ptq",
        source_revision="a" * 40, release="1.1.0-rc.2", run_id="test-run",
        candidate_manifest_sha256="b" * 64,
    )


def test_public_worker_rejects_swapped_bytes_before_model_load(tmp_path, monkeypatch):
    values = worker_inputs(tmp_path)
    (values["onnx_bundle"] / "weights.bin").write_bytes(b"swappe")
    monkeypatch.setattr(
        qualify_onnx, "_execute", lambda **kwargs: pytest.fail("swapped bytes executed")
    )
    with pytest.raises(qualify_onnx.OnnxQualificationError, match="differs from archive"):
        qualify_onnx.run(**values)


def test_public_worker_retains_custody_until_after_execution(tmp_path, monkeypatch):
    values = worker_inputs(tmp_path)
    weight = values["onnx_bundle"] / "weights.bin"

    def execute(**kwargs):
        assert weight.stat().st_nlink == 3
        assert kwargs["fault_workspace"].is_dir()
        return {"test-result": "executed"}

    monkeypatch.setattr(qualify_onnx, "_execute", execute)
    assert qualify_onnx.run(**values) == {"test-result": "executed"}
    assert weight.stat().st_nlink == 1

    def mutate(**kwargs):
        weight.write_bytes(b"swappe")
        return {"test-result": "must not publish"}

    monkeypatch.setattr(qualify_onnx, "_execute", mutate)
    with pytest.raises(qualify_onnx.OnnxQualificationError, match="changed"):
        qualify_onnx.run(**values)


def test_public_worker_preserves_parent_symlink_for_rejection(tmp_path, monkeypatch):
    values = worker_inputs(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(values["onnx_bundle"].parent, target_is_directory=True)
    values["onnx_bundle"] = alias / "unpacked"
    monkeypatch.setattr(
        qualify_onnx, "_execute", lambda **kwargs: pytest.fail("symlink alias executed")
    )
    with pytest.raises(qualify_onnx.OnnxQualificationError, match="custody failed"):
        qualify_onnx.run(**values)

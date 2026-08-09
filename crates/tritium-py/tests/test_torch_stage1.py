"""ADR 0037 Stage 1 gates: tritium.torch optimizers/schedules/EMA/envelope/
contract/DPO.

Two tiers:
- Self-contained unit tests (always run).
- Migration-parity tests against the LamQuant originals and the blut
  contract lint — these import from the source repos by absolute path and
  skip cleanly on machines that lack them (they are the migration gates on
  the dev box, not portable CI).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tritium.torch import checkpoint as ckpt
from tritium.torch import contract, losses, optim, schedules
from tritium.torch.ema import Ema

LAMQUANT_LQ = Path("/mnt/4tb/LamQuant/training/cookbooks/lamquant/python")
LAMQUANT_CORE = Path("/mnt/4tb/LamQuant/training/cookbooks/core/python")
BLUT_LINT = Path.home() / "blut" / "scripts" / "contract_lint.py"

needs_lamquant = pytest.mark.skipif(
    not (LAMQUANT_LQ.is_dir() and LAMQUANT_CORE.is_dir()),
    reason="LamQuant checkout not present (dev-box migration gate)",
)
needs_blut = pytest.mark.skipif(
    not BLUT_LINT.is_file(), reason="blut checkout not present (dev-box migration gate)"
)


def _quadratic_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
    return m


def _train_steps(model, opt_obj, steps: int = 12, seed: int = 1) -> list[float]:
    torch.manual_seed(seed)
    xs = torch.randn(32, 8)
    ys = torch.randn(32, 4)
    hist = []
    for _ in range(steps):
        opt_obj.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xs), ys)
        loss.backward()
        opt_obj.step()
        hist.append(float(loss.detach()))
    return hist


# --------------------------------------------------------------------------- #
# Optimizers                                                                  #
# --------------------------------------------------------------------------- #
def test_soap_reduces_loss():
    # SOAP's first step only initializes the preconditioner (no param update),
    # so give it a longer horizon and require monotone-ish progress, not a
    # fixed ratio — descent here is a sanity check; parity below is the gate.
    model = _quadratic_model()
    hist = _train_steps(model, optim.SOAP(model.parameters(), lr=3e-3), steps=60)
    assert hist[-1] < hist[1]


def test_esoap_routed_groups_and_descent():
    model = _quadratic_model()
    groups = optim.route_param_groups(model.named_parameters(), "esoap", weight_decay=0.01)
    assert [g["method"] for g in groups] == ["esoap", "adamw"]
    assert all(p.ndim == 2 for p in groups[0]["params"])
    assert len(groups[0]["params"]) == 2  # the two Linear weights
    hist = _train_steps(model, optim.ESOAP(groups, lr=1e-2), steps=20)
    assert hist[-1] < hist[0] * 0.9


def test_sinksoaph_descent():
    model = _quadratic_model()
    groups = optim.route_param_groups(model.named_parameters(), "sinksoaph")
    hist = _train_steps(model, optim.SinkSOAPH(groups, lr=5e-3), steps=20)
    assert hist[-1] < hist[0]


def test_cautious_wd_never_fights_the_step():
    p = torch.tensor([1.0, -1.0, 2.0])
    update = torch.tensor([0.5, 0.5, -0.5])  # agrees, fights, fights
    p0 = p.clone()
    optim.cautious_decoupled_wd_(p, update.clone(), lr=0.1, weight_decay=0.5)
    # Entry 0: update*p > 0 → decay folded in; entries 1-2: plain step only.
    assert p[0] == pytest.approx(float(p0[0] - (0.5 + 0.1 * 0.5 * p0[0])))
    assert p[1] == pytest.approx(float(p0[1] - 0.5))
    assert p[2] == pytest.approx(float(p0[2] + 0.5))


def test_create_optimizer_unknown_name():
    with pytest.raises(ValueError, match="unknown optimizer"):
        optim.create_optimizer("madgrad", _quadratic_model())


def test_clip_grad_norm():
    model = _quadratic_model()
    loss = model(torch.randn(4, 8)).square().sum() * 100
    loss.backward()
    total = optim.clip_grad_norm_(model.parameters(), max_norm=0.5)
    assert total > 0.5  # it actually clipped something
    post = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))
    assert float(post) == pytest.approx(0.5, rel=1e-5)


# --------------------------------------------------------------------------- #
# Schedules                                                                   #
# --------------------------------------------------------------------------- #
def _sched_fixture(decay_frac=0.10, groups_lr=(1e-3,)):
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in groups_lr]
    opt_obj = torch.optim.SGD(
        [{"params": [p], "lr": lr} for p, lr in zip(params, groups_lr)], lr=groups_lr[0]
    )
    return schedules.WSDScheduler(
        opt_obj, total_epochs=100, peak_lr=1e-3, warmup_frac=0.10, decay_frac=decay_frac
    )


def test_wsd_phases():
    s = _sched_fixture()
    s.step(epoch=5)
    assert s.phase == "warmup" and 0 < s.get_last_lr()[0] < 1e-3
    s.step(epoch=50)
    assert s.phase == "stable" and s.get_last_lr()[0] == pytest.approx(1e-3)
    s.step(epoch=100)
    assert s.phase == "decay" and s.get_last_lr()[0] == pytest.approx(1e-6, rel=1e-3)


def test_wsd_infinite_and_trigger_decay():
    s = _sched_fixture(decay_frac=0.0)
    s.step(epoch=10_000)
    assert s.phase == "stable∞" and s.get_last_lr()[0] == pytest.approx(1e-3)
    s.trigger_decay(n_epochs=10)
    s.step(epoch=10_010)
    assert s.phase == "decay" and s.get_last_lr()[0] == pytest.approx(1e-6, rel=1e-3)


def test_wsd_preserves_group_scale():
    s = _sched_fixture(groups_lr=(1e-3, 2.5e-4))  # second group at 0.25x
    s.step(epoch=50)
    lrs = s.get_last_lr()
    assert lrs[0] == pytest.approx(1e-3) and lrs[1] == pytest.approx(2.5e-4)


def test_wsd_state_roundtrip_carries_triggered_decay():
    s = _sched_fixture(decay_frac=0.0)
    s.step(epoch=500)
    s.trigger_decay(n_epochs=40)
    state = schedules.wsd_state_dict(s)
    s2 = _sched_fixture(decay_frac=0.0)
    schedules.wsd_load_state_dict(s2, state)
    s.step()
    s2.step()
    assert s2.phase == "decay" and s2.get_last_lr() == s.get_last_lr()


# --------------------------------------------------------------------------- #
# EMA                                                                         #
# --------------------------------------------------------------------------- #
def test_ema_tracks_and_roundtrips():
    model = _quadratic_model()
    ema = Ema(model, decay=0.9)
    ema.update(model)  # first update copies (n_averaged=0): avg == model
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)  # second update lags behind the moved weights
    w_live = next(model.parameters())
    w_avg = next(ema.module.parameters())
    assert not torch.equal(w_live, w_avg)  # lagging
    state = ema.state_dict()
    ema2 = Ema(_quadratic_model(), decay=0.9)
    ema2.load_state_dict(state)
    assert torch.equal(
        next(ema2.module.parameters()), next(ema.module.parameters())
    )
    with pytest.raises(ValueError, match="decay mismatch"):
        Ema(_quadratic_model(), decay=0.5).load_state_dict(state)


# --------------------------------------------------------------------------- #
# Envelope                                                                    #
# --------------------------------------------------------------------------- #
def test_envelope_roundtrip_rng_and_prev_rotation(tmp_path):
    model = _quadratic_model()
    opt_obj = torch.optim.AdamW(model.parameters())
    path = tmp_path / "run.ckpt"
    torch.manual_seed(7)
    _pre = torch.rand(3)  # advance RNG to a nontrivial state
    expected_next = None
    ckpt.save_envelope(
        path,
        model=model.state_dict(),
        opt=opt_obj.state_dict(),
        config={"lr": 1e-3, "seq": 512},
        step=42,
    )
    expected_next = torch.rand(3)  # what the RNG produced after capture

    torch.manual_seed(0)  # scramble
    payload = ckpt.load_envelope(path)
    assert payload["step"] == 42
    ckpt.restore_rng(payload["rng"])
    assert torch.equal(torch.rand(3), expected_next)

    # Second save rotates the first to .prev.
    ckpt.save_envelope(
        path,
        model=model.state_dict(),
        opt=opt_obj.state_dict(),
        config={"lr": 1e-3, "seq": 512},
        step=43,
    )
    assert path.with_suffix(path.suffix + ".prev").exists()
    assert ckpt.load_envelope(path)["step"] == 43


def test_envelope_alias_reads(tmp_path):
    path = tmp_path / "lamquant_style.ckpt"
    torch.save(
        {
            "state_dict": {"w": torch.zeros(1)},
            "optimizer": {"state": {}},
            "training_config_hash": "abc123",
            "step": 7,
            "rng_state": {"torch": torch.get_rng_state()},
            "epoch": 3,
        },
        path,
    )
    payload = ckpt.load_envelope(path)
    assert payload["model"] == {"w": payload["model"]["w"]}
    assert payload["config"] == "abc123"
    assert payload["rng"]["torch"] is not None
    assert payload["epoch"] == 3  # extras preserved


def test_envelope_extra_key_collision(tmp_path):
    with pytest.raises(ValueError, match="collide"):
        ckpt.save_envelope(
            tmp_path / "x.ckpt",
            model={},
            opt={},
            config={},
            step=0,
            extra={"model": {}},
        )


# --------------------------------------------------------------------------- #
# Contract                                                                    #
# --------------------------------------------------------------------------- #
def test_contract_stream_shape(capsys):
    contract.announce()
    contract.heartbeat(phase="model-load")
    contract.step(1, 10, 2.0, 1e-4, 100)
    contract.emit_metric({"val_r": 0.9, "note": "dropped", "flag": True}, phase="quant")
    contract.saved("c/x.pt")
    contract.done(1.5, "c")
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == "BLUT_CONTRACT 1"
    metric = next(l for l in out if l.startswith("BLUT_METRIC "))
    payload = json.loads(metric[len("BLUT_METRIC "):])
    assert payload == {"val_r": 0.9, "kind": "epoch", "phase": "quant"}
    kinds = [json.loads(l)["kind"] for l in out if l.startswith("{")]
    assert kinds == ["heartbeat", "step", "saved", "done"]
    with pytest.raises(ValueError, match="unknown status kind"):
        contract.emit("telemetry")


def test_heartbeat_file(tmp_path):
    state = tmp_path / "state.json"
    contract.write_heartbeat_file(state, {"run": "r1"})
    payload = json.loads(state.read_text())
    assert payload["run"] == "r1" and payload["heartbeat_unix"] > 0
    assert contract.HEARTBEAT_INTERVAL == 60


def test_resume_key_stability_and_sensitivity():
    blake3 = pytest.importorskip("blake3")  # noqa: F841 - contract-pinned dep
    k1 = contract.resume_key({"b": 2, "a": 1}, "digest-x")
    k2 = contract.resume_key({"a": 1, "b": 2}, "digest-x")  # key order irrelevant
    k3 = contract.resume_key({"a": 1, "b": 3}, "digest-x")
    k4 = contract.resume_key({"a": 1, "b": 2}, "digest-y")
    assert k1 == k2 and len(k1) == 64
    assert len({k1, k3, k4}) == 3


@needs_blut
def test_contract_stream_passes_blut_lint(tmp_path):
    stream = tmp_path / "stream.jsonl"
    proc = subprocess.run(
        [sys.executable, "-m", "tritium.torch.contract"],
        capture_output=True,
        text=True,
        check=True,
    )
    stream.write_text(proc.stdout)
    lint = subprocess.run(
        [sys.executable, str(BLUT_LINT), "--stream", str(stream)],
        capture_output=True,
        text=True,
    )
    assert lint.returncode == 0, lint.stdout + lint.stderr


# --------------------------------------------------------------------------- #
# DPO                                                                         #
# --------------------------------------------------------------------------- #
def test_dpo_loss_values():
    z = torch.zeros(4)
    loss, cr, rr = losses.dpo_loss(z, z, z, z, beta=0.1)
    assert float(loss) == pytest.approx(math.log(2.0))  # -logsigmoid(0)
    assert torch.equal(cr, torch.zeros(4)) and torch.equal(rr, torch.zeros(4))

    # Policy prefers chosen more than reference does → loss below log 2,
    # positive reward margin.
    pc, pr = torch.full((4,), -1.0), torch.full((4,), -3.0)
    rc, rr_ = torch.full((4,), -2.0), torch.full((4,), -2.0)
    loss2, cr2, rr2 = losses.dpo_loss(pc, pr, rc, rr_, beta=0.5)
    assert float(loss2) < math.log(2.0)
    assert torch.all(cr2 > rr2)
    with pytest.raises(ValueError, match="label_smoothing"):
        losses.dpo_loss(z, z, z, z, label_smoothing=0.7)


# --------------------------------------------------------------------------- #
# Migration parity vs the LamQuant originals (dev-box gates)                  #
# --------------------------------------------------------------------------- #
def _import_originals():
    for p in (str(LAMQUANT_LQ), str(LAMQUANT_CORE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from blut_core.ingredients.optimizer.soap_optimizer import SOAP as SoapOrig
    from lamquant.ingredients.optimizers.esoap import ESOAP as EsoapOrig
    from lamquant.ingredients.optimizers.sinksoaph import SinkSOAPH as SinkOrig

    return SoapOrig, EsoapOrig, SinkOrig


def _run_pair(make_ours, make_orig, steps=8, seed=3):
    results = []
    for make in (make_ours, make_orig):
        torch.manual_seed(seed)
        model = _quadratic_model(seed=seed)
        opt_obj = make(model)
        _train_steps(model, opt_obj, steps=steps, seed=seed + 1)
        results.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    return results


@needs_lamquant
def test_soap_parity_with_lamquant():
    SoapOrig, _, _ = _import_originals()
    ours, orig = _run_pair(
        lambda m: optim.SOAP(m.parameters(), lr=3e-3),
        lambda m: SoapOrig(m.parameters(), lr=3e-3),
    )
    assert torch.equal(ours, orig)


@needs_lamquant
def test_esoap_parity_with_lamquant():
    _, EsoapOrig, _ = _import_originals()

    def groups_for(m):
        return optim.route_param_groups(m.named_parameters(), "esoap", weight_decay=0.01)

    ours, orig = _run_pair(
        lambda m: optim.ESOAP(groups_for(m), lr=1e-2),
        lambda m: EsoapOrig(groups_for(m), lr=1e-2),
    )
    assert torch.equal(ours, orig)


@needs_lamquant
def test_sinksoaph_parity_with_lamquant():
    _, _, SinkOrig = _import_originals()

    def groups_for(m):
        return optim.route_param_groups(m.named_parameters(), "sinksoaph")

    ours, orig = _run_pair(
        lambda m: optim.SinkSOAPH(groups_for(m), lr=5e-3),
        lambda m: SinkOrig(groups_for(m), lr=5e-3),
    )
    assert torch.equal(ours, orig)


@needs_lamquant
def test_wsd_parity_with_lamquant():
    for p in (str(LAMQUANT_CORE),):
        if p not in sys.path:
            sys.path.insert(0, p)
    from blut_core.ingredients.scheduler.wsd import WSDScheduler as WsdOrig

    def series(cls):
        opt_obj = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        s = cls(opt_obj, total_epochs=200, peak_lr=1e-3, warmup_frac=0.05, decay_frac=0.1)
        return [(_e, s.step(_e), s.get_last_lr()[0])[2] for _e in range(1, 201)]

    assert series(schedules.WSDScheduler) == series(WsdOrig)

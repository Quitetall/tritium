"""tritium.torch.optim — the unified optimizer surface (ADR 0037 Stage 1).

One import point for every optimizer the migrating cookbooks use, plus the
name-suffix/ndim param-group routing they rely on and a gradient-clipping
helper. Native implementations live in sibling modules (`optim_soap`,
`optim_esoap`, `optim_sinksoaph`, `optim_cautious` — verbatim ports, see their
headers); external ones (APOLLO, bitsandbytes 8-bit, schedule-free) resolve
lazily from their packages with actionable errors, mirroring BLUT's
three-tier resolution.
"""

from __future__ import annotations

from typing import Iterable

import torch

from tritium.torch.optim_cautious import cautious_decoupled_wd_
from tritium.torch.optim_esoap import ESOAP
from tritium.torch.optim_sinksoaph import SinkSOAPH
from tritium.torch.optim_soap import SOAP

__all__ = [
    "SOAP",
    "ESOAP",
    "SinkSOAPH",
    "cautious_decoupled_wd_",
    "route_param_groups",
    "create_optimizer",
    "clip_grad_norm_",
    "LINEAR_SUFFIXES",
]

# 2-D weight name-suffixes routed to the preconditioned (SOAP-family) group —
# VERBATIM from LamQuant's `_route_by_suffix` (`_LINEAR_SUFFIXES`), so a
# migrated trainer gets byte-identical group assignments. Deliberately narrow:
# a `.weight` catch-all would sweep embeddings and every other 2-D weight into
# the Muon/Newton-Schulz tail, which the modded-nanogpt lineage avoids. Pass
# `suffixes=` to widen for a different model family.
LINEAR_SUFFIXES: tuple[str, ...] = (
    "in_proj.weight",
    "x_proj.weight",
    "out_proj.weight",
    "spatial_mix.weight",
)


def route_param_groups(
    named_params: Iterable[tuple[str, torch.Tensor]],
    method: str,
    *,
    weight_decay: float = 0.0,
    suffixes: tuple[str, ...] = LINEAR_SUFFIXES,
) -> list[dict]:
    """Split parameters into a `method`-tagged preconditioned group (2-D
    weights whose name ends in one of `suffixes`) and an `adamw` remainder
    group (everything else: 1-D params and any 2-D weight not matching —
    embeddings are excluded only insofar as their names don't match, exactly
    as in LamQuant). This is LamQuant's `_route_by_suffix` contract: one
    optimizer object, per-group `method` tags, one `.step()`/`.zero_grad()`,
    and an LR schedule that scales every group.
    """
    routed, rest = [], []
    for nm, q in named_params:
        if not q.requires_grad:
            continue
        if q.ndim == 2 and nm.endswith(suffixes):
            routed.append(q)
        else:
            rest.append(q)
    return [
        {"params": routed, "method": method, "weight_decay": weight_decay},
        {"params": rest, "method": "adamw", "weight_decay": weight_decay},
    ]


def _require(module: str, hint: str):
    try:
        return __import__(module)
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            f"optimizer requires the {module!r} package ({hint}); "
            f"install it or pick a native optimizer (soap/esoap/sinksoaph/adamw)"
        ) from e


def create_optimizer(name: str, model: torch.nn.Module, /, **cfg) -> torch.optim.Optimizer:
    """Build an optimizer by registry name.

    Native: `adamw`, `sgd`, `soap`, `esoap`, `sinksoaph` (the latter two
    consume `route_param_groups` output automatically). External, resolved
    lazily: `adamw_8bit` (bitsandbytes), `apollo` / `apollo_mini`
    (apollo-torch, rank 4 / rank 1), `schedule_free_adamw` (schedulefree).

    `cfg` is passed through to the optimizer constructor; `weight_decay` and
    `suffixes` additionally steer the routing for the SOAP-family names.
    """
    name = name.lower().replace("-", "_")
    wd = cfg.get("weight_decay", 0.0)
    suffixes = cfg.pop("suffixes", LINEAR_SUFFIXES)

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **cfg)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **cfg)
    if name == "soap":
        return SOAP(model.parameters(), **cfg)
    if name in ("esoap", "sinksoaph"):
        groups = route_param_groups(
            model.named_parameters(), name, weight_decay=wd, suffixes=suffixes
        )
        cls = ESOAP if name == "esoap" else SinkSOAPH
        return cls(groups, **cfg)
    if name == "adamw_8bit":
        bnb = _require("bitsandbytes", "8-bit Adam moments")
        return bnb.optim.AdamW8bit(model.parameters(), **cfg)
    if name in ("apollo", "apollo_mini"):
        apollo = _require("apollo_torch", "low-rank projected-gradient AdamW")
        rank = 1 if name == "apollo_mini" else cfg.pop("rank", 4)
        return apollo.APOLLOAdamW(
            model.parameters(),
            rank=rank,
            update_proj_gap=cfg.pop("update_proj_gap", 200),
            scale=cfg.pop("scale", 1.0),
            **cfg,
        )
    if name == "schedule_free_adamw":
        sf = _require("schedulefree", "schedule-free AdamW")
        return sf.AdamWScheduleFree(model.parameters(), **cfg)
    raise ValueError(
        f"unknown optimizer {name!r} (native: adamw, sgd, soap, esoap, "
        "sinksoaph; external: adamw_8bit, apollo, apollo_mini, "
        "schedule_free_adamw)"
    )


def clip_grad_norm_(
    parameters, max_norm: float, *, norm_type: float = 2.0
) -> torch.Tensor:
    """Global-norm gradient clipping. Thin, name-stable wrapper over
    `torch.nn.utils.clip_grad_norm_` so cookbooks depend on one surface. A
    Rust-tape mirror must reproduce torch's exact rule — scale by
    `min(1, max_norm / (total_norm + 1e-6))` — not an idealized clamp, or the
    two paths diverge bitwise near the threshold.
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

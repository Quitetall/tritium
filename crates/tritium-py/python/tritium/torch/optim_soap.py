"""SOAP optimizer — vendored from nikhilvyas/SOAP (MIT license; attribution retained).

SOAP = Shampoo + Adam in the eigenbasis of Shampoo's preconditioner.
From Vyas et al. 2024 (arXiv:2409.11321).

Reference: https://github.com/nikhilvyas/SOAP
"""
from __future__ import annotations

import torch
import torch.optim as optim
from itertools import chain

from tritium.torch.optim_cautious import cautious_decoupled_wd_


class SOAP(optim.Optimizer):
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321).

    Parameters:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 0.003).
        betas: Adam's betas (default: (0.95, 0.95)).
        shampoo_beta: Beta for preconditioner EMA. -1 = use betas[1].
        eps: Epsilon for numerical stability (default: 1e-8).
        weight_decay: AdamW-style weight decay (default: 0.01).
        precondition_frequency: Steps between eigenbasis updates (default: 10).
        max_precond_dim: Max dimension for preconditioning (default: 10000).
        merge_dims: Merge small dims for better preconditioning (default: False).
        precondition_1d: Whether to precondition 1D gradients (default: False).
        correct_bias: Use Adam bias correction (default: True).
    """

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95),
                 shampoo_beta=-1, eps=1e-8, weight_decay=0.01,
                 precondition_frequency=10, max_precond_dim=10000,
                 merge_dims=False, precondition_1d=False,
                 correct_bias=True, cautious_wd=False):
        defaults = dict(lr=lr, betas=betas, shampoo_beta=shampoo_beta,
                        eps=eps, weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency,
                        max_precond_dim=max_precond_dim,
                        merge_dims=merge_dims,
                        precondition_1d=precondition_1d,
                        correct_bias=correct_bias,
                        cautious_wd=cautious_wd)
        super().__init__(params, defaults)
        self._data_format = "channels_first"

    def merge_dims(self, grad, max_precond_dim):
        shape = grad.shape
        new_shape = []
        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape
        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)
        return grad.reshape(new_shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                if 'Q' not in state:
                    self.init_preconditioner(
                        grad, state,
                        precondition_frequency=group['precondition_frequency'],
                        precondition_1d=group['precondition_1d'],
                        shampoo_beta=(group['shampoo_beta']
                                      if group['shampoo_beta'] >= 0
                                      else group["betas"][1]),
                        max_precond_dim=group['max_precond_dim'],
                        merge_dims=group["merge_dims"])
                    self.update_preconditioner(
                        grad, state,
                        max_precond_dim=group['max_precond_dim'],
                        merge_dims=group["merge_dims"],
                        precondition_1d=group["precondition_1d"])
                    continue

                grad_projected = self.project(
                    grad, state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group['max_precond_dim'])

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(),
                                            alpha=(1.0 - beta2))
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bc1 = 1.0 - beta1 ** state["step"]
                    bc2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * (bc2 ** .5) / bc1

                norm_grad = self.project_back(
                    exp_avg / denom, state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group['max_precond_dim'])

                wd = group["weight_decay"]
                if wd > 0.0 and group.get("cautious_wd", False):
                    # Cautious decoupled WD (ADR 0030, SPECULATIVE, flag-gated,
                    # default off). Fold the decoupled decay lr*wd*p into the
                    # update only on entries where the update already agrees in
                    # sign with the param (update*p > 0), so decay never fights
                    # the step. Must clear an end-to-end A/B before adoption.
                    update = norm_grad.mul(step_size)
                    cautious_decoupled_wd_(p, update, group["lr"], wd)
                else:
                    # Plain decoupled WD path — byte-identical to pre-0030.
                    p.add_(norm_grad, alpha=-step_size)
                    if wd > 0.0:
                        p.add_(p, alpha=(-group["lr"] * wd))

                self.update_preconditioner(
                    grad, state,
                    max_precond_dim=group['max_precond_dim'],
                    merge_dims=group["merge_dims"],
                    precondition_1d=group["precondition_1d"])
        return loss

    def init_preconditioner(self, grad, state, precondition_frequency=10,
                            shampoo_beta=0.95, max_precond_dim=10000,
                            precondition_1d=False, merge_dims=False):
        state['GG'] = []
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state['GG'].append([])
            else:
                state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0],
                                               device=grad.device))
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)
            for sh in grad.shape:
                if sh > max_precond_dim:
                    state['GG'].append([])
                else:
                    state['GG'].append(torch.zeros(sh, sh, device=grad.device))
        state['Q'] = None
        state['precondition_frequency'] = precondition_frequency
        state['shampoo_beta'] = shampoo_beta

    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        original_shape = grad.shape
        if merge_dims:
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
            else:
                perm = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(perm)
        if merge_dims:
            grad = grad.reshape(original_shape)
        return grad

    def update_preconditioner(self, grad, state, max_precond_dim=10000,
                              merge_dims=False, precondition_1d=False):
        if state["Q"] is not None:
            state["exp_avg"] = self.project_back(
                state["exp_avg"], state,
                merge_dims=merge_dims, max_precond_dim=max_precond_dim)
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state['GG'][0].lerp_(
                    grad.unsqueeze(1) @ grad.unsqueeze(0),
                    1 - state['shampoo_beta'])
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim)
            else:
                new_grad = grad
            for idx, sh in enumerate(new_grad.shape):
                if sh <= max_precond_dim:
                    outer = torch.tensordot(
                        new_grad, new_grad,
                        dims=[[*chain(range(idx),
                                      range(idx + 1, len(new_grad.shape)))]] * 2)
                    state['GG'][idx].lerp_(outer, 1 - state['shampoo_beta'])

        if state['Q'] is None:
            state['Q'] = self.get_orthogonal_matrix(state['GG'])
        if state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0:
            state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim,
                                                        merge_dims)
        if state["step"] > 0:
            state["exp_avg"] = self.project(
                state["exp_avg"], state,
                merge_dims=merge_dims, max_precond_dim=max_precond_dim)

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        original_shape = grad.shape
        if merge_dims:
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                perm = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(perm)
        if merge_dims:
            grad = grad.reshape(original_shape)
        return grad

    def get_orthogonal_matrix(self, mat):
        matrix = []
        for m in mat:
            if len(m) == 0:
                matrix.append([])
                continue
            if m.data.dtype != torch.float:
                matrix.append(m.data.float())
            else:
                matrix.append(m.data)
        final = []
        for m in matrix:
            if len(m) == 0:
                final.append([])
                continue
            try:
                _, Q = torch.linalg.eigh(
                    m + 1e-30 * torch.eye(m.shape[0], device=m.device))
            except Exception:
                _, Q = torch.linalg.eigh(
                    m.to(torch.float64) + 1e-30 * torch.eye(
                        m.shape[0], device=m.device))
                Q = Q.to(m.dtype)
            Q = torch.flip(Q, [1])
            final.append(Q)
        return final

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000,
                                  merge_dims=False):
        precond_list = state['GG']
        orth_list = state['Q']
        matrix = []
        orth_matrix = []
        for m, o in zip(precond_list, orth_list):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                continue
            matrix.append(m.data.float())
            orth_matrix.append(o.data.float())

        if merge_dims:
            exp_avg_sq = self.merge_dims(state['exp_avg_sq'], max_precond_dim)
        else:
            exp_avg_sq = state['exp_avg_sq']

        final = []
        for ind, (m, o) in enumerate(zip(matrix, orth_matrix)):
            if len(m) == 0:
                final.append([])
                continue
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o = o[:, sort_idx]
            power_iter = m @ o
            Q, _ = torch.linalg.qr(power_iter)
            final.append(Q)

        if merge_dims:
            exp_avg_sq = exp_avg_sq.reshape(state['exp_avg_sq'].shape)
        state['exp_avg_sq'] = exp_avg_sq
        return final


__all__ = ['SOAP']

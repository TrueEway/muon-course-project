"""Exact polar directions and a CPU adaptation of the reference Muon update.

The quintic coefficients, EMA/Nesterov convention and shape scaling follow
KellerJordan/Muon (MIT). See THIRD_PARTY_NOTICES.md. Float32 is intentional on
CPU; the reference GPU routine uses bfloat16. No distributed code is used.
"""
from __future__ import annotations

import numpy as np
import torch


def polar_numpy(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError('Muon requires a matrix')
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    if not s.size or s[0] == 0:
        return np.zeros_like(matrix)
    keep = s > max(matrix.shape) * np.finfo(matrix.dtype).eps * s[0]
    return u[:, keep] @ vh[keep]


def ns_numpy(matrix: np.ndarray, steps=5, eps=1e-7) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError('Muon requires a matrix')
    if steps < 1:
        raise ValueError('steps must be positive')
    x = matrix.copy()
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (np.linalg.norm(x) + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ x
    return x.T if transpose else x


@torch.no_grad()
def polar_torch(matrix: torch.Tensor) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    if not s.numel() or s[0] == 0:
        return torch.zeros_like(matrix)
    keep = s > max(matrix.shape) * torch.finfo(matrix.dtype).eps * s[0]
    return u[:, keep] @ vh[keep]


@torch.no_grad()
def ns_torch(matrix: torch.Tensor, steps=5, eps=1e-7) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError('Muon requires a matrix')
    if steps < 1:
        raise ValueError('steps must be positive')
    x = matrix.float()
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ x
    return (x.T if transpose else x).to(matrix.dtype)


class MatrixMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, mode='ns', steps=5,
                 weight_decay=0.01, nesterov=True):
        if mode not in {'svd', 'ns'}:
            raise ValueError(mode)
        super().__init__(params, dict(lr=lr, momentum=momentum, mode=mode,
                                     steps=steps, weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError('Only hidden matrix weights belong in MatrixMuon')
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                buf = state['momentum']
                beta = group['momentum']
                buf.lerp_(p.grad, 1 - beta)
                direction = p.grad.lerp(buf, beta) if group['nesterov'] else buf
                q = polar_torch(direction) if group['mode'] == 'svd' else ns_torch(direction, group['steps'])
                q = q * max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(q, alpha=-group['lr'])
        return loss

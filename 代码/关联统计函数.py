"""Within- and between-source association estimates from admitted analysis units (Figure 5)."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import torch
from CODE_TITMC_FIG03_evidence_maturity import gpu_device

def _gpu_rank(values, multiplicity=None, *, device='cuda:0'):
    """Average tie ranks, reranked exactly for each PF bootstrap multiplicity."""
    values = torch.as_tensor(values, dtype=torch.float64, device=gpu_device(device))
    order = torch.argsort(values, stable=True)
    v = values[order]
    first = torch.ones(len(v), dtype=torch.bool, device=v.device)
    first[1:] = v[1:] != v[:-1]
    ties = first.cumsum(0) - 1
    w = torch.ones((1, len(v)), dtype=torch.float64, device=v.device) if multiplicity is None else multiplicity
    mass = torch.zeros_like(w).scatter_add_(1, ties.expand(len(w), -1), w[:, order])
    rank = mass.cumsum(1) - (mass - 1) / 2
    out = torch.empty_like(w)
    out[:, order] = rank[:, ties]
    return out[0] if multiplicity is None else out

def _gpu_corr(x, y, w=None, *, device='cuda:0'):
    x = torch.as_tensor(x, dtype=torch.float64, device=gpu_device(device))
    y = torch.as_tensor(y, dtype=torch.float64, device=x.device)
    w = torch.ones_like(x) if w is None else w
    sw = w.sum(-1, keepdim=True)
    dx = x - (x * w).sum(-1, keepdim=True) / sw
    dy = y - (y * w).sum(-1, keepdim=True) / sw
    den = ((w * dx.square()).sum(-1) * (w * dy.square()).sum(-1)).sqrt()
    return torch.where(den > 1e-20, ((w * dx * dy).sum(-1) / den).clamp(-1, 1), float('nan'))

def _gpu_group_medians(values, group, size, *, device='cuda:0'):
    values = torch.as_tensor(values, dtype=torch.float64, device=gpu_device(device))
    if values.ndim == 1:
        values = values[:, None]
    counts = torch.bincount(group, minlength=size)
    offset = counts.cumsum(0) - counts
    (lo, hi) = (offset + (counts - 1) // 2, offset + counts // 2)
    result = []
    for j in range(values.shape[1]):
        order = torch.argsort(values[:, j], stable=True)
        order = order[torch.argsort(group[order], stable=True)]
        v = values[order, j]
        result.append((v[lo] + v[hi]) / 2)
    return torch.stack(result, dim=1)

def _gpu_replicates(z, x, y, paper, controls, family_multiplicity, *, device='cuda:0'):
    (ids, labels) = pd.factorize(z[paper].astype('string'), sort=False)
    group = torch.as_tensor(ids, device=gpu_device(device), dtype=torch.int64)
    counts = torch.bincount(group, minlength=len(labels)).double()
    multiplier = family_multiplicity[:, group]
    w = multiplier / counts[group]
    values = torch.as_tensor(z[[x, y, *controls]].to_numpy(float), device=group.device, dtype=torch.float64)
    centered = []
    for j in range(values.shape[1]):
        rank = _gpu_rank(values[:, j], multiplier, device=device)
        sums = torch.zeros((len(multiplier), len(labels)), dtype=torch.float64, device=group.device)
        sums.scatter_add_(1, group.expand(len(multiplier), -1), rank)
        centered.append(rank - (sums / counts)[:, group])
    (rx, ry) = centered[:2]
    if controls:
        design = torch.stack(centered[2:], dim=-1)
        gram = design.transpose(1, 2) @ (design * w[..., None])
        target = torch.stack([rx, ry], dim=-1)
        rhs = design.transpose(1, 2) @ (target * w[..., None])
        residual = target - design @ (torch.linalg.pinv(gram, hermitian=True) @ rhs)
        (rx, ry) = residual.unbind(-1)
    return _gpu_corr(rx, ry, w, device=device)

def _effect_once(z, x, y, paper, controls, *, device='cuda:0'):
    (ids, labels) = pd.factorize(z[paper].astype('string'), sort=False)
    device = gpu_device(device)
    group = torch.as_tensor(ids, device=device, dtype=torch.int64)
    v = torch.as_tensor(z[[x, y]].to_numpy(float), device=device, dtype=torch.float64)
    counts = torch.bincount(group, minlength=len(labels))
    low = torch.full((len(labels), 2), float('inf'), device=device, dtype=torch.float64)
    high = torch.full_like(low, -float('inf'))
    low.scatter_reduce_(0, group[:, None].expand(-1, 2), v, reduce='amin')
    high.scatter_reduce_(0, group[:, None].expand(-1, 2), v, reduce='amax')
    informative = int(((counts >= 2) & (high - low > 0).all(1)).sum())
    within = _gpu_replicates(z, x, y, paper, controls, torch.ones((1, len(labels)), device=device, dtype=torch.float64), device=device)[0]
    raw = _gpu_corr(_gpu_rank(v[:, 0], device=device), _gpu_rank(v[:, 1], device=device), device=device)
    med = _gpu_group_medians(v, group, len(labels), device=device)
    between = _gpu_corr(_gpu_rank(med[:, 0], device=device), _gpu_rank(med[:, 1], device=device), device=device) if len(labels) >= 3 else float('nan')
    return dict(effect=float(within), raw_rho=float(raw), between_rho=float(between), informative_paperfamily_n=informative)

def _bootstrap_effect(z, x, y, paper, controls, reps, seed, return_draws=False, *, device='cuda:0', index_device=None, legacy_cpu_probability=False):
    if reps <= 0:
        return (np.nan, np.nan, np.nan)
    (device, nf) = (gpu_device(device), z[paper].nunique())
    draw_device = torch.device(index_device) if index_device is not None else device
    gen = torch.Generator(device=draw_device).manual_seed(seed)
    vals = []
    for begin in range(0, reps, 32):
        nb = min(32, reps - begin)
        draws = torch.randint(nf, (nb, nf), generator=gen, device=draw_device).to(device)
        m = torch.zeros((nb, nf), device=device, dtype=torch.float64)
        m.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
        vals.append(_gpu_replicates(z, x, y, paper, controls, m, device=device))
    a = torch.cat(vals)
    a = a[torch.isfinite(a)]
    if return_draws:
        return a
    if len(a) < max(20, reps // 5):
        return (np.nan, np.nan, np.nan)
    (lo, hi) = torch.quantile(a, torch.tensor([0.025, 0.975], device=device, dtype=torch.float64))
    p = (2 * torch.minimum((a <= 0).sum() + 1, (a >= 0).sum() + 1) / (len(a) + 1)).clamp(max=1)
    if legacy_cpu_probability:
        numerator = 2 * torch.minimum((a <= 0).sum() + 1, (a >= 0).sum() + 1)
        p = (numerator.to(torch.float64) / (len(a) + 1)).to(torch.float32).clamp(max=1)
    return (float(lo), float(hi), float(p))

def _leave_one_family_out_stability(z, x, y, paper, controls, reference_effect, *, device='cuda:0'):
    empty = dict(lofo_n=0, lofo_sign_stability=np.nan, lofo_min=np.nan, lofo_max=np.nan)
    nf = z[paper].nunique()
    if not np.isfinite(reference_effect) or nf < 4:
        return empty
    vals = []
    for begin in range(0, nf, 32):
        nb = min(32, nf - begin)
        m = torch.ones((nb, nf), device=gpu_device(device), dtype=torch.float64)
        m[torch.arange(nb, device=m.device), torch.arange(begin, begin + nb, device=m.device)] = 0
        vals.append(_gpu_replicates(z, x, y, paper, controls, m, device=device))
    a = torch.cat(vals)
    a = a[torch.isfinite(a)]
    if not len(a):
        return empty
    return dict(lofo_n=len(a), lofo_sign_stability=float((a.sign() == np.sign(reference_effect)).double().mean()), lofo_min=float(a.min()), lofo_max=float(a.max()))

def _random_effects_family_meta(z, x, y, paper, min_families=5, *, device='cuda:0'):
    (effects, device) = ([], gpu_device(device))
    for (_, g) in z.groupby(paper, observed=True, sort=False):
        if len(g) < 4:
            continue
        v = torch.as_tensor(g[[x, y]].to_numpy(float), device=device, dtype=torch.float64)
        rho = _gpu_corr(_gpu_rank(v[:, 0], device=device), _gpu_rank(v[:, 1], device=device), device=device)
        if bool(torch.isfinite(rho)):
            effects.append(torch.stack([rho.clamp(-0.999999, 0.999999).atanh(), torch.tensor(1 / (len(g) - 3), device=device, dtype=torch.float64)]))
    k = len(effects)
    if k < min_families:
        return dict(meta_family_n=k, meta_rho=np.nan, meta_ci_low=np.nan, meta_ci_high=np.nan, meta_tau2=np.nan, meta_i2=np.nan, meta_q=np.nan, meta_p=np.nan)
    (zf, variance) = torch.stack(effects).unbind(1)
    w = 1 / variance
    fixed = (w * zf).sum() / w.sum()
    q = (w * (zf - fixed).square()).sum()
    c = w.sum() - w.square().sum() / w.sum()
    tau2 = ((q - (k - 1)) / c).clamp_min(0)
    wr = 1 / (variance + tau2)
    (mu, se) = ((wr * zf).sum() / wr.sum(), wr.sum().rsqrt())
    return dict(meta_family_n=k, meta_rho=float(mu.tanh()), meta_ci_low=float((mu - 1.96 * se).tanh()), meta_ci_high=float((mu + 1.96 * se).tanh()), meta_tau2=float(tau2), meta_i2=float(((q - k + 1) / q).clamp_min(0) * 100), meta_q=float(q), meta_p=float(torch.erfc((mu / se).abs() / math.sqrt(2))))

def benjamini_hochberg(p, *, device='cuda:0'):
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return out
    a = torch.as_tensor(valid.to_numpy(float), dtype=torch.float64, device=gpu_device(device)).clamp(0, 1)
    order = torch.argsort(a, stable=True)
    q = a[order] * len(a) / torch.arange(1, len(a) + 1, device=a.device, dtype=a.dtype)
    q = q.flip(0).cummin(0).values.flip(0).clamp(0, 1)
    restored = torch.empty_like(q)
    restored[order] = q
    out.loc[valid.index] = restored.cpu().numpy()
    return out

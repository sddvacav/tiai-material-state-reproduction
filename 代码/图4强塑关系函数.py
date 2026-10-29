"""Equal-source strength–elongation moments and within-source bootstrap intervals (Figure 4)."""
from __future__ import annotations
import math
import torch

def source_weights(groups):
    (_, g) = torch.unique(groups, return_inverse=True)
    counts = torch.bincount(g).to(torch.float64)
    w = 1 / counts[g]
    return (g, w / w.sum(), len(counts))

def weighted_quantile(values, weights, q, *, device='cuda'):
    order = torch.argsort(values, stable=True)
    cdf = weights[order].cumsum(0) / weights.sum()
    idx = torch.searchsorted(cdf, torch.as_tensor(q, device=device, dtype=torch.float64))
    return values[order][idx.clamp(max=len(values) - 1)]

def association(xy, groups, *, bootstrap_draws=None, generator=None, include_loso=False, moments_only=False, device='cuda'):
    requested = torch.device(device)
    if xy.device.type != requested.type or groups.device != xy.device:
        raise ValueError('Association inputs must match the explicitly selected statistics device')
    device = xy.device
    draws_n = N_BOOT if bootstrap_draws is None else bootstrap_draws
    if draws_n < 100:
        raise ValueError('Source bootstrap needs at least 100 draws')
    (g, w, G) = source_weights(groups)
    mean = (w[:, None] * xy).sum(0)
    cov = (xy - mean).T @ (w[:, None] * (xy - mean))
    raw = xy - xy.mean(0)
    rawcov = raw.T @ raw / len(xy)
    n = torch.bincount(g).to(torch.float64)
    means = torch.zeros((G, 2), device=device, dtype=torch.float64)
    means.scatter_add_(0, g[:, None].expand(-1, 2), xy)
    means /= n[:, None]
    residual = xy - means[g]
    wcov = residual.T @ (w[:, None] * residual)
    bcov = (means - means.mean(0)).T @ (means - means.mean(0)) / G
    result = {}
    for (label, c) in (('pooled', rawcov), ('balanced', cov), ('within', wcov), ('between', bcov)):
        result[label + '_slope_mpa_per_pct'] = float((c[0, 1] / c[0, 0]).item()) if c[0, 0] > 1e-12 else None
        result[label + '_r'] = float((c[0, 1] / torch.sqrt(c[0, 0] * c[1, 1])).item()) if c[0, 0] * c[1, 1] > 1e-12 else None
    if moments_only:
        result['el_between_source_variance_share'] = float(bcov[0, 0] / cov[0, 0])
        result['uts_between_source_variance_share'] = float(bcov[1, 1] / cov[1, 1])
        return result
    result['simpson_sign_reversal'] = bool(result['pooled_slope_mpa_per_pct'] is not None and result['within_slope_mpa_per_pct'] is not None and (result['pooled_slope_mpa_per_pct'] * result['within_slope_mpa_per_pct'] < 0))
    result['sources_with_multiple_states'] = int((n > 1).sum())
    (result['centroid_el'], result['centroid_uts']) = mean.cpu().tolist()
    result['n_eff'] = float(1 / w.square().sum())
    for (column, name) in ((0, 'el'), (1, 'uts')):
        q = weighted_quantile(xy[:, column], w, [0.25, 0.5, 0.75], device=device).cpu().tolist()
        for (suffix, value) in zip(('q25', 'median', 'q75'), q):
            result[name + '_' + suffix] = value
    scaling = torch.tensor([40.0, 1600.0], device=device, dtype=torch.float64)
    (eig, vec) = torch.linalg.eigh(cov / scaling[:, None] / scaling[None, :])
    result['principal_angle_deg'] = float(torch.atan2(vec[1, -1], vec[0, -1]) * 180 / math.pi)
    moments = torch.zeros((G, 2), device=device, dtype=torch.float64)
    terms = torch.stack((residual[:, 0].square(), residual[:, 0] * residual[:, 1]), 1)
    moments.scatter_add_(0, g[:, None].expand(-1, 2), terms / n[g, None])
    draws = torch.randint(G, (draws_n, G), device=device, generator=generator)
    sums = moments[draws].sum(1)
    valid = sums[:, 0] > 1e-12
    if int(valid.sum()) >= draws_n * 0.95 and int((n > 1).sum()) >= 5:
        ci = torch.quantile(sums[valid, 1] / sums[valid, 0], torch.tensor([0.025, 0.975], device=device, dtype=torch.float64))
        (result['within_slope_ci_low'], result['within_slope_ci_high']) = ci.cpu().tolist()
    else:
        result['within_slope_ci_low'] = result['within_slope_ci_high'] = None
    if include_loso:
        within = moments.sum(0)[None, :] - moments
        sx = means[:, 0].sum() - means[:, 0]
        sy = means[:, 1].sum() - means[:, 1]
        remaining_g = max(G - 1, 1)
        bxx = means[:, 0].square().sum() - means[:, 0].square() - sx.square() / remaining_g
        bxy = (means[:, 0] * means[:, 1]).sum() - means[:, 0] * means[:, 1] - sx * sy / remaining_g
        between = torch.stack((bxx, bxy), 1)
        (source_sx, source_sy) = (n * means[:, 0], n * means[:, 1])
        source_sxx = n * (moments[:, 0] + means[:, 0].square())
        source_sxy = n * (moments[:, 1] + means[:, 0] * means[:, 1])
        remaining_n = len(xy) - n
        (px, py) = (source_sx.sum() - source_sx, source_sy.sum() - source_sy)
        pooled = torch.stack((source_sxx.sum() - source_sxx - px.square() / remaining_n, source_sxy.sum() - source_sxy - px * py / remaining_n), 1)
        tensors = {'pooled': pooled, 'balanced': within + between, 'within': within, 'between': between}
        slopes = {name: torch.where((m[:, 0] > 1e-12) & (remaining_n > 0), m[:, 1] / m[:, 0], torch.nan) for (name, m) in tensors.items()}
        values = {name: v.cpu().tolist() for (name, v) in slopes.items()}
        removed = n.to(torch.int64).cpu().tolist()
        result['leave_one_source_out'] = [{'source_code': i, 'n_states_removed': removed[i], **{name + '_slope_mpa_per_pct': value[i] if math.isfinite(value[i]) else None for (name, value) in values.items()}} for i in range(G)]
        finite = slopes['within'][torch.isfinite(slopes['within'])]
        base = result['within_slope_mpa_per_pct']
        result['within_loso_summary'] = {'n_estimable': len(finite), 'n_sources': G, 'minimum_slope': float(finite.min()) if len(finite) else None, 'maximum_slope': float(finite.max()) if len(finite) else None, 'max_absolute_change': float((finite - base).abs().max()) if len(finite) and base is not None else None, 'all_estimable_deletions_preserve_sign': bool((finite * base > 0).all()) if len(finite) and base is not None and (base != 0) else None, 'scope': 'Delete one complete source; fixed admitted states and endpoint definitions.'}
        result['bootstrap_draws'] = draws_n
        result['bootstrap_valid_draws'] = int(valid.sum())
    return result

N_BOOT = 2000

(SEED, N_BOOT, REGION_BOOT) = (20260908, 2000, 200)

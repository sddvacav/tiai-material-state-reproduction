"""Five-year endpoint windows and source-bootstrap intervals (Figure 7)."""
from __future__ import annotations
import torch

(YEAR_LO, YEAR_HI) = (1950, 2026)

(WIN, MIN_PF_Q, N_BOOT) = (5, 20, 300)

GPU = {}

ENDPOINTS = [dict(id='uts', prop='ultimate_tensile_strength', unit='mpa', display_unit='MPa', en='UTS', zh='抗拉强度', main=True, tensile=True), dict(id='ys', prop='yield_strength_unspecified', unit='mpa', display_unit='MPa', en='Yield strength', zh='屈服强度', main=True, tensile=True), dict(id='el', prop='elongation_unspecified', unit='%', display_unit='%', en='Unspecified EL', zh='未指定延伸率', main=True, tensile=True), dict(id='mod', prop='elastic_modulus', unit='gpa', display_unit='GPa', en='Elastic modulus', zh='弹性模量', main=True, tensile=False), dict(id='hv', prop='vickers_hardness', unit='hv', display_unit='HV', en='Vickers hardness', zh='维氏硬度', main=True, tensile=False), dict(id='rp02', prop='proof_stress_rp0_2', unit='mpa', display_unit='MPa', en='Proof stress', zh='规定塑性延伸强度', main=False, tensile=True), dict(id='el_total', prop='total_elongation', unit='%', display_unit='%', en='Total elongation', zh='总延伸率', main=False, tensile=True), dict(id='el_uniform', prop='uniform_elongation', unit='%', display_unit='%', en='Uniform elongation', zh='均匀延伸率', main=False, tensile=True), dict(id='hv_micro', prop='vickers_microhardness', unit='hv', display_unit='HV', en='Microhardness', zh='显微硬度', main=False, tensile=False)]

QS = (0.5, 0.75, 0.9, 0.95)

def inverse_edf(values, weights, qs=QS):
    order = torch.argsort(values, stable=True)
    (v, w) = (values[order], weights[order])
    cdf = torch.cumsum(w, 0) / w.sum()
    q = torch.tensor(qs, dtype=torch.float64, device=v.device)
    indices = torch.searchsorted(cdf + 1e-12, q).clamp(max=len(v) - 1)
    return v[indices]

def balanced_quantiles(values, groups):
    (_, ids) = torch.unique(groups, sorted=True, return_inverse=True)
    sizes = torch.bincount(ids).to(torch.float64)
    return inverse_edf(values, 1 / sizes[ids])

def window_statistics(values, groups, temperatures):
    (unique, ids) = torch.unique(groups, sorted=True, return_inverse=True)
    (n, g) = (len(values), len(unique))
    sizes = torch.bincount(ids).to(torch.float64)
    order = torch.argsort(values, stable=True)
    (v, sorted_ids) = (values[order], ids[order])
    q = balanced_quantiles(values, groups)
    raw = inverse_edf(values, torch.ones_like(values))
    grouped_order = torch.argsort(sorted_ids, stable=True)
    grouped = v[grouped_order]
    ends = sizes.to(torch.int64).cumsum(0)
    starts = ends - sizes.to(torch.int64)
    med = (grouped[starts + (sizes.to(torch.int64) - 1) // 2] + grouped[starts + sizes.to(torch.int64) // 2]) / 2
    median_q = inverse_edf(med, torch.ones_like(med))
    generator = torch.Generator(device=values.device).manual_seed(20260907)
    sampled = torch.randint(g, (N_BOOT, g), device=values.device, generator=generator)
    multiplicity = torch.zeros((N_BOOT, g), dtype=torch.float64, device=values.device)
    multiplicity.scatter_add_(1, sampled, torch.ones_like(sampled, dtype=torch.float64))
    cdf = (multiplicity[:, sorted_ids] / sizes[sorted_ids]).cumsum(1) / g
    targets = torch.tensor(QS, dtype=torch.float64, device=values.device).expand(N_BOOT, -1).contiguous()
    indices = torch.searchsorted((cdf + 1e-12).contiguous(), targets).clamp(max=n - 1)
    draws = v[indices]
    bounds = torch.quantile(draws, torch.tensor([0.025, 0.975], dtype=torch.float64, device=values.device), dim=0)
    keep = ~torch.isin(ids, torch.argsort(sizes, descending=True, stable=True)[:min(5, g)])
    dropped = balanced_quantiles(values[keep], groups[keep]) if keep.any() else None
    exact = temperatures == 25
    q25 = balanced_quantiles(values[exact], groups[exact]) if exact.any() else None
    out = {'n_rows': n, 'n_pf': g, 'plot_supported': g >= MIN_PF_Q, 'n_tail_q90_pf': int(torch.unique(groups[values >= q[2]]).numel()), 'n_pf_exact25': int(torch.unique(groups[exact]).numel()), 'n_pf_drop_top5': int(torch.unique(groups[keep]).numel())}
    for (i, name) in enumerate(('q50', 'q75', 'q90', 'q95')):
        out[name] = float(q[i].item())
        out[name + '_lo'] = float(bounds[0, i].item())
        out[name + '_hi'] = float(bounds[1, i].item())
        out[name + '_row_weighted'] = float(raw[i].item())
        out[name + '_family_median_diagnostic'] = float(median_q[i].item())
        out[name + '_drop_top5'] = float(dropped[i].item()) if dropped is not None else None
        out[name + '_exact25'] = float(q25[i].item()) if q25 is not None else None
    GPU.update(weighted_quantile_device=str(q.device), bootstrap_device=str(draws.device), bootstrap_draws=N_BOOT)
    return out

def temporal_windows(values, years, groups, temperatures, spec, *, screening=False, window_ends=None):
    """Original five-calendar-year windows and 300 source resamples, unchanged."""
    if values.device.type != 'cuda':
        raise RuntimeError('GPU_REQUIRED: temporal windows')
    rows = []
    for end in range(YEAR_LO, YEAR_HI + 1):
        if window_ends is not None and end not in window_ends:
            continue
        inside = (years >= end - WIN + 1) & (years <= end)
        if not inside.any():
            continue
        row = window_statistics(values[inside], groups[inside], temperatures[inside])
        row.update(endpoint_id=spec['id'], lane='SCREENING_CONDITIONS_AND_VALUE_KIND_UNCONTROLLED' if screening else 'MAIN_REPORTED_RT', year=end, window_lo=end - WIN + 1, window_hi=end, property=spec['prop'], unit=spec['display_unit'])
        if screening:
            row['plot_supported'] = False
        rows.append(row)
    return rows

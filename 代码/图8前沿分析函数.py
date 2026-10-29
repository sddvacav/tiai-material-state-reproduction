"""Observed strength–elongation frontier, dated hypervolume and source-balanced portfolios (Figure 8)."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import torch

STEM = 'TITMC_FIG07_pareto_audit'

(ERA_CUT, N_PORTFOLIOS, SEED) = (2016, 400, 20260907)

def pareto(xy):
    keep = torch.ones(xy.shape[0], device='cuda', dtype=torch.bool)
    for start in range(0, len(xy), 256):
        query = xy[start:start + 256]
        dominates = (xy[None] >= query[:, None]).all(-1) & (xy[None] > query[:, None]).any(-1)
        keep[start:start + len(query)] = ~dominates.any(1)
    return keep

def hv(xy):
    """Batched exact union of rectangles from the common physical zero reference."""
    if xy.ndim == 2:
        xy = xy.unsqueeze(0)
    if xy.shape[1] == 0:
        return torch.zeros(xy.shape[0], device='cuda', dtype=torch.float64)
    order = torch.argsort(xy[:, :, 0], dim=1, descending=True)
    e = xy[:, :, 0].gather(1, order)
    u = xy[:, :, 1].gather(1, order).cummax(1).values
    previous = torch.cat((torch.zeros_like(u[:, :1]), u[:, :-1]), 1)
    return (e * (u - previous)).sum(1)

def portfolios(xy, groups):
    """All source families enter each portfolio; one actual pair is drawn per family."""
    (unique, ids) = torch.unique(groups, return_inverse=True)
    counts = torch.bincount(ids)
    order = torch.argsort(ids)
    starts = torch.cat((torch.zeros(1, device='cuda', dtype=torch.long), counts.cumsum(0)[:-1]))
    slots = (torch.rand((N_PORTFOLIOS, len(unique)), device='cuda') * counts).long() + starts
    return hv(xy[order[slots]])

def analyze_admitted_pairs(xy, groups, years, dated, n_sources):
    """Original GPU analyses after pairing/admission; shared by compute and replay."""
    if not torch.cuda.is_available() or any((value.device.type != 'cuda' for value in (xy, groups, years, dated))):
        raise RuntimeError('GPU_REQUIRED: observed paired-state analysis')
    if not bool(torch.isfinite(xy).all()) or len(xy) != len(groups) or len(xy) != len(years) or (len(xy) != len(dated)):
        raise ValueError('Paired-state numerical coordinates differ')
    torch.manual_seed(SEED)
    device = xy.device
    counts = torch.bincount(groups, minlength=n_sources).to(torch.float64)
    if not bool((counts > 0).all()):
        raise ValueError('Empty encoded source group')
    main_front = pareto(xy)
    (old_mask, new_mask) = (dated & (years < ERA_CUT), dated & (years >= ERA_CUT))
    old_front = torch.zeros_like(main_front)
    new_front = torch.zeros_like(main_front)
    old_front[old_mask] = pareto(xy[old_mask])
    new_front[new_mask] = pareto(xy[new_mask])
    denominator = hv(xy[dated]).item()
    if denominator <= 0:
        raise ValueError('Nonpositive dated hypervolume normalizer')
    trends = []
    for year in range(2000, 2027):
        mask = dated & (years <= year)
        n = int(mask.sum().item())
        g = torch.unique(groups[mask]).numel()
        if n == 0:
            continue
        spread = portfolios(xy[mask], groups[mask]) / denominator
        q = torch.quantile(spread, torch.tensor([0.1, 0.5, 0.9], device=device, dtype=torch.float64))
        trends.append(dict(year=year, n_pairs=n, n_paperfamilies=g, raw_hv_ratio=hv(xy[mask]).item() / denominator, portfolio_p10=q[0].item(), portfolio_median=q[1].item(), portfolio_p90=q[2].item()))
    eg = torch.linspace(0, xy[:, 0].max(), 81, device=device, dtype=torch.float64)
    ug = torch.linspace(0, xy[:, 1].max(), 81, device=device, dtype=torch.float64)
    (xx, yy) = torch.meshgrid(eg, ug, indexing='xy')
    grid = torch.stack((xx.flatten(), yy.flatten()), 1)
    onehot = torch.nn.functional.one_hot(groups, n_sources).to(torch.float32)
    fractions = []
    for chunk in grid.split(256):
        dom = (xy[None] >= chunk[:, None]).all(-1).to(torch.float32)
        fractions.append((dom @ onehot > 0).to(torch.float64).mean(1))
    fraction = torch.cat(fractions)
    attainment = pd.DataFrame(dict(el_threshold=grid[:, 0].cpu().tolist(), uts_threshold=grid[:, 1].cpu().tolist(), fraction=fraction.cpu().tolist()))
    return {'on_front': main_front, 'old_front': old_front, 'recent_front': new_front, 'old_mask': old_mask, 'recent_mask': new_mask, 'counts': counts, 'pf_weight': 1 / counts[groups], 'hv_normalizer_dated': denominator, 'history': pd.DataFrame(trends), 'attainment': attainment, 'kish_state_weights': float((1 / counts[groups]).sum().square().div((1 / counts[groups]).square().sum()))}

def reproduce_saved_pairs(source: Path) -> dict:
    """Replay adopted figure analyses from complete saved pairs, never re-admit sources."""
    if not torch.cuda.is_available():
        raise RuntimeError('GPU_REQUIRED: Figure 8 frontier reproduction')
    row = pd.read_csv(source / f'{STEM}_source_data.csv', float_precision='round_trip', dtype={'feature_row_id': 'Int64'})
    states = pd.read_csv(source / f'{STEM}_states.csv', float_precision='round_trip', dtype={'feature_row_id': 'Int64'})
    layers = pd.read_csv(source / f'{STEM}_analysis_matrix.csv', float_precision='round_trip')
    support = pd.read_csv(source / f'{STEM}_paperfamily_support.csv', float_precision='round_trip')
    receipt = json.loads((source / f'{STEM}_claim_receipt.json').read_text(encoding='utf-8'))
    if receipt['n_portfolios'] != N_PORTFOLIOS or receipt['seed'] != SEED or receipt['hv_reference'] != [0.0, 0.0]:
        raise ValueError('Recorded frontier procedure differs')
    if len(row) != receipt['n_pairs'] or len(states) != receipt['n_pairs_checked'] or (not row.observation_id.is_unique) or (not states.observation_id_uts.is_unique):
        raise ValueError('Saved paired-state population differs')
    admitted = states.loc[states.admitted.eq(True)].set_index('observation_id_uts')
    if set(admitted.index) != set(row.observation_id):
        raise ValueError('Admitted pair IDs differ')
    admitted = admitted.loc[row.observation_id].reset_index()
    for (field, paired) in (('paired_observation_id', 'observation_id_el'), ('paper_family_id', 'paper_family_id'), ('doi', 'doi'), ('feature_row_id', 'feature_row_id'), ('material_id', 'material_id')):
        if row[field].astype(str).tolist() != admitted[paired].astype(str).tolist():
            raise ValueError('Same-state identity differs: ' + field)
    tensor = lambda values: torch.as_tensor(values, device='cuda', dtype=torch.float64)
    xy = tensor(row[['el', 'uts']].to_numpy())
    if not bool(torch.equal(xy, tensor(admitted[['elong_pct', 'uts_mpa']].to_numpy()))):
        raise ValueError('Original pair coordinates differ')
    encoded_sources = pa.array(row.paper_family_id.tolist()).dictionary_encode()
    labels = encoded_sources.dictionary.to_pylist()
    groups = torch.tensor(encoded_sources.indices.to_numpy(), device='cuda', dtype=torch.int64)
    years = tensor(row.paper_year.to_numpy())
    dated = torch.tensor(row.year_eligible.to_numpy(dtype=bool), device='cuda')
    valid = torch.isfinite(years) & (years >= 1950) & (years <= 2026) & (years == years.floor())
    if bool((dated & ~valid).any()) or len(labels) != receipt['n_paperfamilies'] or int(dated.sum()) != receipt['n_dated_pairs']:
        raise ValueError('Saved dated-pair population differs')
    actual = analyze_admitted_pairs(xy, groups, years, dated, len(labels))
    for key in ('on_front', 'old_front', 'recent_front'):
        if not bool(torch.equal(actual[key], torch.tensor(row[key].to_numpy(dtype=bool), device='cuda'))):
            raise ValueError('Saved frontier membership differs: ' + key)
    if not bool(torch.isclose(actual['pf_weight'], tensor(row.pf_weight.to_numpy()), atol=1e-12, rtol=1e-12).all()):
        raise ValueError('Source-balanced pair weights differ')
    bound_support = support.set_index('paper_family_id').loc[labels]
    if not bool(torch.equal(actual['counts'], tensor(bound_support.n_pairs.to_numpy()))):
        raise ValueError('Source support differs')
    if not bool((tensor(bound_support.total_weight.to_numpy()) == 1).all()):
        raise ValueError('Source total weight differs')
    comparisons = {}
    for (layer, fields) in (('history', ['year', 'n_pairs', 'n_paperfamilies', 'raw_hv_ratio', 'portfolio_p10', 'portfolio_median', 'portfolio_p90']), ('attainment', ['el_threshold', 'uts_threshold', 'fraction'])):
        expected = layers.loc[layers.layer.eq(layer), fields].reset_index(drop=True)
        result = actual[layer][fields].reset_index(drop=True)
        if len(expected) != len(result):
            raise ValueError('Saved analysis population differs: ' + layer)
        delta = (tensor(result.to_numpy()) - tensor(expected.to_numpy())).abs()
        if not bool(torch.isclose(tensor(result.to_numpy()), tensor(expected.to_numpy()), atol=1e-10, rtol=1e-12).all()):
            raise ValueError('Saved frontier analysis differs: ' + layer + ' max_delta=' + str(float(delta.max())))
        comparisons[layer] = {'rows': len(result), 'maximum_absolute_difference': float(delta.max())}
    facts = {'n_pairs': len(row), 'n_paperfamilies': len(labels), 'n_dated_pairs': int(dated.sum()), 'n_missing_or_conflicted_year_pairs': int((~dated).sum()), 'n_front': int(actual['on_front'].sum()), 'n_old_front': int(actual['old_front'].sum()), 'n_recent_front': int(actual['recent_front'].sum()), 'n_old_pairs': int(actual['old_mask'].sum()), 'n_recent_pairs': int(actual['recent_mask'].sum()), 'hv_normalizer_dated': actual['hv_normalizer_dated'], 'kish_state_weights': actual['kish_state_weights']}
    if not bool(torch.isclose(tensor(list(facts.values())), tensor([receipt[k] for k in facts]), atol=1e-10, rtol=1e-12).all()):
        raise ValueError('Saved observed-pair summary differs')
    frontier_ids = {key: row.loc[actual[key].cpu().numpy(), 'observation_id'].tolist() for key in ('on_front', 'old_front', 'recent_front')}
    excluded = states.loc[~states.admitted.eq(True), ['observation_id_uts', 'observation_id_el', 'reason']].to_dict('records')
    return {**facts, 'source_generation': receipt['source_sha256'], 'n_retained_excluded_pairs': len(excluded), 'excluded_pairs': excluded, 'frontier_observation_ids': frontier_ids, 'analysis_comparisons': comparisons, 'history': actual['history'].to_dict('records'), 'attainment_grid_shape': [81, 81], 'n_portfolios': N_PORTFOLIOS, 'seed': SEED, 'era_cut': ERA_CUT, 'hv_reference': [0.0, 0.0], 'source_encoding': 'Arrow dictionary, original first-seen admitted-row order', 'dated_mask': 'Saved pair-year agreement/eligibility; numeric year validity checked, raw paired-year admission not reconstructed', 'comparison_atol': 1e-10, 'comparison_rtol': 1e-12, 'model_fits': 0, 'model_inference': 0, 'scope': 'Strict observed Pareto, dated hypervolume/400 source portfolios and source attainment after existing admission; no source-text, original pairing or full-table admission replay', 'claim_ceiling': receipt['claim_ceiling'], 'portfolio_interval': '10–90% portfolio spread, not a confidence interval'}

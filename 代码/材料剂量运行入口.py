"""Dose-series regressions, source summaries and bootstrap intervals of the source-equal addition analysis (Figure S23)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch

CUDA_RUNS = {}

def cuda_tensor(values):
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError('GPU_REQUIRED: source-resolved dose statistics require CUDA')
    return torch.as_tensor(np.asarray(values, dtype=np.float64), device='cuda')

def cuda_record(name, start, stop):
    stop.record()
    torch.cuda.synchronize()
    stat = CUDA_RUNS.setdefault(name, dict(calls=0, cuda_elapsed_ms=0.0))
    stat['calls'] += 1
    stat['cuda_elapsed_ms'] += start.elapsed_time(stop)

def cuda_median(values, dim=-1):
    ordered = torch.sort(values, dim=dim).values
    n = ordered.shape[dim]
    return (ordered.select(dim, (n - 1) // 2) + ordered.select(dim, n // 2)) / 2

def cuda_reduce(values, operation):
    """Scalar reductions for this protocol; pandas-compatible missing-value rules."""
    if operation == 'nunique':
        values = pd.factorize(pd.Series(values).dropna(), sort=False)[0]
    (start, stop) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    start.record()
    x = cuda_tensor(values)
    if operation in {'positive_sum', 'negative_sum', 'positive_mean'}:
        x = (x < 0 if operation == 'negative_sum' else x > 0).to(torch.float64)
        operation = 'mean' if operation == 'positive_mean' else 'sum'
    x = x[~torch.isnan(x)]
    if operation == 'nunique':
        result = torch.unique(x).numel()
    elif operation == 'sum':
        result = x.sum().item()
    elif not x.numel():
        result = float('nan')
    elif operation == 'mean':
        result = x.mean().item()
    elif operation == 'median':
        result = cuda_median(x).item()
    elif operation == 'min':
        result = x.min().item()
    elif operation == 'max':
        result = x.max().item()
    else:
        raise ValueError(operation)
    cuda_record(operation, start, stop)
    return result

def cuda_series_metrics(x_values, y_values):
    (start, stop) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    start.record()
    (x, y) = (cuda_tensor(x_values), cuda_tensor(y_values))
    (xc, yc) = (x - x.mean(), y - y.mean())
    slope = torch.dot(xc, yc) / torch.dot(xc, xc)
    sst = yc.square().sum()
    (span, delta) = (x[-1] - x[0], y[-1] - y[0])
    result = dict(dose_span=span.item(), dose_span_to_one_pct_unit=span.item(), one_unit_coefficient_exceeds_observed_dose_span=(span < 1).item(), dose_span_below_0_01_unit=(span < 0.01).item(), minimum_adjacent_dose_difference=torch.diff(x).min().item(), dose_span_fraction_of_midpoint=torch.where(x[0] + x[-1] > 0, span / ((x[0] + x[-1]) / 2), torch.nan).item(), delta_value=delta.item(), relative_change_pct=torch.where(y[0] > 0, 100 * delta / y[0], torch.nan).item(), slope_per_dose_unit=slope.item(), r2=torch.where(sst > 0, 1 - (y - (y.mean() + slope * xc)).square().sum() / sst, torch.nan).item(), all_monotonic_increase=(torch.diff(y) >= 0).all().item(), all_monotonic_decrease=(torch.diff(y) <= 0).all().item())
    cuda_record('dose_OLS_and_changes', start, stop)
    return result

def clean(v):
    if v is None or (isinstance(v, (float, np.floating)) and (not np.isfinite(v))):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): clean(x) for (k, x) in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    return v

def js(v):
    return json.dumps(clean(v), ensure_ascii=False, sort_keys=True, separators=(',', ':'))

SUMMARY_KEYS = ['factor', 'basis', 'dose_role', 'dose_evidence', 'tier', 'endpoint', 'outcome_unit', 'mode', 'temp_band', 'scope']

def summarize(series, bootstrap, seed, *, backend='cuda', group_keys=None, bootstrap_plans=None):
    if backend not in {'cuda', 'cpu'}:
        raise ValueError('Unsupported summary backend: ' + backend)
    reduce = cuda_reduce if backend == 'cuda' else cpu_reduce
    keys = SUMMARY_KEYS if group_keys is None else list(group_keys)
    if series.empty:
        return (pd.DataFrame(), pd.DataFrame())
    source_rows = []
    for (vals, g) in series.groupby(keys + ['source'], dropna=False, sort=True):
        r = dict(zip(keys + ['source'], vals))
        r.update(n_series=len(g), n_rows=int(reduce(g.n_rows, 'sum')), source_mean_slope=reduce(g.slope_per_dose_unit, 'mean'), source_mean_delta=reduce(g.delta_value, 'mean'), source_mean_relative_change_pct=reduce(g.relative_change_pct, 'mean'), source_mean_dose_span=reduce(g.dose_span, 'mean'), source_min_dose_span=reduce(g.dose_span, 'min'), source_max_dose_span=reduce(g.dose_span, 'max'), n_series_span_less_than_one_unit=int(reduce(g.one_unit_coefficient_exceeds_observed_dose_span, 'sum')), n_series_span_below_0_01_unit=int(reduce(g.dose_span_below_0_01_unit, 'sum')), records_with_source_sha=int(reduce(g.records_with_source_sha, 'sum')), records_with_native_locator_and_sha=int(reduce(g.records_with_native_locator_and_sha, 'sum')), mean_non_target_elements_known=reduce(g.non_target_elements_known_min, 'mean'), source_mean_baseline=reduce(g.baseline_value, 'mean'), source_positive_fraction=reduce(g.slope_per_dose_unit, 'positive_mean'), series_ids=js(g.series_id.tolist()), doi=js(sorted(set((x for v in g.doi for x in json.loads(v))))))
        source_rows.append(r)
    sources = pd.DataFrame(source_rows)
    rng = np.random.default_rng(seed)
    summaries = []
    for (vals, g) in sources.groupby(keys, dropna=False, sort=True):
        r = dict(zip(keys, vals))
        n = len(g)
        if bootstrap_plans is not None:
            plan = bootstrap_plans[tuple((str(value) for value in vals))]
            if n != plan['n_sources']:
                raise ValueError('Recorded bootstrap source count differs')
            if n >= 3:
                rng.bit_generator.state = plan['state_before']
        draws = rng.integers(0, n, size=(bootstrap, n)) if n >= 3 else None
        r.update(n_sources=n, n_series=int(reduce(g.n_series, 'sum')), n_rows=int(reduce(g.n_rows, 'sum')), mean_source_slope=reduce(g.source_mean_slope, 'mean'), median_source_slope=reduce(g.source_mean_slope, 'median'), mean_source_delta=reduce(g.source_mean_delta, 'mean'), mean_source_relative_change_pct=reduce(g.source_mean_relative_change_pct, 'mean'), median_source_delta=reduce(g.source_mean_delta, 'median'), median_source_relative_change_pct=reduce(g.source_mean_relative_change_pct, 'median'), mean_dose_span=reduce(g.source_mean_dose_span, 'mean'), median_source_dose_span=reduce(g.source_mean_dose_span, 'median'), min_dose_span=reduce(g.source_min_dose_span, 'min'), max_dose_span=reduce(g.source_max_dose_span, 'max'), n_series_span_less_than_one_unit=int(reduce(g.n_series_span_less_than_one_unit, 'sum')), n_series_span_below_0_01_unit=int(reduce(g.n_series_span_below_0_01_unit, 'sum')), n_sources_with_native_locator_and_sha=int(reduce(g.records_with_native_locator_and_sha, 'positive_sum')), median_non_target_elements_known=reduce(g.mean_non_target_elements_known, 'median'), n_positive_sources=int(reduce(g.source_mean_slope, 'positive_sum')), n_negative_sources=int(reduce(g.source_mean_slope, 'negative_sum')), bootstrap_draws=bootstrap if draws is not None else 0, interval_status='pointwise_source_bootstrap_exploratory' if draws is not None else 'fewer_than_3_sources_no_interval', source_ids=js(g.source.tolist()), estimand='equal-source mean of condition-series slopes; observed condition mixture, not universal causal gain')
        for (metric, outname) in (('source_mean_slope', 'slope'), ('source_mean_delta', 'delta'), ('source_mean_relative_change_pct', 'relative_change_pct')):
            if backend == 'cpu':
                v = g[metric].to_numpy(dtype=np.float64)
                sampled = v[draws] if draws is not None and np.isfinite(v).all() else None
                for (prefix, statistic) in (('', np.mean), ('median_', np.median)):
                    quantiles = np.quantile(statistic(sampled, axis=1), [0.025, 0.975], method='linear') if sampled is not None else [np.nan, np.nan]
                    (r[prefix + outname + '_ci_low'], r[prefix + outname + '_ci_high']) = quantiles
                continue
            (start, stop) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            start.record()
            v = cuda_tensor(g[metric])
            finite = torch.isfinite(v).all().item()
            if draws is None or not finite:
                r[outname + '_ci_low'] = r[outname + '_ci_high'] = np.nan
            else:
                sampled = v[torch.as_tensor(draws, dtype=torch.int64, device='cuda')]
                q = torch.quantile(sampled.mean(1), cuda_tensor([0.025, 0.975]), interpolation='linear')
                (r[outname + '_ci_low'], r[outname + '_ci_high']) = q.cpu().tolist()
            if draws is None or not finite:
                r['median_' + outname + '_ci_low'] = r['median_' + outname + '_ci_high'] = np.nan
            else:
                q = torch.quantile(cuda_median(sampled, dim=1), cuda_tensor([0.025, 0.975]), interpolation='linear')
                (r['median_' + outname + '_ci_low'], r['median_' + outname + '_ci_high']) = q.cpu().tolist()
            cuda_record('source_bootstrap_and_quantiles', start, stop)
        summaries.append(r)
    return (sources, pd.DataFrame(summaries))

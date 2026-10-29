"""Metrics and error distributions of the five saved property models on the qualified queries (Figure 9)."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch

TARGET_ORDER = ['uts_mpa', 'ys_mpa', 'elong_pct', 'modulus_gpa', 'hardness_hv']

COMPUTE_RECEIPT: dict[str, Any] = {}

QUALIFICATION_RECEIPT: dict[str, Any] = {}

def metric_values(frame: pd.DataFrame, *, allow_degenerate: bool=False) -> dict[str, float | None]:
    if not torch.cuda.is_available():
        raise RuntimeError('GPU_REQUIRED: CUDA evidence reduction is unavailable')
    if frame.empty:
        raise ValueError('Paired, nonempty observations required')
    values = torch.as_tensor(frame[['y_true', 'y_pred']].to_numpy(dtype=float), device='cuda', dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError('Non-finite prediction evidence')
    (observed, predicted) = (values[:, 0], values[:, 1])
    residual = predicted - observed
    denominator = ((observed - observed.mean()) ** 2).sum()
    iqr = torch.quantile(observed, 0.75) - torch.quantile(observed, 0.25)
    if not allow_degenerate and (not bool((denominator > 0) & (iqr > 0))):
        raise ValueError('Target variance and IQR must be positive')
    result = torch.stack((torch.where(denominator > 0, 1.0 - (residual ** 2).sum() / denominator, torch.nan), torch.sqrt((residual ** 2).mean()), residual.abs().mean(), torch.quantile(residual.abs(), 0.95), residual.mean(), iqr)).cpu().tolist()
    return dict(zip(('r2', 'rmse', 'mae', 'p95ae', 'bias', 'observed_iqr'), (None if value != value else value for value in result)))

def prediction_errors(frame: pd.DataFrame, observed_iqr: float) -> tuple[pd.DataFrame, float, list[int]]:
    """Saved-point errors, ECDF and equal-source MAE on the same CUDA owner."""
    if not torch.cuda.is_available():
        raise RuntimeError('GPU_REQUIRED: CUDA evidence reduction is unavailable')
    values = torch.as_tensor(frame[['y_true', 'y_pred']].to_numpy(dtype=float), device='cuda', dtype=torch.float64)
    if not len(values) or not bool(torch.isfinite(values).all()):
        raise ValueError('Nonempty finite prediction evidence is required')
    residual = values[:, 1] - values[:, 0]
    absolute = residual.abs()
    normalized = absolute / observed_iqr if observed_iqr > 0 else torch.full_like(absolute, torch.nan)
    order = torch.argsort(absolute, stable=True)
    ecdf = torch.empty_like(absolute)
    ecdf[order] = torch.arange(1, len(values) + 1, device='cuda', dtype=torch.float64) / len(values)
    (codes, sources) = pd.factorize(frame['source'], sort=True)
    if not len(sources) or (codes < 0).any():
        raise ValueError('Every prediction needs a source')
    groups = torch.as_tensor(codes, device='cuda', dtype=torch.int64)
    source_errors = torch.zeros(len(sources), device='cuda', dtype=torch.float64)
    source_errors.scatter_add_(0, groups, absolute)
    macro_mae = (source_errors / torch.bincount(groups, minlength=len(sources))).mean().item()
    errors = pd.DataFrame(torch.stack((residual, absolute, normalized, ecdf), dim=1).cpu().numpy(), columns=['residual', 'abs_error', 'normalized_abs_error', 'ecdf_fraction'], index=frame.index)
    return (errors, macro_mae, order.cpu().tolist())

def load_and_validate(registry_path: Path | None=None, points_path: Path | None=None) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        raise RuntimeError('GPU_REQUIRED: CUDA evidence reduction is unavailable')
    registry = pd.read_csv(REGISTRY if registry_path is None else registry_path, float_precision='round_trip')
    points = pd.read_csv(POINTS if points_path is None else points_path, float_precision='round_trip')
    if set(registry['target']) != set(TARGET_ORDER) or len(registry) != 5:
        raise ValueError('Expected the existing five-target authority, exactly once per target')
    if not set(points['target']).issubset(TARGET_ORDER):
        raise ValueError('Unexpected target in the active prediction table')
    if points.duplicated(['target', 'query_id']).any():
        raise ValueError('Duplicate unique-query identities')
    registry = registry.set_index('target').loc[TARGET_ORDER].reset_index()
    failures: list[str] = []
    checks: list[dict[str, Any]] = []
    (started, ended) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    started.record()
    for target in TARGET_ORDER:
        row = registry.loc[registry['target'] == target].iloc[0]
        mask = points['target'] == target
        target_points = points.loc[mask]
        measured = metric_values(target_points)
        if not all((np.isfinite(value) for value in measured.values())):
            raise ValueError(f'{target}: non-finite reduced prediction evidence')
        threshold = float(row['threshold'])
        gate_flag = str(row['same_boundary_gate_pass']).lower()
        if gate_flag not in {'true', 'false'}:
            failures.append(f'{target}: invalid recorded authority pass flag')
        passed = measured['r2'] >= threshold and gate_flag == 'true'
        if threshold != 0.9:
            failures.append(f'{target}: active authority threshold changed')
        for metric in ('r2', 'rmse', 'mae'):
            declared = float(row[metric])
            if not np.isfinite(declared) or not np.isfinite(measured[metric]):
                failures.append(f'{target}: non-finite {metric} evidence')
                continue
            delta = abs(measured[metric] - declared)
            if delta > 1e-10:
                failures.append(f'{target}: {metric} mismatch ({delta:.3e})')
        if len(target_points) != int(row['n_unique_queries']):
            failures.append(f'{target}: query count mismatch')
        if target_points['source'].nunique() != int(row['n_sources']):
            failures.append(f'{target}: source count mismatch')
        reported = target_points['boundary'].dropna().astype(str)
        conflicts = reported.str.strip().ne('') & reported.ne(str(row['boundary']))
        if conflicts.any():
            failures.append(f'{target}: prediction and registry boundaries conflict')
        points.loc[mask, 'boundary'] = row['boundary']
        (errors, _, _) = prediction_errors(target_points, measured['observed_iqr'])
        points.loc[mask, errors.columns] = errors.to_numpy()
        checks.append({'target': target, **measured, 'threshold': threshold, 'passed': passed, 'same_boundary_gate_pass': str(row['same_boundary_gate_pass']), 'display_status': 'CONSISTENT_SAVED_EVIDENCE', 'authority_promoted': False, 'n_unique_queries': int(row['n_unique_queries']), 'prediction_event_count': int(row['prediction_event_count']), 'n_sources': int(row['n_sources']), 'boundary': str(row['boundary']), 'validation_estimand': str(row['validation_estimand']), 'group_isolation_state': str(row['group_isolation_state'])})
    ended.record()
    torch.cuda.synchronize()
    COMPUTE_RECEIPT.update(backend='PyTorch CUDA float64', device=torch.cuda.get_device_name(), cuda_elapsed_ms=started.elapsed_time(ended), synchronized=True, operations=['metrics', 'quantiles', 'residual_normalization', 'ECDF_sort'], model_fit_calls=0, input_predictions=len(points), cpu_compute_fallback=False)
    if failures:
        raise RuntimeError('Figure evidence validation failed:\n' + '\n'.join(failures))
    historical_checks = [dict(row) for row in checks]
    iop = '10.1088/1757-899x/1270/1/012118'
    matchar = '10.1016/j.matchar.2026.116231'
    el_members = {'3845136729074884472': 'bcdee047-5b4f-5fd7-9a27-9acce7dac3b5', '187385565008268580': 'a8238b86-ba9f-56a7-b7e7-378b24f323c5', '6976702623115430915': '345ac06f-3147-50d2-8942-4e10548e2649', '7417583661406280452': '449e32eb33528f057894', '14643771671271144075': 'd3e14cb4d66f5b0c4fe6', '11138656207309091115': '54f80cb9b7e68b0ba722', '3135147448763357126': '68e83765f2fa7717c46a', '10616373396070805879': '1324a6179f8b4d706481', '1134640324849297659': '208f465bfb348e73e5a6'}
    specifications = [('uts_mpa', iop, '13', 308.0, 'd3e14cb4d66f5b0c4fe6', 'SECONDARY_SOURCE_UNATTRIBUTED'), ('uts_mpa', matchar, '65', 1053.0, '66b96162-705e-589d-9bbe-5f16cb460685', 'WRONG_MATERIAL_IDENTITY')] + [('elong_pct', iop, iop + '::' + key, 1.8, uuid, 'SECONDARY_SOURCE_UNATTRIBUTED') for (key, uuid) in el_members.items()]
    (excluded, records) = (pd.Series(False, index=points.index), [])
    for (target, source, query, value, uuid, reason) in specifications:
        selected = points.target.eq(target) & points.source.eq(source) & points.query_id.astype(str).eq(query)
        if int(selected.sum()) != 1 or float(points.loc[selected, 'y_true'].iloc[0]) != value:
            raise ValueError('Reviewed query identity/value differs: ' + target + ':' + query)
        excluded |= selected
        records.append(dict(target=target, source=source, query_id=query, record_uuid=uuid, original_y_true=value, original_y_pred=float(points.loc[selected, 'y_pred'].iloc[0]), reason=reason))
    if len(points) != 4819 or int(excluded.sum()) != 11:
        raise ValueError('Expected the complete 4819-query authority and exactly 11 source-role exclusions')
    points = points.loc[~excluded].copy()
    checks = []
    for target in TARGET_ORDER:
        mask = points.target.eq(target)
        frame = points.loc[mask]
        measured = metric_values(frame)
        (errors, _, _) = prediction_errors(frame, measured['observed_iqr'])
        points.loc[mask, errors.columns] = errors.to_numpy()
        ix = registry.target.eq(target)
        for key in ('r2', 'rmse', 'mae'):
            registry.loc[ix, key] = measured[key]
        registry.loc[ix, 'n_unique_queries'] = len(frame)
        registry.loc[ix, 'n_sources'] = frame.source.nunique()
        registry.loc[ix, 'source_fold_overlap_count'] = int(frame.groupby('source').fold.nunique().gt(1).sum())
        if target != 'hardness_hv':
            registry.loc[ix, 'prediction_event_count'] = len(frame)
        check = dict(next((row for row in historical_checks if row['target'] == target)))
        check.update(measured, n_unique_queries=len(frame), n_sources=int(frame.source.nunique()), source_fold_overlap_count=int(registry.loc[ix, 'source_fold_overlap_count'].iloc[0]), prediction_event_count=int(registry.loc[ix, 'prediction_event_count'].iloc[0]), display_status='SOURCE_ROLE_QUALIFIED_SAVED_QUERY_SUBSET', authority_promoted=False)
        checks.append(check)
    QUALIFICATION_RECEIPT.update(scope='Finite source-role qualification of saved evaluation queries; not a new model authority', original_query_count=4819, publication_query_count=len(points), excluded_queries=records, historical_authority_checks=historical_checks, original_authority_unchanged=True, labels_predictions_models_unchanged=True, retained_alias_limit={'source': matchar, 'query_ids': ['66', '67'], 'record_uuids': ['2b9453c2-1716-581f-a555-c9cfd40902e3', 'b67fa9f7-d537-5c28-8894-fe2794110ba1'], 'interpretation': 'Two retained queries; independence as two material states is unresolved'})
    COMPUTE_RECEIPT.update(historical_input_predictions=4819, publication_predictions=len(points))
    return (registry, points, checks)

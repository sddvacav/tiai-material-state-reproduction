from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

def recount_target_plane(root):
    analysis = root / 'paper01' / 'analysis'
    source = root / 'paper01' / 'source_data'
    stem = 'TITMC_PAPER01_selection_utility'
    inputs = {'panel_spec': root / 'data/selection_static_spec.json', 'sequential_spec': root / 'data/selection_sequential_spec.json', 'source_inventory': analysis / 'predictive_utility' / 'source_inventory.csv', 'budget_utility': analysis / 'predictive_utility' / 'budget_utility_summary.csv', 'representation_resolution': analysis / 'source_aware_ranking' / 'source_representation_resolution.csv', 'representation_oracle': analysis / 'source_aware_ranking' / 'representation_oracle_by_source.csv', 'representation_blocks': analysis / 'source_aware_ranking' / 'representation_blocks_label_informed.csv', 'nested_top1': analysis / 'source_aware_ranking' / 'top1_utility_summary.csv', 'exact_tie': analysis / 'source_aware_ranking' / 'exact_tie_utility_summary.csv', 'sequential_gains': analysis / 'sequential_design' / 'paired_policy_gains.csv', 'sequential_outcomes': analysis / 'sequential_design' / 'budget_outcome_summary.csv', 'changed_choices': analysis / 'sequential_design' / 'all_changed_second_choices.csv', 'decisions': source / 'TITMC_PAPER01_source_heldout_decisions.csv'}
    (spec, sequential) = (json.loads(inputs[key].read_text(encoding='utf-8')) for key in ('panel_spec', 'sequential_spec'))
    table = {key: pd.read_csv(path, float_precision='round_trip', low_memory=False) for (key, path) in inputs.items() if path.suffix == '.csv'}
    (shared, formulas) = (spec['shared_content'], spec['scientific_formula_semantics'])
    composition = shared['observed_source_composition_diagnostic']['counts']
    resolved = shared['representation_resolution_diagnostic']
    bindings = spec['four_layer_candidate_source_bindings']
    (methods, budgets) = (tuple(spec['filter']['methods']), [int(k) for k in spec['filter']['requested_k']])
    (attain_metric, regret_metric) = (panel['metric'] for panel in spec['panels'])
    labels = shared['method_labels']

    def declared(text, pattern):
        found = re.search(pattern, text)
        if found is None:
            raise ValueError('Declared specification value is missing: ' + pattern)
        return float(found.group(1))
    inventory = table['source_inventory']
    primary = inventory.loc[inventory.primary_cohort.astype(bool)].set_index('paper_family_id').sort_index()
    resolution = table['representation_resolution'].set_index('paper_family_id')
    oracle = table['representation_oracle'].set_index('paper_family_id').sort_index()
    blocks = table['representation_blocks']
    if not primary.index.is_unique or not primary.index.isin(resolution.index).all() or (not oracle.index.equals(primary.index)) or (not blocks.paper_family_id.isin(primary.index).all()):
        raise ValueError('Primary source identities differ between the saved selection tables')
    resolution = resolution.loc[primary.index]
    (counts, hits) = (primary.n_candidates.astype(int), primary.observed_hit_count.astype(int))
    if not (resolution.n_candidates.astype(int).equals(counts) and oracle.n_candidates.astype(int).equals(counts)):
        raise ValueError('Candidate counts differ between the saved selection tables')
    opportunity = pd.Series(np.where(hits.eq(0), 'no_observed_hit', np.where(hits.eq(counts), 'all_candidates_hit', 'mixed_hit_status')), index=primary.index)
    (identical, mixed) = (resolution.all_raw_X_identical.astype(bool), opportunity.eq('mixed_hit_status'))
    checks = {'primary sources': (len(primary), composition['primary']), 'no observed hit': (int(opportunity.eq('no_observed_hit').sum()), composition['no_observed_hit']), 'all candidates hit': (int(opportunity.eq('all_candidates_hit').sum()), composition['all_candidates_hit']), 'mixed hit status': (int(mixed.sum()), composition['mixed_hit_status']), 'any observed hit': (int(hits.gt(0).sum()), composition['any_observed_hit']), 'candidates': (int(counts.sum()), resolved['candidate_count']), 'raw X blocks': (int(resolution.distinct_raw_X_representations.sum()), resolved['source_local_X_blocks']), 'encoded X blocks': (int(resolution.distinct_outer_encoded_X_representations.sum()), resolved['source_local_X_blocks']), 'label-informed X blocks': (len(blocks), resolved['source_local_X_blocks']), 'oracle X blocks': (int(oracle.distinct_X_blocks.sum()), resolved['source_local_X_blocks']), 'all raw X identical': (int(identical.sum()), resolved['all_X_identical_sources']), 'all encoded X identical': (int(resolution.all_outer_encoded_X_identical.astype(bool).sum()), resolved['all_X_identical_sources']), 'mixed and identical': (int((mixed & identical).sum()), resolved['mixed_hit_status_and_all_X_identical_sources']), 'mixed and distinguishable': (int((mixed & ~identical).sum()), resolved['mixed_hit_status_but_distinguishable_X_sources'])}
    failed = [name for (name, (seen, expected)) in checks.items() if seen != int(expected)]
    if failed:
        raise ValueError('Saved selection inputs differ from MAIN_FIGURE_PANEL_SPEC: ' + ', '.join(failed))
    budget = table['budget_utility']
    curve = budget.loc[budget.cohort.eq(spec['filter']['cohort']) & budget.stratum.eq(spec['filter']['stratum'])]

    def budget_row(method, k, metric):
        row = curve.loc[curve.method.eq(method) & curve.requested_k.eq(k) & curve.metric.eq(metric)]
        if len(row) != 1:
            raise ValueError(f'Missing saved budget row: {method} k={k} {metric}')
        return row.iloc[0]
    positions = {int(k): int(v) for (k, v) in shared['actual_selection_positions_by_k'].items()}
    remaining = {int(k): int(v) for (k, v) in shared['sources_with_choice_remaining_by_k'].items()}
    for k in budgets:
        if int(np.minimum(counts, k).sum()) != positions[k] or int(counts.gt(k).sum()) != remaining[k]:
            raise ValueError(f'Actual source budgets differ at k={k}')
        for metric in (attain_metric, regret_metric):
            rows = [budget_row(method, k, metric) for method in methods]
            if any((int(r.n_sources) != len(primary) or int(r.actual_selection_positions) != positions[k] or len(primary) - int(r.n_sources_budget_exhausts_candidate_set) != remaining[k] or (int(r.n_sources_no_observed_hit_diagnostic_only) != int(composition['no_observed_hit'])) for r in rows)):
                raise ValueError(f'Saved budget denominators differ at k={k}: {metric}')
            if max((abs(float(r.uniform_exact_expectation) - float(rows[0].uniform_exact_expectation)) for r in rows)) > 1e-12:
                raise ValueError(f'Exact uniform reference differs between methods at k={k}: {metric}')
    uniform = {metric: np.array([float(budget_row(methods[0], k, metric).uniform_exact_expectation) for k in budgets]) for metric in (attain_metric, regret_metric)}
    n = len(primary)
    ceiling = int(composition['any_observed_hit']) / n
    (uniform_top1, uniform_regret1) = (float(uniform[attain_metric][0]), float(uniform[regret_metric][0]))
    block_reference = float(oracle.label_informed_block_oracle_under_random_tie.mean())
    if abs(ceiling - float(shared['observed_source_opportunity_ceiling'])) > 1e-12 or abs(float(oracle.full_recorded_candidate_opportunity.mean()) - ceiling) > 1e-12 or abs(uniform_top1 - declared(formulas['top1_uniform_decomposition'], '=\\s*([0-9.]+)\\s*$')) > 1e-12 or (abs(float(oracle.uniform_observed_hit_fraction.mean()) - uniform_top1) > 1e-12) or (abs(block_reference - declared(formulas['label_informed_block_oracle'], 'source mean is ([0-9]+\\.[0-9]+)')) > 1e-12):
        raise ValueError('Saved opportunity, exact uniform or X-block reference differs from the specification')
    top1 = {method: float(budget_row(method, 1, attain_metric).observed) for method in methods}
    top1_hits = {method: int(budget_row(method, 1, attain_metric).selected_sources_with_any_hit) for method in methods}
    if any((abs(top1[method] * n - top1_hits[method]) > 1e-09 for method in methods)):
        raise ValueError('Saved Top-1 attainment counts differ from their fractions')
    nested_table = table['nested_top1']

    def nested_row(comparison, metric):
        row = nested_table.loc[nested_table.comparison.eq(comparison) & nested_table.metric.eq(metric)]
        if len(row) != 1:
            raise ValueError(f'Missing nested Top-1 row: {comparison} {metric}')
        return row.iloc[0]
    nested_style = {'source_equal_pointwise_ridge': ('Pointwise ridge', 's'), 'source_centered_ridge': ('Centered ridge', 'v')}
    nested_value = {name: float(nested_row(name, 'attainment').estimate) for name in nested_style}
    nested_hits = {name: int(round(value * n)) for (name, value) in nested_value.items()}
    for name in nested_style:
        if int(nested_row(name, 'attainment').n_sources) != n or abs(nested_value[name] * n - nested_hits[name]) > 1e-09 or abs(float(nested_row(name, 'attainment_gain').estimate) - (nested_value[name] - uniform_top1)) > 1e-12 or (abs(float(nested_row(name, 'regret_gain').estimate) - (uniform_regret1 - float(nested_row(name, 'observed_best_regret').estimate))) > 1e-09):
            raise ValueError('Nested Top-1 results do not share the exact uniform comparator: ' + name)
    seq_cohort = sequential['shared_filter']['cohort']
    seq_budgets = [int(k) for k in sequential['shared_filter']['requested_k']]
    comparators = list(sequential['shared_filter']['comparator'])
    (seq_attain, seq_regret) = (panel['metric'] for panel in sequential['panels'])
    matched = bindings['required_matched_comparator'].split(';')[0].strip()
    if matched not in comparators or len(comparators) != 2:
        raise ValueError('The matched paid-label comparator is not a declared sequential comparator')
    fixed_name = next((name for name in comparators if name != matched))
    gains = table['sequential_gains'].loc[lambda frame: frame.cohort.eq(seq_cohort)]

    def gain_row(comparator, k, metric):
        row = gains.loc[gains.comparator.eq(comparator) & gains.requested_k.eq(k) & gains.metric.eq(metric)]
        if len(row) != 1:
            raise ValueError(f'Missing sequential gain row: {comparator} k={k} {metric}')
        return row.iloc[0]
    for k in seq_budgets:
        for comparator in comparators:
            for metric in (seq_attain, seq_regret):
                row = gain_row(comparator, k, metric)
                if int(row.n_sources) != n or int(row.actual_selection_positions) != positions[k]:
                    raise ValueError(f'Sequential denominators differ at k={k}')
    k2 = gain_row(matched, 2, seq_attain)
    declared_ci = bindings['budget2_gain_ci95_percentage_points']
    if abs(float(k2.paired_gain) * 100 - float(bindings['budget2_attainment_gain_vs_matched_uniform_percentage_points'])) > 1e-09 or abs(float(k2.paired_gain_ci95_low) * 100 - float(declared_ci[0])) > 1e-09 or abs(float(k2.paired_gain_ci95_high) * 100 - float(declared_ci[1])) > 1e-09:
        raise ValueError('Budget-2 matched comparison differs from the declared result')
    outcomes = table['sequential_outcomes']
    outcomes = outcomes.loc[outcomes.cohort.eq(seq_cohort) & outcomes.requested_k.eq(2) & outcomes.metric.eq(seq_attain)].set_index('strategy')
    context = sequential['panels'][0]['required_k2_context']
    if abs(float(outcomes.loc['kernel_residual_greedy', 'sum_source_hit_indicators_or_exact_expectations']) - declared(context, 'Adaptive (\\d+)/')) > 1e-09 or abs(float(outcomes.loc[fixed_name, 'sum_source_hit_indicators_or_exact_expectations']) - declared(context, 'fixed ridge (\\d+)/')) > 1e-09 or abs(float(outcomes.loc[matched, 'estimate']) * 100 - declared(context, 'exact expectation ([0-9.]+)%')) > 0.0005:
        raise ValueError('Budget-2 sequential outcomes differ from the declared context')
    changed = table['changed_choices']
    best_gain = changed.best_q_gain_adaptive_minus_fixed.astype(float)
    (regret_fixed, attain_fixed) = (gain_row(fixed_name, 2, seq_regret), gain_row(fixed_name, 2, seq_attain))
    if len(changed) != int(declared(sequential['diagnostic_annotation'], 'only (\\d+)/\\d+ second choices change')) or not changed.paper_family_id.isin(primary.index).all() or int(best_gain.gt(1e-12).sum()) != int(regret_fixed.sources_positive_gain) or (int(best_gain.lt(-1e-12).sum()) != int(regret_fixed.sources_negative_gain)) or (int(changed.attainment_gain_adaptive_minus_fixed.gt(0).sum()) != int(attain_fixed.sources_positive_gain)) or (int(changed.attainment_gain_adaptive_minus_fixed.lt(0).sum()) != int(attain_fixed.sources_negative_gain)):
        raise ValueError('The changed second choices differ from the paired fixed-ridge diagnostics')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU_REQUIRED: Figure 10 target-plane recount')
    decisions = table['decisions']
    if decisions.admitted.dtype != bool:
        raise ValueError('Saved decision admission flags are not boolean')
    observed_rows = decisions.loc[decisions.method.eq('ridge') & decisions.admitted].set_index('observation_id_uts')
    if not observed_rows.index.is_unique:
        raise ValueError('Saved candidate outcomes repeat an observation identity')
    members = blocks.assign(observation_id_uts=blocks.observation_ids_uts.astype(str).str.split(';')).explode('observation_id_uts')
    member_ids = members.observation_id_uts.str.strip()
    if not member_ids.is_unique or not member_ids.isin(observed_rows.index).all():
        raise ValueError('Input-block members do not map one-to-one onto saved candidate outcomes')
    member_rows = observed_rows.loc[member_ids.to_numpy()]
    if not np.array_equal(member_rows.paper_family_id.astype(str).to_numpy(), members.paper_family_id.astype(str).to_numpy()):
        raise ValueError('Input-block members and saved candidate outcomes name different sources')
    source_position = pd.Series(np.arange(n), index=primary.index)
    (member_source, member_block) = (source_position.loc[members.paper_family_id].to_numpy(), members.index.to_numpy())
    if not np.array_equal(np.bincount(member_source, minlength=n), counts.to_numpy()) or not np.array_equal(np.bincount(member_block, minlength=len(blocks)), blocks.block_size.astype(int).to_numpy()):
        raise ValueError('Input-block membership differs from the saved candidate counts')
    (device, f64) = (torch.device('cuda'), torch.float64)
    (cand_uts, cand_el) = (torch.as_tensor(member_rows[column].to_numpy(float), device=device, dtype=f64) for column in ('uts_mpa', 'elong_pct'))
    if not (torch.isfinite(cand_uts).all() and torch.isfinite(cand_el).all()):
        raise ValueError('Saved candidate outcomes lack UTS or EL')
    (cand_source, cand_block) = (torch.as_tensor(item, device=device) for item in (member_source, member_block))
    block_source = torch.as_tensor(source_position.loc[blocks.paper_family_id].to_numpy(), device=device)
    source_size = torch.as_tensor(counts.to_numpy(float), device=device, dtype=f64)
    block_size = torch.as_tensor(blocks.block_size.to_numpy(float), device=device, dtype=f64)
    plane_uts = 400.0 + 5.0 * torch.arange(201, device=device, dtype=f64)
    plane_el = torch.arange(201, device=device, dtype=f64) / 10.0
    (grid_uts, grid_el) = (item.reshape(-1) for item in torch.meshgrid(plane_uts, plane_el, indexing='ij'))
    hit = ((cand_uts[:, None] >= grid_uts[None, :]) & (cand_el[:, None] >= grid_el[None, :])).to(f64)
    per_source = torch.zeros(n, hit.shape[1], device=device, dtype=f64).index_add_(0, cand_source, hit)
    per_block = torch.zeros(len(blocks), hit.shape[1], device=device, dtype=f64).index_add_(0, cand_block, hit) / block_size[:, None]
    best_block = torch.full_like(per_source, -1.0).scatter_reduce_(0, block_source[:, None].expand_as(per_block), per_block, reduce='amax')
    plane = {'O': (per_source > 0).to(f64).mean(0), 'R': best_block.mean(0), 'U': (per_source / source_size[:, None]).mean(0), 'none': (per_source == 0).to(f64).mean(0), 'all': (per_source == source_size[:, None]).to(f64).mean(0)}
    plane['mixed'] = 1.0 - plane['none'] - plane['all']
    anchor = 120 * len(plane_el) + 100
    recount = {key: float(value[anchor]) for (key, value) in plane.items()}
    saved_fraction = torch.as_tensor(blocks.observed_hit_fraction_label_informed.to_numpy(float), device=device, dtype=f64)
    if float(grid_uts[anchor]) != 1000.0 or float(grid_el[anchor]) != 10.0 or float(best_block.min()) < 0.0 or (not torch.equal(per_source[:, anchor], torch.as_tensor(hits.to_numpy(float), device=device, dtype=f64))) or (float((per_block[:, anchor] - saved_fraction).abs().max()) > 1e-12) or (abs(recount['O'] - ceiling) > 1e-12) or (abs(recount['R'] - block_reference) > 1e-12) or (abs(recount['U'] - uniform_top1) > 1e-12) or (abs(recount['mixed'] * n - int(mixed.sum())) > 1e-09):
        raise ValueError('The target-plane recount does not reproduce the saved references at 1,000 MPa and 10%')
    (plane_uts_np, plane_el_np) = (plane_uts.cpu().numpy(), plane_el.cpu().numpy())
    plane_np = {key: value.reshape(len(plane_uts_np), len(plane_el_np)).cpu().numpy() for (key, value) in plane.items()}
    path_k = np.arange(40, 126)
    path_cells = (2 * path_k - 80, path_k)
    (path_uts_np, path_el_np) = (plane_uts_np[path_cells[0]], plane_el_np[path_cells[1]])
    path_np = {key: value[path_cells] for (key, value) in plane_np.items()}
    path_anchor = int(np.flatnonzero(path_k == 100)[0])
    if not (np.array_equal(path_uts_np, 10.0 * path_k) and np.allclose(path_el_np, path_k / 10.0, rtol=0.0, atol=1e-12) and (path_uts_np[path_anchor] == 1000.0) and (path_el_np[path_anchor] == 10.0)):
        raise ValueError('The target path leaves the recounted plane or misses the declared target')
    mixed_plane = plane_np['mixed']
    peak = np.unravel_index(int(np.argmax(mixed_plane)), mixed_plane.shape)
    (gain_rank, gain_unresolved) = ((path_np['R'] - path_np['U']) * 100, (path_np['O'] - path_np['R']) * 100)
    gain_total = gain_rank + gain_unresolved
    plane_name = f'{stem}_target_plane.csv'
    plane_frame = pd.DataFrame({'uts_target_mpa': np.repeat(plane_uts_np, len(plane_el_np)), 'el_target_pct': np.tile(plane_el_np, len(plane_uts_np)), 'opportunity_pct': plane_np['O'].ravel() * 100, 'best_input_block_pct': plane_np['R'].ravel() * 100, 'uniform_pct': plane_np['U'].ravel() * 100, 'no_hit_sources': np.rint(plane_np['none'].ravel() * n).astype(int), 'mixed_sources': np.rint(mixed_plane.ravel() * n).astype(int), 'all_hit_sources': np.rint(plane_np['all'].ravel() * n).astype(int), 'mixed_sources_pct': mixed_plane.ravel() * 100})
    whole = {key: plane_np[a] - plane_np[b] for (key, (a, b)) in {'attainable': ('O', 'U'), 'ranking': ('R', 'U'), 'unresolved': ('O', 'R')}.items()}
    target_plane = {'grid': {'uts_mpa': [400.0, 1400.0, 5.0], 'el_pct': [0.0, 20.0, 0.1], 'cells': int(mixed_plane.size)}, 'path': {'uts_mpa': '1000 t', 'el_pct': '10 t', 't': [0.4, 1.25, 0.01], 'points': int(path_k.size)}, 'declared_target': {'opportunity_pct': recount['O'] * 100, 'best_input_block_pct': recount['R'] * 100, 'uniform_pct': recount['U'] * 100, 'ranking_room_pp': float(gain_rank[path_anchor]), 'unresolved_states_pp': float(gain_unresolved[path_anchor]), 'mixed_sources': int(round(recount['mixed'] * n))}, 'mixed_sources_max': int(round(float(mixed_plane.max()) * n)), 'mixed_share_max_pct': float(mixed_plane.max() * 100), 'mixed_share_max_at': {'uts_mpa': float(plane_uts_np[peak[0]]), 'el_pct': float(plane_el_np[peak[1]])}, 'pool_fixed_sources_min': n - int(round(float(mixed_plane.max()) * n)), 'path_max_pp': {name: {'value': float(values.max()), 'uts_mpa': float(path_uts_np[int(values.argmax())]), 'el_pct': float(path_el_np[int(values.argmax())])} for (name, values) in (('attainable_gain', gain_total), ('ranking_room', gain_rank), ('unresolved_states', gain_unresolved))}, 'plane_max_pp': {name: float(values.max() * 100) for (name, values) in whole.items()}, 'device': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'dtype': 'float64'}
    return (plane_frame, {'primary_sources': n, 'primary_candidates': int(counts.sum()), 'source_local_X_blocks': len(blocks), 'target_plane': target_plane, 'model_fits': 0, 'figures_rendered': False})

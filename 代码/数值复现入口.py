#!/usr/bin/env python3
"""Reproduce the numerical analyses of this release from the released inputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATOL, RTOL = 1e-10, 1e-9


def read_table(root, relative):
    return pd.read_csv(root/relative, float_precision="round_trip",
                       keep_default_na=False, na_values=[""])


def representation(root: Path, output: Path) -> dict:
    qualification = json.loads((root/"SOURCE_QUALIFICATION.json").read_text("utf-8"))
    config = json.loads((root/"ANALYSIS_CONFIG.json").read_text("utf-8"))
    decisions = read_table(root, "paper01/source_data/TITMC_PAPER01_source_heldout_decisions.csv")
    frozen = decisions.loc[decisions.admitted.eq(True) & decisions.method.eq("ridge")].copy()
    if len(frozen) != 1111 or frozen.paper_family_id.nunique() != 336:
        raise ValueError("Unexpected frozen prediction pool.")
    removed = {r["observation_id_uts"] for r in qualification["removed_pairs"]}
    retained = frozen.loc[~frozen.observation_id_uts.isin(removed)]
    counts = retained.groupby("paper_family_id").size()
    sources = set(counts.loc[counts.ge(2)].index)
    candidates = retained.loc[retained.paper_family_id.isin(sources)].sort_values("observation_id_uts")
    if len(retained) != 1088 or len(candidates) != 931 or len(sources) != 173:
        raise ValueError("The endpoint qualification does not produce the current comparison cohort.")
    features = read_table(root, "paper01/analysis/source_aware_ranking/candidate_inputs.csv")
    if not features.observation_id_uts.is_unique:
        raise ValueError("Duplicate feature identifiers.")
    features = features.set_index("observation_id_uts")
    numeric, categorical = config["numeric_x_columns"], config["categorical_x_columns"]
    if len(numeric) != 39 or len(categorical) != 5:
        raise ValueError("Unexpected feature interface.")
    matrices = {}
    for fold in range(5):
        with np.load(root/f"paper01/analysis/source_aware_ranking/encoded_inputs_fold{fold}.npz",
                     allow_pickle=False) as archive:
            matrices[fold] = (archive["X"].copy(),
                             {str(oid): i for i, oid in enumerate(archive["observation_id_uts"])})
    block_rows, source_rows = [], []
    no_hit = mixed = all_hit = identical = mixed_identical = 0
    for source, group in candidates.groupby("paper_family_id", sort=True):
        group = group.sort_values("observation_id_uts")
        ids = group.observation_id_uts.tolist()
        part = features.loc[ids]
        if not part.paper_family_id.eq(source).all():
            raise ValueError("A feature row belongs to a different source.")
        values = part[numeric].to_numpy(dtype=float)
        categories = part[categorical].fillna("UNKNOWN").astype(str).to_numpy()
        groups = defaultdict(list)
        for index, (row, category) in enumerate(zip(values, categories)):
            signature = tuple(None if not np.isfinite(v) else float(v).hex() for v in row) + tuple(category)
            groups[signature].append(index)
        q = (group[["uts_mpa", "elong_pct"]].to_numpy(float)/[1000.0, 10.0]).min(axis=1)
        hits = q >= 1
        fraction = float(hits.mean())
        no_hit += int(not hits.any())
        all_hit += int(hits.all())
        mixed += int(hits.any() and not hits.all())
        identical += int(len(groups) == 1)
        mixed_identical += int(len(groups) == 1 and hits.any() and not hits.all())
        block_fractions = []
        for block_index, indices in enumerate(groups.values()):
            block_fraction = float(hits[indices].mean())
            block_fractions.append(block_fraction)
            block_rows.append(dict(paper_family_id=source, block_index=block_index,
                n_candidates=len(group), block_size=len(indices),
                observed_hit_fraction_label_informed=block_fraction,
                observed_mean_q_label_informed=float(q[indices].mean()),
                observation_ids_uts=";".join(ids[i] for i in indices)))
        fold = int(group.fold.iloc[0])
        X, positions = matrices[fold]
        encoded = X[[positions[oid] for oid in ids]]
        encoded_groups = len(np.unique(encoded, axis=0))
        source_rows.append(dict(paper_family_id=source, n_candidates=len(group),
            distinct_X_blocks=len(groups), distinct_encoded_X_blocks=encoded_groups,
            all_X_identical=len(groups) == 1, uniform_observed_hit_fraction=fraction,
            label_informed_block_oracle_under_random_tie=max(block_fractions),
            full_recorded_candidate_opportunity=int(hits.any())))

    blocks, oracle = pd.DataFrame(block_rows), pd.DataFrame(source_rows)
    saved_blocks = read_table(root, "paper01/analysis/source_aware_ranking/representation_blocks_label_informed.csv")
    def keyed(frame):
        result = {}
        for row in frame.to_dict("records"):
            key = (row["paper_family_id"], tuple(sorted(row["observation_ids_uts"].split(";"))))
            if key in result:
                raise ValueError("Duplicate representation group.")
            result[key] = row
        return result
    actual_groups, reference_groups = keyed(blocks), keyed(saved_blocks)
    if actual_groups.keys() != reference_groups.keys():
        raise ValueError("The independently reconstructed input groups differ from the saved partition.")
    largest = 0.0
    for key, row in actual_groups.items():
        saved = reference_groups[key]
        for field in ("n_candidates", "block_size", "observed_hit_fraction_label_informed",
                      "observed_mean_q_label_informed"):
            np.testing.assert_allclose(row[field], saved[field], rtol=RTOL, atol=ATOL, err_msg=field)
            largest = max(largest, abs(float(row[field])-float(saved[field])))
    saved_oracle = read_table(root, "paper01/analysis/source_aware_ranking/representation_oracle_by_source.csv")
    actual = oracle.set_index("paper_family_id").sort_index()
    saved = saved_oracle.set_index("paper_family_id").sort_index()
    if not actual.index.equals(saved.index):
        raise ValueError("Representation source identities differ.")
    for field in ("n_candidates", "distinct_X_blocks", "all_X_identical",
                  "uniform_observed_hit_fraction", "label_informed_block_oracle_under_random_tie",
                  "full_recorded_candidate_opportunity"):
        av, bv = actual[field].to_numpy(float), saved[field].to_numpy(float)
        np.testing.assert_allclose(av, bv, rtol=RTOL, atol=ATOL, err_msg=field)
        largest = max(largest, float(np.max(np.abs(av-bv))))
    resolution = read_table(root, "paper01/analysis/source_aware_ranking/source_representation_resolution.csv")
    resolution = resolution.set_index("paper_family_id").loc[actual.index]
    np.testing.assert_array_equal(actual.distinct_encoded_X_blocks,
                                  resolution.distinct_outer_encoded_X_representations)
    reference_values = qualification["reference_values"]
    observed = {"primary_candidates": len(candidates), "primary_sources": len(sources),
                "input_equivalence_blocks": len(blocks), "no_hit_sources": no_hit,
                "mixed_hit_sources": mixed, "all_hit_sources": all_hit,
                "identical_input_sources": identical, "mixed_identical_input_sources": mixed_identical}
    for name, value in observed.items():
        if value != reference_values[name]:
            raise ValueError("Reference count differs: "+name)
    output.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(output/"representation_blocks.csv", index=False, float_format="%.17g")
    oracle.to_csv(output/"representation_by_source.csv", index=False, float_format="%.17g")
    return {"status": "passed", **observed,
            "encoded_input_groups": int(oracle.distinct_encoded_X_blocks.sum()),
            "maximum_absolute_difference": largest,
            "uniform_attainment": float(oracle.uniform_observed_hit_fraction.mean()),
            "label_informed_block_oracle": float(oracle.label_informed_block_oracle_under_random_tie.mean()),
            "recorded_candidate_opportunity": float(oracle.full_recorded_candidate_opportunity.mean()),
            "group_membership": "Reconstructed from the frozen feature values; matched by source and member identifiers."}


def material_relations(root: Path) -> dict:
    """Replay the original Figure 4 equal-source moments on admitted pairs."""
    import torch
    import CODE_TITMC_FIG08_strength_ductility as figure_code
    if not torch.cuda.is_available():
        raise RuntimeError('GPU_REQUIRED: material-relationship replay')
    pairs = read_table(root, 'data/TITMC_FIG08_strength_ductility_states.csv')
    reference = json.loads((root/'data/TITMC_PAPER01_strength_elongation_variance_decomposition.json').read_text('utf-8'))
    result = {}
    for panel, name in (('rt', 'room_temperature'), ('ht', '500_to_800C')):
        frame = pairs.loc[pairs.panel.eq(panel)]
        codes, sources = pd.factorize(frame.paper_family_id, sort=True)
        expected = reference['cohorts'][name]
        if len(frame) != expected['pairs'] or len(sources) != expected['sources']:
            raise ValueError('Figure 4 admitted cohort differs: ' + panel)
        moments = figure_code.association(
            torch.tensor(frame[['el', 'uts']].to_numpy(float), device='cuda', dtype=torch.float64),
            torch.tensor(codes, device='cuda', dtype=torch.int64), moments_only=True)
        fields = {'uts_between_source_variance_share': 'uts_between_source_variance_share',
                  'el_between_source_variance_share': 'el_between_source_variance_share',
                  'slope_within_source_MPa_per_pp': 'within_slope_mpa_per_pct',
                  'slope_between_source_MPa_per_pp': 'between_slope_mpa_per_pct',
                  'slope_equal_source_pooled_MPa_per_pp': 'balanced_slope_mpa_per_pct'}
        actual = torch.tensor([moments[v] for v in fields.values()], device='cuda', dtype=torch.float64)
        saved = torch.tensor([expected[k] for k in fields], device='cuda', dtype=torch.float64)
        if not bool(torch.isclose(actual, saved, rtol=1e-10, atol=1e-9).all()):
            raise ValueError('Figure 4 saved moments differ: ' + panel)
        result[panel] = {'pairs': len(frame), 'sources': len(sources), 'moments': moments,
                         'maximum_absolute_difference': float((actual-saved).abs().max())}
    primary = json.loads((root/'data/TITMC_FIG08_strength_ductility_primary_intervals.json').read_text('utf-8'))
    # The original source-role update seeds once, then visits RT followed by HT.
    # First-seen source codes preserve the original cluster draw identities.
    torch.manual_seed(figure_code.SEED)
    for panel in ('rt', 'ht'):
        frame = pairs.loc[pairs.panel.eq(panel)]
        codes, _ = pd.factorize(frame.paper_family_id, sort=False)
        analysis = figure_code.association(
            torch.tensor(frame[['el','uts']].to_numpy(float),device='cuda',dtype=torch.float64),
            torch.tensor(codes,device='cuda',dtype=torch.int64))
        fields = ('within_slope_ci_low','within_slope_ci_high')
        actual = torch.tensor([analysis[k] for k in fields],device='cuda',dtype=torch.float64)
        saved = torch.tensor([primary['panels'][panel][k] for k in fields],device='cuda',dtype=torch.float64)
        if not bool(torch.isclose(actual,saved,rtol=1e-10,atol=1e-9).all()):
            raise ValueError('Figure 4 primary bootstrap intervals differ: '+panel)
        result[panel]['primary_interval'] = {k:analysis[k] for k in fields}
    return {'backend': 'torch CUDA float64', 'device': torch.cuda.get_device_name(),
            'cohorts': result, 'model_fits': 0, 'bootstrap_intervals_recomputed': True,
            'bootstrap_draws':figure_code.N_BOOT,'bootstrap_seed':figure_code.SEED,
            'scope': 'Original equal-source covariance moments and primary within-source bootstrap intervals after existing admission; no source adjudication or causal inference. The separate 5000-draw sensitivity is not substituted for the primary protocol.'}


def observed_frontier(root: Path) -> dict:
    """Use the original complete saved-pair frontier replay, including portfolios."""
    import CODE_TITMC_FIG07_pareto_audit as figure_code
    return figure_code.reproduce_saved_pairs(root/'data')


def temporal_distributions(root: Path) -> dict:
    """Replay original temporal windows from admitted observations, not summaries."""
    import torch
    import CODE_TITMC_FIG04_temporal_frontier as figure_code
    if not torch.cuda.is_available():
        raise RuntimeError('GPU_REQUIRED: admitted temporal replay')
    states = read_table(root, 'data/TITMC_FIG04_temporal_frontier_states.csv')
    saved = read_table(root, 'data/TITMC_FIG04_temporal_frontier_source_data.csv')
    saved = saved.loc[saved.lane.eq('MAIN_REPORTED_RT')]
    rebuilt = []
    for spec in figure_code.ENDPOINTS:
        frame = states.loc[states.endpoint_id.eq(spec['id'])]
        if frame.empty:
            continue
        tensor = lambda key: torch.tensor(frame[key].to_numpy(float), device='cuda', dtype=torch.float64)
        groups = torch.tensor(frame.source_order_code.to_numpy(), device='cuda', dtype=torch.int64)
        rebuilt.extend(figure_code.temporal_windows(tensor('value'), tensor('year'), groups,
                                                        tensor('temperature_c'), spec))
    actual = pd.DataFrame(rebuilt)
    keys = ['endpoint_id', 'year']
    actual = actual.sort_values(keys).reset_index(drop=True)
    saved = saved.sort_values(keys).reset_index(drop=True)
    if len(actual) != len(saved) or not actual[keys].equals(saved[keys]):
        raise ValueError('Temporal window membership differs')
    maximum = 0.0
    for field in saved.columns:
        if field in keys or field in ('lane', 'property', 'unit', 'plot_supported'):
            if actual[field].astype(str).tolist() != saved[field].astype(str).tolist():
                raise ValueError('Temporal identity differs: ' + field)
            continue
        left = torch.tensor(pd.to_numeric(actual[field]).to_numpy(float), device='cuda', dtype=torch.float64)
        right = torch.tensor(pd.to_numeric(saved[field]).to_numpy(float), device='cuda', dtype=torch.float64)
        if not bool(torch.isclose(left, right, rtol=1e-10, atol=1e-9, equal_nan=True).all()):
            raise ValueError('Temporal numerical field differs: ' + field)
        delta = (left-right).abs()
        finite = delta[torch.isfinite(delta)]
        if len(finite):
            maximum = max(maximum, float(finite.max()))
    return {'backend': 'torch CUDA float64', 'device': torch.cuda.get_device_name(),
            'windows': len(actual), 'observations': len(states), 'maximum_absolute_difference': maximum,
            'bootstrap_draws': figure_code.N_BOOT, 'bootstrap_intervals_recomputed': True,
            'scope': 'Original main-RT endpoint windows and source-resampled intervals after existing admission; source-incidence and upstream admission are separate.'}


def microstructure_coordinates(root: Path) -> dict:
    import CODE_TITMC_FIG16_microstructure as figure_code
    return figure_code.reproduce_saved_scales(root/'data')


def association_cells(root: Path) -> dict:
    """Re-estimate cells from original admitted units, never the complete corpus."""
    import torch
    import build_association_cells as figure_code
    if not torch.cuda.is_available():
        raise RuntimeError('GPU_REQUIRED: association-cell replay')
    units = read_table(root, 'data/TITMC_REF11_admitted_units.csv')
    saved = read_table(root, 'data/TITMC_REF11_SourceData.csv')
    parameters = json.loads((root/'data/TITMC_REF11_publication_parameters.json').read_text('utf-8'))
    limits = parameters['thresholds']
    index_plans={(entry['row_id'],entry['col_id']):entry['index_device']
                 for entry in parameters['bootstrap_index_overrides']}
    controls = ['control_'+str(i) for i in range(len(parameters['controls']))]
    comparisons = []
    regenerated_p = []
    for row in saved.to_dict('records'):
        regenerated_p.append(float('nan'))
        frame = units.loc[units.row_id.eq(row['row_id']) & units.col_id.eq(row['col_id'])]
        if len(frame) != row['raw_n'] or frame.paper_family_id.nunique() != row['paperfamily_n']:
            raise ValueError('Association admitted unit population differs: '+row['row_id']+'/'+row['col_id'])
        if len(frame)<limits['min_raw_n'] or frame.paper_family_id.nunique()<limits['min_paperfamilies']:
            if row['status']!='UNDERPOWERED':
                raise ValueError('Association support status differs')
            continue
        actual = figure_code._effect_once(frame,'x','y','paper_family_id',controls)
        fields = {'raw_rho':actual['raw_rho'],'between_rho':actual['between_rho'],
                  'informative_paperfamily_n':actual['informative_paperfamily_n']}
        if actual['informative_paperfamily_n']>=limits['min_informative_paperfamilies'] and np.isfinite(actual['effect']):
            seed = limits['seed']+int(frame.feature_index.iloc[0])*1009+int(frame.endpoint_index.iloc[0])*9176
            index_device=index_plans.get((row['row_id'],row['col_id']),parameters['bootstrap_index_default']['index_device'])
            lo,hi,p = figure_code._bootstrap_effect(frame,'x','y','paper_family_id',controls,
                limits['bootstrap_reps'],seed,index_device=index_device,
                legacy_cpu_probability=index_device=='cpu')
            fields.update(effect=actual['effect'],ci95_low=lo,ci95_high=hi,p_boot=p)
            regenerated_p[-1]=p
            if np.isfinite(p):
                fields.update(figure_code._leave_one_family_out_stability(
                    frame,'x','y','paper_family_id',controls,actual['effect']))
                fields.update(figure_code._random_effects_family_meta(
                    frame,'x','y','paper_family_id',limits['min_meta_families']))
        else:
            fields['effect']=float('nan')
        left = torch.tensor(list(fields.values()),device='cuda',dtype=torch.float64)
        right = torch.tensor([row[k] for k in fields],device='cuda',dtype=torch.float64)
        if not bool(torch.isclose(left,right,atol=1e-9,rtol=1e-10,equal_nan=True).all()):
            raise ValueError('Association numerical cell differs: '+row['row_id']+'/'+row['col_id'])
        comparisons.append({'row_id':row['row_id'],'col_id':row['col_id'],'fields':list(fields)})
    p_values=pd.Series(regenerated_p,index=saved.index)
    global_q=figure_code.benjamini_hochberg(p_values)
    block_q=pd.Series(np.nan,index=saved.index)
    for _,indices in saved.groupby('col_block',dropna=False).groups.items():
        block_q.loc[indices]=figure_code.benjamini_hochberg(p_values.loc[indices])
    for field,actual in (('q_value',global_q),('q_value_global',global_q),('q_value_block',block_q)):
        left=torch.tensor(actual.to_numpy(float),device='cuda',dtype=torch.float64)
        right=torch.tensor(saved[field].to_numpy(float),device='cuda',dtype=torch.float64)
        if not bool(torch.isclose(left,right,atol=1e-9,rtol=1e-10,equal_nan=True).all()):
            raise ValueError('Association multiplicity adjustment differs: '+field)
    return {'backend':'torch CUDA float64','device':torch.cuda.get_device_name(),
            'reference_cells':len(saved),'estimated_cells':len(comparisons),'model_fits':0,
            'bootstrap_intervals_recomputed':True,'bootstrap_draws':limits['bootstrap_reps'],
            'scope':'Original rank, within/between effects, exact family-reranked intervals, LOFO, random-effects summaries and original global/block FDR from admitted state-unit medians. Raw source adjudication and census admission remain separate.'}


def material_dose(root: Path) -> dict:
    """Replay original dose OLS and complete source-bootstrap summaries on CUDA."""
    import torch
    import material_dose as figure_code
    if not torch.cuda.is_available():
        raise RuntimeError('GPU_REQUIRED: admitted dose-series replay')
    base=root/'data'
    series=pd.read_csv(base/'material_dose_series.csv.gz',float_precision='round_trip')
    points=pd.read_csv(base/'material_dose_series_points.csv.gz',float_precision='round_trip')
    settings=json.loads((base/'material_dose_parameters.json').read_text('utf-8'))
    if not series.series_id.is_unique or set(series.series_id)!=set(points.series_id):
        raise ValueError('Dose point/series identities differ')
    rebuilt=series.copy()
    positions={str(value):index for index,value in enumerate(series.series_id)}
    for identity,frame in points.groupby('series_id',sort=False):
        frame=frame.sort_values('dose',kind='stable')
        actual=figure_code.cuda_series_metrics(frame.dose.to_numpy(),frame.value.to_numpy())
        index=positions[str(identity)]
        for field,value in actual.items():rebuilt.at[index,field]=value
        rebuilt.at[index,'baseline_value']=float(frame.value.iloc[0])
        rebuilt.at[index,'final_value']=float(frame.value.iloc[-1])
        rebuilt.at[index,'n_rows']=int(figure_code.cuda_reduce(frame.n_records,'sum'))
    plans={tuple(row['key']):row for row in settings['bootstrap_plans']}
    if settings['summary_keys']!=figure_code.SUMMARY_KEYS:
        raise ValueError('Dose group definition differs')
    sources,summaries=figure_code.summarize(rebuilt,settings['bootstrap'],settings['seed'],
                                                backend='cuda',bootstrap_plans=plans)
    def compare(actual,reference,keys,label):
        actual=actual.sort_values(keys,kind='stable').reset_index(drop=True)
        reference=reference.sort_values(keys,kind='stable').reset_index(drop=True)
        if list(actual.columns)!=list(reference.columns) or len(actual)!=len(reference):
            raise ValueError('Dose table population/columns differ: '+label)
        maximum=0.0
        for field in reference:
            left=pd.to_numeric(actual[field],errors='coerce');right=pd.to_numeric(reference[field],errors='coerce')
            numeric=(left.notna()|actual[field].isna()).all() and (right.notna()|reference[field].isna()).all()
            if numeric:
                x=torch.tensor(left.to_numpy(float),device='cuda',dtype=torch.float64)
                y=torch.tensor(right.to_numpy(float),device='cuda',dtype=torch.float64)
                if not bool(torch.isclose(x,y,rtol=1e-8,atol=1e-8,equal_nan=True).all()):
                    raise ValueError('Dose numerical field differs: '+label+'/'+field)
                delta=(x-y).abs();delta=delta[torch.isfinite(delta)]
                if len(delta):maximum=max(maximum,float(delta.max()))
            elif actual[field].fillna('').astype(str).tolist()!=reference[field].fillna('').astype(str).tolist():
                raise ValueError('Dose identity field differs: '+label+'/'+field)
        return {'rows':len(actual),'maximum_absolute_difference':maximum}
    results={'series':compare(rebuilt,series,['series_id'],'series'),
             'source_effects':compare(sources,pd.read_csv(base/'material_dose_source_effects.csv.gz',float_precision='round_trip'),
                                      figure_code.SUMMARY_KEYS+['source'],'source_effects'),
             'summary_effects':compare(summaries,pd.read_csv(base/'material_dose_summary_effects.csv',float_precision='round_trip'),
                                       figure_code.SUMMARY_KEYS,'summary_effects')}
    return {'backend':'torch CUDA float64','device':torch.cuda.get_device_name(),'comparisons':results,
            'bootstrap_intervals_recomputed':True,'bootstrap_draws':settings['bootstrap'],'seed':settings['seed'],
            'rng':'Original NumPy PCG64 integer source-index plans only; numerical statistics and quantiles on CUDA',
            'model_fits':0,'scope':'All supplied admitted dose-series OLS, source-level summaries and pointwise mean/median intervals. Upstream raw source admission and distinct main-protocol paired spans remain separate.'}


def addition_map_data(root: Path, output: Path) -> dict:
    """Execute the original numerical-only map producer and compare every CSV."""
    subprocess.run([sys.executable,str(root/'tools/reproduce_addition_maps.py'),
                    '--source',str(root/'data'),'--output',str(output),'--data-only'],
                   check=True,capture_output=True,text=True)
    reference=json.loads((root/'data/addition_maps_receipt.json').read_text('utf-8'))
    generated=json.loads((output/'addition_maps_receipt.json').read_text('utf-8'))
    comparisons={}
    for name,entry in reference['outputs'].items():
        identical=hashlib.sha256((output/name).read_bytes()).hexdigest()==entry['sha256']
        maximum=0.0
        if not identical:
            import torch
            left=pd.read_csv(root/'data'/name,float_precision='round_trip')
            right=pd.read_csv(output/name,float_precision='round_trip')
            if left.shape!=right.shape or left.columns.tolist()!=right.columns.tolist():
                raise ValueError('Addition-map CSV identities differ: '+name)
            for field in left:
                x=pd.to_numeric(left[field],errors='coerce');y=pd.to_numeric(right[field],errors='coerce')
                numeric=(x.notna()|left[field].isna()).all() and (y.notna()|right[field].isna()).all()
                if numeric:
                    x=torch.tensor(x.to_numpy(float),device='cuda',dtype=torch.float64)
                    y=torch.tensor(y.to_numpy(float),device='cuda',dtype=torch.float64)
                    if not bool(torch.isclose(x,y,rtol=1e-12,atol=1e-9,equal_nan=True).all()):
                        raise ValueError('Addition-map numerical field differs: '+name+'/'+field)
                    delta=(x-y).abs();delta=delta[torch.isfinite(delta)]
                    if len(delta):maximum=max(maximum,float(delta.max()))
                elif left[field].fillna('').astype(str).tolist()!=right[field].fillna('').astype(str).tolist():
                    raise ValueError('Addition-map identity field differs: '+name+'/'+field)
        comparisons[name]=dict(byte_identical=identical,maximum_absolute_difference=maximum)
    if generated['counts']!=reference['counts']:
        raise ValueError('Addition-map comparison population differs')
    return {'backend':'torch CUDA float64','device':generated['numeric_gpu']['device'],
            'counts':generated['counts'],'csv_files':len(reference['outputs']),
            'comparisons':comparisons,'comparison_atol':1e-9,'comparison_rtol':1e-12,
            'bootstrap_draws':generated['bootstrap']['draws'],'bootstrap_intervals_recomputed':True,
            'figures_rendered':False,'scope':'Original designed-series, family means, medians, 4000-draw intervals and content bins; no new figure pixels.'}


def saved_model_metrics(root: Path) -> dict:
    import render_five_target_model_authority_atlas as figure_code
    _,points,checks=figure_code.load_and_validate(
        root/'data/model_oof_authority_registry.csv',root/'data/passed_model_point_predictions.csv')
    return dict(prediction_queries=len(points),targets=checks,computation=figure_code.COMPUTE_RECEIPT,
        model_fits=0,model_inference=0,bootstrap_intervals_recomputed=False,
        scope='Figure 9 saved-query metrics, residual normalization and error ECDFs under each recorded support/overlap boundary; not model retraining, recalibration or unseen-source validation.')


def target_plane(root: Path) -> dict:
    """Replay the original fixed-cohort observed-label CUDA threshold recount."""
    import torch
    import selection_target_plane as figure_code
    actual, protocol = figure_code.recount_target_plane(root)
    reference = read_table(root, 'data/selection_target_plane.csv')
    if list(actual.columns) != list(reference.columns) or len(actual) != 40401 or len(reference) != 40401:
        raise ValueError('Target-plane schema or cardinality differs')
    a = torch.as_tensor(actual.to_numpy(float),device='cuda',dtype=torch.float64)
    b = torch.as_tensor(reference.to_numpy(float),device='cuda',dtype=torch.float64)
    if not bool(torch.isclose(a,b,rtol=1e-12,atol=1e-10).all()):
        raise ValueError('Original target-plane recount differs from saved numerical reference')
    return dict(cells=len(actual),maximum_absolute_difference=float((a-b).abs().max()),
                primary_candidates=protocol['primary_candidates'],primary_sources=protocol['primary_sources'],
                source_local_X_blocks=protocol['source_local_X_blocks'],backend='torch CUDA float64',
                model_fits=0,model_inference=0,figures_rendered=False,
                scope='Original observed-label references on the fixed qualified cohort; no new model or deployable policy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analysis", choices=("representation", "utility", "sequential", "all", "material-relations", "frontier", "temporal", "microstructure", "associations", "dose", "addition-maps", "materials", "saved-models", "target-plane"),
                        default="materials")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root/"results"/args.analysis.replace('-', '_')
    if output == root or any(output.is_relative_to(root/p) for p in ("paper01", "tools", ".git")):
        raise ValueError("The output directory must be separate from the released inputs and code.")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "passed", "analysis": args.analysis,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                          "backend": "CPU float64"},
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_fits": 0, "bootstrap_intervals_recomputed": False}
    if args.analysis in ("representation", "all"):
        report["representation"] = representation(root, output/"representation")
    if args.analysis in ("utility", "sequential", "all"):
        import selection_analysis
        report["selection"] = selection_analysis.run(root, output, args.analysis)
    if args.analysis in ('material-relations', 'frontier', 'temporal', 'microstructure', 'associations', 'dose', 'addition-maps', 'materials', 'saved-models', 'target-plane'):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('GPU_REQUIRED: admitted-material replay')
        report['runtime'].update(backend='torch CUDA float64', torch=torch.__version__,
                                 cuda=torch.version.cuda, device=torch.cuda.get_device_name())
        functions = {'material-relations': material_relations, 'frontier': observed_frontier,
                     'temporal': temporal_distributions, 'microstructure': microstructure_coordinates,
                     'associations':association_cells, 'dose':material_dose,
                     'addition-maps':lambda value:addition_map_data(value,output/'addition_maps')}
        if args.analysis=='target-plane':
            report['materials']=target_plane(root)
        elif args.analysis=='saved-models':
            report['materials']=saved_model_metrics(root)
        elif args.analysis=='materials':
            report['status']='running'
            report['materials']={}
            for name,function in functions.items():
                try:
                    report['materials'][name]=function(root)
                except Exception as exc:
                    report.update(status='failed',failed_module=name,error=str(exc))
                    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
                    raise
                (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            report['status']='passed'
        else:
            report['materials'] = functions[args.analysis](root)
        report['bootstrap_intervals_recomputed'] = report['materials'].get('bootstrap_intervals_recomputed',False)
        if args.analysis=='materials':
            report['bootstrap_intervals_recomputed']=any(value.get('bootstrap_intervals_recomputed',False)
                                                        for value in report['materials'].values())
    if (root/"DATA_INPUTS.json").is_file():
        report["data_inputs_sha256"] = hashlib.sha256((root/"DATA_INPUTS.json").read_bytes()).hexdigest()
    text = json.dumps(report, indent=2)+"\n"
    (output/"report.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

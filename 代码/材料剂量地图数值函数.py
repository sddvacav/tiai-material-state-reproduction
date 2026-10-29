"""Figures 11 and 12: whole-domain maps of how additions move titanium properties.

Reads the source-context series supplied under data/. A series is compared when its amount at least
doubles within its source context (additions from zero included). Alloying-element
series are read only in contexts where the reinforcement content and the
reinforcement-forming elements B, C and La stay unchanged, so matrix dilution by
reinforcement cannot appear as an alloying response. Each paper family contributes
the mean of its series changes. Medians, net directions, family-bootstrap intervals and
binned quantiles run on PyTorch CUDA; both figures are composed at their Word inline
width (178 mm) and rasterized by the shared ti_figstyle Skia GPU route.

usage: build_addition_maps.py [--source DIR] [--output DIR] [--log FILE]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source', type=Path, default=PAPER / 'analysis' / 'material_dose')
ap.add_argument('--output', type=Path, default=HERE / 'outputs')
ap.add_argument('--log', type=Path)
ap.add_argument('--data-only', action='store_true', help='Refresh original numerical Source Data without rendering or changing figure pixels')
ap.add_argument('--render-only', action='store_true', help='Draw from the registered numerical CSV files without recomputing or rewriting statistics')
ap.add_argument('--statistics-directory', type=Path, help='Existing addition-map CSV directory for --render-only')
ap.add_argument('--figure', choices=('11', '12', 'all'), default='all', help='Select one existing figure for revision')
args = ap.parse_args()
args.data_only = True
args.render_only = False
args.statistics_directory = None
args.figure = 'all'
if args.data_only and args.render_only:
    ap.error('--data-only and --render-only are mutually exclusive')
if args.statistics_directory and (not args.render_only):
    ap.error('--statistics-directory requires --render-only')
out = args.output
out.mkdir(parents=True, exist_ok=True)
prior_receipt_text = None
prior_csv_text = {}
if args.data_only and (out / 'addition_maps_receipt.json').exists():
    prior_text = (out / 'addition_maps_receipt.json').read_text(encoding='utf-8')
    prior_document = json.loads(prior_text)
    prior_receipt_text = prior_document.get('prior_render_receipt_text', prior_text)
    prior_csv_text = prior_document.get('prior_render_csv_text', {path.name: path.read_text(encoding='utf-8') for path in out.glob('addition_maps_*.csv')})
if args.log:
    sys.stdout = sys.stderr = open(args.log, 'w', encoding='utf-8')
import numpy as np
import pandas as pd
import torch
if not torch.cuda.is_available():
    raise SystemExit('GPU_REQUIRED: CUDA is not available')
DEV = torch.device('cuda')
GEN = torch.Generator(device='cuda')
GEN.manual_seed(20261028)
DRAWS = 4000
CUDA_LOG = []
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
INPUTS = {n: args.source / n for n in ('all_source_context_spans.csv.gz', 'all_paired_strength_elongation_spans.csv.gz', 'all_source_context_series_points.csv.gz')}
INPUT_IDENTITIES = {n: dict(bytes=p.stat().st_size, sha256=sha(p)) for (n, p) in INPUTS.items()}
saved_numerics = None
if args.render_only:
    statistics_directory = args.statistics_directory or out
    saved_numerics = json.loads((statistics_directory / 'addition_maps_receipt.json').read_text(encoding='utf-8'))
    if saved_numerics['inputs'] != INPUT_IDENTITIES:
        raise RuntimeError('Saved addition-map statistics belong to a different input generation')
    frames = {'finger': 'fingerprint', 'finger_all': 'fingerprint_all_series', 'compass': 'compass', 'gsum': 'group_medians', 'design': 'design', 'bands': 'reinforcement_bands', 'dose': 'content_bins', 'cloud': 'reinforcement_cloud', 'pts': 'content_points', 'pcloud': 'designed_series'}
    for (variable, stem) in frames.items():
        path = statistics_directory / ('addition_maps_' + stem + '.csv')
        expected = saved_numerics['outputs'][path.name]
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise RuntimeError('Saved addition-map CSV changed: ' + path.name)
        globals()[variable] = pd.read_csv(path, low_memory=False)
    contrast_path = statistics_directory / 'addition_maps_contrast.csv'
    if sha(contrast_path) != saved_numerics['outputs'][contrast_path.name]['sha256']:
        raise RuntimeError('Saved addition-map contrast table changed')
    print('SAVED_STATISTICS', json.dumps(saved_numerics['counts']), flush=True)
else:
    spans = pd.read_csv(INPUTS['all_source_context_spans.csv.gz'], low_memory=False)
    paired = pd.read_csv(INPUTS['all_paired_strength_elongation_spans.csv.gz'], low_memory=False)
    points = pd.read_csv(INPUTS['all_source_context_series_points.csv.gz'], low_memory=False)
    spans['family'] = spans['paper_family_id'].fillna(spans['doi']).astype(str)
    paired['family'] = paired['paper_family_id_UTS'].fillna(paired['doi_UTS']).astype(str)
REINF_FORMING = ['Total reinforcement', 'B', 'C', 'La']
GROUPS = [('Reinforcement-forming', REINF_FORMING, 2, 'D'), ('Interstitial', ['O', 'N', 'H'], 3, '^'), ('α stabilizer / neutral', ['Al', 'Zr', 'Sn'], 5, 's'), ('β isomorphous', ['Mo', 'V', 'Nb', 'Ta', 'W'], 0, 'o'), ('β eutectoid', ['Fe', 'Cr', 'Ni', 'Cu', 'Si'], 1, 'v')]
ROLE = {s: (g, c, m) for (g, members, c, m) in GROUPS for s in members}
SPECIES = [s for (_, members, _, _) in GROUPS for s in members]
SHORT = {'Total reinforcement': 'Reinf.'}
PROPS = [('ultimate_tensile_strength', 'UTS', 'relative_change_pct', '%'), ('yield_strength_unspecified', 'YS', 'relative_change_pct', '%'), ('elongation_unspecified', 'EL', 'absolute_change', 'pp'), ('vickers_hardness', 'HV', 'relative_change_pct', '%'), ('elastic_modulus', 'E', 'relative_change_pct', '%')]
(ROW_MIN, CELL_MIN, COMPASS_MIN) = (5, 3, 5)
if not args.render_only:
    reinforced_contexts = set(spans.loc[spans['dose_species'] == 'Total reinforcement', 'context_id'])

def compared(frame, cov_col):
    designed = frame['dose_low'] <= 0.5 * frame['dose_high']
    covary = frame[cov_col].fillna('').astype(str).str.split(';').apply(lambda xs: bool({'B', 'C', 'La'} & {x.strip() for x in xs}))
    clean = ~covary & ~frame['context_id'].isin(reinforced_contexts)
    return designed & (frame['dose_species'].isin(REINF_FORMING) | clean)
if not args.render_only:
    spans['compared'] = compared(spans, 'co_varying_elements')
    paired['compared'] = compared(paired, 'co_varying_elements_UTS')

def tband(state, t):
    if state != 'REPORTED' or not np.isfinite(t):
        return 'NR'
    if 15 <= t <= 35:
        return 'RT'
    return 'HT' if t >= 400 else 'MID'
if not args.render_only:
    spans['tband'] = [tband(s, t) for (s, t) in zip(spans['test_temperature_state'], spans['test_temperature_c'].astype(float))]
    paired['tband'] = [tband(s, t) for (s, t) in zip(paired['test_temperature_state_UTS'], paired['test_temperature_c_UTS'].astype(float))]

def family_means(frame, col):
    """Mean of col within each paper family (CUDA scatter-add)."""
    v = frame[col].to_numpy(dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return torch.zeros(0, dtype=torch.float64, device=DEV)
    (codes, uniq) = pd.factorize(frame['family'].to_numpy()[ok])
    t0 = time.perf_counter()
    vals = torch.tensor(v[ok], dtype=torch.float64, device=DEV)
    idx = torch.tensor(codes, dtype=torch.int64, device=DEV)
    sums = torch.zeros(len(uniq), dtype=torch.float64, device=DEV).scatter_add_(0, idx, vals)
    cnt = torch.zeros(len(uniq), dtype=torch.float64, device=DEV).scatter_add_(0, idx, torch.ones_like(vals))
    res = sums / cnt
    torch.cuda.synchronize()
    CUDA_LOG.append(('family_means', int(ok.sum()), time.perf_counter() - t0))
    return res

def summary(fam):
    """Median, 95% family-bootstrap percentile interval and net direction (up - down) / n."""
    n = int(fam.numel())
    if n == 0:
        return dict(n=0, median=np.nan, lo=np.nan, hi=np.nan, direction=np.nan, up=0, down=0)
    t0 = time.perf_counter()
    med = torch.quantile(fam, 0.5)
    (up, down) = (int((fam > 0).sum()), int((fam < 0).sum()))
    lo = hi = torch.tensor(float('nan'), device=DEV)
    if n >= 2:
        idx = torch.randint(0, n, (DRAWS, n), device=DEV, generator=GEN)
        boot = torch.quantile(fam[idx], 0.5, dim=1)
        (lo, hi) = torch.quantile(boot, torch.tensor([0.025, 0.975], dtype=torch.float64, device=DEV))
    torch.cuda.synchronize()
    CUDA_LOG.append(('median_bootstrap', n, time.perf_counter() - t0))
    return dict(n=n, median=float(med), lo=float(lo), hi=float(hi), direction=(up - down) / n, up=up, down=down)
DESIGN = [('Phase', 'final_reinforcement_phase', [('TiB', 'TiB'), ('TiC', 'TiC')]), ('Precursor', 'precursor_phase', [('B4C', 'B$_4$C'), ('TiB2', 'TiB$_2$')]), ('Architecture', 'reinforcement_architecture', [('in_situ', 'In situ'), ('network', 'Network'), ('whisker', 'Whisker'), ('particle', 'Particle')]), ('Route', 'process_route', [('wrought', 'Wrought'), ('cast', 'Cast'), ('pm', 'PM'), ('am_ded', 'DED'), ('am_pbf', 'PBF')])]
if not args.render_only:
    (finger, finger_all) = ([], [])
    for s in SPECIES:
        for (prop, short, col, unit) in PROPS:
            g = spans[(spans['dose_species'] == s) & (spans['property'] == prop)]
            c = g[g['compared']]
            finger.append(dict(species=s, property=short, unit=unit, spans=len(c), **summary(family_means(c, col))))
            finger_all.append(dict(species=s, property=short, unit=unit, spans=len(g), **summary(family_means(g, col))))
    (finger, finger_all) = (pd.DataFrame(finger), pd.DataFrame(finger_all))
    compass = []
    for s in SPECIES:
        g = paired[(paired['dose_species'] == s) & paired['compared']]
        (u, e) = (summary(family_means(g, 'relative_change_pct_UTS')), summary(family_means(g, 'absolute_change_EL')))
        compass.append(dict(species=s, spans=len(g), families=u['n'], uts_median=u['median'], uts_lo=u['lo'], uts_hi=u['hi'], el_median=e['median'], el_lo=e['lo'], el_hi=e['hi']))
    compass = pd.DataFrame(compass)
    rank = {sp_: i for (i, sp_) in enumerate(SPECIES)}
    pcloud = paired[paired['compared'] & paired['dose_species'].isin(SPECIES)].copy()
    pcloud['rank'] = pcloud['dose_species'].map(rank)
    pcloud = pcloud.sort_values('rank').drop_duplicates(['context_id', 'relative_change_pct_UTS', 'absolute_change_EL'])
    pcloud['group'] = pcloud['dose_species'].map(lambda sp_: ROLE[sp_][0])
    gsum = []
    for (gname, members, _, _) in GROUPS:
        g = pcloud[pcloud['dose_species'].isin(members)]
        (u, e) = (summary(family_means(g, 'relative_change_pct_UTS')), summary(family_means(g, 'absolute_change_EL')))
        gsum.append(dict(group=gname, series=len(g), families=u['n'], uts_median=u['median'], uts_lo=u['lo'], uts_hi=u['hi'], el_median=e['median'], el_lo=e['lo'], el_hi=e['hi']))
    gsum = pd.DataFrame(gsum)
    reinf = spans[(spans['dose_kind'] == 'reinforcement') & spans['compared']].copy()
    design = []
    for (head, col, cats) in DESIGN:
        for (key, label) in cats:
            g = reinf[reinf[col] == key]
            u = summary(family_means(g[g['property'] == 'ultimate_tensile_strength'], 'relative_change_pct'))
            e = summary(family_means(g[g['property'] == 'elongation_unspecified'], 'absolute_change'))
            design.append(dict(group=head, key=key, label=label, uts_n=u['n'], uts_median=u['median'], uts_lo=u['lo'], uts_hi=u['hi'], el_n=e['n'], el_median=e['median'], el_lo=e['lo'], el_hi=e['hi']))
    design = pd.DataFrame(design)
    cloud = paired[(paired['dose_species'] == 'Total reinforcement') & paired['compared']].copy()
    bands = []
    for b in ('RT', 'HT'):
        g = cloud[cloud['tband'] == b]
        (u, e) = (summary(family_means(g, 'relative_change_pct_UTS')), summary(family_means(g, 'absolute_change_EL')))
        bands.append(dict(band=b, spans=len(g), families=u['n'], uts_median=u['median'], uts_lo=u['lo'], uts_hi=u['hi'], el_median=e['median'], el_lo=e['lo'], el_hi=e['hi']))
    bands = pd.DataFrame(bands)
    pts = points[points['dose_species'] == 'Total reinforcement'].sort_values(['series_id', 'dose']).copy()
    first = pts.groupby('series_id').head(1).set_index('series_id')
    pts['base_dose'] = pts['series_id'].map(first['dose'])
    pts['base_value'] = pts['series_id'].map(first['value_median'])
    pts = pts[(pts['dose'] > pts['base_dose']) & (pts['base_dose'] == 0)].copy()
    point_values = torch.tensor(pts[['value_median', 'base_value']].to_numpy(float), device=DEV, dtype=torch.float64)
    point_delta = point_values[:, 0] - point_values[:, 1]
    pts['d_rel'] = (point_delta / point_values[:, 1] * 100.0).cpu().numpy()
    pts['d_abs'] = point_delta.cpu().numpy()
    pts['family'] = pts['series_id'].map(spans.drop_duplicates('series_id').set_index('series_id')['family'])
    EDGES = [0, 2, 4, 6, 8, 10, 15, 20, 30]
    dose = []
    for (prop, col) in (('ultimate_tensile_strength', 'd_rel'), ('elongation_unspecified', 'd_abs')):
        g = pts[pts['property'] == prop]
        for (lo, hi) in zip(EDGES[:-1], EDGES[1:]):
            h = g[(g['dose'] > lo) & (g['dose'] <= hi)]
            if h['family'].nunique() < 3:
                continue
            v = family_means(h, col)
            t0 = time.perf_counter()
            q = torch.quantile(v, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64, device=DEV)).tolist()
            torch.cuda.synchronize()
            CUDA_LOG.append(('content_bin_quantiles', int(v.numel()), time.perf_counter() - t0))
            dose.append(dict(property=prop, lo=lo, hi=hi, centre=(lo + hi) / 2, points=len(h), series=h['series_id'].nunique(), families=int(v.numel()), q25=q[0], q50=q[1], q75=q[2]))
    dose = pd.DataFrame(dose)
    for (name, frame) in (('fingerprint', finger), ('fingerprint_all_series', finger_all), ('compass', compass), ('group_medians', gsum), ('design', design), ('reinforcement_bands', bands), ('content_bins', dose)):
        frame.to_csv(out / f'addition_maps_{name}.csv', index=False, encoding='utf-8')
    cloud[['series_id_UTS', 'family', 'context_id', 'dose_low', 'dose_high', 'tband', 'test_temperature_c_UTS', 'relative_change_pct_UTS', 'absolute_change_EL', 'reinforcement_architecture_UTS', 'doi_UTS']].to_csv(out / 'addition_maps_reinforcement_cloud.csv', index=False, encoding='utf-8')
    pts[['series_id', 'family', 'property', 'dose', 'base_dose', 'd_rel', 'd_abs']].to_csv(out / 'addition_maps_content_points.csv', index=False, encoding='utf-8')
    pcloud[['series_id_UTS', 'family', 'context_id', 'dose_species', 'group', 'dose_low', 'dose_high', 'tband', 'relative_change_pct_UTS', 'absolute_change_EL', 'doi_UTS']].to_csv(out / 'addition_maps_designed_series.csv', index=False, encoding='utf-8')
    pd.set_option('display.width', 250)
    print('compared spans', int(spans['compared'].sum()), 'of', len(spans), '; compared paired', int(paired['compared'].sum()), 'of', len(paired), flush=True)
    print('designed paired series after removing duplicates', len(pcloud), '; reinforcement cloud bands', cloud['tband'].value_counts().to_dict(), flush=True)
    for (name, frame) in (('fingerprint', finger), ('compass', compass), ('groups', gsum), ('design', design), ('bands', bands), ('content', dose)):
        print('==', name, flush=True)
        print(frame.round(3).to_string(), flush=True)
if args.data_only:
    rows = [s for s in SPECIES if finger[(finger.species == s) & finger.property.isin(['UTS', 'EL'])]['n'].max() >= ROW_MIN]
    contrast = []
    for prop in ('UTS', 'EL'):
        for s in [value for value in rows if value not in REINF_FORMING]:
            a = finger_all[(finger_all.species == s) & (finger_all.property == prop)].iloc[0]
            d = finger[(finger.species == s) & (finger.property == prop)].iloc[0]
            contrast.append(dict(species=s, property=prop, all_families=a['n'], all_median=a['median'], designed_families=d['n'], designed_median=d['median'], designed_lo=d['lo'], designed_hi=d['hi']))
    pd.DataFrame(contrast).to_csv(out / 'addition_maps_contrast.csv', index=False, encoding='utf-8')
    if any((sha(p) != INPUT_IDENTITIES[n]['sha256'] for (n, p) in INPUTS.items())):
        raise RuntimeError('Addition-map source inputs changed during numerical refresh')
    receipt = dict(mode='DATA_ONLY', purpose='Numerical Source Data of Figures 11 and 12', inputs=INPUT_IDENTITIES, code=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)), comparison=dict(designed='amount at least doubles within source context; zero-baseline additions included', alloying='reinforcement content and B/C/La fixed in the same context', aggregation='paper-family mean of series changes'), bootstrap=dict(draws=DRAWS, seed=20261028, interval='95% percentile of the family-bootstrap median'), counts=dict(spans=len(spans), compared_spans=int(spans['compared'].sum()), paired=len(paired), compared_paired=int(paired['compared'].sum()), families=int(spans['family'].nunique()), designed_series=len(pcloud), reinforcement_cloud=len(cloud), content_points=len(pts), zero_baseline_content_series=int(pts['series_id'].nunique()), zero_baseline_content_sources=int(pts['family'].nunique())), numeric_gpu=dict(backend='PyTorch CUDA', torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0), operations=len(CUDA_LOG), cpu_fallback=False), figures_rendered=False, prior_render_receipt_text=prior_receipt_text, prior_render_csv_text=prior_csv_text)
    receipt['outputs'] = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.glob('addition_maps_*.csv'))}
    (out / 'addition_maps_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('DATA_ONLY_DONE', json.dumps(receipt['counts']), flush=True)
    raise SystemExit(0)

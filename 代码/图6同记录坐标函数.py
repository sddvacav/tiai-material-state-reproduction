"""Same-record microstructure descriptor–property coordinates (Figure 6)."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import torch

SPECS = [dict(id='grain', field='grain_size_um', prop='ultimate_tensile_strength', unit='mpa', en='Grain size (µm)', zh='晶粒尺寸 (µm)', log=True), dict(id='prior_beta', field='prior_beta_grain_size_um', prop='ultimate_tensile_strength', unit='mpa', en='Prior-β grain (µm)', zh='原β晶粒 (µm)', log=True), dict(id='alpha_lath', field='alpha_lath_size_um', prop='ultimate_tensile_strength', unit='mpa', en='α-lath size (µm)', zh='α板条尺寸 (µm)', log=True), dict(id='colony', field='colony_size_um', prop='ultimate_tensile_strength', unit='mpa', en='Colony size (µm)', zh='集束尺寸 (µm)', log=True), dict(id='alpha_fraction', field='alpha_phase_fraction_pct', prop='elongation_unspecified', unit='%', en='α fraction (%)', zh='α相比例 (%)', log=False), dict(id='beta_fraction', field='beta_phase_fraction_pct', prop='elongation_unspecified', unit='%', en='β fraction (%)', zh='β相比例 (%)', log=False)]

def reproduce_saved_scales(source):
    """Reconstruct admitted same-row descriptor/property coordinates on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError('GPU_REQUIRED: admitted microstructure coordinates')
    source = Path(source)
    frame = pd.read_csv(source / 'TITMC_FIG16_microstructure_source_data.csv', float_precision='round_trip')
    reference = json.loads((source / 'TITMC_FIG16_microstructure_claim_receipt.json').read_text(encoding='utf-8'))
    panels = []
    for spec in SPECS:
        part = frame.loc[frame.panel.eq(spec['id'])]
        original = torch.tensor(part[[spec['field'], 'value']].to_numpy(float), device='cuda', dtype=torch.float64)
        recorded = torch.tensor(part[['x', 'y']].to_numpy(float), device='cuda', dtype=torch.float64)
        if not bool(torch.isfinite(original).all()) or not bool(torch.equal(original, recorded)):
            raise ValueError('Microstructure coordinates differ from their original fields: ' + spec['id'])
        if not part.descriptor_field.eq(spec['field']).all() or not part.property.eq(spec['prop']).all():
            raise ValueError('Microstructure descriptor/endpoint identity differs: ' + spec['id'])
        if part.doi.isna().any() or part.paper_family_id.isna().any():
            raise ValueError('An admitted microstructure coordinate has no source identity')
        (codes, _) = pd.factorize(part.paper_family_id, sort=False)
        count = torch.unique(torch.tensor(codes, device='cuda', dtype=torch.int64)).numel()
        expected = next((row for row in reference['panel_denominators'] if row['panel'] == spec['id']))
        if len(part) != expected['plotted_observations'] or count != expected['plotted_sources']:
            raise ValueError('Microstructure panel population differs: ' + spec['id'])
        panels.append(dict(panel=spec['id'], observations=len(part), sources=count, coordinates_equal=True))
    return dict(device=torch.cuda.get_device_name(), backend='torch CUDA float64', panels=panels, model_fits=0, scope='Same-row physical descriptor/property mapping after existing admission; broad raw-field census and categorical label admission are not recomputed.')

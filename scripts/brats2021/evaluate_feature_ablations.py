from __future__ import annotations

import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score

from src.brats2021.controller import fit, predict, evaluate


def parse_grid(s):
    return [float(x) for x in str(s).split(',')]


def drop_variant(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    x = df.copy()
    if variant == 'full':
        return x
    if variant == 'no_error_map':
        return x.drop(columns=[c for c in x.columns if c.startswith('err_')], errors='ignore')
    if variant == 'no_uncertainty':
        needles = ('src_entropy_', 'src_low_conf_', 'src_mc_var_')
        return x.drop(columns=[c for c in x.columns if c.startswith(needles)], errors='ignore')
    if variant == 'no_region_identity':
        return x.drop(columns=[c for c in x.columns if c.startswith('region_')], errors='ignore')
    if variant == 'geometry_action_only':
        keep_meta = {
            'case_id','unit_id','region','split','action_id','action_name','base_dice','base_hd95','base_utility','base_fp','base_fn',
            'candidate_dice','candidate_hd95','candidate_utility','delta_dice','delta_utility','harmful','beneficial',
            'diagnostic_failure_type','oracle_best_action','oracle_best_action_id','oracle_best_gain','spacing_x_mm','spacing_y_mm','spacing_z_mm'
        }
        keep_features = [c for c in x.columns if c.startswith(('src_fg_','src_component_','src_largest_','src_boundary_','act_','region_'))]
        return x[[c for c in x.columns if c in keep_meta or c in keep_features]].copy()
    raise ValueError(variant)


def calibrate(bundle, val: pd.DataFrame, pc: dict):
    pv = predict(bundle, val)
    best = None
    for g,h,q in itertools.product(parse_grid(pc['gain_grid']),parse_grid(pc['harm_grid']),parse_grid(pc['quality_grid'])):
        pol={'min_pred_gain':g,'max_pred_harm':h,'quality_floor':q,'harm_penalty':float(pc.get('harm_penalty',.05)),
             'failure_dice_threshold':float(pc.get('failure_dice_threshold',.80)),'confidence':float(pc.get('confidence',.95))}
        _,s=evaluate(pv,pol,pol['failure_dice_threshold'],pol['confidence'])
        valid=s['auto_failure_upper']<=float(pc.get('target_auto_failure_upper',.15))
        score=(int(valid),s['coverage_region'],-s['harmful_action_rate'],s['auto_mean_dice'] or 0.)
        if best is None or score>best[0]: best=(score,pol,s)
    return best[1], best[2]


def diagnostics(bundle, val: pd.DataFrame):
    pv=predict(bundle,val)
    by=pv.sort_values(['unit_id','action_id']).drop_duplicates('unit_id')
    nz=pv[pv.action_id.astype(int)>0]
    out={'quality_mae':float(mean_absolute_error(by.base_dice,by.pred_base_dice)),
         'gain_mae':float(mean_absolute_error(nz.delta_utility,nz.pred_delta_utility)),
         'gain_spearman':float(spearmanr(nz.delta_utility,nz.pred_delta_utility).statistic)}
    out['harm_auroc']=float(roc_auc_score(nz.harmful,nz.pred_harm_prob)) if nz.harmful.nunique()>1 else None
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-csv',required=True); ap.add_argument('--val-csv',required=True); ap.add_argument('--test-csv',required=True)
    ap.add_argument('--config',default='configs/brats2021_farc.yaml'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--output-dir',required=True)
    args=ap.parse_args(); cfg=yaml.safe_load(Path(args.config).read_text()); pc=cfg['policy']
    tr0=pd.read_csv(args.train_csv); va0=pd.read_csv(args.val_csv); te0=pd.read_csv(args.test_csv)
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    variants=['full','no_error_map','no_uncertainty','no_region_identity','geometry_action_only']
    rows=[]
    for i,v in enumerate(variants):
        tr=drop_variant(tr0,v); va=drop_variant(va0,v); te=drop_variant(te0,v)
        b,_=fit(tr,args.seed+i*100)
        pol,cal=calibrate(b,va,pc)
        td=predict(b,te); _,s=evaluate(td,pol,float(pol['failure_dice_threshold']),float(pol['confidence']))
        diag=diagnostics(b,va)
        rows.append({'variant':v,**diag,**{f'cal_{k}':val for k,val in cal.items() if isinstance(val,(int,float,bool))},**s})
    df=pd.DataFrame(rows); df.to_csv(outdir/'feature_ablation.csv',index=False)
    (outdir/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(df.to_string(index=False))

if __name__=='__main__': main()

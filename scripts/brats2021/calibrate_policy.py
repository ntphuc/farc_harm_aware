from __future__ import annotations
import argparse,json,itertools
from pathlib import Path
import pandas as pd
from src.brats2021.controller import load,predict,evaluate

def parse(s):return [float(x) for x in str(s).split(',')]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--controller',required=True);ap.add_argument('--val-csv',required=True);ap.add_argument('--config',default='configs/brats2021_farc.yaml');ap.add_argument('--output-dir',required=True);args=ap.parse_args()
 import yaml;cfg=yaml.safe_load(Path(args.config).read_text());pc=cfg['policy'];b=load(args.controller);df=predict(b,pd.read_csv(args.val_csv));best=None;trials=[]
 for g,h,q in itertools.product(parse(pc['gain_grid']),parse(pc['harm_grid']),parse(pc['quality_grid'])):
  pol={'min_pred_gain':g,'max_pred_harm':h,'quality_floor':q,'harm_penalty':float(pc.get('harm_penalty',.05)),'failure_dice_threshold':float(pc.get('failure_dice_threshold',.80)),'confidence':float(pc.get('confidence',.95))}
  _,s=evaluate(df,pol,pol['failure_dice_threshold'],pol['confidence']);valid=s['auto_failure_upper']<=float(pc.get('target_auto_failure_upper',.15));score=(int(valid),s['coverage_region'],-s['harmful_action_rate'],s['auto_mean_dice'] or 0)
  trials.append({**pol,**s,'valid':valid,'score':list(score)})
  if best is None or score>best[0]:best=(score,pol,s)
 out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);pol=best[1];pol.update({'calibration_summary':best[2],'target_auto_failure_upper':float(pc.get('target_auto_failure_upper',.15))});(out/'policy.json').write_text(json.dumps(pol,indent=2));pd.DataFrame(trials).to_csv(out/'policy_grid.csv',index=False);print(json.dumps(pol,indent=2))
if __name__=='__main__':main()

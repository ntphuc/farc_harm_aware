from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--test-csv',required=True);ap.add_argument('--decisions-csv',required=True);ap.add_argument('--baseline-csv',required=True);ap.add_argument('--output-dir',required=True);args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(args.test_csv);dec=pd.read_csv(args.decisions_csv);base=pd.read_csv(args.baseline_csv)
 # H1: knowing error type does not imply its matching correction will help every patient.
 units=df.drop_duplicates('unit_id')[['unit_id','diagnostic_failure_type','region']]
 match=[]
 for _,u in units.iterrows():
  g=df[df.unit_id==u.unit_id];name=u.diagnostic_failure_type;r=g[g.action_name==name].iloc[0]
  match.append({'unit_id':u.unit_id,'region':u.region,'failure_type':name,'matched_action_gain':float(r.delta_utility),'matched_action_harmful':int(r.harmful),'matched_action_beneficial':int(r.beneficial),'oracle_gain':float(r.oracle_best_gain),'oracle_action':r.oracle_best_action})
 m=pd.DataFrame(match);m.to_csv(out/'failure_type_vs_action.csv',index=False)
 h1=m.groupby('failure_type').agg(n=('unit_id','size'),matched_benefit_rate=('matched_action_beneficial','mean'),matched_harm_rate=('matched_action_harmful','mean'),mean_matched_gain=('matched_action_gain','mean'),mean_oracle_gain=('oracle_gain','mean')).reset_index()
 h1.to_csv(out/'h1_failure_type_not_enough.csv',index=False)
 # H2/H3 from evaluated controller decisions.
 auto=dec[~dec.human_review].copy(); corrected=auto[auto.action_id>0]
 # Compare base vs final only for automated units.
 valid=auto.final_dice.notna()
 if valid.any():
  delta=auto.loc[valid,'final_dice'].values-auto.loc[valid,'base_dice'].values
  try: stat,p=wilcoxon(delta) if np.any(np.abs(delta)>1e-12) else (0.0,1.0)
  except ValueError: stat,p=np.nan,np.nan
 else:delta=np.array([]);stat=p=np.nan
 summary={
  'H1_interpretation':'If matched_harm_rate > 0 or matched_benefit_rate < 1, failure type alone is insufficient to guarantee a safe/useful correction.',
  'H1_overall_matched_harm_rate':float(m.matched_action_harmful.mean()),'H1_overall_matched_benefit_rate':float(m.matched_action_beneficial.mean()),
  'H2_auto_mean_delta_dice':float(delta.mean()) if len(delta) else None,'H2_wilcoxon_stat':None if np.isnan(stat) else float(stat),'H2_wilcoxon_p':None if np.isnan(p) else float(p),
  'H3_harm_rate_among_corrected':float(corrected.true_harmful.fillna(0).mean()) if len(corrected) else 0.0,
  'H3_review_rate_region':float(dec.human_review.mean()),'H3_review_rate_case':float(dec.groupby('case_id').human_review.max().mean()),
 }
 # Pull direct ablation deltas from baseline table.
 b=base.set_index('method')
 for metric in ['harmful_action_rate','auto_mean_dice','auto_failure_rate','review_rate_case']:
  if 'full_action_harm_referral' in b.index and metric in b.columns:
   summary[f'full_{metric}']=None if pd.isna(b.loc['full_action_harm_referral',metric]) else float(b.loc['full_action_harm_referral',metric])
   if 'gain_only_no_harm' in b.index: summary[f'gain_only_{metric}']=None if pd.isna(b.loc['gain_only_no_harm',metric]) else float(b.loc['gain_only_no_harm',metric])
 (out/'hypothesis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

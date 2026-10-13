from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from src.brats2021.controller import load,predict,evaluate,cp_upper
from src.brats2021.actions import ACTION_NAMES

def summarize(rows,fthr=.8,conf=.95):
 d=pd.DataFrame(rows);auto=d[~d.human_review];fail=int((auto.final_dice<fthr).sum()) if len(auto) else 0;case_review=d.groupby('case_id').human_review.max()
 return {'coverage_region':float(len(auto)/max(len(d),1)),'review_rate_region':float(d.human_review.mean()),'review_rate_case':float(case_review.mean()),'auto_failure_rate':float(fail/max(len(auto),1)) if len(auto) else 1.0,'auto_failure_upper':cp_upper(fail,len(auto),conf),'auto_mean_dice':float(auto.final_dice.mean()) if len(auto) else None,'harmful_action_rate':float(auto.harmful.fillna(0).mean()) if len(auto) else 0.,'correction_rate':float((auto.action_id>0).mean()) if len(auto) else 0.}
def row_for(g,aid,review=False):
 b=g[g.action_id.astype(int)==0].iloc[0]
 if review:return {'unit_id':b.unit_id,'case_id':b.case_id,'region':b.region,'human_review':True,'action_id':-1,'final_dice':np.nan,'harmful':np.nan}
 r=g[g.action_id.astype(int)==int(aid)].iloc[0];return {'unit_id':b.unit_id,'case_id':b.case_id,'region':b.region,'human_review':False,'action_id':int(aid),'final_dice':float(r.candidate_dice),'harmful':float(r.harmful)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--controller',required=True);ap.add_argument('--policy',required=True);ap.add_argument('--val-csv',required=True);ap.add_argument('--test-csv',required=True);ap.add_argument('--output-dir',required=True);args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 b=load(args.controller);pol=json.loads(Path(args.policy).read_text());val=predict(b,pd.read_csv(args.val_csv));test=predict(b,pd.read_csv(args.test_csv));fthr=float(pol.get('failure_dice_threshold',.8));conf=float(pol.get('confidence',.95));full_dec,full=evaluate(test,pol,fthr,conf);rows=[{'method':'full_action_harm_referral',**full}]
 keep=[row_for(g,0) for _,g in test.groupby('unit_id',sort=False)];rows.append({'method':'source_keep_all',**summarize(keep,fthr,conf)})
 # Quality-only referral
 qrows=[]
 for _,g in test.groupby('unit_id',sort=False):
  base=g[g.action_id.astype(int)==0].iloc[0];qrows.append(row_for(g,0,float(base.pred_base_dice)<float(pol['quality_floor'])))
 rows.append({'method':'quality_refer_only',**summarize(qrows,fthr,conf)})
 # Entropy referral matched to validation full review rate
 vd,_=evaluate(val,pol,fthr,conf);target=float(vd.human_review.mean());vc=val.sort_values(['unit_id','action_id']).drop_duplicates('unit_id');thr=float(vc.src_entropy_mean.quantile(max(0,min(1,1-target))))
 er=[]
 for _,g in test.groupby('unit_id',sort=False):
  base=g[g.action_id.astype(int)==0].iloc[0];er.append(row_for(g,0,float(base.src_entropy_mean)>=thr))
 rows.append({'method':'entropy_refer_matched',**summarize(er,fthr,conf),'entropy_threshold':thr})
 # Failure-type routing uses diagnostic failure type: oracle diagnostic only, upper bound for "know the error type" hypothesis.
 fr=[]
 for _,g in test.groupby('unit_id',sort=False):
  name=str(g.iloc[0].diagnostic_failure_type);aid=ACTION_NAMES.index(name);fr.append(row_for(g,aid))
 rows.append({'method':'diagnostic_failure_type_route_GT_only',**summarize(fr,fthr,conf)})
 # Reference-free error-risk heuristic: FN risk vs FP risk; otherwise boundary.
 rr=[]
 for _,g in test.groupby('unit_id',sort=False):
  base=g.iloc[0];fp=float(base.err_fp_mean);fn=float(base.err_fn_mean)
  if max(fp,fn)<.02:aid=0
  elif fn>1.2*fp:aid=1
  elif fp>1.2*fn:aid=2
  else:aid=3
  rr.append(row_for(g,aid))
 rows.append({'method':'error_map_failure_route',**summarize(rr,fthr,conf)})
 # Gain-only action controller.
 gp=dict(pol);gp['max_pred_harm']=1.;gp['harm_penalty']=0.;gd,gs=evaluate(test,gp,fthr,conf);rows.append({'method':'gain_only_no_harm',**gs})
 # Full controller without referral: forced keep original on referred units.
 nr=[]
 for _,r in full_dec.iterrows():
  g=test[test.unit_id.astype(str)==str(r.unit_id)];nr.append(row_for(g,0) if r.human_review else row_for(g,int(r.action_id)))
 rows.append({'method':'full_no_human_review',**summarize(nr,fthr,conf)})
 # Always actions.
 for aid,name in enumerate(ACTION_NAMES[1:],1):rows.append({'method':f'always_{name}',**summarize([row_for(g,aid) for _,g in test.groupby('unit_id',sort=False)],fthr,conf)})
 # Oracle action.
 orows=[]
 for _,g in test.groupby('unit_id',sort=False):orows.append(row_for(g,int(g.iloc[0].oracle_best_action_id)))
 rows.append({'method':'oracle_action_upper_bound',**summarize(orows,fthr,conf)})
 df=pd.DataFrame(rows);df.to_csv(out/'baseline_comparison.csv',index=False);(out/'notes.json').write_text(json.dumps({'GT_failure_type_route_is_diagnostic_only':True,'oracle_is_diagnostic_only':True},indent=2));print(df.to_string(index=False))
if __name__=='__main__':main()

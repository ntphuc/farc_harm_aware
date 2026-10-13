from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error,roc_auc_score
from scipy.stats import spearmanr
from src.brats2021.controller import fit,predict,save

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train-csv',required=True);ap.add_argument('--val-csv',required=True);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--output-dir',required=True);args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 tr=pd.read_csv(args.train_csv);va=pd.read_csv(args.val_csv);b,m=fit(tr,args.seed);save(b,out/'controller.joblib');pv=predict(b,va);pv.to_csv(out/'val_predictions.csv',index=False)
 by=pv.sort_values(['unit_id','action_id']).drop_duplicates('unit_id');nz=pv[pv.action_id.astype(int)>0];vm={'quality_mae':float(mean_absolute_error(by.base_dice,by.pred_base_dice)),'gain_mae':float(mean_absolute_error(nz.delta_utility,nz.pred_delta_utility)),'gain_spearman':float(spearmanr(nz.delta_utility,nz.pred_delta_utility).statistic),'n_units':int(by.unit_id.nunique())}
 vm['harm_auroc']=float(roc_auc_score(nz.harmful,nz.pred_harm_prob)) if nz.harmful.nunique()>1 else None
 s={'train':m,'val':vm,'hypothesis':'Action-specific gain and harm are predictable from deployment-visible signals.'};(out/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
if __name__=='__main__':main()

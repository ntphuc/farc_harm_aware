from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from src.brats2021.controller import load,predict,evaluate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--controller',required=True);ap.add_argument('--policy',required=True);ap.add_argument('--outcomes-csv',required=True);ap.add_argument('--output-dir',required=True);args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 b=load(args.controller);p=json.loads(Path(args.policy).read_text());df=predict(b,pd.read_csv(args.outcomes_csv));df.to_csv(out/'controller_predictions.csv',index=False);dec,s=evaluate(df,p,float(p.get('failure_dice_threshold',.8)),float(p.get('confidence',.95)));dec.to_csv(out/'decisions.csv',index=False);s['decision_counts']=dec.decision.value_counts().to_dict();s['region_summary']=dec.groupby('region').agg(n=('unit_id','size'),review_rate=('human_review','mean'),base_dice=('base_dice','mean'),final_dice=('final_dice','mean')).reset_index().to_dict(orient='records');(out/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
if __name__=='__main__':main()

from __future__ import annotations

import argparse,copy,datetime as dt,json,os,subprocess,sys,time
from pathlib import Path
import pandas as pd,yaml

ROOT=Path(__file__).resolve().parent

def stamp():return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def run(cmd,stage,expected=(),resume=True):
    ex=[Path(x) for x in expected]
    if resume and ex and all(p.exists() for p in ex):print(f'[{stamp()}] SKIP {stage}',flush=True);return
    print(f'[{stamp()}] ==== {stage} ====\n$ '+' '.join(map(str,cmd)),flush=True);env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','');env['PYTHONUNBUFFERED']='1';t=time.time();subprocess.run(cmd,cwd=ROOT,env=env,check=True);print(f'[{stamp()}] done {stage} seconds={time.time()-t:.1f}',flush=True)
def write_yaml(p,obj):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(yaml.safe_dump(obj,sort_keys=False),encoding='utf-8')
def write_json(p,obj):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(obj,indent=2),encoding='utf-8')

def one_run(label,arch,base,seed,cfg,args,resume):
    root=ROOT/cfg['output_root']/label/f'seed{seed}';root.mkdir(parents=True,exist_ok=True)
    rcfg=copy.deepcopy(cfg);rcfg['segmentor']['architecture']=arch;rcfg['segmentor']['base_channels']=int(base);cpath=root/'run_config.yaml';write_yaml(cpath,rcfg)
    manifest=str(ROOT/cfg['manifest']);data_root=str(args.data_root or cfg['data_root']);seg=root/'segmentor';err=root/'error_map';outcomes=root/'action_outcomes';ctrl=root/'controller';policy=root/'policy';test=root/'locked_test'
    run([sys.executable,'-u','train_brats2021_segmentor.py','--config',str(cpath),'--manifest',manifest,'--data-root',data_root,'--seed',str(seed),'--device',args.device,'--output-dir',str(seg)],f'SEGMENTOR {label} seed{seed}',[seg/'best.pt'],resume)
    run([sys.executable,'-u','train_brats2021_error_map.py','--config',str(cpath),'--segmentor-checkpoint',str(seg/'best.pt'),'--manifest',manifest,'--data-root',data_root,'--seed',str(seed),'--device',args.device,'--output-dir',str(err)],f'ERROR MAP {label} seed{seed}',[err/'best.pt'],resume)
    run([sys.executable,'-u','evaluate_brats2021_error_map.py','--config',str(cpath),'--segmentor-checkpoint',str(seg/'best.pt'),'--error-map-checkpoint',str(err/'best.pt'),'--manifest',manifest,'--data-root',data_root,'--split','source_val','--device',args.device,'--output-dir',str(err/'eval_source_val')],f'ERROR MAP SOURCE-VAL {label} seed{seed}',[err/'eval_source_val'/'summary.json'],resume)
    mc=int(cfg['suite']['mc_samples_main'] if label=='primary' else cfg['suite'].get('mc_samples_replication',0))
    for split in ('controller','val'):
        run([sys.executable,'-u','scripts/brats2021/build_action_outcomes.py','--config',str(cpath),'--segmentor-checkpoint',str(seg/'best.pt'),'--error-map-checkpoint',str(err/'best.pt'),'--manifest',manifest,'--data-root',data_root,'--split',split,'--device',args.device,'--mc-samples',str(mc),'--output',str(outcomes/f'{split}.csv')],f'OUTCOMES {split} {label} seed{seed}',[outcomes/f'{split}.csv'],resume)
    run([sys.executable,'-u','train_brats2021_controller.py','--train-csv',str(outcomes/'controller.csv'),'--val-csv',str(outcomes/'val.csv'),'--seed',str(seed),'--output-dir',str(ctrl)],f'CONTROLLER {label} seed{seed}',[ctrl/'controller.joblib'],resume)
    run([sys.executable,'-u','scripts/brats2021/calibrate_policy.py','--controller',str(ctrl/'controller.joblib'),'--val-csv',str(outcomes/'val.csv'),'--config',str(cpath),'--output-dir',str(policy)],f'POLICY FREEZE {label} seed{seed}',[policy/'policy.json'],resume)
    # Locked test is touched only after the policy file exists.
    if not args.skip_test:
        run([sys.executable,'-u','scripts/brats2021/build_action_outcomes.py','--config',str(cpath),'--segmentor-checkpoint',str(seg/'best.pt'),'--error-map-checkpoint',str(err/'best.pt'),'--manifest',manifest,'--data-root',data_root,'--split','test','--device',args.device,'--mc-samples',str(mc),'--output',str(outcomes/'test.csv')],f'LOCKED TEST OUTCOMES {label} seed{seed}',[outcomes/'test.csv'],resume)
        run([sys.executable,'-u','evaluate_brats2021_error_map.py','--config',str(cpath),'--segmentor-checkpoint',str(seg/'best.pt'),'--error-map-checkpoint',str(err/'best.pt'),'--manifest',manifest,'--data-root',data_root,'--split','test','--device',args.device,'--output-dir',str(err/'eval_test')],f'ERROR MAP LOCKED TEST {label} seed{seed}',[err/'eval_test'/'summary.json'],resume)
        run([sys.executable,'-u','evaluate_brats2021_policy.py','--controller',str(ctrl/'controller.joblib'),'--policy',str(policy/'policy.json'),'--outcomes-csv',str(outcomes/'test.csv'),'--output-dir',str(test)],f'FARC LOCKED TEST {label} seed{seed}',[test/'summary.json'],resume)
        run([sys.executable,'-u','evaluate_brats2021_end_to_end.py','--config',str(cpath),'--segmentor-checkpoint',str(seg/'best.pt'),'--error-map-checkpoint',str(err/'best.pt'),'--controller',str(ctrl/'controller.joblib'),'--policy',str(policy/'policy.json'),'--manifest',manifest,'--data-root',data_root,'--split','test','--device',args.device,'--mc-samples',str(mc),'--output-dir',str(test/'end_to_end')],f'END-TO-END NESTED LOCKED TEST {label} seed{seed}',[test/'end_to_end'/'summary.json'],resume)
        run([sys.executable,'-u','scripts/brats2021/evaluate_baselines.py','--controller',str(ctrl/'controller.joblib'),'--policy',str(policy/'policy.json'),'--val-csv',str(outcomes/'val.csv'),'--test-csv',str(outcomes/'test.csv'),'--output-dir',str(test/'baselines')],f'BASELINES {label} seed{seed}',[test/'baselines'/'baseline_comparison.csv'],resume)
        run([sys.executable,'-u','scripts/brats2021/evaluate_feature_ablations.py','--train-csv',str(outcomes/'controller.csv'),'--val-csv',str(outcomes/'val.csv'),'--test-csv',str(outcomes/'test.csv'),'--config',str(cpath),'--seed',str(seed),'--output-dir',str(test/'feature_ablations')],f'FEATURE ABLATIONS {label} seed{seed}',[test/'feature_ablations'/'feature_ablation.csv'],resume)
        run([sys.executable,'-u','scripts/brats2021/analyze_hypotheses.py','--test-csv',str(outcomes/'test.csv'),'--decisions-csv',str(test/'decisions.csv'),'--baseline-csv',str(test/'baselines'/'baseline_comparison.csv'),'--output-dir',str(test/'hypotheses')],f'HYPOTHESES {label} seed{seed}',[test/'hypotheses'/'hypothesis_summary.json'],resume)
    return {'label':label,'architecture':arch,'seed':seed,'root':root}

def summarize(records,cfg):
    rows=[]
    for r in records:
        p=r['root']/'locked_test'/'summary.json';c=r['root']/'controller'/'summary.json';h=r['root']/'locked_test'/'hypotheses'/'hypothesis_summary.json';e=r['root']/'error_map'/'eval_test'/'summary.json'
        if not p.exists():continue
        ps=json.loads(p.read_text());cs=json.loads(c.read_text()) if c.exists() else {};hs=json.loads(h.read_text()) if h.exists() else {};es=json.loads(e.read_text()) if e.exists() else {}
        rows.append({'label':r['label'],'architecture':r['architecture'],'seed':r['seed'],'coverage_region':ps.get('coverage_region'),'review_rate_case':ps.get('review_rate_case'),'auto_mean_dice':ps.get('auto_mean_dice'),'auto_failure_rate':ps.get('auto_failure_rate'),'harmful_action_rate':ps.get('harmful_action_rate'),'controller_gain_mae':(cs.get('val') or {}).get('gain_mae'),'controller_gain_spearman':(cs.get('val') or {}).get('gain_spearman'),'controller_harm_auroc':(cs.get('val') or {}).get('harm_auroc'),'error_auprc':es.get('mean_error_auprc'),'entropy_auprc':es.get('mean_entropy_auprc'),'matched_failure_action_harm_rate':hs.get('H1_overall_matched_harm_rate')})
    out=ROOT/cfg['output_root']/'analysis';out.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows);df.to_csv(out/'by_seed.csv',index=False)
    if len(df):df.groupby(['label','architecture']).mean(numeric_only=True).to_csv(out/'summary_mean.csv')
    write_json(out/'summary.json',{'n_runs':len(rows),'central_hypothesis':'Knowing where/how a segmentation is wrong is not sufficient for safe correction; action-specific benefit and harm should be predicted before choosing automatic correction, otherwise the case should be referred.'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/brats2021_farc.yaml');ap.add_argument('--data-root',default=None);ap.add_argument('--device',default='auto');ap.add_argument('--no-resume',action='store_true');ap.add_argument('--skip-test',action='store_true');args=ap.parse_args();cfg=yaml.safe_load((ROOT/args.config).read_text());resume=not args.no_resume;data_root=args.data_root or cfg['data_root']
    manifest=ROOT/cfg['manifest']
    run([sys.executable,'-u','scripts/brats2021/prepare_brats2021.py','--data-root',data_root,'--output',str(manifest),'--split-seed',str(cfg.get('split_seed',2021)),'--source-val-fraction',str(cfg.get('source_val_fraction',.08)),'--error-train-fraction',str(cfg.get('error_train_fraction',.12)),'--controller-fraction',str(cfg.get('controller_fraction',.12)),'--policy-val-fraction',str(cfg.get('policy_val_fraction',.08)),'--test-fraction',str(cfg.get('test_fraction',.10)),'--label-scheme',str(cfg.get('label_scheme','auto')),'--label-scan-max',str(cfg.get('label_scan_max',128))],'PREPARE BRATS DATA',[manifest],False)
    rec=[]
    for label in ('primary','replication'):
        x=cfg['suite'].get(label)
        if not x:continue
        for seed in x.get('seeds',[]):rec.append(one_run(label,x['architecture'],x.get('base_channels',24),int(seed),cfg,args,resume))
    summarize(rec,cfg);write_json(ROOT/cfg['output_root']/'SUITE_COMPLETE.json',{'complete':True,'finished_at':stamp(),'runs':[{'label':r['label'],'architecture':r['architecture'],'seed':r['seed']} for r in rec]});print(f'[{stamp()}] BRATS2021 FARC-v2 COMPLETE -> {cfg["output_root"]}',flush=True)
if __name__=='__main__':main()

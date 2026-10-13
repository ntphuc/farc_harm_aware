from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from torch.utils.data import DataLoader
from src.brats2021.data import BraTS2021VolumeDataset,REGIONS
from src.brats2021.model import build_brats_segmentor
from src.brats2021.error_map import ErrorMap3D,error_targets,entropy_map

def dev(n):return torch.device('cuda' if n=='auto' and torch.cuda.is_available() else ('cpu' if n=='auto' else n))
def score(y,s):
 y=np.asarray(y).astype(np.uint8).ravel();s=np.asarray(s).astype(np.float32).ravel()
 if len(np.unique(y))<2:return {'auroc':None,'auprc':float(y.mean())}
 return {'auroc':float(roc_auc_score(y,s)),'auprc':float(average_precision_score(y,s))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/brats2021_farc.yaml');ap.add_argument('--segmentor-checkpoint',required=True);ap.add_argument('--error-map-checkpoint',required=True);ap.add_argument('--manifest',default='data/brats2021_manifest.csv');ap.add_argument('--data-root',default='/datastore/phucngt/datasets/BraTS2021');ap.add_argument('--split',default='test');ap.add_argument('--device',default='auto');ap.add_argument('--output-dir',required=True);ap.add_argument('--max-voxels-per-case',type=int,default=200000);args=ap.parse_args()
 import yaml;cfg=yaml.safe_load(Path(args.config).read_text());scfg=cfg['segmentor'];devc=dev(args.device);sck=torch.load(args.segmentor_checkpoint,map_location='cpu',weights_only=False);shape=tuple(sck.get('volume_shape',scfg['volume_shape']));seg=build_brats_segmentor(scfg.get('architecture','unet3d'),int(scfg.get('base_channels',24)),float(scfg.get('dropout',.15))).to(devc);seg.load_state_dict(sck['model']);seg.eval();eck=torch.load(args.error_map_checkpoint,map_location='cpu',weights_only=False);err=ErrorMap3D(int(cfg['error_map'].get('width',8)),float(cfg['error_map'].get('dropout',.1))).to(devc);err.load_state_dict(eck['model']);err.eval();ds=BraTS2021VolumeDataset(args.manifest,args.split,args.data_root,shape,train=False);loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=int(scfg.get('num_workers',2)))
 rows=[];rng=np.random.default_rng(2021)
 with torch.no_grad():
  for im,gt,meta in loader:
   im,gt=im.to(devc),gt.to(devc);z=seg(im);y=error_targets(z,gt).cpu().numpy()[0];risk=torch.sigmoid(err(im,z)).cpu().numpy()[0];p=torch.sigmoid(z);ent=entropy_map(p).cpu().numpy()[0];cid=str(meta['case_id'][0])
   for r in range(3):
    for kind,k in [('FP',2*r),('FN',2*r+1)]:
     yy=y[k].ravel();rr=risk[k].ravel();ee=ent[r].ravel();n=min(len(yy),args.max_voxels_per_case);idx=rng.choice(len(yy),n,replace=False) if n<len(yy) else np.arange(len(yy));a=score(yy[idx],rr[idx]);b=score(yy[idx],ee[idx]);rows.append({'case_id':cid,'region':REGIONS[r],'error_type':kind,'error_auroc':a['auroc'],'error_auprc':a['auprc'],'entropy_auroc':b['auroc'],'entropy_auprc':b['auprc']})
 out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows);df.to_csv(out/'per_case.csv',index=False);summary=df.groupby(['region','error_type']).mean(numeric_only=True).reset_index();summary.to_csv(out/'summary.csv',index=False);(out/'summary.json').write_text(json.dumps({'split':args.split,'n_cases':int(df.case_id.nunique()),'mean_error_auprc':float(df.error_auprc.mean()),'mean_entropy_auprc':float(df.entropy_auprc.mean())},indent=2));print(summary.to_string(index=False))
if __name__=='__main__':main()

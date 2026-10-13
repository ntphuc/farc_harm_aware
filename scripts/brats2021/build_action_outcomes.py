from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.ndimage import label
from torch.utils.data import DataLoader

from src.brats2021.actions import ACTION_NAMES,REGION_NAMES,all_candidates
from src.brats2021.data import BraTS2021VolumeDataset
from src.brats2021.error_map import ErrorMap3D
from src.brats2021.features import source_features,action_features
from src.brats2021.metrics import case_metrics
from src.brats2021.model import build_brats_segmentor,enable_mc_dropout


def dev(n):return torch.device('cuda' if n=='auto' and torch.cuda.is_available() else ('cpu' if n=='auto' else n))
def ncomp(m):return int(label(np.asarray(m).astype(bool))[1])

def failure_type(src,gt):
    s=np.asarray(src).astype(bool); g=np.asarray(gt).astype(bool); fp=int((s&~g).sum()); fn=int((~s&g).sum())
    if fp+fn==0:return 'no_correction'
    if fn>1.25*max(fp,1):return 'recover_fn'
    if fp>1.25*max(fn,1):
        return 'tail_cleanup' if ncomp(s)>ncomp(g) else 'remove_fp'
    return 'boundary_align'


def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/brats2021_farc.yaml'); ap.add_argument('--segmentor-checkpoint',required=True); ap.add_argument('--error-map-checkpoint',default=''); ap.add_argument('--manifest',default='data/brats2021_manifest.csv'); ap.add_argument('--data-root',default='/datastore/phucngt/datasets/BraTS2021'); ap.add_argument('--split',required=True,choices=['train','source_val','error_train','controller','val','test']); ap.add_argument('--device',default='auto'); ap.add_argument('--output',required=True); ap.add_argument('--mc-samples',type=int,default=0); args=ap.parse_args()
 import yaml; cfg=yaml.safe_load(Path(args.config).read_text()); scfg=cfg['segmentor']; acfg=cfg['actions']; devc=dev(args.device)
 ck=torch.load(args.segmentor_checkpoint,map_location='cpu',weights_only=False); shape=tuple(ck.get('volume_shape',scfg['volume_shape'])); seg=build_brats_segmentor(scfg.get('architecture','unet3d'),int(scfg.get('base_channels',24)),float(scfg.get('dropout',.15))).to(devc); seg.load_state_dict(ck['model']); seg.eval()
 err=None
 if args.error_map_checkpoint:
  eck=torch.load(args.error_map_checkpoint,map_location='cpu',weights_only=False); err=ErrorMap3D(int(cfg['error_map'].get('width',8)),float(cfg['error_map'].get('dropout',.1))).to(devc); err.load_state_dict(eck['model']); err.eval()
 ds=BraTS2021VolumeDataset(args.manifest,args.split,args.data_root,shape,train=False); loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=int(scfg.get('num_workers',2)),pin_memory=True)
 rows=[]
 with torch.no_grad():
  for image,gt,meta in loader:
   image=image.to(devc); logits=seg(image); prob=torch.sigmoid(logits); mcvar=torch.zeros_like(prob)
   if args.mc_samples>1:
    enable_mc_dropout(seg); ss=[torch.sigmoid(seg(image)) for _ in range(args.mc_samples)]; st=torch.stack(ss); prob=st.mean(0); mcvar=st.var(0,unbiased=False); seg.eval()
   risks=torch.sigmoid(err(image,logits)).cpu().numpy()[0] if err is not None else None
   p=prob.cpu().numpy()[0]; v=mcvar.cpu().numpy()[0]; g=gt.numpy()[0]; cid=str(meta['case_id'][0])
   sp_meta=meta.get('effective_spacing_mm',(1.0,1.0,1.0))
   if isinstance(sp_meta,(list,tuple)) and len(sp_meta)==3:
    spacing=tuple(float(x[0].item() if hasattr(x[0],'item') else x[0]) if hasattr(x,'__len__') else float(x) for x in sp_meta)
   else:
    spacing=(1.0,1.0,1.0)
   for ridx,rname in enumerate(REGION_NAMES):
    src=p[ridx]; truth=g[ridx]>=.5; base=case_metrics(src>=.5,truth,float(acfg.get('hd95_scale',30)),spacing)
    fp_r=risks[2*ridx] if risks is not None else None; fn_r=risks[2*ridx+1] if risks is not None else None
    shared=source_features(src,v[ridx],fp_r,fn_r,rname); ftype=failure_type(src>=.5,truth); unit=f'{cid}::{rname}'; case_rows=[]
    for aid,aname,cand in all_candidates(src,.5,int(acfg.get('min_component_voxels',64))):
      cm=case_metrics(cand,truth,float(acfg.get('hd95_scale',30)),spacing); du=float(cm['utility']-base['utility']); dd=float(cm['dice']-base['dice'])
      row={'case_id':cid,'unit_id':unit,'region':rname,'split':args.split,'action_id':aid,'action_name':aname,**shared,**action_features(src,cand,aid),
           'base_dice':base['dice'],'base_hd95':base['hd95'],'base_utility':base['utility'],'base_fp':base['fp'],'base_fn':base['fn'],
           'candidate_dice':cm['dice'],'candidate_hd95':cm['hd95'],'candidate_utility':cm['utility'],'delta_dice':dd,'delta_utility':du,
           'harmful':int(du < -float(acfg.get('harm_margin',.002))),'beneficial':int(du > float(acfg.get('benefit_margin',.002))),
           'diagnostic_failure_type':ftype,'spacing_x_mm':spacing[0],'spacing_y_mm':spacing[1],'spacing_z_mm':spacing[2]}
      case_rows.append(row)
    best=max(case_rows,key=lambda x:(x['candidate_utility'],x['candidate_dice']))
    for r in case_rows:r.update({'oracle_best_action':best['action_name'],'oracle_best_action_id':best['action_id'],'oracle_best_gain':best['delta_utility']})
    rows.extend(case_rows)
 df=pd.DataFrame(rows); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
 summary={'split':args.split,'n_cases':int(df.case_id.nunique()),'n_units':int(df.unit_id.nunique()),'rows':len(df),'oracle_action_counts':df.drop_duplicates('unit_id').oracle_best_action.value_counts().to_dict(),'failure_type_counts':df.drop_duplicates('unit_id').diagnostic_failure_type.value_counts().to_dict(),'note':'GT is used only to create training/evaluation outcome targets and diagnostic failure types. Controller features are reference-free.'}
 out.with_suffix('.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

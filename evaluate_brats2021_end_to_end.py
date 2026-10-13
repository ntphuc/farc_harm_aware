from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.brats2021.actions import REGION_NAMES, all_candidates
from src.brats2021.controller import load, predict, select
from src.brats2021.data import BraTS2021VolumeDataset, enforce_nested_regions
from src.brats2021.error_map import ErrorMap3D
from src.brats2021.features import source_features, action_features
from src.brats2021.metrics import case_metrics
from src.brats2021.model import build_brats_segmentor, enable_mc_dropout


def device_from(name):
    return torch.device('cuda' if name == 'auto' and torch.cuda.is_available() else ('cpu' if name == 'auto' else name))


def spacing_from_meta(meta):
    xs=meta.get('effective_spacing_mm',(1.,1.,1.))
    out=[]
    for x in xs:
        if hasattr(x,'__len__'):
            y=x[0]
            out.append(float(y.item() if hasattr(y,'item') else y))
        else:
            out.append(float(x))
    return tuple(out)


def main():
    ap=argparse.ArgumentParser(description='True inference-style FARC evaluation with final BraTS nesting enforced.')
    ap.add_argument('--config',default='configs/brats2021_farc.yaml')
    ap.add_argument('--segmentor-checkpoint',required=True);ap.add_argument('--error-map-checkpoint',required=True)
    ap.add_argument('--controller',required=True);ap.add_argument('--policy',required=True)
    ap.add_argument('--manifest',default='data/brats2021_manifest.csv');ap.add_argument('--data-root',default='/datastore/phucngt/datasets/BraTS2021')
    ap.add_argument('--split',default='test');ap.add_argument('--device',default='auto');ap.add_argument('--mc-samples',type=int,default=5);ap.add_argument('--output-dir',required=True)
    args=ap.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text());scfg=cfg['segmentor'];acfg=cfg['actions'];dev=device_from(args.device)
    sck=torch.load(args.segmentor_checkpoint,map_location='cpu',weights_only=False);shape=tuple(sck.get('volume_shape',scfg['volume_shape']))
    seg=build_brats_segmentor(scfg.get('architecture','unet3d'),int(scfg.get('base_channels',24)),float(scfg.get('dropout',.15))).to(dev);seg.load_state_dict(sck['model']);seg.eval()
    eck=torch.load(args.error_map_checkpoint,map_location='cpu',weights_only=False);err=ErrorMap3D(int(cfg['error_map'].get('width',8)),float(cfg['error_map'].get('dropout',.1))).to(dev);err.load_state_dict(eck['model']);err.eval()
    bundle=load(args.controller);policy=json.loads(Path(args.policy).read_text())
    ds=BraTS2021VolumeDataset(args.manifest,args.split,args.data_root,shape,train=False);loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=int(scfg.get('num_workers',2)))
    region_rows=[];case_rows=[]
    with torch.no_grad():
        for image,gt,meta in loader:
            image=image.to(dev);z=seg(image);prob=torch.sigmoid(z);mcvar=torch.zeros_like(prob)
            if args.mc_samples>1:
                enable_mc_dropout(seg);ss=[torch.sigmoid(seg(image)) for _ in range(args.mc_samples)];st=torch.stack(ss);prob=st.mean(0);mcvar=st.var(0,unbiased=False);seg.eval()
            risk=torch.sigmoid(err(image,z)).cpu().numpy()[0]
            p=prob.cpu().numpy()[0];v=mcvar.cpu().numpy()[0];truth=gt.numpy()[0]>=.5;cid=str(meta['case_id'][0]);spacing=spacing_from_meta(meta)
            base_masks=enforce_nested_regions((p>=.5).astype(np.float32)).astype(bool)
            selected=[];decisions=[]
            for ridx,rname in enumerate(REGION_NAMES):
                fp=risk[2*ridx];fn=risk[2*ridx+1]
                rows=[];cand_by_id={}
                for aid,aname,cand in all_candidates(p[ridx],.5,int(acfg.get('min_component_voxels',64))):
                    cand_by_id[int(aid)]=np.asarray(cand).astype(bool)
                    rows.append({'case_id':cid,'unit_id':f'{cid}::{rname}','region':rname,'action_id':int(aid),'action_name':aname,
                                 **source_features(p[ridx],v[ridx],fp,fn,rname),**action_features(p[ridx],cand,aid)})
                pred=predict(bundle,pd.DataFrame(rows));d=select(pred,policy);decisions.append(d)
                selected.append(base_masks[ridx] if d['human_review'] else cand_by_id[int(d['action_id'])])
            review=any(d['human_review'] for d in decisions)
            if review:
                final_masks=None
            else:
                final_masks=enforce_nested_regions(np.stack(selected,axis=0).astype(np.float32)).astype(bool)
            base_ds=[];final_ds=[];harms=[]
            for ridx,rname in enumerate(REGION_NAMES):
                bm=case_metrics(base_masks[ridx],truth[ridx],float(acfg.get('hd95_scale',30)),spacing)
                d=decisions[ridx]
                row={'case_id':cid,'region':rname,'human_review':bool(review),'region_requested_review':bool(d['human_review']),'decision':d['decision'],'action_id':int(d['action_id']),
                     'pred_base_dice':float(d['pred_base_dice']),'pred_gain':float(d['pred_gain']),'pred_harm':float(d['pred_harm']),'base_dice':bm['dice'],'base_hd95':bm['hd95']}
                base_ds.append(bm['dice'])
                if final_masks is not None:
                    fm=case_metrics(final_masks[ridx],truth[ridx],float(acfg.get('hd95_scale',30)),spacing)
                    delta=fm['utility']-bm['utility'];row.update({'final_dice':fm['dice'],'final_hd95':fm['hd95'],'delta_dice':fm['dice']-bm['dice'],'delta_utility':delta,'harmful_after_nesting':int(delta < -float(acfg.get('harm_margin',.002)))})
                    final_ds.append(fm['dice']);harms.append(row['harmful_after_nesting'])
                else:
                    row.update({'final_dice':np.nan,'final_hd95':np.nan,'delta_dice':np.nan,'delta_utility':np.nan,'harmful_after_nesting':np.nan})
                region_rows.append(row)
            case_rows.append({'case_id':cid,'human_review':bool(review),'base_mean_dice':float(np.mean(base_ds)),'final_mean_dice':float(np.mean(final_ds)) if final_ds else np.nan,
                              'any_harm_after_nesting':float(max(harms)) if harms else np.nan,'actions':'|'.join(d['decision'] for d in decisions)})
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);rdf=pd.DataFrame(region_rows);cdf=pd.DataFrame(case_rows);rdf.to_csv(out/'per_region.csv',index=False);cdf.to_csv(out/'per_case.csv',index=False)
    auto=cdf[~cdf.human_review];r_auto=rdf[~rdf.human_review]
    summary={'n_cases':int(len(cdf)),'review_rate_case':float(cdf.human_review.mean()),'auto_coverage_case':float((~cdf.human_review).mean()),
             'source_mean_dice_all_regions':float(rdf.base_dice.mean()),'auto_final_mean_dice_all_regions':float(r_auto.final_dice.mean()) if len(r_auto) else None,
             'auto_mean_delta_dice_all_regions':float(r_auto.delta_dice.mean()) if len(r_auto) else None,
             'harmful_region_rate_after_nesting':float(r_auto.harmful_after_nesting.mean()) if len(r_auto) else None,
             'harmful_case_rate_after_nesting':float(auto.any_harm_after_nesting.mean()) if len(auto) else None,
             'region_breakdown':r_auto.groupby('region').agg(n=('case_id','size'),base_dice=('base_dice','mean'),final_dice=('final_dice','mean'),harm_rate=('harmful_after_nesting','mean')).reset_index().to_dict(orient='records') if len(r_auto) else []}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

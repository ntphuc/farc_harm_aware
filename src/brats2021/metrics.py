from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_erosion,distance_transform_edt,generate_binary_structure


def dice(mask,gt):
    a=np.asarray(mask).astype(bool); b=np.asarray(gt).astype(bool); den=a.sum()+b.sum()
    if den==0:return 1.0
    return float(2*(a&b).sum()/den)

def precision(mask,gt):
    a=np.asarray(mask).astype(bool); b=np.asarray(gt).astype(bool); d=a.sum()
    return float((a&b).sum()/d) if d else (1.0 if not b.any() else 0.0)

def recall(mask,gt):
    a=np.asarray(mask).astype(bool); b=np.asarray(gt).astype(bool); d=b.sum()
    return float((a&b).sum()/d) if d else 1.0

def hd95(mask,gt,empty_penalty=128.0,spacing=(1.0,1.0,1.0)):
    a=np.asarray(mask).astype(bool); b=np.asarray(gt).astype(bool)
    if not a.any() and not b.any(): return 0.0
    if not a.any() or not b.any(): return float(empty_penalty)
    s=generate_binary_structure(3,1)
    ba=a ^ binary_erosion(a,structure=s,border_value=0); bb=b ^ binary_erosion(b,structure=s,border_value=0)
    sp=tuple(float(v) for v in spacing)
    da=distance_transform_edt(~ba,sampling=sp); db=distance_transform_edt(~bb,sampling=sp)
    d1=db[ba]; d2=da[bb]
    if not len(d1) or not len(d2):return float(empty_penalty)
    return float(np.percentile(np.concatenate([d1,d2]),95))

def case_metrics(mask,gt,hd95_scale=30.0,spacing=(1.0,1.0,1.0)):
    a=np.asarray(mask).astype(bool); b=np.asarray(gt).astype(bool)
    fp=int((a&~b).sum()); fn=int((~a&b).sum()); d=dice(a,b); h=hd95(a,b,spacing=spacing)
    util=float(d-0.05*min(h/float(hd95_scale),2.0))
    return {'dice':d,'hd95':h,'precision':precision(a,b),'recall':recall(a,b),'fp':fp,'fn':fn,'pred_voxels':int(a.sum()),'gt_voxels':int(b.sum()),'utility':util}

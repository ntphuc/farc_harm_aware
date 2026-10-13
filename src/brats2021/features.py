from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation,binary_erosion,generate_binary_structure,label
from .actions import ACTION_NAMES, REGION_NAMES


def entropy(p):
    x=np.clip(np.asarray(p,dtype=np.float32),1e-6,1-1e-6); return -(x*np.log(x)+(1-x)*np.log(1-x))/np.log(2.0)

def boundary(m):
    x=np.asarray(m).astype(bool); return x ^ binary_erosion(x,structure=generate_binary_structure(3,1),border_value=0)

def components(m):
    lab,n=label(np.asarray(m).astype(bool))
    if n==0:return 0,0.0
    c=np.bincount(lab.ravel())[1:]; return int(n),float(c.max()/max(c.sum(),1))

def source_features(prob,mc_var=None,fp_risk=None,fn_risk=None,region='WT'):
    p=np.asarray(prob,dtype=np.float32); m=p>=.5; e=entropy(p); b=boundary(m); band=binary_dilation(b,structure=generate_binary_structure(3,1)) if b.any() else b
    n,largest=components(m); rel=binary_dilation(m,iterations=2) if m.any() else np.ones_like(m,bool)
    out={
      'src_fg_fraction':float(m.mean()),'src_prob_mean':float(p.mean()),'src_prob_std':float(p.std()),
      'src_entropy_mean':float(e[rel].mean()),'src_entropy_boundary':float(e[band].mean()) if band.any() else float(e.mean()),
      'src_low_conf_fraction':float((np.abs(p[rel]-.5)<.15).mean()),'src_component_count':float(n),'src_largest_component_ratio':largest,
      'src_boundary_fraction':float(b.mean()),'src_mc_var_mean':float(np.asarray(mc_var).mean()) if mc_var is not None else 0.0,
      'err_fp_mean':float(np.asarray(fp_risk).mean()) if fp_risk is not None else 0.0,'err_fn_mean':float(np.asarray(fn_risk).mean()) if fn_risk is not None else 0.0,
      'err_fp_p95':float(np.quantile(fp_risk,.95)) if fp_risk is not None else 0.0,'err_fn_p95':float(np.quantile(fn_risk,.95)) if fn_risk is not None else 0.0,
    }
    for r in REGION_NAMES:out[f'region_{r}']=1.0 if region==r else 0.0
    return out

def action_features(prob,candidate,action_id):
    p=np.asarray(prob,dtype=np.float32); src=p>=.5; c=np.asarray(candidate).astype(bool); add=c&~src; rem=src&~c; ch=src^c
    out={'action_id_feature':float(action_id),'act_add_fraction':float(add.mean()),'act_remove_fraction':float(rem.mean()),'act_changed_fraction':float(ch.mean()),'act_fg_delta':float(c.mean()-src.mean())}
    for i,n in enumerate(ACTION_NAMES):out[f'act_is_{n}']=1.0 if i==int(action_id) else 0.0
    return out

def feature_columns(df_columns):
    cols=[c for c in df_columns if c.startswith(('src_','err_','region_','act_'))]
    return sorted(cols)

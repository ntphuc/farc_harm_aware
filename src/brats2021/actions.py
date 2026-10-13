from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening, generate_binary_structure, label

ACTION_NAMES = ("no_correction","recover_fn","remove_fp","boundary_align","tail_cleanup")
REGION_NAMES = ("WT","TC","ET")


def _struct(radius:int=1):
    s=generate_binary_structure(3,1)
    if radius<=1:return s
    out=s
    for _ in range(radius-1): out=binary_dilation(out,structure=s)
    return out


def remove_small(mask:np.ndarray,min_voxels:int=64)->np.ndarray:
    m=np.asarray(mask).astype(bool); lab,n=label(m)
    if n==0:return m
    counts=np.bincount(lab.ravel())
    keep=np.zeros_like(m,bool)
    for i in range(1,n+1):
        if counts[i]>=int(min_voxels): keep[lab==i]=True
    return keep


def action_candidate(prob:np.ndarray,action_id:int,threshold:float=.5,min_component:int=64)->np.ndarray:
    """Model-agnostic 3D action bank for one BraTS region.

    Each action is intentionally simple and interpretable. The paper hypothesis is not that
    these operators are novel, but that the controller can predict which action helps/hurts.
    """
    p=np.asarray(prob,dtype=np.float32); base=p>=float(threshold); aid=int(action_id)
    if aid==0:return base
    if aid==1: # recover_fn: conservative uncertain expansion around current foreground
        near=binary_dilation(base,structure=_struct(1))
        return base | (near & (p>=0.30))
    if aid==2: # remove_fp: require stronger confidence + keep nontrivial components
        strong=(p>=0.62) & base
        return remove_small(strong,max(8,int(min_component//2)))
    if aid==3: # boundary alignment: close pinholes then open small spikes
        return binary_opening(binary_closing(base,structure=_struct(1)),structure=_struct(1))
    if aid==4: # tail cleanup: remove small disconnected components only
        return remove_small(base,int(min_component))
    raise ValueError(action_id)


def all_candidates(prob:np.ndarray,threshold:float=.5,min_component:int=64):
    return [(i,ACTION_NAMES[i],action_candidate(prob,i,threshold,min_component)) for i in range(len(ACTION_NAMES))]

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_map(prob: torch.Tensor) -> torch.Tensor:
    p=prob.float().clamp(1e-6,1-1e-6)
    return (-(p*torch.log(p)+(1-p)*torch.log(1-p))/0.6931471805599453).clamp(0,1)


class ErrorMap3D(nn.Module):
    """Reference-free voxel error predictor for WT/TC/ET.
    Input: 4 MRI + 3 source probabilities + 3 entropies = 10 channels.
    Output: [FP_WT,FN_WT, FP_TC,FN_TC, FP_ET,FN_ET].
    """
    def __init__(self,width:int=8,dropout:float=.1):
        super().__init__(); w=int(width)
        self.net=nn.Sequential(
            nn.Conv3d(10,w,3,padding=1,bias=False),nn.InstanceNorm3d(w,affine=True),nn.LeakyReLU(.01,inplace=True),
            nn.Conv3d(w,w,3,padding=2,dilation=2,bias=False),nn.InstanceNorm3d(w,affine=True),nn.LeakyReLU(.01,inplace=True),
            nn.Dropout3d(float(dropout)),
            nn.Conv3d(w,w,3,padding=4,dilation=4,bias=False),nn.InstanceNorm3d(w,affine=True),nn.LeakyReLU(.01,inplace=True),
            nn.Conv3d(w,6,1))
    def forward(self,image:torch.Tensor,source_logits:torch.Tensor):
        p=torch.sigmoid(source_logits.float()); e=entropy_map(p)
        return self.net(torch.cat([image.float(),p,e],dim=1))


def error_targets(source_logits:torch.Tensor,gt:torch.Tensor,threshold:float=.5):
    pred=torch.sigmoid(source_logits.float())>=float(threshold); truth=gt>=.5
    outs=[]
    for c in range(3):
        outs.extend([(pred[:,c:c+1]&~truth[:,c:c+1]).float(),((~pred[:,c:c+1])&truth[:,c:c+1]).float()])
    return torch.cat(outs,dim=1)


def error_loss(logits,target,max_pos_weight:float=50.0):
    t=target.float(); pos=t.sum(dim=(0,2,3,4)); total=float(t.shape[0]*t.shape[2]*t.shape[3]*t.shape[4]); neg=total-pos
    pw=(neg/pos.clamp_min(1.0)).clamp(1.0,float(max_pos_weight))
    bce=F.binary_cross_entropy_with_logits(logits,t,pos_weight=pw[None,:,None,None,None])
    p=torch.sigmoid(logits); inter=(p*t).sum(dim=(0,2,3,4)); den=p.sum(dim=(0,2,3,4))+t.sum(dim=(0,2,3,4)); dice=(2*inter+1)/(den+1)
    return bce+(1-dice.mean())

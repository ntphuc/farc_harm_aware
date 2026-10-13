from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct3d(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        layers = [
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(float(dropout)))
        super().__init__(*layers)


class Down3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, 2, stride=2)
        self.block = ConvNormAct3d(out_ch, out_ch, dropout)
    def forward(self, x):
        return self.block(self.down(x))


class Up3d(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.block = ConvNormAct3d(out_ch + skip_ch, out_ch, dropout)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class BraTSUNet3D(nn.Module):
    """Compact 3D U-Net for WT/TC/ET multi-label prediction.

    Dropout remains in the bottleneck/decoder so MC-dropout can be used as an uncertainty baseline.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, base: int = 24, dropout: float = 0.15):
        super().__init__()
        b = int(base)
        self.enc0 = ConvNormAct3d(in_channels, b)
        self.enc1 = Down3d(b, b * 2)
        self.enc2 = Down3d(b * 2, b * 4, dropout * 0.5)
        self.enc3 = Down3d(b * 4, b * 8, dropout)
        self.bottleneck = Down3d(b * 8, b * 12, dropout)
        self.up3 = Up3d(b * 12, b * 8, b * 8, dropout)
        self.up2 = Up3d(b * 8, b * 4, b * 4, dropout * 0.5)
        self.up1 = Up3d(b * 4, b * 2, b * 2)
        self.up0 = Up3d(b * 2, b, b)
        self.head = nn.Conv3d(b, out_channels, 1)

    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.bottleneck(s3)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.head(x)


def enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def multilabel_dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5, dice_weight: float = 0.5):
    t = target.float()
    bce = F.binary_cross_entropy_with_logits(logits.float(), t)
    p = torch.sigmoid(logits.float())
    dims = (0, 2, 3, 4)
    inter = (p * t).sum(dim=dims)
    denom = p.sum(dim=dims) + t.sum(dim=dims)
    dice = (2 * inter + 1.0) / (denom + 1.0)
    dl = 1.0 - dice.mean()
    return float(bce_weight) * bce + float(dice_weight) * dl, {"bce": float(bce.detach()), "dice_soft": float(dice.mean().detach())}


def dice_per_region(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    p = torch.sigmoid(logits) >= float(threshold)
    t = target >= 0.5
    dims = (0, 2, 3, 4)
    inter = (p & t).sum(dim=dims).float()
    denom = p.sum(dim=dims).float() + t.sum(dim=dims).float()
    return (2 * inter + 1e-6) / (denom + 1e-6)

class ResBlock3d(nn.Module):
    def __init__(self,in_ch:int,out_ch:int,stride:int=1,dropout:float=0.0):
        super().__init__();self.c1=nn.Conv3d(in_ch,out_ch,3,padding=1,stride=stride,bias=False);self.n1=nn.InstanceNorm3d(out_ch,affine=True);self.c2=nn.Conv3d(out_ch,out_ch,3,padding=1,bias=False);self.n2=nn.InstanceNorm3d(out_ch,affine=True);self.drop=nn.Dropout3d(dropout) if dropout>0 else nn.Identity();self.skip=nn.Conv3d(in_ch,out_ch,1,stride=stride,bias=False) if in_ch!=out_ch or stride!=1 else nn.Identity()
    def forward(self,x):
        y=F.leaky_relu(self.n1(self.c1(x)),.01,inplace=True);y=self.drop(y);y=self.n2(self.c2(y));return F.leaky_relu(y+self.skip(x),.01,inplace=True)

class BraTSResUNet3D(nn.Module):
    """Residual 3D U-Net replication backbone for method-agnostic validation."""
    def __init__(self,in_channels:int=4,out_channels:int=3,base:int=20,dropout:float=.15):
        super().__init__();b=int(base);self.e0=ResBlock3d(in_channels,b);self.e1=ResBlock3d(b,b*2,2);self.e2=ResBlock3d(b*2,b*4,2,dropout*.5);self.e3=ResBlock3d(b*4,b*8,2,dropout);self.bn=ResBlock3d(b*8,b*12,2,dropout);self.u3=Up3d(b*12,b*8,b*8,dropout);self.u2=Up3d(b*8,b*4,b*4,dropout*.5);self.u1=Up3d(b*4,b*2,b*2);self.u0=Up3d(b*2,b,b);self.head=nn.Conv3d(b,out_channels,1)
    def forward(self,x):
        s0=self.e0(x);s1=self.e1(s0);s2=self.e2(s1);s3=self.e3(s2);x=self.bn(s3);x=self.u3(x,s3);x=self.u2(x,s2);x=self.u1(x,s1);x=self.u0(x,s0);return self.head(x)

def build_brats_segmentor(architecture:str='unet3d',base:int=24,dropout:float=.15):
    a=str(architecture).lower().replace('-','_')
    if a in ('unet3d','unet_3d'):return BraTSUNet3D(base=base,dropout=dropout)
    if a in ('resunet3d','resunet_3d'):return BraTSResUNet3D(base=base,dropout=dropout)
    raise ValueError(f'Unknown BraTS architecture: {architecture}')

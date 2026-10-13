from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.brats2021.data import BraTS2021VolumeDataset, REGIONS
from src.brats2021.model import build_brats_segmentor, multilabel_dice_bce_loss, dice_per_region


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def device_from(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def epoch(model, loader, device, opt=None, amp=True, bce_weight=.5, dice_weight=.5):
    train = opt is not None
    model.train(train)
    scaler = epoch.scaler
    losses, dices = [], []
    for image, mask, _ in loader:
        image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        if train: opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=(amp and device.type == 'cuda')):
            logits = model(image)
            loss, _ = multilabel_dice_bce_loss(logits, mask, bce_weight, dice_weight)
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
        losses.append(float(loss.detach().cpu()))
        dices.append(dice_per_region(logits.detach(), mask).cpu().numpy())
    d = np.stack(dices).mean(axis=0)
    return {'loss': float(np.mean(losses)), **{f'dice_{REGIONS[i]}': float(d[i]) for i in range(3)}, 'dice_mean': float(d.mean())}

epoch.scaler = GradScaler('cuda', enabled=torch.cuda.is_available())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/brats2021_farc.yaml')
    ap.add_argument('--manifest', default='data/brats2021_manifest.csv')
    ap.add_argument('--data-root', default='/datastore/phucngt/datasets/BraTS2021')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    scfg = cfg['segmentor']
    seed_all(args.seed)
    device = device_from(args.device)
    shape = tuple(int(v) for v in scfg.get('volume_shape', [128,128,128]))
    tr = BraTS2021VolumeDataset(args.manifest, 'train', args.data_root, shape, train=True)
    va = BraTS2021VolumeDataset(args.manifest, 'source_val', args.data_root, shape, train=False)
    trl = DataLoader(tr, batch_size=int(scfg.get('batch_size',1)), shuffle=True, num_workers=int(scfg.get('num_workers',2)), pin_memory=True)
    val = DataLoader(va, batch_size=1, shuffle=False, num_workers=int(scfg.get('num_workers',2)), pin_memory=True)
    model = build_brats_segmentor(scfg.get('architecture','unet3d'), int(scfg.get('base_channels',24)), float(scfg.get('dropout',0.15))).to(device)
    opt = AdamW(model.parameters(), lr=float(scfg.get('lr',2e-4)), weight_decay=float(scfg.get('weight_decay',1e-5)))
    sch = CosineAnnealingLR(opt, T_max=max(1,int(scfg.get('epochs',60))))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    history=[]; best=-1.0; bad=0
    epochs=int(scfg.get('epochs',60)); patience=int(scfg.get('patience',12))
    for e in range(1,epochs+1):
        a=epoch(model,trl,device,opt,amp=bool(scfg.get('amp',True)))
        b=epoch(model,val,device,None,amp=bool(scfg.get('amp',True)))
        sch.step(); row={'epoch':e,'train':a,'val':b}; history.append(row)
        print(json.dumps(row), flush=True)
        if b['dice_mean'] > best + float(scfg.get('min_delta',1e-4)):
            best=b['dice_mean']; bad=0
            torch.save({'model':model.state_dict(),'config':cfg,'seed':args.seed,'best_val_dice':best,'volume_shape':shape}, out/'best.pt')
        else:
            bad += 1
            if bad >= patience: break
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps({'best_val_dice':best,'seed':args.seed,'device':str(device)},indent=2),encoding='utf-8')

if __name__=='__main__': main()

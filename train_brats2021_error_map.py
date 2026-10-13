from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np, torch
from torch.amp import GradScaler,autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from src.brats2021.data import BraTS2021VolumeDataset
from src.brats2021.model import build_brats_segmentor
from src.brats2021.error_map import ErrorMap3D,error_targets,error_loss

def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def device(n): return torch.device('cuda' if n=='auto' and torch.cuda.is_available() else ('cpu' if n=='auto' else n))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/brats2021_farc.yaml'); ap.add_argument('--segmentor-checkpoint',required=True); ap.add_argument('--manifest',default='data/brats2021_manifest.csv'); ap.add_argument('--data-root',default='/datastore/phucngt/datasets/BraTS2021'); ap.add_argument('--device',default='auto'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--output-dir',required=True); args=ap.parse_args()
 import yaml; cfg=yaml.safe_load(Path(args.config).read_text()); ecfg=cfg['error_map']; scfg=cfg['segmentor']; seed(args.seed); dev=device(args.device)
 ck=torch.load(args.segmentor_checkpoint,map_location='cpu',weights_only=False); shape=tuple(ck.get('volume_shape',scfg['volume_shape'])); seg=build_brats_segmentor(scfg.get('architecture','unet3d'),int(scfg.get('base_channels',24)),float(scfg.get('dropout',.15))).to(dev); seg.load_state_dict(ck['model']); seg.eval(); [p.requires_grad_(False) for p in seg.parameters()]
 net=ErrorMap3D(int(ecfg.get('width',8)),float(ecfg.get('dropout',.1))).to(dev); opt=AdamW(net.parameters(),lr=float(ecfg.get('lr',3e-4)),weight_decay=float(ecfg.get('weight_decay',1e-5))); scaler=GradScaler('cuda',enabled=torch.cuda.is_available())
 tr=BraTS2021VolumeDataset(args.manifest,'error_train',args.data_root,shape,train=False); va=BraTS2021VolumeDataset(args.manifest,'source_val',args.data_root,shape,train=False)
 trl=DataLoader(tr,batch_size=1,shuffle=True,num_workers=int(ecfg.get('num_workers',2)),pin_memory=True); val=DataLoader(va,batch_size=1,shuffle=False,num_workers=int(ecfg.get('num_workers',2)),pin_memory=True)
 out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); best=1e9; hist=[]
 for ep in range(1,int(ecfg.get('epochs',10))+1):
  vals={}
  for phase,loader in [('train',trl),('val',val)]:
   net.train(phase=='train'); ls=[]
   for im,gt,_ in loader:
    im,gt=im.to(dev),gt.to(dev)
    with torch.no_grad(): z=seg(im); y=error_targets(z,gt)
    if phase=='train': opt.zero_grad(set_to_none=True)
    with autocast(device_type=dev.type,enabled=bool(ecfg.get('amp',True) and dev.type=='cuda')): loss=error_loss(net(im,z),y,float(ecfg.get('max_pos_weight',50)))
    if phase=='train': scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(),5); scaler.step(opt); scaler.update()
    ls.append(float(loss.detach().cpu()))
   vals[phase]=float(np.mean(ls))
  hist.append({'epoch':ep,**vals}); print(json.dumps(hist[-1]),flush=True)
  if vals['val']<best: best=vals['val']; torch.save({'model':net.state_dict(),'config':cfg,'segmentor_checkpoint':args.segmentor_checkpoint,'best_val_loss':best},out/'best.pt')
 (out/'history.json').write_text(json.dumps(hist,indent=2)); (out/'summary.json').write_text(json.dumps({'best_val_loss':best},indent=2))
if __name__=='__main__': main()

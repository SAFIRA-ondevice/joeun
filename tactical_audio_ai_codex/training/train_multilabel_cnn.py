#!/usr/bin/env python3
"""Train independent sigmoid outputs with synthetic mixtures.

Directory layout: splits/{train,val}/{speech,drone,gunshot,background}/*.wav
Background is a negative source, not an output class. Target clips are mixed at random
SNRs so speech+drone etc. have multi-hot labels.
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch, torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0,str(Path(__file__).parent)); from audio_model import AudioCNN
from audio_io import load_mono

def load_audio(path,sr,size,rng=None):
    x,old=load_mono(path)
    if old!=sr: x=torchaudio.functional.resample(x,old,sr)
    if len(x)>size:
        start=(rng.randrange(len(x)-size+1) if rng else (len(x)-size)//2); x=x[start:start+size]
    return torch.nn.functional.pad(x[:size],(0,max(0,size-len(x))))

def mix_at_snr(base,other,snr_db):
    br=torch.sqrt(torch.mean(base.square())+1e-9); orms=torch.sqrt(torch.mean(other.square())+1e-9)
    if br < 1e-5:
        return other / other.abs().max().clamp_min(1.0)
    scaled=other*(br/(orms*10**(snr_db/20)+1e-9)); out=base+scaled
    peak=out.abs().max().clamp_min(1.0); return out/peak

class MixDataset(Dataset):
    def __init__(self,root,targets,sr=16000,seconds=1.0,epoch_size=None,mix_probability=.65,max_targets=3,seed=42,training=True):
        self.targets=targets; self.sr=sr; self.size=int(sr*seconds); self.mix_probability=mix_probability; self.max_targets=max_targets; self.seed=seed; self.training=training
        self.by={c:sorted((root/c).glob("*.wav")) for c in targets+["background"]}; self.base=[(p,c) for c,v in self.by.items() for p in v]
        if not self.base or any(not self.by[c] for c in targets+["background"]): raise ValueError("Every target and background folder needs WAV files")
        self.epoch_size=epoch_size or len(self.base)
    def __len__(self): return self.epoch_size
    def __getitem__(self,index):
        rng=random.Random((self.seed+index) if not self.training else random.randrange(2**31))
        path,cls=(rng.choice(self.base) if self.training else self.base[index%len(self.base)])
        x=load_audio(path,self.sr,self.size,rng); y=torch.zeros(len(self.targets)); used=set()
        if cls in self.targets: y[self.targets.index(cls)]=1; used.add(cls)
        if self.training:
            candidates=[c for c in self.targets if c not in used]; rng.shuffle(candidates)
            extra=rng.randint(1,min(len(candidates),self.max_targets-len(used))) if candidates and rng.random()<self.mix_probability else 0
            for c in candidates[:extra]:
                other=load_audio(rng.choice(self.by[c]),self.sr,self.size,rng); x=mix_at_snr(x,other,rng.uniform(-8,8)); y[self.targets.index(c)]=1
            if rng.random()<.5:
                bg=load_audio(rng.choice(self.by["background"]),self.sr,self.size,rng); x=mix_at_snr(x,bg,rng.uniform(-10,10))
            x=x*rng.uniform(.8,1.2)+torch.randn_like(x)*rng.uniform(0,.003)
        mel=torchaudio.transforms.MelSpectrogram(self.sr,n_fft=512,win_length=400,hop_length=160,n_mels=64,f_min=20,f_max=7600)(x)
        z=torchaudio.transforms.AmplitudeToDB(top_db=80)(mel); z=(z-z.mean())/(z.std()+1e-6)
        return z[None],y

def metrics(logits,y,thresholds):
    pred=torch.sigmoid(logits)>=thresholds; truth=y.bool(); tp=(pred&truth).sum(0).float(); fp=(pred&~truth).sum(0).float(); fn=(~pred&truth).sum(0).float()
    f1=(2*tp/(2*tp+fp+fn+1e-9)).mean().item(); exact=(pred==truth).all(1).float().mean().item(); return f1,exact

def validate(model,loader,loss_fn,device,thresholds):
    model.eval(); zs=[]; ys=[]; loss=n=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); z=model(x); loss+=loss_fn(z,y).item()*len(y); n+=len(y); zs.append(z.cpu()); ys.append(y.cpu())
    f1,exact=metrics(torch.cat(zs),torch.cat(ys),thresholds); return loss/n,f1,exact

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--targets",nargs="+",default=["speech","drone","gunshot"]); p.add_argument("--epochs",type=int,default=50); p.add_argument("--epoch-size",type=int,default=12000)
    p.add_argument("--batch-size",type=int,default=32); p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--workers",type=int,default=2); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr=MixDataset(a.data/"train",a.targets,epoch_size=a.epoch_size,seed=a.seed,training=True); va=MixDataset(a.data/"val",a.targets,epoch_size=None,seed=a.seed,training=False)
    tl=DataLoader(tr,a.batch_size,shuffle=True,num_workers=a.workers); vl=DataLoader(va,a.batch_size,num_workers=a.workers)
    model=AudioCNN(len(a.targets)).to(device); loss_fn=nn.BCEWithLogitsLoss(); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4); thresholds=torch.full((len(a.targets),),.5); best=-1
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,a.epochs+1):
        model.train()
        for x,y in tl:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step()
        vl_loss,f1,exact=validate(model,vl,loss_fn,device,thresholds); print(f"epoch={epoch:03d} val_loss={vl_loss:.4f} macro_f1={f1:.4f} exact={exact:.4f}")
        if f1>best:
            best=f1; torch.save({"state_dict":model.state_dict(),"classes":a.targets,"task":"multilabel","activation":"sigmoid","thresholds":{c:.5 for c in a.targets},"sample_rate":16000,"n_mels":64,"seconds":1.0,"model_config":{"base_channels":32},"val_macro_f1":f1},a.output)
    print(json.dumps({"best_val_macro_f1":best,"model":str(a.output)},indent=2))
if __name__=="__main__": main()

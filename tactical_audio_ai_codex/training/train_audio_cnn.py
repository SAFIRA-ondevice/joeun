#!/usr/bin/env python3
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0,str(Path(__file__).parent))
from audio_model import AudioCNN
from audio_io import load_mono

class AudioFolder(Dataset):
    def __init__(self,root,classes,sr=16000,seconds=1.0,augment=False):
        self.classes=classes; self.sr=sr; self.size=int(sr*seconds); self.augment=augment
        self.items=[(p,i) for i,c in enumerate(classes) for p in sorted((root/c).glob("*.wav"))]
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,y=self.items[i]; x,old=load_mono(p)
        if old!=self.sr: x=torchaudio.functional.resample(x,old,self.sr)
        if self.augment and len(x)>self.size:
            start=random.randint(0,len(x)-self.size); x=x[start:start+self.size]
        else: x=x[:self.size]
        x=torch.nn.functional.pad(x,(0,max(0,self.size-len(x))))
        if self.augment:
            x=x+torch.randn_like(x)*random.uniform(0,0.005); x=x*random.uniform(.8,1.2)
        mel=torchaudio.transforms.MelSpectrogram(self.sr,n_fft=512,win_length=400,hop_length=160,n_mels=64,f_min=20,f_max=7600)(x)
        feat=torchaudio.transforms.AmplitudeToDB(top_db=80)(mel)
        feat=(feat-feat.mean())/(feat.std()+1e-6)
        return feat.unsqueeze(0),y

def evaluate(model,loader,loss_fn,device):
    model.eval(); loss=correct=n=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); z=model(x); loss+=loss_fn(z,y).item()*len(y); correct+=(z.argmax(1)==y).sum().item(); n+=len(y)
    return loss/max(n,1),correct/max(n,1)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--classes",nargs="+",default=["speech","drone","gunshot","background"]); p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--batch-size",type=int,default=32); p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--workers",type=int,default=2); a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr=AudioFolder(a.data/"train",a.classes,augment=True); va=AudioFolder(a.data/"val",a.classes)
    if not tr.items or not va.items: raise SystemExit("Empty split; run split_dataset.py first")
    tl=DataLoader(tr,a.batch_size,shuffle=True,num_workers=a.workers); vl=DataLoader(va,a.batch_size,num_workers=a.workers)
    model=AudioCNN(len(a.classes)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss(); best=-1
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,a.epochs+1):
        model.train()
        for x,y in tl:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step()
        vl_loss,acc=evaluate(model,vl,loss_fn,device); print(f"epoch={epoch:03d} val_loss={vl_loss:.4f} val_acc={acc:.4f}")
        if acc>best:
            best=acc; torch.save({"state_dict":model.state_dict(),"classes":a.classes,"sample_rate":16000,"n_mels":64,"seconds":1.0,"model_config":{"base_channels":32},"known_issue":None,"val_accuracy":acc},a.output)
    print(json.dumps({"best_val_accuracy":best,"model":str(a.output),"device":str(device)},indent=2))
if __name__=="__main__": main()

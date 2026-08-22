#!/usr/bin/env python3
import argparse, sys, time
from collections import deque
from pathlib import Path
import numpy as np, serial, torch, torchaudio
sys.path.insert(0,str(Path(__file__).parents[1]/"training")); sys.path.insert(0,str(Path(__file__).parent))
from audio_model import AudioCNN
from m3c0 import sync_packet

def feature(x,sr,n_mels):
    x=torch.from_numpy(x.astype(np.float32)/32768.0)
    mel=torchaudio.transforms.MelSpectrogram(sr,n_fft=512,win_length=400,hop_length=160,n_mels=n_mels,f_min=20,f_max=7600)(x)
    z=torchaudio.transforms.AmplitudeToDB(top_db=80)(mel); z=(z-z.mean())/(z.std()+1e-6); return z[None,None]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--port",default="/dev/ttyUSB1"); p.add_argument("--baud",type=int,default=921600)
    p.add_argument("--model",type=Path,required=True); p.add_argument("--threshold",type=float,default=.60); p.add_argument("--hop",type=float,default=.25)
    p.add_argument("--channels",type=int,default=3); p.add_argument("--no-start-command",action="store_true"); a=p.parse_args()
    ck=torch.load(a.model,map_location="cpu"); classes=ck["classes"]; sr=int(ck.get("sample_rate",16000)); n_mels=int(ck.get("n_mels",64))
    model=AudioCNN(len(classes),**ck.get("model_config",{})); model.load_state_dict(ck["state_dict"]); model.eval()
    ring=deque(maxlen=sr*2); since=0; expected=None; hop=max(1,int(sr*a.hop))
    with serial.Serial(a.port,a.baud,timeout=2) as s:
        s.reset_input_buffer()
        if not a.no_start_command: s.write(b"START_INMP_ONLY\n"); s.flush(); time.sleep(.2)
        while True:
            seq,n,payload=sync_packet(s,a.channels)
            if expected is not None and seq!=expected: print(f"[WARN] sequence jump | expected={expected} | received={seq}")
            expected=(seq+1)&0xffff; pcm=np.frombuffer(payload,dtype="<i2").reshape(n,a.channels)
            mono=((pcm[:,0].astype(np.int32)+pcm[:,1].astype(np.int32))//2).astype(np.int16); ring.extend(mono.tolist()); since+=n
            if len(ring)>=sr and since>=hop:
                since=0; x=np.asarray(ring,dtype=np.int16)[-sr:]; t=time.perf_counter()
                with torch.no_grad(): prob=torch.softmax(model(feature(x,sr,n_mels)),1)[0].numpy()
                ms=(time.perf_counter()-t)*1000; idx=int(prob.argmax()); label=classes[idx].upper() if prob[idx]>=a.threshold else "UNKNOWN"
                scores=" | ".join(f"{c}={v:.3f}" for c,v in zip(classes,prob)); print(f"[AI] {scores} | => {label:<10} ({prob[idx]*100:.1f}%) | inference={ms:.1f} ms")
if __name__=="__main__": main()

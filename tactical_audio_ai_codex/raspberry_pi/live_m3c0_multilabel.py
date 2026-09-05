#!/usr/bin/env python3
import argparse, json, sys, time
from collections import deque
from pathlib import Path
import numpy as np, serial, torch, torchaudio
sys.path.insert(0,str(Path(__file__).parents[1]/"training")); sys.path.insert(0,str(Path(__file__).parent))
from audio_model import AudioCNN
from m3c0 import sync_packet
from decision_logic import decide

def feature(x,sr,n_mels):
    t=torch.from_numpy(x.astype(np.float32)/32768.0); mel=torchaudio.transforms.MelSpectrogram(sr,n_fft=512,win_length=400,hop_length=160,n_mels=n_mels,f_min=20,f_max=7600)(t)
    z=torchaudio.transforms.AmplitudeToDB(top_db=80)(mel); z=(z-z.mean())/(z.std()+1e-6); return z[None,None]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--port",default="/dev/ttyUSB1"); p.add_argument("--baud",type=int,default=921600); p.add_argument("--model",type=Path,required=True)
    p.add_argument("--hop",type=float,default=.25); p.add_argument("--smoothing",type=int,default=4); p.add_argument("--channels",type=int,default=3); p.add_argument("--threshold",type=float,default=None)
    p.add_argument("--yamnet",action="store_true"); p.add_argument("--yamnet-handle",default="https://tfhub.dev/google/yamnet/1"); p.add_argument("--yamnet-class-map"); p.add_argument("--yamnet-confidence",type=float,default=.25)
    p.add_argument("--json",action="store_true"); p.add_argument("--no-start-command",action="store_true"); a=p.parse_args()
    ck=torch.load(a.model,map_location="cpu");
    if ck.get("task")!="multilabel": raise SystemExit("This script requires a checkpoint with task=multilabel")
    classes=ck["classes"]; sr=int(ck.get("sample_rate",16000)); n_mels=int(ck.get("n_mels",64)); thresholds={c:(a.threshold if a.threshold is not None else float(ck.get("thresholds",{}).get(c,.5))) for c in classes}
    model=AudioCNN(len(classes),**ck.get("model_config",{})); model.load_state_dict(ck["state_dict"]); model.eval(); fallback=None
    if a.yamnet:
        from audioset_fallback import YamnetFallback
        fallback=YamnetFallback(a.yamnet_handle,a.yamnet_class_map,a.yamnet_confidence)
    ring=deque(maxlen=sr*2); history=deque(maxlen=max(1,a.smoothing)); since=0; expected=None; hop=max(1,int(sr*a.hop))
    with serial.Serial(a.port,a.baud,timeout=2) as s:
        s.reset_input_buffer()
        if not a.no_start_command: s.write(b"START_INMP_ONLY\n"); s.flush(); time.sleep(.2)
        while True:
            seq,n,payload=sync_packet(s,a.channels)
            if expected is not None and seq!=expected: print(f"[WARN] sequence jump expected={expected} received={seq}")
            expected=(seq+1)&0xffff; pcm=np.frombuffer(payload,dtype="<i2").reshape(n,a.channels); mono=((pcm[:,0].astype(np.int32)+pcm[:,1].astype(np.int32))//2).astype(np.int16); ring.extend(mono); since+=n
            if len(ring)>=sr and since>=hop:
                since=0; x=np.asarray(ring,dtype=np.int16)[-sr:]; t=time.perf_counter()
                with torch.no_grad(): raw=torch.sigmoid(model(feature(x,sr,n_mels)))[0].numpy()
                history.append(raw); prob=np.mean(history,axis=0); out=decide(classes,prob,thresholds,fallback,x,sr); out["inference_ms"]=round((time.perf_counter()-t)*1000,1)
                if a.json: print(json.dumps(out,ensure_ascii=False))
                else:
                    scores=" | ".join(f"{c}={out['targets'][c]:5.1f}%" for c in classes); print(f"[AI] {scores} | => {out['final']} | {out['inference_ms']:.1f} ms")
if __name__=="__main__": main()

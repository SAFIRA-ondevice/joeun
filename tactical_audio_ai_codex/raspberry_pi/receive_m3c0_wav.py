#!/usr/bin/env python3
import argparse, sys, time, wave
from pathlib import Path
import numpy as np, serial
sys.path.insert(0,str(Path(__file__).parent)); from m3c0 import sync_packet

def main():
    p=argparse.ArgumentParser(); p.add_argument("--port",default="/dev/ttyUSB1"); p.add_argument("--baud",type=int,default=921600)
    p.add_argument("--seconds",type=float,default=10); p.add_argument("--sample-rate",type=int,default=16000); p.add_argument("--channels",type=int,default=3)
    p.add_argument("--output",type=Path,default=Path("/home/pi/joeun/recordings/m3c0_capture.wav")); p.add_argument("--no-start-command",action="store_true"); a=p.parse_args()
    target=int(a.seconds*a.sample_rate); chunks=[]; total=0; expected=None
    with serial.Serial(a.port,a.baud,timeout=2) as s:
        s.reset_input_buffer()
        if not a.no_start_command: s.write(b"START_INMP_ONLY\n"); s.flush(); time.sleep(.2)
        while total<target:
            seq,n,payload=sync_packet(s,a.channels)
            if expected is not None and seq!=expected: print(f"[WARN] sequence jump expected={expected} received={seq}")
            expected=(seq+1)&0xffff; x=np.frombuffer(payload,dtype="<i2").reshape(n,a.channels); chunks.append(x); total+=n
    data=np.concatenate(chunks)[:target]; a.output.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(a.output),"wb") as w: w.setnchannels(a.channels); w.setsampwidth(2); w.setframerate(a.sample_rate); w.writeframes(data.astype("<i2").tobytes())
    print(f"saved {a.output} shape={data.shape} rate={a.sample_rate}")
if __name__=="__main__": main()

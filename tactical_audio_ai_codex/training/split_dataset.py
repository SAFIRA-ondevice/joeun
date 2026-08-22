#!/usr/bin/env python3
import argparse, csv, random, shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description="Deterministic class-balanced train/val/test split")
    p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--classes",nargs="+",required=True); p.add_argument("--train",type=float,default=.8)
    p.add_argument("--val",type=float,default=.1); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--balance-min",action="store_true"); p.add_argument("--copy",action="store_true")
    a=p.parse_args(); rng=random.Random(a.seed)
    if a.train+a.val>=1: raise SystemExit("train + val must be < 1")
    by={c:sorted((a.input/c).glob("*.wav")) for c in a.classes}
    if any(not v for v in by.values()): raise SystemExit(f"empty class: {[k for k,v in by.items() if not v]}")
    limit=min(map(len,by.values())) if a.balance_min else None; rows=[]
    for c,files in by.items():
        rng.shuffle(files); files=files[:limit] if limit else files
        n=len(files); ntr=int(n*a.train); nv=int(n*a.val)
        for split,items in (("train",files[:ntr]),("val",files[ntr:ntr+nv]),("test",files[ntr+nv:])):
            for src in items:
                dst=a.output/split/c/src.name; dst.parent.mkdir(parents=True,exist_ok=True)
                if a.copy: shutil.copy2(src,dst)
                else:
                    try: dst.symlink_to(src.resolve())
                    except OSError: shutil.copy2(src,dst)
                rows.append((split,c,str(src),str(dst)))
            print(c,split,len(items))
    with (a.output/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(("split","class","source","destination")); w.writerows(rows)
if __name__=="__main__": main()

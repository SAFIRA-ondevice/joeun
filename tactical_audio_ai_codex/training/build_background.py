#!/usr/bin/env python3
import argparse, hashlib, random, shutil
from pathlib import Path

AUDIO={".wav",".flac",".ogg"}
def main():
    p=argparse.ArgumentParser(description="Build background pool without touching sources")
    p.add_argument("--sources",nargs="+",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--exclude-token",nargs="*",default=["yes_drone","gunshot"]); p.add_argument("--limit",type=int)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--dry-run",action="store_true"); a=p.parse_args()
    files=[]
    for root in a.sources:
        files += [x for x in root.rglob("*") if x.suffix.lower() in AUDIO and not any(t.lower() in str(x).lower() for t in a.exclude_token)]
    files=sorted(set(files)); random.Random(a.seed).shuffle(files); files=files[:a.limit] if a.limit else files
    print(f"selected background candidates={len(files)}")
    if not a.dry_run:
        a.output.mkdir(parents=True,exist_ok=True)
        for src in files:
            key=hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:10]; shutil.copy2(src,a.output/f"{src.stem}_{key}{src.suffix.lower()}")
if __name__=="__main__": main()

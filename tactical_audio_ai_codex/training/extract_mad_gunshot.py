#!/usr/bin/env python3
import argparse, csv, json, shutil
from pathlib import Path

def norm(v): return str(v).strip().lower().replace("-","_").replace(" ","_")
def load_rows(path):
    if path.suffix.lower()==".json":
        data=json.loads(path.read_text(encoding="utf-8")); return data if isinstance(data,list) else data.get("annotations",data.get("data",[]))
    with path.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser(description="Extract only label=gunshot from MAD annotations")
    p.add_argument("--mad-root",type=Path,required=True); p.add_argument("--annotations",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--path-column"); p.add_argument("--label-column")
    p.add_argument("--dry-run",action="store_true"); a=p.parse_args(); rows=load_rows(a.annotations)
    if not rows: raise SystemExit("No annotation rows")
    keys=list(rows[0]); path_col=a.path_column or next((k for k in keys if norm(k) in {"file","filename","filepath","path","audio"}),None)
    label_col=a.label_column or next((k for k in keys if norm(k) in {"label","class","category","sound_class"}),None)
    if not path_col or not label_col: raise SystemExit(f"Specify columns. Available: {keys}")
    selected=[r for r in rows if norm(r.get(label_col,""))=="gunshot"]
    print(f"rows={len(rows)} gunshot={len(selected)} path_column={path_col} label_column={label_col}")
    missing=[]
    for r in selected:
        rel=Path(str(r[path_col])); src=rel if rel.is_absolute() else a.mad_root/rel
        if not src.exists(): missing.append(str(src)); continue
        if not a.dry_run: a.output.mkdir(parents=True,exist_ok=True); shutil.copy2(src,a.output/src.name)
    print(f"copied={0 if a.dry_run else len(selected)-len(missing)} missing={len(missing)} dry_run={a.dry_run}")
    if missing: print("first missing:",*missing[:10],sep="\n")
if __name__=="__main__": main()

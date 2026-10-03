"""Histogram of concern categories per function over the same 12-repo population as funnel12.py.
Answers: of functions the analyzer flags at all (>=1 category), what share carry >=2?"""
import os, ast, json, collections, tempfile, textwrap
import funnel12 as F

if __name__=="__main__":
    fs=F.extract()
    d1=tempfile.mkdtemp(prefix="hist_all_")
    meta={}
    for i,(repo,s) in enumerate(fs):
        src=s
        try: ast.parse(src)
        except Exception:
            src=textwrap.dedent(s)
            try: ast.parse(src)
            except Exception: continue
        name=f"{i}.py"; open(os.path.join(d1,name),"w").write(src); meta[name]=repo
    print(f"parseable: {len(meta)}",flush=True)
    m=F.ruff_map(d1)
    hist=collections.Counter(len(m.get(n,())) for n in meta)
    tot=len(meta); flagged=sum(v for k,v in hist.items() if k>=1); comp=sum(v for k,v in hist.items() if k>=2)
    print("\ncategories per function:")
    for k in sorted(hist): print(f"  {k}: {hist[k]:>7}  ({100*hist[k]/tot:5.2f}% of all)")
    print(f"\nparseable functions      : {tot}")
    print(f"flagged (>=1 category)   : {flagged}  = {100*flagged/tot:.2f}% of all functions")
    print(f"compound (>=2 categories): {comp}  = {100*comp/tot:.2f}% of all functions")
    print(f"COMPOUND SHARE OF FLAGGED: {100*comp/flagged:.2f}%   -> 1 in {flagged/comp:.1f} flagged functions")
    json.dump({"hist":dict(hist),"parseable":tot,"flagged":flagged,"compound":comp,
               "compound_share_of_flagged":comp/flagged}, open("funnel_hist.json","w"), indent=1)

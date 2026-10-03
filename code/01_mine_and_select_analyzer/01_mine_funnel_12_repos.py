"""Full 12-repo funnel with criterion (2) as its own stage, batched so Ruff runs a handful of
times instead of once per function. Mirrors scan_ours_dedup.py's extraction and dedup exactly."""
import os, ast, json, hashlib, shutil, subprocess, textwrap, tempfile, collections
ROOT="/data/home/hrish/kg-adapter/swebench_repos"
SKIP={".git","build","dist","node_modules",".tox","tests","test","__pycache__","doc","docs",".eggs"}
SEL="F841,F811,ERA,C4,PERF,SIM,UP,B,RET,PIE,RSE"
CATS={"cleanup":("F841","F811","ERA"),"comprehension":("C40","C41","PERF401","PERF402","PERF403"),
      "simplify":("SIM",),"modernize":("UP",),"bugbear":("B0","RET","PIE","RSE")}
def cat_of(c):
    for k,ps in CATS.items():
        if any(c.startswith(p) for p in ps): return k

def extract():
    out=[]
    for repo in sorted(os.listdir(ROOT)):
        rp=os.path.join(ROOT,repo)
        if not os.path.isdir(rp): continue
        n0=len(out)
        for dp,dn,fn in os.walk(rp):
            dn[:]=[d for d in dn if d not in SKIP]
            for f in fn:
                if not f.endswith(".py"): continue
                try:
                    src=open(os.path.join(dp,f),encoding="utf-8",errors="ignore").read(); tree=ast.parse(src)
                except Exception: continue
                L=src.split("\n")
                for n in ast.walk(tree):
                    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and getattr(n,"end_lineno",None):
                        out.append((repo,"\n".join(L[n.lineno-1:n.end_lineno])))
        print(f"  {repo:32s} {len(out)-n0}",flush=True)
    seen=set(); uniq=[]
    globals()['fs_raw']=out
    for r,s in out:
        h=hashlib.md5(s.encode()).hexdigest()
        if h in seen: continue
        seen.add(h); uniq.append((r,s))
    print(f"extracted {len(out)} -> unique {len(uniq)}",flush=True)
    return uniq

def ruff_map(d):
    """one ruff invocation over a directory -> {basename: set(categories)}"""
    out=subprocess.run(["ruff","check",d,"--select",SEL,"--output-format","json",
                        "--isolated","--no-cache"],capture_output=True,text=True).stdout
    try: v=json.loads(out) if out.strip() else []
    except Exception: v=[]
    m=collections.defaultdict(set)
    for x in v:
        c=cat_of(x["code"])
        if c: m[os.path.basename(x["filename"])].add(c)
    return m

if __name__=="__main__":
    fs=extract()
    d1=tempfile.mkdtemp(prefix="funnel_all_")
    meta={}
    par=0
    for i,(repo,s) in enumerate(fs):
        src=s
        try: ast.parse(src)
        except Exception:
            src=textwrap.dedent(s)
            try: ast.parse(src)
            except Exception: continue
        par+=1
        name=f"{i}.py"
        open(os.path.join(d1,name),"w").write(src)
        meta[name]=(repo,len(src.splitlines()))
    print(f"parseable written: {par}",flush=True)
    m=ruff_map(d1)
    compound={n for n,c in m.items() if len(c)>=2}
    print(f"criterion (1) compound (>=2 cats): {len(compound)}",flush=True)

    # criterion (2) BEFORE criterion (3), matching the order of the numbered list in the paper
    d2=tempfile.mkdtemp(prefix="funnel_fix_")
    for n in compound: shutil.copy(os.path.join(d1,n),os.path.join(d2,n))
    subprocess.run(["ruff","check",d2,"--select",SEL,"--fix","--isolated","--no-cache"],
                   capture_output=True,text=True)
    after=ruff_map(d2)
    survive={n for n in compound if len(after.get(n,()))>=1}
    print(f"criterion (2) nonfixable: {len(survive)}",flush=True)
    le80={n for n in survive if meta[n][1]<=80}
    print(f"criterion (3) <=80 lines: {len(le80)}",flush=True)

    def per(names): return collections.Counter(meta[n][0] for n in names)
    allnames=set(meta)
    stages={"parseable":per(allnames),"crit1":per(compound),"crit2":per(survive),"crit3":per(le80)}
    extracted=collections.Counter(r for r,_ in fs_raw)
    uniq=collections.Counter(r for r,_ in fs)
    repos=sorted(extracted, key=lambda r:-extracted[r])
    print(f"\n{'repo':30s}{'extr':>8}{'uniq':>8}{'parse':>8}{'(1)':>7}{'(2)':>7}{'(3)':>7}")
    for r in repos:
        print(f"{r:30s}{extracted[r]:>8}{uniq[r]:>8}{stages['parseable'][r]:>8}"
              f"{stages['crit1'][r]:>7}{stages['crit2'][r]:>7}{stages['crit3'][r]:>7}")
    res={"extracted":dict(extracted),"unique":dict(uniq),
         "parseable":dict(stages["parseable"]),"crit1":dict(stages["crit1"]),
         "crit2":dict(stages["crit2"]),"crit3":dict(stages["crit3"])}
    json.dump(res,open("funnel12_byrepo.json","w"),indent=1)
    shutil.rmtree(d1,ignore_errors=True); shutil.rmtree(d2,ignore_errors=True)

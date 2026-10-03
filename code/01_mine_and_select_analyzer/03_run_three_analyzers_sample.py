"""Table 2 re-run on the DEDUPLICATED parseable population (51,300), seed 123456.
Saves per-file counts so capped and uncapped variants come from one Pylint pass."""
import os,ast,json,hashlib,random,shutil,subprocess,textwrap,tempfile,collections,time,math,statistics
ROOT="/data/home/hrish/kg-adapter/swebench_repos"
SKIP={".git","build","dist","node_modules",".tox","tests","test","__pycache__","doc","docs",".eggs"}
SEL="F841,F811,ERA,C4,PERF,SIM,UP,B,RET,PIE,RSE"
CATS={"cleanup":("F841","F811","ERA"),"comprehension":("C40","C41","PERF401","PERF402","PERF403"),
      "simplify":("SIM",),"modernize":("UP",),"bugbear":("B0","RET","PIE","RSE")}
ARTIFACT={"undefined-variable","import-error","used-before-assignment","no-name-in-module",
          "undefined-loop-variable","no-member","relative-beyond-top-level"}
STYLE={"missing-module-docstring","missing-function-docstring","missing-class-docstring",
       "missing-final-newline","line-too-long","invalid-name","trailing-whitespace",
       "trailing-newlines","bad-indentation","protected-access","too-many-arguments",
       "too-many-locals","too-many-branches","too-few-public-methods"}
def cat_of(c):
    for k,ps in CATS.items():
        if any(c.startswith(p) for p in ps): return k
out=[]
for repo in sorted(os.listdir(ROOT)):
    rp=os.path.join(ROOT,repo)
    if not os.path.isdir(rp): continue
    for dp,dn,fn in os.walk(rp):
        dn[:]=[d for d in dn if d not in SKIP]
        for f in fn:
            if not f.endswith(".py"): continue
            try:
                s0=open(os.path.join(dp,f),encoding="utf-8",errors="ignore").read(); tree=ast.parse(s0)
            except Exception: continue
            L=s0.split("\n")
            for n in ast.walk(tree):
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and getattr(n,"end_lineno",None):
                    out.append("\n".join(L[n.lineno-1:n.end_lineno]))
seen=set(); uniq=[]
for s in out:
    h=hashlib.md5(s.encode()).hexdigest()
    if h in seen: continue
    seen.add(h); uniq.append(s)
par=[]
for s in uniq:
    src=s
    try: ast.parse(src)
    except Exception:
        src=textwrap.dedent(s)
        try: ast.parse(src)
        except Exception: continue
    par.append(src)
print(f"population: extracted {len(out)} -> unique {len(uniq)} -> parseable {len(par)}",flush=True)
random.seed(123456); samp=random.sample(par,4000)
lines=[len(s.splitlines()) for s in samp]
d=tempfile.mkdtemp(prefix="t2_"); ps=[]
for i,s in enumerate(samp):
    p=os.path.join(d,f"fn_{i:05d}.py"); open(p,"w").write(s); ps.append(p)

t0=time.time()
r=subprocess.run(["ruff","check",d,"--select",SEL,"--output-format","json","--isolated","--no-cache"],
                 capture_output=True,text=True)
t_ruff=time.time()-t0
rv=json.loads(r.stdout or "[]")
ruff_cats=collections.defaultdict(set); ruff_n=collections.Counter()
for x in rv:
    b=os.path.basename(x["filename"]); ruff_n[b]+=1
    c=cat_of(x["code"])
    if c: ruff_cats[b].add(c)
print(f"RUFF: {len(rv)} findings in {t_ruff:.2f}s",flush=True)

t0=time.time()
p=subprocess.run(["pylint","--output-format=json","--score=n",*ps],capture_output=True,text=True)
t_pyl=time.time()-t0
pm=json.loads(p.stdout or "[]")
print(f"PYLINT: {len(pm)} findings in {t_pyl:.0f}s",flush=True)
json.dump({"t_ruff":t_ruff,"t_pyl":t_pyl,"lines":lines,
           "ruff":[{"f":os.path.basename(x["filename"]),"code":x["code"]} for x in rv],
           "pylint":[{"f":os.path.basename(x["path"]),"sym":x["symbol"]} for x in pm]},
          open("rerun_table2_raw.json","w"))
print("raw saved to rerun_table2_raw.json",flush=True)
shutil.rmtree(d,ignore_errors=True)

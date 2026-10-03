"""VERIFICATION-ONLY benchmark (no gold fix needed). Mine functions from CURRENT HEAD with >=2 concern categories
where >=1 category is NON-autofixable by Ruff. Grading = (a) target concerns gone (Ruff) + (b) repo tests pass;
neither needs a reference. `ruff --fix` scores PARTIAL credit by construction (clears autofixable, leaves the rest)
-> the residual is what a method must earn. This script does the MINING + non-autofixable filter + the ruff --fix
partial-credit baseline, per repo. Coverage/test-execution is a separate stage (grade_docker.py).
Usage: python build_verif_benchmark.py <repo_dir>"""
import os, sys, ast, json, subprocess, tempfile, shutil, collections, textwrap
REPO=sys.argv[1]; NAME=os.path.basename(REPO.rstrip("/"))
SEL="F841,F811,ERA,C4,PERF,SIM,UP,B,RET,PIE,RSE"      # F401 dropped (module-scoped)
CATS={"cleanup":("F841","F811","ERA"),"comprehension":("C40","C41","PERF401","PERF402","PERF403"),
      "simplify":("SIM",),"modernize":("UP",),"bugbear":("B0","RET","PIE","RSE")}
def cat_of(c):
    for k,ps in CATS.items():
        if any(c.startswith(p) for p in ps): return k
def ruff_file(path):
    out=subprocess.run(["ruff","check",path,"--select",SEL,"--output-format","json","--no-cache",
                        "--exclude","*/test*,*/tests/*"],capture_output=True,text=True).stdout
    try: return json.loads(out) if out.strip() else []
    except: return []
def func_ranges(path):
    try: tree=ast.parse(open(path,errors="ignore").read())
    except: return []
    out=[]
    def walk(node,cls=None):
        for n in getattr(node,"body",[]):
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                out.append((n.lineno,getattr(n,"end_lineno",n.lineno),(cls+"." if cls else "")+n.name))
            if isinstance(n,ast.ClassDef): walk(n,(cls+"." if cls else "")+n.name)
    walk(ast.parse(open(path,errors="ignore").read())); return out
def enclosing(fr,line):
    best=None
    for s,e,nm in fr:
        if s<=line<=e and (best is None or s>best[0]): best=(s,e,nm)
    return best

# 1) per-function concern categories + fixability, from full-file ruff (context-aware)
byfunc=collections.defaultdict(lambda: {"cats":set(),"nonfix":set(),"rules":set()})
frc={}
for root,_,fs in os.walk(REPO):
    if "/.git" in root or "/test" in root: continue
    for fn in fs:
        if not fn.endswith(".py"): continue
        p=os.path.join(root,fn)
        for v in ruff_file(p):
            if "/test" in v["filename"]: continue
            c=cat_of(v["code"])
            if not c: continue
            fr=frc.get(v["filename"]) or frc.setdefault(v["filename"],func_ranges(v["filename"]))
            enc=enclosing(fr,v["location"]["row"])
            if not enc: continue
            key=(v["filename"],enc[2],enc[0],enc[1]); d=byfunc[key]
            d["cats"].add(c); d["rules"].add(v["code"])
            if not v.get("fix"): d["nonfix"].add(c)

# 2) qualify: >=2 categories AND >=1 non-autofixable category
items=[]
for (fp,name,s,e),d in byfunc.items():
    if len(d["cats"])<2 or not d["nonfix"]: continue
    src=textwrap.dedent("\n".join(open(fp,errors="ignore").read().split("\n")[s-1:e]))
    try: ast.parse(src)
    except: continue
    items.append({"repo":NAME,"file":os.path.relpath(fp,REPO),"func":name,"start":s,"end":e,
                  "concerns":sorted(d["cats"]),"nonfixable":sorted(d["nonfix"]),
                  "n_lines":e-s+1,"rules":sorted(d["rules"]),"orig_src":src})

# 3) ruff --fix PARTIAL-CREDIT baseline: after autofix, how many target categories remain?
def autofix_residual(fp, s, e):
    tmp=tempfile.NamedTemporaryFile(suffix=".py",delete=False,mode="w");tmp.close()
    shutil.copyfile(fp,tmp.name)
    subprocess.run(["ruff","check","--fix","--select",SEL,"--isolated","--no-cache",tmp.name],capture_output=True)
    fr=func_ranges(tmp.name)
    # locate same-named function nearest original start
    import ast as _a
    cats=set()
    for v in ruff_file(tmp.name):
        enc=enclosing(fr,v["location"]["row"])
        if enc and enc[2]==name_at and abs(enc[0]-s)<50:
            c=cat_of(v["code"]);
            if c: cats.add(c)
    os.unlink(tmp.name); return cats
cleared=[]; files_done={}
for it in items:
    fp=os.path.join(REPO,it["file"]); name_at=it["func"]
    resid=autofix_residual(fp,it["start"],it["end"])
    tgt=set(it["concerns"]); got=tgt-resid            # categories ruff --fix cleared
    it["autofix_cleared"]=sorted(got); it["autofix_residual"]=sorted(tgt&resid)
    cleared.append(len(got)/len(tgt))

json.dump(items, open(f"verif_{NAME}.json","w"), indent=1)
import statistics
pair=collections.Counter(); ncat=collections.Counter(); nc=collections.Counter()
for it in items:
    nc[len(it["concerns"])]+=1
    for c in it["concerns"]: ncat[c]+=1
    for i in range(len(it["concerns"])):
        for j in range(i+1,len(it["concerns"])): pair[(it["concerns"][i],it["concerns"][j])]+=1
print(f"===== VERIFICATION-ONLY benchmark — {NAME} =====")
print(f"  qualified items (>=2 cats, >=1 non-autofixable): {len(items)}")
print(f"  #concerns dist: {dict(sorted(nc.items()))}")
print(f"  per concern: {dict(ncat.most_common())}")
print(f"  top pairs: {dict(pair.most_common(6))}")
print(f"  n_lines: median={statistics.median([it['n_lines'] for it in items]) if items else 0} "
      f"max={max([it['n_lines'] for it in items],default=0)}")
print(f"  ruff --fix PARTIAL CREDIT (frac of target cats cleared): mean={statistics.mean(cleared)*100:.0f}%  "
      f"(<100% by construction -> residual = what the system must earn)")
print(f"  items where ruff --fix clears NOTHING: {sum(1 for c in cleared if c==0)}")
print("DONE",flush=True)

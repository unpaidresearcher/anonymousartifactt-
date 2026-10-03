"""Execution-grade the 220 original_tier_a items (sympy/django) IN THE LOCAL CHECKOUTS.

Both repos' bodies match their checkout at the recorded line ranges (once indentation is normalised), and
both suites run locally: sympy via pytest under bipy_py38, django via tests/runtests.py under kg-blackwell.
Stage 1 finds a GREEN targeted test per item; stage 2 patches the prediction in and re-runs that same test.
Env: SWEEP, METHOD, NTRY(4), TT(300)"""
import os, re, json, time, textwrap, collections, subprocess
REPOS  = "/data/home/hrish/kg-adapter/swebench_repos"
PYBIN  = {"sympy__sympy":  os.path.expanduser("~/miniconda3/envs/bipy_py38/bin/python"),
          "django__django": os.path.expanduser("~/miniconda3/envs/kg-blackwell/bin/python")}
SWEEP  = os.environ.get("SWEEP", "sweep_outputs_p15_loc.json")
METHOD = os.environ.get("METHOD", "single_all_loc")
NTRY   = int(os.environ.get("NTRY", "4")); TT = int(os.environ.get("TT", "300"))

def norm(s):
    return {s.strip(), textwrap.dedent(s).strip(),
            "\n".join(l.strip() for l in s.split("\n") if l.strip())}

def run_test(repo, tgt):
    root, py = os.path.join(REPOS, repo), PYBIN[repo]
    env = dict(os.environ, PYTHONPATH=root)
    if repo == "django__django":
        lbl = (tgt[6:] if tgt.startswith("tests/") else tgt)[:-3].replace("/", ".")
        cmd = [py, "tests/runtests.py", lbl, "--parallel=1"]
    else:
        cmd = [py, "-m", "pytest", tgt, "-q", "-p", "no:cacheprovider"]
    try: r = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=TT)
    except subprocess.TimeoutExpired: return None
    out = r.stdout + r.stderr
    if repo == "django__django":
        return ("OK" in out) and not re.search(r"FAILED|Traceback|No such test|Ran 0 tests", out)
    return bool(re.search(r"\d+ passed", out)) and not re.search(r"\d+ (failed|error)", out)

def index(repo):
    root = os.path.join(REPOS, repo); by = collections.defaultdict(list)
    for dp, dn, fn in os.walk(root):
        if os.sep + ".git" in dp: continue
        for f in fn:
            if not f.endswith(".py"): continue
            rel = os.path.relpath(os.path.join(dp, f), root)
            if f == "tests.py": by[os.path.basename(os.path.dirname(rel))].append(rel); continue
            if not re.match(r"(test_.+|.+_test|unittest_.+)\.py$", f): continue
            b = f[:-3]
            for s in (b[5:] if b.startswith("test_") else None, b[:-5] if b.endswith("_test") else None):
                if s: by[s].append(rel)
    return by

def candidates(by, relfile):
    parts = relfile[:-3].split(os.sep); base = parts[-1]
    keys = [base] + ([f"{parts[-2]}_{base}", parts[-2]] if len(parts) > 1 else [])
    sd = os.path.dirname(relfile).split(os.sep)
    def prox(tf):
        td = os.path.dirname(tf).split(os.sep); n = 0
        for a, b in zip(sd, td):
            if a != b: break
            n += 1
        return -n
    seen = []
    for k in keys:
        for tf in sorted(by.get(k, []), key=prox):
            if tf not in seen: seen.append(tf)
    return seen

def reindent(pred, orig_lines):
    oi = len(orig_lines[0]) - len(orig_lines[0].lstrip()); pl = pred.split("\n")
    while pl and not pl[0].strip(): pl.pop(0)
    if not pl: return pred
    pi = len(pl[0]) - len(pl[0].lstrip()); d = oi - pi
    if d > 0: return "\n".join((" "*d)+l if l.strip() else l for l in pl)
    if d < 0: return "\n".join(l[-d:] if l[:-d].strip()=="" else l.lstrip() for l in pl)
    return "\n".join(pl)

BENCH = [x for x in json.load(open("verif_benchmark_tiered_ext.json"))
         if x.get("tier") == "covered" and x["provenance"]["cohort"] == "original_tier_a"]
K = lambda r: (r["repo"], r["func"], r["start"])
preds = {K(o): o for o in json.load(open(SWEEP))[METHOD]}
todo = [x for x in BENCH if K(x) in preds and preds[K(x)].get("correct")]
print(f"{METHOD}: tier-A grader-correct predictions to execute: {len(todo)}", flush=True)

IDX = {r: index(r) for r in ("sympy__sympy", "django__django")}
for r, v in IDX.items(): print(f"  {r}: {len(v)} test stems indexed", flush=True)

res, rows, t0 = collections.Counter(), [], time.time()
for n, it in enumerate(todo, 1):
    repo, fpath = it["repo"], os.path.join(REPOS, it["repo"], it["file"])
    src = open(fpath, errors="ignore").read(); L = src.split("\n")
    if it["end"] > len(L) or not (norm("\n".join(L[it["start"]-1:it["end"]])) & norm(it["orig_src"])):
        res["body_moved"] += 1; continue
    tgt = None
    for g in candidates(IDX[repo], it["file"])[:NTRY]:
        if run_test(repo, g): tgt = g; break
    if not tgt: res["no_green_test"] += 1; continue
    new = "\n".join(L[:it["start"]-1] + reindent(preds[K(it)]["pred_src"], L[it["start"]-1:it["end"]]).split("\n") + L[it["end"]:])
    try:
        open(fpath, "w").write(new)
        ok = run_test(repo, tgt)
    finally:
        open(fpath, "w").write(src)                      # always restore
    if ok is None: res["timeout"] += 1; continue
    res["pass" if ok else "FAIL"] += 1
    rows.append({"repo": repo, "func": it["func"], "start": it["start"], "passed": bool(ok),
                 "test_target": tgt, "cohort": "original_tier_a"})
    if n % 10 == 0:
        json.dump(rows, open(f"exec_tiera_{METHOD}.json", "w"))
        print(f"  {n}/{len(todo)}  pass={res['pass']} FAIL={res['FAIL']} skip={dict((k,v) for k,v in res.items() if k not in ('pass','FAIL'))}  {(time.time()-t0)/60:.1f} min", flush=True)

json.dump(rows, open(f"exec_tiera_{METHOD}.json", "w"))
g = res["pass"] + res["FAIL"]
print(f"\n===== TIER-A LOCAL EXECUTION, {METHOD} =====")
print(f"  grader-correct  : {len(todo)}")
print(f"  executed        : {g}   pass={res['pass']}  FAIL={res['FAIL']}  "
      f"false-accept={100*res['FAIL']/g if g else 0:.1f}%")
print(f"  skipped         : {dict((k,v) for k,v in res.items() if k not in ('pass','FAIL'))}")
print(f"  wall clock      : {(time.time()-t0)/60:.1f} min")
print("DONE", flush=True)

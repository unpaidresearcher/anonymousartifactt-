"""STAGE 2: execution-grade an arm over ALL 584 items, not just the container cohort.

Merges tiera_recovered.json (stage 1: image + test_target for original_tier_a items) into the benchmark,
then runs the same patch-and-rerun cycle behavior_container.py uses, over every item that has an image.
Reports the execution-validated score for the arm, and states its coverage explicitly.
Env: SWEEP, METHOD, TEST_TIMEOUT(180)"""
import os, re, json, time, collections, subprocess
os.environ.setdefault("DOCKER_HOST", "unix:///run/user/1009/docker.sock")
SWEEP  = os.environ.get("SWEEP", "sweep_outputs_p15_loc.json")
METHOD = os.environ.get("METHOD", "single_all_loc")
TT     = int(os.environ.get("TEST_TIMEOUT", "180"))

def sh(a, t=None):
    try: return subprocess.run(a, capture_output=True, text=True, timeout=t)
    except Exception: return subprocess.CompletedProcess(a, 1, "", "")
def dexec(cid, script, t=None, stdin=None):
    try: return subprocess.run(["docker","exec","-i",cid,"bash","-lc",script], input=stdin,
                               capture_output=True, text=True, timeout=t or TT)
    except Exception: return subprocess.CompletedProcess([], 1, "", "")

K = lambda r: (r["repo"], r["func"], r["start"])
BENCH = {K(x): x for x in json.load(open("verif_benchmark_tiered_ext.json")) if x.get("tier") == "covered"}
try:
    rec = json.load(open("tiera_recovered.json"))
    for x in rec: BENCH[K(x)] = x                       # recovered items carry image + test_target
    print(f"merged {len(rec)} recovered tier-A items", flush=True)
except Exception as e:
    print(f"no tiera_recovered.json ({e}); container cohort only", flush=True)

preds = {K(o): o for o in json.load(open(SWEEP))[METHOD]}
N = len(BENCH)
correct = [k for k in BENCH if k in preds and preds[k].get("correct")]
runnable = [k for k in correct if BENCH[k]["provenance"].get("image") and BENCH[k].get("test_target")]
print(f"{METHOD}: {N} items | grader-correct {len(correct)} ({100*len(correct)/N:.1f}%) | "
      f"executable {len(runnable)} ({100*len(runnable)/max(len(correct),1):.1f}% of correct)", flush=True)

def reindent(pred, orig):
    oi = len(orig[0]) - len(orig[0].lstrip()); pl = pred.split("\n")
    while pl and not pl[0].strip(): pl.pop(0)
    if not pl: return pred
    pi = len(pl[0]) - len(pl[0].lstrip()); d = oi - pi
    if d > 0: return "\n".join((" "*d)+l if l.strip() else l for l in pl)
    if d < 0: return "\n".join(l[-d:] if l[:-d].strip()=="" else l.lstrip() for l in pl)
    return "\n".join(pl)

by_img = collections.defaultdict(list)
for k in runnable: by_img[BENCH[k]["provenance"]["image"]].append(k)
res, rows = collections.Counter(), []
t_all = time.time()
for img, group in sorted(by_img.items(), key=lambda x: -len(x[1])):
    cid = sh(["docker","run","-d",img,"sleep","infinity"], t=300).stdout.strip()
    if not cid: res["no_container"] += len(group); continue
    print(f"[{img.split('.')[-1][:40]}] {len(group)} items", flush=True)
    for k in group:
        it, pr = BENCH[k], preds[k]; f = it["file"]; s, e = it["start"], it["end"]
        orig = dexec(cid, f"cat /testbed/{f}", t=60).stdout
        if not orig: res["file_missing"] += 1; continue
        lines = orig.split("\n")
        if e > len(lines): res["range_bad"] += 1; continue
        new = "\n".join(lines[:s-1] + reindent(pr["pred_src"], lines[s-1:e]).split("\n") + lines[e:])
        dexec(cid, f"cp /testbed/{f} /tmp/orig.bak && cat > /testbed/{f}", t=90, stdin=new)
        ok = dexec(cid, "source /opt/miniconda3/bin/activate testbed && python -c "
                        f"'import ast;ast.parse(open(\"/testbed/{f}\").read())'", t=60).returncode == 0
        if not ok: res["patch_unparseable"] += 1
        else:
            r = dexec(cid, "source /opt/miniconda3/bin/activate testbed && cd /testbed && "
                           f"timeout {TT} python -m pytest {it['test_target']} -x -q 2>&1 | tail -3", t=TT+40)
            lo = r.stdout.lower()
            good = bool(re.search(r"\d+ passed", lo)) and not re.search(r"\d+ (failed|error)", lo)
            res["pass" if good else "FAIL"] += 1
            rows.append({"repo": it["repo"], "func": it["func"], "start": s, "passed": good,
                         "cohort": it["provenance"]["cohort"]})
        dexec(cid, f"cp /tmp/orig.bak /testbed/{f}", t=60)
    sh(["docker","rm","-f",cid], t=120)
    json.dump(rows, open(f"exec584_{METHOD}.json","w"))

json.dump(rows, open(f"exec584_{METHOD}.json","w"))
g = res["pass"] + res["FAIL"]
rate = res["FAIL"]/g if g else 0
print(f"\n===== EXECUTION OVER {N} ITEMS, {METHOD} =====")
print(f"  static grader-correct : {len(correct)}/{N} = {100*len(correct)/N:.1f}%")
print(f"  executed              : {g}/{len(correct)}   false-accept {100*rate:.1f}%")
print(f"  VALIDATED SCORE       : {res['pass'] + (len(correct)-g)*(1-rate):.1f}/{N} = "
      f"{100*(res['pass'] + (len(correct)-g)*(1-rate))/N:.1f}%")
print(f"  (pass={res['pass']} FAIL={res['FAIL']}, other={dict((k,v) for k,v in res.items() if k not in ('pass','FAIL'))})")
print(f"  wall clock: {(time.time()-t_all)/60:.1f} min")
print("DONE", flush=True)

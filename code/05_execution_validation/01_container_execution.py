"""CONTAINER-SIDE BEHAVIORAL VALIDATION for the extension cohort.

`test_exec_grade` patches into LOCAL repo checkouts, so extension items -- mined from SWE-bench containers at
other commits -- were ungradable (only 19/620 of P15's survivors could be tested, almost all sympy/django).
This validates them where they came from: start the item's own image, patch the predicted function into
/testbed at its recorded line range, re-run the SAME targeted test that was GREEN at admission, restore.

A prediction PASSES if that test still passes. Env: SWEEP, METHOD, LIMIT(0=all), TEST_TIMEOUT(180)."""
import os, re, json, subprocess, collections, time, sys
os.environ.setdefault("DOCKER_HOST", "unix:///run/user/1009/docker.sock")
SWEEP = os.environ.get("SWEEP", "sweep_outputs_p15_loc.json")
METHOD = os.environ.get("METHOD", "single_all_loc")
LIMIT = int(os.environ.get("LIMIT", "0")); TT = int(os.environ.get("TEST_TIMEOUT", "180"))

def sh(a, t=None): return subprocess.run(a, capture_output=True, text=True, timeout=t)
def dexec(cid, script, t=None, stdin=None):
    # stdin is used to ship file CONTENT: embedding a base64 file in argv overflows the arg list (E2BIG)
    return subprocess.run(["docker","exec","-i",cid,"bash","-lc",script], input=stdin,
                          capture_output=True, text=True, timeout=t or TT)

bench = {(x["repo"],x["func"],x["start"]): x for x in json.load(open("verif_benchmark_tiered_ext.json"))
         if x["tier"] == "covered" and x["provenance"]["cohort"] != "original_tier_a"}
preds = {(o["repo"],o["func"],o["start"]): o for o in json.load(open(SWEEP))[METHOD]}
todo = [(k, bench[k], preds[k]) for k in bench if k in preds and preds[k].get("correct")]
if LIMIT: todo = todo[:LIMIT]
print(f"{METHOD}: grader-correct EXTENSION items to behaviorally validate: {len(todo)}", flush=True)

def reindent(pred, orig):
    """Model output may lose the original indentation; realign it to the source line it replaces."""
    oi = len(orig[0]) - len(orig[0].lstrip())
    pl = pred.split("\n")
    while pl and not pl[0].strip(): pl.pop(0)
    if not pl: return pred
    pi = len(pl[0]) - len(pl[0].lstrip())
    d = oi - pi
    if d > 0:  return "\n".join((" "*d)+l if l.strip() else l for l in pl)
    if d < 0:  return "\n".join(l[-d:] if l[:-d].strip()=="" else l.lstrip() for l in pl)
    return "\n".join(pl)

by_img = collections.defaultdict(list)
for k, it, pr in todo: by_img[it["provenance"]["image"]].append((k, it, pr))
res = collections.Counter(); rows = []
for img, group in sorted(by_img.items(), key=lambda x: -len(x[1])):
    cid = sh(["docker","run","-d",img,"sleep","infinity"], t=180).stdout.strip()
    if not cid: print(f"  SKIP {img}: container failed"); res["no_container"] += len(group); continue
    print(f"\n[{img.split('.')[-1]}] {len(group)} items  container {cid[:12]}", flush=True)
    for i,(k,it,pr) in enumerate(group):
        f = it["file"]; s, e = it["start"], it["end"]
        orig = dexec(cid, f"cat /testbed/{f}", t=60).stdout
        if not orig: res["file_missing"] += 1; continue
        lines = orig.split("\n")
        if e > len(lines): res["range_bad"] += 1; continue
        patched = reindent(pr["pred_src"], lines[s-1:e])
        new = "\n".join(lines[:s-1] + patched.split("\n") + lines[e:])
        dexec(cid, f"cp /testbed/{f} /tmp/orig.bak && cat > /testbed/{f}", t=90, stdin=new)
        ok_parse = dexec(cid, f"source /opt/miniconda3/bin/activate testbed && python -c "
                              f"'import ast,sys;ast.parse(open(\"/testbed/{f}\").read())'", t=60).returncode == 0
        if not ok_parse:
            res["patch_unparseable"] += 1
        else:
            r = dexec(cid, f"source /opt/miniconda3/bin/activate testbed && cd /testbed && "
                           f"timeout {TT} python -m pytest {it['test_target']} -x -q 2>&1 | tail -3", t=TT+40)
            lo = r.stdout.lower()
            good = bool(re.search(r"\d+ passed", lo)) and not re.search(r"\d+ (failed|error)", lo)
            res["pass" if good else "FAIL"] += 1
            rows.append({"repo":it["repo"],"func":it["func"],"start":s,"passed":good})
        dexec(cid, f"cp /tmp/orig.bak /testbed/{f}", t=60)
        if (i+1) % 20 == 0: print(f"    {i+1}/{len(group)}  pass={res['pass']} FAIL={res['FAIL']}", flush=True)
    sh(["docker","rm","-f",cid], t=120)
    json.dump(rows, open(f"behavior_container_{METHOD}.json","w"))

json.dump(rows, open(f"behavior_container_{METHOD}.json","w"))
g = res["pass"] + res["FAIL"]
print(f"\n===== CONTAINER-SIDE BEHAVIOR, {METHOD} =====")
print(f"  gradable in-container : {g}/{len(todo)} = {100*g/max(len(todo),1):.1f}%   (local-checkout method managed 19/620)")
if g: print(f"  TESTS PASS            : {res['pass']}/{g} = {100*res['pass']/g:.1f}%   (failures {res['FAIL']})")
print(f"  other outcomes: {dict((k,v) for k,v in res.items() if k not in ('pass','FAIL'))}")
print("DONE", flush=True)

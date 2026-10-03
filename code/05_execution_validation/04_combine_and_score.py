"""Combine container-side (extension) and local (tier-A) execution results into ONE exact
execution-validated score over all 584 items, for a given arm. Reports coverage explicitly so any
residual extrapolation is visible rather than hidden."""
import json, os, collections
METHOD = os.environ.get("METHOD", "single_all_loc")
SWEEP  = os.environ.get("SWEEP", "sweep_outputs_p15_loc.json")
K = lambda r: (r["repo"], r["func"], r["start"])

BENCH = {K(x): x for x in json.load(open("verif_benchmark_tiered_ext.json")) if x.get("tier") == "covered"}
preds = {K(o): o for o in json.load(open(SWEEP))[METHOD]}
correct = [k for k in BENCH if k in preds and preds[k].get("correct")]
N = len(BENCH)

ex = {}
for f, tag in ((f"behavior_container_{METHOD}.json", "extension_new"),
               (f"exec_tiera_{METHOD}.json", "original_tier_a")):
    try: rows = json.load(open(f))
    except Exception as e: print(f"  (missing {f}: {e})"); continue
    n = 0
    for r in rows:
        k = K(r)
        if k in BENCH and BENCH[k]["provenance"]["cohort"] == tag:
            ex[k] = bool(r["passed"]); n += 1
    print(f"  {f}: {n} executed rows in-benchmark")

coh = collections.Counter(BENCH[k]["provenance"]["cohort"] for k in correct)
done = [k for k in correct if k in ex]
miss = [k for k in correct if k not in ex]
npass = sum(1 for k in done if ex[k])
nfail = len(done) - npass
rate  = nfail / len(done) if done else 0.0

print(f"\n===== EXECUTION-VALIDATED SCORE, {METHOD}, over {N} items =====")
print(f"  static grader-correct : {len(correct)}/{N} = {100*len(correct)/N:.1f}%   {dict(coh)}")
print(f"  executed              : {len(done)}/{len(correct)} = {100*len(done)/len(correct):.0f}% coverage")
print(f"    pass {npass}   fail {nfail}   false-accept {100*rate:.1f}%")
for c in ("extension_new", "original_tier_a"):
    d = [k for k in done if BENCH[k]["provenance"]["cohort"] == c]
    if d:
        f_ = sum(1 for k in d if not ex[k])
        print(f"    {c:<16} executed {len(d):>3}  fail {f_:>3}  = {100*f_/len(d):.1f}%")
print(f"  not executed          : {len(miss)}")

low  = npass
high = npass + len(miss)
mid  = npass + len(miss) * (1 - rate)
print(f"\n  VALIDATED SCORE : {100*mid/N:.1f}%   (bounds {100*low/N:.1f}% - {100*high/N:.1f}%)")
print(f"  static score    : {100*len(correct)/N:.1f}%")
print(f"  drop            : {100*(len(correct)-mid)/N:.1f} points")
if not miss:
    print("\n  >>> EXACT: every correct prediction was executed, no extrapolation. <<<")

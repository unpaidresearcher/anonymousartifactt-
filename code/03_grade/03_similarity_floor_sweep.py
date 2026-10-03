"""SIMFLOOR sensitivity sweep. The anti-degeneracy similarity floor (grade_verif.py SIMFLOOR=0.35) was a
hand-set default with no recorded derivation. This re-grades every stored prediction for the 7 headline arms
on the corrected n=584 set, caching the SIMFLOOR-independent facts (ruff concerns, function-exists, statement
count, SequenceMatcher ratio) so the threshold can be swept for free.

Validation gate: at 0.35 the arm accuracies MUST reproduce headline_584.log or the harness is wrong.
"""
import json, ast, difflib, tempfile, subprocess, os, collections
from multiprocessing import Pool
import grade_verif as G

BENCH = "verif_benchmark_tiered_ext.json"
ARMS = {  # arm -> file
    "ruff_fix":        "sweep_outputs_ext620.json",
    "single_base":     "sweep_outputs_ext620.json",
    "flat_base":       "sweep_outputs_ext620.json",
    "iterate_base":    "sweep_outputs_ext620.json",
    "single_all_base": "sweep_outputs_ext620_allbase.json",
    "verified_base":   "sweep_outputs_p13_verified.json",
    "single_all_loc":  "sweep_outputs_p15_loc.json",
}
PUBLISHED = {"ruff_fix":0.0,"single_base":12.3,"single_all_base":39.2,"flat_base":42.6,
             "iterate_base":42.8,"verified_base":45.9,"single_all_loc":59.9}
KEY = lambda x: (x["repo"], x["file"], x["func"], x["start"], x["end"])

bench = {KEY(x): x for x in json.load(open(BENCH)) if x.get("tier") == "covered"}
assert len(bench) == 584, len(bench)

def facts(job):
    """Everything the correctness rule needs, EXCEPT the SIMFLOOR comparison itself."""
    k, pred_src, concerns, orig_src, func = job
    try:
        ast.parse(pred_src); parses = True
    except Exception:
        parses = False
    fe = parses and G.has_func(pred_src, func)
    ratio = difflib.SequenceMatcher(None, orig_src, pred_src).ratio()
    stmt_ok = G.n_stmts(pred_src) >= 0.5 * max(G.n_stmts(orig_src), 1)
    resolved = parses and not (G.ruff_cats(pred_src) & set(concerns))
    return k, dict(fe=fe, ratio=ratio, stmt_ok=stmt_ok, resolved=resolved)

if __name__ == "__main__":
    cache = {}
    for arm, path in ARMS.items():
        preds = {KEY(x): x for x in json.load(open(path))[arm]}
        jobs = [(k, preds[k]["pred_src"], bench[k]["concerns"], bench[k]["orig_src"], bench[k]["func"])
                for k in bench if k in preds]
        print(f"{arm}: {len(jobs)} items", flush=True)
        with Pool(32) as p:
            cache[arm] = dict(p.map(facts, jobs, chunksize=8))
    json.dump({a: {str(k): v for k, v in d.items()} for a, d in cache.items()},
              open("simfloor_cache.json", "w"))

    def acc(arm, floor):
        d = cache[arm]
        return 100.0 * sum(1 for f in d.values()
                           if f["resolved"] and f["fe"] and f["stmt_ok"] and f["ratio"] >= floor) / len(d)

    print("\n=== VALIDATION at SIMFLOOR=0.35 (must match headline_584.log) ===")
    ok = True
    for arm, want in PUBLISHED.items():
        got = acc(arm, 0.35)
        good = abs(got - want) < 0.15
        ok &= good
        print(f"  {arm:16s} published {want:5.1f}%   recomputed {got:5.1f}%   {'OK' if good else 'MISMATCH'}")
    print("VALIDATION", "PASSED" if ok else "FAILED")

    FLOORS = [0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
    print("\n=== SWEEP ===")
    print("floor  " + "".join(f"{a:>17s}" for a in ARMS))
    for fl in FLOORS:
        print(f"{fl:4.2f}   " + "".join(f"{acc(a, fl):16.1f}%" for a in ARMS))

    print("\n=== headline gap: single_all_loc - single_all_base ===")
    for fl in FLOORS:
        A, B = cache["single_all_loc"], cache["single_all_base"]
        ok_ = lambda f: f["resolved"] and f["fe"] and f["stmt_ok"] and f["ratio"] >= fl
        b = sum(1 for k in A if ok_(A[k]) and not ok_(B[k]))
        c = sum(1 for k in A if not ok_(A[k]) and ok_(B[k]))
        print(f"  floor {fl:4.2f}  gap {acc('single_all_loc',fl)-acc('single_all_base',fl):+5.1f}pt  b={b:3d} c={c:3d}")
    print("DONE")

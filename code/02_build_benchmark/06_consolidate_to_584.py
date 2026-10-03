"""Merge the extension items into the benchmark. Two hazards handled explicitly:

1. DEDUPE. The same function can be mined twice -- once from a new-commit image and once from an old-commit
   image of the same repo, and it may also already exist in the original benchmark. Identity is
   (repo, file, func): the SAME function at two commits is ONE item, not two, even though its source text
   differs. Counting both would inflate n with near-duplicates and correlate train/eval content.
   Preference order: existing benchmark > older commit (larger pool, more pre-modernization concerns).
2. REPO BALANCE. Report the resulting distribution and flag any repo over 35%, since the whole point of
   extending was to break the sympy monoculture (162/230 = 70%).

Writes verif_benchmark_tiered_ext.json (NON-destructive: original file untouched).
"""
import json, glob, collections, os, hashlib

BENCH = "verif_benchmark_tiered.json"
bench = json.load(open(BENCH))
def ident(x): return (x["repo"], x["file"], x["func"])

N_A0 = sum(1 for x in bench if x["tier"] == "covered")   # capture BEFORE promotion mutates bench
# Explicit cohort on EVERY item so a re-runner can separate original-Tier-A / extension / promoted
# without reverse-engineering which provenance keys are present.
for x in bench:
    x.setdefault("provenance", {})["cohort"] = ("original_tier_a" if x["tier"] == "covered"
                                                else "original_tier_b")
by_id = {ident(x): x for x in bench}
seen = set(by_id)
print(f"original benchmark: {len(bench)} items ({sum(1 for x in bench if x['tier']=='covered')} Tier A), "
      f"{len(seen)} distinct (repo,file,func)")

# old-commit files LAST so that on collision the first-seen (new-commit) wins? No -- prefer OLD (bigger pool,
# and its source is the pre-modernization form we actually want). So load _old first.
# ---- RELABEL concerns with the GRADER'S detector before admitting anything ----------------------
# Mining detects violations in FILE context and attributes them by line range; the grader re-runs ruff on the
# EXTRACTED FUNCTION. Rules needing module imports or class context (B028 warnings.warn, UP006/UP007/UP045
# typing, UP008 super(Foo,self)) therefore get recorded but can NEVER be observed as resolved -- which counts
# them as "resolved for free" and INFLATES every method's score. So label each item exactly as the grader will
# see it, and re-apply the admission criteria to the relabeled set. No container needed: orig_src is stored.
import grade_verif
AUTOFIXABLE_ALONE = set()      # a concern is "nonfixable" per-item; recomputed from the surviving rules below
def relabel(it):
    try: cats = grade_verif.ruff_cats(it["orig_src"])
    except Exception: return None
    keep = sorted(set(it["concerns"]) & cats)
    if len(keep) < 2: return None                      # no longer compound under the grader's view
    nf = [c for c in it.get("nonfixable", []) if c in keep]
    if not nf: return None                             # lost its non-autofixable anchor -> ruff --fix could solve it
    it = dict(it); it["concerns"] = keep; it["nonfixable"] = nf
    it.setdefault("provenance", {})["relabeled"] = "grader_detector_isolated_src"
    return it

files = sorted(glob.glob("tier_a_extend_*_old.json")) + sorted(
        f for f in glob.glob("tier_a_extend_*.json") if not f.endswith("_old.json"))
added, dropped, promoted = [], collections.Counter(), collections.Counter()
dropped_relabel = collections.Counter()
dropped_promotion = collections.Counter()
for f in files:
    d = json.load(open(f)); n0 = len(added)
    for it in d:
        k = ident(it)
        if k in seen:
            # A collision with an existing TIER-B item is not a duplicate to discard: Tier B means
            # "no verified test oracle", and the extension just FOUND one. Promote it in place.
            ex = by_id.get(k)
            # a promotion enters the EVAL set, so it must clear the same relabel bar as a new item
            # PROMOTION IS DISABLED. A promoted item keeps the ORIGINAL benchmark's file/start/end/orig_src
            # (mined from a local HEAD checkout) but inherits the EXTENSION's image + test_target. Item and
            # environment then disagree -- the exact property the extension design declares load-bearing --
            # so its "green test at admission" validated the CONTAINER's version of the function, not the
            # version stored in the item. Diagnosed via container-side validation: 15 of 27 ungradable
            # outputs were region mismatches, all promoted items. The SAME 36 items are also the ones that
            # overlap the adapter training pools, so dropping them fixes both defects at once.
            if ex is not None and ex.get("tier") == "ruff_only":
                dropped_promotion[f] += 1; continue
            if False:
                ex["tier"] = "covered"; ex["test_target"] = it["test_target"]
                ex["baseline_secs"] = it["baseline_secs"]
                ex.setdefault("provenance", {}).update(
                    {**it.get("provenance", {}), "promoted_from": "ruff_only", "source_file": f,
                     "cohort": "extension_promoted"})
                promoted[f] += 1
            else:
                dropped[f] += 1
            continue
        it2 = relabel(it)
        if it2 is None: dropped_relabel[f] += 1; continue
        seen.add(k); it2["provenance"]["source_file"] = f
        it2["provenance"]["cohort"] = "extension_new"
        added.append(it2)
    print(f"  {f:44s} +{len(added)-n0:<4} (promoted {promoted[f]}, deduped {dropped[f]}, relabel-dropped {dropped_relabel[f]})")

print(f"\nNEW items after dedupe: {len(added)}  |  PROMOTED ruff_only->covered: {sum(promoted.values())}"
      f"  |  true duplicates dropped: {sum(dropped.values())}"
      f"  |  DROPPED BY RELABEL: {sum(dropped_relabel.values())}"
      f"  |  PROMOTIONS DISABLED (item/env mismatch + train overlap): {sum(dropped_promotion.values())}")
# The CONFIRMATION set must be uniformly scope-matched, so apply the same relabel to the ORIGINAL items
# *in this output file only* -- the pre-registered verif_benchmark_tiered.json is never touched, and the
# pre-registered analysis keeps its own labels (see the sensitivity footnote in CANONICAL_RESULTS).
# NOTE: by this point promotions have already flipped 36 Tier-B items to "covered", so this loop sees
# original-Tier-A AND promoted items. Count them SEPARATELY -- these numbers are quoted in the paper.
rl = collections.Counter(); dr = collections.Counter()
bench_out = []
for x in bench:
    if x["tier"] != "covered":
        bench_out.append(x); continue
    coh = x["provenance"]["cohort"]
    r = relabel(x)
    if r is None:
        dr[coh] += 1                           # not compound under the grader's detector -> excluded
        continue
    if set(r["concerns"]) != set(x["concerns"]): rl[coh] += 1
    r["provenance"]["cohort"] = coh
    bench_out.append(r)
print("\nrelabel applied to the CONFIRMATION set (pre-registered file untouched):")
for coh in sorted(set(rl) | set(dr)):
    print(f"  {coh:22s} relabeled {rl[coh]:>3}   EXCLUDED {dr[coh]:>3}")
out = bench_out + added
cov = [x for x in out if x["tier"] == "covered"]
print(f"Tier A: {N_A0} -> {len(cov)}  (+{len(cov)-N_A0}: {len(added)} new + {sum(promoted.values())} promoted)")

print("\n=== repo distribution of the new Tier A ===")
c = collections.Counter(x["repo"].split("__")[0] for x in cov)
for r, n in c.most_common():
    pct = 100 * n / len(cov)
    print(f"  {r:16s} {n:>4}  {pct:5.1f}%{'   <- OVER 35%, still concentrated' if pct > 35 else ''}")
print(f"  repos: {len(c)}  (was 2)")
print("\n=== cohort breakdown (provenance.cohort) ===")
for k, v in collections.Counter(x["provenance"]["cohort"] for x in cov).most_common():
    print(f"  {k:22s} {v:>4}")

print("\n=== concern / k distribution ===")
print(f"  k: {dict(sorted(collections.Counter(len(x['concerns']) for x in cov).items()))}")
print(f"  concerns: {dict(collections.Counter(cc for x in cov for cc in x['concerns']).most_common())}")
thin = [k for k, v in collections.Counter(cc for x in cov for cc in x["concerns"]).items() if v < 40]
if thin: print(f"  STILL THIN (<40): {thin}")

# ---- CONTAMINATION FLAG vs the ADAPTER TRAINING POOLS -------------------------------------------------
# The pools (train_candidates_*.json) were built from Tier B + XL and verified disjoint from the ORIGINAL
# Tier A. Extending Tier A broke that in two ways: (1) PROMOTION turned Tier-B training candidates into eval
# items; (2) extension items mined fresh from containers can coincide with XL candidates, which never appear
# in the benchmark file and so were invisible to the dedupe. Base-only methods are unaffected, but ANY
# adapter arm on this set must exclude these items or report with/without them.
train_keys = set()
for tf in glob.glob("train_candidates_*.json"):
    for x in json.load(open(tf)):
        train_keys.add((x["repo"], x["file"], x["func"]))
n_ov = 0
for x in out:
    ov = (x["repo"], x["file"], x["func"]) in train_keys
    x.setdefault("provenance", {})["train_overlap"] = ov
    if ov and x["tier"] == "covered": n_ov += 1
print(f"\n=== ADAPTER-TRAINING CONTAMINATION FLAG ===")
print(f"  Tier-A items overlapping an adapter training pool: {n_ov}/{len(cov)}"
      f"  ({100*n_ov/max(len(cov),1):.1f}%)  -> provenance.train_overlap=true")
print(f"  clean subset for ADAPTER arms: {len(cov)-n_ov} items. Base-only arms may use all {len(cov)}.")

json.dump(out, open("verif_benchmark_tiered_ext.json", "w"), indent=1)
print(f"\nwrote verif_benchmark_tiered_ext.json ({len(out)} items) -- original untouched")
print("NOTE: run the sweep with BENCH=verif_benchmark_tiered_ext.json to use it; all pre-registered")
print("      results stay attached to the ORIGINAL 230 unless explicitly re-run and relabeled.")

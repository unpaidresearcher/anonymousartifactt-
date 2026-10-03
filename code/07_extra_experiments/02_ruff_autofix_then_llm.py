"""Does ruff --fix THEN localized-LLM beat localized alone, on cost or accuracy?

Pipeline: run `ruff check --fix` on the original function, then give Claude Sonnet 5 the
ALREADY-FIXED source plus the localized findings of what remains. Grade against the item's
ORIGINAL admitted concerns, so the pipeline only wins if everything is cleared.
Cost is identical to localized, i.e. one generation, since ruff --fix is free.
Paired against Claude's existing localized run on the same items. Env: N(0=all), SEED.
"""
import os, re, json, random, subprocess, tempfile, time, collections
import grade_verif as G
from mc_prompt import mc_user_prompt_localized
import anthropic

KEY = open(os.path.expanduser("~/.anthropic.key")).read().strip()
MODEL = os.environ.get("MODEL", "claude-sonnet-5")
MAXTOK = int(os.environ.get("MAXTOK", "4096"))      # deliberately above the 512 local cap
N = int(os.environ.get("N", "200")); SEED = int(os.environ.get("SEED", "123456"))
_c = anthropic.Anthropic(api_key=KEY)

def call(prompt):
    r = _c.messages.create(model=MODEL, max_tokens=MAXTOK, thinking={"type": "disabled"},
                           messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

def blk(t):
    m = re.findall(r"```(?:python)?\s*(.*?)```", t, re.S); return (m[-1].strip() if m else t.strip())

def ruff_fix(src):
    t = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w"); t.write(src); t.close()
    subprocess.run(["ruff", "check", "--fix", "--select", G.SEL, "--isolated", "--no-cache", t.name],
                   capture_output=True)
    out = open(t.name).read(); os.unlink(t.name); return out

K = lambda r: (r["repo"], r["func"], r["start"])
B = [x for x in json.load(open("verif_benchmark_tiered_ext.json")) if x.get("tier") == "covered"]
LOC = {K(r): r for r in json.load(open("frontier_outputs_sonnet5.json"))["localized"]}
random.seed(SEED); items = random.sample(B, N) if N else B
print(f"ruff-fix-then-LLM vs localized | {MODEL} | n={len(items)} | max_tokens={MAXTOK}", flush=True)

res, rows, t0 = collections.Counter(), [], time.time()
for i, it in enumerate(items, 1):
    fixed = ruff_fix(it["orig_src"])
    left = G.ruff_cats(fixed) & set(it["concerns"])
    if not left:                                  # ruff alone already cleared everything
        pred, res["ruff_alone_sufficient"] = fixed, res["ruff_alone_sufficient"] + 1
    else:
        try: pred = blk(call(mc_user_prompt_localized(fixed, sorted(left))))
        except Exception as e:
            res["api_error"] += 1; print(f"  api error {e}", flush=True); continue
    g = G.grade(it, pred)
    res["pipeline_correct"] += bool(g["correct"]); res["n"] += 1
    base = LOC.get(K(it))
    res["localized_correct"] += bool(base and base.get("correct"))
    rows.append({"repo": it["repo"].split("__")[0], "func": it["func"], "start": it["start"],
                 "pipeline": bool(g["correct"]), "localized": bool(base and base.get("correct")),
                 "ruff_cleared_all": not left})
    if i % 25 == 0:
        json.dump(rows, open("ruffthen_sonnet5.json", "w"), indent=1)
        print(f"  {i}/{len(items)} pipeline {res['pipeline_correct']} vs localized {res['localized_correct']}"
              f"  ({(time.time()-t0)/60:.1f} min)", flush=True)

json.dump(rows, open("ruffthen_sonnet5.json", "w"), indent=1)
n = max(res["n"], 1)
b = sum(1 for r in rows if r["pipeline"] and not r["localized"])
c = sum(1 for r in rows if r["localized"] and not r["pipeline"])
print(f"\n===== RUFF-FIX THEN LOCALIZED LLM, {MODEL}, n={n} =====")
print(f"  pipeline  : {res['pipeline_correct']}/{n} = {100*res['pipeline_correct']/n:.1f}%")
print(f"  localized : {res['localized_correct']}/{n} = {100*res['localized_correct']/n:.1f}%")
print(f"  net       : {100*(res['pipeline_correct']-res['localized_correct'])/n:+.1f} pts   (b={b} c={c})")
print(f"  ruff --fix alone sufficient on {res['ruff_alone_sufficient']} items")
print(f"  cost      : 1 generation per item for both arms; ruff --fix adds no model call")
print(f"  wall      : {(time.time()-t0)/60:.1f} min")
print("DONE", flush=True)

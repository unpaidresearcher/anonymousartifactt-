"""UNIFORM STaR failure-mining for ANY concern (GPU). Generalizes mine_bugbear_star.py.

Why this exists: the ruff-gold recipe trains on a mechanical rewriter's output for EVERY candidate,
including items the base model already handles, which pulls the adapter off-distribution and damaged
3 of 5 adapters on their OWN concern (p2_gate). STaR trains only where the base FAILS, on the model's
own passing samples, and is the recipe behind the one adapter that passed the gate. Applying it to all
five concerns makes "which adapters survive" a property of the concern rather than of our data recipe.

Contamination-safe by construction: candidates are Tier B + dropped-XL, never Tier A.
Env: CONCERN (required), K(4), MAXLINES(100), MOA_BASE.
"""
import os, re, json, collections, torch
os.environ.setdefault("MOA_BASE", "./basemodel_qwen14b_instruct")
from transformers import AutoModelForCausalLM, AutoTokenizer
import grade_verif

CONCERN = os.environ["CONCERN"]
BASE = os.environ.get("MOA_BASE", "./basemodel_qwen14b_instruct")
K = int(os.environ.get("K", "4")); MAXLINES = int(os.environ.get("MAXLINES", "100"))
# identical wording to sweep.py's DESC so the training prompt cannot drift from the eval prompt
DESC = {"cleanup": "remove unused variables and dead code",
        "comprehension": "replace manual accumulation loops with comprehensions",
        "simplify": "simplify redundant/verbose code (nested ifs, redundant booleans, use ternaries/any/all)",
        "modernize": "modernize outdated syntax (use f-strings, modern Python idioms)",
        "bugbear": "fix the likely-bug/bad-practice pattern (e.g. mutable default argument, unnecessary else after return)"}[CONCERN]

def blk(t):
    m = re.findall(r"```(?:python)?\s*(.*?)```", t, re.S)
    return (m[-1].strip() if m else t.strip())

def passes(item, out):
    g = grade_verif.grade(item, out)
    if not (g["parses"] and g["function_exists"] and g["not_gutted"]): return False
    return CONCERN not in grade_verif.ruff_cats(out)

cands = [x for x in json.load(open(f"train_candidates_{CONCERN}.json")) if x["n_lines"] <= MAXLINES]
print(f"{CONCERN}: {len(cands)} candidates <= {MAXLINES} lines, K={K}", flush=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0},
                                             trust_remote_code=True).eval()
print("loaded FROZEN base", flush=True)

def fix(code, temp):
    u = (f"This function should {DESC}, WITHOUT changing behavior and WITHOUT altering anything else. "
         f"Rewrite it and return the full function in a ```python block.\n\n```python\n{code}\n```")
    p = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    enc = tok(p, return_tensors="pt", truncation=True, max_length=3000).to(model.device)
    kw = dict(max_new_tokens=512, pad_token_id=tok.eos_token_id)
    kw.update(dict(do_sample=False) if temp == 0 else dict(do_sample=True, temperature=temp, top_p=0.95))
    with torch.no_grad(): o = model.generate(**enc, **kw)
    return blk(tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))

prov = collections.Counter(); pairs = []; reps = collections.Counter()
for i, item in enumerate(cands):
    it = {k: item[k] for k in ("repo", "file", "func", "start", "end", "concerns", "orig_src")}
    prov["seen"] += 1
    if passes(it, fix(item["orig_src"], 0)):
        prov["base_right_skip"] += 1; continue      # base already fixes it -> no training value
    prov["base_wrong"] += 1
    trace = None
    for _ in range(K):
        s = fix(item["orig_src"], 0.8)
        if passes(it, s): trace = s; break
    if trace:
        prov["with_trace"] += 1; reps[item["repo"].split("__")[0]] += 1
        pairs.append({"repo": item["repo"], "file": item["file"], "func": item["func"],
                      "defective": item["orig_src"], "fix": trace})
    else:
        prov["no_trace"] += 1
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(cands)}: pairs={len(pairs)} base_right={prov['base_right_skip']} "
              f"base_wrong={prov['base_wrong']}", flush=True)

json.dump(pairs, open(f"{CONCERN}_star_pairs.json", "w"), indent=1)
json.dump({"pool": f"TierB+XL {CONCERN}, <={MAXLINES} lines", "counts": dict(prov),
           "pairs": len(pairs), "repos": dict(reps), "K": K, "recipe": "STaR"},
          open(f"train_provenance_{CONCERN}_star.json", "w"), indent=1)
print(f"\n{CONCERN} STaR: {len(pairs)} pairs  {dict(prov)}\nDONE", flush=True)

"""P8 — PER-ITEM ORDER DEPENDENCE on Tier A (the missing half of the interaction pairing).

For each Tier-A item, apply `flat` twice with the concern order FORWARD (sorted) vs REVERSED, frozen base, same
grader. order_dependent(item) := correct(forward) != correct(reversed). Ships a PER-ITEM order-dependence label
with the benchmark (nobody else does this), and reports a finer signal too: whether the FINAL concern-set differs.

Pre-registered (P8): near-zero (<10%) order dependence. Disconfirming: >=20% => "composition not sequencing" needs
revisiting AND the interaction matrix needs a direction-of-application dimension. Env: MOA_BASE, N(0=all). GPU.
"""
import os, re, ast, json, collections, torch
os.environ.setdefault("MOA_BASE", "./basemodel_qwen14b_instruct")
from transformers import AutoModelForCausalLM, AutoTokenizer
import grade_verif
BASE = os.environ["MOA_BASE"]; N = int(os.environ.get("N", "0")); MXNEW = int(os.environ.get("MXNEW", "512"))
DESC = {"cleanup": "remove unused variables and dead code",
        "comprehension": "replace manual accumulation loops with comprehensions",
        "simplify": "simplify redundant/verbose code (nested ifs, redundant booleans, use ternaries/any/all)",
        "modernize": "modernize outdated syntax (use f-strings, modern Python idioms)",
        "bugbear": "fix the likely-bug/bad-practice pattern (e.g. mutable default argument, unnecessary else after return)"}
def blk(t):
    m = re.findall(r"```(?:python)?\s*(.*?)```", t, re.S); return (m[-1].strip() if m else t.strip())
def parses(s):
    try: ast.parse(s); return True
    except Exception: return False
items = [x for x in json.load(open("verif_benchmark_tiered_ext.json")) if x["tier"] == "covered" and len(x["concerns"]) >= 2]
if N: items = items[:N]
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True).eval()
print(f"order-dependence: {len(items)} Tier-A items (>=2 concerns) | base={BASE}", flush=True)
def fix_one(code, c):
    u = (f"This function should {DESC[c]}, WITHOUT changing behavior and WITHOUT altering anything else. "
         f"Rewrite it and return the full function in a ```python block.\n\n```python\n{code}\n```")
    p = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    enc = tok(p, return_tensors="pt", truncation=True, max_length=3500).to(model.device)
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=MXNEW, do_sample=False, pad_token_id=tok.eos_token_id)
    return blk(tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
def flat(item, order):
    code = item["orig_src"]
    for c in order:
        new = fix_one(code, c)
        if parses(new): code = new
    return code
C = collections.Counter(); byk = collections.defaultdict(collections.Counter); recs = []
try:
    recs = json.load(open("order_dependence584_labels.json"))
    done = {(r["repo"], r["func"], r["start"]) for r in recs}
    for r in recs:
        C["n"] += 1; C["dep"] += r["order_dependent"]; C["setdiff"] += r["final_set_differs"]
        byk[r["k"]]["n"] += 1; byk[r["k"]]["dep"] += r["order_dependent"]
    print(f"resuming, {len(recs)} items already done", flush=True)
except Exception:
    done = set()
items = [x for x in items if (x["repo"].split("__")[0], x["func"], x["start"]) not in done]
for i, it in enumerate(items):
    fwd = sorted(it["concerns"]); rev = list(reversed(fwd))
    of, orv = flat(it, fwd), flat(it, rev)
    gf, gr = grade_verif.grade(it, of), grade_verif.grade(it, orv)
    dep = bool(gf["correct"]) != bool(gr["correct"])
    setf = grade_verif.ruff_cats(of) if gf["parses"] else set(it["concerns"])
    setr = grade_verif.ruff_cats(orv) if gr["parses"] else set(it["concerns"])
    setdiff = setf != setr
    k = len(it["concerns"]); C["n"] += 1; C["dep"] += dep; C["setdiff"] += setdiff
    byk[k]["n"] += 1; byk[k]["dep"] += dep
    recs.append({"repo": it["repo"].split("__")[0], "func": it["func"], "start": it["start"], "k": k,
                 "order_dependent": dep, "final_set_differs": setdiff,
                 "correct_forward": int(gf["correct"]), "correct_reversed": int(gr["correct"])})
    if C["n"] % 20 == 0:
        json.dump(recs, open("order_dependence584_labels.json", "w"), indent=1)
        print(f"  {C['n']}/{len(items)} (order-dependent so far {C['dep']})", flush=True)
json.dump(recs, open("order_dependence584_labels.json", "w"), indent=1)
n = max(C["n"], 1); pct = 100 * C["dep"] // n
print(f"\n===== P8 ORDER DEPENDENCE (Tier A, n={C['n']}) =====")
print(f"  order-dependent (correct flips): {C['dep']}/{C['n']} = {pct}%")
print(f"  final concern-set differs (finer signal): {C['setdiff']}/{C['n']} = {100*C['setdiff']//n}%")
for k in sorted(byk):
    b = byk[k]; print(f"    k={k}: {b['dep']}/{b['n']} = {100*b['dep']//max(b['n'],1)}%")
verdict = ("CONFIRMED near-zero (<10%) -> composition-not-sequencing HOLDS; matrix needs NO order dimension" if pct < 10
           else ("DISCONFIRMED (>=20%) -> revisit composition-not-sequencing; matrix NEEDS a direction-of-application dimension" if pct >= 20
                 else "MIDDLING (10-19%) -> report as a bounded caveat, not a reframe"))
print(f"  => P8 {verdict}")
print("  per-item labels saved -> order_dependence584_labels.json (ships with the benchmark)")
print("DONE", flush=True)

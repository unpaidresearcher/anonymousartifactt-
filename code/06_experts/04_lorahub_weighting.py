"""LoRAHub composition arm (minimal faithful): LoRAHub learns a linear combination of adapters via gradient-free
optimization on a few examples, then applies the combined adapter. With 2 adapters that's a search over 2 weights.
We grid-search (w_bugbear, w_comp) on a HELD-OUT dev set (20 items) to maximize correct under flat structure, build
the best-weighted merged adapter, and eval flat+iterate on the remaining items. Reported honestly — with 2 experts
the search space is tiny, so we expect it to land near base/merge. Env: NDEV(20). GPU."""
import os, re, ast, json, time, random, collections, contextlib, torch
os.environ.setdefault("MOA_BASE","./basemodel_qwen14b_instruct")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import grade_verif
BASE=os.environ["MOA_BASE"]; NDEV=int(os.environ.get("NDEV","20")); MAXPASS=4; random.seed(0)
_def="bugbear=./checkpoints/bugbear_adapter_14b_a2/final_model,comp_rg=./checkpoints/comprehension_ruffgold_adapter_14b/final_model"
ADP=dict(kv.split("=",1) for kv in os.environ.get("ADAPTERS",_def).split(","))
BENCH=os.environ.get("BENCH","verif_benchmark_tiered.json")
OUT=os.environ.get("OUT_TAG","lorahub")
DESC={"cleanup":"remove unused variables and dead code","comprehension":"replace manual accumulation loops with comprehensions",
      "simplify":"simplify redundant/verbose code (nested ifs, redundant booleans, use ternaries/any/all)",
      "modernize":"modernize outdated syntax (use f-strings, modern Python idioms)",
      "bugbear":"fix the likely-bug/bad-practice pattern (e.g. mutable default argument, unnecessary else after return)"}
def blk(t):
    m=re.findall(r"```(?:python)?\s*(.*?)```",t,re.S); return (m[-1].strip() if m else t.strip())
def parses(s):
    try: ast.parse(s); return True
    except: return False
tierA=[x for x in json.load(open(BENCH)) if x["tier"]=="covered"]
random.shuffle(tierA); dev=tierA[:NDEV]; ev=tierA[NDEV:]
print(f"dev={len(dev)} eval={len(ev)}",flush=True)
tok=AutoTokenizer.from_pretrained(BASE,trust_remote_code=True)
_b=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,device_map={"":0},trust_remote_code=True).eval()
_n=list(ADP)
model=PeftModel.from_pretrained(_b,ADP[_n[0]],adapter_name=_n[0]).eval()
for _a in _n[1:]: model.load_adapter(ADP[_a],adapter_name=_a)
print(f"loaded base + {_n}",flush=True)
def build(ws,name):
    try: model.delete_adapter(name)
    except: pass
    model.add_weighted_adapter(_n,list(ws),name,combination_type="linear")
def gen(u,adapter):
    p=tok.apply_chat_template([{"role":"user","content":u}],tokenize=False,add_generation_prompt=True)
    enc=tok(p,return_tensors="pt",truncation=True,max_length=3500).to(model.device)
    if adapter: model.set_adapter(adapter)
    with torch.no_grad(): o=model.generate(**enc,max_new_tokens=512,do_sample=False,pad_token_id=tok.eos_token_id)
    return blk(tok.decode(o[0][enc["input_ids"].shape[1]:],skip_special_tokens=True))
def fix_one(code,c,adapter):
    return gen(f"This function should {DESC[c]}, WITHOUT changing behavior and WITHOUT altering anything else. "
               f"Rewrite it and return the full function in a ```python block.\n\n```python\n{code}\n```",adapter)
def flat(item,adapter):
    code=item["orig_src"]
    for c in sorted(item["concerns"]):
        new=fix_one(code,c,adapter)
        if parses(new): code=new
    return code
def iterate(item,adapter):
    code=item["orig_src"]
    for _ in range(MAXPASS):
        cats=sorted(grade_verif.ruff_cats(code)&set(item["concerns"]))
        if not cats: break
        new=fix_one(code,cats[0],adapter)
        if not parses(new) or (grade_verif.ruff_cats(new)&set(item["concerns"]))>=(grade_verif.ruff_cats(code)&set(item["concerns"])): break
        code=new
    return code
# ---- LoRAHub search: simplex grid over the adapter weights, maximize flat-correct on dev ----
import itertools
def simplex_grid(k):
    """each adapter alone, the uniform blend, and every pair at 0.5/0.5 plus 0.75/0.25 both ways."""
    g=[tuple(1.0 if j==i else 0.0 for j in range(k)) for i in range(k)]
    g.append(tuple(1.0/k for _ in range(k)))
    for i,j in itertools.combinations(range(k),2):
        for wi,wj in ((0.5,0.5),(0.75,0.25),(0.25,0.75)):
            g.append(tuple(wi if x==i else wj if x==j else 0.0 for x in range(k)))
    return list(dict.fromkeys(g))
GRID=simplex_grid(len(_n))
best=None
print(f"== LoRAHub coefficient search on dev, {len(GRID)} points over {_n} ==",flush=True)
for ws in GRID:
    build(ws,"lh"); cor=0
    for it in dev:
        g=grade_verif.grade(it, flat(it,"lh")); cor+=g["correct"]
    print(f"  w={ws} dev flat-correct={100*cor//max(len(dev),1)}%",flush=True)
    if best is None or cor>best[0]: best=(cor,ws)
_,bws=best; print(f"== best weights {dict(zip(_n,bws))} -> eval on {len(ev)} ==",flush=True)
build(bws,"lorahub")
res={m:collections.defaultdict(lambda:collections.Counter()) for m in ("lorahub_flat","lorahub_iterate")}
out={m:[] for m in res}
for i,it in enumerate(ev):
    for m,fn in (("lorahub_flat",flat),("lorahub_iterate",iterate)):
        o=fn(it,"lorahub"); g=grade_verif.grade(it,o)
        res[m][("all",it["repo"])]["n"]+=1; res[m][("all",it["repo"])]["correct"]+=g["correct"]
        for c in it["concerns"]:
            res[m][(c,it["repo"])]["n"]+=1; res[m][(c,it["repo"])]["resolved"]+=(c not in (grade_verif.ruff_cats(o) if g["parses"] else set(it["concerns"])))
        out[m].append({**{q:it[q] for q in ('repo','func','start','concerns')},'pred_src':o,'correct':g['correct']})
    if (i+1)%30==0: print(f"  {i+1}/{len(ev)}",flush=True)
json.dump(out,open(f"lorahub_outputs_{OUT}.json" if OUT!="lorahub" else "lorahub_outputs.json","w"))
reps=set(it["repo"] for it in ev)
print(f"\n===== LoRAHub (best w=({bw1},{bw2})) on n={len(ev)} =====")
for m in res:
    tot=sum(res[m][("all",r)]["n"] for r in reps); cor=sum(res[m][("all",r)]["correct"] for r in reps)
    line=" ".join(f"{r.split('__')[0]}={100*res[m][('all',r)]['correct']//max(res[m][('all',r)]['n'],1)}%" for r in reps)
    bug_n=sum(res[m][("bugbear",r)]["n"] for r in reps); bug_r=sum(res[m][("bugbear",r)]["resolved"] for r in reps)
    print(f"  [{m}] correct={100*cor//max(tot,1)}%  {line}  bugbear-resolved={100*bug_r//max(bug_n,1)}%")
print("  compare: base flat 41 / oracle 48 / uniform-merge 41. LoRAHub expected near merge/base (2-expert search is tiny).")
print("DONE",flush=True)

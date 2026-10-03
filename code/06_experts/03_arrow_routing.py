"""Arrow composition arm (minimal faithful). Arrow (ICML'24) routes to adapters via their OWN prototypes, gradient-
free: no training. Faithful essence for 2 experts = per-input routing by how strongly each adapter's down-projection
(lora_A) FIRES on the input's hidden states. We run the base once per input (output_hidden_states), score each
adapter = sum over q_proj layers of ||A_a · mean-pooled-hidden||, and route (top-1) to the stronger expert, then
flat/iterate with it. SIMPLIFICATION (honest): per-INPUT top-1, not Arrow's per-TOKEN soft mixture — with 2 experts
and known-concern inputs the routing is near-degenerate, as expected; reported as-is. Env: none. GPU."""
import os, re, ast, json, collections, torch
os.environ.setdefault("MOA_BASE","./basemodel_qwen14b_instruct")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import grade_verif
BASE=os.environ["MOA_BASE"]; MAXPASS=4
_def="bugbear=./checkpoints/bugbear_adapter_14b_a2/final_model,comp_rg=./checkpoints/comprehension_ruffgold_adapter_14b/final_model"
ADP=dict(kv.split("=",1) for kv in os.environ.get("ADAPTERS",_def).split(","))
BENCH=os.environ.get("BENCH","verif_benchmark_tiered.json")
OUT=os.environ.get("OUT_TAG","arrow")
DESC={"cleanup":"remove unused variables and dead code","comprehension":"replace manual accumulation loops with comprehensions",
      "simplify":"simplify redundant/verbose code (nested ifs, redundant booleans, use ternaries/any/all)",
      "modernize":"modernize outdated syntax (use f-strings, modern Python idioms)",
      "bugbear":"fix the likely-bug/bad-practice pattern (e.g. mutable default argument, unnecessary else after return)"}
def blk(t):
    m=re.findall(r"```(?:python)?\s*(.*?)```",t,re.S); return (m[-1].strip() if m else t.strip())
def parses(s):
    try: ast.parse(s); return True
    except: return False
items=[x for x in json.load(open(BENCH)) if x["tier"]=="covered"]
import random; random.seed(0); random.shuffle(items)
tok=AutoTokenizer.from_pretrained(BASE,trust_remote_code=True)
_b=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,device_map={"":0},trust_remote_code=True).eval()
_n=list(ADP)
model=PeftModel.from_pretrained(_b,ADP[_n[0]],adapter_name=_n[0]).eval()
for _a in _n[1:]: model.load_adapter(ADP[_a],adapter_name=_a)
# ---- FAITHFUL Arrow prototypes: unit-norm top-right-singular-vector of B·A per q_proj module (no norm bias) ----
Aw={a:{} for a in _n}; Bw={a:{} for a in _n}
for name,p in model.named_parameters():
    if "q_proj" in name and ("lora_A" in name or "lora_B" in name):
        for a in Aw:
            if f".{a}." in name:
                li=int(re.search(r"layers\.(\d+)\.",name).group(1))
                (Aw if "lora_A" in name else Bw)[a][li]=p.detach()
proto={a:{} for a in Aw}                       # unit-norm right singular vector (input space) per layer
for a in Aw:
    for li in Aw[a]:
        if li in Bw[a]:
            M=(Bw[a][li].float() @ Aw[a][li].float())      # out×in, rank<=r
            _,_,V=torch.svd_lowrank(M, q=1)                # V: in×1, unit-norm top right singular vector
            proto[a][li]=V[:,0]
print(f"loaded base + {_n} | SVD prototypes over {len(proto[_n[0]])} q_proj layers",flush=True)
def route(prompt):   # Arrow: align input hidden state with each adapter's UNIT-NORM prototype (magnitude-free)
    enc=tok(prompt,return_tensors="pt",truncation=True,max_length=3500).to(model.device)
    with model.disable_adapter(), torch.no_grad():
        out=_b(**enc,output_hidden_states=True)
    hs=out.hidden_states  # (L+1) x [1,seq,H]
    score={}
    for a,pr in proto.items():
        s=0.0
        for li,v in pr.items():
            h=hs[li].mean(1).squeeze(0).float()            # input to layer li
            hn=h/ (torch.linalg.vector_norm(h)+1e-8)       # normalize input too -> pure direction alignment
            s+=abs(float(hn @ v.to(h.device)))
        score[a]=s
    return max(score,key=score.get), score
def gen(u,adapter):
    p=tok.apply_chat_template([{"role":"user","content":u}],tokenize=False,add_generation_prompt=True)
    enc=tok(p,return_tensors="pt",truncation=True,max_length=3500).to(model.device)
    model.set_adapter(adapter)
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
res={m:collections.defaultdict(lambda:collections.Counter()) for m in ("arrow_flat","arrow_iterate")}
out={m:[] for m in res}; routed=collections.Counter()
for i,it in enumerate(items):
    a,_=route(it["orig_src"]); routed[a]+=1     # Arrow routes per input (once), used for both structures
    for m,fn in (("arrow_flat",flat),("arrow_iterate",iterate)):
        o=fn(it,a); g=grade_verif.grade(it,o)
        res[m][("all",it["repo"])]["n"]+=1; res[m][("all",it["repo"])]["correct"]+=g["correct"]
        for c in it["concerns"]:
            res[m][(c,it["repo"])]["n"]+=1; res[m][(c,it["repo"])]["resolved"]+=(c not in (grade_verif.ruff_cats(o) if g["parses"] else set(it["concerns"])))
        out[m].append({**{q:it[q] for q in ('repo','func','start','concerns')},'pred_src':o,'correct':g['correct'],'routed_to':a})
    if (i+1)%30==0: print(f"  {i+1}/{len(items)} (routing so far: {dict(routed)})",flush=True)
json.dump(out,open(f"arrow_outputs_{OUT}.json" if OUT!="arrow" else "arrow_outputs.json","w"))
reps=set(it["repo"] for it in items)
print(f"\n===== Arrow (per-input prototype routing) n={len(items)} | routed {dict(routed)} =====")
for m in res:
    tot=sum(res[m][("all",r)]["n"] for r in reps); cor=sum(res[m][("all",r)]["correct"] for r in reps)
    line=" ".join(f"{r.split('__')[0]}={100*res[m][('all',r)]['correct']//max(res[m][('all',r)]['n'],1)}%" for r in reps)
    bn=sum(res[m][("bugbear",r)]["n"] for r in reps); br=sum(res[m][("bugbear",r)]["resolved"] for r in reps)
    print(f"  [{m}] correct={100*cor//max(tot,1)}%  {line}  bugbear-resolved={100*br//max(bn,1)}%")
print("  compare: base flat 41 / oracle 48 / uniform-merge 41. Arrow expected near merge/base (2-expert routing degenerate).")
print("DONE",flush=True)

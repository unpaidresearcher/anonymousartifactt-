"""FULL SWEEP on the real benchmark (Tier A + k>=4), 2 adapters (bugbear STaR + comp_rg ruff-gold). Pass-structures x
routers, ruff --fix baseline, per-concern/per-repo (n inline), cost, PER-ITEM OUTPUTS SAVED for every arm. Computes
the P1 clean test: paired iterate-vs-flat b/c on COMPREHENSION items for base AND comp_restricted router (matched
definitions) -> does re-detection's benefit shrink when the CLEAN adapter fixes comprehension (=collateral was the
driver). Env: METHODS(comma-sep or 'essentials'/'full'), NMAX(0=all), MAXPASS(4), SUBSET_BALANCE(1=cap per repo)."""
import os, re, ast, json, time, random, collections, subprocess, tempfile, contextlib, statistics, torch
os.environ.setdefault("MOA_BASE","./basemodel_qwen14b_instruct")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import grade_verif
from mc_prompt import mc_user_prompt, mc_user_prompt_localized, mc_user_prompt_loc_nomsg, mc_user_prompt_loc_desc   # shared so prompts cannot drift
BASE=os.environ["MOA_BASE"]; NMAX=int(os.environ.get("NMAX","0")); MAXPASS=int(os.environ.get("MAXPASS","4")); random.seed(0)
BENCH=os.environ.get("BENCH","verif_benchmark_tiered.json"); MXNEW=int(os.environ.get("MXNEW","512")); MAXLEN=int(os.environ.get("MAXLEN","3500"))
BUG="./checkpoints/bugbear_adapter_14b_a2/final_model"; COMP="./checkpoints/comprehension_ruffgold_adapter_14b/final_model"
# the two adapters the pre-registered gate REJECTED -- used only by the route-everywhere arm (P9)
MODZ="./checkpoints/modernize_ruffgold_adapter_14b/final_model"; CLN="./checkpoints/cleanup_ruffgold_adapter_14b/final_model"
MCRG="./checkpoints/multiconcern_ruffgold_adapter_14b/final_model"   # P10 one-pass multi-concern
# STaR adapter set: the concerns that passed the gate after uniform STaR retraining. Selected with
# METHODS=*_star. cleanup and comprehension are absent because they yielded only 3 STaR pairs each.
STAR={c:f"./checkpoints/{c}_star_adapter_14b/final_model" for c in ("simplify","modernize","bugbear")}
# P12: the SIX task-specialist adapters from the project's prior architecture (different task family:
# compliance / completion / hardware-opt / repair / test-gen / adversarial-breaking).
ADP6={"compliance":"./checkpoints/compliance_real_14b/final_model",
      "completion":"./checkpoints/completion_real_14b/final_model",
      "hwopt":"./checkpoints/hwopt_real_14b/final_model",
      "repair":"./checkpoints/ctxstar_14b/final_model",
      "testgen":"./checkpoints/testgen_real_14b/final_model",
      "breaker":"./checkpoints/breaker_real_14b/final_model"}
DESC={"cleanup":"remove unused variables and dead code",
      "comprehension":"replace manual accumulation loops with comprehensions",
      "simplify":"simplify redundant/verbose code (nested ifs, redundant booleans, use ternaries/any/all)",
      "modernize":"modernize outdated syntax (use f-strings, modern Python idioms)",
      "bugbear":"fix the likely-bug/bad-practice pattern (e.g. mutable default argument, unnecessary else after return)"}
def load_tokenizer(base):
    """AutoTokenizer in transformers 5.x mis-loads some byte-level BPE vocabs (DeepSeek-Coder):
    it substitutes a SentencePiece-style decoder, and BOTH encode and decode then drop every
    space, so `def f(x):` becomes `deff(x):` and nothing parses. Detect it with a round-trip and
    fall back to building the tokenizer straight from tokenizer.json, which is correct."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    import os
    tok = AutoTokenizer.from_pretrained(base)
    probe = "def f(x):\n    return 1\n"
    try:
        ok = tok.decode(tok(probe)["input_ids"], skip_special_tokens=True).strip() == probe.strip()
    except Exception:
        ok = False
    if ok:
        return tok
    tj = os.path.join(base, "tokenizer.json")
    if not os.path.exists(tj):
        raise RuntimeError(f"tokenizer round-trip FAILED for {base} and no tokenizer.json to fall back to")
    fixed = PreTrainedTokenizerFast(tokenizer_file=tj)
    if fixed.decode(fixed(probe)["input_ids"], skip_special_tokens=True).strip() != probe.strip():
        raise RuntimeError(f"tokenizer round-trip FAILED for {base} even from tokenizer.json")
    for a in ("eos_token", "pad_token", "bos_token"):
        if getattr(fixed, a, None) is None and getattr(tok, a, None) is not None:
            setattr(fixed, a, getattr(tok, a))
    if getattr(tok, "chat_template", None):
        fixed.chat_template = tok.chat_template
    print(f"  [tokenizer] AutoTokenizer round-trip failed for {base}; using tokenizer.json directly", flush=True)
    return fixed

def _chat_wrap(tok, u):
    """Apply the model's chat template when it has one, else pass the raw prompt.

    Detection used to test `tok.chat_template`, which is None on MistralCommonBackend even though that
    tokenizer DOES implement apply_chat_template. Mistral models were therefore fed raw, untemplated
    prompts and behaved like base models, returning empty strings on most items. Ask the tokenizer to
    render instead of inspecting an attribute."""
    try:
        p = tok.apply_chat_template([{"role": "user", "content": u}],
                                    tokenize=False, add_generation_prompt=True)
        if isinstance(p, str) and p.strip() and p.strip() != u.strip():
            return p
    except Exception:
        pass
    return u

def blk(t):
    m=re.findall(r"```(?:python)?\s*(.*?)```",t,re.S); return (m[-1].strip() if m else t.strip())
def parses(s):
    try: ast.parse(s); return True
    except: return False
# ---- routers: concern -> adapter name (or None=base) ----
ROUTER={"base":       lambda c: None,
        "oracle":     lambda c: {"bugbear":"bugbear","comprehension":"comp_rg"}.get(c),
        "comp_restr": lambda c: "comp_rg" if c=="comprehension" else None,
        "bug_only":   lambda c: "bugbear" if c=="bugbear" else None,   # fair analogue of LoRAHub w=(1,0): drop comp_rg
        "allroute":   lambda c: {"bugbear":"bugbear","comprehension":"comp_rg",
                                 "modernize":"mod_rg","cleanup":"clean_rg"}.get(c),   # P9: ignore the gate
        "mc":         lambda c: "mc_rg",   # P10: one-pass multi-concern specialist (not per-concern routing)
        # P12 adaptive6: EXACTLY what the trained 6-way router predicts for each concern instruction
        # (deterministic: it embeds the instruction text, not the item). Measured, not chosen by us.
        "adaptive6":  lambda c: {"cleanup":"compliance","comprehension":"breaker","simplify":"breaker",
                                 "modernize":"compliance","bugbear":"compliance"}.get(c),
        # P12 transfer6: the ORACLE-transferable mapping -- the two adapters that plausibly transfer.
        "transfer6":  lambda c: {"comprehension":"hwopt","bugbear":"repair"}.get(c)}
# ---- items ----
tierA=[x for x in json.load(open(BENCH)) if x["tier"]=="covered"]
random.shuffle(tierA)
if os.environ.get("SUBSET_BALANCE","1")=="1" and NMAX:
    per=collections.Counter(); items=[]
    for it in tierA:
        if per[it["repo"]]<NMAX//2: items.append(it); per[it["repo"]]+=1
else:
    items=tierA[:NMAX] if NMAX else tierA
per=collections.Counter(it["repo"] for it in items)
print(f"eval items: {len(items)} | per repo: {dict(per)} | MAXPASS={MAXPASS}",flush=True)
# ---- model + adapters ----
tok=load_tokenizer(BASE)
def _load_base(path):
    """Most bases are plain causal LMs. Some newer families (e.g. Mistral 3) ship only a multimodal
    wrapper class, which AutoModelForCausalLM refuses. Those still generate text normally, so fall back
    to the image-text-to-text auto class rather than skipping the model."""
    kw=dict(dtype=torch.bfloat16, device_map={"":0}, trust_remote_code=True)
    try:
        return AutoModelForCausalLM.from_pretrained(path, **kw).eval()
    except (ValueError, KeyError) as e:
        from transformers import AutoModelForImageTextToText
        print(f"  [loader] AutoModelForCausalLM rejected this config ({type(e).__name__}); "
              f"falling back to AutoModelForImageTextToText", flush=True)
        return AutoModelForImageTextToText.from_pretrained(path, **kw).eval()
_b=_load_base(BASE)
# adapters are Qwen-specific — load ONLY if a selected method needs them (base-only methods run on ANY MOA_BASE model)
_sel=os.environ.get("METHODS","essentials")
if _sel in ("essentials","full"): NEED_ADP=True
else: NEED_ADP=not all(m in {"ruff_fix","single_base","single_fair_base","flat_base","iterate_base","single_all_base","verified_base","single_all_loc","single_all_locnm","single_all_locdesc"} for m in _sel.split(","))
NEED_ALL=any("allroute" in m for m in _sel.split(",")) or _sel=="full"
NEED_MC=any(m=="single_mc" for m in _sel.split(","))
NEED_6=any(("adaptive6" in m) or ("transfer6" in m) for m in _sel.split(","))
NEED_STAR=any("_star" in m for m in _sel.split(","))
if NEED_STAR:
    _sn=list(STAR)
    model=PeftModel.from_pretrained(_b,STAR[_sn[0]],adapter_name=_sn[0]).eval()
    for _a in _sn[1:]: model.load_adapter(STAR[_a],adapter_name=_a)
    ROUTER["star"]=lambda c: c if c in STAR else None          # oracle: concern -> its own specialist
    try:
        model.add_weighted_adapter(_sn,[1.0/len(_sn)]*len(_sn),"star_merge",combination_type="linear")
        ROUTER["star_merge"]=lambda c: "star_merge"
        print(f"built star_merge over {_sn}",flush=True)
    except Exception as e:
        print(f"WARN star_merge failed: {e}",flush=True)
    print(f"loaded base + STaR adapters {_sn}",flush=True)
elif NEED_ADP:
    model=PeftModel.from_pretrained(_b,BUG,adapter_name="bugbear").eval(); model.load_adapter(COMP,adapter_name="comp_rg")
    if NEED_ALL:
        model.load_adapter(MODZ,adapter_name="mod_rg"); model.load_adapter(CLN,adapter_name="clean_rg")
        print("loaded gate-REJECTED adapters mod_rg + clean_rg (P9 route-everywhere arm)",flush=True)
    if NEED_MC:
        model.load_adapter(MCRG,adapter_name="mc_rg")
        print("loaded multi-concern ruff-gold adapter mc_rg (P10 one-pass arm)",flush=True)
    if NEED_6:
        for _n in ("compliance","breaker","hwopt","repair"):     # only those any P12 arm actually routes to
            model.load_adapter(ADP6[_n],adapter_name=_n)
        print("loaded prior-architecture adapters: compliance, breaker, hwopt, repair (P12)",flush=True)
    try:
        model.add_weighted_adapter(["bugbear","comp_rg"],[0.5,0.5],"uniform_merge",combination_type="linear")
        ROUTER["merge"]=lambda c: "uniform_merge"
        print("built uniform_merge adapter (0.5·bugbear + 0.5·comp_rg)",flush=True)
    except Exception as e:
        print(f"WARN uniform_merge failed: {e}",flush=True)
    print("loaded base + bugbear + comp_rg",flush=True)
else:
    model=_b; print(f"BASE-ONLY mode (no adapters) — model={BASE}",flush=True)
def gen(u, adapter, mx=512):
    if adapter and hasattr(model,"set_adapter"): model.set_adapter(adapter)
    p=_chat_wrap(tok,u)
    enc=tok(p,return_tensors="pt",truncation=True,max_length=MAXLEN).to(model.device)
    cm=model.disable_adapter() if (adapter is None and hasattr(model,"disable_adapter")) else contextlib.nullcontext()
    with cm, torch.no_grad(): o=model.generate(**enc,max_new_tokens=MXNEW,do_sample=False,pad_token_id=tok.eos_token_id)
    return blk(tok.decode(o[0][enc["input_ids"].shape[1]:],skip_special_tokens=True))
def fix_one(code, concern, router):
    return gen(f"This function should {DESC[concern]}, WITHOUT changing behavior and WITHOUT altering anything else. "
               f"Rewrite it and return the full function in a ```python block.\n\n```python\n{code}\n```", ROUTER[router](concern))
# ---- pass structures (all take router) ----
def s_single(item, router):   # MoEVD/MoEFix select-one: fix exactly ONE concern
    return fix_one(item["orig_src"], sorted(item["concerns"])[0], router), 1
def fix_one_fair(code, concern, router):
    """Select-one WITHOUT the scope prohibition. The published SE MoE systems route an input to one
    expert and let it work; they never instruct the model to leave other issues alone. Our original
    fix_one appended "WITHOUT altering anything else", which on a >=2-concern item makes perfect
    compliance score zero by construction. This arm keeps behavior preservation and drops the scope
    clause, so it is a faithful instantiation of routing to a single specialist."""
    return gen(f"This function should {DESC[concern]}, WITHOUT changing behavior. "
               f"Rewrite it and return the full function in a ```python block.\n\n```python\n{code}\n```",
               ROUTER[router](concern))
def s_single_fair(item, router):
    return fix_one_fair(item["orig_src"], sorted(item["concerns"])[0], router), 1
def s_flat(item, router):     # fix each concern ONCE, no re-detect (single-pass analog, routed)
    code=item["orig_src"]; passes=0
    for c in sorted(item["concerns"]):
        new=fix_one(code,c,router); passes+=1
        if parses(new): code=new
    return code, passes
def s_iterate(item, router):  # detect -> fix first remaining -> re-detect -> repeat
    code=item["orig_src"]; passes=0
    for _ in range(MAXPASS):
        cats=sorted(grade_verif.ruff_cats(code) & set(item["concerns"]))
        if not cats: break
        new=fix_one(code,cats[0],router); passes+=1
        if not parses(new) or (grade_verif.ruff_cats(new)&set(item["concerns"])) >= (grade_verif.ruff_cats(code)&set(item["concerns"])): break
        code=new
    return code, passes
def s_single_all(item, router):
    """P10 NEW CELL: ONE generation, ALL of the item's concerns named in the prompt. Same oracle knowledge of
    which concerns are present that flat/iterate get. router 'base' -> no adapter (the mandatory control that
    separates a prompt/pass-structure effect from a trained-specialist effect); 'mc' -> the specialist."""
    return gen(mc_user_prompt(item["orig_src"], item["concerns"]), ROUTER[router]("__all__")), 1
def s_verified(item, router):
    """P13: flat + an ACCEPT-OR-REVERT gate. Keep a fix only if it strictly improves (parses, resolves the
    targeted concern, introduces NO new concern, passes the not-gutted guards); else revert and retry once
    with the failure as feedback. The gate -- not the re-detection -- is the new ingredient: `iterate`
    already re-detects with Ruff and only reaches 42%."""
    MAXTRY=int(os.environ.get("MAXTRY","2"))
    code=item["orig_src"]; passes=0
    base_cats=set(item["concerns"])
    for c in sorted(item["concerns"]):
        fb=""
        for _ in range(MAXTRY):
            u=(f"This function should {DESC[c]}, WITHOUT changing behavior and WITHOUT altering anything else."
               f"{fb} Rewrite it and return the full function in a ```python block.\n\n```python\n{code}\n```")
            new=gen(u, ROUTER[router](c)); passes+=1
            if not parses(new):
                fb=" Your previous attempt was not valid Python; return syntactically valid code."; continue
            cats=grade_verif.ruff_cats(new)
            introduced=cats-base_cats
            g=grade_verif.grade({**item,"concerns":[c]}, new)
            if c in cats:
                fb=f" Your previous attempt did not actually fix the issue ({c}); make the change."; continue
            if introduced:
                fb=f" Your previous attempt introduced a new problem ({','.join(sorted(introduced))}); avoid it."; continue
            if not (g["function_exists"] and g["not_gutted"]):
                fb=" Your previous attempt deleted too much; keep the function body intact."; continue
            code=new; break        # accepted
        # not accepted after MAXTRY -> revert (code unchanged)
    return code, passes
def s_single_all_loc(item, router):
    """P15: IDENTICAL to s_single_all (one pass, all concerns named) except the prompt also gives each issue's
    LINE NUMBER and offending source line. Localization is the ONLY variable versus `single_all_base`."""
    return gen(mc_user_prompt_localized(item["orig_src"], item["concerns"]), ROUTER[router]("__all__")), 1
def s_single_all_locnm(item, router):
    """Localization ABLATION: same as s_single_all_loc but the analyzer's MESSAGE is stripped, leaving
    line number + rule code + offending line. Isolates location from prescription."""
    return gen(mc_user_prompt_loc_nomsg(item["orig_src"], item["concerns"]), ROUTER[router]("__all__")), 1
def s_single_all_locdesc(item, router):
    """Property-naming variant: location + a DESCRIPTIVE rule label, never the remedy."""
    return gen(mc_user_prompt_loc_desc(item["orig_src"], item["concerns"]), ROUTER[router]("__all__")), 1
def s_ruff(item, router):     # ruff --fix (safe) baseline — GPU-free, no model
    t=tempfile.NamedTemporaryFile(suffix=".py",delete=False,mode="w");t.write(item["orig_src"]);t.close()
    subprocess.run(["ruff","check","--fix","--isolated","--no-cache",t.name],capture_output=True)
    o=open(t.name).read(); os.unlink(t.name); return o, 0
# name -> (structure_fn, router)
ALL_METHODS={
 "ruff_fix":       (s_ruff,"base"),
 "single_base":    (s_single,"base"),
 "single_fair_base":(s_single_fair,"base"),  # select-one WITHOUT the scope prohibition
 "flat_star":      (s_flat,"star"),          # oracle routing over the 3 gate-passing STaR adapters
 "iterate_star":   (s_iterate,"star"),
 "flat_star_merge":(s_flat,"star_merge"),    # uniform merge of the same 3
 "iterate_star_merge":(s_iterate,"star_merge"),
 "flat_base":      (s_flat,"base"),
 "iterate_base":   (s_iterate,"base"),
 "single_oracle":  (s_single,"oracle"),      # select-one upper bound (adapter for that concern)
 "flat_oracle":    (s_flat,"oracle"),
 "iterate_oracle": (s_iterate,"oracle"),
 "flat_comp":      (s_flat,"comp_restr"),    # P1 clean pair (comprehension->adapter, else base)
 "iterate_comp":   (s_iterate,"comp_restr"),
 "single_merge":   (s_single,"merge"),       # uniform-merge composition arm (merged adapter fixes every concern)
 "flat_merge":     (s_flat,"merge"),
 "iterate_merge":  (s_iterate,"merge"),
 "flat_bugonly":   (s_flat,"bug_only"),       # routing with ONLY bugbear (comp_rg dropped) — fair vs LoRAHub w=(1,0)
 "iterate_bugonly":(s_iterate,"bug_only"),
 "single_all_base":(s_single_all,"base"),
 "single_all_loc": (s_single_all_loc,"base"),
 "single_all_locnm": (s_single_all_locnm,"base"),
 "single_all_locdesc": (s_single_all_locdesc,"base"),   # ablation: location without the message  # P15: same, but told WHERE each issue is    # P10 CONTROL: one-pass all-concern PROMPT on the base model
 "single_mc":      (s_single_all,"mc"),      # P10: one-pass multi-concern SPECIALIST
 "verified_base":  (s_verified,"base"),   # P13: flat + accept-or-revert gate
 "flat_adaptive6": (s_flat,"adaptive6"),     # P12: the trained 6-way router's ACTUAL choices
 "flat_transfer6": (s_flat,"transfer6"),     # P12: oracle-transferable mapping (comprehension->hwopt, bugbear->repair)
 "flat_allroute":  (s_flat,"allroute"),      # P9: ROUTE-EVERYWHERE (adds the 2 gate-failed adapters)
 "single_allroute":(s_single,"allroute"),
 "iterate_allroute":(s_iterate,"allroute"),
}
sel=os.environ.get("METHODS","essentials")
if sel=="essentials": METHODS=["ruff_fix","single_base","iterate_base","iterate_comp"]
elif sel=="full": METHODS=list(ALL_METHODS)
else: METHODS=sel.split(",")
print(f"methods: {METHODS}",flush=True)
res={m:collections.defaultdict(lambda: collections.Counter()) for m in METHODS}
cost={m:{"passes":[],"secs":[]} for m in METHODS}; allout={m:[] for m in METHODS}
for i,it in enumerate(items):
    for m in METHODS:
        fn,router=ALL_METHODS[m]; t0=time.time(); out,passes=fn(it,router); dt=time.time()-t0
        g=grade_verif.grade(it,out); cost[m]["passes"].append(passes); cost[m]["secs"].append(dt)
        k=("all",it["repo"]); res[m][k]["n"]+=1; res[m][k]["correct"]+=g["correct"]
        for c in it["concerns"]:
            kc=(c,it["repo"]); res[m][kc]["n"]+=1
            res[m][kc]["resolved"]+=(c not in (grade_verif.ruff_cats(out) if g["parses"] else set(it["concerns"])))
        allout[m].append({**{q:it[q] for q in ('repo','file','func','start','end','concerns')},'pred_src':out,'correct':g['correct'],'passes':passes})
    if (i+1)%20==0: print(f"  {i+1}/{len(items)} done",flush=True)
json.dump(allout, open(f"sweep_outputs_{os.environ.get('OUT_TAG', sel)}.json","w"))
# ---- report ----
print("\n===== SWEEP (Tier A, grade_verif concerns+guards; behavior 2nd pass) =====")
for m in METHODS:
    tot=sum(res[m][("all",r)]["n"] for r in per); cor=sum(res[m][("all",r)]["correct"] for r in per)
    print(f"\n  [{m}] correct={100*cor//max(tot,1)}%  passes/item={statistics.mean(cost[m]['passes']):.2f}  wall/item={statistics.mean(cost[m]['secs']):.1f}s")
    for r in per:
        n=res[m][("all",r)]["n"]; print(f"      {r.split('__')[0]:8s} n={n:>3} correct={100*res[m][('all',r)]['correct']//max(n,1)}%")
    # partial credit (mean fraction of concerns resolved) — the informative number for ruff_fix
    tc=sum(res[m][(c,r)]["n"] for c in DESC for r in per); rc=sum(res[m][(c,r)]["resolved"] for c in DESC for r in per)
    print(f"      partial-credit (concerns-resolved)={100*rc//max(tc,1)}%")
    for c in DESC:
        n=sum(res[m][(c,r)]["n"] for r in per); rv=sum(res[m][(c,r)]["resolved"] for r in per)
        if n: print(f"      concern {c:13s} resolved={100*rv//n:>3}% (n={n})")
# ---- P1: paired iterate-vs-flat b/c on COMPREHENSION items, base vs comp_restr ----
def paired_bc(iter_m, flat_m):
    if iter_m not in allout or flat_m not in allout: return None
    bi={(o['repo'],o['func'],o['start']):o['correct'] for o in allout[iter_m]}
    bf={(o['repo'],o['func'],o['start']):o['correct'] for o in allout[flat_m]}
    comp=[o for o in allout[iter_m] if 'comprehension' in o['concerns']]
    b=c=0
    for o in comp:
        key=(o['repo'],o['func'],o['start'])
        if bi.get(key) and not bf.get(key): b+=1
        if bf.get(key) and not bi.get(key): c+=1
    return len(comp),b,c
for label,im,fm in [("BASE","iterate_base","flat_base"),("COMP_RESTR(clean adapter)","iterate_comp","flat_comp")]:
    r=paired_bc(im,fm)
    if r: print(f"\n  P1 {label}: comprehension n={r[0]} iterate-only-right b={r[1]} flat-only-right c={r[2]} (gap b-c={r[1]-r[2]})")
print("\n  P1 read: if b-c shrinks BASE->COMP_RESTR, re-detection's benefit was comprehension collateral (mechanism fuses).")
print("DONE",flush=True)

"""GRADER for the verification-only benchmark (with degenerate-output guards). A method's output for an item is
CORRECT iff:
  1. concerns resolved  — target concern categories gone (Ruff-clean, isolated), primary metric;
  2. function-exists    — output parses and contains a FunctionDef of the same name (not deleted);
  3. similarity floor    — output not gutted: SequenceMatcher ratio vs original >= SIMFLOOR AND keeps >= half the
                           statements (blocks the 'empty the body clears all concerns' exploit);
  4. tests pass          — for test-COVERED items, the repo test suite still passes with the output patched in
                           (the real behavioral oracle; handled per-repo by cover_repo infra, hooked here).
`ruff --fix` is reported as a baseline (partial credit by construction). Usage: import grade(), or CLI on a
predictions file [{item..., 'pred_src':...}]."""
import os, ast, json, subprocess, tempfile, difflib, sys, collections
SEL="F841,F811,ERA,C4,PERF,SIM,UP,B,RET,PIE,RSE"
CATS={"cleanup":("F841","F811","ERA"),"comprehension":("C40","C41","PERF401","PERF402","PERF403"),
      "simplify":("SIM",),"modernize":("UP",),"bugbear":("B0","RET","PIE","RSE")}
SIMFLOOR=float(os.environ.get("SIMFLOOR","0.35"))
def cat_of(c):
    for k,ps in CATS.items():
        if any(c.startswith(p) for p in ps): return k
def ruff_cats(src):
    t=tempfile.NamedTemporaryFile(suffix=".py",delete=False,mode="w");t.write(src);t.close()
    out=subprocess.run(["ruff","check",t.name,"--select",SEL,"--output-format","json","--isolated","--no-cache"],
                       capture_output=True,text=True).stdout
    os.unlink(t.name)
    try: v=json.loads(out) if out.strip() else []
    except: v=[]
    return set(filter(None,(cat_of(x["code"]) for x in v)))
def n_stmts(src):
    try: return sum(1 for n in ast.walk(ast.parse(src)) if isinstance(n,ast.stmt))
    except: return 0
def has_func(src, name):
    base=name.split(".")[-1]
    try:
        for n in ast.walk(ast.parse(src)):
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==base: return True
    except: return False
    return False

def grade(item, pred_src, tests_pass=None):
    """returns dict with guard results and 'correct'. tests_pass: True/False/None(=not covered/not run)."""
    g={}
    g["parses"]=False
    try: ast.parse(pred_src); g["parses"]=True
    except: pass
    g["function_exists"]= g["parses"] and has_func(pred_src, item["func"])
    ratio=difflib.SequenceMatcher(None, item["orig_src"], pred_src).ratio()
    g["similarity"]=round(ratio,3)
    g["not_gutted"]= (ratio>=SIMFLOOR) and (n_stmts(pred_src) >= 0.5*max(n_stmts(item["orig_src"]),1))
    g["concerns_left"]= sorted(ruff_cats(pred_src) & set(item["concerns"])) if g["parses"] else item["concerns"]
    g["concerns_resolved"]= (len(g["concerns_left"])==0)
    guards_ok = g["function_exists"] and g["not_gutted"]
    g["tests_pass"]=tests_pass
    # correct = concerns resolved AND guards AND (tests pass if the item is covered)
    g["correct"]= bool(g["concerns_resolved"] and guards_ok and (tests_pass in (True, None)))
    return g

def ruff_fix_baseline(item):
    """partial credit of the free ruff --fix tool: fraction of target categories it clears."""
    t=tempfile.NamedTemporaryFile(suffix=".py",delete=False,mode="w");t.write(item["orig_src"]);t.close()
    subprocess.run(["ruff","check","--fix","--select",SEL,"--isolated","--no-cache",t.name],capture_output=True)
    left=ruff_cats(open(t.name).read()); os.unlink(t.name)
    tgt=set(item["concerns"]); cleared=tgt-left
    return len(cleared)/max(len(tgt),1)

if __name__=="__main__":
    # CLI: grade predictions (or, with --rufffix, report the ruff --fix partial-credit baseline over a set)
    path=sys.argv[1] if len(sys.argv)>1 else "verif_benchmark_final.json"
    data=json.load(open(path))
    if "--rufffix" in sys.argv:
        import statistics
        pc=[ruff_fix_baseline(x) for x in data[:int(os.environ.get('N','200'))]]
        print(f"ruff --fix baseline over n={len(pc)}: mean partial credit = {statistics.mean(pc)*100:.0f}%  "
              f"(fully-solved items: {sum(1 for p in pc if p==1)}/{len(pc)})")
    else:
        ok=sum(grade(x, x.get("pred_src", x["orig_src"]))["correct"] for x in data)
        print(f"graded {len(data)}: correct={ok} (NOTE: default grades orig_src as a sanity check -> should be ~0 since concerns unresolved)")
    print("DONE")

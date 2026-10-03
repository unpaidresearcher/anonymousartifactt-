"""WHY RUFF, CHECKED RATHER THAN ASSERTED.

Reviewers reasonably ask why not Pylint, which is what the closest prior work uses. Two things
disqualify it for THIS benchmark, and both are measurable:
  1. it has no autofixer, so criterion (2) (">=1 concern ruff --fix cannot resolve") is inexpressible;
  2. at FUNCTION scope most of what it reports is not a function-local quality concern at all --
     names defined in the enclosing module become `undefined-variable` the moment the function is
     extracted, which is the same detector-scope trap Section 3.3 describes.

Usage: PYLINT=/path/to/pylint python3 compare_linters.py [N]
"""
import json, subprocess, tempfile, os, sys, collections, time

PYLINT = os.environ.get("PYLINT", "pylint")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SEL = "F841,F811,ERA,C4,PERF,SIM,UP,B,RET,PIE,RSE"
ARTIFACT = {"undefined-variable", "import-error", "used-before-assignment", "no-name-in-module",
            "undefined-loop-variable", "no-member", "relative-beyond-top-level"}
STYLE = {"missing-module-docstring", "missing-function-docstring", "missing-class-docstring",
         "missing-final-newline", "line-too-long", "invalid-name", "trailing-whitespace",
         "trailing-newlines", "bad-indentation", "protected-access", "too-many-arguments",
         "too-many-locals", "too-many-branches", "too-few-public-methods"}

bench = [x for x in json.load(open("verif_benchmark_tiered_ext.json")) if x.get("tier") == "covered"][:N]
paths = []
for x in bench:
    f = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    f.write(x["orig_src"]); f.close(); paths.append(f.name)

t0 = time.time()
r = subprocess.run(["ruff", "check", "--select", SEL, "--output-format", "json",
                    "--isolated", "--no-cache", *paths], capture_output=True, text=True)
t_ruff = time.time() - t0
n_ruff = len(json.loads(r.stdout or "[]"))

t0 = time.time()
p = subprocess.run([PYLINT, "--output-format=json", "--score=n", *paths], capture_output=True, text=True)
t_pyl = time.time() - t0
msgs = json.loads(p.stdout or "[]")
c = collections.Counter(m["symbol"] for m in msgs)
art = sum(v for k, v in c.items() if k in ARTIFACT)
sty = sum(v for k, v in c.items() if k in STYLE)
sem = len(msgs) - art - sty

print(f"{len(bench)} extracted benchmark functions")
print(f"  ruff   : {n_ruff:4d} findings, {t_ruff:5.2f}s")
print(f"  pylint : {len(msgs):4d} findings, {t_pyl:5.2f}s  ({t_pyl/max(t_ruff,1e-9):.0f}x)")
print(f"    scope artifacts of extraction : {art:4d}  {100*art/len(msgs):4.1f}%")
print(f"    style / size conventions      : {sty:4d}  {100*sty/len(msgs):4.1f}%")
print(f"    function-local semantic       : {sem:4d}  {100*sem/len(msgs):4.1f}%")
for q in paths: os.unlink(q)

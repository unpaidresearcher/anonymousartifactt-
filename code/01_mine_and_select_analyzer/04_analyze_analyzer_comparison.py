import json, collections, math, os
raw=json.load(open("rerun_table2_raw.json"))
lines=raw["lines"]
son=json.load(open("sonar_issues_dedup.json"))
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*(c-h),100*(c+h)
# ---- outlier set: the four extraction artifacts
OUT={f"fn_{i:05d}.py" for i in range(4000) if lines[i]>200}
print(f"functions over 200 lines: {len(OUT)}\n")

RCAT={"cleanup":("F841","F811","ERA"),"comprehension":("C40","C41","PERF401","PERF402","PERF403"),
      "simplify":("SIM",),"modernize":("UP",),"bugbear":("B0","RET","PIE","RSE")}
def rcat(c):
    for k,ps in RCAT.items():
        if any(c.startswith(p) for p in ps): return k
PART={"undefined-variable","import-error","used-before-assignment","no-name-in-module",
      "undefined-loop-variable","no-member","relative-beyond-top-level"}
PSTY={"missing-module-docstring","missing-function-docstring","missing-class-docstring",
      "missing-final-newline","line-too-long","invalid-name","trailing-whitespace",
      "trailing-newlines","bad-indentation","protected-access","too-many-arguments",
      "too-many-locals","too-many-branches","too-few-public-methods"}
PCAT={"cleanup":{"unused-variable","unused-import","pointless-statement","pointless-string-statement",
                 "unreachable","unused-private-member"},
      "simplify":{"no-else-return","no-else-raise","no-else-break","no-else-continue",
                  "simplifiable-if-statement","simplifiable-if-expression","consider-using-ternary",
                  "consider-merging-isinstance","consider-using-in","chained-comparison",
                  "simplifiable-condition","consider-using-max-builtin","consider-using-min-builtin",
                  "consider-using-with","unnecessary-negation"},
      "modernize":{"consider-using-f-string","super-with-arguments","consider-using-sys-exit",
                   "useless-object-inheritance","raise-missing-from","consider-using-dict-items",
                   "consider-using-enumerate","deprecated-method","deprecated-module"},
      "comprehension":{"unnecessary-comprehension","consider-using-dict-comprehension",
                       "consider-using-set-comprehension","consider-using-generator",
                       "use-list-literal","use-dict-literal","unnecessary-dict-index-lookup"},
      "bugbear":{"inconsistent-return-statements","dangerous-default-value","broad-exception-caught",
                 "broad-except","bare-except","raising-bad-type","assert-on-tuple",
                 "useless-return","redefined-loop-name"}}
P2C={s:c for c,ss in PCAT.items() for s in ss}
SSTY={"python:S3776","python:S117","python:S1542","python:S1135","python:S1134","python:S107"}
SSCO={"python:S1172","python:S930","python:S1192","python:S5886"}
SCAT={"python:S1481":"cleanup","python:S125":"cleanup","python:S5603":"cleanup",
      "python:S1066":"simplify","python:S1940":"simplify","python:S1871":"simplify","python:S6659":"simplify",
      "python:PrintStatementUsage":"modernize","python:S6660":"modernize",
      "python:S5717":"bugbear","python:S1515":"bugbear","python:S112":"bugbear","python:S5754":"bugbear",
      "python:S3516":"bugbear","python:S2836":"bugbear","python:S1751":"bugbear","python:S3984":"bugbear"}

def report(excl, label):
    n = 4000 - (len(excl))
    print(f"===== {label}  (n={n}) =====")
    rows={}
    # ruff
    rf=[x for x in raw["ruff"] if x["f"] not in excl]
    rc=collections.defaultdict(set)
    for x in rf:
        c=rcat(x["code"])
        if c: rc[x["f"]].add(c)
    comp=sum(1 for v in rc.values() if len(v)>=2); lo,hi=wilson(comp,n)
    rows["Ruff"]=(len(rf),0,0,len(rf),comp,100*comp/n,lo,hi)
    # pylint
    pl=[x for x in raw["pylint"] if x["f"] not in excl]
    art=sum(1 for x in pl if x["sym"] in PART); sty=sum(1 for x in pl if x["sym"] in PSTY)
    sem=len(pl)-art-sty
    pc=collections.defaultdict(set)
    for x in pl:
        c=P2C.get(x["sym"])
        if c: pc[x["f"]].add(c)
    comp=sum(1 for v in pc.values() if len(v)>=2); lo,hi=wilson(comp,n)
    rows["Pylint"]=(len(pl),art,sty,sem,comp,100*comp/n,lo,hi)
    # sonar
    sn=[x for x in son if os.path.basename(x.get("component","")) not in excl]
    art=sum(1 for x in sn if x["rule"] in SSCO); sty=sum(1 for x in sn if x["rule"] in SSTY)
    sem=len(sn)-art-sty
    sc=collections.defaultdict(set)
    for x in sn:
        c=SCAT.get(x["rule"])
        if c: sc[os.path.basename(x["component"])].add(c)
    comp=sum(1 for v in sc.values() if len(v)>=2); lo,hi=wilson(comp,n)
    rows["SonarQube"]=(len(sn),art,sty,sem,comp,100*comp/n,lo,hi)
    print(f"{'':11s}{'find':>8}{'scope':>9}{'style':>9}{'fn-local':>10}{'comp':>6}{'rate':>8}   Wilson")
    for k,(f,a,s,se,c,r,lo,hi) in rows.items():
        ap=f"{a} ({100*a/f:.1f}%)" if f else "-"
        sp=f"{s} ({100*s/f:.1f}%)" if f else "-"
        ep=f"{se} ({100*se/f:.1f}%)" if f else "-"
        print(f"{k:11s}{f:>8}{ap:>14}{sp:>14}{ep:>15}{c:>6}{r:>7.2f}%   [{lo:.2f}, {hi:.2f}]")
    print(f"ruff time {raw['t_ruff']:.2f}s   pylint time {raw['t_pyl']:.0f}s\n")
report(set(), "UNCAPPED, all 4,000")
report(OUT, f"EXCLUDING {len(OUT)} functions over 200 lines")

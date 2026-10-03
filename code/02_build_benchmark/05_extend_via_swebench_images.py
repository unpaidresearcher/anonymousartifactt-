"""EXTEND TIER A using cached SWE-bench Docker images (era-matched envs, already on disk).

Mines FRESH compound items from the image's /testbed (so item and environment are consistent by construction —
promoting our HEAD-mined Tier-B items would mismatch the container's commit), then verifies a behavioral oracle by
running the targeted test file INSIDE the container. Emits new tier="covered" items with full provenance.

Env: REPO, IMAGE, MAXLINES(80), LIMIT(0=all), TEST_TIMEOUT(120), NTRY(3), TAG(output suffix).
"""
import os, random, re, ast, json, subprocess, tempfile, collections, time
os.environ.setdefault("DOCKER_HOST", "unix:///run/user/1009/docker.sock")
REPO = os.environ["REPO"]; IMAGE = os.environ["IMAGE"]
MAXLINES = int(os.environ.get("MAXLINES", "80")); LIMIT = int(os.environ.get("LIMIT", "0"))
MINCATS = int(os.environ.get("MINCATS", "2")); MAXCATS = int(os.environ.get("MAXCATS", "99"))
REQUIRE_NONFIX = os.environ.get("REQUIRE_NONFIX", "1") == "1"
TT = int(os.environ.get("TEST_TIMEOUT", "120"))
CATS = {"cleanup": ("F841", "F811", "ERA"), "comprehension": ("C40", "C41", "PERF401", "PERF402", "PERF403"),
        "simplify": ("SIM",), "modernize": ("UP",), "bugbear": ("B0", "RET", "PIE", "RSE")}
SEL = "F841,F811,ERA,C40,C41,PERF401,PERF402,PERF403,SIM,UP,B0,RET,PIE,RSE"
def cat_of(code):
    for c, pres in CATS.items():
        if any(code.startswith(p) for p in pres): return c
    return None
def sh(cmd, t=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=t)
def dexec(cid, script, t=None):
    return subprocess.run(["docker", "exec", cid, "bash", "-lc", script], capture_output=True, text=True, timeout=t or TT)

# ---------- 1. container up + source out ----------
cid = sh(["docker", "run", "-d", IMAGE, "sleep", "infinity"], t=120).stdout.strip()
assert cid, "container failed to start"
commit = dexec(cid, "git -C /testbed log -1 --format=%H").stdout.strip()
print(f"{REPO}: container {cid[:12]} @ {commit[:10]}", flush=True)
work = tempfile.mkdtemp(prefix=f"ext_{REPO}_")
sh(["docker", "cp", f"{cid}:/testbed", work], t=600)
src_root = os.path.join(work, "testbed")
print(f"  source extracted -> {src_root}", flush=True)

# ---------- 2. mine compound items on the host copy ----------
def ruff_json(path):
    r = sh(["ruff", "check", "--select", SEL, "--output-format", "json", "--isolated", "--no-cache", path])
    try: return json.loads(r.stdout or "[]")
    except Exception: return []
def func_ranges(path):
    try: tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except Exception: return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and getattr(n, "end_lineno", None):
            out.append((n.name, n.lineno, n.end_lineno))
    return out
def ruff_cats_of_src(src, fix=False):
    t = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w"); t.write(src); t.close()
    if fix: sh(["ruff", "check", "--select", SEL, "--fix", "--isolated", "--no-cache", t.name])
    got = {cat_of(v["code"]) for v in ruff_json(t.name) if cat_of(v["code"])}
    os.unlink(t.name); return got

pyfiles = []
for dp, dn, fn in os.walk(src_root):
    # prune by directory BASENAME (an earlier `"/test" in dp` check wrongly matched the /testbed root itself)
    dn[:] = [d for d in dn if d not in {".git", "build", "dist", "node_modules", ".tox", "tests", "test",
                                        "__pycache__", "doc", "docs", ".eggs"}]
    pyfiles += [os.path.join(dp, f) for f in fn if f.endswith(".py")]
print(f"  scanning {len(pyfiles)} non-test .py files", flush=True)

cand = []
for fp in pyfiles:
    viol = ruff_json(fp)
    if len(viol) < 2: continue
    frs = func_ranges(fp)
    if not frs: continue
    per = collections.defaultdict(lambda: {"cats": set(), "rules": set()})
    for v in viol:
        ln = (v.get("location") or {}).get("row")
        c = cat_of(v["code"])
        if not ln or not c: continue
        for (name, s, e) in frs:
            if s <= ln <= e: per[(name, s, e)]["cats"].add(c); per[(name, s, e)]["rules"].add(v["code"]); break
    lines = open(fp, encoding="utf-8", errors="ignore").read().splitlines(keepends=True)
    for (name, s, e), d in per.items():
        # MINCATS/MAXCATS make the CONTROL population minable with the identical pipeline: the discrimination
        # experiment needs single-concern items that differ from the compound set ONLY in concern count.
        if not (MINCATS <= len(d["cats"]) <= MAXCATS) or (e - s + 1) > MAXLINES: continue
        src = "".join(lines[s-1:e])
        import textwrap; src = textwrap.dedent(src)
        left = ruff_cats_of_src(src, fix=True)                          # what safe autofix CANNOT clear
        nonfix = sorted(d["cats"] & left)
        if REQUIRE_NONFIX and not nonfix: continue                     # require >=1 non-autofixable
        cand.append({"repo": REPO, "file": os.path.relpath(fp, src_root), "func": name, "start": s, "end": e,
                     "concerns": sorted(d["cats"]), "nonfixable": nonfix, "rules": sorted(d["rules"]),
                     "n_lines": e - s + 1, "orig_src": src,
                     "provenance": {"image": IMAGE, "commit": commit, "mined": "extend_tier_a"}})
print(f"  candidates ({MINCATS}-{MAXCATS} cats, nonfix_required={REQUIRE_NONFIX}, <={MAXLINES} lines): {len(cand)}", flush=True)
# shuffle BEFORE capping: cand is in file-walk order, so a raw head() would bias the
# sample toward alphabetically-early packages. Seeded => reproducible.
random.Random(0).shuffle(cand)
if LIMIT: cand = cand[:LIMIT]

# ---------- 3. behavioral oracle: does a targeted test file exist AND pass in-container? ----------
# Path-GUESSING fails across repos (pytest keeps its own suite in testing/, pylint names files
# unittest_*.py), so DISCOVER the layout instead: index every test file in the image once, then match
# a source module by basename convention and rank by directory proximity to the source. Repo-agnostic,
# and one docker call per repo instead of four per candidate.
_all = dexec(cid, "find /testbed -name .git -prune -o -name '*.py' -print", t=120).stdout.split()
TESTFILES = [f[len("/testbed/"):] for f in _all
             if re.match(r"(test_.+|.+_test|unittest_.+)\.py$", os.path.basename(f))]
_by_base = collections.defaultdict(list)
for tf in TESTFILES:
    b = os.path.basename(tf)[:-3]
    for stem in (b[5:] if b.startswith("test_") else None,
                 b[:-5] if b.endswith("_test") else None,
                 b[9:] if b.startswith("unittest_") else None):
        if stem: _by_base[stem].append(tf)
print(f"  indexed {len(TESTFILES)} test files ({len(_by_base)} distinct module stems)", flush=True)

def target_candidates(relfile):
    """Test files plausibly covering relfile, nearest-to-the-source first."""
    parts = relfile[:-3].split(os.sep); base = parts[-1]
    keys = [base]                                        # nodes.py     -> test_nodes.py
    if len(parts) > 1:
        keys.append(f"{parts[-2]}_{base}")               # util/nodes.py -> test_util_nodes.py
        keys.append(parts[-2])                           # util/nodes.py -> test_util.py (package-level)
    srcdir = os.path.dirname(relfile).split(os.sep)
    def prox(tf):                                        # shared leading dir components with the source
        td = os.path.dirname(tf).split(os.sep); n = 0
        for a, b in zip(srcdir, td):
            if a != b: break
            n += 1
        return -n
    seen = []
    for k in keys:
        for tf in sorted(_by_base.get(k, []), key=prox):
            if tf not in seen: seen.append(tf)
    return seen
kept = []; stats = collections.Counter()
NTRY = int(os.environ.get("NTRY", "3"))          # candidates to try before giving up on an item
for it in cand:
    cands = target_candidates(it["file"])[:NTRY]
    if not cands: stats["no_test_file"] += 1; continue
    tgt = None; secs = 0.0
    for g in cands:                              # paths came from find, so they exist; the question is
        t0 = time.time()                         # whether the suite is GREEN at this commit
        try: r = dexec(cid, f"source /opt/miniconda3/bin/activate testbed && cd /testbed && timeout {TT} python -m pytest {g} -x -q 2>&1 | tail -3", t=TT + 30)
        except subprocess.TimeoutExpired: stats["timeout"] += 1; continue
        lo = r.stdout.lower()
        if re.search(r"\d+ passed", lo) and not re.search(r"\d+ (failed|error)", lo):
            tgt = g; secs = round(time.time() - t0, 1); break
    if not tgt: stats["baseline_fail"] += 1; continue
    it["tier"] = "covered"; it["test_target"] = tgt; it["baseline_secs"] = secs
    kept.append(it); stats["kept"] += 1
    if stats["kept"] % 10 == 0: print(f"    verified {stats['kept']} (skips {dict(stats)})", flush=True)

TAG = os.environ.get("TAG", "")          # distinguishes runs of the SAME repo at different commits
out = f"tier_a_extend_{REPO}{TAG}.json"
json.dump(kept, open(out, "w"), indent=1)
print(f"\n===== {REPO}: NEW Tier-A items = {len(kept)} =====")
print(f"  funnel: {len(pyfiles)} files -> {len(cand)} compound -> {len(kept)} with verified oracle | skips {dict(stats)}")
if kept:
    print(f"  concern mix: {dict(collections.Counter(c for k in kept for c in k['concerns']))}")
    print(f"  median baseline test time: {sorted(k['baseline_secs'] for k in kept)[len(kept)//2]}s")
print(f"  saved -> {out}")
sh(["docker", "rm", "-f", cid], t=120)
print("DONE", flush=True)

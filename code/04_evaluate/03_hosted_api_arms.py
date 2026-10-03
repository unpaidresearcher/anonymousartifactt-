"""FRONTIER-MODEL RUNNER. Give it an API key and it produces rows directly comparable to Table 6.

Providers: openai | anthropic | gemini. One model per run.

Everything that could make the comparison unfair is shared with the local arms rather than
reimplemented: prompts come from mc_prompt, code extraction matches sweep.py's blk(), and grading is
grade_verif.grade. The only thing that differs from a local row is who generates the text.

Determinism: temperature 0 where the provider allows it. Hosted models are still not reproducible the
way a local greedy run is, so report these as dated rows and say so.

The earlier Gemini pilot lost ~80% of its items to API errors. This runner retries with exponential
backoff and jitter, checkpoints after every item, and resumes by skipping items already in the output
file, so an interrupted or rate-limited run can simply be restarted.

Env:
  PROVIDER   openai | anthropic | gemini          (required)
  MODEL      provider's model id                   (required)
  API_KEY    the key, or API_KEY_FILE pointing at a file containing it
  ARMS       any of: zero_shot,select_one,select_one_scoped,all_concerns,flat,iterate,verified,localized
             (default: all seven, i.e. exactly the columns of Table 6)
  NMAX       0 = all 584                           (default 0)
  RPM        requests per minute budget            (default 30)
  WORKERS    items processed concurrently          (default 8). Items are independent, so this is
             safe; the multi-pass arms stay sequential WITHIN an item, which is what makes
             iterate and verified meaningful.
  OUT_TAG    output goes to frontier_outputs_<tag>.json
  MAXTOK     max output tokens                     (default 1024)
"""
import os, sys, json, re, time, random, collections, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mc_prompt import mc_user_prompt, mc_user_prompt_localized, bare_user_prompt, DESC
import grade_verif as G

PROVIDER = os.environ["PROVIDER"].lower()
MODEL    = os.environ["MODEL"]
ARMS     = os.environ.get("ARMS", "zero_shot,select_one,all_concerns,flat,iterate,verified,localized").split(",")
MAXPASS  = int(os.environ.get("MAXPASS", "4"))
MAXTRY   = int(os.environ.get("MAXTRY", "2"))
NMAX     = int(os.environ.get("NMAX", "0"))
RPM      = float(os.environ.get("RPM", "30"))
WORKERS  = int(os.environ.get("WORKERS", "8"))
MAXTOK   = int(os.environ.get("MAXTOK", "1024"))
BENCH    = os.environ.get("BENCH", "verif_benchmark_tiered_ext.json")
TAG      = os.environ.get("OUT_TAG", f"{PROVIDER}_{MODEL}".replace("/", "_"))
OUT      = f"frontier_outputs_{TAG}.json"
KEY      = os.environ.get("API_KEY") or open(os.environ["API_KEY_FILE"]).read().strip()
GAP      = 60.0 / RPM

def blk(t):
    """Identical to sweep.py's extraction, plus the unterminated-fence guard the pilot needed."""
    m = re.findall(r"```(?:python)?\s*(.*?)```", t, re.S)
    if m: return m[-1].strip()
    m2 = re.search(r"```(?:python)?\s*(.*)", t, re.S)
    return (m2.group(1) if m2 else t).strip()

# ---- one call per provider, each returning plain text -------------------------------------------
if PROVIDER == "openai":
    from openai import OpenAI
    _c = OpenAI(api_key=KEY)
    # GPT-5 is a reasoning model: it rejects temperature and bills reasoning tokens as output.
    # reasoning_effort="minimal" keeps the row comparable to a greedy local decode and keeps cost sane.
    _REASONING = MODEL.startswith(("gpt-5", "o1", "o3", "o4"))
    def call(prompt):
        kw = dict(model=MODEL, max_completion_tokens=MAXTOK,
                  messages=[{"role": "user", "content": prompt}])
        if _REASONING: kw["reasoning_effort"] = os.environ.get("REASONING_EFFORT", "none")
        else:          kw["temperature"] = 0
        r = _c.chat.completions.create(**kw)
        return r.choices[0].message.content or ""
elif PROVIDER == "anthropic":
    import anthropic
    _c = anthropic.Anthropic(api_key=KEY)
    # Sampling params were REMOVED on Sonnet 5 / Opus 5 / Opus 4.7+ -- sending temperature is a 400.
    # Thinking is also billed as output, so we disable it: the local arms are greedy single-shot
    # decodes, and leaving thinking on would compare a reasoning budget against no budget.
    _NO_TEMP = any(t in MODEL for t in ("sonnet-5", "opus-5", "opus-4-7", "opus-4-8", "fable-5", "sonnet-4-6", "opus-4-6"))
    def call(prompt):
        kw = dict(model=MODEL, max_tokens=MAXTOK, messages=[{"role": "user", "content": prompt}])
        if _NO_TEMP:
            kw["thinking"] = {"type": "disabled"}     # accepted on Sonnet 5; keeps output = the code
        else:
            kw["temperature"] = 0
        r = _c.messages.create(**kw)
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
elif PROVIDER == "gemini":
    import urllib.request, urllib.error
    URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}"
    def call(prompt):
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                           "generationConfig": {"temperature": 0, "maxOutputTokens": MAXTOK}}).encode()
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
else:
    sys.exit(f"unknown PROVIDER {PROVIDER!r}, expected openai|anthropic|gemini")

_rate_lock = threading.Lock(); _next_at = [0.0]
def _throttle():
    """One RPM budget shared by every worker."""
    with _rate_lock:
        now = time.time()
        wait = max(0.0, _next_at[0] - now)
        _next_at[0] = max(now, _next_at[0]) + GAP
    if wait: time.sleep(wait)

_io_lock = threading.Lock()

def call_retry(prompt, tries=6):
    """The pilot lost ~80% of items to transient errors. Back off and keep the item instead."""
    for i in range(tries):
        try:
            _throttle()
            return call(prompt), None
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"[:200]
            if i == tries - 1:
                return None, msg
            wait = min(2 ** i * 4, 120) + random.uniform(0, 4)
            print(f"    retry {i+1}/{tries-1} in {wait:.0f}s  ({msg[:90]})", flush=True)
            time.sleep(wait)

def _parses(src):
    import ast
    try: ast.parse(src); return True
    except Exception: return False

def _fix_one(code, concern, scoped=True):
    """Byte-identical to sweep.py's fix_one (scoped) / fix_one_fair (not scoped)."""
    scope = " and WITHOUT altering anything else" if scoped else ""
    return (f"This function should {DESC[concern]}, WITHOUT changing behavior{scope}. "
            f"Rewrite it and return the full function in a ```python block.\n\n"
            f"```python\n{code}\n```")

def run_arm(arm, it):
    """Returns (pred_src, n_calls, err). Single-pass arms cost 1 call; the multi-pass arms
    mirror sweep.py's s_flat / s_iterate / s_verified so the cost column means the same thing."""
    cs = sorted(set(it["concerns"]))

    if arm in ("zero_shot", "all_concerns", "localized", "select_one", "select_one_scoped"):
        if arm == "zero_shot":      pr = bare_user_prompt(it["orig_src"])
        elif arm == "all_concerns": pr = mc_user_prompt(it["orig_src"], it["concerns"])
        elif arm == "localized":    pr = mc_user_prompt_localized(it["orig_src"], it["concerns"])
        # select_one keeps the FAIR prompt these runs originally used. select_one_scoped adds back
        # "and WITHOUT altering anything else", which is what every open-weight row received, so the
        # hosted rows can be compared to them like for like.
        elif arm == "select_one_scoped": pr = _fix_one(it["orig_src"], cs[0], scoped=True)
        else:                       pr = _fix_one(it["orig_src"], cs[0], scoped=False)
        txt, err = call_retry(pr)
        return (None, 1, err) if err else (blk(txt), 1, None)

    if arm == "flat":              # fix each concern once, in order, no re-detection
        code, n = it["orig_src"], 0
        for c in cs:
            txt, err = call_retry(_fix_one(code, c)); n += 1
            if err: return None, n, err
            new = blk(txt)
            if _parses(new): code = new
        return code, n, None

    if arm == "iterate":           # detect -> fix first remaining -> re-detect -> repeat
        code, n = it["orig_src"], 0
        for _ in range(MAXPASS):
            left = sorted(G.ruff_cats(code) & set(it["concerns"]))
            if not left: break
            txt, err = call_retry(_fix_one(code, left[0])); n += 1
            if err: return None, n, err
            new = blk(txt)
            if not _parses(new): break
            if (G.ruff_cats(new) & set(it["concerns"])) >= (G.ruff_cats(code) & set(it["concerns"])): break
            code = new
        return code, n, None

    if arm == "verified":          # flat + accept-or-revert gate, retry once with feedback
        code, n = it["orig_src"], 0
        base = set(it["concerns"])
        for c in cs:
            fb = ""
            for _ in range(MAXTRY):
                pr = _fix_one(code, c).replace("anything else. ", "anything else." + fb + " ")
                txt, err = call_retry(pr); n += 1
                if err: return None, n, err
                new = blk(txt)
                if not _parses(new):
                    fb = " Your previous attempt was not valid Python; return syntactically valid code."; continue
                cats = G.ruff_cats(new)
                g = G.grade({**it, "concerns": [c]}, new)
                if c in cats:
                    fb = f" Your previous attempt did not actually fix the issue ({c}); make the change."; continue
                if cats - base:
                    fb = (f" Your previous attempt introduced a new problem "
                          f"({','.join(sorted(cats - base))}); avoid it."); continue
                if not (g["function_exists"] and g["not_gutted"]):
                    fb = " Your previous attempt deleted too much; keep the function body intact."; continue
                code = new; break
        return code, n, None

    raise SystemExit(f"unknown arm {arm!r}")

K = lambda x: (x["repo"], x["func"], x["start"])
bench = [x for x in json.load(open(BENCH)) if x.get("tier") == "covered"]
if NMAX: bench = bench[:NMAX]

out = json.load(open(OUT)) if os.path.exists(OUT) else {a: [] for a in ARMS}
for a in ARMS: out.setdefault(a, [])
done = {a: {(r["repo"], r["func"], r["start"]) for r in out[a]} for a in ARMS}
todo = sum(1 for a in ARMS for it in bench if K(it) not in done[a])
print(f"{PROVIDER}/{MODEL}: {len(bench)} items x {len(ARMS)} arms, {todo} calls to make "
      f"(resuming, {sum(len(v) for v in out.values())} already done)", flush=True)

errs = collections.Counter(); calls = collections.Counter(); n = 0; t0 = time.time()

def process(arm, it):
    """One item, one arm. Returns a record or None. Safe to run concurrently with other items."""
    pred, ncalls, err = run_arm(arm, it)
    if err is not None:
        return arm, it, None, ncalls, err
    g = G.grade(it, pred)
    rec = {**{q: it[q] for q in ("repo", "file", "func", "start", "end", "concerns")},
           "pred_src": pred, "correct": bool(g["correct"]),
           "n_calls": ncalls, "concerns_left": g["concerns_left"]}
    return arm, it, rec, ncalls, None

for arm in ARMS:
    pending = [it for it in bench if K(it) not in done[arm]]
    if not pending: continue
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for arm_, it, rec, ncalls, err in ex.map(lambda x: process(arm, x), pending):
            n += ncalls; calls[arm] += ncalls
            if err is not None:
                errs[arm] += 1
                print(f"  [{arm}] GAVE UP on {it['repo']}/{it['func']}: {err[:90]}", flush=True)
            else:
                with _io_lock:
                    out[arm].append(rec)
                    if len(out[arm]) % 10 == 0:          # checkpoint in batches, not per item
                        json.dump(out, open(OUT, "w"))
            if n % 40 < ncalls:
                rate = n / max(time.time() - t0, 1) * 60
                print(f"  {n}/{todo} calls  ({rate:.1f}/min)  " +
                      "  ".join(f"{a}:{sum(1 for r in out[a] if r['correct'])}/{len(out[a])}" for a in ARMS),
                      flush=True)
    with _io_lock:
        json.dump(out, open(OUT, "w"))                    # flush at the end of every arm

print(f"\n===== {PROVIDER}/{MODEL} on {len(bench)} items =====")
for a in ARMS:
    v = out[a]; k = sum(1 for r in v if r["correct"])
    miss = len(bench) - len(v)
    gen_per_item = sum(r.get("n_calls", 1) for r in v) / max(len(v), 1)
    print(f"  {a:14s} {k:3d}/{len(v)} = {100*k/max(len(v),1):5.1f}%   {gen_per_item:4.2f} gen/item"
          + (f"   [{miss} items missing, {errs[a]} gave up]" if miss else ""))
print(f"\nwrote {OUT}\nDONE", flush=True)

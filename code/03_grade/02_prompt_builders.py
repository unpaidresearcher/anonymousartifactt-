"""THE one-pass multi-concern prompt, in ONE place.

Training (train_adapter_mc.py) and eval (sweep.py) MUST use a byte-identical prompt; if they drift, the
adapter silently underperforms and we would misread that as the distribution-matching hypothesis failing.
So neither file writes this string -- both import it."""
DESC = {"cleanup": "remove unused variables and dead code",
        "comprehension": "replace manual accumulation loops with comprehensions",
        "simplify": "simplify redundant/verbose code (nested ifs, redundant booleans, use ternaries/any/all)",
        "modernize": "modernize outdated syntax (use f-strings, modern Python idioms)",
        "bugbear": "fix the likely-bug/bad-practice pattern (e.g. mutable default argument, unnecessary "
                   "else after return)"}

def mc_user_prompt(code, concerns):
    """One pass, ALL concerns named. Same oracle knowledge of which concerns are present that flat/iterate
    already get (they call fix_one per known concern), so the comparison is fair."""
    bullets = "\n".join(f"- {DESC[c]}" for c in sorted(concerns))
    return ("This function has several code-quality issues. Fix ALL of them in one rewrite, WITHOUT changing "
            "behavior and WITHOUT altering anything else:\n"
            f"{bullets}\n\n"
            "Rewrite it and return the full function in a ```python block.\n\n"
            f"```python\n{code}\n```")


# ---- P15: LOCALIZED variant -- same one-pass all-concern structure, but the model is TOLD WHERE ----
import subprocess, tempfile, os as _os, json as _json
_SEL = "F841,F811,ERA001,UP,SIM,C4,PERF401,PERF402,PERF403,B0,RET,PIE,RSE"
_CAT = {"F8": "cleanup", "ER": "cleanup", "UP": "modernize", "SI": "simplify",
        "C4": "comprehension", "PE": "comprehension", "B0": "bugbear", "RE": "bugbear",
        "PI": "bugbear", "RS": "bugbear"}

def ruff_violations(src):
    """[(line, code, message)] for the item's concerns, with LINE NUMBERS -- the localization signal."""
    t = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w"); t.write(src); t.close()
    out = subprocess.run(["ruff", "check", t.name, "--select", _SEL, "--output-format", "json",
                          "--isolated", "--no-cache"], capture_output=True, text=True).stdout
    _os.unlink(t.name)
    try: v = _json.loads(out) if out.strip() else []
    except Exception: v = []
    return [((x.get("location") or {}).get("row"), x["code"], x.get("message", "")) for x in v]

def mc_user_prompt_localized(code, concerns):
    """IDENTICAL to mc_user_prompt except each issue is given WITH ITS LINE NUMBER and the offending line.
    The ONLY variable versus the unlocalized arm is location information."""
    lines = code.split("\n")
    viol = [v for v in ruff_violations(code) if v[0]]
    seen = set(); items = []
    for ln, codeid, msg in sorted(viol, key=lambda z: z[0]):
        if (ln, codeid) in seen: continue
        seen.add((ln, codeid))
        src_line = lines[ln-1].strip() if 0 < ln <= len(lines) else ""
        items.append(f"- line {ln} [{codeid}] {msg}    -->  {src_line}")
    if not items:                      # fall back to the unlocalized form if ruff reports nothing
        return mc_user_prompt(code, concerns)
    bullets = "\n".join(f"- {DESC[c]}" for c in sorted(concerns))
    return ("This function has several code-quality issues. Fix ALL of them in one rewrite, WITHOUT changing "
            "behavior and WITHOUT altering anything else.\n\n"
            f"The issues to fix:\n{bullets}\n\n"
            f"They occur at these exact locations:\n" + "\n".join(items) + "\n\n"
            "Rewrite it and return the full function in a ```python block.\n\n"
            f"```python\n{code}\n```")

def mc_user_prompt_loc_nomsg(code, concerns):
    """ABLATION of mc_user_prompt_localized: keeps the line number, the rule code and the offending
    source line, and STRIPS the analyzer's message. Ruff is fix-oriented, so 57.2% of its messages name
    the edit rather than the symptom. This arm separates LOCATION from PRESCRIPTION, which is what decides
    whether the localization result carries to analyzers that report a property instead of an edit."""
    lines = code.split("\n")
    viol = [v for v in ruff_violations(code) if v[0]]
    seen = set(); items = []
    for ln, codeid, _msg in sorted(viol, key=lambda z: z[0]):
        if (ln, codeid) in seen: continue
        seen.add((ln, codeid))
        src_line = lines[ln-1].strip() if 0 < ln <= len(lines) else ""
        items.append(f"- line {ln} [{codeid}]    -->  {src_line}")
    if not items:
        return mc_user_prompt(code, concerns)
    bullets = "\n".join(f"- {DESC[c]}" for c in sorted(concerns))
    return ("This function has several code-quality issues. Fix ALL of them in one rewrite, WITHOUT changing "
            "behavior and WITHOUT altering anything else.\n\n"
            f"The issues to fix:\n{bullets}\n\n"
            f"They occur at these exact locations:\n" + "\n".join(items) + "\n\n"
            "Rewrite it and return the full function in a ```python block.\n\n"
            f"```python\n{code}\n```")

_RULE_NAMES = None
def _rule_name(code):
    """Ruff's own DESCRIPTIVE rule name (e.g. collapsible-if) as opposed to its PRESCRIPTIVE message
    (e.g. 'Use a single if statement instead of nested if statements')."""
    global _RULE_NAMES
    if _RULE_NAMES is None:
        import os, json
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruff_rule_names.json")
        _RULE_NAMES = json.load(open(p))
    return _RULE_NAMES.get(code, "")

def mc_user_prompt_loc_desc(code, concerns):
    """ABLATION between mc_user_prompt_loc_nomsg and mc_user_prompt_localized. Supplies the line, the rule
    code, the offending line AND a DESCRIPTIVE label naming the problem, but never the remedy. This is the
    shape of a property-naming analyzer's output (e.g. a cognitive-complexity threshold), holding coverage
    and location identical to the prescriptive arm, so only the wording varies."""
    lines = code.split("\n")
    viol = [v for v in ruff_violations(code) if v[0]]
    seen = set(); items = []
    for ln, codeid, _msg in sorted(viol, key=lambda z: z[0]):
        if (ln, codeid) in seen: continue
        seen.add((ln, codeid))
        src_line = lines[ln-1].strip() if 0 < ln <= len(lines) else ""
        nm = _rule_name(codeid).replace("-", " ")
        items.append(f"- line {ln} [{codeid}] {nm}    -->  {src_line}")
    if not items:
        return mc_user_prompt(code, concerns)
    bullets = "\n".join(f"- {DESC[c]}" for c in sorted(concerns))
    return ("This function has several code-quality issues. Fix ALL of them in one rewrite, WITHOUT changing "
            "behavior and WITHOUT altering anything else.\n\n"
            f"The issues to fix:\n{bullets}\n\n"
            f"They occur at these exact locations:\n" + "\n".join(items) + "\n\n"
            "Rewrite it and return the full function in a ```python block.\n\n"
            f"```python\n{code}\n```")

def bare_user_prompt(code):
    """NO concern names, NO locations, and NO count. Deliberately does not say "issues" (plural)
    or "all of them": either would tell the model that more than one problem exists, which is the
    very thing the targeting arms supply. The only instructions left are the two that grading
    cannot work without, i.e. return the whole function in a fenced block, and preserve behavior.
    This is what a developer actually types. Every other arm gives the model an oracle list of what
    is wrong; this one gives it nothing, so it measures whether the model finds the problems itself."""
    return ("Improve the code quality of this function, WITHOUT changing behavior and "
            "WITHOUT altering anything else.\n\n"
            "Rewrite it and return the full function in a ```python block.\n\n"
            f"```python\n{code}\n```")

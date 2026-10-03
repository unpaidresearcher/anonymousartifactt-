"""THE BARE ARM -- hand the model the function and nothing else.

Every arm in the paper tells the model which concerns are present; flat/iterate/select-one are even
given them one at a time. That is an oracle a real developer does not have. This arm asks the plain
question ("clean this up") and records, per item, WHICH concerns the model left behind. It exists to
motivate the benchmark: the failure is the model not finishing, not a tool being badly designed.

Env: MOA_BASE, BENCH, NMAX(0=all), OUT_TAG. GPU.
"""
import os, json, re, torch
os.environ.setdefault("MOA_BASE", "./basemodel_qwen14b_instruct")
from transformers import AutoModelForCausalLM, AutoTokenizer
from mc_prompt import bare_user_prompt
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

import grade_verif as G

BASE = os.environ["MOA_BASE"]; BENCH = os.environ.get("BENCH", "verif_benchmark_tiered_ext.json")
NMAX = int(os.environ.get("NMAX", "0")); TAG = os.environ.get("OUT_TAG", "bare")
MXNEW = int(os.environ.get("MXNEW", "512")); MAXLEN = int(os.environ.get("MAXLEN", "3500"))

bench = [x for x in json.load(open(BENCH)) if x.get("tier") == "covered"]
if NMAX: bench = bench[:NMAX]
print(f"bare arm: {len(bench)} items, base={BASE}", flush=True)

tok= load_tokenizer(BASE)

def _load_base(path):
    """Mistral3ForConditionalGeneration is rejected by AutoModelForCausalLM (it is registered as an
    image-text-to-text config even when used text-only). Same fallback sweep.py needed."""
    kw = dict(dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    try:
        return AutoModelForCausalLM.from_pretrained(path, **kw).eval()
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        print("  [loader] AutoModelForCausalLM rejected this config; falling back to AutoModelForImageTextToText", flush=True)
        return AutoModelForImageTextToText.from_pretrained(path, **kw).eval()

model = _load_base(BASE)
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
    m = re.findall(r"```(?:python)?\s*(.*?)```", t, re.S)
    return (m[-1].strip() if m else t.strip())

out = []
for i, it in enumerate(bench, 1):
    _u = bare_user_prompt(it["orig_src"])
    p = _chat_wrap(tok, _u)
    enc = tok(p, return_tensors="pt", truncation=True, max_length=MAXLEN).to(model.device)
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=MXNEW, do_sample=False, pad_token_id=tok.eos_token_id)
    pred = blk(tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
    g = G.grade(it, pred)
    out.append({"repo": it["repo"], "file": it["file"], "func": it["func"], "start": it["start"],
                "concerns": sorted(it["concerns"]), "rules": sorted(it.get("rules", [])),
                "left_behind": g["concerns_left"], "correct": bool(g["correct"]),
                "resolved_all": g["concerns_resolved"], "pred_src": pred})
    if i % 25 == 0:
        ok = sum(1 for x in out if x["correct"])
        print(f"  {i}/{len(bench)}  correct={ok} ({100*ok/i:.1f}%)", flush=True)
        json.dump(out, open(f"bare_outputs_{TAG}.json", "w"))

json.dump(out, open(f"bare_outputs_{TAG}.json", "w"))
ok = sum(1 for x in out if x["correct"])
part = sum(1 for x in out if not x["resolved_all"] and len(x["left_behind"]) < len(x["concerns"]))
none = sum(1 for x in out if len(x["left_behind"]) == len(x["concerns"]))
print(f"\nBARE ARM on {len(out)} items")
print(f"  fully correct                : {ok} ({100*ok/len(out):.1f}%)")
print(f"  fixed SOME, left some behind : {part} ({100*part/len(out):.1f}%)")
print(f"  fixed none                   : {none} ({100*none/len(out):.1f}%)")
print("DONE", flush=True)

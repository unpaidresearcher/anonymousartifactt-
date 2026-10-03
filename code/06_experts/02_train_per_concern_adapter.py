"""SFT a per-concern LoRA adapter on UNIFORM STaR-mined pairs. Generalizes train_bugbear_adapter.py.
frozen — same recipe as the existing 6 adapters. Prompt masked, only the fix tokens supervised. Contamination-safe
by construction (pairs come from Tier B + XL, never Tier A). Env: EPOCHS(3), LR(2e-4), ATTEMPT(1)."""
import os, json, torch
os.environ.setdefault("MOA_BASE","./basemodel_qwen14b_instruct")
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
BASE=os.environ.get("MOA_BASE","./basemodel_qwen14b_instruct")
EPOCHS=float(os.environ.get("EPOCHS","3")); LR=float(os.environ.get("LR","2e-4")); ATTEMPT=os.environ.get("ATTEMPT","1")
CONCERN=os.environ["CONCERN"]
OUT=f"./checkpoints/{CONCERN}_star_adapter_14b/final_model"
DESC={"cleanup":"remove unused variables and dead code","comprehension":"replace manual accumulation loops with comprehensions","simplify":"simplify redundant/verbose code (nested ifs, redundant booleans, use ternaries/any/all)","modernize":"modernize outdated syntax (use f-strings, modern Python idioms)","bugbear":"fix the likely-bug/bad-practice pattern (e.g. mutable default argument, unnecessary else after return)"}[CONCERN]
pairs=json.load(open(f"{CONCERN}_star_pairs.json"))
print(f"training pairs: {len(pairs)} | epochs={EPOCHS} lr={LR} attempt={ATTEMPT} -> {OUT}",flush=True)
tok=AutoTokenizer.from_pretrained(BASE,trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token

def build(ex):
    prompt=tok.apply_chat_template([{"role":"user","content":
        f"This function should {DESC}, WITHOUT changing behavior and WITHOUT altering anything else. "
        f"Rewrite it and return the full function in a ```python block.\n\n```python\n{ex['defective']}\n```"}],
        tokenize=False, add_generation_prompt=True)
    target=f"```python\n{ex['fix']}\n```"+tok.eos_token
    pids=tok(prompt,add_special_tokens=False)["input_ids"]; tids=tok(target,add_special_tokens=False)["input_ids"]
    ids=(pids+tids)[:2048]; labels=([-100]*len(pids)+tids)[:2048]
    return {"input_ids":ids,"labels":labels,"attention_mask":[1]*len(ids)}
data=[build(p) for p in pairs]
class DS(torch.utils.data.Dataset):
    def __len__(self): return len(data)
    def __getitem__(self,i): return data[i]
def collate(b):
    mx=max(len(x["input_ids"]) for x in b); pad=tok.pad_token_id
    import torch as T
    return {"input_ids":T.tensor([x["input_ids"]+[pad]*(mx-len(x["input_ids"])) for x in b]),
            "labels":T.tensor([x["labels"]+[-100]*(mx-len(x["labels"])) for x in b]),
            "attention_mask":T.tensor([x["attention_mask"]+[0]*(mx-len(x["attention_mask"])) for x in b])}

model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,device_map={"":0},trust_remote_code=True)
model=get_peft_model(model, LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
                                       target_modules=["q_proj","k_proj","v_proj","o_proj"]))
model.print_trainable_parameters()
args=TrainingArguments(output_dir=f"./checkpoints/{CONCERN}_star_adapter_14b", num_train_epochs=EPOCHS,
    per_device_train_batch_size=1, gradient_accumulation_steps=8, learning_rate=LR, warmup_steps=3,
    lr_scheduler_type="cosine", logging_steps=5, save_strategy="no", bf16=True, report_to=[])
Trainer(model=model,args=args,train_dataset=DS(),data_collator=collate).train()
model.save_pretrained(OUT)
print(f"saved {CONCERN} STaR adapter -> {OUT}\nDONE",flush=True)

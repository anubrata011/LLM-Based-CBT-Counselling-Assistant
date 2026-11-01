"""
Module 3 - LoRA fine-tuning of a small LLM on CBT dialogues.
Run on a GPU (free Colab T4 is enough):   python -m src.train_generator --epochs 2
Only the LoRA adapter (a few MB) is trained and saved.
"""
import json, argparse, random, os
from src.config import *
from src.prompts import build_messages

def main(epochs=2, bs=8, lr=2e-4, max_len=384, limit=None):
    import torch, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(GENERATOR_BASE)
    model = AutoModelForCausalLM.from_pretrained(GENERATOR_BASE, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])).to(dev)
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open("data/cbt_dialogues.jsonl")]
    random.seed(0); random.shuffle(rows)
    if limit: rows = rows[:limit]
    data = []
    for r in rows:
        stage = 3
        prompt = tok.apply_chat_template(build_messages(r["user"], r["distortion"], stage),
                                         tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(r["assistant"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len]      # loss only on the counsellor reply
        data.append((ids, labels))

    def collate(b):
        m = max(len(x[0]) for x in b)
        ids = torch.tensor([x[0] + [tok.pad_token_id] * (m - len(x[0])) for x in b])
        lab = torch.tensor([x[1] + [-100] * (m - len(x[1])) for x in b])
        return ids, lab, (ids != tok.pad_token_id) | (lab != -100)
    from torch.utils.data import DataLoader
    dl = DataLoader(data, batch_size=bs, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for ep in range(epochs):
        model.train(); tot = 0
        for i, (ids, lab, mask) in enumerate(dl):
            loss = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), labels=lab.to(dev)).loss
            loss.backward(); opt.step(); opt.zero_grad(); tot += loss.item(); losses.append(loss.item())
            if i % 20 == 0: print(f"epoch {ep+1} step {i}/{len(dl)} loss {loss.item():.4f}")
        print(f"== epoch {ep+1} mean loss {tot/len(dl):.4f}")
    os.makedirs(GENERATOR_LORA_DIR, exist_ok=True); model.save_pretrained(GENERATOR_LORA_DIR)
    plt.figure(figsize=(6, 4)); plt.plot(losses); plt.xlabel("step"); plt.ylabel("loss")
    plt.title("LoRA fine-tuning loss"); plt.tight_layout(); plt.savefig("results/generator_loss.png", dpi=150)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="use fewer rows for a quick test")
    a = ap.parse_args(); main(epochs=a.epochs, limit=a.limit)

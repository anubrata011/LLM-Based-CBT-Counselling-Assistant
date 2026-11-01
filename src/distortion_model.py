"""
Module 1 - Cognitive distortion detector.
  * Deep model : DistilBERT fine-tuned for 12-class classification (PyTorch loop)
  * Baseline   : TF-IDF + Logistic Regression (to compare against in the report)

Usage:
    python -m src.make_dataset
    python -m src.distortion_model --baseline         # sklearn baseline
    python -m src.distortion_model --train            # DistilBERT
"""
import json, os, argparse
from src.config import *

def load(split):
    with open(f"data/distortion_{split}.jsonl") as f:
        return [json.loads(l) for l in f]

def train_baseline():
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    tr, te = load("train"), load("test")
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                         LogisticRegression(max_iter=2000, C=5))
    pipe.fit([r["text"] for r in tr], [r["label"] for r in tr])
    acc = pipe.score([r["text"] for r in te], [r["label"] for r in te])
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, BASELINE_PATH)
    print(f"Baseline test accuracy: {acc:.3f}")

def train_bert(epochs=4, bs=32, lr=3e-5, max_len=64):
    import torch, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(CLASSIFIER_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_BASE, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID).to(dev)

    def enc(rows):
        e = tok([r["text"] for r in rows], truncation=True, padding="max_length",
                max_length=max_len, return_tensors="pt")
        y = torch.tensor([LABEL2ID[r["label"]] for r in rows])
        return list(zip(e["input_ids"], e["attention_mask"], y))
    tr = DataLoader(enc(load("train")), batch_size=bs, shuffle=True)
    va = DataLoader(enc(load("val")), batch_size=bs)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * len(tr) * epochs), len(tr) * epochs)
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for ep in range(epochs):
        model.train(); tot = 0
        for ids, mask, y in tr:
            out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), labels=y.to(dev))
            out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(); tot += out.loss.item()
        model.eval(); vl = c = n = 0
        with torch.no_grad():
            for ids, mask, y in va:
                out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), labels=y.to(dev))
                vl += out.loss.item(); c += (out.logits.argmax(-1).cpu() == y).sum().item(); n += len(y)
        hist["train_loss"].append(tot / len(tr)); hist["val_loss"].append(vl / len(va)); hist["val_acc"].append(c / n)
        print(f"epoch {ep+1}: train_loss={hist['train_loss'][-1]:.4f} val_loss={hist['val_loss'][-1]:.4f} val_acc={c/n:.3f}")
    os.makedirs(CLASSIFIER_DIR, exist_ok=True); os.makedirs("results", exist_ok=True)
    model.save_pretrained(CLASSIFIER_DIR); tok.save_pretrained(CLASSIFIER_DIR)
    plt.figure(figsize=(6, 4)); plt.plot(hist["train_loss"], label="train"); plt.plot(hist["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("DistilBERT training curve"); plt.tight_layout()
    plt.savefig("results/training_curve.png", dpi=150)
    json.dump(hist, open("results/training_history.json", "w"), indent=2)

class DistortionClassifier:
    """Loads DistilBERT if trained, else the TF-IDF baseline, else returns 'none'."""
    def __init__(self):
        self.kind = "none"
        if os.path.isdir(CLASSIFIER_DIR):
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.torch = torch
                self.tok = AutoTokenizer.from_pretrained(CLASSIFIER_DIR)
                self.model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_DIR).eval()
                self.kind = "bert"; return
            except Exception as e:
                print("BERT load failed:", e)
        if os.path.exists(BASELINE_PATH):
            import joblib; self.pipe = joblib.load(BASELINE_PATH); self.kind = "baseline"

    def predict(self, text, top_k=2):
        """Returns list of (label, probability)."""
        if self.kind == "bert":
            x = self.tok(text, return_tensors="pt", truncation=True, max_length=64)
            with self.torch.no_grad():
                p = self.torch.softmax(self.model(**x).logits[0], -1)
            top = p.topk(top_k)
            return [(ID2LABEL[i.item()], v.item()) for v, i in zip(top.values, top.indices)]
        if self.kind == "baseline":
            p = self.pipe.predict_proba([text])[0]; cls = self.pipe.classes_
            order = p.argsort()[::-1][:top_k]
            return [(str(cls[i]), float(p[i])) for i in order]
        return [("none", 1.0)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true"); ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()
    if a.baseline: train_baseline()
    if a.train: train_bert(epochs=a.epochs)

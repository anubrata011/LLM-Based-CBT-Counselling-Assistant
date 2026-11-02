
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import CLASSIFIER_DIR, LABELS
from src.distortion_model import load

# Load the trained DistilBERT model
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    CLASSIFIER_DIR
).to(device)

model.eval()

# Load test dataset
test_data = load("test")

texts = [item["text"] for item in test_data]
true_labels = [item["label"] for item in test_data]

# Predict test labels
predictions = []

for text in texts:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_id = outputs.logits.argmax(dim=-1).item()

    predictions.append(model.config.id2label[predicted_id])

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)

macro_f1 = f1_score(
    true_labels,
    predictions,
    labels=LABELS,
    average="macro",
    zero_division=0
)

print("\n===== DistilBERT Test Results =====")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Macro-F1 Score: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    true_labels,
    predictions,
    labels=LABELS,
    zero_division=0
))

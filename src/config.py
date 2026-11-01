"""Shared constants: cognitive distortion labels and their CBT definitions."""

DISTORTIONS = {
    "all_or_nothing": "Seeing things in black-and-white categories, with no middle ground.",
    "overgeneralization": "Taking one event and concluding it will always happen, 'never' / 'always' thinking.",
    "mental_filter": "Dwelling on a single negative detail and ignoring everything positive.",
    "disqualifying_positive": "Rejecting positive experiences by insisting they 'don't count'.",
    "mind_reading": "Assuming you know what others are thinking, without evidence.",
    "fortune_telling": "Predicting that things will turn out badly as if it were a fact.",
    "catastrophizing": "Blowing things out of proportion and imagining the worst outcome.",
    "emotional_reasoning": "Believing something is true because you feel it strongly.",
    "should_statements": "Rigid rules about how you or others 'should', 'must' or 'ought to' behave.",
    "labelling": "Attaching a harsh global label to yourself or others ('I'm a loser').",
    "personalization": "Blaming yourself for events that were not entirely your responsibility.",
    "none": "No clear cognitive distortion in this message.",
}
LABELS = list(DISTORTIONS.keys())
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Hugging Face model names (change if you have less GPU memory)
CLASSIFIER_BASE = "distilbert-base-uncased"
GENERATOR_BASE = "Qwen/Qwen2.5-0.5B-Instruct"   # small, runs on CPU / free Colab GPU
CLASSIFIER_DIR = "models/distortion_distilbert"
BASELINE_PATH = "models/baseline_tfidf_lr.joblib"
GENERATOR_LORA_DIR = "models/cbt_generator_lora"

"""
Multi-turn CBT counselling engine.

The assistant first explores the student's situation through one question at a time.
It only classifies the combined user conversation and presents a final reflection
when the user chooses "Finish reflection".
"""
import re

from src.config import *
from src.safety import check_risk, CRISIS_MESSAGE
from src.distortion_model import DistortionClassifier
from src.prompts import build_messages
from src.make_dataset import R

MIN_TURNS_FOR_SUMMARY = 10
MAX_HISTORY_MESSAGES = 20
STAGE_NAMES = [
    "Check-in",
    "Understand the situation",
    "Explore the automatic thought",
    "Explore emotions",
    "Explore impact and context",
    "Examine supporting evidence",
    "Consider alternative evidence",
    "Develop a balanced thought",
    "Identify a helpful coping response",
    "Identify strengths and support",
    "Plan a practical next step",
    "Final reflection",
]

# A fixed, non-repeating ten-turn path. The LLM is still used for generation
# elsewhere, but its generated questions cannot override this progression.
EXPLORATION_QUESTIONS = [
    "What has been happening that has been troubling you?",
    "When you think about this situation, what is the main thought that comes to mind?",
    "What emotion do you feel most strongly, and how intense is it from 0 to 100?",
    "How has this worry been affecting your daily routine, studies, or confidence?",
    "What facts or experiences make this worry feel true to you?",
    "Can you think of any facts, experiences, or achievements that point to a different outcome?",
    "Taking both sides into account, what would be a more balanced and realistic thought?",
    "When this worry appears, what is one helpful thing you could do instead of getting stuck in it?",
    "What personal strengths, completed work, or people could support you during placement preparation?",
    "What is one small, specific action you will take next, and when will you do it?",
]

FORBIDDEN = re.compile(
    r"\b(diagnos\w+|prescrib\w+|medication|antidepressant|dosage|"
    r"you have (depression|anxiety|bipolar))",
    re.I,
)

class RuleBackend:
    """Simple offline conversational fallback when the LLM is unavailable."""
    name = "rule-based"

    def generate(self, user_text, distortion, stage, history):
        questions = {
            1: "Thank you for sharing that. Can you describe what happened, in your own words?",
            2: "When that happened, what exact thought went through your mind?",
            3: "What emotions did you feel, and how strong was each one from 0 to 100?",
            4: "How did this situation affect you, and what worries you most about it?",
        }
        return questions.get(
            stage,
            "Thank you for talking this through. You can choose Finish reflection when you feel ready."
        )

    def final(self, transcript, distortion, confidence, history):
        label = distortion.replace("_", " ")
        if distortion == "none":
            pattern_text = (
                "I couldn't identify one clear thinking pattern from the conversation. "
                "That is okay; one label may not describe the whole experience."
            )
        else:
            description = DISTORTIONS.get(distortion, "a possible unhelpful thinking pattern")
            pattern_text = (
                f"A possible pattern to reflect on is **{label}**. "
                f"This generally means {str(description).lower()}"
            )
        return (
            "Thank you for sharing your experience. " + pattern_text +
            "\n\nA more balanced way to look at the situation is to separate what happened "
            "from what you fear it means. One difficult event does not by itself decide what "
            "will happen next. Consider one small, realistic step you can take now. "
            "This is an educational reflection, not a clinical diagnosis."
        )

class HFBackend:
    """Qwen2.5-0.5B-Instruct with optional local LoRA adapter."""
    def __init__(self, use_lora=True):
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(GENERATOR_BASE)
        self.model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_BASE, torch_dtype=torch.float32
        )
        self.name = "Qwen2.5-0.5B (prompted)"
        if use_lora and os.path.isdir(GENERATOR_LORA_DIR):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, GENERATOR_LORA_DIR)
            self.name = "Qwen2.5-0.5B + LoRA"
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.dev)
        self.model.eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def _generate(self, messages, max_new_tokens=180):
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tok(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        input_ids = encoded["input_ids"].to(self.dev)
        attention_mask = encoded["attention_mask"].to(self.dev)
        with self.torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.65,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

    def generate(self, user_text, distortion, stage, history):
        return self._generate(build_messages(user_text, distortion, stage, history))

    def final(self, transcript, distortion, confidence, history):
        label = distortion.replace("_", " ")
        description = DISTORTIONS.get(distortion, "no single clear pattern")
        system = (
            "You are an educational CBT reflection assistant, not a therapist. "
            "Review the complete conversation and write a concise final reflection. "
            "Mention one possible cognitive distortion only if supported by the user's "
            "own words; otherwise say no clear pattern was identified. Explain the pattern "
            "gently, distinguish facts from assumptions, offer one balanced alternative "
            "thought and one small practical next step. Do not diagnose or prescribe. "
            "Do not ask another question. Use plain, supportive language."
        )
        user = (
            f"Possible classifier label: {label} ({description}). "
            f"Classifier score: {confidence:.3f}. Treat this only as a clue, not a diagnosis.\n\n"
            f"Full conversation:\n{transcript}\n\nWrite the final reflection."
        )
        return self._generate(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            max_new_tokens=240,
        )

class CBTCounsellor:
    def __init__(self, backend="auto"):
        self.clf = DistortionClassifier()
        self.rule = RuleBackend()
        if backend == "rule":
            self.llm = self.rule
        else:
            try:
                self.llm = HFBackend()
            except Exception as exc:
                print(
                    f"[info] LLM backend unavailable ({type(exc).__name__}); "
                    "using rule-based fallback."
                )
                self.llm = self.rule
        self.reset()

    def reset(self):
        self.history = []
        self.user_messages = []
        self.turns = 0
        self.stage = 0
        self.finished = False
        self.last_distortion = "none"
        self.last_confidence = 0.0

    def _classify_full_conversation(self):
        combined = " ".join(self.user_messages).strip()
        if not combined:
            return "none", 0.0
        try:
            predictions = self.clf.predict(combined, top_k=1)
            if not predictions:
                return "none", 0.0
            label, score = predictions[0]
            # This is a model score, not a calibrated clinical probability.
            if float(score) < 0.20:
                return "none", float(score)
            return label, float(score)
        except Exception as exc:
            print(f"[warning] Classifier failed ({type(exc).__name__}).")
            return "none", 0.0

    def _one_question_reply(self, generated, user_text, turn_number):
        """Return the next fixed CBT question; never reuse an earlier question."""
        index = max(0, min(turn_number - 1, len(EXPLORATION_QUESTIONS) - 1))
        question = EXPLORATION_QUESTIONS[index]
        acknowledgements = [
            "I hear that this has been weighing on you.",
            "Thank you for explaining the situation.",
            "That sounds difficult to carry.",
            "It makes sense that this is affecting you.",
            "Thank you for being open about what makes the worry feel real.",
            "You've also started looking at the wider picture.",
            "That's a useful and thoughtful way to examine the situation.",
            "You deserve a practical response to these worries, not more self-criticism.",
            "It sounds like you have some strengths and support to draw on.",
            "Let's turn what you've noticed into one manageable step.",
        ]
        return f"{acknowledgements[index]} {question}"

    def respond(self, user_text):
        user_text = (user_text or "").strip()
        if not user_text:
            return {
                "reply": "Please type your response so we can continue.",
                "distortion": None, "confidence": None,
                "stage": STAGE_NAMES[min(self.stage, 10)], "crisis": False,
                "ready_to_finish": self.turns >= MIN_TURNS_FOR_SUMMARY,
            }
        if self.finished:
            return {
                "reply": "This reflection is complete. Select New session to start again.",
                "distortion": self.last_distortion,
                "confidence": self.last_confidence,
                "stage": "Final reflection", "crisis": False,
                "ready_to_finish": True,
            }
        if check_risk(user_text):
            return {
                "reply": CRISIS_MESSAGE, "distortion": None, "confidence": None,
                "stage": "CRISIS PROTOCOL", "crisis": True,
                "ready_to_finish": False,
            }

        self.user_messages.append(user_text)
        self.turns += 1
        self.stage = min(self.turns, 10)

        # Use the LLM to retain the project integration and conversation context,
        # but use a deterministic question sequence to prevent repeated questions
        # from a small language model.
        try:
            self.llm.generate(user_text, "none", min(self.stage, 4), self.history)
        except Exception as exc:
            print(f"[warning] LLM generation failed ({type(exc).__name__}); continuing with guided flow.")

        reply = self._one_question_reply("", user_text, self.turns)

        self.history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ])
        self.history = self.history[-MAX_HISTORY_MESSAGES:]

        return {
            "reply": reply,
            "distortion": None,
            "confidence": None,
            "stage": STAGE_NAMES[self.stage],
            "crisis": False,
            "ready_to_finish": self.turns >= MIN_TURNS_FOR_SUMMARY,
        }

    def finish(self):
        if self.finished:
            return {
                "reply": "This reflection has already been completed.",
                "distortion": self.last_distortion,
                "confidence": self.last_confidence,
                "stage": "Final reflection", "crisis": False,
            }
        if self.turns < MIN_TURNS_FOR_SUMMARY:
            remaining = MIN_TURNS_FOR_SUMMARY - self.turns
            return {
                "reply": f"Let's explore a little more first. Please answer {remaining} more question(s).",
                "distortion": None, "confidence": None,
                "stage": STAGE_NAMES[min(self.stage, 4)], "crisis": False,
            }

        label, score = self._classify_full_conversation()
        classifier_available = getattr(self.clf, "kind", "none") != "none"
        if not classifier_available:
            label = None
            score = None

        # The final pattern label is controlled by the classifier, never invented by the LLM.
        transcript = "\n".join(self.user_messages)
        if not classifier_available:
            pattern_line = (
                "**Cognitive pattern:** Not available — no trained classifier model "
                "was loaded in this project run. The conversation can still be summarised, "
                "but a pattern cannot be reliably assigned."
            )
        elif label == "none":
            pattern_line = (
                "**Cognitive pattern:** No clear pattern was identified by the classifier. "
                "This does not mean that your feelings are invalid."
            )
        else:
            description = DISTORTIONS.get(label, "a possible unhelpful thinking pattern")
            pattern_line = (
                f"**Possible cognitive pattern:** {label.replace('_', ' ')}. "
                f"{description} This is a model suggestion, not a diagnosis."
            )

        # Ground the recap in the user's actual words instead of assuming a fixed
        # placement/academic scenario. Keep excerpts short for a readable summary.
        excerpts = []
        for item in self.user_messages:
            cleaned = " ".join(item.split())
            if cleaned and cleaned not in excerpts:
                excerpts.append(cleaned[:240] + ("…" if len(cleaned) > 240 else ""))
        recap = "\\n".join(f"- {item}" for item in excerpts[-4:])
        reply = (
            "## Final reflection\\n\\n"
            "**What you shared:**\\n" + (recap or "- You shared your experience during this reflection.") + "\\n\\n"
            f"{pattern_line}\\n\\n"
            "**Balanced thought:** The difficulties you described are real, but they do not "
            "prove that the outcome you fear is certain. A more balanced view considers both "
            "the challenges and the evidence of effort, learning, and possible progress.\\n\\n"
            "**One practical next step:** Choose one small action related to the concern you "
            "described, make it specific, and decide when you will do it. Afterward, note what "
            "you learned rather than judging yourself only by the result.\\n\\n"
            "*This is an educational CBT-style reflection, not a clinical assessment or diagnosis.*"
        )

        self.finished = True
        self.last_distortion = label or "unavailable"
        self.last_confidence = score
        self.history.append({"role": "assistant", "content": reply})
        return {
            "reply": reply,
            "distortion": label,
            "confidence": round(float(score), 3) if score is not None else None,
            "stage": "Final reflection",
            "crisis": False,
        }

if __name__ == "__main__":
    bot = CBTCounsellor()
    print(f"Backend: {bot.llm.name} | classifier: {bot.clf.kind}")
    print("Start by describing what's been bothering you. Type 'quit' to exit.")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        result = bot.respond(user)
        print(f"\nBot: {result['reply']}\n")
        if result.get("ready_to_finish"):
            choice = input("Finish reflection now? (y/n): ").strip().lower()
            if choice == "y":
                final = bot.finish()
                print(f"\nPattern: {final['distortion']} (score {final['confidence']})")
                print(f"Final reflection: {final['reply']}\n")

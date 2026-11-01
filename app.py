"""Gradio interface for the multi-turn CBT Counselling Assistant."""
import gradio as gr
from src.cbt_engine import CBTCounsellor, MIN_TURNS_FOR_SUMMARY

bot = CBTCounsellor()

DISCLAIMER = (
    "**Disclaimer:** This is an academic project using AI and CBT techniques. "
    "It is NOT a substitute for a licensed professional. If you are in crisis, "
    "call Tele-MANAS 14416 or emergency services 112."
)
WELCOME = (
    "Hi! I'm here to listen and help you reflect on what's been happening. "
    "This is a step-by-step conversation, and I'll ask one question at a time. "
    "What's been bothering you lately?"
)

def chat(message, history):
    message = (message or "").strip()
    history = list(history or [])
    if not message:
        return history, "", "Please enter a message.", gr.update(interactive=False)

    result = bot.respond(message)
    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": result["reply"]},
    ])

    if result["crisis"]:
        info = "**CRISIS PROTOCOL ACTIVATED**"
        finish_update = gr.update(interactive=False)
    else:
        info = (
            f"**Stage:** {result['stage']}\n\n"
            f"**Conversation turns:** {bot.turns}\n\n"
            f"**Pattern detection:** Delayed until final reflection (after 10 user turns)."
        )
        finish_update = gr.update(interactive=bot.turns >= MIN_TURNS_FOR_SUMMARY)
    return history, "", info, finish_update

def finish_reflection(history):
    history = list(history or [])
    result = bot.finish()
    history.append({"role": "assistant", "content": result["reply"]})
    if result["crisis"]:
        info = "**CRISIS PROTOCOL ACTIVATED**"
    else:
        pattern = result["distortion"] or "No clear pattern identified"
        score = result["confidence"]
        info = (
            f"**Stage:** Final reflection\n\n"
            f"**Overall detected pattern:** {pattern.replace('_', ' ')}"
        )
        if score is not None:
            info += f"\n\n**Classifier score:** {score:.3f} (model score, not a clinical probability)"
        else:
            info += "\n\n**Classifier status:** No trained classifier was loaded."
    return history, info, gr.update(interactive=False)

def reset():
    bot.reset()
    return [{"role": "assistant", "content": WELCOME}], "", gr.update(interactive=False)

with gr.Blocks(title="LLM-based CBT Counselling Assistant") as demo:
    gr.Markdown("# LLM-based CBT Counselling Assistant\n" + DISCLAIMER)
    gr.Markdown(
        "### One-to-one reflection\n"
        "Share your experience naturally. The assistant will ask follow-up questions "
        "one at a time. The overall thinking-pattern analysis appears at the end."
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": WELCOME}],
                height=500,
                label="Counselling conversation",
            )
            box = gr.Textbox(
                placeholder="Type your response here...",
                show_label=False,
                lines=2,
            )
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                finish_btn = gr.Button("Finish reflection", interactive=False)
                reset_btn = gr.Button("New session")
        with gr.Column(scale=1):
            info = gr.Markdown(
                f"**Backend:** {bot.llm.name}\n\n"
                f"**Classifier:** {bot.clf.kind}\n\n"
                "**Pattern detection:** Delayed until final reflection (after 10 user turns)."
            )

    box.submit(chat, [box, chatbot], [chatbot, box, info, finish_btn])
    send.click(chat, [box, chatbot], [chatbot, box, info, finish_btn])
    finish_btn.click(finish_reflection, [chatbot], [chatbot, info, finish_btn])
    reset_btn.click(reset, None, [chatbot, info, finish_btn])

if __name__ == "__main__":
    demo.launch()

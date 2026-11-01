"""Rule-based crisis / safety layer. Runs BEFORE any model is called."""
import re

HIGH_RISK = [
    r"\bkill myself\b", r"\bsuicid", r"\bend my life\b", r"\bend it all\b",
    r"\bwant to die\b", r"\bwanna die\b", r"\bbetter off dead\b", r"\bno reason to live\b",
    r"\bhurt myself\b", r"\bharm myself\b", r"\bself[- ]harm", r"\bcut myself\b",
    r"\bdon'?t want to (be alive|live)\b", r"\btake my own life\b", r"\boverdose\b",
]
_RX = [re.compile(p, re.I) for p in HIGH_RISK]

CRISIS_MESSAGE = (
    "I'm really glad you told me, and I'm concerned about your safety. I'm an AI "
    "tool and can't give the help you deserve right now, but trained people can.\n\n"
    "- In India: call Tele-MANAS at 14416 (free, 24x7).\n"
    "- Emergency: call 112 right now if you may act on these thoughts.\n"
    "- Please reach out to someone near you - a friend, family member, or a "
    "counsellor at your college - and stay with them.\n\n"
    "You don't have to go through this alone. I'm here to keep talking if you'd like."
)

def check_risk(text: str) -> bool:
    return any(rx.search(text) for rx in _RX)

"""
Builds a labelled cognitive-distortion dataset + CBT dialogue pairs.

IMPORTANT: this is a *synthetic seed* dataset so the whole pipeline runs end to end.
For your final report, add real data (see README: 'Datasets') and report results on it.
The test split uses template sentences never seen in training, so the score is not
inflated by memorisation of exact sentences.
"""
import json, random, os
from src.config import LABELS

random.seed(42)
TOPICS = ["my exam", "my presentation", "my friend", "my job interview", "my project",
          "my family", "my results", "my relationship", "my assignment", "my lab viva",
          "my placement", "my college"]
PREFIX = ["", "", "honestly, ", "I think ", "Lately ", "I keep feeling that ", "Ugh, "]
SUFFIX = ["", "", " I can't stop thinking about it.", " It's been on my mind all day.", " I don't know what to do."]

T = {
"all_or_nothing": [
 "If I don't score full marks in {t}, I am a complete failure.",
 "Either {t} goes perfectly or it is totally worthless.",
 "I made one mistake in {t}, so the whole thing is ruined.",
 "I'm either the best at {t} or I'm useless.",
 "There is no middle ground, {t} is a total disaster or a perfect success.",
 "If {t} isn't flawless, there's no point in even trying."],
"overgeneralization": [
 "I failed at {t}, so I will always fail at everything.",
 "Nothing ever works out for me, just like {t}.",
 "Everyone always leaves me, same as with {t}.",
 "I never get anything right, {t} proves it.",
 "This always happens to me, every single time with {t}.",
 "I messed up {t} once and now I'm sure I'll never succeed at anything."],
"mental_filter": [
 "Even though most of {t} went well, I can't stop thinking about that one mistake.",
 "Everyone praised {t}, but all I notice is the single criticism.",
 "I only see what went wrong in {t}, nothing else matters.",
 "One bad comment about {t} is all I can think about.",
 "The good parts of {t} don't count, only the flaw I made does.",
 "I keep replaying the one thing that went wrong with {t}."],
"disqualifying_positive": [
 "They said {t} was great, but they were just being nice.",
 "That success with {t} was just luck, it doesn't count.",
 "Anyone could have done {t}, so the praise means nothing.",
 "I got a good result in {t} but it was a fluke.",
 "When people compliment {t}, I know they don't really mean it.",
 "It doesn't matter that {t} went well, it wasn't a real achievement."],
"mind_reading": [
 "I know everyone thinks I'm stupid because of {t}.",
 "She didn't reply, so she must be angry about {t}.",
 "I can tell they secretly judge me for {t}.",
 "My teacher obviously thinks I'm lazy after {t}.",
 "They looked at me strangely, they must think {t} was terrible.",
 "I'm sure my friends are laughing at me about {t}."],
"fortune_telling": [
 "I just know {t} is going to go horribly wrong.",
 "There's no way I'll pass, {t} will end in failure.",
 "I will definitely embarrass myself in {t}.",
 "Nothing will improve; {t} will turn out badly like always.",
 "I'm certain they'll reject me after {t}.",
 "I can already tell {t} won't work out and I'll end up alone."],
"catastrophizing": [
 "If I fail {t}, my whole life is over.",
 "{t} going wrong would be an absolute catastrophe, I couldn't survive it.",
 "One mistake in {t} and I'll lose everything.",
 "This is the worst thing that could ever happen, {t} will ruin my future.",
 "If {t} falls apart, I'll never recover.",
 "I'll be humiliated forever if {t} goes badly."],
"emotional_reasoning": [
 "I feel like a failure after {t}, so I must be one.",
 "I feel so anxious about {t}, which means something terrible will happen.",
 "I feel stupid, so {t} must prove that I am stupid.",
 "I feel guilty about {t}, so it must be my fault.",
 "It feels hopeless, so {t} must be hopeless.",
 "I feel like nobody cares, so it must be true after {t}."],
"should_statements": [
 "I should have done better in {t}.",
 "I must never make mistakes in {t}.",
 "I ought to be able to handle {t} without any stress.",
 "I should always be perfect at {t}.",
 "I shouldn't feel nervous about {t}, there's something wrong with me.",
 "Other people shouldn't ever see me struggle with {t}."],
"labelling": [
 "I'm such an idiot for how {t} went.",
 "I'm a loser because of {t}.",
 "I'm a total failure, {t} proves it.",
 "I am worthless after {t}.",
 "He's a complete jerk for what happened with {t}.",
 "I'm just a stupid person, look at {t}."],
"personalization": [
 "{t} went badly and it's all my fault.",
 "My friend is upset, I must have caused it, it's about {t}.",
 "If I had done something different with {t}, nobody would have been hurt.",
 "The team lost because of me in {t}.",
 "I'm to blame for how {t} turned out.",
 "They are in a bad mood and it must be because of me and {t}."],
"none": [
 "I have {t} tomorrow and I'm planning to revise tonight.",
 "{t} went okay, some parts were good and some can improve.",
 "I felt nervous before {t}, but I managed it.",
 "I'm going to talk to someone about {t}.",
 "I'd like some help thinking about {t}.",
 "{t} was tiring, so I'm taking a short break."],
}

# Socratic CBT-style replies per distortion (used for generator fine-tuning data)
R = {
"all_or_nothing": "It sounds like {t} feels like either a total success or a total failure right now. Let's look for the middle ground: if you rated how it went on a scale of 0 to 100 instead of pass/fail, where would it honestly land, and what parts went well?",
"overgeneralization": "I hear how discouraging {t} has been. Words like 'always' and 'never' can make one event feel like a permanent pattern. Can you recall a time something did go right, even a small one?",
"mental_filter": "It's understandable that the one hard moment in {t} stands out. Let's widen the lens a little: what are two or three things that went reasonably well, even if they feel small?",
"disqualifying_positive": "It seems the good side of {t} is being waved away as luck or politeness. If a close friend had that same result, would you tell them it didn't count? What would you say to them?",
"mind_reading": "That sounds uncomfortable. When it comes to {t}, what actual evidence do you have about what they are thinking, and what other explanations could there be?",
"fortune_telling": "Feeling that {t} will go badly is a heavy thing to carry. How sure are you, from 0 to 100, and have your predictions in the past always come true? What is one small thing you could do to prepare?",
"catastrophizing": "It sounds like your mind is jumping to the worst case with {t}. Let's slow it down: what is the worst that could realistically happen, what is the most likely outcome, and how would you cope with it?",
"emotional_reasoning": "Your feelings about {t} are real and worth listening to. But a feeling is not always a fact. What evidence supports this feeling, and what evidence points the other way?",
"should_statements": "It sounds like you are holding yourself to a very strict rule about {t}. What would change if you replaced 'I should' with 'I would prefer to', and treated yourself as you would a friend?",
"labelling": "That is a harsh label to put on yourself over {t}. One event doesn't define a whole person. Can we describe what actually happened without the label, and what you might do differently next time?",
"personalization": "You seem to be carrying a lot of responsibility for how {t} turned out. Let's draw a pie chart together: what other factors played a part, and how much was truly in your control?",
"none": "Thank you for sharing that about {t}. What emotion is strongest right now, and what thought goes along with it? We can explore it together step by step.",
}

def build(n_per_class_train=140, n_per_class_test=40):
    train, test = [], []
    for lab in LABELS:
        tpls = T[lab]
        tr_t, te_t = tpls[:4], tpls[4:]          # templates held out for test
        for pool, bucket, n in ((tr_t, train, n_per_class_train), (te_t, test, n_per_class_test)):
            for _ in range(n):
                t = random.choice(TOPICS)
                text = random.choice(PREFIX) + random.choice(pool).format(t=t)
                text = text[0].upper() + text[1:]
                text += random.choice(SUFFIX)
                bucket.append({"text": text, "label": lab, "topic": t})
    random.shuffle(train); random.shuffle(test)
    k = int(0.15 * len(train))           # validation = 15% of train
    return train[k:], train[:k], test

def build_dialogues(rows):
    return [{"user": r["text"], "distortion": r["label"],
             "assistant": R[r["label"]].format(t=r["topic"])} for r in rows]

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    tr, va, te = build()
    for name, rows in (("train", tr), ("val", va), ("test", te)):
        with open(f"data/distortion_{name}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    with open("data/cbt_dialogues.jsonl", "w") as f:
        for r in build_dialogues(tr): f.write(json.dumps(r) + "\n")
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

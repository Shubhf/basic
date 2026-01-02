import streamlit as st
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
import joblib

# ========================================
# KNOWLEDGE BASE
# ========================================

KNOWLEDGE = {
    ("Prime Minister", "India"): "Narendra Modi",
    ("Prime Minister", "US"): "Joe Biden",
    ("Prime Minister", "UK"): "Rishi Sunak",

    ("Captain", "India"): "Rohit Sharma",
    ("Captain", "US"): "Varies by team and sport.",
    ("Captain", "UK"): "Depends on the team."
}

DUTIES = {
    "Prime Minister": (
        "Leads the government, sets national policy, chairs the cabinet, "
        "represents the country internationally, and oversees administration."
    ),
    "Captain": (
        "Leads strategy on the field, motivates teammates, coordinates with coaches, "
        "and makes tactical decisions during play."
    )
}

# ========================================
# STATE STRUCTURES
# ========================================

@dataclass
class ContextFrame:
    value: Optional[str] = None
    confidence: float = 0.0

    def decay(self, factor=0.85):
        self.confidence *= factor
        if self.confidence < 0.2:
            self.value = None


@dataclass
class DialogueState:
    domain: ContextFrame = field(default_factory=ContextFrame)
    subject: ContextFrame = field(default_factory=ContextFrame)
    role: ContextFrame = field(default_factory=ContextFrame)
    intent: ContextFrame = field(default_factory=ContextFrame)

    def decay_all(self):
        self.domain.decay()
        self.subject.decay()
        self.role.decay()
        self.intent.decay()

    def snapshot(self):
        return {
            "domain": vars(self.domain),
            "subject": vars(self.subject),
            "role": vars(self.role),
            "intent": vars(self.intent),
        }


# ========================================
# MODELS
# ========================================

clf = joblib.load("topic_classifier.pkl")
vectorizer = joblib.load("topic_vectorizer.pkl")


def predict_topic(text):
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]


# ========================================
# DETECTORS
# ========================================

def detect_dialogue_act(text: str) -> str:
    t = text.lower()

    if any(x in t for x in ["thanks", "wait", "ok", "fine"]):
        return "chit_chat"

    if any(x in t for x in ["now tell me", "switch to", "change topic"]):
        return "topic_shift"

    if any(x in t for x in ["what about", "his", "her", "their", "more about"]):
        return "contextual_continuation"

    return "fresh_query"


def detect_role(text: str):
    t = text.lower()
    if "prime minister" in t or "pm" in t:
        return "Prime Minister"
    if "captain" in t:
        return "Captain"
    if "coach" in t:
        return "Coach"
    return None


def detect_subject(text: str):
    words = text.strip().split()
    if len(words) <= 2:
        return None

    candidate = words[-1].strip("?.!,").title()

    blocked = {
        "who", "what", "about", "is",
        "pm", "captain", "coach",
        "minister", "president", "leader"
    }

    if candidate.lower() in blocked:
        return None

    return candidate


def detect_intent(text: str):
    t = text.lower()
    if "duties" in t or "responsibilities" in t:
        return "duties"
    if "who" in t:
        return "who"
    if "about" in t:
        return "info"
    return None


# ========================================
# STATE UPDATE — FIXED
# ========================================

def update_state_from_text(text, state: DialogueState):
    new_domain = predict_topic(text)

    if new_domain == "Unknown":
        new_domain = None

    # If topic changes → RESET dependent context
    if new_domain and new_domain != state.domain.value:
        state.subject = ContextFrame()
        state.role = ContextFrame()
        state.intent = ContextFrame()

    role = detect_role(text)
    subject = detect_subject(text)
    intent = detect_intent(text)

    if new_domain:
        state.domain.value = new_domain
        state.domain.confidence = 1.0

    if subject:
        state.subject.value = subject
        state.subject.confidence = 1.0

    if role:
        state.role.value = role
        state.role.confidence = 1.0

    if intent:
        state.intent.value = intent
        state.intent.confidence = 1.0


# ========================================
# EXPANSION
# ========================================

def expand_query(user_text, state: DialogueState, act: str):
    if act != "contextual_continuation":
        return None, None

    t = user_text.lower()

    # need context to expand
    if not state.role.value and not state.subject.value:
        return None, "clarify: not enough context"

    if "what about" in t:
        new_subject = detect_subject(user_text)

        if new_subject:
            if state.role.value:
                return f"Who is the {state.role.value} of {new_subject}?", None
            return f"Tell me more about {new_subject}.", None

        if state.role.value and state.subject.value:
            return f"Who is the {state.role.value} of {state.subject.value}?", None

        return None, "clarify: what exactly do you mean?"

    if "duties" in t or "responsibilities" in t:
        if state.role.value and state.subject.value:
            return (
                f"What are the duties of the {state.role.value} of {state.subject.value}?",
                None
            )
        return None, "clarify: whose duties?"

    if state.role.value and state.subject.value:
        return (
            f"Tell me more about the {state.role.value} of {state.subject.value}.",
            None
        )

    return None, "clarify: cannot infer expansion"


# ========================================
# ANSWER LAYER
# ========================================

def answer(expanded, state: DialogueState):
    text = expanded or ""

    role = state.role.value
    subject = state.subject.value

    if "duties" in text.lower():
        if role in DUTIES:
            return DUTIES[role]
        return "I don't have role responsibilities for this case yet."

    if role and subject and (role, subject) in KNOWLEDGE:
        return KNOWLEDGE[(role, subject)]

    if role and subject:
        return f"I don't have the exact answer for {role} of {subject} yet."

    return None


# ========================================
# TOPIC TAG
# ========================================

def assign_topic(text, state: DialogueState):
    if state.domain.value:
        return state.domain.value, (state.subject.value or "NA")
    return "General", "NA"


# ========================================
# MAIN TURN LOOP
# ========================================

def process_turn(user_text, state: DialogueState):
    state.decay_all()
    act = detect_dialogue_act(user_text)

    if act == "chit_chat":
        return None, ("General", "NA"), "chit_chat", None, state.snapshot()

    if act == "contextual_continuation":
        prev_domain = state.domain.value
        update_state_from_text(user_text, state)
        state.domain.value = prev_domain
    else:
        update_state_from_text(user_text, state)

    expanded, note = expand_query(user_text, state, act)
    topic = assign_topic(expanded or user_text, state)
    ans = answer(expanded, state)

    return expanded, topic, note, ans, state.snapshot()


# ========================================
# UI
# ========================================

st.set_page_config(page_title="Context-Aware Expansion", layout="wide")
st.title("Context-Aware Query Expansion (Subject-based)")

if "conversation" not in st.session_state:
    st.session_state.conversation = []
    st.session_state.state = DialogueState()

user_input = st.text_input("User input", key="u")

if st.button("Submit") and user_input:
    expanded, topic, note, ans, snap = process_turn(
        user_input, st.session_state.state
    )

    st.session_state.conversation.append({
        "user": user_input,
        "expanded": expanded,
        "topic": topic,
        "note": note,
        "answer": ans,
        "state": deepcopy(snap)
    })

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Conversation")

    for turn in st.session_state.conversation:
        st.markdown(f"**User:** {turn['user']}")

        if turn["expanded"]:
            st.markdown(f"**Expanded:** {turn['expanded']}")

        st.markdown(f"**Topic:** {turn['topic'][0]} → {turn['topic'][1]}")

        if turn.get("answer"):
            st.success(f"Answer: {turn['answer']}")

        if turn["note"]:
            st.info(turn["note"])

        st.markdown("---")

with col2:
    st.subheader("Dialogue State (latest)")
    if st.session_state.conversation:
        st.json(st.session_state.conversation[-1]["state"])

if st.button("Reset"):
    st.session_state.conversation = []
    st.session_state.state = DialogueState()

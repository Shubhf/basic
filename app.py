import streamlit as st
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

# ========================================
# CORE DATA STRUCTURES
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
    domain: ContextFrame = field(default_factory=ContextFrame)   # politics / sports / general
    subject: ContextFrame = field(default_factory=ContextFrame)  # india, uk, liverpool, etc
    role: ContextFrame = field(default_factory=ContextFrame)     # prime minister, captain …
    intent: ContextFrame = field(default_factory=ContextFrame)   # duties, who, info …

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
# DETECTORS
# ========================================

def detect_dialogue_act(text: str) -> str:
    t = text.lower()

    if any(x in t for x in ["thanks", "wait", "ok", "fine"]):
        return "chit_chat"

    if any(x in t for x in ["now tell me", "switch to", "change topic"]):
        return "topic_shift"

    if any(x in t for x in ["what about", "his", "her", "their", "that", "more about"]):
        return "contextual_continuation"

    return "fresh_query"


def detect_domain(text: str):
    t = text.lower()
    if any(x in t for x in ["pm", "prime minister", "parliament", "government"]):
        return "Politics"
    if any(x in t for x in ["captain", "cricket", "football", "team"]):
        return "Sports"
    return None


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
    # Very generic: take final token-ish subject mention
    # Later ML can replace this
    words = text.strip().split()
    if len(words) <= 2:
        return None

    # crude heuristic — last word often entity
    candidate = words[-1].strip("?.!,").title()

    if candidate.lower() in ["who", "what", "about", "is", "pm", "captain"]:
        return None

    return candidate


def detect_intent(text: str):
    t = text.lower()
    if "duties" in t or "responsibilities" in t:
        return "duties"
    if "who" in t:
        return "who"
    if "tell me" in t or "about" in t:
        return "info"
    return None


# ========================================
# STATE UPDATE
# ========================================

def update_state_from_text(text, state: DialogueState):
    domain = detect_domain(text)
    if domain and domain != state.domain.value:
        state.intent.value = None
       state.intent.confidence = 0.0
    role = detect_role(text)
    subject = detect_subject(text)
    intent = detect_intent(text)

    if domain:
        state.domain.value = domain
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
# EXPANSION ENGINE
# ========================================

def expand_query(user_text, state: DialogueState, act: str):
    """
    Always try to rewrite into a fully explicit question.
    If context missing → return clarify.
    """

    # If it's not contextual, no expansion needed
    if act != "contextual_continuation":
        return None, None

    # Need at least subject or role to expand
    if not state.subject.value and not state.role.value:
        return None, "clarify: not enough context"

    t = user_text.lower()

    # Case: "what about X"
    if "what about" in t:
        new_subject = detect_subject(user_text)

        if new_subject:
            # user explicitly changed subject
            if state.role.value:
                return (
                    f"Who is the {state.role.value} of {new_subject}?",
                    None
                )
            return (
                f"Tell me more about {new_subject}.",
                None
            )

        # No explicit new subject, use old one
        if state.role.value and state.subject.value:
            return (
                f"Who is the {state.role.value} of {state.subject.value}?",
                None
            )

        return None, "clarify: what exactly do you want to know?"

    # Case: pronoun / vague reference like "his duties"
    if "duties" in t or "responsibilities" in t:
        if state.role.value and state.subject.value:
            return (
                f"What are the duties of the {state.role.value} of {state.subject.value}?",
                None
            )
        return None, "clarify: whose duties?"

    # Other implicit references
    if state.role.value and state.subject.value:
        return (
            f"Tell me more about the {state.role.value} of {state.subject.value}.",
            None
        )

    return None, "clarify: cannot infer expansion"


# ========================================
# TOPIC ASSIGNMENT
# ========================================

def assign_topic(text, state: DialogueState):
    if state.domain.value:
        return state.domain.value, state.subject.value or "NA"
    return "General", "NA"


# ========================================
# MAIN TURN HANDLER
# ========================================

def process_turn(user_text, state: DialogueState):
    state.decay_all()
    act = detect_dialogue_act(user_text)

    if act == "chit_chat":
        return None, ("General", "NA"), "chit_chat", state.snapshot()

    # freeze domain on continuation
    if act == "contextual_continuation":
        prev_domain = state.domain.value
        update_state_from_text(user_text, state)
        state.domain.value = prev_domain
    else:
        update_state_from_text(user_text, state)

    expanded, note = expand_query(user_text, state, act)
    topic = assign_topic(expanded or user_text, state)

    return expanded, topic, note, state.snapshot()


# ========================================
# STREAMLIT UI
# ========================================

st.set_page_config(page_title="Context-Aware Expansion", layout="wide")
st.title("Context-Aware Query Expansion (Subject-based)")

if "conversation" not in st.session_state:
    st.session_state.conversation = []
    st.session_state.state = DialogueState()

user_input = st.text_input("User input", key="u")

if st.button("Submit") and user_input:
    expanded, topic, note, snap = process_turn(
        user_input, st.session_state.state
    )

    st.session_state.conversation.append({
        "user": user_input,
        "expanded": expanded,
        "topic": topic,
        "note": note,
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

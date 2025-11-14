# main_new.py
import streamlit as st
import os
import json
import random
from datetime import datetime
from dotenv import load_dotenv

# LangChain / LLM imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# ---------------- Setup ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROFILES_PATH = "profile.json"

# ---------------- Utils ----------------
def load_profiles(path=PROFILES_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_profiles(profiles, path=PROFILES_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)


def generate_dummy_from_profile(profile: dict, seed=None):
    if seed is not None:
        random.seed(seed)

    def pick(base, jitter=0.15, floor=None, ceil=None):
        val = base * (1 + random.uniform(-jitter, jitter))
        if floor is not None:
            val = max(val, floor)
        if ceil is not None:
            val = min(val, ceil)
        return round(val, 2)

    desc = profile.get("profile_description", "") if profile else ""
    lh_base = 5.0
    if "lh mean" in desc.lower():
        import re
        m = re.search(r"lh mean\s*≈\s*([0-9]*\.?[0-9]+)", desc.lower())
        if m:
            try:
                lh_base = float(m.group(1))
            except:
                lh_base = 5.0

    dummy = {
        "id": f"dummy_{random.randint(1000,9999)}",
        "created_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hormonal_rhythm": {"lh_mean": pick(lh_base, 0.25, 0.5, 12)},
        "llm_context": "Automatically generated dummy record derived from profile text."
    }
    return dummy


def build_system_prompt(profile_title: str, profile_description: str, additional_dummy: list = None):
    additional_block = ""
    if additional_dummy:
        additional_block = (
            "\n[Additional Dummy Data]\n"
            + json.dumps(additional_dummy, indent=2).replace("{", "{{").replace("}", "}}")
        )

    system_template = f"""
You are a helpful AI assistant fine-tuned for psychological and behavioral personalization.
Persona: {profile_title}

{profile_description}
{additional_block}

Follow this structure:
- 💬 Summary of user query
- 🧠 Contextual reasoning (persona-based)
- ✅ Concise advice or insight
"""
    return system_template


def create_persona_chatbot_llm(system_prompt: str):
    try:
        system_prompt_template = SystemMessagePromptTemplate.from_template(system_prompt)
        human_prompt = HumanMessagePromptTemplate.from_template("{input}")
        prompt = ChatPromptTemplate.from_messages([system_prompt_template, human_prompt])

        if not OPENAI_API_KEY:
            st.warning("OPENAI_API_KEY not set — using mock responses.")
            return None

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)

        store = {}

        def get_session_history(session_id: str):
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
            return store[session_id]

        return RunnableWithMessageHistory(
            runnable=prompt | llm,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    except Exception as e:
        st.error(f"LLM setup failed: {e}")
        return None


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Persona Chat", layout="centered")
st.markdown("<h1 style='text-align:center;color:#00ffff;'>💬 Persona Chat Assistant</h1>", unsafe_allow_html=True)

# Load profiles
profiles = load_profiles()
if not profiles:
    st.error("No profiles found in profile.json. Please add one.")
    st.stop()

profile_keys = list(profiles.keys())

col1, col2 = st.columns([2, 1])
with col1:
    selected_profile = st.selectbox(
        "Select Persona Profile",
        profile_keys,
        format_func=lambda k: f"{k} — {profiles[k].get('profile_title','')}",
    )
with col2:
    is_existing = st.selectbox("Existing User?", ["No", "Yes"]) == "Yes"

append_dummy = is_existing and st.checkbox("Append 2 dummy records to profile.json?", value=True)

# Start session
if st.button("🚀 Start Session"):
    profile_obj = profiles[selected_profile]
    additional_dummy = None

    if append_dummy:
        d1 = generate_dummy_from_profile(profile_obj, seed=random.randint(1, 99999))
        d2 = generate_dummy_from_profile(profile_obj, seed=random.randint(1, 99999))
        profile_obj.setdefault("dummy_records", []).extend([d1, d2])
        save_profiles(profiles)
        additional_dummy = [d1, d2]
        st.success("Added dummy data ✅")

    system_prompt = build_system_prompt(
        profile_obj.get("profile_title", selected_profile),
        profile_obj.get("profile_description", ""),
        additional_dummy,
    )

    chatbot = create_persona_chatbot_llm(system_prompt)

    st.session_state["chatbot"] = chatbot
    st.session_state["session_id"] = f"{selected_profile}_{random.randint(1000,9999)}"
    st.session_state["messages"] = [
        {"role": "assistant", "text": f"Hey! I'm your {profile_obj.get('profile_title','persona')} assistant 🤖"}
    ]
    st.success(f"Session started — {st.session_state['session_id']}")
    st.rerun()


# ---------------- Chat UI ----------------
if "messages" in st.session_state:
    st.markdown(
        "<div style='background:#0f172a;padding:10px;border-radius:10px;'>",
        unsafe_allow_html=True,
    )

    for m in st.session_state["messages"]:
        role, text = m["role"], m["text"]
        if role == "user":
            st.markdown(
                f"<div style='text-align:right;background:#1e293b;padding:10px;margin:6px;border-radius:10px;color:#e2e8f0;'>{text}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='text-align:left;background:#334155;padding:10px;margin:6px;border-radius:10px;color:#f8fafc;border:1px solid #475569;'>{text}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "text": user_input})

        chatbot = st.session_state.get("chatbot")
        if chatbot:
            try:
                response = chatbot.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": st.session_state["session_id"]}},
                )
                reply_text = getattr(response, "content", str(response))
            except Exception as e:
                st.error(f"LLM error: {e}")
                reply_text = "Mock reply (LLM error)"
        else:
            reply_text = "LLM not connected (mock reply)."

        st.session_state["messages"].append({"role": "assistant", "text": reply_text})
        st.rerun()

# ---------------- Session Info ----------------
with st.expander("🧠 Session Info"):
    st.json(
        {
            "session_id": st.session_state.get("session_id"),
            "messages_count": len(st.session_state.get("messages", [])),
            "chatbot_active": bool(st.session_state.get("chatbot")),
        }
    )

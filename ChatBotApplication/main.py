import os
import json
import random
from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# === Load environment variables ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")


# === Utility: load / save profiles ===
def load_profiles(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    return profiles


def save_profiles(file_path: str, profiles: dict):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)


# === Dummy generator (simple numeric-generator based on a profile) ===
def generate_dummy_from_profile(profile: dict, seed=None):
    """
    Generate a single dummy record using mild randomization around profile values when available.
    This can be replaced by a more sophisticated generator (time series, distributions, etc.)
    """
    if seed is not None:
        random.seed(seed)

    # Try to parse one of the numeric cues from profile_description if present,
    # otherwise use sensible defaults per category.
    def pick(base, jitter=0.15, floor=None, ceil=None):
        val = base * (1 + random.uniform(-jitter, jitter))
        if floor is not None:
            val = max(val, floor)
        if ceil is not None:
            val = min(val, ceil)
        # round floats sensibly
        return round(val, 2)

    # heuristics: attempt to extract an LH mean if present in description text
    desc = profile.get("profile_description", "") if profile else ""
    # fallback bases
    lh_base = 5.0
    if "lh mean" in desc.lower():
        # naive parse: look for first float after 'lh mean'
        import re
        m = re.search(r"lh mean\s*≈\s*([0-9]*\.?[0-9]+)", desc.lower())
        if m:
            try:
                lh_base = float(m.group(1))
            except:
                lh_base = 5.0

    # Build numeric dummy
    dummy = {
        "id": f"dummy_{random.randint(1000,9999)}",
        "created_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hormonal_rhythm": {
            "lh_mean": pick(lh_base, jitter=0.25, floor=0.5, ceil=12),
            "lh_median": pick(max(0.8, lh_base - 1.0), jitter=0.25, floor=0.2),
            "lh_std": pick(1.5, jitter=0.5, floor=0.05),
            "lh_peak_count_30days": int(max(0, round(random.gauss(3, 1.5)))),
        },
        "thermoregulation": {
            "nightly_temperature_skew": pick(-0.8 if "negative" in desc.lower() else 0.3, jitter=1.0, floor=-5, ceil=5),
            "luteal_temp_rise_magnitude_celsius": pick(0.22 if "luteal" in desc.lower() and "rise" in desc.lower() else 0.08, jitter=0.5, floor=-1, ceil=2),
            "luteal_temp_rise_present": random.choice([True, False]),
        },
        "fatigue_mood": {
            "fatigue_prop_VeryLowLittle": pick(0.15 if "fatigue" in desc.lower() else 0.35, jitter=0.4, floor=0.0, ceil=1.0),
            "mood_variability_std": pick(0.25, jitter=0.6, floor=0.0),
            "daytime_sleepiness_score_0_10": pick(4.0, jitter=0.7, floor=0.0, ceil=10.0),
        },
        "sleep_patterns": {
            "restlessness_range_minutes": int(max(0, round(random.gauss(18, 12)))),
            "deep_sleep_in_minutes_range": int(max(0, round(random.gauss(28, 18)))),
            "total_minutes_trend_per_week_minutes": int(round(random.uniform(-30, 30))),
            "sleep_efficiency_percent": int(max(40, min(98, round(random.gauss(85, 8))))),
        },
        "activity": {
            "daily_steps_mean": int(max(500, round(random.gauss(7000, 2500)))),
            "elevationgain_kurt": pick(2.5, jitter=1.5, floor=0.1),
            "infrared_red_std": pick(0.08, jitter=0.6, floor=0.0),
            "burstiness_index": pick(0.4, jitter=0.7, floor=0.0, ceil=1.5),
        },
        "hrv": {
            "rmssd_ms": pick(52, jitter=0.5, floor=8, ceil=200),
            "recovery_curve_slope": pick(0.07, jitter=0.6, floor=-1, ceil=1),
            "hrv_sdnn_ms": int(max(10, round(random.gauss(45, 18)))),
        },
        "llm_context": "Automatically generated dummy record derived from profile text.",
    }
    return dummy


# === Create Chatbot for Selected Persona (updated to accept appended dummy data in system prompt) ===
def create_persona_chatbot(profile_name: str, profiles: dict, additional_dummy: list = None):
    profile = profiles.get(profile_name)
    if not profile:
        raise ValueError(f"Profile '{profile_name}' not found in profiles.json")

    profile_description = profile["profile_description"]
    profile_title = profile["profile_title"]

    # Prepare optional additional dummy block to be injected in the system prompt
    additional_block = ""
    if additional_dummy:
        # include a compact JSON snippet for LLM to use
        compact_dummy = json.dumps(additional_dummy, indent=2)
        additional_block = f"\n[Additional Dummy Data — appended]\n{compact_dummy}\n"

    # --- System prompt injection ---
    system_template = f"""
You are a helpful AI assistant fine-tuned for psychological and behavioral personalization.
You will use the following persona context to tailor your tone, reasoning, and advice:

[Persona Description — {profile_title}]
{profile_description}
{additional_block}

Rules:
1. Always stay consistent with the persona's traits, motivation, and tone.
2. Structure your responses clearly in this order:
   - 💬 Summary of understanding the user query
   - 🧠 Contextual reasoning (based on persona traits)
   - ✅ Actionable, structured advice or answer
3. Keep answers short, clear, and psychologically aligned with the persona.

Start now.
    """

    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    # instantiate the llm
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)

    # === Define chat history memory ===
    store = {}  # store histories by session_id

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # === Combine LLM + Prompt + Memory ===
    chain_with_history = RunnableWithMessageHistory(
        runnable=prompt | llm,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return chain_with_history


# === Run the Chatbot with interactive existing/new user flow ===
if __name__ == "__main__":
    profiles_path = "profile.json"
    profiles = load_profiles(profiles_path)

    print("Available Profiles:")
    for name, p in profiles.items():
        print(f"- {name}: {p.get('profile_title', 'no title')}")

    # Ask whether this is an existing user scenario
    is_existing = input("\nIs this an existing user profile you'd like to update? (y/n): ").strip().lower()
    additional_dummy = None

    if is_existing in ("y", "yes"):
        chosen = input("Enter profile name to update (e.g., Profile1): ").strip()
        if chosen not in profiles:
            print(f"Profile '{chosen}' not found. Exiting.")
            raise SystemExit(1)

        # Ask whether to append generated dummy records (simple flow)
        append_choice = input("Append 2 generated dummy records to this profile? (y/n): ").strip().lower()
        if append_choice in ("y", "yes"):
            # generate two dummy records and append them under a new key 'dummy_records' (create if missing)
            d1 = generate_dummy_from_profile(profiles[chosen], seed=random.randint(1, 99999))
            d2 = generate_dummy_from_profile(profiles[chosen], seed=random.randint(1, 99999))
            profiles[chosen].setdefault("dummy_records", [])
            profiles[chosen]["dummy_records"].extend([d1, d2])

            # persist the updated profiles file
            save_profiles(profiles_path, profiles)
            print(f"Appended 2 dummy records to {chosen} and saved to {profiles_path}.")

            # prepare to inject these into the system prompt
            additional_dummy = [d1, d2]

        # create chatbot with (possibly) appended dummy data
        chatbot = create_persona_chatbot(chosen, profiles, additional_dummy=additional_dummy)
        session_id = chosen

    else:
        # New user flow: choose a profile to drive persona but do not append data to file
        chosen = input("Enter profile name to use for this session (e.g., Profile1): ").strip()
        if chosen not in profiles:
            print(f"Profile '{chosen}' not found. Exiting.")
            raise SystemExit(1)
        chatbot = create_persona_chatbot(chosen, profiles, additional_dummy=None)
        session_id = f"{chosen}_newsession"

    print(f"\nChatbot ready! Persona = {profiles[chosen]['profile_title']}")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = chatbot.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        # response object from RunnableWithMessageHistory may be a complex object; try to extract content
        try:
            content = response.content
        except Exception:
            # fallback: stringify response
            content = str(response)
        print(f"Bot: {content}\n")

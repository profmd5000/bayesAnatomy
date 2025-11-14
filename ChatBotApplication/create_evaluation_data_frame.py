import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from main import (  
    load_profiles,
    create_persona_chatbot
)

# === CONFIG ===
QUESTIONS = [
    "Lately I’ve been feeling more fatigued than usual before my period — could this be linked to my hormonal rhythm or sleep cycle?",
    "I’ve noticed my deep sleep minutes are dropping — what could be causing that, and how can I improve it?",
    "My temperature doesn’t seem to rise much after ovulation — does that mean my luteal phase isn’t normal?",
    "I walk around 6,000–7,000 steps daily, but still feel tired — is that enough activity for stable mood and energy levels?",
    "I’ve been experiencing mood swings lately — can my HRV or sleep data help me understand why?"
]

OUTPUT_FILE_JSON = "persona_responses.json"
OUTPUT_FILE_CSV = "persona_responses.csv"
PROFILES_PATH = "profile.json"

# === Load environment and profiles ===
print("🌱 Loading environment and profiles...")
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env")

profiles = load_profiles(PROFILES_PATH)
print(f"📂 Loaded {len(profiles)} profiles successfully.\n")

# === Prepare result storage ===
records = []

# === Iterate over each persona ===
for persona_name, profile in tqdm(profiles.items(), desc="🤖 Processing personas", colour="cyan"):
    print(f"\n🧩 Creating chatbot for persona: {persona_name} — {profile.get('profile_title','(no title)')}")
    chatbot = create_persona_chatbot(persona_name, profiles)
    session_id = f"auto_{persona_name}"

    # progress bar for questions per persona
    for q in tqdm(QUESTIONS, desc=f"💬 Chatting as {persona_name}", leave=False, colour="green"):
        try:
            response = chatbot.invoke(
                {"input": q},
                config={"configurable": {"session_id": session_id}},
            )
            content = getattr(response, "content", str(response))

            records.append({
                "persona_name": persona_name,
                "persona_title": profile.get("profile_title", ""),
                "question": q,
                "answer": content.strip(),
            })
            tqdm.write(f"✅ [{persona_name}] Answered: {q[:50]}...")

        except Exception as e:
            tqdm.write(f"⚠️ [{persona_name}] Error: {e}")

print("\n🧠 All personas processed. Saving results...")

# === Save results to DataFrame ===
df = pd.DataFrame(records)

df.to_json(OUTPUT_FILE_JSON, orient="records", indent=2, force_ascii=False)
df.to_csv(OUTPUT_FILE_CSV, index=False)

print(f"\n🎉 Done! Saved {len(df)} total Q&A pairs:")
print(f"   📘 JSON → {OUTPUT_FILE_JSON}")
print(f"   📗 CSV  → {OUTPUT_FILE_CSV}")
print("✨ All complete.\n")

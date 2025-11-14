"""
compare_persona_answers.py — with emoji logging & tqdm progress

Same as before, but includes:
✅ tqdm progress bars for major loops
✅ Emoji print logs for clarity
✅ Colored stage updates
"""

import os
import argparse
import json
from collections import defaultdict
import re

import pandas as pd
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
from PIL import Image

# LangChain / OpenAI imports for GPT evaluator
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


# -----------------------------
# Utility: ROUGE-L (LCS-based)
# -----------------------------
def lcs_length(a: str, b: str) -> int:
    a_words, b_words = a.split(), b.split()
    n, m = len(a_words), len(b_words)
    if not n or not m:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if a_words[i - 1] == b_words[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def rouge_l_fscore(hyp: str, ref: str, beta: float = 1.0) -> float:
    if not hyp.strip() or not ref.strip():
        return 0.0
    lcs = lcs_length(hyp, ref)
    prec = lcs / len(hyp.split()) if hyp.split() else 0.0
    rec = lcs / len(ref.split()) if ref.split() else 0.0
    if prec + rec == 0:
        return 0.0
    return (1 + beta**2) * (prec * rec) / ((beta**2) * prec + rec)


# -----------------------------
# GPT Evaluator
# -----------------------------
def gpt_evaluate_similarity(openai_model, a: str, b: str) -> float:
    system = SystemMessage(
        content="You are an assistant that rates how similar two text answers are. "
                "Return only a single number 0–100 (higher means more similar)."
    )
    human = HumanMessage(content=f"Answer A:\n{a}\n\nAnswer B:\n{b}\n\nRate similarity 0–100, reply with only number.")
    resp = openai_model([system, human])
    text = getattr(resp, "content", str(resp)).strip()
    m = re.search(r"(\d{1,3}(?:\.\d+)?)", text)
    if m:
        return float(np.clip(float(m.group(1)), 0, 100))
    return 0.0


# -----------------------------
# Main pipeline
# -----------------------------
def main(input_path: str, output_dir: str, api_model_name: str = "gpt-4o-mini"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"{Fore.GREEN}🌱 Loading persona data...{Style.RESET_ALL}")
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_json(input_path, orient='records')

    required = {"persona_name", "question", "answer"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns {required}")

    persona_list = df["persona_name"].unique().tolist()
    if len(persona_list) < 3:
        raise ValueError("Need at least 3 personas.")
    selected_personas = persona_list[:3]
    print(f"🧩 Comparing personas: {', '.join(selected_personas)}\n")

    print(f"{Fore.YELLOW}🔑 Loading GPT model...{Style.RESET_ALL}")
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY missing")
    openai_model = ChatOpenAI(model=api_model_name, temperature=0.0, api_key=key)
    print("🤖 GPT evaluator ready!\n")

    print(f"{Fore.CYAN}📊 Preparing TF-IDF model...{Style.RESET_ALL}")
    all_answers = df["answer"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer().fit(all_answers)

    results = []
    questions = df["question"].unique().tolist()

    print(f"{Fore.MAGENTA}💬 Comparing answers across {len(questions)} questions...{Style.RESET_ALL}")
    for q in tqdm(questions, desc="🔎 Evaluating Questions", colour="cyan"):
        sub = df[df["question"] == q]
        answers = {p: sub[sub["persona_name"] == p]["answer"].values[0] if len(sub[sub["persona_name"] == p]) else "" 
                   for p in selected_personas}

        for i in range(len(selected_personas)):
            for j in range(i + 1, len(selected_personas)):
                a, b = selected_personas[i], selected_personas[j]
                ans_a, ans_b = answers[a], answers[b]

                cos = 0.0
                rouge = 0.0
                gpt_score = 0.0
                if ans_a.strip() and ans_b.strip():
                    vecs = vectorizer.transform([ans_a, ans_b])
                    cos = float(cosine_similarity(vecs[0], vecs[1])[0, 0])
                    rouge = rouge_l_fscore(ans_a, ans_b)
                    try:
                        gpt_score = gpt_evaluate_similarity(openai_model, ans_a, ans_b)
                    except Exception as e:
                        tqdm.write(f"⚠️ GPT error for {a} vs {b}: {e}")

                results.append({
                    "question": q,
                    "persona_a": a,
                    "persona_b": b,
                    "cosine_sim": cos,
                    "rouge_l_f1": rouge,
                    "gpt_score_0_100": gpt_score
                })

    res_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "persona_comparisons.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved pairwise results to {csv_path}")

    agg = res_df.groupby(["persona_a", "persona_b"])[["cosine_sim", "rouge_l_f1", "gpt_score_0_100"]].mean().reset_index()

    print(f"{Fore.BLUE}📈 Generating graphs...{Style.RESET_ALL}")
    metric_files = {}
    metrics = [
        ("cosine_sim", "Cosine Similarity (TF-IDF)"),
        ("rouge_l_f1", "ROUGE-L F1"),
        ("gpt_score_0_100", "GPT Evaluator (0-100)")
    ]
    x_labels = agg.apply(lambda r: f"{r['persona_a']} vs\n{r['persona_b']}", axis=1).tolist()

    for col, title in tqdm(metrics, desc="🖼️ Plotting", colour="green"):
        vals = agg[col].tolist()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x_labels, vals, color="skyblue", edgecolor="black")
        ax.set_title(f"{title}")
        ax.set_ylim(0, max(1.0, max(vals) * 1.15))
        plt.xticks(rotation=15)
        plt.tight_layout()
        path = os.path.join(output_dir, f"metric_{col}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        metric_files[col] = path
        tqdm.write(f"✅ Saved {title} plot → {path}")

    print(f"{Fore.YELLOW}🧩 Combining plots into one image...{Style.RESET_ALL}")
    imgs = [Image.open(metric_files[m[0]]) for m in metrics]
    widths, heights = zip(*(i.size for i in imgs))
    total_width, max_height = sum(widths), max(heights)
    combined = Image.new("RGB", (total_width, max_height), "white")
    x_offset = 0
    for img in imgs:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width

    final_path = os.path.join(output_dir, "comparison_all_metrics.png")
    combined.save(final_path)
    print(f"🎯 Final combined image saved → {final_path}\n")
    print(f"{Fore.GREEN}✨ All done! Results ready in {output_dir}{Style.RESET_ALL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="persona_responses.csv", help="Input CSV/JSON path")
    parser.add_argument("--output_dir", "-o", type=str, default="./comparisons", help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model for GPT evaluator")
    args = parser.parse_args()

    main(args.input, args.output_dir, api_model_name=args.model)

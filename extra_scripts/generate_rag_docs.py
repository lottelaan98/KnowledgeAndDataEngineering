import os
import json
import re
import pandas as pd
from collections import defaultdict

# ---------------- CONFIG ----------------
CSV_PATH = "data/symptom2disease.csv"
SYMPTOM_PATH = "ontology/symptoms.json"
OUTPUT_DIR = "rag/docs"
# ----------------------------------------


def normalize(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower())


def extract_symptoms(text: str, vocabulary: list[str]) -> set[str]:
    text = normalize(text)
    return {s for s in vocabulary if s in text}


def sentence_join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    with open(SYMPTOM_PATH) as f:
        symptom_vocab = json.load(f)

    disease_symptoms = defaultdict(set)

    # Aggregate symptoms per disease
    for _, row in df.iterrows():
        disease = row["label"].strip()
        symptoms = extract_symptoms(row["text"], symptom_vocab)
        disease_symptoms[disease].update(symptoms)

    # Generate ONE file per disease
    for disease, symptoms in disease_symptoms.items():
        if not symptoms:
            continue

        symptom_list = sorted(symptoms)
        symptom_sentence = sentence_join(symptom_list)

        content = f"""DISEASE: {disease}

KNOWN SYMPTOMS:
- """ + "\n- ".join(symptom_list) + f"""

EXPLANATION:
{disease} is commonly associated with {symptom_sentence}.
These symptoms are frequently reported together in clinical descriptions.
This explanation is based on observed symptom patterns and does not constitute a medical diagnosis.
"""

        filename = disease.replace(" ", "") + ".md"
        path = os.path.join(OUTPUT_DIR, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Generated: {path}")


if __name__ == "__main__":
    main()

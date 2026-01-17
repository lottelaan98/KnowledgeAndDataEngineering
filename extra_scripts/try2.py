import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

CSV_PATH = "data/symptom2disease.csv"
OUTPUT_PATH = "ontology/symptom_candidates2.txt"

STOPWORDS = set(ENGLISH_STOP_WORDS)

# add extra fillers that often slip through
EXTRA_STOPWORDS = {
    "also", "ever", "really", "quite", "just", "like",
    "ive", "im", "dont", "cant", "wont",
    "morning", "today", "yesterday", "recently",
}
STOPWORDS |= EXTRA_STOPWORDS

STOP_PHRASES = {
    "have been", "has been", "had been",
    "been feeling", "been having",
    "there is", "there are",
    "in my", "on my", "for me",
    "the itching", "the skin",
}

def normalize(text: str) -> str:
    return re.sub(r"[^a-z ]", "", str(text).lower())

def main():
    df = pd.read_csv(CSV_PATH)

    counter = Counter()

    # row iteration
    for _, row in df.iterrows():
        disease = str(row["label"]).strip()
        text = normalize(row["text"])
        tokens = text.split()

        for i in range(len(tokens)):
            w1 = tokens[i]

            # unigram filter
            if w1 in STOPWORDS or len(w1) <= 3:
                continue
            counter[(disease, w1)] += 1

            # bigram filter
            if i < len(tokens) - 1:
                w2 = tokens[i + 1]
                bigram = f"{w1} {w2}"

                if (
                    w2 in STOPWORDS
                    or bigram in STOP_PHRASES
                ):
                    continue

                counter[(disease, bigram)] += 1

    # keep only frequent items
    MIN_COUNT = 10
    candidates = [
        (disease, symptom, count)
        for (disease, symptom), count in counter.items()
        if count > MIN_COUNT
    ]

    # sort by disease then symptom
    candidates.sort(key=lambda x: (x[0].lower(), -x[2], x[1]))


    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for disease, symptom, count in candidates:
            f.write(f"{disease} | {symptom} | {count}\n")

    print(f"Wrote {len(candidates)} disease–symptom candidates to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()


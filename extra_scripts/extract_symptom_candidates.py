import pandas as pd
import re
from collections import Counter

CSV_PATH = "data/symptom2disease.csv"
OUTPUT_PATH = "ontology/symptom_candidates.txt"


def normalize(text):
    return re.sub(r"[^a-z ]", "", text.lower())


def main():
    df = pd.read_csv(CSV_PATH)

    counter = Counter()

    for text in df["text"]:
        text = normalize(text)
        tokens = text.split()

        # collect unigrams and bigrams
        for i in range(len(tokens)):
            counter[tokens[i]] += 1
            if i < len(tokens) - 1:
                counter[f"{tokens[i]} {tokens[i+1]}"] += 1

    # filter obvious junk
    candidates = [
        s for s, c in counter.items()
        if c > 10 and len(s) > 3
    ]

    candidates = sorted(candidates)

    with open(OUTPUT_PATH, "w") as f:
        for c in candidates:
            f.write(c + "\n")

    print(f"Wrote {len(candidates)} candidates to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
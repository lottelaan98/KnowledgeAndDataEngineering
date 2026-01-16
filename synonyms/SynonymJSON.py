from pathlib import Path
import json
import numpy as np
import fasttext
import faiss


# --------- paths ----------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "BioWordVec_PubMed_MIMICIII_d200.bin"
INDEX_PATH = BASE_DIR / "faiss_subset" / "index.faiss"
WORDS_PATH = BASE_DIR / "faiss_subset" / "words.json"
OUT_JSON   = BASE_DIR / "faiss_subset" / "symptom_synonyms.json"


# --------- helpers ----------
def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def find_synonyms(model, index, words, term: str, k: int = 10):
    q = np.array(model.get_sentence_vector(term), dtype="float32").reshape(1, -1)
    q = normalize_rows(q)

    D, I = index.search(q, k)
    return [(words[i], float(D[0][j])) for j, i in enumerate(I[0])]


# --------- main ----------
def main():
    print("Loading fastText model...")
    model = fasttext.load_model(str(MODEL_PATH))

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading words...")
    words = json.loads(WORDS_PATH.read_text(encoding="utf-8"))

    print(f"Exporting synonyms for {len(words)} terms...")
    symptom_to_synonyms = {}

    for term in words:
        syns = find_synonyms(model, index, words, term, k=10)

        # optional: remove self-match
        syns = [(w, s) for (w, s) in syns if w != term]

        # tuples -> lists (JSON-safe)
        symptom_to_synonyms[term] = [[w, s] for (w, s) in syns]

    print("Saving JSON...")
    OUT_JSON.write_text(
        json.dumps(symptom_to_synonyms, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Done.")
    print("Saved to:", OUT_JSON)


if __name__ == "__main__":
    main()

from pathlib import Path
import json
import numpy as np
import fasttext
import faiss


# --------- paths ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "BioWordVec_PubMed_MIMICIII_d200.bin"
CAND_PATH  = BASE_DIR / "symptom_candidates.txt"

OUT_DIR = BASE_DIR / "faiss_subset"
INDEX_PATH = OUT_DIR / "index.faiss"
WORDS_PATH = OUT_DIR / "words.json"


# --------- helpers ----------
def read_candidates(path: Path) -> list[str]:
    terms = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t and t not in seen:
            seen.add(t)
            terms.append(t)
    return terms


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def build_subset_index(model, terms: list[str]) -> tuple[faiss.Index, list[str]]:
    """
    Build FAISS index over ONLY the given terms.
    Cosine similarity is done by L2-normalizing vectors + inner product.
    """
    dim = model.get_dimension()

    # Use sentence vectors so multi-word symptoms work out of the box
    X = np.vstack([model.get_sentence_vector(t) for t in terms]).astype("float32")
    X = normalize_rows(X)

    index = faiss.IndexFlatIP(dim)  # exact search, usually fast enough for subset
    index.add(X)

    return index, terms


def query(index: faiss.Index, words: list[str], model, term: str, k: int = 10):
    q = np.array(model.get_sentence_vector(term), dtype="float32").reshape(1, -1)
    q = normalize_rows(q)

    D, I = index.search(q, k)
    return [(words[i], float(D[0][j])) for j, i in enumerate(I[0])]


# --------- main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading fastText model...")
    model = fasttext.load_model(str(MODEL_PATH))

    print("Reading candidates...")
    candidates = read_candidates(CAND_PATH)
    print("Num candidates:", len(candidates))

    print("Building FAISS index (subset only)...")
    index, words = build_subset_index(model, candidates)

    print("Saving index + words...")
    faiss.write_index(index, str(INDEX_PATH))
    WORDS_PATH.write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")

    # Quick tests
    for term in ["fever", "shortness of breath"]:
        print(f"\nTop neighbors for: {term!r}")
        for w, s in query(index, words, model, term, k=10):
            print(f"{s:0.4f}\t{w}")


if __name__ == "__main__":
    main()
